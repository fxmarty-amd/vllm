# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 asm decode layers' folded all-reduces on 4 gfx950 GPUs,
with the real collectives: ``WoBAllReduce`` (wo_b + all-reduce #1) and
``CombAllReduce`` (MoE combine + all-reduce #2), per wire, against vLLM's own
``tensor_model_parallel_all_reduce`` (main's path: AITER's custom all-reduce
at these sizes, no quick-reduce quantization) on the same per-rank partials,
and against the fp64 sum of the partials.

* bf16 wire: bitwise main's all-reduce (both: the bf16 rounding of the fp32
  sum), on every rank;
* MXFP8 wire: within MXFP8_TOLERANCE of the fp64 sum (one E8M0 a 32 elements,
  e4m3 partials, the owner's re-quantization), the same on every rank.

One spawn of 4 rank processes computes every item (seconds); the items check
its results."""

import json
import os
import socket

import pytest
import torch

BF16 = torch.bfloat16
HIDDEN, WO_B_K = 5120, 2048
ROW_COUNTS = (17, 64, 65, 128, 129, 256)  # the kernels' modes change at 64 / 65 and 128 / 129
KINDS = ("wo_b", "combine")
SPREADS = ("gaussian", "outliers")
WIRES = {"bf16": "bf16", "mxfp8": "fp8"}  # test name -> the runners' wire argument
# the MXFP8 wire against the fp64 sum (norm-relative; measured 3.5e-2)
MXFP8_TOLERANCE = 0.05
# main's all-reduce against the fp64 sum: the bf16 rounding of the output
BF16_TOLERANCE = 3e-3


def _mxfp8(x: torch.Tensor):
    """[M, K] f32 → e4m3 codes, E8M0 bytes a 32 (ceil rule)."""
    m, k = x.shape
    blocks = x.view(m, k // 32, 32)
    amax = blocks.abs().amax(-1).clamp_min(2.0**-126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(e).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.view(m, k), (e + 127).to(torch.uint8)


def _norm_relative(out: torch.Tensor, exact: torch.Tensor) -> float:
    return ((out.double() - exact).norm() / exact.norm()).item()


def _rank(rank: int, port: int, path: str) -> None:
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    os.environ.pop("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", None)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        tensor_model_parallel_all_reduce,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(4, rank, f"tcp://127.0.0.1:{port}", rank, backend="nccl")
        ensure_model_parallel_initialized(4, 1)
    communicator = get_tp_group().device_communicator
    taken: list = []
    for attribute, method, name in (
        ("aiter_ar_comm", "custom_all_reduce", "aiter_custom"),
        ("ca_comm", "custom_all_reduce", "vllm_custom"),
        ("qr_comm", "quick_all_reduce", "quick_reduce"),
        ("symm_mem_comm", "all_reduce", "symm_mem"),
        ("pynccl_comm", "all_reduce", "pynccl"),
    ):
        backend = getattr(communicator, attribute, None)
        if backend is not None and hasattr(backend, method):
            original = getattr(backend, method)

            def recorded(*args, _original=original, _name=name, **kwargs):
                out = _original(*args, **kwargs)
                if out is not None:
                    taken.append(_name)
                return out

            setattr(backend, method, recorded)

    gloo = dist.new_group(backend="gloo")
    from vllm.models.deepseek_v41.amd.asm.ar_runner import CombAllReduce, WoBAllReduce
    from vllm.models.deepseek_v41.amd.asm.kernels import gemv8
    from vllm.models.deepseek_v41.amd.asm.kernels.kmod import Kernel

    reduces = {}
    for wire, value in WIRES.items():
        reduces[wire] = (WoBAllReduce(rank, gloo, wire=value), CombAllReduce(rank, gloo, wire=value))
    source = gemv8.source(gemv8.WO_B1, "k")
    gemv = Kernel(source["name"], source["src"], source["ka"])
    unused = torch.zeros(64, dtype=torch.uint8, device=device)

    def gather(x: torch.Tensor) -> list:
        xs: list = [None] * 4
        dist.all_gather_object(xs, x.cpu(), group=gloo)
        return xs

    generator = torch.Generator().manual_seed(1234 + rank)
    weight = torch.randn(HIDDEN, WO_B_K, generator=generator) * 0.05
    weight_codes, weight_scales = _mxfp8(weight)
    weight_scales = weight_scales.view(HIDDEN, WO_B_K // 32)[::32].contiguous()
    w, ws = gemv8.frag_w(weight_codes.to(device), weight_scales.to(device), gemv8.WO_B1)
    results = []
    for spread in SPREADS:
        scale = 11.0 if spread == "outliers" else 1.0
        for m in ROW_COUNTS:
            # wo_b partials: this rank's GEMV
            x = torch.randn(m, WO_B_K, generator=generator) * scale
            if spread == "outliers":
                x[0, :8] = 3e4
            codes, scales = _mxfp8(x)
            z, zs = gemv8.frag_x(codes.to(device), scales.to(device))
            partial = torch.empty(m, HIDDEN, dtype=BF16, device=device)
            gemv(gemv8.WO_B1.grid, 256, w=w, ws=ws, x=z, xs=zs, out=partial, outs=unused,
                 part=unused, cnt=unused, cs=unused, pos=unused, dbg=unused, m=m)  # fmt: skip
            torch.cuda.synchronize()
            wo_b = {wire: pair[0](w, ws, z, zs, m).clone() for wire, pair in reduces.items()}
            # combine partials: 6 picks (5 % dropped) summed, plus the bf16 add
            y = (torch.randn(m * 6, HIDDEN, generator=generator) * scale).to(BF16)
            ids = torch.randint(0, 256, (m * 6,), generator=generator, dtype=torch.int32)
            ids[torch.rand(m * 6, generator=generator) < 0.05] = -1
            add = (torch.randn(m, HIDDEN, generator=generator) * scale).to(BF16)
            picks = torch.where((ids >= 0).view(-1, 1), y.float(), 0.0).view(m, 6, HIDDEN)
            routed = picks[:, 0].clone()
            for pick in range(1, 6):
                routed = routed + picks[:, pick]
            combine_partial = (routed.to(BF16).float() + add.float()).to(BF16).to(device)
            combine = {
                wire: pair[1](y.to(device), ids.to(device), add.to(device), m).clone()
                for wire, pair in reduces.items()
            }
            torch.cuda.synchronize()
            for kind, own, outs in (("wo_b", partial, wo_b), ("combine", combine_partial, combine)):
                taken.clear()
                main = tensor_model_parallel_all_reduce(own.clone())
                torch.cuda.synchronize()
                exact = sum(p.double() for p in gather(own))
                mains = gather(main)
                for wire, out in outs.items():
                    outs_all = gather(out)
                    results.append(dict(
                        spread=spread, m=m, kind=kind, wire=wire, main_backend=list(taken),
                        ranks_agree=all(torch.equal(outs_all[0], o) for o in outs_all),
                        main_ranks_agree=all(torch.equal(mains[0], o) for o in mains),
                        bits_differing_from_main=int((outs_all[0].view(torch.int16) != mains[0].view(torch.int16)).sum()),
                        error=_norm_relative(outs_all[0], exact),
                        main_error=_norm_relative(mains[0], exact),
                    ))
    if rank == 0:
        with open(path, "w") as f:
            json.dump(results, f)
    dist.barrier(group=gloo)


def _skip_unless_four_gfx950() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950() or torch.cuda.device_count() < 4:
        pytest.skip("needs 4 gfx950 GPUs")


@pytest.fixture(scope="module")
def all_reduces(tmp_path_factory) -> dict:
    """Every item's result, from one spawn of 4 rank processes."""
    _skip_unless_four_gfx950()
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    path = str(tmp_path_factory.mktemp("allreduce") / "results.json")
    mp.spawn(_rank, args=(port, path), nprocs=4, join=True)
    with open(path) as f:
        return {(r["spread"], r["m"], r["kind"], r["wire"]): r for r in json.load(f)}


@pytest.mark.parametrize("spread", SPREADS)
@pytest.mark.parametrize("num_rows", ROW_COUNTS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("wire", list(WIRES))
def test_folded_all_reduce(all_reduces, wire, kind, num_rows, spread):
    result = all_reduces[(spread, num_rows, kind, wire)]
    print(json.dumps(result))
    assert result["ranks_agree"] and result["main_ranks_agree"], result
    assert result["main_error"] < BF16_TOLERANCE, result
    if wire == "bf16":
        assert result["bits_differing_from_main"] == 0, result
    else:
        assert result["error"] < MXFP8_TOLERANCE, result
