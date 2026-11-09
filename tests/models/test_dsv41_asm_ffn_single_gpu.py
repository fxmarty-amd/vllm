# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2, the asm FFN half of a DeepSeek-V4.1 decoder layer
(``vllm/models/deepseek_v41/amd/asm/k2_runner.py``), on ONE gfx950 GPU against
the ops vLLM runs for the same step, without an engine.

K2 runs per TP rank up to its folded all-reduce: from the attention's reduced
output to this rank's FFN partial (``comb_in``: routed rows + shared expert),
which ``CombAllReduce`` then sums over the 4 ranks. A TP4 rank's partial is
exactly what vLLM's ``DeepseekV4MoE`` computes at TP1 on that rank's shard,
so the reference is the real module, built in-process at TP1 with the TP4
shard's sizes (``moe_intermediate_size`` 2304 / 4 = 576: the routed experts
padded 576 -> 640, the shared expert 576, as on a TP4 rank) and its weights
taken through ``process_weights_after_loading`` (AITER's MXFP4 MoE method,
its scale shuffles), as the model loader does.

Weights are gaussian, quantized to the checkpoint's formats (MXFP4 experts,
MXFP8 shared expert, one E8M0 a 32-block, scales by the ceil rule: no
saturation, no NaN / Inf codes); ``test_weights_are_gaussian`` checks that.

Compared at every row count where the kernels change mode, each stage on its
own input, so a failure names the stage:
- the seam (mHC post + pre + the norm) against ``MHCPreDelayedOp``;
- the MoE on the kernel's own normed input (routing cannot differ) against
  ``DeepseekV4MoE``;
- the partial against vLLM's whole FFN half, per token where both inputs pick
  the same experts; the picks themselves;
- CUDA-graph replays of several widths back to back, bit for bit.
The all-reduce itself needs 4 GPUs: ``test_dsv41_asm_allreduce.py``."""

import types

import pytest
import torch

from vllm.platforms import current_platform

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
H, HC, TOPK = 5120, 4, 6
INTER = 2304 // 4  # a TP4 rank's routed / shared intermediate
SIGMA = 0.02  # the experts' weight std
ROWS = (17, 64, 65, 128, 129, 255, 256)
REPLAYS = 100
BF, U8 = torch.bfloat16, torch.uint8
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _cos(a, b) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a, b) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


def _assert_ulp(got: torch.Tensor, want: torch.Tensor, frac: float = 1e-4) -> None:
    """bf16 tensors at most 1 ulp apart everywhere (fp32 sums in another
    order round to the neighbour), and that on under ``frac`` of the elements."""
    a = got.contiguous().view(torch.int16).int()
    b = want.contiguous().view(torch.int16).int()
    assert ((a < 0) == (b < 0)).all() or ((a - b).abs() <= 1).all()
    d = (a - b).abs()
    assert d.max() <= 1, d.max()
    assert (d > 0).float().mean() < frac, (d > 0).float().mean()


def _skip_unless_gfx950() -> None:
    if not current_platform.is_rocm():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.models.deepseek_v41.amd.asm.kernels.build import toolchain_missing

    if missing := toolchain_missing():
        pytest.skip(f"needs {missing} to build the kernels")


# ---------------------------------------------------------------- gaussian quantized weights


def _e8m0(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """The smallest power of 2 that brings amax within qmax (ceil rule) → E8M0."""
    e = torch.ceil(torch.log2(amax.clamp_min(2.0**-100) / qmax)).clamp(-127, 127)
    return e


def _gauss_mxfp8(n: int, k: int, g) -> tuple[torch.Tensor, torch.Tensor]:
    """N(0, SIGMA) [n, k] → e4m3 codes, E8M0 [n / 32, k / 32] (one a 32 x 32 block)."""
    w = torch.randn(n, k, generator=g, device=g.device) * SIGMA
    blk = w.view(n // 32, 32, k // 32, 32)
    e = _e8m0(blk.abs().amax(dim=(1, 3)), 448.0)
    q = (blk / torch.exp2(e)[:, None, :, None]).view(n, k).to(torch.float8_e4m3fn)
    return q, (e + 127).to(U8)


def _gauss_mxfp4(rows: int, k: int, g) -> tuple[torch.Tensor, torch.Tensor]:
    """N(0, SIGMA) [rows, k] → packed e2m1 [rows, k / 2] (element 2i in the low
    nibble), E8M0 [rows, k / 32]."""
    w = torch.randn(rows, k, generator=g, device=g.device) * SIGMA
    blk = w.view(rows, k // 32, 32)
    e = _e8m0(blk.abs().amax(-1), 6.0)
    v = (blk / torch.exp2(e)[..., None]).view(rows, k)
    mag = (v.abs()[..., None] - E2M1.to(v.device)).abs().argmin(-1)
    code = (mag | ((v < 0) & (mag > 0)).long() << 3).to(U8)
    return code[:, 0::2] | code[:, 1::2] << 4, (e + 127).to(U8)


def _dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1)
    mag = E2M1.to(packed.device)[(codes & 7).long()]
    v = torch.where(codes >= 8, -mag, mag)
    sc = torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
    return v * sc


def _fill_moe(moe, seed: int) -> None:
    """Gaussian weights in the loaded layout (before post-load processing): the
    MXFP8 shared expert (per-row scale tensors, alike over each 32 rows, as the
    loader writes them), the MXFP4 routed experts (the intermediate past INTER
    zero: the loader's padding), the bf16 router."""
    dev = moe.gate.weight.device
    g = torch.Generator(device=dev).manual_seed(seed)
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n, k = lin.weight.shape
        q, s = _gauss_mxfp8(n, k, g)
        lin.weight.data.copy_(q)
        sc = lin.weight_scale.data.view(U8)
        sc.copy_(s.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else s)
    e = moe.experts.routed_experts
    w13, s13 = e.w13_weight.data.view(U8), e.w13_weight_scale.data.view(U8)
    w2, s2 = e.w2_weight.data.view(U8), e.w2_weight_scale.data.view(U8)
    n_exp, two_pad, _ = w13.shape
    pad = two_pad // 2
    w13.zero_(), w2.zero_(), s13.fill_(127), s2.fill_(127)
    for x in range(n_exp):
        for half in (0, pad):
            q, s = _gauss_mxfp4(INTER, H, g)
            w13[x, half : half + INTER], s13[x, half : half + INTER] = q, s
        q, s = _gauss_mxfp4(H, INTER, g)
        w2[x, :, : INTER // 2], s2[x, :, : INTER // 32] = q, s
    gw = moe.gate.weight
    gw.data.copy_(torch.randn(gw.shape, generator=g, device=dev) * SIGMA)
    bias = moe.gate.e_score_correction_bias
    bias.data.copy_(torch.randn(bias.shape, generator=g, device=dev) * 0.02)


def _check_gaussian(w: torch.Tensor, what: str) -> None:
    w = w.float().reshape(-1)
    assert torch.isfinite(w).all(), what
    mean, std = w.mean().item(), w.std().item()
    kurt = (((w - mean) / std) ** 4).mean().item()
    assert abs(mean) < 0.02 * SIGMA, (what, mean)
    assert abs(std / SIGMA - 1) < 0.1, (what, std)
    assert 2.6 < kurt < 3.6, (what, kurt)  # gaussian: 3


# ---------------------------------------------------------------- the module


@pytest.fixture(scope="module")
def ffn():
    """vLLM's DeepseekV4MoE at TP1 with a TP4 rank's shapes, gaussian weights
    (raw copies kept for test_weights_are_gaussian), post-load processed, the
    FFN seam's parameters, and a K2Runner on the layer."""
    _skip_unless_gfx950()
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v41.amd.asm.k2_runner import K2Runner, raw_from_layer
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.utils.network_utils import get_open_port
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import init_workspace_manager

    mp = pytest.MonkeyPatch()
    mp.setenv("VLLM_ROCM_USE_AITER", "1")
    mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()
    dev = torch.device("cuda:0")
    torch.accelerator.set_device_index(dev)
    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=1,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=4096,
        language_model_only=True,
        hf_overrides={"moe_intermediate_size": INTER},
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    cfg = getattr(cfg, "text_config", cfg)
    with set_current_vllm_config(vc):
        init_distributed_environment(
            1, 0, f"tcp://127.0.0.1:{get_open_port()}", 0, backend="nccl"
        )
        ensure_model_parallel_initialized(1, 1)
    init_workspace_manager(dev)
    with set_current_vllm_config(vc), set_default_torch_dtype(BF), dev:
        moe = DeepseekV4MoE(vc, prefix="model.layers.21.ffn")
    _fill_moe(moe, seed=2001)
    e = moe.experts.routed_experts
    raw = dict(
        w13=_dequant_mxfp4(e.w13_weight.data.view(U8)[:4, :INTER], e.w13_weight_scale.data.view(U8)[:4, :INTER]),
        w2=_dequant_mxfp4(e.w2_weight.data.view(U8)[:4, :, : INTER // 2], e.w2_weight_scale.data.view(U8)[:4, :, : INTER // 32]),
        **{
            n: lin.weight.float() * torch.exp2(lin.weight_scale.view(U8).float() - 127).repeat_interleave(32, -1)
            for n, lin in (("gate_up", moe.shared_experts.gate_up_proj), ("down", moe.shared_experts.down_proj))
        },
    )  # fmt: skip
    process_weights_after_loading(moe, vc.model_config, dev)

    # the FFN seam's parameters, as a decoder layer holds them
    g = torch.Generator().manual_seed(1000)
    f32 = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd).to(dev)  # noqa: E731
    mix = (2 + HC) * HC
    layer = types.SimpleNamespace(
        ffn=moe,
        hc_ffn_fn=f32(mix, HC * H, sd=0.01),
        hc_ffn_scale=f32(3, sd=0.5).abs() + 0.5,
        hc_ffn_base=f32(mix, sd=0.1),
        ffn_norm=types.SimpleNamespace(
            weight=(1 + f32(H, sd=0.1)).to(BF), variance_epsilon=cfg.rms_norm_eps
        ),
        rms_norm_eps=cfg.rms_norm_eps,
        hc_eps=cfg.hc_eps,
        hc_post_alpha=2.0,  # DeepseekV4DecoderLayer's
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
    )
    from vllm.model_executor.layers.mhc import MHCPreDelayedOp

    with set_current_vllm_config(vc):
        mhc = MHCPreDelayedOp()
        run = K2Runner(dev)
    wt = run.weights(raw_from_layer(layer))
    yield types.SimpleNamespace(
        vc=vc, moe=moe, layer=layer, mhc=mhc, run=run, wt=wt, raw=raw, dev=dev
    )
    del run, moe
    destroy_model_parallel()
    destroy_distributed_environment()
    mp.undo()
    rocm_aiter_ops.refresh_env_variables()


def _inputs(m: int, dev, seed: int):
    """The FFN seam's inputs of an m-row step: the attention's reduced output,
    the residual streams, the previous seam's mixes (doubly stochastic comb)."""
    g = torch.Generator().manual_seed(seed)
    res = torch.randn(m, HC, H, generator=g)
    res = (res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True))).to(BF)
    x = (torch.randn(m, H, generator=g) * 0.5).to(BF)
    comb = torch.rand(m, HC, HC, generator=g) + 0.1
    for _ in range(20):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    post = torch.sigmoid(torch.randn(m, HC, 1, generator=g)) * 2
    pre = torch.sigmoid(torch.randn(m, HC, generator=g)) + 1e-6
    return [t.to(dev).contiguous() for t in (x, res, post, comb, pre)]


def _seam(f, x, res, post, comb, pre):
    """vLLM's FFN seam, as DeepseekV4DecoderLayer calls it (fused norm)."""
    ly = f.layer
    return f.mhc(
        res, ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base, ly.rms_norm_eps,
        ly.hc_eps, ly.hc_eps, ly.hc_post_alpha, ly.hc_sinkhorn_iters,
        pre_mix=pre, sublayer_out=x, post_layer_mix=post, comb_res_mix=comb,
        norm_weight=ly.ffn_norm.weight, norm_eps=ly.ffn_norm.variance_epsilon,
    )  # fmt: skip


def _moe(f, x):
    from vllm.forward_context import set_forward_context

    with set_forward_context(None, f.vc):
        ids = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return f.moe(x.clone(), ids)


def _top6(f, x):
    """vLLM's picks: sqrt(softplus(logits)) + bias, top 6; and the 6th / 7th gap."""
    gw = f.moe.gate.weight.float()
    sc = torch.nn.functional.softplus(x.float() @ gw.T).sqrt()
    top = (sc + f.moe.gate.e_score_correction_bias.float()).topk(7, dim=1)
    return top.indices[:, :6].sort(1).values, top.values[:, 5] - top.values[:, 6]


def _partial(run, m):
    """K2's pre-all-reduce partial: comb_ar's combine without the reduce,
    bf16(bf16(Σ_k y[t·6 + k]) + shared[t]), picks with ids < 0 reading 0."""
    y, ids, shared = run.comb_in
    y = y[: m * TOPK].float().view(m, TOPK, H)
    y = torch.where((ids[: m * TOPK] >= 0).view(m, TOPK, 1), y, 0.0)
    s = y[:, 0].clone()
    for k in range(1, TOPK):
        s = s + y[:, k]
    return (s.to(BF).float() + shared[:m].float()).to(BF)


# ---------------------------------------------------------------- tests


def test_weights_are_gaussian(ffn):
    """The quantized weights the kernels and vLLM read hold a gaussian of std
    SIGMA: finite, centered, the right spread and tails (no saturated or
    degenerate blocks)."""
    for name, w in ffn.raw.items():
        _check_gaussian(w, name)


@pytest.mark.parametrize("m", ROWS)
def test_k2_matches_vllm(ffn, m):
    run, wt = ffn.run, ffn.wt
    x, res, post, comb, pre = _inputs(m, ffn.dev, seed=11 + m)
    res2, post2, comb2, xf, pre2 = _seam(ffn, x, res, post, comb, pre)
    out = _moe(ffn, xf)

    _, r_new, post_k, comb_k, pre_k = run(wt, x, res, post, comb, pre)
    torch.accelerator.synchronize()
    part = _partial(run, m)
    for t in (r_new, post_k, comb_k, pre_k, part):
        assert torch.isfinite(t.float()).all()

    # the seam: residual streams and the next mixes
    _assert_ulp(r_new, res2)
    for got, want in ((post_k, post2), (comb_k, comb2), (pre_k, pre2)):
        torch.testing.assert_close(got.view_as(want), want, rtol=1e-4, atol=1e-5)
    # the FFN input: K2 keeps y' = x1 · ffn_norm.w without rstd and applies
    # rstd inside (router, stage 1); y' · rstd is the normed input
    normed = (run.s["y"][:m].float() * run.s["rstd"][:m, None]).to(BF)
    torch.testing.assert_close(normed, xf, rtol=2**-7, atol=1e-6)

    # the MoE on the kernel's own input: routing cannot differ
    same_in = _moe(ffn, normed)
    assert _cos(part, same_in) > 0.9998 and _rel(part, same_in) < 0.02, (
        _cos(part, same_in),
        _rel(part, same_in),
    )
    # the picks: K2's ids against vLLM's, where vLLM's 6th / 7th scores aren't
    # a near tie
    want_ids, gap = _top6(ffn, normed)
    got_ids = run.s["ids"][:m].long().sort(1).values
    clear = gap > 1e-3
    assert clear.float().mean() > 0.9
    assert torch.equal(got_ids[clear], want_ids[clear])
    # the whole FFN half against vLLM's, per token where both inputs pick the
    # same experts
    same = (_top6(ffn, xf)[0] == want_ids).all(1)
    assert same.float().mean() >= 0.9, same.float().mean()
    tok = torch.nn.functional.cosine_similarity(part.float(), out.float(), dim=1)
    assert tok[same].min() > 0.999, tok[same].min()


def test_k2_graph_replay(ffn):
    """Several widths back to back in one CUDA graph (scratch shared by the
    launches, as the layers share it): every replay bit for bit the eager
    run."""
    run, wt = ffn.run, ffn.wt
    ins = {m: _inputs(m, ffn.dev, seed=100 + m) for m in (17, 129, 256, 64)}
    outs: dict = {}

    def step():
        for m, a in ins.items():
            o = run(wt, *a)
            outs[m] = [*o[1:], *(t[: m * TOPK] if i < 2 else t[:m] for i, t in enumerate(run.comb_in))]

    step()
    torch.accelerator.synchronize()
    ref = {m: [t.clone() for t in o] for m, o in outs.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for i in range(REPLAYS):
        graph.replay()
        if i % 25 == 24:
            torch.accelerator.synchronize()
            for m in ins:
                bad = [j for j, (a, b) in enumerate(zip(outs[m], ref[m])) if not torch.equal(a, b)]
                assert not bad, (i, m, bad)
