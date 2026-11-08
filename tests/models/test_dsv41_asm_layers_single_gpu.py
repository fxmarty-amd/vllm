# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 asm decode layers end to end on ONE gfx950 GPU: the
engine at TP1 serving the five-layer model of ``test_dsv41_mono_layers.py``
with one TP4 rank's shard sizes (16 heads, 2 output groups, a 576-wide expert
intermediate), so every decoder layer holds exactly what a TP4 rank holds and
``AsmDecodeLayer.create`` / ``install`` / the seam hooks / the eager breaks /
graph capture run as in production, on vLLM's real attention metadata and KV
caches.

The engine core runs in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, as in
``vllm bench forward``): every request is added before the first step, so the
schedule, and with it each step's row count, is exact and reproducible.

The asm path is TP4 by design; test-only patches, applied around each engine
(in this process), run it on one rank:
- the TP4 gate of ``decode.enabled`` sees 4, and the mono decode layer
  (steps up to 16 rows, TP 2 / 4 / 8 only) is off;
- the two folded all-reduces are the 1-rank all-reduce: wo_b's plain GEMV
  (gemv8 WO_B1, the fused kernel's own GEMV) and the MoE combine in torch
  (comb_ar's documented combine), then the wire's rounding of the reduced
  value, as the owner rank sends it to every rank: none for the bf16 wire
  (stock vLLM's precision), an MXFP8 round trip (one E8M0 a 32 elements,
  vLLM's ceil rule) for the MXFP8 wire. Their 4-rank kernels are tested on 4
  GPUs (``test_dsv41_asm_allreduce.py``).
On TP1 the Engram side chain (TP4 only) stays vLLM's.

Weights: the mono test's random loader, then every quantized weight rewritten
with its scales as a gaussian of a fixed std quantized to its checkpoint
format (MXFP8: one E8M0 a 32 x 32 block; MXFP4: one E8M0 a 32, e2m1 codes;
ceil-rule scales, so no saturation and no NaN / Inf codes), before vLLM's
post-load processing (``process_weights_after_loading``, the MoE oracle's
backend and its shuffles). ``test_weights_are_gaussian`` checks them.

The reference is vLLM's path made deterministic and high precision
(``dsv41_reference_moe.py``): the MoE in fp64 torch on exactly dequantized
weights (AITER's fused MoE is not deterministic), the indexer top-k rows in a
canonical order and the unquantized GEMM past 128 rows in fp32. Against
itself it is exact (the mixed items' control).

Three kinds of checks, all eager unless stated:
- end to end (``test_end_to_end``): the logits of every decode-only step, the
  asm path as served against the reference, per request and step; errors
  compound through the five random layers, so the bound is loose;
- per layer, tighter: in every step of 17-256 rows, a hook runs the
  reference on clones of the layer's exact inputs (the layer's asm off for
  that call), then the asm forward runs as served, the reference's indexer
  top-k given to its attention; their outputs (attention half, residual
  streams, layer output) compare per row, so a layer's error is its own and a
  failure names the layer, the step kind and the row count;
- the indexer (``test_indexer_recall``): asm's own top-k and the reference's,
  each against the exact top-k (unquantized q and keys), on rows with more
  than 512 candidates.
Each layer is a different path (see LAYERS). The decode tokens are forced
(one allowed token a request), so every engine runs the same steps. The graph
engine's logits, per request and step, compare bitwise against the eager
engine's.

Decode workloads by context length (DECODE_CONTEXTS, as `vllm bench
forward`'s): "short" contexts keep every top-k trivial (fewer than 512
compressed entries), "long" ones (past 1024 tokens; chunked prefill, then
decode steps of each row count) make the indexer select.

Items: wire x [decode (short, long) | mixed] x layer x row count (eager),
wire x workload (end to end), wire x workload x index layer (recall), and
wire x decode x row count (graph). Mixed steps are not graph-captured (no
prefill capture sizes, as in ``deepseek_v41/amd_specialized_decode/test_dsv41_asm_layers.py``); their row counts
follow the token budget, so they are checked by ranges. One eager engine per
workload runs the reference once, then each wire; the graph engine is one
per wire (the wire is captured). The folded all-reduces with the real
collectives: ``test_dsv41_asm_allreduce.py`` (4 GPUs)."""

import os
import types

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader

from . import dsv41_reference_moe as reference
from .deepseek_v41.amd_specialized_decode.test_dsv41_asm_layers import (
    STEP_ROWS,
    AsmProbe,
    _prompts,
    _step_rows_max_tokens,
)
from .test_dsv41_mono_layers import MODEL, STEPS, TRUNCATED, RandomWeightLoader

# the mixed workload's scheduler budget (the 4-GPU file used to define it)
MIXED_BUDGET = 256

LOAD_FORMAT = "dsv41_asm_gauss"
# one TP4 rank's shard sizes, served at TP1
SHARD = {"num_attention_heads": 64 // 4, "o_groups": 8 // 4, "moe_intermediate_size": 2304 // 4}
# the weights' std per format (the mono loader's magnitudes: larger random
# layers make the five-layer stack chaotic)
SIGMA = {"mxfp8": 2.0**-10, "mxfp4": 2.0**-6}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
U8 = torch.uint8

# the path each layer takes (the install summary, alike at TP4)
LAYERS = {
    0: "swa-only, vLLM attention, K2",
    1: "ratio-2 index source, K1 (top-k width 2048), K2",
    2: "ratio-2 consumer + Engram, K1, K2",
    3: "ratio-1 index + candidate source, K1 (width 4096), K2",
    4: "ratio-1 reindex over 3's candidates, K1 (width 4096), K2",
}
ROWS_ON_ASM = (17, 256)  # the step sizes the asm path takes
# vLLM's path per compared layer step, on the same inputs (see run_pass)
REFERENCE_PASSES = ("reference", "repeat", "exact_ffn", "exact_attention", "stock")
# TEST_TMP_EXACT_ASM_ROUNDING=1: also the reference with asm's rounding rules
# ("asm_rounding": which vectors are quantized where, as K1 / K2 do), which asm
# should match closely
ASM_ROUNDING = os.environ.get("TEST_TMP_EXACT_ASM_ROUNDING") == "1"
if ASM_ROUNDING:
    REFERENCE_PASSES = (*REFERENCE_PASSES, "asm_rounding")
# the layers with an indexer of their own
INDEX_LAYERS = (1, 3, 4)
# asm's top-k recall (vs the exact top-k) at most this below the reference's
RECALL_MARGIN = 0.02
# the logits of the whole model, asm against the reference (median per step)
# measured: median 3.2-7.2 %, worst (request, step) 28 %
END_TO_END_TOLERANCE = {"bf16": 0.10, "mxfp8": 0.10}
END_TO_END_WORST = 0.35
DECODE_ROW_COUNTS = tuple(sorted(STEP_ROWS))
MIXED_ROW_RANGES = ((17, 64), (65, 128), (129, 256))
WIRES = ("bf16", "mxfp8")
# Per-layer checks, each half on the reference's input (the attention given
# the reference's top-k, the FFN given the reference's attention output), per
# row, relative L2 errors.
#
# Why asm and the reference are several % apart: both quantize the GEMMs'
# activations (MXFP8 / fp8), each ~4.7 % (FFN) / ~3.7 % (attention) from the
# same computation with exact activations; asm rounds other vectors (before the
# row's rstd, vLLM after), so the two roundings are independent and land about
# sqrt(2) x that apart. Stock vLLM shares the reference's rounding, hence its
# small distance. The checks that matter: asm and stock vLLM equally far from
# the exact-activation references (ACCURACY_RATIO), and asm vs the reference
# explained by independent roundings (INDEPENDENT_ROUNDING).
#
# stock vLLM against the reference (vLLM's own bounds): AITER's MoE vs the fp64
# reference with the same MXFP8 rule; the attention is the same code
# (measured max 3.4 % over the rows of 16 mixed contexts; 4.03 % on one row of the
# 6-context sample, layer 4, mixed 65-128 rows, p99 2.8 %: a tail of another
# sample, numerics unchanged; 4 % before)
STOCK_FFN_TOLERANCE = 0.045
STOCK_ATTENTION_TOLERANCE = 1e-3
# asm / stock, of their errors against the exact-activation references (median
# and max over the non-rerouted rows of an item); the MXFP8 wire adds its own
# rounding of the reduced outputs
# (measured: bf16 1.06 / 1.08, MXFP8 1.24 / 1.26)
ACCURACY_RATIO = {"bf16": 1.10, "mxfp8": 1.35}
ACCURACY_RATIO_MAX = {"bf16": 1.15, "mxfp8": 1.40}
# asm vs reference at most this x sqrt(asm_exact^2 + reference_exact^2) (median;
# measured 0.91-1.07)
INDEPENDENT_ROUNDING = 1.2
# sign agreement with the reference (elements >= 0.1 x the row RMS), per row
MIN_SIGN_AGREEMENT = {"attention": 0.99, "output": 0.99}
# a row whose FFN output is this far from the reference: rerouted (an expert
# pick flipped at a near tie of the routing scores)
OUTPUT_TOLERANCE = {"bf16": 0.09, "mxfp8": 0.10}
RESIDUAL_TOLERANCE = {"bf16": 1e-3, "mxfp8": 1e-3}  # the residual streams
# the layers whose halves the asm path takes at TP1 (attention, FFN): layer 0
# has no compressed attention; layer 2's attention follows the asm Engram side
# chain, which needs TP4 (at TP1 vLLM's Engram makes the seam call K1 declines)
ASM_HALVES = {0: (False, True), 1: (True, True), 2: (False, True), 3: (True, True), 4: (True, True)}
# a near tie of vLLM's 6th / 7th routing scores (1e-2 past layer 0): a row
# whose output is past OUTPUT_TOLERANCE must be one, at most
# MAX_REROUTED_FRACTION of the rows
ROUTING_TIE = {0: 5e-3}
MAX_REROUTED_FRACTION = 0.25
# graph against eager: bitwise, for the requests whose prefill (vLLM's path,
# torch.compile'd in the graph engine only) gave bitwise the same logits
MIN_BITWISE_PREFILL_FRACTION = 0.75
# and of the compared tokens, at most this fraction not bitwise, each within
# GRAPH_ROUNDING (relative L2 of the logits)
GRAPH_NOT_BITWISE_FRACTION = 0.25
GRAPH_ROUNDING = 2e-3


# ---------------------------------------------------------------- gaussian quantized weights


def _e8m0(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """Ceil rule: the smallest power of 2 bringing amax within qmax."""
    return torch.ceil(torch.log2(amax.clamp_min(2.0**-100) / qmax)).clamp(-127, 127)


@torch.no_grad()
def _gauss_mxfp8(w: torch.Tensor, ws: torch.Tensor, g) -> torch.Tensor:
    """Fill e4m3 ``w`` [.., n, k] and its E8M0 ``ws`` ([.., n, k / 32] per row, or
    [.., n / 32, k / 32]) with N(0, sigma) quantized by 32 x 32 blocks; → the
    dequantized values."""
    *b, n, k = w.shape
    # fp32 throughout (the loader runs under a bf16 default dtype, whose
    # rounding would push codes past 448: NaN)
    f32 = dict(device=w.device, dtype=torch.float32)
    x = torch.randn(w.shape, generator=g, **f32) * SIGMA["mxfp8"]
    blk = x.view(*b, n // 32, 32, k // 32, 32)
    e = _e8m0(blk.abs().amax(dim=(-3, -1)), 448.0)
    q = (blk / torch.exp2(e)[..., None, :, None]).view(w.shape).to(torch.float8_e4m3fn)
    w.copy_(q)
    s = (e + 127).to(U8)
    ws.view(U8).copy_(s.repeat_interleave(32, -2) if ws.shape[-2] == n else s)
    return q.float().view(*b, n // 32, 32, k // 32, 32) * torch.exp2(e)[..., None, :, None]


@torch.no_grad()
def _gauss_mxfp4(w: torch.Tensor, ws: torch.Tensor, rows: int, cols: int, g):
    """Fill packed e2m1 ``w`` [.., R, K / 2] (element 2i in the low nibble) and
    its E8M0 ``ws`` [.., R, K / 32]: rows < ``rows`` and columns < ``cols``
    with N(0, sigma), the rest zero (the loader's padding); → the dequantized
    real part."""
    wu, su = w.view(U8), ws.view(U8)
    wu.zero_()
    su.fill_(127)
    *b, _, _ = wu.shape
    f32 = dict(device=w.device, dtype=torch.float32)
    x = torch.randn(*b, rows, cols, generator=g, **f32) * SIGMA["mxfp4"]
    blk = x.view(*b, rows, cols // 32, 32)
    e = _e8m0(blk.abs().amax(-1), 6.0)
    v = (blk / torch.exp2(e)[..., None]).view(*b, rows, cols)
    grid = torch.tensor(E2M1, **f32)
    mag = torch.empty(v.shape, dtype=torch.long, device=w.device)
    for i in range(0, rows, 64):  # bounded temporaries
        mag[..., i : i + 64, :] = (
            (v[..., i : i + 64, :].abs()[..., None] - grid).abs().argmin(-1)
        )
    code = (mag | ((v < 0) & (mag > 0)).long() << 3).to(U8)
    wu[..., :rows, : cols // 2] = code[..., 0::2] | code[..., 1::2] << 4
    su[..., :rows, : cols // 32] = (e + 127).to(U8)
    deq = torch.where(code >= 8, -grid[mag], grid[mag])
    return deq.view(*b, rows, cols // 32, 32) * torch.exp2(e)[..., None]


def _stats(x: torch.Tensor) -> dict:
    x = x.double().reshape(-1)
    if x.numel() > 1 << 24:
        x = x[:: x.numel() // (1 << 24) + 1]
    mean, std = x.mean().item(), x.std().item()
    return dict(
        finite=bool(torch.isfinite(x).all()),
        mean=mean,
        std=std,
        kurtosis=(((x - mean) / std) ** 4).mean().item(),
    )


@register_model_loader(LOAD_FORMAT)
class GaussianWeightLoader(RandomWeightLoader):
    """The mono test's random weights, then every quantized weight (a weight
    with a ``weight_scale``) a gaussian quantized to its format, before
    post-load processing; the dequantized weights' statistics are kept for
    ``weight_stats``."""

    stats: dict = {}

    def load_weights(self, model, model_config) -> None:
        super().load_weights(model, model_config)
        inter = model_config.hf_text_config.moe_intermediate_size
        hidden = model_config.hf_text_config.hidden_size
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        g = torch.Generator(device=dev).manual_seed(4242)
        GaussianWeightLoader.stats = {}
        for name, m in model.named_modules():
            w13 = getattr(m, "w13_weight", None)
            if w13 is not None and w13.device.type != "meta":
                pad = w13.shape[1] // 2
                rows = min(inter, pad)  # other experts (DSpark's): their own
                for x in range(w13.shape[0]):  # per expert: bounded temporaries
                    parts = [
                        _gauss_mxfp4(
                            w13.data[x, h : h + pad],
                            m.w13_weight_scale.data[x, h : h + pad],
                            rows, hidden, g,
                        )  # fmt: skip
                        for h in (0, pad)
                    ]
                    parts.append(
                        _gauss_mxfp4(
                            m.w2_weight.data[x], m.w2_weight_scale.data[x], hidden, rows, g
                        )
                    )
                    if x == 0:
                        for t, p in zip(("w1", "w3", "w2"), parts):
                            self.stats[f"{name}.{t}[0]"] = _stats(p)
                continue
            w, ws = getattr(m, "weight", None), getattr(m, "weight_scale", None)
            if (
                isinstance(w, torch.Tensor)
                and isinstance(ws, torch.Tensor)
                and w.device.type != "meta"
                and w.dtype == torch.float8_e4m3fn
                and w.shape[-1] % 32 == 0
                and w.shape[-2] % 32 == 0
            ):
                self.stats[name] = _stats(_gauss_mxfp8(w.data, ws.data, g))
                self.stats[name]["layout"] = (
                    tuple(w.shape), str(ws.dtype), tuple(ws.shape)
                )
        # the reference MoE's raw codes, before the post-load shuffles
        reference.stash_raw(model)


# ---------------------------------------------------------------- one-rank patches

# the wire the one-rank all-reduces apply, set before each run (a graph engine
# captures the one set when it starts)
SERVED_WIRE = {"wire": "bf16"}


def _mxfp8_round_trip(values: torch.Tensor) -> torch.Tensor:
    """bf16 [rows, hidden] → bf16(dequant(MXFP8(values))), one E8M0 a 32
    elements by vLLM's ceil rule: the reduced value as the MXFP8 wire's owner
    rank sends it to every rank."""
    blocks = values.float().view(values.shape[0], -1, 32)
    scale = torch.exp2(_e8m0(blocks.abs().amax(-1, keepdim=True), 448.0))
    codes = (blocks / scale).to(torch.float8_e4m3fn)
    return (codes.float() * scale).view_as(values).to(torch.bfloat16)


def _through_wire(reduced: torch.Tensor) -> torch.Tensor:
    return _mxfp8_round_trip(reduced) if SERVED_WIRE["wire"] == "mxfp8" else reduced


class _OneRankWoB:
    """WoBAllReduce on one rank: wo_b's GEMV alone (gemv8 WO_B1, the GEMV the
    fused kernel runs), out = this rank's partial."""

    def __init__(self):
        from vllm.models.deepseek_v41.amd.asm.kernels import gemv8
        from vllm.models.deepseek_v41.amd.asm.kernels.kmod import Kernel

        d = gemv8.source(gemv8.WO_B1, "k")
        self.k, self.grid = Kernel(d["name"], d["src"], d["ka"]), gemv8.WO_B1.grid
        self.dummy = torch.zeros(64, dtype=U8, device="cuda")

    def __call__(self, weight, weight_scale, activation, activation_scale, num_rows: int):
        partial = torch.empty(num_rows, 5120, dtype=torch.bfloat16, device=activation.device)
        unused = self.dummy
        self.k(self.grid, 256, w=weight, ws=weight_scale, x=activation,
               xs=activation_scale, out=partial, outs=unused, part=unused,
               cnt=unused, cs=unused, pos=unused, dbg=unused, m=num_rows)  # fmt: skip
        return _through_wire(partial)

    def errors(self) -> int:
        return 0


class _OneRankComb:
    """CombAllReduce on one rank: comb_ar's combine without the reduce,
    bf16(bf16(Σ_k y[t·6 + k]) + add[t]), picks with ids < 0 reading 0."""

    k = types.SimpleNamespace(name="comb_one_rank_torch")

    def __call__(self, expert_rows, expert_ids, shared_output, num_rows: int):
        picks = expert_rows[: num_rows * 6].view(num_rows, 6, -1).float()
        picked = (expert_ids[: num_rows * 6] >= 0).view(num_rows, 6, 1)
        picks = torch.where(picked, picks, 0.0)
        routed = picks[:, 0]
        for pick in range(1, 6):
            routed = routed + picks[:, pick]
        combined = routed.to(torch.bfloat16).float() + shared_output.float()
        return _through_wire(combined.to(torch.bfloat16))

    def errors(self) -> int:
        return 0


class _AsmOnlyMono:
    """The mono decode layer's place at TP1: no mono kernels, the asm decode
    layer (``asm``, set by MonoDecodeLayer.create) for its steps, else None
    (vLLM's forward)."""

    ffn_only = False
    unfolded_entry = False

    def __init__(self):
        self.asm = None

    def __call__(self, layer, *args):
        return None if self.asm is None else self.asm(layer, *args)


def _asm_of(layer):
    """The layer's asm decode layer (under the mono decode layer), or None."""
    mono = getattr(layer, "mono", None)
    return None if mono is None else mono.asm


def _patch_one_rank():
    """Apply the one-rank patches; → a function that undoes them."""
    from vllm.models.deepseek_v41.amd.asm import decode
    from vllm.models.deepseek_v41.amd.mono_decode import MonoDecodeLayer

    originals = dict(
        world_size=decode.get_tensor_model_parallel_world_size,
        wob_setup=decode._wob_setup,
        comb_setup=decode._comb_setup,
        mono_create=MonoDecodeLayer.__dict__["_create"],
    )
    decode.get_tensor_model_parallel_world_size = lambda: 4  # enabled()'s gate

    def wob_setup():
        if decode._wob["wob"] is None:
            decode._wob["wob"] = _OneRankWoB()

    def comb_setup():
        if decode._comb["car"] is None:
            decode._comb["car"] = _OneRankComb()

    decode._wob_setup, decode._comb_setup = wob_setup, comb_setup
    # the mono decode layer (steps of up to 16 rows) needs TP 2, 4 or 8: off,
    # as with --monokernel off (those steps take vLLM's path on both sides);
    # MonoDecodeLayer.create still attaches the asm decode layer to what
    # _create returns, so a stand-in that only forwards to it
    MonoDecodeLayer._create = staticmethod(lambda layer, vllm_config: _AsmOnlyMono())

    def undo():
        decode.get_tensor_model_parallel_world_size = originals["world_size"]
        decode._wob_setup = originals["wob_setup"]
        decode._comb_setup = originals["comb_setup"]
        MonoDecodeLayer._create = originals["mono_create"]
        decode._wob["wob"] = decode._comb["car"] = None
        decode._cfg.clear()  # enabled() is decided once per process

    return undo


# ---------------------------------------------------------------- the probe


def _sign_agreement_per_row(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per row: the fraction of elements of the same sign, among the expected
    row's elements of at least 0.1 x its RMS (smaller ones flip under any
    rounding difference)."""
    actual, expected = actual.float().flatten(1), expected.float().flatten(1)
    rms = expected.pow(2).mean(1, keepdim=True).sqrt()
    kept = expected.abs() >= 0.1 * rms
    same = (torch.sign(actual) == torch.sign(expected)) & kept
    return same.sum(1) / kept.sum(1).clamp_min(1)


# layers whose exact-attention calls were printed (once each)
_EXACT_CHECK_PRINTED: set = set()


def _relative_error_per_row(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per row ([rows, ...]): relative L2 error of actual against expected."""
    actual, expected = actual.float().flatten(1), expected.float().flatten(1)
    return (actual - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(1e-30)


def _layer_caches(layer) -> list[torch.Tensor]:
    """Every state tensor the layer's modules hold (not their parameters):
    caches (SWA, compressed, compressor state, indexer), top-k and other
    persistent buffers, once each."""
    caches, seen = [], set()
    for module in layer.modules():
        parameters = {id(parameter) for parameter in module.parameters(recurse=False)}
        for value in vars(module).values():
            if isinstance(value, dict):
                values = list(value.values())
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]
            for tensor in values:
                if (
                    isinstance(tensor, torch.Tensor)
                    and not isinstance(tensor, torch.nn.Parameter)
                    and id(tensor) not in parameters
                    and tensor.is_cuda
                    and tensor.numel()
                    and tensor.data_ptr() not in seen
                ):
                    seen.add(tensor.data_ptr())
                    caches.append(tensor)
    return caches


def _argument(args, kwargs, position: int, name: str):
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


class OneRankProbe(AsmProbe):
    """Worker extension: each layer against vLLM's path on the same inputs
    (``compare_layers_to_vllm``), the sampler's logits (``capture_logits``),
    the results of a run (``results``)."""

    def install_reference(self) -> None:
        """The deterministic reference side: the fp64 MoE (the served path's
        MoE where vLLM's path runs, as AITER's is not deterministic) and the
        exact indexer's recorder."""
        model = self.model_runner.model  # type: ignore[attr-defined]
        self.undo_reference = [reference.install_reference_moe(model), reference.install_exact_attention(model)]
        self.indexer_reference = reference.IndexerReference()
        self.undo_reference.append(self.indexer_reference.install())
        self.comparing, self.inject = False, False

    def uninstall_reference(self) -> None:
        for undo in reversed(getattr(self, "undo_reference", [])):
            undo()
        self.undo_reference = []
        self.indexer_reference = None
        reference.clear()

    def set_modes(self, comparing: bool, inject: bool) -> None:
        """comparing: run each layer's reference pass and compare (else the
        layers just run as served); inject: the asm attention takes the
        reference's indexer top-k (the attention checked given the same
        selection; asm's own top-k is kept for the recall check)."""
        self.comparing, self.inject = comparing, inject

    def clear_results(self) -> None:
        self.layer_errors: dict = {}
        self.recalls: dict = {}
        self.logits_by_token: dict = {}
        # the prefill's logits (vLLM's path only), per prompt length
        self.prefill_logits: dict = {}
        self.steps_per_layer: dict = {}
        # (layer, "attention" | "ffn") -> asm calls that returned the half's output
        self.asm_calls: dict = {}

    def results(self) -> dict:
        return dict(
            layer_errors=self.layer_errors,
            logits_by_token=self.logits_by_token,
            prefill_logits=self.prefill_logits,
            recalls=getattr(self, "recalls", {}),
            asm_calls=self.asm_calls,
            weight_stats=dict(GaussianWeightLoader.stats),
            installed=self.asm_state(),
        )

    def count_asm_calls(self) -> None:
        """Count, per layer and half, the calls the asm path took (returned the
        half's output rather than falling back to vLLM's)."""
        if not hasattr(self, "asm_calls"):
            self.clear_results()

        def counted(layer_id, half, method):
            def call(*args, **kwargs):
                output = method(*args, **kwargs)
                if output is not None:
                    key = (layer_id, half)
                    self.asm_calls[key] = self.asm_calls.get(key, 0) + 1
                return output

            return call

        for layer_id, layer in self._decoder_layers().items():
            asm = _asm_of(layer)
            if asm is not None:
                asm.install()  # done by the profile run already; idempotent
                if asm.attn_fn is not None:
                    asm.attn_fn = counted(layer_id, "attention", asm.attn_fn)
                if asm.ffn_fn is not None:
                    asm.ffn_fn = counted(layer_id, "ffn", asm.ffn_fn)

    def asm_switch(self, on: bool) -> None:
        """The layers' asm path on (as built) or off (vLLM's path)."""
        for layer in self._decoder_layers().values():
            if layer.mono is None:
                continue
            if not hasattr(layer, "_asm_built"):
                layer._asm_built = layer.mono.asm
            layer.mono.asm = layer._asm_built if on else None

    def compare_layers_to_vllm(self) -> None:
        """Per decoder layer, in every step of ROWS_ON_ASM rows (while
        ``comparing``), the asm outputs against the reference (vLLM's path,
        made deterministic by install_reference). Per row, ``layer_errors[(layer,
        step, row)]``: the relative errors of the layer's output, residual
        streams and attention half, the step's kind and row count, the
        indexer top-k difference and the reference's routing gap (6th - 7th
        expert score); per step, the indexer top-k recalls (``recalls``).

        Same inputs: vLLM's path runs first on copies of the layer's inputs
        (the layer's asm off for those calls), once per REFERENCE_PASSES mode,
        with the layer's caches restored after each (a step with prefill
        chunks advances state that a second run would see: compressor
        windows), then the asm forward as served (so its KV writes are the ones
        kept); with ``inject``, the asm attention takes the reference's top-k."""
        from vllm.forward_context import get_forward_context

        self.clear_results()
        self.inside_reference = False
        self.pass_mode = None
        self.reference_outputs: dict = {}
        self.asm_topk: dict = {}  # asm's own top-k where the reference's was injected
        # the reference pass's attention half, fed to asm's and stock's FFN seam
        self.golden_attention: dict = {}
        self.routing_gaps: dict = {}
        # the (wire, layer, step kind, row count) whose reference already ran twice:
        # the determinism check ("repeat") runs once per such shape
        self.repeated: set = set()
        # the attention half's (all-reduced) output, as the FFN seam takes it
        self.attention_outputs: dict = {}

        def step_kind(layer):
            metadata = get_forward_context().attn_metadata
            if not isinstance(metadata, dict):
                return None
            swa = metadata.get(layer.attn.swa_cache_layer.prefix)
            if swa is None:
                return None
            return ("mixed" if swa.num_prefills else "decode"), swa.num_decode_tokens

        def run_vllm_path_first(layer_id, layer):
            def hook(module, args, kwargs):
                if not self.comparing or self.inside_reference or _asm_of(layer) is None:
                    return
                kind, num_rows = step_kind(layer), args[0].shape[0]
                if kind is None or not ROWS_ON_ASM[0] <= num_rows <= ROWS_ON_ASM[1]:
                    return

                def copy(value):
                    return value.clone() if isinstance(value, torch.Tensor) else value

                caches = _layer_caches(layer)
                snapshots = [cache.clone() for cache in caches]
                topk = getattr(layer.attn, "topk_indices_buffer", None)

                def run_pass(mode):
                    """vLLM's path on copies of the inputs, the caches restored:
                    "reference" / "repeat" (the deterministic reference, twice),
                    "exact_ffn" / "exact_attention" (the reference with that
                    half's activations unquantized; the attention given the
                    reference's top-k), "stock" (vLLM as served: AITER's MoE, no
                    determinism patches)."""
                    reference.ENABLED["on"] = mode != "stock"
                    # K1's rules only where asm runs K1 (layers 0 and 2 keep vLLM's attention)
                    reference.ASM_RULE.update(ffn=mode == "asm_rounding",
                                              attention=mode == "asm_rounding" and bool(ASM_HALVES[layer_id][0]))
                    reference.ACTIVATIONS["rule"] = "none" if mode == "exact_ffn" else "mxfp8"
                    reference.ACTIVATIONS["attention"] = "none" if mode == "exact_attention" else "mxfp8"
                    self.pass_mode = mode
                    self.indexer_reference.take_queries()
                    indexer = getattr(layer.attn, "indexer", None)
                    selected = passes.get("reference", {}).get("topk")
                    if mode in ("exact_attention", "stock", "asm_rounding") and self.inject and indexer is not None and selected is not None:
                        def same_selection(*_args, **_kwargs):
                            topk[: selected.shape[0]].copy_(selected)
                            return topk

                        indexer.indexer_op.forward = same_selection
                    reference.CALLS.clear()
                    try:
                        outputs = layer(*[copy(v) for v in args], **{n: copy(v) for n, v in kwargs.items()})
                        if mode == "exact_attention":
                            # the exact-activation attention really ran (no silent
                            # fallback to the served quantized GEMMs)
                            for name in ("fused_wqa_wkv", "norms", "wq_b", "wo_a+wo_b"):
                                assert reference.CALLS.get(name, 0) > 0, (layer_id, name, dict(reference.CALLS))
                            assert max(reference.GEMM_CHECK.values()) < 1e-5, reference.GEMM_CHECK
                            if layer_id not in _EXACT_CHECK_PRINTED:
                                _EXACT_CHECK_PRINTED.add(layer_id)
                                print(f"EXACT-ATTENTION layer {layer_id}: calls {dict(reference.CALLS)}, "
                                      f"fp32 GEMMs vs fp64 worst {max(reference.GEMM_CHECK.values()):.2e} "
                                      f"over {len(reference.GEMM_CHECK)} weights")
                        found = dict(
                            # the indexer's q quantization(s), for the exact top-k
                            queries=self.indexer_reference.take_queries(),
                            topk=topk[:num_rows].clone()  # before the restore
                            if topk is not None and layer.attn.indexer is not None else None,
                            output=outputs[0].clone(),
                            residual=outputs[1].clone(),
                            attention=self.attention_outputs.pop(layer_id),
                        )
                    finally:
                        reference.ENABLED["on"] = True
                        reference.ACTIVATIONS.update(rule="mxfp8", attention="mxfp8")
                        reference.ASM_RULE.update(ffn=False, attention=False)
                        reference.ASM_CONTEXT.clear()
                        if indexer is not None:
                            indexer.indexer_op.__dict__.pop("forward", None)
                        for cache, snapshot in zip(caches, snapshots):
                            cache.copy_(snapshot)
                    return found

                asm_path = layer.mono.asm
                layer.mono.asm = None
                self.inside_reference = True
                passes: dict = {}
                try:
                    for mode in REFERENCE_PASSES:
                        if mode == "repeat":
                            shape = (SERVED_WIRE["wire"], layer_id, kind[0], num_rows)
                            if shape in self.repeated:
                                continue
                            self.repeated.add(shape)
                        passes[mode] = run_pass(mode)
                        if mode == "reference":
                            self.golden_attention[layer_id] = passes[mode]["attention"]
                finally:
                    layer.mono.asm = asm_path
                    self.inside_reference = False
                    self.pass_mode = None
                self.reference_outputs[layer_id] = dict(
                    passes=passes,
                    queries=passes["reference"]["queries"],
                    topk=passes["reference"]["topk"],
                    output=passes["reference"]["output"],
                    residual=passes["reference"]["residual"],
                    attention=passes["reference"]["attention"],
                    kind=kind[0],
                    num_decode_rows=kind[1],
                    num_rows=num_rows,
                )

            return hook

        def store_errors(layer_id, asm, vllm, step_info, topk_differences=None):
            """Per row: asm and stock vLLM against the reference (relative L2
            error and sign agreement) for each half and the residual streams;
            the FFN half of asm, stock and the reference against the exact-
            activation reference; the reference against itself."""
            num_rows = step_info["num_rows"]
            passes = vllm["passes"]
            sides = {"": asm, "stock_": passes["stock"]}
            if "repeat" in passes:  # once per shape
                sides["repeat_"] = passes["repeat"]
            per_row = {}
            for prefix, side in sides.items():
                for name in ("output", "residual", "attention"):
                    per_row[prefix + name] = _relative_error_per_row(side[name], vllm[name]).tolist()
                for name in ("output", "attention"):
                    per_row[prefix + "sign_" + name] = _sign_agreement_per_row(side[name], vllm[name]).tolist()
            if "asm_rounding" in passes:
                # asm against the reference with asm's rounding rules
                rounding = passes["asm_rounding"]
                for name in ("output", "attention"):
                    per_row["asm_rounding_" + name] = _relative_error_per_row(asm[name], rounding[name]).tolist()
                    per_row["rounding_ref_" + name] = _relative_error_per_row(rounding[name], vllm[name]).tolist()
                per_row["rounding_exact_output"] = _relative_error_per_row(
                    rounding["output"], passes["exact_ffn"]["output"]).tolist()
                per_row["rounding_exact_attention"] = _relative_error_per_row(
                    rounding["attention"], passes["exact_attention"]["attention"]).tolist()
            for prefix, side in (("", asm), ("stock_", passes["stock"]), ("reference_", vllm)):
                per_row[prefix + "exact_output"] = _relative_error_per_row(
                    side["output"], passes["exact_ffn"]["output"]
                ).tolist()
                per_row[prefix + "exact_attention"] = _relative_error_per_row(
                    side["attention"], passes["exact_attention"]["attention"]
                ).tolist()
            gaps = self.routing_gaps.pop(layer_id, None)
            gaps = gaps.tolist() if gaps is not None else [float("inf")] * num_rows
            step = self.steps_per_layer[layer_id] = self.steps_per_layer.get(layer_id, -1) + 1
            for row in range(num_rows):
                self.layer_errors[(layer_id, step, row)] = dict(
                    kind=step_info["kind"],
                    num_rows=num_rows,
                    decode_row=row < step_info["num_decode_rows"],
                    topk_difference=topk_differences[row] if topk_differences else None,
                    routing_gap=gaps[row],
                    **{name: values[row] for name, values in per_row.items()},
                )

        def compare_asm_outputs(layer_id):
            def hook(module, args, kwargs, outputs):
                if self.inside_reference or not self.comparing:
                    return
                if layer_id not in self.reference_outputs:
                    return
                reference = self.reference_outputs.pop(layer_id)
                self.golden_attention.pop(layer_id, None)
                asm = dict(
                    output=outputs[0],
                    residual=outputs[1],
                    attention=self.attention_outputs.pop(layer_id),
                )
                topk_differences = None
                # asm's own top-k (the buffer, unless the reference's was injected)
                asm_topk = self.asm_topk.pop(layer_id, None)
                if asm_topk is None and reference["topk"] is not None:
                    asm_topk = module.attn.topk_indices_buffer[: reference["num_rows"]].clone()
                if reference["topk"] is not None:
                    # the indexer's top-k sets: entries in one and not the other
                    topk_differences = [
                        len(set(a[a >= 0].tolist()) ^ set(b[b >= 0].tolist()))
                        for a, b in zip(asm_topk.cpu(), reference["topk"].cpu())
                    ]
                indexer = getattr(module.attn, "indexer", None)
                if indexer is not None and reference["topk"] is not None and reference["queries"]:
                    # each selection against the exact (unquantized) indexer's top-k
                    from vllm.forward_context import get_forward_context

                    metadata = get_forward_context().attn_metadata[indexer.k_cache.prefix]
                    found = self.indexer_reference.recall(
                        indexer.k_cache.kv_cache, metadata, reference["queries"][-1],
                        {"asm": asm_topk, "vllm": reference["topk"]}, reference["topk"].shape[1],
                    )
                    bucket = self.recalls.setdefault((layer_id, reference["kind"], reference["num_rows"]), {})
                    for name, values in found.items():
                        bucket.setdefault(name, []).extend(values)
                store_errors(layer_id, asm, reference, reference, topk_differences=topk_differences)

            return hook

        def record_routing_gap(layer_id):
            def hook(ffn, args, kwargs):
                if self.pass_mode != "reference":
                    return
                hidden = _argument(args, kwargs, 0, "hidden_states").float()
                gate = ffn.gate
                scores = torch.nn.functional.softplus(hidden @ gate.weight.float().T).sqrt()
                scores = scores + gate.e_score_correction_bias.float()
                top7 = scores.topk(7, dim=-1).values
                self.routing_gaps[layer_id] = top7[:, 5] - top7[:, 6]

            return hook

        def inject_topk(layer_id, layer, tk_hook):
            """After an index layer's top-k (asm's or vLLM's op), before the
            attention: in inject mode, keep asm's selection and put the
            reference pass's in its place (asm's lists rebuilt from it)."""
            from vllm.models.deepseek_v41.amd.asm import decode

            def hook():
                pending = self.reference_outputs.get(layer_id)
                if self.comparing and self.inject and pending is not None and pending["topk"] is not None:
                    selected = pending["topk"]
                    buffer = layer.attn.topk_indices_buffer
                    self.asm_topk[layer_id] = buffer[: selected.shape[0]].clone()
                    buffer[: selected.shape[0]].copy_(selected)
                    decode._k1.pop("tk_done", None)  # tk_hook translates the buffer
                return tk_hook()

            return hook

        def capture_attention_output(layer_id, layer):
            """Wrap the layer's seams (mhc_pre_delayed's calls: vLLM's path
            directly, asm's through AsmDecodeLayer._pre, whose fallback calls it
            again: captured once), args = (residual, hc_fn, ...)."""
            # vLLM's seam (the reference passes): the layer's own MHCPreDelayedOp
            # module, wrapped through its instance forward
            seam = layer.mhc_pre_delayed.forward
            busy = [False]

            def capture(run_seam):
                def seam_with_capture(*args, **kwargs):
                    if busy[0]:
                        return run_seam(*args, **kwargs)
                    busy[0] = True
                    try:
                        return capture_once(run_seam, *args, **kwargs)
                    finally:
                        busy[0] = False

                return seam_with_capture

            def capture_once(run_seam, *args, **kwargs):
                if self.comparing and len(args) > 1 and args[1] is layer.hc_ffn_fn:  # the FFN seam
                    self.attention_outputs[layer_id] = kwargs["sublayer_out"].clone()
                    golden = self.golden_attention.get(layer_id)
                    if self.inject and golden is not None and self.pass_mode in (None, "stock", "asm_rounding"):
                        # the FFN half on the reference's input (asm, stock, asm rounding)
                        kwargs = dict(kwargs, sublayer_out=golden.clone())
                    if self.pass_mode == "asm_rounding":
                        # K2's input: y' = bf16(x1 * ffn_norm.w) and rstd, x1 the
                        # seam's collapse (the seam again, without its norm)
                        x1 = seam(*args, **dict(kwargs, norm_weight=None))[3].float()
                        rstd = torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + kwargs["norm_eps"])
                        y_prime = (x1 * layer.ffn_norm.weight.float()).to(torch.bfloat16)
                        reference.ASM_CONTEXT["ffn"] = (y_prime, rstd)
                if (self.comparing and self.pass_mode == "asm_rounding" and len(args) > 1
                        and args[1] is layer.hc_attn_fn and "norm_eps" in kwargs):
                    # K1's input, likewise: y' = bf16(x1 * attn_norm.w) and rstd
                    x1 = seam(*args, **dict(kwargs, norm_weight=None))[3].float()
                    rstd = torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + kwargs["norm_eps"])
                    y_prime = (x1 * layer.attn_norm.weight.float()).to(torch.bfloat16)
                    reference.ASM_CONTEXT["attention"] = (y_prime, rstd)
                return run_seam(*args, **kwargs)

            layer.mhc_pre_delayed.forward = capture(seam)
            asm = _asm_of(layer)
            if asm is not None:
                asm_pre = asm._pre
                asm._pre = lambda lyr, *a, **k: capture(lambda *a2, **k2: asm_pre(lyr, *a2, **k2))(*a, **k)

        for layer_id, layer in self._decoder_layers().items():
            capture_attention_output(layer_id, layer)
            asm = _asm_of(layer)
            if asm is not None and getattr(asm, "tk_hook", None) is not None:
                asm.tk_hook = inject_topk(layer_id, layer, asm.tk_hook)
            layer.register_forward_pre_hook(run_vllm_path_first(layer_id, layer), with_kwargs=True)
            layer.register_forward_hook(compare_asm_outputs(layer_id), with_kwargs=True)
            layer.ffn.register_forward_pre_hook(record_routing_gap(layer_id), with_kwargs=True)

    def capture_logits(self, prompt_lengths: list[int]) -> None:
        """The logits the sampler takes in decode-only steps of ROWS_ON_ASM
        rows, ``logits_by_token[(prompt length, position)] = (step rows,
        logits)`` (taken outside the model's CUDA graph: the graph engine's
        too)."""
        prompt_lengths = sorted(prompt_lengths)
        if not hasattr(self, "layer_errors"):
            self.clear_results()
        runner = self.model_runner  # type: ignore[attr-defined]
        sample = runner.sample

        def sample_and_capture(hidden_states, input_batch, *args, **kwargs):
            num_rows = input_batch.num_tokens
            if input_batch.has_prefill:
                sampled_rows = input_batch.logits_indices
                logits = runner.model_state.compute_sample_logits(
                    hidden_states[sampled_rows]
                ).float()
                for row, position in enumerate(input_batch.positions[sampled_rows].tolist()):
                    if position + 1 in prompt_lengths:  # a prompt's last token
                        self.prefill_logits.setdefault(position + 1, logits[row].cpu())
            if not input_batch.has_prefill and ROWS_ON_ASM[0] <= num_rows <= ROWS_ON_ASM[1]:
                sampled_rows = input_batch.logits_indices
                logits = runner.model_state.compute_sample_logits(
                    hidden_states[sampled_rows]
                ).float()
                positions = input_batch.positions[sampled_rows].tolist()
                for row, position in enumerate(positions):
                    shorter = [length for length in prompt_lengths if length <= position]
                    if not shorter:  # a request outside the workload's prompts
                        continue
                    prompt_length = max(shorter)
                    self.logits_by_token.setdefault(
                        (prompt_length, position), (num_rows, logits[row].cpu())
                    )
            return sample(hidden_states, input_batch, *args, **kwargs)

        runner.sample = sample_and_capture


# ---------------------------------------------------------------- runs


def _skip_unless_gfx950() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the asm decode kernels target gfx950")
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.asm.kernels.build import toolchain_missing

    if missing := toolchain_missing():
        pytest.skip(f"needs {missing} to build the kernels")


# mixed workload: requests over long contexts (past both compression ratios'
# 512 entries), then one joining request at a time whose whole prompt fits the
# next step, next to the decode rows: mixed steps of exactly 6 + 24 = 30,
# 7 + 100 = 107 and 8 + 200 = 208 rows
# (6 since the < 4 min target, one mixed step per MIXED_ROW_RANGES bucket;
# 16 before)
MIXED_CONTEXTS = 6
# decode workloads by context length (as `vllm bench forward`'s): "short" (128-345
# tokens) keeps every indexer top-k trivial (fewer than 512 compressed entries);
# "long" (past 1024 tokens at ratio 2) makes it select
DECODE_CONTEXTS = {"short": 128, "long": int(os.environ.get("DSV41_ASM_TEST_LONG_CONTEXT", "1536"))}
DECODE_WORKLOADS = {"short": "decode", "long": "decode-long"}
# tokens a step for the long prefills (steps past 256 rows: vLLM's path, not
# compared; 32768 since the < 4 min target, 8192 before)
LONG_PREFILL_BUDGET = 32768
JOINING_PROMPT_LENGTHS = (24, 100, 200)


def _workload(name: str):
    """(prompts, max tokens, token budget): "decode": decode-only steps of
    every STEP_ROWS row count; "mixed": the long-context requests of
    ``_run_mixed_schedule``."""
    if name == "decode":
        max_tokens = _step_rows_max_tokens()
        prompts = _prompts(len(max_tokens) // 8, 128, len(STEP_ROWS) + 1, copies=8)
        return prompts, max_tokens, None
    if name == "decode-long":
        rows = max(STEP_ROWS)
        prompts = _prompts(rows // 8, DECODE_CONTEXTS["long"], len(STEP_ROWS) + 1, copies=8)
        return prompts, [4096] * len(prompts), LONG_PREFILL_BUDGET
    prompts = _prompts(MIXED_CONTEXTS, 1100, STEPS + 1)
    return prompts, [1024] * len(prompts), MIXED_BUDGET


def _forced(prompt: list[int], max_tokens: int):
    """Greedy sampling forced to one token (the prompt's first), so every
    engine runs the same steps whatever its numerics."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True, allowed_token_ids=[prompt[0]]
    )


def _run_long_decode_schedule(llm, prompts, label: str) -> None:
    """Step the in-process engine by hand: every request's (chunked) prefill
    until all of them decode, then decode-only steps of each STEP_ROWS row
    count (two each), aborting requests down to it. Aborts everything at the
    end."""
    from vllm.inputs import TokensPrompt

    engine = llm.llm_engine
    request_ids, decoding = [], set()
    for index, prompt in enumerate(prompts):
        request_ids.append(f"{label}-{index}")
        engine.add_request(request_ids[-1], TokensPrompt(prompt_token_ids=prompt), _forced(prompt, 1024))

    def step():
        for output in engine.step():
            if output.outputs and output.outputs[0].token_ids:
                decoding.add(output.request_id)

    while len(decoding) < len(prompts):
        step()
    active = list(request_ids)
    for rows in sorted(STEP_ROWS, reverse=True):
        engine.abort_request(active[rows:])
        active = active[:rows]
        step()
        step()
    engine.abort_request(active)


def _run_mixed_schedule(llm, prompts, max_tokens, label: str) -> None:
    """Step the in-process engine by hand: the long-context requests (their
    chunked prefill takes mixed steps of the whole budget) until all of them
    decode, then each joining request: one mixed step of the decode rows and
    its whole prompt, then a decode-only step. Aborts everything at the end."""
    from vllm.inputs import TokensPrompt

    engine = llm.llm_engine
    request_ids, decoding = [], set()

    def add(name, prompt, tokens):
        request_ids.append(f"{label}-{name}")
        engine.add_request(request_ids[-1], TokensPrompt(prompt_token_ids=prompt), _forced(prompt, tokens))

    def step():
        for output in engine.step():
            if output.outputs and output.outputs[0].token_ids:
                decoding.add(output.request_id)

    for index, (prompt, tokens) in enumerate(zip(prompts, max_tokens)):
        add(f"context-{index}", prompt, tokens)
    while len(decoding) < len(prompts):
        step()
    step()  # decode-only: every context request
    generator = torch.Generator().manual_seed(5)
    for index, length in enumerate(JOINING_PROMPT_LENGTHS):
        prompt = torch.randint(1000, 60000, (length,), generator=generator).tolist()
        add(f"joining-{index}", prompt, 1024)
        step()  # mixed: the decode rows and the whole new prompt
        step()  # decode-only: the new request decodes too
    engine.abort_request(request_ids)


def _engine(vllm_runner, prompts, budget, graph: bool):
    graphs = (
        dict(
            enforce_eager=False,
            compilation_config={
                "decode_cudagraph_capture_request_sizes": sorted(STEP_ROWS),
                "prefill_cudagraph_capture_token_sizes": [],
            },
        )
        if graph
        else dict(enforce_eager=True)
    )
    return vllm_runner(
        MODEL,
        tensor_parallel_size=1,
        load_format=LOAD_FORMAT,
        hf_overrides={**TRUNCATED, **SHARD},
        worker_extension_cls=f"{__name__}.OneRankProbe",
        moe_backend="aiter",
        monokernel=True,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        engram_config={"cpu_offload": True},
        block_size=128,
        max_model_len=4096,
        max_num_seqs=len(prompts) + len(JOINING_PROMPT_LENGTHS),
        max_num_batched_tokens=budget or sum(len(prompt) for prompt in prompts),
        enable_chunked_prefill=budget is not None,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        # a small KV pool: the comparison snapshots each layer's caches
        kv_cache_memory_bytes=4 << 30,
        **graphs,
    )


def _generate(llm, prompts, max_tokens):
    """All the requests at once, with forced decode tokens (``_forced``)."""
    from vllm.inputs import TokensPrompt

    params = [_forced(prompt, tokens) for prompt, tokens in zip(prompts, max_tokens)]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts], params
    )
    for prompt, output in zip(prompts, outputs):
        assert set(output.outputs[0].token_ids) == {prompt[0]}
    return outputs


def _run_engine(vllm_runner, workload: str, wires: tuple[str, ...], graph: bool) -> dict:
    """One engine on ``workload``, run once per wire → {wire: results}. Eager:
    each layer against vLLM's path, and the logits; graph: the logits."""
    prompts, max_tokens, budget = _workload(workload)
    prompt_lengths = [len(prompt) for prompt in prompts]
    from vllm._aiter_ops import rocm_aiter_ops

    results_by_wire = {}
    with pytest.MonkeyPatch.context() as env:
        for name in (
            "VLLM_ROCM_USE_AITER",
            "VLLM_ROCM_USE_AITER_MOE",
            "VLLM_USE_BREAKABLE_CUDAGRAPH",
        ):
            env.setenv(name, "1")
        # the engine core in-process: every request is added before the first
        # step, so the schedule is exact (as in `vllm bench forward`)
        env.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # in this process, AITER's flags were read at import: refresh them
        rocm_aiter_ops.refresh_env_variables()
        undo_patches = _patch_one_rank()
        # the reference side deterministic: canonical top-k order, fp32 GEMM
        # past 128 rows (before the engine: linears bind their GEMM)
        undo_determinism = reference.deterministic_patches()
        SERVED_WIRE["wire"] = wires[0]  # a graph engine captures it at start
        try:
            with _engine(vllm_runner, prompts, budget, graph) as runner:
                call = runner.llm.collective_rpc
                call("install_reference")
                call("count_asm_calls")
                if not graph:
                    call("compare_layers_to_vllm")
                call("capture_logits", args=(prompt_lengths,))

                def run_workload(label):
                    if workload == "mixed":
                        _run_mixed_schedule(runner.llm, prompts, max_tokens, label)
                    elif workload == "decode-long":
                        _run_long_decode_schedule(runner.llm, prompts, label)
                    else:
                        _generate(runner.llm, prompts, max_tokens)

                reference_logits = stock_logits = None
                if not graph:
                    # end to end, the reference: vLLM's (deterministic) path
                    call("asm_switch", args=(False,))
                    call("set_modes", args=(False, False))
                    call("clear_results")
                    run_workload("reference")
                    reference_logits = call("results")[0]["logits_by_token"]
                    # and vLLM as served (AITER's MoE, no determinism patches)
                    reference.ENABLED["on"] = False
                    try:
                        call("clear_results")
                        run_workload("stock")
                        stock_logits = call("results")[0]["logits_by_token"]
                    finally:
                        reference.ENABLED["on"] = True
                    call("asm_switch", args=(True,))
                for wire in wires:
                    SERVED_WIRE["wire"] = wire
                    asm_logits = served = None
                    if not graph:
                        # end to end, the asm path as served
                        call("set_modes", args=(False, False))
                        call("clear_results")
                        run_workload(f"{wire}-asm")
                        served = call("results")[0]
                        asm_logits = served["logits_by_token"]
                        # per layer: replays on the reference's inputs, the
                        # reference's top-k injected into asm's attention
                        call("set_modes", args=(True, True))
                    call("clear_results")
                    run_workload(wire)
                    results_by_wire[wire] = call("results")[0]
                    results_by_wire[wire]["end_to_end"] = dict(
                        asm=asm_logits, reference=reference_logits, stock=stock_logits
                    )
                    # the eager asm path as served (no replays, no injected top-k)
                    results_by_wire[wire]["served"] = served
                call("uninstall_reference")
        finally:
            reference.clear()
            undo_determinism()
            undo_patches()
            # vLLM's layerwise reload info (a WeakKeyDictionary whose values hold
            # the layers' parameters, whose weight loaders hold the layers: the
            # keys) keeps every engine's weights alive in this process (~10 GiB
            # each here); drop it so the next engine finds the GPU free
            from vllm.model_executor.model_loader.reload import layerwise

            layerwise.LAYERWISE_INFO.clear()
            SERVED_WIRE["wire"] = "bf16"
    rocm_aiter_ops.refresh_env_variables()
    return results_by_wire


@pytest.fixture(scope="module")
def runs(vllm_runner):
    """runs(workload, wire, graph=False) → the results of that run. Engines
    start on first use: one eager engine per workload (it runs every wire),
    one graph engine per wire. A failed engine is not started again."""
    engines: dict = {}

    def get(workload: str, wire: str, graph: bool = False) -> dict:
        engine = (workload, "graph", wire) if graph else (workload, "eager")
        if engine not in engines:
            _skip_unless_gfx950()
            try:
                engines[engine] = _run_engine(
                    vllm_runner, workload, (wire,) if graph else WIRES, graph
                )
            except BaseException as error:
                engines[engine] = error
                raise
        if isinstance(engines[engine], BaseException):
            pytest.fail(f"the {engine} engine failed: {engines[engine]!r}")
        return dict(engines[engine][wire])

    return get


def _rows_of(results: dict, layer: int, kind: str, row_range: tuple[int, int]) -> list[dict]:
    """The rows of the ``kind`` steps of ``row_range`` rows at ``layer``."""
    lowest, highest = row_range
    rows = [
        row
        for (layer_id, _, _), row in results["layer_errors"].items()
        if layer_id == layer and row["kind"] == kind and lowest <= row["num_rows"] <= highest
    ]
    seen = sorted({(row["kind"], row["num_rows"]) for row in results["layer_errors"].values()})
    assert rows, f"no {kind} step of {lowest}..{highest} rows reached layer {layer}: {seen}"
    return rows


# TEST_TMP_EXACT_ASM_ROUNDING: asm against the reference with asm's rounding
# rules, relative L2 error per row (MISMATCH_INVESTIGATION.md section 8.5).
# bf16 wire only: the asm-rounding reference does not emulate the MXFP8 wire's
# rounding of the reduced outputs (2.7 % on the FFN half, one MXFP8 rounding).
# FFN half: median measured <= 0.23 %; max up to 2.9 % on rows near routing ties
# below the reroute threshold.
ASM_ROUNDING_FFN = (0.004, 0.04)  # (median, max)
# attention half on K1 layers: decode steps keep K1's attention kernel against
# vLLM's (measured median <= 3.35 %, max 3.9 %); mixed steps run vLLM's kernel on
# inputs matching K1's to rounding ties (median <= 1.82 %, max 2.8 %)
ASM_ROUNDING_ATTENTION = {"decode": (0.037, 0.045), "mixed": (0.022, 0.034)}
# the asm-rounding reference as accurate as stock vLLM against the exact-
# activation references (measured: at most 2.1 % (attention), 5.5 % (FFN)
# above stock's error, median rows)
ASM_ROUNDING_ACCURACY = 1.10


def _check_asm_rounding(rows: list[dict], wire: str, layer: int, label: str) -> None:
    def median(name, kept=rows):
        values = sorted(row[name] for row in kept)
        return values[len(values) // 2]

    for half, stock, own in (("attention", "stock_exact_attention", "rounding_exact_attention"),
                             ("FFN", "stock_exact_output", "rounding_exact_output")):
        assert median(own) <= ASM_ROUNDING_ACCURACY * median(stock), (
            label, half, "asm-rounding reference less accurate than stock", median(own), median(stock))
    if wire != "bf16":
        return
    bound, worst = ASM_ROUNDING_FFN
    assert median("asm_rounding_output") < bound, (label, "FFN asm vs asm-rounding", median("asm_rounding_output"))
    assert max(row["asm_rounding_output"] for row in rows) < worst, (label, "FFN asm vs asm-rounding max")
    if not ASM_HALVES[layer][0]:
        return  # vLLM's attention: covered by the reference checks
    for kind, (bound, worst) in ASM_ROUNDING_ATTENTION.items():
        kept = [row for row in rows if row["kind"] == kind]
        if kept:
            assert median("asm_rounding_attention", kept) < bound, (
                label, kind, "attention asm vs asm-rounding", median("asm_rounding_attention", kept))
            assert max(row["asm_rounding_attention"] for row in kept) < worst, (
                label, kind, "attention asm vs asm-rounding max")


def _check_rows(rows: list[dict], wire: str, layer: int, label: str) -> None:
    """The reference repeats exactly; stock vLLM within its own bounds of the
    reference (STOCK_*); asm as accurate as stock vLLM against the
    exact-activation references (ACCURACY_RATIO*), and its distance to the
    reference that of independent roundings (INDEPENDENT_ROUNDING); sign
    agreement (MIN_SIGN_AGREEMENT); residual streams within
    RESIDUAL_TOLERANCE; rows whose FFN output is past OUTPUT_TOLERANCE only at
    near ties of the routing scores (rerouted), at most MAX_REROUTED_FRACTION
    of the rows."""
    rerouted = [row for row in rows if row["output"] >= OUTPUT_TOLERANCE[wire]]
    worst_residual = max(row["residual"] for row in rows)
    print(f"{label}: {len(rows)} rows ({sum(r['decode_row'] for r in rows)} decode), rerouted {len(rerouted)}")
    for metric, title in (
        ("attention", "attention half vs reference"),
        ("output", "FFN half vs reference"),
        ("exact_output", "FFN half vs exact-activation reference"),
        ("exact_attention", "attention half vs exact-activation ref."),
        ("residual", "residual vs reference"),
        ("sign_attention", "attention sign agreement"),
        ("sign_output", "FFN sign agreement"),
    ):
        line = []
        for side in ("stock_", "", "reference_"):
            values = sorted(row[side + metric] for row in rows if side + metric in row)
            if values:
                p99 = values[min(len(values) - 1, int(0.99 * len(values)))]
                low = metric.startswith("sign")
                line.append(
                    f"{({'stock_': 'stock', '': 'asm', 'reference_': 'ref'})[side]} median "
                    f"{values[len(values) // 2]:.4f} {'p01' if low else 'p99'} "
                    f"{values[len(values) - 1 - int(0.99 * len(values))] if low else p99:.4f} "
                    f"{'min' if low else 'max'} {values[0] if low else values[-1]:.4f}"
                )
        print(f"{label}:   {title:40s} " + " | ".join(line))
    if "asm_rounding_output" in rows[0]:
        kept_rows = [row for row in rows if row not in rerouted]
        line = []
        for metric, title in (
            ("asm_rounding_attention", "attention asm vs asm-rounding ref"),
            ("asm_rounding_output", "FFN asm vs asm-rounding ref"),
            ("rounding_ref_attention", "attention asm-rounding ref vs ref"),
            ("rounding_ref_output", "FFN asm-rounding ref vs ref"),
            ("rounding_exact_attention", "attention asm-rounding ref vs exact"),
            ("rounding_exact_output", "FFN asm-rounding ref vs exact"),
        ):
            values = sorted(row[metric] for row in kept_rows)
            line.append(f"{title} median {100 * values[len(values) // 2]:.2f} % max {100 * values[-1]:.2f} %")
        print(f"{label}: ASM-ROUNDING " + " | ".join(line))
        _check_asm_rounding(kept_rows, wire, layer, label)
    # the reference against itself (the layer run twice on the same inputs): exact
    # (once per layer, step kind and row count: the rows of those steps)
    repeated = [row for row in rows if "repeat_output" in row]
    assert repeated, (label, "no reference repeat")
    for name in ("output", "residual", "attention"):
        assert max(row["repeat_" + name] for row in repeated) == 0.0, (label, "reference repeat", name)
    differences = sorted(row["topk_difference"] for row in rows if row.get("topk_difference") is not None)
    if differences:
        print(
            f"{label}: indexer top-k entries differing per row: median "
            f"{differences[len(differences) // 2]} max {differences[-1]}, rows identical "
            f"{sum(d == 0 for d in differences)}/{len(differences)}"
        )
    for decode_row in (True, False):
        split = sorted(row["attention"] for row in rows if row["decode_row"] == decode_row)
        topk = sorted(  # asm's own top-k against the reference's, per row kind
            row["topk_difference"] for row in rows
            if row["decode_row"] == decode_row and row.get("topk_difference") is not None
        )
        if split:
            print(
                f"{label}: {'decode' if decode_row else 'prefill'} rows {len(split)}, attention "
                f"error median {split[len(split) // 2]:.4f} max {split[-1]:.4f}"
                + (f", top-k differing median {topk[len(topk) // 2]} max {topk[-1]}, identical "
                   f"{sum(d == 0 for d in topk)}/{len(topk)}" if topk else "")
            )
    assert worst_residual < RESIDUAL_TOLERANCE[wire], (label, worst_residual)
    kept = [row for row in rows if row not in rerouted]

    def median(values):
        values = sorted(values)
        return values[len(values) // 2]

    # vLLM's own bounds: stock vLLM against the reference, same inputs
    stock_ffn = max(row["stock_output"] for row in rows)
    stock_attention = max(row["stock_attention"] for row in rows)
    measured = dict(stock_ffn=stock_ffn, stock_attention=stock_attention)
    # asm as accurate as stock vLLM, against the exact-activation references
    for metric in ("exact_attention", "exact_output"):
        asm_median = median(row[metric] for row in kept)
        stock_median = median(row["stock_" + metric] for row in kept)
        asm_max = max(row[metric] for row in kept)
        stock_max = max(row["stock_" + metric] for row in kept)
        measured[metric] = (asm_median / max(stock_median, 1e-9), asm_max / max(stock_max, 1e-9))
    # asm against the reference: two independent roundings of the same values
    for half, exact in (("attention", "exact_attention"), ("output", "exact_output")):
        predicted = median(
            (row[exact] ** 2 + row["reference_" + exact] ** 2) ** 0.5 for row in kept
        )
        measured["independent_" + half] = median(row[half] for row in kept) / max(predicted, 1e-9)
    signs = {name: min(row["sign_" + name] for row in kept) for name in MIN_SIGN_AGREEMENT}
    stock_signs = {name: min(row["stock_sign_" + name] for row in rows) for name in MIN_SIGN_AGREEMENT}
    print(
        f"{label}: CHECKS stock ffn max {stock_ffn:.4f} attention max {stock_attention:.4f} | "
        f"asm/stock vs exact (median, max): attention {measured['exact_attention'][0]:.3f} "
        f"{measured['exact_attention'][1]:.3f}, ffn {measured['exact_output'][0]:.3f} "
        f"{measured['exact_output'][1]:.3f} | asm vs ref / independent-rounding prediction: "
        f"attention {measured['independent_attention']:.3f}, ffn {measured['independent_output']:.3f} | "
        f"sign min asm {signs} stock {stock_signs}"
    )
    assert stock_ffn < STOCK_FFN_TOLERANCE, (label, "stock FFN", stock_ffn)
    assert stock_attention < STOCK_ATTENTION_TOLERANCE, (label, "stock attention", stock_attention)
    for metric in ("exact_attention", "exact_output"):
        ratio_median, ratio_max = measured[metric]
        assert ratio_median < ACCURACY_RATIO[wire], (label, metric, "median ratio", ratio_median)
        assert ratio_max < ACCURACY_RATIO_MAX[wire], (label, metric, "max ratio", ratio_max)
    for half in ("attention", "output"):
        assert measured["independent_" + half] < INDEPENDENT_ROUNDING, (label, half, measured)
    for name, bound in MIN_SIGN_AGREEMENT.items():
        assert signs[name] >= bound and stock_signs[name] >= bound, (label, name, signs, stock_signs)
    tie = ROUTING_TIE.get(layer, 1e-2)
    not_ties = [(row["output"], row["routing_gap"]) for row in rerouted if row["routing_gap"] >= tie]
    assert not not_ties, (label, not_ties)
    assert len(rerouted) <= MAX_REROUTED_FRACTION * len(rows), (label, len(rerouted), len(rows))


# ---------------------------------------------------------------- tests


def test_weights_are_gaussian(runs):
    """Every quantized weight the engine loaded, dequantized before post-load
    processing: finite, centered, its format's std, gaussian tails."""
    stats = runs("decode", "bf16")["weight_stats"]
    assert any(name.endswith("w1[0]") for name in stats) and len(stats) > 10, list(stats)
    for name, weight in stats.items():
        sigma = SIGMA["mxfp4" if name.endswith("]") else "mxfp8"]
        assert weight["finite"], (name, weight.get("layout"))
        assert abs(weight["mean"]) < 0.02 * sigma, (name, weight)
        assert abs(weight["std"] / sigma - 1) < 0.1, (name, weight)
        assert 2.6 < weight["kurtosis"] < 3.6, (name, weight)


@pytest.mark.parametrize("layer", list(LAYERS))
def test_asm_taken(runs, layer):
    """The layer's asm halves are installed as at TP4 (K1 on the
    compressed-attention layers, K2 everywhere) and those of ASM_HALVES ran
    (the others fell back to vLLM's path every time). Eager only: graph
    replays run no Python; test_decode_graph checks them bitwise."""
    results = runs("decode", "bf16")
    assert results["installed"]["layers"][layer] == [layer != 0, True], results["installed"]
    for half, expected in zip(("attention", "ffn"), ASM_HALVES[layer]):
        calls = results["asm_calls"].get((layer, half), 0)
        assert (calls > 0) == expected, (layer, half, calls, results["asm_calls"])


@pytest.mark.parametrize("num_rows", DECODE_ROW_COUNTS)
@pytest.mark.parametrize("layer", list(LAYERS))
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("context", list(DECODE_CONTEXTS))
def test_decode_eager(runs, context, wire, layer, num_rows):
    """Decode-only steps of ``num_rows`` rows on ``context`` contexts, eager:
    the layer's outputs per row against the reference (vLLM's deterministic
    path) on the same inputs, the reference's indexer top-k given to asm's
    attention (asm's own is checked by test_indexer_recall)."""
    rows = _rows_of(runs(DECODE_WORKLOADS[context], wire), layer, "decode", (num_rows, num_rows))
    label = f"{wire} decode {context} {num_rows} rows, layer {layer} ({LAYERS[layer]})"
    _check_rows(rows, wire, layer, label)


# not "mixed": with MIXED_CONTEXTS = 6 its decode-only steps have 6-8 rows,
# below ROWS_ON_ASM (it compared its 17- and 18-row steps with 16 contexts)
@pytest.mark.parametrize("workload", ["decode", "decode-long"])
@pytest.mark.parametrize("wire", WIRES)
def test_end_to_end(runs, workload, wire):
    """The whole model, eager: the logits of every decode-only step of
    ROWS_ON_ASM rows (per request and step), the asm path as served against
    the reference (vLLM's deterministic path), same forced tokens. Errors
    compound through the five random layers (a chaotic stack), so the bound
    is per layer's tolerance compounded; the per-layer items are the tight
    ones."""
    end_to_end = runs(workload, wire)["end_to_end"]
    asm, expected = end_to_end["asm"], end_to_end["reference"]
    keys = sorted(set(asm) & set(expected))
    assert keys, (len(asm), len(expected))
    errors = sorted(
        (_relative_error_per_row(asm[key][1][None], expected[key][1][None]).item(), key) for key in keys
    )
    median, worst = errors[len(errors) // 2][0], errors[-1]
    stock = sorted(
        _relative_error_per_row(end_to_end["stock"][key][1][None], expected[key][1][None]).item()
        for key in keys if key in end_to_end["stock"]
    )
    print(f"{wire} {workload} end to end: {len(errors)} (request, step) logits, relative error "
          f"asm median {median:.4f} max {worst[0]:.4f} at {worst[1]} | stock vLLM median "
          f"{stock[len(stock) // 2]:.4f} max {stock[-1]:.4f}")
    assert median < END_TO_END_TOLERANCE[wire], (median, errors[-5:])
    assert worst[0] < END_TO_END_WORST, errors[-5:]


@pytest.mark.parametrize("layer", INDEX_LAYERS)
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("workload", ["decode-long", "mixed"])
def test_indexer_recall(runs, workload, wire, layer):
    """The indexer top-k of asm and of the reference, each against the exact
    top-k (fp32 scores of the unquantized q and keys the reference's path
    quantizes), on rows with more than 512 candidates: asm's selection as
    good as the reference's, within RECALL_MARGIN. The reference's against
    the same scores with its MXFP4 rounding validates the reconstruction."""
    recalls: dict = {}
    for (layer_id, kind, _), found in runs(workload, wire)["recalls"].items():
        if layer_id == layer:
            for name, values in found.items():
                recalls.setdefault(name, []).extend(values)
    assert recalls.get("vllm"), f"no non-trivial top-k row at layer {layer}"
    mean = {name: sum(values) / len(values) for name, values in recalls.items()}
    print(f"{wire} {workload} layer {layer} indexer recall over {len(recalls['vllm'])} rows: "
          + ", ".join(f"{name} {value:.4f}" for name, value in sorted(mean.items())))
    assert mean["vllm@mxfp4"] > 0.98, mean  # the reference reconstruction
    assert mean["asm"] > mean["vllm"] - RECALL_MARGIN, mean


@pytest.mark.parametrize("row_range", MIXED_ROW_RANGES, ids=lambda r: f"{r[0]}-{r[1]}")
@pytest.mark.parametrize("layer", list(LAYERS))
@pytest.mark.parametrize("wire", WIRES)
def test_mixed_eager(runs, wire, layer, row_range):
    """Mixed steps (prefill chunks next to decode rows) of ``row_range``
    rows, eager: the layer's outputs per row (decode and prefill rows: the asm
    path takes the whole step) against vLLM's path on the same inputs."""
    rows = _rows_of(runs("mixed", wire), layer, "mixed", row_range)
    label = f"{wire} mixed {row_range[0]}-{row_range[1]} rows, layer {layer} ({LAYERS[layer]})"
    _check_rows(rows, wire, layer, label)


@pytest.mark.parametrize("num_rows", DECODE_ROW_COUNTS)
@pytest.mark.parametrize("wire", ["bf16"])  # one graph engine (< 4 min target): the bf16 wire
def test_decode_graph(runs, wire, num_rows):
    """Decode-only steps of ``num_rows`` rows replayed from CUDA graphs: the
    logits per request and step against the eager engine's (same forced
    tokens, same weights, same wire)."""
    eager, graph = runs("decode", wire)["served"], runs("decode", wire, graph=True)
    # the prefill runs vLLM's path, torch.compile'd in the graph engine only:
    # the requests whose prefill logits differ start decode from another state
    same_prefill = {
        length
        for length, logits in graph["prefill_logits"].items()
        if torch.equal(logits, eager["prefill_logits"].get(length, torch.empty(0)))
    }
    tokens = [
        token
        for token, (rows, _) in graph["logits_by_token"].items()
        if rows == num_rows and token in eager["logits_by_token"]
    ]
    seen = sorted({rows for rows, _ in graph["logits_by_token"].values()})
    assert tokens, f"no graph step of {num_rows} rows: {seen}"
    compared = [token for token in tokens if token[0] in same_prefill]
    excluded = sorted({token[0] for token in tokens} - same_prefill)
    print(
        f"{wire} graph {num_rows} rows: {len(compared)} of {len(tokens)} tokens compared; "
        f"excluded (prefill logits not bitwise): prompt lengths {excluded}"
    )
    assert len(compared) >= MIN_BITWISE_PREFILL_FRACTION * len(tokens), (num_rows, excluded)
    not_bitwise = [
        token
        for token in compared
        if not torch.equal(graph["logits_by_token"][token][1], eager["logits_by_token"][token][1])
    ]
    sizes = {
        token: _relative_error_per_row(
            graph["logits_by_token"][token][1][None], eager["logits_by_token"][token][1][None]
        ).item()
        for token in not_bitwise
    }
    # the two engines' prefill (vLLM's path: the eager engine's sequential
    # attention, the graph engine's multi-stream one) can leave KV caches a
    # rounding apart even where the prompt's last logits agree bitwise: a few
    # tokens may differ at rounding level, no more
    print(f"{wire} graph {num_rows} rows: not bitwise {sizes}")
    assert len(not_bitwise) <= GRAPH_NOT_BITWISE_FRACTION * len(compared), (num_rows, sizes)
    assert all(size < GRAPH_ROUNDING for size in sizes.values()), (num_rows, sizes)
