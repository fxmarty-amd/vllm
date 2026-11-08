# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only deterministic, high-precision reference pieces for DeepSeek-V4.1
on ROCm, installed in the process that runs the model (an in-process engine):

* ``install_reference_moe``: the MoE (routed experts and shared expert) in
  fp64 torch on exactly dequantized weights (the raw checkpoint codes,
  ``stash_raw`` before vLLM's post-load shuffles), the activations
  quantize-dequantized to MXFP8 where the served kernels quantize them (x
  before gate_up, h before down; one E8M0 a 32, the ceil rule AITER uses), e4m3
  round-to-nearest-even in fp64, SwiGLU clamp, the router's weights, one
  rounding to the output dtype. Replaces AITER's fused MoE, whose stage-2
  atomics make it non-deterministic.
* ``deterministic_patches``: the indexer top-k rows in a canonical order
  (ascending, -1 last; AITER's top-k returns the set in a run-dependent order)
  and vLLM's unquantized bf16 GEMM past 128 rows in fp32 torch (AITER's tgemm
  picks a split-K FlyDSL kernel there, non-deterministic).
* ``IndexerReference``: the indexer's top-k recomputed exactly (fp32, no MXFP4)
  from the bf16 q and keys vLLM's path quantizes (recorded at its q quant and
  K store), and with vLLM's MXFP4 rule (to validate the reconstruction).

Validated in asm_bench/MISMATCH_INVESTIGATION.md (sections 1.4, 1.5).
"""

import os

import torch

DTYPE = torch.float64
U8 = torch.uint8
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0

# id(module) -> raw weight codes, filled by stash_raw
_RAW: dict = {}


# ---------------------------------------------------------------- formats


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
    """e2m1 codes [..., K / 2] (low nibble first) + E8M0 [..., K / 32] -> [..., K]."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=dtype, device=codes.device)
    codes = codes.view(U8)
    values = torch.stack([table[(codes & 0xF).long()], table[(codes >> 4).long()]], -1).flatten(-2)
    scale = torch.exp2(scales.view(U8).to(dtype) - 127)
    return (values.unflatten(-1, (-1, 32)) * scale[..., None]).flatten(-2)


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
    """e4m3 [N, K] + E8M0 blocks [N / bn, K / bk] -> [N, K]."""
    w = weight.view(torch.float8_e4m3fn).to(dtype)
    s = torch.exp2(scale.view(U8).to(dtype) - 127)
    bn, bk = w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]
    return w * s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)


def _round_e4m3(v: torch.Tensor) -> torch.Tensor:
    """The nearest e4m3fn value (ties to even), saturating at 448."""
    mag = v.abs()
    exponent = torch.floor(torch.log2(mag.clamp_min(2.0**-30))).clamp_min(-6)
    step = torch.exp2(exponent - 3)
    return torch.copysign((torch.round(mag / step) * step).clamp_max(E4M3_MAX), v)


def qdq_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 quantize-dequantize along the last dim (blocks of 32, ceil rule)."""
    blocks = x.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127))
    return (_round_e4m3(blocks / scale) * scale).flatten(-2)


def _ceil_log2(y: torch.Tensor) -> torch.Tensor:
    """ceil(log2(y)) exactly, y > 0 (frexp: y = m 2^e, m in [0.5, 1))."""
    m, e = torch.frexp(y)
    return torch.where(m == 0.5, e - 1, e).to(y.dtype)


def _round_e4m3_exact(v: torch.Tensor) -> torch.Tensor:
    """_round_e4m3 with the exponent from frexp (exact in any float dtype)."""
    mag = v.abs()
    _, e = torch.frexp(mag)
    exponent = (e - 1).clamp_min(-6).to(v.dtype)  # floor(log2 |v|); subnormals: 2^-9 steps
    step = torch.exp2(exponent - 3)
    return torch.copysign((torch.round(mag / step) * step).clamp_max(E4M3_MAX), v)


def qdq_mxfp8_exact(x: torch.Tensor) -> torch.Tensor:
    """qdq_mxfp8 with exact exponents (frexp); the amax / 448 division in x's dtype."""
    blocks = x.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    scale = torch.exp2(_ceil_log2(amax / E4M3_MAX).clamp(-127, 127))
    return (_round_e4m3_exact(blocks / scale) * scale).flatten(-2)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------- the reference MoE


def clear() -> None:
    """Drop the stashed raw codes (they hold GPU memory past the engine)."""
    _RAW.clear()


def stash_raw(model: torch.nn.Module) -> int:
    """Clone the raw codes of every MXFP4 expert set and MXFP8 linear; call it
    before process_weights_after_loading."""
    count = 0
    for module in model.modules():
        w13 = getattr(module, "w13_weight", None)
        if isinstance(w13, torch.Tensor) and w13.device.type != "meta":
            _RAW[id(module)] = dict(
                kind="experts",
                w13=w13.data.view(U8).clone(),
                s13=module.w13_weight_scale.data.view(U8).clone(),
                w2=module.w2_weight.data.view(U8).clone(),
                s2=module.w2_weight_scale.data.view(U8).clone(),
            )
            count += 1
            continue
        w, ws = getattr(module, "weight", None), getattr(module, "weight_scale", None)
        if (
            isinstance(w, torch.Tensor)
            and isinstance(ws, torch.Tensor)
            and w.device.type != "meta"
            and w.dtype == torch.float8_e4m3fn
        ):
            _RAW[id(module)] = dict(kind="linear", w=w.data.clone(), s=ws.data.view(U8).clone())
            count += 1
    return count


def routed_reference(raw, x, topk_weights, topk_ids, limit, expert_map, inter):
    """sum_k w_k * expert_k(x) -> [M, H] in DTYPE."""
    ids = topk_ids.long()
    if expert_map is not None:
        ids = expert_map[ids].long()
    pad = raw["w13"].shape[1] // 2
    xq = qdq_mxfp8(x.to(DTYPE))
    out = torch.zeros(x.shape[0], x.shape[1], dtype=DTYPE, device=x.device)
    rows = torch.cat([torch.arange(inter), pad + torch.arange(inter)]).to(x.device)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        w13 = dequant_mxfp4(raw["w13"][expert, rows], raw["s13"][expert, rows])
        w2 = dequant_mxfp4(raw["w2"][expert, :, : inter // 2], raw["s2"][expert, :, : inter // 32])
        gate_up = xq[tokens] @ w13.T
        h = qdq_mxfp8(_swiglu(gate_up[:, :inter], gate_up[:, inter:], limit))
        out.index_add_(0, tokens, (h @ w2.T) * topk_weights[tokens, slots, None].to(DTYPE))
    return out


def shared_reference(gate_up_raw, down_raw, x, limit):
    """The shared expert (two MXFP8 linears) -> [M, H] in DTYPE."""
    w13 = dequant_mxfp8(gate_up_raw["w"], gate_up_raw["s"])
    w2 = dequant_mxfp8(down_raw["w"], down_raw["s"])
    inter = w13.shape[0] // 2
    gate_up = qdq_mxfp8(x.to(DTYPE)) @ w13.T
    return qdq_mxfp8(_swiglu(gate_up[:, :inter], gate_up[:, inter:], limit)) @ w2.T


# rows per reference chunk (bounds the temporaries of long prefills)
_CHUNK = 4096
# the reference on (installed patches active) or off (stock vLLM, e.g. for a
# stock pass next to the reference pass)
ENABLED = {"on": True}
# experts dequantized per chunk when caching an expert set
_EXPERT_CHUNK = 16
# the activations' quantization in the reference: "mxfp8" (vLLM's rule, as
# AITER's afp8 kernels and the MXFP8 linears) or "none" (exact activations)
ACTIVATIONS = {"rule": "mxfp8", "attention": "mxfp8"}


# test-only: round intermediates to bf16 as the served kernels may (to locate
# where stock vLLM and the fp32 reference part)
INTERMEDIATES = {"gate_up": False, "h": False, "picks": False}
# asm's rounding rules (TEST_TMP_EXACT_ASM_ROUNDING): "ffn" (K2's) and
# "attention" (K1's), and the per-call inputs they need, set by the harness:
# ASM_CONTEXT["ffn"] = (y', rstd) of the FFN seam, y' = bf16(x1 * ffn_norm.w)
# without rstd; ASM_CONTEXT["attention"] likewise for the attention seam
ASM_RULE = {"ffn": False, "attention": False}
ASM_CONTEXT: dict = {}


def _asm_ffn_input(x: torch.Tensor):
    """K2's input for the rows of x: (MXFP8(y'), rstd [rows, 1]) or None."""
    if not ASM_RULE["ffn"]:
        return None
    found = ASM_CONTEXT.get("ffn")
    if found is None or found[0].shape[0] != x.shape[0]:
        return None
    y_prime, rstd = found
    return qdq_mxfp8_exact(y_prime.float()), rstd.float().view(-1, 1)


def _intermediate(x: torch.Tensor, name: str) -> torch.Tensor:
    return x.to(torch.bfloat16).float() if INTERMEDIATES[name] else x


def _activation(x: torch.Tensor) -> torch.Tensor:
    return qdq_mxfp8_exact(x) if ACTIVATIONS["rule"] == "mxfp8" else x


def dequantized_experts(raw: dict, inter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """An expert set's weights dequantized once: w13 [E, 2 inter, H] (gate rows,
    then up rows), w2 [E, H, inter], in bf16, which holds every MXFP4 value
    exactly (an e2m1 value times a power of 2); checked."""
    pad = raw["w13"].shape[1] // 2
    rows = torch.cat([torch.arange(inter), pad + torch.arange(inter)]).to(raw["w13"].device)
    w13, w2 = [], []
    for start in range(0, raw["w13"].shape[0], _EXPERT_CHUNK):
        end = start + _EXPERT_CHUNK
        full13 = dequant_mxfp4(raw["w13"][start:end, rows], raw["s13"][start:end, rows], torch.float32)
        full2 = dequant_mxfp4(
            raw["w2"][start:end, :, : inter // 2], raw["s2"][start:end, :, : inter // 32], torch.float32
        )
        w13.append(full13.to(torch.bfloat16))
        w2.append(full2.to(torch.bfloat16))
        assert torch.equal(w13[-1].float(), full13) and torch.equal(w2[-1].float(), full2), (
            "an MXFP4 weight is not exact in bf16"
        )
    return torch.cat(w13), torch.cat(w2)


def routed_fast(w13, w2, x, topk_weights, topk_ids, limit, expert_map):
    """routed_reference on cached weights: fp32 GEMMs ("highest" precision, no
    TF32), the expert sum in fp64, one host sync. -> [M, H] fp64."""
    ids = topk_ids.long()
    if expert_map is not None:
        ids = expert_map[ids].long()
    num_rows, k = ids.shape
    inter = w2.shape[-1]
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat.clamp_min(0), minlength=w13.shape[0])
    counts[0] -= (flat < 0).sum()  # ids < 0 (masked picks) sort first: skip them
    skipped = int((flat < 0).sum())
    counts = counts.tolist()
    asm = _asm_ffn_input(x)
    # K2: MXFP8 of y' (before rstd), rstd applied to gate / up after the GEMM
    xq, row_scale = asm if asm is not None else (_activation(x.float()), None)
    picks = order[skipped:]
    tokens, slots = picks // k, picks % k
    segments = [(expert, n) for expert, n in enumerate(counts) if n]
    h = torch.empty(picks.numel(), inter, dtype=torch.float32, device=x.device)
    y = torch.empty(picks.numel(), x.shape[1], dtype=torch.float32, device=x.device)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        xs = xq[tokens]  # the picks' rows, by expert
        start = 0
        for expert, n in segments:
            gate_up = _intermediate(xs[start : start + n] @ w13[expert].float().T, "gate_up")
            if row_scale is not None:
                gate_up = gate_up * row_scale[tokens[start : start + n]]
            h[start : start + n] = _swiglu(gate_up[:, :inter], gate_up[:, inter:], limit)
            start += n
        h = _activation(_intermediate(h, "h"))  # per row: one quantization for every pick
        start = 0
        for expert, n in segments:
            torch.matmul(h[start : start + n], w2[expert].float().T, out=y[start : start + n])
            start += n
    finally:
        torch.set_float32_matmul_precision(precision)
    # each pick to its own (token, slot) place, then the sum over the slots in a
    # fixed order (an index_add over duplicate tokens would use atomics)
    picked = torch.zeros(num_rows, k, x.shape[1], dtype=torch.float64, device=x.device)
    picked[tokens, slots] = y.double() * topk_weights[tokens, slots, None].double()
    if row_scale is not None:  # K2: each weighted pick in bf16, their sum in fp32
        picked = picked.to(torch.bfloat16).double()
    if INTERMEDIATES["picks"]:  # each weighted pick in bf16, summed in bf16
        out = picked[:, 0].to(torch.bfloat16)
        for slot in range(1, k):
            out = (out.float() + picked[:, slot].to(torch.bfloat16).float()).to(torch.bfloat16)
        return out.double()
    out = picked[:, 0].clone()
    for slot in range(1, k):
        out += picked[:, slot]
    return out


def shared_fast(w13, w2, x, limit):
    """shared_reference on cached (bf16-exact) MXFP8 weights, fp32 -> [M, H] fp32."""
    inter = w13.shape[0] // 2
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        asm = _asm_ffn_input(x)
        if asm is not None:  # K2: MXFP8(y') before rstd, rstd after the GEMM
            gate_up = (asm[0] @ w13.float().T) * asm[1]
        else:
            gate_up = _intermediate(_activation(x.float()) @ w13.float().T, "gate_up")
        h = _intermediate(_swiglu(gate_up[:, :inter], gate_up[:, inter:], limit), "h")
        return _activation(h) @ w2.float().T
    finally:
        torch.set_float32_matmul_precision(precision)


def install_reference_moe(model: torch.nn.Module):
    """Route ``model``'s routed experts (AiterMxfp4Experts.apply) and shared
    experts (DeepseekV4MLP.forward) through the reference → undo()."""
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterMxfp4Experts
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP

    # the weights dequantized once (bf16, exact), keyed by the served weight
    experts_by_weight = {
        module.w13_weight.data_ptr(): dequantized_experts(
            _RAW[id(module)], module.moe_config.intermediate_size_per_partition
        )
        for module in model.modules()
        if id(module) in _RAW and _RAW[id(module)]["kind"] == "experts"
    }

    def exact_bf16(raw):
        full = dequant_mxfp8(raw["w"], raw["s"], torch.float32)
        assert torch.equal(full.to(torch.bfloat16).float(), full), "an MXFP8 weight is not exact in bf16"
        return full.to(torch.bfloat16)

    shared_by_module = {
        id(module): (exact_bf16(_RAW[id(module.gate_up_proj)]), exact_bf16(_RAW[id(module.down_proj)]))
        for module in model.modules()
        if isinstance(module, DeepseekV4MLP)
        and id(module.gate_up_proj) in _RAW
        and id(module.down_proj) in _RAW
    }
    apply, forward = AiterMxfp4Experts.apply, DeepseekV4MLP.forward

    def reference_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                        activation=None, global_num_experts=None, expert_map=None,
                        a1q_scale=None, a2_scale=None, workspace13=None, workspace2=None,
                        expert_tokens_meta=None, apply_router_weight_on_input=False):
        weights = experts_by_weight.get(w1.data_ptr())
        if weights is None or apply_router_weight_on_input or not ENABLED["on"]:
            return apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                         activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                         workspace13, workspace2, expert_tokens_meta,
                         apply_router_weight_on_input)
        limit = self.moe_config.swiglu_limit
        for start in range(0, hidden_states.shape[0], _CHUNK):
            end = start + _CHUNK
            out = routed_fast(*weights, hidden_states[start:end], topk_weights[start:end],
                              topk_ids[start:end], limit, expert_map)
            output[start:end].copy_(out.to(output.dtype))

    def reference_forward(self, x):
        weights = shared_by_module.get(id(self))
        if weights is None or not ENABLED["on"] or not x.shape[0]:
            return forward(self, x)
        limit = getattr(self.act_fn, "swiglu_limit", None)
        return torch.cat([
            shared_fast(*weights, x[start : start + _CHUNK], limit).to(x.dtype)
            for start in range(0, x.shape[0], _CHUNK)
        ])

    AiterMxfp4Experts.apply = reference_apply
    DeepseekV4MLP.forward = reference_forward

    def undo():
        AiterMxfp4Experts.apply, DeepseekV4MLP.forward = apply, forward

    return undo


# ---------------------------------------------------------------- exact attention


def _exact_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x @ weight.T in fp32, "highest" precision (no TF32) -> fp32."""
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        out = x.float() @ weight.float().T
    finally:
        torch.set_float32_matmul_precision(precision)
    key = (weight.data_ptr(), tuple(weight.shape))
    if key not in GEMM_CHECK and x.shape[0]:
        # once per weight: the fp32 GEMM against fp64 on a few rows (relative L2)
        exact = x[:8].double() @ weight.double().T
        GEMM_CHECK[key] = ((out[:8].double() - exact).norm(dim=-1)
                           / exact.norm(dim=-1).clamp_min(1e-300)).max().item()
    return out


# _exact_gemm's cross-check against fp64, worst relative L2 error per weight
GEMM_CHECK: dict = {}


def install_exact_attention(model: torch.nn.Module):
    """While ACTIVATIONS["attention"] == "none": the ROCm DeepSeek-V4.1
    attention's four GEMMs (fused_wqa_wkv, wq_b, wo_a, wo_b; the indexer's wq_b
    too) in fp32 on the raw MXFP8 weights (dequantized once, bf16-exact) with
    unquantized inputs, the q / kv RMS norms in fp32 (no fp8 quantization of
    qr), the attention output in bf16 (no MXFP8 before wo_a). The KV-cache
    formats and the attention kernel are unchanged. → undo()."""
    from vllm.models.deepseek_v41.amd.rocm import (
        DeepseekV41ROCMAiterMLAAttention,
        DeepseekV41RocmMxfp4Indexer,
    )

    def exact_weight(linear):
        raw = _RAW.get(id(linear))
        if raw is None or raw["kind"] != "linear":
            return None
        full = dequant_mxfp8(raw["w"], raw["s"], torch.float32)
        assert torch.equal(full.to(torch.bfloat16).float(), full), "an MXFP8 weight is not exact in bf16"
        return full.to(torch.bfloat16)

    weights: dict = {}
    for module in model.modules():
        if isinstance(module, DeepseekV41ROCMAiterMLAAttention):
            weights[id(module)] = {
                name: exact_weight(getattr(module, name))
                for name in ("fused_wqa_wkv", "wq_b", "wo_a", "wo_b")
            }
        if isinstance(module, DeepseekV41RocmMxfp4Indexer):
            weights[id(module)] = {"wq_b": exact_weight(module.wq_b)}
    attention, indexer = DeepseekV41ROCMAiterMLAAttention, DeepseekV41RocmMxfp4Indexer
    originals = {
        (attention, name): getattr(attention, name)
        for name in ("_fused_wqa_wkv_gemm", "_split_qkv_and_norm", "_wq_b_proj", "_alloc_attn_out", "_o_proj",
                     "_fused_qnorm_rope_kv_insert")
    }
    originals[(indexer, "_wq_b_proj")] = indexer._wq_b_proj
    CALLS.clear()

    def active(self, name):
        return ACTIVATIONS["attention"] == "none" and weights.get(id(self), {}).get(name) is not None

    def count(name):
        CALLS[name] = CALLS.get(name, 0) + 1

    def asm_rule(self, name):
        """K1's rounding rules apply (TEST_TMP_EXACT_ASM_ROUNDING)."""
        return ASM_RULE["attention"] and weights.get(id(self), {}).get(name) is not None

    def fused_wqa_wkv_gemm(self, hidden_states):
        found = ASM_CONTEXT.get("attention")
        if asm_rule(self, "fused_wqa_wkv") and found is not None and found[0].shape[0] == hidden_states.shape[0]:
            # K1: MXFP8 of y' = x1 * attn_norm.w (before rstd), rstd after the GEMM
            count("asm fused_wqa_wkv")
            y_prime, rstd = found
            out = _exact_gemm(qdq_mxfp8_exact(y_prime.float()), weights[id(self)]["fused_wqa_wkv"])
            # K1 hands wq_b the q rows of this product unscaled, in fp32
            ASM_CONTEXT["qr_prime"] = (out[:, : self.q_lora_rank], rstd.float().view(-1, 1))
            scaled = out * rstd.float().view(-1, 1)
            # K1's kv wave: qr_kv is stored in bf16 unscaled, then x = bf16(x * rstd)
            kv = out[:, self.q_lora_rank:].to(torch.bfloat16).float() * rstd.float().view(-1, 1)
            scaled[:, self.q_lora_rank:] = kv
            return scaled.to(hidden_states.dtype)
        if not active(self, "fused_wqa_wkv"):
            return originals[(attention, "_fused_wqa_wkv_gemm")](self, hidden_states)
        count("fused_wqa_wkv")
        return _exact_gemm(hidden_states, weights[id(self)]["fused_wqa_wkv"]).to(hidden_states.dtype)

    def split_qkv_and_norm(self, qr_kv):
        if asm_rule(self, "wq_b"):
            # K1: kv normed; q handed to wq_b as qr * q_norm.w without the norm's
            # 1 / rms, which wq_b_proj applies after the GEMM
            count("asm norms")
            qr, kv = qr_kv.float().split([self.q_lora_rank, self.head_dim], dim=-1)
            kv = kv * torch.rsqrt(kv.pow(2).mean(-1, keepdim=True) + self.eps) * self.kv_norm.weight.float()
            prime = ASM_CONTEXT.get("qr_prime")
            if prime is not None and prime[0].shape[0] == qr.shape[0]:
                # K1: wq_b's input is qr' * q_norm.w (qr' = MXFP8(y') @ W, no rstd,
                # fp32), the norm rr = rstd * rsqrt(rstd^2 * mean(qr'^2) + eps) after
                qr_prime, rstd = prime
                rrms = rstd * torch.rsqrt(rstd * rstd * qr_prime.pow(2).mean(-1, keepdim=True) + self.eps)
                # K1 keeps qr' in bf16 (its qr_kv buffer) and rounds qr' * q_norm.w
                # to bf16 before MXFP8 (checked stage by stage: the codes then
                # match to 0.2 %, against 3.4 % without these roundings)
                qr_prime = qr_prime.to(torch.bfloat16).float()
                q_in = (qr_prime * self.q_norm.weight.to(torch.bfloat16).float()).to(torch.bfloat16).float()
            else:
                rrms = torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + self.eps)
                q_in = qr * self.q_norm.weight.float()
            rrms.asm_rrms = True
            ASM_CONTEXT["rr"] = rrms
            return q_in, rrms, kv.to(qr_kv.dtype)
        if ACTIVATIONS["attention"] != "none":
            return originals[(attention, "_split_qkv_and_norm")](self, qr_kv)
        count("norms")
        qr, kv = qr_kv.float().split([self.q_lora_rank, self.head_dim], dim=-1)

        def rms(x, weight):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight.float()

        return (rms(qr, self.q_norm.weight).to(qr_kv.dtype), None,
                rms(kv, self.kv_norm.weight).to(qr_kv.dtype))

    def wq_b_proj(self, qr, qr_scale=None):
        if getattr(qr_scale, "asm_rrms", False) and weights.get(id(self), {}).get("wq_b") is not None:
            # K1: MXFP8 of the unnormalised q, the norm's 1 / rms after the GEMM
            count("asm wq_b")
            q_in = qdq_mxfp8_exact(qr.float())
            out = _exact_gemm(q_in, weights[id(self)]["wq_b"])
            if not isinstance(self, indexer):
                ASM_CONTEXT["q"] = out  # before the 1 / rms (fused_qnorm_rope_kv_insert)
            return (out * qr_scale).to(torch.bfloat16)
        if not active(self, "wq_b") or qr_scale is not None or not isinstance(qr, torch.Tensor):
            owner = indexer if isinstance(self, indexer) else attention
            return originals[(owner, "_wq_b_proj")](self, qr, qr_scale)
        count("wq_b")
        return _exact_gemm(qr, weights[id(self)]["wq_b"]).to(torch.bfloat16)

    def alloc_attn_out(self, num_tokens, hidden_states):
        if ACTIVATIONS["attention"] != "none":
            return originals[(attention, "_alloc_attn_out")](self, num_tokens, hidden_states)
        return torch.empty(
            (num_tokens, self.padded_heads, self.head_dim), dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def o_proj(self, attn_out, positions):
        if asm_rule(self, "wo_a") and not isinstance(attn_out, torch.Tensor):
            # K1: the MXFP8 attention output -> wo_a in fp32 -> MXFP8 of z -> wo_b
            count("asm wo_a+wo_b")
            data = attn_out.data.view(torch.float8_e4m3fn).float()
            scale = torch.exp2(attn_out.scale.view(torch.uint8).float() - 127)
            o = (data.unflatten(-1, (-1, 32)) * scale[..., None]).flatten(-2)
            o = o.reshape(o.shape[0], self.n_local_groups, -1)
            wo_a = weights[id(self)]["wo_a"].view(self.n_local_groups, self.o_lora_rank, -1)
            precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            try:
                z = torch.einsum("tgd,grd->tgr", o, wo_a.float()).flatten(1)
            finally:
                torch.set_float32_matmul_precision(precision)
            # K1 rounds z to bf16 before its MXFP8 (stage check: 0.05 % of the
            # codes differ with it, 3.4 % without)
            z = qdq_mxfp8_exact(z.to(torch.bfloat16).float())
            return _exact_gemm(z, weights[id(self)]["wo_b"]).to(torch.bfloat16)
        if not active(self, "wo_a") or not isinstance(attn_out, torch.Tensor):
            return originals[(attention, "_o_proj")](self, attn_out, positions)
        count("wo_a+wo_b")
        # the attention output comes back rotated (gfx950), as for the bf16 path
        o = attn_out[:, : self.n_local_heads, :].reshape(attn_out.shape[0], self.n_local_groups, -1)
        wo_a = weights[id(self)]["wo_a"].view(self.n_local_groups, self.o_lora_rank, -1)
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            z = torch.einsum("tgd,grd->tgr", o.float(), wo_a.float()).flatten(1)
        finally:
            torch.set_float32_matmul_precision(precision)
        return _exact_gemm(z, weights[id(self)]["wo_b"]).to(attn_out.dtype)

    def fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        out = originals[(attention, "_fused_qnorm_rope_kv_insert")](self, q, kv, positions, attn_metadata)
        if ASM_RULE["attention"] and isinstance(attn_metadata, dict) and "rr" in ASM_CONTEXT:
            rows = q.shape[0]
            if ASM_CONTEXT.get("q") is not None and ASM_CONTEXT["q"].shape[0] == rows:
                # K1: RoPE (_rope_k1) on the unscaled wq_b output rounded to bf16, bf16;
                # q_rr then x rr, bf16 again (mixed steps; on decode K1 folds rr
                # in its fp32 score scale, the nearest a bf16 q gets to it)
                heads = q.shape[1]
                k1_q = _rope_k1(ASM_CONTEXT["q"].to(torch.bfloat16).view(rows, heads, -1),
                                self.rotary_emb.cos_sin_cache, positions[:rows])
                rr = ASM_CONTEXT["rr"].float().view(rows, 1, 1)
                out[:rows, :heads].copy_((k1_q.float() * rr).to(torch.bfloat16))
                count("asm q order")
        return out

    attention._fused_qnorm_rope_kv_insert = fused_qnorm_rope_kv_insert
    attention._fused_wqa_wkv_gemm = fused_wqa_wkv_gemm
    attention._split_qkv_and_norm = split_qkv_and_norm
    attention._wq_b_proj = wq_b_proj
    attention._alloc_attn_out = alloc_attn_out
    attention._o_proj = o_proj
    indexer._wq_b_proj = wq_b_proj

    # the attention's compressor (index source layers): asm's comp_gemm runs on
    # y' (no rstd), fp32, and scales each kv_score row by rstd
    compressors = [m.compressor for m in model.modules()
                   if isinstance(m, attention) and getattr(m, "compressor", None) is not None]

    def compressor_forward(compressor):
        forward = compressor.forward

        def wrapped(kv_score, positions, *args, **kwargs):
            found = ASM_CONTEXT.get("attention")
            if ASM_RULE["attention"] and found is not None and found[0].shape[0] == kv_score.shape[0]:
                count("asm compressor gemm")
                y_prime, rstd = found
                weight = compressor.fused_wkv_wgate.weight
                kv_score = torch.mm(y_prime.to(weight.dtype), weight.T, out_dtype=torch.float32)
                kv_score = kv_score * rstd.float().view(-1, 1)
            return forward(kv_score, positions, *args, **kwargs)

        return wrapped

    for compressor in compressors:
        compressor.forward = compressor_forward(compressor)

    def undo():
        for (owner, name), value in originals.items():
            setattr(owner, name, value)
        for compressor in compressors:
            compressor.__dict__.pop("forward", None)

    return undo


# exact-attention calls by kind (to check the exact pass took the patches)
CALLS: dict = {}


# ---------------------------------------------------------------- determinism


def deterministic_patches():
    """Canonical top-k row order and a deterministic unquantized GEMM past 128
    rows; apply before the engine is built (linears bind their GEMM at
    construction) → undo()."""
    import vllm.model_executor.layers.utils as layer_utils
    import vllm.v1.attention.ops.rocm_paged_mxfp4_indexer as indexer_ops

    topk, gemm = indexer_ops._topk, layer_utils.rocm_unquantized_gemm

    def sorted_topk(logits, lengths, out, k):
        rows, width = logits.shape
        if not ENABLED["on"] or not out.numel() or not rows:
            return topk(logits, lengths, out, k)
        # deterministic: a stable descending sort of each row's [0, length) logits,
        # exact ties to the lower index (AITER's top_k_per_row breaks them by its
        # scratch state: one swapped candidate seen, 10.6), then ascending, -1 pads
        n = min(k, width)
        valid = torch.arange(width, device=logits.device)[None, :] < lengths.reshape(-1)[:rows, None]
        masked = torch.where(valid, logits.float(), torch.full_like(logits, float("-inf"), dtype=torch.float32))
        order = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :n]
        picked = torch.where(torch.gather(valid, 1, order), order, -1).to(out.dtype)
        big = torch.iinfo(out.dtype).max
        key = torch.where(picked < 0, torch.full_like(picked, big), picked).sort(dim=1).values
        out[:rows].fill_(-1)
        out[:rows, :n].copy_(torch.where(key == big, torch.full_like(key, -1), key))

    def fp32_gemm(layer, x, weight, bias=None):
        if not x.is_cuda or x.numel() // x.size(-1) <= 128 or not ENABLED["on"]:
            return gemm(layer, x, weight, bias)
        out = x.float() @ weight.float().T
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype)

    indexer_ops._topk, layer_utils.rocm_unquantized_gemm = sorted_topk, fp32_gemm

    def undo():
        indexer_ops._topk, layer_utils.rocm_unquantized_gemm = topk, gemm

    return undo


# ---------------------------------------------------------------- exact indexer


def _rope_gptj(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """GPT-J RoPE on the last 64 dims (the indexer's Q and K), fp32."""
    x = x.float().clone()
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    if x.dim() == 3:
        cos, sin = cos[:, None], sin[:, None]
    even, odd = x[..., -64::2].clone(), x[..., -63::2].clone()
    x[..., -64::2] = even * cos - odd * sin
    x[..., -63::2] = odd * cos + even * sin
    return x


def _rope_k1(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """K1's q RoPE (gemv8 epilogue "rope") on bf16 x, last 64 dims: e' = fma(-o, sin,
    fp32(e cos)), o' = fma(e, sin, fp32(o cos)), each rounded to fp32 once (exact
    in fp64: bf16 x fp32 products fit), then bf16."""
    y = x.float().clone()
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None].double(), sin[:, None].double()
    e, o = x[..., -64::2].double(), x[..., -63::2].double()
    y[..., -64::2] = ((e * cos).float().double() - o * sin).float()
    y[..., -63::2] = ((o * cos).float().double() + e * sin).float()
    return y.to(torch.bfloat16)


def mxfp4_round_trip(x: torch.Tensor) -> torch.Tensor:
    """MXFP4 as the indexer's writers round: scale 2^ceil(log2(amax / 6)) a 32,
    the nearest e2m1 value, ties to even."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2**-126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6)))
    y = (blocks / scale).clamp(-6, 6)
    grid = torch.tensor(E2M1, device=x.device)
    grid = torch.cat([grid[0::2], grid[1::2]])  # even codes first: argmin keeps them on ties
    index = (y.abs()[..., None] - grid).abs().argmin(-1)
    return (grid[index] * y.sign() * scale).flatten(-2)


class IndexerReference:
    """Records, from vLLM's path, every indexer key it stores (bf16 k_pre, per
    cache slot, normed and rotated: exact and MXFP4) and the q quantizations it
    runs (bf16 q, positions, weights), and recomputes a step's top-k exactly.

    ``install()`` wraps the two ops in vllm.models.deepseek_v41.amd.rocm
    → undo(). ``take_queries()`` returns the q quantizations since the last call
    (the harness calls it around a layer's reference pass)."""

    def __init__(self):
        self.keys: dict = {}  # k cache data_ptr -> dict(exact, mxfp4, ratio, entries)
        self.queries: list = []

    def install(self):
        import vllm.models.deepseek_v41.amd.rocm as rocm

        store, quant = rocm.rocm_mxfp4_indexer_k_store, rocm.rocm_mxfp4_indexer_q_quant

        def k_store(k_pre, positions, cos_sin, norm_weight, eps, k_cache, slots, ratio, *args, **kwargs):
            self._record_keys(k_pre, positions, cos_sin, norm_weight, eps, k_cache, slots, ratio)
            return store(k_pre, positions, cos_sin, norm_weight, eps, k_cache, slots, ratio, *args, **kwargs)

        def q_quant(positions, q, cos_sin, weights, softmax_scale, head_scale, *args, **kwargs):
            self.queries.append(dict(
                positions=positions.clone(), q=q.clone(), cos_sin=cos_sin,
                weights=weights.float() * softmax_scale * head_scale,
            ))
            return quant(positions, q, cos_sin, weights, softmax_scale, head_scale, *args, **kwargs)

        rocm.rocm_mxfp4_indexer_k_store, rocm.rocm_mxfp4_indexer_q_quant = k_store, q_quant

        def undo():
            rocm.rocm_mxfp4_indexer_k_store, rocm.rocm_mxfp4_indexer_q_quant = store, quant

        return undo

    def _record_keys(self, k_pre, positions, cos_sin, norm_weight, eps, k_cache, slots, ratio):
        n = slots.numel()
        if n == 0:
            return
        key = k_cache.data_ptr()
        if key not in self.keys:
            slots_total = k_cache.shape[0] * k_cache.shape[1]
            self.keys[key] = dict(
                exact=torch.zeros(slots_total, k_pre.shape[1], device=k_pre.device),
                mxfp4=torch.zeros(slots_total, k_pre.shape[1], device=k_pre.device),
                ratio=ratio, entries=k_cache.shape[1],
            )
        store = self.keys[key]
        pos = positions[:n].long()
        keep = (slots[:n] >= 0) & ((pos + 1) % ratio == 0)
        if not bool(keep.any()):
            return
        k = k_pre[:n][keep].float()
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
        k = _rope_gptj(k, cos_sin, (pos[keep] // ratio) * ratio)
        store["exact"][slots[:n][keep].long()] = k
        store["mxfp4"][slots[:n][keep].long()] = mxfp4_round_trip(k.bfloat16())

    def take_queries(self) -> list:
        queries, self.queries = self.queries, []
        return queries

    def recall(self, k_cache, metadata, query: dict, selections: dict, topk: int) -> dict:
        """Per row of the step (the rows ``query`` holds), for each selection
        (name -> [rows, topk] local compressed indices, -1 padded): the fraction
        of the exact top-k it picked, and of the MXFP4-rule top-k ("<name>@mxfp4").
        Rows with at most ``topk`` candidates (a trivial top-k) are skipped:
        → {name: [recall per non-trivial row]}."""
        store = self.keys.get(k_cache.data_ptr())
        out: dict = {}
        if store is None:
            return out
        ratio, entries = store["ratio"], store["entries"]
        positions = query["positions"].long()
        q_exact = _rope_gptj(query["q"], query["cos_sin"], positions)
        q_mxfp4 = mxfp4_round_trip(q_exact.bfloat16())
        weights = query["weights"]
        tables = _row_block_tables(metadata, positions.numel())
        for row in range(positions.numel()):
            end = int(positions[row] + 1) // ratio
            if end <= topk or tables[row] is None:
                continue
            columns = torch.arange(end, device=positions.device)
            slots = tables[row][columns // entries].long() * entries + columns % entries
            reference = {}
            for rule, keys, q in (("exact", store["exact"], q_exact), ("mxfp4", store["mxfp4"], q_mxfp4)):
                scores = (torch.relu(q[row] @ keys[slots].T) * weights[row][:, None]).sum(0)
                reference[rule] = set(scores.topk(topk).indices.tolist())
            for name, selected in selections.items():
                picked = set(selected[row][selected[row] >= 0].tolist())
                for rule, suffix in (("exact", ""), ("mxfp4", "@mxfp4")):
                    out.setdefault(name + suffix, []).append(len(picked & reference[rule]) / topk)
        return out


def _row_block_tables(metadata, num_rows: int) -> list:
    """Per row of the step, its request's indexer block table (None: unknown)."""
    tables: list = [None] * num_rows
    decode = getattr(metadata, "decode", None)
    if decode is not None:
        num_decode = min(decode.block_table.shape[0], num_rows)
        for row in range(num_decode):
            tables[row] = decode.block_table[row]
    prefill = getattr(metadata, "prefill", None)
    if prefill is not None:
        # the ROCm plans: (first row, end row, request) per request, rows from
        # the chunk's token_start, requests indexing the chunk's block table
        for chunk, plan in zip(prefill.chunks, getattr(metadata, "prefill_plans", [])):
            for first, end, request in plan.requests:
                for row in range(chunk.token_start + first, min(chunk.token_start + end, num_rows)):
                    tables[row] = chunk.block_table[request]
    return tables
