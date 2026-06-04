# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import huggingface_hub
import pytest
import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils import (
    nvfp4_emulation_utils,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_triton_dequantize_nvfp4(monkeypatch) -> None:
    """Test the Triton dequantization kernel against the CPU reference
    using real NVFP4 weights from a checkpoint.

    Tests both 2D (attention projection) and 3D (stacked MoE experts).
    """
    checkpoint_path = huggingface_hub.snapshot_download(
        "nvidia/Qwen3-30B-A3B-NVFP4",
        allow_patterns=["model-00001-of-00004.safetensors"],
    )
    shard_path = f"{checkpoint_path}/model-00001-of-00004.safetensors"
    block_size = 16

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # 2D case: attention projection
        tensor_fp4_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight")
        tensor_sf_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale")
        global_scale_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale_2")

        # 3D case: stack ALL experts for layer 9 up_proj
        expert_prefix = "model.layers.9.mlp.experts."
        expert_indices = sorted(
            int(key.split(".")[5])
            for key in all_keys
            if key.startswith(expert_prefix) and key.endswith(".up_proj.weight")
        )
        assert len(expert_indices) > 0

        all_fp4 = []
        all_sf = []
        all_global_scale = []
        for index in expert_indices:
            name = f"{expert_prefix}{index}.up_proj"
            all_fp4.append(f.get_tensor(f"{name}.weight"))
            all_sf.append(f.get_tensor(f"{name}.weight_scale"))
            all_global_scale.append(f.get_tensor(f"{name}.weight_scale_2"))

    tensor_fp4_3d = torch.stack(all_fp4)
    tensor_sf_3d = torch.stack(all_sf)
    global_scale_3d = torch.stack(all_global_scale)

    test_cases = [
        ("2D base", tensor_fp4_2d, tensor_sf_2d, global_scale_2d),
        (
            "2D 2x rows",
            tensor_fp4_2d.repeat(2, 1),
            tensor_sf_2d.repeat(2, 1),
            global_scale_2d,
        ),
        (
            "2D 4x rows",
            tensor_fp4_2d.repeat(4, 1),
            tensor_sf_2d.repeat(4, 1),
            global_scale_2d,
        ),
        (
            "2D 2x cols",
            tensor_fp4_2d.repeat(1, 2),
            tensor_sf_2d.repeat(1, 2),
            global_scale_2d,
        ),
        ("3D base", tensor_fp4_3d, tensor_sf_3d, global_scale_3d),
        (
            "3D 2x experts",
            tensor_fp4_3d.repeat(2, 1, 1),
            tensor_sf_3d.repeat(2, 1, 1),
            global_scale_3d.repeat(2),
        ),
        (
            "3D 2x rows",
            tensor_fp4_3d.repeat(1, 2, 1),
            tensor_sf_3d.repeat(1, 2, 1),
            global_scale_3d,
        ),
        (
            "3D 2x cols",
            tensor_fp4_3d.repeat(1, 1, 2),
            tensor_sf_3d.repeat(1, 1, 2),
            global_scale_3d,
        ),
    ]

    quantiles = [0.5, 0.001, 0.999]

    # Move the E2M1 lookup table to CUDA ahead of time, as would normally
    # happen during model loading (process_weights_after_loading).  Both the
    # Triton and PyTorch reference paths run on CUDA.
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cuda()
    )

    for label, tensor_fp4, tensor_sf, global_scale in test_cases:
        fp4_cuda = tensor_fp4.cuda()
        sf_cuda = tensor_sf.cuda()
        gs_cuda = global_scale.cuda()

        # Triton path
        triton_result = dequantize_to_dtype(
            fp4_cuda,
            sf_cuda,
            gs_cuda,
            torch.bfloat16,
            block_size,
            swizzle=False,
        )

        # Reference path (PyTorch ops on CUDA, Triton dispatch disabled)
        with monkeypatch.context() as m:
            m.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            reference = dequantize_to_dtype(
                fp4_cuda,
                sf_cuda,
                gs_cuda,
                torch.bfloat16,
                block_size,
                swizzle=False,
            )

        torch.testing.assert_close(triton_result, reference, atol=0, rtol=0)

        # Benchmark
        shape = list(tensor_fp4.shape)

        def _triton_bench(
            fp4_cuda=fp4_cuda,
            scale_cuda=sf_cuda,
            global_scale_cuda=gs_cuda,
            block_size=block_size,
        ):
            return dequantize_to_dtype(
                fp4_cuda,
                scale_cuda,
                global_scale_cuda,
                torch.bfloat16,
                block_size,
                swizzle=False,
            )

        triton_ms, triton_min, triton_max = triton.testing.do_bench(
            _triton_bench, quantiles=quantiles
        )

        def _reference_bench(
            fp4_cuda=fp4_cuda,
            scale_cuda=sf_cuda,
            global_scale_cuda=gs_cuda,
            block_size=block_size,
        ):
            with monkeypatch.context() as m2:
                m2.setattr(
                    nvfp4_emulation_utils.current_platform,
                    "is_cuda_alike",
                    lambda: False,
                )
                dequantize_to_dtype(
                    fp4_cuda,
                    scale_cuda,
                    global_scale_cuda,
                    torch.bfloat16,
                    block_size,
                    swizzle=False,
                )

        ref_ms, ref_min, ref_max = triton.testing.do_bench(
            _reference_bench, quantiles=quantiles
        )

        speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"  dequantize {label} {shape}:")
        print(
            f"    triton:    median={triton_ms:.3f}ms, "
            f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
        )
        print(
            f"    reference: median={ref_ms:.3f}ms, "
            f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
        )
        print(f"    speedup:   {speedup:.2f}x")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
@pytest.mark.parametrize(
    "m, k",
    [
        (1, 16),
        (1, 4096),
        (2, 4096),
        (4, 4096),
        (8, 4096),
        (16, 4096),
        (24, 4096),
        (32, 4096),
        (1, 8192),
        (2, 8192),
        (4, 8192),
        (8, 8192),
        (16, 8192),
        (24, 8192),
        (32, 8192),
        (1, 32),
        (2, 48),
        (7, 64),
        (16, 128),
        (33, 160),
        (128, 256),
        (256, 512),
        (1024, 1024),
        (5120, 2048),
        (2048, 4096),
        (4096, 7168),
        (8192, 8192),
        (128, 16384),
    ],
)
@pytest.mark.parametrize("global_scale_value", [0.5, 1.0, 0.001])
def test_triton_nvfp4_quant_dequant(
    monkeypatch, m: int, k: int, global_scale_value: float
) -> None:
    """Test the Triton quant-dequant kernel against the CPU reference."""
    block_size = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor(global_scale_value, dtype=torch.float32, device="cuda")

    # Triton path
    triton_result = ref_nvfp4_quant_dequant(x, global_scale, block_size)

    # CPU reference path
    with monkeypatch.context() as mp:
        mp.setattr(
            nvfp4_emulation_utils.current_platform,
            "is_cuda_alike",
            lambda: False,
        )
        reference = ref_nvfp4_quant_dequant(x.cpu(), global_scale.cpu(), block_size)

    torch.testing.assert_close(triton_result.cpu(), reference, atol=0, rtol=0)

    # Benchmark (both paths on CUDA tensors for fair comparison)
    quantiles = [0.5, 0.001, 0.999]

    def _triton_bench(
        input_tensor=x, input_global_scale=global_scale, input_block_size=block_size
    ):
        return ref_nvfp4_quant_dequant(
            input_tensor, input_global_scale, input_block_size
        )

    triton_ms, triton_min, triton_max = triton.testing.do_bench(
        _triton_bench, quantiles=quantiles
    )

    def _reference_bench(
        input_tensor=x, input_global_scale=global_scale, input_block_size=block_size
    ):
        with monkeypatch.context() as mp2:
            mp2.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            ref_nvfp4_quant_dequant(input_tensor, input_global_scale, input_block_size)

    ref_ms, ref_min, ref_max = triton.testing.do_bench(
        _reference_bench, quantiles=quantiles
    )

    speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
    print(f"  quant_dequant [{m}x{k}] gs={global_scale_value}:")
    print(
        f"    triton:    median={triton_ms:.3f}ms, "
        f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
    )
    print(
        f"    reference: median={ref_ms:.3f}ms, "
        f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
    )
    print(f"    speedup:   {speedup:.2f}x")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("top_k", [2, 4, 8])
def test_nvfp4_moe_correctness(num_tokens: int, top_k: int) -> None:
    """Compare Nvfp4QuantizationEmulationTritonExperts (unfused, reference)
    against FusedNvfp4EmulationTritonExperts (fused scaffold, under test).

    Both must produce bit-identical results since the fused variant currently
    falls back to the same unfused path.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        Nvfp4QuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe_fused import (
        FusedNvfp4EmulationTritonExperts,
    )

    # ── Load real weights from checkpoint ──
    checkpoint_path = huggingface_hub.snapshot_download(
        "nvidia/Qwen3-30B-A3B-NVFP4",
        allow_patterns=["model-00001-of-00004.safetensors"],
    )
    shard_path = f"{checkpoint_path}/model-00001-of-00004.safetensors"

    expert_prefix = "model.layers.9.mlp.experts."
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        expert_indices = sorted(
            int(key.split(".")[5])
            for key in all_keys
            if key.startswith(expert_prefix)
            and key.endswith(".gate_proj.weight")
        )
        num_experts = len(expert_indices)

        gate_weights, up_weights, down_weights = [], [], []
        gate_scales, up_scales, down_scales = [], [], []
        gate_gscales, up_gscales, down_gscales = [], [], []
        a1_scales, a2_scales = [], []

        for idx in expert_indices:
            prefix = f"{expert_prefix}{idx}"
            gate_weights.append(f.get_tensor(f"{prefix}.gate_proj.weight"))
            gate_scales.append(
                f.get_tensor(f"{prefix}.gate_proj.weight_scale")
            )
            gate_gscales.append(
                f.get_tensor(f"{prefix}.gate_proj.weight_scale_2")
            )
            up_weights.append(f.get_tensor(f"{prefix}.up_proj.weight"))
            up_scales.append(
                f.get_tensor(f"{prefix}.up_proj.weight_scale")
            )
            up_gscales.append(
                f.get_tensor(f"{prefix}.up_proj.weight_scale_2")
            )
            down_weights.append(f.get_tensor(f"{prefix}.down_proj.weight"))
            down_scales.append(
                f.get_tensor(f"{prefix}.down_proj.weight_scale")
            )
            down_gscales.append(
                f.get_tensor(f"{prefix}.down_proj.weight_scale_2")
            )
            a1_scales.append(
                f.get_tensor(f"{prefix}.gate_proj.input_scale")
            )
            a2_scales.append(
                f.get_tensor(f"{prefix}.down_proj.input_scale")
            )

    # Stack into MoE format: w1 = [E, 2*intermediate, hidden//2]
    # gate_proj: [intermediate, hidden//2], up_proj: [intermediate, hidden//2]
    w1 = torch.stack([
        torch.cat([g, u], dim=0)
        for g, u in zip(gate_weights, up_weights)
    ]).cuda()
    w1_scale = torch.stack([
        torch.cat([g, u], dim=0)
        for g, u in zip(gate_scales, up_scales)
    ]).cuda()
    # Use same global scale for gate and up (they share the same value).
    w1_gscale = torch.stack(gate_gscales).cuda()

    w2 = torch.stack(down_weights).cuda()
    w2_scale = torch.stack(down_scales).cuda()
    w2_gscale = torch.stack(down_gscales).cuda()

    a13_scale_raw = torch.stack(a1_scales).cuda()
    a2_scale_raw = torch.stack(a2_scales).cuda()

    # ── Apply EMULATION transforms (matches oracle/nvfp4.py) ──
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cuda()
    )
    a1_gscale = 1.0 / a13_scale_raw.max().to(torch.float32)
    a2_gscale = 1.0 / a2_scale_raw.max().to(torch.float32)

    # ── Derive model dimensions ──
    # w1: [E, 2*intermediate, hidden//2]
    hidden_dim = w1.size(2) * 2  # packed K
    intermediate_size = w1.size(1) // 2

    # ── Build configs ──
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )

    def _make_quant_config():
        return nvfp4_moe_quant_config(
            g1_alphas=w1_gscale.clone(),
            g2_alphas=w2_gscale.clone(),
            a1_gscale=a1_gscale.clone(),
            a2_gscale=a2_gscale.clone(),
            w1_scale=w1_scale.clone(),
            w2_scale=w2_scale.clone(),
        )

    # ── Construct both expert classes ──
    ref_experts = Nvfp4QuantizationEmulationTritonExperts(
        moe_config=moe_config,
        quant_config=_make_quant_config(),
    )
    fused_experts = FusedNvfp4EmulationTritonExperts(
        moe_config=moe_config,
        quant_config=_make_quant_config(),
    )

    # ── Prepare inputs ──
    torch.manual_seed(42)
    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.randn(
        num_tokens, top_k, dtype=torch.float32, device="cuda"
    ).softmax(dim=-1)
    topk_ids = torch.stack([
        torch.randperm(num_experts, device="cuda")[:top_k]
        for _ in range(num_tokens)
    ]).to(torch.int32)

    # ── Allocate workspaces and outputs ──
    N = w1.size(1)  # 2 * intermediate
    K = hidden_dim

    ws13_size = num_tokens * top_k * max(intermediate_size, K)
    ws2_size = num_tokens * top_k * max(N, K)

    workspace13_ref = torch.zeros(
        ws13_size, dtype=torch.bfloat16, device="cuda"
    )
    workspace2_ref = torch.zeros(
        ws2_size, dtype=torch.bfloat16, device="cuda"
    )
    output_ref = torch.zeros(
        num_tokens, K, dtype=torch.bfloat16, device="cuda"
    )

    workspace13_fused = torch.zeros_like(workspace13_ref)
    workspace2_fused = torch.zeros_like(workspace2_ref)
    output_fused = torch.zeros_like(output_ref)

    # ── Run reference (unfused) ──
    ref_experts.apply(
        output=output_ref,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13_ref,
        workspace2=workspace2_ref,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # ── Run fused scaffold ──
    fused_experts.apply(
        output=output_fused,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13_fused,
        workspace2=workspace2_fused,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # ── Assert identical outputs ──
    torch.testing.assert_close(output_fused, output_ref, atol=0, rtol=0)

    # ── Benchmark ──
    quantiles = [0.5, 0.001, 0.999]

    def _ref_bench():
        workspace13_ref.zero_()
        workspace2_ref.zero_()
        output_ref.zero_()
        ref_experts.apply(
            output=output_ref,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13_ref,
            workspace2=workspace2_ref,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    def _fused_bench():
        workspace13_fused.zero_()
        workspace2_fused.zero_()
        output_fused.zero_()
        fused_experts.apply(
            output=output_fused,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13_fused,
            workspace2=workspace2_fused,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    ref_ms, ref_min, ref_max = triton.testing.do_bench(
        _ref_bench, quantiles=quantiles
    )
    fused_ms, fused_min, fused_max = triton.testing.do_bench(
        _fused_bench, quantiles=quantiles
    )

    speedup = ref_ms / fused_ms if fused_ms > 0 else float("inf")
    print(f"  MoE [tokens={num_tokens}, top_k={top_k}]:")
    print(
        f"    unfused:   median={ref_ms:.3f}ms, "
        f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
    )
    print(
        f"    fused:     median={fused_ms:.3f}ms, "
        f"min={fused_min:.3f}ms, max={fused_max:.3f}ms"
    )
    print(f"    speedup:   {speedup:.2f}x")
