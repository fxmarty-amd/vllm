# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import huggingface_hub
import pytest
import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils import (
    nvfp4_emulation_utils,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


@pytest.fixture(scope="module")
def dequant_test_cases():
    """Load real NVFP4 weights and build 2D/3D test cases."""
    checkpoint_path = huggingface_hub.snapshot_download(
        "nvidia/Qwen3-30B-A3B-NVFP4",
        allow_patterns=["model-00001-of-00004.safetensors"],
    )
    shard_path = f"{checkpoint_path}/model-00001-of-00004.safetensors"

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        tensor_fp4_2d = f.get_tensor(
            "model.layers.9.self_attn.k_proj.weight")
        tensor_sf_2d = f.get_tensor(
            "model.layers.9.self_attn.k_proj.weight_scale")
        global_scale_2d = f.get_tensor(
            "model.layers.9.self_attn.k_proj.weight_scale_2")

        expert_prefix = "model.layers.9.mlp.experts."
        expert_indices = sorted(
            int(key.split(".")[5])
            for key in all_keys
            if key.startswith(expert_prefix)
            and key.endswith(".up_proj.weight")
        )
        assert len(expert_indices) > 0

        all_fp4, all_sf, all_global_scale = [], [], []
        for index in expert_indices:
            name = f"{expert_prefix}{index}.up_proj"
            all_fp4.append(f.get_tensor(f"{name}.weight"))
            all_sf.append(f.get_tensor(f"{name}.weight_scale"))
            all_global_scale.append(f.get_tensor(f"{name}.weight_scale_2"))

    tensor_fp4_3d = torch.stack(all_fp4)
    tensor_sf_3d = torch.stack(all_sf)
    global_scale_3d = torch.stack(all_global_scale)

    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cuda()
    )

    return [
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


BLOCK_SIZE = 16


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_triton_dequantize_nvfp4_correctness(
    monkeypatch, dequant_test_cases
) -> None:
    """Test the Triton dequantization kernel against the PyTorch reference
    using real NVFP4 weights from a checkpoint.

    Tests both 2D (attention projection) and 3D (stacked MoE experts).
    """
    for label, tensor_fp4, tensor_sf, global_scale in dequant_test_cases:
        fp4_cuda = tensor_fp4.cuda()
        sf_cuda = tensor_sf.cuda()
        gs_cuda = global_scale.cuda()

        triton_result = dequantize_to_dtype(
            fp4_cuda, sf_cuda, gs_cuda,
            torch.bfloat16, BLOCK_SIZE, swizzle=False,
        )

        with monkeypatch.context() as m:
            m.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            reference = dequantize_to_dtype(
                fp4_cuda, sf_cuda, gs_cuda,
                torch.bfloat16, BLOCK_SIZE, swizzle=False,
            )

        torch.testing.assert_close(triton_result, reference, atol=0, rtol=0)
        print(f"  correctness OK: {label} {list(tensor_fp4.shape)}")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_triton_dequantize_nvfp4_benchmark(dequant_test_cases) -> None:
    """Benchmark the Triton dequantization kernel on real NVFP4 weights."""
    quantiles = [0.5, 0.001, 0.999]
    all_medians: list[float] = []

    for label, tensor_fp4, tensor_sf, global_scale in dequant_test_cases:
        fp4_cuda = tensor_fp4.cuda()
        sf_cuda = tensor_sf.cuda()
        gs_cuda = global_scale.cuda()
        shape = tuple(tensor_fp4.shape)

        def _bench(
            fp4_cuda=fp4_cuda,
            scale_cuda=sf_cuda,
            global_scale_cuda=gs_cuda,
        ):
            return dequantize_to_dtype(
                fp4_cuda, scale_cuda, global_scale_cuda,
                torch.bfloat16, BLOCK_SIZE, swizzle=False,
            )

        median_ms, min_ms, max_ms = triton.testing.do_bench(
            _bench, quantiles=quantiles,
        )
        all_medians.append(median_ms)

        print(f"  dequantize {label} {shape}:")
        print(
            f"    triton: median={median_ms:.3f}ms, "
            f"min={min_ms:.3f}ms, max={max_ms:.3f}ms"
        )
        print(f"  {shape}: {median_ms:.4f} ms")

    geomean_ms = math.exp(sum(math.log(m) for m in all_medians) / len(all_medians))
    print(f"GEAK_RESULT_LATENCY_MS={geomean_ms:.4f}")
