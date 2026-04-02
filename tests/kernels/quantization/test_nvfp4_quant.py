# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
    ref_nvfp4_quant,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
PAD_SHAPES = [
    (90, 64),
    (150, 64),
    (128, 48),
    (128, 80),
    (150, 80),
    (90, 48),
    (90, 128),
    (150, 128),
    (150, 48),
    (90, 80),
    (128, 512),
    (128, 1024),
    (128, 2048),
    (64, 7168),
    (64, 7152),
    (32, 14336),
]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
BLOCK_SIZE = 16


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def recover_swizzled_scales(scale, m, n):
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // BLOCK_SIZE
    rounded_n = round_up(scale_n, 4)
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_quantize_to_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    if not current_platform.has_device_capability(100):
        pytest.skip(
            reason="Nvfp4 Requires compute capability of 10 or above.",
            allow_module_level=True,
        )

    set_random_seed(seed)
    torch.set_default_device(device)

    m, n = shape

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale, block_size=BLOCK_SIZE)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.parametrize(
    "shape",
    [(32, 4096), (128, 4096), (1, 64), (127, 1024), (256, 16384)],
)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@torch.inference_mode()
def test_python_util_matches_cpp_allocation(
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
) -> None:
    """
    Verify that the Python utility (create_fp4_output_tensors) allocates
    tensors with the same shapes and dtypes as the C++ functional variant
    (scaled_fp4_quant_func).
    """
    if not current_platform.has_device_capability(100):
        pytest.skip(
            reason="Nvfp4 Requires compute capability of 10 or above.",
            allow_module_level=True,
        )

    from vllm._custom_ops import create_fp4_output_tensors

    torch.set_default_device("cuda:0")
    m, n = shape
    input_tensor = torch.randn((m, n), dtype=torch.bfloat16)
    input_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda:0")

    # C++ functional variant allocates internally
    cpp_out, cpp_scale = torch.ops._C.scaled_fp4_quant(
        input_tensor, input_scale, is_sf_swizzled_layout
    )

    # Python utility
    py_out, py_scale = create_fp4_output_tensors(
        m, n, torch.device("cuda:0"), is_sf_swizzled_layout
    )

    assert py_out.shape == cpp_out.shape, (
        f"Output shape mismatch: Python {py_out.shape} vs C++ {cpp_out.shape}"
    )
    assert py_out.dtype == cpp_out.dtype, (
        f"Output dtype mismatch: Python {py_out.dtype} vs C++ {cpp_out.dtype}"
    )
    assert py_scale.shape == cpp_scale.shape, (
        f"Scale shape mismatch: Python {py_scale.shape} vs C++ {cpp_scale.shape}"
    )
    assert py_scale.dtype == cpp_scale.dtype, (
        f"Scale dtype mismatch: Python {py_scale.dtype} vs C++ {cpp_scale.dtype}"
    )


@pytest.mark.parametrize("pad_shape", PAD_SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4_padded(pad_shape: tuple[int, int]) -> None:
    if not current_platform.has_device_capability(100):
        pytest.skip(
            reason="Nvfp4 Requires compute capability of 10 or above.",
            allow_module_level=True,
        )

    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = pad_shape

    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale, block_size=BLOCK_SIZE)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.parametrize("pad_shape", PAD_SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4_padded_no_sf_swizzled(pad_shape: tuple[int, int]) -> None:
    if not current_platform.has_device_capability(100):
        pytest.skip(
            reason="Nvfp4 Requires compute capability of 10 or above.",
            allow_module_level=True,
        )

    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = pad_shape

    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale, block_size=BLOCK_SIZE)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale, is_sf_swizzled_layout=False)
    scale_ans = out_scale.to(torch.float32)
    out_ans = cast_from_fp4(out, m, n)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_ref_nvfp4_quant_native_vs_emulated(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """Test that native FP8 QDQ (using torch.float8) and implementation not
    using torch.float8 produce equivalent results.
    """
    torch.set_default_device("cuda:0")

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # Force native FP8 path (sm_90+)
    with patch.object(
        current_platform, "get_device_capability", return_value=DeviceCapability(9, 0)
    ):
        out_native, scale_native = ref_nvfp4_quant(x, global_scale, BLOCK_SIZE)

    # Force emulated FP8 path (sm_80)
    with patch.object(
        current_platform, "get_device_capability", return_value=DeviceCapability(8, 0)
    ):
        out_emulated, scale_emulated = ref_nvfp4_quant(x, global_scale, BLOCK_SIZE)

    # Both implementations should produce identical results
    torch.testing.assert_close(out_native, out_emulated, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scale_native, scale_emulated, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("swizzle", [True, False])
@torch.inference_mode()
def test_dequantize_to_dtype_native_vs_emulated(
    dtype: torch.dtype,
    shape: tuple[int, int],
    swizzle: bool,
) -> None:
    """Test that FP8 dequantization using torch.float8 / not using it
    produce equivalent results.
    """
    torch.set_default_device("cuda:0")

    kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to("cuda:0")

    m, n = shape
    global_scale = torch.tensor(1.0, dtype=torch.float32)

    # Create synthetic quantized FP4 data (uint8, packed)
    # Two FP4 values are packed into one uint8
    packed_n = n // 2
    out_fp4 = torch.randint(0, 256, (m, packed_n), dtype=torch.uint8, device="cuda:0")

    # Create synthetic FP8 E4M3 scale factors (uint8)
    # Avoid creating NaN/Inf values: in FP8 E4M3, exponent=15 represents NaN/Inf
    # Exponent is bits 3-6, so we create values with exponent < 15
    scale_m = m
    scale_n = n // BLOCK_SIZE

    def create_valid_fp8_e4m3_bytes(shape):
        """
        Create random uint8 values that represent valid (non-NaN/Inf)
        FP8 E4M3 values.
        """
        result = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda:0")
        # Filter out NaN/Inf: exponent = (byte >> 3) & 0xF should not be 15
        exponent = (result >> 3) & 0xF
        invalid_mask = exponent == 15
        # Replace invalid values with valid ones (exponent != 15)
        while invalid_mask.any():
            result[invalid_mask] = torch.randint(
                0, 256, (invalid_mask.sum(),), dtype=torch.uint8, device="cuda:0"
            )
            exponent = (result >> 3) & 0xF
            invalid_mask = exponent == 15
        return result

    if swizzle:
        # Create swizzled scale layout
        round_up = lambda x, y: (x + y - 1) // y * y
        rounded_m = round_up(m, 128)
        rounded_n = round_up(scale_n, 4)
        m_tiles = rounded_m // 128
        k_tiles = rounded_n // 4
        out_scale = create_valid_fp8_e4m3_bytes((1, m_tiles, k_tiles, 32, 4, 4))
    else:
        # Non-swizzled layout
        out_scale = create_valid_fp8_e4m3_bytes((scale_m, scale_n))

    # Dequantize using native FP8 path (CUDA sm_90+ or ROCm MI300+)
    with patch.object(current_platform, "has_device_capability", return_value=True):
        dequant_native = dequantize_to_dtype(
            out_fp4,
            out_scale,
            global_scale,
            dtype,
            BLOCK_SIZE,
            swizzle=swizzle,
        )

    # Dequantize using emulated FP8 path (older devices)
    with patch.object(current_platform, "has_device_capability", return_value=False):
        dequant_emulated = dequantize_to_dtype(
            out_fp4,
            out_scale,
            global_scale,
            dtype,
            BLOCK_SIZE,
            swizzle=swizzle,
        )

    # Both implementations should produce identical results
    torch.testing.assert_close(dequant_native, dequant_emulated, rtol=1e-5, atol=1e-5)
