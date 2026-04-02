# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

__all__ = [
    "break_fp4_bytes",
    "dequantize_to_dtype",
    "ref_nvfp4_quant",
]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT4_E2M1_MAX_RECIPROCAL = 1 / FLOAT4_E2M1_MAX

# FP8 constants for non-FP8 emulation
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0
EPS = 1e-12
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2

kE2M1ToFloat_handle = SimpleNamespace(
    val=torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
)


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles
    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()
    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)

    kE2M1 = kE2M1ToFloat_handle.val
    # Device-aware lookup and sign application
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def decode_fp8_e4m3_bytes(tensor_bytes: torch.Tensor) -> torch.Tensor:
    """Decode FP8 E4M3 bytes to float32 without using torch.float8_e4m3fn.

    Args:
        tensor_bytes: Tensor with dtype uint8 containing FP8 E4M3 encoded values.

    Returns:
        Tensor with dtype float32 containing decoded values.
    """
    # FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    # This is a manual decoding to avoid .view(torch.float8_e4m3fn)
    assert tensor_bytes.dtype == torch.uint8

    # Extract sign, exponent, and mantissa
    sign = (tensor_bytes >> 7) & 0x1  # 1 bit
    exponent = (tensor_bytes >> 3) & 0xF  # 4 bits
    mantissa = tensor_bytes & 0x7  # 3 bits

    # Convert to float32
    result = torch.zeros_like(tensor_bytes, dtype=torch.float32)

    # Handle special cases and normal numbers
    # Exponent bias for E4M3 is 7
    bias = 7

    # Subnormal numbers (exponent = 0, mantissa != 0)
    subnormal_mask = (exponent == 0) & (mantissa != 0)
    # Subnormal: (-1)^sign * 2^(-6) * (mantissa / 8)
    result[subnormal_mask] = (2.0 ** (-6)) * (mantissa[subnormal_mask].float() / 8.0)

    # Normal numbers (exponent != 0 and not NaN/Inf)
    normal_mask = (exponent != 0) & (exponent != 0xF)
    # Normal: (-1)^sign * 2^(exponent - bias) * (1 + mantissa/8)
    result[normal_mask] = (2.0 ** (exponent[normal_mask].float() - bias)) * (
        1.0 + mantissa[normal_mask].float() / 8.0
    )

    # NaN and Inf (exponent = 15)
    inf_nan_mask = exponent == 0xF
    # If mantissa = 0, it's infinity, otherwise NaN
    inf_mask = inf_nan_mask & (mantissa == 0)
    nan_mask = inf_nan_mask & (mantissa != 0)
    result[inf_mask] = float("inf")
    result[nan_mask] = float("nan")

    # Apply sign
    result[sign == 1] = -result[sign == 1]

    return result


def dequantize_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor | float,
    dtype: torch.dtype,
    block_size: int = 16,
    swizzle: bool | None = True,
):
    """Dequantize the fp4 tensor back to high precision.

    Dispatches to native FP8 operations or emulated FP8 based on device capability.
    """
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)

    # Dispatch based on device capability
    # Use native FP8 if:
    # - CUDA: device capability >= 9.0 (sm_90+, Hopper)
    # - ROCm: device capability >= 9.4 (MI300+, gfx942+)
    if current_platform.is_rocm():
        use_native_fp8 = current_platform.has_device_capability(94, device_id=tensor_fp4.device.index)
    else:
        use_native_fp8 = current_platform.has_device_capability(90, device_id=tensor_fp4.device.index)

    if use_native_fp8:
        # Use native FP8 operations
        tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
        if swizzle:
            tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
        tensor_sf_dtype = tensor_sf.to(torch.float32) * global_scale
    else:
        # Use emulated FP8 decoding
        tensor_sf_bytes = tensor_sf.view(torch.uint8)
        if swizzle:
            tensor_sf_bytes = convert_swizzled_to_linear(
                tensor_sf_bytes, m, k, block_size
            )
        tensor_sf_f32 = decode_fp8_e4m3_bytes(tensor_sf_bytes)
        tensor_sf_dtype = tensor_sf_f32 * global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype)


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        # torch.where yields operation not permitted when stream is capturing.
        return 1.0 / (x + (x == 0) * 1e8)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


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


def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    scale.copy_(res)
    return scale


def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    """Compute the scale for converting a tensor to FP8.
    Args:
        x: The input tensor.
        float8_dtype: The target float8 dtype.
        dim: The dimension along which to compute the scale.
    """
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values

    return amax_to_scale(amax, float8_dtype, x.dtype)


def to_fp8_saturated(x: torch.Tensor, fp8_dtype: torch.dtype):
    """Clamp and convert a tensor to FP8.
    Args:
        x: The input tensor.
        fp8_dtype: The target float8 dtype.
    """
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)


def float8_qdq_native(scale: torch.Tensor) -> torch.Tensor:
    """FP8 quantize-dequantize using native torch.float8_e4m3fn operations.

    Args:
        scale: Input tensor to quantize and dequantize through FP8.

    Returns:
        Tensor after FP8 quantize-dequantize round-trip.
    """
    return scale.to(torch.float8_e4m3fn).to(torch.float32)


def float8_qdq_no_fp8(scale: torch.Tensor) -> torch.Tensor:
    """FP8 quantize-dequantize without using torch.float8_e4m3fn.

    Emulates the behavior of scale.to(torch.float8_e4m3fn).to(torch.float32)
    using manual encoding/decoding.

    Args:
        scale: Input tensor to quantize and dequantize through FP8.

    Returns:
        Tensor after FP8 quantize-dequantize round-trip.
    """
    # Clamp to FP8 E4M3 range
    scale_clamped = torch.clamp(scale, min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)

    # Compute scale for FP8 conversion
    fp8_scale = amax_to_scale(
        torch.max(torch.abs(scale_clamped)),
        e4m3_type,
        scale.dtype
    )

    # Emulate: scale -> fp8 -> fp32
    scale_fp8_emulated = (
        to_fp8_saturated(scale_clamped * fp8_scale, e4m3_type).to(torch.float32)
        / fp8_scale
    )

    return scale_fp8_emulated


def ref_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor, block_size: int):
    """Reference NVFP4 quantization.

    Dispatches to native FP8 operations or emulated FP8 based on device capability.

    Args:
        x: Input tensor to quantize (2D).
        global_scale: Global scaling factor (float32).
        block_size: Size of blocks for per-block scaling.

    Returns:
        Tuple of (quantized_fp4, scale):
            - quantized_fp4: FP4 quantized values (float32)
            - scale: Per-block scales (float32)
    """
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)

    # Compute scale before FP8 quantize-dequantize
    scale = global_scale * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, max=448, min=-448)

    # Dispatch to native or emulated FP8 quantize-dequantize
    device_capability = current_platform.get_device_capability(device_id=x.device.index)

    # Use native FP8 if:
    # - CUDA: device capability >= 9.0 (sm_90+)
    # - ROCm: device capability >= 9.4 (MI300+, gfx942+)
    if current_platform.is_rocm():
        use_native_fp8 = (
            device_capability is not None and device_capability.to_int() >= 94
        )
    else:
        use_native_fp8 = (
            device_capability is not None and device_capability.to_int() >= 90
        )

    if use_native_fp8:
        scale = float8_qdq_native(scale)
    else:
        scale = float8_qdq_no_fp8(scale)

    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    # both outputs are float32
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out
