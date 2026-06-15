# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

__all__ = [
    "break_fp4_bytes",
    "dequantize_to_dtype",
    "fused_nvfp4_dequant_gemm",
    "ref_nvfp4_quant",
    "triton_bf16_gemm",
]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT4_E2M1_MAX_RECIPROCAL = 1 / FLOAT4_E2M1_MAX

kE2M1ToFloat_handle = SimpleNamespace(
    val=torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
)


@triton.jit
def _e2m1_inline(nibble):
    """Decode an NVFP4 nibble (4 bits: 1 sign + 3 magnitude) to float32.

    Uses direct IEEE 754 bit construction.
    For magnitudes 2-7 the FP32 bit pattern is 0x3F000000 + (mag << 22),
    which is a single shift + add + bitcast.  Magnitudes 0 (zero) and 1
    (E2M1 subnormal = 0.5) are patched with two tl.where ops.
    """
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1

    fp32_bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
    val = fp32_bits.to(tl.float32, bitcast=True)

    val = tl.where(magnitude == 0, 0.0, val)
    val = tl.where(magnitude == 1, 0.5, val)

    return tl.where(sign == 1, -val, val)


@triton.jit
def _dequantize_nvfp4_kernel(
    fp4_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    rows_per_batch: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_batch_global_scale: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
):
    """Triton kernel for NVFP4 dequantization (swizzle=False).

    Optimized with 2D tile processing + interleave for coalesced stores.
    """
    BLOCK_PACKED: tl.constexpr = BLOCK_SIZE // 2

    row_idx = tl.program_id(0).to(tl.int64)
    tile_idx = tl.program_id(1)

    if has_batch_global_scale:
        batch_idx = row_idx // rows_per_batch
        global_scale = tl.load(global_scale_ptr + batch_idx).to(tl.float32)
    else:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

    fp4_row_offset = row_idx * num_blocks * BLOCK_PACKED
    scale_row_offset = row_idx * num_blocks
    output_row_offset = row_idx * num_blocks * BLOCK_SIZE

    start_block = tile_idx * TILE_BLOCKS

    # Load scales for this tile: [TILE_BLOCKS]
    block_offsets = tl.arange(0, TILE_BLOCKS)
    block_mask = (start_block + block_offsets) < num_blocks

    raw_scales = tl.load(
        scale_ptr + scale_row_offset + start_block + block_offsets,
        mask=block_mask,
        other=0,
    )
    scale_f32 = tl.cast(raw_scales, tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_values = (scale_f32 * global_scale)[:, None]

    # Load [TILE_BLOCKS, BLOCK_PACKED] packed bytes
    packed_offsets = tl.arange(0, BLOCK_PACKED)[None, :]
    byte_indices = (
        fp4_row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_PACKED
        + packed_offsets
    )
    elem_mask = block_mask[:, None]
    raw_bytes = tl.load(fp4_ptr + byte_indices, mask=elem_mask, other=0)

    low_nibble = raw_bytes & 0x0F
    high_nibble = (raw_bytes >> 4) & 0x0F

    low_result = _e2m1_inline(low_nibble) * scale_values
    high_result = _e2m1_inline(high_nibble) * scale_values

    # Interleave for coalesced contiguous store
    result = tl.interleave(low_result, high_result)

    elem_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
    out_indices = (
        output_row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_SIZE
        + elem_offsets
    )
    tl.store(output_ptr + out_indices, result, mask=block_mask[:, None])


@triton.jit
def _e2m1_lookup(magnitude):
    """Lookup E2M1 float value from 3-bit magnitude."""
    result = tl.where(magnitude == 1, 0.5, 0.0)
    result = tl.where(magnitude == 2, 1.0, result)
    result = tl.where(magnitude == 3, 1.5, result)
    result = tl.where(magnitude == 4, 2.0, result)
    result = tl.where(magnitude == 5, 3.0, result)
    result = tl.where(magnitude == 6, 4.0, result)
    result = tl.where(magnitude == 7, 6.0, result)
    return result


@triton.jit
def _round_to_fp4(x):
    """Round float values to the nearest E2M1 representable value.

    Matches the thresholds in the Python ``cast_to_fp4`` exactly.
    """
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)
    result = tl.where(abs_x > 5.0, 6.0, 0.0)
    result = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, result)
    result = tl.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, result)
    result = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, result)
    result = tl.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, result)
    result = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, result)
    result = tl.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, result)
    return result * sign


@triton.jit
def _nvfp4_quant_dequant_kernel(
    input_ptr,
    output_ptr,
    global_scale_ptr,
    k: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP4_MAX_RECIPROCAL: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
):
    """Fused NVFP4 quantize-dequantize kernel.

    Uses a 2D grid (rows x tiles) to parallelize across both rows
    and quantization groups within a row. Each program handles
    TILE_BLOCKS groups at once using vectorized 2D operations.
    """
    row_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    row_offset = row_idx * k

    start_block = tile_idx * TILE_BLOCKS
    block_offsets = tl.arange(0, TILE_BLOCKS)
    block_mask = (start_block + block_offsets) < num_blocks

    # Load [TILE_BLOCKS, BLOCK_SIZE] elements
    indices = (
        row_offset
        + (start_block + block_offsets[:, None]) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    mask_2d = block_mask[:, None]
    x = tl.load(input_ptr + indices, mask=mask_2d, other=0.0).to(tl.float32)

    # Per-group scale: [TILE_BLOCKS]
    vec_max = tl.max(tl.abs(x), axis=1)
    scale = global_scale * (vec_max * FP4_MAX_RECIPROCAL)
    scale = tl.clamp(scale, -448.0, 448.0)
    scale = scale.to(tl.float8e4nv).to(tl.float32)

    # Safe reciprocal, broadcast to [TILE_BLOCKS, 1]
    output_scale = tl.where(scale == 0.0, 0.0, global_scale / scale)[:, None]

    # Quantize: scale, clamp, round to FP4
    scaled_x = tl.clamp(x * output_scale, -6.0, 6.0)
    fp4_val = _round_to_fp4(scaled_x)

    # Dequantize: fp4_val * (scale / global_scale)
    dequant_scale = (scale / global_scale)[:, None]
    result = fp4_val * dequant_scale

    tl.store(output_ptr + indices, result, mask=mask_2d)


def _triton_nvfp4_quant_dequant(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton-accelerated NVFP4 quantize-dequantize."""
    x_m, x_k = x.shape

    if not torch.compiler.is_compiling():
        assert x_k % block_size == 0, (
            f"Weight shape K={x_k} is not divisible by block_size={block_size}"
        )

    output_dtype = x.dtype
    num_blocks = x_k // block_size

    output = torch.empty(x_m, x_k, dtype=output_dtype, device=x.device)

    tile_blocks = min(64, triton.next_power_of_2(num_blocks))
    num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
    grid = (x_m, num_tiles)
    _nvfp4_quant_dequant_kernel[grid](
        x,
        output,
        global_scale,
        x_k,
        num_blocks,
        block_size,
        FLOAT4_E2M1_MAX_RECIPROCAL,
        tile_blocks,
    )

    return output


def _triton_dequantize_nvfp4(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 using Triton (swizzle=False only).

    Supports both 2D and 3D inputs:
    - 2D: [m, packed_k] -> [m, k]
    - 3D: [dim0, m, packed_k] -> [dim0, m, k]
    """
    assert tensor_fp4.dtype == torch.uint8

    is_3d = tensor_fp4.ndim == 3
    if is_3d:
        dim0, m_per_batch, packed_k = tensor_fp4.shape
        tensor_fp4_2d = tensor_fp4.reshape(-1, packed_k)
        tensor_sf_2d = tensor_sf.reshape(-1, tensor_sf.shape[-1])
        total_rows_flat = dim0 * m_per_batch
    else:
        m_per_batch, packed_k = tensor_fp4.shape
        tensor_fp4_2d = tensor_fp4
        tensor_sf_2d = tensor_sf
        total_rows_flat = m_per_batch

    k = packed_k * 2
    num_blocks = k // block_size

    output = torch.empty(total_rows_flat, k, dtype=dtype, device=tensor_fp4.device)

    # View as uint8 so Triton can load raw bytes and bitcast to float8_e4m3fn
    scale_raw = tensor_sf_2d.contiguous().view(torch.uint8)

    # Shape-adaptive tile sizing: for large row counts (3D), process
    # entire row in one tile. For small row counts (2D), use smaller
    # tiles to increase parallelism across CUs.
    np2 = triton.next_power_of_2(num_blocks)
    if total_rows_flat >= 4096:
        # Many rows: maximize work per CTA, one tile per row
        tile_blocks = np2
        nw = 1
        ns = 2
    elif total_rows_flat >= 2048:
        # Medium-many rows: full row, 2 warps
        tile_blocks = np2
        nw = 2
        ns = 2
    else:
        # Few rows: use moderate tiles for CU utilization
        tile_blocks = min(64, np2)
        nw = 4
        ns = 2
    num_tiles = (num_blocks + tile_blocks - 1) // tile_blocks
    grid = (total_rows_flat, num_tiles)
    _dequantize_nvfp4_kernel[grid](
        tensor_fp4_2d,
        scale_raw,
        global_scale,
        output,
        m_per_batch,
        num_blocks,
        block_size,
        is_3d,
        tile_blocks,
        num_warps=nw,
        num_stages=ns,
    )

    if is_3d:
        output = output.reshape(dim0, m_per_batch, k)

    return output


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


def dequantize_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 16,
    swizzle: bool | None = True,
):
    """Dequantize the fp4 tensor back to high precision.

    Supports both 2D and 3D inputs:
    - 2D: [m, packed_k] -> [m, k]
    - 3D: [dim0, m, packed_k] -> [dim0, m, k]
    """
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8

    if not swizzle and current_platform.is_cuda_alike():
        return _triton_dequantize_nvfp4(
            tensor_fp4, tensor_sf, global_scale, dtype, block_size
        )

    # We handle 3D tensors reshaping them to 2D.
    is_3d = tensor_fp4.ndim == 3

    if is_3d:
        dim0, m, packed_k = tensor_fp4.shape
        tensor_fp4 = tensor_fp4.reshape(-1, packed_k)
        tensor_sf = tensor_sf.reshape(-1, tensor_sf.shape[-1])
        global_scale = global_scale[:, None, None]
    else:
        m, packed_k = tensor_fp4.shape

    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(-1, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)

    if swizzle:
        tensor_sf = convert_swizzled_to_linear(  # noqa: E501
            tensor_sf, tensor_f32.size(0), k, block_size
        )

    if is_3d:
        tensor_sf = tensor_sf.reshape(dim0, m, k // block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) * global_scale

    if is_3d:
        tensor_f32 = tensor_f32.reshape(dim0, m, -1, block_size)

    # scale the tensor
    out = tensor_f32 * tensor_sf_dtype.unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], -1)

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


def ref_nvfp4_quant(x, global_scale, block_size):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, max=448, min=-448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    # both outputs are float32
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def ref_nvfp4_quant_dequant(
    x: torch.Tensor, global_scale: torch.Tensor, block_size: int
) -> torch.Tensor:
    """
    NVFP4 quantize-dequantize operation.

    `global_scale` is expected to have a single element.
    """
    if current_platform.is_cuda_alike():
        return _triton_nvfp4_quant_dequant(x, global_scale, block_size)

    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, global_scale, block_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // block_size, block_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)

    return x_dq


@triton.jit
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def _pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


_bf16_gemm_configs = [
    triton.Config(
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
         "GROUP_SIZE_M": gm, "NUM_KSPLIT": ks},
        num_stages=ns, num_warps=nw)
    for bm, bn, bk, gm, ks, ns, nw in [
        (16, 64, 64, 8, 1, 4, 4),
        (32, 64, 64, 8, 1, 4, 4),
        (64, 64, 64, 8, 1, 4, 4),
        (128, 64, 64, 8, 1, 4, 4),
        (64, 128, 64, 8, 1, 4, 8),
        (128, 128, 64, 8, 1, 3, 8),
        (64, 256, 64, 8, 1, 3, 8),
        (128, 256, 64, 8, 1, 3, 8),
        # Split-K configs for small M
        (16, 64, 64, 1, 4, 4, 4),
        (16, 64, 64, 1, 8, 4, 4),
        (16, 64, 64, 1, 16, 4, 4),
        (16, 128, 64, 1, 4, 4, 4),
        (16, 128, 64, 1, 8, 4, 4),
        (32, 64, 64, 1, 4, 4, 4),
        (32, 64, 64, 1, 8, 4, 4),
        (32, 128, 64, 1, 4, 4, 4),
        (64, 64, 64, 1, 4, 4, 4),
    ]
]


@triton.autotune(configs=_bf16_gemm_configs, key=["M", "N", "K"])
@triton.heuristics({
    "SPLITK_BLOCK_SIZE": lambda args: triton.cdiv(
        args["K"], args["NUM_KSPLIT"]
    ),
    "EVEN_K": lambda args: (
        triton.cdiv(args["K"], args["NUM_KSPLIT"])
        % args["BLOCK_SIZE_K"] == 0
    ),
    "EVEN_MN": lambda args: (
        args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % args["BLOCK_SIZE_N"] == 0
    ),
})
@triton.jit(do_not_specialize=["M", "N"])
def _triton_bf16_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
):
    """Reference Triton BF16 GEMM: C[M,N] = A[M,K] @ B[K,N]."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ck > 0)

    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = _remap_xcd(pid_unified, num_pid_m * num_pid_n * NUM_KSPLIT)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
        pid_m, pid_n = _pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        if EVEN_MN:
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                k_mask_1d = offs_k < k_span - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs, mask=k_mask_1d[None, :], other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=k_mask_1d[:, None], other=0.0
                )
            accumulator = tl.dot(a, b, acc=accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        if EVEN_MN:
            tl.store(c_ptrs, c)
        else:
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _splitk_reduce_kernel(
    partials_ptr,
    output_ptr,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    ACTUAL_KSPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT_POW2: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(NUM_KSPLIT_POW2):
        if k < ACTUAL_KSPLIT:
            p = tl.load(
                partials_ptr
                + k * stride_pk
                + offs_m[:, None] * stride_pm
                + offs_n[None, :] * stride_pn,
                mask=mask,
                other=0.0,
            )
            acc += p

    tl.store(
        output_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc.to(output_ptr.type.element_ty),
        mask=mask,
    )


def triton_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Reference Triton BF16 GEMM: output[M,N] = x[M,K] @ weight[N,K]^T."""
    M, K = x.shape
    N = weight.shape[0]
    w = weight.T.contiguous()
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    max_ksplit = max(c.kwargs.get("NUM_KSPLIT", 1) for c in _bf16_gemm_configs)

    partials = torch.empty(
        (max_ksplit, M, N), dtype=torch.float32, device=x.device
    )

    grid = lambda META: (
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _triton_bf16_gemm_kernel[grid](
        x, w, partials,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        partials.stride(0), partials.stride(1), partials.stride(2),
    )

    chosen_ksplit = _triton_bf16_gemm_kernel.best_config.kwargs.get(
        "NUM_KSPLIT", 1
    )
    if chosen_ksplit == 1:
        output = partials[0].to(x.dtype)
    else:
        splitk_block_size = triton.cdiv(K, chosen_ksplit)
        actual_ksplit = triton.cdiv(K, splitk_block_size)
        REDUCE_BLOCK_M = 32
        REDUCE_BLOCK_N = 32
        reduce_grid = (
            triton.cdiv(M, REDUCE_BLOCK_M),
            triton.cdiv(N, REDUCE_BLOCK_N),
        )
        _splitk_reduce_kernel[reduce_grid](
            partials, output,
            M, N,
            partials.stride(0), partials.stride(1), partials.stride(2),
            output.stride(0), output.stride(1),
            actual_ksplit,
            REDUCE_BLOCK_M, REDUCE_BLOCK_N,
            triton.next_power_of_2(chosen_ksplit),
        )

    return output


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_nvfp4_dequant_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    w_global_scale_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    block_k_diviable: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    group_size: tl.constexpr,
):
    """Fused NVFP4 weight dequantization + BF16 GEMM.

    Computes C[M, N] = A[M, K] @ dequant(B_packed[N, K//2])^T.
    """
    BLOCK_SIZE_K_PACKED: tl.constexpr = BLOCK_SIZE_K // 2

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K_PACKED)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    b_ptrs = (
        b_ptr + offs_bn[:, None] * stride_bn + offs_k_packed[None, :] * stride_bk
    )

    group_size_packed: tl.constexpr = group_size // 2

    w_global_scale = tl.load(w_global_scale_ptr).to(tl.float32)

    m_mask = offs_am[:, None] < M

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if block_k_diviable:
            a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=m_mask & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        if block_k_diviable:
            raw_bytes = tl.load(b_ptrs)
        else:
            kp_mask = offs_k_packed[None, :] < (K // 2) - k * BLOCK_SIZE_K_PACKED
            raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)

        low_nibble = raw_bytes & 0x0F
        high_nibble = (raw_bytes >> 4) & 0x0F

        low_decoded = _e2m1_inline(low_nibble)
        high_decoded = _e2m1_inline(high_nibble)

        b_scale_ptrs = (
            b_scale_ptr
            + offs_bn[:, None] * stride_bsn
            + (
                (offs_k_packed[None, :] + BLOCK_SIZE_K_PACKED * k)
                // group_size_packed
            )
            * stride_bsk
        )
        if block_k_diviable:
            b_scale_raw = tl.load(b_scale_ptrs)
        else:
            b_scale_raw = tl.load(b_scale_ptrs, mask=kp_mask, other=0.0)

        b_scale = tl.cast(b_scale_raw, tl.float8e4nv, bitcast=True).to(
            tl.float32
        )
        b_scale = b_scale * w_global_scale

        low_scaled = low_decoded * b_scale
        high_scaled = high_decoded * b_scale

        b = tl.trans(tl.interleave(low_scaled, high_scaled)).to(tl.bfloat16)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_PACKED * stride_bk

    c = accumulator.to(tl.bfloat16)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_nvfp4_dequant_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Fused NVFP4 weight dequantization + BF16 GEMM.

    Args:
        x: Activations [M, K] in bf16.
        weight_packed: Packed NVFP4 weights [N, K//2] as uint8.
        weight_scale: Per-block scales [N, K//block_size] as uint8
                      (fp8_e4m3fn view).
        weight_global_scale: Scalar global scale, float32.
        block_size: Quantization group size (default 16).

    Returns:
        Output [M, N] in bf16.
    """
    M, K = x.shape
    N = weight_packed.shape[0]

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    scale_raw = weight_scale.contiguous().view(torch.uint8)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _fused_nvfp4_dequant_gemm_kernel[grid](
        x,
        weight_packed,
        output,
        scale_raw,
        weight_global_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight_packed.stride(0),
        weight_packed.stride(1),
        output.stride(0),
        output.stride(1),
        scale_raw.stride(0),
        scale_raw.stride(1),
        block_k_diviable=K % 64 == 0,
        group_size=block_size,
    )

    return output


def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    output_dtype = x.dtype
    group_size = 16

    x_dq = ref_nvfp4_quant_dequant(x, input_global_scale, block_size=group_size)

    if not swizzle and current_platform.is_cuda_alike():
        w_fp4 = weight.data.view(torch.uint8)
        out = fused_nvfp4_dequant_gemm(
            x_dq,
            w_fp4,
            weight_scale_swizzled.data,
            weight_global_scale,
            block_size=group_size,
        )
        return out

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
    return out
