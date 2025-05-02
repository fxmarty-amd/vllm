import torch

OCP_MX_BLOCK_SIZE = 32


def per_token_group_dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                                  block_k: int,
                                  float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group)
        from quark.torch.utils import pack
    except ImportError as e:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from e

    # TODO: Both arguments are unused.
    pack_method = pack.Pack_fp4(None, dtype="fp4")
    # TODO: Both 'reorder' and 'origin_packed_axis_size' are unused.
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2**(scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    # TODO: `dequantize_fp4_fp6_per_group` and `prepare_inputs_per_group` always return fp32.
    return dequantize_fp4_fp6_per_group(unpacked_x,
                                        scale,
                                        axis=-1,
                                        group_size=block_k,
                                        quant_dtype="fp4").to(float_dtype)


# TODO: technically this is quant + dequant.
def per_token_group_quant_mxfp4(x: torch.Tensor, block_k: int):
    try:
        from quark.torch.kernel import scaled_fake_quantize
        from quark.torch.quantization.utils import (even_round,
                                                    reshape_to_blocks)
    except ImportError:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`. Error: {e}")

    axis = -1
    block_x = reshape_to_blocks(x, block_k, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    # TODO: there are other rounding strategies supported in quark and in the config.json that we do not check for here!
    scale = even_round(amax, "fp4")

    # Apply dequantize(quantize(x)).
    x = scaled_fake_quantize(
        "fp4",
        x,
        scale.to(x.device),
        None,
        axis,
        block_k,
        -1.,  # TODO: useless, to make cleaner
        1.,  # TODO: useless, to make cleaner
        0,  # TODO: useless, to make cleaner
        "per_group",
        'None',  # must be a string in quark hw_emulation_interface.py, why?
    )

    return x, scale
