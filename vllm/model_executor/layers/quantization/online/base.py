# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_layer_name_after_index,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Static,
)

logger = init_logger(__name__)


# Online dispatch tables, keyed by the QuantSpec.weight QuantKey. The
# corresponding method class handles the activation choice via its
# `supported_activation_quant` set.
_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineLinearMethod,
    kMxfp8Dynamic: Mxfp8OnlineLinearMethod,
    kMxfp4Static: Mxfp4OnlineLinearMethod,
}

_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kFp8StaticTensorSym: Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kFp8StaticChannelSym: Fp8PtpcOnlineMoEMethod,
    kMxfp8Dynamic: Mxfp8OnlineMoEMethod,
    kMxfp4Static: Mxfp4OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
    kNvfp4Static: Nvfp4OnlineMoEMethod,
}


class OnlineQuantizationConfig(QuantizationConfig):
    """Model-level config for online quantization (quantize fp16/bf16 weights
    during model loading, without requiring a pre-quantized checkpoint)."""

    def __init__(
        self,
        args: QuantizationConfigArgs,
    ) -> None:
        super().__init__()
        if args.linear is None and args.moe is None:
            raise ValueError(
                "OnlineQuantizationConfig requires at least one of "
                "quantization_config.linear or quantization_config.moe "
                "to be set."
            )
        self.args = args
        self.ignored_layers: list[str] = args.ignore
        self.quantized_layers: dict[str, tuple[str, str, str | None]] = {}

    @property
    def quantized_layer_summaries(self) -> list[str]:
        counts: dict[tuple[str, str, str | None, str], int] = {}
        for layer_name, (
            source,
            quant_key_str,
            target_pattern,
        ) in self.quantized_layers.items():
            key = (
                get_layer_name_after_index(layer_name),
                source,
                target_pattern,
                quant_key_str,
            )
            counts[key] = counts.get(key, 0) + 1

        summaries = []
        # Build summary entries as
        # `self_attn.o_proj: 24 (from targets: re:.*self_attn\.o_proj, mxfp4`
        for (layer_type, source, target_pattern, quant_key_str), count in sorted(
            counts.items()
        ):
            pattern_prefix = f"{target_pattern}, " if target_pattern else ""
            summaries.append(
                f"{layer_type}: {count} "
                f"(from {source}: {pattern_prefix}{quant_key_str})"
            )
        return summaries

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Note: as more online quant schemes will be added, this
        # value will become the minimum across all supported schemes.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        raise NotImplementedError(
            "OnlineQuantizationConfig does not support loading from a "
            "checkpoint config. Use quantization_config or "
            "quantization='fp8_per_tensor'/'fp8_per_block' instead."
        )

    def _get_method_cls(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer: torch.nn.Module | None,
    ) -> type | None:
        if spec is None or spec.weight is None:
            return None
        cls = table.get(spec.weight)
        if cls is None:
            raise ValueError(
                f"online quantization for {type(layer).__name__} with "
                f"weight={spec.weight} is not supported; supported weight "
                f"keys: {sorted(str(k) for k in table)}"
            )
        # Online method classes pick their own activation format internally.
        # Per-class activation overrides are not yet wired through; reject
        # explicit overrides until the relevant method class opts in.
        if spec.activation is not None:
            raise ValueError(
                f"activation override (activation={spec.activation}) is not "
                f"yet supported for online {cls.__name__}"
            )
        return cls

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        _, _, cls = self.get_quant_method_target(prefix, type(layer))
        if cls is None:
            return None
        if cls is UnquantizedLinearMethod:
            return UnquantizedLinearMethod()
        if cls is UnquantizedFusedMoEMethod:
            assert isinstance(layer, RoutedExperts)
            return UnquantizedFusedMoEMethod(layer.moe_config)
        source_and_spec = self.get_source_and_spec(type(layer))
        assert source_and_spec is not None
        source, spec = source_and_spec
        assert spec is not None
        self.quantized_layers[prefix] = (source, str(spec), None)

        if isinstance(layer, RoutedExperts):
            assert issubclass(cls, FusedMoEMethodBase)
            return cls(moe=layer.moe_config)

        if isinstance(layer, LinearBase):
            assert issubclass(cls, OnlineLinearBase)
            linear_method_cls = cast(type[OnlineLinearBase], cls)
            return linear_method_cls()

        raise AssertionError(f"Unsupported online quantization layer: {layer}")

    def get_source_and_spec(
        self, layer_type: type[torch.nn.Module]
    ) -> tuple[str, QuantSpec | None] | None:
        """Return the online quantization source and specification for a layer."""
        if issubclass(layer_type, LinearBase):
            return "linear", self.args.linear
        if issubclass(layer_type, RoutedExperts):
            return "moe", self.args.moe
        return None

    def get_quant_method_target(
        self, prefix: str, layer_type: type[torch.nn.Module]
    ) -> tuple[
        QuantKey | None,
        QuantKey | None,
        type[QuantizeMethodBase]
        | type[UnquantizedLinearMethod]
        | type[UnquantizedFusedMoEMethod]
        | None,
    ]:
        """Return the online quantization target selected for ``prefix``."""
        source_and_spec = self.get_source_and_spec(layer_type)
        if source_and_spec is None:
            return None, None, None
        source, spec = source_and_spec

        table = _ONLINE_LINEAR_METHODS if source == "linear" else _ONLINE_MOE_METHODS
        if should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return (
                None,
                None,
                UnquantizedLinearMethod
                if issubclass(layer_type, LinearBase)
                else UnquantizedFusedMoEMethod,
            )

        method_cls = self._get_method_cls(spec, table, None)
        if method_cls is None:
            return (
                None,
                None,
                UnquantizedLinearMethod
                if issubclass(layer_type, LinearBase)
                else UnquantizedFusedMoEMethod,
            )
        assert spec is not None
        return (
            spec.weight,
            getattr(method_cls, "activation_quant_key", None),
            method_cls,
        )
