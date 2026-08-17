# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineSharedExpertLoader,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts


class OnlineMxfp4SharedExpertLoader(OnlineSharedExpertLoader):
    """Load full-precision expert weights into packed MXFP4 MoE storage."""

    @staticmethod
    def _weight_parameter_name(layer: "RoutedExperts", stem: str) -> str:
        """Resolve the checkpoint method's packed weight parameter name."""
        name = f"{stem}_weight"
        if hasattr(layer, name):
            return name
        packed_name = f"{name}_packed"
        if hasattr(layer, packed_name):
            return packed_name
        raise AttributeError(f"{type(layer).__name__} has no {name!r} parameter")

    @staticmethod
    def _tp_shard(
        layer: "RoutedExperts", loaded_weight: torch.Tensor, shard_id: str
    ) -> torch.Tensor:
        """Select this rank's full-precision projection shard."""
        shard_dim = 1 if shard_id == "w2" else 0
        tp_size = layer.moe_config.moe_parallel_config.tp_size
        if tp_size == 1:
            return loaded_weight
        if loaded_weight.shape[shard_dim] % tp_size != 0:
            raise ValueError(
                f"Cannot TP-shard full-precision {shard_id} weight with shape "
                f"{tuple(loaded_weight.shape)} across {tp_size} ranks."
            )
        shard_size = loaded_weight.shape[shard_dim] // tp_size
        return loaded_weight.narrow(
            shard_dim, layer.moe_config.tp_rank * shard_size, shard_size
        )

    @torch.no_grad()
    def _load(
        self,
        layer: "RoutedExperts",
        *,
        global_expert_id: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, ...]:
        """Quantize one projection and store its packed weight and scales."""
        local_expert_id = layer._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )
        if local_expert_id == -1:
            return ()

        stem = "w2" if shard_id == "w2" else "w13"
        weight_name = self._weight_parameter_name(layer, stem)
        scale_name = f"{stem}_weight_scale"
        weight = getattr(layer, weight_name)[local_expert_id]
        scale = getattr(layer, scale_name)[local_expert_id]

        loaded_weight = self._tp_shard(layer, loaded_weight, shard_id)
        if shard_id == "w2":
            unpacked_shape = (weight.shape[0], weight.shape[1] * 2)
            weight_dst = weight
            scale_dst = scale
        else:
            shard_size = weight.shape[0] // layer.moe_config.w13_num_shards
            shard_index = 0 if shard_id == "w1" else 1
            unpacked_shape = (shard_size, weight.shape[1] * 2)
            weight_dst = weight.narrow(0, shard_index * shard_size, shard_size)
            scale_dst = scale.narrow(0, shard_index * shard_size, shard_size)

        padded_weight = torch.zeros(
            unpacked_shape,
            dtype=loaded_weight.dtype,
            device=weight.device,
        )
        loaded_weight = loaded_weight.to(device=weight.device)
        rows = min(padded_weight.shape[0], loaded_weight.shape[0])
        columns = min(padded_weight.shape[1], loaded_weight.shape[1])
        padded_weight[:rows, :columns].copy_(loaded_weight[:rows, :columns])

        quantized, quantized_scale = mxfp4_quantize(padded_weight)
        weight_dst.copy_(quantized)
        scale_dst.copy_(quantized_scale)
        return weight_name, scale_name
