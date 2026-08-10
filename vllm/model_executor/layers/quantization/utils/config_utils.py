# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def find_matching_patterns(
    layer_name: str,
    patterns: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> list[set[str]]:
    """Return matching patterns for a layer or each shard of a fused layer.

    A pattern matching the fused layer directly takes precedence. Otherwise,
    return one set of matching patterns for every shard.
    """
    patterns = list(patterns)
    matches = [
        pattern for pattern in patterns if is_equal_or_regex_match(layer_name, pattern)
    ]
    if matches:
        return [set(matches)]

    proj_name = layer_name.split(".")[-1]
    if proj_name not in fused_mapping:
        return [set()]

    shard_names = [
        layer_name.replace(proj_name, shard_proj_name)
        for shard_proj_name in fused_mapping[proj_name]
    ]
    per_shard_matches = [
        {
            pattern
            for pattern in patterns
            if is_equal_or_regex_match(shard_name, pattern)
        }
        for shard_name in shard_names
    ]
    return per_shard_matches


def get_layer_name_after_index(layer_name: str) -> str:
    """Return the suffix following the final numeric component of a layer name."""
    parts = layer_name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].isdigit():
            return ".".join(parts[index + 1 :])
    return layer_name


def is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def is_shared_expert_quant_fse_compatible(
    quant_config: "QuantizationConfig | None",
    expert_prefix_pairs: list[tuple[str, str]],
) -> bool:
    """Check whether quantization permits fused shared-expert execution."""
    if quant_config is None:
        return True

    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    if isinstance(quant_config, OnlineQuantizationConfig):
        targets = quant_config.args.targets
        if targets is None:
            return quant_config.args.moe is not None and (
                quant_config.args.linear is None
                or quant_config.args.linear == quant_config.args.moe
            )

        def get_target(prefix: str) -> str | None:
            matches = find_matching_patterns(prefix, targets)
            if any(len(match) > 1 for match in matches):
                raise ValueError(
                    f"Layer {prefix} matches multiple "
                    f"quantization_config.targets patterns: {matches}."
                )
            if any(not match for match in matches):
                return None
            selected = {targets[next(iter(match))] for match in matches}
            return selected.pop() if len(selected) == 1 else None

        for shared_expert_prefix, routed_experts_prefix in expert_prefix_pairs:
            routed_target = get_target(routed_experts_prefix)
            shared_targets = {
                target
                for projection in ("gate_up_proj", "down_proj")
                if (target := get_target(f"{shared_expert_prefix}.{projection}"))
                is not None
            }
            if routed_target is None or (
                shared_targets and shared_targets != {routed_target}
            ):
                return False
        return True
    elif isinstance(quant_config, QuarkConfig):
        return not any(
            "shared_expert." in str(entry)
            for entry in quant_config.quant_config.get("exclude", [])
        )

    raise NotImplementedError(
        "Shared-expert FSE quantization compatibility is not implemented for "
        f"{type(quant_config).__name__}."
    )
