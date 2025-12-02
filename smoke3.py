import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.parameter import ModelWeightParameter

def make_dummy_quant_config(rotation_layers):
    return {
        "name": "rotation",
        "backbone": "model",
        "model_decoder_layers": "model.layers",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "self_attn": "self_attn",
        "mlp": "mlp",
        "r1": True,
        "r2": True,
        "r3": False,
        "r4": True,
        "rotation_size": 64,
        "scaling_layers": {
            "first_layer": [
            {
                "prev_modules": [
                "model.embed_tokens"
                ],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj"
                ]
            },
            {
                "prev_modules": [
                "model.layers.layer_id.self_attn.o_proj"
                ],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                "model.layers.layer_id.mlp.up_proj",
                "model.layers.layer_id.mlp.gate_proj"
                ]
            }
            ],
            "middle_layers": [
            {
                "prev_modules": [
                "model.layers.pre_layer_id.mlp.down_proj"
                ],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj"
                ]
            },
            {
                "prev_modules": [
                "model.layers.layer_id.self_attn.o_proj"
                ],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                "model.layers.layer_id.mlp.up_proj",
                "model.layers.layer_id.mlp.gate_proj"
                ]
            }
            ],
            "last_layer": [
            {
                "prev_modules": [
                "model.layers.layer_id.mlp.down_proj"
                ],
                "norm_module": "model.norm",
                "next_modules": [
                "lm_head"
                ]
            }
            ]
        }
        }

def test_quark_ocp_mx_online_rotation_smoke():
    layer_name = "model.layers.0.mlp.down_proj"
    quant_config = make_dummy_quant_config(rotation_layers=[layer_name])

    # weight_config / input_config aren’t used by __init__ for rotation logic,
    # so we can just pass {} here. If your version requires more, fill it in.
    qc = QuarkConfig(
        quant_config=quant_config,
        kv_cache_group=[],
        kv_cache_config=None,
        pack_method="reorder",
    ).rotation_config

    # 1) rotation got picked up from quant_config
    assert scheme.has_online_rotation()
    assert scheme.rotation_size == 4

    # 2) create a dummy Linear and let scheme.register weights
    layer = nn.Linear(4, 4, bias=False)

    # This must call the code that does:
    #   if self.use_online_rotation: layer.register_parameter("input_rotation", ModelWeightParameter(...))
    scheme.create_weights(layer)

    # Ensure we have an input_rotation parameter of right shape
    assert hasattr(layer, "input_rotation")
    assert isinstance(layer.input_rotation, ModelWeightParameter)
    assert layer.input_rotation.data.shape == (4, 4)

    # 3) Manually set the rotation to something obvious: 2 * I
    with torch.no_grad():
        layer.input_rotation.data.copy_(2.0 * torch.eye(4, dtype=torch.float64))

    # Also make weights something simple: identity
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4))

    # 4) Build an input where we know the answer: x = I (batch=1)
    x = torch.eye(4).unsqueeze(0)  # [1, 4, 4] or [4, 4] depending on your expected shape

    # Whatever apply_weights does, we know that before quantization it calls:
    #   x_rot = activation_transform(layer, x)
    # and activation_transform should compute x @ (2I) = 2x.
    # To keep this a pure smoke test, we can call activation_transform directly:

    x_rot = scheme.activation_transform(layer, x)

    # x_rot should be exactly 2 * x (mod dtype conversions)
    assert torch.allclose(x_rot, 2 * x, atol=1e-6), (
        "activation_transform did not apply rotation correctly"
    )

    # 5) (Optional) If you want to exercise apply_weights as well:
    y = scheme.apply_weights(layer, x)

    # With weight=I and rotation=2I, you expect:
    #   x --(R4)--> 2x  --(linear with W=I)--> 2x
    # Shape might differ (batch collapsed, etc), so adjust as needed:
    expected = 2 * x
    assert torch.allclose(y, expected, atol=1e-5), (
        "apply_weights did not respect activation_transform"
    )

