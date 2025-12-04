from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

# cfg = {
#     "export": {
#       "kv_cache_group": [],
#       "min_kv_scale": 0.0,
#       "pack_method": "reorder",
#       "weight_format": "real_quantized",
#       "weight_merge_groups": None
#     },
#     "algo_config": [
#       {
#         "backbone": "model",
#         "mlp": "mlp",
#         "model_decoder_layers": "model.layers",
#         "name": "rotation",
#         "o_proj": "self_attn.o_proj",
#         "online_r1_rotation": False,
#         "r1": True,
#         "r2": True,
#         "r3": False,
#         "r4": True,
#         "random": None,
#         "random_r1": False,
#         "random_r2": False,
#         "rotation_size": None,
#         "scaling_layers": {
#           "first_layer": [
#             {
#               "next_modules": [
#                 "model.layers.layer_id.self_attn.q_proj",
#                 "model.layers.layer_id.self_attn.k_proj",
#                 "model.layers.layer_id.self_attn.v_proj"
#               ],
#               "norm_module": "model.layers.layer_id.input_layernorm",
#               "prev_modules": [
#                 "model.embed_tokens"
#               ]
#             },
#             {
#               "next_modules": [
#                 "model.layers.layer_id.mlp.up_proj",
#                 "model.layers.layer_id.mlp.gate_proj"
#               ],
#               "norm_module": "model.layers.layer_id.post_attention_layernorm",
#               "prev_modules": [
#                 "model.layers.layer_id.self_attn.o_proj"
#               ]
#             }
#           ],
#           "last_layer": [
#             {
#               "next_modules": [
#                 "lm_head"
#               ],
#               "norm_module": "model.norm",
#               "prev_modules": [
#                 "model.layers.layer_id.mlp.down_proj"
#               ]
#             }
#           ],
#           "middle_layers": [
#             {
#               "next_modules": [
#                 "model.layers.layer_id.self_attn.q_proj",
#                 "model.layers.layer_id.self_attn.k_proj",
#                 "model.layers.layer_id.self_attn.v_proj"
#               ],
#               "norm_module": "model.layers.layer_id.input_layernorm",
#               "prev_modules": [
#                 "model.layers.pre_layer_id.mlp.down_proj"
#               ]
#             },
#             {
#               "next_modules": [
#                 "model.layers.layer_id.mlp.up_proj",
#                 "model.layers.layer_id.mlp.gate_proj"
#               ],
#               "norm_module": "model.layers.layer_id.post_attention_layernorm",
#               "prev_modules": [
#                 "model.layers.layer_id.self_attn.o_proj"
#               ]
#             }
#           ]
#         },
#         "self_attn": "self_attn",
#         "trainable": False,
#         "v_proj": "self_attn.v_proj"
#       }
#     ],
# }

# qc = QuarkConfig.from_config({
#     **cfg,
#     "layer_quant_config": {},
#     "layer_type_quant_config": {},
#     "global_quant_config": {},
#     "exclude": [],
# })



quant_config = {
    "algo_config": [
        {
            "backbone": "model",
            "name": "rotation",
            "r1": True,
            "r2": True,
            "r3": False,
            "r4": True,
            "online_r1_rotation": False,
            "rotation_size": None,
        }
    ],
    "export": {
        "kv_cache_group": [],
        "pack_method": "reorder",
    },
    "layer_quant_config": {},
    "layer_type_quant_config": {},
    "global_quant_config": {},
    "exclude": [],
}

qc = QuarkConfig(
    quant_config=quant_config,
    kv_cache_group=[],
    kv_cache_config=None,
    pack_method="reorder",
)


print("rotation_algo_config:", qc.rotation_algo_config)
print("r1/r2/r3/r4:", qc.r1_enabled, qc.r2_enabled, qc.r3_enabled, qc.r4_enabled)
print("online_r1_rotation:", qc.online_r1_rotation)
print("rotation_size:", qc.rotation_size)
print("has_online_rotation:", qc.has_online_rotation())