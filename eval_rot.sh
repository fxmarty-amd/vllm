#!/bin/bash
set -e

export MODEL_DIR=/amd_models/meta-llama/Llama-3.1-8B-Instruct #/amd_models/facebook/opt-125m # /amd_models/meta-llama/Llama-3.1-8B-Instruct or /amd_models/Qwen/Qwen3-8B
export MODEL_SN=Llama-3.1-8B-Instruct #opt-125m # Llama-3.1-8B-Instruct or Qwen3-8B
export QUANT_SCHEME=w_mxfp4_a_mxfp4 #w_uint4_per_channel_asym #w_uint4_per_token_a_uint4_per_channel # w_int4_per_group_sym or w_mxfp4_a_mxfp4 or w_int3_per_group_asym
export DATASET=wikitext_for_gptq_benchmark #wikitext_for_gptq_benchmark # pileval_for_awq_benchmark or ${DATASET}
export LOG_DIR=${MODEL_SN}-logs
export MAX_EVAL_BATCH_SIZE=32
export BATCH_SIZE=32
export EVAL_BATCH_SIZE=32

# export TASKS='--tasks leaderboard_mmlu_pro,arc_challenge,gsm8k_platinum,gsm8k,mmlu_management'
export TASKS='--tasks wikitext'

mkdir -p ${LOG_DIR}

export QUANT_ALGO=rotation
export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_r1r2r4-quantized

export CUDA_VISIBLE_DEVICES="0"
nohup lm_eval --model vllm \
    --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
    --batch_size ${BATCH_SIZE} \
     ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
echo "vllm Model ${QMODEL_DIR}"

# export QUANT_ALGO1=rotation
# export QUANT_ALGO2=gptq
# export QUANT_ALGO=${QUANT_ALGO1},${QUANT_ALGO2}
# export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO1}_${QUANT_ALGO2}_${MODEL_SN}-quantized

# export CUDA_VISIBLE_DEVICES="2"
# nohup lm_eval --model vllm \
#     --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
#     --batch_size ${BATCH_SIZE} \
#      ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
# echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
# echo "vllm Model ${QMODEL_DIR}"

# export QUANT_ALGO1=rotation
# export QUANT_ALGO2=qronos
# export QUANT_ALGO=${QUANT_ALGO1},${QUANT_ALGO2}
# export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO1}_${QUANT_ALGO2}_${MODEL_SN}-quantized

# export CUDA_VISIBLE_DEVICES="3"
# nohup lm_eval --model vllm \
#     --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
#     --batch_size ${BATCH_SIZE} \
#      ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
# echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
# echo "vllm Model ${QMODEL_DIR}"

# export QUANT_ALGO1=rotation
# export QUANT_ALGO2=gptq
# export QUANT_ALGO=${QUANT_ALGO1},${QUANT_ALGO2}
# export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO1}_${QUANT_ALGO2}_${MODEL_SN}-quantized

# export CUDA_VISIBLE_DEVICES="4"
# nohup lm_eval --model vllm \
#     --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
#     --batch_size ${BATCH_SIZE} \
#      ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
# echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
# echo "vllm Model ${QMODEL_DIR}"

# export QUANT_ALGO1=rotation
# export QUANT_ALGO2=gptaq
# export QUANT_ALGO=${QUANT_ALGO1},${QUANT_ALGO2}
# export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO1}_${QUANT_ALGO2}_${MODEL_SN}-quantized

# export CUDA_VISIBLE_DEVICES="5"
# nohup lm_eval --model vllm \
#     --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
#     --batch_size ${BATCH_SIZE} \
#      ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
# echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
# echo "vllm Model ${QMODEL_DIR}"

# export QUANT_ALGO1=rotation
# export QUANT_ALGO2=awq
# export QUANT_ALGO=${QUANT_ALGO1},${QUANT_ALGO2}
# export QMODEL_DIR=/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/${QUANT_SCHEME}_${QUANT_ALGO1}_${QUANT_ALGO2}_${MODEL_SN}-quantized

# export CUDA_VISIBLE_DEVICES="6"
# nohup lm_eval --model vllm \
#     --model_args pretrained=${QMODEL_DIR},kv_cache_dtype='fp8',quantization='quark',enforce_eager=False \
#     --batch_size ${BATCH_SIZE} \
#      ${TASKS}  > ${LOG_DIR}/${QUANT_SCHEME}_${QUANT_ALGO}_${MODEL_SN}_vllm.out & 2>&1
# echo "${MODEL_SN}-${QUANT_ALGO}-${QUANT_SCHEME}-vllm on ${CUDA_VISIBLE_DEVICES} PID $!"
# echo "vllm Model ${QMODEL_DIR}"
