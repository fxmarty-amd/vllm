import torch
from vllm import LLM, SamplingParams
import multiprocessing
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def test_llama_quark_rotation_runs():
    
    llm = LLM(
        model="/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/w_mxfp4_a_mxfp4_rotation_Llama-3.1-8B-Instruct_r1r2r4-quantized/",
        quantization="quark",
        trust_remote_code=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    # No assertion beyond "it runs without crashing"
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    test_llama_quark_rotation_runs()