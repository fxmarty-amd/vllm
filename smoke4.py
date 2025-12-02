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
sampling_params = SamplingParams(temperature=0.2, top_p=0.95)

def print_model_layers(llm_instance):
    """
    Accesses the internal vLLM model runner to print layer shapes.
    """
    print("\n" + "="*60)
    print("MODEL LAYER DIMENSIONS")
    print("="*60)
    
    # Try to access the underlying PyTorch model.
    # The path to the model depends slightly on vLLM version and if you are using
    # a single GPU or Ray for distributed inference.
    try:
        # Common path for single-GPU execution
        model = llm_instance.llm_engine.model_executor.driver_worker.model_runner.model
        
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
            
    except AttributeError:
        print("Could not easily access model directly via standard attributes.")
        print("Note: If you are using multi-GPU (Ray), the model is distributed across workers")
        print("and cannot be easily inspected from the main driver process like this.")

def test_llama_quark_rotation_runs():
    
    llm = LLM(
        model="/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/w_mxfp4_a_mxfp4_rotation_Llama-3.1-8B-Instruct_r1r2r4-quantized/",
        quantization="quark",
        trust_remote_code=True,
        enforce_eager=False,
    )
    # print_model_layers(llm)
    outputs = llm.generate(prompts, sampling_params)
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