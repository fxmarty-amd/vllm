import torch

def check_rotations(model):
    print("=== Rotation parameters present ===")
    for name, module in model.named_modules():
        rot = getattr(module, "input_rotation", None)
        if rot is None:
            continue
        print(f"{name}: shape={tuple(rot.shape)}, dtype={rot.dtype}, mean_abs={rot.float().abs().mean().item():.4f}")
        # Orthogonality check (small sample to keep it light)
        with torch.no_grad():
            k = min(4, rot.shape[0])
            sub = rot[:k, :k].float()
            eye_err = (sub @ sub.t() - torch.eye(k, device=sub.device)).abs().max().item()
            print(f"  sub-ortho max err: {eye_err:.4e}")

def forward_probe(model, layer_name="mlp.down_proj", hidden_size=4096, batch=1, seq=1):
    model.eval()
    target = None
    for name, module in model.named_modules():
        if layer_name in name and hasattr(module, "scheme"):
            target = module
            target_name = name
            break
    if target is None:
        print(f"Layer containing '{layer_name}' not found")
        return
    x = torch.randn(batch, seq, hidden_size, device=next(model.parameters()).device, dtype=torch.bfloat16)
    pre = None
    def hook(mod, inp, out):
        nonlocal pre
        pre = inp[0].detach()
    h = target.register_forward_hook(hook, with_kwargs=False, prepend=True)
    _ = model(x)
    h.remove()
    if pre is None:
        print("Hook did not fire")
        return
    # Re-run just the layer to compare rotated vs unrotated
    mod = target
    with torch.no_grad():
        rot = getattr(mod, "input_rotation", None)
        if rot is None:
            print(f"{target_name}: no rotation found")
            return
        # manually apply activation_transform
        x_rot = pre.reshape(-1, rot.shape[0]).float() @ rot.float()
        x_rot = x_rot.to(pre.dtype).reshape_as(pre)
        out_rot = mod.scheme.apply_weights(mod, x_rot)
        out_no_rot = mod.scheme.apply_weights(mod, pre)
        diff = (out_rot - out_no_rot).float().norm().item()
        print(f"{target_name}: rotated vs unrotated output norm diff = {diff:.4f}")

# Usage:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/workspaces/Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/w_mxfp4_a_mxfp4_rotation_Llama-3.1-8B-Instruct-quantized", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
check_rotations(model)
forward_probe(model, layer_name="mlp.down_proj", hidden_size=model.config.hidden_size)
