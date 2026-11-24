import json
import os
import torch
from datasets import load_from_disk
from gptqmodel import BACKEND, GPTQModel
from torch.utils.data import DataLoader

from data_collator import DynamicSFTCollator
from qalora_layer import QALoRALinear, inject_qalora
from tokenizer_utils import get_llama32_tokenizer

# --- Configuration ---
QUANT_DIR = "./llama32_3b_gptq_g32"
ADAPTER_DIR = "./checkpoints/python_expert_qalora"
TEST_DATA_DIR = "./llama32_python_tokenized/test"
MERGED_DIR = "./merged_models/llama32_3b_python_int4"
BATCH_SIZE = 4


@torch.no_grad()
def compute_val_loss(model, val_loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += model(**batch).loss.item()
    return total_loss / max(len(val_loader), 1)


def main():
    device = "cuda:0"

    with open(os.path.join(ADAPTER_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = get_llama32_tokenizer()
    test_ds = load_from_disk(TEST_DATA_DIR)
    collator = DynamicSFTCollator(pad_token_id=tokenizer.pad_token_id)
    val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    print(f"Loading calibrated GPTQ base model from {QUANT_DIR}...")
    wrapper = GPTQModel.load(QUANT_DIR, device=device, backend=BACKEND.TORCH)
    model = wrapper.model
    model.config.use_cache = False

    inject_qalora(model, r=cfg["r"], lora_alpha=cfg["lora_alpha"])

    adapter_path = os.path.join(ADAPTER_DIR, "adapter_weights.pt")
    adapter_state = torch.load(adapter_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(adapter_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")
    print(f"Loaded {len(adapter_state)} trained QA-LoRA adapter tensors.")

    print("Evaluating test loss BEFORE merge (with separate LoRA branches)...")
    pre_merge_loss = compute_val_loss(model, val_loader, device)
    print(f"Pre-Merge Test Loss:  {pre_merge_loss:.4f}")

    # Fold every QA-LoRA adapter losslessly into its group zero-points
    merged_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, QALoRALinear):
            module.merge()
            merged_layers[name] = {
                "qweight": module.qweight.detach().cpu(),
                "scales": module.scales.detach().cpu(),
                "zeros": module.zeros.detach().cpu(),
                "bias": module.bias.detach().cpu() if module.bias is not None else None,
                "group_size": module.group_size,
            }

    print(f"Merged {len(merged_layers)} layers into INT4 group zero-points.")
    print("Evaluating test loss AFTER merge (LoRA branches bypassed)...")
    post_merge_loss = compute_val_loss(model, val_loader, device)
    loss_diff = abs(post_merge_loss - pre_merge_loss)
    print(f"Post-Merge Test Loss: {post_merge_loss:.4f} (abs diff: {loss_diff:.6f})")

    os.makedirs(MERGED_DIR, exist_ok=True)
    torch.save(merged_layers, os.path.join(MERGED_DIR, "merged_qalora_layers.pt"))
    with open(os.path.join(MERGED_DIR, "merge_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "quant_dir": QUANT_DIR,
                "adapter_dir": ADAPTER_DIR,
                "group_size": cfg["group_size"],
                "num_merged_layers": len(merged_layers),
                "pre_merge_val_loss": round(pre_merge_loss, 4),
                "post_merge_val_loss": round(post_merge_loss, 4),
                "loss_diff": round(loss_diff, 6),
            },
            f,
            indent=2,
        )

    print(f"\nSaved lossless INT4 merged expert to {MERGED_DIR}")


if __name__ == "__main__":
    main()