import json
import math
import os
import torch
from datasets import load_from_disk
from gptqmodel import BACKEND, GPTQModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_collator import DynamicSFTCollator
from qalora_layer import inject_qalora
from tokenizer_utils import get_llama32_tokenizer

# --- Configuration ---
QUANT_DIR = "./llama32_3b_gptq_g32"
DATA_DIR = "./llama32_python_tokenized"
SAVE_DIR = "./checkpoints/python_expert_qalora"

LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # Effective batch size = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
EVAL_EVERY_STEPS = 100
SEED = 42


def save_qalora_adapter(model, save_dir: str, meta: dict):
    """Save only the trainable QA-LoRA adapter tensors and config."""
    os.makedirs(save_dir, exist_ok=True)
    adapter_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(adapter_state, os.path.join(save_dir, "adapter_weights.pt"))
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
    model.train()
    return total_loss / max(len(val_loader), 1)


def main():
    torch.manual_seed(SEED)
    device = "cuda:0"

    tokenizer = get_llama32_tokenizer()
    collator = DynamicSFTCollator(pad_token_id=tokenizer.pad_token_id)

    train_ds = load_from_disk(os.path.join(DATA_DIR, "train"))
    test_ds = load_from_disk(os.path.join(DATA_DIR, "test"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    print(f"Loading calibrated GPTQ model from {QUANT_DIR}...")
    wrapper = GPTQModel.load(QUANT_DIR, device=device, backend=BACKEND.TORCH)
    model = wrapper.model
    model.config.use_cache = False

    replaced = inject_qalora(model, r=LORA_R, lora_alpha=LORA_ALPHA)
    print(f"Injected QA-LoRA into {replaced} linear layers.")

    # Enable gradient checkpointing to keep VRAM low on longer batches
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Total optimizer steps: {total_steps} | Warmup steps: {warmup_steps}")
    print("Computing baseline validation loss before training...")
    best_val_loss = evaluate(model, val_loader, device)
    print(f"Baseline Val Loss: {best_val_loss:.4f}\n")

    history = [{"step": 0, "train_loss": None, "val_loss": round(best_val_loss, 4)}]
    global_step = 0
    running_loss = 0.0
    micro_steps = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / GRAD_ACCUM_STEPS).backward()

            running_loss += loss.item()
            micro_steps += 1

            is_update_step = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader)
            if not is_update_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            avg_train_loss = running_loss / micro_steps
            pbar.set_postfix(
                step=global_step,
                loss=f"{avg_train_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % EVAL_EVERY_STEPS == 0 or global_step == total_steps:
                val_loss = evaluate(model, val_loader, device)
                print(f"\n[Step {global_step}/{total_steps}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                history.append({
                    "step": global_step,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(val_loss, 4),
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_qalora_adapter(
                        model,
                        SAVE_DIR,
                        {
                            "quant_dir": QUANT_DIR,
                            "r": LORA_R,
                            "lora_alpha": LORA_ALPHA,
                            "group_size": 32,
                            "best_step": global_step,
                            "best_val_loss": round(best_val_loss, 4),
                        },
                    )
                    print(f"Saved new best adapter -> {SAVE_DIR} (val_loss={best_val_loss:.4f})")

                running_loss = 0.0
                micro_steps = 0

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f} | Peak VRAM: {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()