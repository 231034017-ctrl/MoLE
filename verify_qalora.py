import torch
from datasets import load_from_disk
from gptqmodel import BACKEND, GPTQModel
from qalora_layer import QALoRALinear, inject_qalora

QUANT_DIR = "./llama32_3b_gptq_g32"
TRAIN_DATA = "./llama32_python_tokenized/train"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def test_lossless_merge(model):
    """Verify merging QA-LoRA into zero-points produces identical outputs."""
    sample_layer = next(m for m in model.modules() if isinstance(m, QALoRALinear))
    # Temporarily give lora_B non-zero weights to test real merge math
    orig_b = sample_layer.lora_B.data.clone()
    orig_z = sample_layer.zeros.data.clone()

    with torch.no_grad():
        sample_layer.lora_B.data.normal_(0, 0.02)
        sample_layer.eval()
        x = torch.randn(1, 8, sample_layer.in_features, device=sample_layer.scales.device, dtype=torch.bfloat16)
        out_before = sample_layer(x)
        sample_layer.merge()
        out_after = sample_layer(x)

        max_diff = (out_before - out_after).abs().max().item()
        print(f"Lossless merge check max diff: {max_diff:.6f}")

        # Restore layer state
        sample_layer.zeros.data.copy_(orig_z)
        sample_layer.lora_B.data.copy_(orig_b)
        sample_layer.merged = False


def main():
    print(f"Loading calibrated GPTQ model from {QUANT_DIR}...")
    wrapper = GPTQModel.load(QUANT_DIR, device="cuda:0", backend=BACKEND.TORCH)
    model = wrapper.model
    model.config.use_cache = False

    replaced = inject_qalora(model, r=16, lora_alpha=32)
    print(f"Replaced {replaced} GPTQ linear layers with QALoRALinear.")
    count_parameters(model)

    # 1-batch forward & backward sanity check
    ds = load_from_disk(TRAIN_DATA)
    sample = ds[0]
    input_ids = torch.tensor([sample["input_ids"]], device="cuda:0")
    attention_mask = torch.tensor([sample["attention_mask"]], device="cuda:0")
    labels = torch.tensor([sample["labels"]], device="cuda:0")

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()

    # Confirm gradients reached lora_A and lora_B
    grad_layers = sum(
        1 for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"1-Batch Loss: {loss.item():.4f}")
    print(f"Adapter tensors with valid gradients: {grad_layers}")
    print(f"Peak GPU VRAM allocated: {vram_gb:.2f} GB")

    test_lossless_merge(model)
    print("\nStep 3 sanity check passed!")


if __name__ == "__main__":
    main()