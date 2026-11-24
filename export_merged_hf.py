import os
import torch
from transformers import AutoModelForCausalLM

from qalora_layer import unpack_int4_rows
from tokenizer_utils import MODEL_ID, get_llama32_tokenizer

# --- Configuration ---
MERGED_LAYERS_PATH = "./merged_models/llama32_3b_python_int4/merged_qalora_layers.pt"
EXPORT_HF_DIR = "./exported_hf/llama32_3b_python_merged"


def dequantize_merged_layer(buf: dict) -> torch.Tensor:
    """Reconstruct exact [out_features, in_features] weights from merged INT4 buffers."""
    qweight = buf["qweight"]
    scales = buf["scales"].to(torch.float32)
    zeros = buf["zeros"].to(torch.float32)
    group_size = buf["group_size"]

    in_features = qweight.shape[0] * 8
    out_features = qweight.shape[1]
    num_groups = in_features // group_size

    w_int = unpack_int4_rows(qweight).to(torch.float32)
    w_grouped = w_int.view(num_groups, group_size, out_features)
    w_dequant = ((w_grouped - zeros.unsqueeze(1)) * scales.unsqueeze(1))

    # Standard nn.Linear expects [out_features, in_features]
    return w_dequant.reshape(in_features, out_features).T.contiguous()


def main():
    print(f"Loading base HF skeleton ({MODEL_ID}) on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = get_llama32_tokenizer(MODEL_ID)

    print(f"Loading merged QA-LoRA INT4 layers from {MERGED_LAYERS_PATH}...")
    merged_layers = torch.load(MERGED_LAYERS_PATH, map_location="cpu", weights_only=True)

    updated = 0
    with torch.no_grad():
        for name, buf in merged_layers.items():
            module = model.get_submodule(name)
            w_out = dequantize_merged_layer(buf).to(module.weight.dtype)

            if w_out.shape != module.weight.shape:
                raise ValueError(
                    f"Shape mismatch on {name}: got {w_out.shape}, expected {module.weight.shape}"
                )

            module.weight.data.copy_(w_out)
            if buf["bias"] is not None and module.bias is not None:
                module.bias.data.copy_(buf["bias"].to(module.bias.dtype))
            updated += 1

    print(f"Replaced {updated} linear layers with merged QA-LoRA INT4 weights.")

    os.makedirs(EXPORT_HF_DIR, exist_ok=True)
    print(f"Saving clean Hugging Face checkpoint to {EXPORT_HF_DIR}...")
    model.save_pretrained(EXPORT_HF_DIR, safe_serialization=True)
    tokenizer.save_pretrained(EXPORT_HF_DIR)
    print("Done! Ready for GGUF conversion.")


if __name__ == "__main__":
    main()