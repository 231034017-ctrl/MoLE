import ast
import json
import os
import re
import torch
from gptqmodel import BACKEND, GPTQModel
from tqdm import tqdm

from qalora_layer import QALoRALinear
from tokenizer_utils import SYSTEM_PROMPT, get_llama32_tokenizer

# --- Configuration ---
QUANT_DIR = "./llama32_3b_gptq_g32"
MERGED_DIR = "./merged_models/llama32_3b_python_int4"
TEST_JSONL = "test.jsonl"
NUM_PROMPTS = 50
MAX_NEW_TOKENS = 512


def load_test_prompts(path: str, limit: int) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            samples.append({
                "prompt": row["messages"][0]["content"],
                "reference": row["messages"][1]["content"],
            })
            if len(samples) >= limit:
                break
    return samples


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)
    return text.strip()


def is_valid_python(text: str) -> bool:
    code = extract_python_code(text)
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


@torch.no_grad()
def run_generation(model, tokenizer, samples: list[dict], device: str, desc: str) -> list[dict]:
    model.eval()
    model.config.use_cache = True
    eos_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    eos_ids = list({eid for eid in eos_ids if eid is not None})

    results = []
    for item in tqdm(samples, desc=desc):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append({
            "prompt": item["prompt"],
            "response": response,
            "valid_syntax": is_valid_python(response),
        })
    return results


def load_merged_int4_weights(model, merged_dir: str, device: str):
    """Replace GPTQ layers directly with merged QALoRALinear modules (LoRA bypassed)."""
    merged_layers = torch.load(
        os.path.join(merged_dir, "merged_qalora_layers.pt"),
        map_location=device,
        weights_only=True,
    )
    for name, buf in merged_layers.items():
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        layer = QALoRALinear(
            qweight=buf["qweight"].to(device),
            scales=buf["scales"].to(device),
            zeros=buf["zeros"].to(device),
            bias=buf["bias"].to(device) if buf["bias"] is not None else None,
            group_size=buf["group_size"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
        )
        layer.merged = True  # Pure INT4 path; lora_A / lora_B never run
        setattr(parent, child_name, layer)

def main():
    device = "cuda:0"
    tokenizer = get_llama32_tokenizer()
    samples = load_test_prompts(TEST_JSONL, NUM_PROMPTS)

    print(f"Loading Base INT4 GPTQ model from {QUANT_DIR}...")
    wrapper = GPTQModel.load(QUANT_DIR, device=device, backend=BACKEND.TORCH)
    model = wrapper.model

    # 1. Evaluate unmodified Base INT4 model
    base_results = run_generation(model, tokenizer, samples, device, desc="Base INT4")

    # 2. Swap in merged QA-LoRA INT4 weights and evaluate
    print(f"\nLoading Merged QA-LoRA INT4 expert from {MERGED_DIR}...")
    load_merged_int4_weights(model, MERGED_DIR, device)
    merged_results = run_generation(model, tokenizer, samples, device, desc="Merged QA-LoRA INT4")

    base_pass = sum(r["valid_syntax"] for r in base_results)
    merged_pass = sum(r["valid_syntax"] for r in merged_results)

    print("\n" + "=" * 60)
    print(f"PYTHON 3 SYNTAX PASS RATE ({NUM_PROMPTS} Held-Out Test Prompts)")
    print("=" * 60)
    print(f"Base INT4 Model:          {base_pass}/{NUM_PROMPTS} ({100.0 * base_pass / NUM_PROMPTS:.1f}%)")
    print(f"Merged QA-LoRA INT4:      {merged_pass}/{NUM_PROMPTS} ({100.0 * merged_pass / NUM_PROMPTS:.1f}%)")
    print("=" * 60)

    # Save full evaluation log
    eval_out = os.path.join(MERGED_DIR, "eval_results.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_prompts": NUM_PROMPTS,
                "base_syntax_pass_rate": round(100.0 * base_pass / NUM_PROMPTS, 2),
                "merged_syntax_pass_rate": round(100.0 * merged_pass / NUM_PROMPTS, 2),
                "comparisons": [
                    {
                        "prompt": samples[i]["prompt"],
                        "base_valid": base_results[i]["valid_syntax"],
                        "merged_valid": merged_results[i]["valid_syntax"],
                        "base_response": base_results[i]["response"],
                        "merged_response": merged_results[i]["response"],
                    }
                    for i in range(NUM_PROMPTS)
                ],
            },
            f,
            indent=2,
        )
    print(f"Full results saved to {eval_out}\n")

    # Print 3 side-by-side examples
    for i in range(min(3, NUM_PROMPTS)):
        print(f"\n--- EXAMPLE {i + 1} ---")
        print(f"PROMPT: {samples[i]['prompt'][:250]}")
        print(f"\n[BASE INT4 | Valid Syntax: {base_results[i]['valid_syntax']}]")
        print(base_results[i]["response"][:450])
        print(f"\n[MERGED QA-LoRA INT4 | Valid Syntax: {merged_results[i]['valid_syntax']}]")
        print(merged_results[i]["response"][:450])
        print("-" * 60)


if __name__ == "__main__":
    main()