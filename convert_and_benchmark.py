import ast
import json
import os
import re
import subprocess
import sys

from tokenizer_utils import SYSTEM_PROMPT, get_llama32_tokenizer

# --- Configuration ---
LLAMA_CPP_DIR = "./llama.cpp"
EXPORT_HF_DIR = "./exported_hf/llama32_3b_python_merged"
GGUF_OUT_DIR = "./exported_gguf"
TEST_JSONL = "test.jsonl"

# Q4_1 = asymmetric 4-bit with block_size=32 (matches QA-LoRA L=32 grid)
QUANT_TYPE = "Q4_1"
NUM_TEST_PROMPTS = 3


def run_cmd(cmd: list[str], capture: bool = False) -> str:
    print(f"\n> {' '.join(cmd)}")
    if capture:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout
    subprocess.run(cmd, check=True)
    return ""


def file_size_gb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 3)


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


def main():
    os.makedirs(GGUF_OUT_DIR, exist_ok=True)

    f16_gguf = os.path.join(GGUF_OUT_DIR, "llama32_3b_python_f16.gguf")
    quant_gguf = os.path.join(GGUF_OUT_DIR, f"llama32_3b_python_{QUANT_TYPE.lower()}.gguf")

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
    bench_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-bench")
    cli_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-cli")

    # 1. Convert HF checkpoint -> FP16 GGUF
    if not os.path.exists(f16_gguf):
        run_cmd([
            sys.executable, convert_script,
            EXPORT_HF_DIR,
            "--outfile", f16_gguf,
            "--outtype", "f16",
        ])

    # 2. Quantize FP16 GGUF -> Q4_1 GGUF
    run_cmd([quant_bin, f16_gguf, quant_gguf, QUANT_TYPE])

    print("\n" + "=" * 60)
    print(f"FP16 GGUF size:        {file_size_gb(f16_gguf):.2f} GB")
    print(f"{QUANT_TYPE} Mobile GGUF size: {file_size_gb(quant_gguf):.2f} GB")
    print("=" * 60)

    # 3. Benchmark CPU/edge prefill (pp128) & decode (tg128) speed
    print("\nRunning edge CPU benchmark (ngl=0, 128 prompt / 128 gen tokens)...")
    run_cmd([bench_bin, "-m", quant_gguf, "-ngl", "0", "-p", "128", "-n", "128", "-r", "3"])

    # 4. Sanity-check Python generation on 3 held-out test prompts
    tokenizer = get_llama32_tokenizer(EXPORT_HF_DIR)
    prompts = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["messages"][0]["content"])
            if len(prompts) >= NUM_TEST_PROMPTS:
                break

    print("\nVerifying exported GGUF outputs on held-out Python prompts...")
    for idx, user_prompt in enumerate(prompts, start=1):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        out = run_cmd(
            [
                cli_bin,
                "-m", quant_gguf,
                "-p", formatted,
                "-n", "256",
                "--temp", "0",
                "-ngl", "0",
                "--no-display-prompt",
            ],
            capture=True,
        ).strip()

        valid = is_valid_python(out)
        print(f"\n--- GGUF TEST {idx} | Valid Python Syntax: {valid} ---")
        print(f"PROMPT: {user_prompt[:200]}")
        print(out[:450])
        print("-" * 60)

    print(f"\nMobile-ready file saved at: {quant_gguf}")


if __name__ == "__main__":
    main()