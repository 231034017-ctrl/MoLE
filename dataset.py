import ast
import hashlib
import json
import random
import re
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "NickIBrody/python-code-instructions-85k"
TRAIN_SIZE = 5000
TEST_SIZE = 500
SEED = 42

# Length bounds (tuned for ~1024-token mobile context windows)
MIN_PROMPT_CHARS = 20
MAX_PROMPT_CHARS = 1000
MIN_OUTPUT_CHARS = 50
MAX_OUTPUT_CHARS = 2200
MIN_CODE_LINES = 4
MAX_CODE_LINES = 65

PLACEHOLDER_INPUTS = {
    "", "none", "n/a", "na", "null", "nil",
    "not applicable", "not applicable.", "no input", "no input."
}


def merge_instruction_and_input(instruction: str, inp: str) -> str:
    """Combine instruction and input into one clear user prompt."""
    inst_clean = (instruction or "").strip()
    inp_clean = (inp or "").strip()

    if inp_clean.lower() in PLACEHOLDER_INPUTS:
        return inst_clean

    return f"{inst_clean}\n\nInput:\n{inp_clean}"


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown fences if present, else return raw text."""
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)
    return text.strip()


def analyze_and_score_code(code: str) -> tuple[bool, float]:
    """
    Validate Python 3 syntax via AST and compute a quality score.
    Returns (is_valid, score).
    """
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not (MIN_CODE_LINES <= len(lines) <= MAX_CODE_LINES):
        return False, 0.0

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return False, 0.0

    score = 0.0
    has_func_or_class = False
    control_flow_nodes = 0
    has_docstring = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_func_or_class = True
            if ast.get_docstring(node):
                has_docstring = True
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.ListComp, ast.DictComp)):
            control_flow_nodes += 1

    # Reward structured, non-trivial Python code
    if has_func_or_class:
        score += 3.0
    if has_docstring:
        score += 1.5
    if "#" in code:  # Inline comments / explanations
        score += 1.0

    # Reward moderate control flow (capped so overly dense scripts don't dominate)
    score += min(control_flow_nodes, 6) * 0.75

    # Slight penalty for trivial print-only scripts with no functions
    if not has_func_or_class and "print(" in code and control_flow_nodes == 0:
        score -= 2.0

    return True, score


def process_split(split_data, seen_hashes: set) -> list[dict]:
    """Filter, deduplicate, merge prompts, and score a dataset split."""
    curated = []

    for row in split_data:
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = (row.get("output", "") or "").strip()

        user_prompt = merge_instruction_and_input(instruction, inp)

        # Basic length filtering
        if not (MIN_PROMPT_CHARS <= len(user_prompt) <= MAX_PROMPT_CHARS):
            continue
        if not (MIN_OUTPUT_CHARS <= len(output) <= MAX_OUTPUT_CHARS):
            continue

        # Deduplicate on normalized prompt
        norm_key = hashlib.md5(" ".join(user_prompt.lower().split()).encode()).hexdigest()
        if norm_key in seen_hashes:
            continue

        # Validate Python syntax and score quality
        code = extract_python_code(output)
        is_valid, score = analyze_and_score_code(code)
        if not is_valid or score < 2.0:
            continue

        seen_hashes.add(norm_key)
        curated.append({
            "score": score,
            "code_len": len(code),
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output},
            ],
        })

    return curated


def select_best_diverse(examples: list[dict], target_size: int, seed: int) -> list[dict]:
    """
    Pick top-scoring examples while preserving length diversity
    (avoids filling the dataset with only long scripts).
    """
    if len(examples) <= target_size:
        return examples

    rng = random.Random(seed)
    # Sort by quality score descending, with random tie-breaking
    examples.sort(key=lambda x: (x["score"], rng.random()), reverse=True)

    # Take a slightly wider high-quality pool (top 1.5x), then sample evenly across length buckets
    pool_size = min(len(examples), int(target_size * 1.5))
    pool = examples[:pool_size]
    pool.sort(key=lambda x: x["code_len"])

    # Stratified sample across 3 length tiers (short, medium, longer)
    n_buckets = 3
    per_bucket = target_size // n_buckets
    selected = []
    bucket_size = len(pool) // n_buckets

    for i in range(n_buckets):
        start = i * bucket_size
        end = len(pool) if i == n_buckets - 1 else (i + 1) * bucket_size
        bucket = pool[start:end]
        k = target_size - len(selected) if i == n_buckets - 1 else min(per_bucket, len(bucket))
        selected.extend(rng.sample(bucket, k))

    rng.shuffle(selected)
    return selected


def save_jsonl(records: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            json.dump({"messages": r["messages"]}, f, ensure_ascii=False)
            f.write("\n")


def main():
    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME)
    seen_hashes = set()

    # Process train split
    print("Filtering and scoring train split...")
    train_candidates = process_split(ds["train"], seen_hashes)
    print(f"Valid high-quality train candidates: {len(train_candidates)}")

    # Use existing test split if available; otherwise hold out from train candidates
    if "test" in ds:
        print("Filtering and scoring test split...")
        test_candidates = process_split(ds["test"], seen_hashes)
        # Top up from validation split if test has fewer than TEST_SIZE valid rows
        if len(test_candidates) < TEST_SIZE and "validation" in ds:
            test_candidates.extend(process_split(ds["validation"], seen_hashes))
        print(f"Valid high-quality test candidates: {len(test_candidates)}")

        final_train = select_best_diverse(train_candidates, TRAIN_SIZE, SEED)
        final_test = select_best_diverse(test_candidates, TEST_SIZE, SEED + 1)
    else:
        combined = select_best_diverse(train_candidates, TRAIN_SIZE + TEST_SIZE, SEED)
        final_test = combined[:TEST_SIZE]
        final_train = combined[TEST_SIZE:]

    save_jsonl(final_train, "train.jsonl")
    save_jsonl(final_test, "test.jsonl")

    print(f"\nDone! Saved {len(final_train)} examples to train.jsonl and {len(final_test)} examples to test.jsonl.")


if __name__ == "__main__":
    main()