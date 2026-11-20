import os
from datasets import load_dataset
from tokenizer_utils import format_and_mask, get_llama32_tokenizer

OUTPUT_DIR = "./llama32_python_tokenized"


def print_sanity_check(sample: dict, tokenizer):
    masked = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l == -100]
    trained = [t for t, l in zip(sample["input_ids"], sample["labels"]) if l != -100]

    print("\n--- MASKED PROMPT (Ignored in Loss) ---")
    print(tokenizer.decode(masked, skip_special_tokens=False))
    print("\n--- UNMASKED TARGET (Trained) ---")
    print(tokenizer.decode(trained, skip_special_tokens=False))
    print("-" * 50 + "\n")


def main():
    tokenizer = get_llama32_tokenizer()
    raw_ds = load_dataset(
        "json", data_files={"train": "train.jsonl", "test": "test.jsonl"}
    )

    preview = format_and_mask(raw_ds["train"][0], tokenizer)
    print_sanity_check(preview, tokenizer)

    for split in ["train", "test"]:
        ds = raw_ds[split].map(
            lambda ex: format_and_mask(ex, tokenizer),
            remove_columns=raw_ds[split].column_names,
        )
        ds = ds.filter(lambda x: not x["is_truncated"])
        ds = ds.remove_columns(["is_truncated"])

        save_path = os.path.join(OUTPUT_DIR, split)
        ds.save_to_disk(save_path)
        print(f"Saved {split}: {len(ds)} rows -> {save_path}")


if __name__ == "__main__":
    main()