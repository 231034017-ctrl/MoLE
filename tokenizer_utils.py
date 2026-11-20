from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write clean, accurate, and efficient Python 3 code."
)


def get_llama32_tokenizer(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def format_and_mask(example: dict, tokenizer) -> dict:
    user_msg = example["messages"][0]["content"]
    assistant_msg = example["messages"][1]["content"]

    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": assistant_msg}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    input_ids = enc["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    is_truncated = (
        len(input_ids) == MAX_SEQ_LENGTH and input_ids[-1] != tokenizer.eos_token_id
    )
    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "is_truncated": is_truncated,
    }