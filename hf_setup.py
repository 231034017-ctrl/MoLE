import os
from huggingface_hub import login, whoami
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def authenticate():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF_TOKEN not set in environment. Launching interactive login...")
        login()

    user = whoami()
    print(f"Authenticated as: {user['name']}")


def verify_model_access():
    print(f"Checking access to {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Success! Loaded tokenizer (vocab size: {len(tokenizer)}).")


if __name__ == "__main__":
    authenticate()
    verify_model_access()