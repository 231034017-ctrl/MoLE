import torch
from datasets import load_from_disk
from gptqmodel import CalibrationConfig, GPTQModel, QuantizeConfig

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = "./llama32_python_tokenized/train"
OUTPUT_DIR = "./llama32_3b_gptq_g32"
NUM_CALIB_SAMPLES = 128


def load_calibration_data(path: str, num_samples: int) -> list[dict]:
    ds = load_from_disk(path)
    subset = ds.select(range(min(num_samples, len(ds))))
    return [
        {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
        }
        for row in subset
    ]


def main():
    print(f"Loading {NUM_CALIB_SAMPLES} calibration samples from {DATASET_PATH}...")
    calib_data = load_calibration_data(DATASET_PATH, NUM_CALIB_SAMPLES)

    # Asymmetric group-wise INT4 (L=32) with contiguous groups (desc_act=False)
    quant_config = QuantizeConfig(
        bits=4,
        group_size=32,
        sym=False,
        desc_act=False,
    )

    print(f"Loading base model {MODEL_ID} for GPTQ calibration...")
    model = GPTQModel.load(MODEL_ID, quant_config)

    print("Running GPTQ calibration (this takes a few minutes)...")
    model.quantize(CalibrationConfig(dataset=calib_data, batch_size=1))

    print(f"Saving calibrated INT4 base model to {OUTPUT_DIR}...")
    model.save(OUTPUT_DIR)
    torch.cuda.empty_cache()
    print("Done! Reusable GPTQ base model saved.")


if __name__ == "__main__":
    main()