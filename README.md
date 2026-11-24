# QA-LoRA for Mobile/Edge SLMs (Llama-3.2-3B-Instruct)

A modular, end-to-end PyTorch pipeline for quantizing **Llama-3.2-3B-Instruct** to 4-bit integers (**GPTQ**, asymmetric group size $L = 32$), fine-tuning domain-specialized experts with **QA-LoRA (Quantization-Aware Low-Rank Adaptation)**, **losslessly merging** adapters directly into the INT4 quantization zero-points, and exporting a mobile-ready **GGUF (`Q4_1`)** bundle for edge inference.

The first prototype domain expert is trained for **Structured Python Instruction & Code Synthesis** using a quality-filtered slice of `NickIBrody/python-code-instructions-85k`.

---

## Why This Project?

Running Small Language Models (SLMs) locally on mobile and edge devices is constrained by strict RAM ceilings and memory bandwidth bottlenecks, making **4-bit weight quantization** essential. However, combining 4-bit quantization with standard LoRA creates a major problem:

* **The Standard QLoRA Bottleneck:** Standard LoRA learns unconstrained FP16 weight updates for every individual parameter. When you try to merge those FP16 updates back into a 4-bit base model, the merged values no longer align with the 4-bit integer grid—forcing you to either keep the base model in FP16 during merging and re-quantize (losing accuracy) or pay runtime overhead to compute LoRA branches separately.
* **The QA-LoRA Solution:** Every weight row is divided into contiguous groups of **$L = 32$ weights**, each sharing one `scale` ($s$) and one `zero_point` ($z$). By applying **group-wise average pooling** to the input activations before the LoRA adapter, the adapter produces one shared update per 32-weight group. After training, that update folds **losslessly** directly into the group's `zero_point`:

$$z' = z - \frac{\Delta W_{\text{group}}}{s}$$

* **Modular Domain Experts:** Because the GPTQ INT4 base model (`./llama32_3b_gptq_g32`) remains completely frozen during training, each domain expert is stored as a lightweight **~25 MB adapter** (`adapter_weights.pt`) that can be independently trained, swapped, or losslessly merged into a standalone mobile binary.

---

## Pipeline Overview

1. **Data Curation:** Filters `NickIBrody/python-code-instructions-85k` using Python 3 AST syntax validation (`ast.parse`), mobile-friendly length bounds, AST structural richness scoring, deduplication, and length-stratified sampling (5,000 train / 500 test).
2. **Chat Formatting & Loss Masking:** Applies the official Llama 3.2 Instruct chat template, sets `<|finetune_right_pad_id|>` as the pad token (preventing `<|eot_id|>` from being masked), and masks all system/user prompt tokens to `-100` so loss is computed strictly on the assistant's code output.
3. **One-Time GPTQ Calibration ($L=32$):** Runs asymmetric 4-bit GPTQ calibration (`bits=4, group_size=32, sym=False, desc_act=False`) using 128 samples from the training split and saves the reusable INT4 base model.
4. **QA-LoRA Fine-Tuning:** Replaces all **196 linear projections** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` across 28 Transformer blocks) with custom `QALoRALinear` layers (`r=16`, `alpha=32`, `group_size=32`) and trains for 2 epochs with dynamic batch padding and cosine LR scheduling.
5. **Lossless INT4 Merge & Evaluation:** Folds all 196 trained adapters into their respective group `zeros`, verifies near-zero test-loss drift, and benchmarks Base INT4 vs. Merged QA-LoRA INT4 on held-out test prompts.
6. **Mobile GGUF Export (`Q4_1`):** Materializes the merged weights into a standard Hugging Face checkpoint, converts to **GGUF `Q4_1`** (asymmetric 4-bit with `block_size=32`, matching the QA-LoRA $L=32$ grid), and benchmarks CPU prefill/decode speed with `llama.cpp`.

---

## Experimental Results (Python Domain Expert)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Base Model** | `meta-llama/Llama-3.2-3B-Instruct` | 28 layers, GQA, SwiGLU |
| **Quantization Scheme** | Asymmetric GPTQ INT4 (`L = 32`) | ~4.85 bpw (~2.3 GiB base footprint) |
| **Replaced Layers** | 196 `QALoRALinear` layers | All 7 attention + MLP projections |
| **Trainable Parameters** | 12,759,040 (~3.14%) | ~25 MB saved adapter checkpoint |
| **Training Setup** | 2 Epochs, Effective Batch Size 16 | `lr=2e-4`, cosine decay, 5% warmup |
| **Peak Training VRAM** | **8.59 GB** | With activation checkpointing enabled |
| **Baseline Val Loss** | `1.8249` (PPL `6.20`) | Unmodified INT4 GPTQ base on test split |
| **Best Val Loss (Step 626)** | **`1.3694`** (PPL `3.93`) | Steady convergence, zero overfitting |
| **Post-Merge Test Loss** | **`1.3746`** (`abs diff: 0.0051`) | LoRA branches removed; bfloat16 rounding only |
| **50-Prompt Syntax Pass Rate** | Base: `76.0%` \| Merged: `74.0%` | Merged model drops verbose markdown prose in favor of direct, concise Python functions |

---

## Repository Structure

```text
MoLE/
├── curate_dataset.py           # Step 1: AST-filtered & scored train/test JSONL creation
├── hf_setup.py                 # Step 2a: Hugging Face auth & gated model check
├── tokenizer_utils.py          # Step 2b: Llama 3.2 chat template & -100 loss masking
├── prepare_dataset.py          # Step 2c: Tokenize, sanity-check & save Arrow datasets
├── calibrate_gptq.py           # Step 3a: One-time GPTQ INT4 (L=32) calibration
├── qalora_layer.py             # Step 3b: QALoRALinear module & lossless zero-point merge
├── verify_qalora.py            # Step 3c: 1-batch forward/backward & merge math check
├── data_collator.py            # Step 4a: Dynamic batch padding collator
├── train_qalora.py             # Step 4b: 2-epoch QA-LoRA training & validation loop
├── merge_qalora.py             # Step 5a: Lossless INT4 merge & test-loss verification
├── eval_python_expert.py       # Step 5b: Base vs. Merged AST syntax evaluation
├── export_merged_hf.py         # Step 6a: Reconstruct standard HF checkpoint from INT4 buffers
├── convert_and_benchmark.py    # Step 6b: Q4_1 GGUF conversion, llama-bench & CLI tests
├── requirements.txt            # Python dependencies
└── .gitignore                  # Excludes .venv, datasets, checkpoints, GGUFs & llama.cpp
```

> **Note on Weights & Datasets:** Multi-GB model checkpoints (`llama32_3b_gptq_g32/`, `merged_models/`, `exported_hf/`, `exported_gguf/`), datasets (`train.jsonl`, `test.jsonl`, `llama32_python_tokenized/`), and `llama.cpp/` are excluded from Git via `.gitignore`. Running the steps below reproduces all artifacts automatically.

---

## Prerequisites & Setup

### Hardware Requirements
* **Training / Calibration:** NVIDIA GPU with $\ge$ 12 GB VRAM (tested at **8.59 GB peak VRAM** on CUDA 12.4+).
* **Mobile / Edge Inference:** Any Linux/macOS CPU or Android device with $\ge$ 4 GB free RAM.

### 1. Clone & Create Virtual Environment
```bash
git clone <your-repo-url>
cd MoLE

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install CUDA-enabled PyTorch (adjust cu124 if your CUDA version differs)
pip install torch --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# Install core training, dataset, and GGUF conversion packages
pip install transformers datasets accelerate optimum tqdm gguf sentencepiece protobuf
pip install -v gptqmodel --no-build-isolation
```

### 3. Request Access & Authenticate with Hugging Face
`meta-llama/Llama-3.2-3B-Instruct` is a gated model:
1. Request and accept the license at [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
2. Generate a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Authenticate in your terminal:
```bash
hf auth login
python hf_setup.py
```

### 4. Build `llama.cpp` (Required Only for Step 6 GGUF Export)
```bash
git clone --depth 1 [https://github.com/ggml-org/llama.cpp.git](https://github.com/ggml-org/llama.cpp.git)
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --config Release -j --target llama-quantize llama-cli llama-bench
```

---

## Step-by-Step Usage

Run the pipeline sequentially from the repository root with `.venv` activated:

### Step 1 & 2: Curate and Tokenize Dataset
```bash
python curate_dataset.py
python prepare_dataset.py
```
*Produces `train.jsonl`, `test.jsonl`, and `./llama32_python_tokenized/{train,test}` with `-100` prompt loss masking.*

### Step 3: Calibrate INT4 Base Model & Verify QA-LoRA
```bash
python calibrate_gptq.py
python verify_qalora.py
```
*Saves the reusable calibrated GPTQ checkpoint to `./llama32_3b_gptq_g32` and confirms gradients, VRAM usage, and lossless merge math on 1 batch.*

### Step 4: Train the QA-LoRA Python Expert
```bash
python train_qalora.py
```
*Trains for 2 epochs and saves the best ~25 MB adapter to `./checkpoints/python_expert_qalora/adapter_weights.pt` along with `training_history.json`.*

### Step 5: Lossless INT4 Merge & Evaluation
```bash
python merge_qalora.py
python eval_python_expert.py
```
*Folds adapters into group zero-points (`./merged_models/llama32_3b_python_int4`), confirms post-merge test loss, and evaluates 50 held-out test prompts (`eval_results.json`).*

### Step 6: Export to Mobile GGUF (`Q4_1`) & Benchmark
```bash
python export_merged_hf.py
python convert_and_benchmark.py
```
*Outputs `./exported_gguf/llama32_3b_python_q4_1.gguf` (~2.0 GB), runs `llama-bench` (`pp128` / `tg128`), and verifies Python output via `llama-cli`.*

---

## Running on Android / Edge Devices

Once `convert_and_benchmark.py` finishes, transfer `./exported_gguf/llama32_3b_python_q4_1.gguf` to your phone:

* **PocketPal AI (Android / iOS):** Open the app $\rightarrow$ **Models** $\rightarrow$ **Add Local Model** $\rightarrow$ select `llama32_3b_python_q4_1.gguf`.
* **Termux (`llama.cpp` CLI on Android):**
  ```bash
  ./llama-cli -m llama32_3b_python_q4_1.gguf -cnv -p "You are an expert Python programmer. Write clean, accurate, and efficient Python 3 code."
  ```

---

## Adding a Second Domain Expert

Because `./llama32_3b_gptq_g32` is already calibrated and frozen, you do **not** need to re-run `calibrate_gptq.py`:
1. Curate and tokenize your new domain dataset into a new folder (e.g., `./llama32_finance_tokenized`).
2. Update `DATA_DIR` and `SAVE_DIR` in `train_qalora.py` (e.g., `./checkpoints/finance_expert_qalora`) and run `python train_qalora.py`.
3. Merge and export independently using `merge_qalora.py`, `export_merged_hf.py`, and `convert_and_benchmark.py`.

---

## References
* **QA-LoRA:** *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models* (Xu et al., ICLR 2024)
* **GPTQ:** *Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., ICLR 2023)
* **Base Model:** [Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* **Dataset:** [NickIBrody/python-code-instructions-85k](https://huggingface.co/datasets/NickIBrody/python-code-instructions-85k)
