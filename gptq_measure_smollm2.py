"""gptq_measure_smollm2.py — LOCAL machine (NVIDIA + CUDA)

Runs the actual GPTQ quantization on SmolLM2-1.7B and saves a 4-bit
checkpoint to disk. This must be run LOCALLY because auto-gptq is
unreliable on Colab Py3.12.

After this finishes:
1. Zip the output folder (./SmolLM2-1.7B-GPTQ-Int4/)
2. Upload to Google Drive
3. In Colab, mount Drive and run gptq_measure_smollm2_colab.py

Requirements (local install, one-time):
    # Recommended: a fresh Python 3.10 or 3.11 conda/venv environment.
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate optimum
    pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

If auto-gptq install fails on Py3.12, the cleanest fix is to install Py3.11:
    conda create -n gptq python=3.11 -y
    conda activate gptq
    (then run the pip installs above)

Hardware: needs ~8GB VRAM during quantization. Any modern NVIDIA card works
(RTX 3060/4060 and up). Quantization itself takes 10-30 minutes depending
on GPU.
"""
import os
import time
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# auto-gptq import — fail fast with a clear message if missing
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except ImportError as e:
    raise SystemExit(
        "auto-gptq is not installed. Install it with:\n"
        "  pip install auto-gptq --extra-index-url "
        "https://huggingface.github.io/autogptq-index/whl/cu121/\n"
        "If that fails on Py3.12, switch to a Py3.11 environment first.\n"
        f"(import error: {e})"
    )

# Config ----------------------------------------------------------------
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
OUT_DIR = "./SmolLM2-1.7B-GPTQ-Int4"      # quantized checkpoint goes here
CALIB_SAMPLES = 128                         # how many wikitext samples to use
CALIB_MAX_LEN = 2048                        # truncate each sample to this many tokens

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Torch: {torch.__version__}")
print(f"Will quantize {MODEL_ID} -> {OUT_DIR}\n")

# Tokenizer & calibration data -----------------------------------------
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading wikitext-2 train split for calibration...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_texts = [t for t in ds["text"] if len(t.strip()) > 200][:CALIB_SAMPLES]
print(f"  Using {len(calib_texts)} calibration samples")

# auto-gptq wants a list of dicts with input_ids + attention_mask
calib_data = []
for t_str in calib_texts:
    enc = tok(t_str, return_tensors="pt", truncation=True, max_length=CALIB_MAX_LEN)
    calib_data.append({
        "input_ids":      enc.input_ids,
        "attention_mask": enc.attention_mask,
    })

# Quantization config ---------------------------------------------------
# group_size=128 is the standard GPTQ setting (per-group scale/zero).
# desc_act=False because activation reordering ("act-order") adds compile
# complexity for marginal quality gain on small models.
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    damp_percent=0.01,    # numerical stabilization for Hessian inverse
)

print(f"\nLoading fp16 model + initializing GPTQ wrapper...")
model = AutoGPTQForCausalLM.from_pretrained(
    MODEL_ID,
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

print(f"Running GPTQ quantization (~10-30 min on a typical GPU)...")
t0 = time.perf_counter()
model.quantize(calib_data)
elapsed = time.perf_counter() - t0
print(f"  Quantization done in {elapsed/60:.1f} minutes.")

print(f"Saving quantized checkpoint to {OUT_DIR}...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_quantized(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# Report sizes ---------------------------------------------------------
total_bytes = 0
for root, _, files in os.walk(OUT_DIR):
    for f in files:
        total_bytes += os.path.getsize(os.path.join(root, f))
print(f"\nCheckpoint size on disk: {total_bytes / 1e9:.2f} GB")

print("\n=== Done ===")
print(f"Next steps:")
print(f"  1. Zip the folder:  zip -r SmolLM2-1.7B-GPTQ-Int4.zip {OUT_DIR}")
print(f"  2. Upload SmolLM2-1.7B-GPTQ-Int4.zip to Google Drive")
print(f"  3. In Colab, run gptq_measure_smollm2_colab.py")