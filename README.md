# KVScope

# 1.
KV cache compression — sliding-window eviction policies and KIVI-style KV cache quantization, evaluated against custom Triton attention kernels.

# 2.
Weight compression — GPTQ-Int4 weight quantization, benchmarked against fp16 baselines on perplexity and decoding throughput.

The first half runs in a Colab notebook on an A100. The second half runs locally via standalone Python scripts.
# Repository contents

analysis.ipynb: Main colab notebook. Runs setup, executes the kernel/policy/cache-quant sweeps, and produces plots

kernels.py: Custom triton attention kernels 

policies.py: Cache policy implementations

quantizaiton.py: KIVI - quantizes the K and V tensors during inference

benchmark_utils.py: Timing, memory measurement, and result-table helpers used by the notebook

gptq_measure_local: Local weight quantization benchmark. Compares GPTQ-Int4 weights against fp16 on WikiText-2 and throughput for SmolLm2-1.7b and Qwen2-0.5b

# Results
CSV outputs from the notebook and local scripts:

 - results_baseline.csv, results_architecture.csv, results_architecture_measured.csv : baseline and architecture sweeps
 - results_triton_perf.csv, results_fused_perf.csv, results_per_op_profile.csv : kernel-level performance
 - results_eviction.csv, results_sweep.csv, results_composition.csv, results_fusion_analysis.csv : policy and composition experiments
 - results_quality.csv, results_quant_quality.csv, results_quant_quality_L.csv : KV cache quantization quality
 - results_pareto.csv, results_unified.csv : aggregated trade-off tables
 - results_gptq.csv : GPTQ weight quantization results from the local GPU
 - results_kernel_v1_vs_v2.csv : historical kernel comparison from earlier development (v2 has since been removed)


# Plots:

 - results_pareto.png — quality vs. memory Pareto frontier
 - results_sweep.png — sweep across cache sizes
 - per_op_profile.png — per-op timing profile

# Requirements
Colab path (analysis.ipynb)
Developed and tested on Google Colab with an NVIDIA A100 GPU (40 GB):

 - Python 3.12 (Colab default)
 - PyTorch 2.10.0 (CUDA 12.8)
 - Triton 3.6.0
 - Transformers 5.0.0
 - matplotlib, datasets

# Local path (gptq_measure_*.py)
Run on a local machine with a CUDA-capable GPU.
# Python 3.11 is required for the local scripts. 

The auto_gptq library does not support Python 3.12+ and will fail to install on newer versions. Create a 3.11 venv before installing dependencies:

py -3.11 -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate       # macOS / Linux
pip install torch transformers datasets auto_gptq


# Model weights
The quantized SmolLM2 weights (SmolLM2-1.7B-GPTQ-Int4/, several GB) are not included in this repository.

 - Base model: HuggingFaceTB/SmolLM2-1.7B
 - Quantization: GPTQ-Int4

The Qwen comparison in gptq_measure_local.py pulls its quantized weights directly from the Hugging Face Hub (Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4), so no local preparation is needed for that model.

# How to run

Colab notebook (main analysis — kernels, policies, KV cache quantization)

1. Open analysis.ipynb in Google Colab.
2. Connect to an A100 (or other CUDA) runtime.
3. In the Setup section, upload benchmark_utils.py, policies.py, quantization.py, and kernels.py to the runtime — either via the file uploader or by mounting Google Drive and copying them in (the notebook shows both paths).
4. Place the quantized SmolLM2 model directory at /content/drive/MyDrive/cs790/SmolLM2-1.7B-GPTQ-Int4/, or update the path in the notebook to match where you stored it.
Run all cells in order. Result CSVs and plots are written to /content/ and can be zipped for download.

Local GPTQ benchmarks (weight quantization)

1. Activate the Python 3.11 venv described in Requirements.
2. Place the quantized SmolLM2 weights in a folder named SmolLM2-1.7B-GPTQ-Int4/ in the same directory as the script.
3. Run:

python gptq_measure_local.py

4. The script loads each model, measures perplexity on wikitext-2 (last 1024 tokens of a 2048-token window) and decoding throughput (50 new tokens from a 512-token prompt), and writes results to results_gptq_local.csv. The fp16 baselines are pre-recorded constants from earlier runs to keep total runtime short.
