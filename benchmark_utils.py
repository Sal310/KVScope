"""benchmark_utils.py — generic timing, cache I/O, perplexity, decode.

Module-level utilities used across the eviction, quantization, and
architectural-sharing sweeps in KVScope. Every public function takes its
model / tokenizer / tensors explicitly — no module-level globals.

Public API
----------
    benchmark(fn, warmup=10, runs=50)          — median + stddev wall-clock ms
    cache_bytes(cache)                         — total fp16 bytes in a DynamicCache
    prefill_full(model, ids)                   — prefill + return (cache, last_logits)
    decode_bench(model, cache, last, n_new)    — greedy-decode benchmark
    perplexity(model, ids, mask, eval_start)   — masked cross-entropy ppl
    count_cuda_kernels(fn, *args)              — kernel-launch counter
    gpu_time(e), gpu_mem(e)                    — profiler-event accessors

Internal helpers (`_layers`, `_set`) are exported so the other modules
(policies, quantization) can manipulate DynamicCache without re-implementing
the new-vs-old API shim.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "benchmark",
    "cache_bytes", "_layers", "_set",
    "prefill_full", "decode_bench", "perplexity",
    "count_cuda_kernels", "gpu_time", "gpu_mem",
]


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(fn, warmup=10, runs=50):
    """Median + stddev wall-clock latency (ms) for `fn`, with CUDA sync.

    Returns (median_ms, stddev_ms).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(latencies)), float(np.std(latencies))


# ─────────────────────────────────────────────────────────────────────────────
# DynamicCache I/O — handle both old (key_cache list) and new (layers) APIs
# ─────────────────────────────────────────────────────────────────────────────
def _layers(cache):
    """Yield (K, V) tensor pairs for every layer in the cache."""
    if hasattr(cache, "layers"):
        return [(L.keys, L.values) for L in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _set(cache, i, k, v):
    """Replace the K/V tensors at layer `i` in-place."""
    if hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        cache.key_cache[i], cache.value_cache[i] = k, v


def cache_bytes(cache):
    """Total bytes occupied by the KV cache (sum across all layers)."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in _layers(cache))


# ─────────────────────────────────────────────────────────────────────────────
# Decode-loop primitives
# ─────────────────────────────────────────────────────────────────────────────
def prefill_full(model, ids):
    """Prefill `model` with `ids`. Returns (cache, last_token_logits)."""
    from transformers import DynamicCache  # local import to keep HF dep optional at module load
    cache = DynamicCache()
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1]


def decode_bench(model, cache, last_logits, n_new):
    """Greedy-decode `n_new` tokens, measuring cache size, peak GPU memory,
    and throughput.

    Returns {"cache_MB": ..., "peak_GB": ..., "tok_per_s": ...}.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok = last_logits.argmax(-1, keepdim=True)
    for _ in range(n_new):
        with torch.no_grad():
            out = model(tok, past_key_values=cache, use_cache=True)
        cache, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return {
        "cache_MB":  cache_bytes(cache) / 1e6,
        "peak_GB":   torch.cuda.max_memory_allocated() / 1e9,
        "tok_per_s": n_new / (time.perf_counter() - t0),
    }


def perplexity(model, ids, mask=None, eval_start=0):
    """Cross-entropy perplexity of `model` on `ids`, scoring tokens at
    positions >= eval_start (positions before that are ignored).

    `mask` may be a custom attention mask (e.g. one of the policy masks in
    `policies.py`) or None for the default causal mask.
    """
    with torch.no_grad():
        logits = model(ids, attention_mask=mask, use_cache=False).logits
    sl = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    lb = ids[:, 1:].contiguous().clone()
    if eval_start > 0:
        lb[:, :max(0, eval_start - 1)] = -100
    loss = F.cross_entropy(sl, lb.view(-1), ignore_index=-100)
    return math.exp(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# torch.profiler accessors — handle PyTorch 1.x / 2.x rename of cuda_* -> device_*
# ─────────────────────────────────────────────────────────────────────────────
def gpu_time(e):
    """GPU time (µs) for a profiler event, across PyTorch versions."""
    for attr in ("device_time_total", "self_device_time_total",
                 "cuda_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return v
    return 0


def gpu_mem(e):
    """GPU memory (bytes) for a profiler event, across PyTorch versions."""
    for attr in ("device_memory_usage", "cuda_memory_usage"):
        v = getattr(e, attr, None)
        if v:
            return v
    return 0


def count_cuda_kernels(fn, *args):
    """Return (kernel_count, fn_output). Used by the fused-kernel analysis to
    quantify launch-overhead savings (HW3 Part 3)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = fn(*args)
        torch.cuda.synchronize()

    def _dev_time(e):
        return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)

    n = sum(1 for e in prof.key_averages()
            if str(e.device_type) == "DeviceType.CUDA" and _dev_time(e) > 0)
    return n, out
