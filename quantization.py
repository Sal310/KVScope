"""quantization.py — KIVI-style KV-cache quantization (Liu et al. 2024).

Two implementation styles, both useful and both kept:

(A) PHYSICAL — `kivi_compress_cache` + `quant_cache_bytes`.
    Actually packs old cache tensors to uint8 with per-channel-K /
    per-token-V scales, stores metadata on `cache._kivi_meta`, and replaces
    the layer tensors with dequantized fp16 so HF attention still works.
    The byte accounting in `quant_cache_bytes` is what an honest paper-
    style cache_MB number looks like.

(B) HOOK-BASED — `kivi_hooks` + `quantize_per_channel_nbit` (+
    `kivi_perplexity_nbit`).
    Installs forward hooks on every layer's k_proj / v_proj that
    quantize-then-dequantize the projection output before it hits attention.
    Generalises to arbitrary bit-widths (2/3/4/8) and group sizes
    (KIVI paper default: 32). What the perplexity sweeps use because it
    composes cleanly with eviction masks.

The notebook's `cache_MB` numbers from (B) match what (A) would produce on
a physically-compressed cache — they use the same analytical formula
(`bytes = n_tok * per_tok_fp16 * (bits/16) * (1 + 2/group_size) + residual_fp16`),
so the two views agree.
"""

import torch
import torch.nn.functional as F

from benchmark_utils import _layers, _set, cache_bytes, perplexity

__all__ = [
    # n-bit primitives
    "quantize_per_channel_nbit",
    # int8-specific packers (used by the fused kernel)
    "quantize_per_channel", "dequantize", "quant_int8",
    # Hook-based KIVI
    "kivi_hooks", "kivi_perplexity_nbit",
    # Physical KIVI
    "kivi_compress_cache", "quant_cache_bytes",
]


# ─────────────────────────────────────────────────────────────────────────
# Generic n-bit asymmetric quantize-then-dequantize
# ─────────────────────────────────────────────────────────────────────────
def quantize_per_channel_nbit(x, axis, bits=8, group_size=None):
    """Asymmetric n-bit quantize-then-dequantize along `axis`.

    When `group_size` is set, the reduced axis is split into chunks of that
    size, each with its own (scale, zero-point). Matches the KIVI paper's
    grouping (typical value 32) — bounds the dynamic range each scale
    spans, substantially reduces error at low bit-widths.

    Returns a tensor with the same shape and dtype as `x` but with values
    snapped to the nearest reconstructable level.
    """
    if group_size is None:
        levels = 2 ** bits - 1
        xmin = x.amin(dim=axis, keepdim=True)
        xmax = x.amax(dim=axis, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-8) / levels
        q = ((x - xmin) / scale).round().clamp(0, levels)
        return (q * scale + xmin).to(x.dtype)

    # Grouped path: split reduced axis into chunks of `group_size`
    ndim = x.ndim
    red_axis = axis if axis >= 0 else ndim + axis
    G = x.shape[red_axis]
    if G % group_size != 0:
        # Fall back to ungrouped if the dim doesn't divide evenly
        return quantize_per_channel_nbit(x, axis, bits, group_size=None)

    n_groups = G // group_size
    new_shape = list(x.shape)
    new_shape[red_axis] = n_groups
    new_shape.insert(red_axis + 1, group_size)
    xR = x.reshape(new_shape)

    levels = 2 ** bits - 1
    inner = red_axis + 1
    xmin = xR.amin(dim=inner, keepdim=True)
    xmax = xR.amax(dim=inner, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / levels
    q = ((xR - xmin) / scale).round().clamp(0, levels)
    deq = q * scale + xmin
    return deq.reshape(x.shape).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Int8 packers (returned shapes match the fused Triton kernel's expectations)
# ─────────────────────────────────────────────────────────────────────────
def quantize_per_channel(x, axis):
    """Asymmetric uint8 quantization along `axis`. Returns (q, scale, zero)
    with q in uint8 and scale/zero in fp16. Scales keep `axis` as a size-1
    dim (broadcastable against `x`).
    """
    xmin = x.amin(dim=axis, keepdim=True)
    xmax = x.amax(dim=axis, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / 255.0
    zero = xmin
    q = ((x - zero) / scale).round().clamp(0, 255).to(torch.uint8)
    return q, scale.to(torch.float16), zero.to(torch.float16)


def dequantize(q, scale, zero):
    """Inverse of `quantize_per_channel`. Returns fp16."""
    return q.to(torch.float16) * scale + zero


def quant_int8(x, axis):
    """Asymmetric int8 quantization with the `axis` squeezed out of the
    scale/zero tensors. Scales/zeros are fp32 to match the fused kernel's
    expected dtypes.

    Used directly by `fused_swa_dequant` in `kernels.py` and by the
    unfused-int8 reference path it benchmarks against.
    """
    xmin = x.amin(dim=axis, keepdim=True)
    xmax = x.amax(dim=axis, keepdim=True)
    scale = ((xmax - xmin) / 255.0).clamp(min=1e-8)
    zero = xmin
    q = ((x - zero) / scale).round().clamp(0, 255).to(torch.int8)
    return (q,
            scale.squeeze(axis).to(torch.float32),
            zero.squeeze(axis).to(torch.float32))


# ─────────────────────────────────────────────────────────────────────────
# Hook-based KIVI (used by the perplexity sweeps)
# ─────────────────────────────────────────────────────────────────────────
def kivi_hooks(model, bits=8, residual=32, group_size=None):
    """Install per-channel-K / per-token-V quantize-then-dequantize forward
    hooks on every layer's k_proj / v_proj.

    The last `residual` tokens (KIVI's residual buffer) are kept in fp16 —
    only older tokens get quantized. Returns the list of handles so the
    caller can remove them with `for h in handles: h.remove()`.

    Works on every model whose attention exposes separate `k_proj` and
    `v_proj` Linear layers — Llama-style (SmolLM2) and Qwen2-style both
    qualify.
    """
    handles = []

    def hook_kv(axis):
        def fn(_mod, _inp, out):
            T_ = out.shape[1]
            if T_ <= residual:
                return out
            old, buf = out[:, :T_ - residual], out[:, T_ - residual:]
            return torch.cat([
                quantize_per_channel_nbit(old, axis=axis, bits=bits,
                                          group_size=group_size),
                buf,
            ], dim=1)
        return fn

    for layer in model.model.layers:
        handles += [
            layer.self_attn.k_proj.register_forward_hook(hook_kv(1)),   # per-channel K
            layer.self_attn.v_proj.register_forward_hook(hook_kv(-1)),  # per-token V
        ]
    return handles


def kivi_perplexity_nbit(model, ids_eval, eval_start, k_bits=8, v_bits=None,
                          residual=32, group_size=None):
    """Install KIVI hooks (asymmetric K/V bits supported), measure perplexity,
    and remove the hooks even on exception.

    K=2, V=4 with group_size=32 is the paper-recommended setting (per-channel
    K tolerates lower precision than per-token V).
    """
    if v_bits is None:
        v_bits = k_bits

    handles = []

    def make_hook(axis, bits):
        def hook(_m, _i, o):
            T_ = o.shape[1]
            if T_ <= residual:
                return o
            old, buf = o[:, :T_ - residual], o[:, T_ - residual:]
            return torch.cat([
                quantize_per_channel_nbit(old, axis=axis, bits=bits,
                                          group_size=group_size),
                buf,
            ], dim=1)
        return hook

    for layer in model.model.layers:
        handles += [
            layer.self_attn.k_proj.register_forward_hook(make_hook(1,  k_bits)),
            layer.self_attn.v_proj.register_forward_hook(make_hook(-1, v_bits)),
        ]
    try:
        return perplexity(model, ids_eval, mask=None, eval_start=eval_start)
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────
# Physical KIVI (reference for cache-byte accounting)
# ─────────────────────────────────────────────────────────────────────────
def kivi_compress_cache(cache, residual_size=32):
    """Quantize each layer's KV cache in-place:
      - K compressed per-channel (along the token dim)
      - V compressed per-token (along the head_dim)
      - last `residual_size` tokens kept in fp16 (KIVI's residual buffer)

    The int8 + scale/zero metadata is stashed on `cache._kivi_meta`, and
    the layer tensors are replaced with dequantized fp16 versions so HF
    attention still works. True bytes for cost reporting come from
    `quant_cache_bytes`, which reads `_kivi_meta` rather than counting
    the (re-expanded fp16) layer tensors.
    """
    cache._kivi_meta = []
    for idx, (k, v) in enumerate(_layers(cache)):
        T_ = k.shape[-2]
        if T_ <= residual_size:
            cache._kivi_meta.append(None)
            continue

        k_old, k_buf = k[..., :T_ - residual_size, :], k[..., T_ - residual_size:, :]
        v_old, v_buf = v[..., :T_ - residual_size, :], v[..., T_ - residual_size:, :]

        k_q, k_s, k_z = quantize_per_channel(k_old, axis=-2)  # per-channel K
        v_q, v_s, v_z = quantize_per_channel(v_old, axis=-1)  # per-token V

        cache._kivi_meta.append({
            "k_q": k_q, "k_s": k_s, "k_z": k_z,
            "v_q": v_q, "v_s": v_s, "v_z": v_z,
            "k_buf": k_buf, "v_buf": v_buf,
        })

        # Reconstruct dequantized tensors so HF attention can consume them.
        # Real bytes for cost reporting come from `_kivi_meta` below.
        k_deq = torch.cat([dequantize(k_q, k_s, k_z), k_buf], dim=-2)
        v_deq = torch.cat([dequantize(v_q, v_s, v_z), v_buf], dim=-2)
        _set(cache, idx, k_deq, v_deq)


def quant_cache_bytes(cache):
    """True post-compression byte count for a KIVI cache.

    Falls back to `cache_bytes(cache)` if the cache hasn't been compressed
    (no `_kivi_meta`, or every layer is None because every layer was below
    the residual buffer size).
    """
    if not hasattr(cache, "_kivi_meta") or all(m is None for m in cache._kivi_meta):
        return cache_bytes(cache)

    total = 0
    for (k, v), meta in zip(_layers(cache), cache._kivi_meta):
        if meta is None:
            total += k.numel() * 2 + v.numel() * 2  # fp16 fallback
        else:
            total += meta["k_q"].numel() + meta["v_q"].numel()  # uint8 packs
            for key in ("k_s", "k_z", "v_s", "v_z", "k_buf", "v_buf"):
                total += meta[key].numel() * meta[key].element_size()
    return total
