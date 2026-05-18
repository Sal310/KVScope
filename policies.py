"""policies.py — KV cache eviction policies (DSL + implementations + masks).

Three layers, all model-agnostic:

1. DSL — `KVPolicy`, `@kv_policy(...)`, `dispatch_policy(...)`.
   The decorator attaches a policy descriptor to a function so a single
   `@kv_policy(strategy=..., ...)` annotation parameterizes any benchmark.

2. Reference attention (pure PyTorch) — `full_causal_attention`,
   `sliding_window_attention`, `streaming_attention`, `h2o_attention`.
   Used for the DSL unit tests in the notebook; NOT what production decode
   uses (the fused Triton kernel in `kernels.py` is what wins).

3. Eviction operations on a DynamicCache — `evict_cache` (sliding /
   streaming), `snapkv_prune` (Li et al. 2024), `h2o_prune` (Zhang et al.
   2023, one-shot/static variant).

4. Attention masks for perplexity scoring — `make_mask` (causal),
   `make_window_mask` (sliding + optional sink), `snapkv_mask`, `h2o_mask`.
   These let the perplexity sweep simulate eviction without modifying the
   cache, so quality numbers are decoupled from cache-mutation correctness.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils import _layers, _set

__all__ = [
    # DSL
    "KVPolicy", "kv_policy", "dispatch_policy",
    # Reference attention
    "full_causal_attention", "sliding_window_attention",
    "streaming_attention", "h2o_attention",
    # Cache-level eviction
    "evict_cache", "snapkv_prune", "h2o_prune",
    # Masks for perplexity
    "make_mask", "make_window_mask", "snapkv_mask", "h2o_mask",
    # Architectural sharing (MQA/GQA mean-pool surgery)
    "convert_kv_heads_via_meanpool",
]

# Sentinel used in mask construction. Picked to match the dtype of the
# model's logits / scores so masked_fill is a no-op in fp16 arithmetic.
NEG = torch.finfo(torch.float16).min


# ═════════════════════════════════════════════════════════════════════════
# 1. DSL
# ═════════════════════════════════════════════════════════════════════════
@dataclass
class KVPolicy:
    """Declarative description of an eviction policy."""
    strategy:    Literal["full", "sliding_window", "streaming", "h2o"]
    window_size: int = None
    sink_size:   int = 4
    budget:      int = 64


def kv_policy(strategy="full", window_size=None, sink_size=4, budget=64):
    """Decorator that attaches a `KVPolicy` to a function (used in unit tests)."""
    def deco(fn):
        fn.policy = KVPolicy(strategy, window_size, sink_size, budget)
        return fn
    return deco


def dispatch_policy(p: KVPolicy):
    """Map a KVPolicy descriptor to the matching reference-attention callable."""
    if p.strategy == "full":
        return full_causal_attention
    if p.strategy == "sliding_window":
        return lambda q, k, v: sliding_window_attention(q, k, v, p.window_size)
    if p.strategy == "streaming":
        return lambda q, k, v: streaming_attention(q, k, v, p.window_size, p.sink_size)
    if p.strategy == "h2o":
        return lambda q, k, v: h2o_attention(q, k, v, p.budget)
    raise ValueError(f"Unknown strategy: {p.strategy}")


# ═════════════════════════════════════════════════════════════════════════
# 2. Reference attention (pure PyTorch; for unit testing the DSL)
# ═════════════════════════════════════════════════════════════════════════
def full_causal_attention(q, k, v):
    B, H, T, D = q.shape
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    idx = torch.arange(T, device=q.device)
    s = s.masked_fill(~(idx[:, None] >= idx[None, :])[None, None], float("-inf"))
    return torch.softmax(s, -1) @ v


def sliding_window_attention(q, k, v, window_size):
    B, H, T, D = q.shape
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    idx = torch.arange(T, device=q.device)
    keep = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) < window_size)
    s = s.masked_fill(~keep[None, None], float("-inf"))
    return torch.softmax(s, -1) @ v


def streaming_attention(q, k, v, window_size, sink_size=4):
    B, H, T, D = q.shape
    if T > sink_size + window_size:
        k = torch.cat([k[..., :sink_size, :], k[..., -window_size:, :]], dim=-2)
        v = torch.cat([v[..., :sink_size, :], v[..., -window_size:, :]], dim=-2)
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    return torch.softmax(s, -1) @ v


def h2o_attention(q, k, v, budget):
    B, H, T, D = q.shape
    if T <= budget:
        return full_causal_attention(q, k, v)
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    idx = torch.arange(T, device=q.device)
    causal = (idx[:, None] >= idx[None, :])[None, None]
    s = s.masked_fill(~causal, float("-inf"))
    w = torch.softmax(s, -1)
    importance = w.sum(2)
    rec = budget // 2
    recent = torch.zeros(B, H, T, dtype=torch.bool, device=q.device)
    recent[..., -rec:] = True
    importance = importance.masked_fill(recent, -1e9)
    _, top = importance.topk(budget - rec, dim=-1)
    keep = recent.clone(); keep.scatter_(-1, top, True)
    s = s.masked_fill(~(causal & keep.unsqueeze(2)), float("-inf"))
    return torch.softmax(s, -1) @ v


# ═════════════════════════════════════════════════════════════════════════
# 3. Cache-level eviction operations (mutate a DynamicCache in-place)
# ═════════════════════════════════════════════════════════════════════════
def evict_cache(cache, sink, win):
    """Drop everything except the last `win` tokens (plus the first `sink`).
    Mutates `cache` in-place. `sink=0` is sliding-window; `sink>0` is
    StreamingLLM (attention-sink). Layers shorter than `sink + win` are
    left untouched.
    """
    for i, (k, v) in enumerate(_layers(cache)):
        if k.shape[-2] <= sink + win:
            continue
        if sink:
            k = torch.cat([k[..., :sink, :], k[..., -win:, :]], dim=-2)
            v = torch.cat([v[..., :sink, :], v[..., -win:, :]], dim=-2)
        else:
            k, v = k[..., -win:, :], v[..., -win:, :]
        _set(cache, i, k, v)


def snapkv_prune(model, prompt_ids, budget, OW=32, kernel=7):
    """SnapKV one-shot prune (Li et al. 2024).

    Runs the prompt through the model with `output_attentions=True`, scores
    every prompt token by the observation-window's attention over it, and
    keeps the top (`budget - OW`) tokens plus the last `OW` tokens. The
    `kernel`-sized average pool over importances matches the paper's default.

    Returns (pruned_cache, last_token_logits).
    """
    with torch.no_grad():
        out = model(prompt_ids, output_attentions=True, use_cache=True)
    P, cache = prompt_ids.shape[1], out.past_key_values
    if P <= OW or budget - OW >= P - OW:
        return cache, out.logits[:, -1]

    for i, (ks, vs) in enumerate(_layers(cache)):
        a = out.attentions[i]                                  # [B, H, P, P]
        imp = a[:, :, P - OW:P, :P - OW].sum(2).mean(1)        # [B, P-OW]
        imp = F.avg_pool1d(imp.float().unsqueeze(1), kernel, 1, kernel // 2).squeeze(1)
        top = imp.topk(budget - OW, dim=-1).indices[0].sort()[0]
        gi = top.view(1, 1, -1, 1).expand(ks.shape[0], ks.shape[1], -1, ks.shape[-1])
        new_k = torch.cat([torch.gather(ks[:, :, :P - OW], 2, gi), ks[:, :, P - OW:P]], dim=2)
        new_v = torch.cat([torch.gather(vs[:, :, :P - OW], 2, gi), vs[:, :, P - OW:P]], dim=2)
        _set(cache, i, new_k, new_v)
    return cache, out.logits[:, -1]


def h2o_prune(model, prompt_ids, budget, recent_frac=0.5):
    """One-shot Heavy-Hitter Oracle (Zhang et al. 2023, static variant).

    Selects tokens by accumulated prompt attention (summed across queries,
    averaged across layers and heads) and keeps the top-k heavy hitters
    plus a recent window. This is the *static* variant of H2O — it captures
    the heavy-hitter mechanism but skips per-step re-eviction during decode.
    """
    with torch.no_grad():
        out = model(prompt_ids, output_attentions=True, use_cache=True)
    P, cache = prompt_ids.shape[1], out.past_key_values
    if P <= budget:
        return cache, out.logits[:, -1]

    n_recent = max(1, int(budget * recent_frac))
    n_heavy = budget - n_recent
    n_heavy_eff = min(n_heavy, max(0, P - n_recent))
    recent_idx = torch.arange(P - n_recent, P, device=prompt_ids.device)

    for i, (ks, vs) in enumerate(_layers(cache)):
        a = out.attentions[i]                              # [B, H, P, P]
        imp = a.sum(dim=2).mean(dim=1)                     # [B, P]
        imp_mod = imp.clone()
        imp_mod[:, -n_recent:] = -float("inf")
        if n_heavy_eff > 0:
            top = imp_mod.topk(n_heavy_eff, dim=-1).indices[0].sort()[0]
            keep_idx = torch.cat([top, recent_idx])
        else:
            keep_idx = recent_idx
        gi = keep_idx.view(1, 1, -1, 1).expand(ks.shape[0], ks.shape[1], -1, ks.shape[-1])
        _set(cache, i, torch.gather(ks, 2, gi), torch.gather(vs, 2, gi))
    return cache, out.logits[:, -1]


# ═════════════════════════════════════════════════════════════════════════
# 4. Attention masks for perplexity scoring (no cache mutation)
# ═════════════════════════════════════════════════════════════════════════
def make_mask(T, device):
    """Pure causal attention mask, shape (1, 1, T, T), fp16."""
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    keep = cols <= rows
    return (torch.zeros(T, T, device=device, dtype=torch.float16)
              .masked_fill(~keep, NEG).unsqueeze(0).unsqueeze(0))


def make_window_mask(T, device, window=None, sink=0):
    """Causal + sliding-window + optional attention-sink mask.

    `window=None` collapses to pure causal. `sink=0` collapses to plain
    sliding window. `sink>0` keeps the first `sink` tokens visible to all
    queries (StreamingLLM semantics).
    """
    rows = torch.arange(T, device=device).unsqueeze(1)
    cols = torch.arange(T, device=device).unsqueeze(0)
    keep = cols <= rows
    if window is not None:
        keep = keep & (((rows - cols) < window) | (cols < sink))
    return (torch.zeros(T, T, device=device, dtype=torch.float16)
              .masked_fill(~keep, NEG).unsqueeze(0).unsqueeze(0))


def snapkv_mask(model, ids_eval, P, T, budget, OW=32, kernel=7):
    """Build an attention mask that simulates SnapKV pruning at position P.

    Returns None when pruning would be a no-op (P >= T, P <= OW, or the
    budget already covers everything outside the observation window).
    """
    device = ids_eval.device
    if P >= T or P <= OW or budget - OW >= P - OW:
        return None
    with torch.no_grad():
        attns = model(ids_eval[:, :P], output_attentions=True,
                      use_cache=False).attentions
    imp = torch.stack([a[:, :, P - OW:P, :P - OW].sum(2).mean(1)
                       for a in attns]).mean(0)
    imp = F.avg_pool1d(imp.float().unsqueeze(1), kernel, 1, kernel // 2).squeeze(1)
    top = imp.topk(budget - OW, dim=-1).indices[0]
    m = make_mask(T, device).squeeze()
    keep = torch.zeros(P - OW, device=device, dtype=torch.bool); keep[top] = True
    m[P:T, :P - OW] = m[P:T, :P - OW].masked_fill(
        (~keep)[None, :].expand(T - P, -1), NEG)
    return m.unsqueeze(0).unsqueeze(0)


def h2o_mask(model, ids_eval, P, T, budget, recent_frac=0.5):
    """Build an attention mask that simulates H2O pruning at position P."""
    device = ids_eval.device
    if P >= T or budget >= P:
        return None
    with torch.no_grad():
        attns = model(ids_eval[:, :P], output_attentions=True,
                      use_cache=False).attentions

    n_recent = max(1, int(budget * recent_frac))
    n_heavy = budget - n_recent
    imp = torch.stack([a.sum(dim=2).mean(dim=1) for a in attns]).mean(0)
    imp_mod = imp.clone()
    imp_mod[:, -n_recent:] = -float("inf")

    n_heavy_eff = min(n_heavy, max(0, P - n_recent))
    top = (imp_mod.topk(n_heavy_eff, dim=-1).indices[0]
           if n_heavy_eff > 0
           else torch.empty(0, dtype=torch.long, device=device))

    keep_in_prompt = torch.zeros(P, device=device, dtype=torch.bool)
    keep_in_prompt[top] = True
    keep_in_prompt[P - n_recent:] = True

    m = make_mask(T, device).squeeze()
    m[P:T, :P] = m[P:T, :P].masked_fill(
        (~keep_in_prompt)[None, :].expand(T - P, -1), NEG)
    return m.unsqueeze(0).unsqueeze(0)


# ═════════════════════════════════════════════════════════════════════════
# 5. Architectural sharing — MQA/GQA mean-pool surgery
# ═════════════════════════════════════════════════════════════════════════
def convert_kv_heads_via_meanpool(src_model, target_H_kv):
    """Apply the GQA paper's mean-pool recipe (Ainslie et al. 2023): collapse
    `H_kv_old` key/value heads down to `target_H_kv` by averaging groups of
    (`H_kv_old / target_H_kv`) heads together.

    This is a zero-shot conversion — no fine-tuning, no calibration. Quality
    typically degrades catastrophically for aggressive reductions but the
    cache footprint shrinks linearly with `target_H_kv`. The §6b sweep uses
    this to measure cache_MB vs perplexity across MHA → GQA → MQA on a
    single base model.

    Deepcopies on CPU first to avoid two full models on GPU concurrently,
    then moves the result back to the source device.
    """
    src_device = next(src_model.parameters()).device
    m = deepcopy(src_model.cpu())
    src_model.to(src_device)
    cfg = m.config
    H_q = cfg.num_attention_heads
    H_kv_old = cfg.num_key_value_heads
    D = cfg.hidden_size // H_q
    hidden = cfg.hidden_size

    if H_kv_old % target_H_kv != 0:
        raise ValueError(
            f"H_kv_old={H_kv_old} not divisible by target_H_kv={target_H_kv}")
    pool_factor = H_kv_old // target_H_kv

    for layer in m.model.layers:
        attn = layer.self_attn
        for proj_name in ("k_proj", "v_proj"):
            old = getattr(attn, proj_name)
            W = old.weight.data.view(target_H_kv, pool_factor, D, hidden)
            W_pooled = W.mean(dim=1)
            new_out = target_H_kv * D
            new = nn.Linear(hidden, new_out,
                            bias=(old.bias is not None)).to(W_pooled.device,
                                                            W_pooled.dtype)
            new.weight.data = W_pooled.reshape(new_out, hidden).contiguous()
            if old.bias is not None:
                b = old.bias.data.view(target_H_kv, pool_factor, D).mean(dim=1)
                new.bias.data = b.reshape(new_out).contiguous()
            setattr(attn, proj_name, new)
        attn.num_key_value_heads = target_H_kv
        attn.num_key_value_groups = H_q // target_H_kv
    cfg.num_key_value_heads = target_H_kv
    m = m.to(src_device)
    return m
