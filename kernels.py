"""kernels.py — Triton kernels for KVScope.
 
Exports:
 
  triton_sliding_window(Q, K, V, W)
      Sliding-window causal attention. Accepts fp16 or fp32 inputs;
      output is fp32.
 
  fused_swa_dequant(Q, Ki, Vi, Ks, Kz, Vs, Vz, W)
      Novelty contribution. Composes sliding-window eviction with
      in-kernel int8 dequant in a single pass — K and V live in HBM as
      int8 and get dequantized inside SRAM during the inner loop, never
      round-tripping through HBM as fp16.
 
  unfused_int8_path(Q, Ki, Vi, Ks, Kz, Vs, Vz, W)
      Two-kernel reference path used to measure the fusion gain.
      Dequant K/V to fp16 via PyTorch (one kernel each), then attention
      on the fp16 tensors (one Triton kernel). The fused kernel beats
      this by a fusion-bandwidth ratio.
 
Conventions: Q, K, V are (B, H, N, D). Int8 K/V are signed int8 with
fp32 scale/zero tensors — K scale/zero are per-channel (B, H, D),
V scale/zero are per-token (B, H, N).
 
Masked-out attention scores use -1e4 as the sentinel (not -inf or -1e9).
-1e4 is safe in both fp16 (max ~65504) and fp32, and any real attention
score is orders of magnitude smaller in absolute value so the mask still
wins. Using -inf would cause inf-inf=NaN in the streaming softmax update.
"""
 
import torch
import triton
import triton.language as tl
 
__all__ = [
    "sliding_window_kernel",
    "triton_sliding_window",
    "fused_swa_dequant_kernel",
    "fused_swa_dequant",
    "unfused_int8_path",
]
 
 
# Safe masked-out sentinel for both fp16 and fp32.
NEG_LARGE = -1e4
 
 
# ═════════════════════════════════════════════════════════════════════════
# Sliding-window causal attention
# ═════════════════════════════════════════════════════════════════════════
@triton.jit
def sliding_window_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sq_b, sq_h, sq_n, sq_d,
    sk_b, sk_h, sk_n, sk_d,
    sv_b, sv_h, sv_n, sv_d,
    so_b, so_h, so_n, so_d,
    N: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b, h, qb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs = qb * BLOCK_N
    qo = qs + tl.arange(0, BLOCK_N)
    do = tl.arange(0, D)
    Qp = Q_ptr + b * sq_b + h * sq_h
    qm = qo < N
    Q = tl.load(Qp + qo[:, None] * sq_n + do[None, :] * sq_d,
                mask=qm[:, None], other=0.0)
 
    acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_N], -1e4, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
 
    Kp = K_ptr + b * sk_b + h * sk_h
    Vp = V_ptr + b * sv_b + h * sv_h
    ks = max(0, qs - W)
    ke = qs + BLOCK_N
    for kb in range(ks, ke, BLOCK_N):
        ko = kb + tl.arange(0, BLOCK_N)
        km = ko < N
        K = tl.load(Kp + ko[:, None] * sk_n + do[None, :] * sk_d,
                    mask=km[:, None], other=0.0).to(Q.dtype)
        s = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) / (D ** 0.5)
        valid = ((qo[:, None] >= ko[None, :]) &
                 ((qo[:, None] - ko[None, :]) < W) & km[None, :])
        s = tl.where(valid, s, -1e4)
        mn = tl.maximum(m_i, tl.max(s, axis=1))
 
        # Clip exp arguments to keep fp32 from over/underflowing.
        exp_arg_a = tl.maximum(tl.minimum(m_i - mn, 88.0), -150.0)
        exp_arg_p = tl.maximum(tl.minimum(s - mn[:, None], 88.0), -150.0)
        a = tl.exp(exp_arg_a)
        p = tl.exp(exp_arg_p)
 
        V = tl.load(Vp + ko[:, None] * sv_n + do[None, :] * sv_d,
                    mask=km[:, None], other=0.0).to(Q.dtype)
        l_i = a * l_i + tl.sum(p, axis=1)
        # tl.dot requires matching input dtypes; cast p to Q.dtype.
        # fp32 accumulator preserved via out_dtype.
        acc = a[:, None] * acc + tl.dot(p.to(Q.dtype), V, out_dtype=tl.float32)
        m_i = mn
 
    acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    Op = O_ptr + b * so_b + h * so_h
    tl.store(Op + qo[:, None] * so_n + do[None, :] * so_d, acc, mask=qm[:, None])
 
 
def triton_sliding_window(Q, K, V, W):
    B, H, N, D = Q.shape
    O = torch.empty_like(Q, dtype=torch.float32)
    BN = 64
    sliding_window_kernel[(B, H, triton.cdiv(N, BN))](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N=N, D=D, W=W, BLOCK_N=BN,
    )
    return O
 
 
# ═════════════════════════════════════════════════════════════════════════
# Fused kernel: sliding-window + in-kernel int8 dequant (novelty)
# ═════════════════════════════════════════════════════════════════════════
@triton.jit
def fused_swa_dequant_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    Ksc, Kzr, Vsc, Vzr,
    sq_b, sq_h, sq_n, sq_d,
    sk_b, sk_h, sk_n, sk_d,
    sv_b, sv_h, sv_n, sv_d,
    so_b, so_h, so_n, so_d,
    sksb, sksh, sksd,   # K scale/zero strides (B, H, D)
    svsb, svsh, svsd,   # V scale/zero strides (B, H, N)
    N: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b, h, qb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs = qb * BLOCK_N
    qo = qs + tl.arange(0, BLOCK_N)
    do = tl.arange(0, D)
    Qp = Q_ptr + b * sq_b + h * sq_h
    qm = qo < N
    Q = tl.load(Qp + qo[:, None] * sq_n + do[None, :] * sq_d,
                mask=qm[:, None], other=0.0).to(tl.float16)
 
    # K scales/zeros are (B, H, D) — broadcast across all token positions
    Ks = tl.load(Ksc + b * sksb + h * sksh + do * sksd).to(tl.float16)
    Kz = tl.load(Kzr + b * sksb + h * sksh + do * sksd).to(tl.float16)
 
    acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_N], -1e4, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    Kp = K_ptr + b * sk_b + h * sk_h
    Vp = V_ptr + b * sv_b + h * sv_h
 
    ks = max(0, qs - W)
    ke = qs + BLOCK_N
    for kb in range(ks, ke, BLOCK_N):
        ko = kb + tl.arange(0, BLOCK_N)
        km = ko < N
 
        # Load int8 K and dequantize inside SRAM (signed convention).
        Ki = tl.load(Kp + ko[:, None] * sk_n + do[None, :] * sk_d,
                     mask=km[:, None], other=0)
        K = Ki.to(tl.float16) * Ks[None, :] + Kz[None, :]
 
        s = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) / (D ** 0.5)
        valid = ((qo[:, None] >= ko[None, :]) &
                 ((qo[:, None] - ko[None, :]) < W) & km[None, :])
        s = tl.where(valid, s, -1e4)
        mn = tl.maximum(m_i, tl.max(s, axis=1))
 
        # Clip exp arguments to keep fp32 from over/underflowing.
        exp_arg_a = tl.maximum(tl.minimum(m_i - mn, 88.0), -150.0)
        exp_arg_p = tl.maximum(tl.minimum(s - mn[:, None], 88.0), -150.0)
        a = tl.exp(exp_arg_a)
        p = tl.exp(exp_arg_p)
 
        # Load int8 V and its per-token scales/zeros (signed convention).
        Vi = tl.load(Vp + ko[:, None] * sv_n + do[None, :] * sv_d,
                     mask=km[:, None], other=0)
        Vs_ = tl.load(Vsc + b * svsb + h * svsh + ko * svsd,
                      mask=km, other=1.0).to(tl.float16)
        Vz_ = tl.load(Vzr + b * svsb + h * svsh + ko * svsd,
                      mask=km, other=0.0).to(tl.float16)
        V = Vi.to(tl.float16) * Vs_[:, None] + Vz_[:, None]
 
        l_i = a * l_i + tl.sum(p, axis=1)
        # tl.dot requires matching input dtypes in Triton ≥3.6.
        # p is fp32 (from tl.exp), V is fp16. Cast p down; fp32 accumulator
        # preserved via out_dtype. Faster too — fp16 tensor-core matmul
        # with fp32 accumulate is the standard pattern.
        acc = a[:, None] * acc + tl.dot(p.to(tl.float16), V, out_dtype=tl.float32)
        m_i = mn
 
    acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    Op = O_ptr + b * so_b + h * so_h
    tl.store(Op + qo[:, None] * so_n + do[None, :] * so_d, acc, mask=qm[:, None])
 
 
def fused_swa_dequant(Q, Kint, Vint, Ks, Kz, Vs, Vz, W):
    """Sliding-window attention with in-kernel int8 dequant.
 
    Kint, Vint are signed int8 (B, H, N, D). Ks, Kz are fp32 (B, H, D) —
    per-channel K scales/zeros. Vs, Vz are fp32 (B, H, N) — per-token V
    scales/zeros.
    """
    B, H, N, D = Q.shape
    O = torch.zeros_like(Q, dtype=torch.float32)
    BN = 64
    fused_swa_dequant_kernel[(B, H, triton.cdiv(N, BN))](
        Q, Kint, Vint, O, Ks, Kz, Vs, Vz,
        Q.stride(0),    Q.stride(1),    Q.stride(2),    Q.stride(3),
        Kint.stride(0), Kint.stride(1), Kint.stride(2), Kint.stride(3),
        Vint.stride(0), Vint.stride(1), Vint.stride(2), Vint.stride(3),
        O.stride(0),    O.stride(1),    O.stride(2),    O.stride(3),
        Ks.stride(0),   Ks.stride(1),   Ks.stride(2),
        Vs.stride(0),   Vs.stride(1),   Vs.stride(2),
        N=N, D=D, W=W, BLOCK_N=BN,
    )
    return O
 
 
# ═════════════════════════════════════════════════════════════════════════
# Unfused int8 reference path — two kernels, dequant goes through HBM
# ═════════════════════════════════════════════════════════════════════════
def unfused_int8_path(Q, Ki, Vi, Ks, Kz, Vs, Vz, W):
    """Reference path the fused kernel beats. Dequant K/V to fp16 via
    PyTorch (one kernel each), then attention on the fp16 tensors
    (one Triton kernel). The fused kernel saves both kernel launches
    AND the HBM round-trip of the dequantized fp16 K/V.
    """
    K_dq = Ki.to(torch.float16) * Ks.unsqueeze(2) + Kz.unsqueeze(2)
    V_dq = Vi.to(torch.float16) * Vs.unsqueeze(3) + Vz.unsqueeze(3)
    return triton_sliding_window(Q, K_dq, V_dq, W)