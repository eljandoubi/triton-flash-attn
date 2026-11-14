"""
Triton kernels for exact attention that never builds the (seq x seq) score matrix.

q, k, v and the output are (batch, heads, seq, head_dim) tensors whose last dimension is
contiguous. Scores are kept in base 2 (exp2 and log2) because that is what the hardware computes,
so the log-sum-exp saved for the backward pass is in base 2 as well.

The forward kernel gives each program one block of query rows and sweeps over the key blocks with
an online softmax. The backward pass uses two kernels, one program per block of keys for dk and dv
and one program per block of queries for dq, so no program adds into another program's output and
the result does not depend on scheduling. `ops/reference.py` runs the same block algorithm in
plain PyTorch.
"""

import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    Out,
    LSE,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        col_ok = offs_n < N_CTX
        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=col_ok[None, :], other=0.0)
        qk = tl.dot(q, k) * qk_scale
        valid = col_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = Out + off_b * stride_ob + off_h * stride_oh
    o_ptrs += offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(OUT_DTYPE), mask=offs_m[:, None] < N_CTX)
    tl.store(LSE + pid_bh * N_CTX + offs_m, m_i + tl.log2(l_i), mask=offs_m < N_CTX)


@triton.jit
def attn_bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    LSE,
    DELTA,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_gb,
    stride_gh,
    stride_gm,
    stride_gd,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # DO, DK and DV are contiguous and share the strides stride_g*.
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m0 = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = offs_n < N_CTX
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    g_off = off_b * stride_gb + off_h * stride_gh

    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=col_ok[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    lo = 0
    if CAUSAL:
        lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + offs_m0
        row_ok = offs_m < N_CTX
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = DO + g_off + offs_m[:, None] * stride_gm + offs_d[None, :] * stride_gd
        q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=row_ok[:, None], other=0.0)
        lse = tl.load(LSE + pid_bh * N_CTX + offs_m, mask=row_ok, other=0.0)
        delta = tl.load(DELTA + pid_bh * N_CTX + offs_m, mask=row_ok, other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = row_ok[:, None] & col_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    dk = dk * sm_scale
    dk_ptrs = DK + g_off + offs_n[:, None] * stride_gm + offs_d[None, :] * stride_gd
    dv_ptrs = DV + g_off + offs_n[:, None] * stride_gm + offs_d[None, :] * stride_gd
    tl.store(dk_ptrs, dk.to(OUT_DTYPE), mask=col_ok[:, None])
    tl.store(dv_ptrs, dv.to(OUT_DTYPE), mask=col_ok[:, None])


@triton.jit
def attn_bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    LSE,
    DELTA,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_gb,
    stride_gh,
    stride_gm,
    stride_gd,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # DO and DQ are contiguous and share the strides stride_g*.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_CTX
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    g_off = off_b * stride_gb + off_h * stride_gh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = DO + g_off + offs_m[:, None] * stride_gm + offs_d[None, :] * stride_gd
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=row_ok[:, None], other=0.0)
    lse = tl.load(LSE + pid_bh * N_CTX + offs_m, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + pid_bh * N_CTX + offs_m, mask=row_ok, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        col_ok = offs_n < N_CTX
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=col_ok[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = row_ok[:, None] & col_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq = dq * sm_scale
    dq_ptrs = DQ + g_off + offs_m[:, None] * stride_gm + offs_d[None, :] * stride_gd
    tl.store(dq_ptrs, dq.to(OUT_DTYPE), mask=row_ok[:, None])
