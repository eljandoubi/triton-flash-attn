"""
Attention in plain PyTorch.

`naive_attention` is the textbook definition and builds the full score matrix.
`tiled_forward` and `tiled_backward` run the block algorithm of the Triton kernels one program at
a time (online softmax, no score matrix kept), with the same masks and the same base-2 scores, so
the indexing and the maths can be checked on a CPU before any GPU is involved.
"""

import math

import torch

LOG2E = 1.4426950408889634


def _scale(q: torch.Tensor, sm_scale: float | None) -> float:
    return sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """softmax(q k^T * scale) v for (batch, heads, seq, head_dim) tensors, computed in fp32."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * _scale(q, sm_scale)
    if causal:
        n = q.shape[-2]
        keep = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def _rows(x: torch.Tensor, offs: torch.Tensor, n: int) -> torch.Tensor:
    """Rows `offs` of x along the sequence axis, zeros past the end (a masked load)."""
    ok = (offs < n).to(x.dtype)
    return x[:, :, offs.clamp(max=n - 1)] * ok[None, None, :, None]


def _vec(x: torch.Tensor, offs: torch.Tensor, n: int) -> torch.Tensor:
    """The same masked load for a (batch, heads, seq) tensor."""
    return x[:, :, offs.clamp(max=n - 1)] * (offs < n).to(x.dtype)[None, None, :]


def tiled_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 16,
    block_n: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the output and the base-2 log-sum-exp of the scaled scores, shape (B, H, N)."""
    b, h, n, d = q.shape
    qk_scale = _scale(q, sm_scale) * LOG2E
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros(b, h, n, d)
    lse = torch.zeros(b, h, n)
    for pid_m in range(-(-n // block_m)):
        offs_m = pid_m * block_m + torch.arange(block_m)
        row_ok = offs_m < n
        qm = _rows(qf, offs_m, n)
        m_i = torch.full((b, h, block_m), float("-inf"))
        l_i = torch.zeros(b, h, block_m)
        acc = torch.zeros(b, h, block_m, d)
        hi = min((pid_m + 1) * block_m, n) if causal else n
        for start_n in range(0, hi, block_n):
            offs_n = start_n + torch.arange(block_n)
            col_ok = offs_n < n
            kb, vb = _rows(kf, offs_n, n), _rows(vf, offs_n, n)
            s = torch.matmul(qm, kb.transpose(-1, -2)) * qk_scale
            valid = col_ok[None, :]
            if causal:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            s = s.masked_fill(~valid, float("-inf"))
            m_ij = torch.maximum(m_i, s.max(dim=-1).values)
            p = torch.exp2(s - m_ij[..., None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + torch.matmul(p, vb)
            m_i = m_ij
        out[:, :, offs_m[row_ok]] = (acc / l_i[..., None])[:, :, row_ok]
        lse[:, :, offs_m[row_ok]] = (m_i + torch.log2(l_i))[:, :, row_ok]
    return out.to(q.dtype), lse


def tiled_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 16,
    block_n: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of q, k and v. One pass over the key blocks for dk and dv, one over the query
    blocks for dq, so that no program has to add into another program's output."""
    b, h, n, d = q.shape
    scale = _scale(q, sm_scale)
    qk_scale = scale * LOG2E
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    delta = (out.float() * dof).sum(dim=-1)
    dq, dk, dv = (torch.zeros(b, h, n, d) for _ in range(3))

    def probs_and_ds(qm, kb, vb, dom, lse_m, delta_m, offs_m, offs_n):
        s = torch.matmul(qm, kb.transpose(-1, -2)) * qk_scale
        valid = (offs_m < n)[:, None] & (offs_n < n)[None, :]
        if causal:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = torch.where(valid, torch.exp2(s - lse_m[..., None]), torch.zeros(()))
        dp = torch.matmul(dom, vb.transpose(-1, -2))
        return p, p * (dp - delta_m[..., None])

    for pid_n in range(-(-n // block_n)):
        offs_n = pid_n * block_n + torch.arange(block_n)
        col_ok = offs_n < n
        kb, vb = _rows(kf, offs_n, n), _rows(vf, offs_n, n)
        dk_acc = torch.zeros(b, h, block_n, d)
        dv_acc = torch.zeros(b, h, block_n, d)
        lo = (pid_n * block_n // block_m) * block_m if causal else 0
        for start_m in range(lo, n, block_m):
            offs_m = start_m + torch.arange(block_m)
            qm, dom = _rows(qf, offs_m, n), _rows(dof, offs_m, n)
            p, ds = probs_and_ds(
                qm,
                kb,
                vb,
                dom,
                _vec(lse, offs_m, n),
                _vec(delta, offs_m, n),
                offs_m,
                offs_n,
            )
            dv_acc = dv_acc + torch.matmul(p.transpose(-1, -2), dom)
            dk_acc = dk_acc + torch.matmul(ds.transpose(-1, -2), qm)
        dk[:, :, offs_n[col_ok]] = (dk_acc * scale)[:, :, col_ok]
        dv[:, :, offs_n[col_ok]] = dv_acc[:, :, col_ok]

    for pid_m in range(-(-n // block_m)):
        offs_m = pid_m * block_m + torch.arange(block_m)
        row_ok = offs_m < n
        qm, dom = _rows(qf, offs_m, n), _rows(dof, offs_m, n)
        lse_m, delta_m = _vec(lse, offs_m, n), _vec(delta, offs_m, n)
        dq_acc = torch.zeros(b, h, block_m, d)
        hi = min((pid_m + 1) * block_m, n) if causal else n
        for start_n in range(0, hi, block_n):
            offs_n = start_n + torch.arange(block_n)
            kb, vb = _rows(kf, offs_n, n), _rows(vf, offs_n, n)
            _, ds = probs_and_ds(qm, kb, vb, dom, lse_m, delta_m, offs_m, offs_n)
            dq_acc = dq_acc + torch.matmul(ds, kb)
        dq[:, :, offs_m[row_ok]] = (dq_acc * scale)[:, :, row_ok]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
