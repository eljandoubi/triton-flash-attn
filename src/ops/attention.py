"""
Flash attention as a torch op: exact softmax attention with a Triton forward and backward pass.
"""

import torch
import triton
import triton.language as tl
from kernels.attention import attn_bwd_dkdv, attn_bwd_dq, attn_fwd

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)

_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


TILE_BYTES = 8192


def default_config(
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    capability: tuple[int, int] | None = None,
) -> dict:
    """Block sizes and launch options that fit the shared memory of a 64 KB card such as a T4.

    The dk/dv kernel keeps several (block x head_dim) tiles in shared memory at once, so one tile
    is limited to 8 KB. That gives 64 rows up to head_dim 64 in half precision, and fewer rows
    for wider heads or float32 (float32 with head_dim 64 and 64 rows needs 80 KB on a T4).

    Before Ampere (compute capability below 8.0) Triton compiles tl.dot to plain FMA
    instructions, and 64 x 64 tiles at head_dim 64 then spill registers: on a T4 the half
    precision forward pass took 3 to 14 times longer than with 32 x 32 tiles (see the README).
    So the block is capped at 32 rows there.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    block = min(64, TILE_BYTES // (head_dim * itemsize))
    if capability is not None and capability < (8, 0):
        block = min(block, 32)
    return {"block_m": block, "block_n": block, "num_warps": 4 if head_dim <= 64 else 8}


def _check(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_m: int, block_n: int
) -> None:
    if q.ndim != 4:
        raise ValueError("expected tensors of shape (batch, heads, seq, head_dim)")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k and v must have the same shape")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in _TL_DTYPES:
        raise ValueError(
            "q, k and v must share one dtype among float16, bfloat16 and float32"
        )
    if not (q.device == k.device == v.device):
        raise ValueError("q, k and v must be on the same device")
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}")
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("the last dimension of q, k and v must be contiguous")
    for block in (block_m, block_n):
        if block < 16 or block & (block - 1):
            raise ValueError("block sizes must be powers of two, at least 16")


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n, num_warps):
        _check(q, k, v, block_m, block_n)
        b, h, n, d = q.shape
        scale = sm_scale if sm_scale is not None else d**-0.5
        o = torch.empty_like(q)
        lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        attn_fwd[(triton.cdiv(n, block_m), b * h)](
            q,
            k,
            v,
            o,
            lse,
            scale,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            h,
            n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=d,
            CAUSAL=causal,
            OUT_DTYPE=_TL_DTYPES[q.dtype],
            num_warps=num_warps,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.config = (causal, scale, block_m, block_n, num_warps)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal, scale, block_m, block_n, num_warps = ctx.config
        b, h, n, d = q.shape
        do = do.to(q.dtype).contiguous()
        delta = (o.float() * do.float()).sum(dim=-1)
        dq, dk, dv = (
            torch.empty(q.shape, dtype=q.dtype, device=q.device) for _ in range(3)
        )
        options = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "HEAD_DIM": d,
            "CAUSAL": causal,
            "OUT_DTYPE": _TL_DTYPES[q.dtype],
            "num_warps": num_warps,
        }
        shared = (scale, *q.stride(), *k.stride(), *v.stride(), *do.stride(), h, n)
        attn_bwd_dkdv[(triton.cdiv(n, block_n), b * h)](
            q, k, v, do, dk, dv, lse, delta, *shared, **options
        )
        attn_bwd_dq[(triton.cdiv(n, block_m), b * h)](
            q, k, v, do, dq, lse, delta, *shared, **options
        )
        return dq, dk, dv, None, None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """softmax(q k^T * sm_scale) v for (batch, heads, seq, head_dim) tensors.

    sm_scale defaults to head_dim ** -0.5. With causal=True position i only attends to j <= i.
    Memory use grows linearly with the sequence length because the score matrix is never stored.
    """
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else None
    config = default_config(q.shape[-1], q.dtype, capability)
    block_m = block_m or config["block_m"]
    block_n = block_n or config["block_n"]
    num_warps = num_warps or config["num_warps"]
    return FlashAttention.apply(q, k, v, causal, sm_scale, block_m, block_n, num_warps)
