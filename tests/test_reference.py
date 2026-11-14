import math

import pytest
import torch

from ops.reference import LOG2E, naive_attention, tiled_backward, tiled_forward

SHAPES = [(1, 1, 16, 16), (2, 3, 37, 16), (1, 2, 64, 32), (2, 2, 5, 8), (1, 1, 1, 16)]
BLOCKS = [(16, 16), (8, 16), (16, 8), (32, 8)]


def make(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g) for _ in range(4)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("blocks", BLOCKS)
@pytest.mark.parametrize("causal", [False, True])
def test_tiled_forward_matches_the_textbook_formula(shape, blocks, causal):
    q, k, v, _ = make(shape)
    out, _ = tiled_forward(q, k, v, causal, block_m=blocks[0], block_n=blocks[1])
    torch.testing.assert_close(
        out, naive_attention(q, k, v, causal), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("causal", [False, True])
def test_the_stored_log_sum_exp_is_in_base_two(causal):
    q, k, v, _ = make((2, 2, 37, 16))
    _, lse = tiled_forward(q, k, v, causal)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(16)
    if causal:
        scores = scores.masked_fill(
            ~torch.ones(37, 37, dtype=torch.bool).tril(), float("-inf")
        )
    expected = torch.logsumexp(scores, dim=-1) * LOG2E
    torch.testing.assert_close(lse, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("blocks", BLOCKS)
@pytest.mark.parametrize("causal", [False, True])
def test_tiled_backward_matches_autograd(shape, blocks, causal):
    q, k, v, do = make(shape)
    q, k, v = (x.clone().requires_grad_() for x in (q, k, v))
    naive_attention(q, k, v, causal).backward(do)
    out, lse = tiled_forward(
        q.detach(), k.detach(), v.detach(), causal, block_m=blocks[0], block_n=blocks[1]
    )
    dq, dk, dv = tiled_backward(
        q.detach(),
        k.detach(),
        v.detach(),
        out,
        do,
        lse,
        causal,
        block_m=blocks[0],
        block_n=blocks[1],
    )
    torch.testing.assert_close(dq, q.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dk, k.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dv, v.grad, atol=1e-5, rtol=1e-4)


def test_a_custom_scale_is_used_by_both_directions():
    q, k, v, do = make((1, 2, 21, 16))
    q, k, v = (x.clone().requires_grad_() for x in (q, k, v))
    naive_attention(q, k, v, True, sm_scale=0.3).backward(do)
    out, lse = tiled_forward(q.detach(), k.detach(), v.detach(), True, sm_scale=0.3)
    dq, dk, dv = tiled_backward(
        q.detach(), k.detach(), v.detach(), out, do, lse, True, sm_scale=0.3
    )
    torch.testing.assert_close(
        out,
        naive_attention(q.detach(), k.detach(), v.detach(), True, 0.3),
        atol=1e-5,
        rtol=1e-5,
    )
    for got, want in ((dq, q.grad), (dk, k.grad), (dv, v.grad)):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_causal_output_ignores_future_keys():
    q, k, v, _ = make((1, 1, 24, 16))
    before, _ = tiled_forward(q, k, v, True)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 12:], v2[:, :, 12:] = (
        torch.randn_like(k[:, :, 12:]),
        torch.randn_like(v[:, :, 12:]),
    )
    after, _ = tiled_forward(q, k2, v2, True)
    torch.testing.assert_close(before[:, :, :12], after[:, :, :12])
    assert not torch.allclose(before[:, :, 12:], after[:, :, 12:])


def test_large_scores_do_not_overflow():
    q, k, v, _ = make((1, 1, 32, 16))
    out, lse = tiled_forward(q * 50, k * 50, v, True)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(
        out, naive_attention(q * 50, k * 50, v, True), atol=1e-4, rtol=1e-4
    )


def test_half_precision_inputs_keep_their_dtype():
    q, k, v, do = (x.half() for x in make((1, 2, 20, 16)))
    out, lse = tiled_forward(q, k, v, True)
    dq, dk, dv = tiled_backward(q, k, v, out, do, lse, True)
    assert out.dtype == dq.dtype == dk.dtype == dv.dtype == torch.float16
    assert lse.dtype == torch.float32
