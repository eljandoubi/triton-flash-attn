"""
Tests of the Triton attention kernels.

They run on a GPU, or on a CPU through the Triton interpreter: TRITON_INTERPRET=1 pytest tests
"""

import os

import pytest
import torch

pytest.importorskip("triton")

from ops.attention import SUPPORTED_HEAD_DIMS, default_config, flash_attention  # noqa: E402
from ops.reference import naive_attention  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
CUDA = torch.cuda.is_available() and not INTERPRET
DEVICE = "cuda" if CUDA else "cpu"

pytestmark = pytest.mark.skipif(
    not (CUDA or INTERPRET), reason="needs a GPU or TRITON_INTERPRET=1"
)

if INTERPRET:
    DTYPES = [torch.float32]
    TOL = {torch.float32: 1e-4}
    HEAD_DIMS = (16, 32)
    # batch, heads, seq, head_dim, block_m, block_n
    CASES = [(1, 2, 40, 16, 16, 16), (2, 1, 33, 32, 16, 32), (1, 1, 17, 16, 32, 16)]
else:
    DTYPES = [torch.float16, torch.float32]
    TOL = {torch.float16: 2e-2, torch.bfloat16: 5e-2, torch.float32: 5e-3}
    HEAD_DIMS = SUPPORTED_HEAD_DIMS
    if CUDA and torch.cuda.get_device_capability()[0] >= 8:
        DTYPES.append(torch.bfloat16)
    CASES = [
        (1, 2, 128, 64, 64, 64),
        (2, 3, 100, 64, 64, 64),
        (1, 1, 257, 32, 64, 32),
        (2, 2, 96, 128, 32, 32),
        (1, 2, 64, 16, 16, 16),
        (1, 2, 200, 64, 32, 64),
    ]


def inputs(b, h, n, d, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda: torch.randn((b, h, n, d), generator=g).to(DEVICE, dtype)  # noqa: E731
    return make(), make(), make(), make()


def blocks_for(d, dtype, bm, bn):
    """The block sizes of a case, shrunk where this dtype would not fit the shared memory of a T4."""
    cap = default_config(d, dtype)["block_m"]
    return min(bm, cap), min(bn, cap)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_forward_matches_the_reference(case, dtype, causal):
    b, h, n, d, bm, bn = case
    bm, bn = blocks_for(d, dtype, bm, bn)
    q, k, v, _ = inputs(b, h, n, d, dtype)
    out = flash_attention(q, k, v, causal=causal, block_m=bm, block_n=bn)
    ref = naive_attention(q, k, v, causal)
    assert out.shape == q.shape and out.dtype == dtype
    torch.testing.assert_close(
        out.float(), ref.float(), atol=TOL[dtype], rtol=TOL[dtype]
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_backward_matches_autograd_of_the_reference(case, dtype, causal):
    b, h, n, d, bm, bn = case
    bm, bn = blocks_for(d, dtype, bm, bn)
    q, k, v, do = inputs(b, h, n, d, dtype)
    ours = [x.clone().requires_grad_() for x in (q, k, v)]
    flash_attention(*ours, causal=causal, block_m=bm, block_n=bn).backward(do)
    ref = [x.float().clone().requires_grad_() for x in (q, k, v)]
    naive_attention(*ref, causal).backward(do.float())
    for got, want in zip(ours, ref, strict=True):
        assert got.grad.dtype == dtype
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=TOL[dtype] * 2, rtol=TOL[dtype] * 2
        )


def test_default_blocks_keep_every_tile_within_8_kb():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        size = torch.empty((), dtype=dtype).element_size()
        for d in SUPPORTED_HEAD_DIMS:
            config = default_config(d, dtype)
            assert config["block_m"] == config["block_n"]
            assert 16 <= config["block_m"] <= 64
            assert config["block_m"] * d * size <= 8192


def test_default_blocks_stay_at_32_rows_before_ampere():
    assert default_config(64, torch.float16, (7, 5))["block_m"] == 32
    assert default_config(16, torch.float32, (7, 5))["block_m"] == 32
    assert default_config(64, torch.float16, (8, 0))["block_m"] == 64
    assert default_config(16, torch.float32)["block_m"] == 64
    assert default_config(128, torch.float16, (7, 5))["block_m"] == 32


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", HEAD_DIMS)
def test_default_blocks_run_forward_and_backward_for_every_head_dim(d, dtype):
    q, k, v, do = inputs(1, 2, 70, d, dtype)
    ours = [x.clone().requires_grad_() for x in (q, k, v)]
    out = flash_attention(*ours, causal=True)
    out.backward(do)
    ref = [x.float().clone().requires_grad_() for x in (q, k, v)]
    expected = naive_attention(*ref, True)
    expected.backward(do.float())
    tol = TOL[dtype]
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)
    for got, want in zip(ours, ref, strict=True):
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=2 * tol, rtol=2 * tol
        )


@pytest.mark.parametrize("blocks", [(16, 32), (32, 16)])
def test_unequal_block_sizes_with_a_causal_mask(blocks):
    q, k, v, do = inputs(1, 2, 45, 16, torch.float32)
    ours = [x.clone().requires_grad_() for x in (q, k, v)]
    out = flash_attention(*ours, causal=True, block_m=blocks[0], block_n=blocks[1])
    out.backward(do)
    ref = [x.clone().requires_grad_() for x in (q, k, v)]
    expected = naive_attention(*ref, True)
    expected.backward(do)
    tol = TOL[torch.float32] * 2
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)
    for got, want in zip(ours, ref, strict=True):
        torch.testing.assert_close(got.grad, want.grad, atol=tol, rtol=tol)


def test_a_custom_scale_changes_the_result_like_the_reference():
    q, k, v, _ = inputs(1, 2, 24, 16, torch.float32)
    out = flash_attention(q, k, v, causal=True, sm_scale=0.2, block_m=16, block_n=16)
    tol = TOL[torch.float32]
    torch.testing.assert_close(
        out, naive_attention(q, k, v, True, 0.2), atol=tol, rtol=tol
    )
    assert not torch.allclose(
        out, flash_attention(q, k, v, causal=True, block_m=16, block_n=16)
    )


def test_inputs_with_a_transposed_layout_are_accepted():
    q, k, v, _ = inputs(2, 20, 3, 16, torch.float32)  # (batch, seq, heads, head_dim)
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    out = flash_attention(qt, kt, vt, causal=True, block_m=16, block_n=16)
    tol = TOL[torch.float32]
    torch.testing.assert_close(
        out, naive_attention(qt, kt, vt, True), atol=tol, rtol=tol
    )


def test_the_backward_pass_gives_the_same_bits_every_time():
    q, k, v, do = inputs(2, 2, 100, 16, torch.float32)
    grads = []
    for _ in range(2):
        ours = [x.clone().requires_grad_() for x in (q, k, v)]
        flash_attention(*ours, causal=True, block_m=16, block_n=16).backward(do)
        grads.append([x.grad for x in ours])
    for first, second in zip(*grads, strict=True):
        assert torch.equal(first, second)


def test_a_causal_output_does_not_depend_on_future_keys():
    q, k, v, _ = inputs(1, 1, 40, 16, torch.float32)
    before = flash_attention(q, k, v, causal=True, block_m=16, block_n=16)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 20:], v2[:, :, 20:] = 5.0, -5.0
    after = flash_attention(q, k2, v2, causal=True, block_m=16, block_n=16)
    torch.testing.assert_close(before[:, :, :20], after[:, :, :20])


def test_bad_inputs_are_rejected_before_any_kernel_runs():
    q, k, v, _ = inputs(1, 1, 16, 16, torch.float32)
    with pytest.raises(ValueError, match="same shape"):
        flash_attention(q, k[:, :, :8], v)
    with pytest.raises(ValueError, match="dtype"):
        flash_attention(q, k.double(), v)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attention(*(torch.zeros(1, 1, 16, 24, device=DEVICE) for _ in range(3)))
    with pytest.raises(ValueError, match="block sizes"):
        flash_attention(q, k, v, block_m=8)
    with pytest.raises(ValueError, match="block sizes"):
        flash_attention(q, k, v, block_n=24)
    with pytest.raises(ValueError, match="batch, heads"):
        flash_attention(q[0], k[0], v[0])
    wide = torch.zeros(1, 1, 16, 32, device=DEVICE)[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        flash_attention(wide, wide, wide)
