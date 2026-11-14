"""
Speed and memory of the Triton attention against PyTorch's scaled_dot_product_attention and the
textbook formula, on one GPU.

    PYTHONPATH=src python benchmarks/bench.py --out results

Times are medians from triton.testing.do_bench. TFLOP/s uses the usual attention count: 4 * B * H
* N^2 * D for the forward pass (halved when causal) and 3.5 times that for forward plus backward.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from ops.attention import flash_attention
from ops.reference import naive_attention

TUNE_CONFIGS = [
    (64, 64, 4),
    (64, 64, 8),
    (64, 32, 4),
    (32, 64, 4),
    (32, 32, 4),
    (32, 32, 2),
    (32, 32, 8),
    (16, 16, 4),
    (128, 64, 4),
    (128, 64, 8),
]


def sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flops(b, h, n, d, causal, with_backward):
    total = 4 * b * h * n * n * d
    if causal:
        total //= 2
    return total * 3.5 if with_backward else total


def measure(fn, q, k, v, do):
    """Forward ms, forward plus backward ms and the extra peak memory of a training step (MB)."""
    fwd = triton.testing.do_bench(lambda: fn(q, k, v))
    both = triton.testing.do_bench(lambda: fn(q, k, v).backward(do))
    for x in (q, k, v):
        x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn(q, k, v).backward(do)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 2**20
    return fwd, both, peak


def candidates(causal, tune, head_dim):
    """Triton configurations to try: the default one, or a small sweep with --tune."""
    if not tune:
        return [(None, None, None)]
    return [c for c in TUNE_CONFIGS if head_dim <= 64 or c[0] <= 64]


def run_case(b, h, n, d, causal, dtype, tune):
    q, k, v = (
        torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    )
    do = torch.randn_like(q)
    rows = []

    def record(name, fn):
        try:
            fwd, both, peak = measure(fn, q, k, v, do)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"impl": name, "oom": True})
            return
        rows.append(
            {
                "impl": name,
                "fwd_ms": fwd,
                "fwd_tflops": flops(b, h, n, d, causal, False) / fwd / 1e9,
                "both_ms": both,
                "both_tflops": flops(b, h, n, d, causal, True) / both / 1e9,
                "peak_mb": peak,
            }
        )

    best = None
    for bm, bn, warps in candidates(causal, tune, d):
        label = "triton" if bm is None else f"triton {bm}x{bn} w{warps}"
        try:
            record(
                label,
                lambda q, k, v: flash_attention(q, k, v, causal, None, bm, bn, warps),
            )
        except Exception as error:  # a block size that does not fit this card
            print(f"  {label}: skipped ({type(error).__name__})")
            continue
        if "fwd_ms" not in rows[-1]:
            continue
        if tune:
            print(
                f"  {label}: forward {rows[-1]['fwd_ms']:.2f} ms, "
                f"forward + backward {rows[-1]['both_ms']:.2f} ms",
                flush=True,
            )
        if best is None or rows[-1]["both_ms"] < best["both_ms"]:
            best = rows[-1]
    triton_rows = [r for r in rows if r["impl"].startswith("triton")]
    if tune and best is not None:
        rows = [r for r in rows if r not in triton_rows] + [best]
    record("torch sdpa", lambda q, k, v: sdpa(q, k, v, causal))
    record("naive", lambda q, k, v: naive_attention(q, k, v, causal))

    with torch.no_grad():
        ours = flash_attention(q, k, v, causal)
        ref = sdpa(q, k, v, causal)
        error = (ours.float() - ref.float()).abs().max().item()
    return rows, error


def table(results, device, dtype):
    lines = [
        f"{device}, {dtype}, medians of triton.testing.do_bench.",
        "",
        "| seq | causal | implementation | forward ms | forward TFLOP/s | fwd+bwd ms "
        "| fwd+bwd TFLOP/s | extra peak MB |",
        "|--:|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for case in results:
        for r in case["rows"]:
            if r.get("oom"):
                cells = ["out of memory", "", "", "", ""]
            else:
                cells = [
                    f"{r['fwd_ms']:.2f}",
                    f"{r['fwd_tflops']:.1f}",
                    f"{r['both_ms']:.2f}",
                    f"{r['both_tflops']:.1f}",
                    f"{r['peak_mb']:.0f}",
                ]
            lines.append(
                f"| {case['seq']} | {case['causal']} | {r['impl']} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--seq", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument(
        "--tune", action="store_true", help="try a few block sizes per case"
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.cuda.get_device_name()
    results = []
    for causal in (False, True):
        for n in args.seq:
            print(f"seq {n} causal={causal}", flush=True)
            rows, error = run_case(
                args.batch, args.heads, n, args.head_dim, causal, dtype, args.tune
            )
            results.append(
                {
                    "seq": n,
                    "causal": causal,
                    "rows": rows,
                    "max_abs_diff_vs_sdpa": error,
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{device} {args.dtype}".lower()).strip("-")
    meta = {
        "device": device,
        "dtype": args.dtype,
        "batch": args.batch,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
    (args.out / f"bench_{slug}.json").write_text(
        json.dumps({**meta, "results": results}, indent=1)
    )
    text = table(results, device, args.dtype)
    (args.out / f"bench_{slug}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
