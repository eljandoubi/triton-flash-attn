Block sizes and warps on a Tesla T4 (Colab), batch 2, 8 heads, sequence length 2048.
Times in ms (median of triton.testing.do_bench): forward, then forward + backward.
`skip` means the kernel needs more shared memory than the 64 KB of the card.

Command: `PYTHONPATH=src python benchmarks/bench.py --tune --seq 2048 --head-dim D --dtype T`
with the torch 2.14.1+cu130 and triton 3.8.0 of the Colab image, 2026-11-14.

### float16, head_dim 64

| block (m x n), warps | not causal | causal |
|:--|--:|--:|
| 64x64 w4 | 33.38 / 201.12 | 98.29 / 247.90 |
| 64x64 w8 | 13.43 / 67.32 | 17.02 / 38.46 |
| 64x32 w4 | 14.28 / 71.44 | 9.61 / 39.19 |
| 32x64 w4 | 13.99 / 70.89 | 17.57 / 39.29 |
| 32x32 w4 | 12.73 / 62.76 | 6.82 / 33.53 |
| 32x32 w2 | 15.64 / 73.35 | 8.57 / 39.20 |
| 128x64 w4 | skip | skip |
| 128x64 w8 | skip | skip |

### float32, head_dim 64

| block (m x n), warps | not causal | causal |
|:--|--:|--:|
| 64x64 w4 | skip | skip |
| 64x64 w8 | skip | skip |
| 64x32 w4 | 27.82 / 64.60 | 17.23 / 32.81 |
| 32x64 w4 | 13.74 / 63.65 | 8.16 / 34.16 |
| 32x32 w4 | 13.54 / 63.13 | 7.20 / 32.63 |
| 32x32 w2 | 12.92 / 69.15 | 6.85 / 34.49 |
| 128x64 w4 | skip | skip |
| 128x64 w8 | skip | skip |

### float16, head_dim 128

| block (m x n), warps | not causal | causal |
|:--|--:|--:|
| 64x64 w4 | skip | skip |
| 64x64 w8 | skip | skip |
| 64x32 w4 | skip | skip |
| 32x64 w4 | skip | skip |
| 32x32 w4 | 41.78 / 241.37 | 21.11 / 122.36 |
| 32x32 w2 | 143.54 / 493.51 | 105.89 / 406.08 |
| 32x32 w8 | 42.35 / 211.80 | 21.70 / 107.61 |
| 16x16 w4 | 46.83 / 236.12 | 23.80 / 119.12 |

### float16, head_dim 32

| block (m x n), warps | not causal | causal |
|:--|--:|--:|
| 64x64 w4 | 9.84 / 29.24 | 3.14 / 21.13 |
| 64x64 w8 | 7.59 / 34.80 | 4.12 / 18.47 |
| 64x32 w4 | 6.71 / 33.38 | 3.75 / 22.91 |
| 32x64 w4 | 7.68 / 35.15 | 4.14 / 19.03 |
| 32x32 w4 | 7.22 / 32.34 | 3.78 / 16.99 |
| 32x32 w2 | 6.93 / 30.94 | 3.71 / 30.90 |
| 32x32 w8 | 11.61 / 53.89 | 6.09 / 28.27 |
| 16x16 w4 | 13.21 / 59.13 | 6.90 / 30.92 |
| 128x64 w4 | 23.62 / 156.21 | 13.51 / 90.56 |
| 128x64 w8 | 5.63 / 58.00 | 3.17 / 32.19 |
