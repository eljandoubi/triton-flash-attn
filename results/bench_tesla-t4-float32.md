Tesla T4, float32, medians of triton.testing.do_bench.

| seq | causal | implementation | forward ms | forward TFLOP/s | fwd+bwd ms | fwd+bwd TFLOP/s | extra peak MB |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 512 | False | triton | 1.93 | 0.6 | 7.15 | 0.5 | 8 |
| 512 | False | torch sdpa | 0.40 | 2.7 | 2.46 | 1.5 | 12 |
| 512 | False | naive | 0.58 | 1.8 | 1.91 | 2.0 | 68 |
| 1024 | False | triton | 3.46 | 1.2 | 16.26 | 0.9 | 16 |
| 1024 | False | torch sdpa | 1.59 | 2.7 | 9.11 | 1.6 | 24 |
| 1024 | False | naive | 2.64 | 1.6 | 7.79 | 1.9 | 264 |
| 2048 | False | triton | 13.71 | 1.3 | 63.91 | 0.9 | 32 |
| 2048 | False | torch sdpa | 6.49 | 2.6 | 34.05 | 1.8 | 48 |
| 2048 | False | naive | 11.25 | 1.5 | 31.60 | 1.9 | 1040 |
| 4096 | False | triton | 55.05 | 1.2 | 255.50 | 0.9 | 64 |
| 4096 | False | torch sdpa | 25.35 | 2.7 | 134.98 | 1.8 | 97 |
| 4096 | False | naive | 47.48 | 1.4 | 131.50 | 1.8 | 4128 |
| 512 | True | triton | 0.57 | 0.9 | 2.46 | 0.8 | 8 |
| 512 | True | torch sdpa | 0.25 | 2.1 | 1.45 | 1.3 | 12 |
| 512 | True | naive | 0.89 | 0.6 | 2.42 | 0.8 | 68 |
| 1024 | True | triton | 1.96 | 1.1 | 8.74 | 0.9 | 16 |
| 1024 | True | torch sdpa | 0.87 | 2.5 | 4.73 | 1.6 | 24 |
| 1024 | True | naive | 3.65 | 0.6 | 9.70 | 0.8 | 265 |
| 2048 | True | triton | 7.22 | 1.2 | 32.86 | 0.9 | 32 |
| 2048 | True | torch sdpa | 3.47 | 2.5 | 15.06 | 2.0 | 48 |
| 2048 | True | naive | 14.53 | 0.6 | 37.14 | 0.8 | 1044 |
| 4096 | True | triton | 27.95 | 1.2 | 127.94 | 0.9 | 64 |
| 4096 | True | torch sdpa | 13.21 | 2.6 | 55.54 | 2.2 | 97 |
| 4096 | True | naive | 60.11 | 0.6 | 152.73 | 0.8 | 4144 |
