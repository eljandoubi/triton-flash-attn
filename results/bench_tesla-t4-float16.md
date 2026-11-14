Tesla T4, float16, medians of triton.testing.do_bench.

| seq | causal | implementation | forward ms | forward TFLOP/s | fwd+bwd ms | fwd+bwd TFLOP/s | extra peak MB |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 512 | False | triton | 2.30 | 0.5 | 9.68 | 0.4 | 7 |
| 512 | False | torch sdpa | 0.12 | 8.8 | 0.51 | 7.3 | 7 |
| 512 | False | naive | 0.62 | 1.7 | 1.86 | 2.0 | 70 |
| 1024 | False | triton | 3.22 | 1.3 | 15.83 | 0.9 | 14 |
| 1024 | False | torch sdpa | 0.33 | 13.2 | 1.72 | 8.8 | 14 |
| 1024 | False | naive | 2.52 | 1.7 | 7.50 | 2.0 | 268 |
| 2048 | False | triton | 12.87 | 1.3 | 63.13 | 1.0 | 28 |
| 2048 | False | torch sdpa | 1.32 | 13.0 | 6.19 | 9.7 | 28 |
| 2048 | False | naive | 9.85 | 1.7 | 28.17 | 2.1 | 1048 |
| 4096 | False | triton | 51.27 | 1.3 | 254.55 | 0.9 | 56 |
| 4096 | False | torch sdpa | 5.69 | 12.1 | 23.86 | 10.1 | 57 |
| 4096 | False | naive | 40.87 | 1.7 | 115.19 | 2.1 | 4144 |
| 8192 | False | triton | 207.63 | 1.3 | 1023.65 | 0.9 | 112 |
| 8192 | False | torch sdpa | 22.81 | 12.1 | 98.72 | 9.7 | 113 |
| 8192 | False | naive | out of memory |  |  |  |  |
| 512 | True | triton | 0.56 | 1.0 | 2.53 | 0.7 | 7 |
| 512 | True | torch sdpa | 0.07 | 7.3 | 0.37 | 5.1 | 7 |
| 512 | True | naive | 0.93 | 0.6 | 2.47 | 0.8 | 70 |
| 1024 | True | triton | 1.90 | 1.1 | 9.12 | 0.8 | 14 |
| 1024 | True | torch sdpa | 0.20 | 10.5 | 1.12 | 6.7 | 14 |
| 1024 | True | naive | 3.74 | 0.6 | 9.75 | 0.8 | 269 |
| 2048 | True | triton | 6.98 | 1.2 | 34.24 | 0.9 | 28 |
| 2048 | True | torch sdpa | 0.75 | 11.4 | 3.87 | 7.8 | 28 |
| 2048 | True | naive | 14.64 | 0.6 | 37.73 | 0.8 | 1052 |
| 4096 | True | triton | 27.26 | 1.3 | 134.07 | 0.9 | 56 |
| 4096 | True | torch sdpa | 3.03 | 11.3 | 14.15 | 8.5 | 57 |
| 4096 | True | naive | 60.10 | 0.6 | 149.37 | 0.8 | 4160 |
| 8192 | True | triton | 108.01 | 1.3 | 532.01 | 0.9 | 112 |
| 8192 | True | torch sdpa | 12.14 | 11.3 | 56.20 | 8.6 | 113 |
| 8192 | True | naive | out of memory |  |  |  |  |
