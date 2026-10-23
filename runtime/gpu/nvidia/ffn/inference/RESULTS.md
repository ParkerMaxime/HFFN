# Results — Nvidia FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The CUDA kernels of `gpu/nvidia/common/kernels` (policy dispatch) against the dense iso-support PyTorch FFN, one isolated FFN per cell, N = 1, 8, 64 and 512. Machine: Tesla T4 (AWS `g4dn.xlarge`). Run `nvidia-ffn-inference/publication_20261021T205618Z_36126`; `delta` = 0.129, reported (not applied).

## Measured

### Run `nvidia-ffn-inference/publication_20261021T205618Z_36126`

`delta` = 0.129 (512_2048 / N8 / hffn), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 50.7 µs | 29.2 µs | 0.575 | 3/3 | < 1 |
| 512_2048_d005 | N8 | 83.1 µs | 48.8 µs | 0.587 | 3/3 | < 1 |
| 512_2048_d005 | N64 | 129.1 µs | 49.4 µs | 0.383 | 3/3 | < 1 |
| 512_2048_d005 | N512 | 547.5 µs | 262.4 µs | 0.479 | 3/3 | < 1 |
| 512_2048_d010 | N1 | 51.2 µs | 29.4 µs | 0.574 | 3/3 | < 1 |
| 512_2048_d010 | N8 | 83.3 µs | 49.2 µs | 0.590 | 3/3 | < 1 |
| 512_2048_d010 | N64 | 130.8 µs | 70.2 µs | 0.536 | 3/3 | < 1 |
| 512_2048_d010 | N512 | 557.7 µs | 443.3 µs | 0.795 | 3/3 | < 1 |
| 512_2048_d020 | N1 | 50.9 µs | 29.9 µs | 0.587 [0.579–0.591] | 10/10 | < 1 |
| 512_2048_d020 | N8 | 83.6 µs | 83.5 µs | 0.999 [0.998–1.002] | 10/10 | < 1 (inside reported delta) |
| 512_2048_d020 | N64 | 131.7 µs | 123.6 µs | 0.939 [0.939–0.939] | 10/10 | < 1 (inside reported delta) |
| 512_2048_d020 | N512 | 562.5 µs | 771.4 µs | 1.372 [1.369–1.378] | 10/10 | > 1 |
| 512_2048_d030 | N1 | 50.7 µs | 29.5 µs | 0.581 | 3/3 | < 1 |
| 512_2048_d030 | N8 | 83.6 µs | 140.0 µs | 1.674 | 3/3 | > 1 |
| 512_2048_d030 | N64 | 132.2 µs | 202.0 µs | 1.527 | 3/3 | > 1 |
| 512_2048_d030 | N512 | 568.2 µs | 1.07 ms | 1.879 | 3/3 | > 1 |
| 512_2048_d050 | N1 | 51.1 µs | 34.2 µs | 0.670 | 3/3 | < 1 |
| 512_2048_d050 | N8 | 84.0 µs | 263.0 µs | 3.130 | 3/3 | > 1 |
| 512_2048_d050 | N64 | 133.4 µs | 319.3 µs | 2.394 | 3/3 | > 1 |
| 512_2048_d050 | N512 | 579.3 µs | 1.77 ms | 3.057 | 3/3 | > 1 |
| 512_2048_d075 | N1 | 50.7 µs | 49.6 µs | 0.978 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d075 | N8 | 84.2 µs | 395.5 µs | 4.698 | 3/3 | > 1 |
| 512_2048_d075 | N64 | 134.3 µs | 470.6 µs | 3.505 | 3/3 | > 1 |
| 512_2048_d075 | N512 | 590.1 µs | 2.60 ms | 4.403 | 3/3 | > 1 |
| 512_2048_d090 | N1 | 50.8 µs | 58.4 µs | 1.148 | 3/3 | > 1 |
| 512_2048_d090 | N8 | 84.2 µs | 472.3 µs | 5.609 | 3/3 | > 1 |
| 512_2048_d090 | N64 | 134.4 µs | 557.6 µs | 4.149 | 3/3 | > 1 |
| 512_2048_d090 | N512 | 596.5 µs | 3.07 ms | 5.141 | 3/3 | > 1 |
| 512_2048_d100 | N1 | 50.6 µs | 58.8 µs | 1.163 | 3/3 | > 1 |
| 512_2048_d100 | N8 | 84.3 µs | 457.1 µs | 5.423 | 3/3 | > 1 |
| 512_2048_d100 | N64 | 134.4 µs | 642.7 µs | 4.783 | 3/3 | > 1 |
| 512_2048_d100 | N512 | 599.8 µs | 3.40 ms | 5.674 | 3/3 | > 1 |
| 256_1024_d010 | N1 | 50.7 µs | 28.6 µs | 0.565 | 3/3 | < 1 |
| 256_1024_d010 | N8 | 50.2 µs | 48.8 µs | 0.972 | 3/3 | < 1 (inside reported delta) |
| 256_1024_d010 | N64 | 55.1 µs | 49.4 µs | 0.896 | 3/3 | < 1 (inside reported delta) |
| 256_1024_d010 | N512 | 151.8 µs | 118.3 µs | 0.779 | 3/3 | < 1 |
| 1024_4096_d010 | N1 | 136.3 µs | 30.8 µs | 0.226 | 3/3 | < 1 |
| 1024_4096_d010 | N8 | 267.6 µs | 149.8 µs | 0.560 | 3/3 | < 1 |
| 1024_4096_d010 | N64 | 362.0 µs | 290.1 µs | 0.801 | 3/3 | < 1 |
| 1024_4096_d010 | N512 | 2.02 ms | 1.80 ms | 0.890 | 3/3 | < 1 (inside reported delta) |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| 512_2048_d005 | 830 / 871 | 26 / 15 | 8 / 7 | 0.826 | 0 / 0 |
| 512_2048_d010 | 831 / 822 | 26 / 16 | 8 / 8 | 0.901 | 0 / 0 |
| 512_2048_d020 | 834 / 825 | 26 / 19 | 8 / 9 | 1.051 | 0 / 0 |
| 512_2048_d030 | 867 / 856 | 26 / 21 | 8 / 10 | 1.201 | 0 / 0 |
| 512_2048_d050 | 836 / 841 | 26 / 26 | 8 / 13 | 1.501 | 0 / 0 |
| 512_2048_d075 | 839 / 863 | 26 / 33 | 8 / 16 | 1.875 | 0 / 0 |
| 512_2048_d090 | 845 / 865 | 26 / 36 | 8 / 18 | 2.100 | 0 / 0 |
| 512_2048_d100 | 847 / 868 | 26 / 39 | 8 / 19 | 2.250 | 0 / 0 |
| 256_1024_d010 | 835 / 843 | 13 / 7 | 2 / 3 | 1.650 | 0 / 0 |
| 1024_4096_d010 | 847 / 849 | 78 / 37 | 34 / 18 | 0.526 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.587 (N1), 0.999 (N8), 0.939 (N64), 1.372 (N512). N8 and N64 are inside the reported `delta`; N512 is above 1.
- From d50 the N8, N64 and N512 cells are above 1 (3.130, 2.394, 3.057 at d50); N1 stays below 1 up to d75 (0.978, inside `delta`).
- Memory: the HFFN arm retains more than dense from d20 (9 against 8 MB) but samples less device memory at d20 (19 against 26 MB); the two counters disagree in sign and this is not explained.

## Limits

- The dense control is PyTorch eager (separate GELU, per-call allocations); at small N both arms are host-bound, so the ratio partly measures launch count and dispatch.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
