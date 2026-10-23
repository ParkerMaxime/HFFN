# Results — Nvidia FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step (forward with `z` stored, backward, compact AdamW) of one FFN layer on CUDA against the dense masked control, N = 1, 8, 64 and 512. Machine: Tesla T4 (AWS `g4dn.xlarge`). Run `nvidia-ffn-training/publication_20261021T214713Z_43110`; `delta` = 0.018, reported (not applied).

## Measured

### Run `nvidia-ffn-training/publication_20261021T214713Z_43110`

`delta` = 0.018 (512_2048_d050 / N512 / dense), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 816.9 µs | 181.4 µs | 0.222 | 3/3 | < 1 |
| 512_2048_d005 | N8 | 856.4 µs | 176.7 µs | 0.206 | 3/3 | < 1 |
| 512_2048_d005 | N64 | 933.0 µs | 207.9 µs | 0.223 | 3/3 | < 1 |
| 512_2048_d005 | N512 | 2.00 ms | 830.9 µs | 0.416 | 3/3 | < 1 |
| 512_2048_d010 | N1 | 815.6 µs | 261.9 µs | 0.321 | 3/3 | < 1 |
| 512_2048_d010 | N8 | 855.1 µs | 313.4 µs | 0.366 | 3/3 | < 1 |
| 512_2048_d010 | N64 | 934.5 µs | 381.3 µs | 0.408 | 3/3 | < 1 |
| 512_2048_d010 | N512 | 2.00 ms | 1.45 ms | 0.724 | 3/3 | < 1 |
| 512_2048_d020 | N1 | 815.5 µs | 511.8 µs | 0.628 [0.626–0.628] | 10/10 | < 1 |
| 512_2048_d020 | N8 | 856.1 µs | 611.9 µs | 0.715 [0.714–0.715] | 10/10 | < 1 |
| 512_2048_d020 | N64 | 938.6 µs | 711.0 µs | 0.757 [0.757–0.758] | 10/10 | < 1 |
| 512_2048_d020 | N512 | 2.01 ms | 2.61 ms | 1.297 [1.296–1.298] | 10/10 | > 1 |
| 512_2048_d030 | N1 | 818.2 µs | 763.4 µs | 0.933 | 3/3 | < 1 |
| 512_2048_d030 | N8 | 856.2 µs | 916.9 µs | 1.071 | 3/3 | > 1 |
| 512_2048_d030 | N64 | 943.8 µs | 1.05 ms | 1.113 | 3/3 | > 1 |
| 512_2048_d030 | N512 | 2.04 ms | 3.72 ms | 1.824 | 3/3 | > 1 |
| 512_2048_d050 | N1 | 814.0 µs | 1.25 ms | 1.541 | 3/3 | > 1 |
| 512_2048_d050 | N8 | 854.5 µs | 1.49 ms | 1.744 | 3/3 | > 1 |
| 512_2048_d050 | N64 | 944.6 µs | 1.64 ms | 1.739 | 3/3 | > 1 |
| 512_2048_d050 | N512 | 2.05 ms | 5.80 ms | 2.832 | 3/3 | > 1 |
| 512_2048_d075 | N1 | 815.3 µs | 1.87 ms | 2.289 | 3/3 | > 1 |
| 512_2048_d075 | N8 | 854.7 µs | 2.22 ms | 2.599 | 3/3 | > 1 |
| 512_2048_d075 | N64 | 948.0 µs | 2.43 ms | 2.562 | 3/3 | > 1 |
| 512_2048_d075 | N512 | 2.06 ms | 8.32 ms | 4.044 | 3/3 | > 1 |
| 512_2048_d090 | N1 | 816.1 µs | 2.23 ms | 2.737 | 3/3 | > 1 |
| 512_2048_d090 | N8 | 858.0 µs | 2.67 ms | 3.109 | 3/3 | > 1 |
| 512_2048_d090 | N64 | 952.5 µs | 2.90 ms | 3.041 | 3/3 | > 1 |
| 512_2048_d090 | N512 | 2.06 ms | 9.68 ms | 4.691 | 3/3 | > 1 |
| 512_2048_d100 | N1 | 813.7 µs | 2.51 ms | 3.090 | 3/3 | > 1 |
| 512_2048_d100 | N8 | 853.7 µs | 3.02 ms | 3.536 | 3/3 | > 1 |
| 512_2048_d100 | N64 | 944.5 µs | 3.51 ms | 3.720 | 3/3 | > 1 |
| 512_2048_d100 | N512 | 2.06 ms | 10.63 ms | 5.166 | 3/3 | > 1 |
| 256_1024_d010 | N1 | 726.9 µs | 178.4 µs | 0.245 | 3/3 | < 1 |
| 256_1024_d010 | N8 | 641.6 µs | 174.7 µs | 0.272 | 3/3 | < 1 |
| 256_1024_d010 | N64 | 646.9 µs | 174.9 µs | 0.270 | 3/3 | < 1 |
| 256_1024_d010 | N512 | 838.0 µs | 407.2 µs | 0.486 | 3/3 | < 1 |
| 1024_4096_d010 | N1 | 3.09 ms | 980.3 µs | 0.317 | 3/3 | < 1 |
| 1024_4096_d010 | N8 | 3.23 ms | 1.13 ms | 0.351 | 3/3 | < 1 |
| 1024_4096_d010 | N64 | 3.40 ms | 1.41 ms | 0.413 | 3/3 | < 1 |
| 1024_4096_d010 | N512 | 7.13 ms | 5.63 ms | 0.790 | 3/3 | < 1 |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| 512_2048_d005 | 1115 / 1012 | 106 / 65 | 27 / 23 | 0.848 | 0 / 0 |
| 512_2048_d010 | 1114 / 1051 | 110 / 76 | 29 / 29 | 1.004 | 0 / 0 |
| 512_2048_d020 | 1132 / 1094 | 117 / 100 | 32 / 40 | 1.267 | 0 / 0 |
| 512_2048_d030 | 1146 / 1172 | 123 / 125 | 35 / 52 | 1.479 | 0 / 0 |
| 512_2048_d050 | 1183 / 1313 | 137 / 175 | 42 / 76 | 1.803 | 0 / 0 |
| 512_2048_d075 | 1214 / 1364 | 153 / 231 | 50 / 105 | 2.085 | 0 / 0 |
| 512_2048_d090 | 1235 / 1518 | 165 / 267 | 55 / 123 | 2.214 | 0 / 0 |
| 512_2048_d100 | 1260 / 1127 | 170 / 288 | 59 / 134 | 2.287 | 0 / 0 |
| 256_1024_d010 | 1066 / 989 | 41 / 41 | 7 / 11 | 1.595 | 0 / 0 |
| 1024_4096_d010 | 1299 / 1105 | 385 / 184 | 114 / 81 | 0.708 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.630 (N1), 0.715 (N8), 0.757 (N64), 1.297 (N512).
- At d50 and d75 every N is above 1 (1.541 to 4.044). At d10 every N is below 1 (0.321 to 0.724); the 256_1024 and 1024_4096 cells at d10 are below 1 at every N.
- Memory: the HFFN step retains more than dense on the 512_2048 cells from d20 (40 against 32 MB), less on 1024_4096_d010 (81 against 114 MB); device memory at d20 is lower (100 against 117 MB).

## Limits

- A 410M training cell does not exist on this hardware: Nvidia LLM training has no published leaf.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
