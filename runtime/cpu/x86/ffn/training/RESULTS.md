# Results — x86 FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step (forward with `z` stored, backward, AdamW) of one FFN layer: the HFFN step of this leaf against the dense masked control, N = 8, 64 and 512 tokens (no N1 cell: the decode path of the inference kernel does not store `z`). Machine: AMD EPYC 9R14, 8 vCPU. Run `x86-ffn-training/publication_20261021T215219Z_43759`; `delta` = 0.120, applied.

## Measured

### Run `x86-ffn-training/publication_20261021T215219Z_43759`

`delta` = 0.120 (512_2048_d020 / N512 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N8 | 4.95 ms | 426.1 µs | 0.086 | 3/3 | gain |
| 512_2048_d005 | N64 | 7.72 ms | 2.27 ms | 0.294 | 3/3 | gain |
| 512_2048_d005 | N512 | 28.13 ms | 22.41 ms | 0.796 | 3/3 | gain |
| 512_2048_d010 | N8 | 5.26 ms | 696.7 µs | 0.132 | 3/3 | gain |
| 512_2048_d010 | N64 | 7.68 ms | 3.23 ms | 0.421 | 3/3 | gain |
| 512_2048_d010 | N512 | 28.96 ms | 37.36 ms | 1.290 | 3/3 | loss |
| 512_2048_d020 | N8 | 5.35 ms | 1.22 ms | 0.229 [0.228–0.230] | 10/10 | gain |
| 512_2048_d020 | N64 | 8.05 ms | 5.48 ms | 0.680 [0.679–0.693] | 10/10 | gain |
| 512_2048_d020 | N512 | 28.74 ms | 61.39 ms | 2.136 [2.102–2.157] | 10/10 | loss |
| 512_2048_d030 | N8 | 5.70 ms | 1.71 ms | 0.300 | 3/3 | gain |
| 512_2048_d030 | N64 | 8.24 ms | 8.20 ms | 0.995 | 3/3 | within delta |
| 512_2048_d030 | N512 | 28.79 ms | 89.87 ms | 3.121 | 3/3 | loss |
| 512_2048_d050 | N8 | 5.73 ms | 2.51 ms | 0.438 | 3/3 | gain |
| 512_2048_d050 | N64 | 8.41 ms | 12.60 ms | 1.498 | 3/3 | loss |
| 512_2048_d050 | N512 | 29.14 ms | 152.31 ms | 5.228 | 3/3 | loss |
| 512_2048_d075 | N8 | 4.98 ms | 3.34 ms | 0.672 | 3/3 | gain |
| 512_2048_d075 | N64 | 7.60 ms | 18.50 ms | 2.434 | 3/3 | loss |
| 512_2048_d075 | N512 | 28.11 ms | 229.41 ms | 8.160 | 3/3 | loss |
| 512_2048_d090 | N8 | 4.38 ms | 3.89 ms | 0.889 | 3/3 | within delta |
| 512_2048_d090 | N64 | 7.00 ms | 21.90 ms | 3.130 | 3/3 | loss |
| 512_2048_d090 | N512 | 27.64 ms | 268.46 ms | 9.714 | 3/3 | loss |
| 512_2048_d100 | N8 | 3.88 ms | 4.99 ms | 1.285 | 3/3 | loss |
| 512_2048_d100 | N64 | 6.56 ms | 24.49 ms | 3.733 | 3/3 | loss |
| 512_2048_d100 | N512 | 26.96 ms | 306.03 ms | 11.351 | 3/3 | loss |
| 256_1024_d010 | N8 | 1.18 ms | 236.3 µs | 0.201 | 3/3 | gain |
| 256_1024_d010 | N64 | 1.88 ms | 1.19 ms | 0.637 | 3/3 | gain |
| 256_1024_d010 | N512 | 7.64 ms | 10.07 ms | 1.318 | 3/3 | loss |
| 1024_4096_d010 | N8 | 23.64 ms | 2.19 ms | 0.093 | 3/3 | gain |
| 1024_4096_d010 | N64 | 32.39 ms | 10.47 ms | 0.323 | 3/3 | gain |
| 1024_4096_d010 | N512 | 113.17 ms | 120.81 ms | 1.068 | 3/3 | within delta |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| 512_2048_d005 | 540 / 362 | 42 / 3 | 0.061 | 0 / 0 |
| 512_2048_d010 | 544 / 359 | 42 / 5 | 0.121 | 0 / 0 |
| 512_2048_d020 | 583 / 409 | 42 / 10 | 0.241 | 0 / 0 |
| 512_2048_d030 | 611 / 454 | 42 / 15 | 0.361 | 0 / 0 |
| 512_2048_d050 | 661 / 505 | 42 / 25 | 0.601 | 0 / 0 |
| 512_2048_d075 | 695 / 570 | 42 / 38 | 0.901 | 0 / 0 |
| 512_2048_d090 | 754 / 601 | 42 / 45 | 1.081 | 0 / 0 |
| 512_2048_d100 | 845 / 608 | 42 / 50 | 1.200 | 0 / 0 |
| 256_1024_d010 | 394 / 301 | 11 / 1 | 0.122 | 0 / 0 |
| 1024_4096_d010 | 1231 / 582 | 168 / 20 | 0.121 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.229 (N8), 0.680 (N64), 2.136 (N512): a gain at N8 and N64 and a loss at N512.
- N512 is a gain at d05 (0.796) and a loss from d10 (1.290); on every cell N512 has the largest ratio of its row.
- Memory (adapter count): the HFFN step retains 10 MB against 42 MB for the dense step at d20 (0.241), and more than dense from d90 (1.081).

## Limits

- No N1 cell. Large blocks favour the dense control (hypothesis, not isolated: the dense GEMM reaches its throughput regime).
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
