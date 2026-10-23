# Results — x86 FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The x86 compact kernel of this leaf (`hffn` loader, recommended routes, 8 threads) against the dense iso-support PyTorch FFN, one isolated FFN per cell, N = 1, 8, 64 and 512 tokens. Machine: AMD EPYC 9R14, 8 vCPU. Run `x86-ffn-inference/publication_20261021T205721Z_36546`; `delta` = 0.025, applied.

## Measured

### Run `x86-ffn-inference/publication_20261021T205721Z_36546`

`delta` = 0.025 (256_1024_d010 / N8 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 178.3 µs | 12.0 µs | 0.067 | 3/3 | gain |
| 512_2048_d005 | N8 | 169.9 µs | 24.8 µs | 0.146 | 3/3 | gain |
| 512_2048_d005 | N64 | 513.5 µs | 72.8 µs | 0.142 | 3/3 | gain |
| 512_2048_d005 | N512 | 3.13 ms | 549.9 µs | 0.176 | 3/3 | gain |
| 512_2048_d010 | N1 | 176.9 µs | 16.7 µs | 0.094 | 3/3 | gain |
| 512_2048_d010 | N8 | 170.1 µs | 38.9 µs | 0.229 | 3/3 | gain |
| 512_2048_d010 | N64 | 489.1 µs | 106.4 µs | 0.217 | 3/3 | gain |
| 512_2048_d010 | N512 | 2.94 ms | 830.5 µs | 0.283 | 3/3 | gain |
| 512_2048_d020 | N1 | 177.2 µs | 26.2 µs | 0.148 [0.148–0.149] | 10/10 | gain |
| 512_2048_d020 | N8 | 168.4 µs | 68.2 µs | 0.405 [0.406–0.406] | 10/10 | gain |
| 512_2048_d020 | N64 | 526.2 µs | 177.4 µs | 0.337 [0.337–0.337] | 10/10 | gain |
| 512_2048_d020 | N512 | 2.99 ms | 1.41 ms | 0.470 [0.468–0.482] | 10/10 | gain |
| 512_2048_d030 | N1 | 177.4 µs | 35.3 µs | 0.199 | 3/3 | gain |
| 512_2048_d030 | N8 | 167.0 µs | 97.7 µs | 0.585 | 3/3 | gain |
| 512_2048_d030 | N64 | 475.4 µs | 250.0 µs | 0.526 | 3/3 | gain |
| 512_2048_d030 | N512 | 2.91 ms | 1.96 ms | 0.671 | 3/3 | gain |
| 512_2048_d050 | N1 | 175.6 µs | 55.3 µs | 0.315 | 3/3 | gain |
| 512_2048_d050 | N8 | 168.7 µs | 157.4 µs | 0.933 | 3/3 | gain |
| 512_2048_d050 | N64 | 524.1 µs | 392.0 µs | 0.748 | 3/3 | gain |
| 512_2048_d050 | N512 | 3.13 ms | 3.05 ms | 0.976 | 3/3 | within delta |
| 512_2048_d075 | N1 | 175.9 µs | 80.1 µs | 0.456 | 3/3 | gain |
| 512_2048_d075 | N8 | 167.8 µs | 230.9 µs | 1.376 | 3/3 | loss |
| 512_2048_d075 | N64 | 518.1 µs | 560.3 µs | 1.081 | 3/3 | loss |
| 512_2048_d075 | N512 | 2.93 ms | 4.33 ms | 1.479 | 3/3 | loss |
| 512_2048_d090 | N1 | 175.0 µs | 94.6 µs | 0.540 | 3/3 | gain |
| 512_2048_d090 | N8 | 168.9 µs | 271.4 µs | 1.607 | 3/3 | loss |
| 512_2048_d090 | N64 | 497.0 µs | 635.4 µs | 1.278 | 3/3 | loss |
| 512_2048_d090 | N512 | 3.06 ms | 4.98 ms | 1.630 | 3/3 | loss |
| 512_2048_d100 | N1 | 177.2 µs | 103.7 µs | 0.585 | 3/3 | gain |
| 512_2048_d100 | N8 | 168.0 µs | 299.8 µs | 1.785 | 3/3 | loss |
| 512_2048_d100 | N64 | 521.2 µs | 670.6 µs | 1.287 | 3/3 | loss |
| 512_2048_d100 | N512 | 2.94 ms | 5.29 ms | 1.798 | 3/3 | loss |
| 256_1024_d010 | N1 | 66.4 µs | 9.0 µs | 0.135 | 3/3 | gain |
| 256_1024_d010 | N8 | 60.3 µs | 15.9 µs | 0.263 | 3/3 | gain |
| 256_1024_d010 | N64 | 159.2 µs | 38.2 µs | 0.240 | 3/3 | gain |
| 256_1024_d010 | N512 | 944.3 µs | 296.7 µs | 0.314 | 3/3 | gain |
| 1024_4096_d010 | N1 | 868.7 µs | 46.7 µs | 0.054 | 3/3 | gain |
| 1024_4096_d010 | N8 | 897.5 µs | 130.3 µs | 0.145 | 3/3 | gain |
| 1024_4096_d010 | N64 | 2.24 ms | 352.9 µs | 0.158 | 3/3 | gain |
| 1024_4096_d010 | N512 | 12.28 ms | 2.96 ms | 0.241 | 3/3 | gain |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| 512_2048_d005 | 328 / 296 | 8 / 2 | 0.185 | 0 / 0 |
| 512_2048_d010 | 285 / 253 | 8 / 2 | 0.259 | 0 / 0 |
| 512_2048_d020 | 287 / 249 | 8 / 3 | 0.409 | 0 / 0 |
| 512_2048_d030 | 289 / 251 | 8 / 5 | 0.559 | 0 / 0 |
| 512_2048_d050 | 291 / 263 | 8 / 7 | 0.859 | 0 / 0 |
| 512_2048_d075 | 293 / 286 | 8 / 10 | 1.234 | 0 / 0 |
| 512_2048_d090 | 296 / 290 | 8 / 12 | 1.458 | 0 / 0 |
| 512_2048_d100 | 297 / 295 | 8 / 14 | 1.608 | 0 / 0 |
| 256_1024_d010 | 280 / 265 | 2 / 1 | 0.368 | 0 / 0 |
| 1024_4096_d010 | 389 / 306 | 34 / 7 | 0.205 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.148 (N1), 0.405 (N8), 0.337 (N64), 0.470 (N512), a gain at every N beyond `delta`.
- Gain at every N up to d30; at d50 N512 is within `delta` (0.976); from d75, N8, N64 and N512 are losses; N1 stays a gain through d100 (0.585). All three isolated cells at d10 (256_1024, 512_2048, 1024_4096) are gains at every N.
- Memory (measured, adapter count): at d20 the HFFN arm retains 3 MB against 8 MB for the dense pair; from d75 it retains more (10 against 8 MB), because the compact layout carries indices and offsets.

## Limits

- One machine and one boot; the dense control is PyTorch's float32 GEMM on the same threads.
- These ratios compare the isolated FFN only; a whole-model figure is bounded by the non-FFN parts (see the LLM leaf).
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
