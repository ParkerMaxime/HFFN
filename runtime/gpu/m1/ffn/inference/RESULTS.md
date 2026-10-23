# Results — M1 GPU FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One exported FFN on the Metal FFN kernel of `gpu/m1/llm/inference/native_transformer` (one bank, one route) against the dense iso-support PyTorch FFN on MPS, N = 1, 8, 64 and 512. Machine: Apple M1 GPU, 16 GB. Run `m1-gpu-ffn-inference/publication_20261022T003013Z_57076`; `delta` = 0.372, reported (not applied).

## Measured

### Run `m1-gpu-ffn-inference/publication_20261022T003013Z_57076`

`delta` = 0.372 (256_1024_d010 / N1 / dense), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 370.7 µs | 238.0 µs | 0.642 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d005 | N8 | 467.2 µs | 187.0 µs | 0.400 | 3/3 | < 1 |
| 512_2048_d005 | N64 | 645.1 µs | 267.5 µs | 0.415 | 3/3 | < 1 |
| 512_2048_d005 | N512 | 2.25 ms | 824.3 µs | 0.366 | 3/3 | < 1 |
| 512_2048_d010 | N1 | 333.0 µs | 237.2 µs | 0.712 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d010 | N8 | 478.4 µs | 220.1 µs | 0.460 | 3/3 | < 1 |
| 512_2048_d010 | N64 | 605.6 µs | 318.7 µs | 0.526 | 3/3 | < 1 |
| 512_2048_d010 | N512 | 2.16 ms | 1.41 ms | 0.653 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d020 | N1 | 351.8 µs | 256.6 µs | 0.730 [0.720–0.734] | 10/10 | < 1 (inside reported delta) |
| 512_2048_d020 | N8 | 466.0 µs | 277.8 µs | 0.596 [0.582–0.592] | 10/10 | < 1 |
| 512_2048_d020 | N64 | 632.8 µs | 496.5 µs | 0.785 [0.765–0.829] | 10/10 | < 1 (inside reported delta) |
| 512_2048_d020 | N512 | 2.28 ms | 2.54 ms | 1.111 [1.106–1.157] | 10/10 | > 1 (inside reported delta) |
| 512_2048_d030 | N1 | 364.3 µs | 271.2 µs | 0.744 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d030 | N8 | 470.1 µs | 354.8 µs | 0.755 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d030 | N64 | 629.5 µs | 615.6 µs | 0.978 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d030 | N512 | 2.16 ms | 3.67 ms | 1.699 | 3/3 | > 1 |
| 512_2048_d050 | N1 | 330.4 µs | 261.2 µs | 0.791 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d050 | N8 | 457.7 µs | 540.5 µs | 1.181 | 3/3 | > 1 (inside reported delta) |
| 512_2048_d050 | N64 | 597.1 µs | 992.1 µs | 1.661 | 3/3 | > 1 |
| 512_2048_d050 | N512 | 2.29 ms | 5.99 ms | 2.620 | 3/3 | > 1 |
| 512_2048_d075 | N1 | 353.7 µs | 340.9 µs | 0.964 | 3/3 | < 1 (inside reported delta) |
| 512_2048_d075 | N8 | 480.4 µs | 688.5 µs | 1.433 | 3/3 | > 1 |
| 512_2048_d075 | N64 | 605.3 µs | 1.34 ms | 2.214 | 3/3 | > 1 |
| 512_2048_d075 | N512 | 2.29 ms | 8.79 ms | 3.841 | 3/3 | > 1 |
| 512_2048_d090 | N1 | 323.2 µs | 353.3 µs | 1.093 | 3/3 | > 1 (inside reported delta) |
| 512_2048_d090 | N8 | 505.7 µs | 847.8 µs | 1.676 | 3/3 | > 1 |
| 512_2048_d090 | N64 | 591.1 µs | 1.58 ms | 2.679 | 3/3 | > 1 |
| 512_2048_d090 | N512 | 2.31 ms | 10.41 ms | 4.510 | 3/3 | > 1 |
| 512_2048_d100 | N1 | 373.1 µs | 406.5 µs | 1.090 | 3/3 | > 1 (inside reported delta) |
| 512_2048_d100 | N8 | 514.8 µs | 864.2 µs | 1.679 | 3/3 | > 1 |
| 512_2048_d100 | N64 | 609.0 µs | 1.81 ms | 2.973 | 3/3 | > 1 |
| 512_2048_d100 | N512 | 2.17 ms | 11.00 ms | 5.059 | 3/3 | > 1 |
| 256_1024_d010 | N1 | 184.5 µs | 187.4 µs | 1.016 | 3/3 | > 1 (inside reported delta) |
| 256_1024_d010 | N8 | 334.1 µs | 154.8 µs | 0.463 | 3/3 | < 1 |
| 256_1024_d010 | N64 | 422.4 µs | 227.6 µs | 0.539 | 3/3 | < 1 |
| 256_1024_d010 | N512 | 1.03 ms | 443.8 µs | 0.432 | 3/3 | < 1 |
| 1024_4096_d010 | N1 | 841.3 µs | 259.7 µs | 0.309 | 3/3 | < 1 |
| 1024_4096_d010 | N8 | 955.4 µs | 411.1 µs | 0.430 | 3/3 | < 1 |
| 1024_4096_d010 | N64 | 1.23 ms | 850.1 µs | 0.692 | 3/3 | < 1 (inside reported delta) |
| 1024_4096_d010 | N512 | 7.39 ms | 5.44 ms | 0.735 | 3/3 | < 1 (inside reported delta) |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| 512_2048_d005 | 437 / 403 | 42 / 76 | 8 / 1 | 0.116 | 0 / 0 |
| 512_2048_d010 | 439 / 409 | 42 / 76 | 8 / 2 | 0.191 | 0 / 0 |
| 512_2048_d020 | 438 / 417 | 42 / 76 | 8 / 3 | 0.341 | 0 / 0 |
| 512_2048_d030 | 438 / 423 | 42 / 76 | 8 / 4 | 0.490 | 0 / 0 |
| 512_2048_d050 | 438 / 435 | 42 / 76 | 8 / 7 | 0.791 | 0 / 0 |
| 512_2048_d075 | 446 / 472 | 42 / 68 | 8 / 10 | 1.165 | 0 / 0 |
| 512_2048_d090 | 459 / 478 | 42 / 68 | 8 / 12 | 1.390 | 0 / 0 |
| 512_2048_d100 | 446 / 474 | 42 / 101 | 8 / 13 | 1.540 | 0 / 0 |
| 256_1024_d010 | 387 / 368 | 42 / 68 | 2 / 1 | 0.310 | 0 / 0 |
| 1024_4096_d010 | 512 / 502 | 1083 / 101 | 34 / 5 | 0.161 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.730 (N1), 0.596 (N8), 0.785 (N64), 1.111 (N512). Every cell is inside the reported `delta` except N8.
- The `256_1024_d010` and `1024_4096_d010` cells are below 1 at N8 to N512 (0.430 to 0.735), several of them inside `delta`.
- Memory: the HFFN arm retains less but samples more device memory than dense on every cell except `1024_4096_d010` (101 against 1083 MB); the two counters disagree in sign and this is not explained.

## Limits

- `delta` = 0.372 is large and reported, not applied: 25 GPU cells across the GPU entries lie inside the reported band, and their sign is not robust.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
