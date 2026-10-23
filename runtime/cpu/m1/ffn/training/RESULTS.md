# Results — M1 CPU FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of one FFN layer on the Mac (forward on the dual entry of the compact G4 inference kernel with `z` stored; backward and AdamW on the G4 training kernel) against the dense masked control, N = 1, 8, 64 and 512. Machine: Apple M1, 16 GB. Run `m1-cpu-ffn-training/publication_20261021T225431Z_48006`; `delta` = 0.359, applied.

## Measured

### Run `m1-cpu-ffn-training/publication_20261021T225431Z_48006`

`delta` = 0.359 (512_2048_d050 / N1 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 5.24 ms | 338.8 µs | 0.065 | 3/3 | gain |
| 512_2048_d005 | N8 | 6.17 ms | 352.5 µs | 0.057 | 3/3 | gain |
| 512_2048_d005 | N64 | 6.86 ms | 1.88 ms | 0.274 | 3/3 | gain |
| 512_2048_d005 | N512 | 14.27 ms | 16.01 ms | 1.122 | 3/3 | within delta |
| 512_2048_d010 | N1 | 5.39 ms | 550.8 µs | 0.102 | 3/3 | gain |
| 512_2048_d010 | N8 | 6.35 ms | 544.8 µs | 0.086 | 3/3 | gain |
| 512_2048_d010 | N64 | 6.78 ms | 3.61 ms | 0.532 | 3/3 | gain |
| 512_2048_d010 | N512 | 14.59 ms | 30.75 ms | 2.108 | 3/3 | loss |
| 512_2048_d020 | N1 | 5.32 ms | 1.06 ms | 0.199 [0.198–0.201] | 10/10 | gain |
| 512_2048_d020 | N8 | 6.43 ms | 1.02 ms | 0.159 [0.150–0.164] | 10/10 | gain |
| 512_2048_d020 | N64 | 6.97 ms | 7.44 ms | 1.067 [1.033–1.111] | 10/10 | within delta |
| 512_2048_d020 | N512 | 14.88 ms | 76.18 ms | 5.119 [4.886–5.281] | 10/10 | loss |
| 512_2048_d030 | N1 | 5.33 ms | 1.56 ms | 0.293 | 3/3 | gain |
| 512_2048_d030 | N8 | 6.16 ms | 1.41 ms | 0.228 | 3/3 | gain |
| 512_2048_d030 | N64 | 6.93 ms | 11.45 ms | 1.653 | 3/3 | loss |
| 512_2048_d030 | N512 | 14.78 ms | 118.54 ms | 8.018 | 3/3 | loss |
| 512_2048_d050 | N1 | 5.43 ms | 2.62 ms | 0.482 | 3/3 | gain |
| 512_2048_d050 | N8 | 6.53 ms | 2.35 ms | 0.360 | 3/3 | gain |
| 512_2048_d050 | N64 | 7.32 ms | 20.13 ms | 2.748 | 3/3 | loss |
| 512_2048_d050 | N512 | 15.27 ms | 199.94 ms | 13.092 | 3/3 | loss |
| 512_2048_d075 | N1 | 5.53 ms | 4.01 ms | 0.726 | 3/3 | within delta |
| 512_2048_d075 | N8 | 6.62 ms | 3.64 ms | 0.551 | 3/3 | gain |
| 512_2048_d075 | N64 | 7.23 ms | 31.54 ms | 4.361 | 3/3 | loss |
| 512_2048_d075 | N512 | 15.64 ms | 296.44 ms | 18.953 | 3/3 | loss |
| 512_2048_d090 | N1 | 5.38 ms | 4.80 ms | 0.892 | 3/3 | within delta |
| 512_2048_d090 | N8 | 6.47 ms | 4.40 ms | 0.680 | 3/3 | within delta |
| 512_2048_d090 | N64 | 7.01 ms | 37.27 ms | 5.320 | 3/3 | loss |
| 512_2048_d090 | N512 | 15.64 ms | 351.76 ms | 22.490 | 3/3 | loss |
| 512_2048_d100 | N1 | 5.32 ms | 5.37 ms | 1.011 | 3/3 | within delta |
| 512_2048_d100 | N8 | 6.49 ms | 4.58 ms | 0.706 | 3/3 | within delta |
| 512_2048_d100 | N64 | 7.00 ms | 42.14 ms | 6.024 | 3/3 | loss |
| 512_2048_d100 | N512 | 15.61 ms | 389.23 ms | 24.933 | 3/3 | loss |
| 256_1024_d010 | N1 | 1.24 ms | 187.5 µs | 0.151 | 3/3 | gain |
| 256_1024_d010 | N8 | 1.41 ms | 199.0 µs | 0.141 | 3/3 | gain |
| 256_1024_d010 | N64 | 1.81 ms | 1.14 ms | 0.630 | 3/3 | gain |
| 256_1024_d010 | N512 | 4.48 ms | 9.38 ms | 2.093 | 3/3 | loss |
| 1024_4096_d010 | N1 | 22.41 ms | 2.11 ms | 0.094 | 3/3 | gain |
| 1024_4096_d010 | N8 | 24.63 ms | 2.11 ms | 0.086 | 3/3 | gain |
| 1024_4096_d010 | N64 | 27.86 ms | 15.21 ms | 0.546 | 3/3 | gain |
| 1024_4096_d010 | N512 | 63.35 ms | 146.97 ms | 2.320 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| 512_2048_d005 | 684 / 582 | 42 / 3 | 0.061 | 0 / 0 |
| 512_2048_d010 | 689 / 591 | 42 / 5 | 0.121 | 0 / 0 |
| 512_2048_d020 | 724 / 636 | 42 / 10 | 0.241 | 0 / 0 |
| 512_2048_d030 | 759 / 675 | 42 / 15 | 0.361 | 0 / 0 |
| 512_2048_d050 | 856 / 764 | 42 / 25 | 0.601 | 0 / 0 |
| 512_2048_d075 | 963 / 865 | 42 / 38 | 0.901 | 0 / 0 |
| 512_2048_d090 | 986 / 935 | 42 / 45 | 1.081 | 0 / 0 |
| 512_2048_d100 | 1008 / 975 | 42 / 50 | 1.200 | 0 / 0 |
| 256_1024_d010 | 452 / 450 | 11 / 1 | 0.122 | 0 / 0 |
| 1024_4096_d010 | 1216 / 998 | 168 / 20 | 0.121 | 0 / 0 |

## Reading

- Reference cell `512_2048_d020`: 0.199 (N1), 0.159 (N8), 1.067 (N64, within `delta`), 5.119 (N512, a loss).
- N512 is within `delta` at d05 (1.122) and a loss from d10 (2.108), up to 24.933 at d100.
- Memory (adapter count): the HFFN step retains 10 MB against 42 MB for the dense step at d20 (0.241), and more than dense from d90 (1.081).

## Limits

- `delta` = 0.359 is set by `512_2048_d050` N1 hffn, where the process spread is 65 %; many cells read *within delta*.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
