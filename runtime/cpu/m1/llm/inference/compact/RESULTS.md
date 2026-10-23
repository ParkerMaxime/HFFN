# Results — M1 CPU inference: isolated FFN and Pythia (compact G4)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../../RESULTS.md`](../../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The compact G4 runtime of this leaf (four fixed groups of hidden neurons, exact offsets, no padding) against the dense iso-support PyTorch control, on the Mac. It is the HFFN arm of both entries: the isolated FFN (`m1-cpu-ffn-inference`, which has no separate leaf) and the Pythia-410M whole request. Machine: Apple M1, 16 GB.

## Measured

### Run `m1-cpu-ffn-inference/publication_20261021T204952Z_35029`

`delta` = 0.461 (512_2048_d020 / N8 / dense), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| 512_2048_d005 | N1 | 71.0 µs | 68.9 µs | 0.970 | 3/3 | within delta |
| 512_2048_d005 | N8 | 123.6 µs | 108.5 µs | 0.878 | 3/3 | within delta |
| 512_2048_d005 | N64 | 514.2 µs | 315.4 µs | 0.613 | 3/3 | within delta |
| 512_2048_d005 | N512 | 3.80 ms | 3.28 ms | 0.865 | 3/3 | within delta |
| 512_2048_d010 | N1 | 81.0 µs | 100.4 µs | 1.239 | 3/3 | within delta |
| 512_2048_d010 | N8 | 162.8 µs | 152.6 µs | 0.937 | 3/3 | within delta |
| 512_2048_d010 | N64 | 584.4 µs | 571.8 µs | 0.978 | 3/3 | within delta |
| 512_2048_d010 | N512 | 14.01 ms | 5.81 ms | 0.414 | 3/3 | gain |
| 512_2048_d020 | N1 | 90.7 µs | 165.7 µs | 1.826 [1.397–2.016] | 10/10 | loss |
| 512_2048_d020 | N8 | 152.8 µs | 364.4 µs | 2.385 [1.950–2.506] | 10/10 | loss |
| 512_2048_d020 | N64 | 514.5 µs | 1.48 ms | 2.879 [2.312–2.974] | 10/10 | loss |
| 512_2048_d020 | N512 | 4.26 ms | 14.76 ms | 3.461 [2.872–3.580] | 10/10 | loss |
| 512_2048_d030 | N1 | 77.7 µs | 177.2 µs | 2.280 | 3/3 | loss |
| 512_2048_d030 | N8 | 136.0 µs | 388.0 µs | 2.852 | 3/3 | loss |
| 512_2048_d030 | N64 | 481.8 µs | 1.69 ms | 3.497 | 3/3 | loss |
| 512_2048_d030 | N512 | 3.78 ms | 15.26 ms | 4.034 | 3/3 | loss |
| 512_2048_d050 | N1 | 72.1 µs | 331.2 µs | 4.597 | 3/3 | loss |
| 512_2048_d050 | N8 | 131.8 µs | 725.6 µs | 5.505 | 3/3 | loss |
| 512_2048_d050 | N64 | 567.1 µs | 3.91 ms | 6.894 | 3/3 | loss |
| 512_2048_d050 | N512 | 4.31 ms | 30.43 ms | 7.065 | 3/3 | loss |
| 512_2048_d075 | N1 | 76.2 µs | 507.9 µs | 6.664 | 3/3 | loss |
| 512_2048_d075 | N8 | 184.6 µs | 1.16 ms | 6.311 | 3/3 | loss |
| 512_2048_d075 | N64 | 565.2 µs | 5.75 ms | 10.166 | 3/3 | loss |
| 512_2048_d075 | N512 | 4.37 ms | 44.66 ms | 10.214 | 3/3 | loss |
| 512_2048_d090 | N1 | 80.1 µs | 611.2 µs | 7.634 | 3/3 | loss |
| 512_2048_d090 | N8 | 138.2 µs | 1.34 ms | 9.731 | 3/3 | loss |
| 512_2048_d090 | N64 | 528.6 µs | 6.60 ms | 12.483 | 3/3 | loss |
| 512_2048_d090 | N512 | 4.35 ms | 53.60 ms | 12.329 | 3/3 | loss |
| 512_2048_d100 | N1 | 86.2 µs | 708.3 µs | 8.216 | 3/3 | loss |
| 512_2048_d100 | N8 | 159.9 µs | 1.49 ms | 9.315 | 3/3 | loss |
| 512_2048_d100 | N64 | 557.5 µs | 7.23 ms | 12.974 | 3/3 | loss |
| 512_2048_d100 | N512 | 4.30 ms | 63.01 ms | 14.642 | 3/3 | loss |
| 256_1024_d010 | N1 | 21.4 µs | 45.5 µs | 2.128 | 3/3 | loss |
| 256_1024_d010 | N8 | 40.6 µs | 79.6 µs | 1.961 | 3/3 | loss |
| 256_1024_d010 | N64 | 164.2 µs | 233.4 µs | 1.422 | 3/3 | within delta |
| 256_1024_d010 | N512 | 1.33 ms | 2.78 ms | 2.085 | 3/3 | loss |
| 1024_4096_d010 | N1 | 1.04 ms | 396.9 µs | 0.380 | 3/3 | gain |
| 1024_4096_d010 | N8 | 4.17 ms | 842.4 µs | 0.202 | 3/3 | gain |
| 1024_4096_d010 | N64 | 5.02 ms | 3.99 ms | 0.794 | 3/3 | within delta |
| 1024_4096_d010 | N512 | 17.08 ms | 35.94 ms | 2.104 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| 512_2048_d005 | 450 / 462 | 8 / 1 | 0.078 | 0 / 0 |
| 512_2048_d010 | 450 / 463 | 8 / 1 | 0.153 | 0 / 0 |
| 512_2048_d020 | 444 / 467 | 8 / 3 | 0.303 | 0 / 0 |
| 512_2048_d030 | 458 / 469 | 8 / 4 | 0.452 | 0 / 0 |
| 512_2048_d050 | 455 / 478 | 8 / 6 | 0.753 | 0 / 0 |
| 512_2048_d075 | 461 / 510 | 8 / 9 | 1.127 | 0 / 0 |
| 512_2048_d090 | 467 / 514 | 8 / 11 | 1.352 | 0 / 0 |
| 512_2048_d100 | 463 / 512 | 8 / 13 | 1.501 | 0 / 0 |
| 256_1024_d010 | 362 / 385 | 2 / 0 | 0.155 | 0 / 0 |
| 1024_4096_d010 | 688 / 653 | 34 / 5 | 0.152 | 0 / 0 |

### Run `m1-cpu-llm-inference/publication_20261022T010704Z_62621`

`delta` = 0.044 (pythia410m_sparse20 / decode_b1 / dense), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse20 | decode_b1 | 50.24 ms | 35.69 ms | 0.710 [0.703–0.716] | 10/10 | gain |
| pythia410m_sparse20 | decode_b4 | 119.95 ms | 77.17 ms | 0.643 [0.642–0.647] | 10/10 | gain |
| pythia410m_sparse20 | prefill_b1 | 250.50 ms | 332.65 ms | 1.328 [1.315–1.342] | 10/10 | loss |
| pythia410m_sparse20 | prefill_b4 | 655.30 ms | 1.38 s | 2.105 [2.032–2.128] | 10/10 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse20 | 4976 / 4788 | 1621 / 1058 | 0.653 | 0 / 0 |

### Run `m1-cpu-llm-inference/publication_20261022T104753Z_80203`

`delta` = 0.083 (pythia410m_sparse10 / decode_b1 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | decode_b1 | 48.99 ms | 29.58 ms | 0.604 | 3/3 | gain |
| pythia410m_sparse10 | decode_b4 | 115.43 ms | 63.51 ms | 0.550 | 3/3 | gain |
| pythia410m_sparse10 | prefill_b1 | 226.04 ms | 195.31 ms | 0.864 | 3/3 | gain |
| pythia410m_sparse10 | prefill_b4 | 602.45 ms | 767.10 ms | 1.273 | 3/3 | loss |
| pythia410m_sparse80 | decode_b1 | 49.11 ms | 67.16 ms | 1.367 | 3/3 | loss |
| pythia410m_sparse80 | decode_b4 | 114.36 ms | 170.66 ms | 1.492 | 3/3 | loss |
| pythia410m_sparse80 | prefill_b1 | 241.36 ms | 1.04 s | 4.330 | 3/3 | loss |
| pythia410m_sparse80 | prefill_b4 | 638.75 ms | 4.82 s | 7.547 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse10 | 4694 / 4491 | 1621 / 938 | 0.578 | 0 / 0 |
| pythia410m_sparse80 | 4576 / 5384 | 1621 / 1783 | 1.100 | 0 / 0 |

## Reading

- Isolated FFN (`m1-cpu-ffn-inference/publication_20261021T204952Z_35029`, `delta` = 0.461, applied): at `512_2048_d020` the ratio is 1.826 (N1), 2.385 (N8), 2.879 (N64) and 3.461 (N512), a loss at every N; so is every density from d20 up. The only gains beyond `delta` that are not confounded with the episode of limit 3 of the shared file are at 1024_4096_d010, at N1 (0.380) and N8 (0.202).
- Estimated at d20 N512: the dense control runs at about 250 GMAC/s and the HFFN arm at about 15 GMAC/s on its active MACs. Hypothesis, not isolated: the dense control uses the M1 matrix unit through Accelerate, which the compact path cannot use.
- Pythia-410M, d20 (`m1-cpu-llm-inference/publication_20261022T010704Z_62621`, `delta` = 0.044): decode is a gain (0.710, 0.643) and prefill a loss (1.328, 2.105).
- Pythia-410M, d10 and d80 (`m1-cpu-llm-inference/publication_20261022T104753Z_80203`, `delta` = 0.083): at d10 decode and prefill_b1 are gains (0.604, 0.550, 0.864) and prefill_b4 is a loss (1.273); at d80 every cell is a loss (1.367 to 7.547). The d20 and d10/d80 parts ran on different days and are not compared with each other.
- Memory (adapter count): the HFFN arm retains 938 and 1058 MB at d10 and d20 against 1621 MB dense, and 1783 MB at d80.

## Limits

- The isolated-FFN run exposes an episode of slow dense processes (limit 3 of the shared file): its `512_2048_d010` N512 "gain" (0.414) is not admissible and its `delta` is 0.461.
- The isolated FFN measured here is executed by the compact G4 runtime, not by a separate padded FFN kernel.
- The limits of [`../../../../../RESULTS.md`](../../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
