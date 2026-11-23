# Results — M1 CPU FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step (forward, backward, AdamW of the compact weights) of an isolated FFN, N = 1, 8, 64 and 512 tokens, B and A against the dense iso-support control. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-cpu-ffn-training/publication_20261109T215023Z_60616`; `delta` = 0.074.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-cpu-ffn-training/publication_20261109T215023Z_60616`

`delta` = 0.074, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.099 [0.099–0.099] | 1.070 [1.070–1.070] | within delta |
| 1024_4096_d010 | N512 | 3/3/3 | 0.926 [0.926–0.926] | 0.409 [0.409–0.409] | gain |
| 1024_4096_d010 | N64 | 3/3/3 | 0.290 [0.290–0.290] | 0.555 [0.555–0.555] | gain |
| 1024_4096_d010 | N8 | 3/3/3 | 0.059 [0.059–0.059] | 0.864 [0.864–0.864] | gain |
| 256_1024_d010 | N1 | 3/3/3 | 0.148 [0.148–0.148] | 1.059 [1.059–1.059] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 1.088 [1.088–1.088] | 0.690 [0.690–0.690] | gain |
| 256_1024_d010 | N64 | 3/3/3 | 0.281 [0.281–0.281] | 0.650 [0.650–0.650] | gain |
| 256_1024_d010 | N8 | 3/3/3 | 0.113 [0.113–0.113] | 0.943 [0.943–0.943] | within delta |
| 512_2048 | N1 | 3/3/3 | 0.315 [0.315–0.315] | 1.070 [1.070–1.070] | within delta |
| 512_2048 | N512 | 3/3/3 | 2.701 [2.701–2.701] | 0.356 [0.356–0.356] | gain |
| 512_2048 | N64 | 3/3/3 | 0.747 [0.747–0.747] | 0.490 [0.490–0.490] | gain |
| 512_2048 | N8 | 3/3/3 | 0.176 [0.176–0.176] | 0.889 [0.889–0.889] | gain |
| 512_2048_d005 | N1 | 3/3/3 | 0.073 [0.073–0.073] | 1.132 [1.132–1.132] | **loss** |
| 512_2048_d005 | N512 | 3/3/3 | 0.751 [0.751–0.751] | 0.692 [0.692–0.692] | gain |
| 512_2048_d005 | N64 | 3/3/3 | 0.164 [0.164–0.164] | 0.648 [0.648–0.648] | gain |
| 512_2048_d005 | N8 | 3/3/3 | 0.048 [0.048–0.048] | 0.937 [0.937–0.937] | within delta |
| 512_2048_d010 | N1 | 3/3/3 | 0.116 [0.116–0.116] | 1.081 [1.081–1.081] | **loss** |
| 512_2048_d010 | N512 | 3/3/3 | 1.131 [1.131–1.131] | 0.605 [0.605–0.605] | gain |
| 512_2048_d010 | N64 | 3/3/3 | 0.268 [0.268–0.268] | 0.573 [0.573–0.573] | gain |
| 512_2048_d010 | N8 | 3/3/3 | 0.069 [0.069–0.069] | 0.845 [0.845–0.845] | gain |
| 512_2048_d020 | N1 | 10/10/10 | 0.214 [0.213–0.215] | 1.076 [1.076–1.077] | **loss** |
| 512_2048_d020 | N512 | 10/10/10 | 1.926 [1.918–1.950] | 0.430 [0.429–0.436] | gain |
| 512_2048_d020 | N64 | 10/10/10 | 0.507 [0.496–0.528] | 0.539 [0.531–0.561] | gain |
| 512_2048_d020 | N8 | 10/10/10 | 0.118 [0.115–0.118] | 0.885 [0.879–0.888] | gain |
| 512_2048_d050 | N1 | 3/3/3 | 0.525 [0.525–0.525] | 1.043 [1.043–1.043] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 4.484 [4.484–4.484] | 0.340 [0.340–0.340] | gain |
| 512_2048_d050 | N64 | 3/3/3 | 1.318 [1.318–1.318] | 0.489 [0.489–0.489] | gain |
| 512_2048_d050 | N8 | 3/3/3 | 0.311 [0.311–0.311] | 0.871 [0.871–0.871] | gain |
| 512_2048_d075 | N1 | 3/3/3 | 0.790 [0.790–0.790] | 1.069 [1.069–1.069] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 6.683 [6.683–6.683] | 0.343 [0.343–0.343] | gain |
| 512_2048_d075 | N64 | 3/3/3 | 1.834 [1.834–1.834] | 0.446 [0.446–0.446] | gain |
| 512_2048_d075 | N8 | 3/3/3 | 0.493 [0.493–0.493] | 0.847 [0.847–0.847] | gain |
| 512_2048_d090 | N1 | 3/3/3 | 0.972 [0.972–0.972] | 1.048 [1.048–1.048] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 7.895 [7.895–7.895] | 0.346 [0.346–0.346] | gain |
| 512_2048_d090 | N64 | 3/3/3 | 2.184 [2.184–2.184] | 0.429 [0.429–0.429] | gain |
| 512_2048_d090 | N8 | 3/3/3 | 0.607 [0.607–0.607] | 0.854 [0.854–0.854] | gain |
| 512_2048_d100 | N1 | 3/3/3 | 1.073 [1.073–1.073] | 1.051 [1.051–1.051] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 8.951 [8.951–8.951] | 0.351 [0.351–0.351] | gain |
| 512_2048_d100 | N64 | 3/3/3 | 2.511 [2.511–2.511] | 0.443 [0.443–0.443] | gain |
| 512_2048_d100 | N8 | 3/3/3 | 0.670 [0.670–0.670] | 0.907 [0.907–0.907] | gain |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `retained_bytes` | 167,854,096 | 20,392,836 | 20,261,764 | 0.121 | 1.006 | within 2 % |
| 256_1024_d010 | `retained_bytes` | 10,506,256 | 1,317,372 | 1,284,604 | 0.125 | 1.026 | **loss** |
| 512_2048 | `retained_bytes` | 41,984,016 | 15,212,540 | 15,147,004 | 0.362 | 1.004 | within 2 % |
| 512_2048_d005 | `retained_bytes` | 41,984,016 | 2,640,356 | 2,574,820 | 0.063 | 1.025 | **loss** |
| 512_2048_d010 | `retained_bytes` | 41,984,016 | 5,150,492 | 5,084,956 | 0.123 | 1.013 | within 2 % |
| 512_2048_d020 | `retained_bytes` | 41,984,016 | 10,194,332 | 10,128,796 | 0.243 | 1.006 | within 2 % |
| 512_2048_d050 | `retained_bytes` | 41,984,016 | 25,301,012 | 25,235,476 | 0.603 | 1.003 | within 2 % |
| 512_2048_d075 | `retained_bytes` | 41,984,016 | 37,879,868 | 37,814,332 | 0.902 | 1.002 | within 2 % |
| 512_2048_d090 | `retained_bytes` | 41,984,016 | 45,431,852 | 45,366,316 | 1.082 | 1.001 | within 2 % |
| 512_2048_d100 | `retained_bytes` | 41,984,016 | 50,458,700 | 50,393,164 | 1.202 | 1.001 | within 2 % |

## Limits

- B loses time against A beyond `delta` (0.074) on 3 of 40 cells, all at N1 (`d005` 1.132, `d010` 1.081, `d020` 1.076); it gains beyond `delta` on 28 cells (median B/A 0.769) and is within `delta` on 9.
- Memory: B is 2.55 % above A (B/A 1.0255, beyond the 2 % of the criterion) on `512_2048_d005` and `256_1024_d010`. The bench adapter counts in B the work space `y_local` stored by the owner path, which A does not hold (hypothesis, not isolated). `retained_bytes` is not stated by the adapter.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
