# Results — M1 CPU FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The compact G4 FFN runtime of this leaf against the dense iso-support PyTorch FFN, one isolated FFN per cell, N = 1, 8, 64 and 512 tokens. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-cpu-ffn-inference/publication_20261109T210137Z_34842`; `delta` = 0.341.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-cpu-ffn-inference/publication_20261109T210137Z_34842`

`delta` = 0.341, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.258 [0.258–0.258] | 1.003 [1.003–1.003] | within delta |
| 1024_4096_d010 | N512 | 3/3/3 | 1.394 [1.394–1.394] | 1.022 [1.022–1.022] | within delta |
| 1024_4096_d010 | N64 | 3/3/3 | 0.397 [0.397–0.397] | 0.962 [0.962–0.962] | within delta |
| 1024_4096_d010 | N8 | 3/3/3 | 0.096 [0.096–0.096] | 0.997 [0.997–0.997] | within delta |
| 256_1024_d010 | N1 | 3/3/3 | 2.314 [2.314–2.314] | 1.002 [1.002–1.002] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 1.425 [1.425–1.425] | 0.992 [0.992–0.992] | within delta |
| 256_1024_d010 | N64 | 3/3/3 | 0.838 [0.838–0.838] | 1.009 [1.009–1.009] | within delta |
| 256_1024_d010 | N8 | 3/3/3 | 1.785 [1.785–1.785] | 1.002 [1.002–1.002] | within delta |
| 512_2048 | N1 | 3/3/3 | 2.143 [2.143–2.143] | 1.003 [1.003–1.003] | within delta |
| 512_2048 | N512 | 3/3/3 | 3.473 [3.473–3.473] | 1.003 [1.003–1.003] | within delta |
| 512_2048 | N64 | 3/3/3 | 2.395 [2.395–2.395] | 0.997 [0.997–0.997] | within delta |
| 512_2048 | N8 | 3/3/3 | 2.074 [2.074–2.074] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d005 | N1 | 3/3/3 | 0.812 [0.812–0.812] | 1.003 [1.003–1.003] | within delta |
| 512_2048_d005 | N512 | 3/3/3 | 0.761 [0.761–0.761] | 1.004 [1.004–1.004] | within delta |
| 512_2048_d005 | N64 | 3/3/3 | 0.480 [0.480–0.480] | 1.008 [1.008–1.008] | within delta |
| 512_2048_d005 | N8 | 3/3/3 | 0.663 [0.663–0.663] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d010 | N1 | 3/3/3 | 1.230 [1.230–1.230] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d010 | N512 | 3/3/3 | 1.307 [1.307–1.307] | 0.994 [0.994–0.994] | within delta |
| 512_2048_d010 | N64 | 3/3/3 | 0.807 [0.807–0.807] | 1.002 [1.002–1.002] | within delta |
| 512_2048_d010 | N8 | 3/3/3 | 0.999 [0.999–0.999] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d020 | N1 | 10/10/10 | 1.502 [1.503–1.538] | 1.004 [1.004–1.006] | within delta |
| 512_2048_d020 | N512 | 10/10/10 | 2.526 [2.413–2.551] | 1.021 [0.986–1.023] | within delta |
| 512_2048_d020 | N64 | 10/10/10 | 1.656 [1.613–1.703] | 1.006 [0.998–1.005] | within delta |
| 512_2048_d020 | N8 | 10/10/10 | 1.487 [1.462–1.514] | 0.993 [0.995–1.003] | within delta |
| 512_2048_d050 | N1 | 3/3/3 | 3.062 [3.062–3.062] | 0.995 [0.995–0.995] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 5.685 [5.685–5.685] | 1.015 [1.015–1.015] | within delta |
| 512_2048_d050 | N64 | 3/3/3 | 4.804 [4.804–4.804] | 0.984 [0.984–0.984] | within delta |
| 512_2048_d050 | N8 | 3/3/3 | 3.338 [3.338–3.338] | 0.994 [0.994–0.994] | within delta |
| 512_2048_d075 | N1 | 3/3/3 | 4.216 [4.216–4.216] | 0.997 [0.997–0.997] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 8.077 [8.077–8.077] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d075 | N64 | 3/3/3 | 7.196 [7.196–7.196] | 1.001 [1.001–1.001] | within delta |
| 512_2048_d075 | N8 | 3/3/3 | 3.601 [3.601–3.601] | 0.993 [0.993–0.993] | within delta |
| 512_2048_d090 | N1 | 3/3/3 | 5.111 [5.111–5.111] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 9.221 [9.221–9.221] | 0.995 [0.995–0.995] | within delta |
| 512_2048_d090 | N64 | 3/3/3 | 8.134 [8.134–8.134] | 0.992 [0.992–0.992] | within delta |
| 512_2048_d090 | N8 | 3/3/3 | 5.840 [5.840–5.840] | 0.991 [0.991–0.991] | within delta |
| 512_2048_d100 | N1 | 3/3/3 | 6.467 [6.467–6.467] | 1.001 [1.001–1.001] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 10.104 [10.104–10.104] | 1.014 [1.014–1.014] | within delta |
| 512_2048_d100 | N64 | 3/3/3 | 8.513 [8.513–8.513] | 0.988 [0.988–0.988] | within delta |
| 512_2048_d100 | N8 | 3/3/3 | 7.260 [7.260–7.260] | 0.992 [0.992–0.992] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `retained_bytes` | 33,574,912 | 5,087,978 | 5,087,978 | 0.152 | 1.000 | within 2 % |
| 256_1024_d010 | `retained_bytes` | 2,102,272 | 326,792 | 326,792 | 0.155 | 1.000 | within 2 % |
| 512_2048 | `retained_bytes` | 8,398,848 | 3,798,024 | 3,798,024 | 0.452 | 1.000 | within 2 % |
| 512_2048_d005 | `retained_bytes` | 8,398,848 | 654,978 | 654,978 | 0.078 | 1.000 | within 2 % |
| 512_2048_d010 | `retained_bytes` | 8,398,848 | 1,282,512 | 1,282,512 | 0.153 | 1.000 | within 2 % |
| 512_2048_d020 | `retained_bytes` | 8,398,848 | 2,543,472 | 2,543,472 | 0.303 | 1.000 | within 2 % |
| 512_2048_d050 | `retained_bytes` | 8,398,848 | 6,320,142 | 6,320,142 | 0.753 | 1.000 | within 2 % |
| 512_2048_d075 | `retained_bytes` | 8,398,848 | 9,464,856 | 9,464,856 | 1.127 | 1.000 | within 2 % |
| 512_2048_d090 | `retained_bytes` | 8,398,848 | 11,352,852 | 11,352,852 | 1.352 | 1.000 | within 2 % |
| 512_2048_d100 | `retained_bytes` | 8,398,848 | 12,609,564 | 12,609,564 | 1.501 | 1.000 | within 2 % |

## Limits

- `delta` is 0.341 (set by the dense copy at `512_2048` N8): it is applied, so every B-against-A cell reads *within delta*; under such a band that is a weak statement (a B up to 1.34 times slower than A would read the same). It is a limit of this run, not a finding about B.
- Memory is the adapter's `retained_bytes` (form C); B and A are equal on every cell. The adapter does not state what it counts; it is read in the bench adapter, which is not distributed.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
