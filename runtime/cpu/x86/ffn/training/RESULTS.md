# Results — x86 FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of an isolated FFN, N = 8, 64 and 512 tokens (no N = 1 cell), B and A against the dense iso-support control. Machine: an OVH cloud VM, AMD EPYC-Milan, 8 vCPU (shared VM). Run `x86-ffn-training/publication_20261109T222243Z_75788`; `delta` = 0.474.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `x86-ffn-training/publication_20261109T222243Z_75788`

`delta` = 0.474, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N512 | 3/3/3 | 0.506 [0.506–0.506] | 0.450 [0.450–0.450] | gain |
| 1024_4096_d010 | N64 | 3/3/3 | 0.414 [0.414–0.414] | 0.616 [0.616–0.616] | within delta |
| 1024_4096_d010 | N8 | 3/3/3 | 0.283 [0.283–0.283] | 0.987 [0.987–0.987] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 1.208 [1.208–1.208] | 0.802 [0.802–0.802] | within delta |
| 256_1024_d010 | N64 | 3/3/3 | 0.729 [0.729–0.729] | 0.815 [0.815–0.815] | within delta |
| 256_1024_d010 | N8 | 3/3/3 | 0.444 [0.444–0.444] | 1.182 [1.182–1.182] | within delta |
| 512_2048 | N512 | 3/3/3 | 1.559 [1.559–1.559] | 0.404 [0.404–0.404] | gain |
| 512_2048 | N64 | 3/3/3 | 1.048 [1.048–1.048] | 0.684 [0.684–0.684] | within delta |
| 512_2048 | N8 | 3/3/3 | 0.934 [0.934–0.934] | 0.993 [0.993–0.993] | within delta |
| 512_2048_d005 | N512 | 3/3/3 | 0.724 [0.724–0.724] | 0.732 [0.732–0.732] | within delta |
| 512_2048_d005 | N64 | 3/3/3 | 0.481 [0.481–0.481] | 0.820 [0.820–0.820] | within delta |
| 512_2048_d005 | N8 | 3/3/3 | 0.264 [0.264–0.264] | 1.054 [1.054–1.054] | within delta |
| 512_2048_d010 | N512 | 3/3/3 | 0.894 [0.894–0.894] | 0.635 [0.635–0.635] | within delta |
| 512_2048_d010 | N64 | 3/3/3 | 0.664 [0.664–0.664] | 0.773 [0.773–0.773] | within delta |
| 512_2048_d010 | N8 | 3/3/3 | 0.437 [0.437–0.437] | 1.008 [1.008–1.008] | within delta |
| 512_2048_d020 | N512 | 10/10/10 | 1.301 [1.288–1.318] | 0.473 [0.473–0.477] | gain |
| 512_2048_d020 | N64 | 10/10/10 | 0.844 [0.806–0.864] | 0.702 [0.705–0.747] | within delta |
| 512_2048_d020 | N8 | 10/10/10 | 0.719 [0.692–0.745] | 1.018 [0.995–1.024] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 1.508 [1.508–1.508] | 0.272 [0.272–0.272] | gain |
| 512_2048_d050 | N64 | 3/3/3 | 1.482 [1.482–1.482] | 0.717 [0.717–0.717] | within delta |
| 512_2048_d050 | N8 | 3/3/3 | 1.439 [1.439–1.439] | 1.002 [1.002–1.002] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 2.365 [2.365–2.365] | 0.285 [0.285–0.285] | gain |
| 512_2048_d075 | N64 | 3/3/3 | 2.366 [2.366–2.366] | 0.678 [0.678–0.678] | within delta |
| 512_2048_d075 | N8 | 3/3/3 | 2.398 [2.398–2.398] | 1.016 [1.016–1.016] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 2.688 [2.688–2.688] | 0.252 [0.252–0.252] | gain |
| 512_2048_d090 | N64 | 3/3/3 | 2.482 [2.482–2.482] | 0.583 [0.583–0.583] | within delta |
| 512_2048_d090 | N8 | 3/3/3 | 3.043 [3.043–3.043] | 0.975 [0.975–0.975] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 2.693 [2.693–2.693] | 0.212 [0.212–0.212] | gain |
| 512_2048_d100 | N64 | 3/3/3 | 3.192 [3.192–3.192] | 0.603 [0.603–0.603] | within delta |
| 512_2048_d100 | N8 | 3/3/3 | 3.660 [3.660–3.660] | 0.979 [0.979–0.979] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `retained_bytes` | 167,854,096 | 58,518,404 | 58,518,404 | 0.349 | 1.000 | within 2 % |
| 256_1024_d010 | `retained_bytes` | 10,506,256 | 10,848,764 | 10,848,764 | 1.033 | 1.000 | within 2 % |
| 512_2048 | `retained_bytes` | 41,984,016 | 34,275,324 | 34,275,324 | 0.816 | 1.000 | within 2 % |
| 512_2048_d005 | `retained_bytes` | 41,984,016 | 21,703,140 | 21,703,140 | 0.517 | 1.000 | within 2 % |
| 512_2048_d010 | `retained_bytes` | 41,984,016 | 24,213,276 | 24,213,276 | 0.577 | 1.000 | within 2 % |
| 512_2048_d020 | `retained_bytes` | 41,984,016 | 29,257,116 | 29,257,116 | 0.697 | 1.000 | within 2 % |
| 512_2048_d050 | `retained_bytes` | 41,984,016 | 44,363,796 | 44,363,796 | 1.057 | 1.000 | within 2 % |
| 512_2048_d075 | `retained_bytes` | 41,984,016 | 56,942,652 | 56,942,652 | 1.356 | 1.000 | within 2 % |
| 512_2048_d090 | `retained_bytes` | 41,984,016 | 64,494,636 | 64,494,636 | 1.536 | 1.000 | within 2 % |
| 512_2048_d100 | `retained_bytes` | 41,984,016 | 69,521,484 | 69,521,484 | 1.656 | 1.000 | within 2 % |

## Limits

- `delta` is 0.474 (dense copy at `d090` and `d100`, N64). B gains over A beyond it on 7 of 30 cells (median B/A 0.725) and is within it on 23; under such a band "no loss" is a weak statement.
- Memory: B equals A within 2 % on every cell; `retained_bytes` is not stated by the adapter.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
