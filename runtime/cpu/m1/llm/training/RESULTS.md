# Results — M1 CPU LLM training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M (batch 1, 32 or 128 tokens) at 10, 20 and 80 % FFN density. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-cpu-llm-training/publication_20261110T014614Z_99837`; `delta` = 0.034.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-cpu-llm-training/publication_20261110T014614Z_99837`

`delta` = 0.034, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | b1_t128 | 3/3/3 | 0.975 [0.975–0.975] | 1.003 [1.003–1.003] | within delta |
| pythia410m_sparse10 | b1_t32 | 3/3/3 | 0.595 [0.595–0.595] | 1.001 [1.001–1.001] | within delta |
| pythia410m_sparse20 | b1_t128 | 10/10/10 | 1.621 [1.618–1.626] | 0.994 [0.997–1.002] | within delta |
| pythia410m_sparse20 | b1_t32 | 10/10/10 | 0.797 [0.788–0.802] | 0.992 [0.986–0.998] | within delta |
| pythia410m_sparse80 | b1_t128 | 3/3/3 | 5.685 [5.685–5.685] | 1.001 [1.001–1.001] | within delta |
| pythia410m_sparse80 | b1_t32 | 3/3/3 | 2.123 [2.123–2.123] | 0.991 [0.991–0.991] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `footprint_timed_max_bytes` | 5,758,654,848 | 2,817,135,272 | 3,108,950,912 | 0.489 | 0.906 | gain |
| pythia410m_sparse20 | `footprint_timed_max_bytes` | 5,708,118,712 | 3,493,574,020 | 3,633,870,320 | 0.612 | 0.961 | gain |
| pythia410m_sparse80 | `footprint_timed_max_bytes` | 6,306,486,848 | 9,221,514,288 | 9,318,114,448 | 1.462 | 0.990 | within 2 % |

## Limits

- Time: B equals A within `delta` (0.034) on every cell. Memory gains beyond 2 % at `sparse10` (B/A 0.906) and `sparse20` (0.961), within 2 % at `sparse80`.
- Form P (one arm per process); memory is `footprint_timed_max_bytes`.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
