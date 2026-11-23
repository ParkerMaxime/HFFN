# Results — M1 GPU LLM inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../../RESULTS.md`](../../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Whole requests of Pythia-410M at 10, 20 and 80 % FFN density: prefill and decode, batch 1 and 4, dense PyTorch attention around the Metal FFN. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-gpu-llm-inference/publication_20261110T051239Z_55600`; `delta` = 0.037.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-gpu-llm-inference/publication_20261110T051239Z_55600`

`delta` = 0.037, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | decode_b1 | 3/3/3 | 0.535 [0.535–0.535] | 1.003 [1.003–1.003] | within delta |
| pythia410m_sparse10 | decode_b4 | 3/3/3 | 0.464 [0.464–0.464] | 1.014 [1.014–1.014] | within delta |
| pythia410m_sparse10 | prefill_b1 | 3/3/3 | 0.644 [0.644–0.644] | 1.003 [1.003–1.003] | within delta |
| pythia410m_sparse10 | prefill_b4 | 3/3/3 | 0.773 [0.773–0.773] | 1.002 [1.002–1.002] | within delta |
| pythia410m_sparse20 | decode_b1 | 10/10/10 | 0.571 [0.570–0.572] | 0.998 [0.993–0.998] | within delta |
| pythia410m_sparse20 | decode_b4 | 10/10/10 | 0.536 [0.536–0.537] | 1.001 [1.000–1.001] | within delta |
| pythia410m_sparse20 | prefill_b1 | 10/10/10 | 0.775 [0.773–0.781] | 0.999 [0.992–1.008] | within delta |
| pythia410m_sparse20 | prefill_b4 | 10/10/10 | 1.025 [1.025–1.026] | 1.001 [1.000–1.003] | within delta |
| pythia410m_sparse80 | decode_b1 | 3/3/3 | 0.880 [0.880–0.880] | 0.996 [0.996–0.996] | within delta |
| pythia410m_sparse80 | decode_b4 | 3/3/3 | 0.895 [0.895–0.895] | 1.000 [1.000–1.000] | within delta |
| pythia410m_sparse80 | prefill_b1 | 3/3/3 | 1.568 [1.568–1.568] | 1.000 [1.000–1.000] | within delta |
| pythia410m_sparse80 | prefill_b4 | 3/3/3 | 2.524 [2.524–2.524] | 1.003 [1.003–1.003] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `footprint_timed_max_bytes` | 5,574,167,672 | 2,972,862,672 | 2,966,735,272 | 0.533 | 1.002 | within 2 % |
| pythia410m_sparse20 | `footprint_timed_max_bytes` | 5,572,267,092 | 3,238,504,884 | 3,226,020,444 | 0.581 | 1.004 | within 2 % |
| pythia410m_sparse80 | `footprint_timed_max_bytes` | 5,571,054,760 | 6,029,824,608 | 6,062,216,040 | 1.082 | 0.995 | within 2 % |

## Limits

- B equals A within `delta` (0.037) and within 2 % of memory on every cell.
- Form pair, without block lifecycle (written limit of the candidate).
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
