# Results — Nvidia LLM inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Whole requests of Pythia-410M at 10, 20 and 80 % FFN density: prefill and decode, batch 1 and 4. Machine: an OVH cloud VM with one NVIDIA Quadro RTX 5000 (16 GB, compute capability 7.5; shared VM, 4 vCPU). Run `nvidia-llm-inference/publication_20261110T031657Z_5365`; `delta` = 0.055.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `nvidia-llm-inference/publication_20261110T031657Z_5365`

`delta` = 0.055, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | decode_b1 | 3/3/3 | 0.943 [0.943–0.943] | 1.006 [1.006–1.006] | within delta |
| pythia410m_sparse10 | decode_b4 | 3/3/3 | 0.891 [0.891–0.891] | 1.003 [1.003–1.003] | within delta |
| pythia410m_sparse10 | prefill_b1 | 3/3/3 | 1.017 [1.017–1.017] | 1.012 [1.012–1.012] | within delta |
| pythia410m_sparse10 | prefill_b4 | 3/3/3 | 0.992 [0.992–0.992] | 1.002 [1.002–1.002] | within delta |
| pythia410m_sparse20 | decode_b1 | 10/10/10 | 0.951 [0.934–0.959] | 1.010 [0.999–1.020] | within delta |
| pythia410m_sparse20 | decode_b4 | 10/10/10 | 0.919 [0.894–0.930] | 0.999 [0.993–1.009] | within delta |
| pythia410m_sparse20 | prefill_b1 | 10/10/10 | 1.329 [1.320–1.333] | 0.998 [0.993–1.002] | within delta |
| pythia410m_sparse20 | prefill_b4 | 10/10/10 | 1.410 [1.407–1.411] | 0.999 [0.997–0.999] | within delta |
| pythia410m_sparse80 | decode_b1 | 3/3/3 | 0.939 [0.939–0.939] | 0.992 [0.992–0.992] | within delta |
| pythia410m_sparse80 | decode_b4 | 3/3/3 | 1.315 [1.315–1.315] | 1.001 [1.001–1.001] | within delta |
| pythia410m_sparse80 | prefill_b1 | 3/3/3 | 2.866 [2.866–2.866] | 0.998 [0.998–0.998] | within delta |
| pythia410m_sparse80 | prefill_b4 | 3/3/3 | 3.426 [3.426–3.426] | 1.004 [1.004–1.004] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `device_memory_peak` | 3,457,009,664 | 2,120,927,232 | 2,120,927,232 | 0.614 | 1.000 | within 2 % |
| pythia410m_sparse20 | `device_memory_peak` | 3,457,009,664 | 2,375,911,424 | 2,375,911,424 | 0.687 | 1.000 | within 2 % |
| pythia410m_sparse80 | `device_memory_peak` | 3,457,009,664 | 3,885,860,864 | 3,885,860,864 | 1.124 | 1.000 | within 2 % |

## Limits

- B equals A within `delta` (0.055) and within 2 % of `device_memory_peak` on every cell.
- Coverage is by user decision: one export of the family (`O1`) is a failed export and is not covered by these figures.
- Form pair. The VM is shared. The v1 figures were taken on a Tesla T4 and are not comparable.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
