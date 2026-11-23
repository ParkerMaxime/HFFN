# Results — x86 LLM inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Whole requests of Pythia-410M at 10, 20 and 80 % FFN density: prefill and decode, batch 1 and 4. Machine: an OVH cloud VM, AMD EPYC-Milan, 8 vCPU (shared VM). Run `x86-llm-inference/publication_20261109T232405Z_89492`; `delta` = 0.070.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `x86-llm-inference/publication_20261109T232405Z_89492`

`delta` = 0.070, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | decode_b1 | 3/3/3 | 0.549 [0.549–0.549] | 1.036 [1.036–1.036] | within delta |
| pythia410m_sparse10 | decode_b4 | 3/3/3 | 0.648 [0.648–0.648] | 1.028 [1.028–1.028] | within delta |
| pythia410m_sparse10 | prefill_b1 | 3/3/3 | 0.697 [0.697–0.697] | 1.036 [1.036–1.036] | within delta |
| pythia410m_sparse10 | prefill_b4 | 3/3/3 | 0.714 [0.714–0.714] | 1.048 [1.048–1.048] | within delta |
| pythia410m_sparse20 | decode_b1 | 10/10/10 | 0.546 [0.538–0.539] | 1.036 [1.006–1.024] | within delta |
| pythia410m_sparse20 | decode_b4 | 10/10/10 | 0.658 [0.654–0.654] | 0.991 [0.990–0.994] | within delta |
| pythia410m_sparse20 | prefill_b1 | 10/10/10 | 0.741 [0.738–0.752] | 1.034 [1.034–1.042] | within delta |
| pythia410m_sparse20 | prefill_b4 | 10/10/10 | 0.854 [0.851–0.858] | 1.089 [1.081–1.102] | **loss** |
| pythia410m_sparse80 | decode_b1 | 3/3/3 | 0.656 [0.656–0.656] | 1.017 [1.017–1.017] | within delta |
| pythia410m_sparse80 | decode_b4 | 3/3/3 | 1.054 [1.054–1.054] | 1.002 [1.002–1.002] | within delta |
| pythia410m_sparse80 | prefill_b1 | 3/3/3 | 1.236 [1.236–1.236] | 1.114 [1.114–1.114] | **loss** |
| pythia410m_sparse80 | prefill_b4 | 3/3/3 | 1.711 [1.711–1.711] | 1.250 [1.250–1.250] | **loss** |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `retained_bytes` | 1,621,336,128 | 949,900,224 | 941,380,544 | 0.586 | 1.009 | within 2 % |
| pythia410m_sparse20 | `retained_bytes` | 1,621,336,128 | 1,070,696,352 | 1,062,176,672 | 0.660 | 1.008 | within 2 % |
| pythia410m_sparse80 | `retained_bytes` | 1,621,336,128 | 1,795,471,968 | 1,786,952,288 | 1.107 | 1.005 | within 2 % |

## Limits

- **B loses time against A beyond `delta` (0.070) on 3 of 12 cells, all prefill**: `sparse20 prefill_b4` 1.089, `sparse80 prefill_b1` 1.114, `sparse80 prefill_b4` 1.250. Decode is within `delta`. The leaf calls the same x86 FFN kernel as `x86-ffn-inference` (the loss there is the same family; attribution is a hypothesis, not isolated).
- Memory is within 2 % on every cell. Form C; the 410M margin of this form on the 30 GB VM is about 0.5 GiB (peak RSS 31,675,084,800 B in the sizing run), a written limit.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
