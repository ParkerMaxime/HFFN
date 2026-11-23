# Results — x86 LLM training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M (batch 1, 32 or 128 tokens) at 10, 20 and 80 % FFN density. Machine: an OVH cloud VM, AMD EPYC-Milan, 8 vCPU (shared VM). Run `x86-llm-training/publication_20261110T003615Z_94748`; `delta` = 0.035.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `x86-llm-training/publication_20261110T003615Z_94748`

`delta` = 0.035, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | b1_t128 | 3/3/3 | 0.788 [0.788–0.788] | 0.932 [0.932–0.932] | gain |
| pythia410m_sparse10 | b1_t32 | 3/3/3 | 0.724 [0.724–0.724] | 1.023 [1.023–1.023] | within delta |
| pythia410m_sparse20 | b1_t128 | 10/10/10 | 1.045 [1.044–1.046] | 0.981 [0.969–0.984] | within delta |
| pythia410m_sparse20 | b1_t32 | 10/10/10 | 0.799 [0.798–0.805] | 0.973 [0.971–0.976] | within delta |
| pythia410m_sparse80 | b1_t128 | 3/3/3 | 2.587 [2.587–2.587] | 0.897 [0.897–0.897] | gain |
| pythia410m_sparse80 | b1_t32 | 3/3/3 | 1.796 [1.796–1.796] | 0.988 [0.988–0.988] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `retained_bytes` | 7,290,651,856 | 4,178,441,008 | 4,421,507,248 | 0.573 | 0.945 | gain |
| pythia410m_sparse20 | `retained_bytes` | 7,290,651,856 | 4,540,829,392 | 5,025,487,888 | 0.623 | 0.904 | gain |
| pythia410m_sparse80 | `retained_bytes` | 7,290,651,856 | 6,715,156,240 | 8,649,365,968 | 0.921 | 0.776 | gain |

## Limits

- B gains time over A beyond `delta` (0.035) on 2 cells and is within it on 4; memory gains beyond 2 % on all 3 cells.
- Form C with a dense copy at every cell; the 410M margin of this form on the 30 GB VM is about 0.5 GiB (peak RSS 31,675,084,800 B in the sizing run), a written limit.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
