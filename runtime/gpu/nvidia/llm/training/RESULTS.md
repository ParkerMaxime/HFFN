# Results — Nvidia LLM training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-70M and Pythia-410M (batch 1, 32 or 128 tokens) at several FFN densities. First publication of this leaf: B and dense only. Machine: an OVH cloud VM with one NVIDIA Quadro RTX 5000 (16 GB, compute capability 7.5; shared VM, 4 vCPU). Run `nvidia-llm-training/publication_20261110T035523Z_9800`; `delta` = 0.217.
Arms: B (this version) and the dense control only; the leaf is published for the first time. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `nvidia-llm-training/publication_20261110T035523Z_9800`

`delta` = 0.217, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | 
|---|---|---|---|
| pythia410m_sparse10 | b1_t128 | 3/3/- | 0.839 [0.839–0.839] |
| pythia410m_sparse10 | b1_t32 | 3/3/- | 0.824 [0.824–0.824] |
| pythia410m_sparse20 | b1_t128 | 10/10/- | 0.908 [0.844–0.909] |
| pythia410m_sparse20 | b1_t32 | 10/10/- | 0.797 [0.824–0.852] |
| pythia410m_sparse80 | b1_t128 | 3/3/- | 1.691 [1.691–1.691] |
| pythia410m_sparse80 | b1_t32 | 3/3/- | 1.422 [1.422–1.422] |
| pythia70m_sparse10 | b1_t128 | 3/3/- | 1.016 [1.016–1.016] |
| pythia70m_sparse10 | b1_t32 | 3/3/- | 1.001 [1.001–1.001] |
| pythia70m_sparse50 | b1_t128 | 3/3/- | 0.988 [0.988–0.988] |
| pythia70m_sparse50 | b1_t32 | 3/3/- | 0.975 [0.975–0.975] |
| pythia70m_sparse80 | b1_t128 | 3/3/- | 1.004 [1.004–1.004] |
| pythia70m_sparse80 | b1_t32 | 3/3/- | 1.031 [1.031–1.031] |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `retained_bytes` | 6,686,672,080 | 3,748,286,416 | — | 0.561 | — | — |
| pythia410m_sparse20 | `retained_bytes` | 6,686,672,080 | 4,231,470,928 | — | 0.633 | — | — |
| pythia410m_sparse80 | `device_memory_peak` | 7,308,366,848 | 7,370,945,536 | — | 1.009 | — | — |
| pythia70m_sparse10 | `retained_bytes` | 1,139,409,264 | 955,821,744 | — | 0.839 | — | — |
| pythia70m_sparse50 | `retained_bytes` | 1,139,409,264 | 1,076,617,584 | — | 0.945 | — | — |
| pythia70m_sparse80 | `retained_bytes` | 1,139,409,264 | 1,167,214,608 | — | 1.024 | — | — |

## Limits

- `delta` is 0.217 (form C, in-process dense copy). The 70M cells, close to 1 against dense, sit inside that band.
- At `pythia410m_sparse80` the primary memory reading is `device_memory_peak` and B is 1.009 of dense; at 10 and 20 % it is `retained_bytes` (0.561, 0.633). The two readings are not compared.
- Forms: C on the Pythia-70M cells and on 410M at 10 and 20 %, P on 410M at 80 %. The forward is the shared inference launch of the kernels of the sibling inference leaf, the backward and the update are those of `gpu/nvidia/ffn/training`. The VM is shared.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
