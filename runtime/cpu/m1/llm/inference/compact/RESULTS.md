# Results — M1 CPU LLM inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../../RESULTS.md`](../../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Whole requests of Pythia-410M at 10, 20 and 80 % FFN density: prefill and decode, batch 1 and 4. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-cpu-llm-inference/publication_20261109T224516Z_86139`; `delta` = 0.110.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-cpu-llm-inference/publication_20261109T224516Z_86139`

`delta` = 0.110, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | decode_b1 | 3/3/3 | 0.612 [0.612–0.612] | 0.999 [0.999–0.999] | within delta |
| pythia410m_sparse10 | decode_b4 | 3/3/3 | 0.609 [0.609–0.609] | 1.000 [1.000–1.000] | within delta |
| pythia410m_sparse10 | prefill_b1 | 3/3/3 | 0.973 [0.973–0.973] | 0.994 [0.994–0.994] | within delta |
| pythia410m_sparse10 | prefill_b4 | 3/3/3 | 1.316 [1.316–1.316] | 1.003 [1.003–1.003] | within delta |
| pythia410m_sparse20 | decode_b1 | 10/10/10 | 0.706 [0.706–0.707] | 1.004 [0.999–1.005] | within delta |
| pythia410m_sparse20 | decode_b4 | 10/10/10 | 0.694 [0.690–0.693] | 1.001 [0.999–1.000] | within delta |
| pythia410m_sparse20 | prefill_b1 | 10/10/10 | 1.561 [1.553–1.591] | 1.007 [1.001–1.018] | within delta |
| pythia410m_sparse20 | prefill_b4 | 10/10/10 | 2.104 [2.098–2.108] | 1.002 [0.996–1.000] | within delta |
| pythia410m_sparse80 | decode_b1 | 3/3/3 | 1.188 [1.188–1.188] | 0.888 [0.888–0.888] | gain |
| pythia410m_sparse80 | decode_b4 | 3/3/3 | 1.243 [1.243–1.243] | 0.856 [0.856–0.856] | gain |
| pythia410m_sparse80 | prefill_b1 | 3/3/3 | 3.904 [3.904–3.904] | 0.860 [0.860–0.860] | gain |
| pythia410m_sparse80 | prefill_b4 | 3/3/3 | 5.967 [5.967–5.967] | 0.770 [0.770–0.770] | gain |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `retained_bytes` | 1,621,336,128 | 937,612,704 | 937,612,704 | 0.578 | 1.000 | within 2 % |
| pythia410m_sparse20 | `retained_bytes` | 1,621,336,128 | 1,058,408,832 | 1,058,408,832 | 0.653 | 1.000 | within 2 % |
| pythia410m_sparse80 | `footprint_timed_max_bytes` | 1,885,080,936 | 2,131,005,040 | 2,526,220,560 | 1.130 | 0.844 | gain |

## Limits

- B gains time over A beyond `delta` (0.110) on 4 cells (`sparse80`) and is within `delta` elsewhere; memory gains beyond 2 % on `sparse80` (B/A 0.844, `footprint_timed_max_bytes`).
- Forms: C at `sparse10` and `sparse20`, P at `sparse80` (one arm per process; the memory reading follows the form); the memory reading follows the form (`retained_bytes` or `footprint_timed_max_bytes`) and the two are not compared.
- Whole-model ratios include the dense non-FFN parts and never project an FFN-only figure.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
