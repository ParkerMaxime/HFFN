# Results — x86 LLM training (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M with the FFNs as HFFN modules (compact parameters, gradients and AdamW state) against the dense iso-support model, batch 1 with 32 and 128 tokens. Machine: AMD EPYC 9R14, 8 vCPU. Run `x86-llm-training/publication_20261021T234202Z_54659`; `delta` = 0.088, applied.

## Measured

### Run `x86-llm-training/publication_20261021T234202Z_54659`

`delta` = 0.088 (pythia410m_sparse20 / b1_t32 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | b1_t32 | 1.25 s | 668.35 ms | 0.533 | 3/3 | gain |
| pythia410m_sparse10 | b1_t128 | 1.98 s | 1.41 s | 0.713 | 3/3 | gain |
| pythia410m_sparse20 | b1_t32 | 1.32 s | 948.29 ms | 0.717 [0.690–0.744] | 10/10 | gain |
| pythia410m_sparse20 | b1_t128 | 2.07 s | 2.09 s | 1.010 [0.987–1.019] | 10/10 | within delta |
| pythia410m_sparse80 | b1_t32 | 1.10 s | 1.72 s | 1.569 | 3/3 | loss |
| pythia410m_sparse80 | b1_t128 | 1.84 s | 5.34 s | 2.899 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse10 | 8145 / 7923 | 7291 / 3627 | 0.498 | 0 / 0 |
| pythia410m_sparse20 | 7800 / 6482 | 7291 / 3990 | 0.547 | 0 / 0 |
| pythia410m_sparse80 | 12111 / 12540 | 7291 / 6164 | 0.845 | 0 / 0 |

## Reading

- d20: 0.717 at 32 tokens (a gain) and 1.010 at 128 tokens (within `delta`). d10: 0.533 and 0.713, both gains. d80: 1.569 and 2.899, both losses.
- Memory (adapter count): HFFN retains 3627, 3990 and 6164 MB at d10, d20 and d80 against 7291 MB dense.

## Limits

- Two token counts and three densities were timed; this is not a convergence or trajectory claim.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
