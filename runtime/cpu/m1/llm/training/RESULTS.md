# Results — M1 CPU LLM training (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M with the FFNs as compact-G4 HFFN modules against the dense iso-support model on the Mac, batch 1 with 32 and 128 tokens. Machine: Apple M1, 16 GB. Two runs, in two parts that are not compared with each other.

## Measured

### Run `m1-cpu-llm-training/publication_20261022T021302Z_64319`

`delta` = 0.064 (pythia410m_sparse20 / b1_t128 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse20 | b1_t32 | 1.14 s | 952.49 ms | 0.837 [0.823–0.836] | 10/10 | gain |
| pythia410m_sparse20 | b1_t128 | 1.42 s | 2.30 s | 1.616 [1.614–1.632] | 10/10 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse20 | 7770 / 5904 | 7291 / 3990 | 0.547 | 0 / 0 |

### Run `m1-cpu-llm-training/publication_20261022T114358Z_81377`

`delta` = 0.044 (pythia410m_sparse10 / b1_t128 / hffn), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | b1_t32 | 1.13 s | 686.32 ms | 0.605 | 3/3 | gain |
| pythia410m_sparse10 | b1_t128 | 1.42 s | 1.34 s | 0.944 | 3/3 | gain |
| pythia410m_sparse80 | b1_t32 | 1.15 s | 2.61 s | 2.277 | 3/3 | loss |
| pythia410m_sparse80 | b1_t128 | 1.45 s | 8.43 s | 5.800 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse10 | 7471 / 4970 | 7291 / 3627 | 0.498 | 0 / 0 |
| pythia410m_sparse80 | 7333 / 9081 | 7291 / 6164 | 0.845 | 0 / 0 |

## Reading

- d20 (`m1-cpu-llm-training/publication_20261022T021302Z_64319`, `delta` = 0.064): 0.837 at 32 tokens (a gain) and 1.616 at 128 tokens (a loss).
- d10 and d80 (`m1-cpu-llm-training/publication_20261022T114358Z_81377`, `delta` = 0.044): d10 is a gain at both token counts (0.605, 0.944); d80 is a loss (2.277, 5.800).
- Memory (adapter count): the HFFN arm retains 3627, 3990 and 6164 MB at d10, d20 and d80 against 7291 MB dense; its RSS at d80 is higher than dense (9081 against 7333 MB).

## Limits

- Two token counts were timed; this is not a convergence or trajectory claim.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
