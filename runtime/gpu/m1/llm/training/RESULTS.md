# Results — M1 GPU LLM training (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M with the FFNs as compact HFFN modules on Metal (attention and the rest on PyTorch/MPS) against the dense iso-support model, batch 1 with 32 and 128 tokens. Machine: Apple M1 GPU, 16 GB. Two runs, in two parts that are not compared with each other.

## Measured

### Run `m1-gpu-llm-training/publication_20261021T215425Z_43892`

`delta` = 0.166 (pythia410m_sparse20 / b1_t32 / dense), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse20 | b1_t32 | 942.77 ms | 673.62 ms | 0.715 [0.709–0.712] | 10/10 | < 1 |
| pythia410m_sparse20 | b1_t128 | 1.07 s | 1.13 s | 1.056 [1.055–1.064] | 10/10 | > 1 (inside reported delta) |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| pythia410m_sparse20 | 2168 / 3567 | 9479 / 8296 | 7291 / 4474 | 0.614 | 0 / 0 |

### Run `m1-gpu-llm-training/publication_20261022T123442Z_81919`

`delta` = 0.004 (pythia410m_sparse10 / b1_t128 / hffn), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | b1_t32 | 830.08 ms | 551.53 ms | 0.664 | 3/3 | < 1 |
| pythia410m_sparse10 | b1_t128 | 959.10 ms | 877.04 ms | 0.914 | 3/3 | < 1 |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| pythia410m_sparse10 | 2274 / 3292 | 9479 / 6820 | 7291 / 3870 | 0.531 | 0 / 0 |

## Reading

- d20 (`m1-gpu-llm-training/publication_20261021T215425Z_43892`, `delta` = 0.166, reported): 0.715 at 32 tokens, 1.056 at 128 tokens (inside `delta`).
- d10 (`m1-gpu-llm-training/publication_20261022T123442Z_81919`, `delta` = 0.004, reported): 0.664 and 0.914.
- **d80: `OOM` on the HFFN arm.** The cell does not fit the 16 GB Mac (it is swap-bound); no figure exists for either arm. A memory fix is deferred to the next optimisation phase.
- Memory (adapter count): the HFFN arm retains 3870 and 4474 MB at d10 and d20 against 7291 MB dense, and samples less device memory at d20 (8296 against 9479 MB).

## Limits

- Part of the d10 and d20 cells ran in different machine states and are not compared with each other.
- The d80 `OOM` is a limit of this version, not a measurement of the format.
- The figures were measured with a tree that still held, per layer, four device index tensors of unused fused layouts (removed from the published tree; the kernels the training step runs are unchanged). The retained and sampled memory figures are therefore an upper bound for this tree (hypothesis, not re-measured).
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
