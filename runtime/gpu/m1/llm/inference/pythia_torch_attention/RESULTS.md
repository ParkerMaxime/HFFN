# Results — M1 GPU LLM inference (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../../RESULTS.md`](../../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Pythia-410M, FFNs on the Metal HFFN kernels with attention, norms and head on PyTorch/MPS, against the same model with dense iso-support FFNs; prefill and decode of the whole request. Machine: Apple M1 GPU, 16 GB. Two runs, in two parts that are not compared with each other.

## Measured

### Run `m1-gpu-llm-inference/publication_20261021T222112Z_45240`

`delta` = 0.066 (pythia410m_sparse20 / decode_b1 / hffn), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse20 | decode_b1 | 40.12 ms | 24.87 ms | 0.620 [0.561–0.592] | 10/10 | < 1 |
| pythia410m_sparse20 | decode_b4 | 108.31 ms | 57.47 ms | 0.531 [0.495–0.508] | 10/10 | < 1 |
| pythia410m_sparse20 | prefill_b1 | 61.69 ms | 49.92 ms | 0.809 [0.763–0.812] | 10/10 | < 1 |
| pythia410m_sparse20 | prefill_b4 | 111.61 ms | 112.97 ms | 1.012 [0.977–1.019] | 10/10 | > 1 (inside reported delta) |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| pythia410m_sparse20 | 2427 / 1723 | 4446 / 2550 | 1621 / 1058 | 0.653 | 0 / 0 |

### Run `m1-gpu-llm-inference/publication_20261022T002142Z_56021`

`delta` = 0.017 (pythia410m_sparse80 / prefill_b4 / dense), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | decode_b1 | 41.55 ms | 21.56 ms | 0.519 | 3/3 | < 1 |
| pythia410m_sparse10 | decode_b4 | 113.03 ms | 51.28 ms | 0.454 | 3/3 | < 1 |
| pythia410m_sparse10 | prefill_b1 | 66.04 ms | 42.10 ms | 0.637 | 3/3 | < 1 |
| pythia410m_sparse10 | prefill_b4 | 117.55 ms | 86.69 ms | 0.737 | 3/3 | < 1 |
| pythia410m_sparse80 | decode_b1 | 41.63 ms | 37.16 ms | 0.893 | 3/3 | < 1 |
| pythia410m_sparse80 | decode_b4 | 112.20 ms | 96.51 ms | 0.860 | 3/3 | < 1 |
| pythia410m_sparse80 | prefill_b1 | 66.67 ms | 104.98 ms | 1.575 | 3/3 | > 1 |
| pythia410m_sparse80 | prefill_b4 | 117.78 ms | 281.98 ms | 2.394 | 3/3 | > 1 |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| pythia410m_sparse10 | 2417 / 1243 | 4446 / 2466 | 1621 / 938 | 0.578 | 0 / 0 |
| pythia410m_sparse80 | 2428 / 3598 | 4446 / 4462 | 1621 / 1783 | 1.100 | 0 / 0 |

## Reading

- d20 (`m1-gpu-llm-inference/publication_20261021T222112Z_45240`, `delta` = 0.066, reported): decode 0.620 and 0.531, prefill 0.809 and 1.012 (prefill_b4 is inside `delta`).
- d10 and d80 (`m1-gpu-llm-inference/publication_20261022T002142Z_56021`, `delta` = 0.017, reported): d10 is below 1 on every shape (0.454 to 0.737); at d80 decode is below 1 (0.893, 0.860) and prefill is above 1 (1.575, 2.394).
- Memory: the HFFN arm retains 1058 MB at d20 against 1621 MB dense and samples 2550 against 4446 MB of device memory.

## Limits

- The pooled median of a two-session cell can lie outside the [min-max] of its sessions (decode_b1 0.620 against [0.561-0.592]); both are reported.
- The limits of [`../../../../../RESULTS.md`](../../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
