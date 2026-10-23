# Results — x86 LLM inference (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Pythia-410M, FFNs executed by HFFN inside PyTorch on 8 threads against the same model with dense iso-support FFNs; prefill and decode of the whole request. Machine: AMD EPYC 9R14, 8 vCPU. Run `x86-llm-inference/publication_20261021T224108Z_47493`; `delta` = 0.118, applied.

## Measured

### Run `x86-llm-inference/publication_20261021T224108Z_47493`

`delta` = 0.118 (pythia410m_sparse80 / decode_b1 / dense), applied.

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | decode_b1 | 91.13 ms | 45.42 ms | 0.498 | 3/3 | gain |
| pythia410m_sparse10 | decode_b4 | 69.36 ms | 39.32 ms | 0.567 | 3/3 | gain |
| pythia410m_sparse10 | prefill_b1 | 204.73 ms | 124.89 ms | 0.610 | 3/3 | gain |
| pythia410m_sparse10 | prefill_b4 | 589.02 ms | 363.09 ms | 0.616 | 3/3 | gain |
| pythia410m_sparse20 | decode_b1 | 80.52 ms | 47.33 ms | 0.588 [0.581–0.596] | 10/10 | gain |
| pythia410m_sparse20 | decode_b4 | 70.50 ms | 42.74 ms | 0.606 [0.603–0.620] | 10/10 | gain |
| pythia410m_sparse20 | prefill_b1 | 205.27 ms | 142.05 ms | 0.692 [0.693–0.698] | 10/10 | gain |
| pythia410m_sparse20 | prefill_b4 | 583.38 ms | 422.56 ms | 0.724 [0.724–0.725] | 10/10 | gain |
| pythia410m_sparse80 | decode_b1 | 82.54 ms | 66.55 ms | 0.806 | 3/3 | gain |
| pythia410m_sparse80 | decode_b4 | 71.07 ms | 65.56 ms | 0.922 | 3/3 | within delta |
| pythia410m_sparse80 | prefill_b1 | 211.53 ms | 238.56 ms | 1.128 | 3/3 | loss |
| pythia410m_sparse80 | prefill_b4 | 613.58 ms | 863.46 ms | 1.407 | 3/3 | loss |

| Cell | RSS dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|
| pythia410m_sparse10 | 4382 / 2984 | 1621 / 941 | 0.581 | 0 / 0 |
| pythia410m_sparse20 | 4321 / 3156 | 1621 / 1062 | 0.655 | 0 / 0 |
| pythia410m_sparse80 | 7851 / 7924 | 1621 / 1787 | 1.102 | 0 / 0 |

## Reading

- Reference cell d20: decode 0.588 (b1) and 0.606 (b4), prefill 0.692 (b1) and 0.724 (b4): a gain on every shape beyond `delta`.
- d10 is a gain on every shape (0.498 to 0.616). At d80 prefill is a loss (1.128, 1.407); decode_b1 is a gain (0.806) and decode_b4 is within `delta` (0.922).
- Memory (adapter count): HFFN retains 941 and 1062 MB at d10 and d20 against 1621 MB dense, and 1787 MB at d80.

## Limits

- Only the 410M model was timed; the 70M and 160M models are covered by the checks, not by this table.
- `delta` = 0.118 is large for a CPU run: the cells close to 1 read *within delta*.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
