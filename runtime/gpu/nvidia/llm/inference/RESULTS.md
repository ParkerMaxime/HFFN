# Results — Nvidia LLM inference (Pythia)

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

Pythia-410M, FFNs executed by the CUDA HFFN runtime inside PyTorch against the same model with dense iso-support FFNs; prefill and decode of the whole request. Machine: Tesla T4 (AWS `g4dn.xlarge`). Run `nvidia-llm-inference/publication_20261021T223804Z_47324`; `delta` = 0.023, reported (not applied).

## Measured

### Run `nvidia-llm-inference/publication_20261021T223804Z_47324`

`delta` = 0.023 (pythia410m_sparse10 / prefill_b1 / dense), reported (GPU: not applied).

| Cell | Size | dense | hffn | hffn/dense [min–max] | processes d/h | reading |
|---|---|---:|---:|---:|---:|---|
| pythia410m_sparse10 | decode_b1 | 14.39 ms | 13.72 ms | 0.953 | 3/3 | < 1 |
| pythia410m_sparse10 | decode_b4 | 14.91 ms | 13.61 ms | 0.913 | 3/3 | < 1 |
| pythia410m_sparse10 | prefill_b1 | 27.38 ms | 26.02 ms | 0.950 | 3/3 | < 1 |
| pythia410m_sparse10 | prefill_b4 | 99.20 ms | 94.84 ms | 0.956 | 3/3 | < 1 |
| pythia410m_sparse20 | decode_b1 | 14.48 ms | 13.56 ms | 0.937 [0.935–0.938] | 10/10 | < 1 |
| pythia410m_sparse20 | decode_b4 | 14.92 ms | 13.67 ms | 0.916 [0.916–0.916] | 10/10 | < 1 |
| pythia410m_sparse20 | prefill_b1 | 28.14 ms | 36.63 ms | 1.302 [1.297–1.309] | 10/10 | > 1 |
| pythia410m_sparse20 | prefill_b4 | 101.18 ms | 138.26 ms | 1.366 [1.363–1.377] | 10/10 | > 1 |
| pythia410m_sparse80 | decode_b1 | 14.61 ms | 13.45 ms | 0.921 | 3/3 | < 1 |
| pythia410m_sparse80 | decode_b4 | 15.03 ms | 22.76 ms | 1.514 | 3/3 | > 1 |
| pythia410m_sparse80 | prefill_b1 | 28.80 ms | 86.92 ms | 3.018 | 3/3 | > 1 |
| pythia410m_sparse80 | prefill_b4 | 103.10 ms | 336.11 ms | 3.260 | 3/3 | > 1 |

| Cell | RSS dense / hffn (MB) | device dense / hffn (MB) | retained dense / hffn (MB) | retained hffn/dense | swap-outs |
|---|---:|---:|---:|---:|---:|
| pythia410m_sparse10 | 2917 / 2164 | 3303 / 1967 | 1621 / 950 | 0.586 | 0 / 0 |
| pythia410m_sparse20 | 3112 / 3608 | 3303 / 2222 | 1621 / 1071 | 0.660 | 0 / 0 |
| pythia410m_sparse80 | 4314 / 6252 | 3303 / 3732 | 1621 / 1795 | 1.107 | 0 / 0 |

## Reading

- d10 is below 1 on every shape (0.913 to 0.956). d20: decode below 1 (0.937, 0.916), prefill above 1 (1.302, 1.366).
- d80: decode_b1 0.921, decode_b4 1.514, prefill 3.018 and 3.260.
- Memory: the HFFN arm retains 950 and 1071 MB at d10 and d20 against 1621 MB dense, and 1795 MB at d80; the sampled device memory is lower for HFFN at d10 and d20 (1967 and 2222 against 3303 MB).

## Limits

- The kernels and their selection thresholds are the ones first measured on the Apple M1 GPU: they were not tuned for CUDA, so these figures are the boundary of this port, not of the format.
- The limits of [`../../../../RESULTS.md`](../../../../RESULTS.md) apply, in particular the noise band `delta`, the retained-bytes definition and the scope of the evidence.
