# `runtime/gpu/nvidia/llm/training` — LLM training on the Nvidia GPU

A Pythia model with the fixed-support HFFN FFN in every layer, trained for a few updates at batch 1 x sequence 4 on a CUDA
GPU. Parameters, gradients and AdamW states of the FFN are the compact values of the support; attention, norms, embeddings, the
loss and the optimiser of the non-FFN weights are the dense PyTorch of the model (math attention). One call per FFN layer
processes the whole block of `N` tokens.

Results: [RESULTS.md](RESULTS.md). This is the first version of this leaf in `publication/` (v1 had no Nvidia LLM training
leaf); its timing has a dense control and no earlier version to compare with.

## Scope

| | |
|---|---|
| Degree of optimisation | **no kernel of its own.** The forward of every FFN layer is the shared inference launch `HFFNRuntime._launch` of [`../../common/kernels`](../../common/kernels/README.md) (the kernel of `../inference`); the backward, the transposed supports, the gradient partials and the compact AdamW are those of [`../../ffn/training`](../../ffn/training/README.md) (`HFFNTrainRuntime`). This leaf holds the per-layer wiring. The kernels are a port of the Metal ones, not tuned for CUDA; optimisation proposals estimated above 2 % are not done and are not claimed. |
| Hardware | an NVIDIA GPU with CUDA 12, one device, PyTorch's default stream; kernels compiled at run time by NVRTC through CuPy. Measured on one Quadro RTX 5000 (16 GB, compute capability 7.5) on a shared OVH cloud VM (4 vCPU), PyTorch 2.8.0 built for CUDA 12.9. No other GPU was tried. |
| Models | Pythia 70M, 160M and 410M (GPT-NeoX, float32) pruned by WANDA, read from compact exports; the checks of this leaf judge Pythia-70M d10 and Pythia-410M d80 and a set of synthetic contexts. B1 x S4 only. |
| Results | [RESULTS.md](RESULTS.md); each figure carries its evidence class and the name of its qualified run. |

## Contents

| File | Role |
|---|---|
| `hffn_pythia_cuda_training/context_runtime.py` | `ContextTrainRuntime`: per-layer `x_t`, `z`, `h`, `y`, `gx` buffers bound onto the shared runtime (`reserve(n)`, `forward_with_context`, `backward_with_context`, `adamw_all`) |
| `hffn_pythia_cuda_training/model.py` | `HFFNTrainingMLP` (one autograd function per layer), `HFFNTrainingOptimizer` (compact AdamW on the active values plus PyTorch AdamW on the dense weights), `load_pythia` |
| `checks/context_check.py` | the contexts checker (synthetic layers, N from 1 to 2100), smoke grid and run |
| `checks/model_check.py`, `checks/update_gate.py` | the model checker of one export (float64 reference, per-update closed form, decay-only pass, faults) |
| `checks/qualification.py` | the driver: contexts, 70M d10 and 410M d80 as child processes |
| `checks/test_update_gate.py` | CPU unit test of the update gate and of the bound-tree comparison |
| `launch.sh`, `check.sh` | launcher and check record |

The leaf reaches `ffn/training`, `ffn/inference`, `common/kernels`, `common/ffn` and `common/pythia/hffn_pythia` through the
`runtime/` tree it sits in (use and copy the tree, not the leaf alone). Assets are read in place from `HFFN_ASSETS`
(no default); nothing is copied.

## Requirements

Linux, a CUDA 12 GPU, Python 3.10 or later, PyTorch with CUDA, CuPy for CUDA 12, numpy, transformers, safetensors.
`HFFN_CUDA_PYTHON` names the interpreter of the CUDA environment (default `python`). `HFFN_BASE_COMMIT` (or `BASE_COMMIT`) is any
identifier of the source base: it is only recorded in `provenance.json`, with the sha256 of every source file read.

## Launcher

```text
bash launch.sh check                 no GPU, untimed: syntax, imports without CUDA, bound files, planned ids
bash launch.sh unit                  no GPU, untimed: the unit test of the update gate
bash launch.sh smoke [--export <e>]  CUDA: contexts smoke grid, or the model checker (mode smoke) on one export
bash launch.sh run [--attempt N]     CUDA: the qualification (contexts, 70M d10, 410M d80)
./check.sh                           CUDA: unit test, then the qualification, as one record in runs/check_<UTC>_<pid>/
```

A run folder holds `config.json`, `provenance.json`, `raw.jsonl`, `summary.json`, `manifest.json`, `EXIT` (written by the
checker) and `launcher_manifest.json`, `LAUNCH_EXIT` (written by the launcher). A run fails `source_identity` when a bound
file changes between its start and its end. `check.sh` needs `pythia70m_sparse10_ffn_only` and `pythia410m_sparse80_ffn_only`
under `$HFFN_ASSETS/pythia/exports`; `launch.sh smoke --export pythia70m_sparse10_ffn_only` is the bounded smoke (about a
minute).

## Checks

Parity and functional memory are judged against a float64 reference. Nothing here is timed. Per export, at B1 x S4, five
updates, math attention:

- loss within 1e-4 (relative) of float64; each gradient group within ten times the distance of the dense float32 control to
  float64;
- **update gate** (`update_gate.py`): every update of the horizon against the float64 closed form started from that build's
  own pre-update float32 state, scale `max(|w_ref|, |w_prev|, lr)`, bound 1e-5; the cumulative form is recorded, not gating;
  the decay-only pass; on 70M d10, eight reference-side faults and four build-side faults (a hyper-parameter of the HFFN
  optimiser changed) must be detected on both judged groups; at 410M no fault arm runs;
- gradient identity, layer coverage, canonical and transposed supports unchanged, inactive slots never written, finite
  parameters and states, the dense32 initial identity (loss bits and gradient fingerprints), update count;
- memory: peak allocated of every arm within 14 GiB, no growth after a warm step.

## Known limits

- Only B1 x S4 and five updates are checked; N above 4, checkpoint and resume, any convergence or long-horizon claim are not
  covered.
- The dense control is the model's own PyTorch MLP with a fixed-support AdamW; it is not shared with `ffn/training`.
- The cumulative own-gradient replay of the update gate exceeds its former bound on some entries; it is recorded, not gating.
- One GPU, one shared cloud VM: compare only within one execution, never with the M1 or x86 figures.
- At high density the HFFN arm is slower than dense: the expected boundary of the format.

## Licence

MIT: the file `LICENSE` at the root of the `runtime/` tree covers this leaf.
