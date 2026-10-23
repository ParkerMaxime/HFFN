# HFFN CPU runtime on x86_64: Pythia training with compact G4 FFN modules

Training of the Pythia models (70M, 160M, 410M) whose FFNs are pruned by WANDA with a fixed support, on a Linux x86_64 CPU.
Each FFN is a compact module: its parameters (`up_val`, `down_val`, `b_up`, `b_down`), gradients and AdamW states hold the
active weights only, by hidden neuron; attention, embeddings, norms, the head and their optimiser are the dense PyTorch of
the model. There is no dense `W_up`, `W_down`, mask, gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the shared x86 inference kernel** ([`cpu/x86/common`](../../../../cpu/x86/common/README.md)): its slots entry of the dual route stores `z` and `h` for the backward, with a dedicated entry for one token. Backward and compact AdamW: the G4 training kernel of [`cpu/common/llm/training`](../../../../cpu/common/llm/training/README.md) (one copy, the same as the Apple M1 leaf), hidden-owned below 32 tokens and dual route from there. Four fixed groups, one thread each, `-O3 -ffast-math`, no manual micro-kernel and no AVX-512, AMX, BF16 or INT8 path. One block of tokens per call. |
| Hardware | Linux x86_64 with AVX2 and FMA; `g++` 13 or later (C++20 headers of PyTorch). Measured on one AWS `c7a.2xlarge` (AMD EPYC 9R14, 8 vCPU). Other x86 CPUs were not tried. |
| Models | Pythia 70M, 160M and 410M (GPT-NeoX, float32), FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from the compact exports (18). |
| Results | [RESULTS.md](RESULTS.md). |

## Contents

```
check.sh                          verifies the source manifest, then runs the three gates, without timing
sources/manifest.json             the shared files this leaf is built from, with their SHA-256
hffn_training_runtime/            the entry of the leaf on the shared level
  forward_entry.py                binds the shared runtime to the shared x86 inference kernel
  build_entry.py                  build constants (g++, flags) of the native extension
```

The runtime, the model integration, the loader, the checkpoint (parameters and optimiser state) and the checks are at the
shared level `cpu/common/llm/training`, which the package of this leaf joins by its path. Use and copy the `runtime/` tree.

## Requirements and assets

- Linux x86_64, glibc, `g++` 13 or later, PyTorch (CPU), transformers, safetensors. `HFFN_PYTHON` names the interpreter
  (default `python`).
- `HFFN_ASSETS` for the base checkpoints and exports (layout at the root of the tree). The prompt tokens of the model gate
  are generated once, as described there.
- The extension is compiled into `build/` of the leaf on first use (`g++ -O3 -ffast-math -mno-recip -std=c++20 -fPIC
  -shared -pthread`). `-mno-recip` keeps a vectorised `sqrtf` from returning NaN on a denormal second moment when the
  process does not flush denormals, as a Python process does not.

## Use

```python
from hffn_training_runtime import load_pythia_hffn_for_training, HFFNTrainingOptimizer

model, runtime = load_pythia_hffn_for_training(base_checkpoint_dir, export_dir)
optimizer = HFFNTrainingOptimizer(model, runtime, lr=1e-4, weight_decay=0.01)
```

`load_pythia_hffn_for_training` builds the model from the base checkpoint (non-FFN tensors) and the compact export (FFN
values). `HFFNTrainingOptimizer` runs the native compact AdamW on the FFN values and a PyTorch AdamW on every other
parameter (`zero_grad`, `step`, `state_dict`, `load_state_dict` for resume). The fixed-support dense control of the checks is
built by `build_fixed_support_pythia_adamw`; it is a control, not a path of this runtime.

## Checks

```bash
./check.sh <base checkpoints dir> <exports dir> [export name ...]     # absolute paths; without names: every folder */export
```

Without timing. `check.sh` first verifies the SHA-256 of every file listed in `sources/manifest.json`, then for each export:

- **layer gate** (`checks/parity_layer.py`): for every layer and a set of N, the forward, `z`, `h`, `grad_x`, the active
  gradients and the bias gradients against the direct float64 equation, each within a relative error of 1e-4. A negative
  control (`--suppress-z`) must fail.
- **model gate** (`checks/parity_model.py --gate`): the first step on a batch of real text. For the loss and each gradient
  group, the distance of the HFFN arm to float64 is at most ten times that of the dense float32 control, and the loss is
  within 1e-4 (relative) of float64. It needs up to 12.2 GB of memory for Pythia-410M at 80 % density.
- **update gate** (`checks/parity_update.py --gate`): one AdamW step from zero moments against its closed form in float64,
  on the entries whose gradient is at least 1 % of the largest of their tensor, within 1e-2 of the learning rate; the moments
  within ten times the distance of the dense float32 control.

The script ends with `check: PASSED` or `check: FAILED`. `HFFN_CHECK_OUT=<dir>` keeps the JSON of each gate.

## Known limits

- **Dense control:** the fixed-support dense control of the figures is PyTorch on the same support, run by the same loop.
- **One block per call**; blocks larger than the ones measured favour the dense control on this CPU (RESULTS.md).
- **Depends on the shared levels** (`cpu/common`, `cpu/x86/common`): copy the `runtime/` tree, not the leaf alone.
- **One machine type:** the figures come from one cloud instance; x86 figures are compared only within one execution.
- No claim of convergence equivalence: the gates measure the distance to float64 of one step, not a trajectory.

## Provenance

See [RESULTS.md](RESULTS.md) for the run and the evidence class of each figure.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
