# HFFN CPU runtime on Apple M1: Pythia training with compact G4 FFN modules

Training of the Pythia models (70M, 160M, 410M) whose FFNs are pruned by WANDA with a fixed support, on the CPU of an Apple
M1. Each FFN is a compact module: its parameters (`up_val`, `down_val`, `b_up`, `b_down`), its gradients and its AdamW
states hold the active weights only; attention, embeddings, norms, the head and their optimiser are the dense PyTorch of the
model. There is no dense `W_up`, `W_down`, mask, gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the kernel of the sibling inference leaf** ([`../inference/compact`](../inference/compact/README.md), its dual entry, which also stores `z` and `h` for the backward). Backward and compact AdamW: the G4 training kernel of [`cpu/common/llm/training`](../../../../cpu/common/llm/training/README.md), hidden-owned below 32 tokens and dual route from there. Four fixed groups, plain C++ under `-O3 -ffast-math`, no manual micro-kernel. One block of tokens per call. |
| Hardware | Apple M1 (arm64), macOS, `clang++`. One machine, passively cooled. |
| Models | Pythia 70M, 160M and 410M (GPT-NeoX, float32), FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from the compact exports (18). |
| Results | [RESULTS.md](RESULTS.md); |

## Contents

```
check.sh                          layer, model and update gates on exports, without timing (below)
hffn_training_runtime/            the entry of the leaf on the shared level
  forward_entry.py                binds the shared runtime to the inference kernel of the sibling leaf
  build_entry.py                  build constants (compiler, flags) of the native extension
parity_hffn_layer_training.py     one layer: forward and gradients against the direct equation, on one export
```

The runtime, the model integration, the loader, the checkpoint (parameters and optimiser state) and the checks are at the
shared level `cpu/common/llm/training`, which the package of this leaf joins by its path. Use and copy the `runtime/` tree.

## Requirements and assets

- macOS arm64, `clang++`, PyTorch (CPU), transformers, safetensors. `HFFN_PYTHON` names the interpreter (default `python`).
- `HFFN_ASSETS` for the base checkpoints and exports (layout at the root of the tree). The prompt tokens of the model gate
  are generated once, as described there.
- Build: the extension is compiled into `build_compact/` of the leaf on first use (the explicit `clang++` command; with Ninja
  installed, `torch.utils.cpp_extension.load` is tried first with the same flags). The two recipes do not produce the
  same machine code (the 21 kernel function bodies are equal; some cold code and binding symbols differ). The published
  figures come from the command recipe: install nothing that provides Ninja when reproducing a figure.

## Use

```python
from hffn_training_runtime import load_pythia_hffn_for_training, HFFNTrainingOptimizer

model, runtime = load_pythia_hffn_for_training(base_checkpoint_dir, export_dir)
optimizer = HFFNTrainingOptimizer(model, runtime, lr=1e-4, weight_decay=0.01)
```

`load_pythia_hffn_for_training` builds the model from the base checkpoint (non-FFN tensors) and the compact export (FFN
values) and returns it with its runtime. `HFFNTrainingOptimizer` runs the native compact AdamW on the FFN values and a
PyTorch AdamW on every other parameter (`zero_grad`, `step`, `state_dict`, `load_state_dict` for resume).
`build_fixed_support_pythia_adamw` and `install_compact_hffn_values_in_dense_pythia` build the fixed-support dense control of
the checks (dense weights equal to the HFFN values on the support and zero elsewhere, and the update masked to the
support); they are controls, not a path of this runtime.

## Checks

```bash
./check.sh <base checkpoints dir> <exports dir> [export name ...]     # without names: every folder */export of <exports dir>
```

Without timing. For each export:

- **layer gate** (`checks/parity_layer.py`): for every layer, N in 1 to 512 and the synthetic test layers, the forward, `z`,
  `h`, `grad_x`, the active gradients and the bias gradients against the direct float64 equation, each within a relative
  error of 1e-4. A negative control (`--suppress-z`) must fail at N = 1.
- **model gate** (`checks/parity_model.py --gate`): the first step on a batch of real text. For the loss and each gradient
  group, the distance of the HFFN arm to float64 is at most ten times that of the dense float32 control, and the loss is
  within 1e-4 (relative) of float64. Two float32 arms are not compared step by step.
- **update gate** (`checks/parity_update.py --gate`): one AdamW step from zero moments against its closed form in float64,
  on the entries whose gradient is at least 1 % of the largest of their tensor, within 1e-2 of the learning rate; the moments
  within ten times the distance of the dense float32 control.

The script ends with `check: PASSED` or `check: FAILED`. `HFFN_CHECK_OUT=<dir>` keeps the JSON of each gate.

## Known limits

- **Dense control:** the fixed-support dense control of the figures is PyTorch on the same support, run by the same loop; it
  is a control, not a path.
- **One block per call**, and a training loop of the model that is the dense PyTorch one; blocks larger than the ones measured
  favour the dense control on this CPU (RESULTS.md).
- **Legacy checkpoints:** `padded_export` and `padded_import` (shared level) read a checkpoint of the former padded training
  format and convert it once to the compact format; they hold no forward, gradient or update and no training path reaches
  them. They are the only trace of that format.
- **Depends on the sibling inference leaf** for the forward: copy the `runtime/` tree, not the leaf alone.
- No claim of convergence equivalence: the gates measure the distance to float64 of one step, not a trajectory.
- One machine; no other arm64 hardware was tried.

## Provenance

See [RESULTS.md](RESULTS.md) for the run and the evidence class of each figure.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
