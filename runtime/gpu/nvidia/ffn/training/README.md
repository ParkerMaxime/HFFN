# HFFN GPU runtime on NVIDIA (CUDA): FFN training step

One training step of an isolated FFN on CUDA: the forward of the shared inference runtime with `z` stored, the backward,
and a compact AdamW, one call per block of `N` tokens. Gradients and AdamW moments hold the active weights only, each in the
length and order of its array. No dense `W_up`, `W_down`, gradient or moment exists.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the inference kernels of [`gpu/nvidia/common/kernels`](../../common/kernels/README.md)**, generated once at the shared level; training asks for the same up kernels with a `z` store. **Backward:** one kernel per quantity for every `N` (`grad_down`, `grad_hidden`, `grad_up`, a tile reduction when `N` > 256, `grad_x`), with fixed tiles of 256 tokens (128 for `grad_x`), not tuned per shape. A port, not a tuned set. |
| Hardware | An NVIDIA GPU with CUDA 12, one device, PyTorch's default stream; kernels compiled at run time by NVRTC through CuPy. Measured on one Tesla T4 only. |
| Models | An isolated FFN given by its compact arrays; homogeneous and irregular (exact offsets) supports, no padding. Checks on the test layer `homog_256_1024` and on real layers of Pythia exports. Timed cells: see RESULTS.md. |
| Results | [RESULTS.md](RESULTS.md). There is no Nvidia LLM training leaf. |

## Contents

```
kernels_train.py      CUDA source and launch glue of the backward and of the compact AdamW
train_runtime.py      HFFNTrainRuntime: wraps the shared HFFNRuntime, owns gradients, moments and workspaces
train_step.py         the dense control step, the float64 oracle and the gates (check support; not part of the runtime)
check.py              the gates on the synthetic test layer, N = 1 to 2100
check.sh              check.py with its record: runs/check_<UTC>/ (check.log, manifest.json, EXIT)
check_exports.py      the same gates on real FFN layers of the Pythia exports
check_exports.sh      check_exports.py with its record and the hashes of the executed sources
```

The runtime wraps the inference runtime of the sibling leaf ([`ffn/inference`](../inference/README.md)) and the shared kernels;
use and copy the `runtime/` tree.

`backward(index, gy)` uses the state of the last `forward(index, x, z)` that stored `z`; no other forward of the runtime may
come between them (checked). The work buffers are shared by the layers of the runtime, so a stack of layers would need
per-layer activations; that is outside this isolated-FFN step. After `reserve(n)` for the largest `N`, `step`, `forward`,
`backward` and `adamw` allocate no device memory; the default `max_tokens` is 2048.

## Run

Needs a CUDA machine and the environment of the inference leaf (`torch`, `cupy-cuda12x`, `numpy`).

```bash
./check.sh                              # synthetic layer, no asset
HFFN_ASSETS=<assets> ./check_exports.sh [--density 50 70] [--models pythia410m] [--layer -1] [--n 2100]
```

`check.sh` ends with `check: PASSED` or `check: FAILED`. For each `N` in 1, 33, 100, 128, 257, 300, 512, 1024 and 2100:

1. the forward without `z` is bitwise the inference runtime's, for the policy and the nine forced pairs;
2. the gradients and activations per group against the float64 oracle, within ten times the distance of the dense float32
   control;
3. one AdamW update against its float64 closed form, on the conditioned entries (relative 1e-5);
4. the same bits twice from the same state, supports and offsets unchanged;
5. five steps, each update against the float64 closed form from the build's own pre-update state and gradient, plus five
   decay-only steps (local form, tolerance 1e-5). The cumulative own-gradient replay is printed as a record, not a gate.

Then the guards of the partial workspace: no buffer growth after `reserve`, a refusal beyond 32-bit indexing.
`check_exports.sh` runs the same gates on the layers of the exports it is given (by default the first, a middle and the last
layer); gate 5 sets its exit code and the other gates are listed apart.

## Known limits

- **Without a GPU nothing runs.**
- **Isolated FFN only:** a model-level training step on CUDA is not part of this tree.
- **A port:** the thresholds and kernels come from the Apple M1 GPU runtime and were not tuned for CUDA.
- **One Tesla T4** on a shared cloud VM; `delta` is reported, not applied.
- The gates measure the distance to float64 of a few steps, not a trajectory: no claim of convergence equivalence.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
