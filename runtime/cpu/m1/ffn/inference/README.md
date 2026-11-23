# `runtime/cpu/m1/ffn/inference` — compact G4 inference runtime, Apple M1 CPU

The owner of the forward of the M1 CPU: the **compact G4** kernel, its PyTorch binding and the runtime that holds its
workspaces. Hidden neurons are split into four fixed groups (the calling thread owns group 0, three workers own the others),
each group has its own private reduction buffer, the reduction order is fixed, and the supports are exact (offsets, no
padding). One call processes the whole block of `N` tokens. This leaf is the isolated-FFN leaf of the M1 CPU: it holds the
kernel and the dense control the FFN is measured against; the Pythia loader, the model and the parity checks that exercise it
are in its consumer [`../../llm/inference/compact`](../../llm/inference/compact/README.md). The M1 CPU training leaves take
their forward from this leaf, through its `dual_entry.py`; none carries a second forward.

Results: [RESULTS.md](RESULTS.md) (one isolated FFN per cell, N = 1, 8, 64 and 512 tokens).

## Scope

| | |
|---|---|
| Degree of optimisation | one kernel and one deterministic route (`hffn_inference_runtime/forward_route.py`): decode at N = 1 when only `y` is asked; a fused entry for `y` at N >= 2; a stores entry for `{y, z, h}` at exactly N = 8 (`STORES_N`) and a dual entry, which reads the same edges in output order and needs no reduction, at the other N. Plain C++ left to the compiler (`-O3 -ffast-math`), no manual micro-kernel, no matrix unit (the dense control uses Apple's Accelerate matrix unit; the compact path cannot). Not tuned for speed; optimisation proposals estimated above 2 % are not done and are not claimed. |
| Hardware | Apple M1 (arm64), macOS, `clang++`. Measured on one Mac with 16 GB, passively cooled. Not tried on other Apple chips or on other arm64 machines. |
| Models | an FFN given in the compact hidden-major layout (`hffn_training_compact_g4_v2`, read by `common/pythia/hffn_pythia/compact_format.py`), GELU (exact), float32; irregular supports with exact offsets. The Pythia models (70M, 160M, 410M) are covered through the consumer leaf. |
| Results | [RESULTS.md](RESULTS.md); each statement carries its evidence class and the name of its qualified run. |

## Contents

```
dense_control.py                        the dense iso-support control: the dense student FFN and the masked-dense payload with its
                                        forward (a control for timing, never an HFFN path)
hffn_inference_runtime/
  forward_route.py                      the route of a call, a function of N and of the outputs asked
  compact_runtime.py                    HFFNCompactG4InferenceRuntime: workspaces per (layer, N), routes, calls
  compact_model.py                      the Pythia MLP replacement (no autograd state)
  compact_inference_loader.py           direct Pythia loader (non-FFN weights from model.safetensors, FFN from the export)
  compact_torch_extension.py            build and load of the native extension (into build/)
  build_entry.py                        compiler, flags and link arguments of the build
  dual_entry.py                         the dual and stores entries that also store z and h, bound once for the M1 training leaves
native/hffn_compact_g4_inference.cpp            the kernel
torch_extension/hffn_compact_inference_torch_binding.cpp   its PyTorch binding
```

The build recipe is the shared `cpu/common/recipe/compact_extension.py`. The leaf finds the shared levels as the first
ancestor folder that holds `runtime/common/pythia/hffn_pythia`: use it, and copy it, as the `runtime/` tree.

## Requirements and build

macOS arm64, `clang++` (Command Line Tools), PyTorch (CPU). `HFFN_PYTHON` names the interpreter (default `python`). The extension
is built into `build/` at the first call with `clang++ -O3 -ffast-math -std=c++17 -fPIC -dynamiclib -pthread`; with Ninja
installed, `torch.utils.cpp_extension.load` is tried first with the same flags and does not produce the same machine code (the
bodies of the kernel functions are equal, some cold code and binding symbols differ). The measured figures come from the command
recipe: install nothing that provides Ninja when reproducing a figure.

## Verification

This leaf has no `check.sh` of its own: the parity checks (per layer, bit determinism of the fixed groups; end to end, token
identity against the dense float32 arm or the float64 oracle, with a negative control) run through the consumer leaf,
`../../llm/inference/compact/check.sh`, and through the M1 CPU training leaves, which call the same runtime. Nothing is timed.

## HFFN execution

Compact: values, indices, offsets (and the reverse schedule of the dual route). Dense: `x`, `z`, `h`, `y` and their buffers, one set
per (layer, N), allocated once. `W_up`, `W_down` and the masks are never materialised outside the dense control and the oracle
of the checks. The runtime's `execution` is `baseline` (default), `dual_route` or `adaptive` (dual from `dual_route_min_n` = 64
tokens): the last two are labelled controls, not the production configuration.

## Known limits

- HFFN is slower than dense where the FFN is dense enough (high density, large N): the expected boundary of the format,
  reported in [RESULTS.md](RESULTS.md).
- N = 2 to 7 is not a stores range for `{y, z, h}` (no timed cell covers it): written limit.
- The reduction order of the fixed groups differs from the dense float32 matmul: outputs agree to float32 rounding, not bit for
  bit with the control.
- One machine, one boot; no other arm64 hardware was tried.

## Licence

MIT: the file `LICENSE` at the root of the `runtime/` tree covers this leaf.
