# `runtime/gpu/m1/ffn/training` — isolated-FFN training step on the Apple M1 GPU (Metal)

One training step of a fixed-support FFN on the M1 GPU: the forward of the shared inference runtime with `z` stored, the
compact backward, and the compact AdamW, in one persistent native context. One `step()` processes all `N` tokens. Gradients
and AdamW moments hold the active weights only, each in the length and order of its array; no dense `W_up`, `W_down`,
gradient or moment is built.

Results: [RESULTS.md](RESULTS.md). This is the first version of this leaf in `publication/` (v1 had no M1 GPU FFN training
leaf); its timing has a dense control and no earlier version to compare with.

## Scope

| | |
|---|---|
| Degree of optimisation | **not an optimised kernel set.** The forward is the one of the inference runtime (`gpu/m1/llm/inference/native_transformer`, generated through `gpu/m1/common/hffn_m1`); the backward and the AdamW update are the compact Metal kernels of this leaf. The leaf stands as it is measured: optimisation proposals estimated above 2 % are not done, and no gain is claimed beyond the tables of `RESULTS.md`. |
| Hardware | Apple M1 (Metal, Apple Silicon unified memory), macOS with the Command Line Tools (`clang++`, Metal framework). Measured on one M1 machine. |
| Models | An isolated FFN given by an export (`hidden_ffn_format.pt`): homogeneous and irregular (exact offsets) supports, rectangular dimensions, independent up and down degrees, empty hidden rows; a wholly empty projection is refused. GELU. |
| Results | [RESULTS.md](RESULTS.md), with the evidence class and the run of each figure. |

## Native API

`native_context.py` is the thin C ABI owner: one synchronous persistent `Context` over `libhffn_train.dylib`, shared by the
Python check code and the command-line runner. `hffn_train_context.py` re-exports its supported names (`Context`,
`from_export`, `prepare_exact_sources`, `prepare_sources`, `read_export`, `checked`, `parts`) and adds no call.

```python
from hffn_train_context import from_export
context = from_export(library, export_directory, arguments)   # reads hidden_ffn_format.pt, exact up/down supports
context.write_input("x", x); context.write_input("gy", gy)
context.step()                                                # forward (z stored) + backward + AdamW, all N tokens
context.snapshot("up_val")                                    # compact views, returned as bytes
```

Views are borrowed native buffers that expire at `close()`; CPU views require explicitly selected shared storage. There is
no torch conversion or autograd integration here (the model-level binding is `gpu/m1/llm/training`). Exact imports admit the
flat AdamW, the default/persistent/owner schedules, the supported fused and unfused variants, forced codec widths, `K1`/`K2`
and shared/private input storage. Index widths follow the route codec (`hffn_route_codec.h`): compact16 for `H <= 2048` with
degree at most 32, compact32 otherwise.

## Contents

| Files | Role |
|---|---|
| `native_train_context.mm`, `native_train_context_api.h`, `.inc` | the Metal training context (forward, backward, AdamW, buffers, C entry `hffn_train_create_exact`) |
| `native_context.py`, `hffn_train_context.py`, `export_input.py`, `native_train_launcher.py`, `run_native_train_step.py` | Python owner of the C ABI, checked cold export input, command-line launcher |
| `native_exact_support.h`, `hffn_route_codec.h`, `hffn_gelu_exact.h`, `hffn_gelu_erf_accurate.h` | exact-support descriptor, route codec, GELU definitions (with `*_smoke.cpp`) |
| `build_train_step_runner.sh` | builds `build/libhffn_train.dylib` and the train-step runner (generates the shared headers first) |
| `build_training_forward_runner.sh`, `build_backward_runner.sh`, `build_adamw_runner.sh`, `build_validate_train_step_multistep.sh` | the standalone runners and the multi-step validator used by the check |
| `check.sh`, `check_leaf.py`, `export_gate.py` | the untimed every-case check and its in-process every-export gate |
| `validate_homogeneous_runtime.py`, `validate_train_step_multistep.*`, `validate_learning_trajectory_iso_support.py`, `validate_pytorch_mps_dense_iso_support_adamw.py`, `iso_support.py` | parity gates against a CPU reference and against the iso-support dense PyTorch control |

The leaf generates headers with `gpu/m1/common/hffn_m1` and reads the dense control (`FixedSupportAdamW`) from
`common/pythia/hffn_pythia/dense_control.py`: use and copy the `runtime/` tree.

## Labelled controls (not production)

`metal_hffn_*_runner.mm` are standalone runners of earlier milestones (ReLU forward prototype, backward, AdamW). They are
kept as labelled controls of the check and are not the production path, which is the native context above. Options of the
native context that are controls or tests, never defaults: `--tiled-n` (tiles of 32 tokens with deterministic reduction;
more dispatches, slower on the full step), `--accum-steps K` (sums the compact gradients over `K` distinct micro-batches before
one AdamW), `--shared-readonly` (immutable inputs in shared instead of private storage), `--owner-tile` (one SIMD group per
owner with tiles of 128 tokens). The owner-tile route is a known failing gate and is not covered by the check or by the results.

## Requirements

macOS on Apple Silicon, `clang++` with the Metal and Foundation frameworks, Python 3.10 or later with PyTorch (MPS), `numpy`.
`HFFN_PYTHON` names the interpreter (default `python`). `HFFN_ASSETS` (no default) names the folder that holds
`homogeneous/models/<export>/hidden_ffn_format.pt` for the in-process every-export gate; the other gates build their layers
from a deterministic generator. `PYTORCH_ENABLE_MPS_FALLBACK` must stay unset.

## Run and check

```bash
zsh build_train_step_runner.sh                  # from this folder
HFFN_ASSETS=/path/to/assets zsh check.sh check  # builds, runs every gate, writes runs/check_<UTC>_<pid>/ (run.log, EXIT, ...)
```

`check.sh` is untimed: no time and no memory figure is read. Its gates:

| Gate | What it requires |
|---|---|
| `smoke_codec`, `smoke_gelu_exact`, `smoke_gelu_erf` | exit 0: codec boundaries (compact16 `H=2048 D=32`; compact32 `H=2049 D=32` and `H=512 D=51`); GELU derivative within 2e-4; erf-accurate GELU within 5e-7 (value) and 3e-7 (derivative) of the exact one. The Metal GELU uses the Abramowitz-Stegun 7.1.26 erf, named "erf-accurate", not "exact" |
| `build` | the five build scripts exit 0 |
| `train_step` | persistent-N, compact16 and compact32, two distinct micro-batches, ten AdamW steps: max abs difference to the CPU reference at most 2e-3 |
| `forward_relu` | the standalone forward runner against the CPU reference: `z`, `h`, `y` within 1e-5 |
| `backward_runner`, `adamw_runner` | exit 0 and finite errors recorded (these prototypes declare no bound) |
| `dense_va`, `dense_tj` | iso-support dense PyTorch MPS control: finite errors, inactive slots exactly 0; 20-epoch trajectory functionally equivalent (objective and relative RMSE of the outputs), strict closeness of the states recorded |
| `export_inprocess` | `export_gate.py` on ten exports (`512_2048` at 5, 10, 20, 30, 50, 75, 90 and 100 % density, `256_1024_d010`, `1024_4096_d010`) at `N` = 1, 8, 64, 512 with the fault arm: 160 cells, each clean cell within the float64 band set by the dense float32 control (ten times its distance, floor, ceiling 1e-4), every injected fault (index, value, support write) detected |

Gradients reduced by SIMD do not have the order of the sequential CPU sum: the moments agree to FP32 rounding, and a bias with
a near-zero gradient can move its weight by about one AdamW step; the validators therefore also check that the GPU weights
satisfy the AdamW equation from their own moments (error below `8e-9`). Parity comes first: a timing of this leaf is read only
from a qualified run.

## Known limits

- The dense control and the float64 reference build dense matrices from the support; they are controls of the gates, never an
  HFFN path.
- Costs of the native context were checked on four uniform fixtures only; irregular exports are numerically checked.
- Sampled Metal allocation excludes host RSS and true transient peaks; the per-arm memory reading of the publication campaign
  for this leaf is not a primary one (see `RESULTS.md`).
- No convergence equivalence, global speed-up or generalisation to other hardware is claimed.
- At high density the HFFN arm is slower than dense: the expected boundary of the format.

## Licence

MIT: the file `LICENSE` at the root of the `runtime/` tree covers this leaf.
