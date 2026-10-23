# HFFN CPU runtime on Apple M1: compact G4 inference

Inference runtime of the Pythia models (70M, 160M, 410M) pruned by WANDA, and of an isolated FFN, on the CPU of an Apple M1
(arm64, macOS). Only the FFNs are executed by HFFN; embeddings, attention, LayerNorm, KV cache and head stay in PyTorch. The
kernel is the **compact G4** kernel: the hidden neurons are split into four fixed groups (the calling thread owns group 0,
three workers own the others), each group has its own private reduction buffer, the reduction is deterministic, and the
supports are exact (offsets, no padding). It is also the forward of the M1 CPU training leaves (`../../../ffn/training`,
`../../training`): they call it through `hffn_inference_runtime/dual_entry.py`.

Results: [RESULTS.md](RESULTS.md) (the isolated FFN and Pythia-410M).

## Scope

| | |
|---|---|
| Degree of optimisation | one kernel with three entries (decode N = 1, a baseline route and a dual route that avoids the reduction), four fixed groups, vectorised in plain C++ and left to the compiler (`-O3 -ffast-math`). No manual micro-kernel, no matrix unit (the dense control uses Apple's Accelerate matrix unit; the compact path cannot). |
| Hardware | Apple M1 (arm64), macOS, `clang++`. Measured on one Mac with 16 GB, passively cooled. Not tried on other Apple chips or on other arm64 machines. |
| Models | the three Pythia models (GPT-NeoX, float32) with the FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from compact exports (18 exports); an isolated FFN given in the same compact layout. |
| Results | [RESULTS.md](RESULTS.md); each statement carries its evidence class. |

## Contents

```
check.sh                          verification without timing (below)
hffn_inference_runtime/           the runtime
  compact_format.py               reader of the compact layer (hffn_training_compact_g4_v2) and the dual-route tables
  compact_runtime.py              HFFNCompactG4InferenceRuntime: workspaces per (layer, N), routes, calls
  compact_model.py                the Pythia MLP replacement (no autograd state)
  compact_inference_loader.py     direct Pythia loader (non-FFN weights from model.safetensors, FFN from the export)
  compact_torch_extension.py      build and load of the native extension (into build/)
  dual_entry.py                   the dual entry that also stores z and h, bound once for the M1 training leaves
hffn_pythia_m1/                   the shared Pythia loader (common/pythia) bound to this runtime
native/hffn_compact_g4_inference.cpp          the kernel
torch_extension/hffn_compact_inference_torch_binding.cpp   its PyTorch binding
checks/                           parity per layer (gate 1) and end to end (gate 2)
```

The leaf finds `common/pythia` as the first ancestor folder that holds it: use it, and copy it, as the `runtime/` tree.

## Requirements and assets

- macOS arm64, `clang++` (Command Line Tools), PyTorch (CPU), transformers, safetensors. `HFFN_PYTHON` names the interpreter
  (default `python`).
- `HFFN_ASSETS` names the assets folder (layout in the README at the root of the tree). `check.sh` also needs the prompt
  tokens of the shared Pythia level, generated once as described there.
- Build: the extension is built into `build/` at the first call. Without Ninja, the explicit `clang++ -O3 -ffast-math
  -std=c++17 -fPIC -dynamiclib -pthread` command builds it; with Ninja installed, `torch.utils.cpp_extension.load` is tried
  first with the same flags. The two recipes do not produce the same machine code (the bodies of the kernel functions are
  equal, some cold code and binding symbols differ); the published figures come from the command recipe, so install nothing
  that provides Ninja when reproducing a figure.

## Fairness of the two arms

`load_pythia` builds the same model for both arms: same configuration, same non-FFN tensors read from the base checkpoint, same
FFN values read from the same compact export. Only the FFN call differs: `ffn="dense"` is the model's own two `nn.Linear`
and a GELU with weights set from the export on the fixed support and zero elsewhere (the dense iso-support control);
`ffn="hffn"` passes the same activations to this runtime. Neither arm reads the FFN tensors of the checkpoint. PyTorch runs on
four threads in the checks.

## HFFN execution

Per call, on a block of N tokens: the up projection and the GELU per hidden neuron, one group per thread; then the down
projection. N = 1 takes the decode entry. The baseline route accumulates each group's contribution into its private
`y_local[4, d_model, N]` and reduces the four buffers in a fixed order; the dual route reads the same edges in output order
(a reverse schedule built at load time) and needs no reduction. The runtime's `execution` is `baseline` (the default of the
loader), `dual_route` or `adaptive` (dual from `dual_route_min_n` = 64 tokens). Compact: values, indices, offsets (and the
reverse schedule). Dense: `x`, `z`, `h`, `y` and their buffers, one set per (layer, N), allocated once. `W_up`, `W_down`
and the masks are never materialised outside the dense arm and the oracle of the checks.

## Checks

```bash
./check.sh [<base checkpoints dir> <exports dir>]     # without arguments: $HFFN_ASSETS
```

`check.sh` compiles the Python sources, then for each of the 18 exports runs:

- **Gate 1, `checks/parity_layers.py`**: for every FFN of the export and each N, two calls on the same input give the same
  bits (G4 is fixed, so determinism replaces the x86 criterion "same bits for every thread count"). The distance to a float64
  oracle is printed next to that of the float32 dense FFN; it is not a criterion.
- **Gate 2, `checks/parity_model.py --threads 4 --negative-control`**: end to end on text, **each HFFN token is equal to the
  token of the dense float32 arm, or failing that to the token of the same dense arm in float64, or failing that to a token
  explained by the float64 margin** (at most the larger of one float32 step and the dense float32 control's own error at
  that position). A negative control (the attention weight of layer 3 of the HFFN arm multiplied by 1.5) must be detected.

The check ends with `check: PASSED (18 exports)` or `check: FAILED`, and stops when it does not find 18 exports. It times
nothing and runs on a Mac with no other job.

## Open point for the shared level

The shared loader (`common/pythia`) hands the blocks and the layer dictionaries to its runtime. This leaf rebinds the blocks
to its own `HFFNCompactG4InferenceMLP`, which writes into the runtime's reusable output buffer and passes the sequence
length, and re-reads the layers with its own loader, because the shared dictionaries do not carry `group_hidden_offsets`.
Moving both to the shared level is open.

## Known limits

- **The isolated-FFN figures** of `RESULTS.md` were obtained by this runtime on isolated-FFN exports, through a loader of
  those exports that is not published: this leaf's loader reads the Pythia layer files. A layer built in the compact layout
  can be given to `HFFNCompactG4InferenceRuntime` directly.
- **Unused options.** The runtime keeps the options of its constructor (`execution`, `dual_route_min_n`, the batch-splitting
  switches `batch_split32`, `batch_split16`, `batch_split16_min_n`, `batch_split16_t96`, `batch_split_t96_size`). The loader's
  defaults are the configuration that was measured (`execution="baseline"`, `batch_split32` on, a split of 96); the other
  values are not covered by a check or a figure here.
- **High density:** HFFN is slower than dense where the FFN is dense enough, and in prefill on this hardware (see
  RESULTS.md): the expected boundary of the format.
- A float32 execution is not token-identical to another: the criterion rests on the near-tie band of Gate 2.
- One machine, one boot; no other arm64 hardware was tried.
- The `batch_split` policy keys on a sequence length of 32 or 96 and on the token count: other shapes take the unsplit path.

## Provenance

The kernel is the compact G4 kernel of the Apple M1 CPU runtime; the x86 kernel and the training kernel derive from it. See
[RESULTS.md](RESULTS.md) for the runs.

## Licence

MIT: the file `LICENSE` at the root of the `runtime/` tree covers this leaf.
