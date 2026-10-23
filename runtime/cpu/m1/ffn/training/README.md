# HFFN CPU runtime on Apple M1: FFN training

One training step of one isolated FFN with a fixed support, on the CPU of an Apple M1: forward with `z` stored, backward
and AdamW, on persistent buffers. Parameters, gradients and AdamW states are the compact values of the support
(`up_val`, `down_val`, `b_up`, `b_down`); the activations `x`, `z`, `h`, `y` and their gradients are dense. There is no
dense `W_up`, `W_down`, mask, gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the kernel of the sibling inference leaf** ([`../../llm/inference/compact`](../../llm/inference/compact/README.md), dual route, entry that stores `z`, from N = 1), loaded as built there (no second forward). Backward and AdamW: the G4 training kernel of [`cpu/common/llm/training`](../../../../cpu/common/llm/training/README.md). Four fixed groups, plain C++ under `-O3 -ffast-math`, no manual micro-kernel. One block of N tokens per call. |
| Hardware | Apple M1 (arm64), macOS, `clang++`. One machine, passively cooled. |
| Models | an isolated FFN (`d_model`, `d_ff` up to 32767, GELU, float32), with an irregular or homogeneous fixed support, given as a compact layer of the `hffn_training_compact_g4_v2` format. The 18 exports of the Pythia FFNs (70M, 160M, 410M at 6 densities) are the layers of the parity check. |
| Results | [RESULTS.md](RESULTS.md): time and memory of one step against the dense iso-support control, per layer. |

## Contents

```
check.sh                        every-export check, without timing (below)
hffn_ffn_training/              the entry point of the leaf on the shared level
  kernels.py                    binds the inference kernel of the sibling leaf and the training kernel
  build.py                      builds the training kernel into build/; locates the inference library
shared_toy.py                   toy facade: HFFNToyFFN on the shared module, dimensions and support as attributes
models.py                       masked dense reference (the dense control of the toy tools) and its support builder
train_toy_ffn_hffn_cpu.py       toy teacher-student training on the shared step, against the masked dense reference
compare_training_curves.py      the same, over several configurations, with parity checks of the reference
config.py data.py io.py parity.py trainer.py torch_info.py paths.py     helpers of the two toy tools
```

The step, the module, the checkpoint API and the check batch are at the shared level `cpu/common/ffn/training`, which this
leaf puts on its package path; the leaf finds `cpu/common`, `common/ffn` and `common/pythia` as the folders above it, so it
is used and copied as the `runtime/` tree.

## Requirements and assets

- macOS arm64, `clang++`, PyTorch (CPU), numpy. `HFFN_PYTHON` names the interpreter (default `python`).
- `HFFN_ASSETS` names the assets (no default; layout at the root of the tree). `check.sh` reads the compact exports
  `pythia/exports/<model>_sparse<P>_ffn_only/export/` with their `manifest_compact_g4_v2.json`. The toy tools read their
  datasets from `homogeneous/models/<cell>/dataset/`. The synthetic layers of the check come from the deterministic
  generator of `common/ffn`, not from assets.
- The inference library is built by the sibling leaf into its own `build/` the first time, then loaded as it is.

## Checks

```bash
./check.sh            # every export and the synthetic layers; ends with EXIT=0 when all gates passed
./check.sh smoke      # two exports (pythia70m 10 %, pythia410m 80 %), one layer each, and the synthetic layers
```

Without timing; writes its records to `runs/`. For each layer, on a block of N tokens:

- **gate 2**: the forward (`y`, `z`, `h`) and the backward (`grad_x` and the four compact gradients) of the HFFN step
  against the direct equation in float64, for a set of block sizes N (1, 2, 8, 13, 32, 61, 64, 97, 128, 192, 203, 512),
  each quantity within a relative error of 1e-4 and finite;
- **gate 3**: one AdamW update at N = 128 against its closed form in float64, fed with the float64 gradients of gate 2, on
  the entries whose gradient is at least 1 % of the largest of their tensor, within 1e-2 of the learning rate;
- the API: resize of N, leased contexts for several graphs in flight, checkpoint round trip (compact and legacy state),
  refusals that must happen (a context used twice, a corrupted checkpoint, a parameter that no longer shares the storage of
  the step), a check that no caller reaches another forward, and a reader that recomputes the verdict from the raw records
  and is itself tested with corrupted records.

The check stops on a failed record, and fails when a source, an asset or a built library changed during the run.

## Known limits

- **Dense control:** the control of the figures is the masked dense FFN of PyTorch on the same support, run by the same
  adapter; its timing is a control, not a path of this runtime.
- **One block per call:** a step takes one block of N tokens. Larger blocks favour the dense control on this CPU
  (see RESULTS.md).
- **Residue of the toy tools:** `models.py` still holds `HFFNCompactFFN`, a Python per-neuron forward that the toy tools
  once used. No caller reaches it (the check `callers.py` makes any call fail) and it is not a path of this runtime; it is
  kept as the type annotation of the toy helpers and is to be removed with them.
- **This leaf depends on the sibling inference leaf** for its forward kernel: copy the `runtime/` tree, not the leaf alone.
- Float32 trajectories are not compared across builds: the criterion is the distance to float64 (gates 2 and 3), not
  convergence equivalence.

## Provenance

See [RESULTS.md](RESULTS.md) for the run and the evidence class of each figure.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
