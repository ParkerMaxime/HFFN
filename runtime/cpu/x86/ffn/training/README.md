# HFFN CPU runtime on x86_64: FFN training

One training step of one isolated FFN with a fixed support, on a Linux x86_64 CPU: forward with `z` stored, backward and
AdamW, on persistent buffers. Parameters, gradients and AdamW states are the compact values of the support (`up_val`,
`down_val`, `b_up`, `b_down`); the activations and their gradients are dense. There is no dense `W_up`, `W_down`, mask,
gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the shared x86 inference kernel** ([`cpu/x86/common`](../../../../cpu/x86/common/README.md)), slots entry of its dual route with `z` stored; backward and AdamW: the G4 training kernel of [`cpu/common/llm/training`](../../../../cpu/common/llm/training/README.md). AVX2/FMA, four threads (one per group of hidden neurons), `-O3 -ffast-math`, no manual micro-kernel and no AVX-512, AMX, BF16 or INT8 path. One block of N tokens per call, N of at least 2: the decode path of the inference kernel (N = 1) does not store `z`. |
| Hardware | Linux x86_64 with AVX2 and FMA; `g++` 9 or later. Measured on one AWS `c7a.2xlarge` (AMD EPYC 9R14, 8 vCPU). Other x86 CPUs were not tried. |
| Models | an isolated FFN (`d_model`, `d_ff` up to 32767, GELU, float32) with an irregular or homogeneous fixed support given as a compact layer of the `hffn_training_compact_g4_v2` format. The 18 exports of the Pythia FFNs (70M, 160M, 410M at 6 densities) are the layers of the check. |
| Results | [RESULTS.md](RESULTS.md): time and memory of one step against the dense iso-support control, per layer. |

## Contents

```
check.sh                    every-export check, without timing (below)
hffn_ffn_training/          the entry point of the leaf on the shared level
  kernels.py                the slots entry of the shared x86 inference kernel and its smallest N
  build.py                  builds the training kernel into build/ with the flags of the inference kernel
```

The step, the module, the checkpoint API and the check batch are at the shared level `cpu/common/ffn/training`, which this
leaf puts on its package path (the same code serves the M1 leaf). Use and copy the `runtime/` tree.

## Requirements and assets

- Linux x86_64, glibc, `g++`, PyTorch (CPU), numpy. `HFFN_PYTHON` names the interpreter (default `python`).
- `HFFN_ASSETS` names the assets (no default; layout at the root of the tree): the compact exports of the Pythia FFNs with
  their `manifest_compact_g4_v2.json`. The synthetic layers of the check come from the deterministic generator of
  `common/ffn`, not from assets.
- Kernels are built into `build/` folders on first use.

## Checks

```bash
./check.sh            # every export and the synthetic layers; ends with EXIT=0 when all gates passed
./check.sh smoke      # two exports (pythia70m 10 %, pythia410m 80 %), one layer each, and the synthetic layers
```

Without timing; writes its records to `runs/`. For each layer:

- **gate 2**: the forward (`y`, `z`, `h`) and the backward (`grad_x` and the four compact gradients) against the direct
  equation in float64, for a set of block sizes N (2, 8, 13, 32, 61, 97, 128, 203, 512), each quantity within a relative error
  of 1e-4 and finite;
- **gate 3**: one AdamW update at N = 128 against its closed form in float64, fed with the float64 gradients of gate 2, on
  the entries whose gradient is at least 1 % of the largest of their tensor, within 1e-2 of the learning rate;
- the API (resize of N, leased contexts, checkpoint round trip, refusals that must happen) and a reader that recomputes the
  verdict from the raw records and is itself tested with corrupted records.

The check stops on a failed record, and fails when a source, an asset or a built library changed during the run.

## Known limits

- **Dense control:** the control of the figures is the masked dense FFN of PyTorch on the same support, run by the same
  adapter; it is a control, not a path of this runtime.
- **One block per call, N >= 2:** larger blocks favour the dense control on this CPU (see RESULTS.md); a one-token step is
  not covered.
- **One machine type:** the figures come from one cloud instance; x86 figures are compared only within one execution.
- Float32 trajectories are not compared across builds: the criterion is the distance to float64 (gates 2 and 3), not
  convergence equivalence.

## Provenance

See [RESULTS.md](RESULTS.md) for the run and the evidence class of each figure.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
