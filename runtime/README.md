# HFFN runtime

Hidden-FFN (HFFN) stores, executes, trains and updates fixed-support sparse feed-forward networks (FFN) **by hidden
neuron**. For `z = W_up x + b_up`, `h = activation(z)`, `y = W_down h + b_down`, hidden neuron `r` owns its active incoming
connections `(up_idx[r], up_val[r])`, its bias `b_up[r]` and its active outgoing connections `(down_idx[r],
down_val[r])`. The sum of all hidden contributions is the original sparse FFN, up to precision and reduction order: HFFN
changes the layout and the scheduling of the same computation, it approximates neither the activation nor the support.

> Compact when a quantity is bounded by the support (active weight values, indices and offsets, gradients of active
> weights, their AdamW states); dense when it is bounded by the activation space (`x`, `z`, `h`, `y`, their gradients and
> the reduction buffers).

No dense `W_up` or `W_down`, no dense gradient or AdamW moment and no mask exists in an HFFN path; a dense matrix appears
only in the dense control of a check and in the float64 oracle. One call processes the whole block of `N` tokens.

This folder is **one tree, to be used and copied as one tree**: the leaves find the shared levels above them (as the first
ancestor folder that holds them), so they work wherever `runtime/` is placed, with no path to set. It is the second
published version (v2), which replaces v1: a unified runtime of sixteen leaves, measured against its predecessor (A) and
against dense in the same processes, with its losses written as limits (see *Degree of optimisation*). Results:
[`RESULTS.md`](RESULTS.md) for the protocol, the runs and the limits that apply to every leaf, and the `RESULTS.md` of each
leaf for its tables.

## What is here

A leaf is `<hardware>/<ffn|llm>/<inference|training>`: `ffn` is an isolated FFN, `llm` is a model with attention around
the FFN (Pythia, GPT-NeoX). Only the FFN is executed by HFFN; embeddings, attention, norms, residuals, the head, the KV
cache and the optimiser of the non-FFN weights are the dense PyTorch of the model, on every hardware.

| Leaf | Folder | What it runs |
|---|---|---|
| x86 CPU, FFN inference | [`cpu/x86/ffn/inference`](cpu/x86/ffn/inference/README.md) | the AVX2/FMA kernel (T threads) on an isolated FFN, with its deterministic test layers |
| x86 CPU, FFN training | [`cpu/x86/ffn/training`](cpu/x86/ffn/training/README.md) | one FFN training step (forward with `z` stored, backward, AdamW) |
| x86 CPU, LLM inference | [`cpu/x86/llm/inference`](cpu/x86/llm/inference/README.md) | Pythia 70M, 160M and 410M, FFNs on the x86 kernel inside PyTorch |
| x86 CPU, LLM training | [`cpu/x86/llm/training`](cpu/x86/llm/training/README.md) | Pythia with compact G4 FFN modules (parameters, gradients, AdamW state) |
| M1 CPU, FFN inference | [`cpu/m1/ffn/inference`](cpu/m1/ffn/inference/README.md) | the compact G4 kernel (four fixed groups of hidden neurons, exact offsets) on an isolated FFN, with its dense control |
| M1 CPU, LLM inference | [`cpu/m1/llm/inference/compact`](cpu/m1/llm/inference/compact/README.md) | Pythia on the compact G4 kernel of the M1 CPU FFN inference leaf: loader, model, parity checks |
| M1 CPU, FFN training | [`cpu/m1/ffn/training`](cpu/m1/ffn/training/README.md) | one FFN training step on the compact G4 kernel |
| M1 CPU, LLM training | [`cpu/m1/llm/training`](cpu/m1/llm/training/README.md) | Pythia with compact G4 FFN modules |
| M1 GPU, FFN inference | [`gpu/m1/ffn/inference`](gpu/m1/ffn/inference/README.md) | an exported FFN on the Metal FFN kernel |
| M1 GPU, FFN training | [`gpu/m1/ffn/training`](gpu/m1/ffn/training/README.md) | one FFN training step in one native Metal context (forward with `z` stored, backward, AdamW); the owner of the M1 GPU backward and update |
| M1 GPU, LLM inference | [`gpu/m1/llm/inference/pythia_torch_attention`](gpu/m1/llm/inference/pythia_torch_attention/README.md) | Pythia with the FFNs on Metal, the rest on PyTorch/MPS |
| M1 GPU, LLM training | [`gpu/m1/llm/training`](gpu/m1/llm/training/README.md) | Pythia training with the FFNs as compact Metal modules; backward and update through the native context of the FFN training leaf |
| Nvidia GPU, FFN inference | [`gpu/nvidia/ffn/inference`](gpu/nvidia/ffn/inference/README.md) | the CUDA kernels on an isolated FFN |
| Nvidia GPU, FFN training | [`gpu/nvidia/ffn/training`](gpu/nvidia/ffn/training/README.md) | one FFN training step on CUDA |
| Nvidia GPU, LLM inference | [`gpu/nvidia/llm/inference`](gpu/nvidia/llm/inference/README.md) | Pythia, FFNs on the CUDA kernels inside PyTorch |
| Nvidia GPU, LLM training | [`gpu/nvidia/llm/training`](gpu/nvidia/llm/training/README.md) | Pythia training with the FFNs on the shared CUDA kernels; backward and update from the Nvidia FFN training leaf |

Shared levels, each at the highest level it covers (a leaf never carries its own copy):

| Level | Folder | Holds |
|---|---|---|
| FFN, all hardware | [`common/ffn`](common/ffn/README.md) | deterministic generator and fingerprints of the test layers, float64 oracle, dense PyTorch reference |
| Pythia, all hardware | [`common/pythia`](common/pythia/README.md) | export reader, two-arm loader, end-to-end token check, training gates, request-level timing harness, prompt-token generator, source verification |
| CPU FFN training | [`cpu/common/ffn/training`](cpu/common/ffn/training/README.md) | the training step, module, checkpoint API and the check batch of the M1 and x86 FFN training leaves |
| CPU LLM training | [`cpu/common/llm/training`](cpu/common/llm/training/README.md) | the G4 training kernel and its binding, the compact runtime and the gates of the M1 and x86 LLM training leaves |
| CPU build recipes | [`cpu/common/recipe`](cpu/common/recipe/README.md) | the two build calls of the CPU leaves (compact extension, plain dynamic library) |
| x86 CPU | [`cpu/x86/common`](cpu/x86/common/README.md) | the x86 inference kernel, shared by its four leaves, and its Python build and binding |
| Nvidia GPU | [`gpu/nvidia/common/kernels`](gpu/nvidia/common/kernels/README.md) | the CUDA kernels and runtime, shared by the four Nvidia leaves |
| M1 GPU, primitives | `gpu/m1/common/hffn_m1` | the Metal sources and Python emitters shared by the M1 GPU leaves (activation, AdamW, backward plan and sums, fused edges); no README of its own |
| M1 GPU | [`gpu/m1/llm/inference/native_transformer`](gpu/m1/llm/inference/native_transformer/README.md) | the Metal FFN runtime, used by the M1 GPU leaves |

Training leaves take the FFN forward from the kernel of their sibling inference leaf and, for an LLM training leaf, the backward and the AdamW update from the FFN training leaf of its hardware (the M1 GPU LLM training leaf reaches the native context of `gpu/m1/ffn/training` through an adapter); no leaf carries a second copy of a mechanism.

## Requirements

- Python 3.10 or later with PyTorch, `numpy`, `transformers` and `safetensors` (the versions of the Mac checks recorded
  below were PyTorch 2.10.0, transformers 5.9.0, numpy 2.4.3, safetensors 0.7.0). Each leaf with extra requirements has a
  `requirements.txt`. The M1 GPU LLM training leaf binds the native context of the FFN training leaf to PyTorch tensors
  and needs PyTorch 2.10.0 exactly.
- CPU leaves: a C++ compiler (`clang++` on macOS arm64; `g++` 9 or later, 13 or later for the x86 LLM training leaf, on
  Linux x86_64 with AVX2 and FMA). Kernels are built on first use into a `build/` folder of the leaf.
- Nvidia leaves: CUDA 12, a PyTorch build with CUDA and CuPy for CUDA 12 (kernels are compiled at run time by NVRTC); the
  checks recorded below used PyTorch 2.8.0 built for CUDA 12.9.
- M1 GPU leaves: an Apple GPU with PyTorch MPS. Never set `PYTORCH_ENABLE_MPS_FALLBACK`: the scripts refuse it.
- `HFFN_PYTHON` names the interpreter the shell scripts use (default `python` of the `PATH`).

## Assets: nothing is distributed

No checkpoint, export or dataset is part of this tree. A check that needs a real model reads it from the folder named by
the environment variable `HFFN_ASSETS` (no default; the scripts stop when it is not set):

```text
$HFFN_ASSETS/pythia/models/base/<model>/                                     base checkpoint (pythia70m, pythia160m, pythia410m)
$HFFN_ASSETS/pythia/exports/<model>_sparse<P>_ffn_only/export/layer<k>/hffn_training_compact_g4_v2.pt
$HFFN_ASSETS/pythia/exports/<model>_sparse<P>_ffn_only/export/manifest_compact_g4_v2.json   read by the CPU FFN training check
$HFFN_ASSETS/homogeneous/models/<cell>/export/hidden_ffn_format.pt          isolated-FFN exports (cells such as 512_2048_d020)
```

`<P>` is the density in percent (10, 20, 30, 50, 70, 80). **Naming:** in these folder names `sparse20` means a density of
20 % (20 % of the weights are kept), not 20 % of zeros; the published tables write `d20`. The pruned models are WANDA
supports (irregular degree per hidden neuron, exact offsets); the `homogeneous/models` cells are synthetic
teacher-student FFNs on random supports of the stated density (also irregular, despite the historical folder name).

Tests of the leaves that can run without an asset build their layers from the deterministic generator of `common/ffn`
(fixed seed, fingerprints pinned in `common/ffn/reference/fingerprints.json`).

**The prompt tokens are generated, not distributed.** `common/pythia/hffn_pythia/wikitext2_test_tokens.json` derives
from the WikiText-2 raw v1 test split (CC BY-SA). Generate it once, from the base checkpoint's tokenizer and the Arrow file
of the test split:

```bash
python -B common/pythia/tools/make_prompt_tokens.py $HFFN_ASSETS/pythia/models/base/pythia70m <wikitext-test.arrow>
```

It writes the file in place and the checks compare its SHA-256 with the pinned
`a6ea1a41a54531b5b6120f8eaebf24ad7275083e2f7c11ac5f50be7b7f955178`. A `.gitignore` keeps it out of version control.

## Verification state of this tree

This tree is the frozen candidate `candidate_v2` (git commit `58653b0ca904d84e53f7e020fbeb67cd5f6dea38`) cleaned for publication: path resolution, comments, removed switches, probe flags and dead
diagnostics changed, the kernel sources being the measured ones (the `sources/` manifests of the x86 and Nvidia LLM
leaves and `common/pythia/tools/verify_sources.py` check this). The full every-export checks and gates of the
candidate are its prerequisites and are not repeated here. The tree was copied alone to an empty folder, outside any
repository, and on each hardware the native build, the import and a bounded smoke of every leaf were run there, plus the
targeted parity of the leaves whose executed code the cleaning touched. These checks time nothing; no figure of this
file is a performance number.

| Hardware | Check run on the copy | Result |
|---|---|---|
| M1 CPU, FFN and LLM inference (`cpu/m1/llm/inference/compact`) | `check.sh check` on the exports of the copy | passed |
| M1 CPU, FFN training | `check.sh smoke` | passed |
| M1 CPU, LLM training | smoke on Pythia-70M d20; `parity_hffn_layer_training.py` on one layer | passed |
| M1 GPU, FFN inference | `smoke_homogeneous_ffn.py` on `256_1024_d010` and on a synthetic layer | passed |
| M1 GPU, FFN training | `check.sh check` (every gate: build, smokes, forward, backward, AdamW, dense gates, in-process export check) | passed |
| M1 GPU, LLM inference | `check.sh check` on Pythia-70M d20 (6 of 6 layers, max error 3.8e-05) | passed |
| M1 GPU, LLM training | `check.sh check` on Pythia-70M d20 (6 of 6 layers, 1 of 1 model step); `validate_optimizer_resume.py` | passed |
| x86 CPU (EPYC Milan VM), four leaves | FFN inference: 51 tests and 3 layers; FFN training smoke; LLM inference check with its negative control detected; LLM training gates | passed |
| Nvidia GPU (Quadro RTX 5000 VM), four leaves | FFN inference and training checks; LLM inference check on 1 export; LLM training unit test, context check (21 of 21) and model check (11 of 11) | passed |

The LLM leaves ran on one or two exports of the copy (their `check.sh` takes the export names as arguments; with none it
runs every export): the every-export check belongs to the candidate. The Nvidia checks used the pinned virtual
environment of the VM (PyTorch 2.8.0, CUDA 12.9). On the M1 GPU the cleaning removed three switches of the Metal
FFN runtime that were always on (their removal leaves the executed code unchanged in meaning), and the M1 GPU rows above were run on the cleaned tree. On the other
hardware the checks ran before the last edits, which are comments, messages and a renamed constant of the check scripts;
no kernel source changed.

## Degree of optimisation

This is v2 of a unified runtime, not an optimised one. Within each hardware the kernels were chosen by parity and by no
loss of time or memory against the earlier per-leaf versions; they were not tuned for speed, and where a leaf loses against
its published predecessor the loss is written as a limit in its `RESULTS.md`. In particular: the CPU kernels use no matrix
unit, no AVX-512, AMX, BF16 or INT8 path; the Nvidia kernels are a port of the Metal kernels, with the thresholds first
measured on an Apple M1; training runs one block of tokens per call and the larger blocks favour the dense control on CPU.
Several optimisation proposals, each estimated above 2 % on the quantity its leaf is judged on, are not done and are
carried as written limits. The figures are the base against which they would be measured. The boundary of the format is
reported, not hidden: at high density, metadata and indirection make HFFN slower than dense.

## What is not included

- Checkpoints, exports, datasets, the run folders and the bench that produced the published figures. The figures can be
  read in the `RESULTS.md` files, not regenerated from this tree. Each leaf's `check.sh` (or its tests) checks parity and
  does not time.
- The native Metal attention of `native_transformer`: set aside, not developed and not a production path.
- MoE, multilayer and lateral variants: not part of this tree.

## Licence

MIT, one file for the whole tree: [`LICENSE`](LICENSE).
