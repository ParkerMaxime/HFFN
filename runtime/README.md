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
ancestor folder that holds them), so they work wherever `runtime/` is placed, with no path to set. It is the first
published version (v1): a unified runtime, measured and documented as it stands, not an optimised one (see *Degree of
optimisation*). Results: [`RESULTS.md`](RESULTS.md) for the protocol, the runs and the limits that apply to every leaf, and
the `RESULTS.md` of each leaf for its tables.

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
| M1 CPU, FFN and LLM inference | [`cpu/m1/llm/inference/compact`](cpu/m1/llm/inference/compact/README.md) | the compact G4 kernel (four fixed groups of hidden neurons, exact offsets) on the Apple M1 CPU; it also runs an isolated FFN, which has no separate folder |
| M1 CPU, FFN training | [`cpu/m1/ffn/training`](cpu/m1/ffn/training/README.md) | one FFN training step on the compact G4 kernel |
| M1 CPU, LLM training | [`cpu/m1/llm/training`](cpu/m1/llm/training/README.md) | Pythia with compact G4 FFN modules |
| M1 GPU, FFN inference | [`gpu/m1/ffn/inference`](gpu/m1/ffn/inference/README.md) | an exported FFN on the Metal FFN kernel |
| M1 GPU, LLM inference | [`gpu/m1/llm/inference/pythia_torch_attention`](gpu/m1/llm/inference/pythia_torch_attention/README.md) | Pythia with the FFNs on Metal, the rest on PyTorch/MPS |
| M1 GPU, LLM training | [`gpu/m1/llm/training`](gpu/m1/llm/training/README.md) | Pythia training with the FFNs as compact Metal modules |
| Nvidia GPU, FFN inference | [`gpu/nvidia/ffn/inference`](gpu/nvidia/ffn/inference/README.md) | the CUDA kernels on an isolated FFN |
| Nvidia GPU, FFN training | [`gpu/nvidia/ffn/training`](gpu/nvidia/ffn/training/README.md) | one FFN training step on CUDA |
| Nvidia GPU, LLM inference | [`gpu/nvidia/llm/inference`](gpu/nvidia/llm/inference/README.md) | Pythia, FFNs on the CUDA kernels inside PyTorch |

Shared levels, each at the highest level it covers (a leaf never carries its own copy):

| Level | Folder | Holds |
|---|---|---|
| FFN, all hardware | [`common/ffn`](common/ffn/README.md) | deterministic generator and fingerprints of the test layers, float64 oracle, dense PyTorch reference |
| Pythia, all hardware | [`common/pythia`](common/pythia/README.md) | export reader, two-arm loader, end-to-end token check, training gates, request-level timing harness, prompt-token generator, source verification |
| CPU FFN training | [`cpu/common/ffn/training`](cpu/common/ffn/training/README.md) | the training step, module, checkpoint API and the check batch of the M1 and x86 FFN training leaves |
| CPU LLM training | [`cpu/common/llm/training`](cpu/common/llm/training/README.md) | the G4 training kernel and its binding, the compact runtime and the gates of the M1 and x86 LLM training leaves |
| x86 CPU | [`cpu/x86/common`](cpu/x86/common/README.md) | the x86 inference kernel, shared by its four leaves, and its Python build and binding |
| Nvidia GPU | [`gpu/nvidia/common/kernels`](gpu/nvidia/common/kernels/README.md) | the CUDA kernels and runtime, shared by the three Nvidia leaves |
| M1 GPU | [`gpu/m1/llm/inference/native_transformer`](gpu/m1/llm/inference/native_transformer/README.md) | the Metal FFN runtime, used by the three M1 GPU leaves |

Training leaves use the kernel of their sibling inference leaf for the forward; none carries a second forward.

## Requirements

- Python 3.10 or later with PyTorch, `numpy`, `transformers` and `safetensors` (the versions of the checks recorded below
  were PyTorch 2.10.0, transformers 5.9.0, numpy 2.4.3, safetensors 0.7.0). Each leaf with extra requirements has a
  `requirements.txt`.
- CPU leaves: a C++ compiler (`clang++` on macOS arm64; `g++` 9 or later, 13 or later for the x86 LLM training leaf, on
  Linux x86_64 with AVX2 and FMA). Kernels are built on first use into a `build/` folder of the leaf.
- Nvidia leaves: CUDA 12, a PyTorch build with CUDA and CuPy for CUDA 12 (kernels are compiled at run time by NVRTC).
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

This tree was assembled from the versions that were measured, with only path resolution, comments, removed switches and
removed dead diagnostics changed (the kernel sources are the measured ones; `sources/` manifests of the x86 and Nvidia LLM
leaves, and `common/pythia/tools/verify_sources.py`, check this). It was then copied alone to an empty folder, outside
any repository, and its embedded checks were run there. These checks time nothing; no figure of this file is a
performance number.

| Leaf | Check run on the copy | Result |
|---|---|---|
| M1 CPU, FFN and LLM inference (`cpu/m1/llm/inference/compact`) | `check.sh check`, 18 exports | passed |
| M1 CPU, FFN training | `check.sh check`, 21 batch entries (18 exports and 3 synthetic layers) | passed (exit 0) |
| M1 CPU, LLM training | `check.sh check`, 18 exports; `parity_hffn_layer_training.py` on one layer | passed |
| M1 GPU, FFN inference | `smoke_homogeneous_ffn.py --synthetic 128 512 256` | passed |
| M1 GPU, LLM inference | `check.sh check` on 2 exports (Pythia-70M d20, Pythia-410M d80; 30 of 30 layers); the end-to-end smoke; `validate_pythia_inference_matrix.py` on Pythia-70M d20 | passed |
| M1 GPU, LLM training | `check.sh check` on 2 exports (Pythia-70M d10 and d80; 12 of 12 layers, 2 of 2 model steps); the scripts of the leaf on Pythia-70M d20 | passed |
| x86 CPU, four leaves | compiled for syntax and `py_compile` only | the embedded checks were **not run on the tree as published** |
| Nvidia GPU, three leaves | `py_compile` and `sh -n` only | the embedded checks were **not run on the tree as published** |

The M1 GPU checks ran on a subset of the exports (the leaf's `check.sh` takes the export names as arguments; with none it
runs every export). The x86 and Nvidia checks need their hardware: run `check.sh` of each leaf there before relying on it.
Versions of the checks: PyTorch 2.10.0, transformers 5.9.0, numpy 2.4.3, safetensors 0.7.0, on macOS with an Apple M1.

## Degree of optimisation

This is a unified first version. Within each hardware, the kernels were chosen by parity and by no loss of time or memory
against the earlier per-leaf versions; they were not tuned for speed. In particular: the CPU kernels use no matrix unit,
no AVX-512, AMX, BF16 or INT8 path; the Nvidia kernels are a port of the Metal
kernels, with the thresholds first measured on an Apple M1; training runs one block of tokens per call and the larger
blocks favour the dense control on CPU. Several optimisation proposals, each estimated above 2 % on the quantity its leaf is
judged on, are not done. The figures are the base against which they would be measured. The boundary of the format is
reported, not hidden: at high density, metadata and indirection make HFFN slower than dense.

## What is not included

- Checkpoints, exports, datasets, the run folders and the bench that produced the published figures. The figures can be
  read in the `RESULTS.md` files, not regenerated from this tree. Each leaf's `check.sh` (or its tests) checks parity and
  does not time.
- M1 GPU FFN training and Nvidia LLM training: no leaf (see `RESULTS.md`, *Not covered*).
- The legacy padded FFN kernel of the M1 CPU: the compact G4 kernel above replaces it, and its figures are in the
  `RESULTS.md` of the compact leaf.
- Native Metal attention and addition models.
- MoE, multilayer and lateral variants: not part of this tree.

## Licence

MIT, one file for the whole tree: [`LICENSE`](LICENSE).
