# Published versions

Hidden-FFN (HFFN) stores, executes, trains and updates fixed-support sparse feed-forward networks by hidden neuron. This
folder holds the published versions, one of each part. Only `runtime/` is populated so far.

## What is published

`runtime/` is **one tree, to be used and copied as one tree**: the leaves find the shared levels above them. It is the
second version (v2) of a unified runtime on every covered hardware, with sixteen leaves, each measured against its
predecessor (A) and against dense in the same processes, with its losses written as limits; not an optimised runtime. Start with [`runtime/README.md`](runtime/README.md) (what is here, requirements, the assets layout, the
degree of optimisation, what is not included) and [`runtime/RESULTS.md`](runtime/RESULTS.md) (protocol, runs, reference
cells, limits that apply to every leaf).

| Hardware | FFN inference | FFN training | LLM inference | LLM training |
|---|---|---|---|---|
| x86 CPU (AVX2/FMA) | `runtime/cpu/x86/ffn/inference` | `runtime/cpu/x86/ffn/training` | `runtime/cpu/x86/llm/inference` | `runtime/cpu/x86/llm/training` |
| M1 CPU (compact G4) | `runtime/cpu/m1/llm/inference/compact` (also runs an isolated FFN) | `runtime/cpu/m1/ffn/training` | `runtime/cpu/m1/llm/inference/compact` | `runtime/cpu/m1/llm/training` |
| M1 GPU (Metal) | `runtime/gpu/m1/ffn/inference` | `runtime/gpu/m1/ffn/training` | `runtime/gpu/m1/llm/inference/pythia_torch_attention` | `runtime/gpu/m1/llm/training` |
| Nvidia GPU (CUDA) | `runtime/gpu/nvidia/ffn/inference` | `runtime/gpu/nvidia/ffn/training` | `runtime/gpu/nvidia/llm/inference` | `runtime/gpu/nvidia/llm/training` |

Shared levels, each at the highest level it covers: `runtime/common/ffn` (deterministic generator and fingerprints of
the test layers, float64 oracle, dense reference), `runtime/common/pythia` (export reader, two-arm loader, token check,
training gates, request-level timing harness, prompt-token generator), `runtime/cpu/common/ffn/training` and
`runtime/cpu/common/llm/training` (the CPU training steps and gates of the M1 and x86 leaves), `runtime/cpu/x86/common`
(the x86 inference kernel), `runtime/gpu/nvidia/common/kernels` (the CUDA kernels) and
`runtime/gpu/m1/common/hffn_m1` (the Metal primitives of the M1 GPU leaves), `runtime/cpu/common/recipe` (the CPU build
recipes) and `runtime/gpu/m1/llm/inference/native_transformer` (the Metal FFN runtime). Training leaves use the kernel of
their sibling inference leaf for the forward, and an LLM training leaf the backward and update of the FFN training leaf of
its hardware.

Every leaf has a `README.md` (degree of optimisation, hardware scope, supported models, how it is checked, limits) and a
`RESULTS.md` (its figures, each with the name of its qualified run and its evidence class). The figures come from
16 qualified runs of a bench that is not distributed; they can be read, not regenerated from this tree. The losses of v2
against v1 or against dense (x86 FFN inference, the x86 LLM inference prefill, the M1 GPU LLM training, some cells of the
M1 CPU FFN training, and the cells where dense is faster) are published as limits in `runtime/RESULTS.md`; no figure of
v1 is compared with a figure of v2 across hardware (the x86 and Nvidia machines changed).

Scope: the x86 leaves need Linux x86_64 with AVX2/FMA and `g++` (13 or later for the x86 LLM training leaf) and were
measured on one OVH cloud VM (AMD EPYC-Milan, 8 vCPU); the Nvidia leaves need an NVIDIA GPU with CUDA 12 (PyTorch, CuPy)
and were measured on one OVH VM with a Quadro RTX 5000; the M1 leaves need an Apple M1 and were measured on one 16 GB
machine. Nothing is a claim of speed-up on other hardware.

## What is not included

- **No checkpoint, export or dataset.** The base Pythia checkpoints and the compact exports are those of the Pythia
  artifact of this project; a check that needs them reads the folder named by the environment variable `HFFN_ASSETS`
  (layout in `runtime/README.md`). The tests that can run without an asset build the deterministic synthetic layers of
  `runtime/common/ffn`.
- **The prompt tokens are generated, not distributed.** `runtime/common/pythia/hffn_pythia/wikitext2_test_tokens.json`
  derives from the WikiText-2 test split (CC BY-SA); it is produced once with
  `runtime/common/pythia/tools/make_prompt_tokens.py` and checked against a pinned SHA-256.
- **No raw campaign result and no bench.**
- Not covered: the native Metal attention, and the MoE, multilayer and lateral variants (their folders stay as they are).

## Licence

MIT, one file for the whole `runtime/` tree: `runtime/LICENSE`.
