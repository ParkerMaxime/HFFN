# `runtime/gpu/m1/llm/training` — Pythia training on the Apple M1 GPU

Training of the Pythia models (70M, 160M, 410M) whose FFNs are pruned by WANDA with a fixed support, on the Apple M1 GPU.
Each FFN is a compact module: its parameters (`up_val`, `down_val`, `b_up`, `b_down`) live in one flat pool, and its
gradients and AdamW states hold the active weights only. Attention, embeddings, norms, the head and their optimiser are the
dense PyTorch of the model, on MPS. There is no dense `W_up`, `W_down`, mask, gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **one path per mechanism, owned elsewhere.** Forward: the Metal FFN kernel of the M1 GPU shared level ([`native_transformer`](../inference/native_transformer/README.md)), through its one-bank entry that stores `z`. Backward and compact AdamW: the native context of [`gpu/m1/ffn/training`](../../ffn/training/README.md), reached through the adapter `native_ffn_adapter.py` (`NativeBackwardMLP`), which holds no kernel. The move onto that path was judged on parity alone: its time and memory are limits, and the host-side bridging costs of the adapter are declared below. One block of tokens per call, no matrix unit, no fusion with attention. |
| Hardware | Apple M1 GPU through PyTorch MPS, 16 GB. Not tried on another Apple chip. |
| Models | Pythia 70M, 160M and 410M (float32), FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density (18 exports). |
| Results | [RESULTS.md](RESULTS.md): Pythia-410M at d10 in the timed grid. 410M d20 and d80 are outside the timed grid (swap guard of the Mac), see Known limits. |

## Contents

```
check.sh                       layer and one-step model parity on every export, without timing
native_ffn_adapter.py          NativeBackwardMLP: the adapter onto the native backward and AdamW of gpu/m1/ffn/training
bridge.mm, build_bridge.sh     three host services for the adapter (MTLBuffer handle of an MPS tensor, one gradient blit, autograd version bump)
hffn_metal_llm/                compact layer, runtime (Torch forward), model integration, optimizer driver, checkpoint, loader
smoke_mps_interop.py           PyTorch -> custom Metal -> PyTorch with no explicit synchronisation
smoke_pythia_training.py       one complete training step of Pythia, FFNs on Metal and the rest on MPS
validate_wanda_layer.py        one layer, forward and active gradients, against the shared float64 oracle
validate_pythia_parity.py      one-step full-model gate against float64 and the dense float32 control
validate_optimizer_resume.py   atomic save and resume of weights, PyTorch AdamW, compact moments and step counter
validate_configuration_matrix.py   every export, the layout selected at each N
test_layout_policy.py          CPU-only check of the layout policy of the forward
```

The kernels are at the shared levels `gpu/m1/llm/inference/native_transformer` (forward) and `gpu/m1/ffn/training`
(backward, AdamW); the export reader, the oracle and the training gates are in `common/pythia` and
`cpu/common/llm/training/checks`: use and copy the `runtime/` tree.

## Requirements and assets

PyTorch with MPS, transformers, safetensors, `clang++` with the Metal and Foundation frameworks. **The bridge links against
PyTorch's internal MPS header and requires exactly PyTorch 2.10.0** (`build_bridge.sh` and the adapter refuse any other
version). `HFFN_ASSETS` for the base checkpoints and the exports (layout at the root of the tree). `HFFN_PYTHON` names the
interpreter (default `python`). The scripts refuse `PYTORCH_ENABLE_MPS_FALLBACK`.

Build order, from the tree:

```bash
zsh gpu/m1/ffn/training/build_train_step_runner.sh     # the owner library build/libhffn_train.dylib
bash gpu/m1/llm/training/build_bridge.sh               # build/native_ffn_bridge.so
```

## Use

```python
from hffn_metal_llm import HFFNMetalAdamW, load_pythia_hffn_metal_for_training

model, modules = load_pythia_hffn_metal_for_training(
    base_checkpoint_dir, export_dir, native_n=(32,), library=owner_library)   # every block size N the loop will use
optimizer = HFFNMetalAdamW(model, modules, lr=1e-5, weight_decay=1e-2)
```

`native_n` lists every block size `N` of the loop; the native context of an `N` is prepared at its first call, one is kept
resident per layer and the one of the previous `N` is discarded. `HFFNMetalAdamW` runs the owner's compact AdamW on the FFN
pools, through one dedicated optimiser context per layer, and a PyTorch AdamW on every other parameter.
`save_training_checkpoint` and `load_training_checkpoint` save and resume weights, both optimiser states and the step counter.
`HFFNMetalMLP` is forward-only: its grad-enabled forward is refused, the training module is `NativeBackwardMLP`.

## Checks

```bash
HFFN_ASSETS=<assets> ./check.sh check <label> [export name ...]     # without names: every pythia*_sparse*_ffn_only export
```

Without timing; it needs the owner library and the bridge built. For each export it runs:

- **layer gate** (`validate_wanda_layer.py`, every layer file): the Torch forward and the native backward at the layouts
  `auto`, `token_major` and `feature_persistent` and `N` = 1, 8, 64, 128 and 256, forward and active gradients against the
  float64 oracle, gate 5e-4;
- **model gate** (`validate_pythia_parity.py`): one training step of the whole model against the fixed-support dense model,
  judged against a float64 reference of the same model and token ids by the shared training gates (loss, gradient groups,
  first AdamW step against its closed form), with two fault injections that must be detected.

It ends with `check: PASSED` or `check: FAILED` and writes `runs/check_<UTC>_<label>/`.

## Known limits

- **Bridging costs of the adapter (estimated counts, read from the code of `native_ffn_adapter.py` and `bridge.mm`, nothing
  counted at run time).** Per layer and per optimiser step, `K = 1`: 3 strided `copy_` of `N*d` floats (layout conversion of
  `x`, `gy`, `gx`); 1 blit of the four gradient views into `grad_flat` and 1 host memset of the native gradient pool; 2 output
  allocations (`grad_x`, `grad_flat`); 4 view fetches and about 9 data-pointer reads per call; 3 command buffers (staged
  forward, staged backward, blit); about 4 host synchronisations, one of them process-global; the staged forward recomputes
  `z` and `h` while the Torch forward still writes `z`; one autograd node per layer and call. The optimiser costs, per layer and
  step: 1 reverse blit of `flat.grad`, 1 `flat.grad.zero_()`, 3 native waits and 4 ctypes calls. A change of `N` costs, per layer,
  one discard (a synchronisation), three MPS staging allocations, one native context, one blit queue and one source
  preparation (a subprocess), counted by `lazy_prepares` and `contexts_discarded`.
- **Memory of the native contexts (estimated, hypothesis for the support copy).** Each (layer, `N`) backward context allocates
  about `6P` floats, `P = nnz_up + nnz_down + d_ff + d_model`, and uses about `1P` of them; the dedicated optimiser context of
  a layer is allocated the same way. At 410M d80 (`P` = 6 716 006) this is about 134 MB per layer. Each context probably also
  copies the support (hypothesis). The owner-tile partial buffer (from `N` >= 256, up to about 107 MB per layer) is used state.
- **Time and memory against the former Torch backward are not measured**: the move was judged on parity alone. The results
  give B against A for the cells run, with their band.
- **Owner limits:** the owner refuses `N` >= 8192 and does not cover a two-dimensional (2d) support; `accum_steps > 1` is not
  claimed here.
- **Pythia-410M at d20 and d80 are outside the timed grid** (decisions of 2026-11-09: the cells are swap-bound on the 16 GB
  Mac and the guard stops them); no figure exists for them in this version.
- **Torch 2.10.0 only**, because of the bridge's internal header; one Apple M1, passively cooled; `delta`, the noise band of
  each run, is applied as the root `RESULTS.md` states.
- **Depends on the shared levels:** copy the `runtime/` tree, not the leaf alone.
- No claim of convergence equivalence: the gates measure the distance to float64 of one step, not a trajectory.
- The Metal runtime keeps the vocabulary of a multi-bank design (`MoEFFN`, `e1b_hidden_banks_v1`); one bank and one route
  are used.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
