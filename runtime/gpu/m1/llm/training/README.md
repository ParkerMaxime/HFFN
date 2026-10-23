# HFFN GPU runtime on Apple M1: Pythia training

Training of the Pythia models (70M, 160M, 410M) whose FFNs are pruned by WANDA with a fixed support, on the Apple M1 GPU.
Each FFN is a compact Metal module: its parameters (`up_val`, `down_val`, `b_up`, `b_down`) live in one flat pool, and its
gradients and AdamW states hold the active weights only. Attention, embeddings, norms, the head and their optimiser are the
dense PyTorch of the model, on MPS. There is no dense `W_up`, `W_down`, mask, gradient or moment in the HFFN path.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **forward: the Metal FFN kernel of the M1 GPU shared level** ([`native_transformer`](../inference/native_transformer/README.md)), through its one-bank entry that stores `z` for the backward. **Backward:** exact-offset Metal kernels in three layouts chosen by `N` = batch x sequence length: `token_major` for `N` <= 6, `feature_persistent` from 9 to 255, `feature_owner_tile` from 256; for `N` = 7 or 8 the compact degree decides (threshold 64). Compact AdamW as a Metal kernel. One block of tokens per call, no matrix unit, no fusion with attention. |
| Hardware | Apple M1 GPU through PyTorch MPS, 16 GB. Not tried on another Apple chip. |
| Models | Pythia 70M, 160M and 410M (float32), FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density (18 exports). |
| Results | [RESULTS.md](RESULTS.md): Pythia-410M at d10, d20 and d80; **the d80 cell is `OOM` on the HFFN arm** (see Known limits). |

## Contents

```
check.sh                       layer and one-step model parity on every export, without timing
hffn_metal_llm/                compact layer, Metal shaders, runtime, model integration, optimizer, checkpoint, loader
smoke_mps_interop.py           PyTorch -> custom Metal -> PyTorch with no explicit synchronisation
smoke_pythia_training.py       one complete training step of Pythia, FFNs on Metal and the rest on MPS
validate_wanda_layer.py        one layer, forward and active gradients, against the shared float64 oracle
validate_compact_adamw.py      ten steps of the compact AdamW shader against its formula
validate_pythia_parity.py      one-step full-model gate against float64 and the dense float32 control
validate_optimizer_resume.py   atomic save and resume of weights, PyTorch AdamW, compact moments and step counter
validate_configuration_matrix.py   every export, the layout selected at each N
test_layout_policy.py          CPU-only check of the layout policy
```

The forward kernel is at the shared level `gpu/m1/llm/inference/native_transformer`, the export reader, the oracle and the
training gates in `common/pythia` and `cpu/common/llm/training/checks`: use and copy the `runtime/` tree.

## Requirements and assets

PyTorch with MPS, transformers, safetensors. `HFFN_ASSETS` for the base checkpoints and the exports (layout at the root of
the tree). `HFFN_PYTHON` names the interpreter of `check.sh` (default `python`). The scripts refuse
`PYTORCH_ENABLE_MPS_FALLBACK`.

## Use

```python
from hffn_metal_llm import HFFNMetalAdamW, load_pythia_hffn_metal_for_training

model, modules = load_pythia_hffn_metal_for_training(base_checkpoint_dir, export_dir)
optimizer = HFFNMetalAdamW(model, modules, lr=1e-5, weight_decay=1e-2)
```

`HFFNMetalAdamW` runs the compact Metal AdamW on the FFN pools and a PyTorch AdamW on every other parameter.
`save_training_checkpoint` and `load_training_checkpoint` save and resume weights, both optimiser states and the step counter.

## Checks

```bash
HFFN_ASSETS=<assets> ./check.sh check <label> [export name ...]     # without names: every pythia*_sparse*_ffn_only export
```

Without timing. For each export it runs:

- **layer gate** (`validate_wanda_layer.py`, every layer file): the layouts `token_major`, `feature_persistent` and
  `feature_owner_tile` at `N` = 1, 8, 64, 128 and 256, forward and active gradients against the float64 oracle, gate 5e-4;
- **model gate** (`validate_pythia_parity.py`): one training step of the whole model against the fixed-support dense model,
  judged against a float64 reference of the same model and token ids by the shared training gates (loss, gradient groups,
  first AdamW step against its closed form), with two fault injections that must be detected.

It ends with `check: PASSED` or `check: FAILED` and writes `runs/check_<UTC>_<label>/`.

## Known limits

- **Pythia-410M at d80 is `OOM` on the HFFN arm** on the 16 GB Mac: the cell is swap-bound and no figure exists. The memory
  fix is not done in this version.
- **Depends on the shared levels** (`native_transformer`, `common/pythia`, `cpu/common/llm/training/checks`): copy the
  `runtime/` tree, not the leaf alone.
- **One Apple M1**, passively cooled; `delta` is reported, not applied, on GPU entries.
- No claim of convergence equivalence: the gates measure the distance to float64 of one step, not a trajectory.
- The Metal runtime keeps the vocabulary of a multi-bank design (`MoEFFN`, `e1b_hidden_banks_v1`); one bank and one route
  are used.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
