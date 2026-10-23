# HFFN GPU runtime on Apple M1: Pythia inference

Pythia (GPT-NeoX) inference with a fixed, exact WANDA support on the Apple M1 GPU. Only the MLPs are replaced by HFFN
Metal kernels; attention, normalisations, residuals, embeddings, the KV cache and the LM head are the dense PyTorch of the
model, on MPS. No post-attention fusion.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **exact offsets, no padding, persistent workspaces.** The down path is permuted once into output-major order: each active value is present exactly once, with its `int16` hidden owner, which removes the packed decoding and the irregular read of the row offsets from the hot loop. Layout chosen by `N` (below). No matrix unit, no simdgroup-matrix path, no fusion with attention. |
| Hardware | Apple M1 GPU through PyTorch MPS. Not tried on another Apple chip. |
| Models | Pythia 70M, 160M and 410M (float32), FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from the compact exports (18 exports, 252 layers). |
| Results | [RESULTS.md](RESULTS.md): Pythia-410M at d10, d20 and d80. |

Invariants: canonical format `hffn_training_compact_g4_v2`; no dense materialisation of the FFN weights and no dense
fallback; float32, exact-erf GELU and the Pythia biases; non-FFN weights loaded from `model.safetensors`; workspaces kept
alive until the MPS event; compatible with `torch.inference_mode()`, the KV cache and `generate()`.

Layout policy (`select_inference_layout`, a function of `N` = tokens of the call and of the model width): a cooperative
owner-row reduction up to `N` = 4 for the 70M model and `N` = 2 for the 160M and 410M, then `feature_persistent` up to
`N` = 63, then `feature_owner_tile` from `N` = 64. `token_major` exists as a forced layout for the checks.

## Contents

```
check.sh                                  layer gate of every layer of every export, every layout, without timing
hffn_metal_llm_inference/                 compact loader, runtime, shaders, model integration
smoke_pythia_inference.py                 prefill, KV-cache decode and greedy generation against the dense model
validate_wanda_inference_layer.py         one layer, every layout, against the shared float64 oracle
validate_pythia_inference_matrix.py       fresh-process full-model parity over the 18 configurations (`--model-root` is the folder of the base checkpoints, `--export-root` the folder of the exports)
```

The Metal FFN runtime is at the shared level [`native_transformer`](../native_transformer/README.md), the export reader and
the oracle in `common/pythia`: use and copy the `runtime/` tree.

## Requirements and assets

PyTorch with MPS, transformers, safetensors. `HFFN_ASSETS` for the base checkpoints and the exports (layout at the root of
the tree). `HFFN_PYTHON` names the interpreter of `check.sh` (default `python`).

## Use

```python
from hffn_metal_llm_inference import load_pythia_hffn_metal_for_inference
model, runtime, modules = load_pythia_hffn_metal_for_inference(base_checkpoint_dir, export_dir)
out = model.generate(input_ids, do_sample=False, max_new_tokens=16, use_cache=True)
```

## Checks

```bash
HFFN_ASSETS=<assets> ./check.sh check <label> [export name ...]     # without names: every pythia*_sparse*_ffn_only export
```

Without timing. For every layer file it runs `validate_wanda_inference_layer.py`: the four layouts (`token_major`,
`owner_row_reduce`, `feature_persistent`, `feature_owner_tile`) at `N` = 1, 8, 32, 128, 256 and 512 against the shared float64
oracle, maximum absolute error at most 5e-4, outputs finite. It ends with `check: PASSED` or `check: FAILED` and writes
`runs/check_<UTC>_<label>/`. It does not run a model; the model-level check is the smoke:

```bash
HFFN_ASSETS=<assets> python smoke_pythia_inference.py \
  --model-dir $HFFN_ASSETS/pythia/models/base/pythia70m \
  --export-dir $HFFN_ASSETS/pythia/exports/pythia70m_sparse20_ffn_only/export
```

It compares prefill logits and every KV-cache decode step with the dense fixed-support model (maximum difference at most
2e-2, same argmax), checks that `generate()` returns the same tokens, and that the workspace count does not grow during decode.

## Known limits

- **High density is the boundary of the format:** at d80 prefill is slower than dense (RESULTS.md).
- **Only Pythia-410M was timed**; the other models are covered by the checks, not by the tables.
- **Depends on the shared levels** (`native_transformer`, `common/pythia`): copy the `runtime/` tree, not the leaf alone.
- **One Apple M1**, passively cooled; `delta` is reported, not applied, on GPU entries.
- The Metal runtime keeps the vocabulary of a multi-bank design (`MoEFFN`, `e1b_hidden_banks_v1`); one bank and one route
  are used.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
