# Shared Pythia level

What every Pythia leaf shares, whatever the hardware: the export reader, the two-arm loader, the end-to-end token check, the
training gates, the request-level timing harness, the prompt tokens and the source verification. Imported as `hffn_pythia`
(a leaf puts `common/pythia` on `sys.path` and gives the loader the constructor of its HFFN runtime).

| File | Role |
|---|---|
| `hffn_pythia/export.py` | reader of the `hffn_training_compact_g4_v2` exports (one file per FFN layer, hidden-major, exact offsets) |
| `hffn_pythia/model.py` | `load_pythia`: builds the dense arm and the HFFN arm from the same configuration, the same non-FFN tensors of the base checkpoint and the same FFN values of the export. Only the FFN call differs. |
| `hffn_pythia/dense_control.py` | the dense fixed-support control arm (stock FFNs holding the active values, and the AdamW that keeps them on the support); checks and benchmarks only |
| `hffn_pythia/oracle.py` | the dense float64 oracle of one fixed-support FFN, for the checks only |
| `hffn_pythia/parity_model.py` | end-to-end token check of the HFFN arm against the dense float32 arm and, as referee, the dense float64 arm, on text, with the tie band set by the dense control itself and a negative control |
| `hffn_pythia/training_gate.py` | model-level training gate: distance of an evaluation to a float64 reference (loss, gradient groups), and the verdict |
| `hffn_pythia/bench_request.py` | the request-level timing harness a leaf drives through its adapter (diagnostic tooling; the published figures come from a bench that is not distributed) |
| `hffn_pythia/assets.py` | where the assets are: `HFFN_ASSETS` (no default) |
| `hffn_pythia/verify_sources.py` | `python -B -m hffn_pythia.verify_sources <leaf>` checks every file of `<leaf>/sources/manifest.json` against its SHA-256 |
| `tools/make_prompt_tokens.py` | generates `hffn_pythia/wikitext2_test_tokens.json` (4,096 token ids of the WikiText-2 raw v1 test split, CC BY-SA), which is not distributed; see the README at the root of the tree |

Dense models exist in this level only as controls and oracles of the checks; the HFFN runtime never calls them.
