# sparse_attention cloud branch — progress

Updated after each step. Status: `not started` / `README committed` / `running` / `done` / `stopped` (with reason).
Run paths are relative to `experiments/sparse_attention/common/llm/training/`.

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| bootstrap | done | enwik8 extracted from the bundle and verified; torch 2.14.1 (PyPI build) on CPU | — | — |
| shared code + checks | done | `checks/check_model.py` 112/112 | 0 | — |
| E0 | done | `e0/runs/run_20261024T182902Z` (first run `run_20261024T180451Z` stopped after its sweep: optimum at the grid edge) | 0 | discriminating: Δ_attn 1.503 bpb, band 0.0062, ratio 243 (R_min 10); lr 0.032 frozen for every arm |
| E1 | not started | | | |
| E2 | not started | | | |
| E3a | not started | | | |
| E3b | not started | | | |
| IND | not started | | | |
| E4a | not started | | | |
| E4b | not started | | | |
| E5 | not started | | | |
| SUMMARY.md | not started | | | |
