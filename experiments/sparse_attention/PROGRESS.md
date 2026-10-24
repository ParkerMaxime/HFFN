# sparse_attention cloud branch — progress

Updated after each step. Status: `not started` / `README committed` / `running` / `done` / `stopped` (with reason).
Run paths are relative to `experiments/sparse_attention/common/llm/training/`.

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| bootstrap | done | enwik8 extracted from the bundle and verified; torch 2.14.1 (PyPI build) on CPU | — | — |
| shared code + checks | done | `checks/check_model.py` 112/112 | 0 | — |
| E0 | done | `e0/runs/run_20261024T182902Z` (first run `run_20261024T180451Z` stopped after its sweep: optimum at the grid edge) | 0 | discriminating: Δ_attn 1.503 bpb, band 0.0062, ratio 243 (R_min 10); lr 0.032 frozen for every arm |
| E1 | done | `e1/runs/run_20261024T192919Z` | 0 | within bound (median d −0.0045 bpb, bound 0.075) |
| E2 | README committed | | | queued in the chain |
| E3a | README committed | | | queued in the chain |
| E3b | README committed | | | queued in the chain |
| IND | README committed | | | queued in the chain |
| E4a | README committed | | | queued in the chain |
| E4b | README committed | | | queued in the chain |
| E5 | not started | | | |
| SUMMARY.md | not started | | | |
