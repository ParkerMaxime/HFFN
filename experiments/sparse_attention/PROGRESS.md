# sparse_attention cloud branch — progress

Updated after each step. Status: `not started` / `README committed` / `running` / `done` / `stopped` (with reason).
Run paths are relative to `experiments/sparse_attention/common/llm/training/`.

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| bootstrap | done | enwik8 extracted from the bundle and verified; torch 2.14.1 (PyPI build) on CPU | — | — |
| shared code + checks | done | `checks/check_model.py` 112/112 | 0 | — |
| E0 | done | `e0/runs/run_20261024T182902Z` (first run `run_20261024T180451Z` stopped after its sweep: optimum at the grid edge) | 0 | discriminating: Δ_attn 1.503 bpb, band 0.0062, ratio 243 (R_min 10); lr 0.032 frozen for every arm |
| E1 | done | `e1/runs/run_20261024T192919Z` | 0 | within bound (median d −0.0045 bpb, bound 0.075) |
| E2 | done | `e2/runs/run_20261024T200327Z` | 0 | beyond bound (median d +0.2516 bpb, bound 0.075) |
| E3a | done | `e3a/runs/run_20261024T203226Z` | 0 | beyond bound (median d +0.1160 bpb, bound 0.075) |
| E3b | done | `e3b/runs/run_20261024T205712Z` | 0 | beyond bound (median d +0.1131 bpb, bound 0.075); complete constraint +0.477 bpb above E0 |
| IND | done | `ind/runs/run_20261024T211955Z` | 0 | mixing not worth it at L=1: E3b better than IND by +0.0031 bpb median (bound 0.075) — not met |
| E4a | done | `e4a/runs/run_20261024T214033Z` | 0 | within bound, better than E3b in every seed (median d −0.2264 bpb; wide seed spread) |
| E4b | done | `e4b/runs/run_20261024T221415Z` | 0 | inconclusive (median d +0.0825, seeds from −0.022 to +0.167; bimodal) |
| E5 | done | `e5/runs/run_20261024T225110Z` | 0 | E4a_L4 vs REF4 beyond bound (+0.093); E4a_L4 vs E3b_L4 −0.290; rotation vs fixed within bound (≈ −0.003) |
| SUMMARY.md | done | `experiments/sparse_attention/SUMMARY.md` | — | all rows of the ladder run; see its "Not run" list |
