# sparse_attention cloud branch — progress

Updated after each step. Status: `not started` / `README committed` / `running` / `done` / `stopped` (with reason).
Run paths are relative to `experiments/sparse_attention/common/llm/training/`.

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| bootstrap | done | enwik8 extracted from the bundle and verified; torch 2.14.1 (PyPI build) on CPU | — | — |
| shared code + checks | done | `checks/check_model.py` 112/112 at the start, 121/121 after the E5 hash-rotation checks were added | 0 | — |
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

## Second series (E6-E9; plan: PLAN.md section 14; brief: CLOUD_E6_E9.md)

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| prep | done | `prep/runs/run_20261025T093927Z` | 0 | 44 new checks + 179 old pass; branch = uninterrupted run bit for bit; re-estimate of the total ≈ 18.5 h (< 24 h) |
| E6 | done | `e6/runs/run_20261025T095209Z` | 0 | H* = 4,000 (deviations at 2,000: 0.029/0.029 fail; at 4,000: 0.014/0.011 pass); E4a_L4 below REF2 in all six pairs |
| E7 | done | `e7/runs/run_20261025T120721Z` | 0 | rates: REF1 0.016, REF4 0.008, E4a_L2 0.008, E4a_L6 0.008 (one extension each, none at an edge) |
| E8 | done | `e8/runs/run_20261025T163405Z` | 0 | E4a dominates the dense model at 12-14 % more compute in 3/3 pairs and 3/3 seeds (valid and test); beyond bound at equal depth (+0.145, +0.090); IND_L4 worse than E4a_L4 by 0.123 |
| E9 | done | `e9/runs/run_20261025T220720Z` | 0 | N=512: E4a_L2 and E4a_L4 dominate REF1 and REF2; E4a_L6 equivalent within tau to REF3 at -29 % compute (worse in 3/3 seeds); equal-depth loss beyond; IND_L4 worse than E4a_L4 |
| SUMMARY_E6_E9.md | done | `experiments/sparse_attention/SUMMARY_E6_E9.md` | — | series complete: prep, E6, E7, E8, E9 |

## Third series (E10-E13; plan: PLAN.md section 15; brief: CLOUD_E10_E13.md)

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| prep3 | done | `prep3/runs/run_20261026T205721Z` | 0 | 203 + 44 + 34 checks pass; gather = selected; gate 0.310 passed; dense control = executor; total re-estimate ≈ 12.6 h |
| E10 | done | `e10/runs/run_20261026T211323Z` | 0 | counted activation saving (0.50 / 0.25) not reached: measured token-side 1.6-1.9x dense at equal counted MACs, 0.89-1.00 at equal depth; prefill peak 0.78; packed/selected 0.74-0.83 |
| E11 | done | `e11/runs/run_20261026T214914Z` (first run `run_20261026T212801Z` interrupted by a container restart) | 0 | delta 0.0975; layer cells faster (r 0.40-0.60); b cells slower (1.29, 1.13); c1 faster (0.85), c2 not resolved |
| E12 | done | `e12/runs/run_20261026T222859Z` | 0 | rates 0.004 (`REF2_W256`, one extension) and 0.004 (`E4a_L4_W256`); none at an edge |
| E13 | done | `e13/runs/run_20261027T040908Z` | 0 | width 256: within τ at equal counted MACs with 2 heads/group (not dominating); with 4 heads/group dominates (-0.042); equal depth beyond (+0.21); i_s favours four heads/group |
| SUMMARY_E10_E13.md | done | | | summary of the third series |

## Fourth step (E14; plan: PLAN.md section 17; brief: CLOUD_E14.md)

| Step | Status | Run | Exit | Verdict / what remains |
|---|---|---|---|---|
| prep4 | done | `prep4/runs/smoke_20261027T201311Z` | 0 | 219 + 44 + 34 + 82 checks pass; E2h packed = selected; decoded step = full forward (26 arms); gate 0.377 passed; SDPA = flash CPU kernel in all arms; total re-estimate ≈ 3.1 h, no cut |
| E14 prefill | not started | | | |
| E14 decoding | not started | | | |
| E14 memory | not started | | | |
| E14 analysis / RESULTS.md | not started | | | |
| SUMMARY_E14.md | not started | | | |
