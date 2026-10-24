# E2 — top-1 routing of the FFN

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E1 (neurons of group `g` read `S_g`, dense attention), **token `t` uses only the neurons `R_g` of its group
`g = t mod 4`**; the three other groups contribute nothing for that token. Parameters unchanged (93,184); the active FFN
compute per token is divided by 4 (counted analytically, PLAN section 6; no time measured). Question: **the cost of
routing the FFN**. Implementation: packed (the tokens of a group are gathered), checked equal to the selected form.

| Arm | Attention | FFN |
|---|---|---|
| `E1` (control) | dense | `up_idx ⊂ S_g`, no routing |
| `E2` | dense | same supports and values, token `t` uses `R_{t mod 4}` only |

Expected (hypothesis): a noticeable cost, since a token loses three quarters of the hidden neurons (the `moe/` evidence
of PLAN section 9 is unfavourable to top-1). Position classes: the four classes are each served by one group.

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); the training
  protocol, evaluation and mixing diagnostic of `../e0/README.md`. The control is re-run in this campaign (numbers of
  different campaigns are not compared) and its figures are compared with the previous step's as a reproducibility check.
- Statistic: `d_s = bpb(E2, s) − bpb(E1, s)`. **Bound `τ = 0.075` bits per byte** (5 % of `Δ_attn` = 1.503 measured in E0; the E0 seed band is 0.006).
  `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse). Within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive if the seeds straddle it; "better in every seed" if `max_s d_s < 0`. Band reported: the range of `d_s`. Also reported: bits per byte by position class, mixing diagnostic, parameter counts.
- Launch: `./run.sh check | smoke | run`; 2 arms × 4 seeds = 8 jobs, one stage; records as in E0.
