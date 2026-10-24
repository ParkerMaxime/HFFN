# E3b — the complete constraint (attention routed)

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E3a, **token `t` computes only the heads `H_g` of its group `g = t mod 4`** (queries, scores, weighted values of
2 heads of 8) and `o_t` lives on `S_g` only (exactly zero elsewhere, checked). Keys and values stay dense for every token
and every head. Same parameters as E3a (80,896): this step varies only the attention routing. Question: **the cost of
routing the attention at equal parameters**. Implementation: packed, checked equal to the selected form (outputs and
gradients). The saving (PLAN section 6, 52,736 against 124,928 multiply-adds per token) is analytical.

| Arm | Attention | FFN |
|---|---|---|
| `E3a` (control) | every head for every token, `W_O` blocks | routed |
| `E3b` | heads `H_g` only for token `t`, `o_t` on `S_g` | routed |

Expected (hypothesis): a cost, larger than the previous steps (the E0 sweep showed E3b 0.52 bpb above E0 at one seed and
2,000 steps — a sight, not a ladder result): on the query side a token sees through 2 heads of 8.

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); the training
  protocol, evaluation and mixing diagnostic of `../e0/README.md`. The control is re-run in this campaign (numbers of
  different campaigns are not compared) and its figures are compared with the previous step's as a reproducibility check.
- Statistic: `d_s = bpb(E3b, s) − bpb(E3a, s)`. **Bound `τ = 0.075` bits per byte** (5 % of `Δ_attn` = 1.503 measured in E0; the E0 seed band is 0.006).
  `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse). Within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive if the seeds straddle it; "better in every seed" if `max_s d_s < 0`. Band reported: the range of `d_s`. Also reported: bits per byte by position class, mixing diagnostic, parameter counts.
- Launch: `./run.sh check | smoke | run`; 2 arms × 4 seeds = 8 jobs, one stage; records as in E0.
