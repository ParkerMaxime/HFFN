# E4b — learned router

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E3b, the route is **top-1 of a learned router** `ρ_t = W_r ξ_t` (`ξ_t` the input of the attention norm; 512
parameters; total 81,408), one router for the head group and the FFN group of the layer.
- Gate: the outputs of the chosen group (`o_t` after `W_O^g` and `y_t`) are scaled by `G · p` (`p` the softmax probability of
  the chosen group, `G = 4`), so that the forward magnitude is that of E3b at a uniform router and the router receives a
  gradient (PLAN section 8 leaves "scaled by its probability" open; this is the choice).
- Balance: Switch loss `G Σ_g f_g P_g` with coefficient **0.01**, added to the training loss and **reported separately**
  (`balance_loss` in `raw.jsonl`; it is 1 at perfect balance).
- Capacity: `ceil(1.25 · N / G) = 40` tokens per group per window; the tokens beyond the capacity (in order of position,
  hence causal) are **dropped**: no attention output, no FFN output, they pass through the residual; their fraction
  and the load per group are reported (`router_dropped`, `router_load`).
- `W_r` initialised at std 0.02. Implementation: selected form.
Question: **does a learned content route beat position routing** at equal parameters (+512). Under a causal mask a
learned router can unbalance the causal work across cores; only the load is reported.

| Arm | Route |
|---|---|
| `E3b` (control) | `t mod 4` |
| `E4b` | learned top-1 |

Expected (hypothesis): unstable or slightly worse at first (the router starts near random), possibly better late.

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); the training
  protocol, evaluation and mixing diagnostic of `../e0/README.md`. The control is re-run in this campaign (numbers of
  different campaigns are not compared) and its figures are compared with the previous step's as a reproducibility check.
- Statistic: `d_s = bpb(E4b, s) − bpb(E3b, s)`. **Bound `τ = 0.075` bits per byte** (5 % of `Δ_attn` = 1.503 measured in E0; the E0 seed band is 0.006).
  `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse). Within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive if the seeds straddle it; "better in every seed" if `max_s d_s < 0`. Band reported: the range of `d_s`. Also reported: bits per byte by position class, mixing diagnostic, parameter counts.
- Launch: `./run.sh check | smoke | run`; 2 arms × 4 seeds = 8 jobs, one stage; records as in E0.
