# E3a — block-diagonal output projection

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E2, **`W_O` is restricted to its four `32 × 32` blocks**: head group `H_g` (heads `2g, 2g+1`) writes only `S_g`.
Every head is still computed for every token, attention is dense for every token; the FFN is as E2. Parameters:
**80,896** (−12,288 against E2; the loss of parameters is part of this mechanism, the price of the block-diagonal
projection). Question: **the price of the block-diagonal `W_O`** alone.

| Arm | Attention | FFN |
|---|---|---|
| `E2` (control) | dense, `W_O` full | routed |
| `E3a` | dense, `W_O` block-diagonal, every head for every token | routed |

Expected (hypothesis): a small cost. If the optimum learning rate of E0 for the arms with block-diagonal `W_O` had
differed from the dense one, this step would report both rates; E0 found one common rate, so only 0.032 is used.

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); the training
  protocol, evaluation and mixing diagnostic of `../e0/README.md`. The control is re-run in this campaign (numbers of
  different campaigns are not compared) and its figures are compared with the previous step's as a reproducibility check.
- Statistic: `d_s = bpb(E3a, s) − bpb(E2, s)`. **Bound `τ = 0.075` bits per byte** (5 % of `Δ_attn` = 1.503 measured in E0; the E0 seed band is 0.006).
  `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse). Within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive if the seeds straddle it; "better in every seed" if `max_s d_s < 0`. Band reported: the range of `d_s`. Also reported: bits per byte by position class, mixing diagnostic, parameter counts.
- Launch: `./run.sh check | smoke | run`; 2 arms × 4 seeds = 8 jobs, one stage; records as in E0.
