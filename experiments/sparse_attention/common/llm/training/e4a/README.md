# E4a — routing by a balanced byte hash

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E3b, the route of token `t` is no longer `t mod 4` but a **static function of the byte of the token**: the 256
byte values are assigned once to the four groups, greedily (heaviest byte first, to the lightest group) on the byte
counts of the training split, so that each group receives about a quarter of the training bytes (checked: 25 % ± 0.01).
Same parameters (80,896). The route is causal (it depends on the token's own byte). Implementation: the selected form
(every head and neuron computed, the other groups multiplied by zero) since the groups are no longer positional slices;
the selected and packed forms were checked equal on position routing. Question: **does a content-dependent static route
change quality against position routing**. The load per window varies with the text (not enforced).

| Arm | Route |
|---|---|
| `E3b` (control) | `t mod 4` |
| `E4a` | balanced byte hash |

Expected (hypothesis): no large change; hashing may help because a group sees a consistent set of bytes (the group of
the current byte) or hurt because the group of a token is unrelated to its position class.

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); the training
  protocol, evaluation and mixing diagnostic of `../e0/README.md`. The control is re-run in this campaign (numbers of
  different campaigns are not compared) and its figures are compared with the previous step's as a reproducibility check.
- Statistic: `d_s = bpb(E4a, s) − bpb(E3b, s)`. **Bound `τ = 0.075` bits per byte** (5 % of `Δ_attn` = 1.503 measured in E0; the E0 seed band is 0.006).
  `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse). Within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive if the seeds straddle it; "better in every seed" if `max_s d_s < 0`. Band reported: the range of `d_s`. Also reported: bits per byte by position class, mixing diagnostic, parameter counts.
- Launch: `./run.sh check | smoke | run`; 2 arms × 4 seeds = 8 jobs, one stage; records as in E0.
