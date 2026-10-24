# IND — degenerate lower bound (four separated models)

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md` (only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Starting from E3b, the rows of `W_Q`, `W_K`, `W_V` of the heads `H_g` read **only `S_g`** (block-diagonal) and `down_idx[r]` of
each neuron of `R_g` lies **inside `S_g`** (26 of 32); everything else as E3b. The cross-block Jacobian is zero except
through the scalar of the RMS norm and the tied output (PLAN section 5). Parameters: **44,032** against 80,896 for E3b —
IND is a lower-bound control, not a parameter-matched arm, and a gap to E3b mixes the effect of the mixing paths with
the 36,864 fewer parameters (an unattributed limit, stated before the run). Question: **is the mixing of E3b worth
anything over four separated models?**

| Arm | Attention | FFN |
|---|---|---|
| `E3b` (control) | `W_Q`, `W_K`, `W_V` read all `M` | `down_idx` over all `M` |
| `IND` | `W_Q`, `W_K`, `W_V` of `H_g` read `S_g` only | `down_idx ⊂ S_g` |

The direction is reversed: **`d_s = bpb(IND, s) − bpb(E3b, s)`, positive = E3b better** (in `step.json` the comparison is written candidate `E3b`, control `IND`, direction `min_improvement`).
The mixing is **worth it** if `min_s d_s > 0.075` (the proposal beats the degenerate model beyond the bound); **not
worth it** if `max_s d_s ≤ 0.075`; inconclusive otherwise. Bound `τ = 0.075` bpb as in E1. Seeds, steps, rate (0.032),
protocol, reported quantities: as the other steps. The mixing diagnostic of IND must be exactly 0 (checked).
Launch: `./run.sh check | smoke | run`; 8 jobs.
