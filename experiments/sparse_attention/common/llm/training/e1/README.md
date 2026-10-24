# E1 — FFN inputs inside the feature blocks

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks: `../e0/README.md`
(only what differs is stated here). Plan: `current/sparse_attention/PLAN.md` section 8.

## Single mechanism and question

Move the support of the FFN so that the up-projection of each hidden neuron of group `g` (neurons `[128g, 128g+128)`)
reads only the coordinates of the block `S_g = [32g, 32g+32)`: `up_idx[r] ⊂ S_g`, **26 of 32 inputs** per neuron,
outputs `down_idx[r]` still random over all `M = 128` coordinates (26 per neuron), **no routing** (every neuron
processes every token), attention dense. Same parameters (93,184), same density in parameters, nothing saved: this
step isolates **the cost of aligning the hidden neurons on the feature blocks at equal density**.

## Arms and control

| Arm | Attention | FFN |
|---|---|---|
| `E0` (control, re-run in this campaign) | dense | random support, `k_up = k_down = 26` |
| `E1` | dense | `up_idx[r] ⊂ S_g` (26 of 32), `down_idx[r]` random (26 of 128), no routing |

## Fixed before the run

- Seeds 0, 1, 2, 3; 4,000 steps; learning rate **0.032** for both arms (frozen by E0: `../e0/RESULTS.md`); protocol of
  `../e0/README.md`. The control is re-run here because numbers of different campaigns are not compared; its figures
  are then compared with E0's as a reproducibility check (expected identical, same code and threads).
- Statistic: `d_s = bpb(E1, s) − bpb(E0, s)` per seed. **Bound `τ = 0.075` bits per byte** (5 % of
  `Δ_attn = 1.503` measured in E0; the E0 band of the seeds is 0.006, so the bound is reachable). Verdict:
  within bound if `max_s d_s ≤ 0.075`; beyond bound if `min_s d_s > 0.075`; inconclusive otherwise. Band reported:
  the range of `d_s`. Also reported: bpb by position class, mixing diagnostic.
- Expected (hypothesis): a small cost or none; the block alignment only restricts which of the 128 inputs a neuron reads
  (26 of 32 instead of 26 of 128).

## Launch

`./run.sh check | smoke | run`. Jobs: 2 arms × 4 seeds = 8, one stage. Records as in E0.
