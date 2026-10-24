# E2 — results

README: `README.md` (frozen before the run). Control: E1, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e2/runs/run_20261024T200327Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E1 | 93,184 | 2.3936 / 2.3867 / 2.3869 / 2.3886 | 2.3878 | [2.3867–2.3936] | 2.412 / 2.396 / 2.378 / 2.365 |
| E2 | 93,184 | 2.6440 / 2.6472 / 2.6392 / 2.6395 | 2.6418 | [2.6392–2.6472] | 2.666 / 2.648 / 2.632 / 2.622 |

**E2 against E1** (bound 0.075, max_degradation): paired difference per seed s0 +0.2504, s1 +0.2605, s2 +0.2523, s3 +0.2509; median +0.2516, range [+0.2504, +0.2605], paired band 0.0101. **Verdict: beyond bound.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E1 | +2.0627 | +2.9121 | +2.3671 | +9.3105 |
| E2 | +2.3296 | +6.1329 | +0.3432 | +8.3025 |

Reproducibility: the control `E1` of this campaign reproduces the figures of the E1 step (2.3936 / 2.3867 / 2.3869 /
2.3886).

## Reading

- **Verdict against the bound fixed beforehand: beyond bound.** Routing the FFN on top-1 (token `t` uses 128 of 512
  neurons) costs about **0.25 bpb** (median +0.2516, all four seeds between +0.2504 and +0.2605, paired band 0.010),
  i.e. 17 % of the `Δ_attn` of E0, more than three times the bound of 0.075. The cost is stable across seeds.
- It is a cost at **equal parameters** (the three other groups' neurons are idle for that token): the active FFN
  compute per token is divided by 4 (analytical, PLAN section 6). This agrees in direction with the unfavourable
  top-1 evidence of `moe/` (PLAN section 9), which concerns another setting (hypothesis).
- The cost is the same in the four position classes (E1 and E2 by class: +0.255, +0.251, +0.253, +0.257): measured,
  every class is served by one group and loses about the same.
- The mixing diagnostic changes in the expected direction: after routing, zeroing `W_down` cross-block entries costs only
  +0.34 (against +2.37 in E1) — the FFN of E2 contributes less through its cross-block outputs (descriptive, one
  reading of why: a token has 128 neurons instead of 512; hypothesis).

## Limits

- The step does not separate "fewer neurons per token" from "a group sees only one position class": position routing
  ties the two (E4 varies the route). Four seeds; same environment limits as E0.
- The ladder continues: only E0 has a stop rule (CLOUD.md); later arms are compared with their own control, so the
  accumulated cost since E0 is read in SUMMARY.md, not here.
