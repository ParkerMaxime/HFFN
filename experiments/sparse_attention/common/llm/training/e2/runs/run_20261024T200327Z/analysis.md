# Analysis of e2/runs/run_20261024T200327Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E1 | 93184 | 2.3878 | [2.3867-2.3936] | 2.412 / 2.396 / 2.378 / 2.365 |
| E2 | 93184 | 2.6418 | [2.6392-2.6472] | 2.666 / 2.648 / 2.632 / 2.622 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E1 | +2.0627 | +2.9121 | +2.3671 | +9.3105 | |
| E2 | +2.3296 | +6.1329 | +0.3432 | +8.3025 | |

**E2 vs E1** (max_degradation, bound 0.075): per-seed d = s0: +0.2504, s1: +0.2605, s2: +0.2523, s3: +0.2509; median +0.2516, range [+0.2504, +0.2605] → **beyond bound**
