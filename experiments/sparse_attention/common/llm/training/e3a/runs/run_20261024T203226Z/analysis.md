# Analysis of e3a/runs/run_20261024T203226Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E2 | 93184 | 2.6418 | [2.6392-2.6472] | 2.666 / 2.648 / 2.632 / 2.622 |
| E3a | 80896 | 2.7572 | [2.7546-2.7606] | 2.786 / 2.767 / 2.743 / 2.734 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E2 | +2.3296 | +6.1329 | +0.3432 | +8.3025 | |
| E3a | +2.4434 | +8.5730 | +0.0506 | +11.2579 | |

**E3a vs E2** (max_degradation, bound 0.075): per-seed d = s0: +0.1167, s1: +0.1104, s2: +0.1154, s3: +0.1173; median +0.1160, range [+0.1104, +0.1173] → **beyond bound**
