# Analysis of e5/runs/run_20261024T225110Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E4a_L4 | 323200 | 2.1252 | [2.1204-2.1347] | 2.152 / 2.132 / 2.116 / 2.101 |
| REF4 | 372352 | 2.0337 | [2.0275-2.0362] | 2.063 / 2.041 / 2.022 / 2.009 |
| E4a_L4rot | 323200 | 2.1237 | [2.1167-2.1293] | 2.151 / 2.130 / 2.115 / 2.100 |
| E3b_L4 | 323200 | 2.4159 | [2.4100-2.4202] | 2.441 / 2.423 / 2.404 / 2.397 |
| E3b_L4rot | 323200 | 2.4105 | [2.4079-2.4138] | 2.433 / 2.418 / 2.399 / 2.387 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E4a_L4 | +1.8492 | +3.1824 | +3.5632 | +6.2502 | |
| REF4 | +2.2646 | +2.9354 | +2.0477 | +5.1696 | |
| E4a_L4rot | +1.8671 | +2.9277 | +4.4339 | +6.7795 | |
| E3b_L4 | +2.4742 | +4.0481 | +4.3497 | +7.9479 | |
| E3b_L4rot | +2.4123 | +3.4374 | +5.8449 | +8.1834 | |

**E4a_L4 vs REF4** (max_degradation, bound 0.075): per-seed d = s0: +0.0986, s1: +0.0879, s2: +0.0859, s3: +0.1022; median +0.0932, range [+0.0859, +0.1022] → **beyond bound**

**E4a_L4rot vs E4a_L4** (max_degradation, bound 0.075): per-seed d = s0: -0.0054, s1: +0.0084, s2: -0.0021, s3: -0.0130; median -0.0038, range [-0.0130, +0.0084] → **within bound**

**E3b_L4 vs REF4** (max_degradation, bound 0.075): per-seed d = s0: +0.3807, s1: +0.3775, s2: +0.3801, s3: +0.3926; median +0.3804, range [+0.3775, +0.3926] → **beyond bound**

**E3b_L4rot vs E3b_L4** (max_degradation, bound 0.075): per-seed d = s0: -0.0050, s1: -0.0007, s2: -0.0011, s3: -0.0123; median -0.0031, range [-0.0123, -0.0007] → **within bound; candidate better than control in every seed**

**E4a_L4 vs E3b_L4** (max_degradation, bound 0.075): per-seed d = s0: -0.2821, s1: -0.2896, s2: -0.2942, s3: -0.2905; median -0.2900, range [-0.2942, -0.2821] → **within bound; candidate better than control in every seed**
