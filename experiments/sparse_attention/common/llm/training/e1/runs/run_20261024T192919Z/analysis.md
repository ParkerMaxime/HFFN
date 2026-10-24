# Analysis of e1/runs/run_20261024T192919Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E0 | 93184 | 2.3925 | [2.3913-2.3975] | 2.417 / 2.401 / 2.383 / 2.370 |
| E1 | 93184 | 2.3878 | [2.3867-2.3936] | 2.412 / 2.396 / 2.378 / 2.365 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E0 | +1.9876 | +3.2070 | +2.9099 | +13.4634 | |
| E1 | +2.0627 | +2.9121 | +2.3671 | +9.3105 | |

**E1 vs E0** (max_degradation, bound 0.075): per-seed d = s0: -0.0000, s1: -0.0046, s2: -0.0045, s3: -0.0088; median -0.0045, range [-0.0088, -0.0000] → **within bound; candidate better than control in every seed**
