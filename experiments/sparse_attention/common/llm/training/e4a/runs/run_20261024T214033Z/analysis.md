# Analysis of e4a/runs/run_20261024T214033Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E3b | 80896 | 2.8692 | [2.8627-2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |
| E4a | 80896 | 2.6393 | [2.5520-2.6778] | 2.666 / 2.645 / 2.627 / 2.619 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 | |
| E4a | +2.4004 | +3.7724 | +6.5696 | +13.8121 | |

**E4a vs E3b** (max_degradation, bound 0.075): per-seed d = s0: -0.2474, s1: -0.1919, s2: -0.2053, s3: -0.3209; median -0.2264, range [-0.3209, -0.1919] → **within bound; candidate better than control in every seed**
