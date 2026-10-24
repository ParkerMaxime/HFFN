# Analysis of e4b/runs/run_20261024T221415Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E3b | 80896 | 2.8692 | [2.8627-2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |
| E4b | 81408 | 2.9503 | [2.8474-3.0360] | 2.976 / 2.949 / 2.939 / 2.938 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 | |
| E4b | +1.1445 | +1.8779 | +0.9432 | +2.0688 | |

**E4b vs E3b** (max_degradation, bound 0.075): per-seed d = s0: +0.1603, s1: -0.0223, s2: +0.1673, s3: +0.0047; median +0.0825, range [-0.0223, +0.1673] → **inconclusive (seeds straddle the bound)**
