# Analysis of e3b/runs/run_20261024T205712Z

| arm | params | median bpb | [min-max] | by class t mod 4 (median) |
|---|---|---|---|---|
| E3a | 80896 | 2.7572 | [2.7546-2.7606] | 2.786 / 2.767 / 2.743 / 2.734 |
| E3b | 80896 | 2.8692 | [2.8627-2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |

| arm | mixing: Δ(Q) | Δ(K,V) | Δ(W_down) | Δ(all) | (median Δ bpb on 4096 valid windows) |
|---|---|---|---|---|---|
| E3a | +2.4434 | +8.5730 | +0.0506 | +11.2579 | |
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 | |

**E3b vs E3a** (max_degradation, bound 0.075): per-seed d = s0: +0.1021, s1: +0.1121, s2: +0.1140, s3: +0.1160; median +0.1131, range [+0.1021, +0.1160] → **beyond bound**
