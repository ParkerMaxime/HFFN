# Frontier analysis (run_20261025T163405Z, suffix '', tau 0.075)

| arm | valid median [min–max] | test median [min–max] | inference MACs | training MACs | params | KV values/token |
|---|---|---|---|---|---|---|
| IND_L4 | 2.2391 [2.2391–2.2440] | 2.2204 [2.2165–2.2210] | 133,120 | 399,360 | 175,744 | 1,024 |
| E4a_L2 | 2.2928 [2.2847–2.3005] | 2.2702 [2.2616–2.2780] | 138,240 | 414,720 | 161,664 | 512 |
| REF1 | 2.3672 [2.3632–2.3699] | 2.3449 [2.3370–2.3477] | 157,696 | 473,088 | 93,184 | 256 |
| E4a_L4 | 2.1161 [2.1105–2.1289] | 2.1001 [2.0926–2.1046] | 243,712 | 731,136 | 323,200 | 1,024 |
| REF2 | 2.1475 [2.1440–2.1513] | 2.1277 [2.1248–2.1305] | 282,624 | 847,872 | 186,240 | 512 |
| E4a_L6 | 2.0398 [2.0391–2.0438] | 2.0206 [2.0201–2.0267] | 349,184 | 1,047,552 | 484,736 | 1,536 |
| REF3 | 2.0721 [2.0687–2.0733] | 2.0504 [2.0473–2.0530] | 407,552 | 1,222,656 | 279,296 | 768 |
| REF4 | 2.0265 [2.0215–2.0296] | 2.0055 [1.9975–2.0120] | 532,480 | 1,597,440 | 372,352 | 1,024 |

## Equal compute, dominance pairs (measured)

- E4a_L2 vs REF1: d = s0 -0.0703, s1 -0.0667, s2 -0.0851; median -0.0703 [-0.0851, -0.0667] → **within bound**; **dominates**
- E4a_L4 vs REF2: d = s0 -0.0186, s1 -0.0279, s2 -0.0407; median -0.0279 [-0.0407, -0.0186] → **within bound**; **dominates**
- E4a_L6 vs REF3: d = s0 -0.0335, s1 -0.0249, s2 -0.0330; median -0.0330 [-0.0335, -0.0249] → **within bound**; **dominates**

## Equal depth (measured)

- E4a_L2 vs REF2: d = s0 +0.1454, s1 +0.1565, s2 +0.1335; median +0.1454 [+0.1335, +0.1565] → **beyond bound**
- E4a_L4 vs REF4: d = s0 +0.0993, s1 +0.0896, s2 +0.0891; median +0.0896 [+0.0891, +0.0993] → **beyond bound**

## Mixing at depth (measured)

- IND_L4 vs E4a_L4: d = s0 +0.1152, s1 +0.1230, s2 +0.1285; median +0.1230 [+0.1152, +0.1285] → **beyond bound**

## Test split, equal-compute pairs (beside, selects nothing)

- E4a_L2 vs REF1: d = s0 -0.0668, s1 -0.0670, s2 -0.0861; median -0.0670 [-0.0861, -0.0668] → **within bound**; **dominates**
- E4a_L4 vs REF2: d = s0 -0.0202, s1 -0.0276, s2 -0.0379; median -0.0276 [-0.0379, -0.0202] → **within bound**; **dominates**
- E4a_L6 vs REF3: d = s0 -0.0273, s1 -0.0238, s2 -0.0324; median -0.0273 [-0.0324, -0.0238] → **within bound**; **dominates**

## Equal compute, interpolated in log C (estimated)

- E4a_L2: no value (compute outside the range of the dense family)
- E4a_L4 between REF1 and REF2 (weight 0.746 on the upper): d median -0.0846 [-0.0962, -0.0734] → **within bound**; **dominates**
- E4a_L6 between REF2 and REF3 (weight 0.578 on the upper): d median -0.0648 [-0.0665, -0.0567] → **within bound**; **dominates**
