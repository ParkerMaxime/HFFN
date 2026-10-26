# Frontier analysis (run_20261025T220720Z, suffix '_N512', tau 0.075)

| arm | valid median [min–max] | test median [min–max] | inference MACs | training MACs | params | KV values/token |
|---|---|---|---|---|---|---|
| E4a_L2_N512 | 2.2889 [2.2824–2.3041] | 2.2679 [2.2603–2.2816] | 187,392 | 562,176 | 161,664 | 512 |
| IND_L4_N512 | 2.2372 [2.2366–2.2377] | 2.2156 [2.2139–2.2158] | 231,424 | 694,272 | 175,744 | 1,024 |
| REF1_N512 | 2.3498 [2.3478–2.3538] | 2.3276 [2.3239–2.3290] | 256,000 | 768,000 | 93,184 | 256 |
| E4a_L4_N512 | 2.1076 [2.0941–2.1224] | 2.0895 [2.0783–2.1036] | 342,016 | 1,026,048 | 323,200 | 1,024 |
| REF2_N512 | 2.1416 [2.1354–2.1438] | 2.1219 [2.1123–2.1220] | 479,232 | 1,437,696 | 186,240 | 512 |
| E4a_L6_N512 | 2.0330 [2.0231–2.0368] | 2.0108 [1.9995–2.0173] | 496,640 | 1,489,920 | 484,736 | 1,536 |
| REF3_N512 | 2.0139 [2.0101–2.0280] | 1.9871 [1.9865–1.9992] | 702,464 | 2,107,392 | 279,296 | 768 |
| REF4_N512 | 1.9601 [1.9281–1.9763] | 1.9365 [1.9005–1.9534] | 925,696 | 2,777,088 | 372,352 | 1,024 |

## Equal compute, dominance pairs (measured)

- E4a_L2 vs REF1: d = s0 -0.0609, s1 -0.0497, s2 -0.0653; median -0.0609 [-0.0653, -0.0497] → **within bound**; **dominates**
- E4a_L4 vs REF2: d = s0 -0.0498, s1 -0.0279, s2 -0.0192; median -0.0279 [-0.0498, -0.0192] → **within bound**; **dominates**
- E4a_L6 vs REF3: d = s0 +0.0088, s1 +0.0092, s2 +0.0229; median +0.0092 [+0.0088, +0.0229] → **within bound**

## Equal depth (measured)

- E4a_L2 vs REF2: d = s0 +0.1451, s1 +0.1687, s2 +0.1408; median +0.1451 [+0.1408, +0.1687] → **beyond bound**
- E4a_L4 vs REF4: d = s0 +0.1660, s1 +0.1313, s2 +0.1622; median +0.1622 [+0.1313, +0.1660] → **beyond bound**

## Mixing at depth (measured)

- IND_L4 vs E4a_L4: d = s0 +0.1432, s1 +0.1302, s2 +0.1142; median +0.1302 [+0.1142, +0.1432] → **beyond bound**

## Test split, equal-compute pairs (beside, selects nothing)

- E4a_L2 vs REF1: d = s0 -0.0672, s1 -0.0475, s2 -0.0560; median -0.0560 [-0.0672, -0.0475] → **within bound**; **dominates**
- E4a_L4 vs REF2: d = s0 -0.0437, s1 -0.0228, s2 -0.0183; median -0.0228 [-0.0437, -0.0183] → **within bound**; **dominates**
- E4a_L6 vs REF3: d = s0 +0.0181, s1 +0.0123, s2 +0.0243; median +0.0181 [+0.0123, +0.0243] → **within bound**

## Equal compute, interpolated in log C (estimated)

- E4a_L2: no value (compute outside the range of the dense family)
- E4a_L4 between REF1 and REF2 (weight 0.462 on the upper): d median -0.1453 [-0.1606, -0.1302] → **within bound**; **dominates**
- E4a_L6 between REF2 and REF3 (weight 0.093 on the upper): d median -0.0964 [-0.1010, -0.0962] → **within bound**; **dominates**

## Bits per byte by position range (median over seeds)

- E4a_L2_N512: 0-127: 2.3322, 128-255: 2.2737, 256-511: 2.2749
- IND_L4_N512: 0-127: 2.2880, 128-255: 2.2211, 256-511: 2.2200
- REF1_N512: 0-127: 2.3962, 128-255: 2.3349, 256-511: 2.3341
- E4a_L4_N512: 0-127: 2.1578, 128-255: 2.0908, 256-511: 2.0908
- REF2_N512: 0-127: 2.1989, 128-255: 2.1230, 256-511: 2.1215
- E4a_L6_N512: 0-127: 2.0879, 128-255: 2.0152, 256-511: 2.0144
- REF3_N512: 0-127: 2.1074, 128-255: 1.9872, 256-511: 1.9805
- REF4_N512: 0-127: 2.0550, 128-255: 1.9345, 256-511: 1.9255
