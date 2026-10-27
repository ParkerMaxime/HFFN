# Width frontier analysis (run_20261027T040908Z, tau 0.075)

| arm | valid median [min–max] | test median [min–max] | inference MACs | training MACs | params | KV values/token | train. act. values/token |
|---|---|---|---|---|---|---|---|
| E4a_L2_W256 | 2.1925 [2.1721–2.2034] | 2.1643 [2.1470–2.1809] | 552,960 | 1,658,880 | 642,816 | 1,024 | 3,328 |
| REF1_W256 | 2.1821 [2.1815–2.1834] | 2.1605 [2.1603–2.1620] | 696,320 | 2,088,960 | 370,688 | 512 | 6,656 |
| E4a_L4_W256 | 1.9867 [1.9839–2.0045] | 1.9638 [1.9615–1.9821] | 1,040,384 | 3,121,152 | 1,285,376 | 2,048 | 6,656 |
| E4a_L4_W256_h16 | 1.9435 [1.9422–1.9487] | 1.9247 [1.9234–1.9292] | 1,040,384 | 3,121,152 | 1,285,376 | 2,048 | 10,752 |
| REF2_W256 | 1.9880 [1.9853–1.9971] | 1.9664 [1.9644–1.9780] | 1,327,104 | 3,981,312 | 741,120 | 1,024 | 13,312 |
| REF2_W256_h16 | 1.9855 [1.9815–1.9857] | 1.9658 [1.9611–1.9688] | 1,327,104 | 3,981,312 | 741,120 | 1,024 | 21,504 |

## Equal counted multiply-adds, width (measured)

- E4a_L2_W256 vs REF1_W256 (counted MACs saved 20.6%): d = s0 -0.0094, s1 +0.0091, s2 +0.0213; median +0.0091 [-0.0094, +0.0213] → **within bound**
- E4a_L4_W256 vs REF2_W256 (counted MACs saved 21.6%): d = s0 -0.0133, s1 +0.0014, s2 +0.0165; median +0.0014 [-0.0133, +0.0165] → **within bound**

## Equal depth, width (measured)

- E4a_L2_W256 vs REF2_W256 (counted MACs saved 58.3%): d = s0 +0.1750, s1 +0.2072, s2 +0.2153; median +0.2072 [+0.1750, +0.2153] → **beyond bound**

## Equal counted multiply-adds, four heads per group (measured)

- E4a_L4_W256_h16 vs REF2_W256_h16 (counted MACs saved 21.6%): d = s0 -0.0420, s1 -0.0328, s2 -0.0435; median -0.0420 [-0.0435, -0.0328] → **within bound**; **dominates**

## Test split, same pairs (beside, selects nothing)

- E4a_L2_W256 vs REF1_W256 (counted MACs saved 20.6%): d = s0 -0.0135, s1 +0.0022, s2 +0.0206; median +0.0022 [-0.0135, +0.0206] → **within bound**
- E4a_L4_W256 vs REF2_W256 (counted MACs saved 21.6%): d = s0 -0.0165, s1 -0.0026, s2 +0.0177; median -0.0026 [-0.0165, +0.0177] → **within bound**
- E4a_L4_W256_h16 vs REF2_W256_h16 (counted MACs saved 21.6%): d = s0 -0.0440, s1 -0.0318, s2 -0.0424; median -0.0424 [-0.0440, -0.0318] → **within bound**; **dominates**

## Heads per group (measured)

i_s = d_s(E4a_L4_W256_h16 − REF2_W256_h16) − d_s(E4a_L4_W256 − REF2_W256)

- seed 0: d(h8 pair) -0.0133, d(h16 pair) -0.0420, i = -0.0287, |i|/tau = 0.38
- seed 1: d(h8 pair) +0.0014, d(h16 pair) -0.0328, i = -0.0342, |i|/tau = 0.46
- seed 2: d(h8 pair) +0.0165, d(h16 pair) -0.0435, i = -0.0599, |i|/tau = 0.80
- median -0.0342 [-0.0599, -0.0287] → **favours four heads per group**

## Bits per byte by position range (median over seeds)

- E4a_L2_W256: 0-127: 2.2366, 128-255: 2.1780, 256-511: 2.1777
- REF1_W256: 0-127: 2.2351, 128-255: 2.1653, 256-511: 2.1642
- E4a_L4_W256: 0-127: 2.0416, 128-255: 1.9680, 256-511: 1.9685
- E4a_L4_W256_h16: 0-127: 2.0034, 128-255: 1.9250, 256-511: 1.9238
- REF2_W256: 0-127: 2.0509, 128-255: 1.9677, 256-511: 1.9668
- REF2_W256_h16: 0-127: 2.0491, 128-255: 1.9652, 256-511: 1.9635

## Measured memory of E10 for the same arms (bytes per token, fp32, one step of 8,192 bytes)

| arm | saved for backward, token side | peak of a training step | peak of a prefill |
|---|---|---|---|
| E4a_L2_W256 | 37,574 | 40,210 | 10,332 |
| REF1_W256 | 23,612 | 25,928 | 13,312 |
| E4a_L4_W256 | 70,999 | 74,211 | 10,332 |
| E4a_L4_W256_h16 | 70,695 | 73,907 | 10,332 |
| REF2_W256 | 43,108 | 45,680 | 13,312 |
| REF2_W256_h16 | 43,172 | 45,744 | 13,312 |
