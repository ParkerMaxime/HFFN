# E14 analysis (frontier of quality against measured time, pairs, memory)

Gaps are **estimated** (a time and a bits-per-byte combined by interpolation); direct pairs, ratios and memory are **measured**.

## prefill (delta of the run = 0.0623)

### prefill, campaign n128: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2 | 0.729 | on the frontier | +0.0216 [+0.0080, +0.0284] | +0.0705, +0.0790, +0.0576 | -0.0274, -0.0222, -0.0416 |
| E4a_L4 | 1.240 | on the frontier | +0.0147 [+0.0040, +0.0234] | +0.0477, +0.0394, +0.0300 | -0.0010, -0.0100, -0.0219 |
| E4a_L6 | 1.788 | outside the range | pair against REF4 (t 1.855): d median +0.0173 [+0.0102, +0.0177], within bound; time ratio 0.964 | | |
| IND_L4 (control) | 1.096 | on the frontier | +0.1132 [+0.1068, +0.1144] | +0.1387, +0.1379, +0.1328 | +0.0836, +0.0816, +0.0746 |

Direct pairs against the anchor (measured): REF1: t 0.577, d median +0.2186; REF3: t 1.463, d median -0.0753; REF4: t 1.855, d median -0.1179; E4a_L2: t 0.729, d median +0.1454; E4a_L4: t 1.240, d median -0.0279; E4a_L6: t 1.788, d median -0.1076; IND_L4: t 1.096, d median +0.0951

### prefill, campaign n512: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2_N512 | 0.655 | on the frontier | -0.0075 [-0.0119, +0.0068] | +0.0374, +0.0545, +0.0331 | -0.0525, -0.0408, -0.0569 |
| E4a_L4_N512 | 1.161 | on the frontier | +0.0193 [-0.0048, +0.0318] | +0.0327, +0.0586, +0.0744 | -0.0423, -0.0201, -0.0108 |
| E4a_L6_N512 | 1.721 | outside the range | pair against REF4_N512 (t 1.827): d median +0.0728 [+0.0469, +0.1088], inconclusive (seeds straddle the bound); time ratio 0.942 | | |
| IND_L4_N512 (control) | 1.065 | on the frontier | +0.1164 [+0.1123, +0.1222] | +0.1498, +0.1615, +0.1590 | +0.0711, +0.0787, +0.0726 |

Direct pairs against the anchor (measured): REF1_N512: t 0.565, d median +0.2061; REF3_N512: t 1.470, d median -0.1215; REF4_N512: t 1.827, d median -0.1815; E4a_L2_N512: t 0.655, d median +0.1451; E4a_L4_N512: t 1.161, d median -0.0279; E4a_L6_N512: t 1.721, d median -0.1086; IND_L4_N512: t 1.065, d median +0.0949

### prefill, campaign w256: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2_W256 | 0.559 | outside the range | pair against REF1_W256 (t 0.562): d median +0.0091 [-0.0094, +0.0213], within bound; time ratio 0.995 | | |
| E4a_L4_W256 | 0.987 | outside the range | pair against REF2_W256 (t 1.000): d median +0.0014 [-0.0133, +0.0165], within bound; time ratio 0.987 | | |
| E4a_L4_W256_h16 | 1.032 | outside the range | pair against REF2_W256 (t 1.000): d median -0.0458 [-0.0536, -0.0366], within bound; time ratio 1.032 | | |

Direct pairs against the anchor (measured): REF1_W256: t 0.562, d median +0.1940; REF2_W256_h16: t 1.104, d median -0.0038; E4a_L2_W256: t 0.559, d median +0.2072; E4a_L4_W256: t 0.987, d median +0.0014; E4a_L4_W256_h16: t 1.032, d median -0.0458

### prefill, pairs (measured)

| cell | variant / anchor | what it isolates | r median [min–max] | outcome | counted | r / counted |
|---|---|---|---|---|---|---|
| p1 | E4a_L4 / E2h_L4 | the attention alone | 0.715 [0.678–0.758] | gain (variant faster) | 0.538 | 1.33 |
| p2 | E2h_L4 / REF4 | the routing of the FFN alone | 0.902 [0.866–0.947] | gain (variant faster) | 0.850 | 1.06 |
| p3 | E4a_L4_N512 / E2h_L4_N512 | the attention alone | 0.685 [0.670–0.716] | gain (variant faster) | 0.404 | 1.70 |
| p4 | E2h_L4_N512 / REF4_N512 | the routing of the FFN alone | 0.932 [0.877–0.971] | gain (variant faster) | 0.914 | 1.02 |
| p5 | E4a_L4_W256 / E2h_L4_W256 | the attention alone, at width | 0.699 [0.670–0.721] | gain (variant faster) | 0.458 | 1.52 |
| p6 | E2h_L4_W256 / REF4_W256 | the routing of the FFN alone, at width | 0.764 [0.690–0.804] | gain (variant faster) | 0.877 | 0.87 |

Products (estimated): p1xp2: 0.645 (counted 0.458); p3xp4: 0.639 (counted 0.369); p5xp6: 0.533 (counted 0.402)

## decode (delta of the run = 0.0707)

### decode, campaign n128: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2 | 0.829 | on the frontier | +0.0725 [+0.0596, +0.0811] | +0.1273, +0.1378, +0.1152 | +0.0176, +0.0243, +0.0040 |
| E4a_L4 | 1.431 | on the frontier | +0.0461 [+0.0371, +0.0543] | +0.0756, +0.0668, +0.0617 | +0.0255, +0.0169, +0.0063 |
| E4a_L6 | 2.036 | outside the range | pair against REF4 (t 1.934): d median +0.0173 [+0.0102, +0.0177], within bound; time ratio 1.053 | | |
| IND_L4 (control) | 1.410 | below the frontier beyond tau | +0.1660 [+0.1623, +0.1665] | +0.1886, +0.1876, +0.1876 | +0.1377, +0.1368, +0.1316 |

Direct pairs against the anchor (measured): REF1: t 0.573, d median +0.2186; REF3: t 1.440, d median -0.0753; REF4: t 1.934, d median -0.1179; E4a_L2: t 0.829, d median +0.1454; E4a_L4: t 1.431, d median -0.0279; E4a_L6: t 2.036, d median -0.1076; IND_L4: t 1.410, d median +0.0951

### decode, campaign n512: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2_N512 | 0.774 | on the frontier | +0.0700 [+0.0657, +0.0891] | +0.1115, +0.1330, +0.1072 | +0.0285, +0.0451, +0.0242 |
| E4a_L4_N512 | 1.379 | on the frontier | +0.0620 [+0.0359, +0.0780] | +0.0756, +0.0973, +0.1171 | -0.0019, +0.0224, +0.0352 |
| E4a_L6_N512 | 2.189 | outside the range | pair against REF4_N512 (t 2.078): d median +0.0728 [+0.0469, +0.1088], inconclusive (seeds straddle the bound); time ratio 1.053 | | |
| IND_L4_N512 (control) | 1.443 | below the frontier beyond tau | +0.2050 [+0.1913, +0.2061] | +0.2342, +0.2332, +0.2390 | +0.1535, +0.1654, +0.1632 |

Direct pairs against the anchor (measured): REF1_N512: t 0.495, d median +0.2061; REF3_N512: t 1.544, d median -0.1215; REF4_N512: t 2.078, d median -0.1815; E4a_L2_N512: t 0.774, d median +0.1451; E4a_L4_N512: t 1.379, d median -0.0279; E4a_L6_N512: t 2.189, d median -0.1086; IND_L4_N512: t 1.443, d median +0.0949

### decode, campaign w256: answer to decision 16 — **not resolved**

| arm | t (anchor = 1) | outcome | g median [min–max] (estimated) | g+ per seed | g- per seed |
|---|---|---|---|---|---|
| E4a_L2_W256 | 0.641 | on the frontier | +0.0924 [+0.0681, +0.1029] | +0.1021, +0.1289, +0.1386 | +0.0341, +0.0558, +0.0671 |
| E4a_L4_W256 | 1.238 | outside the range | pair against REF2_W256_h16 (t 1.066): d median +0.0052 [-0.0017, +0.0189], within bound; time ratio 1.161 | | |
| E4a_L4_W256_h16 | 1.292 | outside the range | pair against REF2_W256_h16 (t 1.066): d median -0.0420 [-0.0435, -0.0328], within bound; time ratio 1.211 | | |

Direct pairs against the anchor (measured): REF1_W256: t 0.464, d median +0.1940; REF2_W256_h16: t 1.066, d median -0.0038; E4a_L2_W256: t 0.641, d median +0.2072; E4a_L4_W256: t 1.238, d median +0.0014; E4a_L4_W256_h16: t 1.292, d median -0.0458

### decode, pairs (measured)

| cell | variant / anchor | what it isolates | r median [min–max] | outcome | counted | r / counted |
|---|---|---|---|---|---|---|
| p1 | E4a_L4 / E2h_L4 | the attention alone | 0.719 [0.703–0.729] | gain (variant faster) | 0.587 | 1.22 |
| p2 | E2h_L4 / REF4 | the routing of the FFN alone | 1.007 [0.966–1.022] | no difference within delta | 0.829 | 1.22 |
| p3 | E4a_L4_N512 / E2h_L4_N512 | the attention alone | 0.648 [0.642–0.697] | gain (variant faster) | 0.474 | 1.37 |
| p4 | E2h_L4_N512 / REF4_N512 | the routing of the FFN alone | 1.025 [0.997–1.032] | no difference within delta | 0.880 | 1.17 |
| p5 | E4a_L4_W256 / E2h_L4_W256 | the attention alone, at width | 0.652 [0.629–0.699] | gain (variant faster) | 0.521 | 1.25 |
| p6 | E2h_L4_W256 / REF4_W256 | the routing of the FFN alone, at width | 0.934 [0.930–0.947] | no difference within delta | 0.845 | 1.10 |

Products (estimated): p1xp2: 0.724 (counted 0.487); p3xp4: 0.664 (counted 0.417); p5xp6: 0.609 (counted 0.440)

## Memory (measured; bytes per token, fp32; counted beside)

| reading | (a) saved, token side | (b) train peak | (c) prefill peak | (e) decoded step peak | counted MACs | counted KV read |
|---|---|---|---|---|---|---|
| attention alone, 128 (E4a_L4 / E2h_L4) | 1.12 | 1.11 | 1.23 | 1.64 | 0.54 | 0.25 |
| attention alone, 512 (E4a_L4_N512 / E2h_L4_N512) | 1.32 | 1.30 | 1.24 | 4.68 | 0.40 | 0.25 |
| attention alone, width 256 (E4a_L4_W256 / E2h_L4_W256) | 1.15 | 1.14 | 1.22 | 6.02 | 0.46 | 0.25 |
| routing of the FFN alone, 128 (E2h_L4 / REF4) | 0.80 | 0.81 | 0.65 | 1.01 | 0.85 | 1.00 |
| routing of the FFN alone, 512 (E2h_L4_N512 / REF4_N512) | 0.76 | 0.77 | 0.63 | 1.01 | 0.91 | 1.00 |
| routing of the FFN alone, width 256 (E2h_L4_W256 / REF4_W256) | 0.75 | 0.76 | 0.64 | 0.97 | 0.88 | 1.00 |
