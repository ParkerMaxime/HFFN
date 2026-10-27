# E14 prefill: analysis (delta of the run = 0.0623)

| cell | variant / anchor | r median [min–max] | outcome | counted ratio | r / counted | steal fraction |
|---|---|---|---|---|---|---|
| n128-REF1 | REF1 / REF2 | 0.577 [0.557–0.588] | gain (variant faster) | 0.558 | 1.03 | 0.0140 |
| n128-REF3 | REF3 / REF2 | 1.463 [1.435–1.477] | loss (variant slower) | 1.442 | 1.01 | 0.0150 |
| n128-REF4 | REF4 / REF2 | 1.855 [1.827–1.876] | loss (variant slower) | 1.884 | 0.98 | 0.0143 |
| n128-E4a_L2 | E4a_L2 / REF2 | 0.729 [0.690–0.744] | gain (variant faster) | 0.489 | 1.49 | 0.0133 |
| n128-E4a_L4 | E4a_L4 / REF2 | 1.240 [1.192–1.296] | loss (variant slower) | 0.862 | 1.44 | 0.0137 |
| n128-E4a_L6 | E4a_L6 / REF2 | 1.788 [1.738–1.860] | loss (variant slower) | 1.236 | 1.45 | 0.0131 |
| n128-IND_L4 | IND_L4 / REF2 | 1.096 [1.078–1.147] | loss (variant slower) | 0.471 | 2.33 | 0.0130 |
| n512-REF1 | REF1_N512 / REF2_N512 | 0.565 [0.562–0.580] | gain (variant faster) | 0.534 | 1.06 | 0.0121 |
| n512-REF3 | REF3_N512 / REF2_N512 | 1.470 [1.347–1.510] | loss (variant slower) | 1.466 | 1.00 | 0.0128 |
| n512-REF4 | REF4_N512 / REF2_N512 | 1.827 [1.712–1.900] | loss (variant slower) | 1.932 | 0.95 | 0.0118 |
| n512-E4a_L2 | E4a_L2_N512 / REF2_N512 | 0.655 [0.642–0.683] | gain (variant faster) | 0.391 | 1.67 | 0.0125 |
| n512-E4a_L4 | E4a_L4_N512 / REF2_N512 | 1.161 [1.146–1.227] | loss (variant slower) | 0.714 | 1.63 | 0.0116 |
| n512-E4a_L6 | E4a_L6_N512 / REF2_N512 | 1.721 [1.643–1.752] | loss (variant slower) | 1.036 | 1.66 | 0.0115 |
| n512-IND_L4 | IND_L4_N512 / REF2_N512 | 1.065 [1.051–1.090] | loss (variant slower) | 0.483 | 2.21 | 0.0115 |
| w256-REF1_W256 | REF1_W256 / REF2_W256 | 0.562 [0.537–0.572] | gain (variant faster) | 0.525 | 1.07 | 0.0115 |
| w256-REF2_W256_h16 | REF2_W256_h16 / REF2_W256 | 1.104 [1.077–1.154] | loss (variant slower) | 1.000 | 1.10 | 0.0117 |
| w256-E4a_L2_W256 | E4a_L2_W256 / REF2_W256 | 0.559 [0.543–0.567] | gain (variant faster) | 0.417 | 1.34 | 0.0113 |
| w256-E4a_L4_W256 | E4a_L4_W256 / REF2_W256 | 0.987 [0.974–1.025] | no difference within delta | 0.784 | 1.26 | 0.0127 |
| w256-E4a_L4_W256_h16 | E4a_L4_W256_h16 / REF2_W256 | 1.032 [1.002–1.063] | no difference within delta | 0.784 | 1.32 | 0.0118 |
| p1 | E4a_L4 / E2h_L4 | 0.715 [0.678–0.758] | gain (variant faster) | 0.538 | 1.33 | 0.0134 |
| p2 | E2h_L4 / REF4 | 0.902 [0.866–0.947] | gain (variant faster) | 0.850 | 1.06 | 0.0152 |
| p3 | E4a_L4_N512 / E2h_L4_N512 | 0.685 [0.670–0.716] | gain (variant faster) | 0.404 | 1.70 | 0.0144 |
| p4 | E2h_L4_N512 / REF4_N512 | 0.932 [0.877–0.971] | gain (variant faster) | 0.914 | 1.02 | 0.0134 |
| p5 | E4a_L4_W256 / E2h_L4_W256 | 0.699 [0.670–0.721] | gain (variant faster) | 0.458 | 1.52 | 0.0138 |
| p6 | E2h_L4_W256 / REF4_W256 | 0.764 [0.690–0.804] | gain (variant faster) | 0.877 | 0.87 | 0.0121 |

Products of the pairs (estimated):
- p1xp2: 0.645 (counted 0.458)
- p3xp4: 0.639 (counted 0.369)
- p5xp6: 0.533 (counted 0.402)
