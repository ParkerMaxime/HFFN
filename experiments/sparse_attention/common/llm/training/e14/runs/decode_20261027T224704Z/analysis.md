# E14 decode: analysis (delta of the run = 0.0707)

| cell | variant / anchor | r median [min–max] | outcome | counted ratio | r / counted | steal fraction |
|---|---|---|---|---|---|---|
| n128-REF1 | REF1 / REF2 | 0.573 [0.558–0.597] | gain (variant faster) | 0.566 | 1.01 | 0.0085 |
| n128-REF3 | REF3 / REF2 | 1.440 [1.430–1.483] | loss (variant slower) | 1.434 | 1.00 | 0.0097 |
| n128-REF4 | REF4 / REF2 | 1.934 [1.875–1.977] | loss (variant slower) | 1.869 | 1.03 | 0.0092 |
| n128-E4a_L2 | E4a_L2 / REF2 | 0.829 [0.816–0.837] | gain (variant faster) | 0.520 | 1.59 | 0.0083 |
| n128-E4a_L4 | E4a_L4 / REF2 | 1.431 [1.408–1.464] | loss (variant slower) | 0.910 | 1.57 | 0.0100 |
| n128-E4a_L6 | E4a_L6 / REF2 | 2.036 [1.948–2.114] | loss (variant slower) | 1.299 | 1.57 | 0.0099 |
| n128-IND_L4 | IND_L4 / REF2 | 1.410 [1.390–1.459] | loss (variant slower) | 0.467 | 3.02 | 0.0116 |
| n512-REF1 | REF1_N512 / REF2_N512 | 0.495 [0.492–0.507] | gain (variant faster) | 0.547 | 0.91 | 0.0107 |
| n512-REF3 | REF3_N512 / REF2_N512 | 1.544 [1.521–1.639] | loss (variant slower) | 1.453 | 1.06 | 0.0106 |
| n512-REF4 | REF4_N512 / REF2_N512 | 2.078 [1.968–2.151] | loss (variant slower) | 1.906 | 1.09 | 0.0096 |
| n512-E4a_L2 | E4a_L2_N512 / REF2_N512 | 0.774 [0.772–0.809] | gain (variant faster) | 0.444 | 1.74 | 0.0090 |
| n512-E4a_L4 | E4a_L4_N512 / REF2_N512 | 1.379 [1.357–1.479] | loss (variant slower) | 0.794 | 1.74 | 0.0102 |
| n512-E4a_L6 | E4a_L6_N512 / REF2_N512 | 2.189 [2.076–2.235] | loss (variant slower) | 1.144 | 1.91 | 0.0093 |
| n512-IND_L4 | IND_L4_N512 / REF2_N512 | 1.443 [1.385–1.508] | loss (variant slower) | 0.476 | 3.03 | 0.0103 |
| w256-REF1_W256 | REF1_W256 / REF2_W256 | 0.464 [0.448–0.475] | gain (variant faster) | 0.531 | 0.87 | 0.0103 |
| w256-REF2_W256_h16 | REF2_W256_h16 / REF2_W256 | 1.066 [1.054–1.092] | no difference within delta | 1.000 | 1.07 | 0.0111 |
| w256-E4a_L2_W256 | E4a_L2_W256 / REF2_W256 | 0.641 [0.631–0.689] | gain (variant faster) | 0.458 | 1.40 | 0.0101 |
| w256-E4a_L4_W256 | E4a_L4_W256 / REF2_W256 | 1.238 [1.187–1.333] | loss (variant slower) | 0.854 | 1.45 | 0.0098 |
| w256-E4a_L4_W256_h16 | E4a_L4_W256_h16 / REF2_W256 | 1.292 [1.229–1.317] | loss (variant slower) | 0.854 | 1.51 | 0.0098 |
| p1 | E4a_L4 / E2h_L4 | 0.719 [0.703–0.729] | gain (variant faster) | 0.587 | 1.22 | 0.0092 |
| p2 | E2h_L4 / REF4 | 1.007 [0.966–1.022] | no difference within delta | 0.829 | 1.22 | 0.0107 |
| p3 | E4a_L4_N512 / E2h_L4_N512 | 0.648 [0.642–0.697] | gain (variant faster) | 0.474 | 1.37 | 0.0120 |
| p4 | E2h_L4_N512 / REF4_N512 | 1.025 [0.997–1.032] | no difference within delta | 0.880 | 1.17 | 0.0111 |
| p5 | E4a_L4_W256 / E2h_L4_W256 | 0.652 [0.629–0.699] | gain (variant faster) | 0.521 | 1.25 | 0.0122 |
| p6 | E2h_L4_W256 / REF4_W256 | 0.934 [0.930–0.947] | no difference within delta | 0.845 | 1.10 | 0.0120 |

Products of the pairs (estimated):
- p1xp2: 0.724 (counted 0.487)
- p3xp4: 0.664 (counted 0.417)
- p5xp6: 0.609 (counted 0.440)
