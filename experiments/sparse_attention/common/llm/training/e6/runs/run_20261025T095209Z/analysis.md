# E6 analysis

| seed | H | REF2 | E4a_L4 | gap | |gap − gap(8,000)| |
|---|---|---|---|---|---|
| 0 | 2000 | 2.2337 | 2.2133 | -0.0204 | 0.0294 |
| 0 | 4000 | 2.1646 | 2.1289 | -0.0357 | 0.0142 |
| 0 | 8000 | 2.1140 | 2.0641 | -0.0498 | — |
| 1 | 2000 | 2.2395 | 2.2019 | -0.0375 | 0.0291 |
| 1 | 4000 | 2.1714 | 2.1161 | -0.0553 | 0.0114 |
| 1 | 8000 | 2.1212 | 2.0545 | -0.0666 | — |

Rule: smallest of [2000, 4000] with both deviations ≤ 0.025: H=2000: fail, H=4000: pass → **H\* = 4000**

Equal training compute (estimated; E4a_L4 steps = H × 1.1597):

- seed 0, REF2 at 2000: E4a_L4 at 2319 steps → bpb 2.1952, gap -0.0385
- seed 0, REF2 at 4000: E4a_L4 at 4639 steps → bpb 2.1150, gap -0.0495
- seed 0, REF2 at 8000: no value (outside the horizons of E4a_L4)
- seed 1, REF2 at 2000: E4a_L4 at 2319 steps → bpb 2.1836, gap -0.0559
- seed 1, REF2 at 4000: E4a_L4 at 4639 steps → bpb 2.1029, gap -0.0684
- seed 1, REF2 at 8000: no value (outside the horizons of E4a_L4)
