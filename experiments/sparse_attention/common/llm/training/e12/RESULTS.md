# E12 — results

README: `README.md` (frozen before the run). Qualified run: `e12/runs/run_20261026T222859Z/` (EXIT 0; 7 WSD runs of 4,000 steps, seed 100, `M = 256`, `N = 512`, all exit 0, full validation split;
round 1: 6 runs, round 2: 1 extension run; checks 203/203, 44/44, 34/34 passed). Selection record: `sweep/selection.json`. Class: **measured** (one seed: it selects, it proves nothing).
Wall time of the run ≈ 5.5 h (estimate without extension 3.5 h; the extension ran alone on one core); relative information only, not a result.

## Measured: final validation bits per byte by peak rate (WSD, `H* = 4,000`, `N = 512`, seed 100)

| rate | 0.002 | 0.004 | 0.008 | 0.016 | selected |
|---|---|---|---|---|---|
| `REF2_W256` (centre 0.008) | 2.0118 (ext.) | **1.9855** | 1.9959 | 2.0545 | **0.004** |
| `E4a_L4_W256` (centre 0.004) | 1.9869 | **1.9862** | 2.1101 | — | **0.004** |

(ext.) = rate added by the edge rule of the README. `REF2_W256`: the best rate of round 1 (0.004) was at the low edge of its grid; one extension (0.002) ended higher, so the best rate is in the
interior (`at_edge`: false, one extension). `E4a_L4_W256`: the best rate (0.004) was the centre of its grid in round 1; no extension (`at_edge`: false, zero extensions). No arm ends at an edge.

## Rates for E13 (computed from this selection; written in the README of E13 before its runs)

`REF2_W256` 0.004 and `E4a_L4_W256` 0.004 (selected); the others are computed, not swept: `REF1_W256` 0.004 × 1.26 = 0.00504; `E4a_L2_W256` the rate of `E4a_L4_W256`, 0.004; `REF2_W256_h16` 0.004 and
`E4a_L4_W256_h16` 0.004 (the rate of their 8-head counterpart).

## Reading

- The selected rates (0.004 for both families) are lower than those of E7-E8 at `M = 128`: half for `E4a_L4` (0.008) and about a third for `REF2` (0.0127, computed at E8, not swept). The rate falls
  with the width, as the halving of the centres assumed (hypothesis, one seed). At width the two families are **not separated by their rate** (0.004 for both).
- `E4a_L4_W256` is flat between 0.002 and 0.004 (0.0007 bpb apart, well within the seed band of the second series, ≈ 0.01): its rate is not unique to a factor of two. `REF2_W256` is 0.0104 above its
  optimum at 0.008, the size of the seed band: not distinguishable from it by this run. At 0.016 / 0.008 the gaps are large for both families (`E4a_L4_W256` 2.1101 at 0.008, 0.124 above its optimum: the constrained family is
  more sensitive to a high rate than the dense one at this width, a hypothesis for the cause being the masked blocks and the smaller per-block fan-in, not tested).

## Limits

- One seed, 4,000 steps, a horizon calibrated at `M = 128` and not recalibrated at width. The rates of `REF1_W256`, `E4a_L2_W256` and the two 16-head layouts are computed from the rule of the plan, never run at E12.
  The grid steps are a factor of two, so the optimum is known to within that factor only; the validation split serves the choice and the figures of E13.
