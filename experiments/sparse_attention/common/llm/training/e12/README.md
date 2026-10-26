# E12 — the learning rates at width

Written on 2026-10-26 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` section 15.4 (E12); brief:
`experiments/sparse_attention/CLOUD_E10_E13.md`. As E7 (`../e7/README.md`), at width 256. Horizon `H* = 4,000` (E6, calibrated at `M = 128`, `N = 128`; not recalibrated at width: a limit).

## Single question

For each family, at which peak rate does the WSD run of 4,000 steps end lowest at `M = 256`, `N = 512`? E12 selects rates for E13; it proves nothing (one seed).

## Design

- `M = 256`, `H = 1,024`, `k_up = k_down = 52`, `G = 4`, 8 heads of 32 (2 per group), `N = 512` only, 8,192 bytes per step (16 × 512), WSD with horizon 4,000 (200 warm-up steps, constant peak,
  linear decay to zero over the last 800), seed **100** (never used for a result), one plain run per arm and rate, validation bits per byte over the whole valid split at the end, the test
  split not read. The hash-routed arm runs in the packed form (`--impl gather`, equal to `selected` in outputs and gradients: `../prep3/RESULTS.md`).
- **Arms and centres** (the rates of E7 halved for the doubled width, rounded to a power of two; a hypothesis): `REF2_W256` **0.008**, `E4a_L4_W256` **0.004**. **Grid: the centre, half and double**
  (`REF2_W256` {0.004, 0.008, 0.016}, `E4a_L4_W256` {0.002, 0.004, 0.008}); if the best rate of an arm is at an edge of its grid the next rate (× 2 or ÷ 2) in that direction is added, **at most
  twice per arm**; still at an edge after two extensions: the edge is kept and stated as a limit. The extension is decided by the launcher from the run records (`adaptive_sweep`).
- **Selection:** the lowest final validation bits per byte. One seed: it selects, it proves nothing; rates within the seed band of the second series (≈ 0.01 bpb) are not distinguishable.

## Rates of the other arms (computed, written in the README of E13 before its runs)

`REF1_W256`: the rate of `REF2_W256` times 1.26 (the ratio `REF1`/`REF2` of E7-E8, 0.016 / 0.0127); `E4a_L2_W256`: the rate of `E4a_L4_W256` (E7 found one rate for the `E4a` family at every depth);
`REF2_W256_h16` and `E4a_L4_W256_h16`: the rate of their 8-head counterpart. Computed, not swept.

## Limits stated before the run

- One seed, one horizon not recalibrated at width; the rates of the three other arms and of the 16-head layouts are computed; the validation split serves the choice and the figures of E13.

## Launch

`./run.sh check | smoke | run`. Estimated ≈ 3.5 h without extensions (`../prep3/RESULTS.md`). Records as in E7.
