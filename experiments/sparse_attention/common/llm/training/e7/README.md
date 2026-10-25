# E7 — the learning rates

Written on 2026-10-25 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` section 14.4 (E7);
brief: `experiments/sparse_attention/CLOUD_E6_E9.md`. Harness: `../prep/README.md`. Horizon from E6: **`H* = 4,000`** (`../e6/RESULTS.md`).

## Single question

For each family, at which peak rate does the WSD run of `H* = 4,000` steps end lowest, at the two extreme depths? E7 selects rates for E8
and E9; it proves nothing (one seed).

## Design

- `N = 128`, 8,192 bytes per step, WSD with horizon 4,000 (200 warm-up steps, constant peak, linear decay to zero over the last 800),
  seed **100** (never used for a result), one plain WSD run per arm and rate (no branches, no checkpoints), the protocol of the first
  ladder otherwise. Final figure: validation bits per byte over the whole valid split. The test split is not read.
- **Arms and the centres of their grids** (estimated from the first ladder, rounded to the nearest power of two): `REF1` 0.032 (E0),
  `REF4` 0.016 (E5), `E4a_L2` 0.016 (the geometric interpolation in `L` of 0.032 at `L = 1` and 0.008 at `L = 4` gives 0.020), `E4a_L6`
  0.004 (the same, extrapolated, gives 0.0032).
- **Grid: the centre, half and double (3 rates)** per arm: `REF1` {0.016, 0.032, 0.064}, `REF4` {0.008, 0.016, 0.032}, `E4a_L2` {0.008, 0.016,
  0.032}, `E4a_L6` {0.002, 0.004, 0.008}: 12 runs. **If the best rate of an arm is at an edge of its grid, the next rate (× 2 or ÷ 2) in that
  direction is added, at most twice per arm**; if the best is still at an edge after two extensions, the edge is kept and stated as a
  limit. The extension is decided by the launcher from the run records (`lib/campaign.py`, `adaptive_sweep`), rounds one after the other.
- **Selection:** the lowest final validation bits per byte. One seed: it selects, it proves nothing; two rates within the seed band of the
  first ladder (0.006) are not distinguishable, and the selected one is stated as such if so.

## Rates of the other arms (written in the README of E8 before its runs, from this step's result)

`REF2` and `REF3`: geometric in `L` between `REF1` and `REF4`; `E4a_L4`: geometric between `E4a_L2` and `E4a_L6`; `IND_L4`: the rate of `E4a_L4`.
They are computed, not swept. The rates of `N = 128` are kept at `N = 512` (E9) — a hypothesis and a limit.

## Limits stated before the run

- One seed (100) and one horizon; rates are not tested at the middle depths (interpolated, not swept); a rate selected within the seed band is
  not unique; the validation split serves the choice and the figures of E8; the first-ladder rates (cosine, 4,000 steps) are only the centres.

## Launch

`./run.sh check | smoke | run`. Estimated wall ≈ 3.3 h without extensions (`../prep/RESULTS.md`). Records as in the first ladder.
