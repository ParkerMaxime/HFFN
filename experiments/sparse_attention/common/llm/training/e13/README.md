# E13 — the frontier at width 256

Written on 2026-10-27 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 15.4-15.5; brief:
`experiments/sparse_attention/CLOUD_E10_E13.md`. Harness, counter and parity of the packed form: `../prep3/RESULTS.md`; horizon: `../e6/RESULTS.md` (`H* = 4,000`, not recalibrated at width);
rates: `../e12/RESULTS.md`. It is E9 at `M = 256` with two head layouts. Everything not stated here is as `../e9/README.md`.

## Single question

At twice the width (`M = 256`, `N = 512`), is the constrained model (balanced byte hash, fixed at every layer) better, equal or worse than the dense-attention HFFN `d20` model at equal counted
inference multiply-adds, what does it lose at equal depth, and does the result move when the heads per group go from two (8 heads of 32) to four (16 heads of 16)? All quantities are the final validation
bits per byte; the test split is read once and reported beside.

## Design

- **Six arms:** `REF1_W256`, `REF2_W256` (dense attention, HFFN `d20` unrouted, 1 and 2 layers), `E4a_L2_W256`, `E4a_L4_W256` (the complete constraint with the byte hash, 2 and 4 layers),
  `REF2_W256_h16`, `E4a_L4_W256_h16` (the same with 16 heads of 16, 4 per group). `M = 256`, `H = 1,024`, `k_up = k_down = 52`, `G = 4`, `N = 512` for training and evaluation,
  8,192 bytes per step (16 × 512), **seeds 0, 1, 2** (the same bytes in the same order for every arm), **WSD with horizon 4,000** (200 warm-up steps, constant peak, linear decay to zero over the
  last 800), one plain run per arm and seed. Weight decay 0.1 on matrices other than the embedding, clipping 1.0. Hash-routed arms in the packed form (`--impl gather`, equal to the selected form in
  outputs and gradients: `../prep3/RESULTS.md`); the dense arms are unaffected by the option.
- **Peak rates (from E12, computed before the runs):** `REF2_W256` **0.004** and `E4a_L4_W256` **0.004** (selected by E12, one seed each, both interior); the other four are computed, not swept:
  `REF1_W256` 0.004 × 1.26 = **0.00504** (the ratio `REF1`/`REF2` of E7-E8, 0.016 / 0.0127); `E4a_L2_W256` **0.004** (the rate of `E4a_L4_W256`; E7 found one rate for the `E4a` family at every depth);
  `REF2_W256_h16` **0.004** and `E4a_L4_W256_h16` **0.004** (the rate of their 8-head counterpart).
- **Figures:** final validation bits per byte over the whole valid split (39,062 windows) and by position range 0-127, 128-255, 256-511; the **test split** (`--test`) read once on every run, reported
  beside, selecting nothing; the counter of each arm; beside, the memory measured by E10 for the same arms (per token, fp32, one step of 8,192 bytes). The mixing diagnostic is not run (not in the plan).

## Counter (estimated; `lib/count.py`, PLAN 15.5)

| Arm | Inference MACs per token | Parameters outside the embedding | KV values per token | Training activations per token |
|---|---|---|---|---|
| `E4a_L2_W256` | 552,960 | 642,816 | 1,024 | 3,328 |
| `REF1_W256` | 696,320 | 370,688 | 512 | 6,656 |
| `E4a_L4_W256` | 1,040,384 | 1,285,376 | 2,048 | 6,656 |
| `E4a_L4_W256_h16` | 1,040,384 | 1,285,376 | 2,048 | 10,752 |
| `REF2_W256` | 1,327,104 | 741,120 | 1,024 | 13,312 |
| `REF2_W256_h16` | 1,327,104 | 741,120 | 1,024 | 21,504 |

Counted saving of the equal-compute pairs: `E4a_L2_W256`/`REF1_W256` 20.6 % (21 %), `E4a_L4_W256`/`REF2_W256` 21.6 % (22 %), `E4a_L4_W256_h16`/`REF2_W256_h16` 21.6 % (22 %); `E4a_L2_W256` costs 41.7 % of
`REF2_W256` (58 % saved). These match PLAN 15.5 (checked by `check_series3.py`).

## Readings (frozen). `d_s = bpb(candidate) − bpb(control)` per seed, positive = candidate worse; `τ = 0.075` bpb

Outcomes (PLAN 13.1): *within* if `max_s d_s ≤ τ`; *beyond* if `min_s d_s > τ`; otherwise *inconclusive*. Where the candidate costs less counted inference compute than the control, a fourth:
*dominates* if `max_s d_s < 0`.

| Reading | Candidate / control | Class |
|---|---|---|
| Equal counted multiply-adds, width | `E4a_L2_W256`/`REF1_W256` (21 % saved), `E4a_L4_W256`/`REF2_W256` (22 % saved) | measured |
| Equal depth, width | `E4a_L2_W256`/`REF2_W256` | measured |
| Equal counted multiply-adds, 4 heads per group | `E4a_L4_W256_h16`/`REF2_W256_h16` (22 % saved) | measured |
| Heads per group | `i_s = d_s(E4a_L4_W256_h16 − REF2_W256_h16) − d_s(E4a_L4_W256 − REF2_W256)` | measured |

- The heads-per-group reading compares the gap to dense under each layout, so that the dense arm absorbs what the layout does to any model (smaller heads) and only its effect on the constraint is read.
  It is stated *favours four heads per group* if `max_s i_s < 0`, *favours two heads per group* if `min_s i_s > 0`, otherwise *not resolved*; `|i_s|` against `τ` is given beside it. Three seeds: a direction,
  not a size.
- No interpolation in `log C` is read at width (the dense family has two points).
- The frontier itself (median and [min–max] over seeds of validation and test bits per byte against the counted multiply-adds, parameters, KV cache and training activations, and the bits per byte by position range)
  is reported in `RESULTS.md` (`lib/analyze_w256.py`), with the measured memory of E10 for the same arms beside it.
- **Cross-campaign reading:** whether the outcomes of the first two readings agree with those of E9 (`../e9/RESULTS.md`) is stated as a cross-campaign reading; figures are not compared across the campaigns
  (different width, rates and runs).

## Limits stated before the run

- Seeds 0, 1, 2: a direction, not a size. Rates of four arms computed, not swept; the rates of E12 are one seed each and the optimum is known to a factor of two only (`E4a_L4_W256` is flat between 0.002 and
  0.004, E12). The horizon is not recalibrated at width.
- Two depths of the dense family only; one width; one context (512). The harness runs the constraint in the packed form of a PyTorch executor: wall times and memory in this run say nothing about the method
  (time is E11; memory is E10).
- The validation split served the choice of the rates (E12) and the figures of E13; the test split is read once beside.

## Launch

`./run.sh check | smoke | run`: 18 runs, 4 in parallel, 1 thread each. Estimated wall ≈ 7.6 h (`../prep3/RESULTS.md`; E12 took 5.5 h against 3.5 h estimated without extension, the extension running alone on a core).
