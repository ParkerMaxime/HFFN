# E9 — the frontier at N = 512

Written on 2026-10-25 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 14.3-14.5; brief:
`experiments/sparse_attention/CLOUD_E6_E9.md`. It is E8 with the context raised to 512, trained entirely at 512 (no context extension). Everything not
stated here is as `../e8/README.md`: the eight arms (with the suffix `_N512`), seeds 0, 1, 2, WSD with horizon 4,000 (`H*` of `../e6/RESULTS.md`),
8,192 bytes per step (batch 16 × 512), the readings and `τ = 0.075`, the memory paragraph, the test split read once on every run and reported beside.

## What differs from E8

- **Context 512** for training and evaluation: windows of 512 bytes, 9,765 validation windows and 9,765 test windows; intermediate evaluations on 524,288 bytes
  (1,024 windows). Rotary positions to 512; the attention is causal over up to 512 keys.
- **Rates of E8 kept at 512** (`REF1` 0.016, `REF2` 0.0127, `REF3` 0.0101, `REF4` 0.008, the four constrained arms 0.008): not re-tuned at 512 — a hypothesis
  that rates transfer when the bytes per step are held fixed, and a limit of this step.
- **Additional figure:** final validation (and test) bits per byte by position range 0-127, 128-255, 256-511, which shows whether the longer context is used.
  The mean of the ranges weighted by their lengths equals the whole figure (checked in `prep`).
- **Counter at 512 (inference multiply-adds per token):** `E4a_L2` 187,392, `REF1` 256,000, `E4a_L4` 342,016, `REF2` 479,232, `E4a_L6` 496,640, `REF3` 702,464,
  `REF4` 925,696, `IND_L4` 231,424. Savings of the equal-compute-paired constrained models: 27 %, 29 %, 29 %. `E4a_L2` (187,392) still lies below `REF1` (256,000): no
  interpolated value for it.

## Readings (the same as E8; the classes are the same)

Dominance pairs `E4a_L2/REF1`, `E4a_L4/REF2`, `E4a_L6/REF3` (measured); the interpolation of the dense family in `log C` for `E4a_L4` and `E4a_L6` (estimated); equal depth
`E4a_L2/REF2` and `E4a_L4/REF4` (measured); mixing at depth `IND_L4/E4a_L4` (measured); equal training compute is again numerically the interpolation in `log C`.

**The effect of the context** is read only from the outcomes (within, beyond, dominates) of the pairs of each campaign, E8 against E9, never from differences of figures across the two
campaigns (the bits per byte at 128 and at 512 are not comparable: different windows and contexts). Such a reading is stated as a **cross-campaign reading**. Whether the
constraint pays more at 512, where it saves more, is read from whether the dominance holds, and whether the equal-depth loss and the mixing outcome change.

## Limits stated before the run

- As E8, plus: the rates are those of 128; the number of steps is that chosen at 128 (`H*` for stability of the gap at 128); the training sees 4,000 × 8,192 bytes at either context, so the 512 model sees fewer
  independent windows; the harness runs the constraint in the selected form, so wall times and memory in this run say nothing about the method.

## Launch

`./run.sh check | smoke | run`: 24 runs, 4 in parallel, 1 thread each. Estimated wall ≈ 7.5 h (`../prep/RESULTS.md`; E8 took 5.4 h against 5.5 h estimated).
