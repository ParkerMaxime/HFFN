# E6 — results

README: `README.md` (rule frozen before the run). Qualified run: `e6/runs/run_20261025T095209Z/` (EXIT 0; 4 plateau jobs and 12 branch jobs `passed`, exit 0,
full validation split for the 12 branches; plateau evaluations on 1,024 windows are not results). Analysis (new files):
`analysis.json`, `analysis.md` from `lib/analyze_e6.py`. Classes: **measured** unless stated.

## Measured: the twelve final figures (validation bits per byte, WSD, `N = 128`)

| seed | H | REF2 (lr 0.025) | E4a_L4 (lr 0.008) | gap = E4a_L4 − REF2 | `|gap(H) − gap(8,000)|` |
|---|---|---|---|---|---|
| 0 | 2,000 | 2.2337 | 2.2133 | −0.0204 | 0.0294 |
| 0 | 4,000 | 2.1646 | 2.1289 | −0.0357 | 0.0142 |
| 0 | 8,000 | 2.1140 | 2.0641 | −0.0498 | — |
| 1 | 2,000 | 2.2395 | 2.2019 | −0.0375 | 0.0291 |
| 1 | 4,000 | 2.1714 | 2.1161 | −0.0553 | 0.0114 |
| 1 | 8,000 | 2.1212 | 2.0545 | −0.0666 | — |

**Rule of the README:** `H = 2,000`: both deviations (0.0294, 0.0291) exceed 0.025 → fail. `H = 4,000`: both (0.0142, 0.0114) are within
0.025 → pass. **`H* = 4,000`.** The series goes on to E7 (the stop applies only if `H* = 8,000`).

## Reading

- The gap is **negative at every horizon and for both seeds**: `E4a_L4`, which costs 243,712 multiply-adds per token at inference against 282,624 for
  `REF2` (−13.8 %, counter), has the lower bits per byte in all six pairs, by 0.020 to 0.067. This is an observation of the horizon
  study (two seeds, rates not tuned under WSD), not the equal-compute reading of E8, which has its own seeds, rates and rule.
- The gap **keeps widening in favour of the constrained model as the horizon grows** (seed 0: −0.020, −0.036, −0.050; seed 1: −0.038, −0.055,
  −0.067). The rule accepts 4,000 because the gap moves by only 0.014 and 0.011 between 4,000 and 8,000, but it is not constant:
  a gap read at `H* = 4,000` **understates the advantage of the constrained model at 8,000 by about 0.01** (measured here, direction
  favourable to the method being measured; for a claim of the form "the constrained model is worse than X" the horizon would be the
  conservative one). The hypothesis of PLAN 14.4 (the gap grows again in the decay because of different peak rates) is not what is seen
  here: under WSD the horizon effect goes the other way.
- Secondary reading (b), **estimated**: at equal **training** compute (`E4a_L4` 731,136 training multiply-adds per token against 847,872 for `REF2`, hence
  1.1597 × the steps; interpolation linear in log `H` between the three horizons of `E4a_L4`, inside [2,000; 8,000]): gap −0.0385 and −0.0559 at the
  point of `REF2` at 2,000 steps (seeds 0, 1); −0.0495 and −0.0684 at 4,000 steps; no value at 8,000. One pair, two seeds.

## Limits

- Two seeds; one pair of two arms; rates chosen under cosine in the first ladder (the `REF2` rate by interpolation, never swept): the gap is partly a
  gap of rates (E7 tunes them at `H* = 4,000`). The horizon is chosen for the stability of the gap within 0.025, not of the loss, and the gap is
  still moving by about 0.01 at 4,000.
- The plateau, the three branches and the uninterrupted run are the same computation (verified bit for bit in `prep` on a short horizon); the
  branches of one (arm, seed) share the plateau up to their checkpoint, so the three horizons of a pair are not independent runs.
- No harness defect in this step.
