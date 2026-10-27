# E13 — results

README: `README.md` (readings, rates, bound frozen before the run). Qualified run: `e13/runs/run_20261027T040908Z/` (EXIT 0; 18 runs `passed`, exit 0, full validation and test splits of 9,765 windows of 512 bytes;
checks 203/203, 44/44, 34/34). Smoke: `e13/runs/smoke_20261027T040214Z/`. Analysis (new files, `lib/analyze_w256.py`): `analysis.json`, `analysis.md` in the run folder. `M = 256`, `N = 512` trained entirely at 512, WSD
horizon 4,000, 8,192 bytes per step, seeds 0-2, rates of `../e12/RESULTS.md` (0.004 for five arms, 0.00504 for `REF1_W256`). `d_s = bpb(candidate) − bpb(control)`, positive = candidate worse; `τ = 0.075`.
Classes as marked. Wall time of the run ≈ 8.8 h against ≈ 7.6 h estimated (relative information only).

## Measured: the frontier at width 256 (median [min–max] over three seeds)

| arm | valid bpb | test bpb (beside) | inference MACs/token | training MACs/token | params (outside emb.) | KV values/token | training activations/token (counted) |
|---|---|---|---|---|---|---|---|
| E4a_L2_W256 | 2.1925 [2.1721–2.2034] | 2.1643 [2.1470–2.1809] | 552,960 | 1,658,880 | 642,816 | 1,024 | 3,328 |
| REF1_W256 | 2.1821 [2.1815–2.1834] | 2.1605 [2.1603–2.1620] | 696,320 | 2,088,960 | 370,688 | 512 | 6,656 |
| E4a_L4_W256 | 1.9867 [1.9839–2.0045] | 1.9638 [1.9615–1.9821] | 1,040,384 | 3,121,152 | 1,285,376 | 2,048 | 6,656 |
| E4a_L4_W256_h16 | 1.9435 [1.9422–1.9487] | 1.9247 [1.9234–1.9292] | 1,040,384 | 3,121,152 | 1,285,376 | 2,048 | 10,752 |
| REF2_W256 | 1.9880 [1.9853–1.9971] | 1.9664 [1.9644–1.9780] | 1,327,104 | 3,981,312 | 741,120 | 1,024 | 13,312 |
| REF2_W256_h16 | 1.9855 [1.9815–1.9857] | 1.9658 [1.9611–1.9688] | 1,327,104 | 3,981,312 | 741,120 | 1,024 | 21,504 |

## Measured: the readings (frozen in the README)

| Reading | Per seed (0, 1, 2) | Median [min–max] | Outcome |
|---|---|---|---|
| `E4a_L2_W256`/`REF1_W256` (−20.6 % counted compute) | −0.0094, +0.0091, +0.0213 | +0.0091 [−0.0094, +0.0213] | **within** (seeds straddle 0: not dominating) |
| `E4a_L4_W256`/`REF2_W256` (−21.6 %) | −0.0133, +0.0014, +0.0165 | +0.0014 [−0.0133, +0.0165] | **within** (not dominating) |
| equal depth `E4a_L2_W256`/`REF2_W256` (−58.3 %) | +0.1750, +0.2072, +0.2153 | +0.2072 [+0.1750, +0.2153] | **beyond** |
| 4 heads per group `E4a_L4_W256_h16`/`REF2_W256_h16` (−21.6 %) | −0.0420, −0.0328, −0.0435 | −0.0420 [−0.0435, −0.0328] | **within; dominates** |
| heads per group `i_s` = d(h16 pair) − d(h8 pair) | −0.0287, −0.0342, −0.0599 | −0.0342 [−0.0599, −0.0287] | **favours four heads per group** (`max i_s < 0`) |

`|i_s|/τ` = 0.38, 0.46, 0.80: no seed has `|i_s|` above `τ`. Three seeds: a direction, not a size.

Test split, beside (selects nothing): `E4a_L2_W256`/`REF1_W256` +0.0022 [−0.0135, +0.0206] within; `E4a_L4_W256`/`REF2_W256` −0.0026 [−0.0165, +0.0177] within; `E4a_L4_W256_h16`/`REF2_W256_h16` −0.0424
[−0.0440, −0.0318] within, dominates — the same outcomes as on validation.

## Measured: bits per byte by position range (median over seeds, validation)

| arm | 0-127 | 128-255 | 256-511 |
|---|---|---|---|
| E4a_L2_W256 | 2.2366 | 2.1780 | 2.1777 |
| REF1_W256 | 2.2351 | 2.1653 | 2.1642 |
| E4a_L4_W256 | 2.0416 | 1.9680 | 1.9685 |
| E4a_L4_W256_h16 | 2.0034 | 1.9250 | 1.9238 |
| REF2_W256 | 2.0509 | 1.9677 | 1.9668 |
| REF2_W256_h16 | 2.0491 | 1.9652 | 1.9635 |

Descriptive, within this campaign: in every arm the ranges 128-255 and 256-511 are within 0.002 of each other and the first range is 0.06-0.08 higher.

## Measured memory of E10 for the same arms, beside (bytes per token, fp32, one step of 8,192 bytes; `../e10/RESULTS.md`)

| arm | saved for the backward, token side | peak of a training step | peak of a prefill |
|---|---|---|---|
| E4a_L2_W256 | 37,574 | 40,210 | 10,332 |
| REF1_W256 | 23,612 | 25,928 | 13,312 |
| E4a_L4_W256 | 70,999 | 74,211 | 10,332 |
| E4a_L4_W256_h16 | 70,695 | 73,907 | 10,332 |
| REF2_W256 | 43,108 | 45,680 | 13,312 |
| REF2_W256_h16 | 43,172 | 45,744 | 13,312 |

At equal counted multiply-adds the constrained model keeps 1.6-1.7 times the dense model's saved activations (token side) in this harness, where the counter gives 0.5 (`E4a_L2_W256`/`REF1_W256` 1.59,
`E4a_L4_W256`/`REF2_W256` 1.65); the prefill peak is 0.78 of the dense one. As in E10 the counted saving of activations is not reached by the implementation measured; this says nothing of a native one.

## Cross-campaign reading (E13 against E9; from the outcomes of the pairs only, never from differences of figures)

- Equal depth: *beyond* at width 256 and at width 128 (E9) — unchanged.
- Equal counted multiply-adds, 8 heads of 32 (2 per group): at `M = 128` E9 reads *within and dominates* for `E4a_L2`/`REF1` and `E4a_L4`/`REF2` (all seeds negative); at `M = 256` both pairs read *within* but
  **not dominating** (one seed negative, two positive in each pair). The constraint stays within `τ` at 21-22 % less counted compute (smaller saving than at 128: 27-29 %), but the dominance of E9 is not reproduced at width with two heads per group.
- Equal counted multiply-adds, 16 heads of 16 (4 per group): *within and dominates* in all three seeds, at 21.6 % less counted compute. This layout was not in E9 and has no counterpart there.
- Different campaigns (width, rates, runs); no claim is made on the size of any difference across them.

## Reading

- **Measured:** at twice the width the constrained model is within `τ` of dense at equal counted compute with two heads per group (median differences +0.009 and +0.001, seeds of both signs), and dominates dense with four heads per group
  (−0.042 median, three seeds negative). It stays beyond `τ` at equal depth (+0.21). The heads-per-group reading favours four heads per group (median −0.034, all seeds of `i_s` negative).
- **Descriptive, not a frozen reading (stated after the run, medians):** the change of layout moves the constrained arm (`E4a_L4_W256`: 1.9867 → `h16`: 1.9435) and barely the dense one (`REF2_W256` 1.9880 → `h16` 1.9855). So the
  favourable `i_s` comes from the constrained arm gaining with more heads per group, not from the dense arm losing. The mechanism is not tested (**hypothesis:** with two heads per group, the heads of a group see a narrower
  slice of the residual than four heads do; not established).

## Limits

- Three seeds: a direction, not a size. The seed band of the constrained arms is wide: `E4a_L4_W256` spans 0.0206 (seed 2 at 2.0045), `E4a_L2_W256` 0.0313, against 0.0019 for `REF1_W256`; the two within-readings at 8 heads rest on that.
- Rates: four computed, not swept; E12 is one seed each and `E4a_L4_W256` is flat between 0.002 and 0.004. The `h16` layouts use the rate of the 8-head arms, not tuned for 16 heads: a different rate could move `i_s` in either direction.
- Horizon 4,000 not recalibrated at width; two depths of the dense family; one width and one context; a PyTorch executor in the packed form, so memory and time here say nothing of a native implementation.
- The validation split served the choice of the rates and the figures of this step; the test split is read once and selects nothing.
