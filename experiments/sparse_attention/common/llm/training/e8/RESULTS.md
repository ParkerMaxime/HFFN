# E8 — results

README: `README.md` (readings, rates and bound frozen before the run). Qualified run: `e8/runs/run_20261025T163405Z/` (EXIT 0; 24 runs `passed`, exit 0, full validation and
test splits). Analysis (new files): `analysis.json`, `analysis.md` from `lib/analyze_frontier.py`. WSD, horizon 4,000, `N = 128`, seeds 0-2, rates of E7.
Classes as marked. `d_s = bpb(candidate) − bpb(control)`, positive = candidate worse; `τ = 0.075`.

## Measured: the frontier (median [min–max] over three seeds; compute from the counter)

| arm | valid bpb | test bpb (beside) | inference MACs/token | training MACs/token | params (outside emb.) | KV values/token |
|---|---|---|---|---|---|---|
| IND_L4 | 2.2391 [2.2391–2.2440] | 2.2204 [2.2165–2.2210] | 133,120 | 399,360 | 175,744 | 1,024 |
| E4a_L2 | 2.2928 [2.2847–2.3005] | 2.2702 [2.2616–2.2780] | 138,240 | 414,720 | 161,664 | 512 |
| REF1 | 2.3672 [2.3632–2.3699] | 2.3449 [2.3370–2.3477] | 157,696 | 473,088 | 93,184 | 256 |
| E4a_L4 | 2.1161 [2.1105–2.1289] | 2.1001 [2.0926–2.1046] | 243,712 | 731,136 | 323,200 | 1,024 |
| REF2 | 2.1475 [2.1440–2.1513] | 2.1277 [2.1248–2.1305] | 282,624 | 847,872 | 186,240 | 512 |
| E4a_L6 | 2.0398 [2.0391–2.0438] | 2.0206 [2.0201–2.0267] | 349,184 | 1,047,552 | 484,736 | 1,536 |
| REF3 | 2.0721 [2.0687–2.0733] | 2.0504 [2.0473–2.0530] | 407,552 | 1,222,656 | 279,296 | 768 |
| REF4 | 2.0265 [2.0215–2.0296] | 2.0055 [1.9975–2.0120] | 532,480 | 1,597,440 | 372,352 | 1,024 |

Per seed (valid): REF1 2.3632/2.3672/2.3699; REF2 2.1475/2.1440/2.1513; REF3 2.0733/2.0687/2.0721; REF4 2.0296/2.0265/2.0215; E4a_L2 2.2928/2.3005/2.2847;
E4a_L4 2.1289/2.1161/2.1105; E4a_L6 2.0398/2.0438/2.0391; IND_L4 2.2440/2.2391/2.2391 (seeds 0/1/2; the order in `analysis.json`).
Reproducibility: `E4a_L4` seeds 0 and 1 reproduce the `H = 4,000` branches of E6 (same rate, same code) to the last digit (2.1289, 2.1161).

## Measured: the registered readings

| Reading | d_s per seed (0, 1, 2) | median [range] | paired band | Outcome |
|---|---|---|---|---|
| `E4a_L2`/`REF1` (−12.3 % inference compute) | −0.0703, −0.0667, −0.0851 | −0.0703 [−0.0851, −0.0667] | 0.0185 | within; **dominates** |
| `E4a_L4`/`REF2` (−13.8 %) | −0.0186, −0.0279, −0.0407 | −0.0279 [−0.0407, −0.0186] | 0.0221 | within; **dominates** |
| `E4a_L6`/`REF3` (−14.3 %) | −0.0335, −0.0249, −0.0330 | −0.0330 [−0.0335, −0.0249] | 0.0086 | within; **dominates** |
| equal depth `E4a_L2`/`REF2` (−51.1 %) | +0.1454, +0.1565, +0.1335 | +0.1454 [+0.1335, +0.1565] | 0.0230 | **beyond** |
| equal depth `E4a_L4`/`REF4` (−54.2 %) | +0.0993, +0.0896, +0.0891 | +0.0896 [+0.0891, +0.0993] | 0.0102 | **beyond** |
| mixing at depth `IND_L4`/`E4a_L4` (IND −45.4 % compute) | +0.1152, +0.1230, +0.1285 | +0.1230 [+0.1152, +0.1285] | 0.0133 | **beyond** (IND worse) |

The test split gives the same outcomes for the three equal-compute pairs (medians −0.0670, −0.0276, −0.0273; all three seeds negative), read beside, selecting nothing.

## Estimated: equal compute, dense family interpolated linearly in log C (per seed)

`E4a_L2`: no value (138,240 lies below `REF1`, 157,696). `E4a_L4` (between `REF1` and `REF2`, weight 0.746 on `REF2`): d median −0.0846 [−0.0962, −0.0734],
within, dominates. `E4a_L6` (between `REF2` and `REF3`, weight 0.578 on `REF3`): d median −0.0648 [−0.0665, −0.0567], within, dominates.
Reading (b), equal training compute: with every arm trained for the same 4,000 steps on the same tokens and training = 3 × inference, it is
numerically the same interpolation (README); the training column is in the table.

## Reading

- **At equal inference compute, in all three pairs and all three seeds, the constrained model is better than the dense model that costs 12-14 % more**, by
  0.028 to 0.070 bpb (median), on validation and on test (measured, dominates). The estimated interpolation says the same for the two models inside the range of
  the dense family (−0.085 and −0.065). The first-ladder verdict of a loss of quality holds **at equal depth** (+0.145 at `L = 2`, +0.090 at `L = 4`, beyond `τ`, measured),
  and is more than paid back **at equal compute** because depth is cheap in the constrained model.
- **Coupling at depth (question 2 of PLAN 13.5):** the four separated models (`IND_L4`) are **worse than `E4a_L4` by 0.123 bpb** (all seeds +0.115 to +0.129, beyond `τ`,
  at 45 % less inference compute and 46 % fewer parameters). Contrary to `L = 1` (+0.003, first ladder), the coupling of heads and FFN groups (dense queries, keys and values
  reading all of `M`, `W_down` writing all of `M`) is worth a measurable amount at depth 4 against four separated models; the frontier of `E4a` is therefore not only the
  conditional computation of four separated models. The confound stands: IND has fewer parameters and compute (the comparison is not parameter-matched or compute-matched).
- **Observation outside the registered readings (descriptive, not a reading of the plan):** `IND_L4` (133,120 MACs) has lower bits per byte than `REF1` (157,696; −0.128) and
  than `E4a_L2` (138,240; −0.054), at about the same or lower compute. IND is a four-block model whose keys and values are block-diagonal; this suggests that depth matters more than width of mixing in
  this regime. It is proposed to the user as a question (SUMMARY_E6_E9), not claimed.
- The mixing diagnostic (descriptive) is zero for `IND_L4` and +4.4 to +6.1 (all paths) for `E4a`; the dense arms depend on the nominal groups as in the first ladder.

## Memory, stated with the result

The favourable frontier is a statement about **compute**, bought with more weights, AdamW states and a larger KV cache:

| Pair | Parameters (outside emb.) constrained / dense | AdamW state values | KV values per token | Training activation values per token (counter) |
|---|---|---|---|---|
| `E4a_L2` / `REF1` | 161,664 / 93,184 (×1.73) | 323,328 / 186,368 | 512 / 256 (×2) | 1,152 / 2,304 (×0.5) |
| `E4a_L4` / `REF2` | 323,200 / 186,240 (×1.74) | 646,400 / 372,480 | 1,024 / 512 (×2) | 2,304 / 4,608 (×0.5) |
| `E4a_L6` / `REF3` | 484,736 / 279,296 (×1.74) | 969,472 / 558,592 | 1,536 / 768 (×2) | 3,456 / 6,912 (×0.5) |
| `E4a_L4` / `REF4` (equal depth) | 323,200 / 372,352 (×0.87) | 646,400 / 744,704 | 1,024 / 1,024 (×1) | 2,304 / 9,216 (×0.25) |

At equal depth the constrained model saves 13 % of weights and states (block-diagonal `W_O`) and nothing of the KV cache; at equal compute it carries about 1.7 times the
weights and states and twice the KV cache of the dense model it beats. The training activations saved by the constraint (queries, scores, attention outputs and FFN
activations divided by `G`) are counted, not measured: per token and layer the constrained model keeps a quarter of the dense activations, so at equal compute (twice the layers) it keeps half. No time or memory was measured (decision 9); the harness runs the constraint in the selected
form, so wall times say nothing about the method.

## Limits

- Three seeds; rates of the middle depths computed, not swept; rates chosen on seed 100 at one horizon, so the dense family may be slightly under-tuned at
  `L = 2` and `3` (computed rates) — a bias against the dense family at equal compute whose size is not measured (the constrained middle rate 0.008 is also computed).
- `H* = 4,000` was chosen for gap stability (E6): the constrained advantage grew by about 0.011-0.014 between 4,000 and 8,000 steps for the `E4a_L4`/`REF2` pair (E6, two seeds), so the
  dominance read at 4,000 is, if anything, conservative for that pair (measured at E6; not re-measured here).
- Validation selected the rates (E7, seed 100) and reports the figures; the test split agrees but was not an independent selection check for the rate of E7.
- `M = 128`, enwik8, `N = 128`; the hash route only; nothing about other routings, widths, datasets.
