# E9 — results

README: `README.md` (readings, rates, bound frozen before the run). Qualified run: `e9/runs/run_20261025T220720Z/` (EXIT 0; 24 runs `passed`, exit 0, full validation and test splits of 9,765
windows of 512 bytes). Analysis (new files): `analysis.json`, `analysis.md`. `N = 512` trained entirely at 512, WSD horizon 4,000, 8,192 bytes per step (16 × 512), seeds 0-2, rates of E8.
`d_s = bpb(candidate) − bpb(control)`, positive = candidate worse; `τ = 0.075`. Classes as marked.

## Measured: the frontier at N = 512 (median [min–max] over three seeds)

| arm | valid bpb | test bpb (beside) | inference MACs/token | training MACs/token | params (outside emb.) | KV values/token |
|---|---|---|---|---|---|---|
| E4a_L2_N512 | 2.2889 [2.2824–2.3041] | 2.2679 [2.2603–2.2816] | 187,392 | 562,176 | 161,664 | 512 |
| IND_L4_N512 | 2.2372 [2.2366–2.2377] | 2.2156 [2.2139–2.2158] | 231,424 | 694,272 | 175,744 | 1,024 |
| REF1_N512 | 2.3498 [2.3478–2.3538] | 2.3276 [2.3239–2.3290] | 256,000 | 768,000 | 93,184 | 256 |
| E4a_L4_N512 | 2.1076 [2.0941–2.1224] | 2.0895 [2.0783–2.1036] | 342,016 | 1,026,048 | 323,200 | 1,024 |
| REF2_N512 | 2.1416 [2.1354–2.1438] | 2.1219 [2.1123–2.1220] | 479,232 | 1,437,696 | 186,240 | 512 |
| E4a_L6_N512 | 2.0330 [2.0231–2.0368] | 2.0108 [1.9995–2.0173] | 496,640 | 1,489,920 | 484,736 | 1,536 |
| REF3_N512 | 2.0139 [2.0101–2.0280] | 1.9871 [1.9865–1.9992] | 702,464 | 2,107,392 | 279,296 | 768 |
| REF4_N512 | 1.9601 [1.9281–1.9763] | 1.9365 [1.9005–1.9534] | 925,696 | 2,777,088 | 372,352 | 1,024 |

Per seed (valid, seeds 0/1/2): REF1 2.3498/2.3538/2.3478; REF2 2.1438/2.1354/2.1416; REF3 2.0280/2.0139/2.0101; REF4 1.9281/1.9763/1.9601; E4a_L2 2.2889/2.3041/2.2824;
E4a_L4 2.0941/2.1076/2.1224; E4a_L6 2.0368/2.0231/2.0330; IND_L4 2.2372/2.2377/2.2366.

Bits per byte by position range (median over seeds, validation): the first 128 positions against positions 256-511 —
REF1 2.3962 → 2.3341; REF2 2.1989 → 2.1215; REF3 2.1074 → 1.9805; REF4 2.0550 → 1.9255; E4a_L2 2.3322 → 2.2749; E4a_L4 2.1578 → 2.0908; E4a_L6 2.0879 → 2.0144;
IND_L4 2.2880 → 2.2200. The ranges 128-255 and 256-511 are within 0.009 of each other in every arm (full table in `analysis.md`).

## Measured: the registered readings

| Reading | d_s per seed (0, 1, 2) | median [range] | paired band | Outcome |
|---|---|---|---|---|
| `E4a_L2`/`REF1` (−26.8 % inference compute) | −0.0609, −0.0497, −0.0653 | −0.0609 [−0.0653, −0.0497] | 0.0156 | within; **dominates** |
| `E4a_L4`/`REF2` (−28.6 %) | −0.0498, −0.0279, −0.0192 | −0.0279 [−0.0498, −0.0192] | 0.0305 | within; **dominates** |
| `E4a_L6`/`REF3` (−29.3 %) | +0.0088, +0.0092, +0.0229 | +0.0092 [+0.0088, +0.0229] | 0.0141 | within; **equivalent within τ at less compute, not dominating** (worse in the three seeds) |
| equal depth `E4a_L2`/`REF2` (−60.9 %) | +0.1451, +0.1687, +0.1408 | +0.1451 [+0.1408, +0.1687] | 0.0278 | **beyond** |
| equal depth `E4a_L4`/`REF4` (−63.1 %) | +0.1660, +0.1313, +0.1622 | +0.1622 [+0.1313, +0.1660] | 0.0347 | **beyond** |
| mixing at depth `IND_L4`/`E4a_L4` (IND −32.3 % compute) | +0.1432, +0.1302, +0.1142 | +0.1302 [+0.1142, +0.1432] | 0.0290 | **beyond** (IND worse) |

Test split, beside, equal-compute pairs: `E4a_L2`/`REF1` −0.0560 (dominates), `E4a_L4`/`REF2` −0.0228 (dominates), `E4a_L6`/`REF3` +0.0181 [+0.0123, +0.0243] (within, not dominating) — the same outcomes
as validation.

## Estimated: equal compute, dense family interpolated in log C (per seed)

`E4a_L2`: no value (187,392 lies below `REF1`, 256,000). `E4a_L4` (between `REF1` and `REF2`, weight 0.462 on `REF2`): d median −0.1453 [−0.1606, −0.1302], within, dominates.
`E4a_L6` (between `REF2` and `REF3`, weight 0.093 on `REF3`): d median −0.0964 [−0.1010, −0.0962], within, dominates. Note: at 512 `E4a_L6` costs 496,640, slightly **more** than `REF2`
(479,232), so its interpolated neighbour is almost `REF2`; the registered pair `E4a_L6`/`REF3` compares it with a dense model 41 % more expensive. As in E8, reading (b) is numerically the
same interpolation.

## Reading

- **At `N = 512` the constrained family is better than the dense family at equal compute for the two shallower pairs, and as good within `τ` (not better) for the deepest pair**:
  `E4a_L2` and `E4a_L4` dominate `REF1` and `REF2` (27 % and 29 % less compute; all seeds, validation and test), `E4a_L6` is equivalent within `τ` to `REF3` at 29 % less compute
  but worse than it in all three seeds (+0.009 to +0.023; test +0.012 to +0.024). Measured, without interpolation.
- **The loss of quality at equal depth** is beyond `τ` for both pairs (measured): +0.145 (`E4a_L2`/`REF2`) and +0.162 (`E4a_L4`/`REF4`).
- **Coupling at depth**: `IND_L4` is worse than `E4a_L4` beyond `τ` (+0.130), as in E8: the frontier is not only the conditional computation of four separated models.
- **Is the longer context used?** Yes, by every arm, mostly between the first 128 positions and the later ones (all arms gain 0.057-0.130 bpb between positions 0-127 and 256-511); positions 128-255 and 256-511
  are nearly equal (within 0.009) in every arm: a window of 128 bytes of context is nearly all that the models exploit. The gain from the first range to the last is **larger in the deepest dense
  models** (`REF3` −0.127, `REF4` −0.130) than in the constrained models of the same depth or compute (`E4a_L6` −0.073, `E4a_L4` −0.067; `REF2` −0.077). Within this campaign, read at the
  level of the position ranges: the dense models use the longer context more, which agrees with the weaker result of the deepest constrained pair (hypothesis for the cause: a token sees through 2
  heads of 8 on the query side, PLAN section 10).
- **Effect of the context, E9 against E8 (cross-campaign reading, from the outcomes of the pairs only, not from differences of figures):** the equal-depth loss is *beyond* at both contexts; the mixing outcome
  is *beyond* at both; the dominance of the equal-compute pairs holds for the two shallower pairs at both contexts, and the deepest pair goes from **dominates at 128 to equivalent within τ at 512**.
  The constraint therefore saves more compute at 512 (27-29 % against 12-14 % by the counter) but its quality advantage at equal compute does not grow with the context: it is kept for two pairs and lost
  (not reversed beyond `τ`) for the deepest one. No claim is made on the size of any difference across the two campaigns.

## Memory, stated with the result

The favourable result is a statement about **compute**, bought with more weights, AdamW states and a larger KV cache, exactly as in E8 (same models; the KV cache per token is the same, `2 L M`, and the
total cache grows with the context in both families): `E4a_L2` 161,664 against 93,184 parameters (×1.73) and 512 against 256 KV values per token; `E4a_L4`/`REF2` 323,200 against 186,240 (×1.74), 1,024 against
512; `E4a_L6`/`REF3` 484,736 against 279,296 (×1.74), 1,536 against 768; at equal depth `E4a_L4`/`REF4` 0.87× the weights, same KV cache. Training activations per token (counter): constrained about half of the
dense at equal compute, a quarter at equal depth. At `N = 512` the saving per layer is larger (2.89 against 2.37) because the score and value terms, which the constraint divides by `G`, weigh more.
Nothing about time or measured memory is claimed.

## Limits

- Rates of 128 kept at 512 (hypothesis); `H* = 4,000` chosen at 128; three seeds, with a wide seed range for `REF4_N512` ([1.9281–1.9763], 0.048) and for `E4a_L4_N512`
  ([2.0941–2.1224]); the deepest pair's gap (+0.009 to +0.023) is of the order of the paired bands (0.014-0.035).
- 4,000 steps on 8,192 bytes per step is the same token budget as at 128: at 512 the model sees fewer independent windows. The harness runs the constrained arms in the selected form; wall time says nothing.
- `M = 128`, enwik8, hash route only; the position-range reading is descriptive and within one campaign.
