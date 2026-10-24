# IND — results

README: `README.md` (frozen before the run). Control: E3b, re-run in this campaign. Classes: **measured** unless stated.
Sign convention of this step: `d_s = bpb(IND, s) − bpb(E3b, s)`, positive = E3b better; the mixing is worth it if
`min_s d_s > 0.075`.

## Measured

Qualified run: `ind/runs/run_20261024T211955Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E3b | 80,896 | 2.8627 / 2.8697 / 2.8687 / 2.8729 | 2.8692 | [2.8627–2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |
| IND | 44,032 | 2.8821 / 2.8704 / 2.8713 / 2.8765 | 2.8739 | [2.8704–2.8821] | 2.902 / 2.875 / 2.863 / 2.850 |

**E3b against IND** (bound 0.075, min_improvement): paired difference per seed s0 +0.0194, s1 +0.0007, s2 +0.0026, s3 +0.0036; median +0.0031, range [+0.0007, +0.0194], paired band 0.0187. **Verdict: not met.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 |
| IND | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

Reproducibility: the control `E3b` of this campaign reproduces the E3b step on the four seeds (median 2.8692).
The mixing diagnostic of IND is exactly 0 for every path and seed (zero by construction; the checks verify it).

## Reading

- **Verdict against the bound fixed beforehand: not met.** The proposal does **not** beat the degenerate model by 0.075 bpb:
  the median advantage of E3b over IND is **+0.0031 bpb** (per seed +0.0194, +0.0007, +0.0026, +0.0036), of the order
  of the seed band of the arms (0.0062 in E0; ranges here 0.010 for E3b and 0.012 for IND). At one layer on bytes, the
  cross-block paths of E3b (queries, keys and values that read all `M` coordinates, `W_down` that writes all `M`) bring
  no measurable gain over four separated models, although the diagnostic shows that E3b *uses* them (zeroing `W_down`'s
  cross-block part costs +6.03 bpb in E3b): the model uses the mixing paths, and the quality is the same without them.
- Confound stated in the README: IND has 36,864 fewer parameters (44,032 against 80,896). E3b is not better than a model
  with 46 % fewer parameters and no mixing, so the parameters the mixing paths add are not converted into quality here
  (measured); whether a parameter-matched IND would be better than E3b is not measured.
- Hypothesis: with one layer, the output of the FFN is followed only by the final norm and the tied output matrix (PLAN
  section 5), which absorbs the block structure; the mixing between layers, which one layer cannot show, is the
  object of E5.

## Limits

- IND is not parameter-matched, and a single comparison at `L = 1`: the verdict concerns this size and depth only.
  Four seeds; environment limits of E0.
