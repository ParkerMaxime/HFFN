# E1 — results

README: `README.md` (frozen before the run). Control: E0, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e1/runs/run_20261024T192919Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E0 | 93,184 | 2.3936 / 2.3913 / 2.3914 / 2.3975 | 2.3925 | [2.3913–2.3975] | 2.417 / 2.401 / 2.383 / 2.370 |
| E1 | 93,184 | 2.3936 / 2.3867 / 2.3869 / 2.3886 | 2.3878 | [2.3867–2.3936] | 2.412 / 2.396 / 2.378 / 2.365 |

**E1 against E0** (bound 0.075, max_degradation): paired difference per seed s0 -0.0000, s1 -0.0046, s2 -0.0045, s3 -0.0088; median -0.0045, range [-0.0088, -0.0000], paired band 0.0088. **Verdict: within bound; candidate better than control in every seed.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E0 | +1.9876 | +3.2070 | +2.9099 | +13.4634 |
| E1 | +2.0627 | +2.9121 | +2.3671 | +9.3105 |

Reproducibility: the control `E0` of this campaign reproduces the figures of the E0 step to the last digit on the four
seeds (2.3936 / 2.3913 / 2.3914 / 2.3975).

## Reading

- **Verdict against the bound fixed beforehand: within bound** (`max_s d_s = −0.00002 ≤ 0.075`). Aligning the hidden
  neurons on the feature blocks at equal density costs nothing measurable here; the median paired difference is
  −0.0045 bpb and E1 is below E0 in the four seeds (seed 0: −0.00002, i.e. equal to the printed precision; seeds 1-3
  −0.004 to −0.009, of the order of the E0 band of 0.006 — so "better" is read as "not worse", not as a gain).
- Hypothesis: restricting the up-projection of a neuron to a 32-coordinate block with 26 of 32 inputs is a denser
  local connectivity than 26 of 128, which may help a one-layer model; not isolated.

## Limits

- Four seeds; the supports change with the seed, so the paired difference includes support variance.
- Mixing diagnostic of E0 and E1 uses nominal groups (neither routes); descriptive only.
- Same environment limits as E0 (bits per byte at context ≤ 128, PyPI torch on CPU, validation used for the rate choice).
