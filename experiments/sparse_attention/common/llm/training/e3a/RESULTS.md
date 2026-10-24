# E3a — results

README: `README.md` (frozen before the run). Control: E2, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e3a/runs/run_20261024T203226Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E2 | 93,184 | 2.6440 / 2.6472 / 2.6392 / 2.6395 | 2.6418 | [2.6392–2.6472] | 2.666 / 2.648 / 2.632 / 2.622 |
| E3a | 80,896 | 2.7606 / 2.7576 / 2.7546 / 2.7569 | 2.7572 | [2.7546–2.7606] | 2.786 / 2.767 / 2.743 / 2.734 |

**E3a against E2** (bound 0.075, max_degradation): paired difference per seed s0 +0.1167, s1 +0.1104, s2 +0.1154, s3 +0.1173; median +0.1160, range [+0.1104, +0.1173], paired band 0.0070. **Verdict: beyond bound.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E2 | +2.3296 | +6.1329 | +0.3432 | +8.3025 |
| E3a | +2.4434 | +8.5730 | +0.0506 | +11.2579 |

Reproducibility: the control `E2` of this campaign reproduces the figures of the E2 step on the four seeds
(median 2.6418, [2.6392–2.6472]).

## Reading

- **Verdict against the bound fixed beforehand: beyond bound.** Restricting `W_O` to its four `32 × 32` blocks costs
  **+0.116 bpb** (median; per seed +0.110 to +0.117, paired band 0.007), 1.5 times the bound of 0.075 and 7.7 % of
  `Δ_attn`. By position class the cost is +0.120 / +0.119 / +0.112 / +0.112 (measured).
- The step removes 12,288 parameters (13 %) **and** the mixing across head groups at the output of attention in the same
  change (README: "the price of the block-diagonal projection"); the two effects vary together and are not separated here.
- Mixing diagnostic (descriptive): zeroing the cross-block part of `W_down` costs only +0.05 in E3a (+0.34 in E2), the
  `W_K, W_V` zeroing +8.6 (+6.1 in E2): the model leans more on the dense keys and values and less on the cross-block
  outputs of the FFN once `W_O` is block-diagonal (one reading; hypothesis).

## Limits

- Parameter count and structure are changed together by design (no parameter-matched `W_O`). Four seeds; the environment
  limits of E0 apply.
