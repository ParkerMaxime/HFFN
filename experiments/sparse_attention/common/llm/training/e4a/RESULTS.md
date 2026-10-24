# E4a — results

README: `README.md` (frozen before the run). Control: E3b, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e4a/runs/run_20261024T214033Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E3b | 80,896 | 2.8627 / 2.8697 / 2.8687 / 2.8729 | 2.8692 | [2.8627–2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |
| E4a | 80,896 | 2.6153 / 2.6778 / 2.6633 / 2.5520 | 2.6393 | [2.5520–2.6778] | 2.666 / 2.645 / 2.627 / 2.619 |

**E4a against E3b** (bound 0.075, max_degradation): paired difference per seed s0 -0.2474, s1 -0.1919, s2 -0.2053, s3 -0.3209; median -0.2264, range [-0.3209, -0.1919], paired band 0.1290. **Verdict: within bound; candidate better than control in every seed.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 |
| E4a | +2.4004 | +3.7724 | +6.5696 | +13.8121 |

Reproducibility: the control `E3b` of this campaign reproduces the E3b step on the four seeds (2.8627 / 2.8697 / 2.8687 / 2.8729).

## Reading

- **Verdict against the bound fixed beforehand: within bound, and better than the control in every seed.** Routing by the
  balanced byte hash instead of `t mod 4` lowers the validation bpb by **0.226** (median paired difference −0.2264; per
  seed −0.247, −0.192, −0.205, −0.321; paired band 0.129), at the same parameters (80,896). By position class
  −0.233 / −0.219 / −0.227 / −0.234. The gain is of the size of the whole routing cost of the FFN (E2, +0.2516): E4a at
  2.6393 is back at the level of E2 (2.6418, dense attention, position-routed FFN) with the complete constraint.
- **The seeds differ much more than in the other arms**: the range of E4a is [2.5520–2.6778] (0.126), against 0.011 for
  E3b. The hash table is the same for the four seeds, so this variance comes from the initialisation and the supports
  (measured: the spread; its cause is not isolated). The paired differences are all negative, but the band of the
  comparison (0.129) is the widest of the ladder, and a statement on the size of the gain should carry it.
- The route of token `t` is a function of its own byte only (checked: causal; route = table[byte]); no information of the
  future enters. The load per group follows the byte statistics of each window (balanced on average over train:
  25 % ± 0.01), not enforced per window.
- Hypothesis: with position routing each group sees an arbitrary quarter of the positions and the four groups learn the
  same function from a quarter of the data each (PLAN section 10); with a byte-class route each group specialises on a
  subset of the byte vocabulary, so its heads and neurons see a consistent input distribution. Not isolated (no control
  with a random per-token route or with a hash on another statistic).
- Mixing diagnostic (descriptive): E4a leans more on every path than E3b (Q +2.40, K/V +3.77, W_down +6.57).

## Limits

- Four seeds with a wide spread; the route uses the current byte only (a content route on the context was not tested).
  A balanced per-token random route (no content) would separate "content" from "balanced vs unbalanced assignment of
  tokens to groups" — not run. Environment limits of E0.
