# E3b — results

README: `README.md` (frozen before the run). Control: E3a, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e3b/runs/run_20261024T205712Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E3a | 80,896 | 2.7606 / 2.7576 / 2.7546 / 2.7569 | 2.7572 | [2.7546–2.7606] | 2.786 / 2.767 / 2.743 / 2.734 |
| E3b | 80,896 | 2.8627 / 2.8697 / 2.8687 / 2.8729 | 2.8692 | [2.8627–2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |

**E3b against E3a** (bound 0.075, max_degradation): paired difference per seed s0 +0.1021, s1 +0.1121, s2 +0.1140, s3 +0.1160; median +0.1131, range [+0.1021, +0.1160], paired band 0.0139. **Verdict: beyond bound.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E3a | +2.4434 | +8.5730 | +0.0506 | +11.2579 |
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 |

Reproducibility: the control `E3a` of this campaign reproduces the E3a step on the four seeds (median 2.7572).

## Reading

- **Verdict against the bound fixed beforehand: beyond bound.** Routing the attention (token `t` computes the 2 heads of
  its group; `o_t` on `S_g` only) at equal parameters costs **+0.113 bpb** (median; per seed +0.102 to +0.116, paired
  band 0.014), 1.5 times the bound; by position class +0.113 / +0.097 / +0.111 / +0.119.
- Together with the previous steps, the complete constraint E3b stands at 2.8692 against 2.3925 for the dense reference E0:
  **+0.477 bpb**, 32 % of `Δ_attn` (sum of the paired medians of the steps E1 −0.0045, E2 +0.2516, E3a +0.1160,
  E3b +0.1131 = +0.476; the steps are separate campaigns whose controls reproduce each other exactly, so this is a sum of
  measured paired medians, not a new run). The largest part (about half) is the routing of the FFN (E2).
- **Analytical saving of the complete model** (estimated, PLAN section 6): 52,736 against 124,928 multiply-adds per token
  (ratio 2.37; ceiling 3.8 with dense K and V). The measured quality cost of this saving is the figure above, at this size,
  one layer, 4,000 steps.
- Mixing diagnostic (descriptive): in E3b the cross-block `W_down` matters a lot (+6.03) while `W_Q` (+1.28) and `W_K, W_V`
  (+1.97) matter less than in E3a: with a token's attention confined to its block, the outputs of the FFN are where the
  other blocks are reached (one reading; hypothesis, tested by IND).

## Limits

- One layer, 4,000 steps, four seeds, learning rate 0.032 tuned on E0/E3b at seed 100. The step changes only the
  attention routing, but the routed heads see queries from a quarter of the positions; the "four groups learn the same
  function four times" weakness of PLAN section 10 is not separated here. Environment limits of E0.
