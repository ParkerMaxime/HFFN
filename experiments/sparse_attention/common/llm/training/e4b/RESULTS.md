# E4b — results

README: `README.md` (frozen before the run). Control: E3b, re-run in this campaign. Classes: **measured** unless stated.

## Measured

Qualified run: `e4b/runs/run_20261024T221415Z/` (EXIT 0; 8/8 jobs `passed`, exit 0, full validation split). Analysis: `analysis.json`, `analysis.md` in that folder.

| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E3b | 80,896 | 2.8627 / 2.8697 / 2.8687 / 2.8729 | 2.8692 | [2.8627–2.8729] | 2.899 / 2.864 / 2.854 / 2.853 |
| E4b | 81,408 | 3.0230 / 2.8474 / 3.0360 / 2.8776 | 2.9503 | [2.8474–3.0360] | 2.976 / 2.949 / 2.939 / 2.938 |

**E4b against E3b** (bound 0.075, max_degradation): paired difference per seed s0 +0.1603, s1 -0.0223, s2 +0.1673, s3 +0.0047; median +0.0825, range [-0.0223, +0.1673], paired band 0.1896. **Verdict: inconclusive (seeds straddle the bound).**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E3b | +1.2759 | +1.9737 | +6.0290 | +9.2105 |
| E4b | +1.1445 | +1.8779 | +0.9432 | +2.0688 |

Router (E4b, median over seeds, final validation): load per group 0.246 / 0.267 / 0.235 / 0.234; dropped fraction 0.0321.

Router at the end of training, per seed (load per group after capacity, dropped fraction), final validation:
s0 0.246 / 0.278 / 0.221 / 0.234, 2.2 %; s1 0.189 / 0.279 / 0.254 / 0.232, 4.6 %; s2 0.245 / 0.255 / 0.249 / 0.235, 1.6 %;
s3 0.281 / 0.244 / 0.189 / 0.244, 4.2 %. The Switch balance loss stays at 1.00-1.02 along training (1 = perfect balance,
`raw.jsonl`, reported apart from the task loss).

Reproducibility: the control `E3b` of this campaign reproduces the E3b step on the four seeds.

## Reading

- **Verdict against the bound fixed beforehand: inconclusive (the seeds straddle the bound).** Per seed the learned
  router is worse by +0.160 (s0) and +0.167 (s2), equal (+0.005, s3) or better by 0.022 (s1) than position routing; median
  +0.0825, paired band 0.19. With the criterion written beforehand the result is neither "within" nor "beyond" the bound.
- The outcome is bimodal across seeds (measured): two seeds (1, 3) end at the level of E3b (2.847, 2.878), two (0, 2) about
  0.15 worse (3.023, 3.036). The cause (a poor routing equilibrium reached by some initialisations) is a hypothesis, not
  isolated; no mechanism was added to test it (README: one mechanism per step).
- The router is balanced (load 0.19-0.28 per group, balance loss ≈ 1.0, 1.6-4.6 % of tokens dropped by the capacity) and
  nothing indicates a collapse onto one group; yet it does **not** reproduce the large gain of the byte hash (E4a −0.226 against E3b): on average
  it is worse than position routing. Learned content routing, as specified here (top-1 on the normed residual, scale
  `G·p`, balance coefficient 0.01, capacity 1.25, `W_r` std 0.02, one router for heads and FFN), does not find the
  partition that the static byte hash gives (measured comparison across steps with the same control; the reason is a
  hypothesis: the router must discover a good partition from a near-random start, while the hash is given).
- Mixing diagnostic (descriptive): the cross-block `W_down` matters little in E4b (+0.94, against +6.0 in E3b and +6.6 in E4a).

## Limits

- Four seeds with a bimodal outcome; one configuration of the router (gate, balance, capacity, initialisation fixed
  beforehand) and one learning rate (0.032, chosen for E0/E3b; a learned router may prefer another). One attempt, per
  CLOUD.md. Environment limits of E0.
