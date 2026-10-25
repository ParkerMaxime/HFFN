# E5 — results

README: `README.md` (frozen before the run). Control: `REF4` (dense, HFFN `d20`, `L = 4`) for the constrained arms, the fixed
route for the rotations. All arms of one campaign. Classes: **measured** unless stated.

## Learning-rate sweep at L = 4 (seed 100, 1,500 steps, final validation bpb, `sweep/selection.json`)

| lr | 0.002 | 0.004 | 0.008 | 0.016 | 0.032 | 0.064 |
|---|---|---|---|---|---|---|
| REF4 | 2.3135 | 2.2129 | 2.1672 | **2.1583** | 2.2070 | 2.6620 |
| E4a_L4 | 2.4225 | 2.3163 | **2.2711** | 2.3047 | 2.4346 | 2.8690 |

Optima differ and are interior: `REF4` trains at **0.016**; `E4a_L4` and, by the README, the other constrained arms
(`E4a_L4rot`, `E3b_L4`, `E3b_L4rot`) at **0.008** (the rate of `E4a_L4`; it was not tuned for the two position-routed
arms). The L = 1 rate of 0.032 is not used at L = 4.

## Measured (4,000 steps, seeds 0-3)


| arm | params | per seed (s0 / s1 / s2 / s3) | median | [min–max] | by class t mod 4 (median) |
|---|---|---|---|---|---|
| E4a_L4 | 323,200 | 2.1347 / 2.1204 / 2.1207 / 2.1297 | 2.1252 | [2.1204–2.1347] | 2.152 / 2.132 / 2.116 / 2.101 |
| REF4 | 372,352 | 2.0362 / 2.0325 / 2.0348 / 2.0275 | 2.0337 | [2.0275–2.0362] | 2.063 / 2.041 / 2.022 / 2.009 |
| E4a_L4rot | 323,200 | 2.1293 / 2.1288 / 2.1187 / 2.1167 | 2.1237 | [2.1167–2.1293] | 2.151 / 2.130 / 2.115 / 2.100 |
| E3b_L4 | 323,200 | 2.4168 / 2.4100 / 2.4149 / 2.4202 | 2.4159 | [2.4100–2.4202] | 2.441 / 2.423 / 2.404 / 2.397 |
| E3b_L4rot | 323,200 | 2.4118 / 2.4093 / 2.4138 / 2.4079 | 2.4105 | [2.4079–2.4138] | 2.433 / 2.418 / 2.399 / 2.387 |

**E4a_L4 against REF4** (bound 0.075, max_degradation): paired difference per seed s0 +0.0986, s1 +0.0879, s2 +0.0859, s3 +0.1022; median +0.0932, range [+0.0859, +0.1022], paired band 0.0163. **Verdict: beyond bound.**

**E4a_L4rot against E4a_L4** (bound 0.075, max_degradation): paired difference per seed s0 -0.0054, s1 +0.0084, s2 -0.0021, s3 -0.0130; median -0.0038, range [-0.0130, +0.0084], paired band 0.0214. **Verdict: within bound.**

**E3b_L4 against REF4** (bound 0.075, max_degradation): paired difference per seed s0 +0.3807, s1 +0.3775, s2 +0.3801, s3 +0.3926; median +0.3804, range [+0.3775, +0.3926], paired band 0.0152. **Verdict: beyond bound.**

**E3b_L4rot against E3b_L4** (bound 0.075, max_degradation): paired difference per seed s0 -0.0050, s1 -0.0007, s2 -0.0011, s3 -0.0123; median -0.0031, range [-0.0123, -0.0007], paired band 0.0116. **Verdict: within bound; candidate better than control in every seed.**

**E4a_L4 against E3b_L4** (bound 0.075, max_degradation): paired difference per seed s0 -0.2821, s1 -0.2896, s2 -0.2942, s3 -0.2905; median -0.2900, range [-0.2942, -0.2821], paired band 0.0121. **Verdict: within bound; candidate better than control in every seed.**

Mixing diagnostic (descriptive; increase of validation bpb on 4,096 windows when the cross-block part is zeroed, median over seeds):

| arm | W_Q | W_K, W_V | W_down | all |
|---|---|---|---|---|
| E4a_L4 | +1.8492 | +3.1824 | +3.5632 | +6.2502 |
| REF4 | +2.2646 | +2.9354 | +2.0477 | +5.1696 |
| E4a_L4rot | +1.8671 | +2.9277 | +4.4339 | +6.7795 |
| E3b_L4 | +2.4742 | +4.0481 | +4.3497 | +7.9479 |
| E3b_L4rot | +2.4123 | +3.4374 | +5.8449 | +8.1834 |

## Reading

- **Cost of the constraint at depth: `E4a_L4` against `REF4`: beyond bound, but much smaller than at one layer.** +0.0932 bpb
  (median; per seed +0.086 to +0.102; the lowest seed is 0.011 above the bound of 0.075), against +0.247 for the same
  routing at `L = 1` (E4a 2.6393 against E0 2.3925, separate campaigns whose shared arms reproduce exactly). The constraint
  costs 2.6 times less at `L = 4` for the hash route (a cross-campaign reading of two measured gaps; the arms differ in
  depth, not only in the constraint, and the learning rates differ between arms by the README rule).
- **Position routing at depth: `E3b_L4` against `REF4`: +0.380 bpb, beyond bound** (all seeds +0.378 to +0.393), against
  +0.477 at `L = 1`: the gap narrows by 0.10 but remains four times the bound.
- **Hash against position routing at depth: `E4a_L4` against `E3b_L4`: −0.290 bpb, better in every seed** (−0.282 to −0.294;
  paired band 0.012): the advantage of the byte hash found at `L = 1` (−0.226) is kept and larger at `L = 4`.
- **Fixed against rotation: no measurable difference.** Hash: `E4a_L4rot` against `E4a_L4` median −0.0038 (per seed −0.005,
  +0.008, −0.002, −0.013), within bound, not better in every seed. Position: `E3b_L4rot` against `E3b_L4` median −0.0031
  (−0.005, −0.001, −0.001, −0.012), within bound and below the fixed route in all four seeds, but each difference is
  of the order of the seed band (0.006-0.010): read as "rotation is not worse and brings at most a few thousandths",
  not as a gain. The inter-layer mixing that rotation adds is therefore not what limits quality here (hypothesis; the
  mixing diagnostic below shows both variants use the cross-block paths).
- Mixing diagnostic (descriptive; increase of bpb when the cross-block part is zeroed): all constrained arms use the cross-block
  `W_down` (+3.6 to +5.8) and `W_K, W_V` (+2.9 to +4.0); the dense reference `REF4` also depends on them under the nominal groups
  (+2.0 and +2.9). IND was not rerun at `L = 4` (the ladder defines it at `L = 1`): whether the mixing is worth anything at
  depth, which E5 was meant to probe, is **not measured**; the diagnostic shows use, not benefit.
- Analytical saving (estimated, PLAN section 6): the per-layer ratio of 2.37 at the working size applies to each layer; the
  KV cache stays dense. Nothing about time is claimed.

## Limits

- The learned router per layer did not enter (README, reason given). No IND at `L = 4`, so "is the mixing between layers worth it"
  is open. The constrained arms use the learning rate of `E4a_L4`; `E3b_L4` / `E3b_L4rot` may prefer another.
- Four seeds; 4,000 steps (the 4-layer models are likely further from convergence than the 1-layer models at the same step
  count: not measured). Environment limits of E0. The step compares arms of different parameter counts (323,200 against 372,352),
  as the ladder does.
