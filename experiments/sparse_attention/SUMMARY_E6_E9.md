# sparse_attention — summary of the second series (E6-E9), cloud session of 2026-10-25/26

Branch `sparse-attention-cloud`, from `bad5c44` (the plan of the series and its brief). Plan: `current/sparse_attention/PLAN.md` section 14; brief:
`experiments/sparse_attention/CLOUD_E6_E9.md`. The `SUMMARY.md` of the first ladder is a record and is not rewritten. Code and records:
`experiments/sparse_attention/common/llm/training/` (`prep/`, `e6/` … `e9/`, each with the README committed before its first run, `step.json`, `run.sh`, `runs/`, `RESULTS.md`).
Every figure is **measured** in the qualified run named in the step's `RESULTS.md` unless marked **estimated**; nothing is promoted; no time or memory is claimed
(compute and memory are counted by `lib/count.py`). Setting: PyTorch on CPU, enwik8 bytes, `M = 128`, 8,192 bytes per step, WSD schedule, `τ = 0.075` bpb.

## Preparation and cost

44 new checks, 179 old checks (the 121 of the first ladder included) pass. A decay branch from a checkpoint equals the uninterrupted WSD run **bit for bit**; a cosine run
of the first ladder is bit-identical to its record (so the first ladder stays reproducible). The counter equals PLAN 14.5 for the seven models at both contexts; `IND_L4`
counted for the first time (133,120 at 128, 231,424 at 512). Wall time: preparation 0.2 h, E6 2.1 h, E7 4.2 h, E8 5.4 h, E9 7.2 h, about 19 h against the 18.5 h re-estimated after the
smoke (the plan said 16 h); the hash-routed arms use the selected form in this harness and are not faster per step than the dense arms.

## E6, the horizon, and the rule

`REF2` (peak 0.025) against `E4a_L4` (0.008), seeds 0 and 1, one WSD plateau to 6,400 and three decay branches. Rule frozen in the README: smallest of 2,000 and 4,000 with
`|gap(H) − gap(8,000)| ≤ 0.025` for both seeds. At 2,000 the deviations are 0.029 and 0.029 (fail); at 4,000 they are 0.014 and 0.011 (pass): **`H* = 4,000`**. The series went on.
The gap `E4a_L4 − REF2` is negative at every horizon (−0.020 to −0.067) and keeps widening in favour of the constrained model with the horizon by about 0.011-0.014 between 4,000 and 8,000:
the horizon of the series is therefore conservative for the claim that the constrained family does well.

## E7, the rates (seed 100, WSD, `H* = 4,000`, one extension each)

`REF1` 0.016, `REF4` 0.008, `E4a_L2` 0.008, `E4a_L6` 0.008 (none ended at an edge). Computed: `REF2` 0.0127, `REF3` 0.0101, `E4a_L4` 0.008, `IND_L4` 0.008. Several selections
are within the seed band of their neighbour (one seed: it selects, it proves nothing).

## The two frontiers (validation bits per byte, median over three seeds; compute from the counter)

| arm | inference MACs/token (128 / 512) | valid bpb 128 | valid bpb 512 | params | KV values/token |
|---|---|---|---|---|---|
| E4a_L2 | 138,240 / 187,392 | 2.2928 | 2.2889 | 161,664 | 512 |
| IND_L4 | 133,120 / 231,424 | 2.2391 | 2.2372 | 175,744 | 1,024 |
| REF1 | 157,696 / 256,000 | 2.3672 | 2.3498 | 93,184 | 256 |
| E4a_L4 | 243,712 / 342,016 | 2.1161 | 2.1076 | 323,200 | 1,024 |
| REF2 | 282,624 / 479,232 | 2.1475 | 2.1416 | 186,240 | 512 |
| E4a_L6 | 349,184 / 496,640 | 2.0398 | 2.0330 | 484,736 | 1,536 |
| REF3 | 407,552 / 702,464 | 2.0721 | 2.0139 | 279,296 | 768 |
| REF4 | 532,480 / 925,696 | 2.0265 | 1.9601 | 372,352 | 1,024 |

(The two bits-per-byte columns are not comparable across contexts. Test-split figures beside, all with the same orderings, are in the `RESULTS.md` of E8 and E9.)

## Readings, with outcome and class (`d_s = candidate − control`, positive = candidate worse)

| Reading | N = 128 (E8) | N = 512 (E9) | Class |
|---|---|---|---|
| `E4a_L2` / `REF1` (12 % / 27 % less compute) | −0.070 [−0.085, −0.067]: within, **dominates** | −0.061 [−0.065, −0.050]: within, **dominates** | measured |
| `E4a_L4` / `REF2` (14 % / 29 %) | −0.028 [−0.041, −0.019]: within, **dominates** | −0.028 [−0.050, −0.019]: within, **dominates** | measured |
| `E4a_L6` / `REF3` (14 % / 29 %) | −0.033 [−0.034, −0.025]: within, **dominates** | +0.009 [+0.009, +0.023]: within, **equivalent at less compute, not dominating** | measured |
| interpolated in log C: `E4a_L4`, `E4a_L6` | −0.085, −0.065: within, dominates | −0.145, −0.096: within, dominates | estimated |
| interpolated: `E4a_L2` | no value (below `REF1`) | no value | — |
| equal depth `E4a_L2` / `REF2` | +0.145 [+0.134, +0.157]: **beyond** | +0.145 [+0.141, +0.169]: **beyond** | measured |
| equal depth `E4a_L4` / `REF4` | +0.090 [+0.089, +0.099]: **beyond** | +0.162 [+0.131, +0.166]: **beyond** | measured |
| mixing at depth `IND_L4` / `E4a_L4` (IND worse by) | +0.123 [+0.115, +0.129]: **beyond** | +0.130 [+0.114, +0.143]: **beyond** | measured |
| equal training compute (b) | numerically the same interpolation (equal steps and tokens) | idem | estimated |
| E6 secondary: gap at equal training compute (`E4a_L4` vs `REF2`, unequal steps) | −0.039/−0.056 at 2,000, −0.050/−0.068 at 4,000 (seeds 0/1) | — | estimated |

**Effect of the context** (E9 against E8, cross-campaign reading, from the outcomes of the pairs only): the equal-depth loss and the mixing outcome are *beyond* at both contexts; the dominance of
`E4a_L2` and `E4a_L4` holds at both; the deepest pair `E4a_L6`/`REF3` goes from **dominates to equivalent within `τ`** at 512. The constraint saves more compute at 512 (27-29 % against 12-14 %) but
its quality advantage at equal compute does not extend to the deepest pair. At 512, all arms improve between the first 128 positions and the later ones, and the dense deep models (`REF3`, `REF4`) gain the most
(−0.127, −0.130 bpb) against the constrained ones (−0.067 to −0.073): descriptive, within E9.

## What the series says

- **At equal inference compute the constrained model (hash route, 2-6 layers) is better than the dense HFFN `d20` model in 5 of 6 registered pairs (all seeds, validation and test), and equivalent within `τ` in the sixth**, while
  costing 12-29 % less compute (counter). It loses quality **at equal depth** (+0.09 to +0.16 bpb, beyond `τ`). The family therefore buys depth with its saved compute and wins on the
  compute axis, in the range and under the protocol of this series (`M = 128`, enwik8, 4,000 steps, three seeds).
- **That win is bought with memory**: about 1.7 times the weights and AdamW states and twice the KV cache of the dense model it beats at equal compute (E8 and E9 `RESULTS.md`). The statement is about compute only.
- **The coupling is worth it at depth**: the four separated models (`IND_L4`) are worse than `E4a_L4` by 0.12-0.13 bpb at both contexts, at 32-45 % less compute and 46 % fewer parameters (not parameter-matched). At
  `L = 1` the first ladder found +0.003 (not met); at `L = 4` the coupling of heads and FFN groups carries a measurable gain.
- **The longer context is used mostly in its first 128 positions** by all arms; the constrained family uses it less than the deep dense models.

## Observations outside the registered readings (proposals for the user, not claims)

- `IND_L4` (133,120 MACs at 128) has lower bits per byte than `REF1` (−0.128) and `E4a_L2` (−0.054) at about the same compute (E8), and a lower bits per byte than `REF1` at 512 (−0.113) at less
  compute than `REF1` (231,424 against 256,000). A four-block model with block-diagonal keys and values is on the frontier's good side of the shallow models. This suggests that the saved compute is best spent
  on depth; a registered reading (IND against the dense family at equal compute, and an IND family over depth) would test it.
- The deepest pair at 512 is the one weak point; a wider constrained model or more heads per group would test the "2 heads of 8" explanation (PLAN section 10), but width is outside this series.
- Open from PLAN 13.5, waiting for the routing study: questions 1 (cost split between routed FFN and routed attention under the hash) and 5 (does the gain come from the byte content or from leaving the position route: a random balanced route).

## Not run, not verified, limits (PLAN 14.6 included)

- **Not run:** any time or memory measurement (decision 9); another routing, width, dataset or model size; a comparison with other efficient attention; a dense family narrower than `d20` or an unrouted FFN of a quarter of the neurons
  (question 4, second part); IND over depth; rates swept at the middle depths (computed) or at `N = 512` (kept from 128); `E4a_L4`/`REF2` branches at `N = 512`; more than 3 seeds (2 in E6); the learned router.
- **Not verified:** that the horizon `H* = 4,000` chosen for gap stability at `N = 128` also holds at 512; that the computed rates of `REF2`, `REF3`, `E4a_L4`, `IND_L4` are near-optimal; the cause of the weaker deepest pair
  at 512; thread-count dependence of the bits (all runs 4 jobs × 1 thread).
- **Limits of reading:** one validation split serves rate choice (seed 100) and figures (test split read once, beside, selects nothing); the middle-depth comparisons rely on computed rates and may bias slightly against the
  dense family or the constrained one (direction not measured); the constrained arms run in the selected form (wall time and memory in this harness are not evidence); `IND_L4` is not parameter- or compute-matched;
  conclusions hold for `M = 128` and enwik8 only.

## Harness events

None in this series (no check failure, no defect, no stopped run): every launch passed its checks and exited 0. Two things worth knowing: the `prep` smoke ran the checks twice (cost 3 minutes each, run by every launcher); a lost sentence in `e6/RESULTS.md`
due to shell quoting was restored in its own commit.
