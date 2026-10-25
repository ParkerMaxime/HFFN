# sparse_attention — summary of the ladder (cloud session of 2026-10-24/25)

Branch `sparse-attention-cloud`. Plan: `current/sparse_attention/PLAN.md`; brief: `experiments/sparse_attention/CLOUD.md`.
Code: `experiments/sparse_attention/common/llm/training/` (`lib/` one model and one config per arm, `checks/`, one folder per
step with README written and committed before its first run, `run.sh check|smoke|run`, `step.json`, `runs/`, `RESULTS.md`).
Every figure below is **measured** in the qualified run named in the step's `RESULTS.md` (status `passed`, exit 0, full
validation split, 4 seeds), unless marked otherwise. Nothing here is a promotion: no figure enters `current/` or
`publication/` without the user. No time or speed is claimed; compute savings are analytical (PLAN section 6).

Setting: PyTorch (2.14.1, PyPI build on CPU), enwik8 bytes, one layer (four for E5), `M = 128`, context 128, `G = 4`, 4,000 steps,
batch 64, seeds 0-3, validation bits per byte over all positions of the 5 M validation bytes (not comparable to published
enwik8 figures). Pre-training checks: 121 checks, all passing (parameter counts equal to the PLAN table, supports and degrees,
`o_t` exactly zero outside `S_g`, FFN independent of `u_t[j]`, `j ∉ S_g`, packed = selected, causality, determinism).

## The ladder

Bound `τ` = 0.075 bpb (5 % of `Δ_attn`) on the paired difference per seed, fixed before each run. "within" = worst seed ≤ τ;
"beyond" = best seed > τ; "inconclusive" = seeds straddle τ. Positive difference = candidate worse.

| Step | Mechanism (control) | Params | bpb median [min–max] (candidate) | Paired difference, median [range] | Verdict |
|---|---|---|---|---|---|
| E0 | reference dense / no attention | 93,184 / 27,520 | 2.3925 [2.3913–2.3975] / 3.8949 | `Δ_attn` = 1.503, band 0.006, ratio 243 (R_min 10) | **discriminating** |
| E1 | FFN inputs inside blocks (E0) | 93,184 | 2.3878 [2.3867–2.3936] | −0.0045 [−0.0088, −0.00002] | within |
| E2 | top-1 FFN routing (E1) | 93,184 | 2.6418 [2.6392–2.6472] | +0.2516 [+0.2504, +0.2605] | **beyond** |
| E3a | block-diagonal `W_O` (E2) | 80,896 | 2.7572 [2.7546–2.7606] | +0.1160 [+0.1104, +0.1173] | **beyond** |
| E3b | attention routed, complete constraint (E3a) | 80,896 | 2.8692 [2.8627–2.8729] | +0.1131 [+0.1021, +0.1160] | **beyond** |
| IND | four separated models (E3b), mixing worth it if E3b better by > τ | 44,032 | 2.8739 [2.8704–2.8821] | E3b better by +0.0031 [+0.0007, +0.0194] | mixing **not worth it** at L = 1 |
| E4a | balanced byte hash route (E3b) | 80,896 | 2.6393 [2.5520–2.6778] | −0.2264 [−0.3209, −0.1919] | within, better in every seed (wide seed spread) |
| E4b | learned top-1 router (E3b) | 81,408 | 2.9503 [2.8474–3.0360] | +0.0825 [−0.0223, +0.1673] | **inconclusive** (bimodal) |
| E5 | depth `L = 4`, hash route (REF4, dense `d20`) | 323,200 vs 372,352 | 2.1252 [2.1204–2.1347] vs 2.0337 | +0.0932 [+0.0859, +0.1022] | **beyond** (much smaller than at L = 1) |

E5 side results (same campaign): position routing at `L = 4` against `REF4` +0.3804 (beyond); hash against position at `L = 4`
−0.2900 (better in every seed); rotation against fixed: hash −0.0038 (within), position −0.0031 (within, below in all four
seeds but of the order of the seed band).

## What the ladder says

- **The task discriminates** (E0): attention is worth 1.5 bpb here and the seed band is 0.006, so differences of a few hundredths
  can be read.
- **Cost of the complete constraint at one layer with position routing: +0.477 bpb** over dense attention (sum of the paired
  medians E1 to E3b; E3b 2.8692 against E0 2.3925), 32 % of `Δ_attn`. About half comes from routing the FFN to one quarter of its
  neurons (E2), a quarter from the block-diagonal `W_O` (E3a, which also removes 13 % of the parameters), a quarter from routing
  the attention heads (E3b).
- **The mixing paths bring nothing measurable at one layer**: a model with no cross-block path and 46 % fewer parameters (IND)
  is 0.003 bpb behind E3b. The model uses them (zeroing them costs several bpb) but obtains no quality from them over four
  separated models at `L = 1`. Whether the mixing helps between layers is **not measured** (no IND at `L = 4`).
- **The route matters more than the structure**: a static function of the token's byte (balanced byte hash) instead of
  `t mod 4` recovers 0.226 bpb at `L = 1` (E4a, back at the level of E2) and 0.290 at `L = 4`, at no extra parameter. The
  learned router specified here (E4b) did not reproduce it: balanced (load 0.19-0.28, balance loss ≈ 1.0, 1.6-4.6 % dropped) but
  bimodal across seeds, on average worse than position routing. Why the hash helps is a hypothesis (each group sees a consistent
  part of the byte vocabulary instead of an arbitrary quarter of the positions); not isolated.
- **Depth narrows the cost**: the hash-routed constrained model is +0.093 bpb above the dense reference at `L = 4` against +0.247
  at `L = 1`; position routing +0.380 against +0.477. The remaining gap at `L = 4` is just beyond the bound fixed beforehand.
- **Rotation of the route across layers changes nothing measurable** (a few thousandths, within the seed band).
- **Analytical compute (estimated, PLAN section 6)**: 52,736 against 124,928 multiply-adds per token and layer at the working size
  (ratio 2.37, ceiling 3.8 with dense K and V). The quality figures above are the price paid in this setting; no speed was measured.

## Rules, events and defects kept in the records

- E0: first execution of the checks failed one check on a tolerance set too tight (1e-6, set to 1e-5 before any training; harness
  defect). The first E0 run (`e0/runs/run_20261024T180451Z`) was stopped after its sweep because both optima were at the edge of
  the grid, per the README rule (`ABORTED.md`); the rerun on the extended grid (`run_20261024T182902Z`) is the qualified run. The 12
  common sweep points reproduce bit for bit. The control of every step reproduces the previous step's figures to the last digit.
- E5: the learning rate was swept again at `L = 4` (seed 100, 1,500 steps); the optima differ (0.016 for `REF4`, 0.008 for the constrained
  arms) and were applied by the README rule.
- The PLAN table counts "≈ 27,648" parameters without attention; the implementation has 27,520 (the removed attention takes its norm
  gain with it).
- `pip` could not reach the CPU wheel index of PyTorch (403 from the proxy); the PyPI build was installed and run on CPU.

## Not run, not verified, limits

- **Not run:** a learned router per layer at `L = 4` (E5 README: E4b was not better than position routing); IND at `L = 4`; a
  parameter-matched IND; a random (content-free) balanced route to separate "content" from "balanced assignment" in E4a; any
  other learning rate or router configuration for E4b (one attempt, as the brief states); the test split (never read); a depth
  other than 4; a context other than 128; another dataset; any timing.
- **Not verified:** that the optimum learning rate chosen at 1,500-2,000 steps and seed 100 holds for 4,000-step runs; thread-count
  dependence of the bits (all runs: 4 jobs × 1 thread); the literature statements of PLAN section 9 (cited from memory there).
- **Limits of reading:** four seeds per arm; steps compare arms of different parameter counts by construction (E3a, IND, E5); the
  validation split serves both the rate choice and the figures; E4a has a seed range of 0.126 (the paired band of its comparison
  is the widest of the ladder, 0.129); the supports change with the seed, so paired differences include support variance.
- **Nothing here is a promotion** and nothing beyond this branch was changed.
