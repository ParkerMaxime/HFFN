# sparse_attention, second series (E6-E9) on a Claude cloud session — brief

Written on 2026-10-25. `current/sparse_attention/PLAN.md` section 14 is the plan of this series; this file only says
how the cloud session runs it. Read `AGENTS.md` sections 3 and 7, the whole of `PLAN.md` (sections 13 and 14 above
all) and `experiments/sparse_attention/CLOUD.md` (the brief of the first ladder) first. Where this brief and
`PLAN.md` disagree, `PLAN.md` wins and the disagreement is reported to the user.

## What this session is

The same isolated cloud session as the first ladder, on the branch `sparse-attention-cloud`, from the commit that
follows E5 (`8f5deec45`, "PROGRESS: checks count after E5") and adds the plan of the series. Everything `CLOUD.md`
says about the freedom of the session and about what stays (one mechanism per step, README committed before the
first run, evidence classes, run records never edited, no time claim, nothing promoted) still holds. This brief adds:

- **The plan and this brief are already on the branch**: `current/sparse_attention/PLAN.md` with sections 13 and 14,
  and this file, committed by the user's local session on top of `8f5deec45`. Nothing comes from `main` (whose
  remote holds the published v1, a separate history). If `PLAN.md` on the branch has no section 14, **stop and tell
  the user**.
- **Cloud compute is limited** (user decision 13). Cost estimates of `PLAN.md` 14.4: about 16 hours for the whole
  series if E6 retains 4,000 steps. After the smoke of the preparation, re-estimate the total from the measured wall
  times; if it exceeds 24 hours, stop and report the estimate to the user before E6.
- **The frozen design** of `PLAN.md` 14.3 is not reopened by the session: routing (the balanced byte hash of E4a,
  fixed at every layer, no rotation), depth as the axis of compute, `M = 128`, the two contexts, 8,192 bytes per
  step, WSD, `τ = 0.075` bpb.

## Order of work

Chained, one step after the other, each step compared as `PLAN.md` 14.4 says:

```text
preparation -> E6 -> (stop if H* = 8,000) -> E7 -> E8 -> E9 -> summary
```

Each step has its folder `experiments/sparse_attention/common/llm/training/<step>/` (`prep`, `e6`, `e7`, `e8`, `e9`)
with `README.md`, `step.json`, `run.sh` (`check | smoke | run`) and `runs/`, as in the first ladder. Commit and push
after each README (before its first run) and after each `RESULTS.md`. Keep `PROGRESS.md` current (new rows below the
first ladder).

### 1. Preparation (`prep/`, no figure)

Changes to `lib/`, each covered by a check in `checks/` that exits non-zero on failure. **The first ladder must stay
reproducible**: every new behaviour is an option whose default is the behaviour of E0-E5 (cosine schedule, batch 64
at `N = 128`, evaluation by position class). The 121 existing checks still pass.

- **Arms** in `arms.py`: `REF1` (= `E0`), `REF2`, `REF3` (`REF4` exists), `E4a_L2`, `E4a_L6` (`E4a_L4` exists),
  `IND_L4` (`IND` with `routing="hash"`, `n_layers=4`), and the suffix `_N512` (`context=512`) for each of the eight.
  Check: parameter counts outside the embedding equal to `PLAN.md` 14.5 (`REF2` 186,240, `REF3` 279,296, `REF4`
  372,352, `E4a_L2` 161,664, `E4a_L4` 323,200, `E4a_L6` 484,736) and independent of the context; `IND_L4` under the
  hash keeps every support of section 8 (supports inside `S_g`, degree 26, `o_t` zero outside `S_g`).
- **Tokens per step.** The batch becomes `8192 // context` (64 at 128, 16 at 512). Check: 8,192 bytes per step at
  both contexts; at a given context and seed, every arm draws the same batches in the same order.
- **WSD schedule** (`--schedule wsd`, default `cosine`): 200 warm-up steps, constant peak, linear decay to zero over
  the last 20 % of the horizon `H` (from `0.8 H` to `H`).
- **Plateau checkpoints and decay branches.** A plateau run saves, at given steps, the full training state: model,
  AdamW state, step, position of the batch stream, RNG states. A branch restarts from a checkpoint at `0.8 H` and runs
  the decay to `H`. Check: on a short horizon (for example `H = 100`, checkpoint at 80), a branch from the checkpoint
  gives the same weights and the same final bits per byte as one uninterrupted WSD run of horizon `H`, within float32
  rounding (state the tolerance in the README of `prep`). Checkpoints stay out of git.
- **Evaluation.** Final validation over the whole valid split, as now. At `N = 512`, also the bits per byte by
  position range 0-127, 128-255, 256-511. Check: the mean of the ranges weighted by their counts equals the whole
  figure. The intermediate evaluations during training use a fixed number of bytes (4096 × 128 = 524,288 bytes,
  hence 1,024 windows at 512), not a fixed number of windows. `--test` (the test split) is used only in E8 and E9.
- **Counter** `lib/count.py`, per arm and context: inference multiply-adds per token (convention of `PLAN.md` section
  6, `N_k = N`, logits included), training multiply-adds per token (forward plus backward, convention stated in the
  file: three times the forward, an estimate), parameters and AdamW states outside the embedding, KV cache per token
  (`2 L M` values), training activations per token (queries, scores, attention outputs, FFN activations). Check: the
  inference column equals `PLAN.md` 14.5 for the seven models it lists, at both contexts, exactly. `IND_L4` is
  counted here for the first time; its count goes into the README of `prep`.
- **Smoke** of every arm (the eight at 128 and the eight `_N512`), 40 steps under WSD, plus one plateau-and-branch
  pair. It proves nothing about quality; it sizes the runs (wall time per step for each arm, 4 workers × 1 thread as
  in the first ladder) and gives the re-estimate of the total.

### 2. E6, the horizon (`e6/`)

`PLAN.md` 14.4, E6. Pair `REF2` (peak 0.025, the geometric interpolation of 0.032 at `L = 1` and 0.016 at `L = 4`)
and `E4a_L4` (peak 0.008), `N = 128`, seeds 0 and 1.

- One WSD plateau run per arm and seed to step 6,400, checkpoints at 1,600, 3,200 and 6,400; one decay branch from
  each, ending at 2,000, 4,000 and 8,000. Twelve final figures.
- **Rule, frozen in the README:** `gap_s(H) = bpb(E4a_L4) − bpb(REF2)` for seed `s` at horizon `H`. `H*` is the
  smallest of 2,000 and 4,000 such that `|gap_s(H) − gap_s(8,000)| ≤ 0.025` bpb for both seeds. Otherwise
  `H* = 8,000`.
- **If `H* = 8,000`, the series stops after E6**: write `RESULTS.md` and `PROGRESS.md`, push, and report to the user,
  who decides on the budget of E7-E9.
- The README states as limits: the rates were chosen under cosine in the first ladder; two seeds.
- Secondary reading (b), estimated: the gap at equal training compute, from the three horizons of the pair and the
  training column of the counter. One pair only.

### 3. E7, the learning rates (`e7/`)

`PLAN.md` 14.4, E7. At `H*`, WSD, `N = 128`, seed 100, one plain WSD run per arm and rate (no branches).

- Arms and the centres of their grids (estimated from the first ladder, rounded to the nearest power of two):
  `REF1` 0.032 (E0), `REF4` 0.016 (E5), `E4a_L2` 0.016 (geometric interpolation in `L` of 0.032 at `L = 1` and 0.008
  at `L = 4` gives 0.020), `E4a_L6` 0.004 (the same extrapolated gives 0.0032).
- Grid: the centre, half and double (3 rates). If the best rate is at an edge, add the next rate (× 2 or ÷ 2) in that
  direction, at most twice per arm; if it is still at an edge, keep the edge and state it as a limit.
- Selection: the lowest final validation bits per byte. One seed: it selects, it proves nothing.
- Rates of the other arms, computed and written in the README of E8 before its runs: `REF2`, `REF3` geometric in `L`
  between `REF1` and `REF4`; `E4a_L4` geometric between `E4a_L2` and `E4a_L6`; `IND_L4` the rate of `E4a_L4`.

### 4. E8, the frontier at `N = 128` (`e8/`)

`PLAN.md` 14.4, E8. Eight arms `REF1`, `REF2`, `REF3`, `REF4`, `E4a_L2`, `E4a_L4`, `E4a_L6`, `IND_L4`, seeds 0, 1, 2,
WSD at `H*`, rates of E7. `--test` on every run (read once, reported beside, selects nothing).

Readings, every one paired per seed, `d_s = bpb(candidate) − bpb(control)`, `τ = 0.075` bpb, the three outcomes of
`PLAN.md` 13.1 (within: `max_s d_s ≤ τ`; beyond: `min_s d_s > τ`; otherwise inconclusive), and *dominates*
(`max_s d_s < 0`) wherever the candidate costs less inference compute than the control:

| Reading | Candidate / control | Class |
|---|---|---|
| Equal compute (dominance pairs) | `E4a_L2`/`REF1`, `E4a_L4`/`REF2`, `E4a_L6`/`REF3` | measured |
| Equal compute (interpolated) | each `E4a_L<L>` against the dense family interpolated linearly in `log C` at its compute, inside the range of the family only (`E4a_L2` lies below `REF1` at `N = 128`: no value) | estimated |
| Equal depth | `E4a_L2`/`REF2`, `E4a_L4`/`REF4` | measured |
| Mixing at depth | `IND_L4`/`E4a_L4` | measured |
| Equal training compute (b) | the same pairs, against the training column of the counter | estimated |

`RESULTS.md` also gives the frontier itself: per arm, the median and [min–max] over seeds of validation and test bits
per byte, against inference and training multiply-adds, parameters and KV cache per token (from the counter). The
memory paragraph of `PLAN.md` 14.5 is written with the result, not after it: a favourable frontier is a statement
about compute bought with more weights and a larger KV cache.

### 5. E9, the frontier at `N = 512` (`e9/`)

The same as E8 with the eight `_N512` arms and the rates of E8 (kept at 512, a hypothesis and a limit), plus the
bits per byte by position range. Its pairs are read inside E9. The effect of the context is read only from the
outcomes of the pairs of each campaign (E8 against E9), never from differences of figures across the two campaigns,
and is stated as a cross-campaign reading.

### 6. Summary

`experiments/sparse_attention/SUMMARY_E6_E9.md` (the `SUMMARY.md` of the first ladder is a record and is not
rewritten): `H*` and its rule, the rates, the two frontiers, every reading with its outcome and class, what was not
run or not verified, and the limits of `PLAN.md` 14.6.

## What the session does not do

- No new arm, routing, width, context or dataset beyond `PLAN.md` 14.4; an idea that comes up goes into `SUMMARY_E6_E9.md`
  as a proposal for the user.
- No relaxed threshold, no change of `τ`, of the E6 rule or of a grid after its step has run; a change is the user's
  amendment and keeps the earlier verdict.
- No time or memory measurement presented as a result (decision 9): wall times only size the runs.
- A failed or partial run gives no figure, only its cause. A harness failure (a defect of `lib/`, a launcher, a check
  or the environment, while no arm was judged) is not an attempt: its record and cause go into the `RESULTS.md` of
  the step, the defect is fixed, the criteria stay frozen, and the step is relaunched.
- Nothing is promoted; no figure of this branch enters `current/` or `publication/` without the user.

## Reporting

Repository files in English, messages to the user in French. Report plainly: failures with their output, skipped
steps as skipped, harness defects apart from candidate failures, and the stop after E6 or after the re-estimate when
it applies.
