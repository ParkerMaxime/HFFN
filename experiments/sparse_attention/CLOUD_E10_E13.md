# sparse_attention, third series (E10-E13) on a Claude cloud session — brief

Written on 2026-10-26. `current/sparse_attention/PLAN.md` section 15 is the plan of this series; this file only says
how the cloud session runs it. Read `AGENTS.md` sections 3 and 7, the whole of `PLAN.md` (sections 14.8 and 15 above
all), `experiments/sparse_attention/CLOUD.md` and `experiments/sparse_attention/CLOUD_E6_E9.md` (the briefs of the
first two series) first. Where this brief and `PLAN.md` disagree, `PLAN.md` wins and the disagreement is reported to
the user.

## What this session is

The same isolated cloud session as the first two series, on the branch `sparse-attention-cloud`, from the commit that
follows `d2ab42ed2` ("Add SUMMARY_E6_E9.md and complete PROGRESS.md for the second series") and adds the plan of this
series. Everything `CLOUD.md` and `CLOUD_E6_E9.md` say about the freedom of the session and about what stays (one
mechanism per step, README committed before the first run, evidence classes, run records never edited, nothing
promoted) still holds, with one change: **relative time is now admitted** under decision 14 (`PLAN.md` 15.2), in E11
only and under its rule. This brief adds:

- **The plan and this brief are already on the branch**: `current/sparse_attention/PLAN.md` with sections 14.8 and 15,
  and this file, committed by the user's local session on top of `d2ab42ed2`. Nothing comes from `main`. If `PLAN.md`
  on the branch has no section 15, **stop and tell the user**.
- **Cloud compute is limited.** `PLAN.md` 15.4 estimates about 17 to 20 hours. After the smoke of the preparation,
  re-estimate the total from the measured wall times; if it exceeds 24 hours, run E10 and E11, then stop and report
  the estimate to the user before E12.
- **The frozen design** of `PLAN.md` 15.2 and 15.3 is not reopened by the session: the hash route of E4a, `G = 4`,
  dense `K` and `V`, `H* = 4,000`, WSD, 8,192 bytes per step, `τ = 0.075` bpb, the width `M = 256` at `N = 512` with
  its two head layouts, the timing rule of E11.

## Order of work

```text
preparation -> E10 -> E11 -> (stop if the re-estimate exceeds 24 h) -> E12 -> E13 -> summary
```

Each step has its folder `experiments/sparse_attention/common/llm/training/<step>/` (`prep3`, `e10`, `e11`, `e12`,
`e13`) with `README.md`, `step.json`, `run.sh` (`check | smoke | run`) and `runs/`, as before. Commit and push after
each README (before its first run) and after each `RESULTS.md`. Keep `PROGRESS.md` current (new rows below the second
series).

### 1. Preparation (`prep3/`, no figure)

Changes to `lib/`, each covered by a check in `checks/` that exits non-zero on failure. **The first two series must
stay reproducible**: every new behaviour is an option whose default is the current behaviour (`selected` form under
the hash route, no executor, no memory or timing instrumentation). The existing checks still pass.

- **Packed form under the hash route** (`PLAN.md` 15.3). Gather the tokens of each group once per forward; per layer,
  keys and values of `H_g` for every token, then queries, scores (causal mask on the original positions), outputs,
  `W_O^g` and FFN group `g` on the routed tokens, scattered back. For `IND`, the same restricted to `S_g`. Check, for
  `E4a_L2`, `E4a_L4`, `IND_L4` at `N = 128` and `512` and for `E4a_L4_W256` and `E4a_L4_W256_h16`: outputs and the
  gradient of every parameter equal to the `selected` form within float32 rounding (state the tolerance in the
  README), on batches of real bytes. The packed form is used in E10, E11 and for the hash-routed arms of E12-E13.
- **Arms of width 256** in `arms.py`: `REF1_W256`, `REF2_W256`, `E4a_L2_W256`, `E4a_L4_W256` (`d_model=256`,
  `hidden=1024`, `degree=52`, `context=512`, 8 heads), and `REF2_W256_h16`, `E4a_L4_W256_h16` (16 heads). Check: the
  counter gives exactly the table of `PLAN.md` 15.5 (inference multiply-adds, parameters outside the embedding, KV
  values, training activations); the two layouts of a family have equal parameters and multiply-adds.
- **Memory harness** for E10 (`PLAN.md` 15.3, quantities (a) to (d)). Check on a synthetic module whose saved bytes
  and peak live bytes are known by hand: (a) and (b) give them exactly; two fresh processes give identical (a).
- **Executor** for E11 (`PLAN.md` 15.3): four workers, one intra-op thread each, one barrier per layer for `E4a`,
  none through depth for `IND`, and the dense control under the same executor (heads `H_g` over every token, barrier,
  `W_O` and FFN on `N / G` tokens per worker). Choose the method (TorchScript `fork`/`wait` or a C++ extension with
  `at::parallel_for`) and state it in the README. Checks: every arm under the executor gives the same outputs as its
  sequential forward within float32 rounding; the **concurrency gate** (four equal independent tasks of the size of
  cell a4 in at most 0.4 times their sequential time, median of five repetitions) is run and its result written in
  the README of `prep3`. If it fails, E11 is not run; say so and go on.
- **Timing harness** for E11 (`PLAN.md` 15.3, the timing rule): fresh process per cell-process, arms control, copy,
  variant built independently (the copy is a second instance of the control with the same weights), warm-up, 12
  windows of `control, copy, variant, variant, copy, control`, steal and user time per window, load average,
  `lscpu`, `torch` configuration. It writes `raw.jsonl` (one line per slot) and computes the ratios only from it.
- **Smoke** of every arm of E12-E13 (40 steps, WSD) and of every cell of E11 (one process, 2 windows), and of E10 on
  two arms. It proves nothing; it sizes the runs and gives the re-estimate of the total. It also times the two dense
  executions of E11 (the same executor and PyTorch's intra-op parallelism on four threads); **the faster one becomes
  the control of E11**, written in the README of `e11` before its run.

### 2. E10, memory (`e10/`)

`PLAN.md` 15.4, E10. Arms: the eight of E8 at `N = 128` and the eight of E9 at `N = 512` (hash-routed arms in packed
form), `E4a_L4` and `E4a_L4_N512` also in the `selected` form, and the six arms of E13. One training step of 8,192
bytes of the train split per arm, random initialisation of the arm (seed 0), each quantity in two fresh processes.

- Rule, frozen in the README: (a) identical in the two processes, (b) and (c) within 1 %; otherwise a harness
  failure.
- `RESULTS.md`: per arm (a) split token-side / weight-side, (b), (c) per token, (d) descriptive, beside the counter;
  the ratios of `PLAN.md` 15.4 (pairs at equal counted multiply-adds, at equal depth, `selected` against packed),
  measured beside counted. Weights, AdamW states and KV cache are given from the counter and labelled counted.

### 3. E11, relative time (`e11/`)

`PLAN.md` 15.4, E11. The eight cells a1-a4, b1-b2, c1-c2, five fresh processes each, under the timing rule. **Nothing
else runs on the machine during E11**: no training, check, build or second process of the session.

- `RESULTS.md`: per cell `r` median [min–max] over five processes, `delta` of the run, the outcome (faster, slower or
  not resolved), the counted ratio of multiply-adds of the cell and the quotient of the two (descriptive), the steal
  time observed, and the phases of the untimed pass (descriptive).
- Write the limit with the result: a ratio of this PyTorch executor on this shared machine, forward only, with a
  masked dense FFN in every arm. No absolute time in any file meant as a result (the raw slots keep their seconds).

### 4. E12, the rates at width (`e12/`)

`PLAN.md` 15.4, E12. `REF2_W256` (centre 0.008) and `E4a_L4_W256` (centre 0.004), WSD at 4,000 steps, `N = 512`, seed
100, grid centre, half, double, extended at most twice at an edge. Selection: the lowest final validation bits per
byte. One seed: it selects, it proves nothing. The rates of the four other arms are computed as `PLAN.md` 15.4 says
and written in the README of E13 before its runs.

### 5. E13, the frontier at width (`e13/`)

`PLAN.md` 15.4, E13. Six arms, seeds 0, 1, 2, WSD at 4,000 steps, `--test` on every run, bits per byte by position
range. The readings of the table of `PLAN.md` 15.4 with their rules; the heads-per-group reading `i_s` with its three
outcomes. `RESULTS.md` also gives the frontier itself (median and [min–max] over seeds of validation and test bits
per byte, against the counted multiply-adds, parameters, KV cache and training activations per token) and, beside,
the measured memory of E10 for the same arms. Agreement with the outcomes of E9 is a cross-campaign reading, stated as
such; figures are not compared across campaigns.

### 6. Summary

`experiments/sparse_attention/SUMMARY_E10_E13.md` (earlier summaries are records and are not rewritten): what E10
measured against the counter, the ratios of E11 with their `delta` and limits, the rates of E12, every reading of E13
with its outcome and class, what was not run or not verified, and the limits of `PLAN.md` 15.6.

## What the session does not do

- No new arm, routing, `G`, width, context or dataset beyond `PLAN.md` 15.4; no specialised FFN groups (decision 15).
  An idea that comes up goes into `SUMMARY_E10_E13.md` as a proposal for the user.
- No absolute time presented as a result; no ratio compared with another machine, another campaign or the HFFN
  runtime; no timing outside E11; no change of the timing rule, of a grid or of a reading after its step has run.
- No relaxed threshold, no change of `τ`; a change is the user's amendment and keeps the earlier verdict.
- A failed or partial run gives no figure, only its cause. A harness failure (a defect of `lib/`, a launcher, a check,
  the executor or the environment, while no arm was judged) is not an attempt: its record and cause go into the
  `RESULTS.md` of the step, the defect is fixed, the criteria stay frozen, and the step is relaunched. A failed
  concurrency gate is not a harness failure: it is a property of the machine or the executor, reported as such.
- Nothing is promoted; no figure of this branch enters `current/` or `publication/` without the user.

## Reporting

Repository files in English, messages to the user in French. Report plainly: failures with their output, skipped
steps as skipped, harness defects apart from candidate failures, the result of the concurrency gate, and the stop
after E11 when the re-estimate applies.
