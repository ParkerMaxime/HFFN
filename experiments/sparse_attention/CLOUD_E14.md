# sparse_attention, fourth step (E14) on a Claude cloud session — brief

Written on 2026-10-27. `current/sparse_attention/PLAN.md` section 17 is the plan of this step; this file only says how
the cloud session runs it. Read `AGENTS.md` sections 3 and 7, `PLAN.md` (sections 15.8, 16 and 17 above all),
`experiments/sparse_attention/CLOUD.md` and `experiments/sparse_attention/CLOUD_E10_E13.md` (the brief of the third
series, whose rules still hold) first. Where this brief and `PLAN.md` disagree, `PLAN.md` wins and the disagreement is
reported to the user.

## What this session is

The same isolated cloud session as the first three series, on the branch `sparse-attention-cloud`, from the commit
that follows `440da1e17` ("Summary E10-E13: state the E13 overrun without an unverified cause") and adds the plan of
this step. Everything `CLOUD.md`, `CLOUD_E6_E9.md` and `CLOUD_E10_E13.md` say about the freedom of the session and
about what stays (one mechanism per step, README committed before the first run, evidence classes, run records never
edited, nothing promoted) still holds. This brief adds:

- **The plan and this brief are already on the branch**: `PLAN.md` with sections 15.8, 16 and 17 (17.1 to 17.7), and
  this file, committed by the user's local session on top of `440da1e17`. Nothing comes from `main`. If `PLAN.md` on
  the branch has no section 17.7, **stop and tell the user**.
- **No training.** E14 measures time and memory only. The quality axis is the validation bpb per seed already on
  record: `e8/runs/run_20261025T163405Z` (`N = 128`), `e9/runs/run_20261025T220720Z` (`N = 512`),
  `e13/runs/run_20261027T040908Z` (`M = 256`). These records are read, never edited.
- **Cloud compute is limited.** `PLAN.md` 17.5 estimates about 4.5 to 5.5 hours. After the smoke, re-estimate the total
  from the measured durations; above 6 hours, cut the cells at `M = 256` first (frontier and pairs p5-p6), say so in
  the README of `e14` before its runs, and tell the user.
- **The frozen design** of `PLAN.md` 17.2, 17.3 and 17.7 is not reopened: the arms and anchors, the two modes
  (prefill at 8,192 tokens per call; decoding with `B = 64` and a cache of `N / 2`), the depth `L = 4` of the pairs,
  the executor of E11 with the executor as dense control, the timing rule of 15.3, the readings of 17.4 and 17.7,
  `τ = 0.075` bpb.

## Order of work

```text
prep4 -> e14 prefill -> e14 decoding -> e14 memory -> analysis -> summary
```

Folders `experiments/sparse_attention/common/llm/training/prep4/` and `.../e14/`, each with `README.md`, `step.json`,
`run.sh` and `runs/`, as before. `e14/run.sh` has the modes `check | smoke | prefill | decode | memory`; each of the
last three is a run of its own (own folder, `EXIT`, `manifest.json`). Commit and push after each README (before its
first run) and after each `RESULTS.md`. Keep `PROGRESS.md` current (a section "Fourth step (E14; plan: PLAN.md
section 17; brief: CLOUD_E14.md)" below the third series).

### 1. Preparation (`prep4/`, no figure)

Changes to `lib/`, each covered by a check in `checks/` (a new `check_e14.py`) that exits non-zero on failure. **The
first three series stay reproducible**: every new behaviour is an option whose default is the current behaviour; the
existing checks still pass.

- **Arms** in `arms.py` (`PLAN.md` 17.7): `E2h_L4 = E2.with_(name="E2h_L4", n_layers=4, routing="hash")`, its
  `_N512` and `_W256` versions, and `REF4_W256`. `REF4_N512` exists. Check: `E2h_L4` has the parameters of `E4a_L4`
  plus 12,288 per layer (`W_O`) at `M = 128`, and 49,152 per layer at `M = 256`; the counter (`count.py`) gives each
  new arm its inference multiply-adds, parameters, KV values and training activations, written in the README of
  `prep4`.
- **Packed form of `E2h`** (`model.py`, `impl="gather"`): dense attention over every token, then `W_O` and the FFN of
  group `g` on the tokens routed to `g` with the gather of `E4a`. Check: outputs and the gradient of every parameter
  equal to the `selected` form within float32 rounding, at the three sizes, on batches of real bytes.
- **Executor** (`executor.py`): a kind for `E2h`, the heads as in the dense kind (`run_dense_heads`), barrier, then
  `W_O` (dense) and FFN group `g` on worker `g` for the tokens routed to `g`, padded to the largest group count as for
  `E4a`. Check: equal to the sequential forward within float32 rounding at the three sizes.
- **Decoding path** (`PLAN.md` 17.3, new): a KV cache per layer and per sequence for every arm of 17.3 and 17.7, in
  the model (sequential) and under the executor, with the per-worker split of 17.3 and 17.7. The cache is filled by an
  untimed forward of the first `N / 2` positions; the timed step writes the keys and values of the new token into a
  preallocated slot at position `N / 2`, so that repeated calls do the same work. Checks: the logits of the decoded
  step equal those of the full forward at position `N / 2` within float32 rounding (tolerance stated in the README),
  for every arm; the decoded step under the executor equals the sequential decoded step; the per-group counts of the
  64 new tokens are recorded.
- **Timing harness** (`timing.py`): cells of `PLAN.md` 17.3 and 17.7 in both modes, the anchor, its copy and the
  variant in the same process, as in E11. Record which SDPA kernel each arm runs in prefill (`PLAN.md` 17.7, limits),
  from the operator names of a profiled call.
- **Memory harness** (`memory.py`): the quantities (a)-(c) of E10 for `E2h` in the packed form, and (e) the peak of
  live tensor bytes of one decoded step (`B = 64`, cache of `N / 2`) by the method of (c), checked on the synthetic
  module of prep3.
- **Smoke** of every cell in both modes (one process, 2 windows) and of the memory run on two arms. It proves nothing;
  it sizes the runs and gives the re-estimate of the total. The concurrency gate of 15.3 runs at the start of the
  smoke and of each timing run; if it fails, the timing runs are not made: say so and stop.

### 2. E14, the frontier and the pairs (`e14/`)

The README, committed before the first run, freezes from `PLAN.md` 17.3, 17.4 and 17.7: the cells, the timing rule,
the readings and their outcomes, the tolerances of the checks, the SDPA kernels observed in the smoke, the re-estimate
of the total and any cut.

- **Prefill run, then decoding run.** Per mode, 19 frontier cells (17.3: anchor `REF2` at `N = 128` and `N = 512`,
  `REF2_W256` at `M = 256`) and 6 pair cells (17.7: p1-p6), five fresh processes each, one `delta` per run over all its
  cell-processes. **Nothing else runs on the machine during a timing run**: no check, build, smoke or second process.
- **Memory run.** The nine arms of p1-p6, (a)-(c) and (e), two fresh processes each, the rule of E10 (a identical, the
  others within 1 %).
- **Analysis** (new files beside the runs, `analysis.json` and `analysis.md`), then `RESULTS.md`:
  - per mode and campaign: the dense frontier per seed, the gap `g` of every constrained arm with `g⁺` and `g⁻`, its
    outcome (17.4), and the answer to decision 16 (*favourable*, *unfavourable*, *not resolved*); the two modes stated
    apart;
  - per pair cell and mode: the ratio median [min–max], `delta`, the outcome (gain, loss, or no difference within
    `delta`), the counted ratio and the quotient, and the products p1 × p2, p3 × p4, p5 × p6 labelled estimated;
  - memory: the ratios `E4a` / `E2h` and `E2h` / `REF4` measured beside counted; parameters, AdamW states and the KV
    cache counted;
  - classes: times and memory measured (E14), bpb measured (E8, E9, E13), every gap estimated (17.4, *Classes*).

### 3. Summary

`experiments/sparse_attention/SUMMARY_E14.md` (earlier summaries are records and are not rewritten): the answer to
decision 16 per mode and campaign with its gaps, the gain of the attention alone and of the routing of the FFN alone
per mode and context, with `delta` and limits, what was cut, not run or not verified, and the limits of `PLAN.md` 17.6
and 17.7.

## What the session does not do

- No training, no new arm beyond `PLAN.md` 17.3 and 17.7 (in particular `E2h` is not trained: its quality is a
  decision of the user, 17.7), no other depth, batch size, cache length, width or context; no specialised FFN groups
  (decision 15). An idea that comes up goes into `SUMMARY_E14.md` as a proposal for the user.
- No absolute time presented as a result; no ratio compared with another machine, another campaign (E11 included) or
  the HFFN runtime; no change of the timing rule, of a cell or of a reading after its run.
- No relaxed threshold, no change of `τ`; a change is the user's amendment and keeps the earlier verdict.
- A failed or partial run gives no figure, only its cause. A harness failure (a defect of `lib/`, a launcher, a check,
  the executor, the decoding path or the environment, while no arm was judged) is not an attempt: its record and cause
  go into the `RESULTS.md` of the step, the defect is fixed, the criteria stay frozen, and the run is relaunched. A
  failed concurrency gate is not a harness failure: it is a property of the machine or the executor, reported as such.
- Nothing is promoted; no figure of this branch enters `current/` or `publication/` without the user.

## Reporting

Repository files in English, messages to the user in French. Report plainly: failures with their output, skipped
steps as skipped, harness defects apart from candidate failures, the result of the concurrency gate, the re-estimate
and any cut.
