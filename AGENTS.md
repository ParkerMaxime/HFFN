# HFFN — repository rules

This repository studies **Hidden-FFN (HFFN)**: a way to store, execute, train and update fixed-support sparse
feed-forward networks by hidden neuron. This file is the single source of rules. Branch-specific reminders live in
`.claude/rules/` and add to it; they never override it. The date of each user decision and the runs that prompted it
are in `DECISIONS.md`. All rules, READMEs and code comments are written in English.

## 1. What the project is

The goal is not a new Transformer architecture, a new pruning method or a faster dense mask. It is to show that a
sparse FFN with a fixed support can be treated as a set of **independent hidden neurons**, and to keep that
structure through the whole runtime instead of rebuilding masked dense matrices at every step.

The favourable regime is a sufficiently sparse FFN, fixed support, on hardware where indirection and indices cost
less than the multiplications, memory and dense updates they avoid. At high density, metadata and indirection can
make HFFN slower than dense. That is an expected boundary of the format, to be reported, not hidden.

## 2. The invariant

For `z = W_up x + b_up`, `h = activation(z)`, `y = W_down h + b_down`, hidden neuron `r` owns its active incoming
connections `(up_idx[r], up_val[r])`, its bias `b_up[r]` and its active outgoing connections
`(down_idx[r], down_val[r])`. The sum of all hidden contributions is exactly the original sparse FFN, up to
precision and reduction order. HFFN changes the layout and the scheduling of the same computation; it approximates
neither the activation nor the support.

> **Compact when a quantity is bounded by the support; dense when it is bounded by the activation space.**

- Compact: active weight values, indices and offsets, gradients of active weights, their AdamW states.
- Dense: `x`, `z`, `h`, `y`, their activation gradients and the reduction buffers.

Consequences:

- One call processes the whole block of `N` tokens. No call per token or per partial mini-batch.
- CPU: fixed groups of hidden neurons, one private reduction buffer per group, deterministic reduction. GPU: keep
  the dispatch and reduction policies of the current runtime; do not transpose the CPU grouping literally.
- Padding is allowed only for homogeneous supports, and padded slots are never written or trained. Irregular
  (WANDA) supports use exact offsets.
- A different reduction order can produce a long-horizon FP32 drift without any change of support or equations.
  Compare reduction orders before blaming the support.

## 3. Prohibited

- No production format other than HFFN. CSR/COO/BSR, two independent sparse matrices or a masked dense matrix are
  controls, labelled as such, never the path being optimised.
- Never materialise `W_up`, `W_down`, their gradients or their AdamW moments as dense tensors in an HFFN path.
- No allocation, layout conversion or sparse-view reconstruction in the hot path.
- No manual dot-product micro-kernel without an integrated profile: it was tested and regressed.
- One mechanism per trial. A refactor or a second optimisation in the same trial makes the result unattributable.
- No claim of convergence equivalence, global speed-up or hardware generalisation without the matching protocol.
- A threshold is never relaxed to save a result. An amendment is a user decision, written as a general rule, and
  the earlier verdict is kept. Every amendment is prospective unless it says otherwise: earlier runs keep their
  verdicts, it reopens no closed ablation, and a re-reading of an earlier run under it is marked retrospective.

**What is HFFN and what is dense.** Everything that is not specific to the HFFN architecture (attention, norms,
embeddings, the optimiser of non-FFN weights, the training loop) runs as the dense PyTorch of the model, on every
hardware. The difference with dense is sought only where dense cannot follow. A mechanism added around the FFN (a
fused stage, an extra layer, a scheduling step) is admitted on the HFFN arm only if it cannot be applied equally to
the dense control; if it can, it is applied to both arms or not at all. Native non-FFN code (the Metal attention of
`native_transformer`) is set aside: not developed, not a production path.

Guiding question for any change: *does it exploit the hidden neuron and its active support better, without
rebuilding a dense representation and without breaking the parity of the fixed-support model?* If not, it is an
ablation or a control, not an evolution of the runtime.

## 4. Layout

Four top-level folders share one layout.

```text
experiments/   open ablations, tests, instrumentation, measurements, smokes
current/       the last working version of every part, and the checkpoints in use
publication/   clean, isolated, self-sufficient versions only (one version per part)
archive/       superseded versions, judged ablations, their runs, old checkpoints, published Zenodo artifacts
papers/        article sources

<top>/<branch>/<cpu|gpu>/<hardware>/<ffn|llm>/<inference|training>/
   branch    runtime | moe | multilayers | lateral | sparse_attention
   hardware  cpu: x86, m1      gpu: nvidia, m1
   ffn       an isolated FFN            llm   a model with attention around the FFN
```

- `common/` holds what is shared, at the highest level it covers (`runtime/common/`, `runtime/gpu/m1/common/`, …).
- `training/` uses the kernel of its sibling `inference/`, with no exception; it never carries its own copy of the
  forward. Moving a training leaf onto that kernel is judged on parity; its time and memory against the former
  forward are reported as a limit and cannot keep a second forward. A loss of time is first diagnosed as one of two
  causes: the training forward carried a mechanism the inference kernel lacks (it becomes an ablation of the
  inference leaf), or the integration is poor (call path, layout or buffer conversion, writes of `z` or `h` for the
  backward, threads, scheduling), fixed in the entry point.
- `moe/` and `multilayers/` build on the kernels of `runtime/`.
- `current/assets/` is the only place for checkpoints, exports and datasets in use. One copy of each. When a model
  is retrained with similar specifications, the old one goes to `archive/assets/`.
- A leaf holds one version. Where two lineages still coexist, each has its own sub-folder and the pair is listed in
  `UNIFICATION.md`, the list of work that remains to reach the target form; read it before proposing a new copy of
  anything.
- When two copies of one mechanic are unified, the better measured one (time, then memory, on the quantity the leaf
  is judged on) becomes the common base. It is generalised to cover what the other copy did before that copy is
  archived; no capability is dropped by the unification.
- Each populated leaf of `current/` has a `STATUS.md`: what the version is, the run that qualifies it, what is open.

## 5. Life cycle

```text
experiments/  --(criteria met + user verdict)-->  current/  --(promotion manager + user verdict)-->  publication/
                                                     |
                                     previous version, the ablation capsule and its runs --> archive/
```

**An ablation** lives in `experiments/`, in the leaf it targets. It is an overlay: its folder contains only the
files it changes or adds, and it imports everything else from `current/`. Its README is written before any
measurement and states the single hypothesis, the control arm, the measured perimeter, the retention criterion, the
git tag or commit of `current/` it starts from, its time budget and its attempts. The control arm is `current/`.

**Attempts.** The README declares the number of campaigns allowed per machine, two at most. When they are spent
without an admissible run, the ablation is closed as `not resolvable at this bound on this machine`, archived with
that verdict, and no amendment is proposed to save it. Reopening it is a new ablation and a decision of the user. A
smoke is not an attempt. A **harness failure** is not an attempt either: a run that stops because an agent-authored
part of the harness is defective (launcher, import or asset path, checker or gate setup, environment, VM setup)
while the candidate was not judged on the point that failed, whether or not a scientific gate had started. Its run
record and exact cause are kept (`RESULTS.md` of the ablation), the defect is fixed before another launch, and the
criteria stay frozen. A failure of the candidate always counts: a gate or bound it fails, a time or memory it
reaches, a crash in its own code, a ceiling (`EXIT=124`) reached while its arms run. The coordinator classifies the
failure from the record; a classification the record does not show goes to the user.

**Promotion to `current/`.** An ablation is kept when the criteria written beforehand are met. The agent runs
`hffn-promote`: the changed files replace those of `current/`, the previous version and the whole capsule (with its
runs) go to `archive/`, `STATUS.md` and `archive/INDEX.md` are updated, and the commit is tagged. If `current/` moved
since the ablation started, the ablation is revalidated on the new base first. A rejected ablation is archived the
same way, with its verdict.

*Promotion by default.* The promotion needs no verdict of the user when all of the following hold; it is then
reported to the user with its evidence, and the user may revert it:

- (a) the README states, before any measurement, a criterion of parity, of time and of memory (or why memory is not
  touched), with numbers, and none of them was amended since;
- (b) the run is qualified (`passed`, exit code 0, full coverage, its gates passed) and every criterion is met, read
  under section 7 (noise band `delta` included); the coordinator recomputes the criteria from `raw.jsonl` and states
  them met before the promotion;
- (c) the targeted parity of section 7 passed on the promoted path, and `hffn-invariant-auditor` reports no
  blocking reserve;
- (d) the base of `current/` is the one of the ablation, or the ablation was revalidated on the new base;
- (e) for a shared level, the parity holds on every hardware the changed files reach.

A reserve of the auditor is **blocking** when it names a violation of section 2 or 3, a change outside the declared
hypothesis, a case of the change that the parity does not cover, or a criterion the run could not have failed (read
on a quantity fixed by construction, or not measured). Any other reserve is **minor**: it is written as a limit in
`RESULTS.md` and as an open point in `STATUS.md`, and does not stop a promotion by default. The auditor marks each
reserve; the coordinator checks the marking.

A promotion stays a decision of the user when a criterion is not met or not measured, when the run is not
admissible, when an amendment of a threshold or a protocol is proposed, when the change differs from the README,
for the stop criterion below and for any promotion to `publication/`. A rejection under the same conditions (a
criterion clearly not met in a qualified run) is archived by default the same way.

*Commit of a promotion.* Every promotion into `current/` and every archived rejection is committed at once by the
session that executed it, with `.claude/team/commit_promotion.sh`: one commit holding only the paths of that
promotion (the leaf, its `STATUS.md`, the archived previous version and capsule, the old capsule path,
`archive/INDEX.md`), tagged as `hffn-promote` says, the pre-commit hook never bypassed. Every other commit is the
user's; the coordinator proposes them in batches.

**Shared levels.** An ablation of a level shared by several hardwares states its criteria per hardware. It is
promoted for each hardware whose criteria are met, without waiting for the others: the change reaches only that
hardware (its leaf, or an entry point per hardware named in the README), and its parity is checked wherever the
changed files arrive. The other hardwares keep the previous version until their own evidence.

**Promotion to `publication/`** is done by the agent `hffn-publication-manager`, with the user's verdict for that
promotion. What enters is clean: no ablation residue, no dead code, no path leaving the folder, no checkpoint (a test
that needs one reads it from `HFFN_ASSETS`, default `current/assets`), documented scope and results with their
provenance. `publication/` holds one version; the previous one is archived.

- *Checks are not repeated.* The every-export checks and gates that qualified the candidate trees are its
  prerequisites, bound to them by a frozen digest; the publication does not rerun them. On the cleaned copy, taken
  alone outside the repository, each leaf runs only its import and its bounded smoke, plus its targeted parity where
  the cleaning changed executed code (a rename that reaches an import, a removed branch on the kept path).
- *A leaf already published is compared three ways.* Its publication campaign runs A (the published version), B
  (the candidate) and the dense control in the same processes, under the schedule of section 7. B replaces A only if
  it loses no time beyond `delta` and no memory beyond 2 %; a gain over A is stated only beyond `delta`. A B without
  gain over A needs a functional reason (a capability, a fix, a unification) and the user's verdict. The ratios
  against dense fill the results of the leaf. A leaf published for the first time runs B and dense.

**Objective: v1 published, v2 in progress** (since 2026-10-22). v1, the unified runtime, is in `publication/` with
its limits. v2 works in this order of priority:

1. *Functional.* Complete what v1 left open or failing (the open items of `UNIFICATION.md`, the cells v1 reports as
   failed or out of memory, a missing leaf).
2. *Performance and memory.* Optimisation ablations are open. One is kept when its gain on the quantity the leaf is
   judged on exceeds **2 %**, or `delta` when `delta` is above 2 % (time: `1 - ratio > max(0.02, delta)`; memory:
   a gain above 2 %), with no loss elsewhere (time within `delta`, memory within 2 %).
3. *Unified and clean.* A change never reopens a unified item: no second lineage, no copy (`UNIFICATION.md`).

*Secondary: `sparse_attention/`* (since 2026-10-24). An architecture study outside the runtime goal of section 1:
attention partitioned by the HFFN groups, in PyTorch, judged on quality against HFFN `d20` with dense attention
(`current/sparse_attention/PLAN.md`). It comes after the three priorities above: when a machine, a VM session or a
review is needed by both, the runtime takes it. Its quality runs are campaigns under section 6.

**When a leaf stops being optimised.** A leaf stops when, after a step back (`hffn-prior-art` over every hardware,
the ablations accepted elsewhere included, and an adverse review by `hffn-critic`), no remaining proposal is
estimated to bring more than `max(2 %, delta)` on the time, or 2 % on the memory, of the quantity the leaf is judged
on (`delta` the last measured noise band of the leaf). The estimate, its source and the proposals examined are
written in `STATUS.md` before the leaf is proposed for publication; the user gives the verdict.

`archive/` is a record. Its scripts are not repaired and active code never imports from it.

## 6. Launching

A **smoke** checks that the code runs and that memory, performance and quality are within expected bounds. It
proves nothing, but a surprising smoke result is checked before any campaign. A **campaign** is any run whose
numbers will be used.

| Machine | Smoke | Campaign |
|---|---|---|
| Local (the Mac) | agent | **user** — the agent hands over the exact command; or a team agent under a grant; or the agent, for an untimed run |
| VM | agent | agent |

- *Under a grant.* A team agent (`.claude/team/`) may launch a local campaign when the user has declared the machine
  ready (`.claude/team/machine_ready.sh`: mains power, disk, running processes, with an expiry; only the user runs
  it) and the coordinator has issued a single-use grant for that exact launcher and mode (`.claude/team/grant.py`).
- *Untimed.* A local run that times nothing and bounds no memory, judged on parity or identity only (a qualification
  by gates, a leaf's `check.sh`), is launched by the agent without a grant or a declaration. Its launcher lists those
  modes on a line `# hffn-local: untimed <mode> ...`; a mode that reads a time or memory figure used by a criterion or
  reported as a limit is never listed.

Every local campaign, however launched, follows the points below and the checks of `hffn-handoff`.

1. One campaign per machine at a time. Campaigns may run in parallel on different machines.
2. No smoke, build or heavy job on a machine while a campaign runs on it. During a campaign on the Mac, work for the
   VMs goes on: transfers (`rsync`, `scp`), VM launches and commands, reading and analysing runs.
3. Outside publication, a campaign lasts the minimum needed to know the speed, memory and quality of the ablation
   against `current/` before the verdict. Its README states its duration and what each part answers; a part that
   answers no question of the verdict is cut. The Mac is cooled passively, so a local timing campaign is designed
   from its duration: the README states a total budget (warm-ups, builds, re-warm-ups and pauses included) before
   choosing cells and repetitions, and cuts cells rather than stretching it; a design whose untimed overhead
   outweighs the timed part is reconsidered. **A local timing campaign lasts at most 30 minutes** from launch to
   `EXIT`; the launcher stops at the ceiling with `EXIT=124` (a partial run, no number read). If no cut answers the
   question within it, the question goes to the user. A revision that lengthens a local campaign is a user decision,
   stated with the former and the new duration. Untimed runs and publication campaigns are not bound by the ceiling.
4. A publication campaign is prepared in advance and needs the user's approval. It lasts what it must. It is described
   by a frozen candidate (`experiments/runtime/common/publication/CANDIDATE_TEMPLATE.md`); the user approves it by
   running `.claude/team/start_publication.sh <candidate>` under their machine declaration, which checks the
   candidate and issues one single-use grant per local launch. The `hffn-publisher` session it opens launches,
   monitors and pulls every entry (the Mac: its queues with a cooldown between two; the VMs in parallel), stops each
   VM, and ends with a coverage report: statuses and exit codes, no performance number, no `RESULTS.md`, nothing under
   `publication/`. It
   launches no entry twice; a resume or a rerun is the user's command.
5. A smoke never triggers a campaign. A launcher prints unbuffered progress, keeps a log file, writes a status
   manifest and a reliable exit code.
6. VM work follows `AWS.md`: check credit before and after, stop the VM at the end and verify it is stopped.
7. *Chained VM sessions.* Since a VM is slow to start and stop, one session may chain the campaigns, smokes and checks
   of several ablations or leaves, **strictly one after the other, never in parallel**. Each item keeps its own run
   folder, `EXIT`, ceiling and attempt count; a failed item does not stop the chain unless it leaves the VM unusable.
   The checks that several items need are run once in the session and shared. The session cap is the sum of the
   items' measured durations, the setup and the pull window, and is extended (`vm_session.sh extend`) rather than
   restarting the VM. The chain is written before the start (`AWS.md`).

## 7. Evidence

Every statement carries its class: **measured** (read in a qualified run — status `passed`, exit code 0, full
coverage — with the run path), **estimated** (derived from a micro-benchmark, a partial counter or a calculation),
or **hypothesis** (an explanation not isolated by a control).

- A failed or partial run gives no performance number, only the cause of the failure and what is missing.
- Numbers from different campaigns are not compared.
- Two effects that vary together are not attributed.
- One seed, one smoke or one repetition proves neither a gain nor convergence.
- Run records (`config.json`, `provenance.json`, `raw.jsonl`, `summary.json`, `metrics.csv`, `run.log`,
  `manifest.json`, `EXIT`) are never edited. A re-analysis goes in a new file and keeps the original verdict.
- Parity comes before any timing: outputs, active gradients, AdamW states, inactive slots, or generated-token
  identity where the branch policy says so. How parity is judged (exact or against a float64 reference, its levels
  and bands) is the single framework of the branch's rule file (`runtime.md`, *Accuracy gates*).

Two levels of measurement: the light protocol below for `experiments → current`; the full protocol of the branch
(cohorts, more sessions and cells, memory bound, the A/B/dense arms of section 5) for `current → publication`, whose
parity is the every-export check of its prerequisites. Both use the same *Timing and noise* rule below; thresholds,
admissibility and retention criteria keep their form under both; the light protocol changes the means, not the bar.

**Light protocol.**

- *Targeted parity.* Parity on the smallest and the largest export the mechanism touches, plus the synthetic test
  layers of the leaf.
- *Batched every-export check.* The leaf's `check.sh` on every export runs once per leaf for a batch of promotions,
  on the last of them and on every hardware the batch reaches; `STATUS.md` lists the promotions it owes. It is always
  run before the leaf serves as the control arm of a timed campaign, before it enters the publication process, and
  before a promotion whose README needs a figure of it. If it fails, the batch is bisected on assembled trees (never
  by editing `current/`): the first failing promotion is reported to the user with its cause, those before it stand.
- *Timing and noise.* One rule for every timed run, on every machine; the README states each of its choices before
  the run.
  - *Schedule.* The arms are control (`current/`, or A for a publication), copy (the same code as the variant up to
    the switches) and variant. GPU: one process per session, the arms alternating call by call in ABBA order on the
    same inputs. CPU: one fresh process per cell, the arms in ABBA blocks inside it. Each arm is warmed up before its
    first timed call.
  - *On the Mac, in addition* (passive cooling): a long warm-up of each arm in each process, so that the first timed
    call runs in a steady state; every arm rebuilt from scratch (fresh allocations, short re-warm-up) at each block;
    the thermal state recorded before each measurement; the ratio of a cell read on the median of its timed windows.
  - *Noise band.* The copy arm does not gate the reading; it sets the band of the run, `delta` = the largest
    `|copy / control - 1|` over its cell-sessions, each with the statistic of the README; `delta` is reported. A gain
    or a loss is stated only beyond `delta` (gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`). A numeric
    bound `t` of the README is met only with the ratio moved by `delta` in the unfavourable direction
    (`ratio + delta <= t`). A memory ratio is not moved by `delta`. The run must still be qualified.
  - *Design from the known band.* The README quotes the last `delta` measured on that leaf and machine, and checks
    that its criteria are reachable under it (`delta < t - 1` for a bound `t`, `delta < gain` for a gain). A
    criterion the known band makes unreachable is redesigned (cells, windows) or read elsewhere as below, before the
    run, never after it.
  - *Shared CPU level read on x86.* For a level shared by the M1 and x86 CPU leaves, a time bound `t` that the M1 band
    makes unreachable (`delta_M1 >= t - 1`) is read on the x86 run only, under the x86 `delta`. On M1 the timing run
    must still be qualified, and every cell must meet the per-cell bound of the README (at most 1.10) moved by
    `delta_M1`. Memory and parity are read on both machines.
- *Minimal cells.* The README names the few cells (about 4 to 6) that cover the quantity the leaf is judged on and
  the regimes the mechanism targets, with the reason for each; not the full grid of the branch.
- *Identical build, no timing.* A step that leaves the executed code unchanged is judged on identity and parity,
  without timing. Unchanged means both: (1) the native libraries built from the changed tree are identical to those
  of `current/`, byte for byte or, where only paths or build metadata differ, in their code sections (the method is
  stated in the README); (2) between the entry point and the native calls, the Python path runs the same operations
  on the same buffers (a moved import or relocated file is unchanged; an added call, conversion, copy, allocation or
  a changed buffer lifetime is not).
- *Identical called functions, no timing.* A step that only removes or moves code the kept path does not execute (an
  alternative forward, a dead branch, an unused option) is judged on identity and parity, without timing, when:
  (1) the native functions the kept path calls are the same on both trees with identical machine code — the README
  names the entry symbols, the closure of their direct calls (and of entries reached through a pointer, such as the
  thread-pool task, listed by hand) and the method (disassembly per symbol, addresses and relocation targets
  normalised); a function on one side only, or with different instructions, fails it; (2) the kept path runs the
  same operations on the same buffers, and every dispatch decision it takes (mode, threshold, route) has the same
  outcome on every cell of the leaf; (3) it allocates the same buffers with the same sizes and lifetimes (memory held
  only by the removed code may disappear, never appear). A change of placement or alignment of the kept code is not
  timed (hypothesis that it does not matter). A part that does not hold (a path moved onto other code) is timed.
- Under either identity rule, the README states the identity criterion in place of the time and memory criteria
  before the run; the run can fail it, so it meets condition (a) of the promotion by default.

## 8. Environment

- Python: `/Users/Parker/Documents/.conda/envs/ML/bin/python` (mamba `ML`, 3.11). Do not use the `python3` of the
  PATH for torch or MPS. Install nothing in `ML` without the user's agreement.
- Metal/MPS: never with the CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK` stays unset).
- The repository root is found from a file as the first parent that contains `current/`. No absolute path and no
  count of parent folders in code.
- Build outputs go to `build/` folders (ignored by git), never next to sources.
- Git: commit only when the user asks, except the commit of a promotion (section 5). The global `pre-commit` hook
  refuses files over 50 MiB and archives; never bypass it. It makes one git call per staged file, so a commit of tens
  of thousands of files takes tens of minutes: run it in the background.
- The disk is nearly full: never duplicate a checkpoint, an export or a dataset.
