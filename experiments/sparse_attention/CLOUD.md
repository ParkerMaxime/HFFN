# sparse_attention on a Claude cloud session — brief

Written on 2026-10-24. `current/sparse_attention/PLAN.md` is the plan; this file only says how the cloud session
runs it. Read `AGENTS.md` sections 3 and 7 and the whole of `PLAN.md` first.

## What this session is

An isolated, free cloud session working on the branch `sparse-attention-cloud` (a slice of the HFFN repository:
`AGENTS.md`, `PLAN.md`, the enwik8 manifest, this folder; no runtime code, no checkpoints, no hooks).
**User decision of 2026-10-24: the session is free of machine constraints. The agent works and launches what it
wants, when it wants** — smokes and campaigns alike, no grant, no 30-minute ceiling, no attempt bound of
`AGENTS.md` section 6, no waiting for a verdict between smokes and runs.

What stays, because it makes the results readable and not because of the machine:

- one mechanism per step, each compared with the step before it (PLAN section 8);
- a README per step, **committed before the first run of that step**, fixing seeds, statistic, bound on the paired
  difference of bits per byte and the band; frozen once the step has run (a threshold is never relaxed to save a
  result; a change is a new step or the user's amendment, and the earlier verdict is kept);
- every statement carries its class: **measured** (read in a qualified run with its path), **estimated**,
  **hypothesis** (`AGENTS.md` section 7); a failed or partial run gives no figure, only its cause;
- run records (`config.json`, `raw.jsonl`, `summary.json`, `metrics.csv`, `run.log`, `EXIT`) are never edited;
  a re-analysis goes in a new file;
- no time or speed claim: the cloud machine is shared and unknown; compute saved is counted analytically (PLAN
  section 6). Wall-clock is logged only to size runs.
- nothing here is a promotion: no figure of this branch enters `current/` or `publication/` without the user.

## Order of work

1. **Bootstrap.** Python with torch (CPU). Check `python -c "import torch"`; install a CPU wheel if absent. Run
   `python experiments/sparse_attention/common/llm/training/fetch_enwik8.py`. If the network blocks the download,
   **stop and tell the user** (try `HFFN_ENWIK8_URL` only if the user provides a mirror); do not substitute another
   dataset or a sample.
2. **Shared code**, in `experiments/sparse_attention/common/llm/training/lib/`: one model with a config for every arm
   of PLAN section 8 (E0, E1, E2, E3a, E3b, IND, E4a, E4b, E5), the data loader on the manifest split, the training
   loop (PLAN section 7), the evaluation (bits per byte, by position class `t mod 4`, the mixing diagnostic of
   section 5, router load). The masked-dense representation of the FFN is admissible here (PLAN section 0).
3. **Checks before any training** (PLAN section 7, last paragraph) as a test script that exits non-zero on failure:
   parameter counts equal to the table of section 7, supports inside `S_g` with degree 26, `o_t` exactly zero outside
   `S_g`, FFN output independent of `u_t[j]`, `j ∉ S_g` except through the RMS, packed and selected implementations
   identical within float32 rounding. Nothing trains until they pass.
4. **One folder per step** `experiments/sparse_attention/common/llm/training/<step>/` with `README.md`, a launcher
   `run.sh` taking `check | smoke | run`, and `runs/<mode>_<UTC timestamp>/` records. The launcher prints unbuffered
   progress, keeps `run.log`, writes a status manifest and a reliable `EXIT`.
5. **E0 first, and its README fixes the open points of PLAN section 12**: number of seeds (at least 3), number of
   steps (from the speed measured in the smoke), the learning-rate procedure (written before the sweep), and the
   minimum ratio `Δ_attn` / band of the seeds below which the task is declared non-discriminating.
   - If E0 declares the task **non-discriminating, stop the ladder there** and report: the next options (longer
     context, wider model, another dataset) are the user's decision (PLAN section 8).
   - Otherwise continue in order: **E1, E2, E3a, E3b, IND, E4 (a then b, one per attempt), E5.** Each step is
     compared with its control of the table, paired by seed. The README of E5 also fixes, before its runs, the order
     of the fixed-against-rotation comparison and whether a learned router per layer enters (PLAN section 12).
6. **Commit and push to `sparse-attention-cloud` after each step** (README commit before the run, results after), so
   that progress survives the end of the session. Keep `experiments/sparse_attention/PROGRESS.md` current: step,
   status, run paths, exit codes, what remains. Never commit a file over 50 MiB or the enwik8 file itself
   (ignored by `.gitignore`); checkpoints stay out of git.
7. Each step ends with a `RESULTS.md` (sections measured / reading / limits; median and [min–max] over seeds, paired
   difference to its control per seed, control named). The last deliverable is
   `experiments/sparse_attention/SUMMARY.md`: the ladder, verdicts against the bounds fixed beforehand, and what was
   not run or not verified.

## Reporting

Repository files (code, READMEs, comments, commit messages) are in English. Messages to the user are in French.
Report plainly: failures with their output, skipped steps as skipped, harness defects separately from candidate
failures.
