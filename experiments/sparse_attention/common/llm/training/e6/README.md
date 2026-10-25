# E6 — the horizon

Written on 2026-10-25 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` section 14.4
(E6); brief: `experiments/sparse_attention/CLOUD_E6_E9.md`; harness and checks: `../prep/README.md`. Protocol of the first ladder
(`../e0/README.md`) except where stated.

## Single question

At which training horizon `H` does the paired gap between the two families stop moving? E6 chooses the horizon `H*` of E7-E9. It
shows neither the loss nor the gap at convergence.

## Design

- **Pair:** `REF2` (dense attention, HFFN `d20` unrouted, 2 layers, 186,240 parameters outside the embedding) and `E4a_L4` (the
  complete constraint, balanced byte hash fixed at every layer, 4 layers, 323,200 parameters). `N = 128`, 8,192 bytes per step
  (64 × 128). Seeds **0 and 1** (the same bytes in the same order for both arms).
- **Schedule: WSD** (200 warm-up steps, constant peak, linear decay to zero over the last 20 % of the horizon).
  **Peak rates:** `REF2` **0.025** (the geometric interpolation of 0.032 at `L = 1` and 0.016 at `L = 4`), `E4a_L4` **0.008** (the rate
  of the first ladder at `L = 4`). Weight decay 0.1 on the matrices other than the embedding, clipping 1.0, as in the first ladder.
- **Runs.** One plateau run per arm and seed with horizon 8,000, to step 6,400, full training state saved at 1,600, 3,200 and 6,400
  (the starts of the decay of the horizons 2,000, 4,000 and 8,000). From each checkpoint one decay branch runs to its horizon:
  1,600 → 2,000, 3,200 → 4,000, 6,400 → 8,000. **Twelve final figures** (2 arms × 2 seeds × 3 horizons): the final validation bits
  per byte of each branch over the whole valid split (39,062 windows of 128). The plateau runs end with a partial evaluation
  (1,024 windows) that is not a result. Intermediate evaluations: 524,288 bytes, every 800 steps (plateau) or 400 (branches).
  The test split is not read. A branch equals the uninterrupted WSD run of its horizon bit for bit (`../prep/RESULTS.md`).

## Rule, frozen

`gap_s(H) = bpb(E4a_L4, s, H) − bpb(REF2, s, H)` for seed `s` and horizon `H` (positive = constrained worse). **`H*` is the smallest of
2,000 and 4,000 such that `|gap_s(H) − gap_s(8,000)| ≤ 0.025` bpb for both seeds `s = 0, 1`.** Otherwise `H* = 8,000`.
0.025 is about the spread of `d_s` measured in E5 (0.016). **If `H* = 8,000` the series stops after E6**: `RESULTS.md` and
`PROGRESS.md` are written and the user decides the budget of E7-E9. The rule is applied by `lib/analyze_e6.py` from the run
records (a new file, `analysis.json`).

## Secondary reading (b), estimated

The gap at equal training compute. Training multiply-adds per token (the counter, 3 × inference): `REF2` 847,872 and `E4a_L4` 731,136 at
`N = 128`, so at equal tokens `E4a_L4` costs 0.862 of `REF2`. For each `REF2` horizon `H_r`, the equal-compute point of `E4a_L4` is
`H_r × 847,872 / 731,136 = 1.1597 H_r` steps; its bits per byte is interpolated linearly in `log H` between the three horizons of
`E4a_L4`, inside [2,000; 8,000] only (the point of `H_r = 8,000` falls outside and gets no value). One pair, two seeds: an estimate, not
a finding about the families.

## Limits stated before the run

- The two peak rates were chosen under the cosine schedule and fixed-length runs of the first ladder; under WSD they are not tuned
  (E7 tunes them at `H*`). A gap that moves with the horizon may in part be a gap of rates.
- Two seeds; a pair of two arms; `H*` is a pragmatic horizon, not a proof of stability beyond 8,000.
- The first-ladder hypothesis that the gap grows again in the decay (PLAN 14.4) is read from this step's curves, not assumed.

## Launch

`./run.sh check | smoke | run [resume_run_dir]`; 4 plateau jobs then 12 branch jobs, 4 in parallel, 1 thread each. Estimated wall
≈ 2.4 h (`../prep/RESULTS.md`). Records as in the first ladder; the `ckpt_*.pt` stay out of git.
