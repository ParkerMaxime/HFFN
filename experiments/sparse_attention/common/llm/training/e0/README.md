# E0 — reference, calibration and dynamic range of attention

Written on 2026-10-24 before any run of this step. Frozen once the step has run (a change is a new step or an
amendment of the user, and the earlier verdict is kept). Plan: `current/sparse_attention/PLAN.md` (sections 7, 8, 12);
rules: `AGENTS.md` sections 3 and 7; brief: `experiments/sparse_attention/CLOUD.md`.

## Single question

At the working size (one layer, `M = 128`, `N = 128`, bytes of enwik8), **can this task discriminate variants of
attention at all?** E0 measures (i) the quality level and the band of the seeds of the reference (dense attention,
HFFN `d20` FFN with a random support), and (ii) the dynamic range `Δ_attn` of attention: bits per byte without
attention minus bits per byte of the reference. If the range is not large against the band, the ladder stops here
(PLAN section 8).

## Arms (configurations in `lib/arms.py`, one model in `lib/model.py`)

| Arm | Attention | FFN | Parameters outside the embedding |
|---|---|---|---|
| `E0` | dense, 8 heads of 16, rotary, no biases | `H = 512`, GELU, random fixed support, `k_up = k_down = 26` per neuron | 93,184 (PLAN table) |
| `NOATT` | none (calibration; the layer is the FFN alone) | as E0 | 27,520 |
| `E3b` | the complete constraint (used here only for the learning-rate sweep) | | 80,896 (PLAN table) |

`NOATT` has 27,520 parameters against the "≈ 27,648" of PLAN section 7: the table counts the norm gain of the removed
attention (128), the implementation drops it with the attention. Embedding (tied), 256 × 128 = 32,768 for every arm.
Everything outside the FFN is dense (`AGENTS.md` section 3); the FFN is a masked dense matrix, a control
representation admissible here because nothing is timed (PLAN section 0).

## Data and training protocol (shared by every step; later READMEs only list what differs)

- enwik8 bytes, `current/assets/enwik8/split.json`: train `[0, 90M)`, valid `[90M, 95M)`, test `[95M, 100M)`. The test
  split is not read in this step. Training windows: 64 windows of 128 bytes (+1 target byte) at uniformly random
  offsets of train; the sequence of batches depends on the seed only (identical for every arm).
- AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on the 2-D matrices other than the embedding (norm gains and
  biases excluded), gradient clipping at 1.0, fp32, warm-up of 5 % of the steps then cosine decay to 10 % of the
  peak rate.
- Initialisation: every tensor is drawn from a generator seeded by (seed, tensor name), so the shared tensors
  (embedding, `W_Q`, `W_K`, `W_V`, norms) are identical across arms for a seed, and masked tensors are the dense draw
  multiplied by their mask. Supports are random with a fixed degree per neuron and depend on the seed.
- Evaluation: validation bits per byte over **all 128 positions of non-overlapping windows** of the whole valid split
  (39,062 windows) at the end; a fixed subset of 4,096 windows every 500 steps (the curve, not a result); by position
  class `t mod 4`; the mixing diagnostic of PLAN section 5 (on the subset; descriptive, never a gate). These bits per
  byte use a context of at most 128 bytes and are **not comparable** to published enwik8 figures.
- Environment: CPU, PyTorch 2.14.1 (the CPU wheel index of PyTorch was refused by the proxy with a 403; the wheel
  installed is the PyPI build, run on CPU), 4 jobs in parallel with 1 thread each.

## Fixed here (PLAN section 12)

- **Seeds: 0, 1, 2, 3** (four, at least three). Seed 100 is used for the learning-rate sweep only. The same seeds,
  the same data order and the same shared initial tensors are used for every arm of every step.
- **Steps: 4,000** per run (32.8 M bytes seen, 0.36 epoch of train). Sizing: a throwaway timing of a training step
  gave 0.24 s for E0 and 0.14 s for E3b with one thread (4,000 steps ≈ 16 min and ≈ 10 min); with four jobs in
  parallel a step of E0 is a few times slower than that measurement, which is accepted. Wall-clock is used only to
  size the runs, never reported as a result.
- **Learning-rate procedure (written before the sweep).** Grid `{5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}`. For the
  reference `E0` and for `E3b`, one run of 2,000 steps (same schedule shape, warm-up 100) per grid point with seed
  100; the criterion is the validation bits per byte at the end of the run. The best rate of each arm is the grid point
  of the smallest value. If the best rate of an arm is at an edge of the grid, the sweep is declared inconclusive and
  repeated on a grid extended by two points in that direction (a new run of the step; the earlier one is kept).
  If the two optima coincide, **that rate is frozen for every arm of the ladder**. If they differ, each arm uses the
  best rate of the same grid, as in PLAN section 7: the arms with dense attention (E0, NOATT, E1, E2, the L = 4
  reference) the rate of E0, the arms with a block-diagonal `W_O` (E3a, E3b, IND, E4a, E4b, the L = 4 constrained
  arms) the rate of E3b; the README of E3a, which changes family, then also reports E2 and E3a at both rates. The
  rates are copied as numbers into the `step.json` of each later step before its README is committed.
  `NOATT` uses the rate of `E0`.
- **Runs of the final stage:** `E0` and `NOATT` × seeds 0-3 = 8 runs, 4,000 steps, at the rate of `E0`.

## Statistic and decision

- Per arm: median and [min–max] over the four seeds of the final validation bits per byte.
- `d_s = bpb(NOATT, s) − bpb(E0, s)` per seed; **`Δ_attn` = median over seeds of `d_s`**.
- **Band** = the larger of the two seed ranges (max − min of the final validation bits per byte over the four seeds),
  for `E0` and for `NOATT`.
- **Minimum ratio `R_min = 10`.** The task is declared **discriminating** if `Δ_attn / band ≥ 10` **and** `d_s > 0`
  for the four seeds, and every run of the final stage is qualified (status `passed`, exit code 0, full coverage).
  Otherwise it is declared **non-discriminating**, the ladder stops, and the next options (longer context, wider model,
  another dataset) are the user's decision (PLAN section 8). A step where a run fails or diverges is reported with
  its cause, with no figure.
- Reading of the ratio (hypothesis, to be read in the run): a ratio of 10 means that a difference of 10 % of the whole
  gain of attention is about one seed band.

## Statistics used by the later steps (defined here, applied by their READMEs with their own numbers)

For a candidate against its control in the same campaign (the control is re-run in the campaign of each step, with
the same seeds; numbers of different campaigns are not compared, but the re-run is compared with the earlier one as a
reproducibility check): `d_s = bpb(candidate, s) − bpb(control, s)`, positive = candidate worse. With a bound `τ` fixed
by the README of the step before its first run: **within bound** if `max_s d_s ≤ τ`; **beyond bound** if
`min_s d_s > τ`; **inconclusive** if the seeds straddle the bound; "better in every seed" if `max_s d_s < 0`. The
paired band of a comparison is the range of `d_s`. Code: `lib/analyze.py`, which writes `analysis.json` and
`analysis.md` as new files next to the run records.

## Checks before any training (PLAN section 7)

`checks/check_model.py` (112 checks, run by every launcher before anything trains, exit non-zero on failure): parameter
counts against the PLAN table and the formula for every arm; every `up_idx` of a group inside its `S_g`; degree 26;
entries outside the masks exactly zero after optimizer steps; `o_t` exactly zero outside `S_g`; FFN output independent
of `u_t[j]`, `j ∉ S_g`, except through the RMS (a swap of two coordinates outside `S_g` leaves it unchanged within
float32 rounding, a swap inside changes it); packed and selected implementations identical (logits exactly, gradients
within 1e-8); causality of every arm; determinism; the balanced byte hash; the router mechanics; the diagnostic of
mixing is zero by construction for `IND`.
Harness defect recorded: the first execution of the checks failed one check (`FFN independence, IND`, change
1.43e-06 against an absolute tolerance of 1e-6 set too tight for float32 rounding of the RMS; the control swap changes
the output by 2.8). The tolerance was set to 1e-5 before any training and no run existed. It is a defect of the
harness, not a result.

## Launch and records

`./run.sh check | smoke | run` (fetches and verifies enwik8 first). `run` executes `step.json`: the `sweep` stage
(12 runs), the selection of the rates (`runs/run_*/sweep/selection.json`), the `final` stage (8 runs). Records per
run: `config.json`, `provenance.json`, `raw.jsonl`, `metrics.csv`, `summary.json`, `run.log`, `EXIT`; per launch
`manifest.json`, `checks.json`, `EXIT`. They are never edited. Checkpoints (`model.pt`) stay out of git.
`RESULTS.md` is written after the run from `analysis.json`.

## Attempts and limits

No attempt bound (user decision of 2026-10-24: the cloud session is unconstrained; the scientific rules remain).
Nothing here is a promotion. One layer on bytes with a context of 128 relies largely on local statistics; this is
exactly what E0 measures. The validation split serves both the learning-rate choice and the reported figures
(PLAN section 7); the test split is not used.
