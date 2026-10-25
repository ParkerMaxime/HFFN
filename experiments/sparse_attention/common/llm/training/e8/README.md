# E8 — the frontier at N = 128

Written on 2026-10-25 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 14.3-14.5;
brief: `experiments/sparse_attention/CLOUD_E6_E9.md`. Harness and counter: `../prep/README.md`; horizon: `../e6/RESULTS.md`
(**`H* = 4,000`**); rates: `../e7/RESULTS.md`.

## Single question

Is the constrained model (balanced byte hash, fixed at every layer) better, equal or worse than the dense-attention HFFN `d20` model **at
equal inference compute per token**, at `N = 128`; what does it lose at equal depth; and is the coupling worth anything over four separated
models at depth 4? All quantities are the final validation bits per byte; the test split is read once and reported beside.

## Design

- **Eight arms:** `REF1`, `REF2`, `REF3`, `REF4` (dense attention, HFFN `d20` unrouted, 1-4 layers) and `E4a_L2`, `E4a_L4`, `E4a_L6` (the complete
  constraint with the byte hash, 2, 4 and 6 layers), `IND_L4` (the four separated models of section 8, hash route, 4 layers). `N = 128`, 8,192 bytes
  per step, **seeds 0, 1, 2** (the same bytes in the same order for every arm), **WSD with horizon 4,000** (200 warm-up steps, constant peak, linear
  decay to zero over the last 800), one plain run per arm and seed (no branches). Weight decay 0.1 on matrices other than the embedding, clipping 1.0.
- **Peak rates (computed before the runs from E7):** `REF1` 0.016, `REF4` 0.008, `E4a_L2` 0.008, `E4a_L6` 0.008 (selected by E7); `REF2` 0.0127 and
  `REF3` 0.0101 (geometric in `L` between `REF1` and `REF4`); `E4a_L4` 0.008 (geometric between `E4a_L2` and `E4a_L6`); `IND_L4` 0.008 (the rate of
  `E4a_L4`). The middle-depth rates are computed, not swept.
- **Figures:** final validation bits per byte over the whole valid split (39,062 windows); the **test split** (`--test`, 39,062 windows) read once on
  every run, reported beside, selecting nothing; the mixing diagnostic of PLAN section 5 (descriptive); the counter of each arm (inference and
  training multiply-adds, parameters, AdamW states, KV cache, training activations per token).

## Readings (frozen). `d_s = bpb(candidate) − bpb(control)` per seed, positive = candidate worse; `τ = 0.075` bpb

Outcomes (PLAN 13.1): *within* if `max_s d_s ≤ τ`; *beyond* if `min_s d_s > τ`; otherwise *inconclusive*. Where the candidate costs less inference
compute than the control, a fourth: *dominates* if `max_s d_s < 0`.

| Reading | Candidate / control | Class |
|---|---|---|
| Equal compute (dominance pairs) | `E4a_L2`/`REF1`, `E4a_L4`/`REF2`, `E4a_L6`/`REF3` (inference savings 12 %, 14 %, 14 % by the counter) | measured |
| Equal compute (interpolated) | each `E4a_L<L>` against the dense family `REF1..REF4` interpolated linearly in `log C` (`C` = inference multiply-adds per token) at its compute, per seed, inside the range of the family only; `E4a_L2` (138,240) lies below `REF1` (157,696) at `N = 128` and gets no value | estimated |
| Equal depth | `E4a_L2`/`REF2`, `E4a_L4`/`REF4` | measured |
| Mixing at depth | `IND_L4`/`E4a_L4`: `d_s = bpb(IND_L4) − bpb(E4a_L4)`. If IND is *within* (not worse beyond `τ`), a favourable frontier measures the conditional computation of four separated models, not the coupling of heads and FFN groups | measured |
| Equal training compute (b) | the same pairs against the training column of the counter | estimated |

Reading (b): every arm trains for the same 4,000 steps on the same 8,192 bytes, and the training column is 3 × the inference column, so the
interpolation in `log` of training compute is **numerically the same** as the interpolation in `log C`; reading (b) adds no independent
information in this step (the independent information on training compute, with unequal steps, is the secondary reading of E6). The training
column is reported in the frontier table.

## Memory (written with the design, to be completed with the result)

At equal depth the constraint saves weights and AdamW states only through the block-diagonal `W_O` (−13 % outside the embedding, e.g. `E4a_L4` 323,200
against `REF4` 372,352), and does not change the KV cache (`K` and `V` dense). At equal compute it goes the other way: `E4a_L4` has 323,200
parameters against 186,240 for `REF2` and a KV cache of 1,024 values per token against 512; `E4a_L6` 484,736 against 279,296 for `REF3` and 1,536
against 768. **A favourable frontier is a statement about compute, bought with more weights and a larger KV cache.** `IND_L4` has 175,744 parameters.
`RESULTS.md` gives the frontier against compute, parameters and KV cache, not against compute alone.

## Limits stated before the run

- Three seeds; the middle-depth rates are computed, not swept; rates chosen on seed 100 at one horizon; the interpolation is across two
  different families' neighbours (estimated); validation serves the rate choice and the figures, the test split only reports beside.
- The harness runs the constraint in the selected form for the hash route (every head and neuron computed, the other groups multiplied by zero): wall
  time is not evidence of saving, and none is claimed. Compute is counted.
- `IND_L4` has 175,744 parameters against 323,200 for `E4a_L4` and a cheaper counter (133,120 against 243,712): a difference to `E4a_L4` mixes the
  effect of the coupling with the parameters and compute; `IND_L4` is a control, not a parameter-matched arm.

## Launch

`./run.sh check | smoke | run`: 24 runs, 4 in parallel, 1 thread each. Estimated wall ≈ 5.5 h (`../prep/RESULTS.md`, corrected by the wall times of E6 and E7).
