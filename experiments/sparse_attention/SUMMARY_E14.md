# sparse_attention — summary of the fourth step (E14), cloud session of 2026-10-27/28

Branch `sparse-attention-cloud`, from `8e0a7ff` (the plan of the step and its brief). Plan: `current/sparse_attention/PLAN.md` section 17; brief: `experiments/sparse_attention/CLOUD_E14.md`. `SUMMARY.md`, `SUMMARY_E6_E9.md`
and `SUMMARY_E10_E13.md` are records and are not rewritten. Code and records: `experiments/sparse_attention/common/llm/training/` (`prep4/`, `e14/`, each with the README committed before its first run, `step.json`, `run.sh`,
`runs/`, `RESULTS.md`). Every figure is **measured** in the qualified run named in `e14/RESULTS.md` unless marked **estimated** (a gap, a product: two measured axes combined) or **hypothesis**; nothing is promoted to `current/`
or `publication/`. No training: the quality axis is the validation bits per byte per seed of E8, E9, E13 (unedited). Times are ratios of the PyTorch executor of E11 on this shared 4-vCPU machine, forward only, with a masked dense
FFN in every arm; they say nothing of another machine, of a native implementation or of the HFFN runtime.

## Preparation and cost

219 + 44 + 34 + 82 checks pass (the first three series stay reproducible). New: arm `E2h` (`E4a` with the dense attention of `REF`, `W_O` dense: 12,288 more parameters per layer at `M = 128`, 49,152 at 256) at the three sizes
and `REF4_W256`; its packed form (equal to `selected`: logits ≤ 2.4e-6, gradients ≤ 1.2e-7) and its executor kind; a **decoding path** with a KV cache for 26 arms (the decoded step equals the full forward at the same position within
2.4e-6 for every arm; the step under the executor equals the sequential step bit for bit; the filled cache is untouched); the 50 timing cells; the memory (e) of a decoded step. Finding while building (e): the profiler's "Total Allocated"
is not reset between two profiled regions of one process; (e) is taken relative to its first event; the figures of E10 were not affected (checked). The SDPA kernel is the CPU flash kernel in every arm; the causal flag is 0-18 % faster than an
explicit mask on the shapes of the cells (descriptive). Re-estimate after the smoke ≈ 3.1 h against the 6 h limit: **no cell was cut.** Wall time: preparation and smoke about 1 h of compute on top of the development, prefill 1.4 h,
decoding 1.4 h, memory 0.1 h (the first prefill run, 0.4 h, was lost to a restart of the container).

## The answer to decision 16 (frontier of quality against measured time; estimated; the modes apart)

`delta` = 0.0623 (prefill) and 0.0707 (decoding), each set by one cell-process among 125 (median `|copy − 1|` 0.020 and 0.017). Gap `g` = bits per byte of the arm minus the dense frontier at its measured time (positive = worse).

| mode | campaign | `E4a_L2` | `E4a_L4` | `E4a_L6` | answer |
|---|---|---|---|---|---|
| prefill | `N` 128 | on, +0.022 | on, +0.015 | outside the range (vs `REF4`: +0.017, time 0.96) | **not resolved** |
| prefill | `N` 512 | on, −0.008 | on, +0.019 | outside (vs `REF4`: +0.073 inconclusive, time 0.94) | **not resolved** |
| prefill | `M` 256 | outside (vs `REF1_W256`: +0.009, time 0.995) | outside (vs `REF2_W256`: +0.001, time 0.987); `h16`: −0.046, time 1.032 | — | **not resolved** |
| decoding | `N` 128 | on, +0.073 | on, +0.046 | outside (vs `REF4`: +0.017, time 1.05) | **not resolved** |
| decoding | `N` 512 | on, +0.070 | on, +0.062 | outside (vs `REF4`: +0.073 inconclusive, time 1.05) | **not resolved** |
| decoding | `M` 256 | on, +0.092 | outside (vs `REF2_W256_h16`: +0.005, time 1.16); `h16`: −0.042, time 1.21 | — | **not resolved** |

- **No `E4a` arm is above the frontier of measured time**, in either mode, at any context or width tried; none is below it beyond τ either. In prefill at `M = 128` the constrained arms lie on the frontier, within τ of it
  (gaps ≤ 0.022); the dominance read at equal counted multiply-adds (E8-E9) does not survive the change of axis. `IND_L4` (control) is behind the frontier, beyond τ in decoding.
- **Decoding is less favourable than prefill** on this executor (the hypothesis of PLAN 17.4, that a long cache favours the constraint, is contradicted): larger gaps in the same arms (`E4a_L4` +0.046 / +0.062 against +0.015 /
  +0.019), higher time ratios (`E4a_L4` / `REF2` 1.43 and 1.38 against 1.24 and 1.16). Most likely cause (hypothesis, not isolated): the packed decoded step copies, for the routed sequences, the cache slice of the heads of the group and
  pads the groups to the largest count; the measured memory of the decoded step shows the copy (below).
- **At `M = 256`** the frontier has three dense points and every constrained arm falls outside its range under the shifts of `delta`; no frontier verdict. By the pair rule: in prefill `E4a_L4_W256_h16` is as fast as `REF2_W256`
  within `delta` (1.032) with 0.046 fewer bits per byte (descriptive: against `REF2_W256_h16` it is 6.5 % faster and 0.042 lower, a margin of the size of `delta`: not a frozen reading); in decoding the constrained arms are 16-21 % slower
  than the nearest dense arm.

## The attention alone and the routing of the FFN alone (measured; equal depth 4; pairs p1-p6)

| | prefill r | decoding r | memory (a) / (b) / (c) / (e) |
|---|---|---|---|
| attention alone, `E4a_L4` / `E2h_L4`: 128, 512, 256 | 0.72, 0.69, 0.70 (**gain**) | 0.72, 0.65, 0.65 (**gain**) | 1.12-1.32 / 1.11-1.30 / 1.22-1.24 / 1.64-6.02 |
| routing of the FFN alone, `E2h_L4` / `REF4`: 128, 512, 256 | 0.90, 0.93, 0.76 (**gain**) | 1.01, 1.03, 0.93 (no difference within `delta`) | 0.75-0.80 / 0.76-0.81 / 0.63-0.65 / 0.97-1.01 |

- The sparse attention alone is 28-35 % faster at equal depth in both modes (between half and three quarters of the counted gain of 41-60 %), including the scheduling it allows (one barrier per layer instead of two). It does **not** save
  activation memory in this harness: 12-32 % more saved tensors and a 22-24 % higher prefill peak than `E2h` (counted: 0.29-0.38 of the activations), and 1.6 to 6.0 times the peak of a decoded step (growing with the cache).
- The routing of the FFN alone saves time in prefill (7-24 %; against a masked dense FFN, which overstates it for the HFFN runtime, hypothesis) and memory (0.75-0.80 of the saved tensors, 0.63-0.65 of the prefill peak), and is
  indistinguishable from the unrouted model in decoding.
- Products p1 × p2 (estimated): prefill 0.645, 0.639, 0.533, decoding 0.724, 0.664, 0.609, against the counted 0.458, 0.369, 0.402 for `E4a_L4` / `REF4`: the measured time gain is about a half to two thirds of the counted gain, the attention
  carrying the larger share in prefill at `M = 128`.

## Observations outside the registered readings (proposals for the user, not claims)

1. **Decode without the copy.** The decoded step of the constrained arms copies the cache slices of the routed sequences (and pads the groups); a step that reads the cache in place (a per-sequence loop, or a native kernel indexing the cache)
   would test whether the counted reading of the cache (0.25 of `E2h`) appears in time and memory. It changes the executor, so it is a new step with its own README, not a rerun of E14.
2. **Train `E2h_L4`** at `N` 128 and 512 (seeds 0, 1, 2, the rate of `E4a_L4`; about 2 hours, PLAN 17.7): the only way to place the attention alone and the routing alone on the quality axis; today they have time and memory but no bits per byte.
3. **A third dense point at `M = 256`** (`REF3_W256`, a training run) would give the frontier a range wide enough for the `M = 256` arms to be read (all three are outside it now).
4. **Batch size and cache length** in decoding (`B` and the cache of `N/2` are one value each): the attention is linear in the cache, so the pairs p1, p3, p5 are the natural cells to repeat at 2-4 cache lengths, and `B` changes the padding of the packed form.
5. **A native packed form** (compute only the routed rows, no masked dense FFN) is the only way to say anything about the HFFN runtime; every ratio above is this PyTorch executor's.

## Not run, not verified, limits (PLAN 17.6 and 17.7 included)

- Not run: any absolute time, another machine, training time, a native implementation, the HFFN runtime; any quality of `E2h`; another batch size, cache length, depth or width; generation of whole sequences.
- Not verified: the causes named as hypotheses (the gather of the cache slices for the decoding gap and the decoded-step memory; the masked dense FFN overstating the routing gain; the extra saved tensors of the attention alone).
- Limits: one executor, one machine (steal about 1-1.5 %), forward only; `delta` of each run set by one outlier cell-process, a smaller `delta` would resolve more cells (stated, not applied); bits per byte from three seeds with the reserves of E8, E9, E13
  (rates of the middle depths and `_h16` arms computed, `H*` not recalibrated at width); the concurrency gate passed with margins of 0.06-0.08 under the 0.4 threshold (0.325, 0.337; 0.377 at the preparation).

## Harness events

- The first prefill run (`e14/runs/prefill_20261027T205123Z`) was stopped by a restart of the container at 21:18 UTC (7 cells complete, one partial); classified as a harness event, recorded (`ABORTED.md`, `EXIT 137`), no figure taken from it;
  the full relaunch used the same README and rules. No other run was aborted, repeated or excluded; no cell, rule, threshold or reading was changed after a run.
