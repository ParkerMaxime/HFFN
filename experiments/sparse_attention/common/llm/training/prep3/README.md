# prep3 — preparation of the third series (E10-E13)

Written on 2026-10-26, before the smoke of this step and before any run of E10-E13. No figure of quality or of time is produced here.
Plan: `current/sparse_attention/PLAN.md` section 15 (14.8 above it); brief: `experiments/sparse_attention/CLOUD_E10_E13.md`. The plan on
the branch has its section 15 (checked). Starting commit: `a08b259` (the plan and brief on top of the second series).

## What changes in `lib/` (every new behaviour is an option; the defaults are the behaviour of the first two series)

| Item | Default | New behaviour |
|---|---|---|
| `--impl gather` (`lib/model.py`) | `packed` (= `selected` under the hash route) | **the packed form under the hash route**: per layer, worker `g` computes the keys and values of its heads `H_g` for every token, then, for the tokens routed to `g` only (gathered once per forward, padded to the largest count of the batch), queries, scores (causal mask on the original positions), outputs, `W_O^g` and FFN group `g`, scattered back to their rows. For `IND`, restricted to `S_g`. Under position routing `gather` is the packed form of the first ladder |
| Arms of width 256 (`lib/arms.py`) | — | `REF1_W256`, `REF2_W256`, `E4a_L2_W256`, `E4a_L4_W256` (8 heads of 32), `REF2_W256_h16`, `E4a_L4_W256_h16` (16 heads of 16); `d_model=256`, `hidden=1024`, `degree=52`, `context=512` |
| `lib/memory.py` | — | memory harness of E10: (a) saved bytes (distinct storages) split into weight-side and token-side, (b) peak live bytes of the forward and backward, (c) of one forward without gradient, (d) peak resident memory |
| `lib/executor.py` | — | executor of E11 (below) and the concurrency gate |
| `lib/timing.py` | — | timing harness of E11 (below) and the descriptive pass of phases |
| `lib/prep3.py`, `checks/check_series3.py` | — | launcher of this step; 34 new checks |

## Tolerances (stated before the run)

Float32 rounding of the same computation in another order: logits of the packed hash form (`gather`) and of the executor against the
sequential forward within **1e-4** (max absolute difference); the gradient of **every** parameter of `gather` against `selected` within
**1e-5** (max absolute difference; gradients are of order 0.2). Observed: logits 0 (bit for bit) at width 128, 2e-6 at width 256; gradients
at most 6e-8.

## The executor (E11)

- **Method: TorchScript `fork`/`wait` on the inter-op pool** (4 threads), the fork issued **from inside a scripted driver**, one driver
  per fork-join. Two findings from building it, kept here because they change how a measurement is read:
  1. A fork issued from Python on a scripted function does not run concurrently; issued from a scripted driver it does.
  2. `torch.set_num_threads(1)` only sets the calling thread: a thread started by the runtime opens its own OpenMP team (four
     threads each, oversubscribing the four vCPUs). The processes of E11 therefore start with `OMP_NUM_THREADS=1` and
     `MKL_NUM_THREADS=1` (the harness refuses to run otherwise); the intra-op control sets four threads in its own call.
- **Constrained arms (`e4a`)**: one fork-join per layer; the work of group `g` on worker `g` as in the packed form; each worker also writes
  the normalised rows of its tokens for the next layer (the whole normalised residual is then read by every worker's keys and values).
  **`IND` (`ind`)**: the same, the keys, values and queries of worker `g` read `S_g` only (a quarter of the normalised residual).
  **Clarification of the plan:** PLAN 15.3 says `IND` has no barrier through depth. The function of `IND` needs, at every layer, the RMS of
  each token's whole row, whose block `S_j` was just written by worker `j` (PLAN section 5: "except through the single scalar of the RMS
  norm"). An executor without that exchange computes another function, and the check "same outputs as the sequential forward" would
  fail. The executor therefore keeps one fork-join per layer for `IND` too: **the cells c measure the difference in the volume read and
  in the counted work, not the absence of a barrier**. This is written with the results.
- **Dense control under the same executor (`dense`)**: worker `g` computes queries, keys, values, scores and outputs of the heads `H_g`
  over every token; barrier; then `W_O` and the (masked dense, unrouted) FFN on a block of `N / G` rows per worker; barrier. The second
  dense execution, PyTorch's intra-op parallelism on four threads (`torch.set_num_threads(4)` around the call, standard model forward),
  is timed in the smoke. **Rule, written before the smoke:** the execution with the lower geometric mean, over the six cells that have a
  dense control (a1-a4, b1, b2), of the per-call time of the control (smoke, 2 windows) becomes the control of E11; the choice is written
  in the README of `e11` before its run.
- Logits are computed by the same code in both arms of a cell (four workers over blocks of rows); the embedding is a gather in the main
  thread. Weights are the effective masked weights built once (a forward without gradient). Verified equal to the sequential forward
  (checks, 10 arms, both contexts and width 256).

## The concurrency gate

Four equal independent tasks of the size of the largest layer cell (a4: one worker's share of an `E4a` layer at `M = 512`, `N = 2048`,
8,192 tokens), run by the executor and one after the other (alternating, so that a drift of the machine does not favour one): the median
of five repetitions of `parallel / sequential` must be at most 0.4. The result is in `RESULTS.md` and in the addendum below. If it fails,
E11 runs no timing and is reported as not run, with the cause (a property of the machine or the executor, not a harness failure).

## The memory harness (E10)

(a) `torch.autograd.graph.saved_tensors_hooks`, distinct storages; weight-side = storages of parameters, buffers and of the masked weights
`w * mask` that `Layer.eff` builds at every forward (a cost of the control representation), token-side = the rest. (b) and (c): the
"Total Allocated" of the memory events of `torch.profiler`, relative to the start of the profiled region (so the parameters, masks and
batch, alive before it, are not in it; the masked weights, activations and gradients are). (d) `ru_maxrss`, descriptive. Weights, AdamW
states and the KV cache are counted (`lib/count.py`), not measured. Checks: a synthetic module whose saved bytes are known by hand (a)
exactly, a synthetic allocation pattern whose peak is known by hand (b, c) exactly, and two fresh processes of one arm identical in (a),
within 1 % in (b), (c) (observed identical).

## The timing harness (E11), the rule implemented

Per PLAN 15.3: a fresh process per cell-process; the arms control, copy and variant built independently (the copy is a second
instance of the control with the same weights); each arm warmed up (at least 30 calls and 5 s); 12 windows of `control, copy, variant,
variant, copy, control`, each slot repeated so that it lasts at least 0.2 s; steal and user jiffies (`/proc/stat`) per window, load average
at the start of the process, `lscpu` and the torch configuration once; `raw.jsonl` has one line per slot and the ratios are computed from
it only (`ratios_from_raw`). `delta` = the largest `|copy / control − 1|` over the cell-processes of the run. Check: synthetic arms with a
known ratio (variant 1.5 times the control) are recovered from `raw.jsonl` alone within 3 %.

## Smoke and sizing

`./run.sh run` (= smoke here): the checks of the three series; the gate; the six arms of E13 for 40 steps of WSD (hash arms in the
`gather` form, 4 jobs in parallel, 1 thread each); E10 on two arms (two fresh processes each); the eight cells of E11 (one process,
2 windows) and the dense control under both executions; the re-estimate of the total from the measured step times (4 workers, LPT).
It proves nothing about quality or time.

## Addendum, written after the smoke (`runs/run_20261026T205721Z`): gate and choices that the run fixes

- **Concurrency gate: passed.** Median of five repetitions of `parallel / sequential` = 0.310 (threshold 0.4; `gate.json`; three earlier
  development repetitions of five gave 0.29-0.32). E11 runs.
- **Dense control of E11: the executor** (the same fork-join executor as the constrained arms). Rule above: geometric mean over the six
  dense-control cells of the control's per-call time in the smoke, intra-op / executor = 1.08 (the executor is the faster by 8 %; per cell the
  intra-op execution was faster in a1, b1 and b2 and slower in a2, a3, a4 by up to 33 %, 2 windows each). The intra-op execution is not used in E11.
- **Harness events:** none after the checks passed. During construction (before any README was written): a first executor ran its forks
  from Python and its threads in oversubscribed OpenMP teams (the two findings above), found by a micro-benchmark of four equal matmul
  tasks; fixed by the scripted driver and by `OMP_NUM_THREADS=1`.
