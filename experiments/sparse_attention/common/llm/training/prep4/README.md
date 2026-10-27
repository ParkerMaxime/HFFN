# prep4 — preparation of E14 (the frontier of quality against measured inference time)

Written on 2026-10-27, before the smoke of this step and before any run of E14. No figure of quality or of time is a result here.
Plan: `current/sparse_attention/PLAN.md` sections 17.1-17.7 (the plan on the branch has section 17.7: checked); brief:
`experiments/sparse_attention/CLOUD_E14.md`. Starting commit: `8e0a7ff` (the plan and brief on top of the third series, `440da1e`).
Rules still hold: README before the first run, run records never edited, evidence classes, nothing promoted, no absolute time as a result.

## What changes in `lib/` (every new behaviour is an option or a new module; the first three series stay reproducible)

| Item | Default | New behaviour |
|---|---|---|
| Arms (`lib/arms.py`) | — | `E2h_L4` (`E2` with `routing="hash"`, 4 layers: dense attention, dense `W_O`, the routed FFN of `E4a`), `E2h_L4_N512`, `E2h_L4_W256`, `REF4_W256` |
| Packed form of `E2h` (`lib/model.py`, `impl="gather"`) | `selected` under the hash route | dense attention over every token, then `W_O` and FFN group `g` on the tokens routed to `g`, gathered once per forward and padded to the largest count of the batch (the tail of the `E4a` packed form); `E4a`/`IND` code paths unchanged |
| Executor kind `e2h` (`lib/executor.py`) | — | the dense heads (`run_dense_heads`), a barrier, then `W_O` (dense) and FFN group `g` on worker `g` for the tokens routed to `g` (`run_e2h_tail`) |
| Decoding path (`lib/decode.py`, new) | — | a KV cache per layer and per sequence for every arm; `prefill_caches` (untimed fill of the first `N/2` positions), `decode_reference` (PyTorch, every head computed and masked by the route: the independent reference), scripted workers `dec_*` and `Decoder.step(tok, p, parallel)` (packed step; `parallel=False` runs the same workers one after the other, `True` as a fork-join of four) |
| Timing cells (`lib/timing4.py`, new) | — | the 19 frontier cells and the pairs p1-p6 of PLAN 17.3 and 17.7 in both modes, on the timing harness of E11 (`lib/timing.py`, unchanged) |
| Memory (`lib/memory4.py`, new) | — | (a)-(c) of E10 by `lib/memory.py` (unchanged) and (e), the peak of live tensor bytes of one decoded step (sequential packed step) |
| `lib/e14.py`, `lib/prep4.py`, `checks/check_e14.py` | — | launchers of E14 and of this step; 82 new checks |

## Decisions of implementation (stated before the smoke)

- **Decoded step.** `B = 64` sequences, windows of `N` bytes of the valid split at offsets drawn per process (seeded by the process index, equal for
  the three arms of a cell-process); the cache holds the keys (after RoPE) and values of positions `0 … p−1`, `p = N/2`, filled by an untimed forward of
  the first `p` positions; the timed step takes the byte at position `p` of each window, writes the keys and values of the 64 new tokens into slot `p`
  of the preallocated cache (the same slot at every call) and attends to positions `0 … p`. The embedding, the route (hash) and its plan, the layers and
  the logits are inside the timed call, as in prefill (where the plan is inside `forward`).
- **Constrained arms (`E4a`, `IND`).** Per layer, one fork-join: worker `g` computes the keys and values of its heads `H_g` for the 64 tokens (dense,
  decision 3) and writes them into the cache slice of `H_g`; for the sequences routed to it, the queries of `H_g`, the scores against the cache of their
  sequence, `W_O^g` and FFN group `g`, and writes the new residual rows of those sequences. Worker `g` reads and writes only the heads `H_g`, so no
  barrier is needed inside a layer. **Dense arms:** worker `g` computes queries, keys, values and the attention of the heads `H_g` for the 64 tokens;
  barrier; `W_O` and the FFN on 16 rows per worker. **`E2h`:** the dense-arm heads, barrier, then `W_O` (dense) and FFN group `g` on the sequences routed to `g`.
- **Padding (reading of PLAN 17.3, "the packed form pads each group to its largest count at that step").** The sequences routed to the four groups are
  padded to the largest of the four counts of the step (shapes equal for the four workers); the padded rows are computed and not written. The per-group
  counts of every step are recorded in `cell.json`. This conservative reading costs the packed form up to about 30 % of extra rows
  (counts `[14, 21, 13, 16]` at `N = 128`, `[19, 17, 17, 11]` at `N = 512` for process 0); it is part of what is measured.
- **Gather of the cache.** The sequences routed to a group are selected from the cache slice of their heads (`index_select` on the batch axis of
  `K[:, H_g, :p+1]`, a copy of `P × hpg × (p+1) × d_head` values per key and per value tensor and layer). It is the packed form of the routed sequences;
  a native kernel could read the cache in place. It is part of what is measured and a limit.
- **Timing rule.** That of 15.3 unchanged (`lib/timing.py`): five fresh processes per cell, anchor, copy of the anchor and variant built independently in the
  same process, warm-up of at least 30 calls and 5 s per arm, 12 windows of 6 slots (anchor, copy, variant, variant, copy, anchor) of at least 0.2 s;
  per cell the median of the per-process medians and [min–max]; **`delta` = the largest `|copy / anchor − 1|` over all cell-processes of the run**
  (one `delta` per mode); steal time, load average, `lscpu` and the torch configuration recorded; nothing excluded after the fact. No phase pass
  in E14 (it is not in the plan).
- **Memory (e).** The peak of live tensor bytes of one decoded step by the method of (c): the memory events of `torch.profiler`. The model and the
  caches are alive before the region (the cache is counted, not measured: PLAN 17.6); the working buffers of the step are released before the region
  and allocated inside it. The step measured is the sequential packed step (one thread). **Finding while building it:** "Total Allocated" of the profiler is a
  running total that is not reset between two profiled regions of one process (a region that leaves tensors alive, such as the gradients of (b), shifts the next
  one by 3.29 MB); the peak (e) is therefore taken relative to the first event of its region (`lib/memory4.py`, checked on a synthetic pattern after a region that
  leaves tensors alive). (a)-(c) are measured by the method and in the order of E10 (a clean process, (c) first; (b) has baseline 0, checked), so
  the E10 figures are not affected. Rule of E10: two fresh processes, (a) identical, (b), (c), (e) within 1 %, otherwise a harness failure.
- **Checks of the E14 runs.** The preparation runs the four suites (`check_model`, `check_series`, `check_series3`, `check_e14`); each run of E14 runs
  `check_e14.py` only (the code of `lib/` does not change between the preparation and the runs; a harness fix would rerun the four suites).
- **Counted ratios of a decode cell:** the counter with `N_k = N / 2` keys (the cache) for the multiply-adds; the keys and values read per decoded token
  `L × 2 × (M or M/G) × N/2` (`M/G` for the routed attention, `M` for the dense attention and `E2h`); stored `2 L M`.

## Tolerances (stated before the run)

Float32 rounding of the same computation in another order: the logits of the decoded step against the full forward at the same position, of the packed
sequential step against the reference, of the step under the executor against the sequential step, and of `E2h` (packed form, executor) against the
sequential forward, within **1e-4** (max absolute difference); the gradient of every parameter of the packed `E2h` against `selected` within **1e-5**.
Observed: 3e-6 at most for the logits, 1.2e-7 for the gradients; the executor equals the sequential step bit for bit (0.0).

## Counted (estimated; `lib/count.py`)

| Arm | Inference MACs per token (prefill, `N_k = N`) | Parameters outside the embedding | KV values per token (stored) | Training activations per token |
|---|---|---|---|---|
| `E2h_L4` (128) | 452,608 | 372,352 | 1,024 | 6,144 |
| `E2h_L4_N512` | 845,824 | 372,352 | 1,024 | 18,432 |
| `E2h_L4_W256` | 2,269,184 | 1,481,984 | 2,048 | 20,480 |
| `REF4_W256` | 2,588,672 | 1,481,984 | 2,048 | 26,624 |
| `E4a_L4` / `E4a_L4_N512` / `E4a_L4_W256` (reference) | 243,712 / 342,016 / 1,040,384 | 323,200 / 323,200 / 1,285,376 | 1,024 / 1,024 / 2,048 | 2,304 / 5,376 / 6,656 |
| `REF4` / `REF4_N512` (reference) | 532,480 / 925,696 | 372,352 | 1,024 | 9,216 / 21,504 |

`E2h_L4` has the parameters of `E4a_L4` plus 12,288 per layer (`W_O`) at `M = 128` and 49,152 per layer at `M = 256` (checked). Counted ratios of the pairs
(prefill / decode; KV read per decoded token): p1 0.538 / 0.587, 0.25; p2 0.850 / 0.829, 1; p3 0.404 / 0.474, 0.25; p4 0.914 / 0.880, 1; p5 0.458 / 0.521, 0.25;
p6 0.877 / 0.845, 1.

## What this step does

`./run.sh check` (the four suites) and `./run.sh smoke` (= `run`): the checks; the concurrency gate of 15.3; the record of the SDPA operators of 12 arms in
both modes and a descriptive measurement of the causal flag against an explicit mask on the shapes of the cells (PLAN 17.7, limits: "the README of E14
records the SDPA kernel selected in each arm"); the smoke of every cell in both modes (one process, 2 windows, 10 warm-up calls and 1 s); the memory run on two
arms (`E2h_L4`, `E4a_L4_W256`; two fresh processes each); the sizing, from the smoke records, and the re-estimate of the total. The smoke proves nothing;
the ratios of a smoke are not results. **Stop rules (brief):** if the concurrency gate fails, E14 makes no timing and says so; if the re-estimate exceeds
6 hours, the cells at `M = 256` (frontier and p5-p6) are cut first and the cut is written in the README of `e14` before its runs.

## Limits

- The smoke times two windows with short warm-up; the re-estimate extrapolates (build time + warm-up + 12 windows of ≥ 0.2 s slots). Memory processes at
  `M = 256` take longer than the smoked arm at `M = 128`; the estimate averages the two smoked arms.
- The executor is that of E11; the FFN is masked dense in every arm; the decoded step of the constrained arms gathers the cache slices (above).
