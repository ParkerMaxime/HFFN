# E14 — the frontier of quality against measured inference time, and the attention alone

Written on 2026-10-27 after `prep4` (`../prep4/RESULTS.md`) and before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 17.1-17.7; brief:
`experiments/sparse_attention/CLOUD_E14.md`. Harness: `../prep4/README.md` (all decisions of implementation and the tolerances are there and hold here). No training: the quality of an arm
is its validation bits per byte per seed already on record (E8 `N = 128`, E9 `N = 512`, E13 `M = 256`), read, never edited. E14 measures time and memory.

## Single question

Decision 16 (PLAN 17.2): is the constrained family (`E4a`) above the frontier of quality against **measured** inference time of the dense-attention family, in prefill and in batched
decoding, at `M = 128` (`N = 128`, `N = 512`) and at `M = 256` (`N = 512`)? And, separately (PLAN 17.7): what does the sparse attention alone gain or lose in time and memory at equal depth
(`E4a_L4` against `E2h_L4`), and what does the routing of the FFN alone (`E2h_L4` against `REF4`)?

## Design (frozen)

- **Cells** (50 = 25 per mode; `step.json`): the 19 frontier cells — anchor `REF2` with variants `REF1, REF3, REF4, E4a_L2, E4a_L4, E4a_L6, IND_L4` at `N = 128` and (suffix `_N512`) at `N = 512`;
  anchor `REF2_W256` with `REF1_W256, REF2_W256_h16, E4a_L2_W256, E4a_L4_W256, E4a_L4_W256_h16` — and the pairs p1-p6 (PLAN 17.7: p1 `E4a_L4`/`E2h_L4`, p2 `E2h_L4`/`REF4`, p3-p4 the same at
  `N = 512`, p5-p6 at `M = 256`). The time of an arm is `t = t(arm) / t(anchor)`, the anchor at 1. Weights random (seed 0), inputs bytes of the valid split.
- **Two modes, a run of its own each, one `delta` per run**: *prefill* (one forward over whole windows, 8,192 tokens per call, the executor of E11) and *batched decoding* (`B = 64` sequences, one new token each at
  position `N/2`, a cache of `N/2` positions; the step at `N/2` is the mean step over a window of `N` for the attention, estimated). The dense control is the executor (E11, README addendum of `prep3`).
- **Timing rule** of 15.3, unchanged (`../prep4/README.md`): five fresh processes per cell, anchor, copy and variant built in the same process, warm-up ≥ 30 calls and 5 s, 12 windows of
  6 slots ≥ 0.2 s; per cell the median of the per-process medians and [min–max]; `delta` of a run = the largest `|copy / anchor − 1|` over all its cell-processes; steal time and load recorded; nothing excluded.
  **Nothing else runs on the machine during a timing run** (no check, build, smoke or second process; the session waits). The concurrency gate (≤ 0.4) runs at the start of each timing run; if it
  fails, the run makes no timing, says so, and E14 stops (a property of the machine or of the executor, not a harness failure).
- **Order of the runs:** prefill, decoding, memory. Memory: the nine arms of p1-p6, (a)-(c) of E10 and (e) (decoded step), two fresh processes each, the rule of E10 (a identical, the others within 1 %).
  Each run runs `check_e14.py` first (82 checks).
- **Re-estimate after the smoke** (`../prep4/RESULTS.md`): ≈ 1.27 h prefill, 1.36 h decoding, 0.08 h memory, ≈ 3.1 h in all with the checks, **under the 6 h of the brief: no cell is cut.**
- **SDPA kernel** (PLAN 17.7, limits; recorded in `../prep4/RESULTS.md`): `aten::_scaled_dot_product_flash_attention_for_cpu` in every arm, in both modes (the causal flag of the dense heads and the explicit
  mask of the constrained heads alike). The causal flag is 0-18 % faster than the explicit mask on the shapes of the cells (descriptive micro-measurement): the understatement of the gain of the attention alone
  in p1, p3, p5 is small and not applied.

## Readings (frozen; `lib/analyze_e14.py` implements them, tested on the smoke records before this README)

`τ = 0.075` bpb. `d_s` and gaps are per seed; seeds 0, 1, 2 of E8, E9, E13.

**The dense frontier.** Per campaign (n128, n512, w256), mode and seed: the points `(t_d, bpb_s(d))` of the dense arms of the campaign (`REF1..REF4`; at `M = 256` `REF1_W256`, `REF2_W256`, `REF2_W256_h16`); a point is
dropped from the lower envelope if another is at most as slow and at most as high (one strict); the frontier is the piecewise-linear interpolation in `log t` between the points of the envelope, flat at the lowest bpb beyond
the last point of the envelope, defined for `t` between the fastest and the slowest dense time only.

**The gap** of a constrained arm `c` (`E4a_*`; `IND_L4` as a control): `g_{c,s} = bpb_s(c) − frontier_s(t_c)`, positive = worse. `g⁺`: `t_c × (1 + delta)` and every dense time `× (1 − delta)` (the anchor included);
`g⁻`: the reverse (multiplicative shifts, the reading of "moved by `delta`" fixed here). Outcomes: *above the frontier* if `max_s g⁺ < 0`; *below the frontier beyond τ* if `min_s g⁻ > τ`; otherwise *on the frontier*
(median `g` and [min–max] stated against τ); *outside the range* if `t_c` lies beyond the fastest or the slowest dense time under either reading — then no frontier reading, a pair against the nearest dense arm in log
time: `d_s = bpb_s(c) − bpb_s(nearest)` with the outcomes of 13.1 (*within* if `max_s d_s ≤ τ`, *beyond* if `min_s d_s > τ`, otherwise *inconclusive*) and *dominates* if `t_c (1 + delta) < t_nearest (1 − delta)` and
`max_s d_s < 0`; the time ratio beside. **Answer to decision 16**, per mode and campaign, the two modes stated apart and not combined: *favourable* if at least one `E4a` arm is above the frontier; *unfavourable* if every
`E4a` arm is below it beyond τ; otherwise *not resolved*, with the gaps stated.

**Pairs p1-p6**, per cell and mode: the ratio (median [min–max] over the processes) and `delta` of the run; *gain* of the variant only if `1 − ratio > delta`, *loss* only if `ratio > 1 + delta`, otherwise *no difference
within delta*. Beside, descriptive: the counted ratio of the cell and the quotient measured / counted; the products p1 × p2, p3 × p4, p5 × p6 against the counted ratio `E4a_L4` / `REF4` (**estimated**: two cells combined).
In decoding the counted multiply-adds use `N/2` keys, and the keys and values read per decoded token (`E4a` / `E2h` 0.25, `E2h` / `REF4` 1) and stored (1) are counted beside.

**Memory**, as measured, without `delta`: the ratios `E4a_L4` / `E2h_L4` and `E2h_L4` / `REF4` of (a) the tensors saved for the backward (token side), (b) the training-step peak, (c) the prefill peak and (e) the
decoded-step peak per decoded token, at the three sizes, beside the counted activations; parameters, AdamW states and the KV cache stay counted.

**Classes.** Times and memory: measured (E14). Bits per byte: measured (E8, E9, E13), each axis compared only inside its own campaign. Every gap and every product: **estimated** (an interpolation of two measured
axes). The direct pairs (an arm against its anchor in time and in bits per byte) and the cells p1-p6: measured. Expected effects (a more favourable decoding at a long cache, 17.4) are hypotheses: E14 can contradict them.

## Limits stated before the run

- Everything holds for the PyTorch executor of E11 on this shared 4-vCPU machine, forward only, with the FFN masked dense in every arm; a native implementation may move every point either way. A favourable
  answer is not a statement on the HFFN runtime; a masked dense FFN overstates the gain of the routing of the FFN there (p2, p4, p6; hypothesis).
- No new quality figure: the bpb are those of E8, E9, E13 with their reserves (rates of the `_h16` arms and of the middle depths computed, `H*` not recalibrated at width); `E2h` has no bpb and is not on the
  frontier: p1-p6 give no quality per time. At `M = 256` the dense frontier has three points; a constrained arm slower than all of them is outside the range.
- One batch size (64), one cache length per context (`N/2`), no generation of whole sequences; the cache is counted, not measured, outside the peak (e). In decoding the packed form pads the four groups to the largest
  count of the step and gathers the cache slices of the routed sequences (`../prep4/README.md`): a cost of this executor, part of what is measured.
- The concurrency gate passed with a small margin at the preparation (0.377); a run whose gate fails makes no timing.

## Launch

`./run.sh check | smoke | prefill | decode | memory`. Records: `runs/prefill_<UTC>/`, `runs/decode_<UTC>/`, `runs/memory_<UTC>/`, each with `EXIT` and `manifest.json`; the analysis
(`python -m lib.analyze_e14 --step-dir e14 --prefill runs/prefill_… --decode runs/decode_… --memory runs/memory_…`) writes `runs/analysis_<UTC>/`.
