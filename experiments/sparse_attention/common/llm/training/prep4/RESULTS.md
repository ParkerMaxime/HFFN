# prep4 — results

README: `README.md` (committed before the smoke). Run: `prep4/runs/smoke_20261027T201311Z/` (EXIT 0). No figure of quality or time is a result here; wall times only size the runs.

## Measured: harness

- Checks: `check_model.py` **219/219** (the 203 of the first three series plus the new arms), `check_series.py` **44/44**, `check_series3.py` **34/34**, `check_e14.py` **82/82** — all pass.
  The first three series stay reproducible (the second-series smoke bit for bit is a check of `check_series3.py`).
- **Packed form of `E2h`** (`gather`) equals `selected`: logits bit for bit at `M = 128`, within 2.4e-6 at `M = 256`; gradient of every parameter within 1.2e-7.
  **Executor kind `e2h`** equals the sequential forward (0.0 at `M = 128`, 3.1e-6 at `M = 256`); `REF4_W256` 0.0.
- **Decoding path**, 26 arms (every arm of the three campaigns and of the pairs): the decoded step equals the full forward at position `N/2` (reference 0.6e-6 to 2.4e-6;
  packed sequential step against the reference ≤ 1.2e-6), the step under the executor equals the sequential step **bit for bit** (0.0), repeated calls give identical logits,
  the filled part of the cache is untouched by a step. Per-group counts of the 64 new tokens of process 0 (route by byte): `[14, 21, 13, 16]` at `N = 128`, `[19, 17, 17, 11]`
  at `N = 512`.
- **Counter**: `E2h_L4` 452,608 inference multiply-adds (hand count equal), parameters those of `E4a_L4` plus 12,288 per layer at `M = 128` and 49,152 at `M = 256`; KV read per decoded token
  `E4a` / `E2h` 0.25, `E2h` / `REF4` 1.
- **Memory harness (e)**: equals the hand value on a synthetic pattern, also after a region that left tensors alive; two fresh processes of `E2h_L4` agree ((a) identical, (b), (c), (e) within 1 %).
  Finding while building it (README): the profiler's "Total Allocated" is not reset between two profiled regions of one process; (e) is taken relative to the first event of its region; (a)-(c) keep the
  method and the order of E10 (baseline 0 checked, the E10 figures of `E4a_L4` reproduced to the byte).
- **Concurrency gate: median ratio 0.377 ≤ 0.4, passed** (repetitions 0.396, 0.377, 0.427, 0.362, 0.283). The margin is smaller than at prep3 (0.310): the gate is a property of the machine at the moment
  of the run; a run whose gate fails makes no timing and says so.
- **SDPA kernel** (PLAN 17.7, limits): in the 12 arms recorded (`REF2`, `E4a_L4`, `IND_L4`, `E2h_L4` at `N` 128 and 512; `REF2_W256`, `E4a_L4_W256`, `E2h_L4_W256`, `E4a_L4_W256_h16`, `REF2_W256_h16`), in
  prefill and in decoding, the only attention operator is `aten::_scaled_dot_product_flash_attention_for_cpu` (all arms, the causal flag of the dense heads and the explicit mask of the constrained
  ones alike). Descriptive micro-measurement, one thread, 8,192 tokens, 2 heads, ms (`is_causal` / explicit mask): `N` 128, heads of 16: 3.41 / 3.46; `N` 512, heads of 16: 10.4 / 12.1; `N` 512,
  heads of 32: 14.3 / 17.4. The causal flag is 0 to 18 % faster than the explicit mask on these shapes, not the factor two that skipping the masked half would give: the kernel computes most
  of the masked half. The understatement of the gain of the attention alone in p1, p3, p5 (PLAN 17.7, limits) is therefore small on these shapes; it is not zero (measured, not applied).

## Smoke and sizing (estimated)

Smoke of the 25 cells in both modes (one process, 2 windows) and of the memory run on `E2h_L4` and `E4a_L4_W256` (two fresh processes each; the rule holds) passed. The ratios of a smoke are not results.
Re-estimate from the measured durations (build + warm-up of ≥ 30 calls and 5 s per arm + 12 windows of 6 slots ≥ 0.2 s, five processes per cell, nothing else running): **prefill ≈ 1.27 h, decoding ≈ 1.36 h,
memory ≈ 0.08 h, checks and gates ≈ 0.1 h per run; total ≈ 3.1 h** (the plan said 4.5-5.5 h of session with the preparation), under the 6 h of the brief: **no cut**; the cells at `M = 256` stay (they weigh
0.4 h in each mode).

## Limits

- The re-estimate extrapolates a 2-window smoke; the memory estimate averages an `M = 128` arm and an `M = 256` arm.
- The gate passed with a small margin; the real runs are not guaranteed to pass it.
