# prep3 — results

README: `README.md`. Run: `prep3/runs/run_20261026T205721Z/` (EXIT 0). No figure of quality or time; wall times only size the runs.

## Measured: harness

- Checks: `check_model.py` **203/203** (the 121 of the first ladder, the arms of the second series and the width-256 arms included), `check_series.py` **44/44**,
  `check_series3.py` **34/34** all pass. The second series stays reproducible: the default form under the hash route (`selected`) gives, to the last digit, the bits of the
  E8 smoke of `E4a_L4` (5.887834514427231).
- **Packed hash form (`gather`) = `selected`**: logits bit for bit at width 128 (E4a_L2, E4a_L4, IND_L4, both contexts), within 2e-6 at width 256; the gradient of every
  parameter within 6e-8 (tolerances 1e-4 and 1e-5, README), on batches of real bytes.
- **Counter**: the six arms of width 256 equal the table of PLAN 15.5 exactly (inference multiply-adds, parameters, KV values, training activations); the two head layouts of a
  family have equal parameters and multiply-adds.
- **Memory harness**: (a) equals the hand count on a synthetic module (25,088 B; weight-side 4,608 B), (b)/(c) equal the hand peak on a synthetic allocation pattern (6 MiB), and two fresh
  processes of `E4a_L2` give identical (a) and (b), (c) (E10 smoke: `E4a_L4` and `REF2`, identical in two processes).
- **Executor**: outputs equal the sequential forward (bit for bit at width 128, within 2.4e-6 at width 256) for ten arms. **Concurrency gate: median ratio 0.310 ≤ 0.4, passed.**
- **Timing harness**: protocol of 6 slots per window, slots of the required duration, steal/user/lscpu/torch recorded; a synthetic variant 1.5 times the control is recovered from `raw.jsonl`
  alone as 1.500, the copy as 1.000.

## Smoke and sizing (estimated)

Training smoke of the six arms of E13 (40 steps WSD, hash arms `gather`, 4 jobs in parallel) passed; E10 smoke on two arms passed; E11 smoke of the eight cells passed (one process, 2
windows each; the ratios of a smoke are not results). The dense control of E11 is the executor (README addendum). Phase pass of the variant (descriptive, smoke) works for the three kinds.
Re-estimate from the measured step times (4 workers, LPT; wall): **E13 ≈ 7.6 h, E12 ≈ 3.5 h (3 rates plus one extension per arm), E10 ≈ 0.6 h, E11 ≈ 1.0 h: total ≈ 12.6 h**, under the 24 h of the
brief; the series goes on to E12-E13 after E11. The packed form makes the hash-constrained training arms cost about as much per step as the dense arm of the same depth at width 256
(`E4a_L4_W256` against `REF2_W256`: 1.1 times), against 1.5-2 times in the `selected` form (hypothesis on cause: only the routed tokens are computed); not a result.

## Limits

- The smoke times 40 steps and 2 windows with the machine busy with other jobs of the smoke; E11 will run alone. The padded batch of the packed form (largest count per group) computes up to
  about 1.3 times the routed tokens (hypothesis, not measured).
- `IND` keeps a barrier per layer in the executor (README): the cells c do not measure its absence.
