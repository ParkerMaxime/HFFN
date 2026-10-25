# prep — results

README: `README.md`. Run: `prep/runs/run_20261025T093927Z/` (EXIT 0): checks `check_model.py` 179/179 and `check_series.py` 44/44, 16 arms for 40 steps of WSD
(8,192 bytes per step, seed 0, 4 jobs in parallel × 1 thread), one plateau-and-branch pair (`smoke_branch/`). All jobs exit 0.
No figure of quality is drawn from a smoke. Wall times below only size the runs and are not results (decision 9).

## Measured: harness

- Every check passes. The branch from a checkpoint equals an uninterrupted WSD run **bit for bit** (weights and bits per byte, `E4a` and
  `E4b`, `H = 100`, checkpoint at 80; stated tolerance 1e-6, observed 0). The cosine run of the E0 smoke is bit-identical to its
  first-ladder record. The first ladder therefore stays reproducible with the new options off.
- No harness defect in this step.

## Counted and sized (estimates)

| arm | s/step (smoke, 4 jobs) | final valid evaluation, s per 256 windows | inference MACs/token | training MACs/token (3x) | params (outside emb.) | AdamW state values | KV values/token | training activation values/token |
|---|---|---|---|---|---|---|---|---|
| REF1 | 0.236 | 0.3 | 157,696 | 473,088 | 93,184 | 186,368 | 256 | 2,304 |
| REF1_N512 | 0.301 | 2.5 | 256,000 | 768,000 | 93,184 | 186,368 | 256 | 5,376 |
| REF2 | 0.444 | 0.5 | 282,624 | 847,872 | 186,240 | 372,480 | 512 | 4,608 |
| REF2_N512 | 0.644 | 4.1 | 479,232 | 1,437,696 | 186,240 | 372,480 | 512 | 10,752 |
| REF3 | 0.611 | 0.8 | 407,552 | 1,222,656 | 279,296 | 558,592 | 768 | 6,912 |
| REF3_N512 | 0.875 | 6.4 | 702,464 | 2,107,392 | 279,296 | 558,592 | 768 | 16,128 |
| REF4 | 0.871 | 1.2 | 532,480 | 1,597,440 | 372,352 | 744,704 | 1,024 | 9,216 |
| REF4_N512 | 1.122 | 8.6 | 925,696 | 2,777,088 | 372,352 | 744,704 | 1,024 | 21,504 |
| E4a_L2 | 0.469 | 0.7 | 138,240 | 414,720 | 161,664 | 323,328 | 512 | 1,152 |
| E4a_L2_N512 | 0.615 | 5.3 | 187,392 | 562,176 | 161,664 | 323,328 | 512 | 2,688 |
| E4a_L4 | 0.914 | 1.2 | 243,712 | 731,136 | 323,200 | 646,400 | 1,024 | 2,304 |
| E4a_L4_N512 | 1.193 | 11.1 | 342,016 | 1,026,048 | 323,200 | 646,400 | 1,024 | 5,376 |
| E4a_L6 | 1.283 | 1.8 | 349,184 | 1,047,552 | 484,736 | 969,472 | 1,536 | 3,456 |
| E4a_L6_N512 | 1.809 | 15.3 | 496,640 | 1,489,920 | 484,736 | 969,472 | 1,536 | 8,064 |
| IND_L4 | 0.892 | 1.5 | 133,120 | 399,360 | 175,744 | 351,488 | 1,024 | 2,304 |
| IND_L4_N512 | 1.196 | 11.1 | 231,424 | 694,272 | 175,744 | 351,488 | 1,024 | 5,376 |

The inference column equals PLAN 14.5 for the seven models it lists, at both contexts. `IND_L4` (counted for the first time): 133,120
at `N = 128`, 231,424 at `N = 512`. The wall time per step is the time of the 40 steps of the smoke divided by 40 (first-step overhead
included, 4 jobs on 4 cores, 1 thread each): the hash-routed arms use the selected form of the constraint (every head and neuron
computed, the other groups multiplied by zero), so **in this harness they cost about as much per step as the dense arms of the same
depth** (E4a_L4 0.914 against REF4 0.871 s/step). That is a property of the harness, not of the method: compute saved is counted in the
inference column, never timed.

## Re-estimate of the total (from the smoke; wall time, estimated)

Per (arm, seed) the unit costs are the step times above plus the full validation evaluation (39,062 windows at 128, 9,765 at 512).
Assuming `H* = 4,000` (the case the plan costs), four jobs in parallel:

| Step | Work | Wall (estimate) |
|---|---|---|
| E6 | 2 arms × 2 seeds × (6,400 plateau + 2,800 branch) steps; `E4a_L4` 0.914 s/step is the long pole | ≈ 2.4 h |
| E7 | 4 arms × 3 rates (+ up to two extensions per arm) × 4,000 steps | ≈ 3.3 h |
| E8 | 8 arms × 3 seeds × 4,000 steps + validation and test evaluations | ≈ 5.3 h |
| E9 | the same at 512 (step × 1.3-1.45; evaluation of 9,765 windows costs 1.5-6 min per split) | ≈ 7.5 h |
| **Total** | | **≈ 18.5 h** (+ preparation ≈ 0.2 h) |

Below the limit of 24 hours of the brief, so the series goes on to E6. If E6 retains `H* = 2,000` the later steps take about half; if it
retains 8,000 the series stops after E6 and the user decides (brief). The plan's 16 hours is optimistic by about 2.5 hours because the
hash-routed arms are not faster per step in this harness.

## Limits

- The step times come from 40-step smokes with the machine fully loaded by 4 jobs; a plateau of 6,400 steps may run at a slightly
  different rate. Full-validation evaluation times are extrapolated from 256 windows.
- `IND_L4` is counted with block-diagonal keys and values (non-zero entries), which is why it is cheaper than `E4a_L4`.
