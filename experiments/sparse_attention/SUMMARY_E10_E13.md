# sparse_attention — summary of the third series (E10-E13), cloud session of 2026-10-26/27

Branch `sparse-attention-cloud`, from `a08b259` (the plan of the series and its brief). Plan: `current/sparse_attention/PLAN.md` section 15; brief:
`experiments/sparse_attention/CLOUD_E10_E13.md`. `SUMMARY.md` and `SUMMARY_E6_E9.md` are records and are not rewritten. Code and records:
`experiments/sparse_attention/common/llm/training/` (`prep3/`, `e10/` … `e13/`, each with the README committed before its first run, `step.json`, `run.sh`, `runs/`, `RESULTS.md`).
Every figure is **measured** in the qualified run named in the step's `RESULTS.md` unless marked **estimated** or **hypothesis**; nothing is promoted to `current/` or `publication/`. Setting: PyTorch on CPU
(4 vCPU virtual machine, torch 2.14.1 CPU), enwik8 bytes, WSD schedule with horizon 4,000, 8,192 bytes per step, `τ = 0.075` bpb. Time appears only as ratios of E11, and only as ratios of that executor.

## Preparation and cost

203 + 44 + 34 checks pass (the second series stays reproducible bit for bit). The packed hash form (`gather`) equals the `selected` form (logits bit for bit at width 128, within 2e-6 at width 256;
gradients within 6e-8). The counter equals PLAN 15.5 for the six arms of width 256. The memory harness equals hand counts on synthetic modules; the executor equals the sequential forward; the concurrency
gate of the timing harness passed (0.310 ≤ 0.4). Wall time: preparation 0.2 h, E10 0.1 h, E11 0.5 h (after one interrupted launch), E12 5.5 h, E13 8.8 h, about 15 h against the 12.6 h re-estimated after
the smoke (the stop rule of 24 h was not triggered). E12 and E13 ran longer than estimated: an extension round ran alone on one core (E12); the last jobs of E13 left cores idle.

## E10, memory of the implementation against the counter (measured; 24 configurations, two fresh processes each, rule held everywhere)

- **The counted saving of training activations is not reached by this implementation.** The counter gives half the activations of the dense model at equal counted multiply-adds and a quarter at equal depth.
  Measured, the tensors saved for the backward are **1.6-1.9 times** the dense model's at equal counted multiply-adds (1.59-1.65 at width 256) and **0.89-1.00** at equal depth; the prefill peak is **0.78**.
- Hypothesis for the gap (read from the numbers): the dense keys and values, normalised rows, gathered rows and logits are saved and are not in the counter's four terms; the CPU attention kernel does not store
  the scores the counter counts; the packed form pads each group. The counted bound is a bound on four terms, not a prediction for this harness.
- The `selected` form costs more than the packed one (packed/selected 0.74 at 128, 0.83 at 512 for saved bytes; 0.57 / 0.56 in prefill): a quarter of the first two series' harness memory was its form.
- Weights, AdamW states and KV cache stay counted (1.7 times the weights and states, twice the KV cache, at equal counted multiply-adds).

## E11, relative time of the executor (measured ratios; 8 cells × 5 processes; `delta` = 0.0975; forward only)

- **One constrained layer runs in 0.40-0.60 of the time of a dense layer** in the four layer cells (faster beyond `delta`); time follows the count approximately (r / counted 0.96-1.56).
- **At the model level the constrained model is slower than the dense one at equal counted multiply-adds** on this executor: `E4a_L4`/`REF2` 1.29 (N 128) and 1.13 (N 512), both beyond `delta`, against counted
  0.86 and 0.71. Hypothesis: the plan, the gathers and the padding of four routed layers are paid with less work per layer to hide them behind; not isolated.
- `IND_L4`/`E4a_L4`: faster at N 128 (0.85), not resolved at N 512 (0.92; a smaller `delta` would resolve it as faster, stated and not applied).
- Specific to this PyTorch executor on this shared machine; says nothing of another machine, of training time or of the HFFN runtime. The first launch was interrupted by a container restart (harness event,
  `ABORTED.md`, no figure); the full relaunch used the same README.

## E12, the rates at width 256 (seed 100, one seed each: they select, they prove nothing)

`REF2_W256` **0.004** (one extension: 0.002 ended higher), `E4a_L4_W256` **0.004** (no extension); none ends at an edge. Computed for E13: `REF1_W256` 0.00504, the three other arms 0.004. The rates are about half
those of `M = 128` for the constrained family. `E4a_L4_W256` is flat between 0.002 and 0.004 (0.0007 apart) and ends 0.124 higher at 0.008; `REF2_W256` is 0.0104 higher at 0.008.

## E13, the frontier at width 256 (validation bits per byte, median over three seeds; compute from the counter)

| arm | valid bpb | inference MACs/token |
|---|---|---|
| E4a_L2_W256 | 2.1925 | 552,960 |
| REF1_W256 | 2.1821 | 696,320 |
| E4a_L4_W256_h16 | 1.9435 | 1,040,384 |
| E4a_L4_W256 | 1.9867 | 1,040,384 |
| REF2_W256_h16 | 1.9855 | 1,327,104 |
| REF2_W256 | 1.9880 | 1,327,104 |

## Readings, with outcome and class (`d_s = candidate − control`, positive = candidate worse; three seeds: a direction, not a size)

| Reading | Median [min–max] | Outcome | Class |
|---|---|---|---|
| `E4a_L2_W256`/`REF1_W256` (−20.6 % counted compute) | +0.0091 [−0.0094, +0.0213] | within; not dominating | measured |
| `E4a_L4_W256`/`REF2_W256` (−21.6 %) | +0.0014 [−0.0133, +0.0165] | within; not dominating | measured |
| equal depth `E4a_L2_W256`/`REF2_W256` (−58.3 %) | +0.2072 [+0.1750, +0.2153] | beyond | measured |
| `E4a_L4_W256_h16`/`REF2_W256_h16` (−21.6 %), 4 heads per group | −0.0420 [−0.0435, −0.0328] | within; **dominates** | measured |
| heads per group `i_s` | −0.0342 [−0.0599, −0.0287] | **favours four heads per group** (`|i_s|` below `τ` in every seed: 0.38, 0.46, 0.80 of `τ`) | measured |

Test split, beside: the same outcomes. Position ranges: 128-255 and 256-511 within 0.002 of each other in every arm (descriptive).

## What the series says

- **Measured (E13, against the frozen readings):** at twice the width and equal counted multiply-adds (21-22 % saved), the constrained model is within `τ` of dense with two heads per group but does **not** dominate it
  (seeds of both signs), and dominates with four heads per group (three seeds negative). The equal-depth loss remains beyond `τ` (+0.21). The heads-per-group reading favours four.
- **Cross-campaign (outcomes only):** E9 at `M = 128` read *within and dominates* for the two equal-compute pairs; at `M = 256` with the same two-heads layout the pairs read *within*, not dominating. The equal-depth outcome is
  *beyond* in both. The four-heads layout has no counterpart in E9.
- **Descriptive, stated after the run:** the favourable `i_s` comes from the constrained arm gaining with 16 heads (1.9867 → 1.9435) while the dense arm barely moves (1.9880 → 1.9855).
- **Across E10-E11-E13 (primary criterion against implementation):** quality at equal counted compute is within or better; the counted memory saving is not reached by this harness (E10); time at the model level is not
  better on this executor (E11) although it is at the level of one layer. These statements are about this harness and this executor, not about the method or about a native implementation.

## Observations outside the registered readings (proposals for the user, not claims)

1. **Heads per group at `M = 128`**: E9's layout is 8 heads of 16 (2 per group); a 16-head layout at `M = 128` (heads of 8) and, at `M = 256`, a 32-head layout would separate the number of heads per group from the head size and from the width,
   which E13 changes together with the layout (head size 32 → 16 at fixed width). Rates for the 16-head layouts were not tuned (limit of E13); a sweep at their own rate, and more seeds, would test whether `i_s` is a direction or noise.
2. **The model-level slowdown of E11** is a statement about this executor; fusing the plan and the gathers into the keys-values phase, or a persistent padded layout across layers, would be a different executor, and would need its own E11.
3. **The memory gap of E10** points to the saved dense keys and values and the padding as the terms to attack; a counter that separates the stored tensors of the harness from the four counted terms would make the quotient a bound check.
4. **E13 at three depths of the dense family** (a third dense point at width 256) would give the interpolation in `log C` that E8-E9 had and E13 could not read.

## Not run, not verified, limits (PLAN 15.6 included)

- Not run: any absolute time; time on another machine; training time (the backward is not divided by group in this executor); decoding with a KV cache; a native kernel; the HFFN runtime (the FFN is masked dense in every arm);
  measured memory of weights, AdamW states and the KV cache (counted); the mixing diagnostic at width; specialised FFN groups (decision 15).
- Not verified: the hypotheses named above (cause of the memory gap, cause of the model-level slowdown, cause of the favourable `i_s`).
- Limits: three seeds; the rates of four arms (E13) and of the 16-head layouts are computed from E12's single seed, not swept; the horizon 4,000 is not recalibrated at width; one width, one context (512), two depths of the dense family;
  the validation split served the choice of the rates and the figures; the test split was read once and selects nothing.

## Harness events

- The first launch of E11 (`e11/runs/run_20261026T212801Z`) was interrupted by a container restart after cells a1-a4 and b1; classified as a harness event, recorded (`ABORTED.md`), no figure taken from it; the full relaunch used the
  same README and rules. No other run was aborted, repeated or excluded; no threshold, grid, rate rule or reading was changed after its step ran.
