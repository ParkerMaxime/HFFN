# E0 — results

README: `README.md` (frozen before the first run). Qualified run: `runs/run_20261024T182902Z/` (EXIT 0; sweep 16/16 and
final 8/8 jobs `passed`, exit 0, full validation split). Analysis (new files, records untouched):
`runs/run_20261024T182902Z/analysis.json`, `analysis.md`. Classes: **measured** unless stated.

## Measured

Learning-rate sweep (seed 100, 2,000 steps, final validation bpb, `sweep/selection.json` of the qualified run):

| lr | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 | 3.2e-2 | 6.4e-2 |
|---|---|---|---|---|---|---|---|---|
| E0 | 2.9149 | 2.7423 | 2.6129 | 2.5227 | 2.4684 | 2.4509 | **2.4412** | 2.4586 |
| E3b | 3.4462 | 3.2791 | 3.1493 | 3.0621 | 3.0000 | 2.9726 | **2.9665** | 3.3293 |

Both optima are at **0.032**, interior to the grid, and coincide: **one rate, 0.032, is frozen for every arm of the
ladder.** The first sweep of the step (`runs/run_20261024T180451Z/`, grid up to 0.016) had its optimum at the grid
edge for both arms; by the README rule it was declared inconclusive and the grid was extended (see the limits). The
values of the 12 common grid points of the two sweeps are identical to the last digit (the training is reproducible
bit for bit with the same code, seed and thread count).

Final stage (seeds 0-3, 4,000 steps, lr 0.032, validation bits per byte over the 5 M validation bytes):

| arm | params (outside embedding) | seeds 0 / 1 / 2 / 3 | median | [min–max] |
|---|---|---|---|---|
| E0 | 93,184 | 2.3936 / 2.3913 / 2.3914 / 2.3975 | 2.3925 | [2.3913–2.3975] |
| NOATT | 27,520 | 3.8959 / 3.8946 / 3.8952 / 3.8945 | 3.8949 | [3.8945–3.8959] |

Per position class `t mod 4` (median): E0 2.417 / 2.401 / 2.383 / 2.370; NOATT 3.895 / 3.893 / 3.896 / 3.895.

- **`Δ_attn` = 1.5028 bpb** (per seed 1.5024, 1.5032, 1.5039, 1.4970), `d_s > 0` for the four seeds.
- **Band** (larger seed range) = 0.0062 bpb. **Ratio `Δ_attn` / band = 243.5** against `R_min = 10`.
- **Verdict (criterion of the README): the task is discriminating.** The ladder continues with E1.

Mixing diagnostic (descriptive, never a gate; increase of validation bpb on 4,096 windows when the cross-block part
is zeroed; the groups are nominal for E0, which has no group structure): W_Q +1.99, W_K and W_V +3.21, W_down +2.91,
all together +13.46 (median over seeds). NOATT: 0 (no attention, so nothing to zero for Q/K/V; W_down of NOATT is also
reported as zero because the diagnostic skips layers without attention — a limit of the code, not a finding).

## Reading (hypothesis unless stated)

- The gain of attention at this size is large (about 1.5 bits per byte, 39 % of the no-attention level), and the seed
  band is small, so a difference of 5 % of `Δ_attn` (0.075 bpb) is about twelve bands: variants of attention can be
  told apart here (measured ratio; the consequence is a reading).
- The FFN alone behaves as a byte-statistics model (NOATT is equal across the four position classes), which is the
  expected calibration (hypothesis).
- The E3b sweep shows, at the same learning rate, a level 0.52 bpb above E0 (2.9665 against 2.4412 at 2,000 steps,
  one seed): this is the first sight of the cost of the whole constraint; it is **not** a ladder result (E3b is judged
  at its own step, against E3a, at 4,000 steps and four seeds).

## Limits

- Harness events, kept in the records: (i) the first execution of the checks failed one check on a tolerance set too
  tight (README); (ii) the first run of the step (`run_20261024T180451Z`) was stopped by the agent right after its
  sweep, because both optima were at the grid edge (`ABORTED.md` in that folder; its four started final jobs were
  killed and have no figure). Neither is a candidate failure.
- One sweep seed and 2,000 steps choose the rate for 4,000-step runs; the optimum may move with length (not measured).
  The validation split serves for the choice and the report.
- Bits per byte use windows of at most 128 bytes; they are not comparable to published enwik8 figures.
- `torch` 2.14.1 (PyPI build) on CPU, 4 jobs × 1 thread; a different thread count may change the bits (not measured).
