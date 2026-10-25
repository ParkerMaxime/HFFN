# E7 — results

README: `README.md` (frozen before the run). Qualified run: `e7/runs/run_20261025T120721Z/` (EXIT 0; 16 WSD runs of 4,000 steps, seed 100, all `passed`, exit 0,
full validation split; round 1: 12 runs, round 2: 4 extension runs). Selection record: `sweep/selection.json`. Classes: **measured**
(one seed: it selects, it proves nothing).

## Measured: final validation bits per byte by peak rate (WSD, `H* = 4,000`, `N = 128`, seed 100)

| rate | 0.002 | 0.004 | 0.008 | 0.016 | 0.032 | 0.064 | selected |
|---|---|---|---|---|---|---|---|
| `REF1` (centre 0.032) | — | — | 2.3813 (ext.) | **2.3639** | 2.3718 | 2.3960 | **0.016** |
| `REF4` (centre 0.016) | — | 2.0323 (ext.) | **2.0273** | 2.0459 | 2.0703 | — | **0.008** |
| `E4a_L2` (centre 0.016) | — | 2.3063 (ext.) | **2.2940** | 2.2969 | 2.3685 | — | **0.008** |
| `E4a_L6` (centre 0.004) | 2.0824 | 2.0453 | **2.0392** | 2.0625 (ext.) | — | — | **0.008** |

(ext.) = rate added by the edge rule of the README. In round 1 the best rate of all four arms was at an edge of its grid of three
(`REF1` 0.016 (low), `REF4` 0.008 (low), `E4a_L2` 0.008 (low), `E4a_L6` 0.008 (high)); one extension in the direction of the edge each
(`REF1` 0.008, `REF4` 0.004, `E4a_L2` 0.004, `E4a_L6` 0.016) placed the best rate in the interior in every case: **no arm needed a
second extension and none ends at an edge** (`at_edge`: false for the four).

## Rates for E8 and E9 (computed from this selection; written in the README of E8 before its runs)

`REF1` 0.016, `REF4` 0.008 (selected); `E4a_L2` 0.008, `E4a_L6` 0.008 (selected); the other arms are computed, not swept: `REF2` and `REF3`
geometric in `L` between `REF1` and `REF4` (0.016 · 0.5^((L−1)/3): 0.0127 and 0.0101); `E4a_L4` geometric between `E4a_L2` and `E4a_L6`: 0.008;
`IND_L4` the rate of `E4a_L4`: 0.008.

## Reading

- The selected peak rates are **lower than the first-ladder cosine rates** (REF1 0.016 against 0.032; REF4 0.008 against 0.016; the constrained arms
  0.008 at `L = 2` and `L = 6` against 0.032 at `L = 1` and 0.008 at `L = 4`) and, under WSD, **the constrained and dense families are not
  separated by their rate beyond a factor of two** at `L = 4` or more (0.008 for both).
- Several selections lie within the seed band of the first ladder (0.006 bpb) of their neighbour, so the rate is not unique: `REF4`
  0.008 against 0.004 (0.005), `E4a_L2` 0.008 against 0.016 (0.003), `E4a_L6` 0.008 against 0.004 (0.006). The curves are flat near the optimum
  (hypothesis for the cause: AdamW with clipping, a decay to zero); a factor of two in the rate of a middle-depth arm (computed, not swept)
  is therefore a small effect, consistent with what E8 needs.
- The constrained arms are all taken at 0.008; the rates of the dense family decrease with depth from 0.016 to 0.008.

## Limits

- One seed, 4,000 steps; the final bits per byte of different arms at seed 100 are not an arm comparison (no pairing is claimed). The middle depths
  use interpolated rates, never run at E7. Rates at `N = 512` are those of `N = 128` (hypothesis, limit of E9).
- The extension rule ended every arm in the interior; the grid steps are a factor of two, so the optimum is known to within that factor only.
