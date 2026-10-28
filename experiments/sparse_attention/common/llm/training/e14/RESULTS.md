# E14 — results

README: `README.md` (cells, readings, timing rule and limits frozen before the first run). Qualified runs (all `EXIT 0`, `check_e14.py` 82/82 passed at the start of each, nothing else running on the machine):
`runs/prefill_20261027T211841Z/` (25 cells × 5 processes, gate 0.325, **`delta` = 0.0623**), `runs/decode_20261027T224704Z/` (25 cells × 5 processes, gate 0.337, **`delta` = 0.0707**),
`runs/memory_20261028T001128Z/` (9 arms × 2 fresh processes; the rule of E10 holds for every arm: (a) identical, (b), (c), (e) within 1 %). Analysis (new files, `lib/analyze_e14.py`):
`runs/analysis_20261028T001745Z/analysis.json`, `analysis.md` (every table in full; the quality axis is read from `e8/runs/run_20261025T163405Z`, `e9/runs/run_20261025T220720Z`,
`e13/runs/run_20261027T040908Z`, unedited). Smoke: `runs/smoke_20261027T203653Z/`. Classes: **measured** (times, memory, bits per byte); every **gap** and every **product** is **estimated**
(an interpolation of two measured axes); expected effects are **hypotheses**. Times are ratios of this PyTorch executor on this shared 4-vCPU machine (steal ≈ 1.0-1.5 %), forward only, FFN masked dense in every arm.

**Harness event.** A first prefill run (`runs/prefill_20261027T205123Z/`, `EXIT 137`, `ABORTED.md`) was stopped by a restart of the container at 21:18 UTC, in cell `n512-REF1` (7 cells complete, one partial).
It gives no figure; the full run above is the relaunch under the same README, `step.json` and rules. No other run was repeated or excluded; no cell, rule or reading was changed after a run.

**`delta`.** Each run's `delta` is the largest `|copy / anchor − 1|` over its 125 cell-processes: one outlier in each (prefill: cell p3, process 4, 1.0623; decoding: cell `n512-E4a_L4`, process 1, 1.0707).
The median `|copy − 1|` is 0.020 (prefill) and 0.017 (decoding), the 95th percentile 0.046 and 0.052. The rule is fixed; a smaller `delta` would resolve more cells (stated, not applied).

## Measured: time of every frontier cell, `r = t(variant) / t(anchor)` (median [min–max] over 5 processes)

| cell | variant / anchor | prefill r [min–max] | prefill | decoding r [min–max] | decoding | counted ratio (prefill / decode) |
|---|---|---|---|---|---|---|
| n128-REF1 | REF1 / REF2 | 0.577 [0.557–0.588] | faster | 0.573 [0.558–0.597] | faster | 0.558 / 0.566 |
| n128-REF3 | REF3 / REF2 | 1.463 [1.435–1.477] | slower | 1.440 [1.430–1.483] | slower | 1.442 / 1.434 |
| n128-REF4 | REF4 / REF2 | 1.855 [1.827–1.876] | slower | 1.934 [1.875–1.977] | slower | 1.884 / 1.869 |
| n128-E4a_L2 | E4a_L2 / REF2 | 0.729 [0.690–0.744] | faster | 0.829 [0.816–0.837] | faster | 0.489 / 0.520 |
| n128-E4a_L4 | E4a_L4 / REF2 | 1.240 [1.192–1.296] | slower | 1.431 [1.408–1.464] | slower | 0.862 / 0.910 |
| n128-E4a_L6 | E4a_L6 / REF2 | 1.788 [1.738–1.860] | slower | 2.036 [1.948–2.114] | slower | 1.236 / 1.299 |
| n128-IND_L4 | IND_L4 / REF2 | 1.096 [1.078–1.147] | slower | 1.410 [1.390–1.459] | slower | 0.471 / 0.467 |
| n512-REF1 | REF1_N512 / REF2_N512 | 0.565 [0.562–0.580] | faster | 0.495 [0.492–0.507] | faster | 0.534 / 0.547 |
| n512-REF3 | REF3_N512 / REF2_N512 | 1.470 [1.347–1.510] | slower | 1.544 [1.521–1.639] | slower | 1.466 / 1.453 |
| n512-REF4 | REF4_N512 / REF2_N512 | 1.827 [1.712–1.900] | slower | 2.078 [1.968–2.151] | slower | 1.932 / 1.906 |
| n512-E4a_L2 | E4a_L2_N512 / REF2_N512 | 0.655 [0.642–0.683] | faster | 0.774 [0.772–0.809] | faster | 0.391 / 0.444 |
| n512-E4a_L4 | E4a_L4_N512 / REF2_N512 | 1.161 [1.146–1.227] | slower | 1.379 [1.357–1.479] | slower | 0.714 / 0.794 |
| n512-E4a_L6 | E4a_L6_N512 / REF2_N512 | 1.721 [1.643–1.752] | slower | 2.189 [2.076–2.235] | slower | 1.036 / 1.144 |
| n512-IND_L4 | IND_L4_N512 / REF2_N512 | 1.065 [1.051–1.090] | slower | 1.443 [1.385–1.508] | slower | 0.483 / 0.476 |
| w256-REF1_W256 | REF1_W256 / REF2_W256 | 0.562 [0.537–0.572] | faster | 0.464 [0.448–0.475] | faster | 0.525 / 0.531 |
| w256-REF2_W256_h16 | REF2_W256_h16 / REF2_W256 | 1.104 [1.077–1.154] | slower | 1.066 [1.054–1.092] | no difference | 1.000 / 1.000 |
| w256-E4a_L2_W256 | E4a_L2_W256 / REF2_W256 | 0.559 [0.543–0.567] | faster | 0.641 [0.631–0.689] | faster | 0.417 / 0.458 |
| w256-E4a_L4_W256 | E4a_L4_W256 / REF2_W256 | 0.987 [0.974–1.025] | no difference | 1.238 [1.187–1.333] | slower | 0.784 / 0.854 |
| w256-E4a_L4_W256_h16 | E4a_L4_W256_h16 / REF2_W256 | 1.032 [1.002–1.063] | no difference | 1.292 [1.229–1.317] | slower | 0.784 / 0.854 |

(faster / slower: beyond `delta` of the run; "no difference": within `delta`.) Counted multiply-adds of the dense arms follow time closely (`r` / counted 0.87-1.10); those of the constrained arms do
not: `r` / counted is 1.26-1.67 in prefill and 1.40-1.91 in decoding (`E4a`), 2.2-3.0 for `IND_L4`. At equal depth the constrained model is slower than the dense one in this executor wherever the count
promises less than about 0.8 of the dense time (`E4a_L4` / `REF2`: counted 0.86, measured 1.24 prefill, 1.43 decoding).

## Estimated: the frontier and the gaps; answer to decision 16 (per mode and campaign, apart)

Gap `g` = bpb of the constrained arm minus the dense frontier at its time, per seed (positive = worse); `g⁺` / `g⁻` the unfavourable / favourable readings with the time moved by `delta` (README). Full per-seed tables in `analysis.md`.

| mode | campaign | `E4a_L2` | `E4a_L4` | `E4a_L6` | answer |
|---|---|---|---|---|---|
| prefill | `N` 128 | on the frontier, `g` +0.022 [+0.008, +0.028] | on the frontier, +0.015 [+0.004, +0.023] | outside the range; vs `REF4`: d +0.017 within, time 0.964 of `REF4` | **not resolved** |
| prefill | `N` 512 | on the frontier, −0.008 [−0.012, +0.007] | on the frontier, +0.019 [−0.005, +0.032] | outside the range; vs `REF4`: d +0.073 inconclusive, time 0.942 | **not resolved** |
| prefill | `M` 256 | outside the range; vs `REF1_W256`: d +0.009 within, time 0.995 | outside the range; vs `REF2_W256`: d +0.001 within, time 0.987 | (`E4a_L4_W256_h16`) outside the range; vs `REF2_W256`: d −0.046 within, time 1.032 | **not resolved** |
| decoding | `N` 128 | on the frontier, +0.073 [+0.060, +0.081] | on the frontier, +0.046 [+0.037, +0.054] | outside the range; vs `REF4`: d +0.017 within, time 1.053 | **not resolved** |
| decoding | `N` 512 | on the frontier, +0.070 [+0.066, +0.089] | on the frontier, +0.062 [+0.036, +0.078] | outside the range; vs `REF4`: d +0.073 inconclusive, time 1.053 | **not resolved** |
| decoding | `M` 256 | on the frontier, +0.092 [+0.068, +0.103] | outside the range; vs `REF2_W256_h16`: d +0.005 within, time 1.161 | (`E4a_L4_W256_h16`) outside the range; vs `REF2_W256_h16`: d −0.042 within, time 1.211 | **not resolved** |

`IND_L4`, the control: on the frontier behind it in prefill (+0.113, +0.116, `g⁻` > 0 in every seed but not beyond τ), *below the frontier beyond τ* in decoding at `N` 128 and 512 (+0.166, +0.205).

- **No `E4a` arm is above the frontier of measured time in either mode**: the largest favourable value is the median gap of `E4a_L2` at `N` 512 in prefill (−0.008), whose unfavourable reading is positive in every seed (+0.033 … +0.055). In
  prefill at `M = 128` the constrained arms lie on the frontier within τ of it (gaps 0.015-0.022 at `N` 128, −0.008 and +0.019 at `N` 512); at equal counted multiply-adds E8-E9 read them as dominating: the change
  of axis from counted multiply-adds to measured time removes the advantage and leaves them close to the frontier, as the first frontier of PLAN 17.1 estimated (+0.021 and +0.008 for `E4a_L4`; here +0.015 and +0.019).
- **No `E4a` arm is below it beyond τ either**, so the answer is *not resolved* in all six cases (and not *unfavourable*: an `E4a_L6` arm or the `M = 256` arms, outside the range, count as neither).
- **Decoding is less favourable than prefill, not more** (hypothesis of PLAN 17.4 contradicted): the gaps of the same arms are larger in decoding (`E4a_L4` +0.046 and +0.062 against +0.015 and +0.019; `E4a_L2` +0.073 and +0.070
  against +0.022 and −0.008), the time ratios are higher (`E4a_L4` / `REF2`: 1.43 and 1.38 against 1.24 and 1.16), and at `M = 256` the constrained arms are 16-21 % slower than the dense arm at the same quality in decoding but within `delta`
  of it in prefill. The two modes are two runs with their own anchor and `delta`; the comparison is of gaps on the bits-per-byte axis, descriptive.
- **At `M = 256` in prefill** `E4a_L4_W256_h16` lies at 1.032 [1.002–1.063] of `REF2_W256` (no difference within `delta`) with a bits per byte 0.046 lower (d −0.0458, all seeds negative: E13). The other dense arm, `REF2_W256_h16`, is at
  1.104 and 0.004 lower than `REF2_W256`. By the medians the 16-head constrained model is 6.5 % faster than `REF2_W256_h16` and 0.042 lower in bits per byte (the pair of E13), but a ratio of two ratios built through the anchor is a
  margin of the size of `delta` (0.062): descriptive, **not a frozen reading**, not resolved. The frozen rule reads the arm as outside the range and gives no frontier verdict.

## Measured: the pairs p1-p6 (the attention alone, the routing of the FFN alone, at equal depth 4)

| cell | variant / anchor | what it isolates | prefill r [min–max] | prefill | decoding r [min–max] | decoding | counted (prefill / decode) | r / counted (prefill / decode) |
|---|---|---|---|---|---|---|---|---|
| p1 | `E4a_L4` / `E2h_L4` | attention alone, 128 | 0.715 [0.678–0.758] | **gain** | 0.719 [0.703–0.729] | **gain** | 0.538 / 0.587 | 1.33 / 1.22 |
| p2 | `E2h_L4` / `REF4` | routing of the FFN alone, 128 | 0.902 [0.866–0.947] | **gain** | 1.007 [0.966–1.022] | no difference within `delta` | 0.850 / 0.829 | 1.06 / 1.22 |
| p3 | `E4a_L4_N512` / `E2h_L4_N512` | attention alone, 512 | 0.685 [0.670–0.716] | **gain** | 0.648 [0.642–0.697] | **gain** | 0.404 / 0.474 | 1.70 / 1.37 |
| p4 | `E2h_L4_N512` / `REF4_N512` | routing of the FFN alone, 512 | 0.932 [0.877–0.971] | **gain** | 1.025 [0.997–1.032] | no difference within `delta` | 0.914 / 0.880 | 1.02 / 1.17 |
| p5 | `E4a_L4_W256` / `E2h_L4_W256` | attention alone, 256 | 0.699 [0.670–0.721] | **gain** | 0.652 [0.629–0.699] | **gain** | 0.458 / 0.521 | 1.52 / 1.25 |
| p6 | `E2h_L4_W256` / `REF4_W256` | routing of the FFN alone, 256 | 0.764 [0.690–0.804] | **gain** | 0.934 [0.930–0.947] | no difference within `delta` | 0.877 / 0.845 | 0.87 / 1.10 |

Products (estimated, two cells combined), measured against the counted `E4a_L4` / `REF4` (0.458, 0.369, 0.402): prefill p1 × p2 = 0.645, p3 × p4 = 0.639, p5 × p6 = 0.533; decoding 0.724, 0.664, 0.609.

- **The sparse attention alone is faster at equal depth in both modes and at the three sizes**: 0.72, 0.69, 0.70 of `E2h` in prefill and 0.72, 0.65, 0.65 in decoding (gains beyond `delta`), against counted 0.54, 0.40, 0.46 and 0.59, 0.47, 0.52:
  the measured gain is about half of the counted one. It includes the scheduling it allows (one barrier per layer instead of two, README of `prep4`).
- **The routing of the FFN alone** (against a masked dense FFN, which executes the whole matrices: an overstatement of its gain for the HFFN runtime, hypothesis, README) is faster in prefill (0.90, 0.93, 0.76) and **not
  distinguishable from the unrouted model in decoding** (1.007, 1.025, 0.934 with `delta` 0.0707; p6 has `1 − r` = 0.066).
- Time by share, prefill at `N` 128: the attention alone carries the larger share of the time gap of `E4a_L4` against `REF4` (0.715 against 0.902), at `N` 512 likewise (0.685 against 0.932), at `M = 256` the two are closer (0.699, 0.764).

## Measured: memory of the nine arms (bytes per token, fp32, one thread; ratios; counted beside)

| reading (candidate / control) | (a) saved for the backward, token side | (b) training-step peak | (c) prefill peak | (e) decoded-step peak per decoded token | counted MACs | counted KV read per decoded token |
|---|---|---|---|---|---|---|
| attention alone, 128 (`E4a_L4` / `E2h_L4`) | 1.12 | 1.11 | 1.23 | **1.64** | 0.54 | 0.25 |
| attention alone, 512 | 1.32 | 1.30 | 1.24 | **4.68** | 0.40 | 0.25 |
| attention alone, 256 | 1.15 | 1.14 | 1.22 | **6.02** | 0.46 | 0.25 |
| routing of the FFN alone, 128 (`E2h_L4` / `REF4`) | 0.80 | 0.81 | 0.65 | 1.01 | 0.85 | 1.00 |
| routing of the FFN alone, 512 | 0.76 | 0.77 | 0.63 | 1.01 | 0.91 | 1.00 |
| routing of the FFN alone, 256 | 0.75 | 0.76 | 0.64 | 0.97 | 0.88 | 1.00 |

Counted activations (the counter): `E4a` / `E2h` 0.375, 0.292, 0.325; `E2h` / `REF4` 0.667, 0.857, 0.769. Parameters and AdamW states counted: `E4a` / `E2h` 0.868 (0.867 at `M = 256`), `E2h` / `REF4` 1; KV cache stored 1 in both pairs.
Per decoded token (e), bytes: `E4a_L4` 7,633 (128), 21,709 (512), 43,366 (256); `E2h_L4` 4,641, 4,641, 7,201; `REF4` 4,608, 4,608, 7,424. Per token (a)/(b)/(c) of the nine arms: `analysis.json`.

- **The sparse attention alone does not save activation memory in this harness**: the constrained model keeps 11-32 % more saved tensors than `E2h` at equal depth (counted: 0.29-0.38), and its prefill peak is 22-24 % higher; the
  routing of the FFN alone saves memory (0.75-0.81 of the saved tensors, 0.63-0.65 of the prefill peak, against counted 0.67-0.86). The first is the result of E10 seen with the FFN held equal: the dense keys and values, the
  gathered rows and the padding of the packed form are stored (hypothesis, as E10).
- **In decoding the packed constrained step has a peak 1.6 to 6.0 times that of `E2h`**, growing with the cache (`N` 128 → 512: 1.6 → 4.7) although it reads a quarter of the cache (counted): the step copies, for the sequences routed to a
  group, the cache slice of the heads of that group (README of `prep4`: *gather of the cache*, padded to the largest count of the step). It is a cost of this executor and the most likely cause (hypothesis, not isolated) of the
  time ratios `r` / counted above 1.4 in decoding for `E4a`; a native step reading the cache in place would not pay it.

## What E14 shows, and does not

- **Measured:** on the frontier of quality against measured time, the constrained family is close to the dense frontier but never above it, in prefill (gaps ≤ 0.02 at `M = 128`) and further behind it in decoding (gaps 0.04-0.09), at every context and
  width tried; the sparse attention alone is 28-35 % faster than the dense attention at equal depth in both modes while keeping 11-32 % more activation memory in this harness (up to 6 times the memory per decoded token); the routing of the
  FFN alone gains 7-24 % of time in prefill and none distinguishable from noise in decoding.
- **Estimated:** every gap, every product, every answer to decision 16 (*not resolved* in the six cases).
- **Hypotheses, not tested:** the cause of the larger decoding gap and of the memory of the decoded step (the gather of the cache slices and the padding); that a native implementation moves the points; that the masked dense FFN overstates the gain of the routing.
- **Not shown** (PLAN 17.6): any absolute time, another machine, the training time, a native implementation, the HFFN runtime; no new quality figure; `E2h` has no bits per byte (not on the frontier); one batch (64), one cache length per context (`N/2`).

## Limits

- This executor, this machine, forward only; masked dense FFN; the decoded step gathers the cache slices of the routed sequences and pads the four groups to the largest count (counts of process 0: `[14, 21, 13, 16]` at `N` 128, `[19, 17, 17, 11]` at 512).
- Each run's `delta` is set by one cell-process; cells with ratios within a few hundredths of 1 are unresolved by the rule (`w256-E4a_L4_W256`, p2, p4, p6 in decoding, p6 `1 − r` = 0.066 against 0.0707). The concurrency gate passed in the three timing contexts with margins of 0.06-0.11 under 0.4.
- Bits per byte are those of E8, E9, E13 (three seeds each, rates of the middle depths and of the `_h16` arms computed, `H*` not recalibrated at width); at `M = 256` the dense frontier has three points, so all three constrained arms of the campaign
  fall outside its range under the shifts of `delta`; the frontier verdict is not defined there.
