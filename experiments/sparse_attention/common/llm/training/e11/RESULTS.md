# E11 — results

README: `README.md` (cells, rule and dense control frozen before the run). Qualified run: `e11/runs/run_20261026T214914Z/` (EXIT 0; 8 cells × 5 fresh processes, 12 windows each, nothing else running on the machine;
the concurrency gate at the start of the run: median ratio 0.324 ≤ 0.4, passed). Analysis (new files): `analysis.json`, `analysis.md`. Classes: **measured** ratios; counted ratios are an estimate.

**What this is:** a ratio of this PyTorch executor (TorchScript fork/join, 4 workers, one intra-op thread each) on this shared 4-vCPU virtual machine, forward only, without gradient, 8,192 tokens per call,
with a masked dense FFN in every arm. No absolute time is reported; no ratio is compared with another machine, campaign or the HFFN runtime.

## Measured: `r` = time(variant) / time(control)

**`delta` of the run = 0.0975** (the largest `|copy / control − 1|` over the 40 cell-processes; it comes from cell a1, whose five copy ratios are 1.063, 0.902, 0.950, 1.070, 0.933).
Faster if `1 − r > delta`, slower if `r − 1 > delta`.

| Cell | Variant / control | `r` median [min–max] over 5 processes | Outcome | Counted ratio of multiply-adds | `r` / counted | Steal fraction |
|---|---|---|---|---|---|---|
| a1 | one layer `E4a` / dense, `M` 128, `N` 128 | 0.601 [0.559–0.669] | **faster** | 0.422 | 1.42 | 1.1 % |
| a2 | one layer, 128, 512 | 0.541 [0.537–0.565] | **faster** | 0.346 | 1.56 | 1.0 % |
| a3 | one layer, 512, 512 | 0.405 [0.394–0.446] | **faster** | 0.422 | 0.96 | 1.1 % |
| a4 | one layer, 512, 2,048 | 0.496 [0.462–0.508] | **faster** | 0.346 | 1.43 | 1.1 % |
| b1 | model `E4a_L4` / `REF2`, 128 | 1.290 [1.239–1.325] | **slower** | 0.862 | 1.50 | 1.0 % |
| b2 | model `E4a_L4_N512` / `REF2_N512` | 1.134 [1.091–1.228] | **slower** | 0.714 | 1.59 | 0.9 % |
| c1 | model `IND_L4` / `E4a_L4`, 128 | 0.852 [0.825–0.895] | **faster** | 0.546 | 1.56 | 0.9 % |
| c2 | model `IND_L4_N512` / `E4a_L4_N512` | 0.916 [0.871–0.954] | not resolved | 0.677 | 1.35 | 0.9 % |

The copy ratio of every cell-process is within 0.098 of 1; the per-process values of `r` and of the copy ratio are in `analysis.json`. Counted ratio of a layer cell: the layer terms of the counter (logits and embedding outside);
of a model cell: the inference count of the two arms (logits included).

## Descriptive: phases of the variant's first layer (untimed pass, single-threaded, first process of each cell; shares, not times)

| Cell | keys & values share of a worker's time | slowest worker / mean worker | barrier + launch share of the fork-join | plan (route, gather) share of the fork-join |
|---|---|---|---|---|
| a1 | 0.27 | 1.11 | 0.30 | 0.048 |
| a2 | 0.22 | 1.06 | 0.06 | 0.044 |
| a3 | 0.35 | 1.03 | 0.10 | 0.010 |
| a4 | 0.23 | 1.08 | 0.14 | 0.007 |
| b1 | 0.27 | 1.06 | 0.30 | 0.084 |
| b2 | 0.21 | 1.06 | 0.14 | 0.051 |
| c1 (`IND`) | 0.24 | 1.18 | 0.18 | 0.086 |
| c2 (`IND`) | 0.16 | 1.15 | 0.23 | 0.039 |

(the "barrier + launch" share is the join time minus the slowest worker's phases run alone, so it also holds load imbalance and memory contention of four workers running together.)

## Reading

- **One layer (a):** under this executor the constrained layer runs in about **0.4 to 0.6 of the time of the dense layer** in all four cells, outside the noise (faster beyond `delta`), i.e. the saving of counted
  multiply-adds is carried into time at the level of one layer. Time follows the count approximately: `r` / counted is 0.96 (a3, the wide layer, where the products are large) to 1.56 (a2): the constrained
  layer is slower than its count in the small cells (1.4-1.6 times), where the plan, the gathers and the barrier weigh (hypothesis; the phase pass shows the plan at 4-5 % and the barrier plus launch at 6-30 % of the fork-join).
- **Model (b): the constrained model is slower than the dense model at equal counted multiply-adds** (`E4a_L4`/`REF2`: 1.29 at `N` 128, 1.13 at 512, both beyond `delta`), although the count says 0.86 and 0.71.
  Reading (hypothesis, not isolated): the two models have the same number of joins (four: one per layer for the 4-layer constrained model, two per layer for the 2-layer dense one), but the constrained model pays the plan of
  the routed tokens, the gathers and the padding at each of its four layers, and its eight-fold smaller per-layer work leaves less to hide them behind; the logits, embedding and route are paid in both. The gap closes
  with the context (1.29 → 1.13), as the count says (0.86 → 0.71). The factor between time and count (1.5-1.6) is what this executor does to a deeper model of routed layers on this machine.
- **`IND_L4` against `E4a_L4` (c):** faster at `N` 128 (0.85, beyond `delta`), not resolved at 512 (0.92, `delta` 0.098), at 0.55 and 0.68 of the counted multiply-adds. The exchange of the residual (the volume read by the
  keys and values, a quarter for `IND`) is part of this difference with the cheaper keys, values and queries of `IND`; the two are not separated (README: `IND` keeps its barrier). So the barrier common to both is not what
  separates them.
- **In terms of the primary criterion of the series (quality at equal counted multiply-adds):** the quality result of E8-E9 (the constrained model dominates the dense one at equal counted multiply-adds) does **not** come with
  a time advantage on this executor at the model level for the middle pair; it comes with a time advantage at the level of one layer. Both statements are specific to this executor, this machine, forward only.

## Limits

- Forward only, no backward, no decoding. A masked dense FFN in every arm. This PyTorch executor; its barrier, padding of the packed form (the largest group count), and TorchScript overhead are part of what is measured. A
  native implementation may move every ratio either way. Shared virtual machine: steal time was about 1 % of the cpu time observed; `delta` of the run (0.0975) comes from the smallest cell and is large against the
  spread of the other cells' copy ratios (≤ 0.05); a smaller `delta` would resolve c2 as faster (`1 − r` = 0.084) — stated, not applied (the rule is fixed).
- A first launch of the run was interrupted by a restart of the container after cells a1-a4 and b1 (`runs/run_20261026T212801Z`, `ABORTED.md`); it gives no figure. This run is the full relaunch under the same README.
- The counted ratios are estimates; the quotient time / count is descriptive. Five processes per cell; the phase pass is single-threaded and untimed (descriptive).
