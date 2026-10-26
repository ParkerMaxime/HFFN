# E11 — relative time

Written on 2026-10-26 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 15.2-15.4 (decision 14, E11);
brief: `experiments/sparse_attention/CLOUD_E10_E13.md`; executor, gate and timing harness: `../prep3/README.md` (and its addendum, which fixes the dense control).
**Nothing else runs on the machine during E11**: the session launches no training, check, build or second process while it times (the checks of the launcher run
before the first timing process and end before it).

## Single question

What is the division of the work over four workers worth in time, in one process, forward only: the constrained model (keys and values of `H_g` for every token, the rest for the
routed tokens) against dense attention under the same executor, at the level of one layer and of the model; and what does the exchange of the residual cost, `E4a_L4` against `IND_L4`?
**A ratio of this PyTorch executor on this shared machine**, forward only, without gradient, with a masked dense FFN in every arm. It orders nothing on another machine, does not
speak of training time, and is not compared with other campaigns or with the HFFN runtime. Absolute times are never a result (the raw slots keep their seconds).

## Cells (variant / control), 8,192 tokens per call

| Cell | Level | Variant / control | `M` | `N` | Why |
|---|---|---|---|---|---|
| a1 | one layer | `E4a` / dense | 128 | 128 | the layer of the series |
| a2 | one layer | `E4a` / dense | 128 | 512 | the context where the constraint saves more |
| a3 | one layer | `E4a` / dense | 512 | 512 | width: products large against the cost of a call (hypothesis) |
| a4 | one layer | `E4a` / dense | 512 | 2,048 | the scores dominate the counted work |
| b1 | model | `E4a_L4` / `REF2` | 128 | 128 | the middle pair at equal counted multiply-adds |
| b2 | model | `E4a_L4_N512` / `REF2_N512` | 128 | 512 | the same at 512 |
| c1 | model | `IND_L4` / `E4a_L4` | 128 | 128 | the cost of the exchange of the residual at every layer |
| c2 | model | `IND_L4_N512` / `E4a_L4_N512` | 128 | 512 | the same at 512 |

At `M = 512` a layer has 8 heads of 64 (2 per group), `H = 2,048`, `k = 104`; weights random (time does not depend on their values; the route does depend on the bytes, so the inputs
are bytes of the valid split). Layer cells time the plan of the routed tokens (for the constrained arm) and the layer, from the embedded residual (the embedding and the logits are outside);
model cells time the embedding, the layers and the logits, by the same code in both arms of a cell. **The dense control is the executor** (the fork-join executor with the heads over every token,
a barrier, `W_O` and the FFN on blocks of rows; PyTorch's intra-op execution is not used: prep3 addendum). In the c cells the control is `E4a_L4` and the variant `IND_L4`, both under the executor.

## Rule (timing rule of PLAN 15.3, implemented in `lib/timing.py`)

Five fresh processes per cell, started with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. In each process the arms control, copy (a second instance of the control with the same weights) and variant are
built independently, warmed up (at least 30 calls and 5 s each), then timed in 12 windows of `control, copy, variant, variant, copy, control`, each slot repeated so that it lasts at least 0.2 s.
Per window `variant / control` and `copy / control` on the per-call times of the sums of the two slots of each arm; per process the median over windows; per cell the median of the five
processes and [min–max]. **`delta` = the largest `|copy / control − 1|` over the cell-processes of the run** (all cells and processes). Outcome per cell, with `r` the cell's median:
**faster if `1 − r > delta`, slower if `r − 1 > delta`, otherwise not resolved.** Steal and user jiffies per window, the load average at the start of each process, `lscpu` and the torch
configuration are recorded; nothing is excluded after the fact. The concurrency gate is run again at the start of the run; if it fails, the timing is not run and the cause is reported.

## Readings

Per cell: `r` median [min–max] over five processes, `delta` of the run, the outcome; beside it the **counted ratio of multiply-adds of the cell** (estimated: layer terms of `lib/count.py` for the
layer cells, the inference count of the two arms for the model cells) and the quotient `r` / counted ratio, descriptive: how far time follows the count on this executor. The steal fraction observed.
A separate pass, not timed, splits the variant's first layer, worker by worker and single-threaded, into the phase of keys and values, the routed phase and the wait at the barrier (the join time minus
the slowest worker; it includes load imbalance and the launch) — descriptive, in `phases.json` of the first process of each cell.

## Limits stated before the run

- Forward only, without gradient (the backward is not divided by group here); 8,192 tokens per call, not a decoding step with a KV cache (one new token leaves three workers without queries).
- The FFN is a masked dense matrix in every arm; a native HFFN path runs at another rate per operation (PLAN 14.8): counted multiply-adds and this time are two proxies of different things.
- `IND` keeps one fork-join per layer (prep3 README: its function needs the RMS of the whole row at every layer), so the c cells compare the volume read and the counted work, not the absence of a barrier.
- Shared virtual machine (4 vCPU, KVM): steal time and other tenants can move a ratio; `delta` and the five processes are the answer to that. The process-level ordering is fixed (cells a, b, c in this order).
- PyTorch's CPU kernels, TorchScript fork/join (4 inter-op threads, one intra-op thread each), and the padding of the packed form (up to the largest group count of the batch) are part of what is measured.

## Launch

`./run.sh check | smoke | run`: 8 cells × 5 processes, one at a time. Estimated ≈ 1 h.
