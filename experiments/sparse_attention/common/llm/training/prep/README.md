# prep — preparation of the second series (E6-E9)

Written on 2026-10-25, before the smoke of this step and before any run of E6-E9. No figure of quality is produced here.
Plan: `current/sparse_attention/PLAN.md` section 14; brief: `experiments/sparse_attention/CLOUD_E6_E9.md`. The plan on the
branch has its section 14 (checked). Starting commit: `bad5c44` (the plan and brief on top of the first ladder).

## What changes in `lib/` (every new behaviour is an option; the default is the behaviour of E0-E5)

| Option / file | Default (first ladder) | New behaviour |
|---|---|---|
| `--schedule` | `cosine`, warm-up 5 %, floor 10 % | `wsd`: warm-up 200 steps (`--warmup` overrides), constant peak to `0.8 H`, linear decay to zero at `H` |
| `--horizon H` | `--steps` | horizon of the schedule; `--steps` is the last global step of the process |
| `--tokens-per-step` | batch 64 | batch = `tokens_per_step // context` (64 at 128, 16 at 512) |
| `--save-at s1,s2` / `--resume ckpt` | none | full training state saved (model, AdamW, step, position of the batch stream, torch RNG, EMA); a branch restarts from it and runs to `--steps` with horizon `--horizon` |
| `--valid-bytes B` | `--valid-windows 4096` | intermediate evaluations on `B // context` windows (524,288 bytes: 4,096 windows at 128, 1,024 at 512) |
| evaluation at `N > 128` | by class `t mod 4` | also bits per byte by position range 0-127, 128-255, 256-(N−1) (`bpb_by_range`, `range_lengths`) |
| `lib/arms.py` | — | `REF1` (= `E0`), `REF2`, `REF3`, `E4a_L2`, `E4a_L6`, `IND_L4` (`IND`, hash route, 4 layers), and `_N512` (context 512) for the eight arms of the series |
| `lib/count.py` | — | counter per arm and context (below) |
| `lib/campaign.py` | — | stages with branches, series options, several check scripts per step |

The data stream is now a restorable object (`BatchStream`); `train_batches` is unchanged in behaviour. Checkpoints (`ckpt_*.pt`)
stay out of git (`*.pt`).

## Checks (`checks/check_series.py`, 44 checks; `checks/check_model.py` is unchanged and still passes: 179/179, the 121 of the
first ladder included, now over the 21 arms of the configuration table)

- parameter counts outside the embedding equal to PLAN 14.5 (`REF2` 186,240, `REF3` 279,296, `REF4` 372,352, `E4a_L2`
  161,664, `E4a_L4` 323,200, `E4a_L6` 484,736) and independent of the context for the eight arms; `IND_L4` (175,744);
- `IND_L4` under the hash route: degree 26, `up_idx` and `down_idx` inside `S_g`, `W_Q`/`W_K`/`W_V`/`W_O` block-diagonal, `o_t`
  exactly zero outside `S_g` at every layer;
- 8,192 bytes per step at both contexts; at a given seed and context the stream does not depend on the arm; its position is
  restorable; the batches of the first ladder (64 × 128) are unchanged;
- WSD values (step 0, end of warm-up, plateau, half decay, zero at `H`); the checkpoints 1,600 / 3,200 / 6,400 are the starts of the decay of
  the horizons 2,000 / 4,000 / 8,000;
- **branch = uninterrupted run**: on `H = 100`, warm-up 10, checkpoint at 80, a branch gives the same weights and the same final
  bits per byte as one uninterrupted WSD run of horizon 100. Stated tolerance: **1e-6** (float32) on the largest absolute
  weight difference and on the bits per byte; measured 0 (bit for bit) for `E4a` and for `E4b` (learned router);
- position ranges: the mean of the ranges weighted by their lengths equals the whole figure (1e-9); intermediate evaluation on a fixed
  number of bytes (1,024 windows at 512); no ranges at 128;
- the counter equals PLAN 14.5 exactly for the seven models, at both contexts;
- **the first ladder stays reproducible**: a cosine run of E0 (the smoke of the first ladder, 40 steps, seed 0) is bit-identical
  to its record (`e0/runs/smoke_20261024T180423Z`: 4.009533778271087).

## The counter (`lib/count.py`; counted, never timed; estimates)

Inference multiply-adds per token with the convention of PLAN section 6 (`N_k = N`, logits `V M` = 32,768 included, a masked-dense
weight counted by its non-zero entries); training multiply-adds per token = 3 × inference (an estimate that ignores the
optimizer and the softmax); parameters and AdamW states (2 per parameter) outside the embedding; KV cache per token `2 L M`
values; training activations per token and layer (queries, scores, attention outputs, FFN pre-activation and activation of the
active neurons). **`IND_L4` is counted here for the first time**: per layer, keys and values 8,192 (block-diagonal: 2 × 4 × 32²),
queries 1,024, scores `32 N`, weighted values `32 N`, output projection 1,024, FFN 6,656; total **133,120 at `N = 128`** and
**231,424 at `N = 512`**, 175,744 parameters outside the embedding. Under the same convention `E4a_L4` costs 243,712 and 342,016.
(IND is cheaper than the constrained model because its keys and values are block-diagonal, not dense; the counter says so.)

## Smoke and sizing

`./run.sh run` (= smoke here) runs the checks, the 16 arms for 40 steps of WSD (4 jobs in parallel, 1 thread each, seed 0, 8,192 bytes
per step), and one plateau-and-branch pair. It proves nothing about quality. Its wall times size the later steps and feed the
re-estimate of the total of PLAN 14.4 (`RESULTS.md`), which the brief requires before E6: if the estimate exceeds 24 hours the
session stops and reports it. Wall time is never a result.
