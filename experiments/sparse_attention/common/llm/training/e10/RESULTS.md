# E10 — results

README: `README.md` (rule and readings frozen before the run). Qualified run: `e10/runs/run_20261026T211323Z/` (EXIT 0; 24 configurations in 48 fresh processes, all exit 0; **the rule holds for every
configuration**: (a) identical in the two processes, (b) and (c) within 1 %). Analysis (new files): `analysis.json`, `analysis.md`. One training step of 8,192 bytes of the train split (seed 0),
float32, one thread, this PyTorch harness on this machine. Classes: **measured** (bytes) and **counted** (the counter, an estimate) side by side.

## Measured and counted, per token (bytes)

| arm | (a) saved, token-side | (b) peak live, train step | (c) peak live, prefill | counted training activations (values × 4 B) |
|---|---|---|---|---|
| REF1 / REF2 / REF3 / REF4 (128) | 12,348 / 22,116 / 31,884 / 41,652 | 14,444 / 24,276 / 34,108 / 43,940 | 6,656 (each) | 9,216 / 18,432 / 27,648 / 36,864 |
| E4a_L2 / E4a_L4 / E4a_L6 (128, packed) | 19,869 / 37,125 / 54,380 | 22,045 / 39,445 / 56,844 | 5,289 (each) | 4,608 / 9,216 / 13,824 |
| IND_L4 (128, packed) | 34,485 | 36,901 | 5,540 | 9,216 |
| REF1 … REF4 (512) | 12,348 / 22,116 / 31,884 / 41,652 | 14,444 / 24,276 / 34,108 / 43,940 | 6,656 (each) | 21,504 / 43,008 / 64,512 / 86,016 |
| E4a_L2 / L4 / L6 (512, packed) | 22,121 / 41,628 / 61,135 | 24,297 / 43,948 / 63,599 | 5,210 (each) | 10,752 / 21,504 / 32,256 |
| IND_L4 (512, packed) | 39,546 | 41,962 | 5,605 | 21,504 |
| REF1_W256 / REF2_W256 / REF2_W256_h16 | 23,612 / 43,108 / 43,172 | 25,928 / 45,680 / 45,744 | 13,312 (each) | 26,624 / 53,248 / 86,016 |
| E4a_L2_W256 / E4a_L4_W256 / E4a_L4_W256_h16 | 37,574 / 70,999 / 70,695 | 40,210 / 74,211 / 73,907 | 10,332 (each) | 13,312 / 26,624 / 43,008 |
| E4a_L4 in the selected form (128 / 512) | 49,972 / 49,972 | 52,292 / 52,292 | 9,256 | 9,216 / 21,504 |

Weight-side saved bytes (the masked weights `w * mask`, a cost of the control representation, reported apart; total per step, not per token): 1.5 MB (REF1) to 5.4 MB (REF4) at width 128, 2.9 MB (E4a_L2)
to 8.4 MB (E4a_L6); 5.6 MB (REF1_W256) to 22.4 MB (E4a_L4_W256) at width 256.

## Measured against counted: ratios constrained / dense

| Reading | (a) token-side | (b) train peak | (c) prefill peak | counted activations | counted MACs | counted params | counted KV |
|---|---|---|---|---|---|---|---|
| equal counted MACs, 128 (3 pairs) | 1.61, 1.68, 1.71 | 1.53, 1.62, 1.67 | 0.79 | 0.50 | 0.88, 0.86, 0.86 | 1.73-1.74 | 2.00 |
| equal counted MACs, 512 (3 pairs) | 1.79, 1.88, 1.92 | 1.68, 1.81, 1.86 | 0.78 | 0.50 | 0.73, 0.71, 0.71 | 1.73-1.74 | 2.00 |
| equal counted MACs, width 256 (`E4a_L2`/`REF1`, `E4a_L4`/`REF2`, 16 heads) | 1.59, 1.65, 1.64 | 1.55, 1.62, 1.62 | 0.78 | 0.50 | 0.79, 0.78, 0.78 | 1.73 | 2.00 |
| equal depth, 128 (`E4a_L2`/`REF2`, `E4a_L4`/`REF4`) | 0.90, 0.89 | 0.91, 0.90 | 0.79 | 0.25 | 0.49, 0.46 | 0.87 | 1.00 |
| equal depth, 512 | 1.00, 1.00 | 1.00, 1.00 | 0.78 | 0.25 | 0.39, 0.37 | 0.87 | 1.00 |
| `IND_L4`/`E4a_L4`, 128 / 512 | 0.93 / 0.95 | 0.94 / 0.95 | 1.05 / 1.08 | 1.00 | 0.55 / 0.68 | 0.54 | 1.00 |
| packed against selected, `E4a_L4`, 128 / 512 | 0.74 / 0.83 | 0.75 / 0.84 | 0.57 / 0.56 | 1.00 | 1.00 | 1.00 | 1.00 |

## Reading

- **The counted saving of training activations is not reached by this implementation.** The counter says the constrained model keeps half the activations of the dense model at equal counted multiply-adds
  (0.50) and a quarter at equal depth (0.25). Measured, the tensors saved for the backward are **1.6 to 1.9 times** the dense model's at equal counted multiply-adds (the constrained model is twice as deep) and
  **0.89-1.00** at equal depth, and the train-step peak follows (a). The saving in prefill (c) is real but small and independent of depth: **0.78-0.79** (the peak is set by a few layers' transient tensors and the logits).
  (measured; counted beside it.)
- **Why the two differ is a hypothesis read from the numbers:** (i) the saved set holds the keys and values of every head for every token (dense by design, decision 3), the normalised and residual rows, the gathered
  rows and the logits, none of which the counter's four terms (queries, scores, attention outputs, FFN activations) contains; (ii) the counter's scores term (`n_h N` per token) is not stored by the CPU attention kernel at all,
  which saves its inputs and one statistic per row: measured token-side bytes are the same at `N = 128` and `N = 512` for every dense arm (12,348 B/token for `REF1` at both), while the counted term grows by 2.3;
  (iii) the packed form pads each group to the largest count of the batch. The counted `0.25` at equal depth is therefore a bound on the four terms, not a prediction of the memory of this harness.
- **The `selected` form is not what the packed form saves**: packed against `selected` for `E4a_L4` is 0.74 (128) and 0.83 (512) in (a), 0.57 / 0.56 in (c) — a quarter of the memory of the first two series' harness was its
  form, and the rest remains.
- **`IND_L4`** saves 5-7 % of the token-side bytes against `E4a_L4` (block-diagonal keys and values), at 45 % (128) / 32 % (512) fewer counted multiply-adds.
- **Weights, AdamW states and KV cache** stay counted (1.73-1.74 times the weights and states, twice the KV cache, in the pairs at equal counted multiply-adds). The weight-side saved bytes of the harness (the masked weights, rebuilt at
  every forward) are about twice the dense model's in those pairs, a cost of the control representation.

## Limits

- One batch, one seed, float32, one thread, this harness; the FFN is a masked dense matrix in every arm and the masked weights are rebuilt at every forward, so the weight-side figures say nothing of the HFFN runtime. The
  attention kernel is PyTorch's CPU kernel; another kernel stores other tensors. (d), the peak resident memory, is recorded in `analysis.json` (descriptive only).
- Measured and counted are not expected to be equal (the counter's four terms are a subset of what is saved); the quotients above are descriptive.
