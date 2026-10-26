# E10 — memory of the implementation

Written on 2026-10-26 before any run of this step; frozen once it has run. Plan: `current/sparse_attention/PLAN.md` sections 15.3-15.4 (E10);
brief: `experiments/sparse_attention/CLOUD_E10_E13.md`; harness and its checks: `../prep3/README.md`.

## Single question

Is the counted saving of training activations reached by a packed implementation, and how much of the memory of the harness of the first two
series is its `selected` form? Measured per arm, in one training step of 8,192 bytes of the train split (seed 0, the first batch of the stream; the
route depends on the bytes), float32, one intra-op thread:

- (a) bytes of the tensors saved for the backward (distinct storages), **token-side** (activations) and **weight-side** (the masked weights, a cost of the
  control representation, reported apart);
- (b) peak live tensor bytes of the forward and backward; (c) peak live tensor bytes of one forward without gradient (prefill of the same bytes);
  (d) peak resident memory of the process (descriptive only: allocator caching, masked weights).

Weights, AdamW states and the KV cache are **counted** (`lib/count.py`), not measured: the harness stores dense masked weights, so their measured size would be
the dense one. Memory of this PyTorch harness on this machine: it says nothing of the HFFN runtime (the FFN is a masked dense matrix in every arm).

## Arms (24 configurations, each in two fresh processes)

The eight arms of E8 at `N = 128` (`REF1`-`REF4`, `E4a_L2`, `E4a_L4`, `E4a_L6`, `IND_L4`) and the eight of E9 at `N = 512`, the hash-routed arms in the packed form
(`--impl gather`, tokens of each group gathered, `E4a_L4` and `E4a_L4_N512` **also in the `selected` form**, keys `E4a_L4:selected` and `E4a_L4_N512:selected`), and the six arms of E13
(`REF1_W256`, `REF2_W256`, `E4a_L2_W256`, `E4a_L4_W256`, `REF2_W256_h16`, `E4a_L4_W256_h16`; hash arms packed). 8,192 tokens per step: 64 windows of 128, 16 of 512.

## Rule, frozen

The quantities are deterministic: each configuration is measured in two fresh processes; (a) must be identical, (b) and (c) within 1 %; otherwise it is a harness failure
(the cause is fixed and the step relaunched). No threshold on the method: the step measures.

## Readings (measured beside counted)

Per arm: (a) token-side and (b) per token against the training activations of the counter (values × 4 bytes), and (c) per token. The ratios constrained / dense (token-side (a), (b),
(c), and the counted training activations, multiply-adds, parameters and KV values beside them) in the pairs at equal counted multiply-adds (`E4a_L2`/`REF1`, `E4a_L4`/`REF2`,
`E4a_L6`/`REF3` at both contexts; `E4a_L2_W256`/`REF1_W256`, `E4a_L4_W256`/`REF2_W256`, `E4a_L4_W256_h16`/`REF2_W256_h16` at width), at equal depth (`E4a_L2`/`REF2`, `E4a_L4`/`REF4`,
at both contexts), `IND_L4`/`E4a_L4`, and packed against `selected` for `E4a_L4` at both contexts. Classes: measured (bytes), counted (the counter's numbers, an estimate), the ratios of
the two side by side, descriptive.

## Limits stated before the run

One batch, one seed; the counter's "training activations" are four terms (queries, scores, attention outputs, FFN activations) while the saved set also holds keys, values, norms,
residual rows, softmax statistics and the logits: measured and counted are not expected to be equal, and their quotient is descriptive. The packed form pads each group's tokens to the largest
count of the batch. (d) depends on the allocator. Garbage collection and the profiler add no allocation of tensors to (b) and (c).

## Launch

`./run.sh check | smoke | run`: 48 measurement processes, 2 at a time. Estimated 0.6 h.
