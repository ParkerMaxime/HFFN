# Results — M1 GPU FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step (forward from the sibling inference leaf, native backward and AdamW of the compact weights) of an isolated FFN, N = 1, 8, 64 and 512 tokens, B against the dense iso-support control. First publication of this leaf: B and dense only. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-gpu-ffn-training/publication_20261110T041933Z_36349`; `delta` = 0.298.
Arms: B (this version) and the dense control only; the leaf is published for the first time. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-gpu-ffn-training/publication_20261110T041933Z_36349`

`delta` = 0.298, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | 
|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/- | 0.158 [0.158–0.158] |
| 1024_4096_d010 | N512 | 3/3/- | 0.549 [0.549–0.549] |
| 1024_4096_d010 | N64 | 3/3/- | 0.269 [0.269–0.269] |
| 1024_4096_d010 | N8 | 3/3/- | 0.165 [0.165–0.165] |
| 256_1024_d010 | N1 | 3/3/- | 0.407 [0.407–0.407] |
| 256_1024_d010 | N512 | 3/3/- | 0.710 [0.710–0.710] |
| 256_1024_d010 | N64 | 3/3/- | 0.557 [0.557–0.557] |
| 256_1024_d010 | N8 | 3/3/- | 0.424 [0.424–0.424] |
| 512_2048 | N1 | 3/3/- | 0.900 [0.900–0.900] |
| 512_2048 | N512 | 3/3/- | 1.176 [1.176–1.176] |
| 512_2048 | N64 | 3/3/- | 1.118 [1.118–1.118] |
| 512_2048 | N8 | 3/3/- | 1.065 [1.065–1.065] |
| 512_2048_d005 | N1 | 3/3/- | 0.260 [0.260–0.260] |
| 512_2048_d005 | N512 | 3/3/- | 0.370 [0.370–0.370] |
| 512_2048_d005 | N64 | 3/3/- | 0.329 [0.329–0.329] |
| 512_2048_d005 | N8 | 3/3/- | 0.248 [0.248–0.248] |
| 512_2048_d010 | N1 | 3/3/- | 0.392 [0.392–0.392] |
| 512_2048_d010 | N512 | 3/3/- | 0.721 [0.721–0.721] |
| 512_2048_d010 | N64 | 3/3/- | 0.536 [0.536–0.536] |
| 512_2048_d010 | N8 | 3/3/- | 0.405 [0.405–0.405] |
| 512_2048_d020 | N1 | 10/10/- | 0.663 [0.664–0.664] |
| 512_2048_d020 | N512 | 10/10/- | 0.869 [0.789–0.925] |
| 512_2048_d020 | N64 | 10/10/- | 0.980 [0.984–0.998] |
| 512_2048_d020 | N8 | 10/10/- | 0.750 [0.743–0.767] |
| 512_2048_d050 | N1 | 3/3/- | 1.024 [1.024–1.024] |
| 512_2048_d050 | N512 | 3/3/- | 2.178 [2.178–2.178] |
| 512_2048_d050 | N64 | 3/3/- | 1.445 [1.445–1.445] |
| 512_2048_d050 | N8 | 3/3/- | 1.319 [1.319–1.319] |
| 512_2048_d075 | N1 | 3/3/- | 1.257 [1.257–1.257] |
| 512_2048_d075 | N512 | 3/3/- | 3.178 [3.178–3.178] |
| 512_2048_d075 | N64 | 3/3/- | 1.412 [1.412–1.412] |
| 512_2048_d075 | N8 | 3/3/- | 1.434 [1.434–1.434] |
| 512_2048_d090 | N1 | 3/3/- | 1.380 [1.380–1.380] |
| 512_2048_d090 | N512 | 3/3/- | 3.702 [3.702–3.702] |
| 512_2048_d090 | N64 | 3/3/- | 1.667 [1.667–1.667] |
| 512_2048_d090 | N8 | 3/3/- | 1.486 [1.486–1.486] |
| 512_2048_d100 | N1 | 3/3/- | 1.418 [1.418–1.418] |
| 512_2048_d100 | N512 | 3/3/- | 4.881 [4.881–4.881] |
| 512_2048_d100 | N64 | 3/3/- | 1.721 [1.721–1.721] |
| 512_2048_d100 | N8 | 3/3/- | 1.238 [1.238–1.238] |

Memory: no primary per-arm reading (see Limits below).

## Limits

- No primary memory reading per arm: the adapter returns no `retained_bytes` (the context exposes views only) and form C holds every instance in one process, so no process figure is per arm. Only the growth of the process at the first build of each instance remains (estimated; at `512_2048_d020` RSS 2.66, footprint 0.90 and device 2.21 times the dense instance's). No memory claim is made for this leaf.
- `delta` is 0.298: a ratio against dense inside that band is not a gain or a loss.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
