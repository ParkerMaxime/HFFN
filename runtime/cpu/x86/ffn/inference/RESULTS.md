# Results — x86 FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The x86 compact kernel of this leaf (`hffn` loader, recommended routes, 8 threads) against the dense iso-support PyTorch FFN, one isolated FFN per cell, N = 1, 8, 64 and 512 tokens. Machine: an OVH cloud VM, AMD EPYC-Milan, 8 vCPU (shared VM). Run `x86-ffn-inference/publication_20261110T085622Z_76740`; `delta` = 0.068.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `x86-ffn-inference/publication_20261110T085622Z_76740`

`delta` = 0.068, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.071 [0.071–0.071] | 1.425 [1.425–1.425] | **loss** |
| 1024_4096_d010 | N512 | 3/3/3 | 0.414 [0.414–0.414] | 1.307 [1.307–1.307] | **loss** |
| 1024_4096_d010 | N64 | 3/3/3 | 0.294 [0.294–0.294] | 1.169 [1.169–1.169] | **loss** |
| 1024_4096_d010 | N8 | 3/3/3 | 0.227 [0.227–0.227] | 1.137 [1.137–1.137] | **loss** |
| 256_1024_d010 | N1 | 3/3/3 | 0.182 [0.182–0.182] | 1.168 [1.168–1.168] | **loss** |
| 256_1024_d010 | N512 | 3/3/3 | 0.526 [0.526–0.526] | 1.125 [1.125–1.125] | **loss** |
| 256_1024_d010 | N64 | 3/3/3 | 0.356 [0.356–0.356] | 1.257 [1.257–1.257] | **loss** |
| 256_1024_d010 | N8 | 3/3/3 | 0.443 [0.443–0.443] | 1.325 [1.325–1.325] | **loss** |
| 512_2048 | N1 | 3/3/3 | 0.404 [0.404–0.404] | 1.692 [1.692–1.692] | **loss** |
| 512_2048 | N512 | 3/3/3 | 1.037 [1.037–1.037] | 1.381 [1.381–1.381] | **loss** |
| 512_2048 | N64 | 3/3/3 | 0.629 [0.629–0.629] | 1.223 [1.223–1.223] | **loss** |
| 512_2048 | N8 | 3/3/3 | 0.591 [0.591–0.591] | 1.051 [1.051–1.051] | within delta |
| 512_2048_d005 | N1 | 3/3/3 | 0.127 [0.127–0.127] | 1.473 [1.473–1.473] | **loss** |
| 512_2048_d005 | N512 | 3/3/3 | 0.452 [0.452–0.452] | 1.270 [1.270–1.270] | **loss** |
| 512_2048_d005 | N64 | 3/3/3 | 0.317 [0.317–0.317] | 1.267 [1.267–1.267] | **loss** |
| 512_2048_d005 | N8 | 3/3/3 | 0.420 [0.420–0.420] | 1.259 [1.259–1.259] | **loss** |
| 512_2048_d010 | N1 | 3/3/3 | 0.144 [0.144–0.144] | 1.315 [1.315–1.315] | **loss** |
| 512_2048_d010 | N512 | 3/3/3 | 0.581 [0.581–0.581] | 1.311 [1.311–1.311] | **loss** |
| 512_2048_d010 | N64 | 3/3/3 | 0.358 [0.358–0.358] | 1.256 [1.256–1.256] | **loss** |
| 512_2048_d010 | N8 | 3/3/3 | 0.323 [0.323–0.323] | 1.078 [1.078–1.078] | **loss** |
| 512_2048_d020 | N1 | 10/10/10 | 0.263 [0.209–0.387] | 1.638 [1.256–2.432] | **loss** |
| 512_2048_d020 | N512 | 10/10/10 | 0.770 [0.728–0.847] | 1.318 [1.299–1.381] | **loss** |
| 512_2048_d020 | N64 | 10/10/10 | 0.448 [0.427–0.499] | 1.185 [1.179–1.287] | **loss** |
| 512_2048_d020 | N8 | 10/10/10 | 0.470 [0.451–0.494] | 1.125 [1.084–1.147] | **loss** |
| 512_2048_d050 | N1 | 3/3/3 | 0.375 [0.375–0.375] | 1.130 [1.130–1.130] | **loss** |
| 512_2048_d050 | N512 | 3/3/3 | 1.540 [1.540–1.540] | 1.422 [1.422–1.422] | **loss** |
| 512_2048_d050 | N64 | 3/3/3 | 0.775 [0.775–0.775] | 1.117 [1.117–1.117] | **loss** |
| 512_2048_d050 | N8 | 3/3/3 | 0.888 [0.888–0.888] | 1.137 [1.137–1.137] | **loss** |
| 512_2048_d075 | N1 | 3/3/3 | 0.517 [0.517–0.517] | 1.095 [1.095–1.095] | **loss** |
| 512_2048_d075 | N512 | 3/3/3 | 2.143 [2.143–2.143] | 1.434 [1.434–1.434] | **loss** |
| 512_2048_d075 | N64 | 3/3/3 | 1.048 [1.048–1.048] | 1.103 [1.103–1.103] | **loss** |
| 512_2048_d075 | N8 | 3/3/3 | 1.234 [1.234–1.234] | 1.099 [1.099–1.099] | **loss** |
| 512_2048_d090 | N1 | 3/3/3 | 0.632 [0.632–0.632] | 1.104 [1.104–1.104] | **loss** |
| 512_2048_d090 | N512 | 3/3/3 | 2.406 [2.406–2.406] | 1.484 [1.484–1.484] | **loss** |
| 512_2048_d090 | N64 | 3/3/3 | 1.092 [1.092–1.092] | 1.093 [1.093–1.093] | **loss** |
| 512_2048_d090 | N8 | 3/3/3 | 1.402 [1.402–1.402] | 1.029 [1.029–1.029] | within delta |
| 512_2048_d100 | N1 | 3/3/3 | 0.737 [0.737–0.737] | 1.117 [1.117–1.117] | **loss** |
| 512_2048_d100 | N512 | 3/3/3 | 2.666 [2.666–2.666] | 1.562 [1.562–1.562] | **loss** |
| 512_2048_d100 | N64 | 3/3/3 | 1.276 [1.276–1.276] | 1.152 [1.152–1.152] | **loss** |
| 512_2048_d100 | N8 | 3/3/3 | 1.709 [1.709–1.709] | 1.139 [1.139–1.139] | **loss** |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `retained_bytes` | 33,574,912 | 15,561,630 | 6,878,142 | 0.463 | 2.262 | **loss** |
| 256_1024_d010 | `retained_bytes` | 2,102,272 | 2,945,340 | 774,492 | 1.401 | 3.803 | **loss** |
| 512_2048 | `retained_bytes` | 8,398,848 | 9,034,940 | 4,693,212 | 1.076 | 1.925 | **loss** |
| 512_2048_d005 | `retained_bytes` | 8,398,848 | 5,891,894 | 1,550,166 | 0.702 | 3.801 | **loss** |
| 512_2048_d010 | `retained_bytes` | 8,398,848 | 6,519,428 | 2,177,700 | 0.776 | 2.994 | **loss** |
| 512_2048_d020 | `retained_bytes` | 8,398,848 | 7,780,388 | 3,438,660 | 0.926 | 2.263 | **loss** |
| 512_2048_d050 | `retained_bytes` | 8,398,848 | 11,557,058 | 7,215,330 | 1.376 | 1.602 | **loss** |
| 512_2048_d075 | `retained_bytes` | 8,398,848 | 14,701,772 | 10,360,044 | 1.750 | 1.419 | **loss** |
| 512_2048_d090 | `retained_bytes` | 8,398,848 | 16,589,768 | 12,248,040 | 1.975 | 1.354 | **loss** |
| 512_2048_d100 | `retained_bytes` | 8,398,848 | 17,846,480 | 13,504,752 | 2.125 | 1.321 | **loss** |

## Limits

- **B loses time against A beyond `delta` (0.068) on 38 of 40 cells** (B/A median 1.204, range 1.078 to 1.692). Reference cell `512_2048_d020`: N1 1.638 [1.256–2.432], N8 1.125, N64 1.185, N512 1.318. This is the first timed reading of B against A: the leaf's earlier promotions were judged on parity alone. Attribution to the whole-block call (one call for the block of N tokens) or to the AVX2 default build is a **hypothesis**: the two changes reached B together and no control separates them.
- **B loses memory against A beyond 2 % on 10 of 10 cells** (`retained_bytes`, B/A 1.32 to 3.80). B and A are counted by the same adapter function; the difference B − A is 4,341,728 B on every `512_2048` cell, 2,170,848 B at `256_1024_d010` and 8,683,488 B at `1024_4096_d010`: it scales with the hidden width and not with the density, so it is activation-bounded work space (measured differences; the attribution to the work buffers of the whole-block call is a hypothesis).
- B replaces A for a functional reason (the whole-block call of the kernel, which meets the invariant that one call processes the block of N tokens; removal of experimental switches), by the user's verdict of 2026-11-10; the losses are written limits and are not offset.
- The first timed window of the A instance is about ten times slower than the next (reported, absorbed by the per-process median).
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
