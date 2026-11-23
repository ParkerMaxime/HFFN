# Results — M1 GPU FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The Metal FFN kernel of the M1 GPU shared level, one isolated FFN per cell, N = 1, 8, 64 and 512 tokens. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-gpu-ffn-inference/publication_20261110T035231Z_6603`; `delta` = 0.628.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-gpu-ffn-inference/publication_20261110T035231Z_6603`

`delta` = 0.628, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.760 [0.760–0.760] | 1.015 [1.015–1.015] | within delta |
| 1024_4096_d010 | N512 | 3/3/3 | 0.796 [0.796–0.796] | 0.898 [0.898–0.898] | within delta |
| 1024_4096_d010 | N64 | 3/3/3 | 0.868 [0.868–0.868] | 1.013 [1.013–1.013] | within delta |
| 1024_4096_d010 | N8 | 3/3/3 | 0.890 [0.890–0.890] | 1.019 [1.019–1.019] | within delta |
| 256_1024_d010 | N1 | 3/3/3 | 1.328 [1.328–1.328] | 0.998 [0.998–0.998] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 0.630 [0.630–0.630] | 1.022 [1.022–1.022] | within delta |
| 256_1024_d010 | N64 | 3/3/3 | 1.043 [1.043–1.043] | 0.990 [0.990–0.990] | within delta |
| 256_1024_d010 | N8 | 3/3/3 | 0.973 [0.973–0.973] | 1.032 [1.032–1.032] | within delta |
| 512_2048 | N1 | 3/3/3 | 1.351 [1.351–1.351] | 0.985 [0.985–0.985] | within delta |
| 512_2048 | N512 | 3/3/3 | 1.111 [1.111–1.111] | 0.911 [0.911–0.911] | within delta |
| 512_2048 | N64 | 3/3/3 | 1.632 [1.632–1.632] | 0.997 [0.997–0.997] | within delta |
| 512_2048 | N8 | 3/3/3 | 1.500 [1.500–1.500] | 0.964 [0.964–0.964] | within delta |
| 512_2048_d005 | N1 | 3/3/3 | 1.241 [1.241–1.241] | 1.039 [1.039–1.039] | within delta |
| 512_2048_d005 | N512 | 3/3/3 | 0.455 [0.455–0.455] | 1.005 [1.005–1.005] | within delta |
| 512_2048_d005 | N64 | 3/3/3 | 0.754 [0.754–0.754] | 0.997 [0.997–0.997] | within delta |
| 512_2048_d005 | N8 | 3/3/3 | 0.831 [0.831–0.831] | 1.011 [1.011–1.011] | within delta |
| 512_2048_d010 | N1 | 3/3/3 | 1.265 [1.265–1.265] | 1.030 [1.030–1.030] | within delta |
| 512_2048_d010 | N512 | 3/3/3 | 0.751 [0.751–0.751] | 0.939 [0.939–0.939] | within delta |
| 512_2048_d010 | N64 | 3/3/3 | 0.903 [0.903–0.903] | 1.064 [1.064–1.064] | within delta |
| 512_2048_d010 | N8 | 3/3/3 | 0.970 [0.970–0.970] | 0.994 [0.994–0.994] | within delta |
| 512_2048_d020 | N1 | 10/10/10 | 1.258 [1.253–1.263] | 1.000 [0.995–1.001] | within delta |
| 512_2048_d020 | N512 | 10/10/10 | 1.016 [0.994–1.016] | 0.978 [0.973–1.016] | within delta |
| 512_2048_d020 | N64 | 10/10/10 | 1.296 [1.269–1.324] | 1.000 [0.986–1.000] | within delta |
| 512_2048_d020 | N8 | 10/10/10 | 1.214 [1.221–1.232] | 0.998 [0.993–1.017] | within delta |
| 512_2048_d050 | N1 | 3/3/3 | 1.435 [1.435–1.435] | 1.036 [1.036–1.036] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 1.335 [1.335–1.335] | 1.019 [1.019–1.019] | within delta |
| 512_2048_d050 | N64 | 3/3/3 | 2.389 [2.389–2.389] | 1.045 [1.045–1.045] | within delta |
| 512_2048_d050 | N8 | 3/3/3 | 2.098 [2.098–2.098] | 1.032 [1.032–1.032] | within delta |
| 512_2048_d075 | N1 | 3/3/3 | 1.558 [1.558–1.558] | 1.034 [1.034–1.034] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 1.568 [1.568–1.568] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d075 | N64 | 3/3/3 | 3.256 [3.256–3.256] | 1.098 [1.098–1.098] | within delta |
| 512_2048_d075 | N8 | 3/3/3 | 2.820 [2.820–2.820] | 1.006 [1.006–1.006] | within delta |
| 512_2048_d090 | N1 | 3/3/3 | 1.580 [1.580–1.580] | 1.014 [1.014–1.014] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 1.906 [1.906–1.906] | 1.059 [1.059–1.059] | within delta |
| 512_2048_d090 | N64 | 3/3/3 | 3.499 [3.499–3.499] | 0.943 [0.943–0.943] | within delta |
| 512_2048_d090 | N8 | 3/3/3 | 3.207 [3.207–3.207] | 1.048 [1.048–1.048] | within delta |
| 512_2048_d100 | N1 | 3/3/3 | 1.510 [1.510–1.510] | 1.010 [1.010–1.010] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 2.220 [2.220–2.220] | 0.967 [0.967–0.967] | within delta |
| 512_2048_d100 | N64 | 3/3/3 | 3.859 [3.859–3.859] | 0.962 [0.962–0.962] | within delta |
| 512_2048_d100 | N8 | 3/3/3 | 3.079 [3.079–3.079] | 0.979 [0.979–0.979] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `footprint_timed_max_bytes` | 1,400,391,216 | 416,777,896 | 412,812,896 | 0.298 | 1.010 | within 2 % |
| 256_1024_d010 | `footprint_timed_max_bytes` | 311,756,216 | 336,184,904 | 334,759,544 | 1.078 | 1.004 | within 2 % |
| 512_2048 | `footprint_timed_max_bytes` | 318,588,392 | 352,388,704 | 350,045,720 | 1.106 | 1.007 | within 2 % |
| 512_2048_d005 | `footprint_timed_max_bytes` | 319,292,928 | 350,701,128 | 347,129,344 | 1.098 | 1.010 | within 2 % |
| 512_2048_d010 | `footprint_timed_max_bytes` | 317,883,784 | 349,390,336 | 346,539,568 | 1.099 | 1.008 | within 2 % |
| 512_2048_d020 | `footprint_timed_max_bytes` | 319,948,204 | 350,709,284 | 347,137,584 | 1.096 | 1.010 | within 2 % |
| 512_2048_d050 | `footprint_timed_max_bytes` | 318,686,648 | 355,567,272 | 349,046,320 | 1.116 | 1.019 | within 2 % |
| 512_2048_d075 | `footprint_timed_max_bytes` | 317,834,656 | 362,776,208 | 359,614,024 | 1.141 | 1.009 | within 2 % |
| 512_2048_d090 | `footprint_timed_max_bytes` | 318,211,488 | 366,347,896 | 370,411,104 | 1.151 | 0.989 | within 2 % |
| 512_2048_d100 | `footprint_timed_max_bytes` | 319,538,664 | 403,146,360 | 398,820,936 | 1.262 | 1.011 | within 2 % |

## Limits

- `delta` is 0.628 (set by the A copy at `512_2048_d075` N64): the B-against-A readings are all *within delta* and say little; a B up to 1.63 times slower than A would read the same. A limit of this run, not a finding about B.
- Form pair, without block lifecycle (written limit of the candidate).
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
