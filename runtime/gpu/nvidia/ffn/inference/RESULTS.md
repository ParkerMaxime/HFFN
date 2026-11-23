# Results — Nvidia FFN inference

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

The CUDA FFN kernels of this leaf (generated at the shared level), one isolated FFN per cell, N = 1, 8, 64 and 512 tokens. Machine: an OVH cloud VM with one NVIDIA Quadro RTX 5000 (16 GB, compute capability 7.5; shared VM, 4 vCPU). Run `nvidia-ffn-inference/publication_20261110T085635Z_76805`; `delta` = 0.120.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `nvidia-ffn-inference/publication_20261110T085635Z_76805`

`delta` = 0.120, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.352 [0.352–0.352] | 0.987 [0.987–0.987] | within delta |
| 1024_4096_d010 | N512 | 3/3/3 | 0.930 [0.930–0.930] | 1.003 [1.003–1.003] | within delta |
| 1024_4096_d010 | N64 | 3/3/3 | 0.728 [0.728–0.728] | 0.999 [0.999–0.999] | within delta |
| 1024_4096_d010 | N8 | 3/3/3 | 0.653 [0.653–0.653] | 1.006 [1.006–1.006] | within delta |
| 256_1024_d010 | N1 | 3/3/3 | 0.689 [0.689–0.689] | 1.030 [1.030–1.030] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 0.985 [0.985–0.985] | 1.002 [1.002–1.002] | within delta |
| 256_1024_d010 | N64 | 3/3/3 | 1.052 [1.052–1.052] | 1.009 [1.009–1.009] | within delta |
| 256_1024_d010 | N8 | 3/3/3 | 1.093 [1.093–1.093] | 0.990 [0.990–0.990] | within delta |
| 512_2048 | N1 | 3/3/3 | 0.685 [0.685–0.685] | 0.998 [0.998–0.998] | within delta |
| 512_2048 | N512 | 3/3/3 | 1.958 [1.958–1.958] | 0.997 [0.997–0.997] | within delta |
| 512_2048 | N64 | 3/3/3 | 1.422 [1.422–1.422] | 1.015 [1.015–1.015] | within delta |
| 512_2048 | N8 | 3/3/3 | 1.559 [1.559–1.559] | 1.005 [1.005–1.005] | within delta |
| 512_2048_d005 | N1 | 3/3/3 | 0.666 [0.666–0.666] | 0.989 [0.989–0.989] | within delta |
| 512_2048_d005 | N512 | 3/3/3 | 0.567 [0.567–0.567] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d005 | N64 | 3/3/3 | 0.630 [0.630–0.630] | 0.996 [0.996–0.996] | within delta |
| 512_2048_d005 | N8 | 3/3/3 | 0.683 [0.683–0.683] | 0.991 [0.991–0.991] | within delta |
| 512_2048_d010 | N1 | 3/3/3 | 0.691 [0.691–0.691] | 1.052 [1.052–1.052] | within delta |
| 512_2048_d010 | N512 | 3/3/3 | 0.870 [0.870–0.870] | 0.992 [0.992–0.992] | within delta |
| 512_2048_d010 | N64 | 3/3/3 | 0.810 [0.810–0.810] | 1.010 [1.010–1.010] | within delta |
| 512_2048_d010 | N8 | 3/3/3 | 0.919 [0.919–0.919] | 0.896 [0.896–0.896] | within delta |
| 512_2048_d020 | N1 | 10/10/10 | 0.688 [0.686–0.690] | 0.979 [0.973–0.985] | within delta |
| 512_2048_d020 | N512 | 10/10/10 | 1.441 [1.440–1.442] | 1.001 [1.000–1.002] | within delta |
| 512_2048_d020 | N64 | 10/10/10 | 1.034 [1.021–1.038] | 1.000 [0.979–1.022] | within delta |
| 512_2048_d020 | N8 | 10/10/10 | 1.186 [1.172–1.203] | 0.985 [0.972–1.000] | within delta |
| 512_2048_d050 | N1 | 3/3/3 | 0.749 [0.749–0.749] | 0.990 [0.990–0.990] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 3.145 [3.145–3.145] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d050 | N64 | 3/3/3 | 1.950 [1.950–1.950] | 0.969 [0.969–0.969] | within delta |
| 512_2048_d050 | N8 | 3/3/3 | 2.671 [2.671–2.671] | 1.027 [1.027–1.027] | within delta |
| 512_2048_d075 | N1 | 3/3/3 | 0.945 [0.945–0.945] | 1.003 [1.003–1.003] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 4.580 [4.580–4.580] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d075 | N64 | 3/3/3 | 2.732 [2.732–2.732] | 1.010 [1.010–1.010] | within delta |
| 512_2048_d075 | N8 | 3/3/3 | 3.621 [3.621–3.621] | 0.997 [0.997–0.997] | within delta |
| 512_2048_d090 | N1 | 3/3/3 | 1.059 [1.059–1.059] | 1.006 [1.006–1.006] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 5.429 [5.429–5.429] | 0.993 [0.993–0.993] | within delta |
| 512_2048_d090 | N64 | 3/3/3 | 3.212 [3.212–3.212] | 1.015 [1.015–1.015] | within delta |
| 512_2048_d090 | N8 | 3/3/3 | 4.128 [4.128–4.128] | 1.006 [1.006–1.006] | within delta |
| 512_2048_d100 | N1 | 3/3/3 | 1.060 [1.060–1.060] | 1.010 [1.010–1.010] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 6.314 [6.314–6.314] | 0.997 [0.997–0.997] | within delta |
| 512_2048_d100 | N64 | 3/3/3 | 3.832 [3.832–3.832] | 1.005 [1.005–1.005] | within delta |
| 512_2048_d100 | N8 | 3/3/3 | 4.038 [4.038–4.038] | 1.008 [1.008–1.008] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `device_memory_peak` | 94,543,872 | 37,681,152 | 37,681,152 | 0.399 | 1.000 | within 2 % |
| 256_1024_d010 | `device_memory_peak` | 17,442,816 | 7,532,544 | 7,532,544 | 0.432 | 1.000 | within 2 % |
| 512_2048 | `device_memory_peak` | 34,754,560 | 21,351,424 | 21,351,424 | 0.614 | 1.000 | within 2 % |
| 512_2048_d005 | `device_memory_peak` | 34,754,560 | 15,064,064 | 15,064,064 | 0.433 | 1.000 | within 2 % |
| 512_2048_d010 | `device_memory_peak` | 34,754,560 | 16,319,488 | 16,319,488 | 0.470 | 1.000 | within 2 % |
| 512_2048_d020 | `device_memory_peak` | 34,754,560 | 18,840,576 | 18,840,576 | 0.542 | 1.000 | within 2 % |
| 512_2048_d050 | `device_memory_peak` | 34,754,560 | 26,395,648 | 26,395,648 | 0.759 | 1.000 | within 2 % |
| 512_2048_d075 | `device_memory_peak` | 34,754,560 | 32,685,056 | 32,685,056 | 0.940 | 1.000 | within 2 % |
| 512_2048_d090 | `device_memory_peak` | 34,754,560 | 36,460,544 | 36,460,544 | 1.049 | 1.000 | within 2 % |
| 512_2048_d100 | `device_memory_peak` | 34,754,560 | 38,971,392 | 38,971,392 | 1.121 | 1.000 | within 2 % |

## Limits

- B equals A within `delta` (0.120) and within 2 % of `device_memory_peak` on every cell.
- Form pair. The VM is shared; figures are read inside this run only. The v1 figures were taken on a Tesla T4 and are not comparable.
- A first launch of this entry stopped at its ceiling (`EXIT` 124); that run is unqualified and no number is drawn from it. The run here is the rerun.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
