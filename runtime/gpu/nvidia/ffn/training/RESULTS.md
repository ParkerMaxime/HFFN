# Results — Nvidia FFN training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step (forward from the inference kernels, backward, AdamW of the compact weights) of an isolated FFN, N = 1, 8, 64 and 512 tokens. Machine: an OVH cloud VM with one NVIDIA Quadro RTX 5000 (16 GB, compute capability 7.5; shared VM, 4 vCPU). Run `nvidia-ffn-training/publication_20261110T023725Z_3664`; `delta` = 0.234.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `nvidia-ffn-training/publication_20261110T023725Z_3664`

`delta` = 0.234, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| 1024_4096_d010 | N1 | 3/3/3 | 0.306 [0.306–0.306] | 0.999 [0.999–0.999] | within delta |
| 1024_4096_d010 | N512 | 3/3/3 | 0.751 [0.751–0.751] | 0.998 [0.998–0.998] | within delta |
| 1024_4096_d010 | N64 | 3/3/3 | 0.342 [0.342–0.342] | 1.002 [1.002–1.002] | within delta |
| 1024_4096_d010 | N8 | 3/3/3 | 0.352 [0.352–0.352] | 1.001 [1.001–1.001] | within delta |
| 256_1024_d010 | N1 | 3/3/3 | 0.094 [0.094–0.094] | 1.019 [1.019–1.019] | within delta |
| 256_1024_d010 | N512 | 3/3/3 | 0.121 [0.121–0.121] | 1.006 [1.006–1.006] | within delta |
| 256_1024_d010 | N64 | 3/3/3 | 0.080 [0.080–0.080] | 0.998 [0.998–0.998] | within delta |
| 256_1024_d010 | N8 | 3/3/3 | 0.080 [0.080–0.080] | 0.959 [0.959–0.959] | within delta |
| 512_2048 | N1 | 3/3/3 | 0.302 [0.302–0.302] | 1.008 [1.008–1.008] | within delta |
| 512_2048 | N512 | 3/3/3 | 0.986 [0.986–0.986] | 0.999 [0.999–0.999] | within delta |
| 512_2048 | N64 | 3/3/3 | 0.329 [0.329–0.329] | 1.001 [1.001–1.001] | within delta |
| 512_2048 | N8 | 3/3/3 | 0.312 [0.312–0.312] | 0.999 [0.999–0.999] | within delta |
| 512_2048_d005 | N1 | 3/3/3 | 0.096 [0.096–0.096] | 1.024 [1.024–1.024] | within delta |
| 512_2048_d005 | N512 | 3/3/3 | 0.359 [0.359–0.359] | 1.008 [1.008–1.008] | within delta |
| 512_2048_d005 | N64 | 3/3/3 | 0.107 [0.107–0.107] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d005 | N8 | 3/3/3 | 0.084 [0.084–0.084] | 1.032 [1.032–1.032] | within delta |
| 512_2048_d010 | N1 | 3/3/3 | 0.142 [0.142–0.142] | 0.971 [0.971–0.971] | within delta |
| 512_2048_d010 | N512 | 3/3/3 | 0.393 [0.393–0.393] | 0.999 [0.999–0.999] | within delta |
| 512_2048_d010 | N64 | 3/3/3 | 0.146 [0.146–0.146] | 1.003 [1.003–1.003] | within delta |
| 512_2048_d010 | N8 | 3/3/3 | 0.139 [0.139–0.139] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d020 | N1 | 10/10/10 | 0.218 [0.218–0.218] | 1.013 [1.005–1.020] | within delta |
| 512_2048_d020 | N512 | 10/10/10 | 0.727 [0.702–0.738] | 1.002 [1.002–1.003] | within delta |
| 512_2048_d020 | N64 | 10/10/10 | 0.268 [0.264–0.273] | 1.004 [0.997–1.004] | within delta |
| 512_2048_d020 | N8 | 10/10/10 | 0.233 [0.215–0.239] | 1.004 [0.995–1.014] | within delta |
| 512_2048_d050 | N1 | 3/3/3 | 0.462 [0.462–0.462] | 1.004 [1.004–1.004] | within delta |
| 512_2048_d050 | N512 | 3/3/3 | 1.419 [1.419–1.419] | 1.008 [1.008–1.008] | within delta |
| 512_2048_d050 | N64 | 3/3/3 | 0.539 [0.539–0.539] | 0.993 [0.993–0.993] | within delta |
| 512_2048_d050 | N8 | 3/3/3 | 0.551 [0.551–0.551] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d075 | N1 | 3/3/3 | 0.618 [0.618–0.618] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d075 | N512 | 3/3/3 | 2.238 [2.238–2.238] | 0.995 [0.995–0.995] | within delta |
| 512_2048_d075 | N64 | 3/3/3 | 0.755 [0.755–0.755] | 0.998 [0.998–0.998] | within delta |
| 512_2048_d075 | N8 | 3/3/3 | 0.695 [0.695–0.695] | 1.000 [1.000–1.000] | within delta |
| 512_2048_d090 | N1 | 3/3/3 | 0.755 [0.755–0.755] | 0.991 [0.991–0.991] | within delta |
| 512_2048_d090 | N512 | 3/3/3 | 2.772 [2.772–2.772] | 1.001 [1.001–1.001] | within delta |
| 512_2048_d090 | N64 | 3/3/3 | 0.857 [0.857–0.857] | 1.004 [1.004–1.004] | within delta |
| 512_2048_d090 | N8 | 3/3/3 | 0.842 [0.842–0.842] | 1.003 [1.003–1.003] | within delta |
| 512_2048_d100 | N1 | 3/3/3 | 0.831 [0.831–0.831] | 0.999 [0.999–0.999] | within delta |
| 512_2048_d100 | N512 | 3/3/3 | 2.791 [2.791–2.791] | 0.999 [0.999–0.999] | within delta |
| 512_2048_d100 | N64 | 3/3/3 | 1.141 [1.141–1.141] | 0.999 [0.999–0.999] | within delta |
| 512_2048_d100 | N8 | 3/3/3 | 1.143 [1.143–1.143] | 0.997 [0.997–0.997] | within delta |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| 1024_4096_d010 | `device_memory_peak` | 428,694,528 | 184,755,712 | 184,755,712 | 0.431 | 1.000 | within 2 % |
| 256_1024_d010 | `device_memory_peak` | 49,690,624 | 41,001,984 | 41,001,984 | 0.825 | 1.000 | within 2 % |
| 512_2048 | `device_memory_peak` | 141,069,824 | 125,220,864 | 125,220,864 | 0.888 | 1.000 | within 2 % |
| 512_2048_d005 | `device_memory_peak` | 124,295,168 | 64,979,968 | 64,979,968 | 0.523 | 1.000 | within 2 % |
| 512_2048_d010 | `device_memory_peak` | 127,641,600 | 76,693,504 | 76,693,504 | 0.601 | 1.000 | within 2 % |
| 512_2048_d020 | `device_memory_peak` | 134,367,232 | 100,230,144 | 100,230,144 | 0.746 | 1.000 | within 2 % |
| 512_2048_d050 | `device_memory_peak` | 154,511,360 | 174,809,600 | 174,809,600 | 1.131 | 1.000 | within 2 % |
| 512_2048_d075 | `device_memory_peak` | 171,285,504 | 231,431,680 | 231,431,680 | 1.351 | 1.000 | within 2 % |
| 512_2048_d090 | `device_memory_peak` | 183,027,200 | 267,284,992 | 267,284,992 | 1.460 | 1.000 | within 2 % |
| 512_2048_d100 | `device_memory_peak` | 188,051,456 | 288,124,928 | 288,124,928 | 1.532 | 1.000 | within 2 % |

## Limits

- `delta` is 0.234, set by one dense-copy pair at `d020` N1 (only 5 of 444 pairs exceed 0.10). B equals A within it on every cell, a weak statement under that band.
- Form pair. The VM is shared. The v1 figures were taken on a Tesla T4 and are not comparable.
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
