# E10 analysis

Per token (bytes); counted = the counter's training activations × 4 bytes.

| config | (a) saved, token-side | (a) saved, weight-side (total) | (b) train peak | (c) prefill peak | counted activations |
|---|---|---|---|---|---|
| E4a_L4 | 37125 | 5,648,896 | 39445 | 5289 | 9216 |
| REF2 | 22116 | 2,763,264 | 24276 | 6656 | 18432 |

| reading (candidate / control) | measured (a) token-side | (b) | (c) | counted activations | counted MACs | counted params | counted KV |
|---|---|---|---|---|---|---|---|
| equal counted MACs, 128: E4a_L4 / REF2 | 1.68 | 1.62 | 0.79 | 0.50 | 0.86 | 1.74 | 2.00 |
