# Results — M1 GPU LLM training

Evidence class: **measured** unless a line says otherwise, each figure read in a qualified run (status `passed`, exit code 0,
full coverage) named in the heading of its table. The protocol, the use of `delta`, the machines, the runs that were not
qualified and the limits that apply to every leaf are in [`../../../../RESULTS.md`](../../../../RESULTS.md). The run folders and the
bench that produced them are not distributed: the figures can be read here, not regenerated from this tree.

One training step of Pythia-410M at 10 % FFN density (batch 1, 32 or 128 tokens). B is the leaf moved onto the FFN training component of `gpu/m1/ffn/training` for the backward and the AdamW update. Machine: Apple M1 Mac (16 GB, passively cooled). Run `m1-gpu-llm-training/publication_20261110T054806Z_68384`; `delta` = 0.017.
Arms: A (the version published before, v1), B (this version) and the dense control, in the same processes. "B / dense" below 1 means B is faster than the dense control; "B / A" below 1 means B is faster than A. A reading against A is a gain or a loss only beyond `delta` (a gain: `1 - ratio > delta`; no loss: `ratio <= 1 + delta`); a memory ratio against A is read at 2 % and is not moved by `delta`.

## Measured

### Run `m1-gpu-llm-training/publication_20261110T054806Z_68384`

`delta` = 0.017, applied.


| Cell | Size | Processes (dense/B/A) | B / dense median [min–max] | B / A median [min–max] | B vs A |
|---|---|---|---|---|---|
| pythia410m_sparse10 | b1_t128 | 3/3/3 | 1.350 [1.350–1.350] | 1.475 [1.475–1.475] | **loss** |
| pythia410m_sparse10 | b1_t32 | 3/3/3 | 0.925 [0.925–0.925] | 1.396 [1.396–1.396] | **loss** |

Memory, primary reading per cell (median over processes, bytes):

| Cell | Reading | dense | B | A | B / dense | B / A | B vs A (2 %) |
|---|---|---|---|---|---|---|---|
| pythia410m_sparse10 | `footprint_timed_max_bytes` | 9,926,806,288 | 7,438,438,456 | 7,485,279,376 | 0.749 | 0.994 | within 2 % |

## Limits

- **B loses time against A beyond `delta` (0.017) on both timed cells**: B/A 1.396 (`b1_t32`) and 1.475 (`b1_t128`); memory is within 2 % (B/A 0.994). This is the price of the move onto the owner path, judged on parity alone; the move was kept for that functional reason (one path per mechanism). Against dense, B is 0.925 at `b1_t32` and 1.350 at `b1_t128`.
- The host-side costs of bridging the two frameworks stay in the entry point of the leaf (activation layout conversion, staging copies, output allocation, compact-gradient copy, handle conversion, transient command objects, host synchronisation per call, native state allocated and not used); they are declared in the README of the leaf with their **estimated** counts, none of them measured separately. Whether they account for the loss is a hypothesis, not isolated.
- Only 410M at 10 % is on the timed grid. 410M at 20 % and 80 % are out of it by decision (the HFFN timed process fails the swap guard in the smokes of the 16 GB Mac); the check of the leaf covers every export at parity only.
- Form P (one arm per process), without block lifecycle (written limit).
- One campaign, one machine; no figure here is compared with a figure of another entry or of the previous version. No claim of convergence equivalence, global speed-up or hardware generalisation.
