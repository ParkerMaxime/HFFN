# Results — publication bench v1

Measured against the frozen pre-publication code of every leaf (see *Code provenance*), 18 qualified runs, 2026-10-21 and
2026-10-22. This file gives the protocol, the runs, the reference cells, the storage figures, the reading and the limits
shared by every leaf. Each leaf's own `RESULTS.md` holds its full tables.

Every figure carries its evidence class: **measured** (read in a run with status `passed`, exit code 0 and full
coverage, named by its run), **estimated** (a calculation or a partial counter) or **hypothesis** (an explanation that no
control isolates). Unless a line says otherwise, a figure here is measured.

## What was measured

For every covered leaf, the HFFN path against a **dense iso-support control** (the same fixed support, executed as the
dense PyTorch model or layer of the leaf), on the same inputs:

- FFN leaves: one isolated FFN, at block sizes `N` of 1, 8, 64 and 512 tokens (inference), or the training step (forward,
  backward, AdamW), on synthetic homogeneous exports of d_model 256, 512 and 1024, at densities from 5 % to 100 %.
- LLM leaves: the whole request of a Pythia model (70M-410M family, the 410M cells here): prefill and decode for
  inference, a training step for training, at 10, 20 and 80 % FFN density. A whole-model ratio is bounded by the dense
  non-FFN parts (Amdahl): an FFN-only figure is never projected onto the model.

Nothing here is a claim of convergence equivalence, of a global speed-up or of generalisation to other hardware.

## Protocol

- One **fresh process per (cell, arm)**, arms in ABBA order, on the same inputs. An A/A (two instances of the same arm,
  between processes) measures the noise of the run.
- Warm-up: 10 calls and 3 s on CPU, 1 s on GPU, before the first timed window. Windows: 100 ms on CPU, 50 ms on GPU.
- Ratio of a cell: the median over the HFFN processes of their process median, over the same for the dense control.
- Reference cell (FFN `512_2048_d020`, Pythia-410M d20): 5 processes per arm and 2 sessions. Every other cell: 3
  processes per arm, 1 session.
- Gates before timing, in every process: copy identity (or process identity for LLM training), then the float64 gate of the
  leaf. A failed gate fails the run.
- `delta` = the largest `|copy / arm - 1|` of the run. On CPU it is **applied**: a gain is stated only when
  `ratio + delta < 1`, a loss only when `ratio - delta > 1`, anything else is *within delta*. On GPU it is **reported**:
  the ratio is read as it is, and the cells with `|ratio - 1| <= delta` are flagged.
- Memory: `retained_bytes` is the adapter's count of the arm's own tensors (it does not see native buffers); RSS (and
  device memory on GPU) is sampled over the timed phases.
- The Mac 410M entries ran in two parts (d20, then d10 and d80), in different machine states.
- Machines: x86 = AMD EPYC 9R14, 8 vCPU (AWS `c7a.2xlarge`); Nvidia = Tesla T4 (AWS `g4dn.xlarge`); M1 = an Apple M1
  Mac with 16 GB, passively cooled.
- The run folders (configuration, provenance, raw observations) and the bench that produced them are **not distributed**:
  the figures can be read here but not regenerated from this tree. The checks of each leaf are in its `README.md`.

## Runs

| Entry | Run | processes | wall (s) | `delta` | `delta` use |
|---|---|---:|---:|---:|---|
| x86 FFN inference | `x86-ffn-inference/publication_20261021T205721Z_36546` | 74 | 2660 | 0.025 | applied |
| x86 FFN training | `x86-ffn-training/publication_20261021T215219Z_43759` | 74 | 2828 | 0.120 | applied |
| x86 LLM inference | `x86-llm-inference/publication_20261021T224108Z_47493` | 32 | 3424 | 0.118 | applied |
| x86 LLM training | `x86-llm-training/publication_20261021T234202Z_54659` | 32 | 3699 | 0.088 | applied |
| Nvidia FFN inference | `nvidia-ffn-inference/publication_20261021T205618Z_36126` | 74 | 2285 | 0.129 | reported |
| Nvidia FFN training | `nvidia-ffn-training/publication_20261021T214713Z_43110` | 74 | 2385 | 0.018 | reported |
| Nvidia LLM inference | `nvidia-llm-inference/publication_20261021T223804Z_47324` | 32 | 2100 | 0.023 | reported |
| M1 CPU FFN inference | `m1-cpu-ffn-inference/publication_20261021T204952Z_35029` | 74 | 2974 | 0.461 | applied |
| M1 CPU FFN training | `m1-cpu-ffn-training/publication_20261021T225431Z_48006` | 74 | 2103 | 0.359 | applied |
| M1 GPU FFN inference | `m1-gpu-ffn-inference/publication_20261022T003013Z_57076` | 74 | 998 | 0.372 | reported |
| M1 GPU LLM inference d20 | `m1-gpu-llm-inference/publication_20261021T222112Z_45240` | 20 | 1597 | 0.066 | reported |
| M1 GPU LLM inference d10+d80 | `m1-gpu-llm-inference/publication_20261022T002142Z_56021` | 12 | 504 | 0.017 | reported |
| M1 GPU LLM training d20 | `m1-gpu-llm-training/publication_20261021T215425Z_43892` | 20 | 715 | 0.166 | reported |
| M1 GPU LLM training d10 | `m1-gpu-llm-training/publication_20261022T123442Z_81919` | 6 | 404 | 0.004 | reported |
| M1 CPU LLM inference d20 | `m1-cpu-llm-inference/publication_20261022T010704Z_62621` | 20 | 2747 | 0.044 | applied |
| M1 CPU LLM inference d10+d80 | `m1-cpu-llm-inference/publication_20261022T104753Z_80203` | 12 | 2152 | 0.083 | applied |
| M1 CPU LLM training d20 | `m1-cpu-llm-training/publication_20261022T021302Z_64319` | 20 | 2187 | 0.064 | applied |
| M1 CPU LLM training d10+d80 | `m1-cpu-llm-training/publication_20261022T114358Z_81377` | 12 | 1832 | 0.044 | applied |

Three runs stopped at their time ceiling (`EXIT` 124) and were replaced by the d10(+d80) entries above. They are not
qualified and give no number: `m1-gpu-llm-training/publication_20261022T000223Z_55072` (7 of 12 units done),
`m1-cpu-llm-inference/publication_20261022T030940Z_64952` (11 of 12) and
`m1-cpu-llm-training/publication_20261022T040352Z_65979` (11 of 12).

The M1 thermal state read before each process was nominal throughout, and no swap-out was counted in any timed phase.

## Not covered

- M1 GPU FFN training: no published leaf.
- Nvidia LLM training: no leaf.
- x86 FFN training has no N1 cell.
- **M1 GPU, Pythia-410M d80, training, HFFN arm: `OOM`.** The cell does not fit the 16 GB Mac (it is swap-bound); no
  figure exists for either arm. A memory fix is deferred to the next optimisation phase.
- One cell is not admissible as stated: M1 CPU FFN inference, `512_2048_d010` N512 (limit 3 below).

## Measured: reference cells (median ratio hffn/dense; the full tables are in each leaf's `RESULTS.md`)

"Loss" means slower than dense. FFN `512_2048_d020`, 5 processes per arm and 2 sessions:

| Entry | N1 | N8 | N64 | N512 | reading |
|---|---:|---:|---:|---:|---|
| x86 inference | 0.148 | 0.405 | 0.337 | 0.470 | gain at every N |
| x86 training | — | 0.229 | 0.680 | 2.136 | gain N8 and N64, loss N512 |
| Nvidia inference | 0.587 | 0.999 | 0.939 | 1.372 | <1 at N1; N8 and N64 inside `delta`; >1 at N512 |
| Nvidia training | 0.630 | 0.715 | 0.757 | 1.297 | <1 at N1 to N64, >1 at N512 |
| M1 CPU inference | 1.826 | 2.385 | 2.879 | 3.461 | loss at every N |
| M1 CPU training | 0.199 | 0.159 | 1.067 | 5.119 | gain N1 and N8, within N64, loss N512 |
| M1 GPU inference | 0.730 | 0.596 | 0.785 | 1.111 | <1 at N1 to N64; every cell inside `delta` except N8 |

Pythia-410M d20, whole model request, 5 processes per arm and 2 sessions:

| Entry | decode_b1 | decode_b4 | prefill_b1 | prefill_b4 | b1_t32 | b1_t128 | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| x86 inference | 0.588 | 0.606 | 0.692 | 0.724 | | | gain on every shape |
| Nvidia inference | 0.937 | 0.916 | 1.302 | 1.366 | | | decode <1, prefill >1 |
| M1 GPU inference | 0.620 | 0.531 | 0.809 | 1.012 | | | <1 except prefill_b4, which is inside `delta` |
| M1 CPU inference | 0.710 | 0.643 | 1.328 | 2.105 | | | decode gain, prefill loss |
| x86 training | | | | | 0.717 | 1.010 | gain t32, within t128 |
| M1 GPU training | | | | | 0.715 | 1.056 | t32 <1, t128 inside `delta` |
| M1 CPU training | | | | | 0.837 | 1.616 | gain t32, loss t128 |

The HFFN arm of the M1 CPU FFN inference row is the compact G4 runtime of `cpu/m1/llm/inference/compact`, not a separate
padded kernel (limit 4).

## Storage per export (bytes)

Synthetic homogeneous exports. The first column is the dense FP32 pair of matrices; the second is the payload of the
export file (int64 indices, as stored); the third is a calculation (FP32 values, int16 indices, int32 offsets).

| Cell | density | dense FP32, two matrices | export payload (int64 indices) | ideal compact (calculation) |
|---|---|---:|---:|---:|
| 512_2048_d005 | 0.05 | 8 388 608 | 1 509 118 | 644 718 |
| 512_2048_d010 | 0.10 | 8 388 608 | 2 973 364 | 1 272 252 |
| 512_2048_d020 | 0.20 | 8 388 608 | 5 915 604 | 2 533 212 |
| 512_2048_d030 | 0.30 | 8 388 608 | 8 842 892 | 3 787 764 |
| 512_2048_d050 | 0.50 | 8 388 608 | 14 727 834 | 6 309 882 |
| 512_2048_d075 | 0.75 | 8 388 608 | 22 065 500 | 9 454 596 |
| 512_2048_d090 | 0.90 | 8 388 608 | 26 470 824 | 11 342 592 |
| 512_2048_d100 | 1.00 | 8 388 608 | 29 403 152 | 12 599 304 |
| 256_1024_d010 | 0.10 | 2 097 152 | 700 666 | 321 652 |
| 1024_4096_d010 | 0.10 | 33 554 432 | 11 833 670 | 5 067 478 |

- The export file stores int64 indices: its payload exceeds the dense FP32 pair from d30.
- The ideal compact layout is a calculation, not a file. It exceeds the dense pair between d50 and d75.
- The runtime's own resident layout is what `retained_bytes` counts in the leaf tables (adapter count).

## Reading

- **Density decides the sign everywhere (measured, within each run).** On every hardware and both modes, the ratio rises
  with density along the 512_2048 slice, and every entry has a density at and above which HFFN is slower than dense:
  - x86 inference: between d50 and d75 for N >= 8; N1 stays a gain up to d100;
  - x86 training: N512 is a gain at d05 (0.796) and a loss from d10 (1.290);
  - Nvidia inference: from d50 for N >= 8 (N1 stays below 1 up to d75, where 0.978 is inside `delta`);
  - Nvidia training: from d50 at every N;
  - M1 CPU training: N512 is within `delta` at d05 (1.122) and a loss from d10 (2.108).
  This is the boundary expected of the format at high density, reported and not hidden.
- **Large blocks favour dense (measured).**
  - On the training entries, N512 has the largest ratio of its cell on every cell: x86, Nvidia and M1 CPU, 10 of 10
    cells each. On x86 inference it is also 10 of 10.
  - The pattern is weaker on the other inference entries: N512 is the largest on 7 of 10 cells on M1 GPU, 6 of 10 on M1 CPU
    and 5 of 10 on Nvidia.
  - At d20 N512 in training, the ratio is 2.136 on x86, 1.297 on Nvidia and 5.119 on M1 CPU.
  - Hypothesis, not isolated: at large N the dense GEMM reaches its throughput regime, while the gather and scatter of the
    compact path do not.
- **On the M1 CPU, FFN inference with this HFFN arm loses at d20 and above (measured).**
  - The reference cell is a loss at every N, and so is every density from d20 up.
  - The only gains beyond `delta` that are not confounded with the episode of limit 3 are at 1024_4096_d010, at N1 (0.380)
    and N8 (0.202).
  - Arithmetic (estimated): at d20 N512, the dense control runs at about 250 GMAC/s (2 x 512 x 2048 x 512 MAC in 4.26 ms)
    and the HFFN arm at about 15 GMAC/s on its active MACs.
  - Hypothesis, not isolated: the dense control uses the M1 matrix unit through Accelerate, which the compact G4 path
    cannot use.
- **Whole-model LLM requests.** Each figure covers the whole request, so the non-FFN dense parts bound every ratio.
  - Decode at d10 and d20 is below 1 on every hardware; on CPU it is a gain beyond `delta`.
  - Prefill at d10 and d20: below 1 on x86 (gain) and on M1 GPU, except prefill_b4 at d20 (1.012, inside `delta`); on M1
    CPU a gain at d10 prefill_b1 (0.864) and a loss on the other three cells; on Nvidia below 1 at d10 (0.950 and 0.956)
    and above 1 at d20 (1.302 and 1.366).
  - At d80, prefill is slower than dense on every hardware (1.128 to 7.547). Decode is mixed: below 1 on M1 GPU (0.893,
    0.860) and on Nvidia decode_b1 (0.921); a gain on x86 decode_b1 (0.806) with decode_b4 (0.922) within `delta`; a loss
    on M1 CPU (1.367, 1.492) and above 1 on Nvidia decode_b4 (1.514).
  - Training: b1_t32 is below 1 at d10 and d20 on every hardware, a gain beyond `delta` on CPU. b1_t128 is a gain only at
    d10 on x86 (0.713) and on M1 CPU (0.944); it is within `delta` at d20 on x86, inside the reported `delta` at d20 on M1
    GPU (1.056), below 1 at d10 on M1 GPU (0.914) and a loss at d20 on M1 CPU (1.616). d80 is a loss on both CPUs (1.569 to
    5.800) and `OOM` on M1 GPU.
- **Memory: retained bytes of the arm's own tensors (measured, adapter count).**
  - On the CPU entries and on M1 GPU, HFFN retains less than dense at d10 and d20: FFN 512_2048_d020, 3 against 8 MB in
    inference and 10 against 42 MB in training; 410M d20 inference, 1058 against 1621 MB; 410M d20 training, 3990 against
    7291 MB on CPU and 4474 against 7291 MB on M1 GPU.
  - On the Nvidia FFN entries it does not: 512_2048_d020 inference retains 9 against 8 MB, training 40 against 32 MB, and
    256_1024_d010 training 11 against 7 MB. Nvidia device memory is nevertheless lower for HFFN at d20: 19 against 26 MB
    in inference, 100 against 117 MB in training.
  - On M1 GPU FFN inference the opposite holds: HFFN retains less but samples more device memory than dense on every cell
    except 1024_4096_d010 (101 against 1083 MB); at d20 it samples 76 against 42 MB.
  - Hypothesis: the gap is held outside the counted tensors, by buffers of the native path. The reverse sign of the two
    counters is reported, not explained.
  - At d80 the 410M inference model retains more than dense on every hardware (1783 to 1795 against 1621 MB).
- **Why the d80 GPU training is `OOM` (hypothesis).** Measured: at d20 on M1 GPU, HFFN samples less device memory than
  dense (8296 against 9479 MB); at d80 on M1 CPU, HFFN retains less than dense (6164 against 7291 MB) but its RSS is
  higher (9081 against 7333 MB). Hypothesis, not isolated: the memory that exceeds the Mac at d80 is held outside the
  counted tensors, in transient buffers of the d80 training path.

## Limits

1. **Not covered**: see *Not covered* above.
2. **Separate parts are not compared.** The Mac 410M entries ran in two parts, d20 and then d10 + d80, in different
   machine states, on different days. A ratio is read inside its part; the d20 and d10/d80 rows of one entry are not
   compared with each other, and no trend across densities is drawn across parts. VM figures are not compared with M1
   figures, nor with each other.
3. **M1 CPU FFN inference: an episode in the run (measured: the dense process medians).** Processes 9 to 15 are
   consecutive in time and cover 512_2048_d010 and the first half of session 0 of 512_2048_d020. In them, the dense medians
   are about 3.8 times those of the neighbouring dense processes (at N512, 14.0 to 14.6 ms against 3.6 to 3.8 ms); the HFFN
   process 11 is slower too (8.1 ms against 5.7 to 5.8 ms). The thermal state stayed nominal and the cause is not known.
   This episode sets the `delta` of the run (0.461); it makes the 512_2048_d010 N512 "gain" (0.414) a product of the slow
   dense processes, so that row is not admissible as a gain; on the reference cell it widens the session range, while
   session 1 alone gives 3.58 at N512. No process was removed.
4. **The HFFN arm of M1 CPU FFN inference is the compact G4 runtime** of `cpu/m1/llm/inference/compact` (exact offsets, no
   padding), not a separate padded FFN kernel. Its figures describe that runtime on the synthetic exports.
5. **Large `delta` on M1.** M1 CPU FFN inference 0.461; M1 CPU FFN training 0.359, set by `512_2048_d050` N1 hffn, where
   the process spread is 65 %; M1 GPU FFN inference 0.372. On CPU it is applied, so many cells read *within delta*. On GPU
   it is reported and not applied: 25 GPU cells lie inside the reported band and are flagged; their sign is not robust.
6. **Pooled medians.** The ratio of a two-session cell is a pooled median. It can lie outside the [min-max] of the
   per-session ratios when the sessions differ in level (M1 GPU LLM inference d20: decode_b1 0.620 against [0.561-0.592]).
   Both are reported.
7. **Memory figures.** RSS and device figures include the harness and, outside LLM training, both instances of the process
   (arm and copy). `retained_bytes` does not see native buffers. On Nvidia FFN and M1 GPU FFN the counted and sampled
   memories disagree in sign; this is not explained. The peak RSS of the processes includes the gates and is not used.
8. **Provenance.** The VM runs record no git commit (the tree was pushed without git; its hashes are recorded). The Nvidia
   GPU model is not in the provenance of the runs: it is the one of the instance type.
9. **Scope of the evidence.** One campaign, one machine state per part. The bench gates check one step or one request per
   process at the leaf's tolerance; the every-export parity of each leaf is its own `check.sh`. No convergence, trajectory
   or hardware-generalisation claim is made. Whole-model ratios include the dense non-FFN parts.
10. **Hypotheses that are not isolated:** the matrix unit for the M1 CPU dense control, the dense GEMM throughput regime at
    large N, and native buffers outside `retained_bytes`. Each would need its own control.
11. **Degree of optimisation.** This is a unified first version of the runtime. The kernels were judged on parity and on no
    loss of time or memory against the earlier per-leaf versions, not tuned for speed. Several optimisation proposals,
    each estimated above 2 % on the quantity its leaf is judged on, are not done; the figures above are the base they
    would be measured against, not a ceiling.

## Code provenance

The figures were measured on the frozen pre-publication tree of each leaf. The tree published here differs from it only
in path resolution (the folder is found from its own location and the assets from `HFFN_ASSETS`), comments, removed
experimental switches and dropped diagnostics, bench adapters and launchers. The native sources are the same, or differ
by comments only, with identical preprocessed output; the Python files are the same operations on the same buffers. This
is a statement of equivalence by construction, not a re-measurement: the checks of each leaf (`README.md`) were re-run
on the published tree, the timings were not.
