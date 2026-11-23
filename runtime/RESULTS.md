# Results — publication bench v2

Measured on the frozen candidate of every leaf (see *Code provenance*), 16 qualified runs, 2026-11-09 and 2026-11-10. This
file gives the protocol, the runs, the reading and the limits shared by every leaf. Each leaf's own `RESULTS.md` holds its
full tables. v2 replaces v1 as the published version of the runtime; the decision to publish it with the losses written
below as limits is the user's (verdict of 2026-11-10: "Oui, les 16 feuilles").

Every figure carries its evidence class: **measured** (read in a run with status `passed`, exit code 0 and full
coverage, named by its run), **estimated** (a calculation or a partial counter) or **hypothesis** (an explanation that no
control isolates). Unless a line says otherwise, a figure here is measured.

## What was measured

For each of the 16 leaves, three arms in the same processes: **A** (the version published before, v1), **B** (the version
published here) and the **dense control** (the same fixed support executed as the dense PyTorch FFN or model of the leaf).
Two leaves are published for the first time and run B and dense only: `gpu/m1/ffn/training` and `gpu/nvidia/llm/training`.

- FFN leaves: one isolated FFN, at block sizes `N` of 1, 8, 64 and 512 tokens (inference), or the training step
  (forward, backward, AdamW; x86 FFN training has no N = 1 cell), on ten synthetic cells: `512_2048` at 30 %, `512_2048` at
  5, 10, 20, 50, 75, 90 and 100 % (`d005` to `d100`), `256_1024_d010` and `1024_4096_d010`. The reference cell is
  `512_2048_d020`.
- LLM leaves: the whole request of a Pythia model (410M at 10, 20 and 80 % FFN density; the Nvidia training leaf adds
  Pythia-70M at 10, 50 and 80 %): prefill and decode (batch 1 and 4) for inference, a training step (batch 1, 32 or 128
  tokens) for training. The reference cell is `pythia410m_sparse20`. A whole-model ratio is bounded by the dense non-FFN
  parts: an FFN-only figure is never projected onto the model.

Nothing here is a claim of convergence equivalence, of a global speed-up or of generalisation to other hardware.

## Protocol

- Rule: *Timing and noise* of the repository (schedule, noise band, per-arm memory), with the full protocol of the branch
  for a publication. The bench is prepared and frozen before the run and is not distributed.
- Forms (how the arms share processes): **C**, the arms of a cell co-resident in one process and timed in ABBA order over
  windows on the same inputs, an in-process dense copy as the A/A; **P**, one arm per process, A/A between processes of one
  session; **pair**, the GPU form of the Nvidia and M1 GPU inference entries, the arm and its copy alternating call by call.
  Each entry's form per cell is in the *Runs* table; the M1 GPU forms P and pair have no block lifecycle (written limit).
- Before any timed window, in every process, gates: copy identity (form C: process identity of every instance), then the
  float64 or token gate of the leaf. A failed gate fails the run.
- Each arm is warmed up before its first timed call. On the M1 (passive cooling): a long warm-up, every arm rebuilt at each
  block where the entry says so, the thermal state recorded before each process. The ratio of a cell is the median over
  processes of the per-process median of the timed windows; [min–max] is over sessions.
- Reference cell: 5 processes per arm and 2 sessions; every other cell: 3 processes and 1 session. A pooled two-session
  median can lie outside the [min–max] of the per-session ratios when the sessions differ in level.
- `delta` = the largest `|copy / arm - 1|` over the A/A pairs of the run, in-process copy for forms C and pair, between
  processes of one session for form P. It is **applied**: a gain over A is stated only if `1 - ratio > delta`, no loss only
  if `ratio <= 1 + delta`. A memory ratio is read at 2 % and is not moved by `delta`.
- Criterion of replacement of A by B: B loses no time beyond `delta` and no memory beyond 2 % on the primary reading; a
  gain is stated only beyond `delta`; a B that does not meet it needs a functional reason and the user's verdict.
- Memory primary reading per cell: `retained_bytes` (form C), `device_memory_peak` (CUDA, forms P and pair),
  `footprint_timed_max_bytes` (M1, P and pair), `rss_timed_max_bytes` (x86 P and pair). The peak of the process lifetime
  includes the gates and is never read per arm.
- Machines: M1 = an Apple M1 Mac with 16 GB, passively cooled; x86 = an OVH cloud VM, AMD EPYC-Milan, 8 vCPU; Nvidia = an
  OVH cloud VM with one Quadro RTX 5000 (16 GB), 4 vCPU. x86 and Nvidia ran on shared VMs. In v1 the x86 figures came from
  an AMD EPYC 9R14 and the Nvidia ones from a Tesla T4: **no figure of v1 is compared with a figure of v2**.
- The run folders (configuration, provenance, raw observations) and the bench that produced them are **not distributed**:
  the figures can be read here but not regenerated from this tree. The checks of each leaf are in its `README.md`.

## Runs

| Entry | Run | Processes | Duration (s) | Forms | delta | A/A pairs |
|---|---|---|---|---|---|---|
| `m1-cpu-ffn-inference` | `m1-cpu-ffn-inference/publication_20261109T210137Z_34842` | 37 | 2912 | C | 0.341 | 148 |
| `m1-cpu-ffn-training` | `m1-cpu-ffn-training/publication_20261109T215023Z_60616` | 37 | 3282 | C | 0.074 | 148 |
| `m1-cpu-llm-inference` | `m1-cpu-llm-inference/publication_20261109T224516Z_86139` | 22 | 10847 | C + P | 0.110 | 88 |
| `m1-cpu-llm-training` | `m1-cpu-llm-training/publication_20261110T014614Z_99837` | 48 | 6364 | P | 0.034 | 96 |
| `m1-gpu-ffn-inference` | `m1-gpu-ffn-inference/publication_20261110T035231Z_6603` | 111 | 1613 | pair | 0.628 | 444 |
| `m1-gpu-ffn-training` | `m1-gpu-ffn-training/publication_20261110T041933Z_36349` | 37 | 3177 | C | 0.298 | 148 |
| `m1-gpu-llm-inference` | `m1-gpu-llm-inference/publication_20261110T051239Z_55600` | 48 | 2117 | pair | 0.037 | 192 |
| `m1-gpu-llm-training` | `m1-gpu-llm-training/publication_20261110T054806Z_68384` | 9 | 1489 | P | 0.017 | 18 |
| `nvidia-ffn-inference` | `nvidia-ffn-inference/publication_20261110T085635Z_76805` | 111 | 2921 | pair | 0.120 | 444 |
| `nvidia-ffn-training` | `nvidia-ffn-training/publication_20261110T023725Z_3664` | 111 | 2316 | pair | 0.234 | 444 |
| `nvidia-llm-inference` | `nvidia-llm-inference/publication_20261110T031657Z_5365` | 48 | 2228 | pair | 0.055 | 192 |
| `nvidia-llm-training` | `nvidia-llm-training/publication_20261110T035523Z_9800` | 28 | 1638 | C + P | 0.217 | 56 |
| `x86-ffn-inference` | `x86-ffn-inference/publication_20261110T085622Z_76740` | 37 | 3383 | C | 0.068 | 148 |
| `x86-ffn-training` | `x86-ffn-training/publication_20261109T222243Z_75788` | 37 | 1941 | C | 0.474 | 111 |
| `x86-llm-inference` | `x86-llm-inference/publication_20261109T232405Z_89492` | 16 | 4285 | C | 0.070 | 64 |
| `x86-llm-training` | `x86-llm-training/publication_20261110T003615Z_94748` | 16 | 5961 | C | 0.035 | 32 |

Qualification guards, checked on every process record: 10,243 gate results, none with `passed` other than `true`;
`swapouts_timed` 0 in every process; the thermal state of the 349 M1 processes reads no thermal warning. Every ratio of
the run summaries, every `delta` and every primary memory ratio was recomputed from the raw observations by an
independent script and reproduced exactly.

**Unqualified, no number drawn**: `x86-ffn-inference/publication_20261109T211059Z_39791` and
`nvidia-ffn-inference/publication_20261110T015128Z_608`, both `EXIT` 124 (the ceiling was reached under the first sizing;
the entries were rerun under raised ceilings). They count for nothing here.

## Not covered

- `gpu/m1/llm/training`: Pythia-410M at 20 % and 80 % are out of the timed grid, by decision (the HFFN timed process fails
  the swap guard in the smokes on the 16 GB Mac); the leaf's check covers every export at parity only.
- `gpu/nvidia/llm/inference`: one export of the family (`O1`) is a failed export by decision of the user (2026-11-07); the
  timing says nothing on it.
- `gpu/m1/ffn/training`: no primary per-arm memory reading (below, limit 6).
- x86 FFN training has no N = 1 cell.
- MoE, multilayer and lateral variants are not part of this tree.

## B against A, per entry

| Entry | delta | Cells with A | B/A median of cells | gain | within delta | loss | Memory cells with A | loss > 2 % | gain > 2 % |
|---|---|---|---|---|---|---|---|---|---|
| `m1-cpu-ffn-inference` | 0.341 | 40 | 1.000 | 0 | 40 | 0 | 10 | 0 | 0 |
| `m1-cpu-ffn-training` | 0.074 | 40 | 0.769 | 28 | 9 | **3** | 10 | **2** | 0 |
| `m1-cpu-llm-inference` | 0.110 | 12 | 0.999 | 4 | 8 | 0 | 3 | 0 | 1 |
| `m1-cpu-llm-training` | 0.034 | 6 | 0.998 | 0 | 6 | 0 | 3 | 0 | 2 |
| `m1-gpu-ffn-inference` | 0.628 | 40 | 1.006 | 0 | 40 | 0 | 10 | 0 | 0 |
| `m1-gpu-ffn-training` | 0.298 | — (first publication) | | | | | — | | |
| `m1-gpu-llm-inference` | 0.037 | 12 | 1.001 | 0 | 12 | 0 | 3 | 0 | 0 |
| `m1-gpu-llm-training` | 0.017 | 2 | 1.435 | 0 | 0 | **2** | 1 | 0 | 0 |
| `nvidia-ffn-inference` | 0.120 | 40 | 1.000 | 0 | 40 | 0 | 10 | 0 | 0 |
| `nvidia-ffn-training` | 0.234 | 40 | 1.000 | 0 | 40 | 0 | 10 | 0 | 0 |
| `nvidia-llm-inference` | 0.055 | 12 | 1.002 | 0 | 12 | 0 | 3 | 0 | 0 |
| `nvidia-llm-training` | 0.217 | — (first publication) | | | | | — | | |
| `x86-ffn-inference` | 0.068 | 40 | 1.204 | 0 | 2 | **38** | 10 | **10** | 0 |
| `x86-ffn-training` | 0.474 | 30 | 0.725 | 7 | 23 | 0 | 10 | 0 | 0 |
| `x86-llm-inference` | 0.070 | 12 | 1.036 | 0 | 9 | **3** | 3 | 0 | 0 |
| `x86-llm-training` | 0.035 | 6 | 0.977 | 2 | 4 | 0 | 3 | 0 | 3 |

A cell is *gain* or *loss* only beyond `delta`. "Memory cells with A" counts the cells where the primary memory ratio of B
over A is read at 2 %.

## Measured: reference cells (median ratio B/dense; the full tables are in each leaf's `RESULTS.md`)

"Loss" means slower than dense. FFN `512_2048_d020`:

| Entry | N1 | N8 | N64 | N512 |
|---|---:|---:|---:|---:|
| x86 FFN inference | 0.263 | 0.470 | 0.448 | 0.770 |
| x86 FFN training | | 0.719 | 0.844 | 1.301 |
| Nvidia FFN inference | 0.688 | 1.186 | 1.034 | 1.441 |
| Nvidia FFN training | 0.218 | 0.233 | 0.268 | 0.727 |
| M1 CPU FFN inference | 1.502 | 1.487 | 1.656 | 2.526 |
| M1 CPU FFN training | 0.214 | 0.118 | 0.507 | 1.926 |
| M1 GPU FFN inference | 1.258 | 1.214 | 1.296 | 1.016 |
| M1 GPU FFN training | 0.663 | 0.750 | 0.980 | 0.869 |

Pythia-410M `sparse20`, whole request or step:

| Entry | decode_b1 | decode_b4 | prefill_b1 | prefill_b4 | b1_t32 | b1_t128 |
|---|---:|---:|---:|---:|---:|---:|
| x86 LLM inference | 0.546 | 0.658 | 0.741 | 0.854 | | |
| Nvidia LLM inference | 0.951 | 0.919 | 1.329 | 1.410 | | |
| M1 GPU LLM inference | 0.571 | 0.536 | 0.775 | 1.025 | | |
| M1 CPU LLM inference | 0.706 | 0.694 | 1.561 | 2.104 | | |
| x86 LLM training | | | | | 0.799 | 1.045 |
| Nvidia LLM training | | | | | 0.797 | 0.908 |
| M1 CPU LLM training | | | | | 0.797 | 1.621 |

M1 GPU LLM training has no `sparse20` cell in the timed grid (410M at 10 %: 0.925 at `b1_t32`, 1.350 at `b1_t128`).
Whether a ratio against dense is a gain or a loss is read against the `delta` of its run, given in each leaf's table: under
a wide `delta` several of these cells are not distinguishable from 1.

## Storage (measured, `retained_bytes`, `512_2048` slice of M1 CPU FFN inference)

The adapter's count of the arm's own tensors, in bytes: dense 8,398,848; B at d005 654,978, d010 1,282,512, d020
2,543,472, d050 6,320,142, d075 9,464,856, d090 11,352,852, d100 12,609,564 (ratios to dense 0.078, 0.153, 0.303, 0.753,
1.127, 1.352, 1.501). The compact layout is smaller than the dense pair up to a density between 50 and 75 %. It does not see
native buffers; the other leaves' counts are in their tables.

## Reading

- **Density decides the sign (measured, within each run).** Along the `512_2048` slice the ratio against dense rises with
  density (for example M1 CPU FFN inference, `retained_bytes`, above). At the reference density (20 %) the ratio is below 1
  at every N on x86 FFN inference (0.263 to 0.770) and on Nvidia FFN training (0.218 to 0.727), and above 1 at every N on
  M1 CPU FFN inference (1.502 to 2.526) and on M1 GPU FFN inference (1.016 to 1.296); whether a ratio lies beyond `delta` is
  read in each leaf's table. At high density, metadata and indirection make HFFN slower than dense; this is the expected
  boundary of the format, reported and not hidden.
- **Large blocks favour dense at the reference density (measured).** N512 is the largest ratio of the reference cell on
  every FFN entry except `m1-gpu-ffn-inference` and `m1-gpu-ffn-training`. Hypothesis, not isolated: at large N the dense
  GEMM reaches its throughput regime, while the gather and scatter of the compact path do not.
- **Whole-model requests.** Each figure covers the whole request, so the dense non-FFN parts bound every ratio. At
  `sparse20`, the decode ratio against dense is below 1 on every hardware (0.536 to 0.951), the prefill ratio is below 1 on
  x86 and on the M1 GPU (except prefill_b4, 1.025) and above 1 on the M1 CPU and on Nvidia; the `delta` of each run is in the
  leaf tables. At `sparse80` the primary memory reading of B exceeds dense in every LLM entry that has an 80 % cell, except
  `x86-llm-training` (0.921).
- **B against A.** B equals A within `delta` and 2 % on `m1-cpu-ffn-inference`, `m1-gpu-ffn-inference`,
  `m1-gpu-llm-inference`, `nvidia-ffn-inference`, `nvidia-ffn-training` and `nvidia-llm-inference`, and on the time of
  `m1-cpu-llm-training`; B gains over A on `m1-cpu-ffn-training` (28 cells), `x86-ffn-training` (7), `m1-cpu-llm-inference` (4)
  and `x86-llm-training` (2), and on memory in `m1-cpu-llm-inference` (1 cell), `m1-cpu-llm-training` (2) and
  `x86-llm-training` (3). B loses against A in four entries, described under *Limits* 1 to 4.
- **Memory (measured).** On every LLM entry B reads below dense at 10 and 20 % (0.489 to 0.749). The readings differ by form
  (`retained_bytes`, `footprint_timed_max_bytes`, `device_memory_peak`) and are not compared with each other.

## Limits

Written limits carried with their evidence class and run; each is a loss or a weakness of v2 that is **published, not
repaired**. The verdict of the user of 2026-11-10 promoted v2 with these losses.

1. **x86 FFN inference: time loss against A and memory loss against A (measured,
   `x86-ffn-inference/publication_20261110T085622Z_76740`, `delta` 0.068).** B loses time beyond `delta` on 38 of 40 cells
   (B/A median 1.204, range 1.078 to 1.692; reference cell: N1 1.638 [1.256–2.432], N8 1.125, N64 1.185, N512 1.318).
   `retained_bytes` loses beyond 2 % on 10 of 10 cells (B/A 1.32 to 3.80); the difference B − A is 4,341,728 B on every
   `512_2048` cell, 2,170,848 B at `256_1024_d010` and 8,683,488 B at `1024_4096_d010`, scaling with the hidden width and not
   with the density (activation-bounded work space). Attribution to the whole-block call of the kernel (one call for the
   block of N tokens) or to the AVX2 default build is a **hypothesis**: both reached B together and no control separates
   them. The leaf's earlier promotions were judged on parity alone; this is the first timed reading against A.
2. **x86 LLM inference: prefill loss against A (measured, `x86-llm-inference/publication_20261109T232405Z_89492`,
   `delta` 0.070).** 3 of 12 cells: `sparse20 prefill_b4` 1.089, `sparse80 prefill_b1` 1.114, `sparse80 prefill_b4` 1.250;
   decode is within `delta`. The leaf calls the same kernel as limit 1 (**hypothesis**, not isolated).
3. **M1 GPU LLM training: time loss against A (measured, `m1-gpu-llm-training/publication_20261110T054806Z_68384`,
   `delta` 0.017).** 410M at 10 %: B/A 1.396 (`b1_t32`) and 1.475 (`b1_t128`), memory 0.994. It is the price of the move of
   the leaf onto the FFN training component of `gpu/m1/ffn/training`, judged on parity alone; the host-side bridging costs
   (activation layout conversion, staging copies, output allocation, compact-gradient copy, handle conversion, transient
   command objects, host synchronisation per call, native state allocated and not used) stay in the entry point of the leaf
   and are **estimated**, declared in its README, not measured. 410M at 20 and 80 % are out of the timed grid.
4. **M1 CPU FFN training: N1 loss and a memory excess against A (measured,
   `m1-cpu-ffn-training/publication_20261109T215023Z_60616`, `delta` 0.074).** N1 loss on 3 of 40 cells (`d005` 1.132, `d010`
   1.081, `d020` 1.076); memory +2.55 % (B/A 1.0255) on `512_2048_d005` and `256_1024_d010`, where the adapter counts the work
   space `y_local` that A does not hold (**hypothesis**). The same entry gains on 28 cells (median B/A 0.769).
5. **Bands too wide to discriminate (measured).** `delta` is 0.628 (`m1-gpu-ffn-inference`), 0.474 (`x86-ffn-training`),
   0.341 (`m1-cpu-ffn-inference`), 0.298 (`m1-gpu-ffn-training`) and 0.234 (`nvidia-ffn-training`, one pair; 5 of 444 pairs
   exceed 0.10). Under such a band "no loss" is a weak statement: a B up to 1.63 times slower than A in `m1-gpu-ffn-inference`
   would still read "within delta". The rule is applied as written; the width is a limit of these entries.
6. **`m1-gpu-ffn-training`: first publication, no primary memory reading per arm.** The adapter returns no `retained_bytes`
   and form C holds every instance in one process. Only the growth at the first build remains (**estimated**; at
   `512_2048_d020`: RSS 2.66, footprint 0.90, device 2.21 against dense). No memory claim is made for this leaf.
7. **`nvidia-llm-training`: first publication** (B and dense only). **`nvidia-llm-inference`**: O1 is a failed export by user
   verdict (2026-11-07).
8. **Hardware changed between v1 and v2** (Tesla T4 to Quadro RTX 5000; AWS EPYC 9R14 to an OVH EPYC-Milan VM), and the x86
   and Nvidia runs share their VMs. A comparison of v1 and v2 figures is not supported.
9. **Other written limits.** M1 GPU forms P and pair without block lifecycle; the x86 Pythia-410M form C leaves a margin of
   about 0.5 GiB on the 30 GB VM (one reading, peak RSS 31,675,084,800 B); the first timed window of some instances is slow
   (for example the A instance of `x86-ffn-inference`, window 0 about ten times window 1) and is absorbed by the per-process
   median, reported not removed; `retained_bytes` is not stated by the adapter for `m1-cpu-ffn-training`,
   `x86-ffn-inference` and `x86-ffn-training`; a retained opt-in route of `gpu/m1/ffn/training` (`--owner-tile`) is a known
   failing gate and is covered by neither the check nor the results.
10. One campaign per entry, one machine each; no figure of one entry is compared with another's. No convergence
    equivalence, global speed-up or hardware generalisation is claimed.
11. **Degree of optimisation.** The kernels were judged on parity and, against the previous versions, on time and memory
    only where the rule required it; they are not tuned for speed. Several proposals, each estimated above 2 % on the
    quantity its leaf is judged on, are not done and are carried as written limits in the leaf folders' READMEs; the
    figures above are the base they would be measured against, not a ceiling.

## Full tables

Each leaf's `RESULTS.md` holds the time table (B / dense and B / A, median [min–max] over processes and sessions) and the
memory table of its run.

## Code provenance

The figures were measured on the frozen candidate tree of each leaf. The tree published here differs from it in path
resolution (the folder is found from its own location, the assets from `HFFN_ASSETS`), documentation, comments, removed
experimental switches and dropped diagnostics; the native kernel sources are the same or differ by comments only (checked
by a textual diff against the candidate). Where cleaning changed executed code (the switches of the Metal FFN runtime, the path
of a temporary folder, renamed check variables), the targeted parity of the leaf was run on the cleaned copy alone outside
the repository (`README.md`, *Verification state*). This is a statement of equivalence by construction and by targeted
parity, not a re-measurement: the timings were not rerun.
