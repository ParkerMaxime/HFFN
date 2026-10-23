# HFFN GPU runtime on NVIDIA (CUDA) for Pythia (inference)

Inference runtime of the three Pythia models (70M, 160M, 410M) pruned by WANDA, on an NVIDIA GPU. Only the FFNs are
executed by HFFN; embeddings, attention, LayerNorm, KV cache and head stay in PyTorch/CUDA. The six CUDA kernels are a
port of the Metal kernels of the Apple M1 GPU runtime of this project; the export reader, the two-arm loader, the
end-to-end token check, the timing harness and the prompt tokens are shared with the other Pythia inference leaves
(`../../../../common/pythia/hffn_pythia`).

This is a **validated port with its measured boundary**, not a tuned runtime. It is checked on 18 exports (parity per
layer and end to end); its figures, on the one GPU where it was timed (a Tesla T4), are in RESULTS.md. The kernels and
their selection thresholds are those of the Metal reference and were not tuned for CUDA.

Results: [RESULTS.md](RESULTS.md) (this leaf) and [`../../../../RESULTS.md`](../../../../RESULTS.md) (protocol and limits of the whole tree).

## Scope

| | |
|---|---|
| Degree of optimisation | a port, not a tuned kernel set. The six kernels (three up, three down), their work distribution, the order of edges and sums, the `[feature, N]` layout and the selection thresholds are those of the Metal reference; the thresholds (cooperative reduction up to N = 4, up tiles from N = 64, "blocked" down from N = 64 if `d_ff` <= 4096 or N <= 128; the width bound is 4096 on CUDA) were measured on an Apple M1 and are **provisional** here: they were not remeasured on a CUDA GPU, and the kernel pairs were not timed. No kernel or threshold was changed for Pythia or for CUDA. |
| Hardware | an NVIDIA GPU with CUDA 12, one device (`cuda:0`), PyTorch's default CUDA stream (the runtime refuses another one); kernels compiled at run time by NVRTC through CuPy. Measured on one Tesla T4 (compute capability 7.5) only; no other GPU was tried. |
| Models | the three Pythia models (GPT-NeoX, 70M, 160M, 410M), float32, with the FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from compact exports (18 exports). `d_model` must be a multiple of 16 (checked); the compact indices are 16-bit, so `d_model` and `d_ff` must stay below 32,768 (not checked by the code); one call takes 1 to 262,144 tokens. |
| Results | [RESULTS.md](RESULTS.md); each figure carries its evidence class and the name of its qualified run. |

## Contents

This leaf lives in the `runtime/` tree of the publication and reaches the shared Pythia level above it:

```
runtime/
  common/pythia/
    hffn_pythia/        shared by the Pythia inference leaves: export.py (reader of the hffn_training_compact_g4_v2
                        exports), model.py (two-arm loader), parity_model.py (end-to-end token check),
                        bench_request.py (request-level timing harness), assets.py (location of the assets),
                        verify_sources.py, wikitext2_test_tokens.json (prompt tokens: not distributed, generated
                        in place, see "Requirements and assets")
    tools/make_prompt_tokens.py   generates the prompt tokens
  gpu/nvidia/llm/inference/      this folder
      hffn_pythia_cuda/  prepared_runtime.py (PreparedRuntime: the decode `coop`/`coop` FFN call launched from a descriptor; the kernels
                         (kernels.py) and HFFNLayer/HFFNRuntime (runtime.py) are the shared `gpu/nvidia/common/kernels/hffn_cuda`),
                         model.py (`load_pythia`: the shared loader bound to this runtime)
      checks/            parity per layer (float64 oracle) and end to end (the shared check with this loader)
      bench/             bench_request.py (CUDA adapter of the shared harness), bench.sh, memory.py (memory of the two arms)
      tools/provenance.py   hashes of the sources of a run
      sources/manifest.json hashes of the files this leaf depends on outside its folder
      check.sh           sources and parity, no timing
      launch.sh          check.sh with its record (log, exit code, coverage)
```

The leaf finds `common/pythia` as the first ancestor folder that has it: it is meant to be used, and copied, as the
`runtime/` tree.

The prompt tokens, `wikitext2_test_tokens.json`, are 4,096 token ids derived from the WikiText-2 raw v1 test split
(CC BY-SA) with the Pythia tokenizer. The file is **not distributed** with this folder (a `.gitignore` of
`common/pythia` keeps it out of version control): it is generated in place, once, as described below.

## Requirements and assets

- An NVIDIA GPU with a driver for CUDA 12, Linux. `pip install -r requirements.txt` (PyTorch with CUDA, CuPy for CUDA
  12, transformers 5.9.0, safetensors). Versions of the measurements: see RESULTS.md.
- The base checkpoints of the three models and the compact exports are those of the Pythia artifact of this project and
  are not part of this tree. `HFFN_ASSETS` names the folder that holds them:

```
$HFFN_ASSETS/pythia/models/base/pythia70m, pythia160m, pythia410m
$HFFN_ASSETS/pythia/exports/<model>_sparse<density>_ffn_only/export/layer<k>/...
```

`check.sh`, `launch.sh` and `bench/bench.sh` read these two folders from `HFFN_ASSETS` when they are not given on the
command line, and refuse to run when it is not set. The checks need a GPU and the exports: this leaf has no test that
runs without assets.

- **Generate the prompt tokens once**, before `check.sh` or the harness. It needs the `datasets` package (not in
  `requirements.txt`: only this step uses it) and the Arrow file of the WikiText-2 raw v1 test split (for example the
  `wikitext-2-raw-v1` test file in the local cache of the Hugging Face `wikitext` dataset); the tokenizer is that of
  the base checkpoints (the three sizes share one):

```bash
python -B runtime/common/pythia/tools/make_prompt_tokens.py $HFFN_ASSETS/pythia/models/base/pythia70m <wikitext-test.arrow>
```

  It writes `runtime/common/pythia/hffn_pythia/wikitext2_test_tokens.json`. Until the file exists, `check.sh` (through
  its source verification), the end-to-end check and the timing harness stop with a file-not-found error on that path.
  `sources/manifest.json` pins its SHA-256, which `check.sh` verifies:
  `a6ea1a41a54531b5b6120f8eaebf24ad7275083e2f7c11ac5f50be7b7f955178`. The file records the SHA-256 of both of its
  inputs (the Arrow file `2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea` and the tokenizer
  `tokenizer.json` `9e286b858bb00093fa93f50982798afd9f6b31c28e4d7c2d08adcd68d83a2695`), so the pinned hash holds for
  those inputs: a differently serialised copy of the same text gives another hash, and the check then fails by design.
  The tree digest printed by `tools/provenance.py` (22 files) is that of the tree with this file in place.

## Fairness of the two arms

`load_pythia` builds the same model for both arms: same configuration, same non-FFN tensors read from the base
checkpoint, same FFN values read from the same compact export, same device. Only the FFN call differs:

- `ffn="dense"`: the model's `GPTNeoXMLP` (two `nn.Linear` and a GELU), its weights set from the export on the
  fixed support, zero elsewhere (the dense iso-support control);
- `ffn="hffn"`: `HFFNMLP`, which passes the same activations to the HFFN runtime.

Neither arm reads the FFN tensors of the checkpoint. The choice is made at load time: there is no automatic switch to
dense depending on density.

## Reference and deviations

The reference of the kernels is the "one bank, one segment" path of the `full` runtime of the artifact `hffn_moe_v4`
of this project (Apple M1 GPU, Metal), qualified there for an addition Transformer (`d_model` = 256).

Taken from the reference: the three up kernels and the three down kernels, their work distribution (block of 512
threads, 16 warps of 32 lanes; one warp per hidden neuron or per output), the order of edges and sums, the
`[feature, N]` layout of activations, the output written in token order, and the selection thresholds.

Declared deviations:

- `d_model`, `d_ff` and `N` are read at run time (the reference fixes 256);
- GELU through CUDA's `erff`, not through the erf approximation of the Metal prelude;
- compilation by CuPy (`RawModule`, NVRTC); buffers belong to PyTorch, kernels are queued on PyTorch's default CUDA
  stream, hence executed in call order;
- one persistent output buffer, instead of an allocation per call and a call queue protected by events;
- not ported: no rows aligned on 64 bytes, no MoE machinery of banks, routes and gates, no native attention and norms.

## HFFN execution

| N | Up (owner: hidden neuron) | Down (owner: output) |
|---|---|---|
| <= 4 | `coop`: the 32 lanes reduce the edge list, input in token order | `coop`: same, per output |
| 5 to 63 | `lane`: one lane per token | `tiled`: one thread per (output, token) |
| >= 64 | `tile`: tiles of 128 tokens, 4 accumulators | `blocked` (2 tokens per thread) if `d_ff` <= 4096 or N <= 128, otherwise `tiled` (shared `BLOCK_MAX_WIDTH` = 4096, chosen from timings on a T4 at d_ff 3072 and 4096 that are not distributed) |

Each hidden value and each output is written by a single thread: no atomics, no scatter. The `coop` kernels add 32
partial sums through a tree of shuffles; the others add in slot order.

Compact: values, indices, offsets (down permuted once into output order at load time). Dense: `x`, `h`, `y` and their
buffers, shared by all layers and reused. `W_up`, `W_down` and the masks are never materialised outside the dense arm
and the oracle of the checks.

## Checks

**This leaf carries no test that runs without an asset and a GPU**: its checks read the compact exports and the base
checkpoints, so they need the assets (`HFFN_ASSETS`) and a CUDA GPU. The synthetic-layer tests of the Nvidia
kernels are in `../../ffn/inference` and `../../ffn/training`.

```bash
./check.sh [<base checkpoints folder> <exports folder>]     # without arguments: $HFFN_ASSETS
./launch.sh [<base checkpoints folder> <exports folder>]    # the same, with a record in runs/ and a coverage count
```

Criterion, for each of the 18 exports:

- **per layer**, N in {1, 2, 4, 5, 32, 63, 64, 128, 129, 256, 512}, with the runtime policy and with each of the nine
  forced (up, down) pairs:
  1. two consecutive calls on the same input give the same bits;
  2. `max |y - y_float64| / max(1, |y_float64|) <= 1e-3` against a float64 oracle of the same fixed-support FFN. This
     guard detects a wrong kernel; it is not a precision claim.
- **end to end**, on text (consecutive windows of the WikiText-2 test split): **each HFFN token is equal to the token
  of the dense float32 arm, or failing that to the token of the same dense arm in float64, or failing that to a token
  whose float64 logit is within one float32 step of that of the float64 token** (a tie float32 cannot represent).
  Tokens compared: the argmax at each prefill position (four shapes: batch 1 and 4, 32 and 128 tokens, 800
  positions), then at each of the 16 steps of a KV-cache decoding where the three models receive the greedy token of
  float64, hence the same context (160 positions): 960 tokens per export.

HFFN and the dense arm are in float32 and do not add in the same order: on a near-tie of logits, either can flip. The
float64 model exists only in this check, as a referee. Reported without threshold: the absolute per-layer distance to
the float64 oracle (and that of dense float32), the logit gap, the bit identity between the four sequential pairs
(`lane`/`tile` x `tiled`/`blocked`), and the kernels actually called. The kernels are compiled by NVRTC with its
default options (FMA contraction allowed): the bits are not comparable with those of the Metal runtime.

## Request-level timing harness

`bench/bench.sh check|smoke|run [<base checkpoints folder> <exports folder>] [--densities 10,50] [--sessions N]`, on the
GPU machine, from the CUDA environment. The harness is the shared one (`hffn_pythia/bench_request.py`);
`bench/bench_request.py` is its CUDA adapter (GPU state from `nvidia-smi`, `torch.cuda.synchronize()`, the runtime
description). One fresh process per cell (export x shape) holds three models from `load_pythia`: `hffn`, `hffn_copy`
(a second HFFN instance, the A/A) and `dense`. The three models read one copy of the non-FFN weights
(`share_non_ffn`), so that the arms differ by the FFN only. Shapes: a prefill of 128 tokens at batch 1 and 4, and 16
cached decode steps after a prompt of 32 tokens at batch 1 and 4. A window is a number of consecutive requests between
two `torch.cuda.synchronize()`, fixed per cell so that a dense window lasts at least 300 ms; every timed window follows
an untimed window of the same arm; the arms run in both orders in every process. Gates before timing: the logits of
`hffn_copy` are bitwise those of `hffn`, and `hffn` is within 1e-3 of max|logit| of `dense`. A run is admissible when
at least 95 % of the ratios `hffn_copy/hffn` (one per cell and session) are within 1 +- 0.03; only then is
`hffn/dense` read. Outputs in `runs/bench_<mode>_<UTC>_<pid>/` (`config.json`, `provenance.json`, `raw.jsonl`,
`summary.json`, `REPORT.md`, `run.log`, `manifest.json`, `EXIT`). Ratios hold within one execution (shared cloud GPU);
the milliseconds of a report are not to be quoted.

## Memory

`python -B bench/memory.py --out <json> [--models <folder> --exports <folder>] [--sizes 70,160,410] [--densities 10,20,50,80]`,
on the GPU machine. One fresh process per arm (dense, HFFN) and export, 12 exports of the timing grid by default: it reports
the exact bytes of the FFN arrays of the arm and the GPU memory held by PyTorch's allocator after load, after a prefill
(batch 4, 128 tokens) and after a decode (batch 4, prompt of 32 tokens, 16 cached steps), with its peak and what it
reserved from the driver, and the bytes of CuPy's own pool beside them. No threshold: the figures are reported. The memory
of the CUDA context is not measured; one process per measurement, no interval.

## Measurement

`bench/bench.sh` and `bench/memory.py` run the measurements of this leaf on a GPU machine. The figures of this version come from a bench that is not distributed (see [`../../../../RESULTS.md`](../../../../RESULTS.md)); the tooling described here is the one of the leaf, kept to run the same kind of measurement again.

**What it measures.** For the exports of the grid (Pythia 70M, 160M, 410M at chosen densities) and the four request
shapes above, on one GPU: the time of a whole request (prefill, or prompt plus 16 cached decode steps) with the FFNs
executed by HFFN, against the same model with the dense iso-support FFNs, in sessions of fresh processes
(`bench/bench.sh run`); and the memory of the two arms (`bench/memory.py`). It reports where HFFN is faster or slower
than dense; it sets no speed to reach. It does not measure batch sizes other than 1 and 4, sequences other than 128
prefill tokens and 32 + 16 decode tokens, other GPUs, other models, the memory of the CUDA context, training, or the
kernel pairs and thresholds.

**Admissibility, as the tool applies it** (the criteria of the whole tree are in
[`../../../../RESULTS.md`](../../../../RESULTS.md)).

- Gates, in each process of the harness: the logits of `hffn_copy` are bitwise those of `hffn`, and `hffn` is within
  1e-3 of max|logit| of `dense`. A failed gate ends the run.
- The ratios `hffn/dense` are read only if the run is **admissible**: status `passed`, exit code 0, full coverage, and at
  least 95 % of the ratios `hffn_copy/hffn` within 1 +- 0.03. A ratio `hffn_copy/hffn` is that of the medians of the
  windows of a process. Otherwise no ratio is read and the cause is reported.
- There is no speed threshold. For each cell (model, density, shape) the tool reports the median over the sessions of
  `hffn/dense` with the interval [min-max] of the sessions. A cell whose median is within 1 +- 0.03, the tolerance
  within which the harness does not tell two identical HFFN copies apart, is **unresolved**; a cell whose spread,
  (max - min) / median, exceeds 10 % is **unstable** and carries no quantified statement.
- A failed or partial run gives no performance number, only the cause and what is missing.

## Known limits

- One GPU (a shared cloud Tesla T4), one execution per run, one boot: the three sessions follow each other and measure
  repeatability from one process to another, not from one day or one instance to another. The absolute time of an arm
  can change level from one execution to another: ratios are read within an execution, milliseconds are not quoted.
- The dense arm keeps one copy of its FFN weights and has no A/A. On the x86 CPU, two dense models of one process
  differed by up to 7.7 % in batch-1 decode; this was not probed on the GPU. A ratio `hffn/dense` is uncertain by that
  much in a cell of that kind, whatever its interval says; the cells within 3 % of 1 are read "unresolved".
- No kernel or threshold was tuned for CUDA; the HFFN kernels may be far from what this GPU allows. The results are the
  boundary of this port, not of the format.
- Memory: one process per measurement and arm, no interval, one batch size and one sequence length; the memory of the CUDA
  context is not measured.
- One text excerpt (4,096 tokens) for the end-to-end check; Gaussian inputs for the per-layer check; N <= 512 and decoding
  batch <= 4 in the checks; incomplete tiles are covered by N = 63 and 129 only.
- Numerical gaps: the per-layer distance of HFFN to a float64 oracle exceeds that of the dense float32 arm (float32
  accumulation over supports of several hundred edges per output); it is reported, not bounded.
- The token criterion rests on a clause for near-ties that float32 cannot represent; strict token equality between two
  float32 executions is not guaranteed.
- Where HFFN is slower than dense (see RESULTS.md) is the boundary of this port on this GPU, reported as measured, not a
  boundary of the format.

## Provenance

`sources/manifest.json` gives the origin and the SHA-256 of the files this leaf depends on outside its folder.
`python -B tools/provenance.py` prints the tree digest and the number of files that a run records, and `--files` the
hash of each file. On this version it prints `tree_sha256=1cea445e25d855e0e0a87975e601f5311f626070b7299698f63886accc898163` (29 files) once the prompt tokens have
been generated ("Requirements and assets"); without that file the tool prints another digest and one file fewer. The
kernels were ported to CUDA from the Metal kernels of the Apple M1 GPU runtime of this project.

## License

MIT. The license is the file `LICENSE` at the root of the `runtime/` folder (`../../../../LICENSE` from this folder); it
covers the whole `runtime/` tree: the shared Pythia level, the other leaves and this leaf.
