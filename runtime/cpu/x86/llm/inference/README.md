# HFFN CPU runtime on x86_64 for Pythia (inference)

Inference runtime of the three Pythia models (70M, 160M, 410M) pruned by WANDA, on x86_64 (AVX2/FMA). Only the
FFNs are executed by HFFN; embeddings, attention, LayerNorm, KV cache and head stay in PyTorch. The kernel is the
one of the x86 FFN leaf (`../../common/kernels`), built here with its OpenMP scheduler; the export reader, the
two-arm loader, the end-to-end token check, the timing harness and the prompt tokens are shared with the other
Pythia inference leaves (`../../../../common/pythia/hffn_pythia`).

Results: [RESULTS.md](RESULTS.md) (this leaf) and [`../../../../RESULTS.md`](../../../../RESULTS.md) (protocol and limits of the whole tree).

## Scope

| | |
|---|---|
| Degree of optimisation | the FFN kernel of the x86 FFN leaf (AVX2/FMA, vectorised over the token lanes), its groups of hidden neurons and of output features run on the OpenMP team of the process, which is PyTorch's. No tuning for the model: no kernel or threshold was changed for Pythia. |
| Hardware | Linux x86_64 with glibc, `g++` >= 9, a CPU with AVX2 and FMA (`-mavx2 -mfma` is always passed and this leaf has no scalar fallback: on a CPU without them the process stops on an illegal instruction). Measured on one machine only (see RESULTS.md). |
| Models | the three Pythia models (GPT-NeoX, 70M, 160M, 410M), float32, with the FFN of every block pruned by WANDA at 10, 20, 30, 50, 70 and 80 % density, read from compact exports (18 exports). |
| Results | [RESULTS.md](RESULTS.md); each figure carries its evidence class and the name of its qualified run. |

## Contents

This leaf lives in the `runtime/` tree of the publication and reaches two shared levels above it:

```
runtime/
  common/pythia/
    hffn_pythia/        shared by the Pythia inference leaves: export.py (reader of the hffn_training_compact_g4_v2
                        exports), model.py (two-arm loader), parity_model.py (end-to-end token check),
                        bench_request.py (request-level timing harness), assets.py (location of the assets),
                        verify_sources.py, wikitext2_test_tokens.json (prompt tokens: not distributed, generated
                        in place, see "Requirements and assets")
    tools/make_prompt_tokens.py   generates the prompt tokens
  cpu/x86/
    common/kernels/hffn_compact_g4_inference.cpp   the kernel, shared with the x86 FFN leaf
    llm/inference/      this folder
      hffn_pythia_x86/  runtime.py (HFFNLayer, HFFNRuntime: kernel call, partitions, work buffers), model.py
                        (`load_pythia`: the shared loader bound to this runtime), build.py (g++ into build/)
      checks/           parity per layer (float64 oracle) and end to end (the shared check with this loader)
      tests/            pytest: the OpenMP build of the kernel against the private-pool build
      bench/            bench_request.py (adapter of the shared harness), bench.sh, memory.py, campaign.sh,
                        campaign_manifest.py
      tools/provenance.py   hashes of the sources of a run
      sources/manifest.json hashes of the files this leaf depends on outside its folder
      check.sh          sources, compilation and parity, no timing
```

The leaf finds `common/kernels` and `common/pythia` as the first ancestor folders that have them: it is meant to be
used, and copied, as the `runtime/` tree (the x86 FFN leaf, which needs only `cpu/x86`, can sit beside it).

The prompt tokens, `wikitext2_test_tokens.json`, are 4,096 token ids derived from the WikiText-2 raw v1 test split
(CC BY-SA) with the Pythia tokenizer. The file is **not distributed** with this folder (a `.gitignore` of
`common/pythia` keeps it out of version control): it is generated in place, once, as described below.

## Requirements and assets

- Linux x86_64 with AVX2/FMA and glibc, `g++` >= 9. `pip install -r requirements.txt` (PyTorch CPU, transformers
  5.9.0, safetensors, pytest).
- The base checkpoints of the three models and the compact exports are those of the Pythia artifact of this project
  and are not part of this tree. `HFFN_ASSETS` names the folder that holds them:

```
$HFFN_ASSETS/pythia/models/base/pythia70m, pythia160m, pythia410m
$HFFN_ASSETS/pythia/exports/<model>_sparse<density>_ffn_only/export/layer<k>/...
```

`check.sh`, `bench/bench.sh`, `bench/campaign.sh` and `bench/memory.py` read these two folders from `HFFN_ASSETS`
when they are not given on the command line, and refuse to run when it is not set. The tests of `tests/` need no
asset and no prompt tokens: they build a synthetic layer from a fixed seed.

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
  The tree digest printed by `tools/provenance.py` (26 files) is that of the tree with this file in place.

## Fairness of the two arms

`load_pythia` builds the same model for both arms: same configuration, same non-FFN tensors read from the base
checkpoint, same FFN values read from the same compact export. Only the FFN call differs:

- `ffn="dense"`: the model's `GPTNeoXMLP` (two `nn.Linear` and a GELU), its weights set from the export on the
  fixed support, zero elsewhere (the dense iso-support control);
- `ffn="hffn"`: `HFFNMLP`, which passes the same activations to the HFFN runtime.

Neither arm reads the FFN tensors of the checkpoint.

## HFFN execution

T threads of the process's OpenMP team, the one PyTorch operators run on: the kernel has no thread pool of its
own. It must be loaded after PyTorch to share its OpenMP runtime. In a diagnostic session with two processes
per arm on an 8-core machine (not distributed), the kernel on its private thread pool ran the FFN 3 to 6 times slower than dense inside
the model, because that pool competes for the cores with PyTorch's OpenMP threads in active wait (estimated:
diagnostic, one session); this is why this leaf builds the OpenMP scheduler of the kernel. Per call:

- up projection and GELU per hidden neuron, over T contiguous groups of neurons;
- down projection per output feature, over T contiguous groups of outputs, reading the same edges in output order
  (inverted scheme built at load time, no reduction): the compact inverse reduction route of the WANDA article.

N = 1 takes the kernel's `decode` route, N >= 2 the `dual` route; beyond 96 tokens, the batch is processed in
consecutive blocks. Each hidden value and each output is computed by a single thread, in slot order: the result
depends neither on T nor on the partitions. Partitions are balanced by work (up slots per neuron, down edges per
output): WANDA degrees are irregular.

Compact: values, indices, offsets. Dense: `x`, `h`, `y` and their buffers, shared by all layers and reused from one
call to the next. `W_up`, `W_down` and the masks are never materialised outside the dense arm and the oracle of the
checks.

## Checks

```bash
./check.sh [<base checkpoints folder> <exports folder>]     # without arguments: $HFFN_ASSETS
python -B -m pytest -q                                       # the OpenMP build, no asset
```

Criterion, for each of the 18 exports:

- per layer: same bits at T threads as at one thread, for N in 1, 2, 7, 8, 32, 96, 128, 256;
- end to end, on text (consecutive windows of the WikiText-2 test split): **each HFFN token is equal to the token
  of the dense float32 arm, or failing that to the token of the same dense arm in float64, or failing that to a
  token whose float64 logit is within one float32 step of that of the float64 token** (a tie float32 cannot
  represent). Tokens compared: the argmax at each prefill position (four shapes: batch 1 and 4, 32 and 128 tokens,
  800 positions), then at each of the 16 steps of a KV-cache decoding where the three models receive the greedy token
  of float64, hence the same context (160 positions): 960 tokens per export.

HFFN and the dense arm are in float32 and do not add in the same order: on a near-tie of logits, either can flip.
The float64 model exists only in this check, as a referee. Numerical gaps (per-layer distance to a float64 oracle,
absolute and relative logit gap) are reported without threshold.

`tests/test_openmp_build.py` compiles the kernel twice from the shared source, with the OpenMP scheduler (this
leaf) and with the private pool, runs both on a synthetic layer of irregular supports (degrees 0 to 60) and states:
the outputs have the same bits for both schedulers and for T = 1, 2, 3, 8 (N = 1, then N up to 130, split in blocks);
they are within 2e-5 of a float64 oracle; and, once PyTorch is imported, a single OpenMP runtime is mapped in the
process after the OpenMP build is loaded.

## Request-level timing harness

`bench/bench.sh check|smoke|run [<base checkpoints folder> <exports folder>] [--threads T] [--densities 10,50]
[--sessions N]`, on the x86 machine. The harness is the shared one (`hffn_pythia/bench_request.py`); this leaf gives
it its loader, its thread count (PyTorch and HFFN run on `--threads` threads of the same OpenMP team, default the
CPUs of the process, at most 8) and its machine record (load, affinity, stolen CPU time). One fresh process per cell
(export x shape) holds three models from `load_pythia`: `hffn`, `hffn_copy` (a second HFFN instance, the A/A) and
`dense`. The three models read one copy of the non-FFN weights (`share_non_ffn`), so that the arms differ by the FFN
only: two copies of the same weights at different addresses do not run at the same speed on this machine (in a
diagnostic on batch-1 decode of Pythia-410M, 0.2 to 6.4 % between two HFFN models and 7.0 to 9.3 % between two dense
models, each with its own copy). Shapes: a prefill of 128 tokens at batch 1 and 4, and 16 cached decode steps after a
prompt of 32 tokens at batch 1 and 4. A window is a number of consecutive requests, fixed per cell so that a dense window lasts at least
300 ms; every timed window follows an untimed window of the same arm; the arms run in both orders in every process.
Gates before timing: the logits of `hffn_copy` are bitwise those of `hffn`, and `hffn` is within 1e-3 of max|logit|
of `dense`. A run is admissible when at least 95 % of the ratios `hffn_copy/hffn` (one per cell and session) are
within 1 +- 0.03; only then is `hffn/dense` read. Ratios hold within one execution (shared cloud VM); the
milliseconds of a report are not to be quoted.

## Measurement campaign

`bench/campaign.sh` runs the measurements of this leaf on an 8-core x86 machine, step by step; its results go to
`results/<tag>/` and a record says whether the ratios can be read. The figures of this version come from a bench that is not distributed (see [`../../../../RESULTS.md`](../../../../RESULTS.md)); the tooling described here is the one of the leaf, kept to run the same kind of measurement again.

**What it measures.** For the exports of the grid (Pythia 70M, 160M, 410M at chosen densities) and the four request
shapes above, on 8 CPUs with 8 threads for PyTorch and HFFN: the time of a whole request (prefill, or prompt plus 16
cached decode steps) with the FFNs executed by HFFN, against the same model with the dense iso-support FFNs. It reports
where HFFN is faster or slower than dense; it sets no speed to reach. It does not measure batch sizes other than 1 and
4, sequences other than 128 prefill tokens and 32 + 16 decode tokens, other processors, other thread counts for the
dense arm, other models or training.

**Steps** (`campaign.sh run <tag>`; a step that fails ends the campaign):

| Step | What | Cap |
|---|---|---|
| `provenance` | `tools/provenance.py`: hashes of the kernel, of the shared `hffn_pythia` package and of the files of this leaf | 5 min |
| `tests` | `pytest`: the OpenMP build against the private-pool build | 30 min |
| `check`, `check_coverage` | `check.sh` on the 18 exports; the log must hold 18 lines `unexplained HFFN tokens -> PASSED` | 1 h |
| `bench_run` | the request harness on the full grid (3 sizes x 4 densities x 4 shapes x 3 sessions = 144 processes), 8 threads | 4 h |
| `bench_threads4` | the same harness, PyTorch and HFFN on 4 threads, one session (24 processes) | 1 h |
| `memory` | `bench/memory.py`: one fresh process per arm and export: exact bytes of the FFNs, resident and peak memory | 1 h |

`campaign.sh bench <tag>` runs the steps `provenance`, `tests`, `bench_run` and `bench_threads4` only. The campaign may
not start a step after 4 hours and ends with `EXIT=124` (status `partial`) if it reaches them.

**Admissibility, as the tool applies it** (the criteria of the whole tree are in
[`../../../../RESULTS.md`](../../../../RESULTS.md)).

- Gates, in each process of the harness: the logits of `hffn_copy` are bitwise those of `hffn`, and `hffn` is within
  1e-3 of max|logit| of `dense`. A failed gate ends the campaign.
- The ratios `hffn/dense` are read only if the campaign is **admissible**: status `passed`, exit code 0, full coverage,
  CPU time stolen by the hypervisor below 0.5 %, and at least 95 % of the ratios `hffn_copy/hffn` within 1 +- 0.03. A
  ratio `hffn_copy/hffn` is that of the medians of the windows of a process. Otherwise no ratio is read and the cause is
  reported.
- There is no speed threshold. For each cell (model, density, shape) the tool reports the median over the sessions of
  `hffn/dense` with the interval [min-max] of the sessions. A cell whose spread, (max - min) / median, exceeds 10 % is
  **unstable** and carries no quantified statement. A cell whose median is within 1 +- 0.03, the tolerance within which
  the harness does not tell two identical HFFN copies apart, is **unresolved**.
- The 4-thread run is one session, reported without interval.
- Memory: the sizes of the FFN arrays are exact; the resident memory is read after the requests, the peak is a loading
  peak for the HFFN arm (the shared loader builds the dense FFNs before replacing them); one process per measurement,
  no interval.
- A failed or partial campaign gives no performance number, only the cause and what is missing.

## Known limits

- The dense arm keeps one copy of its FFN weights and has no A/A: between two dense models of one process, a gap of
  0.9 to 7.7 % remains in batch-1 decode once the non-FFN weights are shared (diagnostic on the same machine, 8
  processes, not distributed with this folder; the model loaded first being the faster in 8 processes of 8). A ratio
  `hffn/dense` of such a cell is uncertain by that much, whatever its interval says; prefill and batch 4 were not probed.
- One machine type, one boot per run; the sessions follow each other and measure repeatability from one process to
  another, not from one day or one instance to another. Only Pythia-410M is timed in the published figures.
- One text excerpt (4,096 tokens) for the end-to-end check; Gaussian inputs for the per-layer check.
- Numerical gaps: the per-layer distance of HFFN to a float64 oracle exceeds that of the dense float32 arm (float32
  accumulation over supports of several hundred edges per output); it is reported, not bounded.
- The token criterion rests on a clause for near-ties that float32 cannot represent; strict token equality between two
  float32 executions is not guaranteed.
- AVX2/FMA required; 8 threads in the published run; one prompt per shape; the dense arm is PyTorch's float32 GEMM through
  `nn.Linear`, on the same OpenMP team.
- HFFN is slower than dense where the FFN is dense enough and the batch large (long prefill at high density): the
  expected boundary of the format. RESULTS.md gives the cells.
- The `delta` noise band of the published run is large for a CPU run; cells close to 1 read *within delta*.

## Provenance

`sources/manifest.json` gives the origin and the SHA-256 of the files this leaf depends on outside its folder.
`python -B tools/provenance.py` prints the tree digest, the kernel hash and the number of files that a run records;
`--files` the hash of each file. On this version it prints `tree_sha256=5dbc047cb824d79db414fc7b09449f46603d9a2c5fb97cc699b9b38348cd0775` (29 files) once the prompt
tokens have been generated ("Requirements and assets"); without that file the tool prints another digest and one file fewer.
The kernel was ported to x86_64 from the HFFN compact kernel of the Apple M1 CPU runtime.

## License

MIT. The license is the file `LICENSE` at the root of the `runtime/` folder (`../../../../LICENSE` from this folder); it covers the whole `runtime/` tree: the kernel, the x86 FFN leaf, the shared Pythia level and this leaf.
