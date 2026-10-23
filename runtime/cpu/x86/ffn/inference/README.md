# HFFN — CPU runtime on x86_64

CPU inference runtime for fixed-support sparse FFNs in the HFFN (Hidden-FFN) format, on x86_64 (AVX2/FMA). This
folder contains the runtime (a Python loader on top of the C++ kernel of `../../common/kernels`), the parity checks
and the benchmarks. The generator and the fingerprints of the test layers are shared (`../../../../common/ffn`).
The kernel runs on an adjustable number of threads T (1 to 64, at most 8 by default).

Results: [RESULTS.md](RESULTS.md) (this leaf) and [`../../../../RESULTS.md`](../../../../RESULTS.md) (protocol and limits of the whole tree).

## Scope

| | |
|---|---|
| Degree of optimisation | one kernel, vectorised over the token lanes with AVX2/FMA (scalar fallback without them), T threads on a persistent pool, deterministic reductions. No AVX-512, AMX, BF16 or INT8 path. |
| Hardware | Linux x86_64, glibc, g++ >= 9. Measured on one machine only: AMD EPYC 9R14 (Zen 4), 8 vCPU (the figures of RESULTS.md). Not validated with clang or on an Intel processor. |
| Models | An isolated FFN layer given in the compact hidden-major layout (`hffn_ffn_layers.synth.CompactLayer`), GELU (exact), float32. Tested on three synthetic layers (d_model 256 and 512, about 10 % density) and measured on isolated FFN cells of the shared exports (RESULTS.md). No real model here: the LLM use of the kernel is the business of the x86 LLM leaves. |
| Results | [RESULTS.md](RESULTS.md); each figure carries its evidence class and the name of its qualified run. |

## What HFFN computes

```
y = W_down · gelu(W_up · x + b_up) + b_down        (exact GELU, float32)
```

The sparse FFN is stored **per hidden neuron**: each neuron `r` owns its active incoming connections
`(up_idx, up_val)`, its bias `b_up[r]` and its active outgoing connections `(down_idx, down_val)`. The result is
that of the original sparse FFN; only the layout and the scheduling change. `W_up`, `W_down` and dense masks are
never materialised. Only `x`, `h`, `y` and the reduction buffers are dense.

## Contents

```
../../common/kernels/hffn_compact_g4_inference.cpp   the kernel, shared by the x86 leaves: AVX2/FMA, scalar fallback, T threads
hffn/                   Python loader (ctypes) and tools
  compact.py            CompactKernel: routes, split into blocks, work buffers
  build.py              g++ compilation -> build/ (name = hash of the command and the source)
  checks.py             parity contract (python -m hffn.checks)
  provenance.py         hashes of the kernel and of the files of this folder (python -m hffn.provenance)
reference/              sources.json (origin and hash of the kernel)
../../../../common/ffn/      shared test layers, outside this folder, imported as `hffn_ffn_layers` (`import hffn` puts it on `sys.path`): synth.py (generator of the synthetic layers, PCG64, reproducible), layers.py (the test layers, checked against reference/fingerprints.json), reference.py (float64 oracle), dense.py (dense PyTorch reference, benchmarks and tests only), reference/fingerprints.json (the pinned test layers)
tests/                  pytest: parity (including T = 1, 2, 3, 8), bits, memory, absence of PyTorch
bench/                  benchmarks, campaign.sh, aggregate.py, summarize.py, vm_setup.sh
RESULTS.md              the figures of this version, with their provenance
```

The folder reaches the kernel through the nearest ancestor that holds `common/kernels/`: it is meant to be used
(and copied) as part of the `runtime/` tree (`runtime/cpu/x86/common/kernels` and `runtime/common/ffn` above this leaf).

## Requirements

- Linux x86_64, `g++` ≥ 9 (C++17). AVX2 and FMA for the vector paths.
- Runtime: Python ≥ 3.10 and numpy. The kernel depends only on libstdc++, libm/libmvec and libc. PyTorch is
  **not** needed.
- Tests and benchmarks: `torch` (CPU), `scipy`, `pytest`, for the dense references.

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy                         # runtime only
pip install -r requirements.txt           # tests and benchmarks
# (CPU wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu)
```

## Quick start

```bash
export HFFN_CXXFLAGS="-mavx2 -mfma"       # enables the AVX2 paths (otherwise scalar fallback)
python -m pytest -q                        # < 1 min, compilation included
python -m hffn.checks                      # detailed parity per test layer
```

```python
import numpy as np
from hffn import layers
from hffn.compact import CompactKernel

layer, case = layers.load("irreg_512_2048")         # or your own CompactLayer
kernel = CompactKernel(layer, hidden_owned=False)    # compiles the shared kernel at the first call; threads=8 to fix T
x = np.random.default_rng(0).standard_normal((32, layer.d_model), dtype=np.float32)
y = kernel.forward(x)                                # [N, d_model] float32, recommended route
```

A layer is built with `hffn_ffn_layers.synth.CompactLayer`, in the canonical hidden-major layout: `groups` int32[5]
(partition of the neurons into 4 groups), `up_offsets` / `down_offsets` int32[d_ff + 1], `up_idx` / `down_idx`
int16, `up_val` / `down_val` / `b_up` / `b_down` float32.

## The runtime

T groups of hidden neurons (G4 is T = 4), one thread per group: the calling thread plus T - 1 threads of a
persistent pool. Computations are vectorised over the token lanes (contiguous `N` axis in the transposed
`[feature, N]` layout). Up to 96 tokens, the batch of a row stays in ymm registers; batch tails use AVX2 masks.

- **Number of threads**: `CompactKernel(layer, threads=T)`. By default, T is the number of CPUs of the process's
  affinity mask, capped at 8 (`hffn.compact.default_threads`). The loader computes the partitions for T: hidden
  neurons in T contiguous blocks of equal size (for T = 4, these are the G4 groups of the test layers), output
  features in T blocks with the same number of edges.
- **Bits and T**: `decode` and `dual` give exactly the same outputs for every T (each hidden value and each output
  is computed by a single thread, in slot order). `baseline` reduces T partial outputs: its bits depend on T,
  within the tolerance.
- **Bits and batch** (AVX2 builds on glibc): every GELU goes through the same 8-way `erff` function of libmvec.
  The output of a token on the `dual` route depends neither on N nor on its position in the batch.
- **Model memory**: the recommended routes read only the up arrays and the inverted scheme.
  `CompactKernel(layer, hidden_owned=False)` does not keep the down arrays stored per neuron (a third of the
  model); the `baseline` route is then unavailable. `model_bytes(route)` gives the bytes read by a route.
- **Pool**: a single pool per process, shared by all kernels. Its workers are created at the first call that
  needs them. A call on T groups uses the first T - 1; the others ignore it. The task is published through the
  atomic generation counter and read without a lock. One call at a time per process.

| Route | When | Principle |
|---|---|---|
| `decode` | N = 1 (recommended) | up per group of neurons, then down per output feature on the inverted scheme; no transposition, no private buffer, no reduction |
| `dual` | N ≥ 2 (recommended) | up per neuron, then down per output feature following an inverted scheme (no reduction) that carries, per edge, the hidden neuron and the down weight in output order; from N · d = 12,288, the input and output transpositions are done by the output groups, each for the rows it owns |
| `baseline` | N ≥ 2 | up and down per neuron; each group zeroes its slice of `y_local[T, d, N]`, then the reduction is split by output features; slower than `dual` on the measured cases |

The `baseline` route is HFFN execution in the strict sense: one hidden neuron, one thread, for both projections.
The recommended routes keep the up projection per hidden neuron and distribute the down projection per output
feature, over the same edges. This is the mechanism of the compact inverse reduction routes of the WANDA article.

- `forward(x)` picks the recommended route (`hffn.compact.recommended_route`); `forward(x, route)` forces a route.
- Batches of more than 96 tokens are split into balanced blocks, multiples of 8 (`batch_chunks`), processed one
  after the other.
- Work buffers are shared by role, sized for the largest N seen, and allocated outside the hot path. `y_t` reuses
  `x_t`. They take about 10 kB per token of the largest block. They start on a 64-byte boundary
  (`WORK_ALIGNMENT`): with numpy's 16-byte alignment, the kernel time depended on the address returned by malloc.
- Pool threads wait actively at most 100 µs after each call, then sleep. This duration is set at compile time:
  `HFFN_CXXFLAGS="... -DHFFN_POOL_SPIN_NS=<ns>"`.
- `kernel_ns(n)` gives the phase times measured inside the kernel for the last call at batch `n`.

**Compilation**: `hffn/build.py` runs `g++ -O3 -ffast-math -std=c++17 -fPIC -shared -pthread $HFFN_CXXFLAGS`.
Changing flags recompiles: the library name contains the hash of the command. The same source also has a
compile-time scheduler on an OpenMP team (`-DHFFN_GROUPS_OPENMP -fopenmp`), which is the choice of the x86 LLM
leaf running inside PyTorch's OpenMP team; it is neither built nor measured here.

## Parity contract

On each route and each N, the kernel must stay within **2e-5** absolute error (outputs of magnitude ~6) of the
float64 oracle (`hffn_ffn_layers/reference.py`, in `runtime/common/ffn`); the gaps measured on this tree are in RESULTS.md. With `-ffast-math` and FMA
contraction, the bits depend on the compiler: parity is a tolerance, not a binary equality. Do not relax the
tolerance to make a kernel pass.

The tests also check that the generated test layers and the benchmark inputs have the pinned fingerprints
(`common/ffn/reference/fingerprints.json`): every check and every benchmark runs on the same layers and the same inputs.

## Test layers

| Layer | d_model | d_ff | up/down degree | support | up + down edges |
|---|---|---|---|---|---|
| `homog_256_1024` | 256 | 1024 | 26 | homogeneous | 53,248 |
| `homog_512_2048` | 512 | 2048 | 51 | homogeneous | 208,896 |
| `irreg_512_2048` | 512 | 2048 | 0 to 102 (mean ~51) | irregular, WANDA-like | ~209,000 |

These layers are synthetic (about 10 % density) and nothing is stored: `hffn_ffn_layers/synth.py` generates each one from its
case parameters and seed with numpy's PCG64 generator, and `hffn_ffn_layers.layers.load(name)` refuses a layer or an input
whose fingerprint differs from `common/ffn/reference/fingerprints.json`. `layers.make_input(case, n)` regenerates the
benchmark inputs deterministically. `import hffn` then `hffn_ffn_layers.layers.main()` checks the three layers (what `bench/campaign.sh` runs).

## Benchmarks

| Script | Measurement |
|---|---|
| `bench/threads.py` | scaling in T: kernel time per T, speed-up against T = 1, fastest T and smallest T within 5 % of it, per case and per N |
| `bench/routes.py` | `baseline` against `dual` |
| `bench/vs_dense.py --phase perf` | HFFN against four dense variants, warm and cold cache |
| `bench/vs_dense.py --phase memory` | model, buffers held, RSS peak per call, pages touched |
| `bench/memory.py` | runtime memory per route and N, serving scenario |
| `bench/process_footprint.py --all` | memory of the whole process per method |
| `bench/summarize.py <folder>` | `summary.md` tables of a folder |
| `bench/campaign.sh`, `bench/aggregate.py` | the campaign and its multi-session aggregate |

Common options: `--cpus LIST` fixes the affinity of the process (HFFN threads, PyTorch, OpenBLAS); `--threads T`
fixes the number of kernel threads (default: the CPUs of the affinity, 8 at most); `--dense-threads N|all` fixes
the threads of the dense methods (default: the same number as HFFN).

Measurement rules (each rule corrects an observed bias):

- Compare two kernel variants on `kernel_us` (time measured inside the kernel, without the Python/ctypes
  overhead). Compare HFFN with dense in wall time, with the same Python entry point.
- Never interleave, call by call, methods that wait actively (HFFN pool, OpenMP, OpenBLAS): the benchmarks
  alternate by blocks, with a pause between blocks, including after the parity check that precedes the first
  block.
- Warm each block up for a duration (5 ms at least), not only a number of calls: after a pause, all methods run
  about twice as slowly for 1 to 2 ms, and ten calls of a fast method do not cover this ramp.
- Align the work buffers: with numpy's alignment, two instances of the same kernel differed by 0.93 to 1.22
  depending on the buffer address. An A/A test (two identical copies) must return a ratio close to 1 before any
  comparison.
- Fix the glibc `mmap` threshold for memory measurements only: the scripts relaunch themselves with this
  threshold.
- `torch.compile`: the recompilation limit is raised, otherwise PyTorch silently falls back to eager mode after
  8 batch sizes.

## Measurement campaign on a dedicated machine (8 cores)

`bench/campaign.sh` runs the measurements of this leaf on a dedicated multi-core machine, step by step, and writes a
record that says whether the run can be read. The figures of this version come from a bench that is not distributed (see [`../../../../RESULTS.md`](../../../../RESULTS.md)); the tooling described here is the one of the leaf, kept to run the same kind of measurement again.

**What it measures.** On 8 physical cores of one NUMA node, for the three test layers and N from 1 to 128: the compact
kernel at T = 8 (recommended routes) against the best of the four dense methods, warm and cold cache; its memory
against dense (model, buffers held, pages touched per call, whole process); its scaling from T = 1 to T = 8. It does
not measure real models, d >= 1024, densities other than about 10 %, other processors or training.

**Reading rule of the tool.** A cell (test layer, N) counts as "HFFN faster" if the ratio best dense / HFFN is above 1
in **each** session; it is "stable" if (max - min) / median <= 10 % over the sessions; an unstable cell is reported
with its interval and supports no statement.

**Structure.** A session is a process; ratios are taken inside a process.

| Folder of `results/<tag>/` | Contents |
|---|---|
| `s1`, `s2`, `s3` | three sessions: `threads.py`, `routes.py`, `vs_dense.py --phase perf` (HFFN at T, dense at T threads, warm and cold cache) |
| `dense4`, `dense1` | dense at 4 then 1 thread, HFFN unchanged, warm cache: 8 threads are not necessarily the best setting of dense, especially at small N |
| `t4` | HFFN at 4 threads against dense at 4 threads |
| `once` | memory against dense, runtime memory, process footprint |
| `AGGREGATE.md` | ratios as median [min-max] over the sessions, stable cells, smallest ratio over the dense thread counts |
| `manifest.json`, `EXIT`, `steps.tsv`, `*.log`, `topology.txt` | status, exit code, duration and code of each step, logs, machine; the source is recorded as the hashes of `python -m hffn.provenance` |

**What disqualifies a run.** A failed step (`status: failed`) or a step exceeding its cap (`EXIT=124`, `status: partial`):
no performance table is then written. A gap to the float64 oracle above 2e-5 fails the step concerned. A fraction of
time stolen by the hypervisor (`cpu_steal_fraction` of the manifest) above 0.5 % makes the times suspect: it is
reported with the results.

**Launch** (in `tmux`, to survive an SSH cut):

```bash
bench/vm_setup.sh && . .venv/bin/activate     # once: g++, venv, numpy, scipy, pytest, torch CPU
bench/campaign.sh check                        # compilation, tests, parity
bench/campaign.sh smoke                        # each step once, reduced arguments
bench/campaign.sh run <tag>                    # the campaign, about 20 min on 8 cores
```

`smoke` never triggers `run`. An existing results folder is never rewritten.

**Limits of the tool.**

- The inputs `x` and the buffers of `dense_numpy` keep numpy's 16-byte alignment; PyTorch aligns its tensors on
  64 bytes, like the HFFN work buffers. The effect on `dense_numpy` is not measured.
- The sessions at 4 and 1 dense thread are single: their ratio has no interval.
- `-mavx2 -mfma` is the reference configuration. `-march=native` (AVX-512) is not part of it.

## Results

See [RESULTS.md](RESULTS.md): the published run of this leaf, its figures and its limits, and [`../../../../RESULTS.md`](../../../../RESULTS.md) for the protocol, the noise band and the limits that apply to every leaf. The ratios hold for that machine and those cells; they are not general speed-ups.

## Known limits

- **Measurements**: one machine, one boot, isolated FFN cells only. Nothing is validated with clang, on an Intel
  processor, or with AVX-512 with this code; the test layers are synthetic with d <= 512 and about 10 % density.
- **Pool**: after a long idle period, a call pays for waking the threads and for the cores leaving idle (time
  about doubled for 1 to 2 ms).
- **Threads**: partitions are balanced by number of neurons (up) and of edges (output): on these layers, the
  most loaded group exceeds the mean by at most 5 % at T = 8. A real support whose degrees depend on the neuron
  index would need a partition by cost.
- **Model**: the inverted scheme is half of the model read by the recommended routes. It holds, per edge, a 2-byte
  hidden index and the 4-byte down weight in output order (6 bytes). The same weights stored per neuron are only
  used by the `baseline` route; a layer holds the per-neuron order only, and the inverted scheme is built at load
  time.
- **GELU**: glibc returns different bits for scalar `erff` and for vector `erff` (libmvec) on 31 % of float
  values (exhaustive test on glibc 2.39, not repeated by the tests of this folder). AVX2 builds use the 8-way
  version everywhere; their bits therefore differ from those of a scalar-`erf` build, within the tolerance.
  Outside glibc, or without AVX2, the GELU goes back through scalar `erf` and independence from position is no
  longer guaranteed.
- **`baseline` route**: a reference route, slower than `dual` on the cases it was compared on. Part of the gap is
  attributed to accumulation in memory (one read and one write per edge); the rest is not isolated (hypothesis).
- **Not explored**: hand-written AVX-512 path, AMX, BF16 and INT8 (outside the 2e-5 tolerance), T(N) policy,
  output provided by the caller (`out=`).

## Provenance

`reference/sources.json` gives the origin and the SHA-256 of the kernel and the list of changes since the kernel
it derives from. The runtime was ported to x86_64 from the HFFN compact kernel of the Apple M1 CPU runtime.
`python -B -m hffn.provenance` prints the tree digest, the kernel hash and the number of files; on this version it
prints `tree_sha256=283b5dbfffef15c4ea6eaecea525cc39183c56b157f8a9609093653e944b775c` (35 files).

## License

MIT, like the original kernels. The license is the file `LICENSE` at the root of the `runtime/` folder (`../../../../LICENSE` from this folder); it covers the whole `runtime/` tree: the kernel, this leaf, the shared Pythia level and the x86 LLM inference leaf.
