# x86 CPU shared level

The x86_64 inference kernel and its Python build and binding, shared by the four x86 leaves (FFN inference, FFN training, LLM
inference, LLM training).

| File | Role |
|---|---|
| `kernels/hffn_compact_g4_inference.cpp` | the HFFN compact kernel: vectorised over the token lanes (AVX2/FMA, scalar fallback), T threads, deterministic reductions, routes `decode`, `dual` and `baseline`, a slots entry and a pre-activation store used by the training step. Built with `-O3 -ffast-math`: parity is a tolerance, not a binary equality. |
| `kernels/hffn_compact_g4_inference_n1.cpp` | a one-token sidecar that includes the kernel source and exports only the entry that stores the pre-activation (the training step at N = 1) |
| `hffn_x86_common/inference.py` | build into `build/` (the library name carries the hash of the command and of the source) and the ctypes entries; imported as `hffn_x86_common` |

Requirements: Linux x86_64, glibc, `g++` 9 or later, AVX2 and FMA. The x86 FFN inference leaf documents the kernel in detail.
