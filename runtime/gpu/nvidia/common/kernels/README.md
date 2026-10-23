# Nvidia GPU shared level

The CUDA kernels and their host runtime, shared by the three Nvidia leaves (FFN inference, FFN training, LLM inference).

| File | Role |
|---|---|
| `hffn_cuda/kernels.py` | CUDA source of the forward kernels (up `coop`, `lane`, `tile`; down `coop`, `tiled`, `blocked`), generated once for the ordinary forward and for the forward that stores the pre-activation `z`; compiled at run time by NVRTC through CuPy |
| `hffn_cuda/runtime.py` | `HFFNLayer` and `HFFNRuntime`: one call runs a whole block of N tokens, an up kernel (hidden owners) then a down kernel (output owners), selected from N and `d_ff` |
| `hffn_cuda/parity.py` | the per-layer parity of a runtime against a float64 oracle (bit repeatability, every forced kernel pair), shared by the checks of the leaves |
| `hffn_cuda/provenance.py` | hashes that identify the sources of a run without git |

The kernels are a port of the Metal kernels of the M1 GPU runtime; their selection thresholds were first measured on an Apple
M1 and are provisional on CUDA (the Nvidia LLM inference README states them). Imported as `hffn_cuda`.
