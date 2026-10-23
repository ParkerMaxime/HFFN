# Shared FFN level

What every isolated-FFN leaf shares, whatever the hardware. Imported as `hffn_ffn_layers` (a leaf puts `common/ffn` on
`sys.path`). NumPy only, except `dense.py`.

| File | Role |
|---|---|
| `hffn_ffn_layers/synth.py` | `CompactLayer` (the canonical hidden-major compact layout: group boundaries, up and down offsets, 16-bit indices, float32 values and biases) and the seeded generator of synthetic layers (NumPy PCG64). Nothing is stored: every array is regenerated from the case and its seed. |
| `hffn_ffn_layers/layers.py` | the test layers, `load(name)` and `make_input(case, n)`; a layer or an input whose fingerprint differs from the pinned one is refused. `python -c "import hffn_ffn_layers.layers as l; l.main()"` checks them all. |
| `reference/fingerprints.json` | the pinned SHA-256 of the test layers and of their benchmark inputs |
| `hffn_ffn_layers/reference.py` | the float64 oracle: it walks the same hidden-major support as the kernels and never builds `W_up` or `W_down`. A test oracle, not a runtime. |
| `hffn_ffn_layers/dense.py` | the dense iso-support FFN of the checks and benchmarks (`F.linear`, exact GELU, float32). The only file that imports PyTorch. Not part of any runtime. |

Test layers: `homog_256_1024` and `homog_512_2048` (homogeneous supports, degree 26 and 51) and `irreg_512_2048`
(irregular support, exact offsets), about 10 % density.
