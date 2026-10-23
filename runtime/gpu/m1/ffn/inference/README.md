# HFFN GPU runtime on Apple M1: isolated FFN inference

One exported FFN (or one synthetic FFN) executed by the Metal FFN kernel of
[`gpu/m1/llm/inference/native_transformer`](../../llm/inference/native_transformer/README.md): per hidden neuron its
active incoming connections, its bias and its active outgoing connections stay compact; `x`, `h` and `y` are dense. One call
processes the whole block of `N` tokens.

Results: [RESULTS.md](RESULTS.md).

## Scope

| | |
|---|---|
| Degree of optimisation | **the Metal FFN kernel of the M1 GPU shared level**, used here with one bank and one route: hidden-owned up kernel, output-owned down kernel, selected by `N`. No dedicated tuning for isolated FFNs. |
| Hardware | Apple M1 GPU through PyTorch MPS. Not tried on another Apple chip. |
| Models | Any FFN given by its compact tensors, with input, hidden layer and output of any widths (16-bit indices below 32 768, 32-bit from there), GELU (exact erf) or ReLU, float32. Measured on the exports `512_2048` at densities 5 to 100 % and `256_1024` and `1024_4096` at d10 (synthetic teacher-student FFNs on random supports, irregular degree per hidden neuron). |
| Results | [RESULTS.md](RESULTS.md). |

## Contents

```
homogeneous.py            FFNInference, OneBankStore, HomogeneousFFNInference, synthetic_ffn
smoke_homogeneous_ffn.py  the embedded check, against the dense fixed-support FFN in float64
```

The kernel is at the shared level (`native_transformer/hffn`), which this leaf joins by its path: use and copy the
`runtime/` tree.

## Use

```python
from homogeneous import FFNInference, HomogeneousFFNInference, OneBankStore, synthetic_ffn

# an exported FFN (HFFN_ASSETS/homogeneous/models/<cell>/export), activation read from the export or forced
ffn = HomogeneousFFNInference(export_dir)
relu = HomogeneousFFNInference(export_dir, 'relu')

# any FFN given by its compact tensors, with input and output of different widths
compact, dense = synthetic_ffn(128, 512, 256, density=0.2, seed=712)     # `dense` is a control, not a path
rect = FFNInference(OneBankStore(compact, 128, 'relu'))                   # x: [N, 128] -> y: [N, 256]
y = ffn.forward(x)                                                        # x: MPS float32 [N, d_model]
```

One call runs the whole block of tokens; its dense buffers are bounded by a workspace limit, 64 MiB by default (2 048
tokens at `d_ff` = 4 096), which the caller may raise: `HomogeneousFFNInference(export_dir, workspace_limit=512*1024**2)`.

## Checks

Without timing. Needs an Apple GPU, no CPU fallback (the script refuses `PYTORCH_ENABLE_MPS_FALLBACK`).

```bash
python smoke_homogeneous_ffn.py --synthetic 128 512 256 [--density 0.2] [--activation relu]        # no asset needed
HFFN_ASSETS=<assets> python smoke_homogeneous_ffn.py --model 256_1024 --n 1 4 128 [--activation relu]
HFFN_ASSETS=<assets> python smoke_homogeneous_ffn.py --model 512_2048 --n 8192 --workspace-mib 512
```

For each `N`, the output is compared with the dense fixed-support FFN computed in float64 on the CPU: maximum absolute
error at most 5e-4, relative L2 error at most 1e-5, a second call bitwise identical to the first, and no workspace growth
between the two. `--model` reads `$HFFN_ASSETS/homogeneous/models/<model>/export/hidden_ffn_format.pt` and, for the control,
`sparse_dense.pt` of the same folder. The script ends with exit code 0 when every `N` passes.

## Known limits

- **Depends on the M1 GPU shared level**: copy the `runtime/` tree, not the leaf alone.
- **One Apple M1**, passively cooled; the figures are not comparable with other hardware.
- **`delta` is large and reported, not applied** (RESULTS.md): the sign of several cells is not robust.
- No training on this leaf: FFN training on the M1 GPU is not published (see `RESULTS.md`, *Not covered*).
- The runtime keeps the vocabulary of a multi-bank design (`MoEFFN`, route plans, the bank format id
  `e1b_hidden_banks_v1`); here it always runs one bank and one route.

## Licence

MIT: `LICENSE` at the root of the `runtime/` tree.
