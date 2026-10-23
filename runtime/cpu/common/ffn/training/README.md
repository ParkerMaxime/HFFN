# CPU shared level: FFN training

One FFN training step, shared by the M1 and the x86 FFN training leaves (a leaf puts this folder on the package path and
provides its inference kernel through its own `hffn_ffn_training/kernels.py`).

| File | Role |
|---|---|
| `hffn_ffn_training/layer.py` | one layer of a compact `hffn_training_compact_g4_v2` export, its G4 partition and its reverse schedules (index tables over the one canonical storage; no copy of any value) |
| `hffn_ffn_training/step.py` | one training step on persistent buffers: forward on the dual route of the leaf's inference kernel with `z` stored, backward and AdamW on the G4 training kernel of `cpu/common/llm/training` |
| `hffn_ffn_training/module.py` | a PyTorch module on the step: its parameters are the compact values, its autograd function runs the native forward and backward |
| `hffn_ffn_training/contexts.py` | explicitly leased saved activations for several graphs in flight on one step |
| `hffn_ffn_training/checkpoint.py` | cold checkpoint interoperability (support, partition, values); never called by the forward or the backward |
| `checks/` | the batch run by the leaf's `check.sh`: numerical gates on every export and on the shared synthetic layers (gates 2 and 3), API capabilities, callers, controls, reader and a supervisor that writes the status and exit code of the run |

Nothing here is a runtime of its own: the forward is always the inference kernel of the hardware.
