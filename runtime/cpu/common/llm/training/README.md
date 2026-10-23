# CPU shared level: LLM training

What the M1 and the x86 LLM training leaves share (and, through `Step`, the FFN training leaves).

| Folder | Role |
|---|---|
| `native/hffn_compact_g4_training.cpp`, `native/g4_contract.h` | the G4 training kernel (backward and compact AdamW, hidden-owned below a block size and dual route above it); standard C++, one copy for both hardware |
| `torch_extension/hffn_compact_torch_binding.cpp` | its PyTorch binding |
| `hffn_training_runtime/` | `compact_export` (the export reader and the G4 and dual-route tables), `compact_runtime` (the runtime: forward through the sibling inference kernel, native backward, AdamW), `compact_model` and `compact_direct_loader` (Pythia integration and loader), `compact_checkpoint` (save and resume of parameters and optimiser state), `forward_route_check` (load-time structural checks of the tables handed to the inference kernel), `masked_reference` (the fixed-support dense control), `padded_export` and `padded_import` (cold import of the legacy padded checkpoint format, never on a training path), `compact_torch_extension` (the build, one module for x86 and M1) |
| `checks/` | the layer gate (forward and backward against the direct float64 equation, with a negative control), the model gate (distance of the first step to a float64 reference, relative to the dense float32 control) and the update gate (one AdamW step against its closed form); `check_core.sh` runs the three on every export |

Each leaf adds an entry file (`forward_entry.py`, `build_entry.py`) that binds this level to the inference kernel of its
hardware.
