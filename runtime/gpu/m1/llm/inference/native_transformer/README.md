# M1 GPU shared level: the Metal FFN runtime

The Metal FFN runtime used by the three M1 GPU leaves (FFN inference, LLM inference, LLM training). They put this folder on
`sys.path` and import `native_transformer.hffn`.

| File | Role |
|---|---|
| `hffn/kernels.py` | Metal source generators for the hidden-major FFN: up kernels (hidden owners) and down kernels (output owners), selected by the number of tokens `N` |
| `hffn/bank_store.py` | the hidden-major compact banks (validation and device placement); format id `e1b_hidden_banks_v1` |
| `hffn/ffn.py` | the FFN call: one up dispatch, then the down dispatch |
| `hffn/aligned.py` | activation row strides aligned on 64 bytes |
| `hffn/training_entry.py` | the one-bank forward on live compact values and caller buffers, used by the training leaf (saves `z`); buffers are borrowed until all queued GPU consumers finish |
| `hffn/stats.py` | runtime counters (what was executed, compiled, allocated and loaded) |

Naming limit: this runtime was written for models with several banks and routes, and keeps their vocabulary (the class
`MoEFFN`, route plans, the format id above). The three leaves here use it with **one bank and one route**; no other path is
published or tested.
