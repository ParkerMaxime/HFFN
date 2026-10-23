"""Metal FFN of the Apple GPU leaves: the package `hffn` holds the Metal source generators (`kernels.py`), the
hidden-major banks and their device placement (`bank_store.py`), the FFN call (`ffn.py`), the aligned layout
(`aligned.py`) and the saved-forward entry of the training leaf (`training_entry.py`).

The LLM inference leaf (`../pythia_torch_attention`), the isolated FFN leaf (`../../../ffn/inference`) and the LLM
training leaf (`../../training`) put the folder that holds this package on `sys.path` and import `native_transformer.hffn`.
"""
