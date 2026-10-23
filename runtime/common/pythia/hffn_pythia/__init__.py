"""What every Pythia inference leaf shares, whatever the hardware: the export reader, the two-arm loader
(dense and HFFN), the end-to-end token check, its prompt tokens and the source verification.

A leaf puts the folder that holds this package on ``sys.path`` (see its own ``__init__``) and gives the loader
the constructor of its HFFN runtime.
"""
from pathlib import Path

TOKENS = Path(__file__).with_name("wikitext2_test_tokens.json")
