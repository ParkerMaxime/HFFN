"""HFFN (hidden-FFN) CPU inference runtime for x86_64."""
import sys
from pathlib import Path

# The shared test layers (`hffn_ffn_layers`) live in the `common/ffn` of the first level above this leaf that has them.
_shared = next(p / "common" / "ffn" for p in Path(__file__).resolve().parents if (p / "common" / "ffn" / "hffn_ffn_layers").is_dir())
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))
