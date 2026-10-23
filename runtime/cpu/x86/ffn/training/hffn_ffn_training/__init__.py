"""One FFN training step on x86: the forward is the dual route of the shared x86 inference kernel with ``z`` stored;
backward and AdamW are those of the G4 training kernel of ``runtime/cpu/common/llm/training``, unchanged."""
import sys
from pathlib import Path

_CPU = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/cpu"
# layer.py and step.py exist once for x86 and M1, at the shared level: they join this package by its path.
__path__.append(str(_CPU / "common/ffn/training/hffn_ffn_training"))
# `hffn_x86_common` (the inference kernel of x86, its build and its binding).
if str(_CPU / "x86/common") not in sys.path:
    sys.path.insert(0, str(_CPU / "x86/common"))
