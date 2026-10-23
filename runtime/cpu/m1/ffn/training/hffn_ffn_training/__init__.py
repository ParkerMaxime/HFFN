"""One FFN training step on the M1 CPU: the forward is the dual entry of the compact G4 inference kernel of
``runtime/cpu/m1/llm/inference/compact`` with ``z`` stored; backward and AdamW are those of the G4 training kernel of
``runtime/cpu/common/llm/training``, unchanged."""
from pathlib import Path

# layer.py and step.py exist once for x86 and M1, at the shared level: they join this package by its path.
__path__.append(str(next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
                    / "runtime/cpu/common/ffn/training/hffn_ffn_training"))
