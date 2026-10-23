"""End-to-end token parity of the x86 HFFN arm against the dense arm: the shared check, with this leaf's loader.

    python -B -m checks.parity_model --model-dir <base checkpoint> --export-dir <export> [--threads T]
"""
import sys

from hffn_pythia_x86 import load_pythia
from hffn_pythia import parity_model

if __name__ == "__main__":
    sys.exit(parity_model.main(load_pythia, options=(("--threads", dict(type=int, default=None)),)))
