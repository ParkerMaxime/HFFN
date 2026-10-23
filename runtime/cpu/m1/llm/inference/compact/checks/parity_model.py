"""End-to-end token parity of the CPU M1 compact HFFN arm against the dense arm: the shared check, with this
capsule's loader. PyTorch runs on `--threads` threads (README: 4).

    python -B -m checks.parity_model --model-dir <base checkpoint> --export-dir <export> --threads 4
"""
import sys

import torch

from hffn_pythia_m1 import G4_GROUPS, load_pythia
from hffn_pythia import parity_model


def _loader(model_dir, export_dir, ffn, threads=None):
    if threads:
        import torch
        torch.set_num_threads(threads)
    return load_pythia(model_dir, export_dir, ffn)


if __name__ == "__main__":
    sys.exit(parity_model.main(_loader, options=(("--threads", dict(type=int, default=None)),),
                               describe=lambda model: f"torch threads {torch.get_num_threads()}, G4 groups {G4_GROUPS}"))
