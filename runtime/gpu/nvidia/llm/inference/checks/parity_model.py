"""End-to-end token parity of the CUDA HFFN arm against the dense arm: the shared check, with this leaf's loader.

    python -B -m checks.parity_model --model-dir <base checkpoint> --export-dir <export>
"""
import sys

from hffn_pythia_cuda import load_pythia
from hffn_pythia import parity_model


def kernels(model) -> str:
    return "kernels " + ", ".join(f"{up}/{down} x{calls}" for (up, down), calls in sorted(model.hffn_runtime.calls.items()))


if __name__ == "__main__":
    sys.exit(parity_model.main(load_pythia, device="cuda", describe=kernels))
