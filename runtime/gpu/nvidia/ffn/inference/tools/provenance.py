"""Hashes that identify the sources of a run, without git: the shared CUDA kernels, the shared test layers and every
source file of this leaf.

    python -B tools/provenance.py            one line: tree digest, number of files
    python -B tools/provenance.py --files    the hash of each file, one per line

The tree digest covers ``common/kernels/hffn_cuda`` and ``common/ffn`` (the test-layer package and its pinned
fingerprints), each found as the first ancestor folder that has it, and, in this leaf, ``hffn_ffn_cuda/``,
``checks/``, ``tools/``, ``check.sh``, ``launch.sh`` and ``requirements.txt``. Documentation, caches and runs are left
out. Standard library only: no GPU, no compilation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(next(p / "common" / "kernels" for p in ROOT.parents if (p / "common" / "kernels" / "hffn_cuda").is_dir())))

from hffn_cuda import provenance  # noqa: E402

SHARED = {"kernels": ("common", "kernels", "hffn_cuda"), "ffn": ("common", "ffn")}
FOLDERS = ("hffn_ffn_cuda", "checks", "tools")
FILES = ("check.sh", "launch.sh", "requirements.txt")


def files() -> dict[str, str]:
    return provenance.files(ROOT, SHARED, FOLDERS, FILES)


if __name__ == "__main__":
    provenance.main(sys.argv, files())
