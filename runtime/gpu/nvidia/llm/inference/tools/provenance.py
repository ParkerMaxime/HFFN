"""Hashes that identify the sources of a run, without git: the shared ``hffn_pythia`` package, the shared CUDA kernels
and every source file of this leaf.

    python -B tools/provenance.py            one line: tree digest, number of files
    python -B tools/provenance.py --files    the hash of each file, one per line

The tree digest covers the files of ``common/pythia/hffn_pythia`` (token ids included) and of
``common/kernels/hffn_cuda`` (the CUDA kernels and the runtime), each found as the first ancestor folder that has it,
and, in this leaf, ``hffn_pythia_cuda/``, ``checks/``, ``bench/``, ``sources/``, ``tools/``, ``check.sh``,
``launch.sh`` and ``requirements.txt``. The documentation (README, RESULTS), LICENSE, caches, runs and results are
left out, so that filling the results into RESULTS.md does not change the digest. Standard library only: no GPU, no
compilation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(next(p / "common" / "kernels" for p in ROOT.parents if (p / "common" / "kernels" / "hffn_cuda").is_dir())))

from hffn_cuda import provenance  # noqa: E402

SHARED = {"pythia": ("common", "pythia", "hffn_pythia"), "kernels": ("common", "kernels", "hffn_cuda")}
FOLDERS = ("hffn_pythia_cuda", "checks", "bench", "sources", "tools")
FILES = ("check.sh", "launch.sh", "requirements.txt")


def files() -> dict[str, str]:
    return provenance.files(ROOT, SHARED, FOLDERS, FILES)


def record() -> str:
    hashes = files()
    return f"tree_sha256={provenance.digest(hashes)} files={len(hashes)}"


if __name__ == "__main__":
    provenance.main(sys.argv, files())
