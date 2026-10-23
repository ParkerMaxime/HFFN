"""Hashes that identify the sources of a run, without git: the kernel, the shared test layers and every file of this leaf.

    python -m hffn.provenance            one line: tree digest, kernel hash, number of files
    python -m hffn.provenance --files    the hash of each file, one per line

The tree digest covers the kernel source, the shared test layers (``common/ffn``: ``hffn_ffn_layers/`` and
``reference/``, keys ``ffn/...``) and the files of ``hffn/``, ``bench/``, ``tests/``, ``reference/`` and
``pyproject.toml`` and ``requirements.txt``; the documentation (README, RESULTS), build products, caches and results are left out, so
that filling the results into RESULTS.md does not change the digest.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from hffn_ffn_layers import layers as shared

from . import build

ROOT = build.ROOT
FOLDERS = ("hffn", "bench", "tests", "reference")
SHARED_FOLDERS = ("hffn_ffn_layers", "reference")
SKIP = {"__pycache__", "build", "results", ".pytest_cache", ".venv"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path, folders: tuple[str, ...]) -> list[Path]:
    return [p for folder in folders for p in (root / folder).rglob("*") if p.is_file() and not SKIP & set(p.relative_to(root).parts)]


def files() -> dict[str, str]:
    """Relative name -> SHA-256 of the kernel, of the shared test layers and of the source files of the leaf."""
    found = {f"kernel/{build.COMPACT}": sha256(build.source())}
    for path in sorted(_tree(shared.ROOT, SHARED_FOLDERS)):
        found[f"ffn/{path.relative_to(shared.ROOT).as_posix()}"] = sha256(path)
    for path in sorted([ROOT / "pyproject.toml", ROOT / "requirements.txt"] + _tree(ROOT, FOLDERS)):
        found[path.relative_to(ROOT).as_posix()] = sha256(path)
    return found


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{name} {value}\n" for name, value in sorted(hashes.items())).encode()).hexdigest()


def record() -> str:
    hashes = files()
    return f"tree_sha256={digest(hashes)} kernel_sha256={hashes[f'kernel/{build.COMPACT}']} files={len(hashes)}"


if __name__ == "__main__":
    if "--files" in sys.argv:
        print("\n".join(f"{value}  {name}" for name, value in files().items()))
    else:
        print(record())
