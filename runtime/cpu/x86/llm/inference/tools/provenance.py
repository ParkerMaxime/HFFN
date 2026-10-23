"""Hashes that identify the sources of a run, without git: the kernel, the shared ``hffn_pythia`` package and
every source file of this leaf.

    python -B tools/provenance.py            one line: tree digest, kernel hash, number of files
    python -B tools/provenance.py --files    the hash of each file, one per line

The tree digest covers the kernel source (``common/kernels``, found as the first ancestor that has it), the files
of ``common/pythia/hffn_pythia`` (found the same way, token ids included), and, in this leaf, ``hffn_pythia_x86/``,
``checks/``, ``bench/``, ``tests/``, ``sources/``, ``tools/``, ``check.sh``, ``requirements.txt`` and
``pyproject.toml``. The documentation (README, RESULTS), LICENSE, build products, caches, runs and results are
left out, so that filling the results into RESULTS.md does not change the digest. Standard library only.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "hffn_compact_g4_inference.cpp"
FOLDERS = ("hffn_pythia_x86", "checks", "bench", "tests", "sources", "tools")
FILES = ("check.sh", "requirements.txt", "pyproject.toml")
SKIP = {"__pycache__", "build", "runs", "results", ".pytest_cache", ".venv"}


def ancestor(*parts: str) -> Path:
    """``parts`` under the first ancestor of this leaf that has them."""
    for folder in ROOT.parents:
        candidate = folder.joinpath(*parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("/".join(parts))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files() -> dict[str, str]:
    """Name -> SHA-256 of the kernel, of the shared package and of the source files of the leaf."""
    found = {f"kernel/{KERNEL}": sha256(ancestor("common", "kernels", KERNEL))}
    shared = ancestor("common", "pythia", "hffn_pythia")
    for path in sorted(p for p in shared.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(shared).parts)):
        found[f"pythia/{path.relative_to(shared).as_posix()}"] = sha256(path)
    paths = [ROOT / name for name in FILES]
    for folder in FOLDERS:
        paths += [p for p in (ROOT / folder).rglob("*") if p.is_file() and not SKIP & set(p.relative_to(ROOT).parts)]
    for path in sorted(paths):
        found[f"leaf/{path.relative_to(ROOT).as_posix()}"] = sha256(path)
    return found


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{name} {value}\n" for name, value in sorted(hashes.items())).encode()).hexdigest()


def record() -> str:
    hashes = files()
    return f"tree_sha256={digest(hashes)} kernel_sha256={hashes[f'kernel/{KERNEL}']} files={len(hashes)}"


if __name__ == "__main__":
    if "--files" in sys.argv:
        print("\n".join(f"{value}  {name}" for name, value in files().items()))
    else:
        print(record())
