"""Hashes that identify the sources of a run, without git: shared packages found above a leaf, and the source files
of the leaf. Standard library only: no GPU, no compilation. Each leaf states what it covers in its
``tools/provenance.py``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SKIP = {"__pycache__", "runs", "results", ".venv"}


def ancestor(root: Path, *parts: str) -> Path:
    """``parts`` under the first ancestor of ``root`` that has them."""
    for folder in root.parents:
        candidate = folder.joinpath(*parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("/".join(parts))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root: Path, shared: dict[str, tuple[str, ...]], folders: tuple[str, ...], names: tuple[str, ...]) -> dict[str, str]:
    """Key -> SHA-256: ``<label>/<path>`` for every file of each shared folder (``label -> parts`` under an ancestor of
    ``root``), and ``leaf/<path>`` for the files ``names`` and the files of ``folders`` of the leaf."""
    found = {}
    for label, parts in shared.items():
        base = ancestor(root, *parts)
        for path in sorted(p for p in base.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(base).parts)):
            found[f"{label}/{path.relative_to(base).as_posix()}"] = sha256(path)
    paths = [root / name for name in names]
    for folder in folders:
        paths += [p for p in (root / folder).rglob("*") if p.is_file() and not SKIP & set(p.relative_to(root).parts)]
    for path in sorted(paths):
        found[f"leaf/{path.relative_to(root).as_posix()}"] = sha256(path)
    return found


def digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{name} {value}\n" for name, value in sorted(hashes.items())).encode()).hexdigest()


def main(argv: list[str], hashes: dict[str, str]) -> None:
    if "--files" in argv:
        print("\n".join(f"{value}  {name}" for name, value in hashes.items()))
    else:
        print(f"tree_sha256={digest(hashes)} files={len(hashes)}")
