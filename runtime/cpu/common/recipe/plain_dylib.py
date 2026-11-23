"""Compile a plain (torch-free) native library once per command and sources: the one build call shared by the CPU leaves.

The leaf gives the compiler, the flags, the library suffix, the source and the files the source includes; this module
issues the compile command and names the library ``<stem>.<tag>.<suffix>`` in ``build_dir``, where ``tag`` is the first
12 hex digits of the SHA-256 of the command text, the source text and the dependency texts (so a change of flags, of
source or of an included header always rebuilds). The command goes to a ``.<pid>.tmp`` file that is renamed at the end:
a build is atomic for the readers of the final name. No torch import. Not a package: a leaf loads this file by path from
the repository root, as it does ``compact_extension.py``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


def build(source: Path, build_dir: Path, compiler: str, flags, suffix: str, extra_flags=(), dependencies=()) -> Path:
    command = [compiler, *flags, *extra_flags, str(source)]
    content = source.read_text() + "".join(path.read_text() for path in dependencies)
    tag = hashlib.sha256((" ".join(command) + content).encode()).hexdigest()[:12]
    lib = build_dir / f"{source.stem}.{tag}.{suffix}"
    if not lib.exists():
        build_dir.mkdir(exist_ok=True)
        partial = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([*command, "-o", str(partial)], check=True)
        partial.rename(lib)
    return lib
