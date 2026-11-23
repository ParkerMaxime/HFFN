"""Compile and load a compact-G4 native extension: the one recipe shared by the CPU leaves (x86 and M1).

A leaf gives the module name, the compiler, the flags, the link arguments, the optional ``TORCH_LOAD`` and the sources in
its own ``build_entry.py`` and façade; this module issues the compile command, once per command and sources (the library
lives in a folder named by their hash, so a change of flags always rebuilds), and imports the result. ``TORCH_LOAD`` (M1
only) is an optional first attempt through ``torch.utils.cpp_extension.load``; it falls back to the command when Ninja is
missing. Not a package: a leaf's façade loads this file by path (``load_by_path``), found from the repository root.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import subprocess
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import include_paths, library_paths, load

# ``load`` records the module as built before it finds Ninja missing, so a second call in the same process would import a
# library that the command path put elsewhere: once Ninja is known to be missing for a module, the command path is used directly.
_NO_NINJA: set[str] = set()


def _import(name: str, library: Path):
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_command(name: str, compiler: str, flags, link_arguments, sources: list[Path]) -> list[str]:
    """The compile command, output path apart."""
    return ([compiler, *flags, f"-DTORCH_EXTENSION_NAME={name}"] + [f"-I{path}" for path in include_paths()]
            + [f"-I{sysconfig.get_paths()['include']}"] + [str(path) for path in sources]
            + [f"-L{path}" for path in library_paths()] + list(link_arguments)
            + [f"-Wl,-rpath,{path}" for path in library_paths()])


def load_extension(build_dir: Path, name: str, compiler: str, flags, link_arguments, torch_load, sources: list[Path], headers: list[Path]):
    """Compile ``sources`` into ``build_dir`` (once per command and sources) and import the module."""
    build_dir.mkdir(parents=True, exist_ok=True)
    # Without Ninja, a process waiting on the baton of ``load`` imports a library that was never built; a build is not
    # atomic either. Processes of a check load at the same time, so they go through here one at a time.
    with open(build_dir / "load.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if torch_load is not None and name not in _NO_NINJA:
            os.environ.setdefault("CXX", compiler)
            try:
                return load(name=name, sources=[str(path) for path in sources], extra_cflags=torch_load["extra_cflags"],
                            extra_ldflags=torch_load["extra_ldflags"], build_directory=str(build_dir), verbose=False)
            except RuntimeError as exc:
                if "Ninja is required" not in str(exc):
                    raise
                _NO_NINJA.add(name)
        command = compile_command(name, compiler, flags, link_arguments, sources)
        tag = hashlib.sha256((" ".join(command) + "".join(path.read_text() for path in [*sources, *headers])).encode()).hexdigest()[:12]
        folder = build_dir / tag
        library = folder / f"{name}.so"
        if not library.exists():
            folder.mkdir(parents=True, exist_ok=True)
            subprocess.run([*command, "-o", str(library)], check=True)
        return _import(name, library)
