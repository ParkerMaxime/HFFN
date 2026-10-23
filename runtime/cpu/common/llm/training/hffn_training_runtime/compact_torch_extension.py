"""Build and load the compact-G4 native extension: one module for x86 and M1.

The leaf gives the module name, the compiler, the flags and the link arguments in its ``build_entry.py``; this module
issues the compile command, once per command and sources (the library lives in a folder named by their hash, so a change of
flags always rebuilds), and imports the result. ``TORCH_LOAD`` (M1 only) is an optional first attempt through
``torch.utils.cpp_extension.load``; it falls back to the command when Ninja is missing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import include_paths, library_paths, load

from .build_entry import COMPILER, FLAGS, LINK_ARGUMENTS, NAME, TORCH_LOAD

# ``load`` records the module as built before it finds Ninja missing, so a second call in the same process would import a
# library that the command path put elsewhere: once Ninja is known to be missing, the command path is used directly.
_NO_NINJA = False


def _sources() -> list[Path]:
    # One copy of the binding and of the kernel for x86 and M1, at the shared CPU level.
    common = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir()) / "runtime/cpu/common/llm/training"
    return [common / "torch_extension" / "hffn_compact_torch_binding.cpp", common / "native" / "hffn_compact_g4_training.cpp"]


def _headers() -> list[Path]:
    # Read by both sources, so part of the cache tag; never passed to the compiler.
    return [_sources()[1].parent / "g4_contract.h"]


def _import(library: Path):
    spec = importlib.util.spec_from_file_location(NAME, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_command(sources: list[Path]) -> list[str]:
    """The compile command, output path apart."""
    return ([COMPILER, *FLAGS, f"-DTORCH_EXTENSION_NAME={NAME}"] + [f"-I{path}" for path in include_paths()]
            + [f"-I{sysconfig.get_paths()['include']}"] + [str(path) for path in sources]
            + [f"-L{path}" for path in library_paths()] + list(LINK_ARGUMENTS)
            + [f"-Wl,-rpath,{path}" for path in library_paths()])


def load_hffn_compact_torch_extension(build_dir: Path):
    """Compile the binding and the kernel into ``build_dir`` (once per command and sources) and import the module."""
    sources = _sources()
    global _NO_NINJA
    if TORCH_LOAD is not None and not _NO_NINJA:
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CXX", COMPILER)
        try:
            return load(name=NAME, sources=[str(path) for path in sources], extra_cflags=TORCH_LOAD["extra_cflags"],
                        extra_ldflags=TORCH_LOAD["extra_ldflags"], build_directory=str(build_dir), verbose=False)
        except RuntimeError as exc:
            if "Ninja is required" not in str(exc):
                raise
            _NO_NINJA = True
    command = compile_command(sources)
    tag = hashlib.sha256((" ".join(command) + "".join(path.read_text() for path in [*sources, *_headers()])).encode()).hexdigest()[:12]
    folder = build_dir / tag
    library = folder / f"{NAME}.so"
    if not library.exists():
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run([*command, "-o", str(library)], check=True)
    return _import(library)
