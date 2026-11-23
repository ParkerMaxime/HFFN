"""Build and load the compact-G4 native extension: one module for x86 and M1.

The leaf gives the module name, the compiler, the flags and the link arguments in its ``build_entry.py``; the recipe
(command, cache folder named by the hash of the command and the sources, ``TORCH_LOAD`` attempt, Ninja fallback) is the
shared ``runtime/cpu/common/recipe/compact_extension.py``, loaded by path from the repository root. This module
keeps the training sources and the public names of the training call.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .build_entry import COMPILER, FLAGS, LINK_ARGUMENTS, NAME, TORCH_LOAD

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())


def _recipe():
    key = "hffn_cpu_compact_extension"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _ROOT / "runtime/cpu/common/recipe/compact_extension.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


_BUILD = _recipe()


def _sources() -> list[Path]:
    # One copy of the binding and of the kernel for x86 and M1, at the shared CPU level.
    common = _ROOT / "runtime/cpu/common/llm/training"
    return [common / "torch_extension" / "hffn_compact_torch_binding.cpp", common / "native" / "hffn_compact_g4_training.cpp"]


def _headers() -> list[Path]:
    # Read by both sources, so part of the cache tag; never passed to the compiler.
    return [_sources()[1].parent / "g4_contract.h"]


def compile_command(sources: list[Path]) -> list[str]:
    """The compile command, output path apart."""
    return _BUILD.compile_command(NAME, COMPILER, FLAGS, LINK_ARGUMENTS, sources)


def load_hffn_compact_torch_extension(build_dir: Path):
    """Compile the binding and the kernel into ``build_dir`` (once per command and sources) and import the module."""
    return _BUILD.load_extension(build_dir, NAME, COMPILER, FLAGS, LINK_ARGUMENTS, TORCH_LOAD, _sources(), _headers())
