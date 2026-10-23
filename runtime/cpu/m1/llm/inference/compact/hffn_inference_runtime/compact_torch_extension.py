"""Build/load the inference extension, which exposes forward symbols only."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import include_paths, library_paths, load


def _manual(name: str, root: Path, build: Path):
    lib = build / f"{name}.so"
    sources = [root / "torch_extension" / "hffn_compact_inference_torch_binding.cpp", root / "native" / "hffn_compact_g4_inference.cpp"]
    if not lib.exists() or any(source.stat().st_mtime > lib.stat().st_mtime for source in sources):
        command = ["clang++", "-O3", "-ffast-math", "-std=c++17", "-fPIC", "-dynamiclib", "-pthread", f"-DTORCH_EXTENSION_NAME={name}"]
        command += [f"-I{path}" for path in include_paths()] + [f"-I{sysconfig.get_paths()['include']}"] + [str(source) for source in sources]
        command += [f"-L{path}" for path in library_paths()] + ["-ltorch", "-ltorch_cpu", "-lc10", "-undefined", "dynamic_lookup"]
        command += [f"-Wl,-rpath,{path}" for path in library_paths()] + ["-o", str(lib)]
        subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location(name, lib)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_hffn_compact_inference_extension(build_dir: Path):
    root = Path(__file__).resolve().parents[1]
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CXX", "clang++")
    name = "hffn_compact_g4_inference_cpu_v1"
    sources = [str(root / "torch_extension" / "hffn_compact_inference_torch_binding.cpp"), str(root / "native" / "hffn_compact_g4_inference.cpp")]
    try:
        return load(name=name, sources=sources, extra_cflags=["-O3", "-ffast-math", "-std=c++17", "-pthread"], extra_ldflags=["-pthread"], build_directory=str(build_dir), verbose=False)
    except RuntimeError as exc:
        if "Ninja is required" not in str(exc):
            raise
        return _manual(name, root, build_dir)
