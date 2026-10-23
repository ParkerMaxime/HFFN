"""Shared setup of the benchmarks: CPU pinning, thread counts, machine record.

Import this module FIRST in a benchmark script: ``--dense-threads`` must reach OpenMP,
OpenBLAS and MKL through the environment before numpy or PyTorch are imported, and
``--cpus`` pins the process before any thread exists (threads inherit the affinity).

    --cpus 0-7          pin the process (HFFN's threads, PyTorch, OpenBLAS) to these CPUs
    --threads 8         threads of one HFFN compact call (default: CPUs of the affinity mask,
                        at most 8; see hffn.compact.default_threads)
    --dense-threads 8   threads of the dense references (default: the HFFN thread count);
                        "all" = every CPU of the affinity mask
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _option(name: str) -> str | None:
    """Value of the last occurrence of an option (as argparse does)."""
    value = None
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def parse_cpus(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus += range(int(first), int(last or first) + 1)
    return cpus


CPUS = _option("--cpus")
if CPUS:
    os.sched_setaffinity(0, parse_cpus(CPUS))
_hffn_threads = _option("--threads")
HFFN_THREADS = int(_hffn_threads) if _hffn_threads else max(1, min(len(os.sched_getaffinity(0)), 8))
_threads = _option("--dense-threads") or str(HFFN_THREADS)
DENSE_THREADS = len(os.sched_getaffinity(0)) if _threads == "all" else int(_threads)
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(DENSE_THREADS)


def add_arguments(parser) -> None:
    """Declare the options read above (so that --help lists them and argparse accepts them)."""
    parser.add_argument("--cpus", default=None, help="pin the process to these CPUs, e.g. 0-7 or 0,2,4,6")
    parser.add_argument("--threads", type=int, default=None, help="HFFN compact threads (default: affinity CPUs, at most 8)")
    parser.add_argument("--dense-threads", default=None, help='dense reference threads (default: the HFFN thread count; "all")')


def compact_kernel(layer, hidden_owned: bool = True):
    """CompactKernel on --threads threads."""
    from hffn.compact import CompactKernel

    return CompactKernel(layer, threads=HFFN_THREADS, hidden_owned=hidden_owned)


def set_torch_threads() -> None:
    import torch

    torch.set_num_threads(DENSE_THREADS)


def _lscpu() -> dict:
    try:
        output = subprocess.run(["lscpu"], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout
    except FileNotFoundError:
        return {}
    keep = ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "NUMA node(s)",
            "L1d cache", "L2 cache", "L3 cache", "Hypervisor vendor", "CPU max MHz")
    return {key: value.strip() for key, _, value in (line.partition(":") for line in output.splitlines()) if key in keep}


def _cpu_flags() -> list[str]:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            present = set(line.split(":", 1)[1].split())
            wanted = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512vl", "avx512bw", "avx512_bf16", "avx512_fp16", "amx_tile")
            return [flag for flag in wanted if flag in present]
    return []


def _governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text().strip() if path.exists() else None


def machine() -> dict:
    """Everything needed to interpret a result file: CPU, topology, pinning, toolchain, sources."""
    from hffn import build

    record = {
        "machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
        "lscpu": _lscpu(), "cpu_flags": _cpu_flags(), "governor": _governor(),
        "affinity": sorted(os.sched_getaffinity(0)), "hffn_threads": HFFN_THREADS, "dense_threads": DENSE_THREADS,
        "compiler": subprocess.run([build.compiler(), "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "hffn_flags": build.BASE_FLAGS + build.extra_flags(),
        "env": {key: os.environ.get(key) for key in ("HFFN_CXXFLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ATEN_CPU_CAPABILITY")},
        "kernel_sha256": {build.source().name: hashlib.sha256(build.source().read_bytes()).hexdigest()},
    }
    try:
        import numpy
        record["numpy"] = numpy.__version__
    except ImportError:
        pass
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        record["torch"] = {"version": torch.__version__, "threads": torch.get_num_threads(),
                           "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    return record


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
