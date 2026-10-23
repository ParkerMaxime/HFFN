"""Request-level timing harness of the x86 LLM leaf: the shared harness (`hffn_pythia.bench_request`) with what
is specific to a CPU. `--threads T` sets the threads of PyTorch and of HFFN (default: the CPUs of the process, at
most 8): both run on the same OpenMP team.

    python -B bench_request.py --mode check|smoke|run --models <base checkpoints dir> --exports <exports dir> [--out <run folder>] [--threads T]
"""
import os
import resource
import sys
from pathlib import Path

LEAF = Path(__file__).resolve().parents[1]
sys.path[0] = str(LEAF)      # the leaf folder replaces this one, so that `hffn_pythia_x86` is importable

import hffn_pythia_x86  # noqa: E402  (puts the shared package on sys.path)
from hffn_pythia import bench_request  # noqa: E402

_threads = None


def configure(job):
    """In the worker, before any model exists: the thread count of the cell, for PyTorch and for HFFN."""
    global _threads
    _threads = job.get('threads')
    if _threads:
        import torch
        torch.set_num_threads(_threads)


def load_pythia(model_dir, export_dir, ffn):
    return hffn_pythia_x86.load_pythia(model_dir, export_dir, ffn, _threads)


def state():
    """Load, affinity and the cumulative CPU counters (the stolen share of a run is their difference)."""
    user, nice, system, idle, iowait, irq, softirq, steal = (int(v) for v in Path('/proc/stat').read_text().split('\n')[0].split()[1:9])
    return {'loadavg_1min': os.getloadavg()[0], 'affinity': len(os.sched_getaffinity(0)), 'cpu_jiffies_total': user+nice+system+idle+iowait+irq+softirq+steal,
            'cpu_jiffies_steal': steal}


def versions():
    import torch
    import transformers
    return (f'python {sys.version.split()[0]} torch {torch.__version__} threads {torch.get_num_threads()} '
            f'transformers {transformers.__version__} OMP_NUM_THREADS {os.environ.get("OMP_NUM_THREADS")}')


def describe(model):
    import torch
    runtime = model.hffn_runtime
    return {'hffn_threads': runtime.threads, 'torch_threads': torch.get_num_threads(), 'workspace_bytes': runtime.workspace_bytes(),
            'kernel_ns': runtime.kernel_ns, 'max_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


ADAPTER = bench_request.Adapter(
    name='x86 LLM leaf', load_pythia=load_pythia, device='cpu', synchronize=lambda: None, state=state, versions=versions,
    describe=describe, configure=configure,
    runtime_files=[*sorted((LEAF/'hffn_pythia_x86').glob('*.py')), hffn_pythia_x86.build.SOURCE],
    note='shared cloud VM: ratios hold within this execution only')

if __name__ == '__main__':
    sys.exit(bench_request.main(ADAPTER, __file__))
