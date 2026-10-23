"""Request-level timing harness of the Nvidia LLM leaf: the shared harness (`hffn_pythia.bench_request`) with
what is specific to CUDA. Protocol, cells, gates and the A/A criterion: README.md of the leaf, "Timing harness".

    python -B bench_request.py --mode check|smoke|run --models <base checkpoints dir> --exports <exports dir> [--out <run folder>]
"""
import subprocess
import sys
from pathlib import Path

LEAF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEAF))

import hffn_pythia_cuda  # noqa: E402  (puts the shared package on sys.path)
from hffn_pythia import bench_request  # noqa: E402
import hffn_cuda  # noqa: E402  (the kernels and the runtime, hashed in the provenance)


def state():
    query = 'name,temperature.gpu,utilization.gpu,memory.used,clocks.current.graphics,clocks_throttle_reasons.active'
    result = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {'error': result.stderr[-200:]}
    values = dict(zip(query.split(','), (part.strip() for part in result.stdout.strip().split(','))))
    return {key: values[key] for key in ('temperature.gpu', 'clocks.current.graphics', 'clocks_throttle_reasons.active',
                                         'utilization.gpu', 'memory.used', 'name')}


def synchronize():
    import torch
    torch.cuda.synchronize()


def versions():
    import cupy
    import torch
    import transformers
    return f'python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} cupy {cupy.__version__} transformers {transformers.__version__}'


def describe(model):
    import torch
    runtime = model.hffn_runtime
    return {'kernels': {f'{up}/{down}': count for (up, down), count in sorted(runtime.calls.items())},
            'workspace_bytes': runtime.workspace_bytes(), 'max_memory_allocated': torch.cuda.max_memory_allocated()}


ADAPTER = bench_request.Adapter(
    name='Nvidia LLM leaf', load_pythia=hffn_pythia_cuda.load_pythia, device='cuda', synchronize=synchronize, state=state,
    versions=versions, describe=describe, runtime_files=sorted(Path(hffn_cuda.__file__).parent.glob('*.py')),
    note='shared cloud GPU: ratios hold within this execution only')

if __name__ == '__main__':
    sys.exit(bench_request.main(ADAPTER, __file__))
