"""Memory of the two arms of one export, one fresh process per arm: the exact bytes of the FFNs, and the memory
of the GPU while the process serves requests.

For each export (size, density) and each arm (dense, hffn), a fresh process loads the model of the arm on the GPU,
then serves one prefill (batch 4, 128 tokens) and one decode (batch 4, prompt of 32 tokens, 16 cached steps).
``ffn_bytes`` is the exact size of the FFN arrays the arm reads (dense: the weights and biases of the two linear
layers of every block; hffn: the compact arrays of every layer, ``HFFNLayer.model_bytes``). The GPU memory is the
one PyTorch's allocator holds for tensors (``allocated``), its peak, and what it reserved from the driver; the
compact arrays and the work buffers of the HFFN runtime are PyTorch tensors, so they are counted there, and the
bytes of CuPy's own pool are recorded beside them. The dense FFNs that the shared loader builds for the hffn arm are
replaced on the host before the model moves to the GPU, so they never reach it: the peak is reached while serving.
Memory held by the CUDA context itself is not measured.

    python -B bench/memory.py --out <json> [--models <dir> --exports <dir>] [--sizes 70,160,410] [--densities 10,20,50,80]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LEAF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEAF))

import hffn_pythia_cuda  # noqa: E402  (puts the shared package on sys.path)
from hffn_pythia.assets import exports_dir, models_dir  # noqa: E402

PREFILL, DECODE, STEPS = (4, 128), (4, 32), 16


def gpu_bytes() -> dict:
    import cupy
    import torch
    torch.cuda.synchronize()
    return {'allocated': torch.cuda.memory_allocated(), 'peak_allocated': torch.cuda.max_memory_allocated(),
            'reserved': torch.cuda.memory_reserved(), 'cupy_pool': cupy.get_default_memory_pool().total_bytes()}


def worker(job: dict) -> int:
    import torch
    from hffn_pythia import TOKENS
    folder = Path(job['exports']) / f"pythia{job['size']}m_sparse{job['density']}_ffn_only" / 'export'
    model = hffn_pythia_cuda.load_pythia(Path(job['models']) / f"pythia{job['size']}m", folder, job['arm'])
    loaded = gpu_bytes()
    blocks = model.gpt_neox.layers
    if job['arm'] == 'dense':
        ffn_bytes = sum(p.numel() * p.element_size() for block in blocks for p in block.mlp.parameters())
    else:
        ffn_bytes = sum(layer.model_bytes() for layer in model.hffn_runtime.layers)
    stream = torch.tensor(json.loads(TOKENS.read_text())['tokens'], device='cuda')
    with torch.inference_mode():
        batch, length = PREFILL
        model(input_ids=stream[:batch * length].view(batch, length), use_cache=True)
        prefilled = gpu_bytes()
        batch, length = DECODE
        out = model(input_ids=stream[:batch * length].view(batch, length), use_cache=True)
        for _ in range(STEPS):
            out = model(input_ids=out.logits[:, -1:].argmax(-1), past_key_values=out.past_key_values, use_cache=True)
        del out
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    print(json.dumps({'size': job['size'], 'density': job['density'], 'arm': job['arm'], 'ffn_bytes': ffn_bytes, 'parameter_bytes': total,
                      'workspace_bytes': model.hffn_runtime.workspace_bytes() if job['arm'] == 'hffn' else 0,
                      'gpu_bytes': {'after_load': loaded, 'after_prefill': prefilled, 'after_decode': gpu_bytes()}}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--exports', type=Path)
    parser.add_argument('--sizes', default='70,160,410')
    parser.add_argument('--densities', default='10,20,50,80')
    args = parser.parse_args()
    if args.worker:
        return worker(json.loads(args.worker))
    models, exports = args.models or models_dir(), args.exports or exports_dir()
    rows = []
    for size in (int(v) for v in args.sizes.split(',')):
        for density in (int(v) for v in args.densities.split(',')):
            for arm in ('dense', 'hffn'):
                job = {'size': size, 'density': density, 'arm': arm, 'models': str(models), 'exports': str(exports)}
                result = subprocess.run([sys.executable, '-B', '-W', 'ignore', __file__, '--worker', json.dumps(job)], text=True, capture_output=True)
                lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
                if result.returncode != 0 or not lines:
                    print(f'failed: {job}\n{result.stderr[-2000:]}', file=sys.stderr)
                    return 1
                rows.append(json.loads(lines[-1]))
                gpu = rows[-1]['gpu_bytes']['after_decode']
                print(f"pythia{size}m d{density} {arm}: ffn {rows[-1]['ffn_bytes'] / 1e6:.1f} MB, GPU allocated after decode {gpu['allocated'] / 2**20:.0f} MiB, "
                      f"peak {gpu['peak_allocated'] / 2**20:.0f} MiB, reserved {gpu['reserved'] / 2**20:.0f} MiB", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'prefill': PREFILL, 'decode': DECODE, 'decode_steps': STEPS, 'rows': rows}, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
