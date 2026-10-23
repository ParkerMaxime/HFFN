"""Memory of the two arms of one export, one fresh process per arm: the exact bytes of the FFNs, and the memory
of the process while it serves requests.

For each export (size, density) and each arm (dense, hffn), a fresh process imports PyTorch and the leaf, notes
its resident memory, loads the model of the arm, notes the resident memory and the peak, then serves one prefill
(batch 4, 128 tokens) and one decode (batch 4, prompt of 32 tokens, 16 cached steps) and notes both again.
``ffn_bytes`` is the exact size of the FFN arrays the arm reads (dense: the weights and biases of the two linear
layers of every block; hffn: the compact arrays of every layer, ``HFFNLayer.model_bytes``). The peak of the hffn
arm includes the dense FFNs that the shared loader builds before replacing them: it is a loading peak, not the
serving memory, which is the resident memory after the requests.

    python -B bench/memory.py --out <json> [--models <dir> --exports <dir>] [--sizes 70,160,410] [--densities 10,20,50,80] [--threads T]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LEAF = Path(__file__).resolve().parents[1]
sys.path[0] = str(LEAF)      # the leaf folder replaces this one, so that `hffn_pythia_x86` is importable

import hffn_pythia_x86  # noqa: E402  (puts the shared package on sys.path)
from hffn_pythia.assets import exports_dir, models_dir  # noqa: E402

PREFILL, DECODE, STEPS = (4, 128), (4, 32), 16


def status_kb(key: str) -> int:
    return next(int(line.split()[1]) for line in open('/proc/self/status') if line.startswith(key))


def worker(job: dict) -> int:
    import torch
    from hffn_pythia import TOKENS
    if job['threads']:
        torch.set_num_threads(job['threads'])
    before = status_kb('VmRSS:')
    folder = Path(job['exports']) / f"pythia{job['size']}m_sparse{job['density']}_ffn_only" / 'export'
    model = hffn_pythia_x86.load_pythia(Path(job['models']) / f"pythia{job['size']}m", folder, job['arm'], job['threads'])
    loaded, peak_load = status_kb('VmRSS:'), status_kb('VmHWM:')
    blocks = model.gpt_neox.layers
    if job['arm'] == 'dense':
        ffn_bytes = sum(p.numel() * p.element_size() for block in blocks for p in block.mlp.parameters())
    else:
        ffn_bytes = sum(layer.model_bytes() for layer in model.hffn_runtime.layers)
    stream = torch.tensor(json.loads(TOKENS.read_text())['tokens'])
    with torch.inference_mode():
        batch, length = PREFILL
        model(input_ids=stream[:batch * length].view(batch, length), use_cache=True)
        prefilled, peak_prefill = status_kb('VmRSS:'), status_kb('VmHWM:')
        batch, length = DECODE
        out = model(input_ids=stream[:batch * length].view(batch, length), use_cache=True)
        for _ in range(STEPS):
            out = model(input_ids=out.logits[:, -1:].argmax(-1), past_key_values=out.past_key_values, use_cache=True)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    print(json.dumps({'size': job['size'], 'density': job['density'], 'arm': job['arm'], 'ffn_bytes': ffn_bytes, 'parameter_bytes': total,
                      'workspace_bytes': model.hffn_runtime.workspace_bytes() if job['arm'] == 'hffn' else 0,
                      'rss_kb': {'before_model': before, 'after_load': loaded, 'after_prefill': prefilled, 'after_decode': status_kb('VmRSS:')},
                      'peak_kb': {'during_load': peak_load, 'after_prefill': peak_prefill, 'after_decode': status_kb('VmHWM:')}}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--exports', type=Path)
    parser.add_argument('--sizes', default='70,160,410')
    parser.add_argument('--densities', default='10,20,50,80')
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args()
    if args.worker:
        return worker(json.loads(args.worker))
    models, exports = args.models or models_dir(), args.exports or exports_dir()
    rows = []
    for size in (int(v) for v in args.sizes.split(',')):
        for density in (int(v) for v in args.densities.split(',')):
            for arm in ('dense', 'hffn'):
                job = {'size': size, 'density': density, 'arm': arm, 'models': str(models), 'exports': str(exports), 'threads': args.threads}
                result = subprocess.run([sys.executable, '-B', '-W', 'ignore', __file__, '--worker', json.dumps(job)], text=True, capture_output=True)
                lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
                if result.returncode != 0 or not lines:
                    print(f'failed: {job}\n{result.stderr[-2000:]}', file=sys.stderr)
                    return 1
                rows.append(json.loads(lines[-1]))
                row = rows[-1]
                print(f"pythia{size}m d{density} {arm}: ffn {row['ffn_bytes'] / 1e6:.1f} MB, resident after decode {row['rss_kb']['after_decode'] / 1024:.0f} MB, "
                      f"peak {row['peak_kb']['after_decode'] / 1024:.0f} MB", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'threads': args.threads, 'prefill': PREFILL, 'decode': DECODE, 'decode_steps': STEPS, 'rows': rows}, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
