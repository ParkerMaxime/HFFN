#!/usr/bin/env python3
"""Smoke of the leaf: one FFN on the GPU M1 kernel against the dense fixed-support FFN in float64.

    python smoke_homogeneous_ffn.py [--model 256_1024] [--n 1 4 128] [--activation relu]
    python smoke_homogeneous_ffn.py --synthetic 128 512 256 [--density 0.2] [--n 1 4 128] [--activation relu]
    python smoke_homogeneous_ffn.py --model 512_2048 --n 8192 --workspace-mib 512
"""
import argparse
import json
import os
from pathlib import Path

import torch

from homogeneous import FFNInference, HomogeneousFFNInference, OneBankStore, synthetic_ffn

MAX_ABS, REL_L2 = 5e-4, 1e-5
ACTIVATIONS = {'gelu': torch.nn.functional.gelu, 'relu': torch.relu}


def reference(dense, x, activation):
    """The dense fixed-support FFN in float64 on the CPU (a control: dense matrices with zeros)."""
    hidden = ACTIVATIONS[activation](x.double() @ dense['W_up_dense'].double() + dense['b_up'].double())
    return hidden @ dense['W_down_dense'].double() + dense['b_down'].double()


def check(dense, ffn, n, seed=712):
    torch.manual_seed(seed+n)
    x = torch.randn(n, ffn.store.d_model)
    expected = reference(dense, x, ffn.store.activation)
    device_x = x.to('mps')
    first = ffn.forward(device_x).cpu()
    allocated = ffn.workspace_stats()['allocated']
    second = ffn.forward(device_x).cpu()
    error = (first.double()-expected)
    return {'N': n, 'max_abs': float(error.abs().max()), 'rel_l2': float(error.norm()/expected.norm()),
            'repeat_bitwise': bool(torch.equal(first, second)),
            'workspace_growth': ffn.workspace_stats()['allocated']-allocated}


def passed(row):
    return (row['max_abs'] <= MAX_ABS and row['rel_l2'] <= REL_L2 and row['repeat_bitwise']
            and row['workspace_growth'] == 0)


def load_dense(export_dir):
    return torch.load(export_dir/'sparse_dense.pt', map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='256_1024')
    parser.add_argument('--synthetic', type=int, nargs=3, metavar=('D_IN', 'D_FF', 'D_OUT'))
    parser.add_argument('--density', type=float, default=0.2)
    parser.add_argument('--n', type=int, nargs='+', default=[1, 4, 128])
    parser.add_argument('--activation', choices=sorted(ACTIVATIONS))
    parser.add_argument('--workspace-mib', type=int, help='workspace limit of the kernel (default: 64)')
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
        raise SystemExit('MPS without CPU fallback is required')
    limit = None if args.workspace_mib is None else args.workspace_mib*1024**2
    if args.synthetic:
        d_in, d_ff, d_out = args.synthetic
        compact, dense = synthetic_ffn(d_in, d_ff, d_out, args.density, seed=712)
        ffn, name = FFNInference(OneBankStore(compact, d_in, args.activation or 'gelu'), limit), f'synthetic {d_in}x{d_ff}x{d_out}'
    else:
        if 'HFFN_ASSETS' not in os.environ:
            raise SystemExit('HFFN_ASSETS must name the assets folder (homogeneous/models/<model>/export), or use --synthetic')
        export_dir = Path(os.environ['HFFN_ASSETS'])/'homogeneous/models'/args.model/'export'
        ffn, dense, name = HomogeneousFFNInference(export_dir, args.activation, limit), load_dense(export_dir), args.model
    with torch.inference_mode():
        rows = [check(dense, ffn, n) for n in args.n]
    ffn.synchronize()
    ok = all(passed(row) for row in rows)
    print(json.dumps({'model': name, 'activation': ffn.store.activation, 'passed': ok, 'rows': rows}, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
