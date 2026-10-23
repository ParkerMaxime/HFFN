"""Request-level timing harness shared by the Pythia inference leaves: HFFN, a second HFFN instance (A/A), dense.

A leaf calls `main(adapter, script)` from its own `bench/bench_request.py`; `adapter` says what is specific to
its hardware (see `Adapter`), `script` is that file (the fresh process of a cell is started on it). Protocol:
one fresh process per cell (export x shape) with three models from the leaf's `load_pythia`, which read one copy
of the non-FFN weights (two copies of the same weights do not run at the same speed); a window is a
number of consecutive requests between two synchronisations, fixed per cell so that a dense window lasts at
least `window_ms`; every timed window follows an untimed window of the same arm; arms run in both orders in
every process. Gates before timing: the logits of `hffn_copy` are bitwise those of `hffn`, and `hffn` is within
GUARD of max|logit| of `dense`. A run is admissible when at least AA_SHARE of the ratios `hffn_copy/hffn` (one
per cell and session) are within 1 +- AA_TOLERANCE; only then is `hffn/dense` read.

    python -B bench_request.py --mode check|smoke|run [--models <base checkpoints dir> --exports <exports dir>] [--out <run folder>]
                               [--threads T] [--densities 10,50] [--sessions N]
    (without --models and --exports, the folders of $HFFN_ASSETS: see `assets.py`; --densities and --sessions replace
    those of the grid of the mode, which `config.json` of the run records.)
    python -B bench_request.py --worker '<cell json>'        (one cell in a fresh process, prints one JSON line)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import TOKENS
from .assets import exports_dir, models_dir

ARMS = ('hffn', 'dense', 'hffn_copy')
SHAPES = {   # shape: batch, prompt tokens, cached decode steps after the prompt call
    'prefill_b1': dict(batch=1, prompt=128, steps=0),
    'prefill_b4': dict(batch=4, prompt=128, steps=0),
    'decode_b1': dict(batch=1, prompt=32, steps=16),
    'decode_b4': dict(batch=4, prompt=32, steps=16),
}
GRID = {   # mode: sizes, densities, shapes, sessions, blocks, minimum window of the dense arm (ms)
    'run': dict(sizes=(70, 160, 410), densities=(10, 20, 50, 80), shapes=tuple(SHAPES), sessions=3, blocks=2, window_ms=300.0),
    'smoke': dict(sizes=(70,), densities=(20,), shapes=('prefill_b1', 'decode_b1'), sessions=1, blocks=1, window_ms=60.0),
}
GUARD = 1e-3            # hffn against dense, as a fraction of max|logit| (gross-error guard)
AA_TOLERANCE, AA_SHARE = 0.03, 0.95
MAX_REPEATS = 200


def folders(models, exports, size, density):
    return Path(models)/f'pythia{size}m', Path(exports)/f'pythia{size}m_sparse{density}_ffn_only'/'export'


@dataclass
class Adapter:
    """What a hardware leaf gives the harness.

    load_pythia(model_dir, export_dir, ffn): the loader of the leaf; device: where the prompt tensors live;
    synchronize(): wait for queued work (nothing on a CPU); state(): what is recorded before each process
    (clocks and temperature of a GPU, load and stolen time of a CPU); versions(): one line of library versions;
    describe(model): what the leaf reports of its HFFN runtime after a cell; configure(job): applied in the worker
    before any model is loaded (thread counts); runtime_files: sources hashed in the provenance; note: one line
    on the machine class."""
    name: str
    load_pythia: object
    device: str
    synchronize: object
    state: object
    versions: object
    describe: object
    runtime_files: list = field(default_factory=list)
    configure: object = None
    note: str = ''


def runtime_digests(paths):
    """SHA-256 of the runtime files of a leaf, keyed `<folder>/<name>` (as `harness_files`); two files that still share a key are an error, not a lost hash."""
    out = {}
    for p in paths:
        key = f'{Path(p).parent.name}/{Path(p).name}'
        if key in out:
            raise ValueError(f'runtime_files: two files share the provenance key {key}')
        out[key] = sha256(p)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# -- worker ------------------------------------------------------------------------------------------------
def worker(adapter, cell):
    import torch
    if adapter.configure:
        adapter.configure(cell)
    from .model import share_non_ffn
    load_pythia = adapter.load_pythia
    model_dir, export_dir = folders(cell['models'], cell['exports'], cell['size'], cell['density'])
    models = {'dense': load_pythia(model_dir, export_dir, 'dense'), 'hffn': load_pythia(model_dir, export_dir, 'hffn'),
              'hffn_copy': load_pythia(model_dir, export_dir, 'hffn')}
    for arm in ('hffn', 'hffn_copy'):       # one copy of the non-FFN weights in the process: the arms differ by the FFN only
        share_non_ffn(models[arm], models['dense'])
    shape = SHAPES[cell['shape']]
    batch, length, steps = shape['batch'], shape['prompt'], shape['steps']
    stream = json.loads(TOKENS.read_text())['tokens']
    prompt = torch.tensor(stream[:batch*length], device=adapter.device).view(batch, length)

    def request(arm, tokens, keep=False):
        """One request: the prompt call, then the cached decode steps fed with `tokens`. No synchronisation.
        With `keep`, the last-position logits of every call."""
        model = models[arm]
        out = model(input_ids=prompt, use_cache=True)
        logits = [out.logits[:, -1].clone()] if keep else None
        for token in tokens:
            out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
            if keep:
                logits.append(out.logits[:, -1].clone())
        return logits

    def window(arm, tokens, repeats):
        adapter.synchronize()
        start = time.perf_counter_ns()
        for _ in range(repeats):
            request(arm, tokens)
        adapter.synchronize()
        return (time.perf_counter_ns()-start)/repeats/1e6

    with torch.inference_mode():
        out = models['dense'](input_ids=prompt, use_cache=True)      # the greedy tokens of the dense arm, fed to every arm
        tokens = []
        for _ in range(steps):
            tokens.append(out.logits[:, -1:].argmax(-1))
            out = models['dense'](input_ids=tokens[-1], past_key_values=out.past_key_values, use_cache=True)
        del out
        reference = {arm: request(arm, tokens, keep=True) for arm in ARMS}
        bitwise = all(bool(torch.equal(a, b)) for a, b in zip(reference['hffn_copy'], reference['hffn']))
        pairs = list(zip(reference['hffn'], reference['dense']))
        gap = max(float((a-b).abs().max()) for a, b in pairs)/max(float(b.abs().max()) for _, b in pairs)
        argmax_differ = sum(int((a.argmax(-1) != b.argmax(-1)).sum()) for a, b in pairs)
        del reference, pairs
        passed = bitwise and gap <= GUARD
        samples, repeats = {arm: [] for arm in ARMS}, None
        if passed:
            for arm in ARMS:                                         # warm-up: kernels compiled, buffers reserved
                window(arm, tokens, 2)
            once = min(window('dense', tokens, 1) for _ in range(3))
            repeats = max(1, min(MAX_REPEATS, int(cell['window_ms']/once)+1))
            orders = (ARMS+ARMS[::-1], ARMS[::-1]+ARMS)
            for block in range(cell['blocks']):
                for arm in orders[(cell['parity']+block) % 2]:
                    window(arm, tokens, repeats)                     # untimed window of the same arm before each timed one
                    samples[arm].append(window(arm, tokens, repeats))
    print(json.dumps({**{k: v for k, v in cell.items() if k not in ('models', 'exports')}, 'samples_ms': samples, 'repeats': repeats,
                      'bitwise_copy': bitwise, 'hffn_vs_dense_relative_gap': gap, 'argmax_differing_positions': argmax_differ,
                      'gate_passed': passed, 'runtime': adapter.describe(models['hffn'])}))
    return 0 if passed else 4


# -- orchestrator ------------------------------------------------------------------------------------------
def summarize(rows):
    cells = {}
    for row in rows:
        cells.setdefault((row['size'], row['density'], row['shape']), []).append(
            {arm: statistics.median(row['samples_ms'][arm]) for arm in ARMS})
    out = []
    for (size, density, shape), values in sorted(cells.items()):
        ratio = [v['hffn']/v['dense'] for v in values]
        out.append({'size': size, 'density': density, 'shape': shape, 'sessions': len(values),
                    'ms': {arm: statistics.median(v[arm] for v in values) for arm in ARMS},
                    'hffn_over_dense': statistics.median(ratio), 'min': min(ratio), 'max': max(ratio),
                    'aa': [v['hffn_copy']/v['hffn'] for v in values]})
    return out


def verdict(cells, complete, gate_passed):
    """The admissibility and retention criteria of the timing method (leaf README), fixed before any measurement."""
    if not complete:
        return {'admissible': False, 'reason': 'incomplete coverage'}
    aa = [a for c in cells for a in c['aa']]
    inside = sum(abs(a-1) <= AA_TOLERANCE for a in aa)
    return {'admissible': gate_passed and inside >= AA_SHARE*len(aa), 'gate_passed': gate_passed, 'aa_values': len(aa),
            'aa_within_tolerance': inside, 'aa_share': inside/len(aa), 'aa_median': statistics.median(aa),
            'aa_min': min(aa), 'aa_max': max(aa)}


def cells_of(grid):
    return [dict(size=s, density=d, shape=shape) for s in grid['sizes'] for d in grid['densities'] for shape in grid['shapes']]


def missing_assets(grid, models, exports):
    return [str(f) for s in grid['sizes'] for d in grid['densities']
            for f in (folders(models, exports, s, d)[0]/'model.safetensors', folders(models, exports, s, d)[1]/'layer0') if not f.exists()]


def orchestrate(adapter, script, mode, models, exports, out, threads=None, densities=None, sessions=None):
    grid = {**GRID[mode], **({'densities': densities} if densities else {}), **({'sessions': sessions} if sessions else {})}
    missing = missing_assets(grid, models, exports)
    if missing:
        raise SystemExit(f'missing assets: {missing}')
    out.mkdir(parents=True, exist_ok=True)
    cells = cells_of(grid)
    planned = len(cells)*grid['sessions']
    (out/'config.json').write_text(json.dumps({'mode': mode, **grid, 'arms': ARMS, 'shape_definitions': SHAPES, 'guard': GUARD,
                                               'aa_tolerance': AA_TOLERANCE, 'aa_share': AA_SHARE}, indent=1)+'\n')
    here = Path(script).resolve().parent
    (out/'provenance.json').write_text(json.dumps({
        'leaf': adapter.name, 'threads': threads,
        'shared_files': {f'hffn_pythia/{p.name}': sha256(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py', '.json')},
        'harness_files': {f'{here.name}/{p.name}': sha256(p) for p in (Path(script), *sorted(here.glob('*.sh')))},
        'runtime_files': runtime_digests(adapter.runtime_files),
        'versions': adapter.versions(), 'state_at_start': adapter.state(), 'machine': adapter.note,
    }, indent=1)+'\n')
    manifest = {'status': 'running', 'planned': planned, 'done': 0, 'started': datetime.now(timezone.utc).isoformat()}
    write = lambda: (out/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    write()
    rows, failed = [], None
    with (out/'raw.jsonl').open('w') as raw:
        for session in range(grid['sessions']):
            for index, cell in (list(enumerate(cells)) if session % 2 == 0 else list(enumerate(cells))[::-1]):
                job = {**cell, 'session': session, 'parity': (session+index) % 2, 'blocks': grid['blocks'],
                       'window_ms': grid['window_ms'], 'models': str(models), 'exports': str(exports), 'threads': threads}
                state = adapter.state()
                result = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(script), '--worker', json.dumps(job)],
                                        text=True, capture_output=True, cwd=here)
                lines = [line for line in result.stdout.strip().splitlines() if line.startswith('{')]
                if result.returncode not in (0, 4) or not lines:
                    failed = {'cell': cell, 'session': session, 'returncode': result.returncode, 'stderr': result.stderr[-4000:]}
                    break
                row = {**json.loads(lines[-1]), 'state_before': state}
                raw.write(json.dumps(row)+'\n')
                raw.flush()
                rows.append(row)
                if result.returncode == 4:
                    failed = {'cell': cell, 'session': session, 'returncode': 4, 'row': {k: v for k, v in row.items() if k != 'samples_ms'}}
                    break
                manifest['done'] += 1
                write()
                med = {arm: statistics.median(row['samples_ms'][arm]) for arm in ARMS}
                print(f"[{manifest['done']}/{planned}] s{session} pythia{cell['size']}m d{cell['density']} {cell['shape']} x{row['repeats']}: "
                      + ', '.join(f'{arm} {med[arm]:.3f}' for arm in ARMS)
                      + ' ms, ' + ', '.join(f'{k} {v}' for k, v in list(state.items())[:3]), flush=True)
            if failed:
                break
    complete = manifest['done'] == planned
    gate_passed = failed is None or failed['returncode'] != 4
    cells_summary = summarize(rows) if complete else []
    judged = verdict(cells_summary, complete, gate_passed) if cells_summary else None
    summary = {'cells': cells_summary, 'verdict': judged, 'failure': failed, 'state_at_end': adapter.state()}
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    status = 'passed' if complete else 'failed'
    report = [f'# Request-level harness, {adapter.name}, {mode}', '', f"Status: {status}; {manifest['done']}/{planned} processes.", '',
              'Milliseconds per request. `hffn/dense` is read only if the verdict below says admissible.', '',
              '| model | density | shape | dense | hffn | hffn/dense [min-max] | A/A per session |', '|---|---:|---|---:|---:|---|---|']
    for c in cells_summary:
        report.append(f"| {c['size']}M | {c['density']} % | {c['shape']} | {c['ms']['dense']:.3f} | {c['ms']['hffn']:.3f} | "
                      f"{c['hffn_over_dense']:.3f} [{c['min']:.3f}-{c['max']:.3f}] | " + ' '.join(f'{a:.3f}' for a in c['aa']) + ' |')
    report += ['', '```json', json.dumps(judged, indent=1), '```']
    if failed:
        report += ['', 'Failure:', '```', json.dumps(failed, indent=1)[:4000], '```']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    manifest.update(status=status, finished=datetime.now(timezone.utc).isoformat())
    write()
    print('\n'.join(report), flush=True)
    return 0 if complete else 1


def main(adapter, script):
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--mode', choices=['check', *sorted(GRID)])
    parser.add_argument('--models', type=Path)
    parser.add_argument('--exports', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--threads', type=int, default=None, help='thread count given to the leaf (CPU leaves)')
    parser.add_argument('--densities', type=lambda text: tuple(int(v) for v in text.split(',')), default=None,
                        help='densities (percent, comma separated) in place of those of the grid of the mode')
    parser.add_argument('--sessions', type=int, default=None, help='number of sessions in place of that of the grid of the mode')
    args = parser.parse_args()
    if args.worker:
        return worker(adapter, json.loads(args.worker))
    args.models, args.exports = args.models or models_dir(), args.exports or exports_dir()
    if args.mode == 'check':
        missing = missing_assets(GRID['run'], args.models, args.exports)
        print(f"check: {len(cells_of(GRID['run']))} cells; missing assets: {missing or 'none'}")
        return 1 if missing else 0
    return orchestrate(adapter, script, args.mode, args.models, args.exports, args.out, args.threads, args.densities, args.sessions)
