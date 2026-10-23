"""Status manifest and tables of one campaign folder (written by campaign.sh).

    python -B bench/campaign_manifest.py results/<tag> --mode run --exit-code 0 --started <UTC>

Writes ``manifest.json`` (status, steps, planned and done, source, CPU steal, admissibility of the ratios) and, for a
campaign that passed, ``AGGREGATE.md``. Admissibility of the ratios `hffn/dense` (README, "Measurement campaign"):
status passed, exit code 0, full coverage of the request harness, CPU steal below 0.5 %, and the verdict of the
harness (gates of every process, and at least 95 % of the ratios `hffn_copy/hffn` within 1 +- 0.03). There is no
speed threshold: the tables report, for each cell, the median over the sessions of `hffn/dense` with the interval of
the sessions, and a cell whose spread exceeds 10 % of its median is unstable and carries no quantified statement.
The run with 4 threads is a single session, reported without interval. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import provenance  # noqa: E402

STEAL_LIMIT = 0.005
UNSTABLE_SPREAD = 0.10


def steal_fraction(directory: Path):
    try:
        first, last = ([int(v) for v in (directory / f'proc_stat_{k}.txt').read_text().split()[1:9]] for k in ('start', 'end'))
    except (OSError, ValueError):
        return None
    delta = [b - a for a, b in zip(first, last)]
    return delta[7] / sum(delta) if sum(delta) else None


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def table(header: list[str], rows: list[list]) -> list[str]:
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows] + ['']


def cell_rows(summary: dict, interval: bool) -> list[list]:
    rows = []
    for c in summary['cells']:
        spread = (c['max'] - c['min']) / c['hffn_over_dense']
        if interval:
            verdict = 'unstable' if spread > UNSTABLE_SPREAD else ('HFFN faster' if c['max'] < 1 else 'HFFN slower' if c['min'] > 1 else 'mixed')
            ratio = f"{c['hffn_over_dense']:.3f} [{c['min']:.3f}-{c['max']:.3f}]"
            rows.append([f"{c['size']}M", f"{c['density']} %", c['shape'], f"{c['ms']['dense']:.2f}", f"{c['ms']['hffn']:.2f}", ratio, f'{spread:.1%}', verdict,
                         ' '.join(f'{a:.3f}' for a in c['aa'])])
        else:
            rows.append([f"{c['size']}M", f"{c['density']} %", c['shape'], f"{c['ms']['dense']:.2f}", f"{c['ms']['hffn']:.2f}", f"{c['hffn_over_dense']:.3f}",
                         ' '.join(f'{a:.3f}' for a in c['aa'])])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', default='run')
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--started', default=None)
    args = parser.parse_args()
    directory = args.directory
    steps = []
    for line in (directory / 'steps.tsv').read_text().splitlines():
        name, code, seconds = line.split('\t')
        steps.append({'name': name, 'exit_code': int(code), 'seconds': int(seconds)})
    planned = int((directory / 'planned.txt').read_text())
    done = sum(s['exit_code'] == 0 for s in steps)
    steal = steal_fraction(directory)
    status = 'passed' if args.exit_code == 0 and done == planned else ('partial' if args.exit_code == 124 else 'failed')
    bench = {name: load(directory / name / 'manifest.json') for name in ('bench_run', 'bench_threads4')}
    summary = {name: load(directory / name / 'summary.json') for name in ('bench_run', 'bench_threads4')}
    reasons = []
    if status != 'passed':
        reasons.append(f'campaign status {status}')
    if not bench['bench_run'] or bench['bench_run'].get('status') != 'passed' or bench['bench_run']['done'] != bench['bench_run']['planned']:
        reasons.append('request harness: incomplete coverage')
    if steal is None or steal >= STEAL_LIMIT:
        reasons.append(f'CPU steal {steal} not below {STEAL_LIMIT}')
    verdict = (summary['bench_run'] or {}).get('verdict')
    if not verdict or not verdict.get('admissible'):
        reasons.append(f'harness verdict not admissible: {verdict}')
    manifest = {'tag': directory.name, 'mode': args.mode, 'status': status, 'exit_code': args.exit_code, 'started_utc': args.started, 'planned': planned,
                'done': done, 'seconds': sum(s['seconds'] for s in steps), 'cpu_steal_fraction': steal, 'source': provenance.record(),
                'ratios_admissible': not reasons, 'inadmissible_because': reasons, 'harness': {k: (v and {x: v[x] for x in ('status', 'planned', 'done')}) for k, v in bench.items()},
                'steps': steps}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"manifest: status={status} exit_code={args.exit_code} steps={done}/{planned} ratios_admissible={not reasons}")
    if status != 'passed':
        return
    lines = [f'# Campaign `{directory.name}` ({args.mode})', '', f"Status **{status}**; {done}/{planned} steps; CPU steal {steal:.4%}; source `{manifest['source']}`.", '']
    lines += [f"Ratios `hffn/dense` are **{'admissible' if not reasons else 'NOT admissible'}**" + ('' if not reasons else ': ' + '; '.join(reasons)) + '.', '']
    if verdict:
        lines += [f"Harness verdict: {json.dumps(verdict)}", '']
    if not reasons:
        lines += ['## Request harness, 8 threads: `hffn/dense` per cell (median over sessions [min-max]; unstable above 10 % spread)', '']
        lines += table(['model', 'density', 'shape', 'dense ms', 'hffn ms', 'hffn/dense', 'spread', 'reading', 'A/A per session'], cell_rows(summary['bench_run'], True))
        if summary['bench_threads4'] and summary['bench_threads4'].get('cells'):
            lines += ['## Request harness, 4 threads for PyTorch and HFFN: one session, no interval', '']
            lines += table(['model', 'density', 'shape', 'dense ms', 'hffn ms', 'hffn/dense', 'A/A'], cell_rows(summary['bench_threads4'], False))
    memory = load(directory / 'memory.json')
    if memory:
        by = {(r['size'], r['density'], r['arm']): r for r in memory['rows']}
        rows = []
        for (size, density, arm), row in sorted(by.items()):
            if arm != 'dense':
                continue
            other = by[(size, density, 'hffn')]
            rows.append([f'{size}M', f'{density} %', f"{row['ffn_bytes'] / 1e6:.1f}", f"{other['ffn_bytes'] / 1e6:.1f}", f"{row['ffn_bytes'] / other['ffn_bytes']:.2f}",
                         f"{row['rss_kb']['after_decode'] / 1024:.0f}", f"{other['rss_kb']['after_decode'] / 1024:.0f}",
                         f"{row['peak_kb']['after_decode'] / 1024:.0f}", f"{other['peak_kb']['after_decode'] / 1024:.0f}"])
        lines += ['## Memory, one process per arm and export (MB of 10^6 bytes for the FFNs, MiB for the process)', '']
        lines += table(['model', 'density', 'FFN dense', 'FFN hffn', 'dense/hffn', 'resident dense', 'resident hffn', 'peak dense', 'peak hffn'], rows)
    (directory / 'AGGREGATE.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
