"""Operational deadline/signal supervisor; no numerical or performance measurement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20 * 60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.seconds <= 0:
        parser.error('a child command and positive deadline are required')
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get('HFFN_QUAL_STARTED_EPOCH', time.time())) + args.seconds
    launcher = int(os.environ.get('HFFN_LAUNCHER_PID', os.getppid()))
    requested = None

    def interrupted(*_):
        nonlocal requested
        requested = 'signal'

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child = subprocess.Popen(command, start_new_session=True)
    print(f'supervisor: verifier pid={child.pid}, external deadline enabled', flush=True)
    sent_at, forced = None, False
    while child.poll() is None:
        try:
            os.kill(launcher, 0)
        except ProcessLookupError:
            requested = requested or 'launcher_exited'
        if (args.out / 'STOP').exists():
            requested = requested or 'signal'
        if time.time() >= deadline - min(3, args.seconds / 2):
            requested = requested or 'timeout'
        if requested and sent_at is None:
            try:
                os.killpg(child.pid, signal.SIGALRM if requested == 'timeout' else signal.SIGTERM)
            except ProcessLookupError:
                pass
            sent_at = time.monotonic()
        if sent_at is not None and (time.monotonic() - sent_at >= 3 or time.time() >= deadline):
            try:
                os.killpg(child.pid, signal.SIGKILL)
                forced = True
            except ProcessLookupError:
                pass
            child.wait()
            break
        time.sleep(0.1)
    raw_code = child.wait()
    # Cancellation remains a failed run even if it races with a successful child exit.
    if (args.out / 'STOP').exists():
        requested = requested or 'signal'
    try:
        os.kill(launcher, 0)
    except ProcessLookupError:
        requested = requested or 'launcher_exited'
    code = 124 if requested == 'timeout' else 130 if requested else raw_code if raw_code >= 0 else 128 - raw_code
    if code:
        path = args.out / 'manifest.json'
        manifest = json.loads(path.read_text()) if path.exists() else dict(planned=0, done=0)
        if manifest.get('status') != 'failed':
            manifest.update(status='failed', supervisor_error=requested or f'child exit {raw_code}')
            path.write_text(json.dumps(manifest, indent=2))
    record = dict(status='passed' if code == 0 else 'failed', exit=code, child_exit=raw_code,
                  reason=requested, forced_kill=forced)
    (args.out / 'supervisor.json').write_text(json.dumps(record, indent=2))
    (args.out / 'EXIT').write_text(f'{code}\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
