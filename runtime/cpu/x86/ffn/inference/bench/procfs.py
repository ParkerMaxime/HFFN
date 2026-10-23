"""Linux procfs memory probes shared by memory.py and vs_dense.py.

Peak RSS: writing 5 to ``/proc/self/clear_refs`` resets VmHWM, so the peak of one call is
VmHWM - VmRSS read just before it. Pages touched: writing 1 clears the referenced bits, so
the ``Referenced`` pages of ``/proc/self/smaps`` after a call are the pages it touched (the
count around an empty Python call is subtracted; median of ``repeat`` measurements).

Memory measurements need a fixed glibc mmap threshold: glibc raises it after a large free,
after which freed buffers stay in the heap and peaks read 0. ``fix_mmap_threshold()``
re-executes the script with MALLOC_MMAP_THRESHOLD_=65536 (every buffer >= 64 KiB in its own
mmap). Never in a timing run: per-call allocations would all pay fresh page faults.
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
from pathlib import Path


def fix_mmap_threshold() -> None:
    if os.environ.get("MALLOC_MMAP_THRESHOLD_") != "65536":
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "MALLOC_MMAP_THRESHOLD_": "65536"})


def status_kb(key: str) -> int:
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1])
    raise KeyError(key)


def reset_peak() -> int:
    gc.collect()
    Path("/proc/self/clear_refs").write_text("5")
    return status_kb("VmRSS:")


def peak_delta_kb(start_rss: int) -> int:
    return status_kb("VmHWM:") - start_rss


def referenced_kb() -> int:
    return sum(int(line.split()[1]) for line in open("/proc/self/smaps") if line.startswith("Referenced:"))


def touched_kb(call, repeat: int) -> float:
    def once(function) -> int:
        Path("/proc/self/clear_refs").write_text("1")
        before = referenced_kb()
        function()
        return referenced_kb() - before
    noise = statistics.median(once(lambda: None) for _ in range(repeat))
    return statistics.median(once(call) for _ in range(repeat)) - noise
