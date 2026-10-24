"""enwik8 bytes on the manifest split (current/assets/enwik8/split.json); windows of N + 1 bytes."""
import json
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "current").is_dir():
            return parent
    raise SystemExit("repository root not found: no parent folder contains current/")


class Enwik8:
    def __init__(self):
        d = repo_root() / "current" / "assets" / "enwik8"
        self.spec = json.loads((d / "split.json").read_text())
        path = d / self.spec["file"]
        if not path.is_file():
            raise SystemExit(f"{path} missing: run experiments/sparse_attention/common/llm/training/fetch_enwik8.py")
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        assert len(self.data) == self.spec["bytes"]

    def split(self, name):
        s = self.spec["splits"][name]
        return s["start"], s["end"]

    def train_counts(self):
        a, b = self.split("train")
        return np.bincount(np.asarray(self.data[a:b]), minlength=256)

    def train_batches(self, seed: int, batch: int, context: int):
        """Endless generator of (x, y) int64 tensors; the order depends on the seed only (paired across arms)."""
        a, b = self.split("train")
        rng = np.random.RandomState(seed + 7919)
        hi = b - a - context - 1
        while True:
            starts = rng.randint(0, hi, size=batch) + a
            w = np.stack([np.asarray(self.data[s:s + context + 1]) for s in starts]).astype(np.int64)
            t = torch.from_numpy(w)
            yield t[:, :-1], t[:, 1:]

    def eval_windows(self, name: str, context: int, max_windows=None):
        """Non-overlapping windows of the split, in order: (x, y) of shape (n, context)."""
        a, b = self.split(name)
        n = (b - a - 1) // context
        if max_windows:
            n = min(n, max_windows)
        w = np.asarray(self.data[a:a + n * context + 1]).astype(np.int64)
        t = torch.from_numpy(w)
        x = t[:-1].view(n, context)
        y = t[1:].view(n, context)
        return x, y


def balanced_byte_hash(counts: np.ndarray, groups: int) -> np.ndarray:
    """Greedy assignment of the 256 byte values to `groups` groups: heaviest byte first, to the lightest group."""
    table = np.zeros(len(counts), dtype=np.int64)
    load = np.zeros(groups)
    for b in np.argsort(-counts, kind="stable"):
        g = int(np.argmin(load))
        table[b] = g
        load[g] += counts[b]
    return table
