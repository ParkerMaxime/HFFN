"""Check every file listed in ``<leaf>/sources/manifest.json`` against its recorded SHA-256.

    python -B -m hffn_pythia.verify_sources <leaf folder>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main(leaf: str | Path) -> int:
    root, failed = Path(leaf).resolve(), 0
    for entry in json.loads((root / "sources" / "manifest.json").read_text())["files"]:
        actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        ok = actual == entry["sha256"]
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {entry['path']}" + ("" if ok else f" sha256={actual}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
