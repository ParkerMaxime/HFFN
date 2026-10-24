#!/usr/bin/env python3
"""Fetch enwik8 into current/assets/enwik8 and verify it against split.json.

The file and each of the three splits are checked by sha256 against `split.json`; nothing is written unless the
data verifies. Idempotent: a file that already verifies is left as it is. The bundle `enwik8_bundle.zip` next to
`split.json` (the same bytes, recompressed, committed on the cloud branch because the cloud network could not reach
the source) is read first; only if it is absent or does not verify is the file downloaded. `HFFN_ENWIK8_URL`
overrides the source URL of `split.json` (that zip is checked by its recorded sha256).
"""
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


BUNDLE = "enwik8_bundle.zip"


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "current").is_dir():
            return parent
    raise SystemExit("repository root not found: no parent folder contains current/")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def problems(data: bytes, spec: dict) -> list[str]:
    out = []
    if len(data) != spec["bytes"]:
        out.append(f"size {len(data)} != {spec['bytes']}")
    if sha256(data) != spec["sha256"]:
        out.append("sha256 of the file differs from split.json")
    for name, part in spec["splits"].items():
        if sha256(data[part["start"]:part["end"]]) != part["sha256"]:
            out.append(f"sha256 of split {name} differs from split.json")
    return out


def main() -> int:
    dest = repo_root() / "current" / "assets" / "enwik8"
    spec = json.loads((dest / "split.json").read_text())
    target = dest / spec["file"]
    if target.is_file():
        bad = problems(target.read_bytes(), spec)
        if not bad:
            print(f"enwik8 already present and verified: {target}", flush=True)
            return 0
        print(f"existing {target} does not verify ({'; '.join(bad)}), fetching again", flush=True)

    src = spec["source"]
    bundle = dest / BUNDLE
    if bundle.is_file():
        with zipfile.ZipFile(bundle) as archive:
            data = archive.read(src["zip_member"])
        bad = problems(data, spec)
        if not bad:
            target.write_bytes(data)
            print(f"enwik8 extracted from {bundle} and verified: {target}", flush=True)
            return 0
        print(f"{bundle} does not verify ({'; '.join(bad)}), downloading instead", flush=True)

    url = str(os.environ.get("HFFN_ENWIK8_URL") or src["url"])
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        blob = response.read()
    if len(blob) != src["zip_bytes"] or sha256(blob) != src["zip_sha256"]:
        print(f"zip does not match split.json (size {len(blob)}, expected {src['zip_bytes']})", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        data = archive.read(src["zip_member"])
    bad = problems(data, spec)
    if bad:
        print("extracted file does not verify: " + "; ".join(bad), file=sys.stderr)
        return 1
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    print(f"enwik8 written and verified: {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
