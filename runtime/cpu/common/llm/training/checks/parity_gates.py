"""The model gate and the update gate of one export in one process, from one load of the supports and one float64 reference.

    python -B checks/parity_gates.py --model-dir <base checkpoint> --export-dir <export> --tokens <wikitext2_test_tokens.json> [--out-model <json>] [--out-update <json>]

Same criteria as `parity_model.py --gate` and `parity_update.py --gate`, which stay the reference of each gate: the dense and
HFFN arms are built afresh for each gate, only the float64 evaluation and the supports they are compared on are shared.
A gate that raises does not stop the other. Exit code 1 if either fails.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_adapter as adapter  # noqa: E402
import parity_model  # noqa: E402
import parity_update  # noqa: E402


def attempt(gate, *arguments) -> int:
    try:
        return gate(*arguments)
    except Exception:
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--out-model", type=Path)
    parser.add_argument("--out-update", type=Path)
    args = parser.parse_args()
    shared = {key: value for key, value in vars(args).items() if not key.startswith("out_")}
    ids = torch.tensor(json.loads(args.tokens.read_text())["tokens"][:args.batch * args.seq_len]).view(args.batch, args.seq_len)
    torch.set_num_threads(4)
    supports, checksum, initial = parity_update.load_supports(args.model_dir, args.export_dir)
    dense = adapter.base.dense_model(args.model_dir, args.export_dir).double()
    loss64, gradients, other = adapter.base.dense_gradients(dense, ids, supports)
    del dense
    gc.collect()
    model_code = attempt(parity_model.model_gate, SimpleNamespace(**shared, gate=True, out=args.out_model), ids, supports, (loss64, gradients, other))
    del other
    gc.collect()
    update_code = attempt(parity_update.update_gate, SimpleNamespace(**shared, gate=True, out=args.out_update), ids, supports, checksum, initial, loss64, gradients)
    return max(model_code, update_code)


if __name__ == "__main__":
    sys.exit(main())
