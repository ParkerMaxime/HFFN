#!/usr/bin/env python3
"""CPU-only regression gate for the measured Metal schedule policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hffn_metal_llm import load_compact_wanda_layer, select_training_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse10-layer", type=Path, required=True)
    parser.add_argument("--sparse20-layer", type=Path, required=True)
    args = parser.parse_args()
    sparse10 = load_compact_wanda_layer(args.sparse10_layer)
    sparse20 = load_compact_wanda_layer(args.sparse20_layer)
    observed = {
        "d10_n1": select_training_layout(sparse10, 1),
        "d10_n4": select_training_layout(sparse10, 4),
        "d10_n6": select_training_layout(sparse10, 6),
        "d10_n7": select_training_layout(sparse10, 7),
        "d10_n8": select_training_layout(sparse10, 8),
        "d10_n9": select_training_layout(sparse10, 9),
        "d10_n16": select_training_layout(sparse10, 16),
        "d20_n4": select_training_layout(sparse20, 4),
        "d20_n6": select_training_layout(sparse20, 6),
        "d20_n7": select_training_layout(sparse20, 7),
        "d20_n8": select_training_layout(sparse20, 8),
        "d20_n16": select_training_layout(sparse20, 16),
        "d20_n256": select_training_layout(sparse20, 256),
    }
    expected = {
        "d10_n1": "token_major",
        "d10_n4": "token_major",
        "d10_n6": "token_major",
        "d10_n7": "token_major",
        "d10_n8": "token_major",
        "d10_n9": "feature_persistent",
        "d10_n16": "feature_persistent",
        "d20_n4": "token_major",
        "d20_n6": "token_major",
        "d20_n7": "feature_persistent",
        "d20_n8": "feature_persistent",
        "d20_n16": "feature_persistent",
        "d20_n256": "feature_owner_tile",
    }
    if observed != expected:
        raise AssertionError({"expected": expected, "observed": observed})
    print(json.dumps({
        "format": "hffn_metal_layout_policy_gate_v1",
        "passed": True,
        "observed": observed,
    }, indent=2))


if __name__ == "__main__":
    main()
