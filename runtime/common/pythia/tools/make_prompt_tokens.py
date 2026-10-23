"""Write ``wikitext2_test_tokens.json`` (``hffn_pythia``): the first token ids of the WikiText-2 test split.

Non-empty lines in order, each encoded without special tokens and followed by EOS, with the tokenizer of the
Pythia base checkpoints (the three sizes share one). The parity checks and the timing harness read this file, so
that they need neither ``datasets`` nor a tokenizer. The record keeps the SHA-256 of both inputs.

    python -B make_prompt_tokens.py <base checkpoint dir> <wikitext-test.arrow> [<output file>]

Needs ``datasets`` and ``transformers``. The WikiText-2 raw v1 test split is CC BY-SA.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

COUNT = 4096
OUTPUT = Path(__file__).resolve().parents[1] / "hffn_pythia" / "wikitext2_test_tokens.json"


def main() -> None:
    model_dir, arrow = Path(sys.argv[1]), Path(sys.argv[2])
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else OUTPUT
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokens: list[int] = []
    for text in Dataset.from_file(str(arrow))["text"]:
        if text.strip():
            tokens += tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(tokens) >= COUNT:
            break
    record = {
        "source": "WikiText-2 raw v1, test split (CC BY-SA), first non-empty lines",
        "arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "tokens": tokens[:COUNT],
    }
    output.write_text(json.dumps(record) + "\n")
    print(f"{output}: {COUNT} tokens")


if __name__ == "__main__":
    main()
