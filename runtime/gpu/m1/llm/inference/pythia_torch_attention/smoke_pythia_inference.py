#!/usr/bin/env python3
"""Full Pythia smoke: prefill, KV-cache decode, greedy generation and reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
sys.path.insert(0, str(ROOT / "runtime/common/pythia"))

from hffn_pythia.dense_control import install_compact_hffn_values_in_dense_pythia  # noqa: E402
from hffn_metal_llm_inference import load_pythia_hffn_metal_for_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; run outside the GPU sandbox")
    if min(args.prompt_length, args.decode_tokens, args.generation_tokens) < 1:
        raise ValueError("token counts must be positive")

    dense = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, local_files_only=True
    ).eval()
    install_compact_hffn_values_in_dense_pythia(dense, args.export_dir)
    dense.to("mps").requires_grad_(False)
    hffn, runtime, modules = load_pythia_hffn_metal_for_inference(args.model_dir, args.export_dir)
    prompt = torch.arange(1, args.prompt_length + 1, dtype=torch.long, device="mps").unsqueeze(0)
    attention = torch.ones_like(prompt)
    decode_errors = []
    allocated_after_first_decode = None
    with torch.inference_mode():
        dense_prefill = dense(input_ids=prompt, attention_mask=attention, use_cache=True)
        hffn_prefill = hffn(input_ids=prompt, attention_mask=attention, use_cache=True)
        prefill_error = float((dense_prefill.logits - hffn_prefill.logits).abs().max().cpu())
        argmax_equal = bool(torch.equal(dense_prefill.logits.argmax(-1), hffn_prefill.logits.argmax(-1)))
        dense_cache, hffn_cache = dense_prefill.past_key_values, hffn_prefill.past_key_values
        token = hffn_prefill.logits[:, -1:].argmax(dim=-1)
        for step in range(args.decode_tokens):
            dense_decode = dense(input_ids=token, past_key_values=dense_cache, use_cache=True)
            hffn_decode = hffn(input_ids=token, past_key_values=hffn_cache, use_cache=True)
            decode_errors.append(float((dense_decode.logits - hffn_decode.logits).abs().max().cpu()))
            dense_cache, hffn_cache = dense_decode.past_key_values, hffn_decode.past_key_values
            token = hffn_decode.logits[:, -1:].argmax(dim=-1)
            allocated = runtime.workspace_stats()["allocated"]
            if step == 0:
                allocated_after_first_decode = allocated
            elif allocated != allocated_after_first_decode:
                raise AssertionError("workspace count grew during decode")
        generation_args = {
            "attention_mask": attention,
            "do_sample": False,
            "max_new_tokens": args.generation_tokens,
            "use_cache": True,
            "pad_token_id": hffn.config.eos_token_id,
        }
        dense_generated = dense.generate(prompt, **generation_args)
        hffn_generated = hffn.generate(prompt, **generation_args)
        generation_equal = bool(torch.equal(dense_generated, hffn_generated))
    runtime.synchronize()
    failures = []
    if prefill_error > args.atol:
        failures.append(f"prefill error {prefill_error} > {args.atol}")
    if any(error > args.atol for error in decode_errors):
        failures.append(f"decode error {max(decode_errors)} > {args.atol}")
    if not argmax_equal:
        failures.append("prefill argmax differs")
    if not generation_equal:
        failures.append("greedy generation differs")
    stats = runtime.workspace_stats()
    if stats["pending_inputs"] or stats["inflight_calls"]:
        failures.append("runtime has live asynchronous references after synchronize")
    result = {
        "format": "hffn_metal_pythia_inference_smoke_v1",
        "model_dir": str(args.model_dir.resolve()),
        "export_dir": str(args.export_dir.resolve()),
        "prefill_max_abs_error": prefill_error,
        "decode_max_abs_errors": decode_errors,
        "prefill_argmax_equal": argmax_equal,
        "generation_equal": generation_equal,
        "layers": len(modules),
        "workspace": stats,
        "gate": {"atol": args.atol, "passed": not failures, "failures": failures},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
