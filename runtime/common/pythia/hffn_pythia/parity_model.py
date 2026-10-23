"""End-to-end parity of the HFFN arm against the dense arm of the same Pythia model, on text.

Shared by the hardware leaves: each leaf's ``checks/parity_model.py`` calls ``main`` with its own
``load_pythia``. Three models (same checkpoint, same export, same device): the dense arm
and the HFFN arm, both float32, and the dense arm in float64 as the arbiter. Prompts are
consecutive windows of the WikiText-2 test split. Compared tokens: the argmax at every prefill
position, then at every step of a cached decode in which the three models are fed the arbiter's
greedy token, so that they always see the same context.

Criterion: every HFFN token equals the dense float32 token or, failing that, the float64 token
or, failing that, a token whose float64 margin (float64 token's logit minus chosen token's logit)
is at most the larger of one float32 step and the dense float32 control's own error at that
position: |dense32 - float64| on the float64 token's logit plus the same on the chosen token's
logit (the band is set by the control, with no free constant). Two float32
executions that add in different orders can pick different tokens on a near tie; a tie that the
dense float32 control itself cannot resolve is not a fault of the HFFN arm. Logit gaps and what
the leaf reports of its runtime are printed and are not a criterion.

``--negative-control`` also runs the check on an HFFN arm with a fixed fault (attention.dense.weight
of layer 3 scaled by 1.5); it must report unexplained tokens, otherwise the check fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import TOKENS

SHAPES = ((1, 32), (4, 32), (1, 128), (4, 128))  # (batch, prompt length)
DECODE_STEPS = 16


@torch.inference_mode()
def all_logits(models: list, prompt: torch.Tensor) -> list[torch.Tensor]:
    """Per model, its logits [B, T + DECODE_STEPS, vocabulary]; ``models[-1]`` drives the decode."""
    outputs = [model(input_ids=prompt, use_cache=True) for model in models]
    logits = [[output.logits] for output in outputs]
    for _ in range(DECODE_STEPS):
        fed = outputs[-1].logits[:, -1:].argmax(-1)
        outputs = [model(input_ids=fed, past_key_values=output.past_key_values, use_cache=True) for model, output in zip(models, outputs)]
        for kept, output in zip(logits, outputs):
            kept.append(output.logits)
    return [torch.cat(kept, dim=1) for kept in logits]


def float32_step(value: torch.Tensor) -> torch.Tensor:
    """Distance from each |value| to the next float32."""
    magnitude = value.abs().float()
    return (torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude).double()


def main(load_pythia, device: str = "cpu", options=(), describe=None) -> int:
    """``load_pythia(model_dir, export_dir, ffn, **leaf options)`` is the loader of the leaf; ``options`` are
    its extra command-line options, as (flag, argparse keywords) pairs, passed to the HFFN arm; ``describe(model)``
    returns what the leaf prints of its HFFN runtime (a string) or is None."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true",
                        help="also run the check on an HFFN arm with a fault injected; it must report unexplained tokens")
    for flag, keywords in options:
        parser.add_argument(flag, **keywords)
    args = parser.parse_args()
    extra = {flag.lstrip("-").replace("-", "_"): getattr(args, flag.lstrip("-").replace("-", "_")) for flag, _ in options}
    dense = load_pythia(args.model_dir, args.export_dir, "dense")
    hffn = load_pythia(args.model_dir, args.export_dir, "hffn", **extra)
    arbiter = load_pythia(args.model_dir, args.export_dir, "dense").double()
    stream = torch.tensor(json.loads(TOKENS.read_text())["tokens"], device=device)
    unresolved = check_arm(dense, hffn, arbiter, stream)
    note = f"{describe(hffn)}; " if describe else ""
    print(f"{args.export_dir.parent.name}: {note}{unresolved} unexplained HFFN tokens -> {'PASSED' if unresolved == 0 else 'FAILED'}")
    code = 0 if unresolved == 0 else 1
    if args.negative_control:
        faulty = load_pythia(args.model_dir, args.export_dir, "hffn", **extra)
        inject_fault(faulty)
        found = check_arm(dense, faulty, arbiter, stream)
        print(f"negative control (attention.dense.weight of layer {FAULT_LAYER} of the HFFN arm x {FAULT_SCALE}): "
              f"{found} unexplained tokens -> {'DETECTED' if found > 0 else 'NOT DETECTED, the band hides faults: gate FAILED'}")
        code = code or (0 if found > 0 else 1)
    return code


FAULT_LAYER, FAULT_SCALE = 3, 1.5


@torch.inference_mode()
def inject_fault(model) -> None:
    """Predeclared fault of the negative control: one dense weight of the HFFN arm only, scaled."""
    model.gpt_neox.layers[FAULT_LAYER].attention.dense.weight.mul_(FAULT_SCALE)


def check_arm(dense, hffn, arbiter, stream) -> int:
    """The check of one HFFN arm against the dense float32 control and the float64 arbiter; returns the number of
    unexplained tokens and prints the three counts per shape and the largest band in absolute logit units."""
    start = unresolved = 0
    largest = 0.0
    for batch, length in SHAPES:
        prompt = stream[start:start + batch * length].view(batch, length)
        start += batch * length
        dense_logits, hffn_logits, arbiter_logits = all_logits([dense, hffn, arbiter], prompt)
        dense_tokens, hffn_tokens, arbiter_tokens = dense_logits.argmax(-1), hffn_logits.argmax(-1), arbiter_logits.argmax(-1)
        differs = hffn_tokens != dense_tokens
        moved = differs & (hffn_tokens != arbiter_tokens)
        best = arbiter_logits.gather(-1, arbiter_tokens.unsqueeze(-1)).squeeze(-1)
        chosen = arbiter_logits.gather(-1, hffn_tokens.unsqueeze(-1)).squeeze(-1)
        margin, step = best - chosen, float32_step(best)
        dense64 = dense_logits.double()
        control_error = ((dense64.gather(-1, arbiter_tokens.unsqueeze(-1)).squeeze(-1) - best).abs()
                         + (dense64.gather(-1, hffn_tokens.unsqueeze(-1)).squeeze(-1) - chosen).abs())
        band = torch.maximum(step, control_error)
        wrong = moved & (margin > band)
        one_step = moved & (margin <= step)
        by_band = moved & (margin > step) & ~wrong
        unresolved += int(wrong.sum())
        largest = max(largest, float(band.max()))
        gap = float((hffn_logits[:, :length] - dense_logits[:, :length]).abs().max())
        print(f"B={batch} T={length}: {hffn_tokens.numel()} tokens, hffn != dense32 {int(differs.sum())} (of which != float64: {int(moved.sum())}, "
              f"within one float32 step: {int(one_step.sum())}, within the control band: {int(by_band.sum())}, "
              f"unexplained: {int(wrong.sum())}), dense32 != float64 {int((dense_tokens != arbiter_tokens).sum())}; "
              f"prefill max|dlogit| {gap:.2e} ({gap / float(dense_logits[:, :length].abs().max()):.1e} of max|logit|); "
              f"max band {float(band.max()):.2e} logit units", flush=True)
    print(f"largest band over all positions: {largest:.3e} logit units", flush=True)
    return unresolved
