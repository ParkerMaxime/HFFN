"""Training loop (PLAN section 7), evaluation (bits per byte, position classes, router load, mixing diagnostic)
and the single-run entry point `python -m lib.train` used by the step launchers.

A run directory receives: config.json, provenance.json, raw.jsonl, metrics.csv, summary.json, run.log (written by the
launcher), EXIT. Nothing in a run directory is edited afterwards.
"""
import argparse
import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arms import get_arm
from .data import BatchStream, Enwik8, balanced_byte_hash, repo_root
from .model import SparseAttnLM, block_mask

LN2 = math.log(2.0)
BATCH = 64


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, x, y, batch=64):
    """Bits per byte over all positions of the windows, by position class t mod 4, and router statistics."""
    model.eval()
    n, N = x.shape
    tot = torch.zeros(4, dtype=torch.float64)
    cnt = torch.zeros(4, dtype=torch.float64)
    pos_tot = torch.zeros(N, dtype=torch.float64)
    load = None
    dropped = 0.0
    nb = 0
    for i in range(0, n, batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        logits, stats = model(xb)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="none").view(xb.shape)
        per_pos = ce.double().sum(0)  # (N,)
        pos_tot += per_pos
        for c in range(4):
            tot[c] += per_pos[c::4].sum()
            cnt[c] += len(xb) * len(range(c, N, 4))
        for s in stats:
            if "load" in s:
                load = s["load"].double() * len(xb) if load is None else load + s["load"].double() * len(xb)
                dropped += float(s["dropped"]) * len(xb) if "dropped" in s else 0.0
        nb += len(xb)
    model.train()
    out = {
        "bpb": float(tot.sum() / cnt.sum() / LN2),
        "bpb_by_class": [float(tot[c] / cnt[c] / LN2) for c in range(4)],
        "windows": n,
    }
    if N > 128:  # bits per byte by position range 0-127, 128-255, 256-(N-1); the mean weighted by counts is the whole
        edges = [0, 128, 256, N] if N > 256 else [0, 128, N]
        out["range_names"] = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])]
        out["range_lengths"] = [b - a for a, b in zip(edges[:-1], edges[1:])]
        out["bpb_by_range"] = [float(pos_tot[a:b].sum() / (n * (b - a)) / LN2) for a, b in zip(edges[:-1], edges[1:])]
    if load is not None:
        out["router_load"] = [float(v) for v in load / nb]
        out["router_dropped"] = dropped / nb
    return out


@contextlib.contextmanager
def cross_block_zeroed(model, which):
    """Zero the cross-block entries (those connecting S_j to the group g != j) of the named matrices (PLAN section 5)."""
    cfg = model.cfg
    saved = []
    bm = block_mask(cfg.d_model, cfg.groups)  # (M, M) keep mask for W_Q/W_K/W_V rows (heads of g) x columns (S_g)
    keep_down = (torch.arange(cfg.d_model)[:, None] // cfg.m == torch.arange(cfg.hidden)[None, :] // cfg.n).float()
    for layer in model.layers:
        if cfg.attn == "none":
            continue
        for name, keep in (("wq", bm), ("wk", bm), ("wv", bm), ("w_down", keep_down)):
            if name in which:
                p = getattr(layer, name)
                saved.append((p, p.data.clone()))
                p.data.mul_(keep)
    try:
        yield
    finally:
        for p, d in saved:
            p.data.copy_(d)


def mixing_diagnostic(model, x, y):
    """Increase of the validation loss (bits per byte) when the cross-block part of each path is zeroed."""
    base = evaluate(model, x, y)["bpb"]
    out = {"base_bpb": base}
    for label, which in (("q", ("wq",)), ("kv", ("wk", "wv")), ("down", ("w_down",)),
                         ("all", ("wq", "wk", "wv", "w_down"))):
        with cross_block_zeroed(model, which):
            out[f"delta_{label}"] = evaluate(model, x, y)["bpb"] - base
    return out


# --------------------------------------------------------------------------- training
def lr_at(step, steps, peak, warmup, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def lr_wsd(step, horizon, peak, warmup=200, decay_frac=0.2):
    """Warm-up, constant peak, linear decay to zero over the last `decay_frac` of the horizon."""
    if step < warmup:
        return peak * (step + 1) / warmup
    start = (1.0 - decay_frac) * horizon
    if step < start:
        return peak
    return peak * max(0.0, (horizon - step) / (horizon - start))


def provenance():
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=repo_root(), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": (git("status", "--porcelain") or "").count("\n") + (1 if git("status", "--porcelain") else 0),
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_checkpoint(path, model, opt, step, stream, ema, args):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "stream": stream.get_state(),
                "torch_rng": torch.get_rng_state(), "ema": ema, "arm": args.arm, "seed": args.seed}, path)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = Enwik8()
    overrides = json.loads(args.arm_overrides) if args.arm_overrides else {}
    cfg = get_arm(args.arm, **overrides)
    ht = balanced_byte_hash(data.train_counts(), cfg.groups) if cfg.routing == "hash" else None
    model = SparseAttnLM(cfg, args.seed, hash_table=ht, impl=args.impl)
    batch = args.tokens_per_step // cfg.context if args.tokens_per_step else BATCH
    valid_windows = args.valid_bytes // cfg.context if args.valid_bytes else args.valid_windows
    horizon = args.horizon if args.horizon else args.steps
    vx, vy = data.eval_windows("valid", cfg.context, valid_windows)
    fx, fy = data.eval_windows("valid", cfg.context, args.final_windows)
    if args.schedule == "wsd":
        warmup = args.warmup if args.warmup is not None else 200
    else:
        warmup = args.warmup if args.warmup is not None else max(1, int(0.05 * args.steps))
    decay = model.decay_params()
    ids = {id(p) for p in decay}
    other = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": other, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    stream = BatchStream(data, args.seed, batch, cfg.context)
    start, ema = 0, None
    if args.resume:
        ck = torch.load(args.resume, weights_only=False)
        assert ck["arm"] == args.arm and ck["seed"] == args.seed, "checkpoint of another arm or seed"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        stream.set_state(ck["stream"])
        torch.set_rng_state(ck["torch_rng"])
        start, ema = ck["step"], ck["ema"]
    save_at = {int(v) for v in args.save_at.split(",")} if args.save_at else set()
    config = {"arm": cfg.__dict__, "seed": args.seed, "steps": args.steps, "lr": args.lr, "warmup": warmup,
              "batch": batch, "tokens_per_step": batch * cfg.context, "schedule": args.schedule, "horizon": horizon,
              "resume": args.resume, "start_step": start, "save_at": sorted(save_at),
              "weight_decay": args.weight_decay, "clip": 1.0, "impl": args.impl,
              "eval_every": args.eval_every, "valid_windows_in_training": len(vx),
              "params_outside_embedding": model.count_params(), "params_total": model.count_params(False),
              "hash_table": None if ht is None else [int(v) for v in ht]}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "provenance.json").write_text(json.dumps(provenance(), indent=1))
    raw = open(out / "raw.jsonl", "w", buffering=1)
    csv = open(out / "metrics.csv", "w", buffering=1)
    csv.write("step,lr,train_bpb,valid_bpb,wall_s\n")
    t0 = time.time()
    status, cause = "passed", None
    print(f"[{cfg.name} seed={args.seed} lr={args.lr}] params={config['params_outside_embedding']} "
          f"steps={start}->{args.steps} schedule={args.schedule} horizon={horizon} batch={batch}x{cfg.context}",
          flush=True)
    model.train()
    step = start - 1
    lr = args.lr
    for step in range(start, args.steps):
        lr = lr_wsd(step, horizon, args.lr, warmup) if args.schedule == "wsd" else lr_at(step, horizon, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.next()
        logits, stats = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        bal = [s["balance_loss"] for s in stats if "balance_loss" in s]
        if bal and cfg.balance_coef > 0:
            loss = loss + cfg.balance_coef * torch.stack(bal).mean()
        if not torch.isfinite(loss):
            status, cause = "failed", f"non-finite loss at step {step}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tb = float(ce) / LN2
        ema = tb if ema is None else 0.98 * ema + 0.02 * tb
        rec = None
        if (step + 1) % args.log_every == 0 or step == start:
            rec = {"step": step + 1, "lr": lr, "train_bpb": tb, "train_bpb_ema": ema,
                   "balance_loss": float(torch.stack(bal).mean()) if bal else None,
                   "wall_s": time.time() - t0}
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            ev = evaluate(model, vx, vy)
            rec = (rec or {"step": step + 1, "lr": lr, "wall_s": time.time() - t0}) | {"valid": ev}
            csv.write(f"{step + 1},{lr:.6g},{ema:.5f},{ev['bpb']:.5f},{time.time() - t0:.1f}\n")
        if (step + 1) in save_at:
            save_checkpoint(out / f"ckpt_{step + 1}.pt", model, opt, step + 1, stream, ema, args)
        if rec:
            raw.write(json.dumps(rec) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                              if k != "valid"}), flush=True)
    summary = {"arm": cfg.name, "seed": args.seed, "lr": args.lr, "steps_done": step + (status == "passed"),
               "steps": args.steps, "start_step": start, "horizon": horizon, "schedule": args.schedule,
               "status": status, "cause": cause,
               "params_outside_embedding": config["params_outside_embedding"],
               "train_wall_s": time.time() - t0}
    if status == "passed":
        t_eval = time.time()
        final = evaluate(model, fx, fy)
        summary["final_eval_wall_s"] = time.time() - t_eval
        summary["final_eval_windows"] = int(fx.shape[0])
        summary["final_valid"] = final
        summary["final_valid_bpb"] = final["bpb"]
        summary["train_bpb_ema_last"] = ema
        raw.write(json.dumps({"step": args.steps, "final_valid": final}) + "\n")
        csv.write(f"{args.steps},{lr:.6g},{ema:.5f},{final['bpb']:.5f},{time.time() - t0:.1f}\n")
        if args.mixing:
            mx, my = data.eval_windows("valid", cfg.context, valid_windows)
            summary["mixing"] = mixing_diagnostic(model, mx, my)
        if args.test:
            tx, ty = data.eval_windows("test", cfg.context, None)
            summary["final_test"] = evaluate(model, tx, ty)
        torch.save(model.state_dict(), out / "model.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    raw.close()
    csv.close()
    (out / "EXIT").write_text("0\n" if status == "passed" else "2\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("final_valid", "mixing")}), flush=True)
    return 0 if status == "passed" else 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--arm-overrides", default=None, help="JSON dict of ArmConfig overrides")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, required=True, help="last step of this process (global step number)")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--impl", default="packed", choices=["packed", "selected", "gather"],
                   help="gather = the packed form under the hash route (default under the hash route: selected)")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--valid-windows", type=int, default=4096)
    p.add_argument("--valid-bytes", type=int, default=None, help="intermediate evaluations on a fixed number of bytes")
    p.add_argument("--final-windows", type=int, default=None, help="default: the whole valid split")
    p.add_argument("--mixing", action="store_true")
    p.add_argument("--test", action="store_true")
    # second series (E6-E9); the defaults are the behaviour of E0-E5
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--horizon", type=int, default=None, help="horizon H of the schedule (default: --steps)")
    p.add_argument("--tokens-per-step", type=int, default=None, help="batch = tokens-per-step // context (default: 64)")
    p.add_argument("--save-at", default=None, help="comma-separated steps at which the full training state is saved")
    p.add_argument("--resume", default=None, help="checkpoint to continue from (a decay branch)")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
