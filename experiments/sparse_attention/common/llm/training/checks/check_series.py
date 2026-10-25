#!/usr/bin/env python3
"""Checks of the preparation of the second series (E6-E9), PLAN section 14 and CLOUD_E6_E9.md. Exits non-zero on failure.

Covers: arms and parameter counts (independent of the context), IND under the hash route, tokens per step, the WSD
schedule, plateau checkpoint and decay branch against an uninterrupted run, position ranges, the fixed evaluation size,
the counter against PLAN 14.5, and the reproducibility of the first ladder (a cosine run of E0 is bit-identical to its
recorded smoke).
"""
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import train  # noqa: E402
from lib.arms import ARMS, SERIES, SERIES_N512, get_arm  # noqa: E402
from lib.count import count_arm  # noqa: E402
from lib.data import BatchStream, Enwik8  # noqa: E402
from lib.model import SparseAttnLM  # noqa: E402

FAILS, RESULTS = [], []
TRAIN = Path(__file__).resolve().parents[1]


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def check_params():
    plan = {"REF2": 186240, "REF3": 279296, "REF4": 372352, "E4a_L2": 161664, "E4a_L4": 323200, "E4a_L6": 484736}
    for name, exp in plan.items():
        n = SparseAttnLM(get_arm(name), 0).count_params()
        check(f"params {name} equal to PLAN 14.5", n == exp, f"{n} vs {exp}")
    check("REF1 is E0 (same configuration but its name)",
          get_arm("REF1").with_(name="E0") == get_arm("E0"))
    for a in SERIES:
        n128 = SparseAttnLM(a, 0).count_params()
        b = get_arm(a.name + "_N512")
        n512 = SparseAttnLM(b, 0).count_params()
        check(f"params independent of the context {a.name}", n128 == n512 and b.context == 512, f"{n128} / {n512}")


def check_ind_hash():
    cfg = get_arm("IND_L4")
    ok_cfg = cfg.routing == "hash" and cfg.n_layers == 4 and cfg.qkv_blockdiag and cfg.ffn_support == "ind"
    check("IND_L4 is IND with the hash route at 4 layers", ok_cfg)
    ht = torch.arange(256) % 4
    m = SparseAttnLM(cfg, 3, hash_table=ht)
    G, mm, n, k = cfg.groups, cfg.m, cfg.n, cfg.degree
    ok_deg = ok_in = ok_out = ok_qkv = True
    for layer in m.layers:
        um, dm = layer.mask_w_up, layer.mask_w_down
        ok_deg &= bool((um.sum(1) == k).all()) and bool((dm.sum(0) == k).all())
        for g in range(G):
            rows, cols = um[g * n:(g + 1) * n], dm[:, g * n:(g + 1) * n]
            ok_in &= bool(rows[:, :g * mm].sum() == 0 and rows[:, (g + 1) * mm:].sum() == 0)
            ok_out &= bool(cols[:g * mm].sum() == 0 and cols[(g + 1) * mm:].sum() == 0)
        for nm in ("wq", "wk", "wv", "wo"):
            ok_qkv &= bool((getattr(layer, nm).detach() * (1 - getattr(layer, "mask_" + nm))).abs().sum() == 0)
    check("IND_L4 under the hash route: degree 26, up_idx and down_idx inside S_g, W_Q/W_K/W_V/W_O block-diagonal",
          ok_deg and ok_in and ok_out and ok_qkv)
    x = torch.randint(0, 256, (4, 128))
    emb = m.emb(x)
    h = emb
    ok_o = True
    for layer in m.layers:
        p = layer.parts(h, x, m.cos, m.sin, m.hash_table, "selected")
        route = p["route"]
        outside = (torch.arange(cfg.d_model) // mm)[None, None, :] != route[..., None]
        ok_o &= float((p["o"].detach().abs() * outside).max()) == 0.0
        h = p["u"] + p["y"] + layer.b_down
    check("IND_L4 under the hash route: o_t exactly zero outside S_g at every layer", ok_o)


def check_tokens_per_step():
    data = Enwik8()
    for ctx in (128, 512):
        b = 8192 // ctx
        x, y = BatchStream(data, 0, b, ctx).next()
        check(f"8,192 bytes per step at context {ctx} (batch {b})", x.numel() == 8192 and y.shape == (b, ctx))
        s1, s2 = BatchStream(data, 5, b, ctx), BatchStream(data, 5, b, ctx)
        check(f"same seed and context, same batches in the same order, whatever the arm (context {ctx})",
              all(bool((p == q).all()) for _ in range(3) for p, q in zip(s1.next(), s2.next())))
        st = s1.get_state()
        nxt = s1.next()
        s1.set_state(st)
        check(f"stream position restorable (context {ctx})", all(bool((p == q).all()) for p, q in zip(nxt, s1.next())))
    # the batch of the first ladder is untouched: next(train_batches) equals BatchStream(…, 64, 128)
    g = data.train_batches(2, 64, 128)
    s = BatchStream(data, 2, 64, 128)
    check("first-ladder batches (64 x 128) unchanged", all(bool((p == q).all()) for p, q in zip(next(g), s.next())))


def check_wsd():
    H, peak = 1000, 0.01
    lrs = [train.lr_wsd(s, H, peak) for s in range(H + 1)]
    ok = abs(lrs[0] - peak / 200) < 1e-12 and abs(lrs[199] - peak) < 1e-12
    ok &= all(abs(lrs[s] - peak) < 1e-12 for s in range(199, 800))
    ok &= all(lrs[s] >= lrs[s + 1] for s in range(800, H))
    ok &= abs(lrs[H]) < 1e-12 and lrs[H - 1] > 0 and abs(lrs[900] - peak / 2) < 1e-9
    check("WSD: 200 warm-up steps, constant peak to 0.8 H, linear decay to zero at H", ok,
          f"lr[0]={lrs[0]:.2e} lr[199]={lrs[199]:.4f} lr[900]={lrs[900]:.4f} lr[{H}]={lrs[H]:.1e}")
    # the plateau of a longer horizon is the plateau of every shorter one at its own 0.8 H
    check("WSD: branch checkpoints 1600/3200/6400 are the starts of the decay of horizons 2000/4000/8000",
          all(abs(train.lr_wsd(c, h, 1.0) - 1.0) < 1e-12 and train.lr_wsd(c + 1, h, 1.0) < 1.0
              for c, h in ((1600, 2000), (3200, 4000), (6400, 8000))))


def cli(arm, seed, steps, lr, out, *extra):
    return train.main(["--arm", arm, "--seed", str(seed), "--steps", str(steps), "--lr", str(lr), "--out", str(out),
                       "--threads", "1", "--final-windows", "256", "--log-every", "20", "--eval-every", "1000",
                       *extra])


def check_branch():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        common = ["--schedule", "wsd", "--warmup", "10", "--horizon", "100", "--tokens-per-step", "8192"]
        for arm in ("E4a", "E4b"):
            rc1 = cli(arm, 0, 100, 0.01, tmp / f"{arm}_full", *common)
            rc2 = cli(arm, 0, 80, 0.01, tmp / f"{arm}_plateau", *common, "--save-at", "80")
            rc3 = cli(arm, 0, 100, 0.01, tmp / f"{arm}_branch", *common, "--resume", str(tmp / f"{arm}_plateau" / "ckpt_80.pt"))
            a = torch.load(tmp / f"{arm}_full" / "model.pt")
            b = torch.load(tmp / f"{arm}_branch" / "model.pt")
            diff = max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
            sa = json.loads((tmp / f"{arm}_full" / "summary.json").read_text())
            sb = json.loads((tmp / f"{arm}_branch" / "summary.json").read_text())
            dbpb = abs(sa["final_valid_bpb"] - sb["final_valid_bpb"])
            check(f"decay branch from the step-80 checkpoint equals an uninterrupted WSD run (H=100) {arm}",
                  rc1 == rc2 == rc3 == 0 and diff <= 1e-6 and dbpb <= 1e-6,
                  f"max|weights diff| {diff:.2e}, |bpb diff| {dbpb:.2e} (tolerance 1e-6 float32)")
        cfg = json.loads((tmp / "E4a_full" / "config.json").read_text())
        check("config records schedule, horizon and tokens per step",
              cfg["schedule"] == "wsd" and cfg["horizon"] == 100 and cfg["tokens_per_step"] == 8192)


def check_context512_and_eval_size():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rc = cli("E4a_L2_N512", 0, 4, 0.005, tmp / "n512", "--schedule", "wsd", "--warmup", "2", "--tokens-per-step", "8192",
                 "--valid-bytes", "524288", "--eval-every", "2")
        cfg = json.loads((tmp / "n512" / "config.json").read_text())
        s = json.loads((tmp / "n512" / "summary.json").read_text())
        check("N=512: batch 16, 1,024 intermediate windows (524,288 bytes)",
              rc == 0 and cfg["batch"] == 16 and cfg["valid_windows_in_training"] == 1024,
              f"batch {cfg['batch']}, windows {cfg['valid_windows_in_training']}")
        fv = s["final_valid"]
        check("N=512: bits per byte by position range 0-127, 128-255, 256-511",
              fv["range_names"] == ["0-127", "128-255", "256-511"] and len(fv["bpb_by_range"]) == 3)
        w = fv["windows"]
        mean = sum(r * w * ln for r, ln in zip(fv["bpb_by_range"], fv["range_lengths"])) / (w * 512)
        check("N=512: the mean of the ranges weighted by their counts equals the whole figure",
              abs(mean - fv["bpb"]) < 1e-9, f"{mean:.9f} vs {fv['bpb']:.9f}")
        rc = cli("E4a", 0, 4, 0.005, tmp / "n128", "--tokens-per-step", "8192", "--valid-bytes", "524288", "--eval-every", "2")
        cfg = json.loads((tmp / "n128" / "config.json").read_text())
        s = json.loads((tmp / "n128" / "summary.json").read_text())
        check("N=128: batch 64, 4,096 intermediate windows (524,288 bytes), no ranges",
              rc == 0 and cfg["batch"] == 64 and cfg["valid_windows_in_training"] == 4096 and "bpb_by_range" not in s["final_valid"])


def check_counter():
    plan = {"E4a_L2": (138240, 187392), "REF1": (157696, 256000), "E4a_L4": (243712, 342016), "REF2": (282624, 479232),
            "E4a_L6": (349184, 496640), "REF3": (407552, 702464), "REF4": (532480, 925696)}
    for name, (a, b) in plan.items():
        x, y = count_arm(name)["inference_macs_per_token"], count_arm(name + "_N512")["inference_macs_per_token"]
        check(f"counter: inference multiply-adds {name} equal to PLAN 14.5 at both contexts", (x, y) == (a, b), f"{x} / {y}")
    one = count_arm("REF1")
    check("counter: one dense layer 124,928 and one constrained layer 52,736 at N=128 (PLAN section 6)",
          sum(one["inference_macs_per_layer_terms"].values()) == 124928
          and sum(count_arm("E4a_L2")["inference_macs_per_layer_terms"].values()) == 52736)
    ind = count_arm("IND_L4")
    check("counter: IND_L4 counted", ind["inference_macs_per_token"] == 4 * 25088 + 32768,
          f"{ind['inference_macs_per_token']} (N=128), {count_arm('IND_L4_N512')['inference_macs_per_token']} (N=512)")


def check_first_ladder_unchanged():
    ref = sorted(glob.glob(str(TRAIN / "e0" / "runs" / "smoke_*" / "smoke" / "E0_s0" / "summary.json")))
    if not ref:
        check("first ladder: recorded smoke of E0 available", False)
        return
    rec = json.loads(Path(ref[0]).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        rc = train.main(["--arm", "E0", "--seed", "0", "--steps", "40", "--lr", "0.002", "--out", tmp, "--threads", "1",
                         "--eval-every", "20", "--valid-windows", "256", "--final-windows", "512"])
        new = json.loads((Path(tmp) / "summary.json").read_text())
    check("first ladder reproducible: the cosine run of the E0 smoke is bit-identical after the changes",
          rc == 0 and new["final_valid_bpb"] == rec["final_valid_bpb"], f"{new['final_valid_bpb']!r} vs {rec['final_valid_bpb']!r}")


def main():
    t0 = time.time()
    torch.manual_seed(0)
    for fn in (check_params, check_ind_hash, check_tokens_per_step, check_wsd, check_counter,
               check_context512_and_eval_size, check_branch, check_first_ladder_unchanged):
        print(f"--- {fn.__name__}", flush=True)
        fn()
    out = {"checks": RESULTS, "failed": FAILS, "wall_s": time.time() - t0}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"{len(RESULTS) - len(FAILS)}/{len(RESULTS)} checks passed", flush=True)
    if FAILS:
        print("FAILED:", *FAILS, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
