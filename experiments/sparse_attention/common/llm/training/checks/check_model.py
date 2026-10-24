#!/usr/bin/env python3
"""Checks before any training (PLAN section 7, last paragraph). Exits non-zero on any failure.

For every arm: parameter counts; supports inside S_g with the right degree; entries outside the masks stay exactly
zero through optimizer steps; o_t exactly zero outside S_g; FFN output independent of u_t[j], j outside S_g, except
through the RMS; packed and selected implementations identical within float32 rounding (outputs and gradients);
causality (no leak from future tokens); determinism. Plus the balanced byte hash and the router mechanics.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.arms import ARMS, get_arm  # noqa: E402
from lib.data import Enwik8, balanced_byte_hash  # noqa: E402
from lib.model import SparseAttnLM, rms_norm  # noqa: E402
from lib.train import cross_block_zeroed, evaluate  # noqa: E402

FAILS = []
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def batch(seed=0, B=8, N=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, N), generator=g)


def model_for(arm, seed=1, ht=None, **kw):
    cfg = get_arm(arm) if isinstance(arm, str) else arm
    if cfg.routing == "hash" and ht is None:
        ht = torch.arange(256) % cfg.groups
    return SparseAttnLM(cfg, seed, hash_table=ht, **kw)


def expected_params(cfg):
    M, H, k, G, m = cfg.d_model, cfg.hidden, cfg.degree, cfg.groups, cfg.m
    attn = 0
    if cfg.attn != "none":
        qkv = 3 * (G * m * m if cfg.qkv_blockdiag else M * M)
        wo = G * m * m if cfg.wo_blockdiag else M * M
        attn = qkv + wo + M  # + norm1 gain
    ffn = 2 * H * k + H + M + M  # up, down values, b_up, b_down, norm2 gain
    router = G * M if cfg.routing == "learned" else 0
    return cfg.n_layers * (attn + ffn + router) + M  # + final norm gain


def check_params():
    plan = {"E0": 93184, "E1": 93184, "E2": 93184, "E3a": 80896, "E3b": 80896}
    for name, cfg in ARMS.items():
        n = model_for(cfg).count_params()
        exp = expected_params(cfg)
        check(f"params {name}", n == exp, f"{n} (formula {exp}" + (f", PLAN table {plan[name]}" if name in plan else "") + ")")
        if name in plan:
            check(f"params {name} equals PLAN section 7 table", n == plan[name], f"{n} vs {plan[name]}")


def check_supports():
    for name, cfg in ARMS.items():
        m = model_for(cfg, seed=3)
        G, mm, n, k = cfg.groups, cfg.m, cfg.n, cfg.degree
        ok_deg, ok_in, ok_out = True, True, True
        for layer in m.layers:
            um, dm = layer.mask_w_up, layer.mask_w_down
            ok_deg &= bool((um.sum(1) == k).all()) and bool((dm.sum(0) == k).all())
            if cfg.ffn_support in ("blocks", "ind"):
                for g in range(G):
                    rows = um[g * n:(g + 1) * n]
                    ok_in &= bool(rows[:, :g * mm].sum() == 0 and rows[:, (g + 1) * mm:].sum() == 0)
            if cfg.ffn_support == "ind":
                for g in range(G):
                    cols = dm[:, g * n:(g + 1) * n]
                    ok_out &= bool(cols[:g * mm].sum() == 0 and cols[(g + 1) * mm:].sum() == 0)
            for pn, mn in layer._masked.items():
                ok_deg &= bool((getattr(layer, pn).detach() * (1 - getattr(layer, mn))).abs().sum() == 0)
        check(f"supports {name}: degree {k} per neuron (up fan-in, down fan-out)", ok_deg)
        if cfg.ffn_support in ("blocks", "ind"):
            check(f"supports {name}: up_idx of R_g inside S_g", ok_in)
        if cfg.ffn_support == "ind":
            check(f"supports {name}: down_idx of R_g inside S_g", ok_out)


def check_zero_outside_masks():
    for name in ("E0", "E3b", "IND", "E4b"):
        cfg = get_arm(name)
        m = model_for(cfg, seed=2)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
        x = batch(5)
        for _ in range(3):
            logits, _ = m(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        leak = 0.0
        for layer in m.layers:
            for pn, mn in layer._masked.items():
                leak += float((getattr(layer, pn).detach() * (1 - getattr(layer, mn))).abs().sum())
        check(f"entries outside the masks stay exactly zero after 3 AdamW steps {name}", leak == 0.0, f"sum|w|={leak}")


def run_parts(model, x, impl):
    model.impl = impl
    emb = model.emb(x)
    N = x.shape[1]
    return model.layers[0].parts(emb, x, model.cos[:N], model.sin[:N], model.hash_table, impl)


def check_o_support():
    for name in ("E3b", "IND", "E4a", "E4b", "E3b_L4", "E3b_L4rot"):
        cfg = get_arm(name)
        m = model_for(cfg)
        x = batch(7)
        m.eval()
        impls = ["selected"] + (["packed"] if cfg.routing == "position" else [])
        for impl in impls:
            p = run_parts(m, x, impl)
            B, N = x.shape
            if p["route"] is not None:
                route = p["route"]
            else:
                route = ((torch.arange(N) + m.layers[0].shift()) % cfg.groups)[None].expand(B, N)
            col_group = torch.arange(cfg.d_model) // cfg.m
            outside = (col_group[None, None, :] != route[..., None]) | (route[..., None] < 0)
            bad = float((p["o"].detach().abs() * outside).max())
            check(f"o_t exactly zero outside S_g {name} [{impl}]", bad == 0.0, f"max|o| outside = {bad}")


def check_ffn_independence():
    for name in ("E2", "E3a", "E3b", "IND"):
        cfg = get_arm(name)
        m = model_for(cfg, seed=4)
        layer = m.layers[0]
        torch.manual_seed(0)
        layer.g2.data = torch.rand(cfg.d_model) + 0.5
        B, N, M = 4, 16, cfg.d_model
        u = torch.randn(B, N, M)
        route = (torch.arange(N) % cfg.groups)[None].expand(B, N)
        y0 = layer.ffn_selected(rms_norm(u) * layer.g2, route)
        worst_out, worst_in = 0.0, 0.0
        for t in range(N):
            g = t % cfg.groups
            outside = [j for j in range(M) if j // cfg.m != g]
            inside = [j for j in range(M) if j // cfg.m == g]
            u2 = u.clone()
            j1, j2 = outside[0], outside[-1]  # swap two coordinates outside S_g: the RMS is unchanged
            u2[:, t, [j1, j2]] = u[:, t, [j2, j1]]
            u3 = u.clone()
            i1, i2 = inside[0], inside[-1]
            u3[:, t, [i1, i2]] = u[:, t, [i2, i1]]
            y2 = layer.ffn_selected(rms_norm(u2) * layer.g2, route)
            y3 = layer.ffn_selected(rms_norm(u3) * layer.g2, route)
            worst_out = max(worst_out, float((y2[:, t] - y0[:, t]).abs().max()))
            worst_in = max(worst_in, float((y3[:, t] - y0[:, t]).abs().max()))
        check(f"FFN output of t independent of u_t[j], j outside S_g (RMS held fixed) {name}", worst_out < 1e-5,
              f"max change {worst_out:.2e}; control (swap inside S_g) changes by {worst_in:.2e}")
        check(f"  control: a swap inside S_g changes the FFN output {name}", worst_in > 1e-3)


def check_packed_selected():
    for name in ("E2", "E3a", "E3b", "IND", "E3b_L4", "E3b_L4rot", "E0", "E1"):
        cfg = get_arm(name)
        x = batch(11)
        y = batch(12)
        outs = {}
        for impl in ("packed", "selected"):
            m = model_for(cfg, seed=6, impl=impl)
            m.train()
            logits, _ = m(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
            outs[impl] = (logits.detach(), grads, float(loss))
        dl = float((outs["packed"][0] - outs["selected"][0]).abs().max())
        dg = float((outs["packed"][1] - outs["selected"][1]).abs().max())
        scale = float(outs["selected"][1].abs().max())
        check(f"packed == selected, logits {name}", dl < 1e-4, f"max|diff| {dl:.2e}")
        check(f"packed == selected, gradients {name}", dg < 1e-4 * max(1.0, scale), f"max|diff| {dg:.2e} (max|grad| {scale:.2e})")


def check_causality():
    for name, cfg in ARMS.items():
        for impl in ("packed", "selected"):
            if impl == "packed" and cfg.routing != "position":
                continue
            m = model_for(cfg, seed=8, impl=impl)
            m.eval()
            x = batch(21, B=4)
            x2 = x.clone()
            x2[:, 90:] = torch.randint(0, 256, x2[:, 90:].shape)
            with torch.no_grad():
                a, _ = m(x)
                b, _ = m(x2)
            d = float((a[:, :90] - b[:, :90]).abs().max())
            check(f"causality {name} [{impl}]", d < 1e-5, f"max|diff| on positions < 90: {d:.2e}")


def check_determinism():
    for name in ("E0", "E3b", "E4b"):
        x = batch(31)
        a = model_for(name, seed=9)(x)[0]
        b = model_for(name, seed=9)(x)[0]
        check(f"determinism of the initialisation and forward {name}", bool((a == b).all()))
    # shared tensors identical across arms (paired comparisons)
    e0, e3 = model_for("E0", seed=5), model_for("E3b", seed=5)
    same = all(bool((getattr(e0.layers[0], n) == getattr(e3.layers[0], n)).all()) for n in ("wq", "wk", "wv", "g1", "g2")) \
        and bool((e0.emb.weight == e3.emb.weight).all())
    check("shared tensors (embedding, W_Q, W_K, W_V, norms) identical across arms for a seed", same)
    e1, e2 = model_for("E1", seed=5), model_for("E2", seed=5)
    same = all(bool((getattr(e1.layers[0], n) == getattr(e2.layers[0], n)).all()) for n in ("w_up", "w_down", "wo"))
    check("E1 and E2 have the same supports and values", same)


def check_hash_and_router():
    try:
        data = Enwik8()
        counts = data.train_counts()
        table = balanced_byte_hash(counts, 4)
        share = [float(counts[table == g].sum() / counts.sum()) for g in range(4)]
        check("balanced byte hash: each group receives about a quarter of the training bytes",
              all(abs(s - 0.25) < 0.01 for s in share), f"shares {[round(s, 4) for s in share]}")
    except SystemExit as e:
        check("balanced byte hash (needs enwik8)", False, str(e))
    cfg = get_arm("E4b")
    m = model_for(cfg, seed=2)
    x = batch(41, B=16)
    p = run_parts(m, x, "selected")
    r = p["route"]
    check("learned router: routes in {-1..G-1}, capacity respected",
          bool(((r >= -1) & (r < 4)).all()) and all(int((r == g).sum(1).max()) <= 40 for g in range(4)),
          f"capacity {-(-int(1.25 * 128) // 4)}, max per window {[int((r == g).sum(1).max()) for g in range(4)]}")
    dropped = r < 0
    check("learned router: dropped tokens receive no attention and no FFN contribution",
          float(p["o"][dropped].abs().max() if dropped.any() else 0.0) == 0.0
          and float(p["y"][dropped].abs().max() if dropped.any() else 0.0) == 0.0,
          f"dropped fraction {float(dropped.float().mean()):.3f}")
    m.train()
    logits, stats = m(x)
    F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1)).backward()
    g = m.layers[0].w_route.grad
    check("learned router receives a gradient through the gate", g is not None and float(g.abs().sum()) > 0)
    check("balance loss is 1 for a perfectly balanced router, >= 1 otherwise",
          float(stats[0]["balance_loss"]) >= 0.99, f"{float(stats[0]['balance_loss']):.3f}")
    # hash routing: route depends on the byte of the token itself only
    h = model_for("E4a", seed=2)
    xh = batch(42)
    ph = run_parts(h, xh, "selected")
    check("hash routing: route = table[byte of the token]", bool((ph["route"] == h.hash_table[xh]).all()))
    hr = model_for("E4a_L4rot", seed=2)
    l1 = hr.layers[1].parts(hr.emb(xh), xh, hr.cos, hr.sin, hr.hash_table, "selected")
    check("hash routing with rotation: route at layer l = (table[byte] + l) mod G",
          bool((l1["route"] == (hr.hash_table[xh] + 1) % 4).all()))


def check_mixing_zero_for_ind():
    m = model_for("IND", seed=3)
    for layer in m.layers:  # make the weights non-trivial
        pass
    x = batch(51, B=8)
    y = batch(52, B=8)
    base = evaluate(m, x, y)["bpb"]
    ok = True
    for which in (("wq",), ("wk", "wv"), ("w_down",)):
        with cross_block_zeroed(m, which):
            ok &= evaluate(m, x, y)["bpb"] == base
    check("IND: zeroing the cross-block entries changes nothing (they are zero by construction)", ok)
    e3 = model_for("E3b", seed=3)
    b0 = evaluate(e3, x, y)["bpb"]
    with cross_block_zeroed(e3, ("wq",)):
        b1 = evaluate(e3, x, y)["bpb"]
    check("E3b: the mixing diagnostic does modify the model", b1 != b0, f"{b1 - b0:+.4f}")


def main():
    t0 = time.time()
    torch.manual_seed(0)
    for fn in (check_params, check_supports, check_zero_outside_masks, check_o_support, check_ffn_independence,
               check_packed_selected, check_causality, check_determinism, check_hash_and_router,
               check_mixing_zero_for_ind):
        print(f"--- {fn.__name__}", flush=True)
        fn()
    out = {"checks": RESULTS, "failed": FAILS, "wall_s": time.time() - t0}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"{len(RESULTS) - len(FAILS)}/{len(RESULTS)} checks passed", flush=True)
    if FAILS:
        print("FAILED:", *FAILS, sep="\n  ", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
