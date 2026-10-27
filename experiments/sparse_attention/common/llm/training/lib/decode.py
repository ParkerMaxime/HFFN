"""Batched decoding path of E14 (PLAN 17.3 and 17.7): a KV cache per layer and per sequence, forward only, without gradient.

One decoded step = `B` sequences (64 windows of the valid split), one new token each at position `p = N / 2`, a cache of `p`
positions per sequence filled by an untimed forward of the first `p` positions. The timed step computes the keys and values of the
new token and writes them into the preallocated slot `p` (the same slot at every call, so that repeated calls do the same work),
then attends to the `p + 1` positions of its sequence. Weights are random (time does not depend on their values); the route of a
token (balanced byte hash) depends on its byte, so the bytes are those of the valid split.

Three layers of code, as in `executor.py`:
* `prefill_caches` and `decode_reference` (PyTorch, sequential, "selected" semantics: every head is computed and the heads of the
  other groups are masked): the independent reference of the checks, equal to the full forward at position `p`;
* scripted workers (`dec_*`): the packed decoded step of the constrained arms and of E2h, and the dense kinds, each worker owning the
  heads `H_g` (and their slice of the cache) and, for the tokens routed to it, the tail of the layer;
* `Decoder`: runs the workers one after the other (`parallel=False`: the sequential decoded step, used for the memory peak (e) and as
  the sequential reference of the executor) or as a fork-join of four workers (`parallel=True`: the executor of E11).

The packed form of the routed phases gathers the sequences routed to a group (padded to the largest count of the step, over the
groups: PLAN 17.3), including the slice of the cache of their heads; the per-group counts of the step are recorded.
"""
from typing import List

import torch
import torch.nn.functional as F

from .executor import (Executor, _rms, _rope, dense_tail_worker, logits_worker, run_dense_tail, run_logits)
from .model import apply_rope, rms_norm


# --------------------------------------------------------------------------- windows of a process
def decode_windows(data, N, B, process, split="valid"):
    """B windows of N bytes of the split at offsets drawn per process (seeded by the process), as (B, N) int64."""
    import numpy as np
    a, b = data.split(split)
    rng = np.random.RandomState(4242 + process)
    starts = rng.randint(a, b - N - 1, size=B)
    w = np.stack([np.asarray(data.data[s:s + N]) for s in starts]).astype(np.int64)
    return torch.from_numpy(w)


# --------------------------------------------------------------------------- reference (PyTorch, sequential)
@torch.no_grad()
def prefill_caches(model, prefix, capacity):
    """Per layer (K, V) of shape (B, n_heads, capacity, d_head), positions [0, p) filled (K after RoPE) by a forward of `prefix`."""
    cfg = model.cfg
    assert cfg.layer_routing == "fixed"
    B, p = prefix.shape
    x = model.emb(prefix)
    caches = []
    for layer in model.layers:
        xi = rms_norm(x) * layer.g1
        _, k, v = layer.qkv(xi)
        k = apply_rope(k, model.cos[:p], model.sin[:p])
        K = torch.zeros(B, cfg.n_heads, capacity, cfg.d_head)
        V = torch.zeros(B, cfg.n_heads, capacity, cfg.d_head)
        K[:, :, :p], V[:, :, :p] = k, v
        caches.append((K, V))
        x, _ = layer(x, prefix, model.cos[:p], model.sin[:p], model.hash_table, "selected")
    return caches


@torch.no_grad()
def decode_reference(model, tok, caches, p):
    """The decoded step as PyTorch ops, every head computed and masked by the route: logits (B, vocab) at position p. Writes slot p."""
    cfg = model.cfg
    B = tok.shape[0]
    x = model.emb(tok)[:, None, :]
    route = model.hash_table[tok] if cfg.routing == "hash" else None
    cos, sin = model.cos[p:p + 1], model.sin[p:p + 1]
    for layer, (K, V) in zip(model.layers, caches):
        xi = rms_norm(x) * layer.g1
        q, k, v = layer.qkv(xi)                                                  # (B, n_heads, 1, d_head)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        K[:, :, p], V[:, :, p] = k[:, :, 0], v[:, :, 0]
        a = F.scaled_dot_product_attention(q, K[:, :, :p + 1], V[:, :, :p + 1]).transpose(1, 2)   # (B, 1, n_heads, d_head)
        if cfg.attn == "routed":
            hm = (layer.head_group[None, None, :] == route[:, None, None]).to(a.dtype)
            a = a * hm[..., None]
        u = x + a.reshape(B, 1, cfg.d_model) @ layer.eff("wo").t()
        z = rms_norm(u) * layer.g2
        y = layer.ffn_selected(z, route[:, None]) if cfg.ffn_routed else layer.ffn_full(z)
        x = u + y + layer.b_down
    return ((rms_norm(x) * model.g_final) @ model.emb.weight.t())[:, 0]


# --------------------------------------------------------------------------- scripted workers
@torch.jit.script
def dec_kv(xi: torch.Tensor, Kc: torch.Tensor, Vc: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, cos: torch.Tensor,
           sin: torch.Tensor, g: int, hpg: int, dh: int, m: int, p: int, ind: bool) -> int:
    """Keys and values of the heads H_g for the B new tokens, written into slot p of the cache of those heads."""
    B = xi.shape[0]
    if ind:
        xs = xi[:, :, g * m:(g + 1) * m]
        wk_g = wk[g * m:(g + 1) * m, g * m:(g + 1) * m]
        wv_g = wv[g * m:(g + 1) * m, g * m:(g + 1) * m]
    else:
        xs = xi
        wk_g = wk[g * m:(g + 1) * m]
        wv_g = wv[g * m:(g + 1) * m]
    Kc[:, g * hpg:(g + 1) * hpg, p:p + 1] = _rope((xs @ wk_g.t()).view(B, 1, hpg, dh).transpose(1, 2), cos, sin)
    Vc[:, g * hpg:(g + 1) * hpg, p:p + 1] = (xs @ wv_g.t()).view(B, 1, hpg, dh).transpose(1, 2)
    return 0


@torch.jit.script
def dec_routed(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, Kc: torch.Tensor, Vc: torch.Tensor,
               wq: torch.Tensor, wo: torch.Tensor, w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor,
               g2: torch.Tensor, b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               idx: torch.Tensor, valid: torch.Tensor, g: int, hpg: int, dh: int, m: int, n: int, p: int, ind: bool,
               ind_down: bool, final: bool) -> int:
    """For the sequences routed to g: queries of H_g, scores against the cache of the sequence, W_O^g and FFN group g."""
    B = x.shape[0]
    M = x.shape[2]
    P = idx.shape[0]
    if P == 0:
        return 0
    xs = xi.index_select(0, idx)
    if ind:
        xs = xs[:, :, g * m:(g + 1) * m]
        wq_g = wq[g * m:(g + 1) * m, g * m:(g + 1) * m]
    else:
        wq_g = wq[g * m:(g + 1) * m]
    q = _rope((xs @ wq_g.t()).view(P, 1, hpg, dh).transpose(1, 2), cos, sin)               # (P, hpg, 1, dh)
    k = Kc[:, g * hpg:(g + 1) * hpg, :p + 1].index_select(0, idx)
    v = Vc[:, g * hpg:(g + 1) * hpg, :p + 1].index_select(0, idx)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(P, 1, m)
    og = a @ wo[g * m:(g + 1) * m, g * m:(g + 1) * m].t()
    u = x.index_select(0, idx).clone()
    u[:, :, g * m:(g + 1) * m] = u[:, :, g * m:(g + 1) * m] + og
    z = _rms(u) * g2
    h = F.gelu(z[:, :, g * m:(g + 1) * m] @ w_up[g * n:(g + 1) * n, g * m:(g + 1) * m].t() + b_up[g * n:(g + 1) * n])
    if ind_down:
        y = torch.zeros_like(u)
        y[:, :, g * m:(g + 1) * m] = h @ w_down[g * m:(g + 1) * m, g * n:(g + 1) * n].t()
    else:
        y = h @ w_down[:, g * n:(g + 1) * n].t()
    xn_rows = (u + y + b_down).reshape(P, M)[valid]
    flat = idx[valid]
    x_out.view(B, M).index_copy_(0, flat, xn_rows)
    if not final:
        xi_out.view(B, M).index_copy_(0, flat, _rms(xn_rows) * g_next)
    return 0


@torch.jit.script
def dec_constrained_worker(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, Kc: torch.Tensor,
                           Vc: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
                           w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor,
                           g_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor,
                           g: int, hpg: int, dh: int, m: int, n: int, p: int, ind: bool, ind_down: bool, final: bool) -> int:
    dec_kv(xi, Kc, Vc, wk, wv, cos, sin, g, hpg, dh, m, p, ind)
    return dec_routed(xi, x, x_out, xi_out, Kc, Vc, wq, wo, w_up, b_up, w_down, g2, b_down, g_next, cos, sin, idx, valid, g,
                      hpg, dh, m, n, p, ind, ind_down, final)


@torch.jit.script
def dec_dense_heads(xi: torch.Tensor, a_out: torch.Tensor, Kc: torch.Tensor, Vc: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
                    wv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, g: int, hpg: int, dh: int, m: int, p: int) -> int:
    """Dense kinds: queries, keys and values of the heads H_g for the B new tokens, attention against the cache of the heads H_g."""
    B = xi.shape[0]
    q = _rope((xi @ wq[g * m:(g + 1) * m].t()).view(B, 1, hpg, dh).transpose(1, 2), cos, sin)
    Kc[:, g * hpg:(g + 1) * hpg, p:p + 1] = _rope((xi @ wk[g * m:(g + 1) * m].t()).view(B, 1, hpg, dh).transpose(1, 2), cos, sin)
    Vc[:, g * hpg:(g + 1) * hpg, p:p + 1] = (xi @ wv[g * m:(g + 1) * m].t()).view(B, 1, hpg, dh).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, Kc[:, g * hpg:(g + 1) * hpg, :p + 1], Vc[:, g * hpg:(g + 1) * hpg, :p + 1])
    a_out[:, :, g * m:(g + 1) * m] = a.transpose(1, 2).reshape(B, 1, m)
    return 0


@torch.jit.script
def dec_e2h_tail_worker(a: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, wo: torch.Tensor,
                        w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor,
                        g_next: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor, g: int, m: int, n: int, final: bool) -> int:
    """E2h after the barrier: W_O (dense) and FFN group g for the sequences routed to g."""
    B = x.shape[0]
    M = x.shape[2]
    P = idx.shape[0]
    if P == 0:
        return 0
    u = x.index_select(0, idx) + a.index_select(0, idx) @ wo.t()
    z = _rms(u) * g2
    h = F.gelu(z[:, :, g * m:(g + 1) * m] @ w_up[g * n:(g + 1) * n, g * m:(g + 1) * m].t() + b_up[g * n:(g + 1) * n])
    xn_rows = (u + h @ w_down[:, g * n:(g + 1) * n].t() + b_down).reshape(P, M)[valid]
    flat = idx[valid]
    x_out.view(B, M).index_copy_(0, flat, xn_rows)
    if not final:
        xi_out.view(B, M).index_copy_(0, flat, _rms(xn_rows) * g_next)
    return 0


# one fork-join per phase (the fork is issued from inside a scripted driver: see executor.py)
@torch.jit.script
def run_dec_constrained(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, Kc: torch.Tensor,
                        Vc: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
                        w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor,
                        g_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: List[torch.Tensor],
                        valid: List[torch.Tensor], hpg: int, dh: int, m: int, n: int, p: int, ind: bool, ind_down: bool,
                        final: bool):
    futs: List[torch.jit.Future[int]] = []
    for g in range(len(idx)):
        futs.append(torch.jit.fork(dec_constrained_worker, xi, x, x_out, xi_out, Kc, Vc, wq, wk, wv, wo, w_up, b_up, w_down, g2,
                                   b_down, g_next, cos, sin, idx[g], valid[g], g, hpg, dh, m, n, p, ind, ind_down, final))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def run_dec_dense_heads(xi: torch.Tensor, a_out: torch.Tensor, Kc: torch.Tensor, Vc: torch.Tensor, wq: torch.Tensor,
                        wk: torch.Tensor, wv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, G: int, hpg: int, dh: int,
                        m: int, p: int):
    futs: List[torch.jit.Future[int]] = []
    for g in range(G):
        futs.append(torch.jit.fork(dec_dense_heads, xi, a_out, Kc, Vc, wq, wk, wv, cos, sin, g, hpg, dh, m, p))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def run_dec_e2h_tail(a: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, wo: torch.Tensor,
                     w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor,
                     g_next: torch.Tensor, idx: List[torch.Tensor], valid: List[torch.Tensor], m: int, n: int, final: bool):
    futs: List[torch.jit.Future[int]] = []
    for g in range(len(idx)):
        futs.append(torch.jit.fork(dec_e2h_tail_worker, a, x, x_out, xi_out, wo, w_up, b_up, w_down, g2, b_down, g_next, idx[g],
                                   valid[g], g, m, n, final))
    for f in futs:
        torch.jit.wait(f)


# --------------------------------------------------------------------------- the decoder
class Decoder:
    """One decoded step of `B` sequences over G workers. `ex` is the `Executor` of the model (kind e4a / ind / dense / e2h)."""

    def __init__(self, ex: Executor, B: int, capacity: int):
        self.ex, self.B, self.cap = ex, B, capacity
        cfg = ex.cfg
        self.caches = [(torch.zeros(B, cfg.n_heads, capacity, cfg.d_head), torch.zeros(B, cfg.n_heads, capacity, cfg.d_head))
                       for _ in ex.layers]
        self.bufs = {}
        self.last_counts = None

    def load_caches(self, caches, p):
        """Copy the filled positions of the caches of `prefill_caches` (untimed)."""
        for (K, V), (K0, V0) in zip(self.caches, caches):
            K[:, :, :p], V[:, :, :p] = K0[:, :, :p], V0[:, :, :p]

    def _buf(self, name, like):
        b = self.bufs.get(name)
        if b is None or b.shape != like.shape:
            b = torch.empty_like(like)
            self.bufs[name] = b
        return b

    def plan(self, tok):
        """Per group: the sequences routed to it, padded to the largest count over the groups, and the validity mask."""
        G = self.ex.G
        route = self.ex.hash_table[tok]
        counts = [int((route == g).sum()) for g in range(G)]
        P = max(counts)
        out = []
        for g in range(G):
            sel = route == g
            order = torch.argsort((~sel).to(torch.int8), stable=True)
            idx = order[:P]
            out.append((idx, torch.arange(P) < counts[g]))
        self.last_counts = counts
        return out

    def step(self, tok, p, parallel=True):
        """Logits (B, 1, vocab) of the new tokens `tok` (B,) at position p; keys and values written into slot p of every layer."""
        ex, cfg = self.ex, self.ex.cfg
        G, m, n, dh = ex.G, cfg.m, cfg.n, cfg.d_head
        hpg = cfg.n_heads // G
        kind = ex.kind
        plan = None if kind == "dense" else self.plan(tok)
        x = ex.emb[tok][:, None, :]
        xi = _rms(x) * ex.layers[0]["g1"]
        cos, sin = ex.cos[p:p + 1], ex.sin[p:p + 1]
        nl = len(ex.layers)
        for l in range(nl):
            w = ex.layers[l]
            Kc, Vc = self.caches[l]
            final = l == nl - 1
            g_next = w["g1"] if final else ex.layers[l + 1]["g1"]
            x_out, xi_out = self._buf(f"x_out{l % 2}", x), self._buf(f"xi_out{l % 2}", x)
            if kind in ("e4a", "ind"):
                ind = kind == "ind"
                idx, valid = [q[0] for q in plan], [q[1] for q in plan]
                args = (xi, x, x_out, xi_out, Kc, Vc, w["wq"], w["wk"], w["wv"], w["wo"], w["w_up"], w["b_up"], w["w_down"],
                        w["g2"], w["b_down"], g_next, cos, sin)
                if parallel:
                    run_dec_constrained(*args, idx, valid, hpg, dh, m, n, p, ind, cfg.ffn_support == "ind", final)
                else:
                    for g in range(G):
                        dec_constrained_worker(*args, idx[g], valid[g], g, hpg, dh, m, n, p, ind, cfg.ffn_support == "ind", final)
            else:
                a = self._buf("a", x)
                if parallel:
                    run_dec_dense_heads(xi, a, Kc, Vc, w["wq"], w["wk"], w["wv"], cos, sin, G, hpg, dh, m, p)
                else:
                    for g in range(G):
                        dec_dense_heads(xi, a, Kc, Vc, w["wq"], w["wk"], w["wv"], cos, sin, g, hpg, dh, m, p)
                if kind == "dense":
                    if parallel:
                        run_dense_tail(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"], w["b_down"],
                                       g_next, G, final)
                    else:
                        for g in range(G):
                            dense_tail_worker(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"],
                                              w["b_down"], g_next, g, G, final)
                else:
                    idx, valid = [q[0] for q in plan], [q[1] for q in plan]
                    if parallel:
                        run_dec_e2h_tail(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"], w["b_down"],
                                         g_next, idx, valid, m, n, final)
                    else:
                        for g in range(G):
                            dec_e2h_tail_worker(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"],
                                                w["b_down"], g_next, idx[g], valid[g], g, m, n, final)
            x, xi = x_out, xi_out
        out = self._buf("logits", torch.empty(self.B, 1, cfg.vocab))
        if parallel:
            run_logits(x, out, ex.g_final, ex.emb, G)
        else:
            for g in range(G):
                logits_worker(x, out, ex.g_final, ex.emb, g, G)
        return out


def kind_of(cfg):
    if cfg.attn == "dense":
        return "e2h" if cfg.routing == "hash" else "dense"
    return "ind" if cfg.qkv_blockdiag else "e4a"


def build_decoder(cfg, ht, tokens, seed=0):
    """A decoder of the arm `cfg` for the windows `tokens` (B, N): caches filled by the untimed forward of the first N/2 positions.
    Returns (decoder, model, tok (B,), p). The model is in the selected form (the reference)."""
    from .model import SparseAttnLM
    B, N = tokens.shape
    p = N // 2
    model = SparseAttnLM(cfg, seed, hash_table=ht, impl="selected").eval()
    ex = Executor(model, kind_of(cfg))
    dec = Decoder(ex, B, N)
    with torch.no_grad():
        dec.load_caches(prefill_caches(model, tokens[:, :p], N), p)
    return dec, model, tokens[:, p].contiguous(), p


# --------------------------------------------------------------------------- parity (checks of E14)
def parity(arms, B=8, process=0):
    """For each arm: the decoded step against the full forward at position p (`decode_reference`), the packed sequential decoded
    step against the reference, the decoded step under the executor against the sequential one, two consecutive calls, and the
    cache positions below p unchanged by a step. Max absolute differences of the logits. Run in a process started with
    OMP_NUM_THREADS=1."""
    from .arms import get_arm
    from .data import Enwik8, balanced_byte_hash
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), 4)
    out = {}
    with torch.no_grad():
        for name in arms:
            cfg = get_arm(name)
            toks = decode_windows(data, cfg.context, B, process)
            dec, model, tok, p = build_decoder(cfg, ht, toks)
            full = model(toks[:, :p + 1])[0][:, p]
            ref = decode_reference(model, tok, prefill_caches(model, toks[:, :p], cfg.context), p)
            before = [(K[:, :, :p].clone(), V[:, :, :p].clone()) for K, V in dec.caches]
            seq = dec.step(tok, p, parallel=False)[:, 0].clone()
            counts = list(dec.last_counts) if dec.last_counts else None
            par = dec.step(tok, p, parallel=True)[:, 0].clone()
            again = dec.step(tok, p, parallel=True)[:, 0].clone()
            kept = all(bool((K[:, :, :p] == k0).all()) and bool((V[:, :, :p] == v0).all())
                       for (K, V), (k0, v0) in zip(dec.caches, before))
            d = lambda a, b: float((a - b).abs().max())  # noqa: E731
            out[name] = {"reference_vs_full_forward": d(ref, full), "sequential_vs_reference": d(seq, ref),
                         "executor_vs_sequential": d(par, seq), "repeat": d(again, par), "prefix_of_cache_unchanged": kept,
                         "group_counts_B": counts}
    return out


if __name__ == "__main__":
    import json
    import sys
    torch.set_num_threads(1)
    torch.set_num_interop_threads(4)
    if "--parity" in sys.argv:
        print(json.dumps(parity(sys.argv[sys.argv.index("--arms") + 1].split(","))))
