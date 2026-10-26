"""Executor of E11 (PLAN 15.3): forward only, without gradient, over G = 4 workers, one intra-op thread each.

Method: TorchScript `fork` / `wait` on the inter-op pool (4 threads). A scripted worker function runs a group's work without
the GIL; the join of the four futures is the barrier. Weights are the effective (masked) weights, built once; the FFN is a
masked dense matrix in every arm (PLAN section 0).

Three arms of the executor, the same structure of calls (one fork-join per layer for the constrained arms, two for the dense):
* `e4a`: worker g computes the keys and values of its heads H_g for every token (reading the whole normalised residual), then
  queries, scores, outputs, W_O^g and the FFN group R_g for the tokens routed to g (the packed form under the hash route), and
  writes the new residual rows of its tokens and their normalised rows for the next layer. One barrier per layer.
* `ind`: the same, the keys, values and queries of worker g read S_g only. The new residual of a token is complete in its
  owner (the other coordinates are the embedding plus the bias); what the workers exchange is the normalised block S_g of every
  token, through the scalar of the RMS norm of the owner. The barrier per layer cannot be removed without changing the function
  (the RMS of a row needs the block its owner has just written), so the executor keeps it.
* `dense`: worker g computes queries, keys, values, scores and outputs of the heads H_g over every token; barrier; then W_O and the
  (unrouted, masked dense) FFN on a block of rows per worker; barrier.
"""
from typing import List

import torch
import torch.nn.functional as F

from .model import SparseAttnLM, rms_norm, rope_tables


# --------------------------------------------------------------------------- scripted workers
@torch.jit.script
def _rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    h = t.shape[-1] // 2
    t1 = t[..., :h]
    t2 = t[..., h:]
    return torch.cat((t1 * cos - t2 * sin, t1 * sin + t2 * cos), dim=-1)


@torch.jit.script
def _rms(x: torch.Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


@torch.jit.script
def kv_phase(xi: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
             g: int, hpg: int, dh: int, m: int, ind: bool):
    B = xi.shape[0]
    N = xi.shape[1]
    if ind:
        xs = xi[:, :, g * m:(g + 1) * m]
        wk_g = wk[g * m:(g + 1) * m, g * m:(g + 1) * m]
        wv_g = wv[g * m:(g + 1) * m, g * m:(g + 1) * m]
    else:
        xs = xi
        wk_g = wk[g * m:(g + 1) * m]
        wv_g = wv[g * m:(g + 1) * m]
    k = _rope((xs @ wk_g.t()).view(B, N, hpg, dh).transpose(1, 2), cos, sin)
    v = (xs @ wv_g.t()).view(B, N, hpg, dh).transpose(1, 2)
    return k, v


@torch.jit.script
def routed_phase(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor, wq: torch.Tensor, wo: torch.Tensor, w_up: torch.Tensor, b_up: torch.Tensor,
                 w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor,
                 sin: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, flat: torch.Tensor,
                 g: int, hpg: int, dh: int, m: int, n: int, ind: bool, ind_down: bool, final: bool) -> int:
    B = x.shape[0]
    N = x.shape[1]
    M = x.shape[2]
    P = pos.shape[1]
    if P == 0:
        return 0
    if ind:
        xs = xi[:, :, g * m:(g + 1) * m]
        wq_g = wq[g * m:(g + 1) * m, g * m:(g + 1) * m]
    else:
        xs = xi
        wq_g = wq[g * m:(g + 1) * m]
    idx = pos.unsqueeze(-1).expand(-1, -1, xs.shape[-1])
    q = (xs.gather(1, idx) @ wq_g.t()).view(B, P, hpg, dh).transpose(1, 2)
    q = _rope(q, cos[pos].unsqueeze(1), sin[pos].unsqueeze(1))
    keys = torch.arange(N)
    mask = (keys.unsqueeze(0).unsqueeze(0) <= pos.unsqueeze(-1)).unsqueeze(1)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, P, m)
    og = a @ wo[g * m:(g + 1) * m, g * m:(g + 1) * m].t()                        # (B, P, m), on S_g
    xr = x.gather(1, pos.unsqueeze(-1).expand(-1, -1, M))                         # residual rows of the routed tokens
    u = xr.clone()
    u[:, :, g * m:(g + 1) * m] = u[:, :, g * m:(g + 1) * m] + og
    z = _rms(u) * g2
    h = F.gelu(z[:, :, g * m:(g + 1) * m] @ w_up[g * n:(g + 1) * n, g * m:(g + 1) * m].t() + b_up[g * n:(g + 1) * n])
    if ind_down:
        y = torch.zeros_like(u)
        y[:, :, g * m:(g + 1) * m] = h @ w_down[g * m:(g + 1) * m, g * n:(g + 1) * n].t()
    else:
        y = h @ w_down[:, g * n:(g + 1) * n].t()
    xn = u + y + b_down                                                            # (B, P, M)
    xn_rows = xn.reshape(B * P, M)[valid.reshape(-1)]
    x_out.view(B * N, M).index_copy_(0, flat, xn_rows)
    if not final:
        xi_out.view(B * N, M).index_copy_(0, flat, _rms(xn_rows) * g_next)
    return 0


@torch.jit.script
def constrained_worker(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor,
                       wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
                       w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor,
                       b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       pos: torch.Tensor, valid: torch.Tensor, flat: torch.Tensor,
                       g: int, hpg: int, dh: int, m: int, n: int, ind: bool, ind_down: bool, final: bool) -> int:
    k, v = kv_phase(xi, wk, wv, cos, sin, g, hpg, dh, m, ind)
    return routed_phase(xi, x, x_out, xi_out, k, v, wq, wo, w_up, b_up, w_down, g2, b_down, g_next, cos, sin, pos, valid,
                        flat, g, hpg, dh, m, n, ind, ind_down, final)


@torch.jit.script
def dense_heads_worker(xi: torch.Tensor, a_out: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor, g: int, hpg: int, dh: int, m: int) -> int:
    B = xi.shape[0]
    N = xi.shape[1]
    q = _rope((xi @ wq[g * m:(g + 1) * m].t()).view(B, N, hpg, dh).transpose(1, 2), cos, sin)
    k = _rope((xi @ wk[g * m:(g + 1) * m].t()).view(B, N, hpg, dh).transpose(1, 2), cos, sin)
    v = (xi @ wv[g * m:(g + 1) * m].t()).view(B, N, hpg, dh).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, N, m)
    a_out[:, :, g * m:(g + 1) * m] = a
    return 0


@torch.jit.script
def dense_tail_worker(a: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, wo: torch.Tensor,
                      w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor,
                      b_down: torch.Tensor, g_next: torch.Tensor, g: int, G: int, final: bool) -> int:
    M = x.shape[2]
    rows = a.shape[0] * a.shape[1]
    r0 = g * rows // G
    r1 = (g + 1) * rows // G
    ar = a.view(rows, M)[r0:r1]
    u = x.view(rows, M)[r0:r1] + ar @ wo.t()
    z = _rms(u) * g2
    y = F.gelu(z @ w_up.t() + b_up) @ w_down.t()
    xn = u + y + b_down
    x_out.view(rows, M)[r0:r1] = xn
    if not final:
        xi_out.view(rows, M)[r0:r1] = _rms(xn) * g_next
    return 0


@torch.jit.script
def logits_worker(x: torch.Tensor, out: torch.Tensor, g_final: torch.Tensor, emb: torch.Tensor, g: int, G: int) -> int:
    M = x.shape[2]
    rows = x.shape[0] * x.shape[1]
    r0 = g * rows // G
    r1 = (g + 1) * rows // G
    out.view(rows, -1)[r0:r1] = (_rms(x.view(rows, M)[r0:r1]) * g_final) @ emb.t()
    return 0


# one fork per worker inside TorchScript (a fork issued from Python does not run concurrently); the join is the barrier
@torch.jit.script
def run_constrained(xi: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, wq: torch.Tensor,
                    wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, w_up: torch.Tensor, b_up: torch.Tensor,
                    w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor, pos: List[torch.Tensor], valid: List[torch.Tensor], flat: List[torch.Tensor],
                    hpg: int, dh: int, m: int, n: int, ind: bool, ind_down: bool, final: bool):
    futs: List[torch.jit.Future[int]] = []
    for g in range(len(pos)):
        futs.append(torch.jit.fork(constrained_worker, xi, x, x_out, xi_out, wq, wk, wv, wo, w_up, b_up, w_down, g2, b_down,
                                   g_next, cos, sin, pos[g], valid[g], flat[g], g, hpg, dh, m, n, ind, ind_down, final))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def run_dense_heads(xi: torch.Tensor, a_out: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor, G: int, hpg: int, dh: int, m: int):
    futs: List[torch.jit.Future[int]] = []
    for g in range(G):
        futs.append(torch.jit.fork(dense_heads_worker, xi, a_out, wq, wk, wv, cos, sin, g, hpg, dh, m))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def run_dense_tail(a: torch.Tensor, x: torch.Tensor, x_out: torch.Tensor, xi_out: torch.Tensor, wo: torch.Tensor,
                   w_up: torch.Tensor, b_up: torch.Tensor, w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor,
                   g_next: torch.Tensor, G: int, final: bool):
    futs: List[torch.jit.Future[int]] = []
    for g in range(G):
        futs.append(torch.jit.fork(dense_tail_worker, a, x, x_out, xi_out, wo, w_up, b_up, w_down, g2, b_down, g_next,
                                   g, G, final))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def run_logits(x: torch.Tensor, out: torch.Tensor, g_final: torch.Tensor, emb: torch.Tensor, G: int):
    futs: List[torch.jit.Future[int]] = []
    for g in range(G):
        futs.append(torch.jit.fork(logits_worker, x, out, g_final, emb, g, G))
    for f in futs:
        torch.jit.wait(f)


# --------------------------------------------------------------------------- the executor
class Executor:
    """Forward pass of a `SparseAttnLM` over G workers. `kind` is "e4a" / "ind" (hash route, packed form) or "dense"."""

    def __init__(self, model: SparseAttnLM, kind: str):
        self.cfg, self.kind, self.G = model.cfg, kind, model.cfg.groups
        assert kind in ("e4a", "ind", "dense")
        assert (kind == "dense") == (model.cfg.attn == "dense")
        self.hash_table = model.hash_table
        self.emb = model.emb.weight.detach()
        self.g_final = model.g_final.detach()
        self.cos, self.sin = model.cos.clone(), model.sin.clone()
        with torch.no_grad():
            self.layers = []
            for layer in model.layers:
                self.layers.append({k: layer.eff(k).detach().contiguous() for k in ("wq", "wk", "wv", "wo", "w_up", "w_down")}
                                   | {"g1": layer.g1.detach(), "g2": layer.g2.detach(), "b_up": layer.b_up.detach(),
                                      "b_down": layer.b_down.detach()})
        self.model = model
        self.buffers = {}

    def _buf(self, name, like):
        b = self.buffers.get(name)
        if b is None or b.shape != like.shape:
            b = torch.empty_like(like)
            self.buffers[name] = b
        return b

    # -- plan of the routed tokens (hash route): positions per group, padded to the largest count of the batch
    def plan(self, tokens):
        G = self.G
        route = self.hash_table[tokens]
        B, N = tokens.shape
        out = []
        for g in range(G):
            sel = route == g
            count = sel.sum(1)
            P = int(count.max())
            order = torch.argsort((~sel).to(torch.int8), dim=1, stable=True)
            pos = order[:, :P]
            valid = torch.arange(P)[None, :] < count[:, None]
            flat = (torch.arange(B)[:, None] * N + pos)[valid]
            out.append((pos, valid, flat))
        return out

    def embed(self, tokens):
        x = self.emb[tokens]
        return x, _rms(x) * self.layers[0]["g1"] if self.cfg.attn != "none" else x

    def layer(self, l, x, xi, plan=None):
        """One layer: returns (x_new, xi_next). The buffers are reused between calls."""
        cfg, w = self.cfg, self.layers[l]
        final = l == len(self.layers) - 1
        g_next = w["g1"] if final else self.layers[l + 1]["g1"]
        x_out, xi_out = self._buf(f"x_out{l % 2}", x), self._buf(f"xi_out{l % 2}", x)  # double buffer: a layer reads the previous one
        G, m, n, dh = self.G, cfg.m, cfg.n, cfg.d_head
        hpg = cfg.n_heads // G
        if self.kind == "dense":
            a = self._buf("a", x)
            run_dense_heads(xi, a, w["wq"], w["wk"], w["wv"], self.cos[:x.shape[1]], self.sin[:x.shape[1]], G, hpg, dh, m)
            run_dense_tail(a, x, x_out, xi_out, w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"], w["b_down"], g_next, G,
                           final)
        else:
            ind = self.kind == "ind"
            cos, sin = self.cos[:x.shape[1]], self.sin[:x.shape[1]]
            run_constrained(xi, x, x_out, xi_out, w["wq"], w["wk"], w["wv"], w["wo"], w["w_up"], w["b_up"], w["w_down"],
                            w["g2"], w["b_down"], g_next, cos, sin, [p[0] for p in plan], [p[1] for p in plan],
                            [p[2] for p in plan], hpg, dh, m, n, ind, cfg.ffn_support == "ind", final)
        return x_out, xi_out

    def layers_forward(self, x, xi, plan=None):
        for l in range(len(self.layers)):
            x, xi = self.layer(l, x, xi, plan)
        return x

    def forward(self, tokens):
        plan = self.plan(tokens) if self.kind != "dense" else None
        x, xi = self.embed(tokens)
        x = self.layers_forward(x, xi, plan)
        out = self._buf("logits", torch.empty(*tokens.shape, self.cfg.vocab))
        run_logits(x, out, self.g_final, self.emb, self.G)
        return out


# --------------------------------------------------------------------------- the concurrency gate (PLAN 15.3)
@torch.jit.script
def _gate_parallel(xi: torch.Tensor, x: torch.Tensor, x_outs: List[torch.Tensor], xi_outs: List[torch.Tensor], wq: torch.Tensor,
                   wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, w_up: torch.Tensor, b_up: torch.Tensor,
                   w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor,
                   sin: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, flat: torch.Tensor,
                   hpg: int, dh: int, m: int, n: int):
    futs: List[torch.jit.Future[int]] = []
    for i in range(len(x_outs)):
        futs.append(torch.jit.fork(constrained_worker, xi, x, x_outs[i], xi_outs[i], wq, wk, wv, wo, w_up, b_up, w_down, g2,
                                   b_down, g_next, cos, sin, pos, valid, flat, 0, hpg, dh, m, n, False, False, False))
    for f in futs:
        torch.jit.wait(f)


@torch.jit.script
def _gate_sequential(xi: torch.Tensor, x: torch.Tensor, x_outs: List[torch.Tensor], xi_outs: List[torch.Tensor], wq: torch.Tensor,
                     wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, w_up: torch.Tensor, b_up: torch.Tensor,
                     w_down: torch.Tensor, g2: torch.Tensor, b_down: torch.Tensor, g_next: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, flat: torch.Tensor,
                     hpg: int, dh: int, m: int, n: int):
    for i in range(len(x_outs)):
        constrained_worker(xi, x, x_outs[i], xi_outs[i], wq, wk, wv, wo, w_up, b_up, w_down, g2, b_down, g_next, cos, sin, pos,
                           valid, flat, 0, hpg, dh, m, n, False, False, False)


def concurrency_gate(M=512, N=2048, repetitions=5, threshold=0.4, tasks=4):
    """Four equal independent tasks of the size of the largest layer cell (a4: one worker's share of an E4a layer at M = 512,
    N = 2048, 8,192 tokens), run by the executor and one after the other: the median of `repetitions` repetitions of the ratio
    must be at most `threshold` times the sequential time. Returns the measurements; the caller records them."""
    import statistics
    import time

    from .arms import ArmConfig
    from .data import Enwik8, balanced_byte_hash
    cfg = ArmConfig("gate", ffn_support="blocks", ffn_routed=True, wo_blockdiag=True, attn="routed", routing="hash",
                    d_model=M, hidden=4 * M, degree=13 * M // 64, context=N, n_layers=1)
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), cfg.groups)
    tokens, _ = data.eval_windows("valid", N, 8192 // N)
    with torch.no_grad():
        ex = Executor(SparseAttnLM(cfg, 0, hash_table=ht, impl="packed").eval(), "e4a")
        x, xi = ex.embed(tokens)
        plan = ex.plan(tokens)
        w = ex.layers[0]
        pos, valid, flat = plan[0]
        args = (xi, x, [torch.empty_like(x) for _ in range(tasks)], [torch.empty_like(x) for _ in range(tasks)], w["wq"], w["wk"],
                w["wv"], w["wo"], w["w_up"], w["b_up"], w["w_down"], w["g2"], w["b_down"], w["g1"], ex.cos[:N], ex.sin[:N], pos,
                valid, flat, cfg.n_heads // cfg.groups, cfg.d_head, cfg.m, cfg.n)
        for _ in range(3):
            _gate_parallel(*args)
            _gate_sequential(*args)
        par, seq = [], []
        for _ in range(repetitions):  # alternate, so that a drift of the machine does not favour one of them
            t = time.perf_counter()
            _gate_sequential(*args)
            seq.append(time.perf_counter() - t)
            t = time.perf_counter()
            _gate_parallel(*args)
            par.append(time.perf_counter() - t)
    ratios = [p / s for p, s in zip(par, seq)]
    return {"cell": f"a4 (M={M}, N={N})", "tasks": tasks, "repetitions": repetitions, "threshold": threshold,
            "ratio_per_repetition": ratios, "median_ratio": statistics.median(ratios),
            "passed": statistics.median(ratios) <= threshold, "interop_threads": torch.get_num_interop_threads(),
            "intraop_threads": torch.get_num_threads()}


# --------------------------------------------------------------------------- parity of the executor with the sequential forward
def parity(arms=("E4a_L2", "E4a_L4", "IND_L4", "REF2", "E4a_L2_N512", "IND_L4_N512", "REF2_N512", "E4a_L4_W256",
                 "E4a_L4_W256_h16", "REF2_W256_h16")):
    """Outputs of every arm under the executor against the sequential forward (`selected` form under the hash route, `packed`
    for the dense arms), on batches of real bytes. Run in a process started with OMP_NUM_THREADS=1."""
    from .arms import get_arm
    from .data import Enwik8, balanced_byte_hash
    data = Enwik8()
    ht = balanced_byte_hash(data.train_counts(), 4)
    out = {}
    with torch.no_grad():
        for name in arms:
            cfg = get_arm(name)
            kind = "dense" if cfg.attn == "dense" else ("ind" if cfg.qkv_blockdiag else "e4a")
            tokens, _ = data.eval_windows("valid", cfg.context, 8192 // cfg.context)
            model = SparseAttnLM(cfg, 3, hash_table=ht, impl="selected" if kind != "dense" else "packed").eval()
            ref, _ = model(tokens)
            got = Executor(model, kind).forward(tokens)
            out[name] = float((ref - got).abs().max())
    return out


if __name__ == "__main__":
    import json
    import sys
    torch.set_num_threads(1)
    torch.set_num_interop_threads(4)
    if "--parity" in sys.argv:
        print(json.dumps(parity()))
    if "--gate" in sys.argv:
        print(json.dumps(concurrency_gate()))
