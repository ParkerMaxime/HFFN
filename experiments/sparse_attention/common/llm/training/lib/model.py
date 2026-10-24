"""The sparse_attention model: one implementation, one `ArmConfig` per arm (PLAN sections 2, 3, 7).

Representation: the fixed-support FFN and the block-diagonal projections are masked dense matrices (PLAN section 0,
a control representation in the sense of AGENTS.md section 3, admissible because nothing is timed). A masked tensor
is stored dense and multiplied by its 0/1 mask at every forward, so its gradient is zero outside the support and the
entries outside the support are zero from the initialisation on.

Two implementations of the constraint (C2) coexist and give the same function and the same gradients (PLAN section 7):
* `packed`: the tokens of group g (positions with (t + shift) mod G = g) are gathered, only the heads H_g and the
  neurons R_g are computed for them. Only for position routing.
* `selected`: every head and every neuron is computed, the contribution of the groups other than the route of the
  token is multiplied by zero. Used for content routing (hash, learned) and as the reference of the checks.
"""
import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arms import ArmConfig


def gen(seed: int, tag: str) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(zlib.crc32(f"{seed}/{tag}".encode()))
    return g


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def block_mask(size: int, groups: int) -> torch.Tensor:
    g = torch.arange(size) // (size // groups)
    return (g[:, None] == g[None, :]).float()


def rope_tables(positions: torch.Tensor, d_head: int, base: float = 10000.0):
    inv = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    ang = positions.float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rope(t, cos, sin):
    h = t.shape[-1] // 2
    t1, t2 = t[..., :h], t[..., h:]
    return torch.cat((t1 * cos - t2 * sin, t1 * sin + t2 * cos), dim=-1)


def build_supports(cfg: ArmConfig, seed: int, layer: int):
    """Fixed random supports with a fixed degree per neuron: `up_mask` (H, M) and `down_mask` (M, H)."""
    H, M, k, m, n = cfg.hidden, cfg.d_model, cfg.degree, cfg.m, cfg.n
    group = torch.arange(H) // n
    g_up = gen(seed, f"l{layer}.upsupport")
    if cfg.ffn_support == "random":
        up_idx = torch.rand(H, M, generator=g_up).topk(k, dim=1).indices
    else:
        up_idx = torch.rand(H, m, generator=g_up).topk(k, dim=1).indices + group[:, None] * m
    g_dn = gen(seed, f"l{layer}.downsupport")
    if cfg.ffn_support == "ind":
        dn_idx = torch.rand(H, m, generator=g_dn).topk(k, dim=1).indices + group[:, None] * m
    else:
        dn_idx = torch.rand(H, M, generator=g_dn).topk(k, dim=1).indices
    up_mask = torch.zeros(H, M).scatter_(1, up_idx, 1.0)
    down_mask = torch.zeros(H, M).scatter_(1, dn_idx, 1.0).t().contiguous()
    return up_mask, down_mask


class Layer(nn.Module):
    def __init__(self, cfg: ArmConfig, seed: int, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        M, H, G = cfg.d_model, cfg.hidden, cfg.groups
        tag = f"l{layer}"
        self._masked = {}

        def dense(name, shape, std):
            return torch.randn(*shape, generator=gen(seed, f"{tag}.{name}")) * std

        def reg(name, value, mask=None):
            if mask is not None:
                value = value * mask
                self.register_buffer(f"mask_{name}", mask)
                self._masked[name] = f"mask_{name}"
            setattr(self, name, nn.Parameter(value))

        if cfg.attn != "none":
            reg("g1", torch.ones(M))
            bm = block_mask(M, G)
            for nm in ("wq", "wk", "wv"):
                reg(nm, dense(nm, (M, M), M ** -0.5), bm if cfg.qkv_blockdiag else None)
            reg("wo", dense("wo", (M, M), M ** -0.5), bm if cfg.wo_blockdiag else None)
        reg("g2", torch.ones(M))
        up_mask, down_mask = build_supports(cfg, seed, layer)
        reg("w_up", dense("w_up", (H, M), cfg.degree ** -0.5), up_mask)
        reg("b_up", torch.zeros(H))
        reg("w_down", dense("w_down", (M, H), cfg.degree ** -0.5), down_mask)
        reg("b_down", torch.zeros(M))
        if cfg.routing == "learned":
            reg("w_route", dense("w_route", (G, M), 0.02))
        neuron_group = torch.arange(H) // cfg.n
        head_group = torch.arange(cfg.n_heads) // (cfg.n_heads // G)
        self.register_buffer("neuron_group", neuron_group, persistent=False)
        self.register_buffer("head_group", head_group, persistent=False)

    def eff(self, name):
        w = getattr(self, name)
        if name in self._masked:
            w = w * getattr(self, self._masked[name])
        return w

    # ---------------------------------------------------------------- routing
    def shift(self):
        return self.layer if self.cfg.layer_routing == "rotation" else 0

    def route_of(self, xi, tokens, hash_table):
        """Route of every token, shape (B, N) with values in [0, G) or -1 (dropped); gate (B, N) or None."""
        cfg = self.cfg
        B, N = tokens.shape
        G = cfg.groups
        stats = {}
        if cfg.routing == "position":
            r = (torch.arange(N) + self.shift()) % G
            return r[None, :].expand(B, N), None, stats
        if cfg.routing == "hash":
            return (hash_table[tokens] + self.shift()) % G, None, stats
        logits = xi @ self.eff("w_route").t()
        probs = logits.softmax(-1)
        p_top, top = probs.max(-1)
        onehot = F.one_hot(top, G).float()
        f = onehot.mean((0, 1))
        P = probs.mean((0, 1))
        stats["balance_loss"] = G * (f * P).sum()
        stats["load_pre"] = f.detach()
        route = top
        if cfg.capacity_factor > 0:
            cap = math.ceil(cfg.capacity_factor * N / G)
            rank = ((onehot.cumsum(1) - 1) * onehot).sum(-1)  # earlier tokens of the same group: causal-safe
            keep = rank < cap
            route = torch.where(keep, top, torch.full_like(top, -1))
            stats["dropped"] = (~keep).float().mean().detach()
        else:
            keep = torch.ones_like(top, dtype=torch.bool)
        stats["load"] = F.one_hot(route.clamp(min=0), G).float().mul(keep[..., None]).mean((0, 1)).detach()
        if cfg.gate == "none":
            gate = None
        else:
            gate = p_top * (G if cfg.gate == "Gp" else 1.0) * keep
        return route, gate, stats

    # ---------------------------------------------------------------- attention
    def qkv(self, xi):
        cfg = self.cfg
        B, n, M = xi.shape
        dh = cfg.d_head
        q = xi @ self.eff("wq").t()
        k = xi @ self.eff("wk").t()
        v = xi @ self.eff("wv").t()
        sp = lambda t: t.view(B, n, cfg.n_heads, dh).transpose(1, 2)
        return sp(q), sp(k), sp(v)

    def attention_full(self, xi, cos, sin):
        """Every head for every token (dense attention). Returns the head outputs (B, N, M) before W_O."""
        B, N, M = xi.shape
        q, k, v = self.qkv(xi)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return a.transpose(1, 2).reshape(B, N, M)

    def attention_selected(self, xi, cos, sin, route):
        B, N, M = xi.shape
        q, k, v = self.qkv(xi)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)  # (B, N, nh, dh)
        hm = (self.head_group[None, None, :] == route[..., None]).to(a.dtype)
        return (a * hm[..., None]).reshape(B, N, M)

    def attention_packed(self, xi, cos, sin):
        """o_t on S_g for the tokens of each group; zeros elsewhere. Heads H_g only are queried for group g."""
        cfg = self.cfg
        B, N, M = xi.shape
        G, m, dh = cfg.groups, cfg.m, cfg.d_head
        hpg = cfg.n_heads // G
        wq, wk, wv, wo = self.eff("wq"), self.eff("wk"), self.eff("wv"), self.eff("wo")
        k = (xi @ wk.t()).view(B, N, cfg.n_heads, dh).transpose(1, 2)
        v = (xi @ wv.t()).view(B, N, cfg.n_heads, dh).transpose(1, 2)
        k = apply_rope(k, cos, sin)
        o = torch.zeros(B, N, M, dtype=xi.dtype)
        keys = torch.arange(N)
        for g in range(G):
            start = (g - self.shift()) % G
            pos = torch.arange(start, N, G)
            rows = slice(g * m, (g + 1) * m)
            hs = slice(g * hpg, (g + 1) * hpg)
            q = (xi[:, start::G] @ wq[rows].t()).view(B, len(pos), hpg, dh).transpose(1, 2)
            q = apply_rope(q, cos[pos], sin[pos])
            mask = keys[None, :] <= pos[:, None]
            a = F.scaled_dot_product_attention(q, k[:, hs], v[:, hs], attn_mask=mask)
            a = a.transpose(1, 2).reshape(B, len(pos), m)
            o[:, start::G, rows] = a @ wo[rows, rows].t()
        return o

    # ---------------------------------------------------------------- FFN
    def ffn_full(self, z):
        h = F.gelu(z @ self.eff("w_up").t() + self.b_up)
        return h @ self.eff("w_down").t()

    def ffn_selected(self, z, route):
        h = F.gelu(z @ self.eff("w_up").t() + self.b_up)
        hm = (self.neuron_group[None, None, :] == route[..., None]).to(h.dtype)
        return (h * hm) @ self.eff("w_down").t()

    def ffn_packed(self, z):
        cfg = self.cfg
        B, N, M = z.shape
        G, m, n = cfg.groups, cfg.m, cfg.n
        w_up, w_down = self.eff("w_up"), self.eff("w_down")
        y = torch.zeros(B, N, M, dtype=z.dtype)
        for g in range(G):
            start = (g - self.shift()) % G
            ns, ms = slice(g * n, (g + 1) * n), slice(g * m, (g + 1) * m)
            zg = z[:, start::G, ms]
            h = F.gelu(zg @ w_up[ns, ms].t() + self.b_up[ns])
            y[:, start::G] = h @ w_down[:, ns].t()
        return y

    # ---------------------------------------------------------------- layer
    def parts(self, x, tokens, cos, sin, hash_table, impl):
        """One layer, with its intermediate quantities: u = x + o (attention), y (FFN), route, gate, router stats."""
        cfg = self.cfg
        stats = {}
        packed = impl == "packed" and cfg.routing == "position"
        route = gate = None
        xi = rms_norm(x) * self.g1 if cfg.attn != "none" else None
        if cfg.ffn_routed and not packed:
            route, gate, stats = self.route_of(xi, tokens, hash_table)
        if cfg.attn == "none":
            o = torch.zeros_like(x)
        elif cfg.attn == "dense":
            o = self.attention_full(xi, cos, sin) @ self.eff("wo").t()
        elif packed:
            o = self.attention_packed(xi, cos, sin)
        else:
            o = self.attention_selected(xi, cos, sin, route) @ self.eff("wo").t()
            if gate is not None:
                o = o * gate[..., None]
        u = x + o
        z = rms_norm(u) * self.g2
        if not cfg.ffn_routed:
            y = self.ffn_full(z)
        elif packed:
            y = self.ffn_packed(z)
        else:
            y = self.ffn_selected(z, route)
            if gate is not None:
                y = y * gate[..., None]
        return {"u": u, "o": o, "y": y, "route": route, "gate": gate, "stats": stats}

    def forward(self, x, tokens, cos, sin, hash_table, impl):
        p = self.parts(x, tokens, cos, sin, hash_table, impl)
        return p["u"] + p["y"] + self.b_down, p["stats"]


class SparseAttnLM(nn.Module):
    def __init__(self, cfg: ArmConfig, seed: int, hash_table=None, impl="packed"):
        super().__init__()
        self.cfg, self.seed, self.impl = cfg, seed, impl
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(cfg.vocab, cfg.d_model, generator=gen(seed, "emb")) * 0.05)
        self.layers = nn.ModuleList(Layer(cfg, seed, l) for l in range(cfg.n_layers))
        self.g_final = nn.Parameter(torch.ones(cfg.d_model))
        ht = torch.zeros(cfg.vocab, dtype=torch.long) if hash_table is None else torch.as_tensor(hash_table).long()
        self.register_buffer("hash_table", ht, persistent=True)
        cos, sin = rope_tables(torch.arange(cfg.context), cfg.d_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens):
        N = tokens.shape[1]
        x = self.emb(tokens)
        stats = []
        for layer in self.layers:
            x, s = layer(x, tokens, self.cos[:N], self.sin[:N], self.hash_table, self.impl)
            stats.append(s)
        logits = rms_norm(x) * self.g_final @ self.emb.weight.t()
        return logits, stats

    def count_params(self, exclude_embedding=True):
        total = 0
        for name, p in self.named_parameters():
            if exclude_embedding and name.startswith("emb."):
                continue
            mask = None
            if name.startswith("layers."):
                _, l, pname = name.split(".")
                layer = self.layers[int(l)]
                if pname in layer._masked:
                    mask = getattr(layer, layer._masked[pname])
            total += int(mask.sum()) if mask is not None else p.numel()
        return total

    def decay_params(self):
        """Matrices other than the embedding (norm gains and biases are 1-D)."""
        return [p for n, p in self.named_parameters() if p.ndim == 2 and not n.startswith("emb.")]
