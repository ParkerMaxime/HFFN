"""Counter of compute and memory per token, by arm and context (PLAN sections 6 and 14.5). Counted, never timed.

Conventions (estimated, not measured):
* Multiply-adds of one forward pass per token, convention of PLAN section 6: every key a query attends to is counted,
  `N_k = N` (the last position of a window; the causal mean of the score and value terms is about half), the logits
  `V M` included. A masked-dense weight is counted by its non-zero entries (the structure the constraint keeps).
* Training multiply-adds per token: three times the forward (forward, backward to the activations, backward to the
  weights). An estimate; it ignores the optimizer step and the softmax.
* Parameters outside the embedding: `SparseAttnLM.count_params()`. AdamW states: two fp32 values per parameter
  (first and second moments).
* KV cache per token: `2 L M` values (keys and values of every head for every layer, dense in every arm).
* Training activations per token and layer, in values kept for the backward pass: the queries, the scores, the
  attention outputs and the FFN activations (the pre-activation `z` and the activation `h` of the active hidden
  neurons). Dense: `M + n_h N + M + 2 H`. Routed attention and FFN: the same divided by `G` (the token's group only):
  `m + (n_h / G) N + m + 2 n`. Under IND the keys and values are block-diagonal and not kept per token here.
"""
from .arms import ArmConfig


def layer_macs(cfg: ArmConfig):
    """Multiply-adds per token for one layer, by term."""
    M, m, n, k, N, G = cfg.d_model, cfg.m, cfg.n, cfg.degree, cfg.context, cfg.groups
    H = cfg.hidden
    terms = {}
    if cfg.attn != "none":
        kv_entries = G * m * m if cfg.qkv_blockdiag else M * M
        terms["keys_values"] = 2 * kv_entries
        if cfg.attn == "routed":
            q_entries = m * m if cfg.qkv_blockdiag else m * M
            terms["queries"] = q_entries
            terms["scores"] = N * m
            terms["weighted_values"] = N * m
            terms["output_projection"] = m * m
        else:
            terms["queries"] = G * m * m if cfg.qkv_blockdiag else M * M
            terms["scores"] = N * M
            terms["weighted_values"] = N * M
            terms["output_projection"] = G * m * m if cfg.wo_blockdiag else M * M
    terms["ffn"] = 2 * n * k if cfg.ffn_routed else 2 * H * k
    return terms


def count(cfg: ArmConfig, params_outside_embedding: int):
    L, M, V, N, G = cfg.n_layers, cfg.d_model, cfg.vocab, cfg.context, cfg.groups
    per_layer = layer_macs(cfg)
    inference = L * sum(per_layer.values()) + V * M
    routed = cfg.attn == "routed"
    heads = cfg.n_heads // G if routed else cfg.n_heads
    width = cfg.m if routed else M
    hidden = cfg.n if cfg.ffn_routed else cfg.hidden
    activations = L * (width + heads * N + width + 2 * hidden)
    return {
        "arm": cfg.name, "context": N, "layers": L,
        "inference_macs_per_token": inference,
        "inference_macs_per_layer_terms": per_layer,
        "training_macs_per_token": 3 * inference,
        "params_outside_embedding": params_outside_embedding,
        "adamw_state_values_outside_embedding": 2 * params_outside_embedding,
        "kv_cache_values_per_token": 2 * L * M,
        "training_activation_values_per_token": activations,
    }


def count_arm(name: str):
    from .arms import get_arm
    from .model import SparseAttnLM
    cfg = get_arm(name)
    return count(cfg, SparseAttnLM(cfg, 0).count_params())


if __name__ == "__main__":
    import json
    import sys
    from .arms import SERIES, SERIES_N512
    names = sys.argv[1:] or [a.name for a in (*SERIES, *SERIES_N512)]
    print(json.dumps([count_arm(n) for n in names], indent=1))
