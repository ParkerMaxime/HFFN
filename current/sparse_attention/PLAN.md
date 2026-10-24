# Sparse attention by HFFN groups — idea, theory and plan

Written on 2026-10-24 at the user's request, from the discussion of the same day. Nothing in this file is measured:
every number is **estimated** (a calculation) and every explanation of an expected effect is a **hypothesis**, until
a qualified run says otherwise (`AGENTS.md` section 7).

## 0. Status and scope

- **What it is.** A quality study, in PyTorch, of a Transformer whose attention is partitioned by the same groups as
  the hidden neurons of its HFFN feed-forward network, so that each token is processed, in attention and in the FFN,
  by one group only. It is a **different model** from a dense-attention Transformer, trained from scratch; it is an
  approximation of dense attention in the sense that it is a restriction of it, not a compression of a trained model.
  Comparisons are on **quality**, at matched parameters wherever the constraint allows.
- **What it is not.** It is not an evolution of the HFFN runtime (`runtime/`), and no time or speed is measured in
  this phase. The compute it saves is counted analytically (section 6). A native implementation, if any, comes later
  and then follows every runtime rule of `AGENTS.md`.
- **Governance.** The branch was opened by the user on 2026-10-24 as a **secondary** study: the runtime keeps
  priority (`AGENTS.md` sections 4 and 5, *Objective*; `DECISIONS.md`). When a machine, a VM session or a review is
  needed by both, the runtime takes it. Like `lateral/`, it is an architecture study decided by the user, outside the
  runtime goal of section 1.
- **Representation of the FFN in this phase.** There is no HFFN runtime path here. A fixed-support FFN may be
  parameterised as a masked dense matrix or as compact index/value tensors; a masked dense matrix is a control
  representation in the sense of `AGENTS.md` section 3, admissible because nothing is timed and no runtime claim is
  drawn. The model and its gradients are the same either way.

### Decisions of the user recorded here (2026-10-24)

1. The unit the constraint applies to is the **vector of dimension `M` of one token** (one row of `X ∈ R^{N×M}`). In
   the first formulations of the idea it was called a *column*; this file says *token vector* throughout.
2. **`G = 4` groups**, one per CPU core. An expert of one hidden neuron is not retained.
3. **`K` and `V` stay dense**: they are the dominant cost and the known floor of the model.
4. The reference is **HFFN `d20` with dense attention**: the dense Transformer is not a reference, its parameter count
   is not comparable.
5. **One layer (`L = 1`)** for the first tests; the inter-layer rule (section 4) keeps the same constraint at every
   layer.
6. Working size of section 7; data: **enwik8, bytes, standard 90M/5M/5M split** (`current/assets/enwik8/`).

## 1. Background: what HFFN already partitions

An FFN computes, for each token, `z = W_up x + b_up`, `h = act(z)`, `y = W_down h + b_down`. HFFN stores it by
**hidden neuron**: neuron `r` owns its active inputs `(up_idx[r], up_val[r])`, its bias `b_up[r]` and its active
outputs `(down_idx[r], down_val[r])`, and the FFN output is exactly the sum of the contributions of all neurons:

```text
y = b_down + Σ_r  down_val[r] ⊗ act( up_val[r] · x[up_idx[r]] + b_up[r] )      (scattered to down_idx[r])
```

The CPU runtime already splits the hidden neurons into **fixed groups**, one per thread. Each group processes the
whole block of `N` tokens, reads `x` in shared mode (no conflict), and writes its contributions into a **private
reduction buffer** of size `N × M`; the buffers are then reduced deterministically. The groups are independent on the
input side; they are coupled on the output side, because every group writes every token.

The idea below starts from these groups and asks: what if each group also owned a **slice of the attention**, and
each **token** were assigned to one group only?

## 2. The idea

### 2.1 Notation

| Symbol | Meaning | Working value |
|---|---|---|
| `V` | vocabulary (bytes) | 256 |
| `M` | model width (`d_model`) | 128 |
| `N` | context length (tokens per sequence) | 128 |
| `N_k` | number of keys a query attends to (`≤ N`, causal: `t + 1`) | — |
| `n_h`, `d_h` | number of heads, width of a head (`n_h · d_h = M`) | 8, 16 |
| `G` | number of groups (= CPU cores) | 4 |
| `m = M / G` | width of a feature block | 32 |
| `S_g` | feature block of group `g`, a set of `m` coordinates of `R^M` | `{32g, …, 32g + 31}` |
| `H_g` | head group of `g`: the `n_h / G` heads whose outputs belong to `g` | 2 heads |
| `H` | number of hidden neurons of the FFN | 512 |
| `n = H / G` | hidden neurons per group, `R_g` their set | 128 |
| `k_up`, `k_down` | fan-in and fan-out of a hidden neuron (`d20`: `≈ 0.2 M`) | 26, 26 |
| `r_l(t)` | group of token `t` at layer `l` (the routing) | `t mod 4` at first |

Contiguous blocks `S_g` are chosen without loss of generality: any partition of the coordinates is the same model up
to a permutation of the residual coordinates.

### 2.2 The constraint

At every layer, the matrix that attention hands to the FFN, `N × M`, satisfies:

- **(C1)** every token vector carries **one group index** `g = r_l(t)`;
- **(C2)** only the coordinates `S_g` of that token vector are **non-zero, and only they are computed**. The other
  coordinates are not computed and then ignored: the goal is to save their computation.

The FFN group `g` then processes exactly the tokens of index `g`, reading only `S_g`. With `G = 4` cores, core `g`
owns head group `H_g`, feature block `S_g`, hidden neurons `R_g`, and the tokens routed to `g`.

### 2.3 What the constraint forces

The constraint is short, but four structural consequences follow from it if the saving is real.

**(a) Heads aligned with groups, and `W_O` block-diagonal.** The attention output of token `t` is
`o_t = W_O a_t`, where `a_t ∈ R^M` concatenates the outputs of the `n_h` heads. Under (C2), only `o_t[S_g]` is
computed: `o_t[S_g] = W_O[S_g, :] a_t`. If the rows `S_g` of `W_O` read heads outside `H_g`, those heads must be
computed for token `t`, and nothing is saved in attention. The saving therefore requires `W_O[S_g, C_j] = 0` for
`j ≠ g`, where `C_j` are the channels of the heads `H_j`: `W_O` is **block-diagonal**, one `m × m` block `W_O^g` per
group, and the head group `H_g` writes only into `S_g`. It also requires `d_h` to divide `m` and `G` to divide `n_h`.

**(b) Queries only for `H_g`.** Token `t` needs the queries of its own heads only: `m · M` multiply-adds instead of
`M²`. Scores and the weighted sum of values are computed for the `n_h / G` heads of `H_g` only.

**(c) Keys and values dense, for every token and every head.** A query of group `g` attends to **all** previous tokens,
whatever their group. The keys and values of heads `H_g` are therefore needed for every token `s`, and since every
head group is queried by some tokens, `K` and `V` are needed for every head and every token: `2 M²` per token, the
same as dense attention. They are the dominant cost and the floor of the model (user decision 3). They are also, with
the outputs of the FFN, the paths that carry information between groups (section 5).

**(d) The FFN of group `g` reads only `S_g`.** Its neurons have their inputs inside `S_g`
(`up_idx[r] ⊂ S_g`, `|up_idx[r]| = k_up ≤ m`). Their outputs are not constrained by (C2): `down_idx[r]` ranges over all
`M` coordinates, as in the reference.

**(e) The residual stream stays complete.** The residual `x_t` keeps its `M` coordinates. At a layer, attention adds
to `S_g` only; the FFN adds to the coordinates its neurons write. (C2) is a property of what is handed to the FFN, not
of the residual stream.

## 3. One layer, written out

Pre-norm layer, token `t`, group `g = r_l(t)`, residual `x_t ∈ R^M`:

```text
ξ_s   = RMSNorm(x_s)                                   for every token s          (M coordinates)
k_s,h = W_K^h ξ_s ,  v_s,h = W_V^h ξ_s                  for every token s, every head h   (dense: 2 M² per token)

for each head h ∈ H_g:
  q_t,h  = W_Q^h ξ_t                                    (rotary position applied to q and k)
  α_t,s,h = softmax_{s ≤ t}( q_t,h · k_s,h / √d_h )
  a_t,h  = Σ_{s ≤ t} α_t,s,h v_s,h

o_t[S_g] = W_O^g · concat_{h ∈ H_g} a_t,h               (m × m block)
o_t[j]   = 0 for j ∉ S_g                                (not computed)
u_t      = x_t + o_t

ζ_t  = (γ_2 ⊙ u_t / rms(u_t)) [S_g]                     (RMS over the M coordinates, input restricted to S_g)
for each neuron r ∈ R_g:
  h_r = act( up_val[r] · ζ_t[up_idx[r]] + b_up[r] )    (up_idx[r] ⊂ S_g)
  y_t[down_idx[r]] += down_val[r] h_r                   (down_idx[r] ⊂ {0, …, M−1})
x'_t = u_t + y_t + b_down
```

After the last layer: `logits_t = W_E^T RMSNorm(x_t^L)`, with the embedding `W_E` tied to the output.

**Choice of normalisation.** The RMS of `u_t` is taken over the `M` coordinates (as in the reference), then the FFN
input is restricted to `S_g`. This costs `M` operations, needs the full `u_t`, which is available (`x_t` is complete,
`o_t` lives on `S_g`), and keeps the normalisation of the reference. The alternative, an RMS over `S_g` only, changes
the model and is not retained at first.

**Relation to the HFFN invariant.** Within group `g`, the FFN is an exact HFFN over the neurons `R_g`. The unrouted
FFN of the reference sums the contributions of the four groups for every token; the routed FFN keeps, for token `t`,
the contribution of group `r_l(t)` only. Each group is a fixed set of hidden neurons with a fixed support, as in the
runtime; what changes is that a group processes the tokens routed to it instead of all the tokens.

## 4. Several layers: the same constraint at every layer

The constraint (C1)-(C2) applies independently at each layer, with a routing `r_l(t)` that may depend on the layer.
The residual stream is complete between layers; the next layer reads it whole through its norm, its queries and its
dense keys and values.

Three routings satisfy the constraint:

| Routing | Definition | Effect (hypothesis) |
|---|---|---|
| fixed | `r_l(t) = r(t)` | a token uses the same 2 of 8 heads and the same quarter of the FFN at every layer: on the query side it lives in a model of width `M / G` through the whole depth |
| rotation | `r_l(t) = (r(t) + l) mod G` | each token visits the four groups every `G` layers, the analogue of the channel shuffle of grouped convolutions |
| learned, per layer | `r_l(t) = argmax W_r^l ξ_t` | content-dependent, needs balancing (section 8, E4) |

At `L = 1` the three coincide for position routing. The base is the **fixed** routing (the simplest); the first test
at `L > 1` compares fixed and rotation, at `L = 4` so that the rotation completes one cycle (`L = 2` covers half of
it). The keys and values of every layer stay dense, so the KV cache of the model is the dense one, `2 L N M` values.

## 5. Information mixing: why this is not four independent Transformers

The risk named by the user is a model that degenerates into four Transformers that never exchange information. The
paths that carry information between feature blocks and between tokens are:

| Path | Between feature blocks | Between tokens | Cost per token and layer |
|---|---|---|---|
| queries read the full `ξ_t` | yes | — | `m M` |
| keys and values read the full `ξ_s` of every token | **yes** | **yes**: a token of group `g` attends to the tokens of every group | `2 M²`, the floor |
| `W_down` writes over all `M` coordinates | yes | — | `n k_down` per routed token |
| residual stream and tied output | yes | — | — |

**The degenerate model, defined.** If the rows of `W_Q`, `W_K`, `W_V` for the heads `H_g` read only `S_g`, and every
`down_idx[r]` of `R_g` lies in `S_g`, then `∂x'_t[S_g] / ∂x_s[S_j] = 0` for `j ≠ g` at every layer, except through
the single scalar of the RMS norm: the four feature blocks evolve independently and meet only at the output. This is the "four independent Transformers" case. In the
proposal, `W_Q`, `W_K`, `W_V` read all `M` coordinates and `W_down` writes all `M`, so the cross-block Jacobian is
non-zero by construction. The degenerate model is kept as an explicit **lower-bound control** (arm `IND`, section 8):
the proposal must beat it beyond the band of the seeds.

**What one layer cannot show.** With `L = 1`, the output of the FFN is followed only by the final norm and the tied
output matrix. Whether `W_down` writes all of `R^M` or only `S_g` then makes little difference: the output matrix
absorbs most of it (hypothesis). One layer therefore tests the mixing *inside* a layer (queries, keys and values that
read everything), not the mixing *between* layers, where a block written by group `g` is read by the other groups of
the next layer. The risk of separation is underestimated at `L = 1`; it is reassessed at `L = 4` before any
conclusion on the architecture.

**Diagnostic of mixing (descriptive, never a gate).** At evaluation, zero the cross-block part of `W_Q`, then of `W_K`
and `W_V`, then of `W_down` (the entries that connect `S_j` to group `g ≠ j`), and report the increase of the
validation loss for each. A large increase means the model uses the mixing; a null increase means it behaves as the
degenerate model on that path.

## 6. Compute model (estimated)

Multiply-adds per token and per layer, `N_k` keys attended. Numbers at the working size, `N_k = 128` (last position of
a window; the causal mean is about half for the score and value terms):

| Term | Reference (HFFN `d20`, dense attention) | Proposal (constraint complete, E3b) |
|---|---|---|
| keys, values | `2 M²` = 32,768 | `2 M²` = 32,768 (unchanged) |
| queries | `M²` = 16,384 | `m M` = 4,096 |
| scores | `N_k M` = 16,384 | `N_k m` = 4,096 |
| weighted values | `N_k M` = 16,384 | `N_k m` = 4,096 |
| output projection | `M²` = 16,384 | `m²` = 1,024 |
| FFN (`d20`) | `2 H k` = 26,624 | `2 n k` = 6,656 |
| **total** | **124,928** | **52,736** |

- Ratio reference / proposal: **2.37** at this size. Keys and values are 62 % of the remaining cost.
- Ceiling while `K` and `V` are dense: 124,928 / 32,768 = **3.8**, even if every other term vanished.
- **Long context.** The score and value terms grow with `N_k`; the constraint divides them by `G`. As `N_k` grows the
  ratio tends to `M / m = G = 4`.
- Larger illustration (`M = 1024`, `H = 4096`, `n_h = 16`, `G = 4`, `N_k = 512`, `k = 205`): reference 6.92 M,
  proposal 3.11 M, ratio 2.23, keys and values 68 % of the remaining cost.

Intermediate arms of the plan, same size: E1 124,928 (supports moved, nothing saved), E2 104,960 (FFN routed),
E3a 92,672 (`W_O` block-diagonal, every head computed), E3b 52,736.

**Mapping onto four CPU cores (target of a later runtime, not of this phase).** Core `g` computes, for **every**
token, the keys and values of its own heads `H_g` (`2 m M` per token, disjoint writes, shared reads of the residual).
It then computes, for the tokens routed to `g`, their queries, their attention over its own keys and values, `W_O^g`
and the FFN group `R_g`. Every step of a layer is local to the core. Each token row is written by one core only (its
attention block and its FFN outputs), so the private reduction buffers of the current runtime disappear. One barrier
per layer remains: the residual of every token must be complete before the next layer computes its keys and values.
With `r(t) = t mod G`, the routed tokens of a causal window are interleaved, so the causal work is balanced across
cores; a contiguous assignment of positions would not be. In decoding (`N = 1` new token), only one core runs the
query, attention and FFN of that token; the keys and values of the new token are still split over the four cores.

## 7. Model, data and training (proposal, fixed in the README of E0)

**Model.** One layer, `M = 128`, 8 heads of 16 with rotary positions, `G = 4` (2 heads per group, `m = 32`), FFN
`H = 512` with GELU, `n = 128` neurons per group, fixed random support with a fixed degree per neuron
`k_up = k_down = 26` (`d20`, 26/128 = 0.203), attention without biases, pre-norm RMSNorm, embedding tied to the output.

| Arm | Parameters outside the embedding (estimated) |
|---|---|
| reference E0 | 65,536 (attention) + 27,264 (FFN, biases included) + 384 (norms) = 93,184 |
| E1, E2 | 93,184 |
| E3a, E3b | 80,896 (`W_O` block-diagonal: −12,288) |
| without attention (calibration) | ≈ 27,648 |
| embedding, tied | 32,768 |

**Data.** enwik8 as raw bytes, standard split by offset, train `[0, 90M)`, valid `[90M, 95M)`, test `[95M, 100M)`
(`current/assets/enwik8/split.json`). A run of a few thousand steps sees a few tens of millions of bytes, less than one
epoch, so the validation loss reflects capacity rather than memorisation (hypothesis). Bits per byte here use windows
of 128 bytes and are **not comparable** to the published enwik8 numbers, which use long contexts.

**Training.** AdamW (β = 0.9, 0.95, weight decay on matrices only), warm-up then cosine decay, gradient clipping at
1.0, batch of 64 windows of 128 bytes sampled from train, fp32. The same seeds, the same data order and the same
initialisation of shared tensors for every arm (paired comparisons). The number of steps is fixed from the speed
measured in the smoke of E0. The learning rate is chosen once by a short sweep on the validation split for the
reference and for E3b; if their optima coincide, one rate is frozen for every arm, otherwise each arm uses the best
rate of the same grid. This choice is written in the README of E0 before the sweep.

**Metrics.** Validation bits per byte at the end and along training; the test split only for the final reading of a
retained arm. Per arm: median and [min–max] over the seeds, and the paired difference to its control per seed.
Diagnostics: bits per byte by position class `t mod 4`, the mixing diagnostic of section 5, and, for a learned router,
the load per group and the dropped tokens.

**Implementation of the constraint in PyTorch.** Since only quality is measured, (C2) may be implemented by packing
the tokens of each group (positions `t ≡ g mod 4` form a contiguous slice after a reshape, the route-major packing of
`moe/`) or by computing every head and selecting those of `H_g`. Both give the same function and the same gradients:
an unselected head of token `t` multiplies a zero and receives a zero gradient. The packed form is preferred because it
also saves the computation. The saving in time is not measured in this phase; it is counted with section 6.

**Checks before any training (smokes, by the agent).** For each arm: parameter counts equal to the table; every
`up_idx` of `R_g` inside `S_g` and every degree equal to 26; `o_t` exactly zero outside `S_g`; the FFN output of token
`t` independent of `u_t[j]`, `j ∉ S_g`, except through the RMS; packed and selected implementations identical within
float32 rounding on a fixed batch.

## 8. The ladder of steps

One mechanism per step (`AGENTS.md` section 3), each compared with the step before it. Each step is an ablation with
its own README, written before any measurement, that fixes its seeds, statistic, bound and band.

| Step | Single mechanism | Attention | FFN | Control | Question |
|---|---|---|---|---|---|
| **E0** | reference and calibration | dense | HFFN `d20`, random support | — | quality level, band of the seeds, **dynamic range** of attention |
| **E1** | FFN inputs inside the blocks | dense | `up_idx[r] ⊂ S_g`, `k_up = 26` of 32, outputs over `M`, no routing | E0 | cost of aligning the hidden neurons on the feature blocks, at equal density |
| **E2** | top-1 routing of the FFN | dense | token `t` uses `R_{t mod 4}` only | E1 | cost of routing the FFN (active FFN compute / 4, same parameters) |
| **E3a** | `W_O` block-diagonal | head group `H_g` writes `S_g`, every head computed for every token | as E2 | E2 | price of the block-diagonal output projection (−12,288 parameters) |
| **E3b** | **the complete constraint** | token `t` computes `H_{t mod 4}` only, `o_t` on `S_g` | as E2 | E3a | cost of routing the attention, same parameters: the step that varies only the attention |
| **IND** | degenerate lower bound | `W_Q`, `W_K`, `W_V` of `H_g` read `S_g` only | `down_idx ⊂ S_g` | E3b | is the mixing of E3b worth anything over four separated models |
| **E4** | routing | as E3b | as E3b | E3b | content routing: (a) balanced byte hash, (b) learned router; one per attempt |
| **E5** | depth | `L = 4` | as the best of E3b/E4 | HFFN `d20` at `L = 4` | inter-layer mixing; fixed routing against rotation |

**E0 decides whether the task can discriminate.** One layer on bytes relies largely on local statistics. The gap
`Δ_attn` = bits per byte without attention − bits per byte of the reference is the only range in which variants of
attention can differ. The README of E0 fixes, before its runs, the minimum ratio of `Δ_attn` to the band of the seeds
below which the task is declared non-discriminating; the next options are then a longer context, a wider model or
another dataset, a decision of the user.

**E4, routing.**

- (a) *Balanced byte hash*: the 256 byte values are assigned to the four groups once, greedily, so that each group
  receives about a quarter of the training bytes (a plain hash would overload the group of the space and of the
  frequent letters). Content-dependent and static.
- (b) *Learned router*: `ρ_t = W_r ξ_t` (`G M` = 512 parameters per layer), top-1, one router per layer for both the
  head group and the FFN group. The output of the chosen group is scaled by its probability so that the router
  receives a gradient; a balance loss of the Switch type (`G Σ_g f_g P_g`) is reported separately from the task loss;
  a capacity per group (the `token_cap` of `moe/`) with the dropped tokens passing through the residual and counted.
  Whether both `o_t` and `y_t` are scaled, and the capacity factor, are fixed in its README. Under a causal mask a
  learned router can unbalance the causal work across cores; the load per group is reported.

**Not in the plan.** Reducing the cost of `K` and `V` (user decision 3). A routing where a token computes two groups:
the `moe/` results below point to it, but it violates (C1) and would be an amendment decided by the user.

## 9. Prior art

**In this repository** (from `current/moe/gpu/m1/docs/ROADMAP.md`, read but not re-verified on its runs): in the
`moe/` branch, top-1 selection was *strongly worse* than top-2 on layers 3 and 4, read as complementary pairs rather
than autonomous experts; the best fixed disjoint partition found after training (coactivation E32) still left an
error 2.34 times that of hidden selection at the same budget, which rejected fixed post-hoc partitions. Those results
concern partitions built after training, on the FFN alone, without attention; they do not refute a model trained
from scratch under the constraint, but they are the closest evidence and it is unfavourable to top-1. What can be
reused from `moe/`: route-major packing of the tokens, a capacity per route, a balance measured on active connection
work rather than on the number of neurons, the auxiliary loss reported apart from the task loss
(`current/moe/cpu/m1/ffn/training/conditional_hffn/`).

**In the literature** (cited from memory, to be checked before any text that relies on it): selection of heads per
token (Mixture of Attention Heads, Zhang et al. 2022; SwitchHead, Csordás et al. 2023; MoH, 2024); top-1 expert
routing (Switch Transformer, Fedus et al. 2021); static hash routing (Hash Layers, Roller et al. 2021); experts of a
single neuron with a learned product-key router (PEER, He 2024); grouped convolutions and the channel shuffle that
restores mixing between groups (ShuffleNet, Zhang et al. 2018). What would be specific here is the coupling: the head
group and the FFN group of a token are the same, and the feature block written by the heads is exactly the input
support of the FFN group.

## 10. Expected weaknesses (hypotheses, to be read in the runs)

- On the query side, a token sees through 2 heads of 8 at each layer; with fixed routing, through the whole depth.
- With position routing, each head group serves one class of positions: the four groups may learn the same function
  four times, each from a quarter of the queries.
- The ceiling of the saving is set by the dense keys and values: 3.8 at the working size, `G` at long context.
- The routing of a single token in decoding leaves three cores idle for the attention and FFN of that token.
- One layer underestimates the separation risk between groups (section 5).

## 11. Launching and layout

- **Smokes and checks** (section 7) are run by the agent. **Quality campaigns** are campaigns, since their numbers are
  used (`AGENTS.md` section 6): on the Mac they are launched by the user, or under a grant; on a VM by the agent. They
  are not timing campaigns, so the 30-minute ceiling does not apply, but each run is sized to minutes.
- Proposed layout, following section 4 for a PyTorch study: `experiments/sparse_attention/common/llm/training/<step>/`
  for the ablations, `current/sparse_attention/` for this plan and, later, the retained model code and its
  `STATUS.md`. The existing empty folders `current/sparse_attention/` and `experiments/sparse_attention/` are used.

## 12. Open points

- Fix in the README of E0: number of seeds (at least 3), number of steps after the smoke, learning-rate procedure,
  minimum ratio `Δ_attn` / band.
- Fix in each later README: its bound on the paired difference of bits per byte and its band, before its first run.
- Before E5: the order of the comparison fixed against rotation at `L = 4`, and whether a learned router per layer
  enters.
