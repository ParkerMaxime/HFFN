# Sparse attention by HFFN groups — idea, theory and plan

Written on 2026-10-24 at the user's request, from the discussion of the same day. Sections 1 to 12 were written before
any run: every number in them is **estimated** (a calculation) and every explanation of an expected effect is a
**hypothesis** (`AGENTS.md` section 7). Section 13, added on 2026-10-25, reports the first ladder run on a cloud
session; its figures are **measured** where it says so, with their run, and its readings carry their class.
Section 14, added on 2026-10-25 before any of its runs, plans the second series (E6-E9): quality against compute.

## 0. Status and scope

- **What it is.** A quality study, in PyTorch, of a Transformer whose attention is partitioned by the same groups as
  the hidden neurons of its HFFN feed-forward network, so that each token is processed, in attention and in the FFN,
  by one group only. It is a **different model** from a dense-attention Transformer, trained from scratch; it is an
  approximation of dense attention in the sense that it is a restriction of it, not a compression of a trained model.
  Comparisons are on **quality**, at matched parameters wherever the constraint allows; since 2026-10-25 also at
  equal inference compute per token (user decision 8, section 14.2, prospective).
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

The open points of the first ladder were all fixed in the READMEs of its steps before their runs (section 13.1). The
points that remain open after it are listed in section 13.5; the second series that answers part of them is
section 14.

## 13. Results of the first ladder (2026-10-24/25)

**Provenance.** A Claude cloud session on the branch `sparse-attention-cloud`, under the brief
`experiments/sparse_attention/CLOUD.md` (no machine constraint, the scientific rules kept). That branch is an orphan
slice of the repository: it shares no history with `main` and is **not merged**; the records cited below live on it,
under `experiments/sparse_attention/common/llm/training/<step>/runs/`, with one `README.md` (committed before the
first run of the step) and one `RESULTS.md` per step, and `experiments/sparse_attention/SUMMARY.md`. Nothing was
promoted. Reviewed on 2026-10-25 (code of `lib/`, READMEs, results, validation curves); the observations of that
review are marked as such.

### 13.1 Protocol as fixed by E0 and the later READMEs

- Model and data of section 7. PyTorch 2.14.1 (PyPI build, CPU; the CPU wheel index was refused by the proxy),
  4 jobs × 1 thread. FFN and block-diagonal projections as masked dense matrices (section 0). Constraint (C2) in the
  packed form for position routing, in the selected form for content routing; 121 pre-training checks pass
  (parameter counts, supports and degree 26, `o_t` exactly zero outside `S_g`, FFN independence, packed = selected,
  causality, determinism).
- Seeds 0-3 for every arm; seed 100 for the learning-rate sweeps only. 4,000 steps, batch 64 × 128 bytes
  (32.8 M bytes, 0.36 epoch). Warm-up 5 %, cosine to 10 % of the peak.
- Learning rate at `L = 1`: sweep on E0 and E3b, seed 100, 2,000 steps, grid `5e-4 … 6.4e-2`. Both optima at
  **0.032**, interior, coincident: one rate for every arm at `L = 1`. At 0.064, E3b jumps from 2.97 to 3.33 bpb (sweep,
  one seed): the constrained arms train one grid step below that jump. At `L = 4` (E5): new sweep, 1,500 steps,
  optimum 0.016 for `REF4`, 0.008 for the constrained arms, each applied by the README rule.
- Statistic: final validation bits per byte over the whole 5 M valid split (windows of 128, not comparable to
  published enwik8 figures); per arm median [min–max] over the four seeds; paired difference
  `d_s = bpb(candidate) − bpb(control)` per seed, the control rerun in the campaign of each step.
- `R_min = 10` for E0. Bound **`τ = 0.075` bpb** (5 % of the `Δ_attn` measured in E0) for every later step:
  within if `max_s d_s ≤ τ`, beyond if `min_s d_s > τ`, inconclusive otherwise.
- Harness events (no candidate failure, no figure drawn from them): one check tolerance set too tight (1e-6 → 1e-5)
  before any training; a first E0 run stopped after its sweep because both optima sat at the grid edge
  (`e0/runs/run_20261024T180451Z/ABORTED.md`), its four started final jobs killed by hand.

### 13.2 Measured

Every row is measured in the qualified run named (status `passed`, exit 0, 8/8 jobs or more, full validation split).
Positive difference = candidate worse.

| Step | Run | Candidate bpb, median [min–max] | Control | `d_s` median [min–max] | Verdict |
|---|---|---|---|---|---|
| E0 | `e0/runs/run_20261024T182902Z` | E0 2.3925 [2.3913–2.3975]; NOATT 3.8949 [3.8945–3.8959] | — | `Δ_attn` 1.5028, band 0.0062, ratio 243 | discriminating |
| E1 | `e1/runs/run_20261024T192919Z` | 2.3878 [2.3867–2.3936] | E0 | −0.0045 [−0.0088, −0.0000] | within, better in every seed |
| E2 | `e2/runs/run_20261024T200327Z` | 2.6418 [2.6392–2.6472] | E1 | +0.2516 [+0.2504, +0.2605] | beyond |
| E3a | `e3a/runs/run_20261024T203226Z` | 2.7572 [2.7546–2.7606] | E2 | +0.1160 [+0.1104, +0.1173] | beyond |
| E3b | `e3b/runs/run_20261024T205712Z` | 2.8692 [2.8627–2.8729] | E3a | +0.1131 [+0.1021, +0.1160] | beyond |
| IND | `ind/runs/run_20261024T211955Z` | IND 2.8739 [2.8704–2.8821] | E3b better than IND by | +0.0031 [+0.0007, +0.0194] | not met (needs > 0.075) |
| E4a | `e4a/runs/run_20261024T214033Z` | 2.6393 [2.5520–2.6778] | E3b | −0.2264 [−0.3209, −0.1919] | within, better in every seed |
| E4b | `e4b/runs/run_20261024T221415Z` | 2.9503 [2.8474–3.0360] | E3b | +0.0825 [−0.0223, +0.1673] | inconclusive |
| E5 | `e5/runs/run_20261024T225110Z` | E4a_L4 2.1252 [2.1204–2.1347]; REF4 2.0337 [2.0275–2.0362] | REF4 | +0.0932 [+0.0859, +0.1022] | beyond |

Parameters outside the embedding: E0, E1, E2 93,184; E3a, E3b, E4a 80,896; E4b 81,408; IND 44,032; NOATT 27,520
(the "≈ 27,648" of section 7 counted the norm gain of the removed attention); `L = 4`: constrained 323,200, `REF4`
372,352.

Comparisons inside the E5 run (all measured, same campaign):

| Candidate | Control | `d_s` median [min–max] | Verdict |
|---|---|---|---|
| E3b_L4 (position) | REF4 | +0.3804 [+0.3775, +0.3926] | beyond |
| E4a_L4 (hash) | E3b_L4 (position) | −0.2900 [−0.2942, −0.2821] | better in every seed |
| E4a_L4rot | E4a_L4 | −0.0038 [−0.0130, +0.0084] | within |
| E3b_L4rot | E3b_L4 | −0.0031 [−0.0123, −0.0007] | within, below in every seed, of the order of the seed band |

E4b router (measured, final validation): load per group 0.19-0.28, Switch balance loss 1.00-1.02 along training,
1.6-4.6 % of tokens dropped by the capacity. Mixing diagnostic of section 5 (descriptive, never a gate): zeroing the
cross-block parts raises the bpb by several units in every constrained arm (E3b: `W_Q` +1.3, `W_K`/`W_V` +2.0,
`W_down` +6.0) and is exactly zero for IND by construction.

Reproducibility (measured): the control of every step reproduces the figures of the previous step to the last digit,
and the 12 grid points common to the two E0 sweeps are identical.

### 13.3 Reading against the questions of this plan

- **E0, can the task discriminate?** Yes: `Δ_attn` is 243 seed bands (measured).
- **Cost of the complete constraint, position routing, `L = 1`:** E3b 2.8692 against E0 2.3925, **+0.477 bpb**, 32 %
  of `Δ_attn`. This is a reading across the campaigns of E0 and E3b, admissible here only because the controls
  reproduce bit for bit across campaigns (training is deterministic for the same code, seed and thread count); it is
  not a paired difference. Its split into E2 (+0.25), E3a (+0.12), E3b (+0.11) holds **for position routing only**:
  under the hash route, which E4a shows to be the better one, the split is not measured.
- **E1, aligning the hidden neurons on the feature blocks:** free at equal density (measured).
- **Section 5, mixing (IND):** at `L = 1` the proposal does not beat the four separated models beyond `τ`; it is ahead
  in every seed by 0.001-0.019, while IND has 46 % fewer parameters (measured). Reading from the review (hypothesis, structural):
  at the first layer the input of the queries, keys and values of every token is `RMSNorm(emb[byte])`, a function of
  the byte only, and a learned slice of 32 coordinates can encode the 256 bytes; restricting `W_Q`, `W_K`, `W_V` to
  `S_g` therefore removes almost no information at `L = 1`. The IND step at one layer could hardly show a benefit of
  mixing; the risk named in section 5 is **not measured**, since IND was not run at `L = 4`. The mixing diagnostic shows
  that the paths are used, not that they are worth anything.
- **Section 4, fixed against rotation at `L = 4`:** no measurable difference under either route (measured, within `τ`
  and within a few thousandths).
- **E4, routing:** the balanced byte hash beats position routing at `L = 1` (−0.226) and at `L = 4` (−0.290), at no
  extra parameter (measured). Why is a hypothesis: with a byte-class route each group serves a consistent part of the
  byte vocabulary, whereas with `t mod 4` the four groups learn the same function from a quarter of the positions
  each (section 10); not isolated (no random balanced route). The learned router of E4b, as specified in its README,
  is inconclusive against position routing.
- **E5, depth:** the best constrained model (hash, `L = 4`) is +0.093 bpb above `REF4`, just beyond `τ`; position
  routing +0.380 (measured). That the constraint costs less at depth (+0.093 against +0.247 for the hash route at
  `L = 1`) is a cross-campaign reading with confounds: the learning rates differ (0.032 at `L = 1`; 0.016 and 0.008 at
  `L = 4`), `REF4` has 15 % more parameters, and `τ` was sized on the `Δ_attn` of `L = 1` (`Δ_attn` not measured at
  `L = 4`).
- **Section 6, compute:** unchanged, estimated: 52,736 against 124,928 multiply-adds per token and layer (2.37). No
  time was measured.

### 13.4 Limits found in the review (2026-10-25)

- **Training horizon.** Every validation curve still falls at the last point (descriptive, curve on 4,096 windows:
  −0.024 for E0, −0.021 for E2, −0.010 for E3b over the last 500 steps). The costs above are costs at a fixed budget of
  4,000 steps, not at convergence; whether the gaps narrow or widen with training is not measured.
- **Plateaus of the content-routed arms** (descriptive, curves; hypothesis for the cause). E4a starts above E3b
  (3.59 against 3.34 at step 500, seed 0) and leaves a plateau at a step that depends on the seed (about 1,000 for
  seed 3, 2,000-2,500 for seeds 1 and 2). In E4b, seeds 1 and 3 leave theirs near step 3,000, seeds 0 and 2 not before
  the end. The spread of E4a (0.126) and the "bimodal" outcome of E4b are therefore probably the time of escape from a
  plateau cut by a fixed horizon, rather than distinct routing equilibria; the statement that the learned router is
  "on average worse than position routing" is not supported beyond the inconclusive verdict.
- **Learning rate.** One seed and 1,500-2,000 steps chose the rate of 4,000-step runs. The constrained arms at `L = 1`
  run at 0.032, one grid point below a jump; E4a and E4b were never swept; at `L = 4` the optimum of the constrained
  arms is four times lower. The validation split serves both the choice and the figures; the test split was never read.
- **Parameters.** E3a, IND and E5 compare arms of different parameter counts by construction (decision 4).
- **Not run:** IND at `L = 4`; a parameter-matched IND; a random balanced route; the hash route on the FFN alone
  (E2 with hash); any other router configuration or a router per layer; other depths, contexts or datasets; any
  comparison at equal compute (section 0 fixes comparisons at matched parameters); any timing.

### 13.5 Open questions after the first ladder (discussed with the user on 2026-10-25; outcome in section 14.6)

1. Under the hash route, how does the cost split between the routed FFN and the routed attention? (control named in
   the review: E2 with the hash route against E1, then E4a against it.)
2. At depth, is the constrained model worth more than four separated models? (IND at `L = 4` under the hash route,
   against E4a_L4.)
3. Do the gaps and the plateaus of the content-routed arms survive a longer training and a learning rate chosen for
   them?
4. Is the constrained model better than a dense HFFN model at equal compute (estimated: E4a_L4 about 210,944
   multiply-adds per token, a dense `d20` model at `L = 2` about 249,856), and is a top-1 routed FFN better than an
   unrouted FFN of a quarter of the neurons? This changes the comparison basis of section 0 and would be an amendment
   decided by the user.
5. Does the hash gain come from the content of the byte or only from leaving the position route? (a random balanced
   route).

## 14. Second series: the trade-off between quality and compute (E6-E9, written 2026-10-25)

Written after the discussion of 2026-10-25 with the user, before any run of the series. Every number in this section
is **estimated** (a calculation, or a cost extrapolated from the wall times of section 13) and every expected effect
is a **hypothesis**. Each step gets its own README before its runs, which freezes what this section proposes.

### 14.1 Why a second series

The first ladder (section 13) shows that the method is **functional**: a model under the complete constraint trains
stably from scratch at one and at four layers, and with the balanced byte hash its loss stays bounded (+0.093 bpb
against `REF4` at `L = 4`, measured). It does not say whether the method is **worth** anything. A restriction of
dense attention is expected to lose quality; the question is whether the compute it saves pays for that loss. The
first ladder cannot answer it, for three reasons:

- every comparison was made at matched parameters (section 0), never at matched compute;
- every gap was read at a fixed budget of 4,000 steps on curves that still fall, and the gap moves with the phase of
  the schedule (section 14.4, E6);
- the saving was counted for one layer and one context only.

The second series establishes three things, cleanly and on one fixed design:

1. **The loss of quality** of the constrained model, at the same depth as the dense model.
2. **The compute and memory saved**, counted (not timed) per token, for every model of the series.
3. **The quality at equal inference compute**: for the same multiply-adds per token at inference, is the constrained
   model better, equal or worse than the dense-attention HFFN `d20` model?

The third is the question that decides whether the method is useful. The first two are what it is made of.

### 14.2 Decisions of the user recorded here (2026-10-25)

7. **Routing is frozen** to the route of E4a: the balanced byte hash (`balanced_byte_hash`, built once from the byte
   counts of the train split, the same for every seed), fixed at every layer, without rotation. Routing is a later
   study of its own; the learned router of E4b and the questions 1 and 5 of section 13.5 wait for it.
8. **New comparison basis, prospective.** Besides matched parameters (section 0), models are compared at **equal
   inference compute per token** (multiply-adds of one forward pass per token, logits included). This is the primary
   criterion of the series, because inference compute is what decides whether the method is useful. Equal
   **training** compute is a secondary reading. The verdicts of section 13 are unchanged.
9. **Compute and memory are counted, not measured**, in this phase. The cloud machine is not suited to timing; real
   performance is measured once the architecture and its effects are understood and fixed.
10. **The axis of compute is depth**, at `M = 128`.
11. **Two contexts**: `N = 128`, and `N = 512` trained entirely at 512 (no context extension).
12. **IND is kept**, as one control inside the series.
13. **Trainings are short**: cloud compute is limited and the series has many runs. The horizon is the shortest at
    which the gap is stable, not the horizon of convergence of the loss (E6).

User decision 5 (one layer for the first tests) is fulfilled; the series runs at two to six layers.

### 14.3 Fixed design of the series

- **Model.** That of section 7 (`M = 128`, 8 heads of 16 with rotary positions, `G = 4`, `m = 32`, HFFN `d20` with
  `H = 512`, `k_up = k_down = 26`, tied embedding, RMSNorm), at `L` layers. Data: enwik8 bytes, the split of
  section 7.
- **Two families**, the same at every depth:
  - `REF<L>`, the reference: dense attention, HFFN `d20` unrouted (E0 at `L` layers; `REF4` of E5).
  - `E4a_L<L>`, the constrained model: the complete constraint (E3b) with the frozen hash route.
  - One control, `IND_L4`, the four separated models of section 8 at `L = 4`, under the same hash route.
  - A model at `N = 512` carries the suffix `_N512`.
- **Why depth.** Depth keeps everything else constant: embedding and logits, `d_head`, the number of heads per group,
  the density `d20`. Only `L` varies, and the two families get the same knob. Width would also change the cost of the
  logits and the heads per group, so two effects would vary together.
- **Tokens per step fixed**: 8,192 bytes per step at every context (batch 64 × 128, or 16 × 512). Every arm sees the
  same bytes in the same order for a given seed.
- **Schedule: warm-up, stable, decay (WSD)** instead of the cosine of section 13:
  - warm-up of 200 steps;
  - constant peak rate;
  - linear decay to zero over the last 20 % of the horizon.

  Why: a point taken in the middle of a cosine run is not the end of a schedule. A WSD run, checkpointed on its
  plateau, gives several horizons, each a true end of schedule, for about 1.2 times the cost of the longest.
- **Figures.** Final validation bits per byte over the whole valid split. At `N = 512` also by position range
  (0-127, 128-255, 256-511), which shows whether the longer context is used. The test split is read once, at the end
  of E8 and E9, for the arms of the frontier, and reported beside; it selects nothing.
- **Statistic.** Paired per seed, `d_s = bpb(candidate) − bpb(control)`, with the three-outcome rule of section 13.1
  and its bound kept unchanged, `τ = 0.075` bpb. For a pair where the candidate costs less compute, a fourth outcome
  is named: *the candidate dominates* if `max_s d_s < 0`.

### 14.4 The steps

| Step | What it runs | Why | Seeds | Cost (estimated) |
|---|---|---|---|---|
| **Preparation** | harness: WSD with plateau checkpoints and decay branches; tokens per step instead of a fixed batch; position ranges in the evaluation; arms `REF<L>`, `E4a_L<L>`, `IND_L4` with hash; MAC and memory counter checked against section 6; checks and a smoke of every arm, `L = 2` and `N = 512` included | makes the series possible; no figure | — | < 0.5 h |
| **E6** | horizon: `REF2` against `E4a_L4`, `N = 128`, one WSD run each to step 6,400, decay branches ending at 2,000, 4,000 and 8,000 steps | chooses the shortest horizon at which the gap is stable | 0, 1 | ≈ 2.2 h |
| **E7** | learning rates at that horizon: 3 rates (grid × 2, extended if the optimum is at an edge) for `REF1`, `REF4`, `E4a_L2`, `E4a_L6` | one fair rate per family and depth | 100 | ≈ 2.2 h |
| **E8** | frontier at `N = 128`: `REF1`, `REF2`, `REF3`, `REF4`, `E4a_L2`, `E4a_L4`, `E4a_L6`, `IND_L4` | answers the three questions of 14.1 at `N = 128` | 0, 1, 2 | ≈ 4.2 h |
| **E9** | frontier at `N = 512`: the same eight arms, `_N512` | the same at a longer context, where the constraint saves more | 0, 1, 2 | ≈ 7.2 h |

Costs extrapolate the wall times measured in section 13 (about 13 minutes per layer for 4,000 steps, 4 jobs × 1 thread;
`× 1.7` at `N = 512` for the larger share of the scores). The total is **about 16 hours** if E6 retains 4,000 steps.
It doubles for E7 to E9 if E6 retains 8,000 steps; the series then stops after E6 and the user decides.

**E6, the horizon.**

- *Why.* In the E5 run the paired gap `E4a_L4 − REF4` was not stable along training. Its mean over the four seeds went
  +0.092 at step 500, +0.050 at 2,000 and +0.094 at 4,000, in the same direction for every seed (descriptive, from
  `e5/runs/run_20261024T225110Z/main/*/metrics.csv`, intermediate points of a cosine schedule).
  - Hypothesis: the gap grows again in the decay, because the two arms ran at different peak rates (0.016 and
    0.008).
  - A short horizon chosen blindly could read any point of that curve.
- *What it must show.* The horizon `H*` at which the gap between the two families no longer moves. Only that horizon
  makes the short runs of E7-E9 admissible. It shows neither the loss at convergence nor the gap at convergence.
- *Rule, frozen in its README.* `H*` is the smallest of 2,000 and 4,000 steps such that
  `|gap(H) − gap(8,000)| ≤ 0.025` bpb for both seeds, the gap being paired per seed. 0.025 is about the spread of
  `d_s` measured in E5, 0.016. If neither passes, `H* = 8,000` and the user decides on the budget before E7.
- *Pair.* `REF2` and `E4a_L4` are the pair of nearest compute in the middle of the frontier. Their rates are
  taken from section 13: `REF2` interpolated geometrically between 0.032 (`L = 1`) and 0.016 (`L = 4`), about 0.025;
  `E4a_L4` 0.008. These rates were chosen under cosine; this is a limit of E6, stated in its README.
- *Secondary reading (b).* The three horizons of the pair also give, by interpolation, the reading at equal
  **training** compute: a constrained model may take more steps for the same training compute. This reading is
  estimated, on one pair only.

**E7, the learning rates.**

- *Why.* In section 13 one rate was chosen on 1,500-2,000 steps for runs of 4,000. The constrained arms at
  `L = 4` had an optimum four times lower than at `L = 1`. E4a was never swept, and the schedule changes to WSD.
  A frontier whose arms run at unfair rates would compare the rates, not the models.
- *What it must show.* For each family, the best rate of the grid at `H*` at the two extreme depths (`REF1` and `REF4`,
  `E4a_L2` and `E4a_L6`). The middle depths take the geometric interpolation in `L`, and `IND_L4` takes the rate of `E4a_L4`. One seed, so it selects;
  it proves nothing.
- The rates of `N = 128` are kept at `N = 512`, which holds the bytes per step fixed (hypothesis; a limit of E9).

**E8 and E9, the frontier.**

- *Why.* A single pair of models cannot say whether one family is better at equal compute, because no two models of
  the two families have exactly the same cost. A frontier of each family over depth can: the families interleave on
  the axis of compute (14.5), so each constrained model has a dense neighbour just above it in compute.
- *What they must show:*
  - **Quality at equal compute, measured.** Each constrained model has a dense model that costs more:

    | Constrained | Dense | Compute saved at `N = 128` | At `N = 512` |
    |---|---|---|---|
    | `E4a_L2` | `REF1` | 12 % | 27 % |
    | `E4a_L4` | `REF2` | 14 % | 29 % |
    | `E4a_L6` | `REF3` | 14 % | 29 % |

    If the constrained model does as well or better in a pair, at less compute, the statement is measured without
    interpolation: *dominates*, or *equivalent within `τ` at less compute*. If it is worse, the size of the loss at
    equal compute needs an interpolation.
  - **Quality at equal compute, estimated.** The dense family is interpolated linearly in `log C` at the compute of
    each constrained model, inside its range only. At `N = 128`, `E4a_L2` lies below `REF1` and gets no interpolated
    value.
  - **Loss of quality at equal depth**, measured, with the same rule, on two pairs in each campaign: `E4a_L2` against
    `REF2` and `E4a_L4` against `REF4` (E8), and their `_N512` counterparts (E9). `REF4` is in the frontier for this
    reading only (user decision, 2026-10-25); it also extends the dense family above `E4a_L6`. The pair `E4a_L4`/`REF4`
    of E5 was measured under the former protocol and is not compared with this one (different campaigns).
  - **Mixing at depth** (question 2 of section 13.5): `IND_L4` against `E4a_L4`, the same rule. If IND is not worse
    beyond `τ`, a favourable frontier measures the conditional computation of four separated models, not the coupling
    of heads and FFN groups that this plan proposes (section 5). The IND of section 13 could not answer this, being
    at one layer (13.3).
  - **Effect of the context**, E9 against E8 inside each family: a cross-campaign reading, stated as such. E9 holds
    its own pairs; whether the constraint pays more at `N = 512`, where it saves more, is read from the outcomes of
    the pairs of each campaign, not from differences of figures across them.

### 14.5 What is counted (estimated)

Inference multiply-adds per token, with the convention of section 6 (`N_k = N`) and the logits (`V M` = 32,768):

| Model | `N = 128` | `N = 512` |
|---|---|---|
| `E4a_L2` | 138,240 | 187,392 |
| `REF1` | 157,696 | 256,000 |
| `E4a_L4` | 243,712 | 342,016 |
| `REF2` | 282,624 | 479,232 |
| `E4a_L6` | 349,184 | 496,640 |
| `REF3` | 407,552 | 702,464 |
| `REF4` | 532,480 | 925,696 |

Per layer, the saving goes from 2.37 at `N = 128` to 2.89 at `N = 512` (223,232 against 77,312), towards `G = 4` at
long context (section 6). Counting `IND_L4` is part of the preparation.

**Memory, and why it must be stated plainly.** At **equal depth** the constraint saves memory of weights and AdamW
states only through the block-diagonal `W_O` (−13 % outside the embedding). It does not change the KV cache, since `K`
and `V` stay dense (decision 3). The training activations it saves are counted by the preparation (queries, scores,
attention outputs and FFN activations per token divided by `G`), and nothing of it is measured. At **equal compute**,
memory goes the other way: the constrained model is deeper. Parameters outside the embedding (computed from the
per-layer counts of section 13):

| Pair | Constrained | Dense |
|---|---|---|
| `E4a_L4` / `REF2` | 323,200 | 186,240 |
| `E4a_L6` / `REF3` | 484,736 | 279,296 |
| `E4a_L4` / `REF4` (equal depth, for comparison) | 323,200 | 372,352 |

Its KV cache per token (`2 L M` values) is twice that of `REF2`. A favourable frontier would therefore be a statement
about **compute**, bought with **more weights and a larger KV cache**. This is the usual price of conditional
computation and must be written with the result, not after it.

### 14.6 What the series will not show

- No time and no measured memory (decision 9).
- No convergence: the horizon is chosen for the stability of the gap, not of the loss.
- No other routing (decision 7), width, dataset or model size.
- No comparison with other efficient attention mechanisms.
- No statement that holds beyond `M = 128` and enwik8.
- At equal compute, the dense family is the only control. A narrower dense model, or an unrouted FFN of a quarter of
  the neurons (question 4 of section 13.5, second part), are not in the series.

**Status of the questions of section 13.5.**

| Question | Status |
|---|---|
| 2 | E8, `IND_L4`, under the hash route |
| 3 | E6 for the horizon, E7 for the rates. The plateaus of `L = 1` are not revisited: the series starts at `L = 2`, and a plateau there would show as a spread of the seeds. |
| 4, first part | E8, E9 |
| 1 and 5 | wait for the routing study |
| 4, second part | not planned |

### 14.7 Launching

- On a Claude cloud session, as in section 13. One brief for the whole series, chained in this order: preparation,
  E6, then a stop if `H* = 8,000`, then E7, E8, E9.
- Each step has its own folder `experiments/sparse_attention/common/llm/training/<step>/` and a README committed
  before its first run. It has its own run records and its own `RESULTS.md`.
- A step that fails its harness is not an attempt (`AGENTS.md` section 5).
- Nothing is promoted: the series measures, it retains no model.
