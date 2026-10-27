# Sparse attention by HFFN groups — idea, theory and plan

Written on 2026-10-24 at the user's request, from the discussion of the same day. Sections 1 to 12 were written before
any run: every number in them is **estimated** (a calculation) and every explanation of an expected effect is a
**hypothesis** (`AGENTS.md` section 7). Section 13, added on 2026-10-25, reports the first ladder run on a cloud
session; its figures are **measured** where it says so, with their run, and its readings carry their class.
Section 14, added on 2026-10-25 before any of its runs, plans the second series (E6-E9): quality against compute.
Section 14.8, added on 2026-10-26, reports that series and the reserves of its review; 14.1 to 14.7 are left as
written before the runs. Section 15, added on 2026-10-26 before any of its runs, plans the third series (E10-E13):
memory and relative time of the implementation, and width; it amends decision 9 (decision 14, section 15.2).
Section 15.8, added on 2026-10-27, reports that series and the reserves of its review; 15.1 to 15.7 are left as
written before the runs.
Section 16, added on 2026-10-27 outside the series and before any run, analyses how the partition could make the
HFFN FFN faster on CPU and GPU; it is estimates and hypotheses only, and changes no step of the plan.
Section 17, added on 2026-10-27 before any of its runs, records the user's criterion of quality per inference time
(decision 16) and plans E14, the measured frontier of quality against inference time, in prefill and batched decoding.

## 0. Status and scope

- **What it is.** A quality study, in PyTorch, of a Transformer whose attention is partitioned by the same groups as
  the hidden neurons of its HFFN feed-forward network, so that each token is processed, in attention and in the FFN,
  by one group only. It is a **different model** from a dense-attention Transformer, trained from scratch; it is an
  approximation of dense attention in the sense that it is a restriction of it, not a compression of a trained model.
  Comparisons are on **quality**, at matched parameters wherever the constraint allows; since 2026-10-25 also at
  equal inference compute per token (user decision 8, section 14.2, prospective).
- **What it is not.** It is not an evolution of the HFFN runtime (`runtime/`), and no absolute time or speed is
  measured in this phase. The compute it saves is counted analytically (section 6). Since 2026-10-26 the relative
  time of two models of this study, in one process under the timing rule of `AGENTS.md` section 7, is admitted on
  the cloud session (decision 14, section 15.2). A native implementation, if any, comes later and then follows every
  runtime rule of `AGENTS.md`.
- **Governance.** The branch was opened by the user on 2026-10-24 as a **secondary** study: the runtime keeps
  priority (`AGENTS.md` sections 4 and 5, *Objective*; `DECISIONS.md`). When a machine, a VM session or a review is
  needed by both, the runtime takes it. Like `lateral/`, it is an architecture study decided by the user, outside the
  runtime goal of section 1.
- **Representation of the FFN in this phase.** There is no HFFN runtime path here. A fixed-support FFN may be
  parameterised as a masked dense matrix or as compact index/value tensors; a masked dense matrix is a control
  representation in the sense of `AGENTS.md` section 3, admissible because no runtime claim is drawn: where the third
  series times or measures (section 15), the FFN is masked dense in every arm alike. The model and its gradients are
  the same either way.

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
is a **hypothesis**. Each step gets its own README before its runs, which freezes what this section proposes. The
results are in 14.8, added after the series.

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

### 14.8 Results of the second series (2026-10-25/26)

**Provenance.** The same cloud session and orphan branch `sparse-attention-cloud` as section 13, under the brief
`experiments/sparse_attention/CLOUD_E6_E9.md`, from `bad5c440f` to `d2ab42ed2`. Records under
`experiments/sparse_attention/common/llm/training/{prep,e6,e7,e8,e9}/`, each with its README (committed before its
first run) and its `RESULTS.md`; summary `experiments/sparse_attention/SUMMARY_E6_E9.md`. The branch is **not
merged**. Nothing was promoted. Reviewed on 2026-10-26: the paired differences below were recomputed from the
`summary.json` of the 48 final runs of E8 and E9; the reserves of that review are in 14.8.4.

**What "compute" means here.** In this section, *compute* and *equal compute* mean inference multiply-adds per token
counted by `lib/count.py` (section 6): an **estimated proxy**, not a time. Counted operations do not convert into time
at a fixed rate: an HFFN path runs at a much lower operation rate than dense kernels and stays competitive because it
processes far fewer operations (the MoE HFFN measurements, recalled by the user on 2026-10-26). Two models equal in
counted multiply-adds, or in parameters, are therefore not equal in time, and no reading below orders the arms by
cost. No time was measured (decision 9). How the arms divide their work over the workers is in 14.8.6.

#### 14.8.1 The protocol as run

- **Preparation** (`prep/runs/run_20261025T093927Z`, exit 0): 44 new checks and the 179 earlier ones pass. A decay
  branch from a plateau checkpoint equals the uninterrupted WSD run bit for bit, and a cosine run of the first ladder
  reproduces its record bit for bit. The counter equals the inference table of 14.5 for the seven models at both
  contexts. `IND_L4` counted for the first time: 133,120 multiply-adds per token at `N = 128`, 231,424 at 512, 175,744
  parameters outside the embedding.
- **Cost:** about 19 hours of wall time, against 16 in 14.4 and 18.5 re-estimated after the smoke (under the stop at
  24 hours of the brief). The hash-routed arms run in the selected form and are not faster per step in this harness
  (not evidence of anything).
- **Harness events:** none (no check failure, no defect, no stopped run).
- **E6** (`e6/runs/run_20261025T095209Z`, exit 0): `REF2` at 0.025 against `E4a_L4` at 0.008, seeds 0 and 1.
- **E7** (`e7/runs/run_20261025T120721Z`, exit 0, seed 100, `H* = 4,000`): `REF1` 0.016, `REF4` 0.008, `E4a_L2`
  0.008, `E4a_L6` 0.008, each interior after one extension. Computed: `REF2` 0.0127, `REF3` 0.0101, `E4a_L4` and
  `IND_L4` 0.008.
- **E8** (`e8/runs/run_20261025T163405Z`) and **E9** (`e9/runs/run_20261025T220720Z`): exit 0, 24 runs each, every
  one `passed` at 4,000 steps, full validation and test splits, seeds 0-2. In E8, `E4a_L4` seeds 0 and 1 reproduce
  the `H = 4,000` branches of E6 to the last digit.

#### 14.8.2 Measured

E6, validation bits per byte at the end of each decay branch, gap `E4a_L4 − REF2`:

| Seed | `H = 2,000` | `H = 4,000` | `H = 8,000` | Deviation from 8,000 at 2,000 / 4,000 |
|---|---|---|---|---|
| 0 | −0.0204 | −0.0357 | −0.0498 | 0.0294 (fail) / 0.0142 (pass) |
| 1 | −0.0375 | −0.0553 | −0.0666 | 0.0291 (fail) / 0.0114 (pass) |

**`H* = 4,000`** under the frozen rule, and the series went on.

The frontiers (validation bits per byte, median over three seeds; compute, parameters and KV cache from the counter,
estimated). The two contexts are different campaigns and their figures are not compared:

| Arm | Inference MACs per token (128 / 512) | E8, `N = 128` | E9, `N = 512` | Parameters | KV values per token |
|---|---|---|---|---|---|
| `IND_L4` | 133,120 / 231,424 | 2.2391 | 2.2372 | 175,744 | 1,024 |
| `E4a_L2` | 138,240 / 187,392 | 2.2928 | 2.2889 | 161,664 | 512 |
| `REF1` | 157,696 / 256,000 | 2.3672 | 2.3498 | 93,184 | 256 |
| `E4a_L4` | 243,712 / 342,016 | 2.1161 | 2.1076 | 323,200 | 1,024 |
| `REF2` | 282,624 / 479,232 | 2.1475 | 2.1416 | 186,240 | 512 |
| `E4a_L6` | 349,184 / 496,640 | 2.0398 | 2.0330 | 484,736 | 1,536 |
| `REF3` | 407,552 / 702,464 | 2.0721 | 2.0139 | 279,296 | 768 |
| `REF4` | 532,480 / 925,696 | 2.0265 | 1.9601 | 372,352 | 1,024 |

The registered readings, `d_s = bpb(candidate) − bpb(control)` per seed (sorted), validation; the test split gives
the same signs and outcomes everywhere:

| Reading | E8, `N = 128` | E9, `N = 512` | Outcome (E8 / E9) |
|---|---|---|---|
| `E4a_L2` / `REF1` (12 % / 27 % fewer counted MACs) | −0.085, −0.070, −0.067 | −0.065, −0.061, −0.050 | dominates / dominates |
| `E4a_L4` / `REF2` (14 % / 29 %) | −0.041, −0.028, −0.019 | −0.050, −0.028, −0.019 | dominates / dominates |
| `E4a_L6` / `REF3` (14 % / 29 %) | −0.033, −0.033, −0.025 | +0.009, +0.009, +0.023 | dominates / within `τ`, not dominating (worse in every seed) |
| equal depth `E4a_L2` / `REF2` | +0.133, +0.145, +0.156 | +0.141, +0.145, +0.169 | beyond / beyond |
| equal depth `E4a_L4` / `REF4` | +0.089, +0.090, +0.099 | +0.131, +0.162, +0.166 | beyond / beyond |
| `IND_L4` / `E4a_L4`, equal depth (IND worse by) | +0.115, +0.123, +0.129 | +0.114, +0.130, +0.143 | beyond / beyond |

Estimated readings (the dense family interpolated in `log C` per seed, inside its range; median over seeds):
`E4a_L4` −0.085 and `E4a_L6` −0.065 at `N = 128`, −0.145 and −0.096 at 512; `E4a_L2` lies below `REF1` and has no
value. Equal counted training multiply-adds (b) give the same numbers, since every arm runs the same steps and bytes and
the training count is three times the forward. E6 gives the other reading (b) on its one pair, at unequal steps: −0.0385
and −0.0559 at 2,000 steps of `REF2`, −0.0495 and −0.0684 at 4,000 (seeds 0 and 1).

By position range at `N = 512` (E9, median, validation, descriptive): from positions 0-127 to 256-511, `REF3` and
`REF4` gain 0.127 and 0.130 bpb; every other arm 0.057 to 0.077 (`REF1` 0.062, `REF2` 0.077, `E4a_L2` 0.057,
`E4a_L4` 0.067, `E4a_L6` 0.073, `IND_L4` 0.068). The ranges 128-255 and 256-511 lie within 0.009 of each other in
every arm.

#### 14.8.3 Reading against the questions of 14.1

1. **Loss of quality at equal depth:** beyond `τ` in both pairs and both contexts, +0.09 to +0.17 bpb (measured).
2. **Counted multiply-adds and memory** (estimated): 12-14 % fewer inference multiply-adds per token at `N = 128` and
   27-29 % at 512 in the equal-compute pairs; what this is worth in time is not known (14.8.6). In those pairs the
   constrained model carries about 1.7 times the weights and AdamW states and twice the KV cache of the dense model
   it is paired with; at equal depth 0.87 times the weights and the same KV cache. Training activations per token
   (counter): about half of the dense in the equal-compute pairs.
3. **Quality at equal inference compute:** the constrained model is better than the dense HFFN `d20` model in all
   three pairs at `N = 128` and in two of three at 512, in every seed, with fewer counted multiply-adds; in the
   deepest pair at 512 it is equivalent within `τ` at 29 % fewer counted multiply-adds, while worse in every seed
   (measured). This is a
   statement of quality at equal counted operations, under the protocol of this series only (`M = 128`, enwik8,
   4,000 steps, three seeds, the rates of E7); it says nothing on which model runs faster.

That the advantage does not extend to the deepest pair at 512, while the constraint saves more there, is a reading
across the two campaigns, from the outcomes of their pairs only. The position ranges agree with it descriptively:
only the dense models of three and four layers gain much from the positions beyond 128. A cause is a hypothesis
(each token computes two of the eight heads, section 10); nothing isolated it.

#### 14.8.4 Reserves of the review (2026-10-26)

- **The coupling is not shown at equal counted multiply-adds.**
  - `RESULTS.md` of E8 and `SUMMARY_E6_E9.md` read `IND_L4` against `E4a_L4` as "the coupling is worth it at depth".
    That holds **at equal depth only**, where `IND_L4` counts 45 % (128) and 32 % (512) fewer multiply-adds.
  - Under the primary criterion of the series (decision 8), the relevant comparison is at equal counted
    multiply-adds. The series
    holds one near pair, outside the registered readings: at `N = 128`, `IND_L4` (133,120 multiply-adds, 175,744
    parameters) against `E4a_L2` (138,240, 161,664), close in compute and in parameters. `IND_L4` is better in every
    seed: −0.061, −0.049, −0.046 on validation, −0.058, −0.049, −0.045 on test (measured figures, an unregistered
    comparison: descriptive, no verdict).
  - At 512 `IND_L4` counts 23 % more multiply-adds than `E4a_L2`. The `E4a` family interpolated in `log C` at its
    count is about 0.012 bpb better than `IND_L4` (estimated, on the medians), within `τ`.
  - `IND_L4` took the rate of `E4a_L4` without a sweep, and it is one pair: a registered `IND` family over depth is
    needed to decide the question of quality. It decides nothing on cost (14.8.6).
- **Learning rates.**
  - E6 ran `REF2` at 0.025, about twice the rate computed after E7 (0.0127). `H*` was chosen on a pair whose dense arm
    was off its rate (the limit was declared in 14.4).
  - In the E7 sweeps a factor of two away from the selected rate costs 0.003 to 0.023 bpb (one seed, estimated).
    That is the order of the smallest dominance margin, 0.019 (`E4a_L4`/`REF2` at both contexts), whose control runs at
    a computed rate.
  - Only `E4a_L2`/`REF1` has both rates swept; the middle depths of both families are computed.
- **Horizon.** `H*` was calibrated on one pair at `N = 128` only. In that pair the gap still moved by 0.011-0.014
  towards the constrained model between 4,000 and 8,000 steps. Its direction for the other pairs, for the
  equal-depth losses and at 512 is not measured.
- **Control at equal counted multiply-adds.** The dense family varies in depth only. The result is "the constrained
  model of depth `2L` beats the dense model of depth `L`". A dense model that spends the same counted multiply-adds
  otherwise (narrower, or an unrouted FFN of a quarter of the neurons, question 4, second part) was not run.
- **Minor.** `SUMMARY_E6_E9.md` gives the gain of the constrained arms between position ranges as 0.067-0.073; the
  E9 figures give 0.057-0.073 (`E4a_L2` 0.057). The record is not edited; the figures above are those of
  `e9/RESULTS.md`.

#### 14.8.5 Status of the questions after the series

| Question | Status |
|---|---|
| 14.1, 1 to 3 | answered under the protocol of the series (14.8.3), with the reserves of 14.8.4 |
| 13.5, 2 (mixing at depth) | beyond `τ` in favour of the coupling at equal depth (measured); open at equal counted multiply-adds, where the one near pair favours `IND_L4` (descriptive) |
| 13.5, 3 (horizon, rates) | not answered for the content-routed arms, which the series did not run; for the hash route it adds a horizon calibrated on one pair at `N = 128` and rates swept at the extreme depths only (14.8.4) |
| 13.5, 4, first part | answered (14.8.3, point 3) |
| 13.5, 4, second part | not run |
| 13.5, 1 and 5 | wait for the routing study |
| New: coupling at equal counted multiply-adds | does `E4a` beat the separated models (`IND`) in quality at equal counted multiply-adds, over depth? Open; it needs a registered `IND` family over depth (14.8.4) |
| Time | not measured; needs a native implementation and a timed protocol (14.8.6) |

#### 14.8.6 `IND` against `E4a`: how each divides the work over the workers

The purpose of the constraint is to move the dense work of attention, which the current runtime does not distribute over
the HFFN workers (user, 2026-10-26), onto the `G` workers, each with its HFFN group (section 2.2: worker `g` owns the
heads `H_g`, the block `S_g`, the neurons `R_g` and the tokens routed to `g`). The two arms are read here by what they
give each worker to compute and to exchange. The division follows from the definitions of sections 2 to 5 (by
construction); its cost in time is **not measured**, and nothing below orders the arms by speed.

| Per layer | `E4a` | `IND` |
|---|---|---|
| Keys and values | worker `g` computes those of its heads `H_g` for **every** token, reading the full normalised residual: `1/G` of the dense work, the same share for every worker | worker `g` computes those of `H_g` for every token, reading `S_g` only |
| Queries, scores, `W_O^g`, FFN group `g` | for the tokens routed to `g` (about `N / G` under the balanced hash); queries read the full residual | for the tokens routed to `g`; queries read `S_g` only |
| Writes | attention into `S_g` of its tokens; the FFN over all `M` coordinates of its tokens. Each token is processed by one worker, so no two workers write the same token: no reduction | attention and FFN into `S_g` of its tokens only |
| Exchange between workers | every worker reads the residual written by the others: one synchronisation per layer | none through depth, except the RMS scalar of each token (section 5); the blocks meet at the output |

Reading (hypothesis, no time measured):

- Both arms split the dense keys and values over the workers, which is the transfer the constraint aims at; `E4a`
  keeps a dense read of the residual by every worker and one synchronisation per layer, `IND` removes both and reads
  a quarter of the residual.
- `IND` against `E4a` is therefore a comparison between workers that exchange their residual at every layer and
  workers that do not. The series measures what the exchange is worth in **quality**: beyond `τ` at equal depth
  (measured); open at equal counted multiply-adds (14.8.4). What it costs in **time** (the synchronisation, the reads
  of the whole residual, the rate each part reaches on the hardware) is set by the implementation and the machine,
  not by the counted multiply-adds.
- A time claim for either arm, against the other or against dense attention, needs a native implementation and a
  timed protocol under `AGENTS.md` sections 6 and 7 (section 0 of this plan).

## 15. Third series: memory and relative time of the implementation, and width (E10-E13, written 2026-10-26)

Written after the discussion of 2026-10-26 with the user, before any run of the series. Every number in this section
is **estimated** (a calculation, or a cost extrapolated from the wall times of section 14.8) and every expected effect
is a **hypothesis**. Each step gets its own README before its runs, which freezes what this section proposes.

### 15.1 Why a third series

The second series shows that, at equal counted inference multiply-adds, the constrained model is better in quality
than the dense-attention HFFN `d20` model in five pairs of six (14.8.3, measured). It leaves three questions, raised
by the user on 2026-10-26:

1. **Memory.** In the pairs at equal counted multiply-adds the constrained model carries about 1.7 times the weights
   and AdamW states and twice the KV cache of the dense model (14.5, counted). The cause is the pairing, not the
   attention: at equal counted multiply-adds the constrained model is twice as deep. Per layer, only `W_O` becomes
   block-diagonal; `W_Q` is stored whole although each token uses `H_g` of it, `W_K` and `W_V` stay dense (decision 3),
   and every neuron of the FFN is stored although each token uses `R_g` (the storage pattern of a mixture of experts:
   compute per token falls, stored weights do not). The training activations per token are counted at about half of
   the dense model's. **None of this is measured**, and the harness cannot measure it: the hash-routed arms run in
   the `selected` form, which computes every head and every neuron and masks them, so its memory and its time say
   nothing of the method.
2. **Time.** Counted multiply-adds are not time (14.8). The constraint is meant to spread the dense work of attention
   over the `G` workers with their HFFN groups (14.8.6). Whether a model divided that way over four workers runs
   faster or slower than dense attention under the same executor, and what the exchange of the residual at every
   layer costs (`E4a` against `IND`), is not known.
3. **Scale.** Every run so far is at `M = 128`, 8 heads, 2 heads per group. Two levers of scale can be studied at
   `G = 4`: the width `M`, and the number of heads each group owns (each token computes `n_heads / G` heads). The third
   lever, `G` itself (the number of workers), is fixed by decision 2 and is not in this series.

The series answers, under its protocol only:

- **E10**: does a packed implementation reach the counted saving of training activations, and how far from it is
  the `selected` form of the harness?
- **E11**: the ratio of time, in one process, between the constrained model divided over four workers and dense
  attention under the same executor, at the level of one layer and of the model; and between `IND_L4` and `E4a_L4`.
- **E12-E13**: does the result at equal counted multiply-adds hold at twice the width, and does the gap to dense move
  when each group owns four heads instead of two?

### 15.2 Decisions of the user recorded here (2026-10-26)

14. **Amendment of decision 9, prospective.** On the cloud session the **relative** time of two models is admitted:
    both measured in the same process, one after the other, under the timing rule of `AGENTS.md` section 7 (arms in
    ABBA blocks, each warmed up before its first timed call, a copy arm that sets the noise band `delta`, a gain or a
    loss stated only beyond `delta`, several fresh processes per cell). An absolute time is never a result; a ratio
    is not compared across machines, campaigns or with the figures of the HFFN runtime, and holds for the executor
    and the machine it was read on. Measuring the memory of the implementation (E10) belongs to the same amendment;
    the user's words of 2026-10-26 name the time only, so the memory part is proposed by this section and stands when
    the user approves the series. The verdicts of sections 13 and 14 are unchanged and no run of E6-E9 is re-read.
    Decision 9 otherwise stands: absolute performance waits for a native implementation.
15. **Specialising the HFFN groups** (an FFN whose groups are specialised by content, beside the attention) is not
    the subject for now. The FFN stays HFFN `d20` as in the second series.

Unchanged: routing frozen to the balanced byte hash (decision 7), `G = 4` (decision 2), `K` and `V` dense (decision 3),
short trainings (decision 13) with `H* = 4,000` (E6), WSD, 8,192 bytes per step, `τ = 0.075` bpb, the three outcomes of
13.1 and *dominates* of 14.3. Decision 10 (depth as the axis of compute at `M = 128`) governed the second series; this
series adds one width, `M = 256`, read in its own campaign, inside which depth stays the axis of compute.

### 15.3 Fixed design of the series

- **Packed form for the hash route** (preparation). Today the `packed` form exists only for the position route
  (`lib/model.py`). Under the hash route, per layer: worker `g` computes the keys and values of its heads `H_g` for
  every token; then, for the tokens routed to `g` only (gathered once per forward, since the route of a byte is the
  same at every layer), the queries of `H_g`, the scores against the keys of every earlier token (causal mask on the
  original positions), the outputs, `W_O^g` and FFN group `g`, scattered back to their rows. It must equal the
  `selected` form within float32 rounding, in outputs and in the gradients of every parameter (check, tolerance in
  the README of the preparation). `selected` stays the default, so E6-E9 stay reproducible.
- **The FFN is a masked dense matrix in every arm** (section 0, a control representation), the same in the
  constrained and the dense models. No figure of this series says anything of the HFFN runtime.
- **E10, what is measured.** Per arm, in one training step of 8,192 bytes of the train split (real bytes: the route
  depends on them), float32:
  - (a) the bytes of the tensors saved for the backward, counted with `torch.autograd.graph.saved_tensors_hooks`
    (distinct storages), split into token-side tensors (activations) and weight-side tensors (the masked weights, a
    cost of the control representation, reported apart);
  - (b) the peak of live tensor bytes over the forward and backward, from the memory events of `torch.profiler`;
  - (c) the peak of live tensor bytes of one forward without gradient (prefill of the same bytes);
  - (d) the peak resident memory of the process, descriptive only (allocator caching, masked weights).

  The methods of (a) and (b) are checked on a synthetic module whose saved and peak bytes are known. Weights, AdamW
  states and the KV cache stay **counted** (by construction; the harness stores dense masked weights, so their
  measured size would be the dense one).
- **E11, the executor.** `G = 4` workers, one per vCPU, one intra-op thread each, forward only, without gradient.
  - Constrained arms: the packed form, the work of group `g` on worker `g`; `E4a` waits at a barrier at the end of
    every layer (every worker reads the residual written by the others); `IND` keeps each worker on its block `S_g`
    and meets at the output (14.8.6).
  - The dense control runs **under the same executor**, as `AGENTS.md` section 3 requires of a scheduling added on
    one arm: worker `g` computes the queries, keys, values, scores and outputs of the heads `H_g` over every token;
    barrier; then `W_O` and the FFN on a block of `N / G` tokens per worker. A second dense execution, PyTorch's own
    intra-op parallelism on four threads, is timed in the smoke; the faster of the two in the smoke is the control of
    E11, chosen and written in its README before the run.
  - The method of parallel execution (TorchScript `fork`/`wait` on the inter-op pool, or a small C++ extension with
    `at::parallel_for` over the groups) is chosen in the preparation and stated in its README; it is the same for
    every arm. A **concurrency gate** checks it: four equal independent tasks of the size of the largest layer cell
    take at most 0.4 times their sequential time (median of five repetitions). If the gate fails, E11 runs no timing
    and is reported as not run, with the cause; the series goes on.
  - Logits and embedding are computed by the same code in both arms of a cell.
- **E11, the timing rule.** One cell = one pair of arms. Five fresh processes per cell, nothing else running on the
  machine. In each process every arm (control, copy, variant) is built, then warmed up (at least 30 calls and 5
  seconds), then timed in 12 windows; a window is one block `control, copy, variant, variant, copy, control`, each
  slot repeated so that it lasts at least 0.2 second. Per window, the ratio `variant / control` and `copy / control`
  on the sums of their two slots; per process, the median over windows; per cell, the median of the five processes
  and [min–max]. `delta` = the largest `|copy / control − 1|` over the cell-processes of the run. Steal time and user
  time (`/proc/stat`) are recorded per window, the load average at the start of each process, `lscpu` and the
  `torch` configuration once; nothing is excluded after the fact.
- **Width** (E12-E13): `M = 256`, `H = 1,024`, `k_up = k_down = 52` (the density `d20`: 26 of 128, 52 of 256), `G = 4`,
  `N = 512` only, tied embedding of 256 bytes, the training protocol of 14.3. Two head layouts: 8 heads of 32 (2 per
  group, the layout of the second series) and 16 heads of 16 (4 per group). The two layouts have the same counted
  multiply-adds, parameters and KV cache (15.5).

### 15.4 The steps

| Step | What it runs | Why | Seeds | Cost (estimated) |
|---|---|---|---|---|
| **Preparation** | packed hash form and its parity; executor and concurrency gate; memory and timing harnesses with their checks; arms of width 256 and their counts; smoke of every arm and of every E11 cell | makes the series possible; no figure | — | 2-4 h of work, < 1 h of compute |
| **E10** | memory (a)-(d) of the eight arms of E8 at `N = 128` and of E9 at `N = 512` (hash arms packed), `E4a_L4` also in the `selected` form at both contexts, and the six arms of E13 | is the counted saving of activations reached; how much of the harness memory is the `selected` form | — | < 0.5 h |
| **E11** | relative time, 8 cells (below) | what the division over four workers is worth in time against dense attention under the same executor, and what the exchange of `E4a` costs against `IND` | — | ≈ 1 h |
| **E12** | learning rates at width: `REF2_W256` and `E4a_L4_W256`, 3 rates (extended at most twice at an edge) | one fair rate per family at the new width | 100 | ≈ 3.5-6 h |
| **E13** | width frontier at `N = 512`: `REF1_W256`, `REF2_W256`, `E4a_L2_W256`, `E4a_L4_W256`, `REF2_W256_h16`, `E4a_L4_W256_h16` | equal counted multiply-adds at twice the width; four heads per group against two | 0, 1, 2 | ≈ 9 h |

Costs extrapolate the wall times of the second series (about 19 hours for 16 planned, so about 6.6 minutes per layer
for 4,000 steps at `N = 512`), times 2.8 at `M = 256` (the counted multiply-adds of one layer; time need not follow),
times about 0.5 for the hash-routed arms in packed form (hypothesis, measured by the smoke). Total **about 17 to 20
hours**; about 23 if the packed form saves nothing in training. After the smoke the total is re-estimated from the
measured wall times; above 24 hours, the session stops after E11 and the user decides on E12-E13.

**E10, memory.**

- *Readings.* Per arm, (a) and (b) per token against the training activations of the counter, and (c) per token; the
  ratios constrained / dense in the pairs at equal counted multiply-adds (`E4a_L2`/`REF1`, `E4a_L4`/`REF2`,
  `E4a_L6`/`REF3`, and the two pairs of width) and at equal depth (`E4a_L2`/`REF2`, `E4a_L4`/`REF4`), measured beside
  counted. `selected` against packed for `E4a_L4` shows how much of the memory of the harness is its form.
- *Rule.* The quantities are deterministic: each is measured in two fresh processes and must be identical for (a) and
  within 1 % for (b) and (c); otherwise it is a harness failure. No threshold on the method: the step measures.

**E11, relative time.** Cells (variant / control), all at 8,192 tokens per call:

| Cell | Level | Variant / control | `M` | `N` | Why |
|---|---|---|---|---|---|
| a1 | one layer | `E4a` / dense | 128 | 128 | the layer of the series |
| a2 | one layer | `E4a` / dense | 128 | 512 | the context where the constraint saves more |
| a3 | one layer | `E4a` / dense | 512 | 512 | width: products large against the cost of a call (hypothesis) |
| a4 | one layer | `E4a` / dense | 512 | 2,048 | the scores dominate the counted work |
| b1 | model | `E4a_L4` / `REF2` | 128 | 128 | the middle pair at equal counted multiply-adds |
| b2 | model | `E4a_L4_N512` / `REF2_N512` | 128 | 512 | the same at 512 |
| c1 | model | `IND_L4` / `E4a_L4` | 128 | 128 | the cost of the exchange of the residual at every layer |
| c2 | model | `IND_L4_N512` / `E4a_L4_N512` | 128 | 512 | the same at 512 |

At `M = 512` a layer has 8 heads of 64 (2 per group), `H = 2,048`, `k = 104`; weights random (time does not depend on
their values; the route does depend on the bytes, so the inputs are bytes of the valid split).

- *Readings.* Per cell, `r = t(variant) / t(control)`, median and [min–max] over the five processes, with `delta` of
  the run: faster if `1 − r > delta`, slower if `r − 1 > delta`, otherwise not resolved. Beside it, the counted ratio of
  multiply-adds of the cell (estimated) and the quotient of the two, descriptive: how far time follows the count on
  this executor. A separate pass, not timed, splits the time of the variant into the phase of keys and values, the
  routed phase and the wait at the barrier (descriptive).
- *What it is not.* A ratio of this PyTorch executor on this shared machine, forward only. A native implementation
  may move it either way. It orders nothing on another machine and says nothing of training time.

**E12, the rates at width.** As E7, at `H*`, WSD, `N = 512`, seed 100, one plain run per arm and rate.

- Centres (the rates of E7 halved for the doubled width, rounded to a power of two; a hypothesis): `REF2_W256`
  0.008, `E4a_L4_W256` 0.004. Grid: the centre, half and double; at an edge, the next rate, at most twice; still at an
  edge, keep it and state it as a limit.
- Rates of the other arms, computed and written in the README of E13 before its runs: `REF1_W256` the rate of
  `REF2_W256` times 1.26 (the ratio `REF1`/`REF2` of E7-E8, 0.016 / 0.0127); `E4a_L2_W256` the rate of `E4a_L4_W256`
  (E7 found one rate for the `E4a` family at every depth); the two `_h16` arms the rate of their 8-head counterpart.

**E13, the frontier at width.** Six arms, seeds 0, 1, 2, `--test` on every run (read once, selects nothing), bits
per byte by position range. Readings, paired per seed, `d_s = bpb(candidate) − bpb(control)`:

| Reading | Candidate / control | Counted multiply-adds saved | Class |
|---|---|---|---|
| Equal counted multiply-adds, width | `E4a_L2_W256`/`REF1_W256`, `E4a_L4_W256`/`REF2_W256` | 21 %, 22 % | measured |
| Equal depth, width | `E4a_L2_W256`/`REF2_W256` | — | measured |
| Equal counted multiply-adds, 4 heads per group | `E4a_L4_W256_h16`/`REF2_W256_h16` | 22 % | measured |
| Heads per group | `i_s = d_s(h16 pair) − d_s(h8 pair)`, the pairs `E4a_L4`/`REF2` at width | — | measured |

- The first three use the outcomes of 13.1 and *dominates* (14.3).
- The heads-per-group reading compares the gap to dense under each layout, so that the dense arm absorbs what the
  layout does to any model (smaller heads), and only its effect on the constraint is read. It is stated *favours four
  heads per group* if `max_s i_s < 0`, *favours two* if `min_s i_s > 0`, otherwise not resolved, and `|i_s|` against
  `τ` beside it. Three seeds: a direction, not a size.
- E13 is not compared with E9 figure by figure (different campaigns); whether the outcomes of the pairs agree with
  those of E9 is a cross-campaign reading, stated as such.

### 15.5 What is counted (estimated)

Arms of width 256 at `N = 512` (`lib/count.py` with the parameters of each arm; logits `V M` = 65,536):

| Arm | Inference MACs per token | Parameters outside the embedding | KV values per token | Training activations per token |
|---|---|---|---|---|
| `E4a_L2_W256` | 552,960 | 642,816 | 1,024 | 3,328 |
| `REF1_W256` | 696,320 | 370,688 | 512 | 6,656 |
| `E4a_L4_W256` | 1,040,384 | 1,285,376 | 2,048 | 6,656 |
| `E4a_L4_W256_h16` | 1,040,384 | 1,285,376 | 2,048 | 10,752 |
| `REF2_W256` | 1,327,104 | 741,120 | 1,024 | 13,312 |
| `REF2_W256_h16` | 1,327,104 | 741,120 | 1,024 | 21,504 |

The saving of counted multiply-adds is smaller than at `M = 128` and the same context (21-22 % against 27-29 %): the
dense keys and values grow as `M²` and take a larger share of the layer. The memory paragraph of 14.5 holds at
width: in the pairs at equal counted multiply-adds the constrained model carries 1.7 times the weights and twice the
KV cache. The preparation checks these figures against the counter of the branch.

### 15.6 What the series will not show

- No absolute time, no time on another machine, no training time (the backward is not divided by group in this
  executor), no decoding with a KV cache (one token per step leaves three workers without queries; it needs batched
  decoding and is not run).
- No native kernel and no statement on the HFFN runtime: the FFN is masked dense in every arm.
- No measured memory of weights, AdamW states or KV cache (counted, 15.3).
- No other routing (decision 7), no other `G` (decision 2), no specialised FFN groups (decision 15), no other dataset.
- One width and one context for the width; `H*` is not recalibrated at width, and only two arms are swept.
- Left open from 14.8.5, not in this series: the registered `IND` family over depth (coupling at equal counted
  multiply-adds), a dense control spending the same counted multiply-adds otherwise (narrower, or an unrouted FFN of
  a quarter of the neurons), and the deepest pair at 512. With `G > 4`, they are proposals for the user.

### 15.7 Launching

- On a Claude cloud session, on the branch `sparse-attention-cloud`, under the brief
  `experiments/sparse_attention/CLOUD_E10_E13.md`, chained: preparation, E10, E11, then the stop if the re-estimate
  exceeds 24 hours, then E12, E13, and a summary.
- E11 runs alone on the machine: no training, check or build of the session runs while it times.
- Each step has its own folder `experiments/sparse_attention/common/llm/training/<step>/`, a README committed before
  its first run, its own run records and its own `RESULTS.md`. A harness failure is not an attempt.
- Nothing is promoted: the series measures, it retains no model.

### 15.8 Results of the third series (2026-10-26/27)

**Provenance.** The same cloud session and orphan branch `sparse-attention-cloud` as sections 13 and 14.8, under the
brief `experiments/sparse_attention/CLOUD_E10_E13.md`, from `a08b25968` to `440da1e17`. Records under
`experiments/sparse_attention/common/llm/training/{prep3,e10,e11,e12,e13}/`, each with its README (committed before its
first run) and its `RESULTS.md`; summary `experiments/sparse_attention/SUMMARY_E10_E13.md`. The branch is **not
merged**. Nothing was promoted. Read on 2026-10-27 from the `RESULTS.md` of each step and the `raw.jsonl` of E11; the
paired differences of E13 were recomputed from its `summary.json`. The reading of 15.8.3 and the reserves of 15.8.4 are
those of this review, not of the cloud summary.

**What "time" and "memory" mean here.** Under decision 14, a time is the ratio of two models measured in the same
process on the cloud machine (a shared virtual machine of 4 vCPU, torch 2.14.1 for CPU), forward only, by a PyTorch
executor (TorchScript fork and join, four workers, one intra-op thread each), with a masked dense FFN in every arm. A
memory is the bytes per token that this PyTorch harness keeps (float32, one thread, one step of 8,192 bytes). Both hold
for that harness and that machine only. Neither says anything of the HFFN runtime or of a native implementation of
the constraint, and no ratio below is compared with another machine, campaign or runtime figure.

#### 15.8.1 The protocol as run

- **Preparation** (`prep3/runs/run_20261026T205721Z`, exit 0):
  - checks 203/203, 44/44 and 34/34 pass;
  - the packed hash form (`--impl gather`) equals the `selected` form: logits bit for bit at width 128, within 2e-6
    at width 256, gradients within 6e-8;
  - the counter equals the table of 15.5;
  - the executor equals the sequential forward;
  - the concurrency gate passes (median ratio 0.310 ≤ 0.4).
  - The dense control of E11 is the executor itself (README addendum before the run).
- **E10** (`e10/runs/run_20261026T211323Z`, exit 0): 24 configurations, two fresh processes each. The rule holds
  everywhere: (a) is identical in both processes, (b) and (c) are within 1 %.
- **E11** (`e11/runs/run_20261026T214914Z`, exit 0): 8 cells, 5 fresh processes per cell, 12 ABBA windows each,
  nothing else running. The concurrency gate at the start passes (0.324). Steal time is about 1 %.
- **Harness event.** A first launch of E11 (`e11/runs/run_20261026T212801Z`, `EXIT 137`, `ABORTED.md`) stopped on a
  restart of the container after cells a1-a4 and b1. It gives no figure. The run above is the full relaunch under the
  same README.
- **E12** (`e12/runs/run_20261026T222859Z`, exit 0): 7 runs of 4,000 steps, seed 100, `M = 256`, `N = 512`; one
  extension (`REF2_W256` at 0.002).
- **E13** (`e13/runs/run_20261027T040908Z`, exit 0): 18 runs `passed` at 4,000 steps, seeds 0-2, full validation and
  test splits of 9,765 windows of 512 bytes. Smoke: `e13/runs/smoke_20261027T040214Z`.
- **Cost:** about 15 hours of wall time (preparation 0.2, E10 0.1, E11 0.5, E12 5.5, E13 8.8), against 17-20 in 15.4 and
  12.6 re-estimated after the smoke. The stop at 24 hours was not reached.

#### 15.8.2 Measured

**E10, memory per token of this harness**, ratio constrained / dense (bytes in `e10/RESULTS.md`). The columns
are (a) the tensors saved for the backward, token side, (b) the peak of a training step and (c) the peak of a prefill.
Counted figures (estimated) are beside them:

| Reading | (a) | (b) | (c) | Counted activations | Counted MACs |
|---|---|---|---|---|---|
| Equal counted MACs, `N = 128` (`E4a_L2`/`REF1`, `E4a_L4`/`REF2`, `E4a_L6`/`REF3`) | 1.61, 1.68, 1.71 | 1.53, 1.62, 1.67 | 0.79 | 0.50 | 0.88, 0.86, 0.86 |
| Equal counted MACs, `N = 512` | 1.79, 1.88, 1.92 | 1.68, 1.81, 1.86 | 0.78 | 0.50 | 0.73, 0.71, 0.71 |
| Equal counted MACs, `M = 256` (`L2`/`REF1`, `L4`/`REF2`, `h16` pair) | 1.59, 1.65, 1.64 | 1.55, 1.62, 1.62 | 0.78 | 0.50 | 0.79, 0.78, 0.78 |
| Equal depth, `N = 128` (`E4a_L2`/`REF2`, `E4a_L4`/`REF4`) | 0.90, 0.89 | 0.91, 0.90 | 0.79 | 0.25 | 0.49, 0.46 |
| Equal depth, `N = 512` | 1.00, 1.00 | 1.00, 1.00 | 0.78 | 0.25 | 0.39, 0.37 |
| `IND_L4`/`E4a_L4`, `N` 128 / 512 | 0.93 / 0.95 | 0.94 / 0.95 | 1.05 / 1.08 | 1.00 | 0.55 / 0.68 |
| Packed / `selected`, `E4a_L4`, `N` 128 / 512 | 0.74 / 0.83 | 0.75 / 0.84 | 0.57 / 0.56 | 1.00 | 1.00 |

- The dense arms keep the same token-side bytes at `N = 128` and `N = 512` (`REF1`: 12,348 B per token at both),
  while the counted scores term grows with `N`.
- Weights, AdamW states and KV cache were not measured. They stay counted at 1.73-1.74 times the weights and states and
  twice the KV cache in the pairs at equal counted multiply-adds.

**E11, relative time** `r` = time(variant) / time(control), median over 5 processes of the per-process median over 12
windows, [min–max]. The noise band is `delta` = 0.0975, set by cell a1, whose copy ratios are 1.063, 0.902, 0.950,
1.070 and 0.933. The other cells keep their copy ratios within 0.05 of 1.

| Cell | Variant / control | `r` | Outcome | Counted ratio | `r` / counted |
|---|---|---|---|---|---|
| a1 | one `E4a` layer / one dense layer, `M` 128, `N` 128 | 0.601 [0.559–0.669] | faster | 0.422 | 1.42 |
| a2 | one layer, `M` 128, `N` 512 | 0.541 [0.537–0.565] | faster | 0.346 | 1.56 |
| a3 | one layer, `M` 512, `N` 512 | 0.405 [0.394–0.446] | faster | 0.422 | 0.96 |
| a4 | one layer, `M` 512, `N` 2,048 | 0.496 [0.462–0.508] | faster | 0.346 | 1.43 |
| b1 | model `E4a_L4` / `REF2`, `N` 128 | 1.290 [1.239–1.325] | slower | 0.862 | 1.50 |
| b2 | model `E4a_L4` / `REF2`, `N` 512 | 1.134 [1.091–1.228] | slower | 0.714 | 1.59 |
| c1 | model `IND_L4` / `E4a_L4`, `N` 128 | 0.852 [0.825–0.895] | faster | 0.546 | 1.56 |
| c2 | model `IND_L4` / `E4a_L4`, `N` 512 | 0.916 [0.871–0.954] | not resolved | 0.677 | 1.35 |

The quotient `r` / counted is descriptive. The untimed phase pass (single-threaded, first process of each cell,
descriptive) puts the plan of the routed tokens at 0.7-8.6 % of the fork-join, and the barrier plus launch at 6-30 %.

**E12, rates at width** (seed 100, final validation bpb):

| Arm | 0.002 | 0.004 | 0.008 | 0.016 | Selected |
|---|---|---|---|---|---|
| `REF2_W256` (centre 0.008) | 2.0118 (extension) | **1.9855** | 1.9959 | 2.0545 | 0.004 |
| `E4a_L4_W256` (centre 0.004) | 1.9869 | **1.9862** | 2.1101 | — | 0.004 |

- Both selected rates are interior; no arm ends at an edge.
- `E4a_L4_W256` is flat between 0.002 and 0.004, and 0.124 higher at 0.008.
- The other rates are computed under the rule of 15.4: `REF1_W256` 0.00504; `E4a_L2_W256` and the two `_h16` arms 0.004.

**E13, the frontier at width** (`M = 256`, `N = 512`; median [min–max] over three seeds; counts estimated):

| Arm | Valid bpb | Test bpb (beside) | Inference MACs per token | Parameters outside the embedding |
|---|---|---|---|---|
| `E4a_L2_W256` | 2.1925 [2.1721–2.2034] | 2.1643 | 552,960 | 642,816 |
| `REF1_W256` | 2.1821 [2.1815–2.1834] | 2.1605 | 696,320 | 370,688 |
| `E4a_L4_W256` | 1.9867 [1.9839–2.0045] | 1.9638 | 1,040,384 | 1,285,376 |
| `E4a_L4_W256_h16` | 1.9435 [1.9422–1.9487] | 1.9247 | 1,040,384 | 1,285,376 |
| `REF2_W256` | 1.9880 [1.9853–1.9971] | 1.9664 | 1,327,104 | 741,120 |
| `REF2_W256_h16` | 1.9855 [1.9815–1.9857] | 1.9658 | 1,327,104 | 741,120 |

The registered readings, `d_s` per seed 0, 1, 2 (validation):

| Reading | Counted MACs saved | `d_s` | Outcome |
|---|---|---|---|
| `E4a_L2_W256` / `REF1_W256` | 20.6 % | −0.0094, +0.0091, +0.0213 | within `τ`, not dominating |
| `E4a_L4_W256` / `REF2_W256` | 21.6 % | −0.0133, +0.0014, +0.0165 | within `τ`, not dominating |
| equal depth `E4a_L2_W256` / `REF2_W256` | 58.3 % | +0.1750, +0.2072, +0.2153 | beyond `τ` |
| 4 heads per group `E4a_L4_W256_h16` / `REF2_W256_h16` | 21.6 % | −0.0420, −0.0328, −0.0435 | within `τ`, dominates |
| `i_s` = `d_s`(h16 pair) − `d_s`(h8 pair) | — | −0.0287, −0.0342, −0.0599 | favours four heads per group; `|i_s|/τ` 0.38, 0.46, 0.80 |

- The test split gives the same outcomes. Its medians are +0.0022, −0.0026 and −0.0424 for the three pairs at equal
  counted multiply-adds.
- By position range (median, validation, descriptive): in every arm the ranges 128-255 and 256-511 lie within 0.002
  of each other, and 0-127 is 0.06-0.08 higher.

#### 15.8.3 Reading against the questions of 15.1

1. **Memory: the counted saving of training activations is not reached by this harness** (measured, E10).
   - At equal counted multiply-adds the constrained model keeps 1.6 to 1.9 times the dense model's saved tensors,
     where the counter says 0.5. At equal depth it keeps 0.89-1.00, where the counter says 0.25.
   - The prefill peak is lower, 0.78-0.79, at every depth and context (measured).
   - The measurements refute the four-term activation model of the counter (queries, scores, attention outputs, FFN
     activations) as a prediction of the memory of an implementation. They do not refute the constraint, of which no
     native implementation exists. The advantage of training activations stated in 14.5 and 15.1 is now supported by
     no measurement.
   - Why (hypothesis, read from the numbers):
     - the dense keys and values of every head, the norm and residual rows, the gathered rows and the logits are saved
       but are not in the four terms;
     - the CPU attention kernel stores no scores (the dense bytes do not move with `N`);
     - the packed form pads each group to the largest count.
   - About a quarter of the memory of the `selected` form of the first two series was its form (packed / `selected`
     0.74-0.83, measured); the rest remains.
2. **Time: one layer is faster, the model at equal counted multiply-adds is slower** (measured, E11, this executor and
   machine).
   - One constrained layer runs in 0.41-0.60 of the time of one dense layer, beyond `delta` in all four cells.
   - The model `E4a_L4` runs in 1.29 (`N = 128`) and 1.13 (`N = 512`) of the time of `REF2`, beyond `delta`, although
     the count says 0.86 and 0.71.
   - **This slowdown follows to first order from the layer cells and the depth** (estimated, within the same run). At
     equal counted multiply-adds the constrained model has twice the layers of the dense one, so its time ratio is
     about `2 r_layer` when the shared costs (embedding, logits, route) are small:
     - 2 × 0.601 = 1.20 against b1 1.29;
     - 2 × 0.541 = 1.08 against b2 1.13.
     The shared costs would pull the model ratio towards 1, so the residual of about 0.09 (`N = 128`) and 0.05 (512)
     is what remains to explain beyond the layer cells. The hypothesis of the E11 summary (less work per layer to hide
     the plan, the gathers and the padding behind) does not explain the bulk of the gap, since the layer cells already
     pay those per layer. At most it bears on that residual (hypothesis, not isolated).
   - A deeper constrained model breaks even in time only where `r_layer < 0.5`. In this run that holds only at
     `M = 512`: a3 at 0.405; a4 at 0.496, at the bound (estimated). There is no cell at `M = 256`, so the time of the
     pairs of E13 is not measured.
   - At the layer level the quotient `r` / counted is 1.4-1.6 at `M = 128` and 0.96 at a3: the constrained layer reaches
     its count only when the layer is wide (descriptive).
   - `IND_L4` against `E4a_L4`: faster at `N = 128` (0.85, beyond `delta`) and not resolved at 512 (0.92). The cheaper
     keys, values and queries of `IND` and the absence of the exchange vary together, and the executor keeps a barrier
     per layer for `IND`, so neither effect is attributed.
3. **Scale: at twice the width the constrained model stays within `τ` of dense at equal counted multiply-adds**
   (measured, E13).
   - With two heads per group (8 heads of 32) both pairs are within `τ` and neither dominates; each has one seed
     negative and two positive.
   - With four heads per group (16 heads of 16) the constrained model dominates in all three seeds, at 21.6 % fewer
     counted multiply-adds.
   - At equal depth it stays beyond `τ` (+0.18 to +0.22).
   - **Cross-campaign** (outcomes only, against E9):
     - equal depth is beyond in both campaigns;
     - the two pairs at equal counted multiply-adds with two heads per group go from *dominates* at `M = 128` to
       *within, not dominating* at 256.
     - Width, rates (0.004 against 0.008-0.016) and a horizon not recalibrated vary together between the campaigns,
       so the change is not attributed to the width.
   - The non-dominance at 8 heads rests largely on one seed per arm: `E4a_L4_W256` spans 0.0206 (seed 2 at 2.0045),
     `E4a_L2_W256` 0.0313, against 0.0019 for `REF1_W256`. The outcome *within* holds with a margin (largest `d_s`
     +0.0213 against `τ` = 0.075); the absence of dominance does not have that margin.
4. **Heads per group: the reading favours four heads per group** (measured, three seeds: a direction, not a size).
   - Descriptive: the change of layout moves the constrained arm by −0.043 (median, 1.9867 → 1.9435) and the
     dense arm by −0.003 (1.9880 → 1.9855).
   - Per seed, from the `summary.json`: dense −0.0116, −0.0038, −0.0024; constrained −0.0403, −0.0380,
     −0.0623.
   - At fixed `M` and `G` the layout changes two things together: the heads each group owns (2 → 4) and the size of a
     head (32 → 16). The dense arm absorbs what smaller heads do to any model, as 15.4 intended. What `i_s` cannot
     separate is whether the constrained model gains from owning more heads per group or from an interaction between
     the constraint and smaller heads.
   - A cause is a hypothesis: under the constraint a token attends with the heads of its group only, two patterns at
     8 heads and four at 16, while the dense model goes from 8 to 16.
   - The rate of both `_h16` arms was not tuned (15.8.4).

The series therefore answers its three questions under its protocol only:

- the harness does not reach the counted saving of activations;
- under this executor the deeper constrained model is slower than dense at equal counted multiply-adds, by about what
  its doubled depth costs at the measured layer ratio;
- at width 256 the quality result holds within `τ`, and dominates only with four heads per group.

#### 15.8.4 Reserves of the review (2026-10-27)

- **Rates.**
  - E12 is one seed per rate.
  - The rate of `E4a_L4_W256` is flat to a factor of two.
  - Four of the six rates of E13 are computed, not swept: `REF1_W256`, `E4a_L2_W256` and both `_h16` arms.
  - The `_h16` arms took the rate of their 8-head counterparts. A rate tuned for 16 heads could move `i_s` either way,
    and the dominance of the `h16` pair rests on it.
- **Horizon.** `H* = 4,000` was calibrated at `M = 128` on one pair (14.8.4) and is not recalibrated at width.
- **E11.**
  - `delta` comes from the smallest cell (a1). Under the fixed rule c2 stays not resolved (`1 − r` = 0.084); this is
    stated, not re-read.
  - Forward only: no backward, no decoding.
  - The FFN is masked dense, and the padding of the packed form and the TorchScript overhead are part of the time.
  - A native implementation may move every ratio either way.
  - There is no cell at `M = 256`, so the time of the E13 pairs is not measured.
- **E10.** One batch, one seed, float32, one thread, PyTorch's CPU attention kernel. Another kernel stores other
  tensors, so the gap to the counter is a fact of this harness.
- **Minor.**
  - The README of E13 states the validation split as 39,062 windows, which is the count at `N = 128`. Its runs read
    9,765 windows of 512 bytes, the same 5 MB split as E9.
  - The record is not edited and no figure depends on it.

#### 15.8.5 Status of the questions after the series

| Question | Status |
|---|---|
| 15.1, 1 (memory) | answered for this harness: activations measured above dense at equal counted multiply-adds, prefill peak below (15.8.3, point 1); weights, AdamW states and KV cache still counted only |
| 15.1, 2 (time) | answered for this executor and machine, forward only: layer faster, model at equal counted multiply-adds slower, by about what the doubled depth costs (15.8.3, point 2); `IND` against `E4a` resolved at `N = 128` only |
| 15.1, 3 (scale) | answered under the protocol: within `τ` at width with two heads per group, dominates with four (15.8.3, points 3 and 4), with the reserves of 15.8.4 |
| 14.8.5, coupling at equal counted multiply-adds | still open (not in this series) |
| Absolute time, native implementation | not measured (decision 9 as amended by decision 14) |

Proposals for the user, none started:

- **Heads per group at `M = 128`.** `E4a_L4`/`REF2` with 16 heads of 8, with a rate swept for the `h16` layouts.
- **Separating heads per group from head size.** At fixed `M` and `G` the two are tied, so separating them needs an
  attention width other than `M`, which changes the counted multiply-adds. Its design is a question for the user.
- **Time at the width of E13.** A layer cell at `M = 256` under the same executor.
- **A fused or persistent executor**, without one join per layer. It would be a new E11 with its own README; it would
  not re-read this one.
- **A counter that separates the stored tensors** (keys and values, norm and residual rows, gathers, logits), so
  that a counted memory can be compared with a measured one.
- **A third dense point at width 256** (`REF3_W256`), so that the frontier at width can be interpolated as in 14.8.2.
- Left open from 15.6: the `IND` family over depth, and a dense control spending the same counted multiply-adds
  otherwise.

## 16. Using the partition to run the HFFN FFN faster on CPU and GPU (analysis, written 2026-10-27)

Written at the user's request from the discussion of the same day, before any run and outside the series of section
15. Nothing here is measured: every cost is **estimated** (a calculation or a reading of the kernels of `current/`),
every explanation of an expected effect is a **hypothesis**. It changes no step, decision or criterion of the plan;
what it proposes is runtime work (`AGENTS.md` sections 5 and 6), not a step of this study.

### 16.1 What the partition gives the FFN

- After the constraint (C1)-(C2), group `g` receives, for each token routed to it, the dense vector `ζ_t[S_g]` of
  `m = M / G = 32` coordinates. The FFN does not read the attention output itself: `ζ_t = RMSNorm(x_t + o_t)`, the
  residual plus the norm over the `M` coordinates, restricted to `S_g`.
- In E1 and E4a, `up_idx[r] ⊂ S_g` for every `r ∈ R_g` (`build_supports`: `torch.rand(H, m).topk(k)` plus `g m`). `S_g`
  is a contiguous block of features and `R_g` a contiguous block of `n = 128` neurons. Renumbering the up indices
  locally (`i − g m ∈ [0, 32)`) is exact: same parameters, same values, same order of slots.
- Three consequences hold without any change of kernel (estimated):
  - the input rows a group reads shrink from `M` to `m` (by `G`);
  - a local index fits in one byte (`uint8`) instead of the `int16` of the runtime;
  - with top-1 routing, the output `y_t` of a token comes from one group only. On CPU the runtime runs every group
    over the same `N` tokens, writes one private buffer per group (`y_local`, `G × M × N`) and reduces the four
    (`reduce_hidden_owned`). Under routing each group works on its own tokens, so each output column is written by one
    group: the private buffers and their reduction are no longer needed, each group writes the columns of its tokens.
    This holds for E4a (down over `M`) and for `IND` (down in `S_g`) alike; `IND` also limits the written rows to `m`.

### 16.2 Why the up of a group is not a dense block

The partition makes the *input* of a group dense; it does not make the *support* of its up dense. Each neuron draws
its own 26 coordinates out of the 32 of `S_g`, independently of the others: the density inside the block is
`26 / 32 = 81.25 %` (20 % over `M`), and two neurons of a group generally miss different coordinates. Writing the up
of the group as one matrix `n × m` therefore puts 6 zeros per row, at different places in each row.

| Form of the up of one group | Parameters | Exact | Multiply-adds per routed token per layer |
|---|---|---|---|
| compact list, 26 local indices per neuron (the HFFN form) | unchanged | yes | 128 × 26 = 3,328, plus the index reads |
| dense block `128 × 32` with 6 frozen zeros per row | unchanged | yes, if the zeros are never trained (masked gradient and AdamW) | 4,096 (+23 %) |
| `k_up = 32` (full block) | +3,072 per layer | no: another model | 4,096 |

- A dense block without zeros needs `k_up = m`, that is a density of `1 / G` (`d25` at `G = 4`), or every neuron of a
  group sharing one support. Both are new arms of the study (new models), not a new layout of the same model.
- "One compact attention vector per neuron" is the gather itself: `n × k_up = 3,328` floats per routed token and
  group instead of 32. Folding the per-neuron gather into the attention is not possible either: if the attention
  produced, linearly, one output per neuron, it would fold into a dense `n × m` up; the residual and the RMS norm over
  `M` sit between the attention and the FFN and block that fold.

### 16.3 CPU (M1 G4 and x86)

*Where the time of the up goes (estimated, from `hffn_compact_g4_inference.cpp`, lines 98-123).* Each worker walks
its neurons one after the other. For each slot it loads one index and one weight, then runs a NEON FMA over the
contiguous row of 128 tokens of the feature-major input; the 128 accumulators of the neuron sit in a stack array, so
each group of 4 tokens costs a load of the accumulators, a load of the input, the FMA and a store. Per slot that is
two scalar loads against about a hundred vector memory operations: the index is about 1 % of the memory accesses of
the up. The zeros of 16.2 are never read. The index read is therefore not the cost to remove on CPU (estimated).

*What a dense GEMM would buy.* Its advantage is not avoiding the index but reusing each loaded input vector in
registers across several neurons (a register tile of neurons × tokens). That reuse needs neurons that share their
inputs, which is what the zeros pay for: +23 % of multiply-adds (16.2) against fewer loads and stores (hypothesis on
the balance).

*Levers, in the order of their risk.*

1. *Locality, free.* The rows `S_g` of one group are 32 × 128 × 4 B = 16 KB per 128-token batch, against 64 KB for
   all of `M`. At `M = 128` both fit in the L1 data cache of the M1 performance cores; the gain should appear only at
   larger `M` (hypothesis).
2. *`uint8` local indices.* 5 bytes per up slot instead of 6 (−17 % of the bytes of the up parameter stream). Same
   operations in the same order: exact bit for bit.
3. *Routed down without reduction* (16.1): no private buffer and no reduction of four of them. The terms of an output
   are the same (the other groups contributed zeros); keeping the place of the bias in the sum keeps it exact bit for
   bit (estimated: the reduction reads `G M N` floats and the buffers hold as many; the direct write needs neither).
4. *A register tile of neurons over the dense block with frozen zeros.* Not exact bit for bit (reduction order),
   judged under a band against float64 (`runtime.md`, *Accuracy gates*). It reads the zeros of 16.2 and stores the up
   of a group as an `n × m` block: under `AGENTS.md` section 3 that is a masked dense matrix, a control, not an HFFN
   path, and at 81 % density it is outside the favourable regime of section 1. Running it as a production path is a
   decision of the user, not an ablation the rules admit as such.

### 16.4 GPU (Metal M1, CUDA)

*Current structure.* Metal `hffn_inference_up` (generated by `current/runtime/gpu/m1/ffn/training/run_native_train_step.py`
and run by `metal_hffn_training_forward_runner.mm`): one simdgroup of 32 lanes per neuron, 16 neurons per
threadgroup; each lane owns 4 tokens of a 128-token tile; the index and the weight are uniform loads of the
simdgroup, the input reads `x[i * N + t]` are coalesced, the loop is unrolled by 4. CUDA
(`current/runtime/gpu/nvidia/common/kernels/hffn_cuda/kernels.py`): the same layout in `up_lane` and `up_tile`
(`UNROLL = 4`, `UP_TILE = 128`), and `up_coop` (lanes over the slots, warp reduction) for `N ≤ 4`.

*Where the index costs (hypothesis).* Not in bandwidth (a uniform 2-byte load per slot) but in latency: the input
read depends on the index read, so each group of four slots waits for one dependent load before its four input loads
can issue. The unroll by 4 hides part of it; nothing measures the rest.

*Levers, in the order of their risk.*

1. *The input tile of the group in threadgroup or shared memory.* The rows `S_g` of a 128-token tile are
   32 × 128 × 4 B = 16 KB; the full `M = 128` tile is 64 KB. 16 KB fits in the 32 KB of threadgroup memory of the M1
   (64 KB does not); every simdgroup of the threadgroup then reads its 26 inputs per token from on-chip memory with a
   `uint8` local index. The HFFN format is kept, no zero is read, the order of the additions is unchanged: exact bit
   for bit. It requires a threadgroup to hold neurons of one group only, which the contiguous `R_g` allows. This is
   the most solid GPU lever of the partition (hypothesis). At width 256 (`m = 64`), the tile is 32 KB, the whole M1
   limit: the token tile would have to shrink to 64 (calculation).
2. *`uint8` local indices*: as on CPU; smaller effect if the cost is latency rather than bytes (hypothesis).
3. *A block GEMM* on `simdgroup_matrix` (Metal) or tensor cores (CUDA) over the `n × m` block with frozen zeros. The
   same objection under `AGENTS.md` section 3 as on CPU; in addition, tensor cores compute in TF32 or BF16, so FP32
   parity would need a band against float64 that may not hold, and blocks of `128 × 32` are small for these units
   (hypothesis).

*Two costs the partition adds on GPU.* Routing needs a gather of the tokens of each group before the FFN and a
scatter after it (route-major packing, as in `moe/`), two passes over `M × N` floats per layer (estimated). And the
natural unit of parallelism on a GPU is the threadgroup, not one of four cores: the partition would be used there with
a larger `G`, which is another model (decision 2).

### 16.5 What E11 can and cannot say about this

E11 times the packed form of the harness (`ffn_packed`: `zg @ w_up[ns, ms].t()`), that is the dense block of 16.2 with
its zeros, in PyTorch, in every arm. It measures what the division of the work over four workers is worth against
dense attention; it says nothing of the HFFN compact loop of 16.3-16.4 and does not choose between the forms of 16.2
(15.6).

### 16.6 What would settle it, in order

1. *A profile of the current up*, before any change: on CPU the share of the up in the FFN step (the existing probes
   measure the transposes, not the up), on Metal a System Trace and on CUDA Nsight Compute of `up_tile` (stalls on
   the dependent load, cache hit rates), at `M = 128` and at a larger `M`. Without it, the index cost of 16.3-16.4
   stays an estimate and a hypothesis, and no ablation is written from it (`AGENTS.md` section 3: no micro-kernel
   without an integrated profile).
2. *Single-mechanism ablations in `experiments/runtime/`*, each with its README, control `current/` and the light
   protocol, on a model whose up supports lie in the blocks `S_g` (E1 or E4a): the shared-memory tile (16.4, lever 1),
   the routed down without reduction (16.1), `uint8` local indices. They are exact bit for bit and judged on time and
   memory.
3. *The dense block with frozen zeros* (16.3 lever 4, 16.4 lever 3) is a control: it can be timed beside the HFFN path
   to bound what a GEMM would give at 81 % density, never promoted without a decision of the user under section 3.

No statement of speed-up on the HFFN runtime is made before step 2 has qualified runs.

## 17. Fourth step: the frontier of quality against measured inference time (E14, written 2026-10-27)

Written at the user's request on 2026-10-27, before any run. Every number in this section is **estimated** (a
calculation, or figures of different campaigns combined) and every expected effect is a **hypothesis**. The README
of E14 freezes what this section proposes, before its first run.

### 17.1 Why

The series so far read the constrained model at equal depth, at equal parameters and at equal **counted**
multiply-adds. On 2026-10-27 the user set the criterion that decides whether the approximation is useful (decision 16,
17.2): a mixture of experts also loses against a single dense model at equal depth or at equal quality; what makes it
useful is a better quality per unit of compute it demands at inference. The same holds here: the question is whether
the constrained family lies **above the frontier of quality against inference time** of the dense-attention family,
not whether a constrained model matches a dense one of the same depth.

Counted multiply-adds are not that time (15.8.3, point 2). A first frontier was built on 2026-10-27 from the figures
already on record (**estimated**: bpb medians of E8 and E9, times from E11 only for `E4a_L4` against `REF2` and for
`IND_L4` against `E4a_L4`; every dense time other than `REF2` derived from its counted multiply-adds or from its
depth, `E4a_L2` and `E4a_L6` extrapolated as half and 1.5 times `E4a_L4`). Gap = bpb of the arm minus the dense family
interpolated in log time at the arm's time; positive is worse:

| Arm | `N = 128`, dense time by counted MACs | by depth | `N = 512`, by counted MACs | by depth |
|---|---|---|---|---|
| `E4a_L4` (time from E11) | +0.021 | +0.016 | +0.008 | +0.006 |
| `IND_L4` (time from E11) | +0.111 | +0.109 | +0.108 | +0.108 |
| `E4a_L2` (time extrapolated) | −0.020 | +0.006 | −0.041 | −0.023 |
| `E4a_L6` (time extrapolated) | slower and worse than `REF4` | +0.008 | +0.048 | +0.043 |

- The gains at equal counted multiply-adds of 14.8.2 (−0.085 and −0.145 for `E4a_L4`) do not survive the change of
  axis: on time, `E4a_L4` lies on the dense frontier or just behind it, within `τ`, and `IND_L4` is behind it beyond
  `τ` (estimated).
- The sign of `E4a_L2` depends on the time assumed. With part of the cost shared between depths (time 0.70 of `REF2`
  at `N = 128`, 0.65 at 512) its gaps become +0.011 and +0.004 (estimated).
- None of this is a result: five of the eight times are not measured, the figures come from three campaigns, the
  executor is that of E11 and only a forward over whole windows (prefill) was timed. E14 measures the time axis.

### 17.2 Decision of the user recorded here (2026-10-27)

16. **The criterion of usefulness is quality per inference time**, read on the frontier of quality against measured
    inference time, as for a mixture of experts. Readings at equal depth, equal parameters and equal counted
    multiply-adds stay as descriptions; counted multiply-adds are a proxy of time, not time. The decision is
    prospective: the verdicts of sections 13 to 15 are unchanged and no run is re-read under it. The user speaks of
    accuracy; the study records bits per byte (no next-byte accuracy is in the `summary.json` of E8, E9 or E13), so
    the quality axis is the validation bpb, lower is better.

Unchanged: decisions 2, 3, 7, 13, 14, 15; `τ = 0.075` bpb; the timing rule of 15.3 (decision 14).

### 17.3 Fixed design

- **No training.** The quality of every arm is its validation bpb per seed, **measured** in its own campaign: E8
  (`N = 128`), E9 (`N = 512`), E13 (`M = 256`, `N = 512`). E14 measures only time.
- **Arms and anchor.** Every arm of those campaigns is timed against one anchor per campaign, in the same process:
  - `M = 128`, `N = 128` and `N = 512`: anchor `REF2`; variants `REF1`, `REF3`, `REF4`, `E4a_L2`, `E4a_L4`, `E4a_L6`,
    `IND_L4` (7 cells per context);
  - `M = 256`, `N = 512`: anchor `REF2_W256`; variants `REF1_W256`, `REF2_W256_h16`, `E4a_L2_W256`, `E4a_L4_W256`,
    `E4a_L4_W256_h16` (5 cells).

  The time of an arm is `t = t(arm) / t(anchor)`, the anchor at 1. Weights are random (time does not depend on their
  values); inputs are bytes of the valid split, since the route depends on them.
- **Executor.** That of E11 (`lib/executor.py`): four workers, one intra-op thread each, the packed hash form for the
  constrained arms (`E4a` with a barrier per layer, `IND` on its blocks), the dense arms as in E11 (heads `H_g` of
  every token on worker `g`, barrier, `W_O` and the FFN on a block of the tokens). The FFN is masked dense in every
  arm. The concurrency gate of 15.3 runs at the start of each run.
- **Two modes**, each a run of its own with its own `delta`:
  - *Prefill*: one forward over whole windows, 8,192 tokens per call, as in E11.
  - *Batched decoding*: one step of `B = 64` sequences (64 windows of the valid split at offsets drawn per process),
    one new token each, with a KV cache of `N / 2` positions per sequence. The attention of a step is linear in the
    length of the cache and the rest is constant, so the step at `N / 2` is the mean step over a window of `N`
    (estimated). Per layer, constrained: worker `g` computes the keys and values of its heads `H_g` for the 64 tokens
    (dense, decision 3) and appends them to the cache, then for the tokens routed to it the queries of `H_g`, the
    scores against the cache of their sequence, `W_O^g` and FFN group `g`; dense: worker `g` computes the heads `H_g`
    of the 64 tokens against the cache, barrier, then `W_O` and the FFN on 16 tokens. `B = 64` is the batch of the
    training at `N = 128`; one value keeps the cells few, and the effect of `B` is not studied.
- **Decoding path, new in the harness** (no KV cache exists in `lib/`): a preparation adds it, with two checks before
  any timing: the logits of a decoded step equal those of the full forward at the same position within float32
  rounding, for every arm of the three campaigns; the decoded step under the executor equals the sequential decoded
  step. The packed form pads each group to its largest count at that step; the per-group counts are recorded.
- **Timing rule.** That of 15.3 unchanged: five fresh processes per cell, every arm (anchor, copy of the anchor,
  variant) built and warmed up (at least 30 calls and 5 seconds), 12 ABBA windows, slots of at least 0.2 second; per
  cell the median over processes of the per-process median and [min–max]; `delta` = the largest
  `|copy / anchor − 1|` over the cell-processes of the run; steal time and load recorded, nothing excluded after the
  fact.

### 17.4 Readings

**The dense frontier.** Per campaign, mode and seed, the dense-attention arms of the campaign (`REF<L>`, and at
`M = 256` also `REF2_W256_h16`) placed at (time, bpb of that seed); the frontier is their lower envelope (a dense arm
slower and worse than another is not on it), interpolated linearly in log time between neighbouring points, as 14.8.2
interpolated in `log C`. No extrapolation outside the range of dense times.

**The gap of a constrained arm** `c` (`E4a_*`, `IND_L4`) for seed `s`: `g_{c,s} = bpb_s(c) − frontier_s(t_c)`.

- Unfavourable reading `g⁺`: the time of `c` moved up by `delta` and every dense time moved down by `delta`.
- Favourable reading `g⁻`: the reverse.

**Outcomes**, per arm, mode and campaign:

| Outcome | Rule |
|---|---|
| *above the frontier* | `max_s g⁺_{c,s} < 0` |
| *below the frontier beyond `τ`* | `min_s g⁻_{c,s} > τ` |
| *on the frontier* | otherwise; median `g` and [min–max] stated against `τ` |
| *outside the range* | `t_c` beyond the fastest or slowest dense time under either reading: no frontier reading; read only as a pair against the nearest dense arm, `d_s` with the outcomes of 13.1 and *dominates* of 14.3, its time ratio beside |

**The answer to decision 16**, per mode and campaign: *favourable* if at least one `E4a` arm is above the frontier;
*unfavourable* if every `E4a` arm is below it beyond `τ`; otherwise *not resolved*, with the gaps stated. `IND_L4` is
read the same way, as a control. The answers of the two modes are stated apart and not combined.

**Beside, descriptive.** The counted multiply-adds of each arm and the quotient `t` / counted ratio (how far time
follows the count on this executor); for decoding, the KV cache read and stored per decoded token (counted). At equal
counted multiply-adds the constrained model is twice as deep and reads the keys and values of `n_heads / G` heads per
layer, so it **reads** about `2 / G = 0.5` of the cache of the dense model per decoded token while it **stores** twice
as much (calculation). Hence the hypothesis that decoding with a long cache is more favourable to the constraint than
prefill; E14 is the first measurement that can contradict it.

**Classes.** Times are measured (E14), bpb are measured (E8, E9, E13); each axis is compared only inside its own
campaign. A gap combines the two by interpolation and is therefore **estimated**, as the frontier readings of 14.8.2
were; the direct pairs (an arm against its anchor in time and in bpb) are measured.

### 17.5 Cost and launching

- Cells: 19 per mode for the frontier and 6 per mode for the pairs of 17.7, 50 in all. E11 took about 0.5 hour for 8
  cells, so about **3.2 hours** of timing (deeper arms lengthen the build and warm-up, not the windows), and under
  0.25 hour for the memory of 17.7; preparation (decoding path, arm `E2h`, their checks, smoke of every cell) about
  1.5-2.5 hours of work and under 0.5 hour of compute. Total **about 4.5-5.5 hours** of session (estimated). After the
  smoke the total is re-estimated from measured durations; above 6 hours the cells at `M = 256` are cut first and the
  user is told.
- On a Claude cloud session, on the branch `sparse-attention-cloud`, under a brief
  `experiments/sparse_attention/CLOUD_E14.md` written before the start: preparation, prefill run, decoding run, a
  summary. Each timing run alone on the machine. Folder `experiments/sparse_attention/common/llm/training/e14/`, a
  README committed before the first run, its own records and `RESULTS.md`. A harness failure is not an attempt.
- Nothing is promoted: E14 measures, it retains no model.

### 17.6 What E14 will not show

- No absolute time, no time on another machine, no training time, no native implementation: every ratio holds for
  the PyTorch executor of E11 on the cloud machine, and a native implementation (16, or the fused executor of 15.8.5)
  may move every point either way. A favourable answer here is not a statement on the HFFN runtime.
- No new quality figure: the bpb are those of E8, E9 and E13, with their reserves (rates of the `_h16` arms, `H*` not
  recalibrated at width, 14.8.4 and 15.8.4). No accuracy (decision 16).
- At `M = 256` the dense frontier has three points (`REF1_W256`, `REF2_W256`, `REF2_W256_h16`); a constrained arm
  slower than all of them is outside the range. `REF3_W256` (15.8.5) would extend it; it is a training run, not part
  of E14.
- One batch size, one cache length per context, no generation of whole sequences; the memory of the cache is counted,
  not measured, outside the peak (e) of the pairs of 17.7.
- No quality of the attention alone under the hash route: `E2h` is not trained (17.7, *Quality*).

### 17.7 The attention alone at equal depth: arm `E2h` (added 2026-10-27, before any run)

**Why.** No figure on record isolates the sparse attention, in time or in memory. The ratios of E10 and E11 compare
`E4a` with `REF` (or the constrained layer with the dense one), where two things change together: the attention, and
the routing of the FFN (routed in `E4a`, unrouted in `REF`). On the executor of E11 the FFN of a dense arm executes
131,072 multiply-adds per token and layer (masked dense, 512 × 128 twice), the routed FFN 20,480; at `N = 128` about
two thirds of the multiply-adds that `E4a` saves against dense come from the routing of the FFN, about half at
`N = 512` (estimated, counted on the executor). Counted on the model, the attention alone of a constrained layer is
0.47 of the dense one at `N = 128` and 0.36 at `N = 512`, and the whole layer 0.50 and 0.38 once the routed FFN is
the same in both (estimated). The step of the ladder that varies only the attention is `E2` → `E3b` (section 8); under
the hash route it was never built (13.4, *not run*; 13.5, question 1).

**The arm.** `E2h` is `E4a` with the dense attention of `REF`: every head for every token, `W_O` dense; the FFN is that
of `E4a` (inputs inside the blocks, as E1; token `t` uses group `g` given by the balanced byte hash of E4 (a)); same
`M`, heads, `H`, `k`, embedding and output head. It has the parameters of `E2`: 12,288 more per layer than `E4a` at
`M = 128` (`W_O`). Under the executor of E11: the heads as in the dense arms (heads `H_g` of every token on worker `g`,
barrier), then the tail of `E4a` (`W_O`, dense, and FFN group `g` on the tokens routed to `g`, packed form padded as
for `E4a`). In decoding: worker `g` computes the keys, values and queries of `H_g` for the 64 tokens against the cache,
barrier, then `W_O` and FFN group `g` on the tokens routed to `g`. The checks of 17.3 apply to `E2h` (full forward
against the sequential reference, decoded step against the full forward and against the sequential decoded step).

Thus the pair `E4a` / `E2h` differs by the attention only: queries of `H_g` computed for the tokens routed to `g`,
`W_O` block-diagonal, and the scheduling this allows (one barrier per layer instead of two). The scheduling is
counted as part of the attention on purpose: spreading the attention over the workers of the HFFN groups is what the
constraint is for (15.1, point 2). The pair `E2h` / `REF` differs by the routing of the FFN only.

**Cells.** One depth, `L = 4` (the depth of `E4a_L4`, the central constrained arm of E8, E9 and E13, and of the cells
b1-c2 of E11); the ratio at equal depth moves with depth only through the embedding and the output head (hypothesis),
and a second depth answers no question of the decision. Per mode (prefill and batched decoding, as in 17.3):

| Cell | Variant / anchor | `M` | `N` | What it isolates |
|---|---|---|---|---|
| p1 | `E4a_L4` / `E2h_L4` | 128 | 128 | the attention alone |
| p2 | `E2h_L4` / `REF4` | 128 | 128 | the routing of the FFN alone |
| p3 | `E4a_L4` / `E2h_L4` | 128 | 512 | the attention alone |
| p4 | `E2h_L4` / `REF4` | 128 | 512 | the routing of the FFN alone |
| p5 | `E4a_L4_W256` / `E2h_L4_W256` | 256 | 512 | the attention alone, at width |
| p6 | `E2h_L4_W256` / `REF4_W256` | 256 | 512 | the routing of the FFN alone, at width |

The heads per group are those of `E4a_L4_W256` in the three arms of p5-p6. `REF4_W256` and every `E2h` have no
training run; time and memory need none (random weights, 17.3). The timing rule is that of 17.3: the anchor, its copy
and the variant in the same processes, `delta` of the run (the two modes are two runs; the pairs may share them with
the frontier cells).

**Memory.** The quantities (a)-(c) of E10 (15.3) for the nine arms of p1-p6, measured again in one run (the E10 figures
of `E4a_L4` and `REF4` are of another campaign and are not reused), plus (e) the peak of live tensor bytes of one
decoded step (`B = 64`, cache of `N / 2`) by the method of (c). Rule of E10: two fresh processes, (a) identical, (b),
(c) and (e) within 1 %, otherwise a harness failure. Counted beside: parameters and AdamW states (`E4a` / `E2h` 0.868
per layer at `M = 128`), the KV cache stored (1 in both pairs, decision 3) and read per decoded token (`E4a` / `E2h`
0.25 at equal depth; `E2h` / `REF4` 1), the activations of the counter.

**Readings.** Per cell and mode: the ratio, median over processes and [min–max], and `delta`. A gain of the variant is
stated only when `1 − ratio > delta`, a loss only when `ratio > 1 + delta`; otherwise *no difference within `delta`*.
The memory ratios are stated as measured, without `delta`. Beside, descriptive: the counted ratio of each cell and the
quotient measured / counted; the product p1 × p2 (and p3 × p4, p5 × p6), **estimated** since it combines two cells, read
against the counted `E4a_L4` / `REF4`. The answers of the two modes and of the three contexts are stated apart.

**Quality.** `E2h` has no bpb, so it is not on the frontier of 17.4 and the pairs give no quality per time. The price
of the attention alone is measured only at `L = 1` under the position route (13.2: `E2` → `E3a` +0.116,
`E3a` → `E3b` +0.113, each beyond `τ`); under the hash route and at depth it is not isolated. Training `E2h_L4` at
`N = 128` and `N = 512` (seeds 0, 1, 2, the rate of `E4a_L4`) would place it, about 2 hours from the durations of 15.4
(estimated); it is not part of E14 and is a decision of the user.

**Limits.**

- p2, p4 and p6 read the routing of the FFN against a masked dense FFN in PyTorch (15.3), which executes the whole
  512 × 128 matrices; the unrouted FFN of the HFFN runtime executes its compact support only. These cells say nothing
  of the HFFN runtime and overstate the gain of the routing there (hypothesis).
- In prefill the dense heads call SDPA with `is_causal=True` and the constrained ones with an explicit mask over all
  `N` positions; if the causal kernel skips the masked half of the scores, p1, p3 and p5 understate the gain of the
  attention in prefill (hypothesis). The README of E14 records the SDPA kernel selected in each arm.
- The ratios hold for the executor of E11 on the cloud machine (17.6, first point).
