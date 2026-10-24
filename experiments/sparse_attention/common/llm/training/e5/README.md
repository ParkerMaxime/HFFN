# E5 — depth: four layers

Written on 2026-10-24 before any run of this step; frozen once it has run. Protocol, statistics and checks:
`../e0/README.md`; bounds and sign convention: `../e1/README.md`. Plan: `current/sparse_attention/PLAN.md` sections 4, 5, 8, 12.

## Single mechanism and question

**Depth: `L = 4`.** One layer cannot show the mixing between layers (PLAN section 5): IND (E3b without cross-block paths)
was within 0.003 bpb of E3b at `L = 1` (`../ind/RESULTS.md`), so the mixing paths gave no measurable gain there. At
`L = 4` a block written by group `g` is read by the other groups of the next layer. Question: **does the constraint
cost more or less at depth, and does a rotation of the route across layers help.**

## Decisions of PLAN section 12 fixed here

- **Which constrained model.** "As the best of E3b/E4": E4a (balanced byte hash) was better than E3b in every seed
  (`../e4a/RESULTS.md`, median −0.226 bpb); E4b (learned router) was inconclusive (`../e4b/RESULTS.md`). The
  constrained model of E5 is therefore **E4a at `L = 4`** (`E4a_L4`: the same hash route at every layer, i.e. fixed
  routing). E3b (position routing) at `L = 4` is also run, because the fixed-against-rotation comparison of PLAN
  section 4 is defined on position routing, and as the link to the earlier steps.
- **Order of the comparison fixed against rotation.** First the main comparison, fixed routing against the dense
  reference (stage `main`); then rotation against fixed on the hash route (stage `rot`: route at layer `l` =
  (hash group + `l`) mod 4); then rotation against fixed on position routing (stage `pos`: route at layer `l` =
  (`t` + `l`) mod 4, against `t mod 4`). All in one campaign, so every difference is paired and in one campaign.
- **Learned router per layer: not entered.** E4b did not beat position routing (bimodal, inconclusive) and cost a
  selected-form implementation; a per-layer router would add a second unvalidated mechanism. It is a candidate for a
  later step if the user wishes (not run, said so in SUMMARY).

## Arms

| Arm | Layers | Attention | FFN | Route | Parameters outside embedding |
|---|---|---|---|---|---|
| `REF4` (control) | 4 | dense | HFFN `d20`, random support | — | 372,352 |
| `E4a_L4` | 4 | routed, `W_O` blocks | blocks, routed | byte hash, same at every layer | 323,200 |
| `E4a_L4rot` | 4 | as above | as above | byte hash + layer index (mod 4) | 323,200 |
| `E3b_L4` | 4 | as above | as above | `t mod 4` at every layer | 323,200 |
| `E3b_L4rot` | 4 | as above | as above | `(t + l) mod 4` | 323,200 |

Each layer has its own supports and weights (seed-dependent, generator named by layer). Keys and values stay dense in
every layer (KV cache dense, PLAN section 4). Implementations: packed for position routing, selected for the hash.

## Learning rate and runs

- Learning rate: the rate of E0 was chosen at `L = 1`; at `L = 4` a sweep is run first, with the rules of
  `../e0/README.md`: `REF4` and `E4a_L4`, grid `{0.002, 0.004, 0.008, 0.016, 0.032, 0.064}`, seed 100, **1,500 steps**,
  criterion = final validation bpb. `REF4` uses its own best rate; `E4a_L4` and the other constrained arms
  (`E4a_L4rot`, `E3b_L4`, `E3b_L4rot`) use the best rate of `E4a_L4`. If the best rate of an arm is at an edge of the
  grid, the sweep is inconclusive and repeated on a wider grid (new run, earlier one kept).
- Seeds 0, 1, 2, 3; 4,000 steps; otherwise the protocol of E0. Stages: `main` (`REF4`, `E4a_L4`), `rot` (`E4a_L4rot`),
  `pos` (`E3b_L4`, `E3b_L4rot`): 20 jobs after the sweep.

## Statistic and bounds (all fixed now)

For each pair, `d_s = bpb(candidate, s) − bpb(control, s)` per seed (positive = candidate worse), bound `τ = 0.075` bpb
(as in E1-E4; 5 % of the `Δ_attn` of E0 at `L = 1`; the dynamic range at `L = 4` is not measured here), verdict by the
rule of E1 (within / beyond / inconclusive; "better in every seed"):

| Candidate | Control | Question |
|---|---|---|
| `E4a_L4` | `REF4` | cost of the constraint at depth |
| `E4a_L4rot` | `E4a_L4` | does rotating the hash route help |
| `E3b_L4` | `REF4` | same, with position routing |
| `E3b_L4rot` | `E3b_L4` | does rotating the position route help |
| `E4a_L4` | `E3b_L4` | does the hash keep its advantage at depth |

The mixing diagnostic (PLAN section 5) is reported for every arm: at `L = 4` it is the descriptive reading of
inter-layer mixing. Analytical saving per token and layer: as PLAN section 6 (estimated, not measured here).

## Launch

`./run.sh check | smoke | run`; sweep (12 jobs) then the stages. Records as in E0. A 4-layer run is several times
slower than a 1-layer run; the wall-clock is not a result.
