"""One configuration per arm of the ladder (PLAN section 8).

An arm is a frozen `ArmConfig`; the model code in `model.py` has no other switch. The nominal sizes are those of
PLAN section 7 (M = 128, 8 heads of 16, G = 4, H = 512, degree 26).
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ArmConfig:
    name: str
    n_layers: int = 1
    # attention: "dense" (every head for every token), "none" (calibration), "routed" (token t computes the head
    # group H_g of its route only, o_t lives on S_g: needs wo_blockdiag and ffn_routed)
    attn: str = "dense"
    wo_blockdiag: bool = False      # W_O restricted to the m x m blocks (head group H_g writes S_g)
    qkv_blockdiag: bool = False     # IND: W_Q, W_K, W_V of H_g read S_g only
    # FFN support: "random" (E0), "blocks" (up_idx inside S_g of the neuron's group, down over M),
    # "ind" (up_idx and down_idx inside S_g)
    ffn_support: str = "random"
    ffn_routed: bool = False        # token uses the neurons R_g of its route only
    routing: str = "position"       # "position" (t mod G), "hash" (balanced byte hash), "learned" (top-1 router)
    layer_routing: str = "fixed"    # "fixed" or "rotation" (route shifted by the layer index)
    gate: str = "none"              # learned router: "none", "p" (scale by p), "Gp" (scale by G * p)
    capacity_factor: float = 0.0    # learned router: capacity per group = ceil(cf * N / G); 0 = no capacity
    balance_coef: float = 0.0       # learned router: coefficient of the Switch balance loss in the training loss
    # nominal sizes
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 8
    groups: int = 4
    hidden: int = 512
    degree: int = 26
    context: int = 128

    def __post_init__(self):
        assert self.attn in ("dense", "none", "routed")
        assert self.ffn_support in ("random", "blocks", "ind")
        assert self.routing in ("position", "hash", "learned")
        assert self.layer_routing in ("fixed", "rotation")
        assert self.gate in ("none", "p", "Gp")
        if self.attn == "routed":
            assert self.wo_blockdiag and self.ffn_routed, "routed attention needs W_O blocks and a routed FFN"
        if self.ffn_routed:
            assert self.ffn_support in ("blocks", "ind"), "a routed FFN needs neurons that read one block"
        if self.qkv_blockdiag:
            assert self.attn == "routed"
        if self.routing != "position":
            assert self.ffn_routed
        assert self.d_model % self.groups == 0 and self.n_heads % self.groups == 0
        assert self.hidden % self.groups == 0

    @property
    def m(self):
        return self.d_model // self.groups

    @property
    def n(self):
        return self.hidden // self.groups

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    def with_(self, **kw):
        return replace(self, **kw)


E0 = ArmConfig("E0")
NOATT = ArmConfig("NOATT", attn="none")
E1 = ArmConfig("E1", ffn_support="blocks")
E2 = ArmConfig("E2", ffn_support="blocks", ffn_routed=True)
E3A = ArmConfig("E3a", ffn_support="blocks", ffn_routed=True, wo_blockdiag=True)
E3B = ArmConfig("E3b", ffn_support="blocks", ffn_routed=True, wo_blockdiag=True, attn="routed")
IND = ArmConfig("IND", ffn_support="ind", ffn_routed=True, wo_blockdiag=True, attn="routed", qkv_blockdiag=True)
E4A = E3B.with_(name="E4a", routing="hash")
E4B = E3B.with_(name="E4b", routing="learned", gate="Gp", capacity_factor=1.25, balance_coef=0.01)

# depth (E5): the dense reference and the constrained model at L = 4
REF4 = E0.with_(name="REF4", n_layers=4)
E3B4 = E3B.with_(name="E3b_L4", n_layers=4)
E3B4R = E3B.with_(name="E3b_L4rot", n_layers=4, layer_routing="rotation")

E4A_L4 = E4A.with_(name="E4a_L4", n_layers=4)
E4A_L4R = E4A.with_(name="E4a_L4rot", n_layers=4, layer_routing="rotation")

# second series (E6-E9, PLAN section 14): two families by depth, one control, two contexts
REF1 = E0.with_(name="REF1")  # = E0 under another name
REF2 = E0.with_(name="REF2", n_layers=2)
REF3 = E0.with_(name="REF3", n_layers=3)
E4A_L2 = E4A.with_(name="E4a_L2", n_layers=2)
E4A_L6 = E4A.with_(name="E4a_L6", n_layers=6)
IND_L4 = IND.with_(name="IND_L4", n_layers=4, routing="hash")

SERIES = (REF1, REF2, REF3, REF4, E4A_L2, E4A_L4, E4A_L6, IND_L4)
SERIES_N512 = tuple(a.with_(name=a.name + "_N512", context=512) for a in SERIES)

ARMS = {a.name: a for a in (E0, NOATT, E1, E2, E3A, E3B, IND, E4A, E4B, REF4, E3B4, E3B4R, E4A_L4, E4A_L4R,
                            REF1, REF2, REF3, E4A_L2, E4A_L6, IND_L4, *SERIES_N512)}


def get_arm(name: str, **overrides) -> ArmConfig:
    arm = ARMS[name]
    return replace(arm, **overrides) if overrides else arm
