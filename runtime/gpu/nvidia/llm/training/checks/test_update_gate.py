"""CPU unit test of `update_gate` of the leaf `llm/training`: no GPU, no export, no model.

A fake build holds float32 parameters, moments and step counters for two layers as the runtime does (`out_val` stored in the
output-major order of a non-trivial permutation) and applies the AdamW update of the compact kernel's form in float32 (scalars
rounded to float32) on gradients drawn per step over five decades; a zero loss gradient gives a zero gradient, as the equations
do. The driver below follows the order of `model_check.arm_run`: snapshot before every step, one step, snapshot after, comparison
of that update; the decay-only pass runs on a fresh build.

Checks (README, Update gate): the correct build passes both passes in both representations; each of the eight reference-side faults
exceeds 1e-5 on both judged groups in both representations; each of the four build-side faults is detected on both groups; an
empty mask fails closed; a non-finite parameter fails; a counter off by one fails; a gradient read from another place than the
one the update used fails; a non-zero gradient in the decay-only pass fails; a conditioned weight of 3e-6 (below lr) perturbed
by 5e-11 passes and by 2e-8 fails; the scalar form equals `ts.adamw_closed_form` when fed double-derived scalars; an incomplete
horizon fails; the bound-tree comparison (identical tree, a changed, missing or added file, an empty start). Run: python -B test_update_gate.py (any Python with torch; imports `train_step` of `ffn/training`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
sys.path.insert(0, str(REPO / "runtime/common/ffn"))             # hffn_ffn_layers, imported by train_step
sys.path.insert(0, str(REPO / "runtime/gpu/nvidia/ffn/training"))   # train_step
sys.path.insert(0, str(HERE))
import train_step as ts  # noqa: E402
import update_gate as ug  # noqa: E402

SIZES = {"up_val": 4096, "b_up": 64, "out_val": 4096, "b_down": 32}
GRADS_OF = {"up_val": "g_up", "b_up": "gb_up", "out_val": "g_down", "b_down": "gb_down"}
SMALL = 7                      # index of the held weight of the small-weight cases
LAYERS = 2
PERM = torch.randperm(SIZES["out_val"], generator=torch.Generator().manual_seed(5))
FULL = {g: torch.ones(SIZES[k], dtype=torch.bool) for k, g in GRADS_OF.items()}
BUILD_FAULTS = {"build_weight_decay_zero": {"weight_decay": 0.0}, "build_eps_times_1000": {"eps": ts.HYPER["eps"] * 1000},
                "build_beta2_0.99": {"betas": (0.9, 0.99)}, "build_lr_times_1.0001": {"lr": ts.HYPER["lr"] * 1.0001}}


class FakeBuild:
    """Float32 parameters, moments and counters of `LAYERS` layers; `step` applies the kernel's AdamW form on a drawn gradient."""

    def __init__(self, hyper: dict | None = None, nan: bool = False, counter_skew: int = 0, perturb: float = 0.0, leak: bool = False):
        self.h, self.nan, self.skew, self.perturb, self.leak = hyper or ts.HYPER, nan, counter_skew, perturb, leak
        self.p, self.mv, self.g, self.steps = [], [], [], [0] * LAYERS
        for layer in range(LAYERS):
            gen = torch.Generator().manual_seed(100 + layer)
            hm = {k: torch.randn(n, generator=gen) * 10 ** (torch.rand(n, generator=gen) * 3 - 3.5) for k, n in SIZES.items()}
            if perturb and layer == 0:
                hm["up_val"][SMALL] = 3e-6
            hm["out_val"] = hm["out_val"][PERM]                 # stored output-major: stored[i] = hidden-major[PERM[i]]
            self.p.append(hm)
            self.mv.append({k: (torch.zeros(n), torch.zeros(n)) for k, n in SIZES.items()})
            self.g.append({})
        self.perturbed = None

    def snapshot(self, layer: int) -> dict:
        return {k: (self.p[layer][k].clone(), *(t.clone() for t in self.mv[layer][k])) for k in SIZES}

    def counters(self, layer: int) -> list:
        return [self.steps[layer] + self.skew] * 4

    def step(self, k: int, live: bool) -> None:
        live = live or self.leak
        b1, b2 = self.h["betas"]
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        decay, step_size, root = f(self.h["lr"] * self.h["weight_decay"]), f(self.h["lr"] / (1 - b1 ** k)), f(math.sqrt(1 - b2 ** k))
        for layer in range(LAYERS):
            self.steps[layer] += 1
            gen = torch.Generator().manual_seed(1000 * layer + k)
            g = {GRADS_OF[n]: torch.randn(s, generator=gen) * 10 ** (torch.rand(s, generator=gen) * 5 - 7) * live for n, s in SIZES.items()}
            if self.perturb and layer == 0:
                g["g_up"][SMALL] = 0.0
            self.g[layer] = g                                    # hidden-major, what the checker reads
            for name, p in self.p[layer].items():
                m, v = self.mv[layer][name]
                grad = g[GRADS_OF[name]]
                grad = grad[PERM] if name == "out_val" else grad
                m.mul_(f(b1)).add_(f(1 - b1) * grad)
                v.mul_(f(b2)).add_(f(1 - b2) * grad * grad)
                p.sub_(decay * p).sub_(step_size * m / (v.sqrt() / root + f(self.h["eps"])))
            if self.perturb and live and k == 3 and layer == 0:   # the held weight, after the update of step 3 (pass A only)
                self.perturbed = float(self.p[0]["up_val"][SMALL])
                self.p[0]["up_val"][SMALL] += self.perturb
            if self.nan and layer == 1:
                self.p[1]["up_val"][0] = float("nan")

    def reported_gradient(self, layer: int, shadow: bool = False) -> dict:
        """What the checker reads. `shadow` is another buffer than the one the update used (a recomputation)."""
        g = self.g[layer]
        return {n: t * (1.001 if shadow else 1.0) for n, t in g.items()}


def pass_a(arm: str, build: FakeBuild, faults=(), masks=FULL, steps: int = ug.STEPS, shadow: bool = False, skip=(), expected=LAYERS) -> dict:
    gate = ug.UpdateGate(arm, "cpu", faults)
    for k in range(1, steps + 1):
        pre = [build.snapshot(layer) for layer in range(LAYERS)]
        build.step(k, True)
        for layer in range(LAYERS):
            if layer not in skip:
                gate.update(layer, k, pre[layer], build.snapshot(layer), build.reported_gradient(layer, shadow), masks, PERM, build.counters(layer))
    return gate.result(expected)


def pass_b(arm: str, build: FakeBuild, faults=(), masks=FULL, skip=(), expected=LAYERS) -> dict:
    gate = ug.DecayGate(arm, "cpu", faults)
    pre = [build.snapshot(layer) for layer in range(LAYERS)]
    for k in range(1, ug.STEPS + 1):
        build.step(k, False)
        gate.gradient(max(float(t.abs().max()) for layer in range(LAYERS) for t in build.g[layer].values()))
    for layer in range(LAYERS):
        if layer not in skip:
            gate.layer(layer, pre[layer], build.snapshot(layer), masks, PERM, build.counters(layer))
    return gate.result(expected)


def both(arm: str, faults=(), masks=FULL, **change) -> tuple[dict, dict]:
    return pass_a(arm, FakeBuild(**change), faults, masks), pass_b(arm, FakeBuild(**change), faults, masks)


def main() -> int:
    results, ok = [], True

    def record(name: str, got: bool, want: bool, note: str = "") -> None:
        nonlocal ok
        results.append((name, got, want, note))
        ok &= got == want

    for arm in ("hffn", "dense32"):
        a, b = both(arm, faults=ug.FAULT_NAMES)
        record(f"{arm}:none_passes_both_passes", a["pass"] and b["pass"], True,
               f"A={max(r[e] for r in a['groups'] for e in ('value', 'm', 'v')):.2e} B={max(r['decay'] for r in b['groups']):.2e}")
        det = ug.detection(a, b)
        record(f"{arm}:every_fault_detected_on_both_groups", det["pass"], True,
               " ".join(f"{n}={min(r['margin'] for r in row.values()):.1e}x" for n, row in det["faults"].items()))
        for name, row in det["faults"].items():
            record(f"{arm}:fault:{name}", all(r["detected"] for r in row.values()), True)
        if arm == "hffn":
            record("hffn:nominal_form_recorded", set(a["nominal_recorded"]) == {"up_val", "down_val"}, True,
                   f"{a['nominal_recorded']}")
        record(f"{arm}:counters_and_steps_complete", a["counters_ok"] and a["steps_complete"] and b["counters_ok"], True)
        record(f"{arm}:decay_moments_exactly_zero", b["moments_exactly_zero"] and b["zero_gradient"], True)
        record(f"{arm}:all_layers_covered", a["layers"] == LAYERS and b["layers"] == LAYERS and a["layers_complete"] and b["layers_complete"], True)
    for name, change in BUILD_FAULTS.items():                  # the gate keeps the nominal hyper-parameters: a fault of the build
        a, b = both("hffn", hyper={**ts.HYPER, **change})
        det = ug.build_fault_detected(a, b)
        record(name, det["pass"], True, " ".join(f"{p}={'fail' if not a['group_pass'][p] or not b['group_pass'][p] else 'pass'}" for p in a["group_pass"]))
    empty = {**FULL, "g_up": torch.zeros(SIZES["up_val"], dtype=torch.bool)}
    a, b = both("hffn", masks=empty)
    record("empty_mask_fails_closed", a["pass"] or b["pass"], False)
    record("empty_mask_fails_the_group_only_in_the_judged_group", not a["group_pass"]["up_val"] and a["group_pass"]["down_val"], True)
    a, b = both("hffn", nan=True)
    record("non_finite_fails", a["pass"] or b["pass"], False)
    record("non_finite_recorded", a["finite"], False)
    a, b = both("hffn", counter_skew=1)
    record("counter_off_by_one_fails", a["counters_ok"] or b["counters_ok"], False)
    a, b = both("hffn", counter_skew=-1)
    record("counter_one_behind_fails", a["counters_ok"] or b["counters_ok"], False)
    a = pass_a("hffn", FakeBuild(), shadow=True)
    record("gradient_from_another_buffer_fails", a["pass"], False)
    b = pass_b("hffn", FakeBuild(leak=True))
    record("decay_pass_gradient_fails", b["pass"], False, f"gradient_max={b['gradient_max']:.2e}")
    a, b = pass_a("hffn", FakeBuild(), skip=(1,)), pass_b("hffn", FakeBuild(), skip=(1,))
    record("skipped_layer_fails_pass_a", a["pass"], False, f"layers={a['layers']}/{a['expected_layers']}")
    record("skipped_layer_fails_decay_pass", b["pass"], False, f"layers={b['layers']}/{b['expected_layers']}")
    record("skipped_layer_other_checks_still_pass", all(a["group_pass"].values()) and all(b["group_pass"].values()), True)
    a, b = pass_a("hffn", FakeBuild(), skip=(0,)), pass_b("hffn", FakeBuild(), skip=(0,))
    record("skipped_first_layer_fails", a["pass"] or b["pass"], False)
    a, b = pass_a("hffn", FakeBuild(), expected=LAYERS + 1), pass_b("hffn", FakeBuild(), expected=LAYERS + 1)
    record("model_with_more_layers_than_judged_fails", a["pass"] or b["pass"], False)
    a, b = pass_a("hffn", FakeBuild(), expected=0), pass_b("hffn", FakeBuild(), expected=0)
    record("zero_expected_layers_fails_closed", a["pass"] or b["pass"], False)
    a = pass_a("hffn", FakeBuild(), steps=4)
    record("incomplete_horizon_fails", a["pass"], False)
    small = FakeBuild(perturb=5e-11)
    a = pass_a("hffn", small)
    record("small_weight_rounding_passes", a["pass"], True, f"perturbed weight {small.perturbed}")
    record("small_weight_is_below_lr", small.perturbed is not None and 2.9e-6 < small.perturbed < 3e-6, True)
    a = pass_a("hffn", FakeBuild(perturb=2e-8))
    record("small_weight_update_error_fails", a["pass"], False)
    record("small_weight_update_error_fails_up_group_only", (not a["group_pass"]["up_val"]) and a["group_pass"]["down_val"], True)
    gen = torch.Generator().manual_seed(7)
    p, g = torch.randn(512, generator=gen).double(), torch.randn(512, generator=gen).double() * 1e-3
    m, v = torch.randn(512, generator=gen).double() * 1e-4, torch.rand(512, generator=gen).double() * 1e-8
    exact = ug.adamw(p, g, m, v, 3, scalars=False, **ts.HYPER)
    plain = ts.adamw_closed_form(p, g, m, v, 3, **ts.HYPER)
    scaled = ug.adamw_closed_form_scalars(p, g, m, v, 3)
    gap = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(exact, plain))
    rounded = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(scaled, plain))
    record("double_scalar_form_equals_closed_form", gap < 1e-12, True, f"gap={gap:.2e}")
    record("float32_scalar_form_within_float32_rounding", 0 < rounded < 1e-6, True, f"gap={rounded:.2e}")
    scalars = ug.float32_scalars(3)
    record("eight_float32_scalars", len(scalars) == 8 and all(ug.f32(x) == x for x in scalars.values()), True)
    import model_check as mc      # stdlib-only until import_runtime; the ids below come from the code `main` runs, not from `planned_ids`
    fault_keys = set(mc.build_faults(ts.HYPER))
    record("build_fault_keys_equal_planned_names", fault_keys == set(mc.BUILD_FAULT_NAMES), True, str(sorted(fault_keys)))
    produced = ["dense64", "dense32_probe", "hffn", "dense32"] + [mc.decay_id(name) for name in mc.JUDGED_ARMS]
    for stage, faulted in (("70M", True), ("410M", False)):
        run_ids = produced + ([mc.fault_id(name) for name in mc.build_faults(ts.HYPER)] if faulted else []) + ["model_gate"]
        planned = mc.planned_ids(faulted)
        record(f"planned_ids_equal_recorded_ids_{stage}", sorted(planned) == sorted(run_ids) and len(set(planned)) == len(planned), True,
               f"{len(planned)} ids")
    record("planned_id_counts", (len(mc.planned_ids(True)), len(mc.planned_ids(False))), (11, 7))
    changes = mc.build_faults(ts.HYPER)
    record("build_faults_change_one_hyperparameter_each",
           all(len([k for k, v in {**ts.HYPER, **change}.items() if v != ts.HYPER[k]]) == 1 for change in changes.values()), True)
    files = {"runtime/a.py": "aa", "runtime/b.md": "bb", "runtime/c.json": "cc"}
    record("bound_identical_tree_has_no_problem", mc.compare_bound(files, dict(files)), [])
    record("bound_changed_file", mc.compare_bound(files, {**files, "runtime/b.md": "xx"}), ["changed: runtime/b.md"])
    record("bound_missing_file", mc.compare_bound(files, {k: v for k, v in files.items() if k != "runtime/a.py"}),
           ["missing: runtime/a.py"])
    record("bound_added_file", mc.compare_bound(files, {**files, "runtime/new.py": "dd"}), ["added: runtime/new.py"])
    record("bound_empty_start_fails_closed", len(mc.compare_bound({}, dict(files))), len(files))
    record("bound_digest_ignores_order", mc.digest_of(files), mc.digest_of(dict(reversed(list(files.items())))))
    for name, got, want, note in results:
        print(f"{'ok ' if got == want else 'BAD'} {name:58s} got={got!s:5s} expected={want!s:5s} {note}")
    print("test_update_gate: PASSED" if ok else "test_update_gate: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
