"""Check-only CPU training API coverage, using the existing gate2 oracle.

No production path imports this module. Numerical local-form helpers are
self-contained; there is no second layer equation. Refusal/fault suites and
actual M1 callers are separate, not claimed by these seven per-layer rows.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path
import sys

import torch

from hffn_ffn_training.module import HFFNTrainFFN
from hffn_ffn_training.contexts import HFFNContextFFN
from hffn_ffn_training import checkpoint

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'runtime' / 'common' / 'pythia' / 'hffn_pythia').is_dir())
COMMON = ROOT / 'runtime/cpu/common/ffn/training/hffn_ffn_training'
P = ('up_val', 'down_val', 'b_up', 'b_down')
G = ('grad_up', 'grad_down', 'grad_b_up', 'grad_b_down')
NATIVE_G = ('g_up', 'g_down', 'g_b_up', 'g_b_down')
QUANTITIES = ('y', 'z', 'h', 'grad_x', *G)
RESERVATION = 128


def bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int32) if a.dtype == torch.float32 else a,
        b.contiguous().view(torch.int32) if b.dtype == torch.float32 else b)


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item: visit(element)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def raw_step(owner): return owner._step if isinstance(owner, HFFNContextFFN) else owner.step


def support(step):
    return {name: tensor.clone() for name, tensor in vars(step.layer).items()
            if isinstance(tensor, torch.Tensor) and name not in P}


def supports(step, saved): return all(bits(getattr(step.layer, name), value) for name, value in saved.items())


def ownership(step):
    values = {name: getattr(step.layer, name) for name in P}
    values.update({'moment/' + str(i): value for i, value in enumerate(step.moments)})
    values.update({'reserved/' + name: value[0] for name, value in step._reserved.items()})
    values.update({name: getattr(step, name) for name in (*NATIVE_G, '_gx_local', 'timing')})
    values.update({'support/' + name: value for name, value in vars(step.layer).items()
                   if isinstance(value, torch.Tensor) and name not in P})
    return {name: (value.data_ptr(), str(value.dtype), tuple(value.shape)) for name, value in values.items()}


def layer_equal(a, b):
    aa = {name: value for name, value in vars(a).items() if isinstance(value, torch.Tensor)}
    bb = {name: value for name, value in vars(b).items() if isinstance(value, torch.Tensor)}
    return (a.d_model, a.d_ff) == (b.d_model, b.d_ff) and set(aa) == set(bb) and all(bits(value, bb[name]) for name, value in aa.items())


def same_equation(a, b):
    if (a.d_model, a.d_ff) != (b.d_model, b.d_ff): return False
    for side in ('up', 'down'):
        if not torch.equal(getattr(a, side + '_offsets'), getattr(b, side + '_offsets')): return False
        owner = torch.repeat_interleave(torch.arange(a.d_ff), getattr(a, side + '_offsets').diff().long())
        keys = [owner * a.d_model + getattr(layer, side + '_idx').long() for layer in (a, b)]
        orders = [torch.argsort(key) for key in keys]
        if not torch.equal(keys[0][orders[0]], keys[1][orders[1]]) or not bits(
                getattr(a, side + '_val')[orders[0]], getattr(b, side + '_val')[orders[1]]): return False
    return bits(a.b_up, b.b_up) and bits(a.b_down, b.b_down)


class Trace:
    def __init__(self, kernel, events): self.kernel, self.events = kernel, events
    def __getattr__(self, name):
        target = getattr(self.kernel, name)
        def called(*args):
            if name == 'forward_dual_slots_nt': n = args[13]
            elif name == 'backward_dual_nt': n = args[15]
            elif name in ('backward_nt', 'backward_weights_nt'): n = args[11]
            else: n = None
            self.events.append(dict(name=name, n=n, scalars=list(args[-6:]) if name == 'adamw' else None))
            return target(*args)
        return called


def expected_trace(n, need):
    route = ('backward_dual_nt' if n >= 32 else 'backward_nt') if need else 'backward_weights_nt'
    return [dict(name='forward_dual_slots_nt', n=n, scalars=None), dict(name=route, n=n, scalars=None)]


def adamw_trace(lr, wd, number): return dict(name='adamw', n=None, scalars=[lr, .9, .999, 1e-8, wd, number])


def snap(step):
    return [(getattr(step.layer, p).clone(), getattr(step, g).clone(),
             step.moments[2 * i].clone(), step.moments[2 * i + 1].clone()) for i, (p, g) in enumerate(zip(P, NATIVE_G))]


def local_form(step, before, lr, wd, number):
    """Existing own-gradient native-float32-scalar FP64 local update form."""
    b1, b2, eps = [ctypes.c_float(value).value for value in (.9, .999, 1e-8)]
    native_lr, native_wd = ctypes.c_float(lr).value, ctypes.c_float(wd).value
    groups = {}
    for i, (name, (p, g, m, v)) in enumerate(zip(P, before)):
        actual, am, av = getattr(step.layer, name), step.moments[2 * i], step.moments[2 * i + 1]
        row = dict(finite=False, count=p.numel(), conditioned=0, gradient_max=None, weight_max_abs=None,
                   zero_lr_exact=None, moment_m_abs=None, moment_m_scale=None, moment_v_abs=None, moment_v_scale=None)
        if p.numel() and all(bool(torch.isfinite(value).all()) for value in (p, g, m, v, actual, am, av)):
            pd, gd, md, vd = [value.double() for value in (p, g, m, v)]
            nm, nv = b1 * md + (1 - b1) * gd, b2 * vd + (1 - b2) * gd * gd
            expected = pd * (1 - native_lr * native_wd) - (native_lr / (1 - b1 ** number)) * nm / (nv.sqrt() / math.sqrt(1 - b2 ** number) + eps)
            maximum = float(gd.abs().max()) if gd.numel() else 0.
            mask = gd.abs() >= .01 * maximum; count = int(mask.sum()) if maximum > 0 else 0
            row.update(finite=all(bool(torch.isfinite(value).all()) for value in (expected, nm, nv)),
                conditioned=count, gradient_max=maximum,
                weight_max_abs=float((actual.double()[mask] - expected[mask]).abs().max()) if count else None,
                moment_m_abs=float((am.double() - nm).abs().max()), moment_m_scale=float(nm.abs().max()),
                moment_v_abs=float((av.double() - nv).abs().max()), moment_v_scale=float(nv.abs().max()))
        groups[name] = row
    return groups


def groups_good(groups, lr):
    if set(groups) != set(P) or not math.isfinite(lr) or lr <= 0: return False
    for row in groups.values():
        if row['finite'] is not True or not 0 < row['conditioned'] <= row['count'] or row['gradient_max'] <= 0: return False
        numbers = [value for name, value in row.items() if name not in ('finite', 'zero_lr_exact')]
        if not all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in numbers): return False
        if row['zero_lr_exact'] is not None or row['weight_max_abs'] / lr > .01: return False
        if any(row['moment_' + name + '_abs'] / max(row['moment_' + name + '_scale'], 1e-30) > 1e-4 for name in ('m', 'v')): return False
    return True


def aliases(module):
    step = raw_step(module)
    return all(getattr(module, name).data_ptr() == getattr(step.layer, name).data_ptr() for name in P)


def quantities(module, x, y, need, *, native_gradients=False):
    step = raw_step(module)
    actual = dict(y=y.detach(), z=step.z.T, h=step.act.T)
    actual.update({name: getattr(step, native) if native_gradients else getattr(module, parameter).grad
                   for name, native, parameter in zip(G, NATIVE_G, P)})
    if need: actual['grad_x'] = x.grad
    return actual


class APIState:
    """One layer's persistent pools and existing FP64 references, released at close."""

    def __init__(self, layer, training, inference, step_type, relative, minimum):
        self.layer, self.kernels, self.Step, self.relative, self.minimum = layer, (training, inference), step_type, relative, minimum
        self.selected = {minimum, 13, 128}; self.cache, self.rows = {}, []
        self.bindings = {}
        for name, item, filename in (('step', step_type, 'step.py'), ('module', HFFNTrainFFN, 'module.py'),
                                     ('contexts', HFFNContextFFN, 'contexts.py')):
            actual = Path(sys.modules[item.__module__].__file__).resolve()
            if actual != (COMMON / filename).resolve(): raise RuntimeError('API check requires the shared ' + name)
            self.bindings[name] = str(actual.relative_to(ROOT))
        if Path(checkpoint.__file__).resolve() != (COMMON / 'checkpoint.py').resolve(): raise RuntimeError('API checkpoint import is not the shared one')
        self.bindings['checkpoint'] = str(Path(checkpoint.__file__).resolve().relative_to(ROOT))
        self.module_events, self.control_events = [], []
        mk = tuple(Trace(kernel, self.module_events) for kernel in self.kernels)
        sk = tuple(Trace(kernel, self.control_events) for kernel in self.kernels)
        self.module = HFFNTrainFFN(layer, RESERVATION, n_min=minimum, kernels=mk)
        self.control = step_type(layer, RESERVATION, *sk, n_min=minimum)
        self.module_ptrs, self.control_ptrs = ownership(self.module.step), ownership(self.control)
        self.meta = support(self.control)

    def errors(self, actual, reference):
        values = {name: self.relative(value, reference[name]) for name, value in actual.items()}
        return {name: value if math.isfinite(value) else None for name, value in values.items()}

    @staticmethod
    def errors_good(errors):
        return bool(errors) and all(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1e-4 for value in errors.values())

    def direct(self, n, x, gy, need):
        step = self.control; step.resize(n)
        gx = (step._reserved['gx_nt'][0], step._reserved['gx_t'][0], step._gx_local)
        if not need:
            for i, value in enumerate(gx): value.fill_(.25 * (i + 1))
        before = [value.clone() for value in gx] if not need else []
        start = len(self.control_events)
        step.forward(x); step.backward(gy, input_grad=need)
        result = step.results()
        if not need: result.pop('grad_x')
        checks = dict(gx_untouched=need or all(bits(a, b) for a, b in zip(before, gx)),
            native_trace=self.control_events[start:] == expected_trace(n, need),
            storage=ownership(step) == self.control_ptrs, support=supports(step, self.meta),
            finite=all(value.numel() > 0 and bool(torch.isfinite(value).all()) for value in result.values()))
        return result, checks, self.control_events[start:]

    def module_case(self, n, x, gy, reference, need, baseline):
        module = self.module; module.zero_grad(set_to_none=True)
        xx = x.detach().clone().requires_grad_(need); start = len(self.module_events)
        y = module(xx); y.backward(gy)
        actual = quantities(module, xx, y, need)
        errors = self.errors(actual, reference)
        owned = y.data_ptr() != module.step.y_nt.data_ptr() and all(
            getattr(module, p).grad.data_ptr() != getattr(module.step, g).data_ptr() for p, g in zip(P, NATIVE_G))
        if need: owned &= xx.grad.data_ptr() != module.step.gx_nt.data_ptr()
        checks = dict(exact_step=all(bits(value, baseline[name]) for name, value in actual.items()),
            finite=all(value.numel() > 0 and bool(torch.isfinite(value).all()) for value in actual.values()),
            input_request=(xx.grad is not None) == need, owned_autograd=owned,
            storage=ownership(module.step) == self.module_ptrs, support=supports(module.step, self.meta),
            aliases=aliases(module), native_trace=self.module_events[start:] == expected_trace(n, need))
        row = dict(needs_x=need, checks=checks, errors=errors, native_trace=self.module_events[start:],
                   passed=all(checks.values()) and self.errors_good(errors))
        return row, y

    def observe(self, layer, x, gy, reference, n):
        if n not in self.selected: return
        if layer is not self.layer or n in self.cache: raise ValueError('wrong or repeated API layer/N')
        baselines, cases, directs = {}, [], []
        for need in (True, False):
            baseline, checks, native_trace = self.direct(n, x, gy, need)
            baselines[need] = baseline; directs.append(dict(needs_x=need, checks=checks, native_trace=native_trace, passed=all(checks.values())))
            row, _ = self.module_case(n, x, gy, reference, need, baseline)
            cases.append(row)
        small_equal = all(bits(baselines[True][name], baselines[False][name]) for name in G) if n < 32 else None
        # Store the original reference object, not another FP64 oracle/copy.
        self.cache[n] = dict(x=x, gy=gy, reference=reference, baseline=baselines)
        checks = dict(reservation=self.module.step.n_max == RESERVATION and self.module.step.n == n,
            same_route_parity=all(row['passed'] for row in cases), direct_step=all(row['passed'] for row in directs))
        if n < 32: checks['small_gradient_exact'] = small_equal
        self.rows.append(dict(api='module', n=n, checks=checks, cases=cases, direct=directs,
            full_weights_exact=small_equal, bindings=self.bindings,
            passed=all(checks.values())))

    def checkpoint_row(self):
        module = self.module; saved = checkpoint.export_checkpoint(module)
        restored = checkpoint.from_checkpoint(saved, name='api-roundtrip')
        legacy = checkpoint.from_legacy_state_dict(saved['state_dict'], name='api-legacy')
        original_pointers = {value.data_ptr() for value in (*saved['state_dict'].values(), saved['groups'],
            *[value for value in vars(module.step.layer).values() if isinstance(value, torch.Tensor)]) if value.numel()}
        loaded_owned = all(value.data_ptr() not in original_pointers for value in vars(restored).values()
                           if isinstance(value, torch.Tensor) and value.numel())
        # Close the persistent pool's nonmonotone 128 -> 13 -> minimum -> 128 cycle.
        large = self.cache[128]
        cycle, _ = self.module_case(128, large['x'], large['gy'], large['reference'], True, large['baseline'][True])
        live = HFFNTrainFFN(restored, RESERVATION, n_min=self.minimum, kernels=self.kernels, lr=.001, weight_decay=0.)
        optimizer = torch.optim.SGD(live.parameters(), lr=.001)
        pointers, meta = ownership(live.step), support(live.step)
        with torch.no_grad():
            for i, moment in enumerate(live.step.moments): moment.fill_((i + 1) * .001)
            for name in P: getattr(live, name).add_(.1)
        live.step.steps = 7
        before_moments = [value.clone() for value in live.step.moments]
        optimizer_parameters = list(optimizer.param_groups[0]['params'])
        checkpoint.copy_parameters(live, saved)
        minimum = self.cache[self.minimum]
        legacy_step = self.Step(legacy, RESERVATION, *self.kernels, n_min=self.minimum)
        legacy_step.resize(self.minimum); legacy_step.forward(minimum['x'])
        legacy_errors = self.errors({'y': legacy_step.y_nt}, minimum['reference'])
        checks = dict(exact_roundtrip=layer_equal(module.step.layer, restored), actual_g4=bits(saved['groups'], module.step.layer.groups),
            owned_roundtrip=loaded_owned, legacy_equation=same_equation(module.step.layer, legacy),
            live_copy=layer_equal(live.step.layer, restored), live_storage=ownership(live.step) == pointers,
            live_aliases=aliases(live), optimizer_bindings=all(a is b for a, b in zip(optimizer_parameters, live.parameters())),
            existing_moments=all(bits(a, b) for a, b in zip(before_moments, live.step.moments)),
            existing_counter=live.step.steps == 7, support=supports(live.step, meta), cycle=cycle['passed'],
            configuration_unchanged=(live.step.lr, live.step.weight_decay) == (.001, 0.))
        source_hash, restored_hash = fingerprint(saved), fingerprint(checkpoint.export_checkpoint(live))
        checks['state_hashes'] = source_hash == restored_hash
        row = dict(api='checkpoint', n=self.minimum, checks=checks, cycle=cycle, legacy_errors=legacy_errors,
            source_sha256=source_hash, restored_sha256=restored_hash,
            optimizer_resume=False, passed=all(checks.values()) and self.errors_good(legacy_errors))
        del optimizer, live, legacy_step, legacy, restored, saved, before_moments
        return row

    def context_row(self, small_need, large_need):
        events = []; traced = tuple(Trace(kernel, events) for kernel in self.kernels)
        module = HFFNContextFFN(self.layer, RESERVATION, contexts=2, n_min=self.minimum,
                                kernels=traced, lr=.001, weight_decay=0.)
        step = raw_step(module); pointers, meta = ownership(step), support(step)
        slot_pointers = [[value.data_ptr() for value in slot['buffers'].values()] for slot in module._slots]
        leases, inputs, outputs = [], {}, {}
        requests = {13: small_need, 128: large_need}
        module.zero_grad(set_to_none=True)
        try:
            for n in (13, 128):
                lease = module.acquire_context(); leases.append(lease)
                xx = self.cache[n]['x'].detach().clone().requires_grad_(requests[n])
                inputs[n] = xx; outputs[n] = module(xx, context=lease)
            ordered = None; backward_rows = []
            for n in (128, 13):
                need = requests[n]; data = self.cache[n]; baseline = data['baseline'][need]
                outputs[n].backward(data['gy'])
                native = quantities(module, inputs[n], outputs[n], need, native_gradients=True)
                errors = self.errors(native, data['reference'])
                gradients = [baseline[name] for name in G]
                ordered = [value.clone() for value in gradients] if ordered is None else [a.add_(b) for a, b in zip(ordered, gradients)]
                checks = dict(exact_step=all(bits(value, baseline[name]) for name, value in native.items()),
                    accumulated=all(bits(getattr(module, name).grad, expected) for name, expected in zip(P, ordered)),
                    accumulated_finite=all(bool(torch.isfinite(getattr(module, name).grad).all()) for name in P),
                    last_native=all(bits(getattr(step, name), expected) for name, expected in zip(NATIVE_G, gradients)),
                    input_request=(inputs[n].grad is not None) == need,
                    owned_autograd=outputs[n].data_ptr() != step.y_nt.data_ptr() and
                        all(getattr(module, p).grad.data_ptr() != getattr(step, g).data_ptr() for p, g in zip(P, NATIVE_G)) and
                        (not need or inputs[n].grad.data_ptr() != step.gx_nt.data_ptr()),
                    finite=all(value.numel() > 0 and bool(torch.isfinite(value).all()) for value in native.values()),
                    storage=ownership(step) == pointers, support=supports(step, meta), aliases=aliases(module))
                backward_rows.append(dict(n=n, needs_x=need, checks=checks, errors=errors,
                    passed=all(checks.values()) and self.errors_good(errors)))
            distinct = any(not bits(getattr(module, p).grad, getattr(step, g)) for p, g in zip(P, NATIVE_G))
        finally:
            for lease in reversed(leases): lease.release()
        before = snap(step); module.step.update(); groups = local_form(step, before, .001, 0., 1)
        routes = [dict(name='forward_dual_slots_nt', n=n, scalars=None) for n in (13, 128)] + [
            expected_trace(n, requests[n])[1] for n in (128, 13)] + [adamw_trace(.001, 0., 1)]
        shapes = {'x_t': self.layer.d_model * RESERVATION, 'z': self.layer.d_ff * RESERVATION, 'act': self.layer.d_ff * RESERVATION}
        compact_pointers = {getattr(step.layer, name).data_ptr() for name in P} | {value.data_ptr() for value in step.moments} | {
            getattr(step, name).data_ptr() for name in NATIVE_G} | {value[0].data_ptr() for value in step._reserved.values()}
        slot_flat = [value for slot in module._slots for value in slot['buffers'].values()]
        checks = dict(backwards=all(row['passed'] for row in backward_rows),
            last_gradient_distinct_from_accumulated=distinct, native_trace=events == routes,
            storage=ownership(step) == pointers, support=supports(step, meta), aliases=aliases(module),
            released=all(not slot['leased'] for slot in module._slots), counter=step.steps == 1,
            slots_activation_only=all(set(slot['buffers']) == set(shapes) and all(value.numel() == shapes[name] and
                value.dtype == torch.float32 and value.device.type == 'cpu' and value.is_contiguous()
                for name, value in slot['buffers'].items()) for slot in module._slots),
            slots_separate_state=len({value.data_ptr() for value in slot_flat}) == 6 and
                all(value.data_ptr() not in compact_pointers for value in slot_flat),
            owned_outputs=outputs[13].data_ptr() != outputs[128].data_ptr(),
            slots_storage=[[value.data_ptr() for value in slot['buffers'].values()] for slot in module._slots] == slot_pointers)
        row = dict(api='contexts', small_needs_x=small_need, large_needs_x=large_need, backward_order=[128, 13],
            checks=checks, backwards=backward_rows, native_trace=events,
            update=dict(n=13, lr=.001, weight_decay=0., update=1, groups=groups, gradient_source='last_native_backward'),
            passed=all(checks.values()) and groups_good(groups, .001))
        del module, step, inputs, outputs, ordered, before
        return row

    def configuration_row(self):
        # Default scalar arguments are observed at the existing Step binding.
        data = self.cache[128]; start = len(self.control_events)
        self.control.resize(128); self.control.forward(data['x']); self.control.backward(data['gy'], input_grad=True)
        errors = self.errors(self.control.results(), data['reference'])
        before = snap(self.control); self.control.update()
        default_groups = local_form(self.control, before, 1e-4, .01, 1)
        default_checks = dict(configuration=(self.control.lr, self.control.weight_decay) == (1e-4, .01),
            native_trace=self.control_events[start:] == expected_trace(128, True) + [adamw_trace(1e-4, .01, 1)],
            storage=ownership(self.control) == self.control_ptrs, support=supports(self.control, self.meta), counter=self.control.steps == 1)
        default = dict(configuration='default', n=128, update=1, lr=1e-4, weight_decay=.01,
            checks=default_checks, errors=errors, groups=default_groups, native_trace=self.control_events[start:],
            passed=all(default_checks.values()) and self.errors_good(errors) and groups_good(default_groups, 1e-4))
        del before
        events = []; traced = tuple(Trace(kernel, events) for kernel in self.kernels)
        module = HFFNTrainFFN(self.layer, RESERVATION, n_min=self.minimum, kernels=traced, lr=.001, weight_decay=0.)
        copied = HFFNTrainFFN(checkpoint.from_checkpoint(checkpoint.export_checkpoint(module)), RESERVATION,
            n_min=self.minimum, kernels=self.kernels, lr=module.step.lr, weight_decay=module.step.weight_decay)
        clone = dict(configuration=(copied.step.lr, copied.step.weight_decay) == (.001, 0.),
            exact_state=layer_equal(module.step.layer, copied.step.layer),
            owned=all(getattr(module, name).data_ptr() != getattr(copied, name).data_ptr() for name in P),
            no_optimizer_resume=copied.step.steps == 0 and all(bool((value == 0).all()) for value in copied.step.moments))
        del copied
        pointers, meta = ownership(module.step), support(module.step)
        updates, previous_state = [], None
        for number, (n, need) in enumerate(((13, True), (128, False)), 1):
            data = self.cache[n]; start = len(events); module.zero_grad(set_to_none=True)
            xx = data['x'].detach().clone().requires_grad_(need)
            y = module(xx); y.backward(data['gy'])  # second backward uses its actual updated weights
            actual = quantities(module, xx, y, need)
            initial_errors = self.errors(actual, data['reference']) if number == 1 else None
            before = snap(module.step)
            carry = any(bool((value != 0).any()) for group in before for value in group[2:])
            before_state = fingerprint({name: [values[0], values[2], values[3]] for name, values in zip(P, before)})
            module.step.update(); groups = local_form(module.step, before, .001, 0., number)
            after_state = fingerprint({name: [getattr(module, name), module.step.moments[2 * i], module.step.moments[2 * i + 1]] for i, name in enumerate(P)})
            checks = dict(configuration=(module.step.lr, module.step.weight_decay) == (.001, 0.),
                native_trace=events[start:] == expected_trace(n, need) + [adamw_trace(.001, 0., number)],
                counter=module.step.steps == number, carry=(carry == (number > 1)),
                carry_state=(previous_state is None if number == 1 else before_state == previous_state),
                storage=ownership(module.step) == pointers, support=supports(module.step, meta), aliases=aliases(module),
                finite_gradient=all(value.numel() > 0 and bool(torch.isfinite(value).all()) for value in actual.values()))
            updates.append(dict(update=number, n=n, needs_x=need, lr=.001, weight_decay=0., checks=checks,
                errors=initial_errors, gradient_accuracy='initial_cached_fp64' if number == 1 else 'local_mechanics_only_after_update',
                before_state_sha256=before_state, previous_after_state_sha256=previous_state, after_state_sha256=after_state,
                groups=groups, native_trace=events[start:],
                passed=all(checks.values()) and groups_good(groups, .001) and (number != 1 or self.errors_good(initial_errors))))
            previous_state = after_state
        checks = dict(default=default['passed'], constructor_clone=all(clone.values()), configured_updates=all(row['passed'] for row in updates))
        row = dict(api='configuration', checks=checks, default=default, clone=clone, updates=updates,
            clone_optimizer_resume=False, passed=all(checks.values()))
        del module, before
        return row

    def finalize(self):
        if set(self.cache) != self.selected: raise RuntimeError('API coverage missing selected N')
        self.rows.append(self.checkpoint_row())
        self.rows.append(self.context_row(False, True))
        self.rows.append(self.context_row(True, False))
        self.rows.append(self.configuration_row())
        if len(self.rows) != 7: raise RuntimeError('expected exactly seven API observations per layer')
        return list(self.rows)

    def close(self):
        self.cache.clear(); self.rows.clear(); self.meta.clear()
        self.module_events.clear(); self.control_events.clear()
        self.module = self.control = self.layer = None
