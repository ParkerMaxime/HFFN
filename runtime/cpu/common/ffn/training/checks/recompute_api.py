"""Independent stdlib reader for the seven API rows per layer.

This validates API evidence only. Source admission, the numerical gates, callers,
refusal suites and reliable run completion belong to the outer batch reader.
No runtime or harness code is imported here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'runtime' / 'common' / 'pythia' / 'hffn_pythia').is_dir())
COMMON = ROOT / 'runtime/cpu/common/ffn/training/hffn_ffn_training'
PARAMETERS = ('up_val', 'down_val', 'b_up', 'b_down')
QUANTITIES = {'y', 'z', 'h', 'grad_x', 'grad_up', 'grad_down', 'grad_b_up', 'grad_b_down'}
MODULE_CHECKS = {'exact_step', 'finite', 'input_request', 'owned_autograd', 'storage', 'support', 'aliases', 'native_trace'}
DIRECT_CHECKS = {'gx_untouched', 'native_trace', 'storage', 'support', 'finite'}
CHECKPOINT_CHECKS = {'exact_roundtrip', 'actual_g4', 'owned_roundtrip', 'legacy_equation', 'live_copy',
    'live_storage', 'live_aliases', 'optimizer_bindings', 'existing_moments', 'existing_counter',
    'support', 'cycle', 'configuration_unchanged', 'state_hashes'}
BACKWARD_CHECKS = {'exact_step', 'accumulated', 'accumulated_finite', 'last_native', 'input_request',
    'owned_autograd', 'finite', 'storage', 'support', 'aliases'}
CONTEXT_CHECKS = {'backwards', 'last_gradient_distinct_from_accumulated', 'native_trace', 'storage',
    'support', 'aliases', 'released', 'counter', 'slots_activation_only', 'slots_separate_state',
    'owned_outputs', 'slots_storage'}
UPDATE_CHECKS = {'configuration', 'native_trace', 'counter', 'carry', 'carry_state', 'storage',
    'support', 'aliases', 'finite_gradient'}
DEFAULT_CHECKS = {'configuration', 'native_trace', 'storage', 'support', 'counter'}
CLONE_CHECKS = {'configuration', 'exact_state', 'owned', 'no_optimizer_resume'}
GROUP_KEYS = {'finite', 'count', 'conditioned', 'gradient_max', 'weight_max_abs', 'zero_lr_exact',
    'moment_m_abs', 'moment_m_scale', 'moment_v_abs', 'moment_v_scale'}


def numeric(value):
    if type(value) not in (int, float): return False
    try: return math.isfinite(value)
    except OverflowError: return False


def native_trace(n, need):
    route = ('backward_dual_nt' if n >= 32 else 'backward_nt') if need else 'backward_weights_nt'
    return [dict(name='forward_dual_slots_nt', n=n, scalars=None), dict(name=route, n=n, scalars=None)]


def adamw(lr, wd, number):
    return dict(name='adamw', n=None, scalars=[lr, .9, .999, 1e-8, wd, number])


class Reader:
    def __init__(self):
        self.issues = []
        self.stats = dict(records=0, layers=0, updates=0, groups=0, oracle_components=0,
                          max_oracle=0., max_weight_lr=0., max_moment_rel=0.)

    def require(self, predicate, path, message):
        if not predicate: self.issues.append(path + ': ' + message)
        return bool(predicate)

    def mapping(self, value, keys, path):
        if not self.require(type(value) is dict, path, 'expected object'): return {}
        self.require(set(value) == set(keys), path, 'keys differ; expected ' + ','.join(sorted(keys)))
        return value

    def sequence(self, value, size, path):
        if not self.require(type(value) is list, path, 'expected list'): return []
        self.require(len(value) == size, path, 'expected ' + str(size) + ' entries')
        return value

    def literal(self, value, expected, path):
        # bool must never pass as 0/1, and native update counters remain integers.
        valid = (value is expected if expected is None or type(expected) is bool else
                 type(value) is int and value == expected if type(expected) is int else
                 numeric(value) and value == expected if type(expected) is float else
                 type(value) is type(expected) and value == expected)
        return self.require(valid, path, 'expected ' + repr(expected))

    def checks(self, value, keys, path):
        item = self.mapping(value, keys, path)
        for key in keys: self.literal(item.get(key), True, path + '.' + key)

    def passed(self, value, start, path):
        # Reconstruct the predicate from all nested evidence inspected since start.
        valid = len(self.issues) == start
        self.require(type(value) is bool and value == valid, path, 'passed disagrees with recomputed predicate')
        self.require(valid, path, 'criterion failed')

    def errors(self, value, need, path, *, only_y=False):
        keys = {'y'} if only_y else QUANTITIES if need else QUANTITIES - {'grad_x'}
        item = self.mapping(value, keys, path)
        for key in keys:
            number = item.get(key)
            if self.require(numeric(number) and 0 <= number <= 1e-4, path + '.' + key, 'oracle relative error outside [0,1e-4]'):
                self.stats['oracle_components'] += 1
                self.stats['max_oracle'] = max(self.stats['max_oracle'], number)

    def trace(self, value, expected, path):
        events = self.sequence(value, len(expected), path)
        for i, (actual, wanted) in enumerate(zip(events, expected)):
            here = path + '[' + str(i) + ']'
            actual = self.mapping(actual, {'name', 'n', 'scalars'}, here)
            self.literal(actual.get('name'), wanted['name'], here + '.name')
            self.literal(actual.get('n'), wanted['n'], here + '.n')
            if wanted['scalars'] is None:
                self.literal(actual.get('scalars'), None, here + '.scalars')
            else:
                scalars = self.sequence(actual.get('scalars'), 6, here + '.scalars')
                for j, (a, b) in enumerate(zip(scalars, wanted['scalars'])):
                    self.literal(a, b, here + '.scalars[' + str(j) + ']')

    def hash(self, value, path):
        return self.require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, path, 'expected lowercase SHA256')

    def groups(self, value, lr, path):
        data = self.mapping(value, PARAMETERS, path)
        for name in PARAMETERS:
            here = path + '.' + name
            group = self.mapping(data.get(name), GROUP_KEYS, here)
            self.literal(group.get('finite'), True, here + '.finite')
            count, conditioned = group.get('count'), group.get('conditioned')
            self.require(type(count) is int and count > 0, here + '.count', 'group must be nonempty')
            self.require(type(conditioned) is int and type(count) is int and 0 < conditioned <= count,
                         here + '.conditioned', 'need nonempty conditioned subset')
            for key in GROUP_KEYS - {'finite', 'count', 'conditioned', 'zero_lr_exact'}:
                number = group.get(key)
                self.require(numeric(number) and number >= 0, here + '.' + key, 'expected finite nonnegative number')
            gmax = group.get('gradient_max')
            self.require(numeric(gmax) and gmax > 0, here + '.gradient_max', 'need nonzero gradient scale')
            self.literal(group.get('zero_lr_exact'), None, here + '.zero_lr_exact')
            weight = group.get('weight_max_abs')
            if numeric(weight) and weight >= 0:
                ratio = weight / lr
                self.require(math.isfinite(ratio) and ratio <= .01, here + '.weight_max_abs', 'conditioned weight error/lr exceeds .01')
                if math.isfinite(ratio): self.stats['max_weight_lr'] = max(self.stats['max_weight_lr'], ratio)
            for moment in ('m', 'v'):
                error, scale = group.get('moment_' + moment + '_abs'), group.get('moment_' + moment + '_scale')
                if numeric(error) and error >= 0 and numeric(scale) and scale >= 0:
                    ratio = error / max(scale, 1e-30)
                    self.require(math.isfinite(ratio) and ratio <= 1e-4, here + '.moment_' + moment, 'moment relative error exceeds 1e-4')
                    if math.isfinite(ratio): self.stats['max_moment_rel'] = max(self.stats['max_moment_rel'], ratio)
            self.stats['groups'] += 1
        self.stats['updates'] += 1

    def case(self, value, n, need, path):
        start = len(self.issues)
        row = self.mapping(value, {'needs_x', 'checks', 'errors', 'native_trace', 'passed'}, path)
        self.literal(row.get('needs_x'), need, path + '.needs_x')
        self.checks(row.get('checks'), MODULE_CHECKS, path + '.checks')
        self.errors(row.get('errors'), need, path + '.errors')
        self.trace(row.get('native_trace'), native_trace(n, need), path + '.native_trace')
        self.passed(row.get('passed'), start, path + '.passed')

    def module(self, row, n, path):
        start = len(self.issues)
        row = self.mapping(row, {'export', 'layer', 'api', 'n', 'checks', 'cases', 'direct',
                                 'full_weights_exact', 'bindings', 'passed'}, path)
        keys = {'reservation', 'same_route_parity', 'direct_step'} | ({'small_gradient_exact'} if n < 32 else set())
        self.checks(row.get('checks'), keys, path + '.checks')
        self.literal(row.get('n'), n, path + '.n')
        self.literal(row.get('full_weights_exact'), True if n < 32 else None, path + '.full_weights_exact')
        bindings = self.mapping(row.get('bindings'), {'step', 'module', 'contexts', 'checkpoint'}, path + '.bindings')
        for key in ('step', 'module', 'contexts', 'checkpoint'):
            self.literal(bindings.get(key), str((COMMON / (key + '.py')).relative_to(ROOT)), path + '.bindings.' + key)
        for i, (case, need) in enumerate(zip(self.sequence(row.get('cases'), 2, path + '.cases'), (True, False))):
            self.case(case, n, need, path + '.cases[' + str(i) + ']')
        for i, (direct, need) in enumerate(zip(self.sequence(row.get('direct'), 2, path + '.direct'), (True, False))):
            here = path + '.direct[' + str(i) + ']'; substart = len(self.issues)
            direct = self.mapping(direct, {'needs_x', 'checks', 'native_trace', 'passed'}, here)
            self.literal(direct.get('needs_x'), need, here + '.needs_x')
            self.checks(direct.get('checks'), DIRECT_CHECKS, here + '.checks')
            self.trace(direct.get('native_trace'), native_trace(n, need), here + '.native_trace')
            self.passed(direct.get('passed'), substart, here + '.passed')
        self.passed(row.get('passed'), start, path + '.passed')

    def checkpoint(self, row, minimum, path):
        start = len(self.issues)
        row = self.mapping(row, {'export', 'layer', 'api', 'n', 'checks', 'cycle', 'legacy_errors',
                                 'source_sha256', 'restored_sha256', 'optimizer_resume', 'passed'}, path)
        self.literal(row.get('n'), minimum, path + '.n')
        self.checks(row.get('checks'), CHECKPOINT_CHECKS, path + '.checks')
        self.case(row.get('cycle'), 128, True, path + '.cycle')
        self.errors(row.get('legacy_errors'), False, path + '.legacy_errors', only_y=True)
        for key in ('source_sha256', 'restored_sha256'): self.hash(row.get(key), path + '.' + key)
        self.require(row.get('source_sha256') == row.get('restored_sha256'), path, 'checkpoint state hashes differ')
        self.literal(row.get('optimizer_resume'), False, path + '.optimizer_resume')
        self.passed(row.get('passed'), start, path + '.passed')

    def contexts(self, row, small, large, path):
        start = len(self.issues)
        row = self.mapping(row, {'export', 'layer', 'api', 'small_needs_x', 'large_needs_x', 'backward_order',
                                 'checks', 'backwards', 'native_trace', 'update', 'passed'}, path)
        self.literal(row.get('small_needs_x'), small, path + '.small_needs_x')
        self.literal(row.get('large_needs_x'), large, path + '.large_needs_x')
        order = self.sequence(row.get('backward_order'), 2, path + '.backward_order')
        for i, (a, b) in enumerate(zip(order, (128, 13))): self.literal(a, b, path + '.backward_order[' + str(i) + ']')
        self.checks(row.get('checks'), CONTEXT_CHECKS, path + '.checks')
        for i, (backward, (n, need)) in enumerate(zip(self.sequence(row.get('backwards'), 2, path + '.backwards'), ((128, large), (13, small)))):
            here = path + '.backwards[' + str(i) + ']'; substart = len(self.issues)
            backward = self.mapping(backward, {'n', 'needs_x', 'checks', 'errors', 'passed'}, here)
            self.literal(backward.get('n'), n, here + '.n'); self.literal(backward.get('needs_x'), need, here + '.needs_x')
            self.checks(backward.get('checks'), BACKWARD_CHECKS, here + '.checks')
            self.errors(backward.get('errors'), need, here + '.errors')
            self.passed(backward.get('passed'), substart, here + '.passed')
        trace = [dict(name='forward_dual_slots_nt', n=n, scalars=None) for n in (13, 128)]
        trace += [native_trace(128, large)[1], native_trace(13, small)[1], adamw(.001, 0., 1)]
        self.trace(row.get('native_trace'), trace, path + '.native_trace')
        update = self.mapping(row.get('update'), {'n', 'lr', 'weight_decay', 'update', 'groups', 'gradient_source'}, path + '.update')
        for key, expected in dict(n=13, lr=.001, weight_decay=0., update=1, gradient_source='last_native_backward').items():
            self.literal(update.get(key), expected, path + '.update.' + key)
        self.groups(update.get('groups'), .001, path + '.update.groups')
        self.passed(row.get('passed'), start, path + '.passed')

    def configuration(self, row, path):
        start = len(self.issues)
        row = self.mapping(row, {'export', 'layer', 'api', 'checks', 'default', 'clone', 'updates', 'clone_optimizer_resume', 'passed'}, path)
        self.checks(row.get('checks'), {'default', 'constructor_clone', 'configured_updates'}, path + '.checks')
        self.literal(row.get('clone_optimizer_resume'), False, path + '.clone_optimizer_resume')
        self.checks(row.get('clone'), CLONE_CHECKS, path + '.clone')
        here = path + '.default'; substart = len(self.issues)
        default = self.mapping(row.get('default'), {'configuration', 'n', 'update', 'lr', 'weight_decay',
                               'checks', 'errors', 'groups', 'native_trace', 'passed'}, here)
        for key, expected in dict(configuration='default', n=128, update=1, lr=1e-4, weight_decay=.01).items():
            self.literal(default.get(key), expected, here + '.' + key)
        self.checks(default.get('checks'), DEFAULT_CHECKS, here + '.checks')
        self.errors(default.get('errors'), True, here + '.errors')
        self.groups(default.get('groups'), 1e-4, here + '.groups')
        self.trace(default.get('native_trace'), native_trace(128, True) + [adamw(1e-4, .01, 1)], here + '.native_trace')
        self.passed(default.get('passed'), substart, here + '.passed')
        previous = None
        for i, (update, (n, need)) in enumerate(zip(self.sequence(row.get('updates'), 2, path + '.updates'), ((13, True), (128, False))), 1):
            here = path + '.updates[' + str(i - 1) + ']'; substart = len(self.issues)
            update = self.mapping(update, {'update', 'n', 'needs_x', 'lr', 'weight_decay', 'checks', 'errors',
                'gradient_accuracy', 'before_state_sha256', 'previous_after_state_sha256', 'after_state_sha256',
                'groups', 'native_trace', 'passed'}, here)
            for key, expected in dict(update=i, n=n, needs_x=need, lr=.001, weight_decay=0.).items():
                self.literal(update.get(key), expected, here + '.' + key)
            self.checks(update.get('checks'), UPDATE_CHECKS, here + '.checks')
            self.literal(update.get('gradient_accuracy'), 'initial_cached_fp64' if i == 1 else 'local_mechanics_only_after_update', here + '.gradient_accuracy')
            if i == 1: self.errors(update.get('errors'), True, here + '.errors')
            else: self.literal(update.get('errors'), None, here + '.errors')
            for key in ('before_state_sha256', 'after_state_sha256'): self.hash(update.get(key), here + '.' + key)
            if i == 1: self.literal(update.get('previous_after_state_sha256'), None, here + '.previous_after_state_sha256')
            else:
                self.hash(update.get('previous_after_state_sha256'), here + '.previous_after_state_sha256')
                self.require(update.get('before_state_sha256') == previous == update.get('previous_after_state_sha256'), here, 'parameter/eight-moment state did not carry')
            self.require(update.get('before_state_sha256') != update.get('after_state_sha256'), here, 'update did not change parameter/moment state')
            previous = update.get('after_state_sha256')
            self.groups(update.get('groups'), .001, here + '.groups')
            self.trace(update.get('native_trace'), native_trace(n, need) + [adamw(.001, 0., i)], here + '.native_trace')
            self.passed(update.get('passed'), substart, here + '.passed')
        self.passed(row.get('passed'), start, path + '.passed')


def inspect(api_rows, expected_layers, n_min):
    """Return API-only verdict, issues and recomputed stats; never import runtime.

    expected_layers is the outer reader's pinned {export: [layer_name, ...]}.
    n_min is the hardware minimum, exactly 1 (M1) or 2 (x86).
    """
    reader = Reader()
    if not reader.require(type(n_min) is int and n_min in (1, 2), 'n_min', 'expected hardware minimum 1 or 2'):
        return dict(passed=False, issues=reader.issues, stats=reader.stats)
    if not reader.require(type(expected_layers) is dict and bool(expected_layers), 'expected_layers', 'expected nonempty export/layer mapping'):
        return dict(passed=False, issues=reader.issues, stats=reader.stats)
    layers = []
    for export, names in expected_layers.items():
        if not reader.require(type(export) is str and bool(export) and type(names) is list and bool(names), 'expected_layers', 'nonempty export and layer list required'): continue
        valid = all(type(name) is str and bool(name) for name in names)
        reader.require(valid and len(names) == len(set(names)) if valid else False, 'expected_layers.' + export, 'layer names must be unique nonempty strings')
        if valid: layers.extend((export, name) for name in names)
    expected = Counter()
    for export, layer in layers:
        for n in (n_min, 13, 128): expected[(export, layer, 'module', n)] += 1
        expected[(export, layer, 'checkpoint', n_min)] += 1
        for small, large in ((False, True), (True, False)): expected[(export, layer, 'contexts', small, large)] += 1
        expected[(export, layer, 'configuration')] += 1
    if not reader.require(type(api_rows) is list, 'api_rows', 'expected list'):
        return dict(passed=False, issues=reader.issues, stats=reader.stats)
    actual = Counter()
    for i, row in enumerate(api_rows):
        path = 'api_rows[' + str(i) + ']'
        if not reader.require(type(row) is dict, path, 'expected object'): continue
        export, layer, api = row.get('export'), row.get('layer'), row.get('api')
        if not reader.require(type(export) is str and type(layer) is str and type(api) is str, path, 'invalid export/layer/api identity'): continue
        identity = (export, layer, api)
        if api in ('module', 'checkpoint'):
            n = row.get('n')
            if not reader.require(type(n) is int, path + '.n', 'expected integer'): continue
            identity += (n,)
        elif api == 'contexts':
            small, large = row.get('small_needs_x'), row.get('large_needs_x')
            if not reader.require(type(small) is bool and type(large) is bool, path, 'context requests must be booleans'): continue
            identity += (small, large)
        actual[identity] += 1
        if not reader.require(identity in expected, path, 'unexpected canonical row key'): continue
        if api == 'module': reader.module(row, row['n'], path)
        elif api == 'checkpoint': reader.checkpoint(row, n_min, path)
        elif api == 'contexts': reader.contexts(row, row['small_needs_x'], row['large_needs_x'], path)
        elif api == 'configuration': reader.configuration(row, path)
    reader.require(actual == expected, 'coverage', 'canonical row multiset differs (missing, duplicate or unexpected row)')
    reader.stats['records'], reader.stats['layers'] = len(api_rows), len(layers)
    reader.require(reader.stats['records'] == 7 * len(layers), 'records', 'expected seven API rows per layer')
    reader.require(reader.stats['updates'] == 5 * len(layers), 'updates', 'expected five native local updates per layer')
    reader.require(reader.stats['groups'] == 20 * len(layers), 'groups', 'expected four groups for each of five updates')
    reader.require(reader.stats['oracle_components'] == 100 * len(layers), 'oracle_components', 'expected 100 finite oracle components per layer')
    return dict(passed=not reader.issues, issues=reader.issues, stats=reader.stats)


def corruption_controls(api_rows, expected_layers, n_min):
    """Reject eighteen independent in-memory corruptions of admitted API rows.

    A valid baseline is mandatory, so an already failing list cannot make
    every control appear successful. No runtime, disk fixture or native call
    is needed. Malformed-evidence exceptions count as rejection, but mutation
    setup errors propagate rather than being counted as a successful guard.
    """
    if not inspect(api_rows, expected_layers, n_min)['passed']:
        raise ValueError('API corruption controls require an admitted baseline')

    def first(rows, api): return next(row for row in rows if row['api'] == api)
    def module(rows): return first(rows, 'module')
    def config(rows): return first(rows, 'configuration')
    def group(rows): return config(rows)['default']['groups']['up_val']
    def grad_error(rows, value): module(rows)['cases'][0]['errors']['grad_up'] = value
    def missing_layer(rows):
        selected = rows[0]['export'], rows[0]['layer']
        rows[:] = [row for row in rows if (row['export'], row['layer']) != selected]
    def duplicate(rows): rows.append(deepcopy(module(rows)))
    def extra(rows): module(rows)['unexpected'] = True
    def groups_missing(rows): config(rows)['default']['groups'].pop('b_down')
    def empty(rows): group(rows)['count'] = 0
    def nonfinite(rows): group(rows)['gradient_max'] = float('inf')
    def weight_bound(rows): group(rows)['weight_max_abs'] = config(rows)['default']['lr'] * .02
    def moment_bound(rows):
        item = group(rows)
        item['moment_v_abs'] = max(item['moment_v_scale'], 1e-30) * 2e-4
    def trace(rows): module(rows)['cases'][0]['native_trace'][1]['name'] = 'wrong_backward'
    def scalar(rows): config(rows)['default']['native_trace'][-1]['scalars'][0] = .002
    def counter(rows): config(rows)['updates'][1]['native_trace'][-1]['scalars'][-1] = 1
    def carryhash(rows):
        update = config(rows)['updates'][1]
        update['before_state_sha256'] = ('0' if update['previous_after_state_sha256'] != '0' * 64 else '1') * 64
    def contextaccum(rows): first(rows, 'contexts')['backwards'][1]['checks']['accumulated'] = False
    def checkpoint_g4(rows): first(rows, 'checkpoint')['checks']['actual_g4'] = False
    def bindingpath(rows): module(rows)['bindings']['step'] = 'wrong/step.py'

    mutations = {
        'coverage': missing_layer,
        'duplicate': duplicate,
        'extra': extra,
        'graderror_nan': lambda rows: grad_error(rows, float('nan')),
        'graderror_bool': lambda rows: grad_error(rows, True),
        'graderror_bound': lambda rows: grad_error(rows, 2e-4),
        'groups_missing': groups_missing,
        'empty': empty,
        'nonfinite': nonfinite,
        'weight_bound': weight_bound,
        'moment_bound': moment_bound,
        'trace': trace,
        'scalar': scalar,
        'counter': counter,
        'carryhash': carryhash,
        'contextaccum': contextaccum,
        'checkpoint_g4': checkpoint_g4,
        'bindingpath': bindingpath,
    }
    result = {}
    for name, mutate in mutations.items():
        changed = deepcopy(api_rows)
        mutate(changed)
        try: rejected = inspect(changed, expected_layers, n_min)['passed'] is False
        except Exception: rejected = True
        result[name] = rejected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate2', type=Path, help='JSON containing api_rows, or the API list itself')
    parser.add_argument('--expected', required=True, type=Path, help='Pinned export -> layer-name-list JSON')
    parser.add_argument('--n-min', required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    value = json.loads(args.gate2.read_text())
    rows = value.get('api_rows') if type(value) is dict else value
    result = inspect(rows, json.loads(args.expected.read_text()), args.n_min)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
