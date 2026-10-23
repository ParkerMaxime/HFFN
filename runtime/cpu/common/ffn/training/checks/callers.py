"""Untimed M1 caller checks for the final export batch; no production runtime.

Small generated data/checkpoints stay in memory. The actual CLI,
autograd, native FFN and optimizer calls execute; instrumentation observes
them and never supplies their result.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
PKG = 'runtime.cpu.m1.ffn.training'
PARAMS = ('up_val', 'down_val', 'b_up', 'b_down')
MEMORY = ('tensor_nbytes', 'optimizer_state_nbytes', 'activation_bytes_estimate',
          'reference_memory_profile', 'hffn_runtime_buffer_profile', 'hffn_memory_profile', 'memory_snapshot')


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((str(x.dtype), tuple(x.shape))).encode()); h.update(x.detach().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def finite(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)): return all(finite(x) for x in value)
    return True


def bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.detach().contiguous().view(torch.int32), b.detach().contiguous().view(torch.int32))


def unmeasured(value):
    if not isinstance(value, dict): return True
    for key, item in value.items():
        if (key.endswith('_seconds') or key.endswith('_per_second') or 'speedup' in key or
                key.startswith('memory_') and key != 'memory_accounting_enabled') and item is not None:
            return False
        if isinstance(item, dict) and not unmeasured(item): return False
        if isinstance(item, list) and any(isinstance(x, dict) and not unmeasured(x) for x in item): return False
    return True


class Trace:
    def __init__(self, kernel): self.kernel, self.events = kernel, []
    def __getattr__(self, name):
        target = getattr(self.kernel, name)
        def call(*args):
            self.events.append(dict(name=name, scalars=list(args[-6:]) if name == 'adamw' else None))
            return target(*args)
        return call


@contextmanager
def no_legacy(models):
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == PKG + '.CPU' or fullname.startswith(PKG + '.CPU.'):
                raise RuntimeError('CPU-lineage import reached by the caller')
    def forbidden(*args, **kwargs): raise RuntimeError('CPU-lineage HFFN execution reached by the caller')
    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        if any(name == PKG + '.CPU' or name.startswith(PKG + '.CPU.') for name in sys.modules):
            raise RuntimeError('legacy CPU already imported')
        with patch.object(models.HFFNCompactFFN, '__init__', forbidden), \
                patch.object(models.HFFNCompactFFN, 'forward', forbidden):
            yield
    finally:
        sys.meta_path.remove(finder)


def storage(model):
    step = model.step
    tensors = {name: getattr(model, name) for name in PARAMS}
    tensors.update({'moment/' + str(i): x for i, x in enumerate(step.moments)})
    tensors.update({'reserved/' + name: x[0] for name, x in step._reserved.items()})
    tensors.update({'gradient/' + name: getattr(step, name) for name in ('g_up', 'g_down', 'g_b_up', 'g_b_down', '_gx_local', 'timing')})
    tensors.update({'support/' + name: x for name, x in vars(step.layer).items()
                    if name not in PARAMS and isinstance(x, torch.Tensor)})
    return {name: (x.data_ptr(), str(x.dtype), list(x.shape)) for name, x in tensors.items()}


def case(kind, optimizer, mode, kernels):
    """One actual main call, three training batches, both old/new weight loads."""
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    name = 'train_toy_ffn_hffn_cpu' if kind == 'toy' else 'compare_training_curves'
    entry = importlib.import_module(PKG + '.' + name)
    toy = importlib.import_module(PKG + '.shared_toy')
    models = importlib.import_module(PKG + '.models')
    checkpoint = importlib.import_module(PKG + '.hffn_ffn_training.checkpoint')
    wanted = (ROOT / 'runtime/cpu/m1/ffn/training' / (name + '.py')).resolve()
    if Path(entry.__file__).resolve() != wanted: raise ValueError('the caller of the leaf must be imported')
    gen = torch.Generator().manual_seed(617)
    data = {axis + '_' + split: torch.randn(n, 128, generator=gen)
            for split, n in (('train', 71), ('val', 37), ('test', 38)) for axis in ('x', 'y')}
    original_data = digest(data)
    captured, saves, updates, h_calls, r_calls, handles, wrapped_opts = {}, {}, [], [], [], [], []
    native_ctor, reference_ctor, make_opt, real_save = entry.HFFNToyFFN, entry.SparseMaskedReferenceFFN, entry.build_optimizer, torch.save
    real_evaluate, real_parity = entry.evaluate, entry.parity_snapshot
    evaluations, parity_requests = [], []
    need = kind == 'curve' and mode == 'full'
    native_adamw = kind == 'curve' and optimizer == 'adamw'

    def forbidden(*args, **kwargs): raise RuntimeError('untimed caller read a timer or memory helper')

    class ObservedReference(reference_ctor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); captured['reference'] = self
            def hook(module, inputs, y):
                r_calls.append(dict(n=len(inputs[0]), gradient=y.requires_grad, needs_x=inputs[0].requires_grad))
            handles.append(self.register_forward_hook(hook))

    def checked_update(perform):
        m, trace = captured['model'], captured['trace']
        number = len(updates) + 1; result = perform()
        n = captured['last_n']; events = trace.events[captured['event_start']:]
        route = ('backward_dual_nt' if n >= 32 else 'backward_nt') if need else 'backward_weights_nt'
        expected = [dict(name=route, scalars=None)]
        if native_adamw: expected.append(dict(name='adamw', scalars=[.0007, .9, .999, 1e-8, .03, number]))
        checks = dict(route=events == expected, counter=m.step.steps == (number if native_adamw else 0),
                      request=captured['last_need'] == need, storage=storage(m) == captured['storage'],
                      actual_input_gradient=(captured['last_x'].grad is not None) == need and
                          (not need or bits(captured['last_x'].grad, m.step.gx_nt)),
                      native_gradient_binding=all(getattr(m, p).grad is not None and bits(getattr(m, p).grad, getattr(m.step, g))
                          for p, g in zip(PARAMS, ('g_up', 'g_down', 'g_b_up', 'g_b_down'))),
                      support=digest({key: value for key, value in vars(m.step.layer).items()
                          if key not in PARAMS and isinstance(value, torch.Tensor)}) == captured['support'],
                      aliases=all(getattr(m, key).data_ptr() == getattr(m.step.layer, key).data_ptr() for key in PARAMS),
                      inactive_reference=all(bool((part.weight.detach()[part.mask == 0] == 0).all())
                          for part in (captured['reference'].up, captured['reference'].down)),
                      finite=all(bool(torch.isfinite(p).all()) for p in [*m.parameters(), *m.step.moments]))
        updates.append(dict(update=number, n=n, needs_x=need, native_trace=events, checks=checks, passed=all(checks.values())))
        return result

    def model_factory(*args, **kwargs):
        trace = Trace(kernels[0]); kwargs['kernels'] = (trace, kernels[1])
        model = native_ctor(*args, **kwargs)
        captured.update(model=model, trace=trace, storage=storage(model), support=digest({key: value
            for key, value in vars(model.step.layer).items() if key not in PARAMS and isinstance(value, torch.Tensor)}))
        def hook(module, inputs, y):
            captured.update(last_n=len(inputs[0]), last_need=inputs[0].requires_grad,
                            last_x=inputs[0], event_start=len(trace.events))
            h_calls.append(dict(n=len(inputs[0]), gradient=y.requires_grad, needs_x=inputs[0].requires_grad))
        handles.append(model.register_forward_hook(hook))
        if native_adamw:
            update = model.step.update
            model.step.update = lambda: checked_update(update)
        return model

    def optimizer_factory(name, params, lr, wd):
        params = list(params); opt = make_opt(name, params, lr, wd)
        if params[0] is captured['model'].up_val:
            wanted_params = list(captured['model'].parameters())
            actual_params = [p for group in opt.param_groups for p in group['params']]
            if len(actual_params) != len(wanted_params) or any(a is not b for a,b in zip(actual_params,wanted_params)):
                raise AssertionError('caller optimizer did not bind every compact Parameter')
            if any((group['lr'],group['weight_decay']) != (.0007,.03) for group in opt.param_groups):
                raise AssertionError('caller optimizer scalar mismatch')
            original = opt.step
            opt.step = lambda *args, **kwargs: checked_update(lambda: original(*args, **kwargs))
            wrapped_opts.append(opt)
        return opt

    def save_in_memory(value, path):
        buffer = io.BytesIO(); real_save(value, buffer); buffer.seek(0)
        saves[Path(path).name] = torch.load(buffer, weights_only=True, map_location='cpu')

    def evaluate(model, x, y, batch):
        evaluations.append(dict(arm='hffn' if model is captured['model'] else 'reference',n=len(x),batch=batch))
        return real_evaluate(model,x,y,batch)

    def parity(reference, model, x, y):
        parity_requests.append(dict(n=len(x),reference=reference is captured['reference'],hffn=model is captured['model']))
        return real_parity(reference,model,x,y)

    try:
        with tempfile.TemporaryDirectory(prefix='hffn_final_caller_') as tmp, ExitStack() as stack, no_legacy(models):
            argv = [name, '--untimed', '--arch', '128_512', '--dataset_path', 'in-memory.pt', '--out_dir', tmp,
                    '--batch_size', '32', '--epochs', '1', '--optimizer', optimizer, '--lr', '.0007',
                    '--weight_decay', '.03', '--parity_samples', '5', '--max_train_samples', '65',
                    '--max_val_samples', '33', '--max_test_samples', '34']
            if kind == 'curve': argv += ['--hffn_mode', mode, '--parity_every', '1']
            for key, value in dict(load_dataset=lambda path: data, HFFNToyFFN=model_factory,
                    SparseMaskedReferenceFFN=ObservedReference, build_optimizer=optimizer_factory,
                    evaluate=evaluate, parity_snapshot=parity,
                    time=SimpleNamespace(perf_counter=forbidden)).items():
                stack.enter_context(patch.object(entry, key, value))
            if kind == 'curve':
                for key in MEMORY: stack.enter_context(patch.object(entry, key, forbidden))
                stack.enter_context(patch.object(entry, 'maybe_write_plot', lambda path, rows: None))
            stack.enter_context(patch.object(torch, 'save', save_in_memory))
            with patch.object(sys, 'argv', argv), patch('sys.stdout', io.StringIO()): entry.main()
            logs = list(Path(tmp).rglob('training_compare_log.json' if kind == 'curve' else 'hffn_train_log.json'))
            if len(logs) != 1: raise ValueError('actual caller did not write exactly one log')
            log = json.loads(logs[0].read_text())
            model = captured['model']; packed = saves['hffn_native_student.pt' if kind == 'curve' else 'hffn_student.pt']
            loaded = toy.load_toy_checkpoint(packed, 32, kernels=kernels)
            legacy = copy.deepcopy(packed); legacy.pop('hffn_runtime')
            old_loaded = toy.load_toy_checkpoint(legacy, 32, kernels=kernels)
            source_state = checkpoint.export_checkpoint(model)
            checks = dict(
                actual_source=Path(entry.__file__).resolve() == wanted,
                all_updates=len(updates) == 3 and all(row['passed'] for row in updates),
                batches=[row['n'] for row in updates] == [32, 32, 1],
                symmetric_requests=[x for x in h_calls if x['gradient']] == [x for x in r_calls if x['gradient']] ==
                    [dict(n=n, gradient=True, needs_x=need) for n in (32, 32, 1)],
                evaluation_schedule=evaluations == [dict(arm=arm,n=n,batch=32) for n in (33,34) for arm in ('reference','hffn')],
                parity_schedule=parity_requests == [dict(n=5,reference=True,hffn=True)]*3,
                untimed=log['timing_enabled'] is False and unmeasured(log),
                curve_memory=kind == 'toy' or (log['memory_accounting_enabled'] is False and
                    log['memory']['initial'] is None and log['memory']['final'] is None),
                save_support=set(packed['state_dict']) == set(PARAMS) | {'up_ptr', 'up_idx', 'down_ptr', 'down_idx'} and
                    set(packed['hffn_runtime']) == {'format', 'd_model', 'd_ff', 'groups'},
                configuration=(log['config']['lr'], log['config']['weight_decay']) == (.0007, .03),
                saved_configuration=packed['config'] == log['config'],
                saved_architecture=packed['architecture'] == dict(type='HFFNToyFFN',d_model=128,d_ff=512,
                    activation='gelu_exact',layout='hidden_major_compact'),
                saved_summary=packed['extra'] == (log['summary'] if kind=='curve' else
                    dict(test=log['test']['hffn'],final_parity=log['final_parity'])),
                new_roundtrip=digest(source_state) == digest(checkpoint.export_checkpoint(loaded)),
                old_roundtrip=digest(source_state) == digest(checkpoint.export_checkpoint(old_loaded)),
                owned_restore=all(getattr(loaded, key).data_ptr() != packed['state_dict'][key].data_ptr() for key in PARAMS),
                weights_only_load=loaded.step.steps == old_loaded.step.steps == 0 and
                    all(bool((x == 0).all()) for x in [*loaded.step.moments, *old_loaded.step.moments]),
                finite_log=finite(log), data_unchanged=digest(data) == original_data)
            return dict(gate='caller', caller=kind, optimizer=optimizer, mode=mode,
                        checks=checks, updates=updates, configuration=[.0007, .03],
                        hffn_calls=h_calls, reference_calls=r_calls,
                        evaluations=evaluations, parity_requests=parity_requests,
                        input_gradient=need, source=str(wanted.relative_to(ROOT)), source_sha256=hashlib.sha256(wanted.read_bytes()).hexdigest(),
                        payload_sha256=digest(packed), log_sha256=digest(log), passed=all(checks.values()))
    finally:
        for handle in handles: handle.remove()
        model = captured.get('model')
        if native_adamw and model is not None and 'update' in vars(model.step): del model.step.update
        for opt in wrapped_opts:
            if 'step' in vars(opt): del opt.step


def run(kernels):
    for optimizer in ('adamw', 'sgd'):
        yield case('toy', optimizer, 'weights_only', kernels)
    for optimizer in ('adamw', 'sgd'):
        for mode in ('full', 'weights_only'):
            yield case('curve', optimizer, mode, kernels)
