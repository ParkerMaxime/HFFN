"""Independent standard-library evidence reader; no Torch/runtime imports."""
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from recompute_api import inspect as inspect_api, corruption_controls as api_corruption_controls

QUANTITIES=set('y z h grad_x grad_up grad_down grad_b_up grad_b_down'.split())
MODELS={'pythia70m':6,'pythia160m':12,'pythia410m':24}
DENSITIES=(10,20,30,50,70,80)
SYNTH=('homog_256_1024','homog_512_2048','irreg_512_2048')
CALLER_CHECKS=set('actual_source all_updates batches symmetric_requests evaluation_schedule parity_schedule untimed curve_memory save_support configuration saved_configuration saved_architecture saved_summary new_roundtrip old_roundtrip owned_restore weights_only_load finite_log data_unchanged'.split())
UPDATE_CHECKS=set('route counter request storage actual_input_gradient native_gradient_binding support aliases inactive_reference finite'.split())
FAULTS={'contexts': ['capacity_0', 'capacity_-1', 'capacity_True', 'capacity_1.5', 'missing_lease', 'index_-1', 'index_1', 'index_True', 'index_1.5', 'exhaustion', 'occupied_index', 'consumed', 'no_overwrite_on_refusal', 'no_grad_live_preserved', 'direct_live_resize', 'direct_live_forward', 'direct_live_backward', 'direct_live_update', 'direct_live_step', 'direct_live_results', 'retain_graph_repeat', 'weights_only_sentinel', 'released_lease', 'stale_backward', 'reuse_old_output_retained', 'abandoned_release', 'foreign', 'torch_version', 'checkpoint_version', 'native_count', 'storage', 'dtype', 'tokens', 'width', 'concurrent_acquire', 'concurrent_forward', 'concurrent_backward', 'direct_backward_without_forward', 'direct_results_without_forward', 'direct_after_release', 'direct_results_coherent', 'direct_results_after_update', 'frozen_up_val', 'frozen_down_val', 'frozen_b_up', 'frozen_b_down', 'frozen_up_val_down_val_b_up_b_down', 'all_frozen_no_lease', 'detect_slot_mixing', 'detect_generation_fault'], 'checkpoint': ['format', 'missing_key', 'extra_key', 'float_index', 'negative_index', 'large_index', 'offset_overflow', 'offset_endpoint', 'offset_order', 'nonfinite', 'parameter_dtype', 'non_cpu_tensor', 'non_strided_tensor', 'parameter_shape', 'groups_shape', 'wrapped_groups', 'groups_endpoint', 'dimension_bound', 'hidden_dimension_bound', 'export_up_idx', 'export_groups', 'atomic_invalid_last', 'atomic_changed_support', 'atomic_changed_partition', 'aliased_swap', 'overlapping_source', 'version_guard', 'value_fault', 'partition_fault']}
FAULTS={key:set(value) for key,value in FAULTS.items()}
FAULTS['configuration']={f'invalid/{kind}/{field}/{i}' for kind in ('step','module','context') for field in ('lr','weight_decay') for i in range(10)}|{f'readonly/{kind}/{field}' for kind in ('step','context') for field in ('lr','weight_decay')}


def number(v):return type(v) in (int,float) and math.isfinite(v) and v>=0

def true_checks(row,keys):
    values=row.get('checks',{})
    return set(values)==keys and all(v is True for v in values.values())

def digest(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)

def identical(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


def canonical(mode):
    plan={}
    for model,count in MODELS.items():
        for density in DENSITIES:
            if mode=='smoke' and (model,density) not in [('pythia70m',10),('pythia410m',80)]:continue
            plan[f'{model}/d{density}']=[f'{model}_sparse{density}_ffn_only/layer{i}' for i in range(1 if mode=='smoke' else count)]
    plan.update({'synth/'+name:[name] for name in SYNTH})
    return plan


def read_rows(rows,p):
    issues=[];mode=p.get('mode');hardware=p.get('hardware');minimum=1 if hardware=='m1' else 2
    plan=canonical(mode);grid=[8,32,128,13,61,97,203,512]+([1,2,64,192] if hardware=='m1' else [2])
    if mode not in ('run','smoke') or hardware not in ('m1','x86') or type(p.get('minimum_n')) is not int or p.get('minimum_n')!=minimum:issues.append('mode/hardware/domain')
    if p.get('expected_layers')!=plan or not identical(p.get('export_grid'),grid) or not identical(p.get('synthetic_grid'),[128,13,minimum]):issues.append('canonical plan/grid')
    expected=Counter()
    for export,layers in plan.items():
        ns=[128,13,minimum] if mode=='smoke' or export.startswith('synth/') else grid
        for layer in layers:
            for n in ns:expected[('2',export,layer,n)]+=1
            expected[('3',export,layer,128)]+=1
    numeric=[r for r in rows if r.get('gate') in ('2','3')]
    if Counter((r.get('gate'),r.get('export'),r.get('layer'),r.get('n')) for r in numeric)!=expected:issues.append('old gate coverage/duplicates')
    for r in numeric:
        if type(r.get('n')) is not int:issues.append('old gate exact integer N')
        if r['gate']=='2':
            keys={'gate','export','layer','n','finite','passed'}|QUANTITIES
            ok=set(r)==keys and r.get('finite') is True and all(number(r.get(k)) and r[k]<=1e-4 for k in QUANTITIES)
        else:
            ok=set(r)=={'gate','export','layer','n','finite','passed','weight_lr','moment_rel'} and r.get('finite') is True and number(r.get('weight_lr')) and r['weight_lr']<=.01 and number(r.get('moment_rel'))
        if not ok or r.get('passed') is not True:issues.append('failed old numeric gate')
    api=[{k:v for k,v in r.items() if k!='gate'} for r in rows if r.get('gate')=='api']
    api_report=inspect_api(api,plan,minimum)
    if api_report.get('passed') is not True:issues.append('API '+str(api_report.get('issues')))
    refusals=[r for r in rows if r.get('gate')=='refusals']
    if Counter(r.get('suite') for r in refusals)!=Counter(FAULTS.keys()):issues.append('refusal suites coverage')
    for r in refusals:
        if set(r)!={'gate','suite','checks','passed'} or not true_checks(r,FAULTS.get(r.get('suite'),set())) or r.get('passed') is not True:issues.append('refusal suite predicates')
    callers=[r for r in rows if r.get('gate')=='caller']
    wanted=[('toy',o,'weights_only') for o in ('adamw','sgd')]+[('curve',o,m) for o in ('adamw','sgd') for m in ('full','weights_only')] if hardware=='m1' else []
    if Counter((r.get('caller'),r.get('optimizer'),r.get('mode')) for r in callers)!=Counter(wanted):issues.append('caller matrix')
    for r in callers:
        need=r.get('caller')=='curve' and r.get('mode')=='full';native=r.get('caller')=='curve' and r.get('optimizer')=='adamw'
        expected_source='runtime/cpu/m1/ffn/training/'+('train_toy_ffn_hffn_cpu.py' if r.get('caller')=='toy' else 'compare_training_curves.py')
        ok=(true_checks(r,CALLER_CHECKS) and r.get('passed') is True and r.get('configuration')==[.0007,.03] and r.get('input_gradient') is need and
            r.get('source')==expected_source and all(digest(r.get(k)) for k in ('source_sha256','payload_sha256','log_sha256')))
        ok &= r.get('evaluations')==[dict(arm=arm,n=n,batch=32) for n in (33,34) for arm in ('reference','hffn')]
        ok &= r.get('parity_requests')==[dict(n=5,reference=True,hffn=True)]*3
        hc,rc=r.get('hffn_calls',[]),r.get('reference_calls',[])
        requested=[dict(n=n,gradient=True,needs_x=need) for n in (32,32,1)]
        ok &= ([x for x in hc if x.get('gradient')]==[x for x in rc if x.get('gradient')]==requested)
        for calls in (hc,rc):
            ok &= bool(calls) and all(set(x)=={'n','gradient','needs_x'} and type(x['n']) is int and x['n']>0 and type(x['gradient']) is bool and type(x['needs_x']) is bool for x in calls)
        updates=r.get('updates',[]);ok &= len(updates)==3
        for i,u in enumerate(updates,1):
            n=(32,32,1)[min(i-1,2)];route=('backward_dual_nt' if n>=32 else 'backward_nt') if need else 'backward_weights_nt'
            trace=[dict(name=route,scalars=None)]+([dict(name='adamw',scalars=[.0007,.9,.999,1e-8,.03,i])] if native else [])
            ok &= set(u)=={'update','n','needs_x','native_trace','checks','passed'} and type(u.get('update')) is int and u.get('update')==i and type(u.get('n')) is int and u.get('n')==n and u.get('needs_x') is need and identical(u.get('native_trace'),trace) and true_checks(u,UPDATE_CHECKS) and u.get('passed') is True
        if not ok:issues.append('caller binding')
    if any(r.get('gate') not in ('2','3','api','refusals','caller') for r in rows):issues.append('unexpected gate')
    if any(r.get('passed') is not True for r in rows):issues.append('failed record')
    return dict(qualified=not issues,issues=issues,records=len(rows),layers=sum(map(len,plan.values())),api=api_report)


def corruption_controls(rows,p):
    api=[{k:v for k,v in r.items() if k!='gate'} for r in rows if r.get('gate')=='api']
    controls={'api/'+k:v for k,v in api_corruption_controls(api,p['expected_layers'],p['minimum_n']).items()}
    def reject(label,edit):
        copied=copy.deepcopy(rows);prov=copy.deepcopy(p);edit(copied,prov)
        try:controls[label]=read_rows(copied,prov)['qualified'] is False
        except (KeyError,TypeError,ValueError,IndexError):controls[label]=True
    reject('missing_record',lambda r,p:r.pop())
    reject('duplicate_record',lambda r,p:r.append(copy.deepcopy(r[0])))
    reject('unexpected_gate',lambda r,p:r[0].update(gate='unknown'))
    reject('canonical_domain',lambda r,p:p.update(minimum_n=9))
    reject('canonical_export',lambda r,p:p['expected_layers'].pop(next(iter(p['expected_layers']))))
    reject('old_output',lambda r,p:next(x for x in r if x['gate']=='2').update(y=.1))
    reject('old_nonfinite',lambda r,p:next(x for x in r if x['gate']=='2').update(y=float('nan')))
    reject('old_bool_distance',lambda r,p:next(x for x in r if x['gate']=='2').update(y=True))
    reject('old_bool_n',lambda r,p:next(x for x in r if x['gate']=='2').update(n=True))
    reject('old_update',lambda r,p:next(x for x in r if x['gate']=='3').update(weight_lr=.1))
    reject('api_missing',lambda r,p:r.remove(next(x for x in r if x['gate']=='api')))
    reject('api_failed',lambda r,p:next(x for x in r if x['gate']=='api').update(passed=False))
    reject('refusal_missing',lambda r,p:next(x for x in r if x['gate']=='refusals')['checks'].popitem())
    reject('refusal_false',lambda r,p:next(x for x in r if x['gate']=='refusals')['checks'].update(injected=False))
    if p['hardware']=='m1':
        reject('caller_missing',lambda r,p:r.remove(next(x for x in r if x['gate']=='caller')))
        reject('caller_grad_binding',lambda r,p:next(x for x in r if x['gate']=='caller')['updates'][0]['checks'].update(native_gradient_binding=False))
        reject('caller_scalar',lambda r,p:next(x for x in r if x['gate']=='caller').update(configuration=[.001,0.]))
        reject('caller_requests',lambda r,p:next(x for x in r if x['gate']=='caller')['hffn_calls'].clear())
    return controls


def read(run,overrides=None):
    overrides=overrides or {}
    def data(name):return copy.deepcopy(overrides[name]) if name in overrides else json.loads((run/name).read_text())
    p=data('provenance.json');rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    report=read_rows(rows,p);issues=report['issues']
    expected=canonical(p.get('mode'));manifest=data('manifest.json')
    if not identical(manifest,dict(status='passed',planned=len(expected),done=len(expected))) or str(overrides.get('EXIT',(run/'EXIT').read_text())).strip()!='0':issues.append('not passed/full/EXIT0')
    supervisor=data('supervisor.json')
    if not identical(supervisor,dict(status='passed',exit=0,child_exit=0,reason=None,forced_kill=False)):issues.append('supervisor completion')
    rechecks=data('rechecks.json')
    if not identical(rechecks,dict(sources=True,checkers=True,assets=True,artifacts=True)):issues.append('source/artifact/asset recheck')
    if len(p.get('artifacts',{}))!=2 or not p.get('sources') or not p.get('checkers') or len(p.get('assets',{}))!=271:issues.append('incomplete provenance')
    if not p.get('cpu_model') or p.get('architecture')!=('arm64' if p.get('hardware')=='m1' else 'x86_64'):issues.append('host model/architecture')
    if p.get('leaf')!='runtime/cpu/'+p.get('hardware','')+'/ffn/training':issues.append('actual leaf')
    suffixes={f'{model}_sparse{density}_ffn_only/export/layer{i}/hffn_training_compact_g4_v2.pt' for model,count in MODELS.items() for density in DENSITIES for i in range(count)}
    suffixes|={f'{model}_sparse{density}_ffn_only/export/manifest_compact_g4_v2.json' for model in MODELS for density in DENSITIES}
    suffixes.add('runtime/common/ffn/reference/fingerprints.json')
    if any(sum(path.endswith('/'+suffix) for path in p.get('assets',{}))!=1 for suffix in suffixes):issues.append('canonical asset provenance')
    root=next(q for q in Path(__file__).resolve().parents if (q/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
    def asset_digest(suffix):
        values=[value for path,value in p.get('assets',{}).items() if path.endswith('/'+suffix)]
        return values[0] if len(values)==1 else None
    for model,count in MODELS.items():
        for density in DENSITIES:
            prefix=f'{model}_sparse{density}_ffn_only/export/'
            manifest=Path(os.environ['HFFN_ASSETS'])/'pythia/exports'/prefix/'manifest_compact_g4_v2.json'
            entries=json.loads(manifest.read_text())['layers']
            if asset_digest(prefix+'manifest_compact_g4_v2.json')!=hashlib.sha256(manifest.read_bytes()).hexdigest():issues.append('original manifest identity')
            if len(entries)!=count:issues.append('original manifest layer coverage')
            for i,entry in enumerate(entries):
                if entry['layer']!=i or entry['path']!=f'layer{i}/hffn_training_compact_g4_v2.pt' or asset_digest(prefix+entry['path'])!=entry['sha256']:issues.append('original asset identity')
    fingerprint=root/'runtime/common/ffn/reference/fingerprints.json'
    if asset_digest('runtime/common/ffn/reference/fingerprints.json')!=hashlib.sha256(fingerprint.read_bytes()).hexdigest():issues.append('synthetic fingerprint identity')
    imports=data('imports_final.json')
    common='runtime/cpu/common/ffn/training/hffn_ffn_training/'
    wanted={name:common+name+'.py' for name in ('step','module','contexts','checkpoint')}
    for name,path in wanted.items():
        if imports.get('hffn_ffn_training.'+name)!=path:issues.append('actual shared import '+name)
    if p.get('hardware')=='m1':
        prefix='runtime.cpu.m1.ffn.training.'
        for name in ('train_toy_ffn_hffn_cpu','compare_training_curves','shared_toy','parity','trainer','io','models','config'):
            path='runtime/cpu/m1/ffn/training/'+name+'.py'
            if imports.get(prefix+name)!=path or path not in p.get('sources',{}):issues.append('actual caller helper import '+name)
    if any('.CPU.' in name or name.endswith('.CPU') for name in imports):issues.append('legacy caller import')
    for values in (p.get('sources',{}),p.get('checkers',{}),p.get('artifacts',{}),p.get('assets',{})):
        if not all(digest(v) for v in values.values()):issues.append('bad source hash')
    checker_names={'batch.py','callers.py','capabilities.py','controls.py','gates.py','reader.py','recompute_api.py','supervise.py'}
    if set(p.get('checkers',{}))!=checker_names:issues.append('checker source inventory')
    cap=next((q for q in Path(__file__).resolve().parents if (q/'base_hashes.json').exists()),None)
    if cap is not None:
        admission=data('admission.json');pins=json.loads((cap/'base_hashes.json').read_text())
        expected_checks={'base_unchanged','runtime_inventory','overlay_scope'}|{'unchanged_'+name for name in ('oracle','gate3','closed_form','inputs','relative','kernels','pairs','targets')}
        if admission.get('base')!='bd1f4460d' or not true_checks(admission,expected_checks):issues.append('source/static admission')
        bound=admission.get('sources',{})
        suffix='/overlay/runtime/cpu/common/ffn/training/checks'
        prefix=p.get('checker_root','')[:-len(suffix)] if p.get('checker_root','').endswith(suffix) else ''
        own={'README.md','SCHEMA.md','base_hashes.json','runtime_hashes.json','native_hashes.json','verify.py','freeze.py','checker_faults.py','launch.sh'}
        own|={'overlay/runtime/cpu/common/ffn/training/checks/'+name for name in checker_names}
        own|={'overlay/runtime/cpu/'+hw+'/ffn/training/check.sh' for hw in ('m1','x86')}
        if set(bound)!=set(pins)|{prefix+'/'+name for name in own}:issues.append('complete capsule/freeze inventory')
        if any(bound.get(prefix+'/'+name)!=hashlib.sha256((cap/name).read_bytes()).hexdigest() for name in own):issues.append('frozen capsule source content')
        if not all(bound.get(k)==v for k,v in pins.items()) or any(p.get('sources',{}).get(k)!=v for k,v in pins.items() if k in p.get('sources',{})):issues.append('base source pins')
        if p.get('sources')!=json.loads((cap/'runtime_hashes.json').read_text()):issues.append('complete bound runtime inventory')
        if p.get('declared_base')!='bd1f4460d':issues.append('declared base commit')
        if any(bound.get(p['checker_root']+'/'+name)!=value for name,value in p.get('checkers',{}).items()):issues.append('checker/admission hashes')
        if json.loads((cap/'native_hashes.json').read_text()).get(p.get('hardware'))!=p.get('artifacts'):issues.append('qualified native artifact identity')
        if not identical(data('capsule_recheck.json'),{'unchanged':True}):issues.append('capsule source recheck')
        if p.get('mode')=='run':
            if json.loads((cap/'FROZEN.json').read_text())!={'sources':bound}:issues.append('source freeze')
            attempt=data('attempt.json')
            if type(attempt.get('number')) is not int or attempt['number'] not in (1,2) or attempt.get('scientific_started') is not True:issues.append('attempt admission')
    controls=data('reader_controls.json')
    wanted={'api/'+name for name in 'coverage duplicate extra graderror_nan graderror_bool graderror_bound groups_missing empty nonfinite weight_bound moment_bound trace scalar counter carryhash contextaccum checkpoint_g4 bindingpath'.split()}
    wanted|=set('missing_record duplicate_record unexpected_gate canonical_domain canonical_export old_output old_nonfinite old_bool_distance old_bool_n old_update api_missing api_failed refusal_missing refusal_false'.split())
    if p.get('hardware')=='m1':wanted|=set('caller_missing caller_grad_binding caller_scalar caller_requests'.split())
    if set(controls)!=wanted or not all(v is True for v in controls.values()):issues.append('missing/failed reader controls')
    raw_hash=hashlib.sha256((run/'raw.jsonl').read_bytes()).hexdigest()
    if data('summary.json').get('raw_sha256')!=raw_hash:issues.append('raw identity')
    report.update(qualified=not issues,raw_sha256=raw_hash)
    return report


if __name__=='__main__':
    try:report=read(Path(sys.argv[1]));code=0 if report['qualified'] else 1
    except Exception as e:report=dict(qualified=False,issues=[type(e).__name__+': '+str(e)]);code=1
    print(json.dumps(report,indent=2));raise SystemExit(code)
