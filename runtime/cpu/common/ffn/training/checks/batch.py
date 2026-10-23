"""Source-bound every-export CPU training/API check; untimed numerical evidence.

The parent launcher owns the operational deadline. No timer, RSS or byte
quantity is read here. Validation controls may use dense FP64 equations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'runtime'/'common'/'pythia'/'hffn_pythia').is_dir())
MODELS={'pythia70m':6,'pythia160m':12,'pythia410m':24}
DENSITIES=(10,20,30,50,70,80)
SYNTH=('homog_256_1024','homog_512_2048','irreg_512_2048')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def source_inventory():
    roots=('runtime/common','runtime/cpu/common','runtime/cpu/m1/ffn/training',
           'runtime/cpu/m1/llm/inference/compact','runtime/cpu/x86/common',
           'runtime/cpu/x86/ffn/training')
    paths=[]
    for rel in roots:
        for p in (ROOT/rel).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.sh','.cpp','.h','.hpp','.json') and not any(
                    x in ('build','runs','__pycache__','dataset','datasets') for x in p.relative_to(ROOT/rel).parts):
                paths.append(p)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def plan(assets,mode,gates):
    inventory={};targets={};groups=[]
    for model,count in MODELS.items():
        for density in DENSITIES:
            directory=assets/f'pythia/exports/{model}_sparse{density}_ffn_only/export'
            manifest=directory/'manifest_compact_g4_v2.json'; data=json.loads(manifest.read_text())
            expected=[directory/f'layer{i}/hffn_training_compact_g4_v2.pt' for i in range(count)]
            if (data['format']!='hffn_training_compact_g4_v2' or
                    [e['layer'] for e in data['layers']]!=list(range(count)) or
                    [directory/e['path'] for e in data['layers']]!=expected or gates.export_layers(directory)!=expected):
                raise ValueError('noncanonical/incomplete manifest '+str(manifest))
            inventory[str(manifest)]=sha(manifest)
            for p,e in zip(expected,data['layers']):
                inventory[str(p)]=sha(p)
                if inventory[str(p)]!=e['sha256']:raise ValueError('asset digest mismatch '+str(p))
            if mode=='smoke' and (model,density) not in [('pythia70m',10),('pythia410m',80)]:continue
            selected=expected[:1] if mode=='smoke' else expected
            name=f'{model}/d{density}'
            targets[name]=[f'{p.parents[2].name}/{p.parent.name}' for p in selected]
            groups.append((name,selected))
    from hffn_ffn_layers import layers
    if tuple(layers.names())!=SYNTH:raise ValueError('synthetic inventory drift')
    inventory[str(layers.FINGERPRINTS)]=sha(layers.FINGERPRINTS)
    for name in SYNTH:
        targets['synth/'+name]=[name];groups.append(('synth/'+name,[name]))
    return targets,groups,inventory


def run(args):
    import torch
    torch.set_num_threads(1)
    import gates
    from hffn_ffn_training import build
    from hffn_ffn_layers import layers
    from controls import run as refusal_rows
    from reader import read_rows, corruption_controls
    os.environ['HFFN_ASSETS']=str(args.assets)    # read by reader.py
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'raw.jsonl').exists():raise FileExistsError('refused existing evidence')
    expected,groups,assets=plan(args.assets,args.mode,gates)
    before=source_inventory(); checker={p.name:sha(p) for p in HERE.glob('*.py')}
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in (build.training_lib(),build.inference_lib())}
    hardware='m1' if gates.INFERENCE_MIN_N==1 else 'x86'
    if hardware!=args.hardware:raise ValueError('hardware minimum/leaf mismatch')
    if (hardware=='m1' and platform.machine()!='arm64') or (hardware=='x86' and platform.machine() not in ('x86_64','AMD64')):
        raise ValueError('host architecture mismatch')
    grid=tuple(gates.GATE_N)
    if gates.INFERENCE_MIN_N not in grid:grid+=(gates.INFERENCE_MIN_N,)
    cpu_model=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if hardware=='m1' else next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    provenance=dict(hardware=hardware,architecture=platform.machine(),cpu_model=cpu_model,
        declared_base=os.environ.get('HFFN_CHECK_BASE'),minimum_n=gates.INFERENCE_MIN_N,
        mode=args.mode,leaf=str(gates.LEAF.relative_to(ROOT)),source_root=str(ROOT),sources=before,
        checkers=checker,checker_root=str(HERE.relative_to(ROOT)),artifacts=artifacts,assets=assets,
        expected_layers=expected,export_grid=list(grid),synthetic_grid=[128,13,gates.INFERENCE_MIN_N],
        imported={name:str(Path(mod.__file__).resolve().relative_to(ROOT)) for name,mod in sys.modules.copy().items()
            if getattr(mod,'__file__',None) and (name=='gates' or name.startswith('hffn_ffn_training'))})
    write(out/'provenance.json',provenance)
    write(out/'manifest.json',dict(status='running',planned=len(expected),done=0))
    rows=[]
    def emit(row):
        with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row)
        if row.get('passed') is not True:raise AssertionError('failed record '+str(row))
    original_targets,original_grid=gates.targets,gates.GATE_N
    try:
        for index,(name,items) in enumerate(groups):
            def selected(_):
                return [(name,(gates.load(item) if isinstance(item,Path) else gates.from_compact(item,layers.load(item)[0]) for item in items))]
            gates.targets=selected
            gates.GATE_N=tuple([128,13,gates.INFERENCE_MIN_N]) if args.mode=='smoke' or name.startswith('synth/') else original_grid
            gargs=SimpleNamespace(api_check=True)
            if index==0 and args.mode=='run' and os.environ.get('HFFN_PENDING_ATTEMPT'):
                write(out/'attempt.json',json.loads(os.environ['HFFN_PENDING_ATTEMPT']))
            ok,payload=gates.gate2(gargs)
            write(out/f'gate2_{index:02d}.json',dict(passed=ok,**payload))
            for row in payload['rows']:emit(dict(gate='2',**row))
            for row in payload['api_rows']:emit(dict(gate='api',**row))
            if not ok:raise AssertionError('gate2/API failed '+name)
            ok,payload=gates.gate3(SimpleNamespace())
            write(out/f'gate3_{index:02d}.json',dict(passed=ok,**payload))
            for row in payload['rows']:emit(dict(gate='3',**row))
            if not ok:raise AssertionError('gate3 failed '+name)
            write(out/'manifest.json',dict(status='running',planned=len(expected),done=index+1))
            print(f'batch coverage {index+1}/{len(expected)} {name}',flush=True)
        kernels=gates.kernels()
        fixture=gates.from_compact(SYNTH[0],layers.load(SYNTH[0])[0])
        for row in refusal_rows(fixture,kernels):emit(row)
        if hardware=='m1':
            from callers import run as caller_rows
            for row in caller_rows(kernels):emit(row)
        final_imports={name:str(Path(mod.__file__).resolve().relative_to(ROOT)) for name,mod in sys.modules.copy().items()
            if getattr(mod,'__file__',None) and (name.startswith('hffn_ffn_training') or name.startswith('runtime.cpu.m1.ffn.training'))}
        write(out/'imports_final.json',final_imports)
        rechecks=dict(sources=before==source_inventory(),checkers=checker=={p.name:sha(p) for p in HERE.glob('*.py')},
            assets=all(sha(p)==v for p,v in assets.items()),artifacts=all(sha(ROOT/p)==v for p,v in artifacts.items()))
        write(out/'rechecks.json',rechecks)
        if not all(rechecks.values()):raise AssertionError('source/asset/artifact drift')
        report=read_rows(rows,provenance)
        write(out/'recomputed.json',report)
        if report['qualified'] is not True:raise AssertionError(str(report))
        controls=corruption_controls(rows,provenance);write(out/'reader_controls.json',controls)
        if not controls or not all(controls.values()):raise AssertionError('reader corruption control failed')
        write(out/'summary.json',dict(status='passed',records=len(rows),layers=sum(map(len,expected.values())),raw_sha256=sha(out/'raw.jsonl')))
        write(out/'manifest.json',dict(status='passed',planned=len(expected),done=len(expected)))
    finally:
        gates.targets,gates.GATE_N=original_targets,original_grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--hardware',choices=('m1','x86'),required=True)
    p.add_argument('--mode',choices=('smoke','run'),required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except BaseException:
        traceback.print_exc();path=args.out/'manifest.json'
        state=json.loads(path.read_text()) if path.exists() else dict(planned=0,done=0)
        state['status']='failed';write(path,state);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
