"""Cold native-host bridge to the sibling inference source generator.

No GPU work, value copy or compilation in this launcher: native host owns its
Metal buffers/queue. Source generation and metadata are prepared before exec.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,sys
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
 parser.add_argument('--input',type=int,default=128);parser.add_argument('--hidden',type=int,default=512)
 parser.add_argument('--output-size',type=int,default=128);parser.add_argument('--n',type=int,default=64)
 parser.add_argument('--hffn-forward-prototype',action='store_true')
 parser.add_argument('--hffn-prepare-only',action='store_true')
 args,_=parser.parse_known_args()
 root=next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
 sys.path.insert(0,str(root/'runtime/gpu/m1/llm/inference'))
 from native_transformer.hffn.training_entry import sources
 from native_transformer.hffn import kernels as K
 up,down=sources(args.input,args.hidden,args.output_size,args.n,layout='feature_major',wide=True)
 if args.hffn_forward_prototype:up=K.with_activation(up,'relu')
 # Native backward/output diagnostics use feature-major y. Only its destination
 # address differs; all arithmetic and policies remain the inference generator.
 count=0
 def feature(match):
  nonlocal count
  expression=match.group(1);suffix=f'*{args.output_size}+o'
  if not expression.endswith(suffix):raise RuntimeError('inference output expression changed')
  count+=1
  return 'y[o*ns[0]+'+expression[:-len(suffix)]+']'
 down=re.sub(r'y\[((?:[^\[\]]|\[[^\[\]]*\])*)\]',feature,down)
 expected=2 if K.down_mode(args.n,args.hidden)=='blocked' else 1
 if count!=expected:raise RuntimeError('inference output store count changed')
 prefix='#include <metal_stdlib>\nusing namespace metal;\n'
 up=prefix+up.replace('kernel void up(','kernel void hffn_inference_up(')
 down=prefix+down.replace('kernel void down(','kernel void hffn_inference_down(')
 plan=dict(input=args.input,hidden=args.hidden,output=args.output_size,n=args.n,
   up_groups=K.up_groups((((0,),(args.hidden,)),),(args.n,)),
   down_groups=K.down_groups(args.n,K.down_mode(args.n,args.hidden),args.output_size))
 leaf=Path(__file__).resolve().parent
 key=hashlib.sha256((up+down+json.dumps(plan,sort_keys=True)).encode()).hexdigest()
 directory=leaf/'build/forward_sources'/key
 directory.mkdir(parents=True,exist_ok=True)
 for name,value in (('up.metal',up),('down.metal',down),('dispatch.json',json.dumps(plan))):
  p=directory/name
  if p.exists():
   if p.read_text()!=value:raise RuntimeError('source cache mismatch')
  else:p.write_text(value)
 if args.hffn_prepare_only:
  print(directory,flush=True);return
 os.environ['HFFN_INFERENCE_FORWARD_DIR']=str(directory)
 name='metal_hffn_training_forward_runner' if args.hffn_forward_prototype else 'metal_hffn_train_step_runner'
 native=leaf/'build'/(name+'.native')
 os.execv(str(native),[str(native),*(v for v in sys.argv[1:] if v!='--hffn-forward-prototype')])
if __name__=='__main__':main()
