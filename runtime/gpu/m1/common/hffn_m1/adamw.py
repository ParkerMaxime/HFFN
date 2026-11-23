"""Shared compact Metal update with native and Torch argument-binding ABIs."""
from pathlib import Path
import json

_CORE=Path(__file__).with_suffix('.metal').read_text()

def torch_source():
    return _CORE+'''
kernel void adamw_compact(
    device float* param, device float* grad,
    device float* moment1, device float* moment2,
    constant float& lr, constant float& beta1, constant float& beta2,
    constant float& eps, constant float& weight_decay,
    constant float& step_size, constant float& beta2_scale,
    uint gid [[thread_position_in_grid]]) {
  hffn_adamw_slot<true,true>(param[gid],grad[gid],moment1[gid],moment2[gid],
      lr,beta1,beta2,eps,weight_decay,step_size,beta2_scale);
}
'''

def native_source():
    return _CORE+'''
struct AdamW{float lr,beta1,beta2,eps,wd,step_size,beta2_scale;};
kernel void adamw(device float*p[[buffer(0)]],device float*g[[buffer(1)]],
    device float*m[[buffer(2)]],device float*v[[buffer(3)]],
    constant AdamW&q[[buffer(4)]],constant uint&count[[buffer(5)]],
    uint gid[[thread_position_in_grid]]) {
  if(gid>=count)return;
  hffn_adamw_slot<false,false>(p[gid],g[gid],m[gid],v[gid],
      q.lr,q.beta1,q.beta2,q.eps,q.wd,q.step_size,q.beta2_scale);
}
'''

def emit_header(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('#pragma once\n#define HFFN_M1_ADAMW_MSL '+json.dumps(native_source())+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--emit-header',required=True)
    emit_header(p.parse_args().emit_header)
