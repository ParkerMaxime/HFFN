#!/usr/bin/env python3
"""Replays the HFFN GELU train step on CPU and compares the exported GPU pools."""
import argparse
import math
from pathlib import Path

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x).astype(np.float32)
    t = np.float32(1) / (np.float32(1) + np.float32(.3275911) * ax)
    p = (((((np.float32(1.061405429) * t - np.float32(1.453152027)) * t + np.float32(1.421413741)) * t - np.float32(.284496736)) * t + np.float32(.254829592)) * t)
    erf = np.where(x < 0, -(np.float32(1) - p * np.exp(-ax * ax)), np.float32(1) - p * np.exp(-ax * ax))
    return (np.float32(.5) * x * (np.float32(1) + erf)).astype(np.float32)


def gelu_grad(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x).astype(np.float32)
    t = np.float32(1) / (np.float32(1) + np.float32(.3275911) * ax)
    p = (((((np.float32(1.061405429) * t - np.float32(1.453152027)) * t + np.float32(1.421413741)) * t - np.float32(.284496736)) * t + np.float32(.254829592)) * t)
    erf = np.where(x < 0, -(np.float32(1) - p * np.exp(-ax * ax)), np.float32(1) - p * np.exp(-ax * ax))
    return (np.float32(.5) * (np.float32(1) + erf) + x * np.float32(.3989422804014327) * np.exp(np.float32(-.5) * x * x)).astype(np.float32)


def adamw(p, g, m, v, step):
    lr, b1, b2, eps, wd = map(np.float32, (1e-3, .9, .999, 1e-8, .01))
    m[:] = b1 * m + (np.float32(1) - b1) * g
    v[:] = b2 * v + (np.float32(1) - b2) * g * g
    step_size = np.float32(float(lr) / (1 - math.pow(float(b1), step)))
    b2scale = np.float32(math.sqrt(1 - math.pow(float(b2), step)))
    p[:] = p * (np.float32(1) - lr * wd) - step_size * (m / (np.sqrt(v) / b2scale + eps))


def sum_f32(values) -> np.float32:
    total = np.float32(0)
    for value in values:
        total = np.float32(total + value)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=int, required=True); p.add_argument('--hidden', type=int, required=True)
    p.add_argument('--output-size', type=int, required=True); p.add_argument('--n', type=int, required=True)
    p.add_argument('--degree', type=int, required=True); p.add_argument('--steps', type=int, required=True)
    p.add_argument('--dump-prefix', type=Path, required=True); a = p.parse_args()
    I,H,O,N,D=a.input,a.hidden,a.output_size,a.n,a.degree; E=H*D
    x=np.empty((I,N),np.float32); gy=np.empty((O,N),np.float32)
    for i in range(I):
        for n in range(N): x[i,n]=np.float32(((i*17+n*13)%29-14)*.02)
    for o in range(O):
        for n in range(N): gy[o,n]=np.float32(((o*11+n*7)%23-11)*.03 + .0007)
    idx=np.empty(E,np.int32); target=np.empty(E,np.int32); up=np.empty(E,np.float32); down=np.empty(E,np.float32)
    bu=np.empty(H,np.float32); bd=np.empty(O,np.float32)
    for r in range(H):
        bu[r]=np.float32((r%9-4)*.01)
        for e in range(D):
            q=r*D+e; idx[q]=(r*19+e*7)%I; target[q]=(r*17+e*5)%O
            up[q]=np.float32(((q*3)%17-8)*.015); down[q]=np.float32(((q*5)%19-9)*.012)
    for o in range(O): bd[o]=np.float32((o%11-5)*.008)
    um=np.zeros(E,np.float32); uv=np.zeros(E,np.float32); dm=np.zeros(E,np.float32); dv=np.zeros(E,np.float32)
    bum=np.zeros(H,np.float32); buv=np.zeros(H,np.float32); bdm=np.zeros(O,np.float32); bdv=np.zeros(O,np.float32)
    for step in range(1,a.steps+1):
        z=np.empty((H,N),np.float32)
        for r in range(H):
            for n in range(N):
                value=bu[r]
                for e in range(D):
                    q=r*D+e; value=np.float32(value + up[q]*x[idx[q],n])
                z[r,n]=value
        h=gelu(z); gz=np.empty_like(z); gd=np.empty(E,np.float32); gu=np.empty(E,np.float32); gx=np.zeros((I,N),np.float32)
        for r in range(H):
            for e in range(D):
                q=r*D+e; gd[q]=sum_f32(h[r,n]*gy[target[q],n] for n in range(N))
            for n in range(N):
                gh=np.float32(0)
                for e in range(D):
                    q=r*D+e; gh=np.float32(gh + down[q]*gy[target[q],n])
                gz[r,n]=np.float32(gh*gelu_grad(z[r,n]))
            for e in range(D):
                q=r*D+e; gu[q]=sum_f32(gz[r,n]*x[idx[q],n] for n in range(N))
                for n in range(N): gx[idx[q],n]=np.float32(gx[idx[q],n] + up[q]*gz[r,n])
        gbu=np.array([sum_f32(gz[r,n] for n in range(N)) for r in range(H)],np.float32)
        gbd=np.array([sum_f32(gy[o,n] for n in range(N)) for o in range(O)],np.float32)
        if step == 1:
            gpu_gu=np.fromfile(str(a.dump_prefix)+'_pre_gup.f32',dtype=np.float32)
            gpu_gd=np.fromfile(str(a.dump_prefix)+'_pre_gdown.f32',dtype=np.float32)
            print('pre_gup', float(np.max(np.abs(gu-gpu_gu))), 'pre_gdown', float(np.max(np.abs(gd-gpu_gd))))
        adamw(up,gu,um,uv,step); adamw(down,gd,dm,dv,step); adamw(bu,gbu,bum,buv,step); adamw(bd,gbd,bdm,bdv,step)
    pools={'up':up,'down':down,'bup':bu,'bdown':bd,'um':um,'uv':uv,'dm':dm,'dv':dv,'bum':bum,'buv':buv,'bdm':bdm,'bdv':bdv}
    errors={}
    for name,ref in pools.items():
        got=np.fromfile(str(a.dump_prefix)+'_'+name+'.f32',dtype=np.float32); errors[name]=float(np.max(np.abs(ref-got)))
        if name in {'up', 'um', 'down', 'dm'}:
            k=int(np.argmax(np.abs(ref-got)))
            print(name, 'ref', ref[:4], 'gpu', got[:4], 'delta', (ref-got)[:4], 'max_at', k, ref[k], got[k])
    print(errors)
    if max(errors.values()) > 3e-6: raise SystemExit(1)

if __name__ == '__main__': main()
