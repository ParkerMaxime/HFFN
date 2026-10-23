"""Inference-owned one-bank forward on live compact values and caller buffers.

Construction compiles the kept inference arithmetic and fixes support/N.
forward_into borrows buffers until all queued GPU consumers finish. There is
no private activation scratch or value permutation to refresh after updates.
"""
from __future__ import annotations

import re
from functools import lru_cache
import torch

from . import kernels as K


def sources(d_in, d_ff, d_out, n, *, layout='token_major', wide=False):
    """Generate from the kept up/down bodies; no separate arithmetic kernel."""
    segments = (((0,), (d_ff,)),)
    reg, mode = K.regime(n), K.down_mode(n, d_ff)
    if layout not in ('token_major', 'feature_major'):
        raise ValueError('unsupported input layout')
    if reg == 0:
        up = K.up_source(segments, (reg,), d_model=d_in)
        if layout == 'feature_major':
            old = f'x0[t*{d_in}+uint(i0[e])]'
            if up.count(old) != 1:
                raise RuntimeError('inference cooperative input expression changed')
            up = up.replace(old, 'x0[uint(i0[e])*ns[0]+t]')
    else:
        x = (lambda i, t: f'x0[{t}*{d_in}+{i}]') if layout == 'token_major' else \
            (lambda i, t: f'x0[{i}*ns[0]+{t}]')
        store = lambda r, t: f'a0[({0})+{r}*(ns[0])+{t}]'
        head, total = K._up_block(0, d_ff, 'ns[0]', reg, '0')
        body = K._up_body(0, d_ff, 'ns[0]', reg, x, store)
        args = 'constant uint* ns,device const int* o0,device const short* i0,device const float* w0,device const float* x0,device float* a0'
        code = f'uint lane=tid&31u,simd=tid>>5u;if(tg>=0 && tg<(0+{total})){{{head}{body}}}'
        up = K.PRELUDE + f'kernel void up({args},{K.IDS}){{{code}}}'
    up = up.replace('device const float* w0', 'device const float* w0,device const float* bias_up')
    up = up.replace('device float* a0', 'device float* a0,device float* z0')
    up = up.replace('w0[r]', 'bias_up[r]').replace(f'w0[{d_ff}+', 'w0[')
    # Balanced one-level index brackets include ns[0]; RHS contains no semicolon.
    pattern = r'(a0\[(?:[^\[\]]|\[[^\[\]]*\])*\])=gelu\(([^;]+)\);'
    count = 0
    def save(match):
        nonlocal count
        count += 1
        dst, expression = match.groups()
        return '{float saved_z=' + expression + ';' + dst.replace('a0[', 'z0[') + '=saved_z;' + dst + '=gelu(saved_z);}'
    up = re.sub(pattern, save, up)
    if count != (4 if reg == 2 else 1):
        raise RuntimeError('inference activation stores changed')
    down = K.down_source(segments, (mode,), d_model=d_out)
    down = down.replace('device const float* v0', 'device const float* v0,device const int* slots0')
    down, count = re.subn(r'v0\[([^\]]+)\]', r'v0[slots0[\1]]', down)
    if count < 1:
        raise RuntimeError('inference down value reads changed')
    if wide:
        up, down = K.with_wide_indices(up), K.with_wide_indices(down)
    return up, down


@lru_cache(maxsize=128)
def _libraries(d_in,d_ff,d_out,n,layout,wide):
    up,down=sources(d_in,d_ff,d_out,n,layout=layout,wide=wide)
    return torch.mps.compile_shader(up),torch.mps.compile_shader(down)


class TrainingForward:
    """Fixed support/N; values and all dense call buffers belong to caller.

    Metadata is prepared once; x may be token-major or feature-major. z/act
    are [hidden,N], y is [N,d_out]. No output may alias input/values or another
    output. Writes to parameters must be ordered after previous GPU consumers.
    """
    def __init__(self, *, d_in, d_ff, d_out, n, up_offsets, up_idx, up_val,
                 b_up, down_offsets, down_hidden, down_slots, down_val, b_down,
                 layout='token_major'):
        if any(type(v) is not int or v <= 0 for v in (d_in,d_ff,d_out,n)):
            raise ValueError('positive integer dimensions/N required')
        if n*max(d_in,d_ff,d_out)>=2**32:
            raise ValueError('activation indexing exceeds uint32')
        if layout not in ('token_major', 'feature_major'):
            raise ValueError('unsupported input layout')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS required')
        values = (up_val, b_up, down_val, b_down)
        for v in values:
            self._tensor(v, torch.float32)
            if v.dim() != 1:
                raise ValueError('compact values must be vectors')
        if b_up.numel()!=d_ff or b_down.numel()!=d_out:
            raise ValueError('bias dimension mismatch')
        for v in (up_offsets,down_offsets,down_slots): self._tensor(v,torch.int32)
        if up_idx.dtype not in (torch.int16,torch.int32) or down_hidden.dtype!=up_idx.dtype:
            raise TypeError('matching int16/int32 compact indices required')
        for v in (up_idx,down_hidden):self._tensor(v,up_idx.dtype)
        if up_offsets.shape!=(d_ff+1,) or down_offsets.shape!=(d_out+1,) or up_idx.shape!=up_val.shape \
                or down_slots.shape!=down_hidden.shape or down_slots.numel()!=down_val.numel():
            raise ValueError('support extent mismatch')
        # Validate immutable support once on CPU before the first native call.
        uo,ui,do,dh,ds=(v.cpu().long() for v in (up_offsets,up_idx,down_offsets,down_hidden,down_slots))
        for offsets,count in ((uo,up_val.numel()),(do,down_val.numel())):
            if offsets[0]!=0 or offsets[-1]!=count or bool((offsets.diff()<0).any()):
                raise ValueError('invalid support offsets')
        for index,bound in ((ui,d_in),(dh,d_ff),(ds,down_val.numel())):
            if index.numel() and (int(index.min())<0 or int(index.max())>=bound):
                raise ValueError('invalid support index')
        if not torch.equal(torch.sort(ds).values,torch.arange(down_val.numel())):
            raise ValueError('down slots must be a permutation')
        self.d_in,self.d_ff,self.d_out,self.n,self.layout=d_in,d_ff,d_out,n,layout
        self.parameters=values
        self.support=(up_offsets,up_idx,down_offsets,down_hidden,down_slots)
        self._identity=tuple(self._signature(v) for v in (*values,*self.support))
        self.ns=torch.tensor([n],dtype=torch.int32,device='mps')
        self.gates=torch.ones(1,dtype=torch.float32,device='mps')  # one bank never reads gates
        self.up_lib,self.down_lib=_libraries(d_in,d_ff,d_out,n,layout,up_idx.dtype==torch.int32)
        segments=(((0,),(d_ff,)),)
        self.up_threads=K.up_groups(segments,(n,))*512
        self.down_threads=K.down_groups(n,K.down_mode(n,d_ff),d_out)*512
        self.calls=0

    @staticmethod
    def _tensor(v,dtype):
        if not isinstance(v,torch.Tensor) or v.device.type!='mps' or v.dtype!=dtype:
            raise TypeError('expected MPS tensor with declared dtype')
        if not v.is_contiguous():raise ValueError('contiguous buffer required')

    @staticmethod
    def _signature(v):
        return (v.untyped_storage()._cdata,v.storage_offset(),v.dtype,tuple(v.shape),v.device)

    @staticmethod
    def _overlap(a,b):
        # MPS data_ptr identifies a Metal buffer; it is not a byte-address
        # interval comparable with another allocation. Views share Storage.
        if a.untyped_storage()._cdata != b.untyped_storage()._cdata:
            return False
        first,last=a.storage_offset()*a.element_size(),b.storage_offset()*b.element_size()
        return max(first,last) < min(first+a.numel()*a.element_size(),last+b.numel()*b.element_size())

    def forward_into(self,x,z,act,y):
        values,support=self.parameters,self.support
        if tuple(self._signature(v) for v in (*values,*support))!=self._identity:
            raise RuntimeError('borrowed compact storage changed')
        expected=((self.n,self.d_in) if self.layout=='token_major' else (self.d_in,self.n),
                  (self.d_ff,self.n),(self.d_ff,self.n),(self.n,self.d_out))
        buffers=(x,z,act,y)
        for v,shape in zip(buffers,expected):
            self._tensor(v,torch.float32)
            if tuple(v.shape)!=shape:raise ValueError('call buffer shape mismatch')
        for i,dst in enumerate(buffers[1:]):
            if any(self._overlap(dst,src) for src in (*values,*support,*buffers[:i+1])):
                raise ValueError('call buffers overlap')
        up_val,b_up,down_val,b_down=values
        up_offsets,up_idx,down_offsets,down_hidden,down_slots=support
        self.up_lib.up(self.ns,up_offsets,up_idx,up_val,b_up,x,act,z,
                       threads=self.up_threads,group_size=512)
        self.down_lib.down(self.ns,down_offsets,down_hidden,down_val,down_slots,act,
                           self.gates,b_down,y,1,threads=self.down_threads,group_size=512)
        self.calls+=1
