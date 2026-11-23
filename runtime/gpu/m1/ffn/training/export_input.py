"""Cold checked native compact input; no converted export file is created.

Admission covers rectangular/wide dimensions. Native independently validates
both the sized C descriptor and the CLI compact transport.
"""
from collections.abc import Mapping
from pathlib import Path
import struct

import torch

INTEGER = (torch.int16, torch.int32, torch.int64)


def checked(payload):
    if not isinstance(payload, Mapping) or payload.get('format') != 'hidden_ffn':
        raise ValueError('expected hidden_ffn export')
    meta = payload.get('metadata')
    if not isinstance(meta, Mapping) or str(meta.get('activation', 'gelu')).lower() != 'gelu':
        raise ValueError('GELU metadata required')
    f = meta.get('d_ff')
    i = meta.get('d_in', meta.get('d_model'))
    o = meta.get('d_out', meta.get('d_model'))
    if any(type(v) is not int or not 0 < v < 2**31 for v in (i, f, o)):
        raise ValueError('positive signed-int32 dimensions required')
    def vector(name, dtype=None):
        t = payload.get(name)
        if not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or t.layout != torch.strided or t.ndim != 1:
            raise ValueError(name + ': expected CPU vector')
        if dtype is not None and t.dtype != dtype:
            raise ValueError(name + ': wrong dtype')
        return t
    max_degree = 1
    for name, size in (('b_up', f), ('b_down', o)):
        t = vector(name, torch.float32)
        if t.numel() != size or not bool(torch.isfinite(t).all()):
            raise ValueError(name + ': bias extent/nonfinite')
    for side, bound in (('up', i), ('down', o)):
        off, idx, val = vector(side+'_ptr'), vector(side+'_idx'), vector(side+'_val', torch.float32)
        if off.dtype not in INTEGER or idx.dtype not in INTEGER:
            raise ValueError(side + ': integer support required')
        if off.numel() != f+1 or idx.numel() != val.numel() or not idx.numel():
            raise ValueError(side + ': invalid/empty projection extent')
        if int(off.min()) < 0 or int(off.max()) >= 2**31 or int(off[0]) != 0 \
                or int(off[-1]) != idx.numel() or bool((off[1:] < off[:-1]).any()):
            raise ValueError(side + ': invalid exact offsets')
        if int(idx.min()) < 0 or int(idx.max()) >= bound:
            raise ValueError(side + ': feature index outside dimension')
        if not bool(torch.isfinite(val).all()):
            raise ValueError(side + ': nonfinite value')
        max_degree = max(max_degree, int(off.diff().max()))
        for a, b in zip(off[:-1].tolist(), off[1:].tolist()):
            if torch.unique(idx[a:b]).numel() != b-a:
                raise ValueError(side + ': duplicate connection')
    ue, de = payload['up_val'].numel(), payload['down_val'].numel()
    if ue+de+f+o >= 2**31:
        raise ValueError('compact pool exceeds int32')
    bits = max(1, (max_degree-1).bit_length())
    if ((f-1) << bits) + max_degree-1 >= 2**32:
        raise ValueError('native route codec overflow')
    return i, f, o, ue, de


def parts(payload):
    """Check before unsigned narrowing and emit compact buffers in wire order."""
    i, f, o, ue, de = checked(payload)
    yield struct.pack('=8I', 0x4846464e, 1, i, f, o, ue, de, 0)
    for name in ('up_ptr', 'down_ptr', 'up_idx', 'down_idx'):
        yield payload[name].contiguous().numpy().astype('uint32').tobytes()
    for name in ('up_val', 'down_val', 'b_up', 'b_down'):
        yield payload[name].detach().contiguous().numpy().tobytes()


def read_export(directory):
    payload = torch.load(Path(directory)/'hidden_ffn_format.pt', map_location='cpu', weights_only=True)
    checked(payload)
    return payload


# Sized native-endian arrays are borrowed only during the cold constructor.
import ctypes as C
class Span(C.Structure):
    _fields_=[('data',C.c_void_p),('elements',C.c_uint64)]
class Support(C.Structure):
    _fields_=[('version',C.c_uint32),('input',C.c_uint32),('hidden',C.c_uint32),('output',C.c_uint32)]+[(name,Span) for name in ('up_offsets','down_offsets','up_idx','down_idx','up_val','down_val','b_up','b_down')]
class OwnedSupport:
    def __init__(self,payload):
        i,h,o,_,_=checked(payload)
        self.arrays=[]
        for name in ('up_ptr','down_ptr','up_idx','down_idx','up_val','down_val','b_up','b_down'):
            array=payload[name].detach().contiguous().numpy()
            if name.endswith(('_ptr','_idx')):array=array.astype('uint32',copy=False)
            self.arrays.append(array)
        self.descriptor=Support(1,i,h,o,*(Span(array.ctypes.data,array.size) for array in self.arrays))
