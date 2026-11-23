"""Thin C ABI owner. CPU views require explicitly selected Shared storage.

No torch conversion, autograd integration or hidden Private-buffer upload.
Borrowed native views expire at close; snapshots below return Python bytes.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import threading
import subprocess
import sys

def prepare_sources(arguments):
    root=next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
    launcher=root/"runtime/gpu/m1/ffn/training/run_native_train_step.py"
    command=[sys.executable,"-B",str(launcher),*map(str,arguments)]
    if "--hffn-prepare-only" not in command:command.append("--hffn-prepare-only")
    return Path(subprocess.check_output(command,text=True).strip())

class View(C.Structure):
    _fields_=[('metal_buffer',C.c_void_p),('data',C.c_void_p),('elements',C.c_uint64),
              ('byte_offset',C.c_uint64),('storage_mode',C.c_uint32)]

class Context:
    def __init__(self, library: str | Path, bundle: str | Path, arguments=(), *, support=None, parameters=None):
        """`parameters=(metal_buffer, byte_offset, count)` creates an exact context whose compact parameter pool is the
        caller's Metal buffer (borrowed handle, see `bind_flat_parameters`); it requires `support`."""
        if parameters is not None and support is None: raise ValueError('an external parameter pool requires an exact support')
        self._lock=threading.RLock()
        self._handle=None
        self._lib=C.CDLL(str(library))
        l=self._lib
        l.hffn_train_create.argtypes=[C.c_int,C.POINTER(C.c_char_p),C.c_char_p]
        l.hffn_train_create.restype=C.c_void_p
        l.hffn_train_step.argtypes=[C.c_void_p];l.hffn_train_step.restype=C.c_int
        for name in ('hffn_train_stage_begin','hffn_train_stage_update'):
            fn=getattr(l,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_int
        for name in ('hffn_train_stage_forward','hffn_train_stage_backward'):
            fn=getattr(l,name);fn.argtypes=[C.c_void_p,C.c_uint32];fn.restype=C.c_int
        l.hffn_train_stage_clear_gradients.argtypes=[C.c_void_p];l.hffn_train_stage_clear_gradients.restype=C.c_int
        l.hffn_train_stage_view.argtypes=[C.c_void_p,C.c_char_p,C.c_uint32,C.POINTER(View)]
        l.hffn_train_stage_view.restype=C.c_int
        l.hffn_train_view.argtypes=[C.c_void_p,C.c_char_p,C.c_uint32,C.POINTER(View)]
        l.hffn_train_view.restype=C.c_int
        l.hffn_train_reset.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_uint64]
        l.hffn_train_reset.restype=C.c_int
        l.hffn_train_completed_updates.argtypes=[C.c_void_p]
        l.hffn_train_completed_updates.restype=C.c_uint32
        l.hffn_train_error.restype=C.c_char_p
        l.hffn_train_destroy.argtypes=[C.c_void_p];l.hffn_train_destroy.restype=None
        args=[str(a).encode() for a in arguments]
        argv=(C.c_char_p*len(args))(*args)
        if support is None:
            self._handle=l.hffn_train_create(len(args),argv,str(bundle).encode())
        elif parameters is None:
            from export_input import OwnedSupport,Support
            owned=OwnedSupport(support)
            l.hffn_train_create_exact.argtypes=[C.c_int,C.POINTER(C.c_char_p),C.c_char_p,C.POINTER(Support)]
            l.hffn_train_create_exact.restype=C.c_void_p
            self._handle=l.hffn_train_create_exact(len(args),argv,str(bundle).encode(),C.byref(owned.descriptor))
        else:
            from export_input import OwnedSupport,Support
            owned=OwnedSupport(support)
            l.hffn_train_create_exact_with_parameters.argtypes=[C.c_int,C.POINTER(C.c_char_p),C.c_char_p,C.POINTER(Support),
                                                                 C.c_void_p,C.c_uint64,C.c_uint64]
            l.hffn_train_create_exact_with_parameters.restype=C.c_void_p
            buffer,offset,count=parameters
            self._owned=owned
            self._handle=l.hffn_train_create_exact_with_parameters(len(args),argv,str(bundle).encode(),C.byref(owned.descriptor),
                                                                    buffer,offset,count)
        if not self._handle: self._raise()
    def _raise(self):
        raise RuntimeError(self._lib.hffn_train_error().decode())
    def _check(self):
        if not self._handle: raise RuntimeError('context is closed')
    def step(self):
        with self._lock:
            self._check()
            if self._lib.hffn_train_step(self._handle): self._raise()
            return self.completed_updates
    def stage_begin(self):
        with self._lock:
            self._check()
            if self._lib.hffn_train_stage_begin(self._handle): self._raise()
    def stage_forward(self,slot):
        with self._lock:
            self._check()
            if not isinstance(slot,int) or not 0<=slot<2**32: raise ValueError('invalid slot')
            if self._lib.hffn_train_stage_forward(self._handle,slot): self._raise()
    def stage_backward(self,slot):
        with self._lock:
            self._check()
            if not isinstance(slot,int) or not 0<=slot<2**32: raise ValueError('invalid slot')
            if self._lib.hffn_train_stage_backward(self._handle,slot): self._raise()
    def stage_update(self):
        with self._lock:
            self._check()
            if self._lib.hffn_train_stage_update(self._handle): self._raise()
            return self.completed_updates
    def stage_clear_gradients(self):
        with self._lock:
            self._check()
            if self._lib.hffn_train_stage_clear_gradients(self._handle): self._raise()
    def stage_view(self,name,slot=0):
        with self._lock:
            self._check()
            if not isinstance(slot,int) or not 0<=slot<2**32: raise ValueError('invalid slot')
            result=View()
            if self._lib.hffn_train_stage_view(self._handle,str(name).encode(),slot,C.byref(result)): self._raise()
            return result
    @property
    def completed_updates(self):
        with self._lock:
            self._check()
            return self._lib.hffn_train_completed_updates(self._handle)
    def view(self,name,slot=0):
        with self._lock:
            self._check()
            if not isinstance(slot,int) or not 0<=slot<2**32: raise ValueError('invalid slot')
            result=View()
            if self._lib.hffn_train_view(self._handle,str(name).encode(),slot,C.byref(result)): self._raise()
            return result
    def snapshot(self,name,slot=0):
        with self._lock:
            view=self.view(name,slot)
            if not view.data: raise RuntimeError('Private buffer has no CPU view')
            return C.string_at(view.data,view.elements*4)
    def write_input(self,name,data,slot=0):
        if name not in ('x','gy'): raise ValueError('only input slots may be written')
        with self._lock:
            view=self.view(name,slot)
            if not view.data: raise RuntimeError('Private input requires an explicit native producer')
            data=bytes(data)
            if len(data)!=view.elements*4: raise ValueError('input size mismatch')
            C.memmove(view.data,data,len(data))
    def reset(self,packed_parameters):
        data=bytes(packed_parameters)
        if len(data)%4: raise ValueError('parameters must be float32 bytes')
        values=(C.c_float*(len(data)//4)).from_buffer_copy(data)
        with self._lock:
            self._check()
            if self._lib.hffn_train_reset(self._handle,values,len(values)): self._raise()
    def stage_optimizer_begin(self):
        """Optimizer-only step: `stage_begin` with no micro-batch; the caller then writes the four compact gradients through
        `stage_view("g_up" ...)` and calls `stage_update` (see `hffn_train_stage_optimizer_begin`)."""
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_stage_optimizer_begin.argtypes=[C.c_void_p];l.hffn_train_stage_optimizer_begin.restype=C.c_int
            if l.hffn_train_stage_optimizer_begin(self._handle): self._raise()
    def set_update_scalars(self,values):
        """The seven scalars of the next update: lr, beta1, beta2, eps, weight decay, step_size, beta2_scale (not validated by
        the owner; see `hffn_train_set_update_scalars`)."""
        values=list(values)
        if len(values)!=7: raise ValueError('seven scalars required: lr, beta1, beta2, eps, weight decay, step_size, beta2_scale')
        packed=(C.c_float*7)(*values)
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_set_update_scalars.argtypes=[C.c_void_p,C.POINTER(C.c_float)];l.hffn_train_set_update_scalars.restype=C.c_int
            if l.hffn_train_set_update_scalars(self._handle,packed): self._raise()
    def restore_optimizer(self,first,second,completed):
        """Cold import of the compact moments (float32 bytes, pool size) and of the update counter (see
        `hffn_train_restore_optimizer`); the pool size is checked by the owner."""
        first,second=bytes(first),bytes(second)
        if len(first)!=len(second) or len(first)%4: raise ValueError('moments must be float32 bytes of one size')
        if not isinstance(completed,int) or not 0<=completed<2**32: raise ValueError('invalid update count')
        count=len(first)//4
        m=(C.c_float*count).from_buffer_copy(first);v=(C.c_float*count).from_buffer_copy(second)
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_restore_optimizer.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_uint64,C.c_uint32]
            l.hffn_train_restore_optimizer.restype=C.c_int
            if l.hffn_train_restore_optimizer(self._handle,m,v,count,completed): self._raise()
    def enable_caller_managed_step(self):
        """Cold, irreversible for this context: the caller's step counter alone drives the bias corrections (see
        `hffn_train_enable_caller_managed_step`)."""
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_enable_caller_managed_step.argtypes=[C.c_void_p];l.hffn_train_enable_caller_managed_step.restype=C.c_int
            if l.hffn_train_enable_caller_managed_step(self._handle): self._raise()
    def bind_flat_parameters(self,metal_buffer,byte_offset,count):
        """Cold binding of a caller-owned compact parameter pool (a Private MTLBuffer, e.g. an MPS tensor view) to a fresh
        context. The caller keeps its tensor and synchronises the producer and consumer queues; no upload, no copy."""
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_bind_flat_parameters.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64,C.c_uint64]
            l.hffn_train_bind_flat_parameters.restype=C.c_int
            if l.hffn_train_bind_flat_parameters(self._handle,metal_buffer,byte_offset,count): self._raise()
    def bind_activations(self,x=None,gy=None,gx=None):
        """Cold, once per kind: `x` / `gy` are lists of `(metal_buffer, byte_offset)`, one per accumulation slot, `gx` is one
        `(metal_buffer, byte_offset)`; None keeps that kind native. The caller keeps the buffers alive and synchronises its
        own queue; the context retains no copy (see `hffn_train_bind_activations`)."""
        with self._lock:
            self._check()
            l=self._lib
            l.hffn_train_bind_activations.argtypes=[C.c_void_p,C.c_uint32,C.POINTER(C.c_void_p),C.POINTER(C.c_uint64),
                                                    C.POINTER(C.c_void_p),C.POINTER(C.c_uint64),C.c_void_p,C.c_uint64]
            l.hffn_train_bind_activations.restype=C.c_int
            def pack(items):
                if items is None: return None,None,0
                handles=(C.c_void_p*len(items))(*[h for h,_ in items])
                offsets=(C.c_uint64*len(items))(*[o for _,o in items])
                return handles,offsets,len(items)
            xh,xo,nx=pack(x);gh,go,ng=pack(gy)
            if x is not None and gy is not None and nx!=ng: raise ValueError('x and gy need the same slot count')
            gbuf,goff=gx if gx is not None else (None,0)
            if l.hffn_train_bind_activations(self._handle,nx or ng,xh,xo,gh,go,gbuf,goff): self._raise()
    def close(self):
        with self._lock:
            if self._handle:
                self._lib.hffn_train_destroy(self._handle)
                self._handle=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def __del__(self):
        if getattr(self,'_handle',None): self.close()


def prepare_exact_sources(payload,arguments=()):
    """Only cold source generation; descriptor dimensions stay authoritative."""
    from export_input import checked
    i,h,o,_,_=checked(payload)
    args=list(map(str,arguments))
    forbidden=('--input','--hidden','--output-size','--degree','--export-dir','--support-stdin')
    if any(value in forbidden for value in args):raise ValueError('exact dimensions come from support')
    return prepare_sources([*args,'--input',str(i),'--hidden',str(h),'--output-size',str(o),'--degree','1'])

def from_export(library,directory,arguments=()):
    """Read an existing export, prepare sources and return its actual context."""
    from export_input import read_export
    payload=read_export(directory)
    return Context(library,prepare_exact_sources(payload,arguments),arguments,support=payload)
