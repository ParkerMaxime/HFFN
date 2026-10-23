"""Pythia integration for compact-G4 HFFN training."""
from __future__ import annotations
from pathlib import Path
import time
import torch
from torch import nn
from .compact_export import HFFNCompactG4Layer, load_hffn_compact_g4_layer
from .compact_runtime import HFFNCompactG4TrainingRuntime

class _Fn(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,up,down,bup,bdown,module):
  shape=tuple(x.shape); y,cid=module.runtime.forward_with_context(module.layer_idx,x.reshape(-1,shape[-1]));ctx.module,ctx.shape,ctx.cid=module,shape,cid;return y.reshape(shape)
 @staticmethod
 def backward(ctx,gy):
  d=ctx.module.runtime.backward_with_context(ctx.cid,ctx.module.layer_idx,gy.reshape(-1,gy.shape[-1]));return d["grad_x"].reshape(ctx.shape),d["grad_up"],d["grad_down"],d["grad_b_up"],d["grad_b_down"],None

class HFFNCompactG4MLP(nn.Module):
 def __init__(self,index:int,layer:HFFNCompactG4Layer,runtime:HFFNCompactG4TrainingRuntime):
  super().__init__();self.layer_idx=index;self.runtime=runtime;self.register_buffer("group_hidden_offsets",layer.group_hidden_offsets);self.register_buffer("up_row_offsets",layer.up_row_offsets);self.register_buffer("up_idx",layer.up_idx);self.register_buffer("down_row_offsets",layer.down_row_offsets);self.register_buffer("down_idx",layer.down_idx);self.up_val=layer.up_val;self.down_val=layer.down_val;self.b_up=layer.b_up;self.b_down=layer.b_down
 def forward(self,x):
  if not torch.is_grad_enabled():
   shape=tuple(x.shape);y,cid=self.runtime.forward_with_context(self.layer_idx,x.reshape(-1,shape[-1]));out=y.clone();self.runtime.discard_context(cid);return out.reshape(shape)
  return _Fn.apply(x,self.up_val,self.down_val,self.b_up,self.b_down,self)

def _params(x:HFFNCompactG4Layer)->HFFNCompactG4Layer:
 return HFFNCompactG4Layer(x.d_model,x.d_ff,x.group_hidden_offsets, x.up_row_offsets,x.up_idx,nn.Parameter(x.up_val),nn.Parameter(x.b_up), x.down_row_offsets,x.down_idx,nn.Parameter(x.down_val),nn.Parameter(x.b_down))

def install_pythia_hffn_compact_g4_mlps(model:nn.Module,export_dir:str|Path)->HFFNCompactG4TrainingRuntime:
 blocks=list(getattr(getattr(model,"gpt_neox",None),"layers",()));root=Path(export_dir)
 if not blocks: raise ValueError("expected Pythia GPT-NeoX")
 layers=[_params(load_hffn_compact_g4_layer(root/f"layer{i}"/"hffn_training_compact_g4_v2.pt")) for i in range(len(blocks))]
 runtime=HFFNCompactG4TrainingRuntime(layers)
 for i,(block,layer) in enumerate(zip(blocks,layers)):
  block.mlp=HFFNCompactG4MLP(i,layer,runtime)
 return runtime

class HFFNCompactG4Optimizer:
 def __init__(self,model:nn.Module,runtime:HFFNCompactG4TrainingRuntime,*,lr:float,weight_decay:float):
  self.runtime,self.lr,self.weight_decay=runtime,float(lr),float(weight_decay);hp=[p for b in model.gpt_neox.layers for p in (b.mlp.up_val,b.mlp.down_val,b.mlp.b_up,b.mlp.b_down)];ids={id(p) for p in hp};self.base=torch.optim.AdamW([p for p in model.parameters() if id(p) not in ids],lr=lr,weight_decay=weight_decay);self._hffn_params=hp
 def zero_grad(self,set_to_none:bool=True):
  self.base.zero_grad(set_to_none=set_to_none);self.runtime.zero_grad()
  for p in self._hffn_params:  # torch.optim.Optimizer.zero_grad semantics on the FFN Parameters, as the base AdamW on the others
   if p.grad is None: continue
   if set_to_none: p.grad=None
   else:
    if p.grad.grad_fn is not None: p.grad.detach_()
    else: p.grad.requires_grad_(False)
    p.grad.zero_()
 def step(self):
  self.base.step();self.runtime.adamw_all(lr=self.lr,weight_decay=self.weight_decay)
  torch.autograd.graph.increment_version(self._hffn_params)  # the native AdamW wrote the FFN Parameters through data_ptr()
 def state_dict(self)->dict:
  """Cold whole-optimizer snapshot at a completed update boundary: non-FFN AdamW, lr/decay and the compact runtime."""
  from .compact_checkpoint import optimizer_state_dict
  return optimizer_state_dict(self)
 def load_state_dict(self,payload:dict)->None:
  """Cold restore; the whole payload is validated and staged before the first write."""
  from .compact_checkpoint import load_optimizer_state_dict
  load_optimizer_state_dict(self,payload)
