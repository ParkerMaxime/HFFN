"""Once-per-batch cold API refusal suites, copied with source provenance.

Context lifecycle faults and checkpoint faults: each one must be refused by the shared module under test.
"""
import copy
import math
import torch
import gates
from hffn_ffn_training import contexts as ctx, checkpoint as cp
from hffn_ffn_training.module import HFFNTrainFFN
from capabilities import bits, layer_equal
P=('up_val','down_val','b_up','b_down')
G=('grad_up','grad_down','grad_b_up','grad_b_down')
def snapshots(m):
 values={p:getattr(m,p) for p in P}
 values.update({'moment'+str(i):v for i,v in enumerate(m._step.moments)})
 values.update({'support/'+k:v for k,v in vars(m._step.layer).items() if isinstance(v,torch.Tensor) and k not in P})
 values.update({f'slot{i}/{k}':v for i,s in enumerate(m._slots) for k,v in s['buffers'].items()})
 values.update({'scratch/'+k:v[0] for k,v in m._step._reserved.items()})
 return {k:(v.data_ptr(),v.dtype,tuple(v.shape)) for k,v in values.items()}

def exact(a,b):
 return set(a)==set(b) and all(bits(v,b[k]) for k,v in a.items())

def direct(step,x,gy,need):
 step.resize(x.shape[0]);step.forward(x);step.backward(gy,input_grad=need)
 out={k:v.clone() for k,v in step.results().items() if k!='grad_x' or need}
 return out

def grads(m):
 return {key:getattr(m,p).grad for key,p in zip(G,P)}

def context_faults(layer,kernels):
 checks={}
 def fresh(cap=1):return ctx.HFFNContextFFN(layer,129,n_min=8,contexts=cap,kernels=kernels)
 def reject(name,fn,message,exc=RuntimeError):
  try:fn();checks[name]=False
  except exc as e:checks[name]=message in str(e)
 x,gy=gates.inputs(layer,8,99)
 for capacity in (0,-1,True,1.5):reject('capacity_'+str(capacity),lambda:fresh(capacity),'positive integer',ValueError)
 m=fresh();reject('missing_lease',lambda:m(x),'acquire a context')
 for index in (-1,1,True,1.5):reject('index_'+str(index),lambda:m.acquire_context(index),'invalid context index',ValueError)
 m._step._reserved['gx_nt'][0].fill_(12345)
 l=m.acquire_context();y=m(x,context=l)
 saved={k:v.clone() for k,v in m._slots[0]['buffers'].items()};ptrs=snapshots(m)
 reject('exhaustion',lambda:m.acquire_context(),'reservation exhausted')
 reject('occupied_index',lambda:m.acquire_context(0),'already leased')
 reject('consumed',lambda:m(x+1,context=l),'one forward')
 checks['no_overwrite_on_refusal']=all(bits(v,m._slots[0]['buffers'][k]) for k,v in saved.items())
 with torch.no_grad():z=m(x+2)
 checks['no_grad_live_preserved']=not z.requires_grad and all(bits(v,m._slots[0]['buffers'][k]) for k,v in saved.items())
 for name,args in [('resize',(31,)),('forward',(x,)),('backward',(gy,)),('update',()),('step',(x,gy)),('results',())]:
  reject('direct_live_'+name,lambda name=name,args=args:getattr(m.step,name)(*args),'release saved contexts')
 y.backward(gy,retain_graph=True);first={k:v.clone() for k,v in grads(m).items()};gx=m._step.gx_nt.clone()
 y.backward(gy,retain_graph=True)
 checks['retain_graph_repeat']=all(bits(v,2*first[k]) for k,v in grads(m).items())
 checks['weights_only_sentinel']=bool(torch.equal(gx,m._step.gx_nt)) and bool((m._step._reserved['gx_nt'][0]==12345).all())
 l.release();l.release();old=y
 reject('released_lease',lambda:m(x,context=l),'foreign or released')
 l2=m.acquire_context();y2=m(x+1,context=l2)
 reject('stale_backward',lambda:old.backward(gy),'stale backward')
 y2.backward(gy);l2.release();checks['reuse_old_output_retained']=old is y and snapshots(m)==ptrs
 l3=m.acquire_context();abandoned=m(x,context=l3);del abandoned;l3.release()
 with m.acquire_context() as l4:m(x,context=l4).backward(gy)
 checks['abandoned_release']=not m._slots[0]['leased']
 other=fresh();foreign=other.acquire_context();mine=m.acquire_context()
 reject('foreign',lambda:m(x,context=foreign),'foreign or released');foreign.release();mine.release()
 for kind in ('torch','checkpoint'):
  m=fresh();l=m.acquire_context();y=m(x,context=l)
  if kind=='torch':
   with torch.no_grad():m.up_val.add_(1)
  else:cp.copy_parameters(m,cp.export_checkpoint(m))
  reject(kind+'_version',lambda:y.backward(gy),'modified by an inplace operation');l.release()
 m=fresh();l=m.acquire_context();y=m(x,context=l);m._step.steps+=1
 reject('native_count',lambda:y.backward(gy),'native update');l.release()
 m=fresh();l=m.acquire_context();y=m(x,context=l)
 m.up_val=torch.nn.Parameter(m.up_val.detach().clone())
 reject('storage',lambda:y.backward(gy),'no longer shares');l.release()
 m=fresh();l=m.acquire_context()
 reject('dtype',lambda:m(x.double(),context=l),'float32',TypeError)
 reject('tokens',lambda:m(x[:1],context=l),'token count',ValueError)
 reject('width',lambda:m(x[:,:-1],context=l),'feature width',ValueError);l.release()
 m._lock.acquire()
 try:
  reject('concurrent_acquire',lambda:m.acquire_context(),'concurrent')
  with torch.no_grad():reject('concurrent_forward',lambda:m(x),'concurrent')
 finally:m._lock.release()
 with m.acquire_context() as l:
  y=m(x,context=l);m._lock.acquire()
  try:reject('concurrent_backward',lambda:y.backward(gy,retain_graph=True),'concurrent')
  finally:m._lock.release()
  y.backward(gy)
 reject('direct_backward_without_forward',lambda:m.step.backward(gy),'direct Step forward')
 reject('direct_results_without_forward',lambda:m.step.results(),'direct Step results')
 m.step.resize(8);m.step.forward(x);m.step.backward(gy,input_grad=False)
 ref=gates.Step(layer,8,*kernels);direct(ref,x,gy,False)
 checks['direct_after_release']=all(bits(getattr(m.step,k),getattr(ref,k)) for k in ('g_up','g_down','g_b_up','g_b_down'))
 checks['direct_results_coherent']=exact(m.step.results(),ref.results())
 m.step.update();checks['direct_results_after_update']=exact(m.step.results(),ref.results())
 for frozen in [(p,) for p in P]+[P]:
  m=fresh()
  for p in frozen:getattr(m,p).requires_grad_(False)
  inp=x.clone().requires_grad_(True)
  with m.acquire_context() as l:m(inp,context=l).backward(gy)
  checks['frozen_'+'_'.join(frozen)]=all((getattr(m,p).grad is None) if p in frozen else bits(getattr(m,p).grad,getattr(ref,k)) for p,k in zip(P,('g_up','g_down','g_b_up','g_b_down'))) and bool(torch.isfinite(inp.grad).all())
 m=fresh()
 for p in P:getattr(m,p).requires_grad_(False)
 checks['all_frozen_no_lease']=not m(x).requires_grad and not any(s['leased'] for s in m._slots)
 m=fresh(2);a,b=m.acquire_context(),m.acquire_context();ya=m(x,context=a);yb=m(x+1,context=b)
 good={k:v.clone() for k,v in m._slots[0]['buffers'].items()}
 m._slots[0]['buffers']['z'].copy_(m._slots[1]['buffers']['z'])
 ya.backward(gy);wrong=any(not bits(getattr(m,p).grad,getattr(ref,k)) for p,k in zip(P,('g_up','g_down','g_b_up','g_b_down')))
 checks['detect_slot_mixing']=wrong
 a.release();b.release()
 m=fresh();a=m.acquire_context();y=m(x,context=a);m._slots[0]['generation']+=1
 reject('detect_generation_fault',lambda:y.backward(gy),'stale backward')
 return checks

def call(m,x,gy,need):
 m.zero_grad(set_to_none=True)
 leaf=x.clone().requires_grad_(need)
 y=m(leaf); y.backward(gy)
 out={'y':y.detach(),'z':m.step.z.T,'h':m.step.act.T}
 out.update({k:getattr(m,p).grad for k,p in zip(G,P)})
 if need:out['grad_x']=leaf.grad
 return out

def checkpoint_faults(layer,kernels):
 m=HFFNTrainFFN(layer,129,n_min=8,kernels=kernels)
 x,gy=gates.inputs(layer,8,17);call(m,x,gy,False);m.step.update()
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4);opt.step()
 payload=cp.export_checkpoint(m)
 out={}
 def rejected(name,edit,loader=cp.from_checkpoint):
  bad=copy.deepcopy(payload); edit(bad)
  try:loader(bad)
  except (ValueError,TypeError,RuntimeError):out[name]=True
  else:out[name]=False
 rejected('format',lambda p:p.update(format='wrong'))
 rejected('missing_key',lambda p:p['state_dict'].pop('b_down'))
 rejected('extra_key',lambda p:p.update(extra=0))
 rejected('float_index',lambda p:p['state_dict'].update(up_idx=p['state_dict']['up_idx'].float()))
 rejected('negative_index',lambda p:p['state_dict']['up_idx'].__setitem__(0,-1))
 rejected('large_index',lambda p:p['state_dict']['up_idx'].__setitem__(0,p['d_model']))
 rejected('offset_overflow',lambda p:p['state_dict']['up_ptr'].__setitem__(1,1<<40))
 rejected('offset_endpoint',lambda p:p['state_dict']['up_ptr'].__setitem__(-1,0))
 rejected('offset_order',lambda p:p['state_dict']['up_ptr'].__setitem__(2,0))
 rejected('nonfinite',lambda p:p['state_dict']['b_down'].__setitem__(0,float('nan')))
 rejected('parameter_dtype',lambda p:p['state_dict'].update(up_val=p['state_dict']['up_val'].double()))
 rejected('non_cpu_tensor',lambda p:p['state_dict'].update(up_val=torch.empty(p['state_dict']['up_val'].shape,device='meta')))
 rejected('non_strided_tensor',lambda p:p['state_dict'].update(up_val=torch.sparse_coo_tensor(torch.tensor([[0]]),torch.ones(1),p['state_dict']['up_val'].shape)))
 rejected('parameter_shape',lambda p:p['state_dict'].update(up_val=p['state_dict']['up_val'][:-1]))
 rejected('groups_shape',lambda p:p.update(groups=torch.tensor([0,p['d_ff']],dtype=torch.int32)))
 rejected('wrapped_groups',lambda p:p.update(groups=p['groups'].long()+(1<<32)))
 rejected('groups_endpoint',lambda p:p['groups'].__setitem__(-1,p['d_ff']-1))
 def dimension(p):
  p['state_dict']['b_down']=torch.zeros(32768);p['d_model']=32768
 rejected('dimension_bound',dimension)
 def hidden_dimension(p):
  p['state_dict']['b_up']=torch.zeros(32768);p['d_ff']=32768
 rejected('hidden_dimension_bound',hidden_dimension)
 # Invalid export originals must be checked before their narrowing conversion.
 for key,value in (('up_idx',m.step.layer.up_idx.float()),('groups',m.step.layer.groups.long()+(1<<32))):
  saved=getattr(m.step.layer,key)
  try:
   setattr(m.step.layer,key,value)
   try:cp.export_checkpoint(m)
   except (ValueError,TypeError):out['export_'+key]=True
   else:out['export_'+key]=False
  finally:setattr(m.step.layer,key,saved)
 def unchanged_after(name,bad):
  before=[p.clone() for p in m.parameters()]; ids=[p.data_ptr() for p in m.parameters()]
  versions=[p._version for p in m.parameters()]
  state=[(v,v.clone(),v.data_ptr()) for v in m.step.moments]+[(v,v.clone(),v.data_ptr()) for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
  steps=m.step.steps
  try:cp.copy_parameters(m,bad)
  except (ValueError,TypeError,RuntimeError):failed=True
  else:failed=False
  out[name]=failed and ids==[p.data_ptr() for p in m.parameters()] and versions==[p._version for p in m.parameters()] and all(bits(a,b) for a,b in zip(before,m.parameters())) and steps==m.step.steps and all(bits(v,w) and v.data_ptr()==ptr for v,w,ptr in state)
 bad=copy.deepcopy(payload);bad['state_dict']['up_val'].add_(.1);bad['state_dict']['b_down'][0]=float('nan')
 unchanged_after('atomic_invalid_last',bad)
 bad=copy.deepcopy(payload);bad['state_dict']['up_val'].add_(.1);bad['state_dict']['up_idx'][0]=(int(bad['state_dict']['up_idx'][0])+1)%layer.d_model
 unchanged_after('atomic_changed_support',bad)
 bad=copy.deepcopy(payload);bad['state_dict']['up_val'].add_(.1);bad['groups'][1]=0
 unchanged_after('atomic_changed_partition',bad)
 # Alias a value source to another destination parameter of equal length.
 state=payload['state_dict'].copy(); before={p:getattr(m,p).detach().clone() for p in P}
 if m.up_val.numel()!=m.down_val.numel():raise RuntimeError('fault fixture must have equal up/down counts')
 state.update(up_val=m.down_val.detach(),down_val=m.up_val.detach())
 cp.copy_parameters(m,state)
 out['aliased_swap']=bits(m.up_val,before['down_val']) and bits(m.down_val,before['up_val'])
 # Expanded/overlapping source must be staged before writes as well.
 state=payload['state_dict'].copy();state['up_val']=m.up_val[:1].expand_as(m.up_val)
 expected=state['up_val'].clone();cp.copy_parameters(m,state)
 out['overlapping_source']=bits(m.up_val,expected)
 cp.copy_parameters(m,payload)
 # Saved autograd version must reject an in-place checkpoint load.
 x,gy=gates.inputs(layer,8,17); y=m(x);cp.copy_parameters(m,payload)
 try:y.backward(gy)
 except RuntimeError as e:out['version_guard']='modified by an inplace operation' in str(e) or 'modified by an inplace' in str(e)
 else:out['version_guard']=False
 # The ordinary identity predicate must reject actual bit/group corruption.
 r=cp.from_checkpoint(payload);r.up_val.view(torch.int32)[0]^=1
 out['value_fault']=not layer_equal(m.step.layer,r)
 r=cp.from_checkpoint(payload);r.groups[1]=r.groups[0]
 out['partition_fault']=not layer_equal(m.step.layer,r)
 return out

def configuration_faults(layer,kernels):
    from hffn_ffn_training.step import Step
    out={}
    invalid=(float('nan'),float('inf'),-float('inf'),-1.,1e40,1e-300,True,'0.1',None,1j)
    class Poison:
        def clone(self): raise RuntimeError('invalid constructor reached clone')
    for name,ctor in [('step',Step),('module',HFFNTrainFFN),('context',ctx.HFFNContextFFN)]:
        for field in ('lr','weight_decay'):
            for i,value in enumerate(invalid):
                try:
                    kw={field:value}
                    if name=='step':ctor(Poison(),8,None,None,**kw)
                    else:ctor(Poison(),8,kernels=kernels,**({'contexts':1} if name=='context' else {}),**kw)
                except (TypeError,ValueError):out[f'invalid/{name}/{field}/{i}']=True
                else:out[f'invalid/{name}/{field}/{i}']=False
    context_module=ctx.HFFNContextFFN(layer,8,contexts=1,kernels=kernels)
    for name,owner in [('step',Step(layer,8,*kernels)),('context',context_module.step)]:
        for field in ('lr','weight_decay'):
            prior=getattr(owner,field)
            try:setattr(owner,field,.2)
            except AttributeError:out[f'readonly/{name}/{field}']=getattr(owner,field)==prior
            else:out[f'readonly/{name}/{field}']=False
    return out


def run(layer,kernels):
    for label,fn in [('contexts',context_faults),('checkpoint',checkpoint_faults),('configuration',configuration_faults)]:
        checks=fn(layer,kernels)
        yield dict(gate='refusals',suite=label,checks=checks,passed=bool(checks) and all(v is True for v in checks.values()))
