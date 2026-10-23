"""Metal source generators for adaptive exact-offset Pythia training."""

from __future__ import annotations


def exact_offset_source(
    d_model: int,
    d_ff: int,
    n: int,
    up_edge_bits: int,
    down_edge_bits: int,
    up_nnz: int,
    down_nnz: int,
) -> str:
    if min(d_model, d_ff, n, up_edge_bits, down_edge_bits, up_nnz, down_nnz) < 1:
        raise ValueError("shader constants must be positive")
    return f"""
constant uint I = {d_model}u;
constant uint H = {d_ff}u;
constant uint O = {d_model}u;
constant uint N = {n}u;
constant uint UEB = {up_edge_bits}u;
constant uint DEB = {down_edge_bits}u;
constant uint UEM = {(1 << up_edge_bits) - 1}u;
constant uint DEM = {(1 << down_edge_bits) - 1}u;
constant uint UN = {up_nnz}u;
constant uint DN = {down_nnz}u;
constant uint OWNER_TILE_N = 128u;
constant uint OWNER_TILES = {(n + 127) // 128}u;

inline float erf_accurate(float x) {{
  float ax=metal::abs(x),t=1.0f/(1.0f+0.3275911f*ax);
  float p=(((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t);
  float y=1.0f-p*metal::exp(-ax*ax);
  return x<0.0f?-y:y;
}}
inline float gelu_grad(float x) {{
  return 0.5f*(1.0f+erf_accurate(x*0.7071067811865475f))+x*0.3989422804014327f*metal::exp(-0.5f*x*x);
}}

// PyTorch boundary layout is token-major [N,feature]. Dense hidden state is
// feature-major [H,N], preserving coalesced token traversal.
kernel void hidden_backward(
    device const float* gy, device const float* z,
    device const int* offsets, device const short* index,
    device const float* weight, device float* gz,
    uint gid [[thread_position_in_grid]]) {{
  if (gid >= H*N) return;
  uint r=gid/N,n=gid-r*N; float sum=0.0f;
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]);
  for(uint p=begin;p<end;++p) sum+=weight[p]*gy[n*O+uint(index[p])];
  gz[gid]=sum*gelu_grad(z[gid]);
}}

kernel void grad_down(
    device const float* gy, device const float* act,
    device const short* slot_hidden, device const short* index,
    device float* grad, uint p [[thread_position_in_grid]]) {{
  uint r=uint(slot_hidden[p]),o=uint(index[p]); float sum=0.0f;
  for(uint n=0;n<N;++n) sum+=act[r*N+n]*gy[n*O+o];
  grad[p]=sum;
}}

kernel void grad_up(
    device const float* gz, device const float* x,
    device const short* slot_hidden, device const short* index,
    device float* grad, uint p [[thread_position_in_grid]]) {{
  uint r=uint(slot_hidden[p]),i=uint(index[p]); float sum=0.0f;
  for(uint n=0;n<N;++n) sum+=gz[r*N+n]*x[n*I+i];
  grad[p]=sum;
}}

kernel void grad_b_up(
    device const float* gz, device float* grad,
    uint r [[thread_position_in_grid]]) {{
  if (r>=H) return; float sum=0.0f;
  for(uint n=0;n<N;++n) sum+=gz[r*N+n]; grad[r]=sum;
}}

kernel void grad_b_down(
    device const float* gy, device float* grad,
    uint o [[thread_position_in_grid]]) {{
  if (o>=O) return; float sum=0.0f;
  for(uint n=0;n<N;++n) sum+=gy[n*O+o]; grad[o]=sum;
}}

kernel void grad_x_token_major(
    device const float* gz, device const int* owner_offsets,
    device const int* packed, device const int* row_offsets,
    device const float* weight, device float* gx,
    uint gid [[thread_position_in_grid]]) {{
  if (gid>=N*I) return;
  uint n=gid/I,i=gid-n*I; float sum=0.0f;
  uint begin=uint(owner_offsets[i]),end=uint(owner_offsets[i+1]);
  for(uint q=begin;q<end;++q) {{
    uint code=uint(packed[q]),r=code>>UEB,e=code&UEM;
    uint p=uint(row_offsets[r])+e; sum+=weight[p]*gz[r*N+n];
  }}
  gx[gid]=sum;
}}

// Exact-offset adaptation of the persistent-N schedule.  One SIMD
// group owns one hidden/output/input owner and lanes traverse N contiguously.
kernel void hidden_backward_feature_persistent(
    device const float* gy, device const float* z,
    device const int* offsets, device const short* index,
    device const float* weight, device float* gz, device float* grad_bias,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,r=tg*16u+(tid>>5u); if(r>=H) return;
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]); float bias_sum=0.0f;
  for(uint n=lane;n<N;n+=32u) {{
    float sum=0.0f;
    for(uint p=begin;p<end;++p) sum+=weight[p]*gy[uint(index[p])*N+n];
    float value=sum*gelu_grad(z[r*N+n]); gz[r*N+n]=value; bias_sum+=value;
  }}
  float reduced=metal::simd_sum(bias_sum); if(lane==0u) grad_bias[r]=reduced;
}}

kernel void down_grad_feature_persistent(
    device const float* act, device const float* gy,
    device const int* owner_offsets, device const int* packed,
    device const int* row_offsets, device float* grad_weight,
    device float* grad_bias,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,o=tg*16u+(tid>>5u); if(o>=O) return;
  float bias_sum=0.0f;
  for(uint n=lane;n<N;n+=32u) bias_sum+=gy[o*N+n];
  float bias_reduced=metal::simd_sum(bias_sum); if(lane==0u) grad_bias[o]=bias_reduced;
  uint begin=uint(owner_offsets[o]),end=uint(owner_offsets[o+1]);
  for(uint q=begin;q<end;++q) {{
    uint code=uint(packed[q]),r=code>>DEB,e=code&DEM;
    uint p=uint(row_offsets[r])+e; float sum=0.0f;
    for(uint n=lane;n<N;n+=32u) sum+=act[r*N+n]*gy[o*N+n];
    float reduced=metal::simd_sum(sum); if(lane==0u) grad_weight[p]=reduced;
  }}
}}

kernel void input_grad_feature_persistent(
    device const float* x, device const float* gz,
    device const int* owner_offsets, device const int* packed,
    device const int* row_offsets, device const float* weight,
    device float* grad_weight, device float* gx,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,i=tg*16u+(tid>>5u); if(i>=I) return;
  uint begin=uint(owner_offsets[i]),end=uint(owner_offsets[i+1]);
  for(uint n=lane;n<N;n+=32u) {{
    float sum=0.0f;
    for(uint q=begin;q<end;++q) {{
      uint code=uint(packed[q]),r=code>>UEB,e=code&UEM;
      uint p=uint(row_offsets[r])+e; sum+=weight[p]*gz[r*N+n];
    }}
    gx[i*N+n]=sum;
  }}
  for(uint q=begin;q<end;++q) {{
    uint code=uint(packed[q]),r=code>>UEB,e=code&UEM;
    uint p=uint(row_offsets[r])+e; float sum=0.0f;
    for(uint n=lane;n<N;n+=32u) sum+=gz[r*N+n]*x[i*N+n];
    float reduced=metal::simd_sum(sum); if(lane==0u) grad_weight[p]=reduced;
  }}
}}

// Exact-offset owner-tile backward. Every compact parameter gets one partial
// per token tile; the final reduction writes the flat compact gradient.
kernel void hidden_backward_owner_tile(
    device const float* gy, device const float* z,
    device const int* offsets, device const short* index,
    device const float* weight, device float* gz, device float* partial,
    uint tid [[thread_position_in_threadgroup]],
    uint flat_tg [[threadgroup_position_in_grid]]) {{
  constexpr uint groups=(H+15u)/16u;
  uint lane=tid&31u,simd=tid>>5u,tile=flat_tg/groups;
  uint owner_group=flat_tg-tile*groups,r=owner_group*16u+simd;
  if(r>=H||tile>=OWNER_TILES) return;
  uint base=tile*OWNER_TILE_N,end=metal::min(N,base+OWNER_TILE_N);
  uint begin=uint(offsets[r]),edge_end=uint(offsets[r+1]); float bias_sum=0.0f;
  for(uint n=base+lane;n<end;n+=32u) {{
    float sum=0.0f;
    for(uint p=begin;p<edge_end;++p) sum+=weight[p]*gy[uint(index[p])*N+n];
    float value=sum*gelu_grad(z[r*N+n]); gz[r*N+n]=value; bias_sum+=value;
  }}
  float reduced=metal::simd_sum(bias_sum);
  if(lane==0u) partial[(UN+DN+r)*OWNER_TILES+tile]=reduced;
}}

kernel void down_grad_owner_tile(
    device const float* act, device const float* gy,
    device const int* owner_offsets, device const int* packed,
    device const int* row_offsets, device float* partial,
    uint tid [[thread_position_in_threadgroup]],
    uint flat_tg [[threadgroup_position_in_grid]]) {{
  constexpr uint groups=(O+15u)/16u;
  uint lane=tid&31u,simd=tid>>5u,tile=flat_tg/groups;
  uint owner_group=flat_tg-tile*groups,o=owner_group*16u+simd;
  if(o>=O||tile>=OWNER_TILES) return;
  uint base=tile*OWNER_TILE_N,end=metal::min(N,base+OWNER_TILE_N); float bias_sum=0.0f;
  for(uint n=base+lane;n<end;n+=32u) bias_sum+=gy[o*N+n];
  float bias_reduced=metal::simd_sum(bias_sum);
  if(lane==0u) partial[(UN+DN+H+o)*OWNER_TILES+tile]=bias_reduced;
  uint begin=uint(owner_offsets[o]),edge_end=uint(owner_offsets[o+1]);
  for(uint q=begin;q<edge_end;++q) {{
    uint code=uint(packed[q]),r=code>>DEB,e=code&DEM;
    uint p=uint(row_offsets[r])+e; float sum=0.0f;
    for(uint n=base+lane;n<end;n+=32u) sum+=act[r*N+n]*gy[o*N+n];
    float reduced=metal::simd_sum(sum);
    if(lane==0u) partial[(UN+p)*OWNER_TILES+tile]=reduced;
  }}
}}

kernel void input_grad_owner_tile(
    device const float* x, device const float* gz,
    device const int* owner_offsets, device const int* packed,
    device const int* row_offsets, device const float* weight,
    device float* partial, device float* gx,
    uint tid [[thread_position_in_threadgroup]],
    uint flat_tg [[threadgroup_position_in_grid]]) {{
  constexpr uint groups=(I+15u)/16u;
  uint lane=tid&31u,simd=tid>>5u,tile=flat_tg/groups;
  uint owner_group=flat_tg-tile*groups,i=owner_group*16u+simd;
  if(i>=I||tile>=OWNER_TILES) return;
  uint base=tile*OWNER_TILE_N,end=metal::min(N,base+OWNER_TILE_N);
  uint begin=uint(owner_offsets[i]),edge_end=uint(owner_offsets[i+1]);
  for(uint n=base+lane;n<end;n+=32u) {{
    float sum=0.0f;
    for(uint q=begin;q<edge_end;++q) {{
      uint code=uint(packed[q]),r=code>>UEB,e=code&UEM;
      uint p=uint(row_offsets[r])+e; sum+=weight[p]*gz[r*N+n];
    }}
    gx[i*N+n]=sum;
  }}
  for(uint q=begin;q<edge_end;++q) {{
    uint code=uint(packed[q]),r=code>>UEB,e=code&UEM;
    uint p=uint(row_offsets[r])+e; float sum=0.0f;
    for(uint n=base+lane;n<end;n+=32u) sum+=gz[r*N+n]*x[i*N+n];
    float reduced=metal::simd_sum(sum);
    if(lane==0u) partial[p*OWNER_TILES+tile]=reduced;
  }}
}}

kernel void reduce_owner_tile_grad(
    device const float* partial, device float* grad,
    uint p [[thread_position_in_grid]]) {{
  if(p>=UN+DN+H+O) return; float sum=0.0f;
  for(uint tile=0;tile<OWNER_TILES;++tile) sum+=partial[p*OWNER_TILES+tile];
  grad[p]=sum;
}}
"""


def adamw_source() -> str:
    """Compact AdamW formula of the GPU FFN runtime."""
    return """
kernel void adamw_compact(
    device float* param, device float* grad,
    device float* moment1, device float* moment2,
    constant float& lr, constant float& beta1, constant float& beta2,
    constant float& eps, constant float& weight_decay,
    constant float& step_size, constant float& beta2_scale,
    uint gid [[thread_position_in_grid]]) {
  float g=grad[gid];
  if (!metal::isfinite(g)) g=0.0f;
  float m=moment1[gid]+(1.0f-beta1)*(g-moment1[gid]);
  float v=beta2*moment2[gid];
  v+=(1.0f-beta2)*g*g;
  moment1[gid]=m; moment2[gid]=v;
  float p=param[gid]*(1.0f-lr*weight_decay);
  float denom=metal::sqrt(v)/beta2_scale+eps;
  p-=step_size*(m/denom);
  param[gid]=p;
  grad[gid]=0.0f;
}
"""
