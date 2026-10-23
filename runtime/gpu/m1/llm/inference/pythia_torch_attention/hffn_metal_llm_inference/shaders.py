"""Forward-only exact-offset Metal shader generator."""

from __future__ import annotations


def inference_source(
    d_model: int,
    d_ff: int,
    n: int,
    down_edge_bits: int,
    up_nnz: int,
    down_nnz: int,
) -> str:
    if min(d_model, d_ff, n, down_edge_bits, up_nnz, down_nnz) < 1:
        raise ValueError("shader constants must be positive")
    return f"""
constant uint I = {d_model}u;
constant uint H = {d_ff}u;
constant uint O = {d_model}u;
constant uint N = {n}u;
constant uint DEB = {down_edge_bits}u;
constant uint DEM = {(1 << down_edge_bits) - 1}u;
constant uint OWNER_TILES = {(n + 127) // 128}u;

inline float erf_accurate(float x) {{
  float ax=metal::abs(x),t=1.0f/(1.0f+0.3275911f*ax);
  float p=(((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t);
  float y=1.0f-p*metal::exp(-ax*ax);
  return x<0.0f?-y:y;
}}
inline float gelu(float x) {{ return 0.5f*x*(1.0f+erf_accurate(x*0.7071067811865475f)); }}

kernel void up_token_major_inference(
    device const float* x, device const int* offsets,
    device const short* index, device const float* weight,
    device const float* bias, device float* act,
    uint gid [[thread_position_in_grid]]) {{
  if(gid>=H*N) return;
  uint r=gid/N,n=gid-r*N; float sum=bias[r];
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]);
  for(uint p=begin;p<end;++p) sum+=weight[p]*x[n*I+uint(index[p])];
  act[gid]=gelu(sum);
}}

kernel void down_token_major_inference(
    device const float* act, device const int* owner_offsets,
    device const short* route_hidden,
    device const float* weight, device const float* bias, device float* y,
    uint gid [[thread_position_in_grid]]) {{
  if(gid>=N*O) return;
  uint n=gid/O,o=gid-n*O; float sum=bias[o];
  uint begin=uint(owner_offsets[o]),end=uint(owner_offsets[o+1]);
  for(uint q=begin;q<end;++q) {{
    uint r=uint(route_hidden[q]); sum+=weight[q]*act[r*N+n];
  }}
  y[gid]=sum;
}}

// Small-N inference specialization: one SIMD owns one hidden/output row and
// lanes cooperatively reduce its exact irregular edge list.  This restores
// lane occupancy during KV-cache decode without padding or duplicated values.
kernel void up_owner_row_reduce_inference(
    device const float* x, device const int* offsets,
    device const short* index, device const float* weight,
    device const float* bias, device float* act,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,r=tg*16u+(tid>>5u); if(r>=H) return;
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]);
  for(uint n=0;n<N;++n) {{
    float partial=0.0f;
    for(uint p=begin+lane;p<end;p+=32u) partial+=weight[p]*x[n*I+uint(index[p])];
    float sum=metal::simd_sum(partial);
    if(lane==0u) act[r*N+n]=gelu(bias[r]+sum);
  }}
}}

kernel void down_owner_row_reduce_inference(
    device const float* act, device const int* owner_offsets,
    device const short* route_hidden,
    device const float* weight, device const float* bias, device float* y,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,o=tg*16u+(tid>>5u); if(o>=O) return;
  uint begin=uint(owner_offsets[o]),end=uint(owner_offsets[o+1]);
  for(uint n=0;n<N;++n) {{
    float partial=0.0f;
    for(uint q=begin+lane;q<end;q+=32u) {{
      uint r=uint(route_hidden[q]); partial+=weight[q]*act[r*N+n];
    }}
    float sum=metal::simd_sum(partial);
    if(lane==0u) y[n*O+o]=bias[o]+sum;
  }}
}}

kernel void up_feature_persistent_inference(
    device const float* x, device const int* offsets,
    device const short* index, device const float* weight,
    device const float* bias, device float* act,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,r=tg*16u+(tid>>5u); if(r>=H) return;
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]);
  for(uint n=lane;n<N;n+=32u) {{
    float sum=bias[r];
    for(uint p=begin;p<end;++p) sum+=weight[p]*x[uint(index[p])*N+n];
    act[r*N+n]=gelu(sum);
  }}
}}

kernel void down_feature_persistent_inference(
    device const float* act, device const int* owner_offsets,
    device const short* route_hidden,
    device const float* weight, device const float* bias, device float* y,
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {{
  uint lane=tid&31u,o=tg*16u+(tid>>5u); if(o>=O) return;
  uint begin=uint(owner_offsets[o]),end=uint(owner_offsets[o+1]);
  for(uint n=lane;n<N;n+=32u) {{
    float sum=bias[o];
    for(uint q=begin;q<end;++q) {{
      uint r=uint(route_hidden[q]); sum+=weight[q]*act[r*N+n];
    }}
    y[o*N+n]=sum;
  }}
}}

kernel void up_feature_owner_tile_inference(
    device const float* x, device const int* offsets,
    device const short* index, device const float* weight,
    device const float* bias, device float* act,
    uint tid [[thread_position_in_threadgroup]],
    uint flat_tg [[threadgroup_position_in_grid]]) {{
  constexpr uint groups=(H+15u)/16u;
  uint lane=tid&31u,simd=tid>>5u,tile=flat_tg/groups;
  uint owner_group=flat_tg-tile*groups,r=owner_group*16u+simd;
  if(r>=H||tile>=OWNER_TILES) return;
  uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;
  float a0=bias[r],a1=a0,a2=a0,a3=a0;
  uint begin=uint(offsets[r]),end=uint(offsets[r+1]);
  for(uint p=begin;p<end;++p) {{
    uint i=uint(index[p]); float w=weight[p];
    if(n0<N) a0+=w*x[i*N+n0]; if(n1<N) a1+=w*x[i*N+n1];
    if(n2<N) a2+=w*x[i*N+n2]; if(n3<N) a3+=w*x[i*N+n3];
  }}
  if(n0<N) act[r*N+n0]=gelu(a0); if(n1<N) act[r*N+n1]=gelu(a1);
  if(n2<N) act[r*N+n2]=gelu(a2); if(n3<N) act[r*N+n3]=gelu(a3);
}}

kernel void down_feature_owner_tile_inference(
    device const float* act, device const int* owner_offsets,
    device const short* route_hidden,
    device const float* weight, device const float* bias, device float* y,
    uint tid [[thread_position_in_threadgroup]],
    uint flat_tg [[threadgroup_position_in_grid]]) {{
  constexpr uint groups=(O+15u)/16u;
  uint lane=tid&31u,simd=tid>>5u,tile=flat_tg/groups;
  uint owner_group=flat_tg-tile*groups,o=owner_group*16u+simd;
  if(o>=O||tile>=OWNER_TILES) return;
  uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;
  float a0=bias[o],a1=a0,a2=a0,a3=a0;
  uint begin=uint(owner_offsets[o]),end=uint(owner_offsets[o+1]);
  for(uint q=begin;q<end;++q) {{
    uint r=uint(route_hidden[q]); float w=weight[q];
    if(n0<N) a0+=w*act[r*N+n0]; if(n1<N) a1+=w*act[r*N+n1];
    if(n2<N) a2+=w*act[r*N+n2]; if(n3<N) a3+=w*act[r*N+n3];
  }}
  if(n0<N) y[o*N+n0]=a0; if(n1<N) y[o*N+n1]=a1;
  if(n2<N) y[o*N+n2]=a2; if(n3<N) y[o*N+n3]=a3;
}}
"""
