#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <stdexcept>
#include "native_train_context_api.h"
#include "hffn_m1_activation_generated.h"
#include "hffn_m1_adamw_generated.h"
#include "hffn_m1_fused_edges_generated.h"
#include "hffn_m1_backward_sums_generated.h"
#include "hffn_m1_backward_plan_generated.h"

#include "hffn_gelu_erf_accurate.h"
#include "hffn_route_codec.h"
#include "native_exact_support.h"

// One self-contained erf-GELU training step. All active parameter data and
// route metadata stay compact; x/z/h/y and activation gradients stay dense
// [feature,N].
struct Args {
  uint32_t input = 128, hidden = 512, output = 128, n = 64, degree = 13,
           warmup = 30, repeats = 200, accum_steps = 1, verify_steps = 0,
           owner_tile_n = 128, packet_up_block = 0, packet_down_block = 0;
  uint32_t forward_owner_tile_n = 0;
  bool untimed = false;
  bool exact_support = false;
  uint32_t up_count=0, down_count=0;
  bool flat_adamw_pool = true;
  bool llm_adamw_policy = false;
  bool compact_indices = true;
  bool wide_route = false;
  // Immutable data is uploaded through Shared staging then kept Private.
  bool private_readonly = true;
  bool tiled_n = false;
  // Owner-tile keeps one SIMD group per HFFN owner, but exposes batch tiles as
  // the Y dispatch dimension.  It is deliberately distinct from --tiled-n,
  // whose scalar edge-tile kernels are retained only as a historical control.
  bool owner_tile = false;
  bool persistent_n = false;
  bool forward_owner_tile = false;
  bool persistent_forward = false;
  bool fup_fused = false;
  bool fdown_fused = false;
  bool fdown_unfused = false;
  // Opt-in, non-default forward-only inverse route: entries are local to one of the
  // four canonical hidden groups, restoring a 16-bit slot representation when
  // the global H*D route no longer fits in 16 bits (e.g. H=2048,D=51).
  bool fdown_segmented_route = false;
  // Exact ownership-preserving realization of a hidden-group forward fusion:
  // four private [output,N] contributions followed by one deterministic sum.
  // This is a labelled control, not the production route: it quantifies the unavoidable dense
  // activation reduction needed to avoid output atomics on irregular support.
  bool fdown_group_private = false;
  // Exact offline layout transform: permute hidden rows by their leading up
  // input (then down output) while moving both projections and b_up together.
  bool hidden_permute_up = false;
  // Forward-only output micro-tile route.  Each SIMD owns four output rows
  // and groups their route entries by hidden neuron to reuse h[r,:].
  bool fdown_output_tile4 = false;
  // Reuse the dense z allocation as the exact GELU derivative cache.  The
  // preactivation has no consumer after fup; b_hidden consumes dGELU/dz.
  bool cache_gelu_grad = false;
  bool hidden_of_slot = true;
  bool route_owner = true;
  bool binput_edge_block4 = false;
  bool binput_split2 = false;
  bool binput_split4 = false;
  bool binput_fused2 = false;
  bool binput_fused1 = false;
  bool packet_binput_staged = false;
  bool packet_binput_staged64 = false;
  bool binput_unfused = false;
  bool bdown_fused = false;
  bool bdown_unfused = false;
  bool bdown_gup_fused = false;
  bool binput_separate = false;
  bool binput_gup_hidden = false;
  bool profile_phases = false;
  const char *path = nullptr, *dump_prefix = nullptr;
};
struct AdamWParams {
  float lr, beta1, beta2, eps, wd, step_size, beta2_scale;
};
static AdamWParams adamw_params(uint32_t count, uint32_t step = 1) {
  constexpr float lr = 1e-3f, beta1 = .9f, beta2 = .999f;
  return {lr,
          beta1,
          beta2,
          1e-8f,
          .01f,
          float(double(lr) / (1.0 - std::pow(double(beta1), double(step)))),
          float(std::sqrt(1.0 - std::pow(double(beta2), double(step))))};
}
static uint32_t num(const char *s) {
  return uint32_t(std::strtoul(s, nullptr, 10));
}
static bool opt(int &i, int argc, const char *argv[], const char *key,
                uint32_t &v) {
  if (std::strcmp(argv[i], key) || ++i >= argc)
    return false;
  v = num(argv[i]);
  return true;
}
static bool parse(int argc, const char *argv[], Args &a) {
  for (int i = 1; i < argc; ++i) {
    if (opt(i, argc, argv, "--input", a.input) ||
        opt(i, argc, argv, "--hidden", a.hidden) ||
        opt(i, argc, argv, "--output-size", a.output) ||
        opt(i, argc, argv, "--n", a.n) ||
        opt(i, argc, argv, "--degree", a.degree) ||
        opt(i, argc, argv, "--warmup", a.warmup) ||
        opt(i, argc, argv, "--repeats", a.repeats) ||
        opt(i, argc, argv, "--accum-steps", a.accum_steps) ||
        opt(i, argc, argv, "--verify-steps", a.verify_steps) ||
        opt(i, argc, argv, "--owner-tile-n", a.owner_tile_n) ||
        opt(i, argc, argv, "--forward-owner-tile-n", a.forward_owner_tile_n) ||
        opt(i, argc, argv, "--packet-up-block", a.packet_up_block) ||
        opt(i, argc, argv, "--packet-down-block", a.packet_down_block))
      continue;
    if (!std::strcmp(argv[i], "--output") && ++i < argc) {
      a.path = argv[i];
      continue;
    }
    if (!std::strcmp(argv[i], "--dump-prefix") && ++i < argc) {
      a.dump_prefix = argv[i];
      continue;
    }
    if (!std::strcmp(argv[i], "--support-stdin")) { a.exact_support=true; continue; }
    if (!std::strcmp(argv[i], "--untimed")) { a.untimed=true; continue; }
    if (!std::strcmp(argv[i], "--flat-adamw-pool")) {
      a.flat_adamw_pool = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--segmented-adamw")) {
      a.flat_adamw_pool = false;
      continue;
    }
    if (!std::strcmp(argv[i], "--compact-indices")) {
      a.compact_indices = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--wide-indices")) {
      a.compact_indices = false;
      continue;
    }
    if (!std::strcmp(argv[i], "--wide-route")) {
      a.wide_route = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--adamw-llm-policy")) {
      a.llm_adamw_policy = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--private-readonly")) {
      a.private_readonly = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--shared-readonly")) {
      a.private_readonly = false;
      continue;
    }
    if (!std::strcmp(argv[i], "--tiled-n")) {
      a.tiled_n = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--owner-tile")) {
      a.owner_tile = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--persistent-n")) {
      a.persistent_n = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--forward-owner-tile")) {
      a.forward_owner_tile = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--persistent-forward")) {
      a.persistent_forward = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fup-fused")) {
      a.fup_fused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fdown-fused")) {
      a.fdown_fused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fdown-unfused")) {
      a.fdown_unfused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fdown-segmented-route")) {
      a.fdown_segmented_route = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fdown-group-private")) {
      a.fdown_group_private = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--hidden-permute-up")) {
      a.hidden_permute_up = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--fdown-output-tile4")) {
      a.fdown_output_tile4 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--cache-gelu-grad")) {
      a.cache_gelu_grad = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--forward-fused")) {
      a.fup_fused = true;
      a.fdown_fused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--hidden-of-slot")) {
      a.hidden_of_slot = true;
      a.route_owner = false;
      continue;
    }
    if (!std::strcmp(argv[i], "--decode-hidden-from-slot")) {
      a.hidden_of_slot = false;
      continue;
    }
    if (!std::strcmp(argv[i], "--route-owner")) {
      a.route_owner = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-edge-block4")) {
      a.binput_edge_block4 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-split2")) {
      a.binput_split2 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-fused2")) {
      a.binput_fused2 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-fused1")) {
      a.binput_fused1 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--packet-binput-staged")) {
      a.packet_binput_staged = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--packet-binput-staged64")) {
      a.packet_binput_staged64 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-unfused")) {
      a.binput_unfused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--bdown-fused")) {
      a.bdown_fused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--bdown-unfused")) {
      a.bdown_unfused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--bdown-gup-fused")) {
      a.bdown_fused = true;
      a.bdown_gup_fused = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-split4")) {
      a.binput_split4 = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-separate")) {
      a.binput_separate = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--binput-gup-hidden")) {
      a.binput_separate = true;
      a.binput_gup_hidden = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--profile-phases")) {
      a.profile_phases = true;
      continue;
    }
    return false;
  }
  if (!a.tiled_n && !a.owner_tile && !a.persistent_n)
    a.owner_tile = a.n >= 256u;
  if (!a.forward_owner_tile && !a.persistent_forward)
    // Persistent forward has only H/16 threadgroups.  At H=512 this is
    // adequate through N=2048, but underfills the GPU once N reaches 8192.
    // Owner tiles restore batch parallelism without changing HFFN storage.
    a.forward_owner_tile = (a.hidden >= 1024u && a.n >= 2048u) ||
                           (a.hidden >= 512u && a.n >= 8192u);
  if (a.forward_owner_tile && a.owner_tile_n == 128u && !a.fdown_unfused)
    a.fdown_fused = true;
  if (a.owner_tile && a.owner_tile_n == 128u && !a.binput_unfused &&
      !a.binput_edge_block4 && !a.binput_split2 && !a.binput_split4 &&
      !a.binput_separate && !a.binput_gup_hidden && !a.binput_fused2)
    a.binput_fused1 = true;
  if (a.owner_tile && a.owner_tile_n == 128u && !a.bdown_unfused)
    a.bdown_fused = true;
  // The 16 KiB two-pass packet staging option does not preserve parity;
  // refuse it rather than expose a path that does not hold parity.
  if (a.packet_binput_staged64)
    return false;
  return a.path && a.input && a.hidden && a.output && a.n && a.accum_steps &&
         !((a.tiled_n && a.owner_tile) || (a.tiled_n && a.persistent_n) ||
           (a.owner_tile && a.persistent_n) ||
           (a.owner_tile && !a.flat_adamw_pool)) &&
         (a.owner_tile_n == 64u || a.owner_tile_n == 128u ||
          a.owner_tile_n == 256u) &&
         (!a.forward_owner_tile_n || a.forward_owner_tile_n == 64u ||
          a.forward_owner_tile_n == 128u || a.forward_owner_tile_n == 256u) &&
         a.degree <= a.input && a.degree <= a.output &&
         (!a.fdown_segmented_route ||
          (a.forward_owner_tile && a.owner_tile_n == 128u &&
           a.hidden % 4u == 0u &&
           uint64_t(a.hidden / 4u) * a.degree <= 65535u)) &&
         (!a.fdown_group_private ||
          (a.forward_owner_tile && a.owner_tile_n == 128u &&
           a.hidden % 4u == 0u &&
           uint64_t(a.hidden / 4u) * a.degree <= 65535u)) &&
         (!a.fdown_output_tile4 ||
          (a.forward_owner_tile && a.owner_tile_n == 128u &&
           a.output % 4u == 0u)) &&
         (!a.packet_up_block ||
          (a.packet_up_block <= a.input && a.degree <= a.packet_up_block &&
           a.input % a.packet_up_block == 0 &&
           a.hidden % (a.input / a.packet_up_block) == 0)) &&
         (!a.packet_down_block ||
          (a.packet_down_block <= a.output && a.degree <= a.packet_down_block &&
           a.output % a.packet_down_block == 0 &&
           a.hidden % (a.output / a.packet_down_block) == 0)) &&
         (!(a.packet_binput_staged || a.packet_binput_staged64) ||
          (a.packet_up_block == 64u && a.input == 512u && a.hidden == 1024u &&
           a.owner_tile && a.owner_tile_n == 128u));
}

static std::string msl(const Args &a, const HffnRouteCodec &c) {
  const uint32_t forward_tile =
      a.forward_owner_tile_n ? a.forward_owner_tile_n : a.owner_tile_n;
  const std::string rt = c.compact16 ? "ushort" : "uint";
  const std::string it =
      a.compact_indices && a.input <= 65535u && a.output <= 65535u ? "ushort"
                                                                   : "uint";
  const std::string ht = a.hidden <= 65535u ? "ushort" : "uint";
  const std::string hidden_r = a.route_owner
      ? "(uint(route[q])>>EB)"
      : (a.hidden_of_slot ? "uint(hidden[p])" : "p/D");
  const std::string block_hidden_r = a.route_owner
      ? "(uint(route[q0+j])>>EB)"
      : (a.hidden_of_slot ? "uint(hidden[p])" : "p/D");
  const std::string decode =
      c.compact16
          ? "inline uint slot(Route x){return (uint(x)>>EB)*D+(uint(x)&EM);}\n"
          : "inline uint slot(Route x){return (x>>EB)*D+(x&EM);}\n";
  const std::string gelu = HFFN_M1_ACTIVATION_MSL;
  // `z` is a dense activation buffer in both variants.  In the cache
  // variant, fup stores dGELU/dz in that same allocation instead of a value
  // that is dead after forward; the backward then only loads it.  This shares
  // the erf evaluation between h and its derivative and adds no workspace.
  const std::string activation_cache = a.cache_gelu_grad
      ? "inline void gelu_store(device float*z,device float*h,uint p,float x){float e=erf_accurate(x*0.7071067811865475f);h[p]=0.5f*x*(1.0f+e);z[p]=0.5f*(1.0f+e)+x*0.3989422804014327f*exp(-0.5f*x*x);}\ninline float activation_grad(float stored){return stored;}\n"
      : "inline void gelu_store(device float*z,device float*h,uint p,float x){z[p]=x;h[p]=gelu(x);}\ninline float activation_grad(float stored){return gelu_grad(stored);}\n";
  const std::string adamw = HFFN_M1_ADAMW_MSL;
  const std::string split4 = !(a.binput_split4 && a.owner_tile) ? "" :
      "kernel void binput_owner_tile_split4(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){threadgroup float gxpart[16u*OWNER_TILE_N];uint lane=tid.x&31u,sg=tid.x>>5u,group=sg>>2u,part=sg&3u,i=tg.x*4u+group,tile=tg.y,base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);bool valid=i<I;uint qb=valid?off[i]:0u,qe=valid?off[i+1u]:0u,qs=qb+(qe-qb)*part/4u,qt=qb+(qe-qb)*(part+1u)/4u;for(uint n=base+lane;n<end;n+=32u){float s=0.0f;if(valid)" HFFN_M1_REDUCE_NATIVE_001(hidden_r) "gxpart[sg*OWNER_TILE_N+n-base]=s;}threadgroup_barrier(mem_flags::mem_threadgroup);if(valid&&part==0u)for(uint n=base+lane;n<end;n+=32u){uint t=n-base;gx[i*N+n]=gxpart[(group*4u)*OWNER_TILE_N+t]+gxpart[(group*4u+1u)*OWNER_TILE_N+t]+gxpart[(group*4u+2u)*OWNER_TILE_N+t]+gxpart[(group*4u+3u)*OWNER_TILE_N+t];}if(valid)for(uint q=qs;q<qt;++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_002 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\n";
  const std::string binput_separate = !(a.binput_separate && a.owner_tile) ? "" :
      "kernel void binput_gx_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);for(uint n=base+lane;n<end;n+=32u){float s=0;" HFFN_M1_REDUCE_NATIVE_003(hidden_r) "gx[i*N+n]=s;}}\nkernel void binput_gup_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);for(uint q=off[i];q<off[i+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_004 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\n";





  const std::string bdown_gup_fused = !(a.bdown_gup_fused && a.owner_tile && a.owner_tile_n == 128u) ? "" :
      "kernel void bdown_bhidden_gup_fused_owner_tile(device const float*z[[buffer(0)]],device const float*h[[buffer(1)]],device const float*gy[[buffer(2)]],device const Index*target[[buffer(3)]],device const float*down[[buffer(4)]],device float*gz[[buffer(5)]],device float*partial[[buffer(6)]],device const Index*upidx[[buffer(7)]],device const float*x[[buffer(8)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,r=tg.x*16u+(tid.x>>5u),tile=tg.y;if(r>=F)return;uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float h0=n0<N?h[r*N+n0]:0.0f,h1=n1<N?h[r*N+n1]:0.0f,h2=n2<N?h[r*N+n2]:0.0f,h3=n3<N?h[r*N+n3]:0.0f;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint e=0;e<D;++e){uint p=r*D+e,o=target[p];" HFFN_M1_FUSED_DOWN_OP "}float bs=0.0f;if(n0<N){a0*=activation_grad(z[r*N+n0]);gz[r*N+n0]=a0;bs+=a0;}if(n1<N){a1*=activation_grad(z[r*N+n1]);gz[r*N+n1]=a1;bs+=a1;}if(n2<N){a2*=activation_grad(z[r*N+n2]);gz[r*N+n2]=a2;bs+=a2;}if(n3<N){a3*=activation_grad(z[r*N+n3]);gz[r*N+n3]=a3;bs+=a3;}float bt=simd_sum(bs);if(lane==0u)partial[(2u*F*D+r)*OWNER_TILES+tile]=bt;for(uint e=0;e<D;++e){uint p=r*D+e,i=upidx[p];float dot=0.0f;if(n0<N)dot+=a0*x[i*N+n0];if(n1<N)dot+=a1*x[i*N+n1];if(n2<N)dot+=a2*x[i*N+n2];if(n3<N)dot+=a3*x[i*N+n3];float st=simd_sum(dot);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\nkernel void binput_gx_only_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint q=off[i];q<off[i+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float w=up[p];if(n0<N)a0+=w*gz[r*N+n0];if(n1<N)a1+=w*gz[r*N+n1];if(n2<N)a2+=w*gz[r*N+n2];if(n3<N)a3+=w*gz[r*N+n3];}if(n0<N)gx[i*N+n0]=a0;if(n1<N)gx[i*N+n1]=a1;if(n2<N)gx[i*N+n2]=a2;if(n3<N)gx[i*N+n3]=a3;}\n";
  const std::string bdown_fused = !(a.bdown_fused && a.owner_tile && a.owner_tile_n == 128u) ? "" :
      "kernel void bdown_bias_owner_tile(device const float*h[[buffer(0)]],device const float*gy[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device float*partial[[buffer(4)]],device const Hidden*hidden[[buffer(5)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,o=tg.x*16u+(tid.x>>5u),tile=tg.y;if(o>=O)return;uint base=tile*128u,end=min(N,base+128u);float s=0;" HFFN_M1_REDUCE_NATIVE_005 "float st=simd_sum(s);if(lane==0u)partial[(2u*F*D+F+o)*OWNER_TILES+tile]=st;}\nkernel void bdown_bhidden_fused_owner_tile(device const float*z[[buffer(0)]],device const float*h[[buffer(1)]],device const float*gy[[buffer(2)]],device const Index*target[[buffer(3)]],device const float*down[[buffer(4)]],device float*gz[[buffer(5)]],device float*partial[[buffer(6)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,r=tg.x*16u+(tid.x>>5u),tile=tg.y;if(r>=F)return;uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float h0=n0<N?h[r*N+n0]:0.0f,h1=n1<N?h[r*N+n1]:0.0f,h2=n2<N?h[r*N+n2]:0.0f,h3=n3<N?h[r*N+n3]:0.0f;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint e=0;e<D;++e){uint p=r*D+e,o=target[p];" HFFN_M1_FUSED_DOWN_OP "}float bs=0.0f;if(n0<N){float v=a0*activation_grad(z[r*N+n0]);gz[r*N+n0]=v;bs+=v;}if(n1<N){float v=a1*activation_grad(z[r*N+n1]);gz[r*N+n1]=v;bs+=v;}if(n2<N){float v=a2*activation_grad(z[r*N+n2]);gz[r*N+n2]=v;bs+=v;}if(n3<N){float v=a3*activation_grad(z[r*N+n3]);gz[r*N+n3]=v;bs+=v;}float bt=simd_sum(bs);if(lane==0u)partial[(2u*F*D+r)*OWNER_TILES+tile]=bt;}\n";
  const std::string fused1 = !(a.binput_fused1 && a.owner_tile && a.owner_tile_n == 128u) ? "" :
      "kernel void binput_owner_tile_fused1(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint q=off[i];q<off[i+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";" HFFN_M1_FUSED_INPUT_OP "}if(n0<N)gx[i*N+n0]=a0;if(n1<N)gx[i*N+n1]=a1;if(n2<N)gx[i*N+n2]=a2;if(n3<N)gx[i*N+n3]=a3;}\n";
  const std::string packet_binput_staged =
      !(a.packet_binput_staged && a.owner_tile && a.owner_tile_n == 128u)
          ? ""
          : "kernel void binput_packet_staged_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],device const uint*mid[[buffer(8)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){threadgroup float stage[64u*128u];uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;bool valid=i<I;uint block=valid?i/64u:0u,rbase=block*128u,n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint chunk=0;chunk<2u;++chunk){for(uint k=tid.x;k<64u*128u;k+=512u){uint rr=k/128u,nn=k-rr*128u,n=tile*128u+nn;stage[k]=n<N?gz[(rbase+chunk*64u+rr)*N+n]:0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);if(valid){uint qs=chunk==0u?off[i]:mid[i],qe=chunk==0u?mid[i]:off[i+1u];for(uint q=qs;q<qe;++q){uint p=slot(route[q]),r=" + hidden_r + ",lr=r-(rbase+chunk*64u);float w=up[p],dot=0.0f;if(n0<N){float v=stage[lr*128u+lane];a0+=w*v;dot+=v*x[i*N+n0];}if(n1<N){float v=stage[lr*128u+lane+32u];a1+=w*v;dot+=v*x[i*N+n1];}if(n2<N){float v=stage[lr*128u+lane+64u];a2+=w*v;dot+=v*x[i*N+n2];}if(n3<N){float v=stage[lr*128u+lane+96u];a3+=w*v;dot+=v*x[i*N+n3];}float st=simd_sum(dot);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}threadgroup_barrier(mem_flags::mem_threadgroup);}if(valid){if(n0<N)gx[i*N+n0]=a0;if(n1<N)gx[i*N+n1]=a1;if(n2<N)gx[i*N+n2]=a2;if(n3<N)gx[i*N+n3]=a3;}}\n";
  const std::string packet_binput_staged64 =
      !(a.packet_binput_staged64 && a.owner_tile && a.owner_tile_n == 128u)
          ? ""
          : "kernel void binput_packet_staged64_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],device const uint*mid[[buffer(8)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){threadgroup float stage[64u*64u];uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;bool valid=i<I;uint block=valid?i/64u:0u,rbase=block*128u,n0=tile*128u+lane,n1=n0+32u,n2=n0+64u,n3=n0+96u;float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;for(uint chunk=0;chunk<2u;++chunk)for(uint half=0;half<2u;++half){for(uint k=tid.x;k<64u*64u;k+=512u){uint rr=k/64u,nn=k-rr*64u,n=tile*128u+half*64u+nn;stage[k]=n<N?gz[(rbase+chunk*64u+rr)*N+n]:0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);if(valid){uint qs=chunk==0u?off[i]:mid[i],qe=chunk==0u?mid[i]:off[i+1u];for(uint q=qs;q<qe;++q){uint p=slot(route[q]),r=" + hidden_r + ",lr=r-(rbase+chunk*64u);float w=up[p],dot=0.0f;if(half==0u){if(n0<N){float v=stage[lr*64u+lane];a0+=w*v;dot+=v*x[i*N+n0];}if(n1<N){float v=stage[lr*64u+lane+32u];a1+=w*v;dot+=v*x[i*N+n1];}}else{if(n2<N){float v=stage[lr*64u+lane];a2+=w*v;dot+=v*x[i*N+n2];}if(n3<N){float v=stage[lr*64u+lane+32u];a3+=w*v;dot+=v*x[i*N+n3];}}float st=simd_sum(dot);if(lane==0u){uint dst=p*OWNER_TILES+tile;if(chunk==0u&&half==0u)partial[dst]=st;else partial[dst]+=st;}}}threadgroup_barrier(mem_flags::mem_threadgroup);}if(valid){if(n0<N)gx[i*N+n0]=a0;if(n1<N)gx[i*N+n1]=a1;if(n2<N)gx[i*N+n2]=a2;if(n3<N)gx[i*N+n3]=a3;}}\n";
  const std::string fused2 = !(a.binput_fused2 && a.owner_tile) ? "" :
      "kernel void binput_owner_tile_fused2(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);float gxv[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};for(uint q0=off[i];q0<off[i+1u];q0+=2u){uint qn=min(2u,off[i+1u]-q0);float dot[2]={0.0f,0.0f};uint k=0;for(uint n=base+lane;n<end;n+=32u,++k){float xv=x[i*N+n];for(uint j=0;j<qn;++j){uint p=slot(route[q0+j]),r=" + block_hidden_r + ";float v=gz[r*N+n];gxv[k]+=up[p]*v;dot[j]+=v*xv;}}for(uint j=0;j<qn;++j){uint p=slot(route[q0+j]);float st=simd_sum(dot[j]);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}uint k=0;for(uint n=base+lane;n<end;n+=32u,++k)gx[i*N+n]=gxv[k];}\n";
  const std::string split2 = !(a.binput_split2 && a.owner_tile) ? "" :
      "kernel void binput_owner_tile_split2(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){threadgroup float gxpart[16u*OWNER_TILE_N];uint lane=tid.x&31u,sg=tid.x>>5u,group=sg>>1u,part=sg&1u,i=tg.x*8u+group,tile=tg.y,base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);bool valid=i<I;uint qb=valid?off[i]:0u,qe=valid?off[i+1u]:0u,qs=qb+(qe-qb)*part/2u,qt=qb+(qe-qb)*(part+1u)/2u;for(uint n=base+lane;n<end;n+=32u){float s=0.0f;if(valid)" HFFN_M1_REDUCE_NATIVE_006(hidden_r) "gxpart[sg*OWNER_TILE_N+n-base]=s;}threadgroup_barrier(mem_flags::mem_threadgroup);if(valid&&part==0u)for(uint n=base+lane;n<end;n+=32u){uint t=n-base;gx[i*N+n]=gxpart[(group*2u)*OWNER_TILE_N+t]+gxpart[(group*2u+1u)*OWNER_TILE_N+t];}if(valid)for(uint q=qs;q<qt;++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_007 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\n";
  const std::string binput_gup_hidden = !(a.binput_gup_hidden && a.owner_tile) ? "" :
      "kernel void binput_gup_hidden_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const Index*idx[[buffer(2)]],device const Route*unused[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,r=tg.x*16u+(tid.x>>5u),tile=tg.y;if(r>=F)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);for(uint e=0;e<D;++e){uint p=r*D+e,i=idx[p];float s=0;" HFFN_M1_REDUCE_NATIVE_008 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\n";
  return "#include <metal_stdlib>\nusing namespace metal;\nconstant uint I=" +
         std::to_string(a.input) + "u,F=" + std::to_string(a.hidden) +
         "u,O=" + std::to_string(a.output) + "u,N=" + std::to_string(a.n) +
         "u,D=" + std::to_string(a.degree) + "u,TILES=" +
         std::to_string((a.n + 31u) / 32u) +
         "u,OWNER_TILE_N=" + std::to_string(a.owner_tile_n) +
         "u,OWNER_TILES=" +
         std::to_string((a.n + a.owner_tile_n - 1u) / a.owner_tile_n) +
         "u,FORWARD_TILE_N=" + std::to_string(forward_tile) + "u,EB=" +
         std::to_string(c.edge_bits) + "u,EM=" +
         std::to_string((1u << c.edge_bits) - 1u) +
         "u,GROUP_SLOTS=" +
         std::to_string((a.hidden / 4u) * a.degree) +
         "u;\nusing Route=" + rt + ";\nusing Index=" + it + ";\nusing Hidden=" + ht + ";\n" + decode +
         gelu + activation_cache + adamw + split4 + split2 + bdown_gup_fused + bdown_fused + fused1 + packet_binput_staged + packet_binput_staged64 + fused2 + binput_separate + binput_gup_hidden +
         "kernel void bdown(device const float*h[[buffer(0)]],device const float*gy[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device float*gd[[buffer(4)]],device float*gb[[buffer(5)]],device const Hidden*hidden[[buffer(6)]],uint tid[[thread_position_in_threadgroup]],uint tg[[threadgroup_position_in_grid]]){uint lane=tid&31u,o=tg*16u+(tid>>5u);if(o>=O)return;float bs=0;" HFFN_M1_REDUCE_NATIVE_009 "float bt=simd_sum(bs);if(lane==0u)gb[o]+=bt;for(uint q=off[o];q<off[o+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_010 "float st=simd_sum(s);if(lane==0u)gd[p]+=st;}}\nkernel void bhidden(device const float*z[[buffer(0)]],device const float*gy[[buffer(1)]],device const Index*target[[buffer(2)]],device const float*down[[buffer(3)]],device float*gz[[buffer(4)]],device float*gb[[buffer(5)]],uint tid[[thread_position_in_threadgroup]],uint tg[[threadgroup_position_in_grid]]){uint lane=tid&31u,r=tg*16u+(tid>>5u);if(r>=F)return;float bs=0;for(uint n=lane;n<N;n+=32u){float gh=0;" HFFN_M1_REDUCE_NATIVE_011 "float v=gh*activation_grad(z[r*N+n]);gz[r*N+n]=v;bs+=v;}float bt=simd_sum(bs);if(lane==0u)gb[r]+=bt;}\nkernel void binput(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gu[[buffer(5)]],device float*gx[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint tid[[thread_position_in_threadgroup]],uint tg[[threadgroup_position_in_grid]]){uint lane=tid&31u,i=tg*16u+(tid>>5u);if(i>=I)return;for(uint n=lane;n<N;n+=32u){float s=0;" HFFN_M1_REDUCE_NATIVE_012(hidden_r) "gx[i*N+n]=s;}for(uint q=off[i];q<off[i+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_013 "float st=simd_sum(s);if(lane==0u)gu[p]+=st;}}\n" +
         (a.owner_tile ?
         // A1: same owner/SIMD mapping as persistent-N, but each grid-Y
         // position owns 128 batch elements.  The only reductions are compact
         // parameter/bias gradients in a single flat partial pool; gz and gx
         // remain direct owner writes to dense activation-gradient buffers.
         "kernel void bdown_owner_tile(device const float*h[[buffer(0)]],device const float*gy[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device float*partial[[buffer(4)]],device const Hidden*hidden[[buffer(5)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,o=tg.x*16u+(tid.x>>5u),tile=tg.y;if(o>=O)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);float bs=0;" HFFN_M1_REDUCE_NATIVE_014 "float bt=simd_sum(bs);if(lane==0u)partial[(2u*F*D+F+o)*OWNER_TILES+tile]=bt;for(uint q=off[o];q<off[o+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_015 "float st=simd_sum(s);if(lane==0u)partial[(F*D+p)*OWNER_TILES+tile]=st;}}\nkernel void bhidden_owner_tile(device const float*z[[buffer(0)]],device const float*gy[[buffer(1)]],device const Index*target[[buffer(2)]],device const float*down[[buffer(3)]],device float*gz[[buffer(4)]],device float*partial[[buffer(5)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,r=tg.x*16u+(tid.x>>5u),tile=tg.y;if(r>=F)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);float bs=0;for(uint n=base+lane;n<end;n+=32u){float gh=0;" HFFN_M1_REDUCE_NATIVE_016 "float v=gh*activation_grad(z[r*N+n]);gz[r*N+n]=v;bs+=v;}float bt=simd_sum(bs);if(lane==0u)partial[(2u*F*D+r)*OWNER_TILES+tile]=bt;}\nkernel void binput_owner_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);for(uint n=base+lane;n<end;n+=32u){float s=0;" HFFN_M1_REDUCE_NATIVE_017(hidden_r) "gx[i*N+n]=s;}for(uint q=off[i];q<off[i+1u];++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_018 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\nkernel void reduce_flat_partials(device const float*partial[[buffer(0)]],device float*g[[buffer(1)]],uint gid[[thread_position_in_grid]]){uint total=2u*F*D+F+O;if(gid>=total)return;float s=0;" HFFN_M1_REDUCE_NATIVE_019 "g[gid]+=s;}\n" : a.tiled_n ?
         "kernel void bdown_tile(device const float*h[[buffer(0)]],device const float*gy[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device float*pg[[buffer(4)]],device float*pb[[buffer(5)]],uint gid[[thread_position_in_grid]]){uint edge=gid/TILES,t=gid%TILES;if(edge>=off[O])return;uint p=slot(route[edge]),r=p/D,o=0;while(off[o+1u]<=edge)++o;uint base=t*32u,e=min(N,base+32u);float s=0;" HFFN_M1_REDUCE_NATIVE_020 "pg[p*TILES+t]=s;if(edge==off[o]){float b=0;" HFFN_M1_REDUCE_NATIVE_021 "pb[o*TILES+t]=b;}}\nkernel void reduce_edge(device const float*pg[[buffer(0)]],device float*g[[buffer(1)]],uint gid[[thread_position_in_grid]]){if(gid>=F*D)return;float s=0;" HFFN_M1_REDUCE_NATIVE_022 "g[gid]+=s;}\nkernel void reduce_output(device const float*pb[[buffer(0)]],device float*g[[buffer(1)]],uint gid[[thread_position_in_grid]]){if(gid>=O)return;float s=0;" HFFN_M1_REDUCE_NATIVE_023 "g[gid]+=s;}\nkernel void bhidden_tile(device const float*z[[buffer(0)]],device const float*gy[[buffer(1)]],device const Index*target[[buffer(2)]],device const float*down[[buffer(3)]],device float*gz[[buffer(4)]],device float*pb[[buffer(5)]],uint gid[[thread_position_in_grid]]){uint r=gid/TILES,t=gid%TILES;if(r>=F)return;uint base=t*32u,e=min(N,base+32u);float b=0;for(uint n=base;n<e;++n){float gh=0;" HFFN_M1_REDUCE_NATIVE_024 "float v=gh*activation_grad(z[r*N+n]);gz[r*N+n]=v;b+=v;}pb[r*TILES+t]=b;}\nkernel void reduce_hidden(device const float*pb[[buffer(0)]],device float*g[[buffer(1)]],uint gid[[thread_position_in_grid]]){if(gid>=F)return;float s=0;" HFFN_M1_REDUCE_NATIVE_025 "g[gid]+=s;}\nkernel void binput_tile(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*pg[[buffer(5)]],device float*gx[[buffer(6)]],uint gid[[thread_position_in_grid]]){uint edge=gid/TILES,t=gid%TILES;if(edge>=off[I])return;uint p=slot(route[edge]),r=p/D,i=0;while(off[i+1u]<=edge)++i;uint base=t*32u,e=min(N,base+32u);float w=0;" HFFN_M1_REDUCE_NATIVE_026 "pg[p*TILES+t]=w;if(edge==off[i])for(uint n=base;n<e;++n){float s=0;" HFFN_M1_REDUCE_NATIVE_027 "gx[i*N+n]=s;}}\n" : "") + (a.binput_edge_block4 && a.owner_tile ?
         "kernel void binput_owner_tile_block4(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){uint lane=tid.x&31u,i=tg.x*16u+(tid.x>>5u),tile=tg.y;if(i>=I)return;uint base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);for(uint n=base+lane;n<end;n+=32u){float s=0;" HFFN_M1_REDUCE_NATIVE_028(hidden_r) "gx[i*N+n]=s;}for(uint q0=off[i];q0<off[i+1u];q0+=4u){uint qn=min(4u,off[i+1u]-q0);float ss[4]={0.0f,0.0f,0.0f,0.0f};for(uint n=base+lane;n<end;n+=32u){float xv=x[i*N+n];" HFFN_M1_REDUCE_NATIVE_029(block_hidden_r) "}for(uint j=0;j<qn;++j){uint p=slot(route[q0+j]);float st=simd_sum(ss[j]);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}}\n" : "");
         "kernel void binput_owner_tile_split4(device const float*x[[buffer(0)]],device const float*gz[[buffer(1)]],device const uint*off[[buffer(2)]],device const Route*route[[buffer(3)]],device const float*up[[buffer(4)]],device float*gx[[buffer(5)]],device float*partial[[buffer(6)]],device const Hidden*hidden[[buffer(7)]],uint3 tid[[thread_position_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){threadgroup float gxpart[16u*OWNER_TILE_N];uint lane=tid.x&31u,sg=tid.x>>5u,group=sg>>2u,part=sg&3u,i=tg.x*4u+group,tile=tg.y,base=tile*OWNER_TILE_N,end=min(N,base+OWNER_TILE_N);bool valid=i<I;uint qb=valid?off[i]:0u,qe=valid?off[i+1u]:0u,qs=qb+(qe-qb)*part/4u,qt=qb+(qe-qb)*(part+1u)/4u;for(uint n=base+lane;n<end;n+=32u){float s=0.0f;if(valid)" HFFN_M1_REDUCE_NATIVE_030(hidden_r) "gxpart[sg*OWNER_TILE_N+n-base]=s;}threadgroup_barrier(mem_flags::mem_threadgroup);if(valid&&part==0u)for(uint n=base+lane;n<end;n+=32u){uint t=n-base;gx[i*N+n]=gxpart[(group*4u)*OWNER_TILE_N+t]+gxpart[(group*4u+1u)*OWNER_TILE_N+t]+gxpart[(group*4u+2u)*OWNER_TILE_N+t]+gxpart[(group*4u+3u)*OWNER_TILE_N+t];}if(valid)for(uint q=qs;q<qt;++q){uint p=slot(route[q]),r=" + hidden_r + ";float s=0;" HFFN_M1_REDUCE_NATIVE_031 "float st=simd_sum(s);if(lane==0u)partial[p*OWNER_TILES+tile]=st;}}\n";
}
static id<MTLBuffer>
buf(id<MTLDevice> d, const void *x, size_t n,
    MTLResourceOptions options = MTLResourceStorageModeShared) {
  id<MTLBuffer> result=[d newBufferWithBytes:x length:n options:options];
  if(!result)throw std::runtime_error("Metal buffer allocation failed");
  return result;
}
// Metal cannot initialize a Private allocation from CPU memory directly.
// Immutable runtime metadata uses a short-lived Shared staging allocation and
// a blit; mutable training state intentionally remains Shared for this runner.
static id<MTLBuffer> upload(id<MTLDevice> d, id<MTLCommandQueue> q,
                            const void *x, size_t n,
                            MTLResourceOptions options) {
  if (options != MTLResourceStorageModePrivate)
    return buf(d, x, n, options);
  id<MTLBuffer> staging = buf(d, x, n, MTLResourceStorageModeShared);
  id<MTLBuffer> target = [d newBufferWithLength:n
                                         options:MTLResourceStorageModePrivate];
  if(!target)throw std::runtime_error("Private buffer allocation failed");
  id<MTLCommandBuffer> command = [q commandBuffer];
  if(!command)throw std::runtime_error("Metal upload command creation failed");
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  if(!blit)throw std::runtime_error("Metal upload encoder creation failed");
  [blit copyFromBuffer:staging sourceOffset:0 toBuffer:target destinationOffset:0 size:n];
  [blit endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "Metal upload: %s\n",
                 [[command.error.localizedDescription description] UTF8String]);
    throw std::runtime_error("native Metal construction/execution failed");
  }
  return target;
}
struct BufferBinding {
  void *buffer;
  NSUInteger offset;
};
static NSValue *view(id<MTLBuffer> buffer, NSUInteger offset = 0) {
  BufferBinding b{(__bridge void *)buffer, offset};
  return [NSValue valueWithBytes:&b objCType:@encode(BufferBinding)];
}
struct NativeDiagnostics {
  bool untimed = true;
  FILE *dispatch_trace = nullptr;
  std::map<void *, std::string> pipeline_names;
  ~NativeDiagnostics() { if (dispatch_trace) std::fclose(dispatch_trace); }
};
static void enc(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                NSArray *b, MTLSize g, NativeDiagnostics &diagnostics) {
  [e setComputePipelineState:p];
  for (NSUInteger i = 0; i < b.count; ++i) {
    id entry = b[i];
    if ([entry isKindOfClass:[NSValue class]]) {
      BufferBinding v;
      [(NSValue *)entry getValue:&v];
      [e setBuffer:(__bridge id<MTLBuffer>)v.buffer offset:v.offset atIndex:i];
    } else {
      [e setBuffer:entry offset:0 atIndex:i];
    }
  }
  if(diagnostics.dispatch_trace)std::fprintf(diagnostics.dispatch_trace,"dispatch %s %lu %lu %lu\n",diagnostics.pipeline_names.at((__bridge void*)p).c_str(),(unsigned long)g.width,(unsigned long)g.height,(unsigned long)g.depth);
  [e dispatchThreadgroups:g threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
}
static double run(id<MTLCommandQueue> q,
                  NSArray<id<MTLComputePipelineState>> *p,
                  NSArray<NSArray<id<MTLBuffer>> *> *b, NSArray<NSValue *> *g,
                  uint32_t r, NativeDiagnostics &diagnostics) {
  id<MTLCommandBuffer> c = [q commandBuffer];
  if(!c)throw std::runtime_error("Metal command buffer creation failed");
  id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
  if(!e)throw std::runtime_error("Metal encoder creation failed");
  for (uint32_t k = 0; k < r; ++k)
    for (NSUInteger j = 0; j < p.count; ++j) {
      MTLSize s;
      [g[j] getValue:&s];
      enc(e, p[j], b[j], s, diagnostics);
    }
  [e endEncoding];
  [c commit];
  [c waitUntilCompleted];
  if (c.status == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "Metal: %s\n",
                 [[c.error.localizedDescription description] UTF8String]);
    throw std::runtime_error("native Metal construction/execution failed");
  }
  if (diagnostics.untimed) return 0.0;
  return (c.GPUEndTime - c.GPUStartTime) * 1000.0 / r;
}
static double err(const std::vector<float> &a, const float *b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { std::fprintf(stderr,"nonfinite parity quantity\n"); throw std::runtime_error("native Metal construction/execution failed"); }
    e = std::max(e, std::abs(double(a[i] - b[i])));
  }
  return e;
}
static bool dump(const char *prefix, const char *suffix, const float *data,
                 size_t count) {
  std::string path = std::string(prefix) + suffix;
  FILE *f = std::fopen(path.c_str(), "wb");
  return f && std::fwrite(data, sizeof(float), count, f) == count &&
         !std::fclose(f);
}

#include "native_train_context_core.inc"

static int prepare_and_run(int argc, const char *argv[], HffnTrainContext **created, const char *inference_bundle,
                           const HffnTrainSupport *supplied=nullptr, void *external_parameter_handle=nullptr,
                           uint64_t external_parameter_offset=0, uint64_t external_parameter_count=0) {
  @autoreleasepool {
    // "--optimizer-only" is consumed here, before parse, so that Args keeps its layout.
    bool optimizer_only = false;
    std::vector<const char *> kept_argv;
    for (int i = 0; i < argc; ++i) {
      if (i > 0 && argv[i] && !std::strcmp(argv[i], "--optimizer-only")) { optimizer_only = true; continue; }
      kept_argv.push_back(argv[i]);
    }
    argc = int(kept_argv.size());
    argv = kept_argv.data();
    Args a;
    if (!parse(argc, argv, a)) {
      std::fprintf(stderr, "usage: --input I --hidden F --output-size O --n N "
                           "--degree D --output file\n");
      return 2;
    }
    auto diagnostics = std::make_unique<NativeDiagnostics>();
    if(created && !supplied && !a.degree)throw std::invalid_argument("zero-degree context is unsupported");
    diagnostics->untimed = created ? true : a.untimed;
    if(!created && a.untimed && std::getenv("HFFN_DISPATCH_TRACE")){
      diagnostics->dispatch_trace=std::fopen(std::getenv("HFFN_DISPATCH_TRACE"),"w");
      if(!diagnostics->dispatch_trace)return 2;
    }
    FILE *dispatch_trace=diagnostics->dispatch_trace;
    if (!created && a.untimed && !a.verify_steps) { std::fprintf(stderr,"--untimed requires --verify-steps\n"); return 2; }
    // Forward scheduling is owned by inference. Former schedule flags are
    // compatibility aliases; they no longer select a private forward.
    a.forward_owner_tile = false;
    a.fup_fused = a.fdown_fused = a.fdown_unfused = false;
    a.fdown_group_private = a.fdown_segmented_route = a.fdown_output_tile4 = false;
    a.cache_gelu_grad = false;  // inference saves z; backward evaluates its derivative
    uint64_t edges64 = uint64_t(a.hidden) * a.degree;
    if (edges64 >= uint64_t(INT32_MAX) || uint64_t(a.n) * std::max({a.input,a.hidden,a.output}) >= (uint64_t(1)<<32)) return 2;
    NativeExactSupport imported;
    if(supplied)a.exact_support=true;
    if(a.exact_support) {
      if(!a.flat_adamw_pool || a.tiled_n || a.packet_up_block || a.packet_down_block ||
         a.hidden_permute_up || !a.route_owner || a.binput_edge_block4 || a.binput_split2 ||
         a.binput_split4 || a.binput_separate || a.binput_fused2 || a.bdown_gup_fused ||
         a.packet_binput_staged) { std::fprintf(stderr,"unsupported imported-support option combination\n"); return 2; }
      try { if(supplied)imported.load(*supplied); else imported.load(a.input,a.hidden,a.output); }
      catch(const std::exception& e){std::fprintf(stderr,"support admission: %s\n",e.what());return 2;}
      a.degree=imported.degree;
    }
    const bool exact_offsets = a.exact_support && !imported.uniform_shared_degree();
    uint32_t E = a.exact_support ? uint32_t(imported.up_val.size()) : a.hidden*a.degree;
    uint32_t ED = a.exact_support ? uint32_t(imported.down_val.size()) : E;
    a.up_count=E;a.down_count=ED;
    if (optimizer_only) {
      // the dedicated optimizer context of a consumer. It holds only what the compact AdamW
      // update reads (the caller's flat parameter pool, the flat gradient pool and both moments, the update scalars) and
      // the one "adamw" pipeline, bound and dispatched exactly as the flat row of the full context. No forward or backward
      // state is built: accum_steps is 0, so every forward, backward and accumulation entry refuses (fail closed).
      if (!created || !supplied || !external_parameter_handle || !a.flat_adamw_pool || !a.llm_adamw_policy ||
          a.tiled_n || a.owner_tile || a.persistent_n) {
        std::fprintf(stderr, "--optimizer-only requires an exact context over an external flat pool with --adamw-llm-policy "
                             "and no tiled, owner-tile or persistent-N regime\n");
        return 2;
      }
      id<MTLDevice> d = MTLCreateSystemDefaultDevice();
      if (!d)
        return 1;
      NSError *er = nil;
      std::string update_source = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + HFFN_M1_ADAMW_MSL;
      const std::string from = "hffn_adamw_slot<false,false>";
      const std::string to = "hffn_adamw_slot<true,true>";
      const auto pos = update_source.find(from);
      if (pos == std::string::npos || update_source.find(from, pos + from.size()) != std::string::npos)
        throw std::runtime_error("unexpected shared AdamW source");
      update_source.replace(pos, from.size(), to);
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.fastMathEnabled = NO;
      id<MTLLibrary> precise_adamw = [d newLibraryWithSource:[NSString stringWithUTF8String:update_source.c_str()] options:opts error:&er];
      if (!precise_adamw) throw std::runtime_error("precise AdamW shader compilation failed");
      id<MTLCommandQueue> q = [d newCommandQueue];
      if(!q)throw std::runtime_error("Metal command queue creation failed");
      NSError *pe = nil;
      id<MTLComputePipelineState> adamw =
          [d newComputePipelineStateWithFunction:[precise_adamw newFunctionWithName:@"adamw"] error:&pe];
      if (!adamw) {
        std::fprintf(stderr, "pipe: %s\n", [[pe.localizedDescription description] UTF8String]);
        throw std::runtime_error("native Metal construction/execution failed");
      }
      diagnostics->pipeline_names[(__bridge void*)adamw]="adamw";
      id<MTLBuffer> external_parameter_buffer = (__bridge id<MTLBuffer>)external_parameter_handle;
      AdamWParams pu = adamw_params(E), pd = adamw_params(ED),
                  pbu = adamw_params(a.hidden), pbd = adamw_params(a.output);
      id<MTLBuffer> pub = buf(d, &pu, sizeof(pu)), pdb = buf(d, &pd, sizeof(pd)),
                    pbub = buf(d, &pbu, sizeof(pbu)),
                    pbdb = buf(d, &pbd, sizeof(pbd));
      const uint32_t flat_total = E + ED + a.hidden + a.output;
      if (external_parameter_buffer.device != d || external_parameter_count != flat_total ||
          external_parameter_offset % sizeof(float) || external_parameter_offset > external_parameter_buffer.length ||
          uint64_t(flat_total) * sizeof(float) > external_parameter_buffer.length - external_parameter_offset)
        throw std::invalid_argument("invalid external compact parameter pool");
      std::vector<float> flat_g(flat_total), flat_m(flat_total), flat_v(flat_total);
      id<MTLBuffer> flat_pb = external_parameter_buffer,
                    flat_gb = buf(d, flat_g.data(), flat_total * sizeof(float)),
                    flat_mb = buf(d, flat_m.data(), flat_total * sizeof(float)),
                    flat_vb = buf(d, flat_v.data(), flat_total * sizeof(float));
      AdamWParams flat_q = adamw_params(flat_total);
      id<MTLBuffer> flat_qb = buf(d, &flat_q, sizeof(flat_q)),
                    flat_count_b = buf(d, &flat_total, sizeof(flat_total));
      // The stage table keeps the shape the context expects (five verify rows, then the update row); the five verify rows
      // are placeholders that no entry reaches with accum_steps 0.
      MTLSize af = MTLSizeMake((flat_total + 511) / 512, 1, 1);
      NSValue *grid = [NSValue valueWithBytes:&af objCType:@encode(MTLSize)];
      NSArray *ps = @[ [NSNull null], [NSNull null], [NSNull null], [NSNull null], [NSNull null], adamw ];
      NSArray *bs = @[ @[], @[], @[], @[], @[],
                       @[ ::view(flat_pb, NSUInteger(external_parameter_offset)), ::view(flat_gb), ::view(flat_mb),
                          ::view(flat_vb), flat_qb, flat_count_b ] ];
      NSArray *gs = @[ grid, grid, grid, grid, grid, grid ];
      // The constructor reads the update row at index verify_count (5: neither tiled nor owner-tile) after the verify rows.
      // NSMakeRange is named here, before the construction, on purpose: clang emits deferred functions in the order of
      // their first reference, and NSMakeRange must be emitted (and found nounwind) before HffnTrainContext's constructor,
      // as in the full path, or the constructor's prefix lambda compiles to other machine code (cmpfn.py, README).
      const NSRange update_row = NSMakeRange(5, 1);
      if (ps.count != update_row.location + update_row.length || ps[update_row.location] != adamw)
        throw std::logic_error("inconsistent optimizer-only stage table");
      a.accum_steps = 0;
      NSDictionary *owners = @{ @"flat_pb": flat_pb, @"flat_gb": flat_gb, @"flat_mb": flat_mb, @"flat_vb": flat_vb,
                                @"flat_qb": flat_qb, @"flat_count_b": flat_count_b, @"pub": pub, @"pdb": pdb,
                                @"pbub": pbub, @"pbdb": pbdb, @"x_micro": @[], @"gy_micro": @[] };
      // Same argument types as the full construction below ([owners copy] is an id rvalue): one make_unique instance.
      auto core = std::make_unique<HffnTrainContext>(a, q, ps, bs, gs, [owners copy], std::move(diagnostics));
      core->external_parameters = external_parameter_buffer;
      core->external_parameter_offset = external_parameter_offset / sizeof(float);
      core->enable_caller_managed_step();
      *created = core.release();
      return 0;
    }
    std::vector<uint32_t> up_hidden_offsets(a.hidden+1),down_hidden_offsets(a.hidden+1);
    for(uint32_t r=0;r<=a.hidden;++r)up_hidden_offsets[r]=down_hidden_offsets[r]=r*a.degree;
    if(a.exact_support){up_hidden_offsets=imported.up_offsets;down_hidden_offsets=imported.down_offsets;}
    auto route_slot=[&](uint32_t packed,const std::vector<uint32_t>& offsets,const HffnRouteCodec& codec){
      return offsets[packed>>codec.edge_bits]+(packed&((1u<<codec.edge_bits)-1u));
    };
    const bool compact_indices =
        a.compact_indices && a.input <= 65535u && a.output <= 65535u;
    HffnRouteCodec c(a.hidden, a.degree);
    if (a.wide_route) c.compact16 = false;
    bool c16 = c.compact16;
    std::vector<float> x(a.input * a.n), gy(a.output * a.n), up(E), down(ED),
        bu(a.hidden), bd(a.output), z(a.hidden * a.n), h(a.hidden * a.n),
        y(a.output * a.n), rz(a.hidden * a.n), rdgelu(a.hidden * a.n), rh(a.hidden * a.n),
        ry(a.output * a.n), rgz(a.hidden * a.n), rgx(a.input * a.n), rgu(E),
        rgd(ED), rgbu(a.hidden), rgbd(a.output);
    std::vector<uint32_t> idx(E), target(ED), doff(a.output + 1),
        uoff(a.input + 1);
    for (uint32_t i = 0; i < a.input; ++i)
      for (uint32_t n = 0; n < a.n; ++n)
        x[i * a.n + n] = float(int((i * 17 + n * 13) % 29) - 14) * .02f;
    for (uint32_t o = 0; o < a.output; ++o) {
      bd[o] = float(int(o % 11) - 5) * .008f;
      for (uint32_t n = 0; n < a.n; ++n)
        gy[o * a.n + n] = float(int((o * 11 + n * 7) % 23) - 11) * .03f +
                           .0007f;
    }
    if (!a.exact_support) {
    for (uint32_t r = 0; r < a.hidden; ++r) {
      bu[r] = float(int(r % 9) - 4) * .01f;
      for (uint32_t e = 0; e < a.degree; ++e) {
        uint32_t p = r * a.degree + e;
        if (a.packet_up_block) {
          const uint32_t blocks = a.input / a.packet_up_block;
          const uint32_t hidden_per_block = a.hidden / blocks;
          const uint32_t block = (r / hidden_per_block) * a.packet_up_block;
          // Fixed-support packet layout: each hidden packet uses only its
          // input block, while the affine rotation preserves uniform local
          // fan-in and distinct edges per hidden neuron.
          idx[p] = block + (r * 19 + e * 7) % a.packet_up_block;
        } else {
          idx[p] = (r * 19 + e * 7) % a.input;
        }
        if (a.packet_down_block) {
          const uint32_t blocks = a.output / a.packet_down_block;
          const uint32_t hidden_per_block = a.hidden / blocks;
          const uint32_t block = (r / hidden_per_block) * a.packet_down_block;
          target[p] = block + (r * 17 + e * 5) % a.packet_down_block;
        } else {
          target[p] = (r * 17 + e * 5) % a.output;
        }
        up[p] = float(int((p * 3) % 17) - 8) * .015f;
        down[p] = float(int((p * 5) % 19) - 9) * .012f;
      }
    }
    if (a.hidden_permute_up) {
      std::vector<uint32_t> order(a.hidden);
      for (uint32_t r = 0; r < a.hidden; ++r) order[r] = r;
      std::sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
        const uint32_t li = idx[lhs * a.degree], ri = idx[rhs * a.degree];
        if (li != ri) return li < ri;
        const uint32_t lo = target[lhs * a.degree], ro = target[rhs * a.degree];
        return lo != ro ? lo < ro : lhs < rhs;
      });
      std::vector<uint32_t> new_idx(E), new_target(E);
      std::vector<float> new_up(E), new_down(E), new_bu(a.hidden);
      for (uint32_t r = 0; r < a.hidden; ++r) {
        const uint32_t old = order[r];
        new_bu[r] = bu[old];
        for (uint32_t e = 0; e < a.degree; ++e) {
          const uint32_t dst = r * a.degree + e, src = old * a.degree + e;
          new_idx[dst] = idx[src]; new_target[dst] = target[src];
          new_up[dst] = up[src]; new_down[dst] = down[src];
        }
      }
      idx.swap(new_idx); target.swap(new_target); up.swap(new_up);
      down.swap(new_down); bu.swap(new_bu);
    }
    } else {
      up=imported.up_val;down=imported.down_val;bu=imported.b_up;bd=imported.b_down;
      idx=imported.up_idx;target=imported.down_idx;
    }
    for(uint32_t p=0;p<ED;++p)++doff[target[p]+1u];
    for(uint32_t p=0;p<E;++p)++uoff[idx[p]+1u];
    // A real gradient-accumulation step receives distinct microbatches.  The
    // deterministic perturbation preserves the K=1 fixture bit-for-bit while
    // making K>1 exercise accumulation rather than K copies of one batch.
    std::vector<std::vector<float>> x_micro(a.accum_steps, x),
        gy_micro(a.accum_steps, gy);
    for (uint32_t k = 1; k < a.accum_steps; ++k) {
      for (uint32_t i = 0; i < a.input; ++i)
        for (uint32_t n = 0; n < a.n; ++n)
          x_micro[k][i * a.n + n] +=
              float(int((i * 5 + n * 3 + k * 7) % 11) - 5) * .001f;
      for (uint32_t o = 0; o < a.output; ++o)
        for (uint32_t n = 0; n < a.n; ++n)
          gy_micro[k][o * a.n + n] +=
              float(int((o * 3 + n * 5 + k * 11) % 13) - 6) * .0015f +
              .0007f * float(k);
    }
    for (uint32_t i = 0; i < a.output; ++i)
      doff[i + 1] += doff[i];
    for (uint32_t i = 0; i < a.input; ++i)
      uoff[i + 1] += uoff[i];
    std::vector<uint16_t> dr16(ED),ur16(E);
    std::vector<uint32_t> dr32(ED),ur32(E),dc=doff,uc=uoff;
    auto reverse=[&](const auto& offsets,const auto& index,auto& cursor,auto& narrow,auto& wide){
      for(uint32_t r=0;r<a.hidden;++r)for(uint32_t p=offsets[r];p<offsets[r+1];++p){
        uint32_t v=c16?uint32_t(c.encode16(r,p-offsets[r])):c.encode32(r,p-offsets[r]);
        uint32_t q=cursor[index[p]]++;
        if(c16)narrow[q]=uint16_t(v);else wide[q]=v;
      }
    };
    reverse(down_hidden_offsets,target,dc,dr16,dr32);
    reverse(up_hidden_offsets,idx,uc,ur16,ur32);
    // [output-tile -> entries] keeps every tile in canonical hidden order.
    // An entry stores (slot, local-output) in 32 bits; unlike packet support,
    // this is a derived execution route and leaves the graph unchanged.
    std::vector<uint32_t> otile_off(a.output / 4u + 1u, 0), otile_route(E);
    if (a.fdown_output_tile4) {
      for (uint32_t p = 0; p < E; ++p) ++otile_off[target[p] / 4u + 1u];
      for (uint32_t t = 0; t < a.output / 4u; ++t)
        otile_off[t + 1u] += otile_off[t];
      std::vector<uint32_t> cursor = otile_off;
      for (uint32_t p = 0; p < E; ++p) {
        const uint32_t o = target[p], t = o / 4u;
        otile_route[cursor[t]++] = (p << 2u) | (o & 3u);
      }
    }
    // A global route needs 17 bits at H=2048,D=51.  For the forward output
    // route only, retain the canonical order but split each output list into
    // four hidden-group segments.  A slot is then relative to its group and
    // fits in u16; segment boundaries make the group implicit, not per-edge
    // metadata.  The original routes remain authoritative for backward.
    std::vector<uint16_t> drseg16(E);
    std::vector<uint32_t> dseg(a.output * 5u, 0);
    if (a.fdown_segmented_route || a.fdown_group_private) {
      const uint32_t group_slots = (a.hidden / 4u) * a.degree;
      for (uint32_t o = 0; o < a.output; ++o) {
        uint32_t counts[4] = {0, 0, 0, 0};
        for (uint32_t q = doff[o]; q < doff[o + 1u]; ++q) {
          const uint32_t p = route_slot(c16?uint32_t(dr16[q]):dr32[q],down_hidden_offsets,c);
          ++counts[p / group_slots];
        }
        uint32_t cursor = doff[o];
        dseg[o * 5u] = cursor;
        for (uint32_t g = 0; g < 4u; ++g) {
          cursor += counts[g];
          dseg[o * 5u + g + 1u] = cursor;
        }
        uint32_t write[4] = {dseg[o * 5u], dseg[o * 5u + 1u],
                             dseg[o * 5u + 2u], dseg[o * 5u + 3u]};
        for (uint32_t q = doff[o]; q < doff[o + 1u]; ++q) {
          const uint32_t p = route_slot(c16?uint32_t(dr16[q]):dr32[q],down_hidden_offsets,c);
          const uint32_t g = p / group_slots;
          drseg16[write[g]++] = uint16_t(p - g * group_slots);
        }
      }
    }
    // The packet-staged binput kernel consumes two contiguous 64-hidden
    // halves.  Routes are already sorted by hidden slot; this one compact
    // cursor per input avoids scanning the other half in the hot kernel.
    std::vector<uint32_t> upacket_mid(a.input, 0);
    if (a.packet_binput_staged) {
      for (uint32_t i = 0; i < a.input; ++i) {
        const uint32_t rcut = (i / 64u) * 128u + 64u;
        uint32_t q = uoff[i];
        for (; q < uoff[i + 1u]; ++q) {
          const uint32_t p = route_slot(c16?uint32_t(ur16[q]):ur32[q],up_hidden_offsets,c);
          if (p / a.degree >= rcut)
            break;
        }
        upacket_mid[i] = q;
      }
    }
    if (!created) {
    for (uint32_t r = 0; r < a.hidden; ++r)
      for (uint32_t n = 0; n < a.n; ++n) {
        float v = bu[r];
        for (uint32_t p = up_hidden_offsets[r]; p < up_hidden_offsets[r+1]; ++p) {
          v += up[p] * x[idx[p] * a.n + n];
        }
        rz[r * a.n + n] = v;
        rdgelu[r * a.n + n] = hffn_gelu_erf_accurate_grad(v);
        rh[r * a.n + n] = hffn_gelu_erf_accurate(v);
      }
    for (uint32_t o = 0; o < a.output; ++o)
      for (uint32_t n = 0; n < a.n; ++n) {
        float v = bd[o];
        for (uint32_t q = doff[o]; q < doff[o + 1]; ++q) {
          uint32_t p = route_slot(c16?uint32_t(dr16[q]):dr32[q],down_hidden_offsets,c);
          v += down[p] * rh[((c16?uint32_t(dr16[q]):dr32[q])>>c.edge_bits)*a.n+n];
        }
        ry[o * a.n + n] = v;
        rgbd[o] += gy[o * a.n + n];
      }
    for (uint32_t r = 0; r < a.hidden; ++r) {
      for (uint32_t p = down_hidden_offsets[r]; p < down_hidden_offsets[r+1]; ++p) {
        for (uint32_t n = 0; n < a.n; ++n)
          rgd[p] += rh[r * a.n + n] * gy[target[p] * a.n + n];
      }
      for (uint32_t n = 0; n < a.n; ++n) {
        float gh = 0;
        for (uint32_t p = down_hidden_offsets[r]; p < down_hidden_offsets[r+1]; ++p) {
          gh += down[p] * gy[target[p] * a.n + n];
        }
        float v = gh * (a.cache_gelu_grad ? rdgelu[r * a.n + n]
                                            : hffn_gelu_erf_accurate_grad(rz[r * a.n + n]));
        rgz[r * a.n + n] = v;
        rgbu[r] += v;
      }
      for (uint32_t p = up_hidden_offsets[r]; p < up_hidden_offsets[r+1]; ++p) {
        for (uint32_t n = 0; n < a.n; ++n) {
          rgu[p] += rgz[r * a.n + n] * x[idx[p] * a.n + n];
          rgx[idx[p] * a.n + n] += up[p] * rgz[r * a.n + n];
        }
      }
    }
    }
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d)
      return 1;
    NSError *er = nil;
    std::string backward_source=msl(a,c);
    if(exact_offsets)backward_source=exact_support_source(backward_source,E,ED);
    if(!created && a.untimed && a.dump_prefix) {
      std::string source_path=std::string(a.dump_prefix)+"_backward.metal";
      FILE* source_file=std::fopen(source_path.c_str(),"wb");
      if(!source_file)return 2;
      bool source_ok=std::fwrite(backward_source.data(),1,backward_source.size(),source_file)==backward_source.size();
      source_ok=std::fclose(source_file)==0 && source_ok;
      if(!source_ok)return 2;
    }
    id<MTLLibrary> l = [d
        newLibraryWithSource:[NSString stringWithUTF8String:backward_source.c_str()]
                     options:nil
                       error:&er];
    if (!l) {
      std::fprintf(stderr, "MSL: %s\n",
                   [[er.localizedDescription description] UTF8String]);
      return 1;
    }
    id<MTLLibrary> precise_adamw = nil;
    if (a.llm_adamw_policy) {
      std::string update_source = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + HFFN_M1_ADAMW_MSL;
      const std::string from = "hffn_adamw_slot<false,false>";
      const std::string to = "hffn_adamw_slot<true,true>";
      const auto pos = update_source.find(from);
      if (pos == std::string::npos || update_source.find(from, pos + from.size()) != std::string::npos)
        throw std::runtime_error("unexpected shared AdamW source");
      update_source.replace(pos, from.size(), to);
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.fastMathEnabled = NO;
      precise_adamw = [d newLibraryWithSource:[NSString stringWithUTF8String:update_source.c_str()] options:opts error:&er];
      if (!precise_adamw) throw std::runtime_error("precise AdamW shader compilation failed");
    }
    const char *forward_dir = inference_bundle ? inference_bundle : std::getenv("HFFN_INFERENCE_FORWARD_DIR");
    if (!forward_dir) { std::fprintf(stderr,"use the build/ runner launcher to prepare inference sources\n"); return 2; }
    auto forward_library = [&](const char *file) -> id<MTLLibrary> {
      NSString *path = [[NSString stringWithUTF8String:forward_dir] stringByAppendingPathComponent:[NSString stringWithUTF8String:file]];
      NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&er];
      id<MTLLibrary> library = source ? [d newLibraryWithSource:source options:nil error:&er] : nil;
      if (!library) { std::fprintf(stderr,"inference MSL: %s\n",er.localizedDescription.UTF8String); throw std::runtime_error("native Metal construction/execution failed"); }
      return library;
    };
    id<MTLLibrary> inference_up = forward_library("up.metal"), inference_down = forward_library("down.metal");
    id<MTLCommandQueue> q = [d newCommandQueue];
    if(!q)throw std::runtime_error("Metal command queue creation failed");
    NativeDiagnostics *diagnostic_state=diagnostics.get();
    auto pp = ^(NSString *n) {
      NSError *e = nil;
      id<MTLLibrary> selected = [n isEqualToString:@"fup"] ? inference_up : [n isEqualToString:@"fdown"] ? inference_down : ([n isEqualToString:@"adamw"] && precise_adamw ? precise_adamw : l);
      NSString *symbol = [n isEqualToString:@"fup"] ? @"hffn_inference_up" : [n isEqualToString:@"fdown"] ? @"hffn_inference_down" : n;
      id<MTLComputePipelineState> p =
          [d newComputePipelineStateWithFunction:[selected newFunctionWithName:symbol]
                                           error:&e];
      if (!p) {
        std::fprintf(stderr, "pipe: %s\n",
                     [[e.localizedDescription description] UTF8String]);
        throw std::runtime_error("native Metal construction/execution failed");
      }
      diagnostic_state->pipeline_names[(__bridge void*)p]=symbol.UTF8String;
      return p;
    };
    std::vector<float> ze(z.size()), he(h.size()), ye(y.size()), gz(z.size()),
        gx(x.size()), gu(E), gd(ED), gbu(a.hidden), gbd(a.output);
    // Segmented AdamW moments exist only without the flat pool, which holds the moments otherwise.
    const bool segmented_moments = !a.flat_adamw_pool;
    std::vector<float> um(segmented_moments ? E : 0), uv(segmented_moments ? E : 0),
        dm(segmented_moments ? ED : 0), dv(segmented_moments ? ED : 0),
        bum(segmented_moments ? a.hidden : 0), buv(segmented_moments ? a.hidden : 0),
        bdm(segmented_moments ? a.output : 0), bdv(segmented_moments ? a.output : 0);
    const void *dr =
        c16 ? (const void *)dr16.data() : (const void *)dr32.data();
    const void *ur =
        c16 ? (const void *)ur16.data() : (const void *)ur32.data();
    size_t rb = E * (c16 ? 2 : 4), dr_bytes=ED*(c16?2:4);
    std::vector<uint16_t> idx16, target16;
    if (compact_indices) {
      idx16.assign(idx.begin(), idx.end());
      target16.assign(target.begin(), target.end());
    }
    const void *idx_data = compact_indices
                               ? static_cast<const void *>(idx16.data())
                               : static_cast<const void *>(idx.data());
    const void *target_data = compact_indices
                                  ? static_cast<const void *>(target16.data())
                                  : static_cast<const void *>(target.data());
    const size_t index_bytes =
        E * (compact_indices ? sizeof(uint16_t) : sizeof(uint32_t));
    std::vector<uint16_t> hidden16;
    std::vector<uint32_t> hidden32;
    const void *hidden_data = nullptr;
    size_t hidden_bytes = 0;
    if (a.route_owner) {
      // The route codec already supplies r. Keep a one-element binding only
      // because the shared MSL signature also serves the A3 control kernel.
      hidden16.resize(1);
      hidden_data = hidden16.data(); hidden_bytes = sizeof(uint16_t);
    } else if (a.hidden <= 65535u) {
      hidden16.resize(E);
      for (uint32_t p = 0; p < E; ++p) hidden16[p] = uint16_t(p / a.degree);
      hidden_data = hidden16.data(); hidden_bytes = E * sizeof(uint16_t);
    } else {
      hidden32.resize(E);
      for (uint32_t p = 0; p < E; ++p) hidden32[p] = p / a.degree;
      hidden_data = hidden32.data(); hidden_bytes = E * sizeof(uint32_t);
    }
    const MTLResourceOptions ro = a.private_readonly
                                      ? MTLResourceStorageModePrivate
                                      : MTLResourceStorageModeShared;
    // When the caller supplies the compact pool, these initial construction
    // placeholders are replaced by flat descriptors before any dispatch.
    // Keep the native-owned and segmented paths byte-for-byte in behavior.
    id<MTLBuffer> external_parameter_buffer = (__bridge id<MTLBuffer>)external_parameter_handle;
    id<MTLBuffer> xb = upload(d, q, x_micro[0].data(), x.size() * 4, ro),
                  gyb = upload(d, q, gy_micro[0].data(), gy.size() * 4, ro),
                  ib = upload(d, q, idx_data, index_bytes, ro),
                  tb = upload(d, q, target_data, ED*(compact_indices?sizeof(uint16_t):sizeof(uint32_t)), ro),
                  upb = external_parameter_buffer ? external_parameter_buffer : buf(d, up.data(), up.size() * 4),
                  db = external_parameter_buffer ? external_parameter_buffer : buf(d, down.data(), down.size() * 4),
                  bub = external_parameter_buffer ? external_parameter_buffer : buf(d, bu.data(), bu.size() * 4),
                  bdb = external_parameter_buffer ? external_parameter_buffer : buf(d, bd.data(), bd.size() * 4),
                  zb = buf(d, ze.data(), ze.size() * 4),
                  hb = buf(d, he.data(), he.size() * 4),
                  yb = buf(d, ye.data(), ye.size() * 4),
                  dob = upload(d, q, doff.data(), doff.size() * 4, ro),
                  uob = upload(d, q, uoff.data(), uoff.size() * 4, ro),
                  drb = upload(d, q, dr, dr_bytes, ro), urb = upload(d, q, ur, rb, ro),
                  dsegb = upload(d, q, dseg.data(), dseg.size() * sizeof(uint32_t), ro),
                  drsegb = upload(d, q, drseg16.data(), drseg16.size() * sizeof(uint16_t), ro),
                  otileob = upload(d, q, otile_off.data(), otile_off.size() * sizeof(uint32_t), ro),
                  otilerb = upload(d, q, otile_route.data(), otile_route.size() * sizeof(uint32_t), ro),
                  upacket_mid_b = upload(d, q, upacket_mid.data(),
                                          upacket_mid.size() * sizeof(uint32_t), ro),
                  hob = upload(d, q, hidden_data, hidden_bytes, ro),
                  gzb = buf(d, gz.data(), gz.size() * 4),
                  gxb = buf(d, gx.data(), gx.size() * 4),
                  gub = buf(d, gu.data(), gu.size() * 4),
                  gdb = buf(d, gd.data(), gd.size() * 4),
                  gbub = buf(d, gbu.data(), gbu.size() * 4),
                  gbdb = buf(d, gbd.data(), gbd.size() * 4);
    id<MTLBuffer> umb = nil, uvb = nil, dmb = nil, dvb = nil, bumb = nil, buvb = nil, bdmb = nil, bdvb = nil;
    if (segmented_moments) {
      umb = buf(d, um.data(), um.size() * 4);
      uvb = buf(d, uv.data(), uv.size() * 4);
      dmb = buf(d, dm.data(), dm.size() * 4);
      dvb = buf(d, dv.data(), dv.size() * 4);
      bumb = buf(d, bum.data(), bum.size() * 4);
      buvb = buf(d, buv.data(), buv.size() * 4);
      bdmb = buf(d, bdm.data(), bdm.size() * 4);
      bdvb = buf(d, bdv.data(), bdv.size() * 4);
    }
    std::vector<float> ylocal;
    id<MTLBuffer> ylocalb = nil;
    if (a.fdown_group_private) {
      ylocal.resize(size_t(4u) * a.output * a.n);
      ylocalb = buf(d, ylocal.data(), ylocal.size() * sizeof(float));
    }
    std::vector<id<MTLBuffer>> x_micro_b{xb}, gy_micro_b{gyb};
    for (uint32_t k = 1; k < a.accum_steps; ++k) {
      x_micro_b.push_back(upload(d, q, x_micro[k].data(), x.size() * 4, ro));
      gy_micro_b.push_back(upload(d, q, gy_micro[k].data(), gy.size() * 4, ro));
    }
    AdamWParams pu = adamw_params(E), pd = adamw_params(ED),
                pbu = adamw_params(a.hidden), pbd = adamw_params(a.output);
    id<MTLBuffer> pub = buf(d, &pu, sizeof(pu)), pdb = buf(d, &pd, sizeof(pd)),
                  pbub = buf(d, &pbu, sizeof(pbu)),
                  pbdb = buf(d, &pbd, sizeof(pbd));
    const uint32_t count_e = E, count_h = a.hidden, count_o = a.output;
    id<MTLBuffer> ceb = buf(d, &count_e, sizeof(count_e)),
                  chb = buf(d, &count_h, sizeof(count_h)),
                  cob = buf(d, &count_o, sizeof(count_o));
    const uint32_t flat_down = E, flat_bup = E + ED,
                   flat_bdown = E + ED + a.hidden,
                   flat_total = flat_bdown + a.output;
    std::vector<float> flat_p, flat_g(flat_total), flat_m(flat_total), flat_v(flat_total);
    if (external_parameter_buffer) {
      if (!created || !supplied || !a.flat_adamw_pool || external_parameter_buffer.device != d ||
          external_parameter_count != flat_total || external_parameter_offset % sizeof(float) ||
          external_parameter_offset > external_parameter_buffer.length ||
          uint64_t(flat_total) * sizeof(float) > external_parameter_buffer.length - external_parameter_offset)
        throw std::invalid_argument("invalid external compact parameter pool");
    } else {
      flat_p.resize(flat_total);
      std::memcpy(flat_p.data(), up.data(), E * sizeof(float));
      std::memcpy(flat_p.data() + flat_down, down.data(), ED * sizeof(float));
      std::memcpy(flat_p.data() + flat_bup, bu.data(), a.hidden * sizeof(float));
      std::memcpy(flat_p.data() + flat_bdown, bd.data(), a.output * sizeof(float));
    }
    id<MTLBuffer> flat_pb = external_parameter_buffer ? external_parameter_buffer : buf(d, flat_p.data(), flat_total * sizeof(float)),
                  flat_gb = buf(d, flat_g.data(), flat_total * sizeof(float)),
                  flat_mb = buf(d, flat_m.data(), flat_total * sizeof(float)),
                  flat_vb = buf(d, flat_v.data(), flat_total * sizeof(float));
    const NSUInteger flat_pb_offset = external_parameter_buffer ? NSUInteger(external_parameter_offset) : 0;
    auto flat_param_view = [&](NSUInteger offset = 0) { return ::view(flat_pb, flat_pb_offset + offset); };
    const uint32_t tiles = (a.n + 31u) / 32u;
    std::vector<float> partial_edge, partial_out, partial_hidden;
    id<MTLBuffer> pdownb = nil, pupb = nil, poutb = nil, phidb = nil,
                  owner_partial_b = nil;
    if (a.tiled_n) {
      partial_edge.resize(E * tiles);
      partial_out.resize(a.output * tiles);
      partial_hidden.resize(a.hidden * tiles);
      pdownb = buf(d, partial_edge.data(), partial_edge.size() * sizeof(float));
      pupb = buf(d, partial_edge.data(), partial_edge.size() * sizeof(float));
      poutb = buf(d, partial_out.data(), partial_out.size() * sizeof(float));
      phidb = buf(d, partial_hidden.data(), partial_hidden.size() * sizeof(float));
    }
    if (a.owner_tile) {
      // [gup | gdown | gbup | gbdown, tile].  This is bounded by a single
      // batch here; the next A1 step will reuse it per chunk for very large N.
      const uint32_t owner_tiles = (a.n + a.owner_tile_n - 1u) / a.owner_tile_n;
      std::vector<float> owner_partial(size_t(flat_total) * owner_tiles);
      owner_partial_b = buf(d, owner_partial.data(),
                            owner_partial.size() * sizeof(float));
    }
    AdamWParams flat_q = adamw_params(flat_total);
    id<MTLBuffer> flat_qb = buf(d, &flat_q, sizeof(flat_q)),
                  flat_count_b = buf(d, &flat_total, sizeof(flat_total));
    NSArray *ps = HFFN_M1_PLAN_PERSISTENT;
    NSArray *bs = @[
      @[ xb, ib, upb, bub, zb, hb ], @[ hb, dob, drb, db, bdb, yb, hob ],
      @[ hb, gyb, dob, drb, gdb, gbdb, hob ], @[ zb, gyb, tb, db, gzb, gbub ],
      @[ xb, gzb, uob, urb, upb, gub, gxb, hob ]
    ];
    if (segmented_moments)
      bs = [bs arrayByAddingObjectsFromArray:@[
        @[ upb, gub, umb, uvb, pub, ceb ], @[ db, gdb, dmb, dvb, pdb, ceb ],
        @[ bub, gbub, bumb, buvb, pbub, chb ], @[ bdb, gbdb, bdmb, bdvb, pbdb, cob ]
      ]];
    MTLSize f = MTLSizeMake((a.hidden + 15) / 16, 1, 1),
            o = MTLSizeMake((a.output + 15) / 16, 1, 1),
            i = MTLSizeMake((a.input + 15) / 16, 1, 1),
            ae = MTLSizeMake((E + 511) / 512, 1, 1),
            ah = MTLSizeMake((a.hidden + 511) / 512, 1, 1),
            ao = MTLSizeMake((a.output + 511) / 512, 1, 1);
    NSArray *gs = @[
      [NSValue valueWithBytes:&f objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&f objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&i objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&ae objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&ae objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&ah objCType:@encode(MTLSize)],
      [NSValue valueWithBytes:&ao objCType:@encode(MTLSize)]
    ];
    if (a.flat_adamw_pool) {
      const NSUInteger eu = NSUInteger(E) * sizeof(float), ed=NSUInteger(ED)*sizeof(float),
                       hu = NSUInteger(a.hidden) * sizeof(float),
                       ou = NSUInteger(a.output) * sizeof(float);
      ps = HFFN_M1_PLAN_PERSISTENT_FLAT;
      bs = @[
        @[ xb, ib, flat_param_view(), flat_param_view( (eu + ed)), zb, hb ],
        @[ hb, dob, drb, flat_param_view( eu), flat_param_view( (eu + ed) + hu), yb, hob ],
        @[ hb, gyb, dob, drb, view(flat_gb, eu), view(flat_gb, (eu + ed) + hu), hob ],
        @[ zb, gyb, tb, flat_param_view( eu), gzb, view(flat_gb, (eu + ed)) ],
        @[ xb, gzb, uob, urb, flat_param_view(), view(flat_gb), gxb, hob ],
        @[
          flat_param_view(), view(flat_gb), view(flat_mb), view(flat_vb), flat_qb,
          flat_count_b
        ]
      ];
      MTLSize af = MTLSizeMake((flat_total + 511) / 512, 1, 1);
      gs = @[
        [NSValue valueWithBytes:&f objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&f objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&i objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&af objCType:@encode(MTLSize)]
      ];
    }
    if (a.owner_tile) {
      const uint32_t owner_tiles = (a.n + a.owner_tile_n - 1u) / a.owner_tile_n;
      const NSUInteger eu = NSUInteger(E) * sizeof(float), ed=NSUInteger(ED)*sizeof(float),
                       hu = NSUInteger(a.hidden) * sizeof(float);
      MTLSize ft = MTLSizeMake((a.hidden + 15) / 16, owner_tiles, 1),
              ot = MTLSizeMake((a.output + 15) / 16, owner_tiles, 1),
              it = MTLSizeMake(a.binput_split4 ? (a.input + 3) / 4
                               : a.binput_split2 ? (a.input + 7) / 8
                                                : (a.input + 15) / 16,
                               owner_tiles, 1),
              ar = MTLSizeMake((flat_total + 511) / 512, 1, 1),
              af = MTLSizeMake((flat_total + 511) / 512, 1, 1);
      // owner-tile requires the canonical flat parameter pool: its compact
      // partial layout is deliberately identical to that pool's segment order.
      ps = a.bdown_fused && a.owner_tile_n == 128u ?
          HFFN_M1_PLAN_OWNER_FUSED : a.binput_separate ?
          HFFN_M1_PLAN_OWNER_SEPARATE :
          HFFN_M1_PLAN_OWNER;
      NSArray *owner_common_bs = @[
        @[ xb, ib, flat_param_view(), flat_param_view( (eu + ed)), zb, hb ],
        @[ hb, dob, drb, flat_param_view( eu), flat_param_view( (eu + ed) + hu), yb, hob ],
        @[ hb, gyb, dob, drb, owner_partial_b, hob ],
        @[ zb, gyb, tb, flat_param_view( eu), gzb, owner_partial_b ],
        @[ xb, gzb, uob, urb, flat_param_view(), gxb, owner_partial_b, hob ],
        @[ owner_partial_b, view(flat_gb) ],
        @[ flat_param_view(), view(flat_gb), view(flat_mb), view(flat_vb), flat_qb,
           flat_count_b ]
      ];
      NSArray *owner_gup_hidden_bs =
          @[ xb, gzb, ib, uob, flat_param_view(), gxb, owner_partial_b, hob ];
      NSArray *owner_packet_binput_bs =
          @[ xb, gzb, uob, urb, flat_param_view(), gxb, owner_partial_b, hob,
             upacket_mid_b ];
      NSArray *owner_bdown_fused_bs =
          @[ zb, hb, gyb, tb, flat_param_view( eu), gzb, owner_partial_b ];
      NSArray *owner_bdown_gup_fused_bs =
          @[ zb, hb, gyb, tb, flat_param_view( eu), gzb, owner_partial_b, ib, xb ];
      bs = a.bdown_fused && a.owner_tile_n == 128u ? @[
        owner_common_bs[0], owner_common_bs[1], owner_common_bs[2],
        (a.bdown_gup_fused ? owner_bdown_gup_fused_bs : owner_bdown_fused_bs),
        ((a.packet_binput_staged || a.packet_binput_staged64)
             ? owner_packet_binput_bs : owner_common_bs[4]), owner_common_bs[5],
        owner_common_bs[6]
      ] : a.binput_separate ? @[
        owner_common_bs[0], owner_common_bs[1], owner_common_bs[2],
        owner_common_bs[3], owner_common_bs[4],
        (a.binput_gup_hidden ? owner_gup_hidden_bs : owner_common_bs[4]),
        owner_common_bs[5], owner_common_bs[6]
      ] : owner_common_bs;
      NSArray *owner_common_gs = @[
        [NSValue valueWithBytes:&f objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&ot objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&ft objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&it objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&ar objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&af objCType:@encode(MTLSize)]
      ];
      gs = a.bdown_fused && a.owner_tile_n == 128u ? owner_common_gs : a.binput_separate ? @[
        owner_common_gs[0], owner_common_gs[1], owner_common_gs[2],
        owner_common_gs[3], owner_common_gs[4],
        (a.binput_gup_hidden ? owner_common_gs[3] : owner_common_gs[4]),
        owner_common_gs[5], owner_common_gs[6]
      ] : owner_common_gs;
    }
    if (a.tiled_n) {
      const NSUInteger eu = NSUInteger(E) * sizeof(float), ed=NSUInteger(ED)*sizeof(float),
                       hu = NSUInteger(a.hidden) * sizeof(float),
                       ou = NSUInteger(a.output) * sizeof(float);
      NSValue *pup = a.flat_adamw_pool ? flat_param_view() : view(upb);
      NSValue *pdown = a.flat_adamw_pool ? flat_param_view( eu) : view(db);
      NSValue *gup = a.flat_adamw_pool ? view(flat_gb) : view(gub);
      NSValue *gdown = a.flat_adamw_pool ? view(flat_gb, eu) : view(gdb);
      NSValue *gbup = a.flat_adamw_pool ? view(flat_gb, (eu + ed)) : view(gbub);
      NSValue *gbdown = a.flat_adamw_pool ? view(flat_gb, (eu + ed) + hu) : view(gbdb);
      const MTLSize td = MTLSizeMake((E * tiles + 511) / 512, 1, 1),
                    th = MTLSizeMake((a.hidden * tiles + 511) / 512, 1, 1),
                    ti = MTLSizeMake((E * tiles + 511) / 512, 1, 1),
                    re = MTLSizeMake((E + 511) / 512, 1, 1),
                    rhg = MTLSizeMake((a.hidden + 511) / 512, 1, 1),
                    rog = MTLSizeMake((a.output + 511) / 512, 1, 1),
                    af = MTLSizeMake((flat_total + 511) / 512, 1, 1);
      ps = HFFN_M1_PLAN_TILED;
      bs = @[
        @[ xb, ib, pup, a.flat_adamw_pool ? flat_param_view( (eu + ed)) : view(bub), zb, hb ],
        @[ hb, dob, drb, pdown, a.flat_adamw_pool ? flat_param_view( (eu + ed) + hu) : view(bdb), yb, hob ],
        @[ hb, gyb, dob, drb, pdownb, poutb ], @[ pdownb, gdown ], @[ poutb, gbdown ],
        @[ zb, gyb, tb, pdown, gzb, phidb ], @[ phidb, gbup ],
        @[ xb, gzb, uob, urb, pup, pupb, gxb ], @[ pupb, gup ],
        a.flat_adamw_pool ? @[ flat_param_view(), view(flat_gb), view(flat_mb), view(flat_vb), flat_qb, flat_count_b ] : @[ upb, gub, umb, uvb, pub, ceb ]
      ];
      if (segmented_moments)
        bs = [bs arrayByAddingObjectsFromArray:@[
          @[ db, gdb, dmb, dvb, pdb, ceb ], @[ bub, gbub, bumb, buvb, pbub, chb ], @[ bdb, gbdb, bdmb, bdvb, pbdb, cob ]
        ]];
      gs = @[
        [NSValue valueWithBytes:&f objCType:@encode(MTLSize)], [NSValue valueWithBytes:&o objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&td objCType:@encode(MTLSize)], [NSValue valueWithBytes:&re objCType:@encode(MTLSize)], [NSValue valueWithBytes:&rog objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&th objCType:@encode(MTLSize)], [NSValue valueWithBytes:&rhg objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&ti objCType:@encode(MTLSize)], [NSValue valueWithBytes:&re objCType:@encode(MTLSize)],
        [NSValue valueWithBytes:&af objCType:@encode(MTLSize)], [NSValue valueWithBytes:&ae objCType:@encode(MTLSize)], [NSValue valueWithBytes:&ae objCType:@encode(MTLSize)], [NSValue valueWithBytes:&ah objCType:@encode(MTLSize)], [NSValue valueWithBytes:&ao objCType:@encode(MTLSize)]
      ];
      if (a.flat_adamw_pool) { ps = [ps subarrayWithRange:NSMakeRange(0, 10)]; bs = [bs subarrayWithRange:NSMakeRange(0, 10)]; gs = [gs subarrayWithRange:NSMakeRange(0, 10)]; }
    }
    // Immutable compact routes are built once; values remain live views of
    // the same optimizer-owned pools used by backward and AdamW.
    std::vector<int32_t> forward_offsets(up_hidden_offsets.begin(),up_hidden_offsets.end()), forward_slots(ED), forward_hidden(ED);
    for (uint32_t qi=0;qi<ED;++qi) {
      uint32_t slot = route_slot(c16?uint32_t(dr16[qi]):dr32[qi],down_hidden_offsets,c);
      forward_slots[qi]=int32_t(slot); forward_hidden[qi]=int32_t((c16?uint32_t(dr16[qi]):dr32[qi])>>c.edge_bits);
    }
    id<MTLBuffer> fuoff=buf(d,forward_offsets.data(),forward_offsets.size()*4),
      fslots=buf(d,forward_slots.data(),forward_slots.size()*4),
      fhidden=buf(d,forward_hidden.data(),forward_hidden.size()*4),
      findex=buf(d,idx.data(),idx.size()*4), nsbuf=buf(d,&a.n,sizeof(a.n));
    uint32_t one=1; float gate=1;
    id<MTLBuffer> stepsbuf=buf(d,&one,sizeof(one)), gatebuf=buf(d,&gate,sizeof(gate));
    NSMutableArray *inference_ps=[ps mutableCopy], *inference_bs=[bs mutableCopy], *inference_gs=[gs mutableCopy];
    inference_ps[0]=pp(@"fup"); inference_ps[1]=pp(@"fdown");
    id up_values=a.flat_adamw_pool ? (id)flat_param_view(0) : (id)upb;
    id down_values=a.flat_adamw_pool ? (id)flat_param_view(NSUInteger(E)*sizeof(float)) : (id)db;
    id up_bias=a.flat_adamw_pool ? (id)flat_param_view(NSUInteger(E+ED)*sizeof(float)) : (id)bub;
    id down_bias=a.flat_adamw_pool ? (id)flat_param_view(NSUInteger(E+ED+a.hidden)*sizeof(float)) : (id)bdb;
    inference_bs[0]=@[nsbuf,fuoff,findex,up_values,up_bias,xb,hb,zb];
    inference_bs[1]=@[nsbuf,dob,fhidden,down_values,fslots,hb,gatebuf,down_bias,yb,stepsbuf];
    NSString *planpath=[[NSString stringWithUTF8String:forward_dir] stringByAppendingPathComponent:@"dispatch.json"];
    NSData *plandata=[NSData dataWithContentsOfFile:planpath];
    NSDictionary *plan=plandata ? [NSJSONSerialization JSONObjectWithData:plandata options:0 error:&er] : nil;
    if (!plan || [plan[@"input"] unsignedIntValue]!=a.input || [plan[@"hidden"] unsignedIntValue]!=a.hidden || [plan[@"output"] unsignedIntValue]!=a.output || [plan[@"n"] unsignedIntValue]!=a.n) return 2;
    uint32_t up_groups=[plan[@"up_groups"] unsignedIntValue],down_groups=[plan[@"down_groups"] unsignedIntValue];
    MTLSize up_grid=MTLSizeMake(up_groups,1,1),down_grid=MTLSizeMake(down_groups,1,1);
    inference_gs[0]=[NSValue valueWithBytes:&up_grid objCType:@encode(MTLSize)];
    inference_gs[1]=[NSValue valueWithBytes:&down_grid objCType:@encode(MTLSize)];
    ps=inference_ps;bs=inference_bs;gs=inference_gs;
    if(exact_offsets){
      id<MTLBuffer> down_row_buffer=upload(d,q,down_hidden_offsets.data(),down_hidden_offsets.size()*4,ro);
      const std::set<std::string> down_kernels={"bdown","bhidden","bdown_owner_tile","bhidden_owner_tile","bdown_bias_owner_tile","bdown_bhidden_fused_owner_tile"};
      NSMutableArray* extended=[bs mutableCopy];
      for(NSUInteger j=2;j<ps.count;++j){
        std::string name=diagnostic_state->pipeline_names.at((__bridge void*)ps[j]);
        if(name=="adamw" || name=="reduce_flat_partials")continue;
        NSMutableArray* bindings=[bs[j] mutableCopy];
        [bindings addObject:down_kernels.count(name)?down_row_buffer:fuoff];
        extended[j]=bindings;
      }
      bs=extended;
    }
    NSMutableDictionary *owners=[NSMutableDictionary dictionary];
    if (xb) owners[@"xb"]=xb;
    if (gyb) owners[@"gyb"]=gyb;
    if (ib) owners[@"ib"]=ib;
    if (tb) owners[@"tb"]=tb;
    if (upb) owners[@"upb"]=upb;
    if (db) owners[@"db"]=db;
    if (bub) owners[@"bub"]=bub;
    if (bdb) owners[@"bdb"]=bdb;
    if (zb) owners[@"zb"]=zb;
    if (hb) owners[@"hb"]=hb;
    if (yb) owners[@"yb"]=yb;
    if (dob) owners[@"dob"]=dob;
    if (uob) owners[@"uob"]=uob;
    if (drb) owners[@"drb"]=drb;
    if (urb) owners[@"urb"]=urb;
    if (dsegb) owners[@"dsegb"]=dsegb;
    if (drsegb) owners[@"drsegb"]=drsegb;
    if (otileob) owners[@"otileob"]=otileob;
    if (otilerb) owners[@"otilerb"]=otilerb;
    if (upacket_mid_b) owners[@"upacket_mid_b"]=upacket_mid_b;
    if (hob) owners[@"hob"]=hob;
    if (gzb) owners[@"gzb"]=gzb;
    if (gxb) owners[@"gxb"]=gxb;
    if (gub) owners[@"gub"]=gub;
    if (gdb) owners[@"gdb"]=gdb;
    if (gbub) owners[@"gbub"]=gbub;
    if (gbdb) owners[@"gbdb"]=gbdb;
    if (umb) owners[@"umb"]=umb;
    if (uvb) owners[@"uvb"]=uvb;
    if (dmb) owners[@"dmb"]=dmb;
    if (dvb) owners[@"dvb"]=dvb;
    if (bumb) owners[@"bumb"]=bumb;
    if (buvb) owners[@"buvb"]=buvb;
    if (bdmb) owners[@"bdmb"]=bdmb;
    if (bdvb) owners[@"bdvb"]=bdvb;
    if (ylocalb) owners[@"ylocalb"]=ylocalb;
    if (pub) owners[@"pub"]=pub;
    if (pdb) owners[@"pdb"]=pdb;
    if (pbub) owners[@"pbub"]=pbub;
    if (pbdb) owners[@"pbdb"]=pbdb;
    if (ceb) owners[@"ceb"]=ceb;
    if (chb) owners[@"chb"]=chb;
    if (cob) owners[@"cob"]=cob;
    if (flat_pb) owners[@"flat_pb"]=flat_pb;
    if (flat_gb) owners[@"flat_gb"]=flat_gb;
    if (flat_mb) owners[@"flat_mb"]=flat_mb;
    if (flat_vb) owners[@"flat_vb"]=flat_vb;
    if (pdownb) owners[@"pdownb"]=pdownb;
    if (pupb) owners[@"pupb"]=pupb;
    if (poutb) owners[@"poutb"]=poutb;
    if (phidb) owners[@"phidb"]=phidb;
    if (owner_partial_b) owners[@"owner_partial_b"]=owner_partial_b;
    if (flat_qb) owners[@"flat_qb"]=flat_qb;
    if (flat_count_b) owners[@"flat_count_b"]=flat_count_b;
    if (fuoff) owners[@"fuoff"]=fuoff;
    if (fslots) owners[@"fslots"]=fslots;
    if (fhidden) owners[@"fhidden"]=fhidden;
    if (findex) owners[@"findex"]=findex;
    if (nsbuf) owners[@"nsbuf"]=nsbuf;
    if (stepsbuf) owners[@"stepsbuf"]=stepsbuf;
    if (gatebuf) owners[@"gatebuf"]=gatebuf;
    NSMutableArray *input_owners=[NSMutableArray array], *gy_owners=[NSMutableArray array];
    for (id<MTLBuffer> b : x_micro_b) [input_owners addObject:b];
    for (id<MTLBuffer> b : gy_micro_b) [gy_owners addObject:b];
    owners[@"x_micro"]=[input_owners copy]; owners[@"gy_micro"]=[gy_owners copy];
    auto core=std::make_unique<HffnTrainContext>(a,q,ps,bs,gs,[owners copy],std::move(diagnostics));
    if (external_parameter_buffer) {
      core->external_parameters=external_parameter_buffer;
      core->external_parameter_offset=external_parameter_offset/sizeof(float);
    }
    if (created) { *created=core.release(); return 0; }
    auto p_up = [&] {
      return a.flat_adamw_pool ? (float *)flat_pb.contents
                               : (float *)upb.contents;
    };
    auto p_down = [&] {
      return a.flat_adamw_pool ? (float *)flat_pb.contents + flat_down
                               : (float *)db.contents;
    };
    auto p_bup = [&] {
      return a.flat_adamw_pool ? (float *)flat_pb.contents + flat_bup
                               : (float *)bub.contents;
    };
    auto p_bdown = [&] {
      return a.flat_adamw_pool ? (float *)flat_pb.contents + flat_bdown
                               : (float *)bdb.contents;
    };
    auto g_up = [&] {
      return a.flat_adamw_pool ? (float *)flat_gb.contents
                               : (float *)gub.contents;
    };
    auto g_down = [&] {
      return a.flat_adamw_pool ? (float *)flat_gb.contents + flat_down
                               : (float *)gdb.contents;
    };
    auto g_bup = [&] {
      return a.flat_adamw_pool ? (float *)flat_gb.contents + flat_bup
                               : (float *)gbub.contents;
    };
    auto g_bdown = [&] {
      return a.flat_adamw_pool ? (float *)flat_gb.contents + flat_bdown
                               : (float *)gbdb.contents;
    };
    auto m_up = [&] {
      return a.flat_adamw_pool ? (float *)flat_mb.contents
                               : (float *)umb.contents;
    };
    auto m_down = [&] {
      return a.flat_adamw_pool ? (float *)flat_mb.contents + flat_down
                               : (float *)dmb.contents;
    };
    auto m_bup = [&] {
      return a.flat_adamw_pool ? (float *)flat_mb.contents + flat_bup
                               : (float *)bumb.contents;
    };
    auto m_bdown = [&] {
      return a.flat_adamw_pool ? (float *)flat_mb.contents + flat_bdown
                               : (float *)bdmb.contents;
    };
    auto v_up = [&] {
      return a.flat_adamw_pool ? (float *)flat_vb.contents
                               : (float *)uvb.contents;
    };
    auto v_down = [&] {
      return a.flat_adamw_pool ? (float *)flat_vb.contents + flat_down
                               : (float *)dvb.contents;
    };
    auto v_bup = [&] {
      return a.flat_adamw_pool ? (float *)flat_vb.contents + flat_bup
                               : (float *)buvb.contents;
    };
    auto v_bdown = [&] {
      return a.flat_adamw_pool ? (float *)flat_vb.contents + flat_bdown
                               : (float *)bdvb.contents;
    };
    if(a.exact_support && a.dump_prefix){
      auto dump_u32=[&](const char* suffix,const std::vector<uint32_t>& values){
        std::string name=std::string(a.dump_prefix)+suffix;FILE* file=std::fopen(name.c_str(),"wb");
        if(!file)return false;bool ok=std::fwrite(values.data(),4,values.size(),file)==values.size();return std::fclose(file)==0&&ok;
      };
      if(!dump_u32("_up_offsets.u32",up_hidden_offsets)||!dump_u32("_down_offsets.u32",down_hidden_offsets)||
         !dump_u32("_up_idx.u32",idx)||!dump_u32("_down_idx.u32",target))return 1;
    }
    if (a.dump_prefix &&
        (!dump(a.dump_prefix, "_initial_down.f32", p_down(), ED) ||
         !dump(a.dump_prefix, "_initial_bdown.f32", p_bdown(), a.output))) {
      std::fprintf(stderr, "initial dump failed\n");
      return 1;
    }
    const NSUInteger forward_count = a.fdown_group_private ? 3u : 2u;
    NSArray *verify_ps=core->verify_ps, *verify_bs=core->verify_bs, *verify_gs=core->verify_gs;
    NSArray *update_ps=core->update_ps, *update_bs=core->update_bs, *update_gs=core->update_gs;
    NSArray *forward_ps=core->forward_ps, *forward_bs=core->forward_bs, *forward_gs=core->forward_gs;
    NSArray *backward_ps=core->backward_ps, *backward_bs=core->backward_bs, *backward_gs=core->backward_gs;
    auto bindings_for_microbatch = [&](uint32_t k) { return core->micro_bindings[k]; };
    auto train_step = [&] { return core->step(); };
    core->execute(forward_ps, forward_bs, forward_gs, 1);
    // Guaranteed nonfinite qualification control, after Metal fast-math.
    if(a.untimed && std::getenv("HFFN_QUAL_NAN_FAULT"))((float*)yb.contents)[0]=NAN;
    const double ez = err(a.cache_gelu_grad ? rdgelu : rz, (float *)zb.contents),
                 eh = err(rh, (float *)hb.contents),
                 ey = err(ry, (float *)yb.contents);
    core->execute(backward_ps, backward_bs, backward_gs, 1);
    const double egz = err(rgz, (float *)gzb.contents),
                 egx = err(rgx, (float *)gxb.contents), egu = err(rgu, g_up()),
                 egd = err(rgd, g_down()), egbu = err(rgbu, g_bup()),
                 egbd = err(rgbd, g_bdown());
    if (a.dump_prefix &&
        (!dump(a.dump_prefix, "_pre_gup.f32", g_up(), E) ||
         !dump(a.dump_prefix, "_pre_gdown.f32", g_down(), ED))) {
      std::fprintf(stderr, "pre-update dump failed\n");
      return 1;
    }
    std::memset(a.flat_adamw_pool ? flat_gb.contents : gub.contents, 0,
                a.flat_adamw_pool ? flat_total * sizeof(float)
                                  : E * sizeof(float));
    if (!a.flat_adamw_pool) {
      std::memset(gdb.contents, 0, ED * sizeof(float));
      std::memset(gbub.contents, 0, a.hidden * sizeof(float));
      std::memset(gbdb.contents, 0, a.output * sizeof(float));
    }
    std::vector<double> s;
    if (a.verify_steps) {
      for (uint32_t step = 1; step <= a.verify_steps; ++step) {
        if (a.untimed && a.dump_prefix) {
          core->begin();
          // Observe actual accumulated gradients before native AdamW clears
          // them. No scientific time/memory read; same kernels/buffer order.
          if(dispatch_trace)std::fprintf(dispatch_trace,"step %u begin\n",step);
          std::string prefix=std::string(a.dump_prefix)+"_step"+std::to_string(step);
          for (uint32_t k=0;k<a.accum_steps;++k) {
            core->accumulate(k);
            std::string mb=prefix+"_mb"+std::to_string(k);
            if(!dump(mb.c_str(),"_z.f32",(float*)zb.contents,a.hidden*a.n) ||
               !dump(mb.c_str(),"_h.f32",(float*)hb.contents,a.hidden*a.n) ||
               !dump(mb.c_str(),"_y.f32",(float*)yb.contents,a.output*a.n) ||
               !dump(mb.c_str(),"_gx.f32",(float*)gxb.contents,a.input*a.n) ||
               !dump(mb.c_str(),"_gz.f32",(float*)gzb.contents,a.hidden*a.n))return 1;
          }
          if (!dump(prefix.c_str(),"_scalars.f32",(float*)(a.flat_adamw_pool ? flat_qb.contents : pub.contents),sizeof(AdamWParams)/sizeof(float))) return 1;
          const char *activation_names[]={"x","gy","z","h","y","gx","gz","x0","gy0"};
          float *activation_values[]={x_micro.back().data(),gy_micro.back().data(),(float*)zb.contents,(float*)hb.contents,(float*)yb.contents,(float*)gxb.contents,(float*)gzb.contents,x_micro.front().data(),gy_micro.front().data()};
          uint32_t activation_counts[]={a.input*a.n,a.output*a.n,a.hidden*a.n,a.hidden*a.n,a.output*a.n,a.input*a.n,a.hidden*a.n,a.input*a.n,a.output*a.n};
          for(int part=0;part<9;++part)if(!dump(prefix.c_str(),("_"+std::string(activation_names[part])+".f32").c_str(),activation_values[part],activation_counts[part]))return 1;
          const char *names[]={"up","down","bup","bdown"};
          uint32_t counts[]={E,ED,a.hidden,a.output};
          float *parameters[]={p_up(),p_down(),p_bup(),p_bdown()};
          float *gradients[]={g_up(),g_down(),g_bup(),g_bdown()};
          float *first[]={m_up(),m_down(),m_bup(),m_bdown()};
          float *second[]={v_up(),v_down(),v_bup(),v_bdown()};
          for (int part=0;part<4;++part) {
            std::string base="_"+std::string(names[part]);
            if (!dump(prefix.c_str(),(base+"_pre.f32").c_str(),parameters[part],counts[part]) ||
                !dump(prefix.c_str(),(base+"_grad.f32").c_str(),gradients[part],counts[part]) ||
                !dump(prefix.c_str(),(base+"_m0.f32").c_str(),first[part],counts[part]) ||
                !dump(prefix.c_str(),(base+"_v0.f32").c_str(),second[part],counts[part])) return 1;
          }
          uint32_t fault_slot=0;
          if(std::getenv("HFFN_QUAL_UPDATE_FAULT"))for(uint32_t slot=1;slot<E;++slot)if(std::abs(g_up()[slot])>std::abs(g_up()[fault_slot]))fault_slot=slot;
          core->update();
          if(a.exact_support)for(int part=0;part<4;++part)
            if(!dump(prefix.c_str(),("_"+std::string(names[part])+"_after_grad.f32").c_str(),gradients[part],counts[part]))return 1;
          // Qualification-only wrong-update control, on the real compact pool.
          if(std::getenv("HFFN_QUAL_UPDATE_FAULT"))p_up()[fault_slot]+=.1f;
          if(dispatch_trace){std::fprintf(dispatch_trace,"step %u end\n",step);std::fflush(dispatch_trace);}
          for (int part=0;part<4;++part) {
            std::string base="_"+std::string(names[part]);
            if (!dump(prefix.c_str(),(base+"_post.f32").c_str(),parameters[part],counts[part]) ||
                !dump(prefix.c_str(),(base+"_m1.f32").c_str(),first[part],counts[part]) ||
                !dump(prefix.c_str(),(base+"_v1.f32").c_str(),second[part],counts[part])) return 1;
          }
        } else train_step();
      }
      s.assign(7, 0.0);
    } else {
      // Benchmark exactly one AdamW-correct training step from the same state.
      // Reusing a fixed step-1 bias-correction across a multi-step dispatch
      // would be a faster-looking but invalid training benchmark.
      auto reset_step_state = [&] {
        core->completed=0;
        if (a.flat_adamw_pool) {
          std::memcpy(flat_pb.contents, flat_p.data(),
                      flat_total * sizeof(float));
          std::memset(flat_gb.contents, 0, flat_total * sizeof(float));
          std::memset(flat_mb.contents, 0, flat_total * sizeof(float));
          std::memset(flat_vb.contents, 0, flat_total * sizeof(float));
          return;
        }
        std::memcpy(upb.contents, up.data(), E * sizeof(float));
        std::memcpy(db.contents, down.data(), ED * sizeof(float));
        std::memcpy(bub.contents, bu.data(), a.hidden * sizeof(float));
        std::memcpy(bdb.contents, bd.data(), a.output * sizeof(float));
        std::memset(gub.contents, 0, E * sizeof(float));
        std::memset(gdb.contents, 0, ED * sizeof(float));
        std::memset(gbub.contents, 0, a.hidden * sizeof(float));
        std::memset(gbdb.contents, 0, a.output * sizeof(float));
        std::memset(umb.contents, 0, E * sizeof(float));
        std::memset(uvb.contents, 0, E * sizeof(float));
        std::memset(dmb.contents, 0, ED * sizeof(float));
        std::memset(dvb.contents, 0, ED * sizeof(float));
        std::memset(bumb.contents, 0, a.hidden * sizeof(float));
        std::memset(buvb.contents, 0, a.hidden * sizeof(float));
        std::memset(bdmb.contents, 0, a.output * sizeof(float));
        std::memset(bdvb.contents, 0, a.output * sizeof(float));
      };
      for (uint32_t k = 0; k < a.warmup; ++k) {
        reset_step_state();
        train_step();
      }
      for (uint32_t k = 0; k < a.repeats; ++k) {
        reset_step_state();
        s.push_back(train_step());
      }
    }
    double mean_train_step_ms = 0.0, std_train_step_ms = 0.0;
    for (double sample : s)
      mean_train_step_ms += sample;
    mean_train_step_ms /= double(s.size());
    if (s.size() > 1) {
      for (double sample : s)
        std_train_step_ms += (sample - mean_train_step_ms) *
                             (sample - mean_train_step_ms);
      std_train_step_ms = std::sqrt(std_train_step_ms / double(s.size() - 1));
    }
    std::sort(s.begin(), s.end());
    std::string phase_json = "";
    if (a.profile_phases) {
      auto profile = [&](NSArray *phase_ps, NSArray *phase_bs, NSArray *phase_gs) {
        constexpr uint32_t samples = 3;
        const uint32_t warm = std::min<uint32_t>(a.warmup, 10u);
        const uint32_t repeats = std::min<uint32_t>(a.repeats, 5u);
        for (uint32_t k = 0; k < warm; ++k)
          core->execute(phase_ps, phase_bs, phase_gs, 1);
        std::vector<double> t;
        for (uint32_t k = 0; k < samples; ++k)
          t.push_back(core->execute(phase_ps, phase_bs, phase_gs, repeats));
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
      };
      const double fup_ms = profile([verify_ps subarrayWithRange:NSMakeRange(0, 1)],
                                    [verify_bs subarrayWithRange:NSMakeRange(0, 1)],
                                    [verify_gs subarrayWithRange:NSMakeRange(0, 1)]);
      const double fdown_ms = profile([verify_ps subarrayWithRange:NSMakeRange(1, forward_count - 1u)],
                                      [verify_bs subarrayWithRange:NSMakeRange(1, forward_count - 1u)],
                                      [verify_gs subarrayWithRange:NSMakeRange(1, forward_count - 1u)]);
      const double bdown_kernel_ms = profile([verify_ps subarrayWithRange:NSMakeRange(forward_count, 1)],
                                             [verify_bs subarrayWithRange:NSMakeRange(forward_count, 1)],
                                             [verify_gs subarrayWithRange:NSMakeRange(forward_count, 1)]);
      const double bdown_bias_ms = a.bdown_fused ? bdown_kernel_ms : 0.0;
      const double bdown_bhidden_fused_ms = a.bdown_fused ?
          profile([verify_ps subarrayWithRange:NSMakeRange(forward_count + 1u, 1)],
                  [verify_bs subarrayWithRange:NSMakeRange(forward_count + 1u, 1)],
                  [verify_gs subarrayWithRange:NSMakeRange(forward_count + 1u, 1)]) : 0.0;
      const double bdown_ms = a.bdown_fused ? bdown_bias_ms + bdown_bhidden_fused_ms
          : bdown_kernel_ms;
      const double bhidden_ms = a.bdown_fused ? 0.0 :
          profile([verify_ps subarrayWithRange:NSMakeRange(forward_count + 1u, 1)],
                  [verify_bs subarrayWithRange:NSMakeRange(forward_count + 1u, 1)],
                  [verify_gs subarrayWithRange:NSMakeRange(forward_count + 1u, 1)]);
      const double binput_gx_ms = profile([verify_ps subarrayWithRange:NSMakeRange(forward_count + 2u, 1)],
                                          [verify_bs subarrayWithRange:NSMakeRange(forward_count + 2u, 1)],
                                          [verify_gs subarrayWithRange:NSMakeRange(forward_count + 2u, 1)]);
      const double binput_gup_ms = a.binput_separate ?
          profile([verify_ps subarrayWithRange:NSMakeRange(forward_count + 3u, 1)],
                  [verify_bs subarrayWithRange:NSMakeRange(forward_count + 3u, 1)],
                  [verify_gs subarrayWithRange:NSMakeRange(forward_count + 3u, 1)]) : 0.0;
      const double binput_ms = binput_gx_ms + binput_gup_ms;
      const double adamw_ms = profile(update_ps, update_bs, update_gs);
      char phase_text[896];
      std::snprintf(phase_text, sizeof(phase_text),
                    ",\n  \"phase_median_gpu_ms\": {\"fup\": %.9f, \"fdown\": %.9f, \"bdown\": %.9f, \"bhidden\": %.9f, \"bdown_bias\": %.9f, \"bdown_bhidden_fused\": %.9f, \"binput\": %.9f, \"binput_gx\": %.9f, \"binput_gup\": %.9f, \"adamw\": %.9f}",
                    fup_ms, fdown_ms, bdown_ms, bhidden_ms, bdown_bias_ms,
                    bdown_bhidden_fused_ms, binput_ms,
                    binput_gx_ms, binput_gup_ms, adamw_ms);
      phase_json = phase_text;
    }
    if (a.dump_prefix &&
        (!dump(a.dump_prefix, "_up.f32", p_up(), E) ||
         !dump(a.dump_prefix, "_down.f32", p_down(), ED) ||
         !dump(a.dump_prefix, "_bup.f32", p_bup(), a.hidden) ||
         !dump(a.dump_prefix, "_bdown.f32", p_bdown(), a.output) ||
         !dump(a.dump_prefix, "_um.f32", m_up(), E) ||
         !dump(a.dump_prefix, "_uv.f32", v_up(), E) ||
         !dump(a.dump_prefix, "_dm.f32", m_down(), ED) ||
         !dump(a.dump_prefix, "_dv.f32", v_down(), ED) ||
         !dump(a.dump_prefix, "_bum.f32", m_bup(), a.hidden) ||
         !dump(a.dump_prefix, "_buv.f32", v_bup(), a.hidden) ||
         !dump(a.dump_prefix, "_bdm.f32", m_bdown(), a.output) ||
         !dump(a.dump_prefix, "_bdv.f32", v_bdown(), a.output))) {
      std::fprintf(stderr, "dump failed\n");
      return 1;
    }
    if (a.untimed) {
      NSDictionary *result=@{@"kind":@"hffn_metal_train_step_gelu_erf_accurate",@"untimed":@YES,
        @"route":c16 ? @"compact16" : @"compact32", @"forward_schedule":@"inference_policy",
        @"memory_bytes":[NSNull null],@"median_train_step_gpu_ms":[NSNull null],
        @"mean_train_step_gpu_ms":[NSNull null],@"std_train_step_gpu_ms":[NSNull null],
        @"dimensions":@{@"input":@(a.input),@"hidden":@(a.hidden),@"output":@(a.output),@"N":@(a.n),@"degree":@(a.degree),@"accum_steps":@(a.accum_steps)},
        @"max_abs_error":@{@"z":@(ez),@"h":@(eh),@"y":@(ey),@"gz":@(egz),@"gx":@(egx),@"gup":@(egu),@"gdown":@(egd),@"gbup":@(egbu),@"gbdown":@(egbd)}};
      if(a.exact_support){
        NSMutableDictionary* result_with_support=[result mutableCopy];
        result_with_support[@"addressing"]=exact_offsets ? @"exact_offsets" : @"uniform_degree";
        result_with_support[@"support"]=@{@"up_edges":@(E),@"down_edges":@(ED),@"padded_slots":@0,@"exact_offsets":@YES};
        result=result_with_support;
      }
      NSData *encoded=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&er];
      if (!encoded || ![encoded writeToFile:[NSString stringWithUTF8String:a.path] options:NSDataWritingAtomic error:&er]) return 1;
      std::printf("%s\n",[[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding] UTF8String]);
      return 0;
    }
    const uint64_t compact_parameter_bytes = uint64_t(flat_total) * sizeof(float),
                   compact_training_state_bytes = compact_parameter_bytes * 4u,
                   dense_parameter_reference_bytes =
                       uint64_t(a.hidden) * (a.input + a.output + 1u) * sizeof(float) +
                       uint64_t(a.output) * sizeof(float),
                   dense_activation_workspace_bytes =
                       uint64_t(a.n) * (uint64_t(2u * a.input) +
                                        uint64_t(2u * a.output) +
                                        uint64_t(3u * a.hidden)) * sizeof(float) +
                       (a.fdown_group_private
                            ? uint64_t(4u) * a.output * a.n * sizeof(float)
                            : 0u),
                   route_metadata_bytes = uint64_t(E)*12u + uint64_t(a.hidden+1u)*4u + 12u + uint64_t(E) *
                           (uint64_t(compact_indices ? 4u : 8u) +
                            uint64_t(c16 ? 4u : 8u)) +
                       uint64_t(a.input + a.output + 2u) * sizeof(uint32_t) +
                       ((a.fdown_segmented_route || a.fdown_group_private)
                            ? uint64_t(E) * sizeof(uint16_t) +
                                  uint64_t(a.output) * 5u * sizeof(uint32_t)
                            : 0u) +
                       (a.fdown_output_tile4
                            ? uint64_t(E) * sizeof(uint32_t) +
                                  uint64_t(a.output / 4u + 1u) * sizeof(uint32_t)
                            : 0u);
    const std::string schedule = a.owner_tile
        ? "owner_tile_N" + std::to_string(a.owner_tile_n)
        : (a.tiled_n ? "tiled_N32" : "persistent_N");
    const uint32_t forward_tile =
        a.forward_owner_tile_n ? a.forward_owner_tile_n : a.owner_tile_n;
    const std::string forward_schedule = "inference_policy";
    const char *fdown_schedule = a.fdown_group_private
        ? "group_private_G4_then_reduce_N128"
        : a.fdown_output_tile4
        ? "output_tile4_hidden_grouped_N128"
        : a.fdown_segmented_route
        ? "segmented_route16_output_owner_N128"
        : a.fdown_fused && a.owner_tile_n == 128u
        ? "fused_output_owner_N128" : "separate_output_owner";
    const char *binput_schedule = a.packet_binput_staged
        ? "packet_staged_owner_N128"
        : a.packet_binput_staged64
        ? "packet_staged64_owner_N128"
        : a.binput_fused1 && a.owner_tile_n == 128u
        ? "fused_inverse_owner_N128" : "separate_inverse_owner";
    const char *bdown_schedule = a.bdown_fused && a.owner_tile_n == 128u
        ? "fused_hidden_owner_N128" : "separate_output_hidden_owner";
    NSString *j = [NSString
        stringWithFormat:
            @"{\n  \"kind\": \"hffn_metal_train_step_gelu_erf_accurate\",\n  "
            @"\"route\": "
            @"\"%s\",\n  \"dimensions\": {\"input\": %u, \"hidden\": %u, "
            @"\"output\": %u, \"N\": %u, \"degree\": %u, \"accum_steps\": "
            @"%u, \"adamw_pool\": \"%s\", \"backward_schedule\": \"%s\", \"forward_schedule\": \"%s\", \"fdown_schedule\": \"%s\", \"binput_schedule\": \"%s\", \"bdown_schedule\": \"%s\"},\n  "
            @"\"support\": {\"active_slots\": %u, \"padded_slots\": 0, \"up_packet_block\": %u, \"down_packet_block\": %u},\n  "
            @"\"memory_bytes\": {\"compact_parameters\": %llu, \"compact_training_state\": %llu, \"dense_parameter_reference\": %llu, \"dense_activation_workspace\": %llu, \"route_metadata\": %llu},\n  "
            @"\"median_train_step_gpu_ms\": %.9f,\n  \"mean_train_step_gpu_ms\": %.9f,\n  \"std_train_step_gpu_ms\": %.9f%s,\n  \"max_abs_error\": "
            @"{\"z\": %.9g, \"h\": %.9g, \"y\": %.9g, \"gz\": %.9g, \"gx\": "
            @"%.9g, \"gup\": %.9g, \"gdown\": %.9g, \"gbup\": %.9g, "
            @"\"gbdown\": %.9g},\n  \"activation_cache\": \"%s\"\n}\n",
            c16 ? "compact16" : "compact32", a.input, a.hidden, a.output, a.n,
            a.degree, a.accum_steps, a.flat_adamw_pool ? "flat" : "segmented",
            schedule.c_str(), forward_schedule.c_str(), fdown_schedule,
            binput_schedule, bdown_schedule,
            E, a.packet_up_block, a.packet_down_block, compact_parameter_bytes, compact_training_state_bytes,
            dense_parameter_reference_bytes, dense_activation_workspace_bytes,
            route_metadata_bytes,
            s[s.size() / 2], mean_train_step_ms, std_train_step_ms, phase_json.c_str(),
            ez, eh, ey, egz, egx, egu, egd, egbu, egbd,
            a.cache_gelu_grad ? "gelu_grad" : "preactivation"];
    if(a.exact_support){
      // The historical uniform formula is not an allocation measurement for
      // irregular pools/routes. Dedicated memory passes use the device query.
      NSMutableDictionary *report=[[NSJSONSerialization JSONObjectWithData:[j dataUsingEncoding:NSUTF8StringEncoding] options:0 error:&er] mutableCopy];
      if(!report)return 1;
      report[@"support"]=@{@"up_edges":@(E),@"down_edges":@(ED),@"padded_slots":@0,@"exact_offsets":@YES};
      report[@"addressing"]=exact_offsets?@"exact_offsets":@"uniform_degree";
      report[@"memory_bytes"]=[NSNull null];
      report[@"memory_scope"]=@"not measured by CLI; use separate native-device allocation pass";
      NSData *encoded=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&er];
      if(!encoded)return 1;
      j=[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding];
    }
    if (![j writeToFile:[NSString stringWithUTF8String:a.path]
             atomically:YES
               encoding:NSUTF8StringEncoding
                  error:&er])
      return 1;
    std::printf("%s", j.UTF8String);
    return 0;
  }
}

#include "native_train_context_api.inc"
