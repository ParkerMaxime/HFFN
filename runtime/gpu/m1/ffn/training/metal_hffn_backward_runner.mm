#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hffn_route_codec.h"

// Exact HFFN backward, ReLU variant. Primary edges remain hidden-major
// [hidden, degree]. The two reverse routes contain slot ids only: they never
// duplicate values, gradients or optimizer state.
struct Args {
  uint32_t input = 128, hidden = 512, output = 128, n = 64, degree = 13,
           warmup = 30, repeats = 200;
  bool tiled_n = false;
  const char *path = nullptr;
};
static uint32_t number(const char *x) {
  return uint32_t(std::strtoul(x, nullptr, 10));
}
static bool opt(int &i, int argc, const char *argv[], const char *name,
                uint32_t &v) {
  if (std::strcmp(argv[i], name) || ++i >= argc)
    return false;
  v = number(argv[i]);
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
        opt(i, argc, argv, "--repeats", a.repeats))
      continue;
    if (!std::strcmp(argv[i], "--output") && ++i < argc) {
      a.path = argv[i];
      continue;
    }
    if (!std::strcmp(argv[i], "--tiled-n")) {
      a.tiled_n = true;
      continue;
    }
    return false;
  }
  return a.path && a.input && a.hidden && a.output && a.n &&
         a.degree <= a.input && a.degree <= a.output;
}

static std::string source(const Args &a) {
  const HffnRouteCodec codec(a.hidden, a.degree);
  const bool compact16 = codec.compact16;
  const std::string route_type = compact16 ? "ushort" : "uint";
  const std::string decode =
      compact16
          ? "inline uint decode_route(Route r){return "
            "(uint(r)>>EB)*D+(uint(r)&EM);}\n"
          : "inline uint decode_route(Route r){return (r>>EB)*D+(r&EM);}\n";
  return "#include <metal_stdlib>\nusing namespace metal;\nconstant uint I=" +
         std::to_string(a.input) + "u,F=" + std::to_string(a.hidden) +
         "u,O=" + std::to_string(a.output) + "u,N=" + std::to_string(a.n) +
         "u,D=" + std::to_string(a.degree) +
         "u,EB=" + std::to_string(codec.edge_bits) +
         "u,EM=" + std::to_string((1u << codec.edge_bits) - 1u) +
         "u,TILES=" + std::to_string((a.n + 31u) / 32u) +
         "u;\nusing Route=" + route_type + ";\n" + decode +
         "kernel void forward_up(device const float*x[[buffer(0)]],device "
         "const uint*up_index[[buffer(1)]],device const "
         "float*up[[buffer(2)]],device const float*bup[[buffer(3)]],device "
         "float*z[[buffer(4)]],device float*h[[buffer(5)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,sg=tid>>5u,r=tg*16u+sg;if(r>=F)return;for(uint "
         "n=lane;n<N;n+=32u){float v=bup[r];for(uint e=0u;e<D;++e){uint "
         "p=r*D+e;v+=up[p]*x[up_index[p]*N+n];}z[r*N+n]=v;h[r*N+n]=max(v,0.0f);"
         "}}\n"
         "kernel void down_weight(device const float*h[[buffer(0)]],device "
         "const float*gy[[buffer(1)]],device const "
         "uint*off[[buffer(2)]],device const Route*slot[[buffer(3)]],device "
         "float*gdown[[buffer(4)]],device float*gbdown[[buffer(5)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,sg=tid>>5u,o=tg*16u+sg;if(o>=O)return;float "
         "bs=0.0f;for(uint b=lane;b<N;b+=32u)bs+=gy[o*N+b];float "
         "bt=simd_sum(bs);if(lane==0u)gbdown[o]=bt;for(uint "
         "q=off[o];q<off[o+1u];++q){uint p=decode_route(slot[q]),r=p/D;float "
         "s=0.0f;for(uint b=lane;b<N;b+=32u)s+=h[r*N+b]*gy[o*N+b];float "
         "st=simd_sum(s);if(lane==0u)gdown[p]=st;}}\n"
         "kernel void hidden_delta(device const float*z[[buffer(0)]],device "
         "const float*gy[[buffer(1)]],device const "
         "uint*target[[buffer(2)]],device const float*down[[buffer(3)]],device "
         "float*gz[[buffer(4)]],device float*gbup[[buffer(5)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,sg=tid>>5u,r=tg*16u+sg;if(r>=F)return;float "
         "bs=0.0f;for(uint b=lane;b<N;b+=32u){float gh=0.0f;for(uint "
         "e=0u;e<D;++e){uint p=r*D+e;gh+=down[p]*gy[target[p]*N+b];}float "
         "v=z[r*N+b]>0.0f?gh:0.0f;gz[r*N+b]=v;bs+=v;}float "
         "bt=simd_sum(bs);if(lane==0u)gbup[r]=bt;}\n"
         "kernel void input_weight_x(device const float*x[[buffer(0)]],device "
         "const float*gz[[buffer(1)]],device const "
         "uint*off[[buffer(2)]],device const Route*slot[[buffer(3)]],device "
         "const float*up[[buffer(4)]],device float*gup[[buffer(5)]],device "
         "float*gx[[buffer(6)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,sg=tid>>5u,i=tg*16u+sg;if(i>=I)return;for(uint "
         "b=lane;b<N;b+=32u){float s=0.0f;for(uint "
         "q=off[i];q<off[i+1u];++q){uint "
         "p=decode_route(slot[q]);s+=up[p]*gz[(p/"
         "D)*N+b];}gx[i*N+b]=s;}for(uint q=off[i];q<off[i+1u];++q){uint "
         "p=decode_route(slot[q]),r=p/D;float s=0.0f;for(uint "
         "b=lane;b<N;b+=32u)s+=gz[r*N+b]*x[i*N+b];float "
         "st=simd_sum(s);if(lane==0u)gup[p]=st;}}\n"
         "kernel void down_tile(device const float*h[[buffer(0)]],device const "
         "float*gy[[buffer(1)]],device const uint*off[[buffer(2)]],device "
         "const Route*slot[[buffer(3)]],device float*pg[[buffer(4)]],device "
         "float*pb[[buffer(5)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,work=tg*16u+(tid>>5u),o=work/"
         "TILES,t=work%TILES,n=t*32u+lane;if(o>=O)return;float "
         "bs=n<N?gy[o*N+n]:0.0f,bt=simd_sum(bs);if(lane==0u)pb[o*TILES+t]=bt;"
         "for(uint q=off[o];q<off[o+1u];++q){uint "
         "p=decode_route(slot[q]),r=p/D;float "
         "s=n<N?h[r*N+n]*gy[o*N+n]:0.0f,st=simd_sum(s);if(lane==0u)pg[p*TILES+"
         "t]=st;}}\n"
         "kernel void reduce_edge(device const float*p[[buffer(0)]],device "
         "float*g[[buffer(1)]],uint "
         "id[[thread_position_in_grid]]){if(id>=F*D)return;float "
         "s=0.0f;for(uint t=0;t<TILES;++t)s+=p[id*TILES+t];g[id]=s;}\n"
         "kernel void reduce_output(device const float*p[[buffer(0)]],device "
         "float*g[[buffer(1)]],uint "
         "id[[thread_position_in_grid]]){if(id>=O)return;float s=0.0f;for(uint "
         "t=0;t<TILES;++t)s+=p[id*TILES+t];g[id]=s;}\n"
         "kernel void hidden_tile(device const float*z[[buffer(0)]],device "
         "const float*gy[[buffer(1)]],device const "
         "uint*target[[buffer(2)]],device const float*down[[buffer(3)]],device "
         "float*gz[[buffer(4)]],device float*pb[[buffer(5)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,work=tg*16u+(tid>>5u),r=work/"
         "TILES,t=work%TILES,n=t*32u+lane;if(r>=F)return;float "
         "v=0.0f;if(n<N){float gh=0.0f;for(uint e=0;e<D;++e){uint "
         "p=r*D+e;gh+=down[p]*gy[target[p]*N+n];}v=z[r*N+n]>0.0f?gh:0.0f;gz[r*"
         "N+n]=v;}float s=simd_sum(v);if(lane==0u)pb[r*TILES+t]=s;}\n"
         "kernel void reduce_hidden(device const float*p[[buffer(0)]],device "
         "float*g[[buffer(1)]],uint "
         "id[[thread_position_in_grid]]){if(id>=F)return;float s=0.0f;for(uint "
         "t=0;t<TILES;++t)s+=p[id*TILES+t];g[id]=s;}\n"
         "kernel void input_tile(device const float*x[[buffer(0)]],device "
         "const float*gz[[buffer(1)]],device const "
         "uint*off[[buffer(2)]],device const Route*slot[[buffer(3)]],device "
         "const float*up[[buffer(4)]],device float*pg[[buffer(5)]],device "
         "float*gx[[buffer(6)]],uint "
         "tid[[thread_position_in_threadgroup]],uint "
         "tg[[threadgroup_position_in_grid]]){uint "
         "lane=tid&31u,work=tg*16u+(tid>>5u),i=work/"
         "TILES,t=work%TILES,n=t*32u+lane;if(i>=I)return;if(n<N){float "
         "s=0.0f;for(uint q=off[i];q<off[i+1u];++q){uint "
         "p=decode_route(slot[q]);s+=up[p]*gz[(p/"
         "D)*N+n];}gx[i*N+n]=s;}for(uint q=off[i];q<off[i+1u];++q){uint "
         "p=decode_route(slot[q]),r=p/D;float "
         "v=n<N?gz[r*N+n]*x[i*N+n]:0.0f,s=simd_sum(v);if(lane==0u)pg[p*TILES+t]"
         "=s;}}\n";
}

static id<MTLBuffer> buffer(id<MTLDevice> d, const void *data, size_t bytes) {
  id<MTLBuffer> b = [d newBufferWithLength:bytes
                                   options:MTLResourceStorageModeShared];
  std::memcpy(b.contents, data, bytes);
  return b;
}
static void bind(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                 NSArray<id<MTLBuffer>> *b, MTLSize groups) {
  [e setComputePipelineState:p];
  for (NSUInteger i = 0; i < b.count; ++i)
    [e setBuffer:b[i] offset:0 atIndex:i];
  [e dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
}
static void forward_once(id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                         NSArray<id<MTLBuffer>> *b, MTLSize groups) {
  id<MTLCommandBuffer> cmd = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
  bind(e, p, b, groups);
  [e endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "Metal: %s\n",
                 [[cmd.error.localizedDescription description] UTF8String]);
    std::exit(1);
  }
}
static double run_one(id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                      NSArray<id<MTLBuffer>> *b, MTLSize groups,
                      uint32_t repeats) {
  id<MTLCommandBuffer> cmd = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
  for (uint32_t k = 0; k < repeats; ++k)
    bind(e, p, b, groups);
  [e endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "Metal: %s\n",
                 [[cmd.error.localizedDescription description] UTF8String]);
    std::exit(1);
  }
  return (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0 / double(repeats);
}
static double run(id<MTLCommandQueue> q, id<MTLComputePipelineState> a,
                  id<MTLComputePipelineState> b, id<MTLComputePipelineState> c,
                  NSArray<id<MTLBuffer>> *ab, NSArray<id<MTLBuffer>> *bb,
                  NSArray<id<MTLBuffer>> *cb, MTLSize ag, MTLSize bg,
                  MTLSize cg, uint32_t repeats) {
  const double down = run_one(q, a, ab, ag, repeats),
               hidden = run_one(q, b, bb, bg, repeats),
               input = run_one(q, c, cb, cg, repeats);
  std::fprintf(
      stderr, "BACKWARD_PHASE_MS down=%.8f hidden=%.8f input=%.8f total=%.8f\n",
      down, hidden, input, down + hidden + input);
  return down + hidden + input;
}
static double error(const std::vector<float> &ref, const float *got) {
  double e = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    e = std::max(e, std::abs(double(ref[i] - got[i])));
  return e;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    Args a;
    if (!parse(argc, argv, a)) {
      std::fprintf(stderr, "usage: --input I --hidden F --output-size O --n N "
                           "--degree D --output file\n");
      return 2;
    }
    const uint32_t edges = a.hidden * a.degree;
    const HffnRouteCodec codec(a.hidden, a.degree);
    const bool compact16 = codec.compact16;
    std::vector<float> x(a.input * a.n), gy(a.output * a.n), bup(a.hidden),
        up(edges), down(edges), z(a.hidden * a.n), h(a.hidden * a.n);
    std::vector<uint32_t> ui(edges), target(edges), downoff(a.output + 1, 0),
        upoff(a.input + 1, 0);
    for (uint32_t i = 0; i < a.input; ++i)
      for (uint32_t n = 0; n < a.n; ++n)
        x[i * a.n + n] = float(int((i * 17u + n * 13u) % 29u) - 14) * .02f;
    for (uint32_t o = 0; o < a.output; ++o)
      for (uint32_t n = 0; n < a.n; ++n)
        gy[o * a.n + n] = float(int((o * 11u + n * 7u) % 23u) - 11) * .03f;
    for (uint32_t r = 0; r < a.hidden; ++r) {
      bup[r] = float(int(r % 9u) - 4) * .01f;
      for (uint32_t e = 0; e < a.degree; ++e) {
        uint32_t p = r * a.degree + e;
        ui[p] = (r * 19u + e * 7u) % a.input;
        target[p] = (r * 17u + e * 5u) % a.output;
        up[p] = float(int((p * 3u) % 17u) - 8) * .015f;
        down[p] = float(int((p * 5u) % 19u) - 9) * .012f;
        ++downoff[target[p] + 1u];
        ++upoff[ui[p] + 1u];
      }
    }
    for (uint32_t i = 0; i < a.output; ++i)
      downoff[i + 1] += downoff[i];
    for (uint32_t i = 0; i < a.input; ++i)
      upoff[i + 1] += upoff[i];
    std::vector<uint16_t> downslot16(edges), upslot16(edges);
    std::vector<uint32_t> downslot32(edges), upslot32(edges), dc = downoff,
                                                              uc = upoff;
    for (uint32_t p = 0; p < edges; ++p) {
      const uint32_t r = p / a.degree, e = p % a.degree;
      const uint32_t encoded =
          compact16 ? uint32_t(codec.encode16(r, e)) : codec.encode32(r, e);
      const uint32_t di = dc[target[p]]++, ui_slot = uc[ui[p]]++;
      if (compact16) {
        downslot16[di] = uint16_t(encoded);
        upslot16[ui_slot] = uint16_t(encoded);
      } else {
        downslot32[di] = encoded;
        upslot32[ui_slot] = encoded;
      }
    }
    for (uint32_t r = 0; r < a.hidden; ++r)
      for (uint32_t n = 0; n < a.n; ++n) {
        float s = bup[r];
        for (uint32_t e = 0; e < a.degree; ++e) {
          uint32_t p = r * a.degree + e;
          s += up[p] * x[ui[p] * a.n + n];
        }
        z[r * a.n + n] = s;
        h[r * a.n + n] = std::max(s, 0.0f);
      }
    std::vector<float> rgz(a.hidden * a.n), rgx(a.input * a.n), rgup(edges),
        rgdown(edges), rgbup(a.hidden), rgbdown(a.output);
    for (uint32_t o = 0; o < a.output; ++o)
      for (uint32_t n = 0; n < a.n; ++n)
        rgbdown[o] += gy[o * a.n + n];
    for (uint32_t r = 0; r < a.hidden; ++r) {
      for (uint32_t e = 0; e < a.degree; ++e) {
        uint32_t p = r * a.degree + e;
        for (uint32_t n = 0; n < a.n; ++n)
          rgdown[p] += h[r * a.n + n] * gy[target[p] * a.n + n];
      }
      for (uint32_t n = 0; n < a.n; ++n) {
        float gh = 0;
        for (uint32_t e = 0; e < a.degree; ++e) {
          uint32_t p = r * a.degree + e;
          gh += down[p] * gy[target[p] * a.n + n];
        }
        float v = z[r * a.n + n] > 0 ? gh : 0;
        rgz[r * a.n + n] = v;
        rgbup[r] += v;
      }
      for (uint32_t e = 0; e < a.degree; ++e) {
        uint32_t p = r * a.degree + e;
        for (uint32_t n = 0; n < a.n; ++n) {
          rgup[p] += rgz[r * a.n + n] * x[ui[p] * a.n + n];
          rgx[ui[p] * a.n + n] += up[p] * rgz[r * a.n + n];
        }
      }
    }
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d)
      return 1;
    NSError *err = nil;
    id<MTLLibrary> lib = [d
        newLibraryWithSource:[NSString stringWithUTF8String:source(a).c_str()]
                     options:nil
                       error:&err];
    if (!lib) {
      std::fprintf(stderr, "MSL: %s\n",
                   [[err.localizedDescription description] UTF8String]);
      return 1;
    }
    auto pipe = ^(NSString *n) {
      NSError *e = nil;
      id<MTLComputePipelineState> p =
          [d newComputePipelineStateWithFunction:[lib newFunctionWithName:n]
                                           error:&e];
      if (!p) {
        std::fprintf(stderr, "pipeline: %s\n",
                     [[e.localizedDescription description] UTF8String]);
        std::exit(1);
      }
      return p;
    };
    id<MTLComputePipelineState> pd = pipe(@"down_weight"),
                                ph = pipe(@"hidden_delta"),
                                pi = pipe(@"input_weight_x"),
                                pdt = pipe(@"down_tile"),
                                pre = pipe(@"reduce_edge"),
                                pro = pipe(@"reduce_output"),
                                pht = pipe(@"hidden_tile"),
                                prh = pipe(@"reduce_hidden"),
                                pit = pipe(@"input_tile");
    const void *downslots = compact16
                                ? static_cast<const void *>(downslot16.data())
                                : static_cast<const void *>(downslot32.data());
    const void *upslots = compact16
                              ? static_cast<const void *>(upslot16.data())
                              : static_cast<const void *>(upslot32.data());
    const size_t route_bytes =
        size_t(edges) * (compact16 ? sizeof(uint16_t) : sizeof(uint32_t));
    id<MTLBuffer> xb = buffer(d, x.data(), x.size() * 4),
                  gyb = buffer(d, gy.data(), gy.size() * 4),
                  zb = buffer(d, z.data(), z.size() * 4),
                  hb = buffer(d, h.data(), h.size() * 4),
                  tb = buffer(d, target.data(), target.size() * 4),
                  db = buffer(d, down.data(), down.size() * 4),
                  ub = buffer(d, up.data(), up.size() * 4),
                  dob = buffer(d, downoff.data(), downoff.size() * 4),
                  dsb = buffer(d, downslots, route_bytes),
                  uib = buffer(d, upoff.data(), upoff.size() * 4),
                  usb = buffer(d, upslots, route_bytes);
    std::vector<float> zeroe(edges), zerof(a.hidden), zeroo(a.output),
        zeroi(a.input * a.n), zerogz(a.hidden * a.n);
    id<MTLBuffer> gdownb = buffer(d, zeroe.data(), zeroe.size() * 4),
                  gbdownb = buffer(d, zeroo.data(), zeroo.size() * 4),
                  gzb = buffer(d, zerogz.data(), zerogz.size() * 4),
                  gbupb = buffer(d, zerof.data(), zerof.size() * 4),
                  gupb = buffer(d, zeroe.data(), zeroe.size() * 4),
                  gxb = buffer(d, zeroi.data(), zeroi.size() * 4);
    const uint32_t tiles = (a.n + 31u) / 32u;
    std::vector<float> zerop_e(edges * tiles), zerop_o(a.output * tiles),
        zerop_h(a.hidden * tiles);
    id<MTLBuffer> pgdownb = buffer(d, zerop_e.data(), zerop_e.size() * 4),
                  pgbdownb = buffer(d, zerop_o.data(), zerop_o.size() * 4),
                  pgbupb = buffer(d, zerop_h.data(), zerop_h.size() * 4),
                  pgupb = buffer(d, zerop_e.data(), zerop_e.size() * 4);
    NSArray<id<MTLBuffer>> *ab = @[ hb, gyb, dob, dsb, gdownb, gbdownb ];
    NSArray<id<MTLBuffer>> *tab = @[ hb, gyb, dob, dsb, pgdownb, pgbdownb ];
    NSArray<id<MTLBuffer>> *reb = @[ pgdownb, gdownb ];
    NSArray<id<MTLBuffer>> *rob = @[ pgbdownb, gbdownb ];
    NSArray<id<MTLBuffer>> *thb = @[ zb, gyb, tb, db, gzb, pgbupb ];
    NSArray<id<MTLBuffer>> *rhb = @[ pgbupb, gbupb ];
    NSArray<id<MTLBuffer>> *tib = @[ xb, gzb, uib, usb, ub, pgupb, gxb ];
    NSArray<id<MTLBuffer>> *rib = @[ pgupb, gupb ];
    NSArray<id<MTLBuffer>> *bb = @[ zb, gyb, tb, db, gzb, gbupb ];
    NSArray<id<MTLBuffer>> *cb = @[ xb, gzb, uib, usb, ub, gupb, gxb ];
    MTLSize dg = MTLSizeMake((a.output + 15u) / 16u, 1, 1),
            hg = MTLSizeMake((a.hidden + 15u) / 16u, 1, 1),
            ig = MTLSizeMake((a.input + 15u) / 16u, 1, 1);
    id<MTLCommandQueue> q = [d newCommandQueue];
    MTLSize tdg = MTLSizeMake((a.output * tiles + 15u) / 16u, 1, 1),
            thg = MTLSizeMake((a.hidden * tiles + 15u) / 16u, 1, 1),
            tig = MTLSizeMake((a.input * tiles + 15u) / 16u, 1, 1),
            reg = MTLSizeMake((edges + 511u) / 512u, 1, 1),
            rog = MTLSizeMake((a.output + 511u) / 512u, 1, 1),
            rhg = MTLSizeMake((a.hidden + 511u) / 512u, 1, 1);
    auto run_tiled_down = [&](uint32_t repeats) {
      const double down = run_one(q, pdt, tab, tdg, repeats) +
                          run_one(q, pre, reb, reg, repeats) +
                          run_one(q, pro, rob, rog, repeats);
      const double hidden = run_one(q, pht, thb, thg, repeats) +
                            run_one(q, prh, rhb, rhg, repeats);
      const double input = run_one(q, pit, tib, tig, repeats) +
                           run_one(q, pre, rib, reg, repeats);
      return down + hidden + input;
    };
    if (a.tiled_n)
      run_tiled_down(a.warmup);
    else
      run(q, pd, ph, pi, ab, bb, cb, dg, hg, ig, a.warmup);
    std::vector<double> s;
    for (int k = 0; k < 7; ++k)
      s.push_back(a.tiled_n
                      ? run_tiled_down(a.repeats)
                      : run(q, pd, ph, pi, ab, bb, cb, dg, hg, ig, a.repeats));
    std::sort(s.begin(), s.end());
    double egz = error(rgz, (float *)gzb.contents),
           egx = error(rgx, (float *)gxb.contents),
           egup = error(rgup, (float *)gupb.contents),
           egdown = error(rgdown, (float *)gdownb.contents),
           ebup = error(rgbup, (float *)gbupb.contents),
           ebdown = error(rgbdown, (float *)gbdownb.contents);
    NSString *json = [NSString
        stringWithFormat:
            @"{\n  \"kind\": \"hffn_metal_backward_dual_route_relu\",\n  "
            @"\"route\": \"%s\",\n  \"dimensions\": {\"input\": %u, "
            @"\"hidden\": %u, \"output\": %u, \"N\": %u, \"degree\": %u},\n  "
            @"\"schedule\": {\"threads_per_threadgroup\": 512, "
            @"\"simdgroups_per_threadgroup\": 16, \"batch_lanes_per_owner\": "
            @"32, \"persistent_N\": %@},\n  \"median_backward_gpu_ms\": "
            @"%.9f,\n  \"max_abs_error\": {\"gz\": %.9g, \"gx\": %.9g, "
            @"\"gup\": %.9g, \"gdown\": %.9g, \"gbup\": %.9g, \"gbdown\": "
            @"%.9g}\n}\n",
            compact16 ? "compact16" : "compact32", a.input, a.hidden, a.output,
            a.n, a.degree, a.tiled_n ? @"false" : @"true", s[3], egz, egx, egup,
            egdown, ebup, ebdown];
    if (![json writeToFile:[NSString stringWithUTF8String:a.path]
                atomically:YES
                  encoding:NSUTF8StringEncoding
                     error:&err]) {
      std::fprintf(stderr, "write: %s\n",
                   [[err.localizedDescription description] UTF8String]);
      return 1;
    }
    std::printf("%s", json.UTF8String);
  }
}
