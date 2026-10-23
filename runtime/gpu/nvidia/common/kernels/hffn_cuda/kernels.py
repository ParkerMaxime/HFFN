"""CUDA source of the HFFN forward kernels: one compact fixed-support FFN, N tokens per call.

Port of the one-bank path of the Metal generators of the M1 GPU runtime (`gpu/m1/llm/inference/native_transformer/hffn/kernels.py`), with d_model, d_ff and N read at run
time. A block has 512 threads: 16 warps of 32 lanes, as the Metal threadgroup of 16 SIMD groups.

up    a[r*N + t] = gelu(b_up[r] + <w_up[r], x[t]>), activations hidden-major [d_ff, N].
      `up_coop` (N <= 4): a warp owns a hidden neuron and reduces its edge list over its 32
      lanes, for each token; the input is token-major [N, d_model].
      `up_lane` (N < 64): a warp owns a hidden neuron, lane l owns tokens l, l + 32.
      `up_tile` (N >= 64): 128-token tiles, lane l owns tokens l, l + 32, l + 64, l + 96 of its
      tile, four accumulators.
      `up_lane` and `up_tile` read the input feature-major [d_model, N] and add in the same
      order: the bias, then the edges in slot order, four at a time (loads first).
down  y[t*d_model + o] = <v[o], a[.][t]> + b_down[o], output owner, edges in output order.
      `down_coop` (N <= 4): a warp owns an output and reduces its edge list over its lanes.
      `down_tiled`: one thread per (output, token), in tiles of 64 tokens.
      `down_blocked`: two tokens per thread, lane l owns tokens l and l + 32 of a 64-token tile.
      `down_tiled` and `down_blocked` add the edges in slot order, then the bias.

Every hidden value and every output is written by one thread: no atomic, no scatter. The two
cooperative kernels add 32 partial sums with a warp shuffle tree; the other kernels add
sequentially.

Not ported from the reference: the erf approximation of the Metal prelude (CUDA has `erff`),
the `volatile` stores of the gated bank sum (one bank here), and the 64-byte row strides of the Metal runtime.
"""
UNROLL = 4          # edges per loop iteration
BLOCK = 512         # threads per block
UP_TILE = 128       # tokens per up tile
DOWN_TILE = 64      # tokens per down tile


def _each(template: str, count: int = UNROLL, separator: str = " ") -> str:
    return separator.join(template.format(u=u) for u in range(count))


def _activation_store(value: str, token: str, store_z: bool) -> str:
    """Generate the one activation store, optionally retaining its preactivation."""
    if store_z:
        return (f"{{ float zv = {value}; zs[r*n + {token}] = zv; "
                f"a[r*n + {token}] = gelu(zv); }}")
    return f"a[r*n + {token}] = gelu({value});"


def _up_tile_body(*, store_z: bool = False) -> str:
    tokens = range(UP_TILE // 32)
    edges = "".join(f"if (t{k} < n) {{ " + _each(f"s{k} += w{{u}}*x[i{{u}}*n + t{k}];") + " }\n      " for k in tokens)
    tail = " ".join(f"if (t{k} < n) s{k} += ww*x[i*n + t{k}];" for k in tokens)
    store = " ".join(f"if (t{k} < n) " + _activation_store(f"s{k}", f"t{k}", store_z) for k in tokens)
    return f"""
    unsigned t0 = blockIdx.y*{UP_TILE}u + lane, t1 = t0 + 32u, t2 = t0 + 64u, t3 = t0 + 96u;
    float s0 = b[r], s1 = s0, s2 = s0, s3 = s0;
    unsigned e = begin;
    for (; e + {UNROLL}u <= end; e += {UNROLL}u) {{
      {_each("unsigned i{u} = (unsigned)idx[e + {u}u]; float w{u} = w[e + {u}u];")}
      {edges}}}
    for (; e < end; ++e) {{
      unsigned i = (unsigned)idx[e]; float ww = w[e];
      {tail}
    }}
    {store}"""


def _down_blocked_loop(second: bool) -> str:
    loads = _each("float v{u} = v[q + {u}u]; unsigned h{u} = (unsigned)hid[q + {u}u];")
    first = _each("s0 += v{u}*a[h{u}*n + tb];")
    other = _each("s1 += v{u}*a[h{u}*n + tb + 32u];") if second else ""
    tail = "s0 += v[q]*a[(unsigned)hid[q]*n + tb];" + (" s1 += v[q]*a[(unsigned)hid[q]*n + tb + 32u];" if second else "")
    return f"""
      for (; q + {UNROLL}u <= qe; q += {UNROLL}u) {{
        {loads}
        {first}
        {other}
      }}
      for (; q < qe; ++q) {{ {tail} }}"""


def forward_source(*, store_z: bool = False) -> str:
    """Canonical inference forward source, with optional saved preactivations.

    Both variants share every support, reduction, arithmetic and dispatch
    implementation. The training entry calls this generator; it rewrites no CUDA.
    """
    suffix = "_z" if store_z else ""
    zarg = "float* zs, " if store_z else ""
    return f"""
__device__ __forceinline__ float gelu(float z) {{
  return 0.5f*z*(1.0f + erff(z*0.70710678118654752f));
}}

__device__ __forceinline__ float warp_sum(float value) {{
  value += __shfl_xor_sync(0xffffffffu, value, 16);
  value += __shfl_xor_sync(0xffffffffu, value, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}}

extern "C" {{

// x token-major [n, d]; grid: ceil(H/16) blocks.
__global__ void up_coop{suffix}(const float* x, const int* off, const short* idx, const float* w,
                        const float* b, float* a, {zarg}unsigned d, unsigned H, unsigned n) {{
  unsigned lane = threadIdx.x & 31u, r = blockIdx.x*16u + (threadIdx.x >> 5);
  if (r >= H) return;
  unsigned begin = (unsigned)off[r], end = (unsigned)off[r + 1u];
  for (unsigned t = 0u; t < n; ++t) {{
    float partial = 0.0f;
    for (unsigned e = begin + lane; e < end; e += 32u) partial += w[e]*x[t*d + (unsigned)idx[e]];
    float sum = warp_sum(partial);
    if (lane == 0u) {_activation_store("b[r] + sum", "t", store_z)}
  }}
}}

// x feature-major [d, n]; grid: ceil(H/16) blocks.
__global__ void up_lane{suffix}(const float* x, const int* off, const short* idx, const float* w,
                        const float* b, float* a, {zarg}unsigned d, unsigned H, unsigned n) {{
  unsigned lane = threadIdx.x & 31u, r = blockIdx.x*16u + (threadIdx.x >> 5);
  if (r >= H) return;
  unsigned begin = (unsigned)off[r], end = (unsigned)off[r + 1u];
  for (unsigned t = lane; t < n; t += 32u) {{
    float s = b[r];
    unsigned e = begin;
    for (; e + {UNROLL}u <= end; e += {UNROLL}u) {{
      {_each("unsigned i{u} = (unsigned)idx[e + {u}u]; float w{u} = w[e + {u}u];")}
      {_each("float x{u} = x[i{u}*n + t];")}
      {_each("s += w{u}*x{u};")}
    }}
    for (; e < end; ++e) s += w[e]*x[(unsigned)idx[e]*n + t];
    {_activation_store("s", "t", store_z)}
  }}
}}

// x feature-major [d, n]; grid: ceil(H/16) x ceil(n/{UP_TILE}) blocks.
__global__ void up_tile{suffix}(const float* x, const int* off, const short* idx, const float* w,
                        const float* b, float* a, {zarg}unsigned d, unsigned H, unsigned n) {{
  unsigned lane = threadIdx.x & 31u, r = blockIdx.x*16u + (threadIdx.x >> 5);
  if (r >= H) return;
  unsigned begin = (unsigned)off[r], end = (unsigned)off[r + 1u];
  {{{_up_tile_body(store_z=store_z)}
  }}
}}

// grid: ceil(d/16) blocks.
__global__ void down_coop(const float* a, const int* off, const short* hid, const float* v,
                          const float* bias, float* y, unsigned d, unsigned n) {{
  unsigned lane = threadIdx.x & 31u, o = blockIdx.x*16u + (threadIdx.x >> 5);
  if (o >= d) return;
  unsigned begin = (unsigned)off[o], end = (unsigned)off[o + 1u];
  for (unsigned t = 0u; t < n; ++t) {{
    float partial = 0.0f;
    for (unsigned q = begin + lane; q < end; q += 32u) partial += v[q]*a[(unsigned)hid[q]*n + t];
    float sum = warp_sum(partial);
    if (lane == 0u) y[t*d + o] = sum + bias[o];
  }}
}}

// grid: ceil(n*d/{BLOCK}) blocks.
__global__ void down_tiled(const float* a, const int* off, const short* hid, const float* v,
                           const float* bias, float* y, unsigned d, unsigned n) {{
  unsigned gid = blockIdx.x*{BLOCK}u + threadIdx.x;
  if (gid >= n*d) return;
  unsigned span = d*{DOWN_TILE}u, tile = gid/span, rest = gid - tile*span;
  unsigned left = n - tile*{DOWN_TILE}u, width = left < {DOWN_TILE}u ? left : {DOWN_TILE}u;
  unsigned o = rest/width, t = tile*{DOWN_TILE}u + rest%width;
  float s = 0.0f;
  unsigned q = (unsigned)off[o], qe = (unsigned)off[o + 1u];
  for (; q + {UNROLL}u <= qe; q += {UNROLL}u) {{
    {_each("float v{u} = v[q + {u}u]; float a{u} = a[(unsigned)hid[q + {u}u]*n + t];")}
    {_each("s += v{u}*a{u};")}
  }}
  for (; q < qe; ++q) s += v[q]*a[(unsigned)hid[q]*n + t];
  y[t*d + o] = s + bias[o];
}}

// grid: ceil(ceil(n/{DOWN_TILE})*d*32/{BLOCK}) blocks.
__global__ void down_blocked(const float* a, const int* off, const short* hid, const float* v,
                             const float* bias, float* y, unsigned d, unsigned n) {{
  unsigned gid = blockIdx.x*{BLOCK}u + threadIdx.x, span = d*32u, tile = gid/span, rest = gid - tile*span;
  unsigned o = rest/32u, tb = tile*{DOWN_TILE}u + rest%32u;
  if (tb >= n) return;
  float s0 = 0.0f, s1 = 0.0f;
  unsigned q = (unsigned)off[o], qe = (unsigned)off[o + 1u];
  if (tb + 32u < n) {{{_down_blocked_loop(True)}
    y[tb*d + o] = s0 + bias[o];
    y[(tb + 32u)*d + o] = s1 + bias[o];
  }} else {{{_down_blocked_loop(False)}
    y[tb*d + o] = s0 + bias[o];
  }}
}}

}}
"""


SOURCE = forward_source()
