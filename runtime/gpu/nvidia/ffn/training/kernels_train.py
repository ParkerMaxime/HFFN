"""CUDA source and launch glue of the HFFN training step on the shared inference kernels (`hffn_cuda`), N tokens per call.

Forward. The common inference generator owns both ordinary and saved-preactivation kernels.
Training requests forward_source(store_z=True), which adds the dense z pointer and store to the same up bodies.
There is no training-owned forward or CUDA text rewrite. Every support traversal, bias, reduction, activation
and down body is generated once at the shared inference level. Saved-z names/signatures preserve the native
up_<mode>_z ABI consumed below; ordinary inference keeps its original source and dispatch.

Layouts. Hidden-major activations `[d_ff, N]` (`a`, `z`, `gz`: element `r*n + t`), feature-major inputs `[d_model, N]`
(`x_t`, `gy_t`: element `i*n + t`), token-major outputs `[N, d_model]` (`y`, `gx`). Compact arrays exactly as the
runtime holds them: up slots hidden-major (`up_offsets`, `up_idx`, `up_val`), down edges output-major
(`out_offsets`, `out_hidden`, `out_val`). Two transposed supports, built once by `train_runtime`, are positions into
those arrays (no second copy of any value): the up slots input-major (`in_offsets`, `in_pos`, `in_hidden`) and the
down edges hidden-major (`hid_offsets`, `hid_pos`, `hid_output`). Every kernel walks the exact range
`[offsets[k], offsets[k + 1])`: there is no padded slot in this layout, homogeneous or irregular, so none can be
read or written.

Dispatch per N.
  forward   as `HFFNRuntime.forward`: up `coop` N <= 4, `lane` N < 64, `tile` N >= 64; down `coop` N <= 4,
            `blocked` N >= 64 and (d_ff <= 4096 or N <= 128), `tiled` otherwise (the runtime's `up_mode`,
            `down_mode` and `force`). The input is also copied feature-major for every N (the backward reads it).
  backward  one kernel per quantity for every N, the grid only grows with N; `tiles = ceil(N / GRAD_TILE)`,
            GRAD_TILE = 256 tokens (fixed, not tuned per shape or N):
            grad_down    ceil(d/16) x tiles blocks of 16 warps, warp = output o on token tile k, lane owns tokens
                         256k + lane + 32j, j < 8, held in registers across the edges of o
            grad_hidden  ceil(H/16) x ceil(N/128) blocks, warp = neuron r, lane owns tokens 128c + lane + 32k, k < 4
            grad_up      ceil(H/16) x tiles blocks, warp = neuron r on token tile k, lane tokens as grad_down
            reduce_tiles only when tiles > 1, after grad_up: ceil(S/512) blocks, thread = one column of the partial
                         workspace (S = nnz_down + d + nnz_up + H columns per layer)
            grad_x       ceil(ceil(N/GX_TILE)*32*d/512) blocks, GX_TILE = 128 tokens (fixed), warp =
                         input i on token tile c, lane owns tokens 128c + lane + 32k, k < 4, one accumulator each
  adamw     ceil(count/512) blocks per array, thread = one compact entry.
  For tiles == 1 (N <= 256) grad_down and grad_up write the gradients themselves and reduce_tiles is not launched.
  For tiles > 1 they write the partial of tile k into row k of the workspace `[tiles, S]` (columns: g_down in
  output-major edge order, gb_down, g_up in hidden-major slot order, gb_up), and reduce_tiles writes the gradients.
  The gathered rows (`h` for grad_down, `x_t` for grad_up) are read from global memory (left to L2), not staged.

Reduction order (float32; no atomic; every result written by one thread or by lane 0 of one warp).
  gb_down[o]  = sum_t gy[t, o]               per tile k: lane l adds its tokens 256k + l + 32j (t < N) in
                                             ascending j from 0.0f, then the xor-shuffle tree 16, 8, 4, 2, 1
                                             (`warp_sum`) gives the partial P_k; then the tiles in ascending k:
                                             ((P_0 + P_1) + P_2) + ... (`reduce_tiles`; one tile: P_0)
  g_down[q]   = sum_t gy[t, o] * h[r, t]     same, one warp reduction per edge q and tile, edges in output-major
                                             slot order
  dh[r, t]    = sum_q v[q] * gy[t, o_q]      one thread: the down edges of r in hidden-major order (outputs
                                             ascending), sequential; gz = dh * gelu'(z) in the same thread,
                                             dh is not stored
  gb_up[r]    = sum_t gz[r, t]               as gb_down
  g_up[e]     = sum_t gz[r, t] * x[t, i_e]   as g_down, one warp reduction per up slot e and tile in slot order
  gx[t, i]    = sum_e w[e] * gz[r_e, t]      one thread: the up slots of input i in hidden order (ascending),
                                             sequential
  AdamW       elementwise, no reduction.
The token reductions depend on N only (which tokens a lane owns, how many tiles), not on the grid or on the launch
order of the blocks: same bits run to run on one GPU and build. They differ from those of a one-tile order in which each lane
adds all N tokens before one shuffle tree. Compiled by NVRTC with its defaults (as the
inference module): `a += b*c` may be contracted to an FMA.

gelu'(z) is exact: 0.5 (1 + erf(z / sqrt 2)) + z exp(-z^2 / 2) / sqrt(2 pi), with `erff` and `expf`, the form of
PyTorch's GELU backward (approximate="none").

AdamW (decoupled weight decay, bias correction, compact moments of the length of their array), in the order of
PyTorch's fused AdamW: p -= lr*wd*p; m = b1*m + (1 - b1)*g; v = b2*v + (1 - b2)*g*g;
p -= (lr / (1 - b1^k)) * m / (sqrt(v) / sqrt(1 - b2^k) + eps). The step-dependent scalars are computed in float64
on the host and passed as float32.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _share_kernels() -> None:
    root = next(p for p in Path(__file__).resolve().parents if (p / "runtime" / "common" / "pythia" / "hffn_pythia").is_dir())
    shared = root / "runtime" / "gpu" / "nvidia" / "common" / "kernels"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))


_share_kernels()

from hffn_cuda.kernels import BLOCK, DOWN_TILE, UP_TILE, forward_source  # noqa: E402
from hffn_cuda.runtime import UP_MODES  # noqa: E402

TOKENS_PER_LANE = UP_TILE // 32   # grad_hidden: tokens of one lane in a 128-token tile
GRAD_TILE = 256                   # grad_down / grad_up: tokens per tile (fixed; N <= 256 is one tile, no reduce_tiles)
GRAD_TOKENS_PER_LANE = GRAD_TILE // 32
GX_TILE = 128                     # grad_x only: tokens per tile (fixed; its own constant, not a shared tile)
assert GX_TILE % 32 == 0
GX_TOKENS_PER_LANE = GX_TILE // 32
assert UP_TILE % 32 == 0 and GRAD_TILE % 32 == 0 and BLOCK == 512
assert GX_TOKENS_PER_LANE == 4    # four tokens per lane, one accumulator each (fixed)


def grad_tiles(n: int) -> int:
    """Token tiles of grad_down / grad_up for N tokens; 1 means no partial workspace and no reduce_tiles launch."""
    return (n + GRAD_TILE - 1) // GRAD_TILE




_TRAIN = r"""
__device__ __forceinline__ float gelu_grad(float z) {
  return 0.5f*(1.0f + erff(z*0.70710678118654752f)) + z*0.39894228040143268f*expf(-0.5f*z*z);
}

extern "C" {

// Down gradients, output owners, token-tiled. gy_t [d, n], a = h [H, n]. grid: ceil(d/16) x ceil(n/GRAD_TILE); warp w
// of block (b, k) owns o = 16b + w on tile k; lane l holds gy[t, o] for t = GRAD_TILE*k + l + 32j in registers across
// the edges of o. Writes the tile partials at gb[k*stride + o] and g[k*stride + q] (one tile: the gradients).
__global__ void grad_down(const float* gy_t, const float* a, const int* off, const short* hid,
                          float* g, float* gb, unsigned d, unsigned n, unsigned stride) {
  unsigned lane = threadIdx.x & 31u, o = blockIdx.x*16u + (threadIdx.x >> 5);
  if (o >= d) return;
  unsigned t0 = blockIdx.y*GRAD_TILE + lane, part = blockIdx.y*stride;
  const float* row = gy_t + o*n;
  float own[GRAD_TOKENS_PER_LANE];
  float partial = 0.0f;
  #pragma unroll
  for (unsigned j = 0u; j < GRAD_TOKENS_PER_LANE; ++j) {
    unsigned t = t0 + 32u*j;
    own[j] = t < n ? row[t] : 0.0f;
    if (t < n) partial += own[j];
  }
  float total = warp_sum(partial);
  if (lane == 0u) gb[part + o] = total;
  unsigned end = (unsigned)off[o + 1u];
  for (unsigned q = (unsigned)off[o]; q < end; ++q) {
    const float* act = a + (unsigned)hid[q]*n;
    partial = 0.0f;
    #pragma unroll
    for (unsigned j = 0u; j < GRAD_TOKENS_PER_LANE; ++j) {
      unsigned t = t0 + 32u*j;
      if (t < n) partial += own[j]*act[t];
    }
    total = warp_sum(partial);
    if (lane == 0u) g[part + q] = total;
  }
}

// gz = (sum of the down edges of r of v * gy) * gelu'(z), hidden owners. v is the output-major `out_val`, read
// through the hidden-major positions `hpos`; `hout` is the output of each. grid: ceil(H/16) x ceil(n/TOKEN_TILE).
__global__ void grad_hidden(const float* gy_t, const float* z, const int* hoff, const int* hpos, const short* hout,
                            const float* v, float* gz, unsigned H, unsigned n) {
  unsigned lane = threadIdx.x & 31u, r = blockIdx.x*16u + (threadIdx.x >> 5);
  if (r >= H) return;
  unsigned t0 = blockIdx.y*TOKEN_TILE + lane;
  float s[TOKENS_PER_LANE];
  #pragma unroll
  for (unsigned k = 0u; k < TOKENS_PER_LANE; ++k) s[k] = 0.0f;
  unsigned end = (unsigned)hoff[r + 1u];
  for (unsigned j = (unsigned)hoff[r]; j < end; ++j) {
    float w = v[(unsigned)hpos[j]];
    const float* row = gy_t + (unsigned)hout[j]*n;
    #pragma unroll
    for (unsigned k = 0u; k < TOKENS_PER_LANE; ++k) {
      unsigned t = t0 + 32u*k;
      if (t < n) s[k] += w*row[t];
    }
  }
  #pragma unroll
  for (unsigned k = 0u; k < TOKENS_PER_LANE; ++k) {
    unsigned t = t0 + 32u*k;
    if (t < n) gz[r*n + t] = s[k]*gelu_grad(z[r*n + t]);
  }
}

// Up gradients, hidden owners, token-tiled. x_t [d, n], gz [H, n]. grid: ceil(H/16) x ceil(n/GRAD_TILE); warp w of
// block (b, k) owns r = 16b + w on tile k; lane l holds gz[r, t] for t = GRAD_TILE*k + l + 32j in registers across
// the up slots of r. Writes the tile partials at gb[k*stride + r] and g[k*stride + e] (one tile: the gradients).
__global__ void grad_up(const float* x_t, const float* gz, const int* off, const short* idx,
                        float* g, float* gb, unsigned H, unsigned n, unsigned stride) {
  unsigned lane = threadIdx.x & 31u, r = blockIdx.x*16u + (threadIdx.x >> 5);
  if (r >= H) return;
  unsigned t0 = blockIdx.y*GRAD_TILE + lane, part = blockIdx.y*stride;
  const float* row = gz + r*n;
  float own[GRAD_TOKENS_PER_LANE];
  float partial = 0.0f;
  #pragma unroll
  for (unsigned j = 0u; j < GRAD_TOKENS_PER_LANE; ++j) {
    unsigned t = t0 + 32u*j;
    own[j] = t < n ? row[t] : 0.0f;
    if (t < n) partial += own[j];
  }
  float total = warp_sum(partial);
  if (lane == 0u) gb[part + r] = total;
  unsigned end = (unsigned)off[r + 1u];
  for (unsigned e = (unsigned)off[r]; e < end; ++e) {
    const float* input = x_t + (unsigned)idx[e]*n;
    partial = 0.0f;
    #pragma unroll
    for (unsigned j = 0u; j < GRAD_TOKENS_PER_LANE; ++j) {
      unsigned t = t0 + 32u*j;
      if (t < n) partial += own[j]*input[t];
    }
    total = warp_sum(partial);
    if (lane == 0u) g[part + e] = total;
  }
}

// Fixed-order sum of the tile partials of grad_down and grad_up, launched once per backward when tiles > 1.
// ws [tiles, stride]; columns [0, c1) g_down, [c1, c2) gb_down, [c2, c3) g_up, [c3, stride) gb_up. One thread per
// column: ((ws[0][j] + ws[1][j]) + ws[2][j]) + ... in ascending tile order. grid: ceil(stride/TRAIN_BLOCK) blocks.
__global__ void reduce_tiles(const float* ws, unsigned stride, unsigned tiles, unsigned c1, unsigned c2, unsigned c3,
                             float* g_down, float* gb_down, float* g_up, float* gb_up) {
  unsigned j = blockIdx.x*TRAIN_BLOCK + threadIdx.x;
  if (j >= stride) return;
  float s = ws[j];
  for (unsigned k = 1u; k < tiles; ++k) s += ws[k*stride + j];
  if (j < c1) g_down[j] = s;
  else if (j < c2) gb_down[j - c1] = s;
  else if (j < c3) g_up[j - c2] = s;
  else gb_up[j - c3] = s;
}

// gx[t, i] = sum of the up slots of input i (hidden ascending) of w * gz[r, t], input owners. w is the hidden-major
// `up_val`, read through the input-major positions `ipos`. Warp w = c*d + i owns input i on the GX_TILE-token tile c;
// lane l holds one accumulator per token t = GX_TILE*c + l + 32k, k < GX_TOKENS_PER_LANE, each summed over the same
// slots in the same order; tokens t >= n are skipped. grid: ceil(ceil(n/GX_TILE)*32*d/BLOCK) blocks.
__global__ void grad_x(const float* gz, const int* ioff, const int* ipos, const short* ihid, const float* w,
                       float* gx, unsigned d, unsigned n) {
  unsigned gid = blockIdx.x*TRAIN_BLOCK + threadIdx.x, warp = gid >> 5u, tile = warp/d, i = warp - tile*d;
  unsigned t0 = tile*GX_TILE + (gid & 31u);
  if (t0 >= n) return;
  float s[GX_TOKENS_PER_LANE];
  #pragma unroll
  for (unsigned k = 0u; k < GX_TOKENS_PER_LANE; ++k) s[k] = 0.0f;
  unsigned end = (unsigned)ioff[i + 1u];
  for (unsigned j = (unsigned)ioff[i]; j < end; ++j) {
    float wj = w[(unsigned)ipos[j]];
    const float* row = gz + (unsigned)ihid[j]*n;
    #pragma unroll
    for (unsigned k = 0u; k < GX_TOKENS_PER_LANE; ++k) {
      unsigned t = t0 + 32u*k;
      if (t < n) s[k] += wj*row[t];
    }
  }
  #pragma unroll
  for (unsigned k = 0u; k < GX_TOKENS_PER_LANE; ++k) {
    unsigned t = t0 + 32u*k;
    if (t < n) gx[t*d + i] = s[k];
  }
}

// Compact AdamW on one array, one thread per entry. grid: ceil(count/BLOCK) blocks.
__global__ void adamw(float* p, const float* g, float* m, float* v, unsigned count, float lr_decay,
                      float beta1, float one_minus_beta1, float beta2, float one_minus_beta2,
                      float step_size, float bias2_sqrt, float eps) {
  unsigned i = blockIdx.x*TRAIN_BLOCK + threadIdx.x;
  if (i >= count) return;
  float gi = g[i], pi = p[i];
  pi -= lr_decay*pi;
  float mi = beta1*m[i] + one_minus_beta1*gi;
  float vi = beta2*v[i] + one_minus_beta2*gi*gi;
  m[i] = mi;
  v[i] = vi;
  p[i] = pi - step_size*mi/(sqrtf(vi)/bias2_sqrt + eps);
}

}
"""

TRAIN_SOURCE = (forward_source(store_z=True)
                + f"\n#define TRAIN_BLOCK {BLOCK}u\n#define TOKEN_TILE {UP_TILE}u\n#define TOKENS_PER_LANE {TOKENS_PER_LANE}u\n"
                + f"#define INPUT_TILE {DOWN_TILE}u\n#define GRAD_TILE {GRAD_TILE}u\n"
                + f"#define GRAD_TOKENS_PER_LANE {GRAD_TOKENS_PER_LANE}u\n"
                + f"#define GX_TILE {GX_TILE}u\n#define GX_TOKENS_PER_LANE {GX_TOKENS_PER_LANE}u\n" + _TRAIN)


def _blocks(count: int, size: int) -> int:
    return (count + size - 1) // size


class TrainKernels:
    """The compiled training module (z-storing up kernels, backward, AdamW) and the grid of each launch.

    Arguments are passed as the inference runtime passes them: CuPy views of persistent device buffers, or
    `numpy.uint64` device pointers, and `numpy.uint32` / `numpy.float32` scalars. Every launch is on the default stream.
    """

    def __init__(self) -> None:
        import cupy
        module = cupy.RawModule(code=TRAIN_SOURCE)
        self.up_z = {mode: module.get_function(f"up_{mode}_z") for mode in UP_MODES}   # called by `HFFNRuntime._launch` as the inference up kernels, plus the `z` pointer
        self._grad_down = module.get_function("grad_down")
        self._grad_hidden = module.get_function("grad_hidden")
        self._grad_up = module.get_function("grad_up")
        self._reduce_tiles = module.get_function("reduce_tiles")
        self._grad_x = module.get_function("grad_x")
        self._adamw = module.get_function("adamw")

    def grad_down(self, n: int, d_model: int, args: tuple) -> None:
        """args: (gy_t, a, out_offsets, out_hidden, g, gb, d, n, stride); g, gb the gradients (one tile) or the
        g_down / gb_down columns of the partial workspace (stride = its row length)."""
        self._grad_down((_blocks(d_model, 16), grad_tiles(n)), (BLOCK,), args)

    def grad_hidden(self, n: int, d_ff: int, args: tuple) -> None:
        """args: (gy_t, z, hid_offsets, hid_pos, hid_output, out_val, gz, H, n)."""
        self._grad_hidden((_blocks(d_ff, 16), _blocks(n, UP_TILE)), (BLOCK,), args)

    def grad_up(self, n: int, d_ff: int, args: tuple) -> None:
        """args: (x_t, gz, up_offsets, up_idx, g, gb, H, n, stride); g, gb as grad_down, for g_up / gb_up."""
        self._grad_up((_blocks(d_ff, 16), grad_tiles(n)), (BLOCK,), args)

    def reduce_tiles(self, stride: int, args: tuple) -> None:
        """args: (ws, stride, tiles, c1, c2, c3, g_down, gb_down, g_up, gb_up). Only when tiles > 1."""
        self._reduce_tiles((_blocks(stride, BLOCK),), (BLOCK,), args)

    def grad_x(self, n: int, d_model: int, args: tuple) -> None:
        """args: (gz, in_offsets, in_pos, in_hidden, up_val, gx, d, n)."""
        self._grad_x((_blocks(_blocks(n, GX_TILE) * 32 * d_model, BLOCK),), (BLOCK,), args)

    def adamw(self, count: int, args: tuple) -> None:
        """args: (p, g, m, v, count, lr_decay, beta1, 1 - beta1, beta2, 1 - beta2, step_size, bias2_sqrt, eps)."""
        self._adamw((_blocks(count, BLOCK),), (BLOCK,), args)
