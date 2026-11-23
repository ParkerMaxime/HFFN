// Forward-only compact hidden-major HFFN kernel with fixed G4 workers.
// Deliberately contains no autograd, backward, gradient, or optimizer state.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <ATen/native/cpu/Gelu.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HFFN_HAS_NEON 1
#else
#define HFFN_HAS_NEON 0
#endif

namespace {
constexpr int kGroups = 4;

inline float gelu_none_scalar(float value) { return 0.5f * value * (1.0f + std::erf(value * 0.7071067811865475f)); }
// PyTorch's GELU(approximate="none") CPU primitive.  A fixed HFFN tile is
// contiguous on the token axis: it therefore maps to independent SIMD lanes
// without changing hidden-major compact storage or compact-edge ordering.
// On NEON the vector width is four float32 values; the scalar tail also keeps
// the same erf-form GELU semantics for an arbitrary tile width.
inline void gelu_none_tile(float* values, int count) {
    using Vec = at::vec::Vectorized<float>;
    constexpr int width = Vec::size();
    int token = 0;
    for (; token + width <= count; token += width) {
        const Vec input = Vec::loadu(values + token);
        at::native::vectorized_gelu(input).store(values + token);
    }
    for (; token < count; ++token) values[token] = gelu_none_scalar(values[token]);
}
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline void nt_to_t(const float* source, float* destination, int n, int h) { for (int token = 0; token < n; ++token) for (int feature = 0; feature < h; ++feature) destination[static_cast<size_t>(feature) * n + token] = source[static_cast<size_t>(token) * h + feature]; }
inline void t_to_nt(const float* source, float* destination, int n, int h) { for (int token = 0; token < n; ++token) for (int feature = 0; feature < h; ++feature) destination[static_cast<size_t>(token) * h + feature] = source[static_cast<size_t>(feature) * n + token]; }
inline int tile_for(int n) { return std::min(n, 128); }
inline void validate(const int32_t* groups, int n, int h, int f) {
    if (n <= 0 || h <= 0 || f <= 0 || groups[0] != 0 || groups[kGroups] != f) throw std::runtime_error("invalid compact G4 dimensions");
    for (int group = 0; group < kGroups; ++group) if (groups[group] > groups[group + 1]) throw std::runtime_error("non-monotonic compact G4 groups");
}

struct HiddenJob {
    const float* x; const int32_t* groups; const float* b_up; const int32_t* up_offsets; const int16_t* up_idx; const float* up_val;
    const int32_t* down_offsets; const int16_t* down_idx; const float* down_val;
    int n, h, begin, end; float* z; float* activation; float* y_local; bool write_down; bool materialize_activation;
};

struct OutputJob {
    const float* activation; const int32_t* offsets; const int32_t* slots; const int16_t* hidden; const float* down_val; const float* b_down;
    const int32_t* groups; int n, h; float* y_t;
};

// Fixed compact paths preserve the group-owned full-batch traversal. Candidate
// tiling policies are not part of this kernel.
template <int kBatch, int kTile>
inline void hidden_group_fixed(const HiddenJob& job, int group) {
    for (int tile_start = 0; tile_start < kBatch; tile_start += kTile) {
        for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
            const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
            const int down_start = job.write_down ? job.down_offsets[hidden] : 0;
            const int down_end = job.write_down ? job.down_offsets[hidden + 1] : 0;
            float accumulated[kTile];
            for (int token = 0; token < kTile; ++token) accumulated[token] = job.b_up[hidden];
            for (int slot = up_start; slot < up_end; ++slot) {
                const float weight = job.up_val[slot];
                const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * kBatch + tile_start;
                for (int token = 0; token < kTile; ++token) accumulated[token] += weight * input[token];
            }
            float* z = job.z + static_cast<size_t>(hidden) * kBatch + tile_start;
            float* active = job.activation + static_cast<size_t>(hidden) * kBatch + tile_start;
            if (job.materialize_activation) for (int token = 0; token < kTile; ++token) z[token] = accumulated[token];
            gelu_none_tile(accumulated, kTile);
            if (job.materialize_activation) for (int token = 0; token < kTile; ++token) active[token] = accumulated[token];
            if (job.write_down) for (int slot = down_start; slot < down_end; ++slot) {
                float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * kBatch + tile_start;
                const float weight = job.down_val[slot];
                const float* source = job.materialize_activation ? active : accumulated;
                for (int token = 0; token < kTile; ++token) output[token] += source[token] * weight;
            }
        }
    }
}

// Batch policy inherited from the validated FFN kernel: one G4 group owns a
// hidden row for the full batch.  Each compact edge is therefore fetched once
// and reused over every tile, rather than being reread for every token.  This
// NEON is applied only across independent token lanes; compact slots retain
// their original order.  This is the micro-kernel of the kept path.
template <int kBatch, int kTile>
inline void hidden_group_slot_inner_fixed(const HiddenJob& job, int group) {
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        const int down_start = job.write_down ? job.down_offsets[hidden] : 0;
        const int down_end = job.write_down ? job.down_offsets[hidden + 1] : 0;
        float accumulated[kBatch];
        for (int token = 0; token < kBatch; ++token) accumulated[token] = job.b_up[hidden];
        for (int slot = up_start; slot < up_end; ++slot) {
            const float weight = job.up_val[slot];
            const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * kBatch;
            for (int tile_start = 0; tile_start < kBatch; tile_start += kTile) {
#if HFFN_HAS_NEON
                for (int token = 0; token < kTile; token += 4) {
                    float32x4_t value = vld1q_f32(accumulated + tile_start + token);
                    value = vmlaq_n_f32(value, vld1q_f32(input + tile_start + token), weight);
                    vst1q_f32(accumulated + tile_start + token, value);
                }
#else
                for (int token = 0; token < kTile; ++token) accumulated[tile_start + token] += weight * input[tile_start + token];
#endif
            }
        }
        float* z = job.z + static_cast<size_t>(hidden) * kBatch;
        float* active = job.activation + static_cast<size_t>(hidden) * kBatch;
        if (job.materialize_activation) for (int token = 0; token < kBatch; ++token) z[token] = accumulated[token];
        gelu_none_tile(accumulated, kBatch);
        if (job.materialize_activation) for (int token = 0; token < kBatch; ++token) active[token] = accumulated[token];
        if (job.write_down) for (int slot = down_start; slot < down_end; ++slot) {
            float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * kBatch;
            const float weight = job.down_val[slot];
            const float* source = job.materialize_activation ? active : accumulated;
            for (int tile_start = 0; tile_start < kBatch; tile_start += kTile) {
#if HFFN_HAS_NEON
                for (int token = 0; token < kTile; token += 4) {
                    float32x4_t value = vld1q_f32(output + tile_start + token);
                    value = vmlaq_n_f32(value, vld1q_f32(source + tile_start + token), weight);
                    vst1q_f32(output + tile_start + token, value);
                }
#else
                for (int token = 0; token < kTile; ++token) output[tile_start + token] += source[tile_start + token] * weight;
#endif
            }
        }
    }
}

// a 128-token tile of a call of n tokens (row stride n, first token job.begin), same
// structure and same slot order as hidden_group_slot_inner_fixed.
// the body of one hidden neuron on one tile (first token `begin`) is a function of its
// own, so that the whole-block walk (hidden_group_token_tiles) runs the same body per neuron and per tile; the group
// loop below calls it with job.begin, as before.
template <int kTileWidth, int kTile>
inline void slot_inner_tile_neuron(const HiddenJob& job, int group, int hidden, int begin) {
    const size_t stride = static_cast<size_t>(job.n);
    const size_t first = static_cast<size_t>(begin);
    {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        const int down_start = job.write_down ? job.down_offsets[hidden] : 0;
        const int down_end = job.write_down ? job.down_offsets[hidden + 1] : 0;
        float accumulated[kTileWidth];
        for (int token = 0; token < kTileWidth; ++token) accumulated[token] = job.b_up[hidden];
        for (int slot = up_start; slot < up_end; ++slot) {
            const float weight = job.up_val[slot];
            const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * stride + first;
            for (int tile_start = 0; tile_start < kTileWidth; tile_start += kTile) {
#if HFFN_HAS_NEON
                for (int token = 0; token < kTile; token += 4) {
                    float32x4_t value = vld1q_f32(accumulated + tile_start + token);
                    value = vmlaq_n_f32(value, vld1q_f32(input + tile_start + token), weight);
                    vst1q_f32(accumulated + tile_start + token, value);
                }
#else
                for (int token = 0; token < kTile; ++token) accumulated[tile_start + token] += weight * input[tile_start + token];
#endif
            }
        }
        float* z = job.z + static_cast<size_t>(hidden) * stride + first;
        float* active = job.activation + static_cast<size_t>(hidden) * stride + first;
        if (job.materialize_activation) for (int token = 0; token < kTileWidth; ++token) z[token] = accumulated[token];
        gelu_none_tile(accumulated, kTileWidth);
        if (job.materialize_activation) for (int token = 0; token < kTileWidth; ++token) active[token] = accumulated[token];
        if (job.write_down) for (int slot = down_start; slot < down_end; ++slot) {
            float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * stride + first;
            const float weight = job.down_val[slot];
            const float* source = job.materialize_activation ? active : accumulated;
            for (int tile_start = 0; tile_start < kTileWidth; tile_start += kTile) {
#if HFFN_HAS_NEON
                for (int token = 0; token < kTile; token += 4) {
                    float32x4_t value = vld1q_f32(output + tile_start + token);
                    value = vmlaq_n_f32(value, vld1q_f32(source + tile_start + token), weight);
                    vst1q_f32(output + tile_start + token, value);
                }
#else
                for (int token = 0; token < kTile; ++token) output[tile_start + token] += source[tile_start + token] * weight;
#endif
            }
        }
    }
}
template <int kTileWidth, int kTile>
inline void hidden_group_slot_inner_tile(const HiddenJob& job, int group) {
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) slot_inner_tile_neuron<kTileWidth, kTile>(job, group, hidden, job.begin);
}
#if HFFN_HAS_NEON
// the fixed paths N = 8, 32, 64 and 96 keep the kBatch accumulators of a hidden neuron
// in NEON registers over its whole up, and its activation in registers over its whole down, instead of loading and
// storing the accumulator row at every compact slot and reloading the activation row at every down slot.  Per token
// lane the operations are those of the paths they replace (b_up, then one vmlaq_n_f32 per up slot in slot
// order, the same GELU primitive on the same values, one vmlaq_n_f32 per down slot in slot order), so z, h and
// every y_local entry are bitwise those of the previous kernel.  The GELU runs on a stack copy of the accumulators
// (gelu_none_tile, once per neuron).
template <int kBatch>
inline void hidden_group_registers(const HiddenJob& job, int group) {
    static_assert(kBatch % 4 == 0 && kBatch <= 96, "register path holds at most 24 float32x4 accumulators");
    constexpr int kVectors = kBatch / 4;
    alignas(16) float active_tile[kBatch];
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        float32x4_t accumulated[kVectors];
        const float32x4_t bias = vdupq_n_f32(job.b_up[hidden]);
#pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = bias;
        for (int slot = up_start; slot < up_end; ++slot) {
            const float weight = job.up_val[slot];
            const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * kBatch;
#pragma clang loop unroll(full)
            for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = vmlaq_n_f32(accumulated[vector], vld1q_f32(input + 4 * vector), weight);
        }
        float* z = job.z + static_cast<size_t>(hidden) * kBatch;
        float* active = job.activation + static_cast<size_t>(hidden) * kBatch;
#pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectors; ++vector) vst1q_f32(active_tile + 4 * vector, accumulated[vector]);
        if (job.materialize_activation) for (int token = 0; token < kBatch; ++token) z[token] = active_tile[token];
        gelu_none_tile(active_tile, kBatch);
        if (job.materialize_activation) for (int token = 0; token < kBatch; ++token) active[token] = active_tile[token];
        if (!job.write_down) continue;
#pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = vld1q_f32(active_tile + 4 * vector);
        for (int slot = job.down_offsets[hidden]; slot < job.down_offsets[hidden + 1]; ++slot) {
            float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * kBatch;
            const float weight = job.down_val[slot];
#pragma clang loop unroll(full)
            for (int vector = 0; vector < kVectors; ++vector) vst1q_f32(output + 4 * vector, vmlaq_n_f32(vld1q_f32(output + 4 * vector), accumulated[vector], weight));
        }
    }
}
// a 128-token tile (row stride n, first token job.begin) is traversed, inside each hidden
// neuron, as kTileWidth / kSubTile sub-tiles of kSubTile tokens, one after the other.  The kSubTile accumulators of a
// sub-tile stay in NEON registers over the whole up of the neuron and its activation stays in registers over its whole
// down, instead of loading and storing the accumulator row at every compact slot.  The compact edges of a neuron are
// read once per sub-tile.  Per token lane the operations are those of hidden_group_slot_inner_tile (b_up, one
// vmlaq_n_f32 per up slot in slot order, the same GELU primitive on the same values, one vmlaq_n_f32 per down
// slot in slot order), so z, h and every y_local entry are bitwise those of the previous kernel.
// the body of one hidden neuron on one tile (first token `begin`) is a function of its
// own, so that the whole-block walk (hidden_group_token_tiles) runs the same body per neuron and per tile; the group
// loop below calls it with job.begin, as before.
template <int kTileWidth, int kSubTile>
inline void tile_registers_neuron(const HiddenJob& job, int group, int hidden, int begin) {
    static_assert(kSubTile % 4 == 0 && kSubTile <= 64 && kTileWidth % kSubTile == 0, "a sub-tile holds at most 16 float32x4 accumulators");
    constexpr int kVectors = kSubTile / 4;
    const size_t stride = static_cast<size_t>(job.n);
    alignas(16) float active_tile[kSubTile];
    {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        const float32x4_t bias = vdupq_n_f32(job.b_up[hidden]);
        for (int sub = 0; sub < kTileWidth; sub += kSubTile) {
            const size_t first = static_cast<size_t>(begin) + sub;
            float32x4_t accumulated[kVectors];
#pragma clang loop unroll(full)
            for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = bias;
            for (int slot = up_start; slot < up_end; ++slot) {
                const float weight = job.up_val[slot];
                const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * stride + first;
#pragma clang loop unroll(full)
                for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = vmlaq_n_f32(accumulated[vector], vld1q_f32(input + 4 * vector), weight);
            }
            float* z = job.z + static_cast<size_t>(hidden) * stride + first;
            float* active = job.activation + static_cast<size_t>(hidden) * stride + first;
#pragma clang loop unroll(full)
            for (int vector = 0; vector < kVectors; ++vector) vst1q_f32(active_tile + 4 * vector, accumulated[vector]);
            if (job.materialize_activation) for (int token = 0; token < kSubTile; ++token) z[token] = active_tile[token];
            gelu_none_tile(active_tile, kSubTile);
            if (job.materialize_activation) for (int token = 0; token < kSubTile; ++token) active[token] = active_tile[token];
            if (!job.write_down) continue;
#pragma clang loop unroll(full)
            for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = vld1q_f32(active_tile + 4 * vector);
            for (int slot = job.down_offsets[hidden]; slot < job.down_offsets[hidden + 1]; ++slot) {
                float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * stride + first;
                const float weight = job.down_val[slot];
#pragma clang loop unroll(full)
                for (int vector = 0; vector < kVectors; ++vector) vst1q_f32(output + 4 * vector, vmlaq_n_f32(vld1q_f32(output + 4 * vector), accumulated[vector], weight));
            }
        }
    }
}
template <int kTileWidth, int kSubTile>
inline void hidden_group_tile_registers(const HiddenJob& job, int group) {
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) tile_registers_neuron<kTileWidth, kSubTile>(job, group, hidden, job.begin);
}
#endif

// the generic path (N = 1, 2, the other N without a fixed path, the tail tile) computes
// the up of kInterleave consecutive hidden neurons of the group together, token by token, so that their accumulation
// chains overlap instead of running one neuron at a time.  Each neuron keeps the exact operation sequence of the
// single-neuron loop it replaces, as the compiler emitted it for the previous kernel: an even chain from b_up and an odd chain
// from zero over the pairs of slots, joined as odd + even when the degree is at least two, then the last odd slot,
// every step one fused multiply-add.  z and the activation are written as before and the downs of the block follow in
// neuron order, so z, h and every y_local entry are bitwise those of the previous kernel.
constexpr int kInterleave = 4;

template <int kCount>
inline void generic_up_block(const HiddenJob& job, int first) {
#pragma clang fp reassociate(off)
    int start[kCount], paired_end[kCount], end[kCount];
    bool paired[kCount];
    int common = job.up_offsets[first + 1] - job.up_offsets[first];
    for (int member = 0; member < kCount; ++member) {
        start[member] = job.up_offsets[first + member];
        end[member] = job.up_offsets[first + member + 1];
        const int degree = end[member] - start[member];
        paired[member] = degree >= 2;
        paired_end[member] = start[member] + (degree & ~1);
        common = std::min(common, degree & ~1);
    }
    const size_t stride = static_cast<size_t>(job.n);
    for (int token = job.begin; token < job.end; ++token) {
        const float* x = job.x + token;
        float even[kCount], odd[kCount];
        for (int member = 0; member < kCount; ++member) { even[member] = job.b_up[first + member]; odd[member] = 0.0f; }
        for (int pair = 0; pair < common; pair += 2) {
            for (int member = 0; member < kCount; ++member) {
                const int slot = start[member] + pair;
                even[member] = std::fma(x[static_cast<size_t>(job.up_idx[slot]) * stride], job.up_val[slot], even[member]);
                odd[member] = std::fma(x[static_cast<size_t>(job.up_idx[slot + 1]) * stride], job.up_val[slot + 1], odd[member]);
            }
        }
        for (int member = 0; member < kCount; ++member) {
            float value = even[member], other = odd[member];
            for (int slot = start[member] + common; slot < paired_end[member]; slot += 2) {
                value = std::fma(x[static_cast<size_t>(job.up_idx[slot]) * stride], job.up_val[slot], value);
                other = std::fma(x[static_cast<size_t>(job.up_idx[slot + 1]) * stride], job.up_val[slot + 1], other);
            }
            if (paired[member]) value = other + value;
            for (int slot = paired_end[member]; slot < end[member]; ++slot) value = std::fma(x[static_cast<size_t>(job.up_idx[slot]) * stride], job.up_val[slot], value);
            job.z[static_cast<size_t>(first + member) * stride + token] = value;
            job.activation[static_cast<size_t>(first + member) * stride + token] = gelu_none_scalar(value);
        }
    }
}

// The down of one hidden neuron of the generic path, unchanged.
inline void generic_down(const HiddenJob& job, int group, int hidden) {
    if (!job.write_down) return;
    const int down_start = job.down_offsets[hidden], down_end = job.down_offsets[hidden + 1];
    const float* active = job.activation + static_cast<size_t>(hidden) * job.n;
    for (int slot = down_start; slot < down_end; ++slot) {
        float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * job.n;
        for (int token = job.begin; token < job.end; ++token) output[token] += active[token] * job.down_val[slot];
    }
}

// a block wider than one token tile (n > tile_for(n) = 128) in one pool dispatch.  The
// group walks its hidden neurons once, in the order of the generic path (blocks of kInterleave neurons from
// groups[group], then the remaining ones one by one); per neuron it runs the full 128-token tiles of the block with the
// tile body of the n = 128 regime (tile_registers_neuron<128, 64> on NEON, slot_inner_tile_neuron<128, 16> otherwise),
// and per block of neurons the tail of n % 128 tokens with the generic path (generic_up_block, then generic_down in
// neuron order), on a stack copy of the job whose begin is the first tail token.  Every z, activation and y_local entry
// receives the operations it received when the tiles were dispatched one by one (same neuron order per token, same slot
// order, same per-lane arithmetic); only the order in which tiles and neurons are visited changes.
inline void hidden_group_token_tiles(const HiddenJob& job, int group) {
    constexpr int kTokenTile = 128;
    const int full_end = job.begin + (job.end - job.begin) / kTokenTile * kTokenTile;
    HiddenJob tail = job;
    tail.begin = full_end;
    const bool has_tail = full_end < job.end;
    const int last = job.groups[group + 1];
    int hidden = job.groups[group];
    for (; hidden + kInterleave <= last; hidden += kInterleave) {
        for (int member = 0; member < kInterleave; ++member) {
            for (int first = job.begin; first < full_end; first += kTokenTile) {
#if HFFN_HAS_NEON
                tile_registers_neuron<kTokenTile, 64>(job, group, hidden + member, first);
#else
                slot_inner_tile_neuron<kTokenTile, 16>(job, group, hidden + member, first);
#endif
            }
        }
        if (!has_tail) continue;
        generic_up_block<kInterleave>(tail, hidden);
        for (int member = 0; member < kInterleave; ++member) generic_down(tail, group, hidden + member);
    }
    for (; hidden < last; ++hidden) {
        for (int first = job.begin; first < full_end; first += kTokenTile) {
#if HFFN_HAS_NEON
            tile_registers_neuron<kTokenTile, 64>(job, group, hidden, first);
#else
            slot_inner_tile_neuron<kTokenTile, 16>(job, group, hidden, first);
#endif
        }
        if (!has_tail) continue;
        generic_up_block<1>(tail, hidden);
        generic_down(tail, group, hidden);
    }
}

inline void hidden_group(const HiddenJob& job, int group) {
    // checked first, so that the fixed paths of n = 192 .. 1536 below stay unreached,
    // as they were when run_hidden split such blocks into 128-token tiles.
    if (job.end - job.begin > tile_for(job.n)) { hidden_group_token_tiles(job, group); return; }
#if HFFN_HAS_NEON
    if (job.begin == 0 && job.end == 8 && job.n == 8) { hidden_group_registers<8>(job, group); return; }
    if (job.begin == 0 && job.end == 32 && job.n == 32) { hidden_group_registers<32>(job, group); return; }
    if (job.begin == 0 && job.end == 64 && job.n == 64) { hidden_group_registers<64>(job, group); return; }
    if (job.begin == 0 && job.end == 96 && job.n == 96) { hidden_group_registers<96>(job, group); return; }
#else
    if (job.begin == 0 && job.end == 8 && job.n == 8) { hidden_group_fixed<8, 8>(job, group); return; }
    if (job.begin == 0 && job.end == 32 && job.n == 32) { hidden_group_slot_inner_fixed<32, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 64 && job.n == 64) { hidden_group_slot_inner_fixed<64, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 96 && job.n == 96) { hidden_group_slot_inner_fixed<96, 16>(job, group); return; }
#endif
    if (job.begin == 0 && job.end == 192 && job.n == 192) { hidden_group_slot_inner_fixed<192, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 256 && job.n == 256) { hidden_group_slot_inner_fixed<256, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 384 && job.n == 384) { hidden_group_slot_inner_fixed<384, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 768 && job.n == 768) { hidden_group_slot_inner_fixed<768, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 1536 && job.n == 1536) { hidden_group_slot_inner_fixed<1536, 16>(job, group); return; }
#if HFFN_HAS_NEON
    if (job.end - job.begin == 128) { hidden_group_tile_registers<128, 64>(job, group); return; }
#else
    if (job.end - job.begin == 128) { hidden_group_slot_inner_tile<128, 16>(job, group); return; }
#endif
    const int last = job.groups[group + 1];
    int hidden = job.groups[group];
    for (; hidden + kInterleave <= last; hidden += kInterleave) {
        generic_up_block<kInterleave>(job, hidden);
        for (int member = 0; member < kInterleave; ++member) generic_down(job, group, hidden + member);
    }
    for (; hidden < last; ++hidden) {
        generic_up_block<1>(job, hidden);
        generic_down(job, group, hidden);
    }
}

template <int kBatch>
inline void output_group_fixed(const OutputJob& job, int group) {
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        float* destination = job.y_t + static_cast<size_t>(output) * kBatch;
        for (int token = 0; token < kBatch; ++token) destination[token] = job.b_down[output];
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * kBatch;
            const float weight = job.down_val[job.slots[slot_index]];
            for (int token = 0; token < kBatch; ++token) destination[token] += active[token] * weight;
        }
    }
}

template <int kBatch>
inline void output_group_neon_fixed(const OutputJob& job, int group) {
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        float* destination = job.y_t + static_cast<size_t>(output) * kBatch;
#if HFFN_HAS_NEON
        for (int token = 0; token < kBatch; token += 4) vst1q_f32(destination + token, vdupq_n_f32(job.b_down[output]));
#else
        for (int token = 0; token < kBatch; ++token) destination[token] = job.b_down[output];
#endif
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * kBatch;
            const float weight = job.down_val[job.slots[slot_index]];
#if HFFN_HAS_NEON
            for (int token = 0; token < kBatch; token += 4) {
                float32x4_t value = vld1q_f32(destination + token);
                value = vmlaq_n_f32(value, vld1q_f32(active + token), weight);
                vst1q_f32(destination + token, value);
            }
#else
            for (int token = 0; token < kBatch; ++token) destination[token] += active[token] * weight;
#endif
        }
    }
}

#if HFFN_HAS_NEON
// the accumulators of kWidth tokens of an output row (row stride n, first token `first`)
// stay in NEON registers over all the slots of the row, instead of being loaded and stored at every slot.  Per token lane
// the operations are those of the loops they replace (b_down, then one vmlaq_n_f32 per slot in slot order), so y is
// bitwise that of the previous kernel.
template <int kWidth>
inline void output_row_registers(const OutputJob& job, int output, int first) {
    static_assert(kWidth % 4 == 0 && kWidth <= 96, "register path holds at most 24 float32x4 accumulators");
    constexpr int kVectors = kWidth / 4;
    const size_t stride = static_cast<size_t>(job.n);
    float32x4_t accumulated[kVectors];
    const float32x4_t bias = vdupq_n_f32(job.b_down[output]);
#pragma clang loop unroll(full)
    for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = bias;
    for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
        const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * stride + first;
        const float weight = job.down_val[job.slots[slot_index]];
#pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectors; ++vector) accumulated[vector] = vmlaq_n_f32(accumulated[vector], vld1q_f32(active + 4 * vector), weight);
    }
    float* destination = job.y_t + static_cast<size_t>(output) * stride + first;
#pragma clang loop unroll(full)
    for (int vector = 0; vector < kVectors; ++vector) vst1q_f32(destination + 4 * vector, accumulated[vector]);
}
#endif

inline void output_group(const OutputJob& job, int group) {
    if (job.n == 8) { output_group_fixed<8>(job, group); return; }
#if HFFN_HAS_NEON
    if (job.n == 32 || job.n == 96) {
        for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
            if (job.n == 32) output_row_registers<32>(job, output, 0); else output_row_registers<96>(job, output, 0);
        }
        return;
    }
    const int tiled = job.n >= 64 ? job.n / 64 * 64 : 0;
#else
    if (job.n == 32) { output_group_fixed<32>(job, group); return; }
    if (job.n == 64) { output_group_neon_fixed<64>(job, group); return; }
    const int tiled = 0;
#endif
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
#if HFFN_HAS_NEON
        for (int first = 0; first < tiled; first += 64) output_row_registers<64>(job, output, first);
#endif
        if (tiled == job.n) continue;
        float* destination = job.y_t + static_cast<size_t>(output) * job.n;
        for (int token = tiled; token < job.n; ++token) destination[token] = job.b_down[output];
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * job.n;
            const float weight = job.down_val[job.slots[slot_index]];
            for (int token = tiled; token < job.n; ++token) destination[token] += active[token] * weight;
        }
    }
}

// Three persistent workers plus the calling thread own exactly four groups.
class G4Pool {
public:
    G4Pool() { for (int group = 1; group < kGroups; ++group) workers[group - 1] = std::thread(&G4Pool::loop, this, group); }
    ~G4Pool() { { std::lock_guard<std::mutex> lock(mutex); stop = true; } start.notify_all(); for (auto& worker : workers) if (worker.joinable()) worker.join(); }
    void hidden(const HiddenJob& job) { run(job, OutputJob{}, Kind::Hidden); }
    void output(const OutputJob& job) { run(HiddenJob{}, job, Kind::Output); }
private:
    enum class Kind { Hidden, Output };
    void run(const HiddenJob& hidden_job, const OutputJob& output_job, Kind next_kind) {
        { std::lock_guard<std::mutex> lock(mutex); hidden_current = hidden_job; output_current = output_job; kind = next_kind; done = 0; ++generation; }
        start.notify_all();
        if (next_kind == Kind::Hidden) hidden_group(hidden_job, 0); else output_group(output_job, 0);
        std::unique_lock<std::mutex> lock(mutex); finish.wait(lock, [&] { return done == 3; });
    }
    void loop(int group) {
        uint64_t seen = 0;
        for (;;) {
            Kind local_kind; HiddenJob hidden_job{}; OutputJob output_job{};
            { std::unique_lock<std::mutex> lock(mutex); start.wait(lock, [&] { return stop || generation != seen; }); if (stop) return; seen = generation; local_kind = kind; hidden_job = hidden_current; output_job = output_current; }
            if (local_kind == Kind::Hidden) hidden_group(hidden_job, group); else output_group(output_job, group);
            { std::lock_guard<std::mutex> lock(mutex); ++done; }
            finish.notify_one();
        }
    }
    std::mutex mutex; std::condition_variable start, finish; std::thread workers[3];
    HiddenJob hidden_current{}; OutputJob output_current{}; Kind kind = Kind::Hidden; uint64_t generation = 0; int done = 0; bool stop = false;
};

inline G4Pool& g4_pool() { static G4Pool pool; return pool; }

// one pool dispatch for the whole block of n >= 3 tokens; a block wider than one
// 128-token tile is walked group by group, tiles inside each neuron (hidden_group_token_tiles), instead of one pool
// dispatch per tile.
inline void run_hidden(const HiddenJob& base) {
    if (base.n <= 2) { for (int group = 0; group < kGroups; ++group) hidden_group(base, group); return; }
    g4_pool().hidden(base);
}

template <int kBatch>
inline void reduce_hidden_owned_fixed(const float* b_down, const float* y_local, float* y_t, int h) {
    for (int output = 0; output < h; ++output) {
        const float* group0 = y_local + static_cast<size_t>(output) * kBatch;
        const float* group1 = y_local + (static_cast<size_t>(h) + output) * kBatch;
        const float* group2 = y_local + (static_cast<size_t>(2 * h) + output) * kBatch;
        const float* group3 = y_local + (static_cast<size_t>(3 * h) + output) * kBatch;
        float* destination = y_t + static_cast<size_t>(output) * kBatch;
        for (int token = 0; token < kBatch; ++token) destination[token] = b_down[output] + group0[token] + group1[token] + group2[token] + group3[token];
    }
}

inline void reduce_hidden_owned(const float* b_down, const float* y_local, float* y_t, int n, int h) {
    if (n == 8) { reduce_hidden_owned_fixed<8>(b_down, y_local, y_t, h); return; }
    if (n == 32) { reduce_hidden_owned_fixed<32>(b_down, y_local, y_t, h); return; }
    if (n == 64) { reduce_hidden_owned_fixed<64>(b_down, y_local, y_t, h); return; }
    for (int output = 0; output < h; ++output) for (int token = 0; token < n; ++token) { float value = b_down[output]; for (int group = 0; group < kGroups; ++group) value += y_local[(static_cast<size_t>(group) * h + output) * n + token]; y_t[static_cast<size_t>(output) * n + token] = value; }
}
}

// Strict decode specialization.  With one token the baseline policy already
// runs groups sequentially; write the final output directly instead of paying
// for [4,H,1] zeroing/reduction and two layout transposes.
extern "C" void hffn_compact_g4_forward_decode_nt_f32(const float* x, const int32_t* groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const int32_t* down_offsets, const int16_t* down_idx, const float* down_val, const float* b_down, int h, int f, float* y, float* z, float* activation, int64_t* timing) {
    validate(groups, 1, h, f); const int64_t begin = now_ns();
    for (int output = 0; output < h; ++output) y[output] = b_down[output];
    for (int group = 0; group < kGroups; ++group) for (int hidden = groups[group]; hidden < groups[group + 1]; ++hidden) {
        float value = b_up[hidden];
        for (int slot = up_offsets[hidden]; slot < up_offsets[hidden + 1]; ++slot) value += up_val[slot] * x[up_idx[slot]];
        z[hidden] = value; const float active = gelu_none_scalar(value); activation[hidden] = active;
        for (int slot = down_offsets[hidden]; slot < down_offsets[hidden + 1]; ++slot) y[down_idx[slot]] += active * down_val[slot];
    }
    const int64_t end = now_ns(); timing[0] = 0; timing[1] = end - begin; timing[2] = 0;
}

// One-token decode on the G4 pool.  The four hidden groups run on the pool, each into its own
// private accumulator y_local[g] (4 * h floats, zero-filled here, one writer per group), then the fixed-order
// reduction y = b_down + y_local[0] + .. + y_local[3] (the one of the N >= 3 path).  Per group the generic stride-1
// path runs (slot order of the sequential entry above); only the association of the output sum differs from it.
extern "C" void hffn_compact_g4_forward_decode_nt_groups_f32(const float* x, const int32_t* groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const int32_t* down_offsets, const int16_t* down_idx, const float* down_val, const float* b_down, int h, int f, float* y, float* z, float* activation, float* y_local, int64_t* timing) {
    validate(groups, 1, h, f); const int64_t begin = now_ns();
    std::fill(y_local, y_local + static_cast<size_t>(kGroups) * h, 0.0f);
    g4_pool().hidden(HiddenJob{x, groups, b_up, up_offsets, up_idx, up_val, down_offsets, down_idx, down_val, 1, h, 0, 1, z, activation, y_local, true, false});
    reduce_hidden_owned_fixed<1>(b_down, y_local, y, h);
    const int64_t end = now_ns(); timing[0] = 0; timing[1] = end - begin; timing[2] = 0;
}

extern "C" void hffn_compact_g4_forward_nt_f32(const float* x_nt, const int32_t* groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const int32_t* down_offsets, const int16_t* down_idx, const float* down_val, const float* b_down, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, float* y_local, int64_t* timing) {
    validate(groups, n, h, f); const int64_t begin = now_ns(); nt_to_t(x_nt, x_t, n, h); const int64_t transposed = now_ns();
    std::fill(y_local, y_local + static_cast<size_t>(kGroups) * h * n, 0.0f);
    // Hidden-owned down consumes each activation immediately.  Do not write
    // diagnostic [F,N] intermediates during inference.
    run_hidden(HiddenJob{x_t, groups, b_up, up_offsets, up_idx, up_val, down_offsets, down_idx, down_val, n, h, 0, n, z, activation, y_local, true, false});
    reduce_hidden_owned(b_down, y_local, y_t, n, h);
    const int64_t reduced = now_ns(); t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = reduced - transposed; timing[2] = end - reduced;
}

// the fused hidden-owned route of `hffn_compact_g4_forward_nt_f32` that also stores z and the activation
// (materialize_activation = true), as the dual entry does for the training forward.  Same hidden stage, same reduction, one pool
// dispatch; `hffn_compact_g4_forward_nt_f32` itself is unchanged.
extern "C" void hffn_compact_g4_forward_stores_nt_f32(const float* x_nt, const int32_t* groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const int32_t* down_offsets, const int16_t* down_idx, const float* down_val, const float* b_down, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, float* y_local, int64_t* timing) {
    validate(groups, n, h, f); const int64_t begin = now_ns(); nt_to_t(x_nt, x_t, n, h); const int64_t transposed = now_ns();
    std::fill(y_local, y_local + static_cast<size_t>(kGroups) * h * n, 0.0f);
    run_hidden(HiddenJob{x_t, groups, b_up, up_offsets, up_idx, up_val, down_offsets, down_idx, down_val, n, h, 0, n, z, activation, y_local, true, true});
    reduce_hidden_owned(b_down, y_local, y_t, n, h);
    const int64_t reduced = now_ns(); t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = reduced - transposed; timing[2] = end - reduced;
}

extern "C" void hffn_compact_g4_forward_dual_nt_f32(const float* x_nt, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* down_val, const float* b_down, const int32_t* down_output_offsets, const int32_t* down_slots, const int16_t* down_hidden, const int32_t* output_groups, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    validate(hidden_groups, n, h, f); const int64_t begin = now_ns(); nt_to_t(x_nt, x_t, n, h); const int64_t transposed = now_ns();
    run_hidden(HiddenJob{x_t, hidden_groups, b_up, up_offsets, up_idx, up_val, nullptr, nullptr, down_val, n, h, 0, n, z, activation, nullptr, false, true});
    const int64_t hidden_done = now_ns(); g4_pool().output(OutputJob{activation, down_output_offsets, down_slots, down_hidden, down_val, b_down, output_groups, n, h, y_t});
    const int64_t reduced = now_ns(); t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = hidden_done - transposed; timing[2] = reduced - hidden_done; timing[3] = end - reduced;
}
