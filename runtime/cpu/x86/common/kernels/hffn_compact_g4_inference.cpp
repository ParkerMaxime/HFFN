// HFFN compact inference kernel, x86_64 (AVX2/FMA, scalar fallback).
//
// Forward only. A caller that trains passes a z buffer distinct from `activation` to the dual route
// and receives the pre-activation z next to h = gelu(z); inference callers pass z == activation. A fixed sparse FFN y = W_down gelu(W_up x + b_up) + b_down
// is stored and executed per hidden neuron: compact incoming slots (up_idx, up_val), bias,
// compact outgoing slots (down_idx, down_val). T groups of hidden neurons (T = `threads`,
// 1..kMaxThreads; G4 is T = 4), one thread per group, private buffers per group,
// deterministic reductions in group order. Every call states T and passes partitions of
// T + 1 offsets; for a given T and partition the result does not depend on scheduling.
//
// Vectorisation runs over token lanes (the N axis, contiguous in the transposed [feature, N]
// layout); compact slots keep their order. AVX2/FMA paths are selected at compile time
// (-mavx2 -mfma or -march=native); without them the scalar paths are compiled.
//
// The groups run on a private persistent pool (default, -pthread), or, with
// -DHFFN_GROUPS_OPENMP -fopenmp, on the OpenMP team of the process: the choice of a caller
// that already runs such a team (PyTorch). Same computation per group in both.
#include <algorithm>
#ifndef HFFN_GROUPS_OPENMP
#include <atomic>
#endif
#include <chrono>
#include <cmath>
#ifndef HFFN_GROUPS_OPENMP
#include <condition_variable>
#endif
#include <cstdint>
#ifndef HFFN_GROUPS_OPENMP
#include <mutex>
#endif
#include <stdexcept>
#ifndef HFFN_GROUPS_OPENMP
#include <thread>
#include <vector>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HFFN_HAS_AVX2 1
#else
#define HFFN_HAS_AVX2 0
#endif
// glibc's 8-lane erff (libmvec, x86-64 vector ABI). Its bits differ from the scalar erff on
// about a third of the floats: AVX2 builds use it for every GELU, so that the result of a
// token depends neither on N nor on its position in the batch.
#if HFFN_HAS_AVX2 && defined(__GLIBC__)
extern "C" __m256 _ZGVdN8v_erff(__m256);
#define HFFN_VECTOR_ERF 1
#else
#define HFFN_VECTOR_ERF 0
#endif

namespace {
constexpr int kMaxThreads = 64;

inline float gelu_none_scalar(float value) { return 0.5f * value * (1.0f + std::erf(value * 0.7071067811865475f)); }
// Exact GELU (erf form, PyTorch's approximate="none") over a contiguous token tile.
// With -ffast-math, g++ vectorises this loop through glibc's libmvec erf. It
// replaces ATen's generic vectorized_gelu (same expression order), which applied
// the scalar erff per element: results differ by at most ~1e-6, and the kernel
// no longer needs PyTorch headers or libc10. Out of line, as ATen's was: inlined
// into the N = 8 path it slowed that path by 25-70 %.
__attribute__((noinline)) void gelu_none_tile(float* values, int count) {
    for (int token = 0; token < count; ++token) {
        const float value = values[token];
        values[token] = value * 0.5f * (1.0f + std::erf(value * 0.70710678118654752f));
    }
}
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// Layout transposes walk blocks of features so that the strided side ([feature, N],
// stride N floats) stays within a few cache lines per pass; same data, same layout.
constexpr int kTransposeBlock = 16;
// Smallest N * h for which the input transpose of the dual route runs on the groups: below
// it the transpose costs less than the pool round trip (measured neutral on an 8-core Zen 4).
constexpr int64_t kGroupTransposeMin = 12288;
#if HFFN_HAS_AVX2
// Full 8x8 tiles go through registers (unpack/shuffle/permute); pure data movement.
inline void transpose8x8(const float* source, size_t source_stride, float* destination, size_t destination_stride) {
    const __m256 r0 = _mm256_loadu_ps(source), r1 = _mm256_loadu_ps(source + source_stride);
    const __m256 r2 = _mm256_loadu_ps(source + 2 * source_stride), r3 = _mm256_loadu_ps(source + 3 * source_stride);
    const __m256 r4 = _mm256_loadu_ps(source + 4 * source_stride), r5 = _mm256_loadu_ps(source + 5 * source_stride);
    const __m256 r6 = _mm256_loadu_ps(source + 6 * source_stride), r7 = _mm256_loadu_ps(source + 7 * source_stride);
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1), t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5), t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)), s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)), s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0)), s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0)), s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(destination, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(destination + destination_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(destination + 2 * destination_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(destination + 3 * destination_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(destination + 4 * destination_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(destination + 5 * destination_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(destination + 6 * destination_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(destination + 7 * destination_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
}
#endif
// Transpose rows x columns (row-major, stride `columns`) into columns x rows.
inline void transpose_blocked(const float* source, float* destination, int rows, int columns) {
    int column_start = 0;
#if HFFN_HAS_AVX2
    const int full_rows = rows - rows % 8;
    for (; column_start + 8 <= columns; column_start += 8) {
        for (int row = 0; row < full_rows; row += 8)
            transpose8x8(source + static_cast<size_t>(row) * columns + column_start, columns, destination + static_cast<size_t>(column_start) * rows + row, rows);
        for (int row = full_rows; row < rows; ++row) for (int column = column_start; column < column_start + 8; ++column)
            destination[static_cast<size_t>(column) * rows + row] = source[static_cast<size_t>(row) * columns + column];
    }
#endif
    for (; column_start < columns; column_start += kTransposeBlock) {
        const int end = std::min(column_start + kTransposeBlock, columns);
        for (int row = 0; row < rows; ++row) for (int column = column_start; column < end; ++column) destination[static_cast<size_t>(column) * rows + row] = source[static_cast<size_t>(row) * columns + column];
    }
}
// Features [begin, end) of x[N, h] into the rows [begin, end) of x_t[h, N]: the part of the
// input transpose owned by one group. Same data movement as transpose_blocked.
inline void nt_to_t_features(const float* source, float* destination, int n, int h, int begin, int end) {
    int feature = begin;
#if HFFN_HAS_AVX2
    const int full_tokens = n - n % 8;
    for (; feature + 8 <= end; feature += 8) {
        for (int token = 0; token < full_tokens; token += 8)
            transpose8x8(source + static_cast<size_t>(token) * h + feature, h, destination + static_cast<size_t>(feature) * n + token, n);
        for (int token = full_tokens; token < n; ++token) for (int column = feature; column < feature + 8; ++column)
            destination[static_cast<size_t>(column) * n + token] = source[static_cast<size_t>(token) * h + column];
    }
#endif
    for (; feature < end; feature += kTransposeBlock) {
        const int stop = std::min(feature + kTransposeBlock, end);
        for (int token = 0; token < n; ++token) for (int column = feature; column < stop; ++column) destination[static_cast<size_t>(column) * n + token] = source[static_cast<size_t>(token) * h + column];
    }
}
// Rows [begin, end) of y_t[h, N] into the features [begin, end) of y[N, h]: the part of the
// output transpose owned by one group.
inline void t_to_nt_features(const float* source, float* destination, int n, int h, int begin, int end) {
    int feature = begin;
#if HFFN_HAS_AVX2
    const int full_tokens = n - n % 8;
    for (; feature + 8 <= end; feature += 8) {
        for (int token = 0; token < full_tokens; token += 8)
            transpose8x8(source + static_cast<size_t>(feature) * n + token, n, destination + static_cast<size_t>(token) * h + feature, h);
        for (int token = full_tokens; token < n; ++token) for (int column = feature; column < feature + 8; ++column)
            destination[static_cast<size_t>(token) * h + column] = source[static_cast<size_t>(column) * n + token];
    }
#endif
    for (; feature < end; ++feature) for (int token = 0; token < n; ++token) destination[static_cast<size_t>(token) * h + feature] = source[static_cast<size_t>(feature) * n + token];
}
inline void nt_to_t(const float* source, float* destination, int n, int h) { transpose_blocked(source, destination, n, h); }
inline void t_to_nt(const float* source, float* destination, int n, int h) { transpose_blocked(source, destination, h, n); }
#if HFFN_HAS_AVX2
// Mask of the first min(count, 8) lanes, for masked loads/stores of a token tail.
inline __m256i lane_mask(int count) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(std::min(count, 8)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
#endif
#if HFFN_HAS_AVX2
// Exact GELU (erf form) of 8 lanes held in a register.
inline __m256 gelu_lanes(__m256 value) {
#if HFFN_VECTOR_ERF
    const __m256 erf = _ZGVdN8v_erff(_mm256_mul_ps(value, _mm256_set1_ps(0.70710678118654752f)));
    return _mm256_mul_ps(_mm256_mul_ps(value, _mm256_set1_ps(0.5f)), _mm256_add_ps(_mm256_set1_ps(1.0f), erf));
#else
    alignas(32) float lanes[8]; _mm256_store_ps(lanes, value);
    for (int lane = 0; lane < 8; ++lane) lanes[lane] = gelu_none_scalar(lanes[lane]);
    return _mm256_load_ps(lanes);
#endif
}
#endif
inline int tile_for(int n) { return std::min(n, 128); }
// `groups` holds threads + 1 offsets partitioning [0, size).
inline void validate(int threads, const int32_t* groups, int size, int n, int h, int f) {
    if (threads < 1 || threads > kMaxThreads || n <= 0 || h <= 0 || f <= 0 || groups[0] != 0 || groups[threads] != size) throw std::runtime_error("invalid compact dimensions or partition");
    for (int group = 0; group < threads; ++group) if (groups[group] > groups[group + 1]) throw std::runtime_error("non-monotonic compact partition");
}

struct HiddenJob {
    const float* x; const int32_t* groups; const float* b_up; const int32_t* up_offsets; const int16_t* up_idx; const float* up_val;
    const int32_t* down_offsets; const int16_t* down_idx; const float* down_val;
    int n, h, begin, end; float* z; float* activation; float* y_local; bool write_down; bool materialize_activation;
};

struct OutputJob {
    // Reverse schedule: per output feature, its hidden neurons and its down weights (output order).
    const float* activation; const int32_t* offsets; const int16_t* hidden; const float* weights; const float* b_down;
    const int32_t* groups; int n, h; float* y_t; float* y_nt;
    // Optional: when set, the down weight of slot s is weights[slots[s]] (weights in hidden-major order,
    // as a training runtime holds them); the reverse schedule, its slot order and the sums are unchanged.
    const int32_t* slots;
};

// Work on feature rows split over the groups: the input transpose (group g fills the rows of
// x_t in its `groups` range) or the reduction of the hidden-owned slabs (uniform split in `parts`).
struct FeatureJob { const float* source; float* destination; const int32_t* groups; int n, h; const float* bias; int parts; };

// N = 8, scalar builds only (AVX2 builds take hidden_group_lanes for every N up to 96).
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
#if HFFN_HAS_AVX2
                static_assert(kTile % 8 == 0, "AVX2 token lanes need a tile multiple of 8");
                const __m256 broadcast = _mm256_set1_ps(weight);
                for (int token = 0; token < kTile; token += 8) {
                    float* lane = accumulated + token;
                    _mm256_storeu_ps(lane, _mm256_fmadd_ps(_mm256_loadu_ps(input + token), broadcast, _mm256_loadu_ps(lane)));
                }
#else
                for (int token = 0; token < kTile; ++token) accumulated[token] += weight * input[token];
#endif
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
#if HFFN_HAS_AVX2
                const __m256 broadcast = _mm256_set1_ps(weight);
                for (int token = 0; token < kTile; token += 8) {
                    float* lane = output + token;
                    _mm256_storeu_ps(lane, _mm256_fmadd_ps(_mm256_loadu_ps(source + token), broadcast, _mm256_loadu_ps(lane)));
                }
#else
                for (int token = 0; token < kTile; ++token) output[token] += source[token] * weight;
#endif
            }
        }
    }
}

// N = 32, 64, 96, scalar builds only: one group owns a hidden row for the full batch, so each
// compact slot is read once and reused over every token (slot-inner traversal,
// faster on x86 than tile-outer policies). AVX2 runs across token lanes only;
// compact slots keep their order.
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
#if HFFN_HAS_AVX2
                static_assert(kTile % 8 == 0, "AVX2 token lanes need a tile multiple of 8");
                const __m256 broadcast = _mm256_set1_ps(weight);
                for (int token = 0; token < kTile; token += 8) {
                    float* lane = accumulated + tile_start + token;
                    _mm256_storeu_ps(lane, _mm256_fmadd_ps(_mm256_loadu_ps(input + tile_start + token), broadcast, _mm256_loadu_ps(lane)));
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
#if HFFN_HAS_AVX2
                const __m256 broadcast = _mm256_set1_ps(weight);
                for (int token = 0; token < kTile; token += 8) {
                    float* lane = output + tile_start + token;
                    _mm256_storeu_ps(lane, _mm256_fmadd_ps(_mm256_loadu_ps(source + tile_start + token), broadcast, _mm256_loadu_ps(lane)));
                }
#else
                for (int token = 0; token < kTile; ++token) output[tile_start + token] += source[tile_start + token] * weight;
#endif
            }
        }
    }
}

#if HFFN_HAS_AVX2
// Generic batch sizes up to 96 tokens: slot-inner traversal (each compact slot
// read once for the whole batch, as the fixed paths) with the batch held in
// kChunks ymm accumulators; the last chunk is masked.
template <int kChunks, bool kPartial>
inline void hidden_group_lanes(const HiddenJob& job, int group) {
    const int n = job.n;
    // Full chunks use plain loads/stores; only a partial last chunk is masked.
    const __m256i tail = lane_mask(n - 8 * (kChunks - 1));
    auto load = [&](const float* source, int chunk) { return chunk + 1 < kChunks || !kPartial ? _mm256_loadu_ps(source + 8 * chunk) : _mm256_maskload_ps(source + 8 * chunk, tail); };
    auto store = [&](float* destination, int chunk, __m256 value) { if (chunk + 1 < kChunks || !kPartial) _mm256_storeu_ps(destination + 8 * chunk, value); else _mm256_maskstore_ps(destination + 8 * chunk, tail, value); };
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        __m256 value[kChunks];
        for (int chunk = 0; chunk < kChunks; ++chunk) value[chunk] = _mm256_set1_ps(job.b_up[hidden]);
        for (int slot = job.up_offsets[hidden]; slot < job.up_offsets[hidden + 1]; ++slot) {
            const float* input = job.x + static_cast<size_t>(job.up_idx[slot]) * n;
            const __m256 weight = _mm256_set1_ps(job.up_val[slot]);
            for (int chunk = 0; chunk < kChunks; ++chunk) value[chunk] = _mm256_fmadd_ps(load(input, chunk), weight, value[chunk]);
        }
        // Without materialisation (baseline route) the activations of one hidden
        // row live in a stack scratch: the [F, N] activation rows are never touched.
        alignas(32) float scratch[8 * kChunks];
        float* active = job.materialize_activation ? job.activation + static_cast<size_t>(hidden) * n : scratch;
        // GELU, 8 lanes per erf call for every N. Full chunks are stored first and processed in
        // place, so that no accumulator is live across the calls; a partial last chunk goes
        // from its register (reloading a masked store is slow), its lanes past N not stored.
        const int full_chunks = kPartial ? kChunks - 1 : kChunks;
        // A caller that passes a z buffer distinct from the activation (training) also keeps the pre-activation.
        if (job.materialize_activation && job.z != job.activation) {
            float* pre = job.z + static_cast<size_t>(hidden) * n;
            for (int chunk = 0; chunk < kChunks; ++chunk) store(pre, chunk, value[chunk]);
        }
        for (int chunk = 0; chunk < full_chunks; ++chunk) _mm256_storeu_ps(active + 8 * chunk, value[chunk]);
        if (kPartial) _mm256_maskstore_ps(active + 8 * full_chunks, tail, gelu_lanes(value[full_chunks]));
        for (int chunk = 0; chunk < full_chunks; ++chunk) _mm256_storeu_ps(active + 8 * chunk, gelu_lanes(_mm256_loadu_ps(active + 8 * chunk)));
        if (job.write_down) for (int slot = job.down_offsets[hidden]; slot < job.down_offsets[hidden + 1]; ++slot) {
            float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * n;
            const __m256 weight = _mm256_set1_ps(job.down_val[slot]);
            for (int chunk = 0; chunk < kChunks; ++chunk) store(output, chunk, _mm256_fmadd_ps(load(active, chunk), weight, load(output, chunk)));
        }
    }
}

// Returns false when the batch is outside the register-blocked range. Out of line
// (keeps hidden_group small).
__attribute__((noinline)) bool hidden_group_lanes_dispatch(const HiddenJob& job, int group) {
    if (job.begin != 0 || job.end != job.n || job.n > 96) return false;
    switch ((job.n + 7) / 8) {
        case 1: if (job.n % 8) hidden_group_lanes<1, true>(job, group); else hidden_group_lanes<1, false>(job, group); return true;
        case 2: if (job.n % 8) hidden_group_lanes<2, true>(job, group); else hidden_group_lanes<2, false>(job, group); return true;
        case 3: if (job.n % 8) hidden_group_lanes<3, true>(job, group); else hidden_group_lanes<3, false>(job, group); return true;
        case 4: if (job.n % 8) hidden_group_lanes<4, true>(job, group); else hidden_group_lanes<4, false>(job, group); return true;
        case 5: if (job.n % 8) hidden_group_lanes<5, true>(job, group); else hidden_group_lanes<5, false>(job, group); return true;
        case 6: if (job.n % 8) hidden_group_lanes<6, true>(job, group); else hidden_group_lanes<6, false>(job, group); return true;
        case 7: if (job.n % 8) hidden_group_lanes<7, true>(job, group); else hidden_group_lanes<7, false>(job, group); return true;
        case 8: if (job.n % 8) hidden_group_lanes<8, true>(job, group); else hidden_group_lanes<8, false>(job, group); return true;
        case 9: if (job.n % 8) hidden_group_lanes<9, true>(job, group); else hidden_group_lanes<9, false>(job, group); return true;
        case 10: if (job.n % 8) hidden_group_lanes<10, true>(job, group); else hidden_group_lanes<10, false>(job, group); return true;
        case 11: if (job.n % 8) hidden_group_lanes<11, true>(job, group); else hidden_group_lanes<11, false>(job, group); return true;
        case 12: if (job.n % 8) hidden_group_lanes<12, true>(job, group); else hidden_group_lanes<12, false>(job, group); return true;
        default: return false;
    }
}
#endif

// Decode up projection, one group per thread; each group writes only its own
// z/activation entries. The slot sum uses four independent partial sums (slot s
// in partial s % 4), written out, out of line and with auto-vectorisation
// disabled: the fastest form measured on x86 (a 16-lane gather is 15-35 % slower,
// a single scalar chain up to 16 %), and one that does not depend on what g++
// decides for the surrounding code.
__attribute__((noinline, optimize("no-tree-vectorize"))) void decode_up_group(const HiddenJob& job, int group) {
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        const int end = job.up_offsets[hidden + 1];
        int slot = job.up_offsets[hidden];
        float partial0 = 0.0f, partial1 = 0.0f, partial2 = 0.0f, partial3 = 0.0f;
        for (; slot + 4 <= end; slot += 4) {
            partial0 += job.up_val[slot] * job.x[job.up_idx[slot]];
            partial1 += job.up_val[slot + 1] * job.x[job.up_idx[slot + 1]];
            partial2 += job.up_val[slot + 2] * job.x[job.up_idx[slot + 2]];
            partial3 += job.up_val[slot + 3] * job.x[job.up_idx[slot + 3]];
        }
        for (; slot < end; ++slot) partial0 += job.up_val[slot] * job.x[job.up_idx[slot]];
        const float value = job.b_up[hidden] + ((partial0 + partial1) + (partial2 + partial3));
        job.activation[hidden] = value;
    }
    float* active = job.activation;
    const int end = job.groups[group + 1];
#if HFFN_HAS_AVX2
    int hidden = job.groups[group];
    for (; hidden + 8 <= end; hidden += 8) _mm256_storeu_ps(active + hidden, gelu_lanes(_mm256_loadu_ps(active + hidden)));
    if (hidden < end) { const __m256i mask = lane_mask(end - hidden); _mm256_maskstore_ps(active + hidden, mask, gelu_lanes(_mm256_maskload_ps(active + hidden, mask))); }
#else
    for (int hidden = job.groups[group]; hidden < end; ++hidden) active[hidden] = gelu_none_scalar(active[hidden]);
#endif
}

inline void hidden_group(const HiddenJob& job, int group) {
    // Hidden-owned down: the group clears its private y_local slab itself.
    if (job.write_down && job.begin == 0) { float* slab = job.y_local + static_cast<size_t>(group) * job.h * job.n; std::fill(slab, slab + static_cast<size_t>(job.h) * job.n, 0.0f); }
    if (job.n == 1 && !job.write_down) { decode_up_group(job, group); return; }
#if HFFN_HAS_AVX2
    // Up to 96 tokens the batch of a hidden row stays in ymm registers, for every N.
    if (hidden_group_lanes_dispatch(job, group)) return;
#else
    if (job.begin == 0 && job.end == 8 && job.n == 8) { hidden_group_fixed<8, 8>(job, group); return; }
    if (job.begin == 0 && job.end == 32 && job.n == 32) { hidden_group_slot_inner_fixed<32, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 64 && job.n == 64) { hidden_group_slot_inner_fixed<64, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 96 && job.n == 96) { hidden_group_slot_inner_fixed<96, 16>(job, group); return; }
#endif
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        const int down_start = job.write_down ? job.down_offsets[hidden] : 0;
        const int down_end = job.write_down ? job.down_offsets[hidden + 1] : 0;
#if HFFN_HAS_AVX2
        // Generic batch sizes: 8 token lanes per chunk, masked tail (never reads
        // past the [feature, N] row); compact slots keep their order per token.
        for (int token = job.begin; token < job.end; token += 8) {
            const __m256i mask = lane_mask(job.end - token);
            __m256 value = _mm256_set1_ps(job.b_up[hidden]);
            for (int slot = up_start; slot < up_end; ++slot)
                value = _mm256_fmadd_ps(_mm256_maskload_ps(job.x + static_cast<size_t>(job.up_idx[slot]) * job.n + token, mask), _mm256_set1_ps(job.up_val[slot]), value);
            float* active = job.activation + static_cast<size_t>(hidden) * job.n + token;
            if (job.z != job.activation) _mm256_maskstore_ps(job.z + static_cast<size_t>(hidden) * job.n + token, mask, value);
            _mm256_maskstore_ps(active, mask, gelu_lanes(value));
        }
#else
        for (int token = job.begin; token < job.end; ++token) {
            float value = job.b_up[hidden];
            for (int slot = up_start; slot < up_end; ++slot) value += job.up_val[slot] * job.x[static_cast<size_t>(job.up_idx[slot]) * job.n + token];
            job.z[static_cast<size_t>(hidden) * job.n + token] = value;
            job.activation[static_cast<size_t>(hidden) * job.n + token] = gelu_none_scalar(value);
        }
#endif
        if (job.write_down) {
            const float* active = job.activation + static_cast<size_t>(hidden) * job.n;
            for (int slot = down_start; slot < down_end; ++slot) {
                float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * job.n;
#if HFFN_HAS_AVX2
                const __m256 weight = _mm256_set1_ps(job.down_val[slot]);
                for (int token = job.begin; token < job.end; token += 8) {
                    const __m256i mask = lane_mask(job.end - token);
                    _mm256_maskstore_ps(output + token, mask, _mm256_fmadd_ps(_mm256_maskload_ps(active + token, mask), weight, _mm256_maskload_ps(output + token, mask)));
                }
#else
                for (int token = job.begin; token < job.end; ++token) output[token] += active[token] * job.down_val[slot];
#endif
            }
        }
    }
}

#if HFFN_HAS_AVX2
// Generic batch sizes up to 96 tokens on the reverse slot schedule: each slot
// read once for the whole batch, kChunks ymm accumulators per output row, plain
// loads for full chunks and a mask only for a partial last chunk.
template <int kChunks, bool kPartial, bool kSlots>
inline void output_group_lanes(const OutputJob& job, int group) {
    const int n = job.n;
    const __m256i tail = lane_mask(n - 8 * (kChunks - 1));
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        __m256 value[kChunks];
        for (int chunk = 0; chunk < kChunks; ++chunk) value[chunk] = _mm256_set1_ps(job.b_down[output]);
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * n;
            const __m256 weight = _mm256_set1_ps(job.weights[kSlots ? job.slots[slot_index] : slot_index]);
            for (int chunk = 0; chunk < kChunks; ++chunk)
                value[chunk] = _mm256_fmadd_ps(chunk + 1 < kChunks || !kPartial ? _mm256_loadu_ps(active + 8 * chunk) : _mm256_maskload_ps(active + 8 * chunk, tail), weight, value[chunk]);
        }
        float* destination = job.y_t + static_cast<size_t>(output) * n;
        for (int chunk = 0; chunk < kChunks; ++chunk) {
            if (chunk + 1 < kChunks || !kPartial) _mm256_storeu_ps(destination + 8 * chunk, value[chunk]);
            else _mm256_maskstore_ps(destination + 8 * chunk, tail, value[chunk]);
        }
    }
}

template <bool kSlots>
inline bool output_group_lanes_dispatch(const OutputJob& job, int group) {
    if (job.n > 96) return false;
    switch ((job.n + 7) / 8) {
        case 1: if (job.n % 8) output_group_lanes<1, true, kSlots>(job, group); else output_group_lanes<1, false, kSlots>(job, group); return true;
        case 2: if (job.n % 8) output_group_lanes<2, true, kSlots>(job, group); else output_group_lanes<2, false, kSlots>(job, group); return true;
        case 3: if (job.n % 8) output_group_lanes<3, true, kSlots>(job, group); else output_group_lanes<3, false, kSlots>(job, group); return true;
        case 4: if (job.n % 8) output_group_lanes<4, true, kSlots>(job, group); else output_group_lanes<4, false, kSlots>(job, group); return true;
        case 5: if (job.n % 8) output_group_lanes<5, true, kSlots>(job, group); else output_group_lanes<5, false, kSlots>(job, group); return true;
        case 6: if (job.n % 8) output_group_lanes<6, true, kSlots>(job, group); else output_group_lanes<6, false, kSlots>(job, group); return true;
        case 7: if (job.n % 8) output_group_lanes<7, true, kSlots>(job, group); else output_group_lanes<7, false, kSlots>(job, group); return true;
        case 8: if (job.n % 8) output_group_lanes<8, true, kSlots>(job, group); else output_group_lanes<8, false, kSlots>(job, group); return true;
        case 9: if (job.n % 8) output_group_lanes<9, true, kSlots>(job, group); else output_group_lanes<9, false, kSlots>(job, group); return true;
        case 10: if (job.n % 8) output_group_lanes<10, true, kSlots>(job, group); else output_group_lanes<10, false, kSlots>(job, group); return true;
        case 11: if (job.n % 8) output_group_lanes<11, true, kSlots>(job, group); else output_group_lanes<11, false, kSlots>(job, group); return true;
        case 12: if (job.n % 8) output_group_lanes<12, true, kSlots>(job, group); else output_group_lanes<12, false, kSlots>(job, group); return true;
        default: return false;
    }
}
#endif

template <bool kSlots>
inline void output_group(const OutputJob& job, int group) {
#if HFFN_HAS_AVX2
    if (output_group_lanes_dispatch<kSlots>(job, group)) return;
#endif
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        float* destination = job.y_t + static_cast<size_t>(output) * job.n;
#if HFFN_HAS_AVX2
        // Generic batch sizes: 8 token lanes per chunk held in a register across
        // the reverse slot schedule, masked tail; slot order unchanged per token.
        for (int token = 0; token < job.n; token += 8) {
            const __m256i mask = lane_mask(job.n - token);
            __m256 value = _mm256_set1_ps(job.b_down[output]);
            for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index)
                value = _mm256_fmadd_ps(_mm256_maskload_ps(job.activation + static_cast<size_t>(job.hidden[slot_index]) * job.n + token, mask), _mm256_set1_ps(job.weights[kSlots ? job.slots[slot_index] : slot_index]), value);
            _mm256_maskstore_ps(destination + token, mask, value);
        }
#else
        for (int token = 0; token < job.n; ++token) destination[token] = job.b_down[output];
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * job.n;
            const float weight = job.weights[kSlots ? job.slots[slot_index] : slot_index];
            for (int token = 0; token < job.n; ++token) destination[token] += active[token] * weight;
        }
#endif
    }
}

// Decode down projection per output feature, on the reverse schedule: each output is the
// sum of its own edges (four partial sums, slot s in partial s % 4, as the up projection),
// written once by one thread. No private buffer, no reduction: the result does not depend on T.
__attribute__((noinline, optimize("no-tree-vectorize"))) void decode_out_group(const OutputJob& job, int group) {
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        const int end = job.offsets[output + 1];
        int slot = job.offsets[output];
        float partial0 = 0.0f, partial1 = 0.0f, partial2 = 0.0f, partial3 = 0.0f;
        for (; slot + 4 <= end; slot += 4) {
            partial0 += job.weights[slot] * job.activation[job.hidden[slot]];
            partial1 += job.weights[slot + 1] * job.activation[job.hidden[slot + 1]];
            partial2 += job.weights[slot + 2] * job.activation[job.hidden[slot + 2]];
            partial3 += job.weights[slot + 3] * job.activation[job.hidden[slot + 3]];
        }
        for (; slot < end; ++slot) partial0 += job.weights[slot] * job.activation[job.hidden[slot]];
        job.y_t[output] = job.b_down[output] + ((partial0 + partial1) + (partial2 + partial3));
    }
}

#ifndef HFFN_GROUPS_OPENMP
// Persistent pool: for a call on T groups, the calling thread runs group 0 and
// worker g runs group g (g = 1..T-1). Workers are created the first time a call
// needs them and kept; workers beyond T skip the call. Waits spin for at most
// HFFN_POOL_SPIN_NS on an atomic before falling back to the condition variable:
// a condition-variable round trip costs 40-50 us on the cloud VM (90-125 us after
// an idle period), the bounded spin about 2 us when calls follow each other.
// Workers never spin longer than the bound, so an idle runtime does not keep the
// CPUs busy. One call at a time per process.
#ifndef HFFN_POOL_SPIN_NS
#define HFFN_POOL_SPIN_NS 100000
#endif
inline void spin_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}
#endif
// y_t[o, :] = b_down[o] + y_local[0, o, :] + ... + y_local[T-1, o, :], in group order, for the
// output features [h p / parts, h (p + 1) / parts): each feature is reduced by one thread.
inline void reduce_features(const FeatureJob& job, int part) {
    const float* y_local = job.source; float* y_t = job.destination; const int n = job.n, h = job.h;
    const int begin = static_cast<int>(static_cast<int64_t>(h) * part / job.parts), end = static_cast<int>(static_cast<int64_t>(h) * (part + 1) / job.parts);
    for (int output = begin; output < end; ++output) {
        float* destination = y_t + static_cast<size_t>(output) * n;
        const float* group0 = y_local + static_cast<size_t>(output) * n;
        for (int token = 0; token < n; ++token) destination[token] = job.bias[output] + group0[token];
        for (int group = 1; group < job.parts; ++group) {
            const float* partial = y_local + (static_cast<size_t>(group) * h + output) * n;
            for (int token = 0; token < n; ++token) destination[token] += partial[token];
        }
    }
}
enum class Kind { Hidden, Output, OutputSlots, DecodeOut, TransposeIn, Reduce };
inline void dispatch(Kind kind, const HiddenJob& hidden_job, const OutputJob& output_job, const FeatureJob& feature_job, int group) {
    if (kind == Kind::Hidden) hidden_group(hidden_job, group);
    else if (kind == Kind::Output || kind == Kind::OutputSlots) {
        if (kind == Kind::Output) output_group<false>(output_job, group); else output_group<true>(output_job, group);
        if (output_job.y_nt != nullptr) t_to_nt_features(output_job.y_t, output_job.y_nt, output_job.n, output_job.h, output_job.groups[group], output_job.groups[group + 1]);
    }
    else if (kind == Kind::DecodeOut) decode_out_group(output_job, group);
    else if (kind == Kind::TransposeIn) nt_to_t_features(feature_job.source, feature_job.destination, feature_job.n, feature_job.h, feature_job.groups[group], feature_job.groups[group + 1]);
    else reduce_features(feature_job, group);
}
#ifndef HFFN_GROUPS_OPENMP
class GroupPool {
public:
    ~GroupPool() { { std::lock_guard<std::mutex> lock(mutex); stop.store(true); generation.fetch_add(kActiveSpan, std::memory_order_release); } start.notify_all(); for (auto& worker : workers) if (worker.joinable()) worker.join(); }
    void run(Kind next_kind, const HiddenJob& hidden_job, const OutputJob& output_job, int threads, const FeatureJob& feature_job = FeatureJob{}) {
        if (threads == 1) { dispatch(next_kind, hidden_job, output_job, feature_job, 0); return; }
        const int expected = threads - 1;
        while (static_cast<int>(workers.size()) < expected) workers.emplace_back(&GroupPool::loop, this, static_cast<int>(workers.size()) + 1, generation.load());
        // The job is published by the release store of the generation word, which also
        // carries the number of groups of the call: workers read the job without the
        // mutex, and workers beyond T never read it. The caller rewrites the job only
        // after every active worker has finished (and copied it).
        hidden_current = hidden_job; output_current = output_job; feature_current = feature_job; kind = next_kind; done.store(0, std::memory_order_relaxed);
        const uint64_t word = (generation.load(std::memory_order_relaxed) / kActiveSpan + 1) * kActiveSpan + static_cast<uint64_t>(threads);
        { std::lock_guard<std::mutex> lock(mutex); generation.store(word, std::memory_order_release); }  // under the mutex: no lost wake-up
        start.notify_all();
        dispatch(next_kind, hidden_job, output_job, feature_job, 0);
        const int64_t spin_start = now_ns();
        while (done.load(std::memory_order_acquire) != expected) {
            if (now_ns() - spin_start > HFFN_POOL_SPIN_NS) { std::unique_lock<std::mutex> lock(mutex); finish.wait(lock, [&] { return done.load(std::memory_order_acquire) == expected; }); break; }
            spin_pause();
        }
    }
private:
    static constexpr uint64_t kActiveSpan = 128;  // generation word = sequence * kActiveSpan + T
    void loop(int group, uint64_t seen) {
        for (;;) {
            const int64_t spin_start = now_ns();
            uint64_t word = generation.load(std::memory_order_acquire);
            while (word == seen && now_ns() - spin_start < HFFN_POOL_SPIN_NS) { spin_pause(); word = generation.load(std::memory_order_acquire); }
            if (word == seen) { std::unique_lock<std::mutex> lock(mutex); start.wait(lock, [&] { return generation.load(std::memory_order_acquire) != seen; }); word = generation.load(std::memory_order_acquire); }
            if (stop.load(std::memory_order_acquire)) return;
            seen = word;
            const int local_active = static_cast<int>(word % kActiveSpan);
            if (group >= local_active) continue;
            const Kind local_kind = kind; const HiddenJob hidden_job = hidden_current; const OutputJob output_job = output_current; const FeatureJob feature_job = feature_current;
            dispatch(local_kind, hidden_job, output_job, feature_job, group);
            // Lock before notifying: the caller checks `done` under the mutex before it sleeps.
            if (done.fetch_add(1, std::memory_order_acq_rel) == local_active - 2) { std::lock_guard<std::mutex> lock(mutex); finish.notify_one(); }
        }
    }
    std::mutex mutex; std::condition_variable start, finish; std::vector<std::thread> workers;
    HiddenJob hidden_current{}; OutputJob output_current{}; FeatureJob feature_current{}; Kind kind = Kind::Hidden;
    std::atomic<uint64_t> generation{0}; std::atomic<int> done{0}; std::atomic<bool> stop{false};
};

inline GroupPool& pool() { static GroupPool instance; return instance; }
#define HFFN_RUN_GROUPS pool().run
#else
// One group per thread of an OpenMP team: the team of the process, which is PyTorch's when the
// kernel is loaded after it (same libgomp). The FFN then runs on the threads that ran the
// previous PyTorch operator instead of competing with them. Static schedule, one group per
// iteration: for a given T and partition the result does not depend on scheduling.
inline void run_groups(Kind kind, const HiddenJob& hidden_job, const OutputJob& output_job, int threads, const FeatureJob& feature_job = FeatureJob{}) {
    #pragma omp parallel for schedule(static, 1) num_threads(threads) if(threads > 1)
    for (int group = 0; group < threads; ++group) dispatch(kind, hidden_job, output_job, feature_job, group);
}
#define HFFN_RUN_GROUPS run_groups
#endif

inline void run_hidden(const HiddenJob& base, int threads) {
    const int tile = tile_for(base.n);
    // N = 1 runs its groups on the calling thread; from N = 2 the groups run in
    // parallel (a pool round trip costs ~2 us).
    if (base.n <= 1) { for (int group = 0; group < threads; ++group) hidden_group(base, group); return; }
    if (tile >= base.n) { HFFN_RUN_GROUPS(Kind::Hidden, base, OutputJob{}, threads); return; }
    for (int begin = 0; begin < base.n; begin += tile) { HiddenJob tile_job = base; tile_job.begin = begin; tile_job.end = std::min(begin + tile, base.n); HFFN_RUN_GROUPS(Kind::Hidden, tile_job, OutputJob{}, threads); }
}

}

// Decode (N = 1), no layout transposes: up + GELU per hidden group, then down per output
// feature on `output_groups`. It only needs the up arrays and the reverse schedule of the
// dual route, and its result does not depend on T.
extern "C" void hffn_compact_forward_decode_nt_f32(int threads, const float* x, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* b_down, const int32_t* down_output_offsets, const int16_t* down_hidden, const float* down_weights, const int32_t* output_groups, int h, int f, float* y, float* activation, int64_t* timing) {
    validate(threads, hidden_groups, f, 1, h, f); validate(threads, output_groups, h, 1, h, f); const int64_t begin = now_ns();
    HFFN_RUN_GROUPS(Kind::Hidden, HiddenJob{x, hidden_groups, b_up, up_offsets, up_idx, up_val, nullptr, nullptr, nullptr, 1, h, 0, 1, activation, activation, nullptr, false, true}, OutputJob{}, threads);
    HFFN_RUN_GROUPS(Kind::DecodeOut, HiddenJob{}, OutputJob{activation, down_output_offsets, down_hidden, down_weights, b_down, output_groups, 1, h, y}, threads);
    const int64_t end = now_ns(); timing[0] = 0; timing[1] = end - begin; timing[2] = 0;
}

// Hidden-owned route: up and down per hidden neuron into a private y_local[T, h, N] slab that
// each group clears itself, then a reduction in group order, split over the output features.
extern "C" void hffn_compact_forward_nt_f32(int threads, const float* x_nt, const int32_t* groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const int32_t* down_offsets, const int16_t* down_idx, const float* down_val, const float* b_down, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, float* y_local, int64_t* timing) {
    validate(threads, groups, f, n, h, f); const int64_t begin = now_ns(); nt_to_t(x_nt, x_t, n, h); const int64_t transposed = now_ns();
    // Hidden-owned down consumes each activation immediately.  Do not write
    // diagnostic [F,N] intermediates during inference.
    run_hidden(HiddenJob{x_t, groups, b_up, up_offsets, up_idx, up_val, down_offsets, down_idx, down_val, n, h, 0, n, z, activation, y_local, true, false}, threads);
    HFFN_RUN_GROUPS(Kind::Reduce, HiddenJob{}, OutputJob{}, threads, FeatureJob{y_local, y_t, nullptr, n, h, b_down, threads});
    const int64_t reduced = now_ns(); t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = reduced - transposed; timing[2] = end - reduced;
}

// Dual route: up per hidden neuron on `hidden_groups`, then down per output feature on
// `output_groups` (reverse schedule, no reduction). `down_weights` holds the down values in
// output order, next to `down_hidden`: the output phase reads both sequentially. The result
// does not depend on T.
namespace {
template <bool kSlots>
void forward_dual(int threads, const float* x_nt, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* b_down, const int32_t* down_output_offsets, const int16_t* down_hidden, const float* down_weights, const int32_t* down_slots, const int32_t* output_groups, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    validate(threads, hidden_groups, f, n, h, f); validate(threads, output_groups, h, n, h, f);
    const int64_t begin = now_ns();
    // The rows of x_t are those of y_t (shared buffer), last written by the output groups:
    // each group transposes the input rows it owns instead of the calling thread pulling
    // every row back. Data movement only: the result does not change.
    const bool by_groups = threads > 1 && static_cast<int64_t>(n) * h >= kGroupTransposeMin;
    if (by_groups) HFFN_RUN_GROUPS(Kind::TransposeIn, HiddenJob{}, OutputJob{}, threads, FeatureJob{x_nt, x_t, output_groups, n, h});
    else nt_to_t(x_nt, x_t, n, h);
    const int64_t transposed = now_ns();
    run_hidden(HiddenJob{x_t, hidden_groups, b_up, up_offsets, up_idx, up_val, nullptr, nullptr, nullptr, n, h, 0, n, z, activation, nullptr, false, true}, threads);
    const int64_t hidden_done = now_ns();
    HFFN_RUN_GROUPS(kSlots ? Kind::OutputSlots : Kind::Output, HiddenJob{}, OutputJob{activation, down_output_offsets, down_hidden, down_weights, b_down, output_groups, n, h, y_t, by_groups ? y_nt : nullptr, down_slots}, threads);
    const int64_t reduced = now_ns(); if (!by_groups) t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = hidden_done - transposed; timing[2] = reduced - hidden_done; timing[3] = end - reduced;
}
}

extern "C" void hffn_compact_forward_dual_nt_f32(int threads, const float* x_nt, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* b_down, const int32_t* down_output_offsets, const int16_t* down_hidden, const float* down_weights, const int32_t* output_groups, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    forward_dual<false>(threads, x_nt, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down, down_output_offsets, down_hidden, down_weights, nullptr, output_groups, n, h, f, x_t, y_nt, y_t, z, activation, timing);
}

// Dual route on the hidden-major down values: `down_val` in hidden-major order (the order a training
// runtime updates) and `down_slots[s]`, the hidden-major index of slot s of the reverse schedule. Same
// schedule, same slot order and same sums as `hffn_compact_forward_dual_nt_f32` fed `down_val[down_slots]`:
// the result is bitwise the same, without a copy of the down values in output order.
extern "C" void hffn_compact_forward_dual_slots_nt_f32(int threads, const float* x_nt, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* b_down, const int32_t* down_output_offsets, const int16_t* down_hidden, const float* down_val, const int32_t* down_slots, const int32_t* output_groups, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    if (down_slots == nullptr) throw std::runtime_error("dual slots entry: down_slots is null");
    forward_dual<true>(threads, x_nt, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down, down_output_offsets, down_hidden, down_val, down_slots, output_groups, n, h, f, x_t, y_nt, y_t, z, activation, timing);
}
