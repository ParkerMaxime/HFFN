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

// Fixed compact paths preserve the group-owned full-batch traversal.
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
// their original order.
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

// A 128-token tile of a call of n tokens (row stride n, first token job.begin), same
// structure and same slot order as hidden_group_slot_inner_fixed.
template <int kTileWidth, int kTile>
inline void hidden_group_slot_inner_tile(const HiddenJob& job, int group) {
    const size_t stride = static_cast<size_t>(job.n);
    const size_t first = static_cast<size_t>(job.begin);
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
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
inline void hidden_group(const HiddenJob& job, int group) {
    if (job.begin == 0 && job.end == 8 && job.n == 8) { hidden_group_fixed<8, 8>(job, group); return; }
    if (job.begin == 0 && job.end == 32 && job.n == 32) { hidden_group_slot_inner_fixed<32, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 64 && job.n == 64) { hidden_group_slot_inner_fixed<64, 32>(job, group); return; }
    if (job.begin == 0 && job.end == 96 && job.n == 96) { hidden_group_slot_inner_fixed<96, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 192 && job.n == 192) { hidden_group_slot_inner_fixed<192, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 256 && job.n == 256) { hidden_group_slot_inner_fixed<256, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 384 && job.n == 384) { hidden_group_slot_inner_fixed<384, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 768 && job.n == 768) { hidden_group_slot_inner_fixed<768, 16>(job, group); return; }
    if (job.begin == 0 && job.end == 1536 && job.n == 1536) { hidden_group_slot_inner_fixed<1536, 16>(job, group); return; }
    if (job.end - job.begin == 128) { hidden_group_slot_inner_tile<128, 16>(job, group); return; }
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        const int up_start = job.up_offsets[hidden], up_end = job.up_offsets[hidden + 1];
        const int down_start = job.write_down ? job.down_offsets[hidden] : 0;
        const int down_end = job.write_down ? job.down_offsets[hidden + 1] : 0;
        for (int token = job.begin; token < job.end; ++token) {
            float value = job.b_up[hidden];
            for (int slot = up_start; slot < up_end; ++slot) value += job.up_val[slot] * job.x[static_cast<size_t>(job.up_idx[slot]) * job.n + token];
            job.z[static_cast<size_t>(hidden) * job.n + token] = value;
            job.activation[static_cast<size_t>(hidden) * job.n + token] = gelu_none_scalar(value);
        }
        if (job.write_down) {
            const float* active = job.activation + static_cast<size_t>(hidden) * job.n;
            for (int slot = down_start; slot < down_end; ++slot) {
                float* output = job.y_local + (static_cast<size_t>(group) * job.h + job.down_idx[slot]) * job.n;
                for (int token = job.begin; token < job.end; ++token) output[token] += active[token] * job.down_val[slot];
            }
        }
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

inline void output_group(const OutputJob& job, int group) {
    if (job.n == 8) { output_group_fixed<8>(job, group); return; }
    if (job.n == 32) { output_group_fixed<32>(job, group); return; }
    if (job.n == 64) { output_group_neon_fixed<64>(job, group); return; }
    for (int output = job.groups[group]; output < job.groups[group + 1]; ++output) {
        float* destination = job.y_t + static_cast<size_t>(output) * job.n;
        for (int token = 0; token < job.n; ++token) destination[token] = job.b_down[output];
        for (int slot_index = job.offsets[output]; slot_index < job.offsets[output + 1]; ++slot_index) {
            const float* active = job.activation + static_cast<size_t>(job.hidden[slot_index]) * job.n;
            const float weight = job.down_val[job.slots[slot_index]];
            for (int token = 0; token < job.n; ++token) destination[token] += active[token] * weight;
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

inline void run_hidden(const HiddenJob& base) {
    const int tile = tile_for(base.n);
    if (base.n <= 2) { for (int group = 0; group < kGroups; ++group) hidden_group(base, group); return; }
    if (tile >= base.n) { g4_pool().hidden(base); return; }
    for (int begin = 0; begin < base.n; begin += tile) { HiddenJob tile_job = base; tile_job.begin = begin; tile_job.end = std::min(begin + tile, base.n); g4_pool().hidden(tile_job); }
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

extern "C" void hffn_compact_g4_forward_dual_nt_f32(const float* x_nt, const int32_t* hidden_groups, const float* b_up, const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* down_val, const float* b_down, const int32_t* down_output_offsets, const int32_t* down_slots, const int16_t* down_hidden, const int32_t* output_groups, int n, int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    validate(hidden_groups, n, h, f); const int64_t begin = now_ns(); nt_to_t(x_nt, x_t, n, h); const int64_t transposed = now_ns();
    run_hidden(HiddenJob{x_t, hidden_groups, b_up, up_offsets, up_idx, up_val, nullptr, nullptr, down_val, n, h, 0, n, z, activation, nullptr, false, true});
    const int64_t hidden_done = now_ns(); g4_pool().output(OutputJob{activation, down_output_offsets, down_slots, down_hidden, down_val, b_down, output_groups, n, h, y_t});
    const int64_t reduced = now_ns(); t_to_nt(y_t, y_nt, n, h); const int64_t end = now_ns(); timing[0] = transposed - begin; timing[1] = hidden_done - transposed; timing[2] = reduced - hidden_done; timing[3] = end - reduced;
}
