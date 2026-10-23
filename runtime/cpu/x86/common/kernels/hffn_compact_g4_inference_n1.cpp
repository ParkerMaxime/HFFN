// A training-only N = 1 entry, compiled as an isolated sidecar. The ordinary
// inference library is built from hffn_compact_g4_inference.cpp unchanged.
#include "hffn_compact_g4_inference.cpp"

namespace {
inline void hidden_group_slots_n1(const HiddenJob& job, int group) {
#if HFFN_HAS_AVX2
    hidden_group_lanes<1, true>(job, group);
#else
    for (int hidden = job.groups[group]; hidden < job.groups[group + 1]; ++hidden) {
        float value = job.b_up[hidden];
        for (int slot = job.up_offsets[hidden]; slot < job.up_offsets[hidden + 1]; ++slot)
            value += job.up_val[slot] * job.x[job.up_idx[slot]];
        job.z[hidden] = value;
        job.activation[hidden] = gelu_none_scalar(value);
    }
#endif
}

inline void run_hidden_slots_n1(const HiddenJob& job, int threads) {
    for (int group = 0; group < threads; ++group) hidden_group_slots_n1(job, group);
}

void forward_dual_slots_n1(int threads, const float* x_nt, const int32_t* hidden_groups, const float* b_up,
                           const int32_t* up_offsets, const int16_t* up_idx, const float* up_val,
                           const float* b_down, const int32_t* down_output_offsets, const int16_t* down_hidden,
                           const float* down_val, const int32_t* down_slots, const int32_t* output_groups,
                           int h, int f, float* x_t, float* y_nt, float* y_t, float* z, float* activation,
                           int64_t* timing) {
    validate(threads, hidden_groups, f, 1, h, f);
    validate(threads, output_groups, h, 1, h, f);
    const int64_t begin = now_ns();
    const bool by_groups = threads > 1 && static_cast<int64_t>(h) >= kGroupTransposeMin;
    if (by_groups) HFFN_RUN_GROUPS(Kind::TransposeIn, HiddenJob{}, OutputJob{}, threads,
                                   FeatureJob{x_nt, x_t, output_groups, 1, h});
    else nt_to_t(x_nt, x_t, 1, h);
    const int64_t transposed = now_ns();
    run_hidden_slots_n1(HiddenJob{x_t, hidden_groups, b_up, up_offsets, up_idx, up_val, nullptr, nullptr, nullptr,
                                  1, h, 0, 1, z, activation, nullptr, false, true}, threads);
    const int64_t hidden_done = now_ns();
    HFFN_RUN_GROUPS(Kind::OutputSlots, HiddenJob{},
                    OutputJob{activation, down_output_offsets, down_hidden, down_val, b_down, output_groups,
                              1, h, y_t, by_groups ? y_nt : nullptr, down_slots}, threads);
    const int64_t reduced = now_ns();
    if (!by_groups) t_to_nt(y_t, y_nt, 1, h);
    const int64_t end = now_ns();
    timing[0] = transposed - begin;
    timing[1] = hidden_done - transposed;
    timing[2] = reduced - hidden_done;
    timing[3] = end - reduced;
}
}

extern "C" __attribute__((visibility("default"))) void hffn_compact_forward_dual_slots_n1_nt_f32(
        int threads, const float* x_nt, const int32_t* hidden_groups, const float* b_up,
        const int32_t* up_offsets, const int16_t* up_idx, const float* up_val, const float* b_down,
        const int32_t* down_output_offsets, const int16_t* down_hidden, const float* down_val,
        const int32_t* down_slots, const int32_t* output_groups, int n, int h, int f,
        float* x_t, float* y_nt, float* y_t, float* z, float* activation, int64_t* timing) {
    if (down_slots == nullptr) throw std::runtime_error("dual slots N = 1 entry: down_slots is null");
    if (n != 1) throw std::runtime_error("dual slots N = 1 entry requires n == 1");
    forward_dual_slots_n1(threads, x_nt, hidden_groups, b_up, up_offsets, up_idx, up_val, b_down,
                         down_output_offsets, down_hidden, down_val, down_slots, output_groups, h, f,
                         x_t, y_nt, y_t, z, activation, timing);
}
