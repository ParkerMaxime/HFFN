#include <torch/extension.h>
#include <cstdint>

extern "C" void hffn_compact_g4_forward_nt_f32(const float*, const int32_t*, const float*, const int32_t*, const int16_t*, const float*, const int32_t*, const int16_t*, const float*, const float*, int, int, int, float*, float*, float*, float*, float*, float*, int64_t*);
extern "C" void hffn_compact_g4_forward_decode_nt_f32(const float*, const int32_t*, const float*, const int32_t*, const int16_t*, const float*, const int32_t*, const int16_t*, const float*, const float*, int, int, float*, float*, float*, int64_t*);
extern "C" void hffn_compact_g4_forward_decode_nt_groups_f32(const float*, const int32_t*, const float*, const int32_t*, const int16_t*, const float*, const int32_t*, const int16_t*, const float*, const float*, int, int, float*, float*, float*, float*, int64_t*);
extern "C" void hffn_compact_g4_forward_dual_nt_f32(const float*, const int32_t*, const float*, const int32_t*, const int16_t*, const float*, const float*, const float*, const int32_t*, const int32_t*, const int16_t*, const int32_t*, int, int, int, float*, float*, float*, float*, float*, int64_t*);

namespace {
void check(const torch::Tensor& tensor, torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == dtype && tensor.is_contiguous(), name, " must be contiguous CPU tensor with expected dtype");
}
void forward_nt(torch::Tensor x, torch::Tensor groups, torch::Tensor bup, torch::Tensor uoff, torch::Tensor uidx, torch::Tensor up, torch::Tensor doff, torch::Tensor didx, torch::Tensor down, torch::Tensor bdown, torch::Tensor xt, torch::Tensor y, torch::Tensor yt, torch::Tensor z, torch::Tensor act, torch::Tensor local, torch::Tensor timing) {
  for (auto tensor : {&x, &bup, &up, &down, &bdown, &xt, &y, &yt, &z, &act, &local}) check(*tensor, torch::kFloat32, "float");
  for (auto tensor : {&groups, &uoff, &doff}) check(*tensor, torch::kInt32, "offset");
  for (auto tensor : {&uidx, &didx}) check(*tensor, torch::kInt16, "index");
  check(timing, torch::kInt64, "timing");
  TORCH_CHECK(groups.numel() == 5 && uoff.numel() == z.size(0) + 1 && doff.numel() == z.size(0) + 1 && x.dim() == 2 && y.sizes() == x.sizes() && x.size(1) == bdown.numel() && timing.numel() == 3, "invalid compact inference forward shapes");
  hffn_compact_g4_forward_nt_f32(x.data_ptr<float>(), groups.data_ptr<int32_t>(), bup.data_ptr<float>(), uoff.data_ptr<int32_t>(), uidx.data_ptr<int16_t>(), up.data_ptr<float>(), doff.data_ptr<int32_t>(), didx.data_ptr<int16_t>(), down.data_ptr<float>(), bdown.data_ptr<float>(), static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<int>(z.size(0)), xt.data_ptr<float>(), y.data_ptr<float>(), yt.data_ptr<float>(), z.data_ptr<float>(), act.data_ptr<float>(), local.data_ptr<float>(), timing.data_ptr<int64_t>());
}
void forward_decode_nt(torch::Tensor x, torch::Tensor groups, torch::Tensor bup, torch::Tensor uoff, torch::Tensor uidx, torch::Tensor up, torch::Tensor doff, torch::Tensor didx, torch::Tensor down, torch::Tensor bdown, torch::Tensor y, torch::Tensor z, torch::Tensor act, torch::Tensor timing) {
  for (auto tensor : {&x, &bup, &up, &down, &bdown, &y, &z, &act}) check(*tensor, torch::kFloat32, "decode float");
  for (auto tensor : {&groups, &uoff, &doff}) check(*tensor, torch::kInt32, "decode offset");
  for (auto tensor : {&uidx, &didx}) check(*tensor, torch::kInt16, "decode index");
  check(timing, torch::kInt64, "decode timing");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && y.sizes() == x.sizes() && groups.numel() == 5 && uoff.numel() == z.numel() + 1 && doff.numel() == z.numel() + 1 && timing.numel() == 3, "invalid compact inference decode shapes");
  hffn_compact_g4_forward_decode_nt_f32(x.data_ptr<float>(), groups.data_ptr<int32_t>(), bup.data_ptr<float>(), uoff.data_ptr<int32_t>(), uidx.data_ptr<int16_t>(), up.data_ptr<float>(), doff.data_ptr<int32_t>(), didx.data_ptr<int16_t>(), down.data_ptr<float>(), bdown.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(z.numel()), y.data_ptr<float>(), z.data_ptr<float>(), act.data_ptr<float>(), timing.data_ptr<int64_t>());
}
// Same arguments as forward_decode_nt, plus `local` (the [4, H, 1] y_local of the (layer, 1) workspace) before `timing`.
void forward_decode_nt_groups(torch::Tensor x, torch::Tensor groups, torch::Tensor bup, torch::Tensor uoff, torch::Tensor uidx, torch::Tensor up, torch::Tensor doff, torch::Tensor didx, torch::Tensor down, torch::Tensor bdown, torch::Tensor y, torch::Tensor z, torch::Tensor act, torch::Tensor local, torch::Tensor timing) {
  for (auto tensor : {&x, &bup, &up, &down, &bdown, &y, &z, &act, &local}) check(*tensor, torch::kFloat32, "decode groups float");
  for (auto tensor : {&groups, &uoff, &doff}) check(*tensor, torch::kInt32, "decode groups offset");
  for (auto tensor : {&uidx, &didx}) check(*tensor, torch::kInt16, "decode groups index");
  check(timing, torch::kInt64, "decode groups timing");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && y.sizes() == x.sizes() && groups.numel() == 5 && uoff.numel() == z.numel() + 1 && doff.numel() == z.numel() + 1 && act.numel() == z.numel() && bdown.numel() == x.size(1) && timing.numel() == 3, "invalid compact inference decode groups shapes");
  TORCH_CHECK(local.numel() == 4 * x.size(1), "y_local must hold 4 * H float32 values");
  hffn_compact_g4_forward_decode_nt_groups_f32(x.data_ptr<float>(), groups.data_ptr<int32_t>(), bup.data_ptr<float>(), uoff.data_ptr<int32_t>(), uidx.data_ptr<int16_t>(), up.data_ptr<float>(), doff.data_ptr<int32_t>(), didx.data_ptr<int16_t>(), down.data_ptr<float>(), bdown.data_ptr<float>(), static_cast<int>(x.size(1)), static_cast<int>(z.numel()), y.data_ptr<float>(), z.data_ptr<float>(), act.data_ptr<float>(), local.data_ptr<float>(), timing.data_ptr<int64_t>());
}
void forward_dual_nt(torch::Tensor x, torch::Tensor groups, torch::Tensor bup, torch::Tensor uoff, torch::Tensor uidx, torch::Tensor up, torch::Tensor down, torch::Tensor bdown, torch::Tensor doff, torch::Tensor slots, torch::Tensor hidden, torch::Tensor output_groups, torch::Tensor xt, torch::Tensor y, torch::Tensor yt, torch::Tensor z, torch::Tensor act, torch::Tensor timing) {
  for (auto tensor : {&x, &bup, &up, &down, &bdown, &xt, &y, &yt, &z, &act}) check(*tensor, torch::kFloat32, "dual float");
  for (auto tensor : {&groups, &uoff, &doff, &slots, &output_groups}) check(*tensor, torch::kInt32, "dual offset");
  for (auto tensor : {&uidx, &hidden}) check(*tensor, torch::kInt16, "dual index");
  check(timing, torch::kInt64, "dual timing");
  TORCH_CHECK(groups.numel() == 5 && output_groups.numel() == 5 && uoff.numel() == z.size(0) + 1 && doff.numel() == x.size(1) + 1 && slots.numel() == down.numel() && hidden.numel() == down.numel() && y.sizes() == x.sizes() && timing.numel() == 4, "invalid compact inference dual forward shapes");
  hffn_compact_g4_forward_dual_nt_f32(x.data_ptr<float>(), groups.data_ptr<int32_t>(), bup.data_ptr<float>(), uoff.data_ptr<int32_t>(), uidx.data_ptr<int16_t>(), up.data_ptr<float>(), down.data_ptr<float>(), bdown.data_ptr<float>(), doff.data_ptr<int32_t>(), slots.data_ptr<int32_t>(), hidden.data_ptr<int16_t>(), output_groups.data_ptr<int32_t>(), static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), static_cast<int>(z.size(0)), xt.data_ptr<float>(), y.data_ptr<float>(), yt.data_ptr<float>(), z.data_ptr<float>(), act.data_ptr<float>(), timing.data_ptr<int64_t>());
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("forward_nt", &forward_nt); module.def("forward_decode_nt", &forward_decode_nt); module.def("forward_decode_nt_groups", &forward_decode_nt_groups);module.def("forward_dual_nt", &forward_dual_nt); }
