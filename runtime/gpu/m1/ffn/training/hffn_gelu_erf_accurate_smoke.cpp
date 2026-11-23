#include "hffn_gelu_erf_accurate.h"
#include "hffn_gelu_exact.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
  float max_value_error = 0.0f, max_grad_error = 0.0f;
  for (int i = -12000; i <= 12000; ++i) {
    const float x = float(i) * 0.001f;
    max_value_error = std::max(max_value_error, std::abs(hffn_gelu_erf_accurate(x) - hffn_gelu_exact(x)));
    max_grad_error = std::max(max_grad_error, std::abs(hffn_gelu_erf_accurate_grad(x) - hffn_gelu_exact_grad(x)));
  }
  std::printf("gelu_erf_accurate value=%g grad=%g\n", max_value_error, max_grad_error);
  return (max_value_error <= 5e-7f && max_grad_error <= 3e-7f) ? 0 : 1;
}
