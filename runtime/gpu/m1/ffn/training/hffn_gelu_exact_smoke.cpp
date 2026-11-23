#include "hffn_gelu_exact.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

int main() {
  for (float x : {-8.f, -2.f, -.1f, 0.f, .1f, 2.f, 8.f}) {
    const float eps = 1e-3f;
    const float finite_difference = (hffn_gelu_exact(x + eps) - hffn_gelu_exact(x - eps)) / (2 * eps);
    if (std::abs(finite_difference - hffn_gelu_exact_grad(x)) > 2e-4f) return 1;
  }
  std::puts("exact GELU and derivative ok");
}
