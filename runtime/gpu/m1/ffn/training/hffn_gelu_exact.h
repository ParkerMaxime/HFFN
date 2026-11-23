#pragma once

#include <cmath>

// Exact GELU, not the tanh approximation. The Metal counterpart uses the same
// equations on float4 vectors: erf and exp are elementwise in MSL.
inline float hffn_gelu_exact(float x) {
  constexpr float inv_sqrt2 = 0.7071067811865475f;
  return 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
}
inline float hffn_gelu_exact_grad(float x) {
  constexpr float inv_sqrt2 = 0.7071067811865475f;
  constexpr float inv_sqrt_2pi = 0.3989422804014327f;
  return 0.5f * (1.0f + std::erf(x * inv_sqrt2)) + x * inv_sqrt_2pi * std::exp(-0.5f * x * x);
}
