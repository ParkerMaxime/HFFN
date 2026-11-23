#pragma once

#include <cmath>

// Abramowitz-Stegun 7.1.26. This follows the erf GELU definition, unlike the
// tanh GELU approximation. Its erf absolute error is bounded near 1.5e-7.
inline float hffn_erf_accurate(float x) {
  const float ax = std::abs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
  const float y = 1.0f - poly * std::exp(-ax * ax);
  return x < 0.0f ? -y : y;
}
inline float hffn_gelu_erf_accurate(float x) {
  return 0.5f * x * (1.0f + hffn_erf_accurate(x * 0.7071067811865475f));
}
inline float hffn_gelu_erf_accurate_grad(float x) {
  return 0.5f * (1.0f + hffn_erf_accurate(x * 0.7071067811865475f)) + x * 0.3989422804014327f * std::exp(-0.5f * x * x);
}
