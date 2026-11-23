// One compact slot update. Compile-time policy preserves each existing FP32
// realization and its exceptional-gradient contract during source unification.
// No layout or host ownership is encoded here; both ABIs call this function.
template<bool LerpMoment, bool SanitizeGradient>
inline void hffn_adamw_slot(
    device float& parameter, device float& gradient,
    device float& first, device float& second,
    float lr, float beta1, float beta2, float eps, float decay,
    float step_size, float beta2_scale) {
  float g = gradient;
  if (SanitizeGradient && !metal::isfinite(g)) g = 0.0f;
  float m;
  if (LerpMoment) m = first + (1.0f-beta1)*(g-first);
  else m = beta1*first + (1.0f-beta1)*g;
  float v;
  if (LerpMoment) {
    v = beta2*second;
    v += (1.0f-beta2)*g*g;
  } else v = beta2*second + (1.0f-beta2)*g*g;
  first = m;
  second = v;
  if (LerpMoment) {
    float p = parameter*(1.0f-lr*decay);
    float denom = metal::sqrt(v)/beta2_scale+eps;
    p -= step_size*(m/denom);
    parameter = p;
  } else parameter = parameter*(1.0f-lr*decay)-step_size*(m/(metal::sqrt(v)/beta2_scale+eps));
  gradient = 0.0f;
}
