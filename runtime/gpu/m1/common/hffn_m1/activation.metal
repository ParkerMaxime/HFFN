// BEGIN erf
inline float erf_accurate(float x) {
  float ax=metal::abs(x),t=1.0f/(1.0f+0.3275911f*ax);
  float p=(((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t);
  float y=1.0f-p*metal::exp(-ax*ax);
  return x<0.0f?-y:y;
}
// END erf
// BEGIN gelu
inline float gelu(float x) { return 0.5f*x*(1.0f+erf_accurate(x*0.7071067811865475f)); }
// END gelu
// BEGIN gradient
inline float gelu_grad(float x) {
  return 0.5f*(1.0f+erf_accurate(x*0.7071067811865475f))+x*0.3989422804014327f*metal::exp(-0.5f*x*x);
}
// END gradient
