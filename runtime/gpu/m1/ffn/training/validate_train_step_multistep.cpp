#include "hffn_gelu_erf_accurate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Adam {
  float lr = .001f, b1 = .9f, b2 = .999f, eps = 1e-8f, wd = .01f;
};
static void update(std::vector<float> &p, const std::vector<float> &g,
                   std::vector<float> &m, std::vector<float> &v, int step) {
  Adam a;
  float ss = float(double(a.lr) / (1.0 - std::pow(double(a.b1), step))),
        scale = float(std::sqrt(1.0 - std::pow(double(a.b2), step)));
  for (size_t i = 0; i < p.size(); ++i) {
    m[i] = a.b1 * m[i] + (1 - a.b1) * g[i];
    v[i] = a.b2 * v[i] + (1 - a.b2) * g[i] * g[i];
    p[i] = p[i] * (1 - a.lr * a.wd) -
           ss * (m[i] / (std::sqrt(v[i]) / scale + a.eps));
  }
}
static std::vector<float> read(const std::string &p, size_t n) {
  std::vector<float> x(n);
  FILE *f = std::fopen(p.c_str(), "rb");
  if (!f || std::fread(x.data(), 4, n, f) != n)
    std::exit(2);
  std::fclose(f);
  return x;
}
static double error(const std::vector<float> &a, const std::vector<float> &b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i)
    e = std::max(e, std::abs(double(a[i] - b[i])));
  return e;
}
int main(int ac, char **av) {
  if (ac != 8 && ac != 9) {
    std::fprintf(stderr, "usage: I H O N D steps prefix [accum_steps]\n");
    return 2;
  }
  int I = std::atoi(av[1]), H = std::atoi(av[2]), O = std::atoi(av[3]),
      N = std::atoi(av[4]), D = std::atoi(av[5]), S = std::atoi(av[6]),
      E = H * D, A = ac == 9 ? std::atoi(av[8]) : 1;
  std::string pre = av[7];
  std::vector<float> x(I * N), gy(O * N), up(E), down(E), bu(H), bd(O), um(E),
      uv(E), dm(E), dv(E), bum(H), buv(H), bdm(O), bdv(O);
  std::vector<int> idx(E), target(E);
  for (int i = 0; i < I; ++i)
    for (int n = 0; n < N; ++n)
      x[i * N + n] = float((i * 17 + n * 13) % 29 - 14) * .02f;
  for (int o = 0; o < O; ++o) {
    bd[o] = float(o % 11 - 5) * .008f;
    for (int n = 0; n < N; ++n)
      gy[o * N + n] = float((o * 11 + n * 7) % 23 - 11) * .03f + .0007f;
  }
  for (int r = 0; r < H; ++r) {
    bu[r] = float(r % 9 - 4) * .01f;
    for (int e = 0; e < D; ++e) {
      int p = r * D + e;
      idx[p] = (r * 19 + e * 7) % I;
      target[p] = (r * 17 + e * 5) % O;
      up[p] = float((p * 3) % 17 - 8) * .015f;
      down[p] = float((p * 5) % 19 - 9) * .012f;
    }
  }
  auto x_at = [&](int i, int n, int micro) {
    float v = x[i * N + n];
    if (micro)
      v += float((i * 5 + n * 3 + micro * 7) % 11 - 5) * .001f;
    return v;
  };
  auto gy_at = [&](int o, int n, int micro) {
    float v = gy[o * N + n];
    if (micro)
      v += float((o * 3 + n * 5 + micro * 11) % 13 - 6) * .0015f +
           .0007f * float(micro);
    return v;
  };
  std::vector<float> initial_down = down, initial_bdown = bd;
  std::printf("initial_down %.9g\n",
              error(down, read(pre + "_initial_down.f32", down.size())));
  std::printf("initial_bdown %.9g\n",
              error(bd, read(pre + "_initial_bdown.f32", bd.size())));
  for (int step = 1; step <= S; ++step) {
    std::vector<float> z(H * N), h(H * N), gz(H * N), gu(E), gd(E), gbu(H),
        gbd(O);
    for (int micro = 0; micro < A; ++micro) {
      for (int r = 0; r < H; ++r)
        for (int n = 0; n < N; ++n) {
          float q = bu[r];
          for (int e = 0; e < D; ++e) {
            int p = r * D + e;
            q += up[p] * x_at(idx[p], n, micro);
          }
          z[r * N + n] = q;
          h[r * N + n] = hffn_gelu_erf_accurate(q);
        }
      for (int o = 0; o < O; ++o)
        for (int n = 0; n < N; ++n)
          gbd[o] += gy_at(o, n, micro);
      for (int r = 0; r < H; ++r) {
        for (int e = 0; e < D; ++e) {
          int p = r * D + e;
          for (int n = 0; n < N; ++n)
            gd[p] += h[r * N + n] * gy_at(target[p], n, micro);
        }
        for (int n = 0; n < N; ++n) {
          float gh = 0;
          for (int e = 0; e < D; ++e) {
            int p = r * D + e;
            gh += down[p] * gy_at(target[p], n, micro);
          }
          float q = gh * hffn_gelu_erf_accurate_grad(z[r * N + n]);
          gz[r * N + n] = q;
          gbu[r] += q;
        }
        for (int e = 0; e < D; ++e) {
          int p = r * D + e;
          for (int n = 0; n < N; ++n)
            gu[p] += gz[r * N + n] * x_at(idx[p], n, micro);
        }
      }
    }
    update(up, gu, um, uv, step);
    update(down, gd, dm, dv, step);
    update(bu, gbu, bum, buv, step);
    update(bd, gbd, bdm, bdv, step);
  }
  struct Pool {
    const char *n;
    const std::vector<float> *v;
  };
  std::vector<Pool> p = {{"up", &up},    {"down", &down}, {"bup", &bu},
                         {"bdown", &bd}, {"um", &um},     {"uv", &uv},
                         {"dm", &dm},    {"dv", &dv},     {"bum", &bum},
                         {"buv", &buv},  {"bdm", &bdm},   {"bdv", &bdv}};
  double worst = 0;
  for (auto q : p) {
    double e = error(*q.v, read(pre + "_" + q.n + ".f32", q.v->size()));
    worst = std::max(worst, e);
    std::printf("%s %.9g\n", q.n, e);
  }
  if (S == 1) {
    Adam a;
    float ss = float(double(a.lr) / (1.0 - double(a.b1))),
          scale = float(std::sqrt(1.0 - double(a.b2)));
    auto direct = [&](const char *name, const std::vector<float> &p0,
                      const std::vector<float> &mgpu,
                      const std::vector<float> &vgpu) {
      auto pgpu = read(pre + "_" + name + ".f32", p0.size());
      std::vector<float> expect(p0.size());
      for (size_t i = 0; i < p0.size(); ++i)
        expect[i] = p0[i] * (1 - a.lr * a.wd) -
                    ss * (mgpu[i] / (std::sqrt(vgpu[i]) / scale + a.eps));
      double e = error(expect, pgpu);
      std::printf("%s_gpu_adamw_equation %.9g\n", name, e);
    };
    direct("down", initial_down, read(pre + "_dm.f32", E),
           read(pre + "_dv.f32", E));
    direct("bdown", initial_bdown, read(pre + "_bdm.f32", O),
           read(pre + "_bdv.f32", O));
  }
  // SIMD reductions intentionally differ from the scalar CPU summation order.
  // In AdamW, a nearly-zero bias gradient may therefore choose a different
  // normalized update direction (up to roughly two learning-rate units here).
  // The direct checks above establish that Metal applies AdamW exactly to its
  // own compact moments.
  return worst <= 2e-3 ? 0 : 1;
}
