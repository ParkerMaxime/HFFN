#include "hffn_route_codec.h"

#include <cstdio>

static void check(uint32_t h, uint32_t d) {
  HffnRouteCodec c(h, d);
  for (uint32_t r = 0; r < h; ++r) for (uint32_t e = 0; e < d; ++e) {
    const uint32_t p = r * d + e;
    if (c.decode32(c.encode32(r, e)) != p) std::abort();
    if (c.compact16 && c.decode16(c.encode16(r, e)) != p) std::abort();
  }
  std::printf("H=%u D=%u route=%s ok\n", h, d, c.compact16 ? "compact16" : "compact32");
}
int main() { check(2048, 32); check(2049, 32); check(512, 51); }
