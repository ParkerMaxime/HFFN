#pragma once

#include <cstdint>
#include <stdexcept>

// Canonical reverse-route slot codec. A route stores only a reference to the
// hidden-major slot p = hidden * degree + edge; values and gradients remain in
// their canonical compact arrays.
struct HffnRouteCodec {
  uint32_t hidden, degree, edge_bits;
  bool compact16;

  HffnRouteCodec(uint32_t h, uint32_t d)
      : hidden(h), degree(d), edge_bits(1), compact16(false) {
    if (!h || !d) throw std::invalid_argument("invalid HFFN route dimensions");
    while ((1u << edge_bits) < d) ++edge_bits;
    // A route stores (hidden, local-edge).  It is compact16 exactly when their
    // minimal bit fields fit, not only when the local edge fits a legacy 5-bit
    // layout.  For example H=1024,D=51 uses 10+6=16 bits.
    compact16 = edge_bits < 16u && h <= (1u << (16u - edge_bits));
  }

  uint32_t encode32(uint32_t h, uint32_t e) const {
    if (h >= hidden || e >= degree) throw std::out_of_range("HFFN route slot");
    return (h << edge_bits) | e;
  }
  uint16_t encode16(uint32_t h, uint32_t e) const {
    if (!compact16) throw std::logic_error("compact16 unavailable");
    return uint16_t((h << edge_bits) | e);
  }
  uint32_t decode16(uint16_t r) const {
    return uint32_t(r >> edge_bits) * degree +
           uint32_t(r & ((1u << edge_bits) - 1u));
  }
  uint32_t decode32(uint32_t r) const { return (r >> edge_bits) * degree + (r & ((1u << edge_bits) - 1u)); }
};
