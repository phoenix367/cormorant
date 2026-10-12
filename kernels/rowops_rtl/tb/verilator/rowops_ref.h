// ---------------------------------------------------------------------------
// rowops_ref.h — the RowOpsKernel's oracle: a C++ port of the integer
// specifications of inference-scheduler/src/rowops_spec.py (the file that
// defines what the unit computes; its 1-LSB tests and model gates are in
// HOSTOPS_PLAN.md §5), applied to a job's register values on the testbench's
// DDR image.  Every output byte the job writes is produced as (address, bytes);
// the testbench compares them with the memory after the run and treats every
// other written byte as a stray write.
//
// Arithmetic: 64-bit integers where the specification's values fit, 128-bit
// for the norms' sums and products, exactly as the Python's integers.
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

namespace ro {

using i64  = int64_t;
using u64  = uint64_t;
using i128 = __int128;

enum Op : uint32_t { COPY = 1, ADDX = 2, DEQUANT = 3, MULACT = 4, NORM = 5, ROPE = 6, XPOSE = 7, TLOAD = 8 };
enum Kind : uint32_t { I16 = 0, I32 = 1, F32 = 2 };
enum Flag : uint32_t { F_B_EN = 1u << 12, F_P_BCAST = 1u << 13, F_P_KEEP = 1u << 14,
                       F_RELU = 1u << 15, F_NORM_LN = 1u << 16 };

constexpr int P1_BITS = 24, RM_BITS = 47, ROPE_BITS = 23, SILU_BITS = 20;
constexpr int G_BITS = 14, SIG_BITS = 24, SIG_STEP = 8, SIG_N = 4096;

struct Job {
  uint32_t op = 0, n = 0, c1 = 0, c2 = 0;
  u64      a = 0, b = 0, c = 0, p = 0;
  uint32_t a_s1 = 0, a_s2 = 0, b_s1 = 0, b_s2 = 0, c_s1 = 0, c_s2 = 0;
  u64      eps = 0;
  uint32_t aux = 0, bq_sh = 0, win_lo = 0, win_hi = 0xFFFFFFFFu;

  uint32_t code() const { return op & 15; }
  Kind ak() const { return Kind((op >> 4) & 3); }
  Kind bk() const { return Kind((op >> 6) & 3); }
  Kind ck() const { return Kind((op >> 8) & 3); }
  bool b_en() const { return op & F_B_EN; }
  bool p_bcast() const { return op & F_P_BCAST; }
  bool relu() const { return op & F_RELU; }
  bool ln() const { return op & F_NORM_LN; }
};

struct Param { int32_t g = 0; int32_t bq = 0; uint8_t sh_a = 0, sh_b = 0, sh_c = 0; };

// DDR access the oracle needs: bytes at an address (the testbench's sparse memory).
struct Mem {
  std::function<uint8_t(u64)> rd8;
  i64 rd16(u64 a) const { return (int16_t)(rd8(a) | (rd8(a + 1) << 8)); }
  i64 rd32(u64 a) const {
    return (int32_t)((uint32_t)rd8(a) | ((uint32_t)rd8(a + 1) << 8) | ((uint32_t)rd8(a + 2) << 16) |
                     ((uint32_t)rd8(a + 3) << 24));
  }
  i64 elem(u64 a, Kind k) const { return k == I16 ? rd16(a) : rd32(a); }
  static int bytes(Kind k) { return k == I16 ? 2 : 4; }
};

struct Write { u64 addr; std::vector<uint8_t> bytes; };

// ---- helpers (rowops_spec) ---------------------------------------------------------
inline i128 rne_shr(i128 x, int k) {
  if (k <= 0) return x << (-k);
  i128 q = x >> k;                     // floor (arithmetic shift)
  i128 r = x - (q << k);
  i128 half = (i128)1 << (k - 1);
  bool up = (r > half) || (r == half && (q & 1));
  return q + (up ? 1 : 0);
}
inline i64 sat16(i128 x) { return x > 32767 ? 32767 : x < -32768 ? -32768 : (i64)x; }
inline i64 sat32(i128 x) { return x > 2147483647 ? 2147483647 : x < -2147483648LL ? -2147483648LL : (i64)x; }

inline const std::vector<uint32_t>& rsqrt_table() {
  static std::vector<uint32_t> t;
  if (t.empty()) {
    t.resize(8192, 0);
    for (int i = 2048; i < 8192; i++) {
      double y = std::rint(std::ldexp(1.0, RM_BITS) / std::sqrt((i + 0.5) * 524288.0));
      t[i] = (uint32_t)std::min(y, 4294967295.0);
    }
  }
  return t;
}

inline int bit_length(i128 v) { int n = 0; while (v) { v >>= 1; n++; } return n; }

// 1/sqrt(D) ~= Rm * 2^-sh (rowops_spec.rsqrt_bf)
inline void rsqrt_bf(i128 D, u64& Rm, int& sh) {
  int e = bit_length(D);
  int d = e - 32;
  if (d & 1) d += 1;
  i128 Dm = d >= 0 ? (D >> d) : (D << (-d));
  u64 y0 = rsqrt_table()[(size_t)(Dm >> 19)];
  u64 t  = (u64)(((i128)Dm * (i128)y0) >> 32);
  u64 u  = (u64)(((i128)t * (i128)y0) >> 32);
  i128 y1 = (i128)y0 + ((((i128)y0) * ((i128)((i64)1 << 30) - (i128)u)) >> 31);
  if (y1 < ((i128)1 << 31)) y1 = (i128)1 << 31;
  if (y1 > ((i128)1 << 32) - 1) y1 = ((i128)1 << 32) - 1;
  Rm = (u64)y1;
  sh = RM_BITS + d / 2;
}

inline const std::vector<i64>& sigmoid_table() {
  static std::vector<i64> t;
  if (t.empty()) {
    t.resize(SIG_N + 1);
    for (int i = 0; i <= SIG_N; i++)
      t[i] = (i64)std::rint(std::ldexp(1.0, SIG_BITS) / (1.0 + std::exp(-(double)i / 256.0)));
  }
  return t;
}

// silu(g) at 2^-20 for g = gv * 2^-14 (rowops_spec.silu_q)
inline i64 silu_q(i64 gv) {
  const auto& T = sigmoid_table();
  i64 a = gv < 0 ? -gv : gv;
  bool big = a >= ((i64)16 << G_BITS);
  i64 ai = big ? ((i64)16 << G_BITS) - 1 : a;
  i64 idx = ai >> (G_BITS - SIG_STEP);
  i64 frac = ai & ((1 << (G_BITS - SIG_STEP)) - 1);
  i64 sig = T[idx] + (((T[idx + 1] - T[idx]) * frac) >> (G_BITS - SIG_STEP));
  if (big) sig = (i64)1 << SIG_BITS;
  if (gv < 0) sig = ((i64)1 << SIG_BITS) - sig;
  return (i64)rne_shr((i128)gv * sig, G_BITS + SIG_BITS - SILU_BITS);
}

inline uint32_t float_bits(i64 v, int sh_c) {   // float32(v * 2^-sh_c), IEEE round to nearest even
  if (v == 0) return 0;
  bool neg = v < 0;
  u64 mag = neg ? (u64)(-v) : (u64)v;
  int lz = 63 - __builtin_clzll(mag);
  u64 norm = lz >= 31 ? mag >> (lz - 31) : mag << (31 - lz);   // leading one at bit 31 (mag < 2^32 here)
  uint32_t m = (uint32_t)(norm >> 8) & 0x7FFFFF;
  uint32_t m24 = (uint32_t)(norm >> 8);                        // with the hidden bit
  bool up = (norm & 0x80) && ((norm & 0x7F) || (norm & 0x100));
  m24 += up;
  int exp = 127 + lz - sh_c + ((m24 >> 24) & 1);
  m = m24 & 0x7FFFFF;
  return ((uint32_t)neg << 31) | ((uint32_t)exp << 23) | m;
}

// ---- the parameter block -------------------------------------------------------------
inline Param read_param(const Mem& m, u64 p, uint32_t c, bool bcast) {
  u64 a = p + 16ull * (bcast ? 0 : c);
  Param q;
  q.g    = (int32_t)m.rd32(a);
  q.bq   = (int32_t)((int32_t)((uint32_t)m.rd32(a + 4) << 10) >> 10);   // the int22 in the int32, sign-extended
  q.sh_a = m.rd8(a + 8) & 63;
  q.sh_b = m.rd8(a + 9) & 63;
  q.sh_c = m.rd8(a + 10) & 63;
  return q;
}

inline void put(std::vector<uint8_t>& v, i64 x, Kind k) {
  int nb = Mem::bytes(k);
  for (int i = 0; i < nb; i++) v.push_back((uint8_t)((u64)x >> (8 * i)));
}

// the whole words of a row of n elements of kind k, lanes past n zero
inline Write row_write(u64 addr, const std::vector<i64>& vals, Kind k) {
  Write w{addr, {}};
  for (i64 x : vals) put(w.bytes, x, k);
  while (w.bytes.size() % 16) w.bytes.push_back(0);
  return w;
}

// ---- the job -------------------------------------------------------------------------
// The table image OP_TLOAD reads (ROWOPS_RTL_KERNEL.md §1.4): 4096 words of 16 bytes —
// words 0 .. 2047 two sigmoid entries {D[i], T[i]} (49 bits in each 64-bit half,
// i = 2w, 2w + 1), words 2048 .. 4095 four rsqrt entries (32 bits each, i = 4(w - 2048) ..).
inline std::vector<uint8_t> table_image() {
  std::vector<uint8_t> img(65536, 0);
  const auto& T = sigmoid_table();
  const auto& R = rsqrt_table();
  auto put64 = [&](size_t off, u64 v) { for (int b = 0; b < 8; b++) img[off + b] = (uint8_t)(v >> (8 * b)); };
  for (int i = 0; i < SIG_N; i++) {
    const u64 d = (u64)(T[i + 1] - T[i]) & 0xFFFFFF, t = (u64)T[i] & 0x1FFFFFF;
    put64((size_t)i * 8, (d << 25) | t);
  }
  for (int i = 0; i < 8192; i++) {
    const size_t off = 32768 + (size_t)i * 4;
    for (int b = 0; b < 4; b++) img[off + b] = (uint8_t)(R[i] >> (8 * b));
  }
  return img;
}

inline std::vector<Write> run(const Job& j, const Mem& m) {
  if (j.code() == TLOAD) return {};               // the tables: no DDR output
  std::vector<Write> out;
  const uint32_t op = j.code();
  const uint32_t n = j.n;
  if (n == 0 || j.c1 == 0 || j.c2 == 0 || n > 4096) return out;
  const Kind ak = j.ak(), bk = j.bk(), ck = j.ck();
  const bool ben = j.b_en(), bcast = j.p_bcast();
  std::vector<Param> P;
  auto param = [&](uint32_t c) -> const Param& {
    if (P.empty()) {
      P.resize(bcast ? 1 : n);
      for (uint32_t i = 0; i < P.size(); i++) P[i] = read_param(m, j.p, i, bcast);
    }
    return P[bcast ? 0 : c];
  };

  if (op == COPY || op == ADDX || op == DEQUANT || op == MULACT) {
    for (u64 i2 = 0; i2 < j.c2; i2++)
      for (u64 i1 = 0; i1 < j.c1; i1++) {
        const u64 ra = j.a + i2 * j.a_s2 + i1 * j.a_s1;
        const u64 rb = j.b + i2 * j.b_s2 + i1 * j.b_s1;
        const u64 rc = j.c + i2 * j.c_s2 + i1 * j.c_s1;
        const bool zero_a = (op == COPY || op == ADDX) && (i1 < j.win_lo || i1 >= j.win_hi);
        std::vector<i64> vals(n);
        for (uint32_t c = 0; c < n; c++) {
          const Param& q = param(c);
          i64 x = zero_a ? 0 : m.elem(ra + (u64)Mem::bytes(ak) * c, ak);
          // B contributes whenever B_EN is set (COPY is ADDX without B); DEQUANT never reads it
          i64 y = (ben && op != DEQUANT) ? m.elem(rb + (u64)Mem::bytes(bk) * c, bk) : 0;
          i64 r = 0;
          if (op == COPY || op == ADDX) {
            i128 s = ((i128)x << q.sh_a) + ((i128)y << q.sh_b) + ((i128)q.bq << j.bq_sh);
            i128 v = rne_shr(s, q.sh_c);
            r = ck == I16 ? sat16(v) : sat32(v);
            if (j.relu() && r < 0) r = 0;
          } else if (op == DEQUANT) {
            r = (i64)float_bits(x, q.sh_c);
          } else {                                          // MULACT
            i64 gv = x << q.sh_a;
            r = sat16(rne_shr((i128)silu_q(gv) * y, q.sh_c));
          }
          vals[c] = r;
        }
        out.push_back(row_write(rc, vals, ck));
      }
    return out;
  }

  if (op == NORM) {
    const bool ln = j.ln();
    for (u64 i2 = 0; i2 < j.c2; i2++)
      for (u64 i1 = 0; i1 < j.c1; i1++) {
        const u64 ra = j.a + i2 * j.a_s2 + i1 * j.a_s1;
        const u64 rc = j.c + i2 * j.c_s2 + i1 * j.c_s1;
        std::vector<i64> x(n);
        i128 S1 = 0, S2 = 0;
        for (uint32_t c = 0; c < n; c++) {
          x[c] = m.elem(ra + (u64)Mem::bytes(ak) * c, ak);
          S1 += x[c];
          S2 += (i128)x[c] * x[c];
        }
        i128 D = ln ? ((i128)n * S2 - S1 * S1 + (i128)(i64)j.eps) : (S2 + (i128)(i64)j.eps);
        if (D <= 0) D = 1;
        u64 Rm; int sh;
        rsqrt_bf(D, Rm, sh);
        std::vector<i64> vals(n);
        for (uint32_t c = 0; c < n; c++) {
          const Param& q = param(c);
          i128 A = ln ? ((i128)n * x[c] - S1) : (i128)x[c];
          i128 P1 = (A * (i128)Rm) >> (sh - P1_BITS);           // floor
          i128 s = P1 * (i128)q.g + ((i128)q.bq << j.bq_sh);
          vals[c] = sat16(rne_shr(s, q.sh_c));
        }
        out.push_back(row_write(rc, vals, I16));
      }
    return out;
  }

  if (op == ROPE) {
    const uint32_t HD = j.aux & 0xFFF, half = HD / 2;
    if (HD == 0 || (HD & 15) || HD > 128 || n % HD) return out;
    for (u64 i2 = 0; i2 < j.c2; i2++)
      for (u64 i1 = 0; i1 < j.c1; i1++) {
        const u64 ra = j.a + i2 * j.a_s2 + i1 * j.a_s1;
        const u64 rb = j.b + i2 * j.b_s2 + i1 * j.b_s1;
        const u64 rc = j.c + i2 * j.c_s2 + i1 * j.c_s1;
        std::vector<i64> x(n), vals(n);
        for (uint32_t c = 0; c < n; c++) x[c] = m.rd16(ra + 2ull * c);
        for (uint32_t c = 0; c < n; c++) {
          const Param& q = param(c);
          uint32_t d = c % HD;
          i64 cs = m.rd32(rb + 4ull * d), sn = m.rd32(rb + 4ull * (HD + d));
          i64 rot = d < half ? -x[c + half] : x[c - half];
          i128 s = (((i128)x[c] * cs) << q.sh_a) + (((i128)rot * sn) << q.sh_b);
          vals[c] = sat16(rne_shr(s, q.sh_c));
        }
        out.push_back(row_write(rc, vals, I16));
      }
    return out;
  }

  if (op == XPOSE) {
    const uint32_t R = j.aux & 15, L = n;
    if (R == 0 || R > 8) return out;
    for (u64 i2 = 0; i2 < j.c2; i2++)
      for (u64 i1 = 0; i1 < j.c1; i1++) {
        const u64 ta = j.a + i2 * j.a_s2 + i1 * j.a_s1;
        const u64 tc = j.c + i2 * j.c_s2 + i1 * j.c_s1;
        if (j.b_s2 == 2 * R) {
          // contiguous granules: the tile is one run of L * R elements written as whole
          // words, the lanes past the last element zero (the stream ops' contract)
          Write w{tc, {}};
          for (uint32_t d = 0; d < L; d++)
            for (uint32_t r = 0; r < R; r++) put(w.bytes, m.rd16(ta + (u64)r * j.b_s1 + 2ull * d), I16);
          w.bytes.resize((w.bytes.size() + 15) & ~size_t(15), 0);
          out.push_back(std::move(w));
        } else
          for (uint32_t d = 0; d < L; d++) {
            Write w{tc + (u64)d * j.b_s2, {}};
            for (uint32_t r = 0; r < R; r++) put(w.bytes, m.rd16(ta + (u64)r * j.b_s1 + 2ull * d), I16);
            out.push_back(std::move(w));
          }
      }
    return out;
  }
  return out;
}

}  // namespace ro
