// ---------------------------------------------------------------------------
// PoolingKernel.cpp — 2-D pooling kernel.
//
// Implements the ONNX MaxPool / AveragePool / LpPool operators (and their
// Global variants) in a channel-tiled structure that maps cleanly to Vitis
// HLS synthesis.
//
// Top-level dataflow (HLS DATAFLOW) — post-§2.14 (128-bit x AND y, 8 lanes
// per cycle through every stage):
//
//   x (DDR gmem0, 128-bit)
//     │ burst_maxi read: one request per (row, channel) run, kReadAhead in flight
//     ▼
//   row_loader ──row_word_pipe (128-bit words)──► window_emitter
//                                                   │ owns line_buf (LUTRAM column banks)
//                                                   │ writes 8 lanes / cycle, reads
//                                                   │ kOwParallel columns × kTileC channels / cycle
//                                                   ├──window_pipe (MultiWindow)──► process_pool_kernel_tile
//                                                   └──denom_pipe  (MultiDenom) ──►   │ reduce ‖ finalise(prev group)
//                                                                                     ▼
//                                     acc_stream (FinBundle = kOwParallel outputs of one channel)
//                                                                                     ▼
//                                                   write_output_tile ── ping-pong row buffer, one
//                                                   byte-strobed burst per (row, channel) run ──► y (DDR gmem1, 128-bit)
//
// Stage responsibilities:
//   * row_loader              — pure DDR reader; pushes every word of every
//                              (row, channel) run of a chunk in one flattened
//                              II=1 loop with requests running ahead.
//   * window_emitter          — owns line_buf.  One flattened II=1 loop per
//                              output row loads the NEXT row's input words
//                              (8 lanes per cycle into column banks) while
//                              emitting the current row's pool_h*pool_w
//                              MultiWindows per ow-group and one MultiDenom
//                              per group.  Out-of-bounds positions, padded
//                              lanes and channel-padding lanes carry the
//                              pool-type identity.
//   * process_pool_kernel_tile — one flattened II=1 loop per output row:
//                              reduces one MultiWindow per cycle into
//                              acc[kOwParallel][kTileC] and, in the same
//                              iterations, finalises the PREVIOUS group's
//                              snapshot kOwParallel lanes per cycle onto
//                              acc_stream (saturated Data_t).
//   * write_output_tile      — one flattened II=1 loop per output row fills
//                              a row buffer with the row's bundles while
//                              draining the previous row as 128-bit words
//                              with byte strobes on the run edges.
//
// Global pool support:
//   No special code.  Caller passes pool_h=in_h, pool_w=in_w, stride=1,
//   pad_top=pad_left=0.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include "hls_stream.h"

#include "PoolingKernel.h"
#include "PoolingKernelDebug.h"

// ---------------------------------------------------------------------------
// Debug-only DDR-read tracking.  Enabled automatically for C-simulation
// builds; disabled (zero-cost) under HLS synthesis.  Mirrors the pattern in
// kernels/conv/kernel/ConvKernel.cpp so the testbench can flag any cell
// read from DDR more than once per invocation.
// ---------------------------------------------------------------------------
#ifndef __SYNTHESIS__
#define DEBUG_LOAD_DATA_CACHING
#endif

#ifdef DEBUG_LOAD_DATA_CACHING
static unsigned g_pool_debug_duplicate_reads = 0;
#endif

void pool_debug_reset_duplicate_reads() {
#ifdef DEBUG_LOAD_DATA_CACHING
    g_pool_debug_duplicate_reads = 0;
#endif
}

unsigned pool_debug_duplicate_read_count() {
#ifdef DEBUG_LOAD_DATA_CACHING
    return g_pool_debug_duplicate_reads;
#else
    return 0;
#endif
}

#ifdef DEBUG_LOAD_DATA_CACHING
#include <cstddef>
#include <iostream>
#include <list>
#include <map>

// One record per DDR fetch — captures the loop-nest position so a duplicate
// dump can show *where* the kernel went back to the same address.
struct PoolReadCounters {
    unsigned ni;
    unsigned ct;
    unsigned c_l;
    unsigned oh;
    unsigned ow;
    unsigned khi;
    unsigned kwi;
};

inline std::ostream& operator<<(std::ostream& os, const PoolReadCounters& c) {
    os << "{ni=" << c.ni
       << " ct=" << c.ct
       << " c_l=" << c.c_l
       << " oh=" << c.oh
       << " ow=" << c.ow
       << " khi=" << c.khi
       << " kwi=" << c.kwi
       << "}";
    return os;
}

typedef std::map<std::size_t, std::list<PoolReadCounters>> PoolAddressMap_t;
#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// WindowLanes — kTileC parallel pixel lanes for a SINGLE ow position.
//
// Conceptually the per-position slice that survived from the pre-§2.9
// implementation; now nested inside MultiWindow so the FIFO word covers
// kOwParallel adjacent ow positions per cycle (see below).  The consumer
// still lays out its acc[p][c1] grid on the same shape.
//
// HLS packs the array into a single field (kTileC * sizeof(Data_t) * 8 b,
// e.g. 128 b for ap_fixed<16,8> × 8).  Plain C array inside a POD struct
// keeps the type trivially copyable so hls::stream can pass it efficiently
// in C-sim and infer a wide FIFO bus in HW.
// ---------------------------------------------------------------------------
struct WindowLanes {
    Data_t lanes[kTileC];
};

// ---------------------------------------------------------------------------
// MultiWindow — kOwParallel × kTileC pixels carried through window_pipe per
// (khi, kwi) cycle, spanning kOwParallel adjacent output positions.
//
// Producer emits one MultiWindow per (khi, kwi) of the pool window.
// Consumer reads one MultiWindow per cycle and updates kOwParallel ×
// kTileC accumulators in parallel — the linear-in-kOwParallel speedup of
// §2.9.  Each per-position WindowLanes has the kTileC channel lanes laid
// out exactly as before, so the consumer's acc[p][c1] update path is a
// straight unroll of the prior acc[c1] loop.
//
// Total FIFO word: kOwParallel * kTileC * sizeof(Data_t) * 8 b
// (256 b at default kOwParallel=2, kTileC=8, ap_fixed<16,8>).
// ---------------------------------------------------------------------------
struct MultiWindow {
    WindowLanes lanes[kOwParallel];
};

// ---------------------------------------------------------------------------
// MultiDenom — kOwParallel valid-pixel counts (or pool_h*pool_w when
// count_include_pad=1) carried through denom_pipe per ow-group.
//
// Each of the kOwParallel positions in a group can have a different denom
// because windows touching different sides of the input have different
// in-bounds counts (corner = fewer valid pixels than centre).  Bundling
// them in one struct keeps the producer/consumer to one denom_pipe
// transaction per ow-group, matching the MultiWindow rate.
// ---------------------------------------------------------------------------
struct MultiDenom {
    unsigned d[kOwParallel];
};

// ---------------------------------------------------------------------------
// poly_sqrt — fixed-point sqrt approximation for LP-Pool p=2 finalize.
//
// Replaces sqrtf((float)acc) → AccData_t round-trip with a fully fixed-point
// pipeline so HLS does not instantiate the FP square-root unit (which costs
// dedicated DSP slices and adds 12+ cycle latency).
//
// Strategy — 3rd-order polynomial with range reduction:
//   1. Decompose x = m * 4^k with m ∈ [1, 4), k = ⌊log₄ x⌋.
//      Implementation: locate MSB of the raw 32-bit fixed-point bits via
//      a priority encoder; shift to land m_raw in [2^16, 2^18).
//   2. Approximate √m by Lagrange interpolation through (1,1), (2,√2),
//      (3,√3), (4,2):
//        √m ≈ 0.4434 + 0.6432·m − 0.0943·m² + 0.0077·m³
//      Max error on [1, 4]: ~0.22% (well under Data_t's 1/256 ≈ 0.4% LSB).
//      Evaluated via Horner's scheme — 3 multiplies, 3 adds.
//   3. √x = √m × 2^k.  Single variable shift (barrel) to scale.
//
// Latency: ~6-8 cycles (priority encoder + Horner chain + final shift).
// Throughput: II=1.  No DSP for FP units; ~3 fabric multipliers for the
// polynomial.
// ---------------------------------------------------------------------------
#ifdef POOL_HAVE_APFIXED
static inline AccData_t poly_sqrt(AccData_t acc) {
    #pragma HLS INLINE
    if (acc <= AccData_t(0)) return AccData_t(0);

    // Step 1 — range reduction.  Reinterpret as ap_ufixed<32,16>; raw
    // 32-bit value satisfies raw = acc × 2^16.
    ap_ufixed<32, 16> ux = acc;
    ap_uint<32> raw = ux.range();

    // Priority encoder: find position P of the highest set bit in raw.
    // "Last write wins" pattern unrolls to a clog2(32)≈5 LUT-level tree.
    ap_uint<5> P = 0;
    for (int i = 0; i < 32; i++) {
        #pragma HLS UNROLL
        if (raw[i]) P = (ap_uint<5>)i;
    }

    // 2k = even integer ≤ (P − 16); k = 2k / 2.
    // Bit-AND with ~1 rounds toward −∞ in two's complement (correct for
    // both positive and negative values).
    const int two_k = ((int)P - 16) & ~1;
    const int k     = two_k >> 1;

    // Shift raw to land m_raw ∈ [2^16, 2^18) — represents m ∈ [1, 4).
    ap_uint<32> m_raw;
    if (two_k >= 0) {
        m_raw = raw >> (unsigned)two_k;
    } else {
        m_raw = raw << (unsigned)(-two_k);
    }
    ap_ufixed<18, 2> m;
    m.range() = m_raw.range(17, 0);

    // Step 2 — 3rd-order polynomial via Horner's scheme.
    // Coefficients in ap_fixed<16,1> (15 fractional bits, range [-1, 1)).
    // Lagrange-interpolated through (1,1), (2,√2), (3,√3), (4,2).
    const ap_fixed<16, 1> c0 =  0.4434;
    const ap_fixed<16, 1> c1 =  0.6432;
    const ap_fixed<16, 1> c2 = -0.0943;
    const ap_fixed<16, 1> c3 =  0.0077;

    const ap_fixed<24, 4> t1     = c3 * m + c2;     // [-0.094, -0.063]
    const ap_fixed<24, 4> t2     = t1 * m + c1;     // [ 0.297,  0.643]
    const ap_fixed<24, 4> sqrt_m = t2 * m + c0;     // [ 1.000,  2.000]

    // Step 3 — apply 2^k scaling via variable shift (barrel).
    AccData_t result = sqrt_m;
    if (k >= 0) {
        result <<= (unsigned)k;
    } else {
        result >>= (unsigned)(-k);
    }
    return result;
}
#else  // float fallback for non-ap_fixed builds
static inline AccData_t poly_sqrt(AccData_t acc) {
    return AccData_t(sqrtf((float)acc));
}
#endif

// ---------------------------------------------------------------------------
// inv_denom_lookup — fixed-point reciprocal lookup for AVG-pool finalisation.
//
// The previous AVG path was:
//     const float inv_denom = 1.0f / (float)denom_u;
//     result = AccData_t((float)acc[c1] * inv_denom);
// which forced HLS to instantiate an FP divider (28 cyc / ~5 DSP), an FP
// multiplier (~3 DSP) and two convert units, just to scale the accumulator
// by 1/N.  This replaces the round-trip with a pure ap_fixed pipeline:
//
//   1. A compile-time-constant ROM, indexed by denom_u, returns 1/denom_u
//      pre-encoded as ap_ufixed<24, 1> raw bits (`raw = round(2^23 / d)`).
//      23 fractional bits give an LSB of 2^-23 ≈ 1.19e-7 — five+ orders of
//      magnitude below Data_t's 1/256 LSB — so the LUT's quantisation error
//      is invisible at the saturated Data_t output.  Encoding as uint32_t
//      lets the table be `constexpr`-initialised at translation time; HLS
//      synthesises it as a ROM (≈3 KiB → 1 BRAM18) with no init loop in
//      hardware.
//
//   2. The use site reconstructs an ap_ufixed<24, 1> by raw .range()
//      assignment — a direct bit copy, no FP→fixed converter.
//
//   3. `acc[c1] * inv_denom` is a native ap_fixed multiply: ap_fixed<32,16> ×
//      ap_ufixed<24,1> → ap_fixed<56,17>, then implicit-cast back to
//      AccData_t.  One fabric multiplier; no DSP-FPU.  Overflow on the
//      narrowing cast is impossible in normal use because |acc/d| ≤
//      max(|acc|/1) which already fits AccData_t — i.e. dividing can only
//      shrink magnitude.
//
// Sizing — kMaxAvgDenom = kMaxLineBufRows * kMaxLineBufCols (1024 with the
// default config).  The line buffers cap pool window area, so denom_u (which
// is at most pool_h * pool_w when count_include_pad=1, and the count of
// in-bounds positions otherwise) cannot exceed this bound.  The clamp on the
// lookup is defensive — in normal flow `d` is always in [1, kMaxAvgDenom].
//
// Numerical stability vs the prior float path — the LUT holds 1/d to 24-bit
// fixed-point precision (relative error ≤ 2^-24 / (1/d) = d / 2^24, ≈ 6e-5
// at d=1024).  The float reciprocal had ~24-bit mantissa (similar precision)
// but the round-trip through float lost ~8 bits of acc precision on the
// (float)acc cast (24-bit float mantissa < 32-bit AccData_t).  So the new
// path is strictly more accurate.
// ---------------------------------------------------------------------------
#ifdef POOL_HAVE_APFIXED
typedef ap_ufixed<24, 1> InvDenom_t;
static constexpr unsigned kMaxAvgDenom = kMaxLineBufRows * kMaxLineBufCols;

namespace {

constexpr uint32_t encode_inv_denom_bits(unsigned d) {
    // ap_ufixed<24,1> stores value v as floor(v * 2^23).  We want
    //   raw = round(2^23 / d)
    // implemented in pure integer arithmetic via the standard "+d/2"
    // round-to-nearest trick.  d=0 is unreachable in normal flow; the
    // sentinel raw=0 ensures multiplying by it produces 0 rather than
    // garbage if a caller ever did pass 0.
    return (d == 0u)
        ? 0u
        : (((1u << 23) + d / 2u) / d);
}

template <unsigned... Is>
constexpr std::array<uint32_t, sizeof...(Is)>
make_inv_denom_lut_helper(std::integer_sequence<unsigned, Is...>) {
    return std::array<uint32_t, sizeof...(Is)>{ encode_inv_denom_bits(Is)... };
}

constexpr auto kInvDenomLutBits = make_inv_denom_lut_helper(
    std::make_integer_sequence<unsigned, kMaxAvgDenom + 1>{});

}  // namespace

static inline InvDenom_t inv_denom_lookup(unsigned d) {
    #pragma HLS INLINE
    const unsigned idx = (d <= kMaxAvgDenom) ? d : 0u;
    InvDenom_t r;
    r.range() = ap_uint<24>(kInvDenomLutBits[idx]);
    return r;
}
#else  // float fallback — keep prior FP-domain reciprocal/multiply.
typedef float InvDenom_t;
static inline InvDenom_t inv_denom_lookup(unsigned d) {
    return (d > 0u) ? 1.0f / (float)d : 0.0f;
}
#endif

// ---------------------------------------------------------------------------
// compute_ow_tile — runtime W-tile width.
//
// Returns the largest ow chunk size whose loaded input-column span fits in
// kMaxLineBufCols.  When in_w <= kMaxLineBufCols the result is out_w (single tile, no
// duplication — same as the zero-duplication path).  When the input is
// wider than the cache we split out_w into multiple tiles; adjacent tiles
// re-read overlapping boundary columns from DDR — the explicit relaxation
// that lets the kernel handle in_w > kMaxLineBufCols.
//
// Span(OW) = (OW - 1) * stride_w + (pool_w - 1) * dil_w + 1
// Solve Span(OW) <= kMaxLineBufCols for the largest OW.
// ---------------------------------------------------------------------------
static inline unsigned compute_ow_tile(
    unsigned out_w,
    unsigned pool_w,
    unsigned stride_w,
    unsigned dil_w
) {
    const unsigned win_w_span = (pool_w - 1) * dil_w + 1;
    unsigned ow_tile = 0;
    if (win_w_span <= kMaxLineBufCols && stride_w > 0) {
        ow_tile = (kMaxLineBufCols - win_w_span) / stride_w + 1;
    }
    if (ow_tile == 0) ow_tile = 1;
    if (ow_tile > out_w) ow_tile = out_w;
    return ow_tile;
}

// ---------------------------------------------------------------------------
// Per-invocation geometry (§2.11 / §2.14).
//
// Computed ONCE in PoolingKernel and passed to every stage, so the runtime-
// divisor divisions (÷ stride_w in compute_ow_tile, ÷ ow_tile) are
// synthesised once.  The §2.14 fields:
//
//   gw            ow positions per group.  kOwParallel, unless the stride
//                 maps the group's columns onto the same line-buffer column
//                 bank (stride_w a multiple of 2·kLanes/kOwParallel), in
//                 which case groups are one column wide and lane p ≥ 1 is
//                 identity-padded (the bank is 1W1R LUTRAM: one read per
//                 bank per cycle).
//   rows_per_chunk input rows loaded per (ni, ct, owt) chunk — every row from
//                 0 up to the last one any output row needs, each once.
//   red_len       pool_h · pool_w window reads per group.
//   slot_len      consumer iterations per group: max(red_len, kTileC) — the
//                 finalise of the previous group (kTileC steps of kOwParallel
//                 lanes) overlaps the reduce of the current one.
//   prefetch      the rows of output row oh+1 are loaded while oh is emitted;
//                 needs (pool_h-1)·dil_h + 1 + stride_h ≤ kMaxLineBufRows so
//                 the incoming slots never alias the window in use.  Otherwise
//                 each row's loads precede its emit inside the same loop.
// ---------------------------------------------------------------------------
static constexpr unsigned kLanes   = kPoolPortElems;              // elements per port word
static constexpr unsigned kLbWords = kMaxLineBufCols / kLanes;    // line-buffer words per row slot
static_assert(kMaxLineBufCols % kLanes == 0, "kMaxLineBufCols must be a whole number of port words");
static_assert(kOwParallel <= kLanes && (kOwParallel & (kOwParallel - 1)) == 0,
              "kOwParallel must be a power of two <= the port lane count");
static_assert((kMaxLineBufRows & (kMaxLineBufRows - 1)) == 0, "kMaxLineBufRows must be a power of two");

struct PoolGeometry {
    unsigned c_tiles;
    unsigned ow_tile;
    unsigned ow_tiles_w;
    unsigned gw;
    unsigned rows_per_chunk;
    unsigned red_len;
    unsigned slot_len;
    bool     prefetch;
};

static inline PoolGeometry compute_pool_geometry(
    unsigned channels,
    unsigned in_h,
    unsigned out_h,
    unsigned out_w,
    unsigned pool_h,
    unsigned pool_w,
    unsigned stride_h,
    unsigned stride_w,
    unsigned pad_top,
    unsigned dil_h,
    unsigned dil_w
) {
    PoolGeometry g;
    g.c_tiles    = (channels + kTileC - 1) / kTileC;
    g.ow_tile    = compute_ow_tile(out_w, pool_w, stride_w, dil_w);
    g.ow_tiles_w = (g.ow_tile > 0)
        ? ((out_w + g.ow_tile - 1) / g.ow_tile)
        : 1u;
    // Column banks are (column mod kLanes); kOwParallel columns spaced by
    // stride_w land on distinct banks iff gcd(stride_w, kLanes) <= kLanes /
    // kOwParallel, i.e. stride_w is not a multiple of 2·kLanes/kOwParallel.
    g.gw = (stride_w % (2u * kLanes / kOwParallel) != 0u) ? kOwParallel : 1u;
    const int ih_last = (int)((out_h - 1) * stride_h + (pool_h - 1) * dil_h) - (int)pad_top;
    g.rows_per_chunk = (ih_last < 0) ? 0u
                     : ((unsigned)ih_last >= in_h ? in_h : (unsigned)ih_last + 1u);
    g.red_len  = pool_h * pool_w;
    g.slot_len = (g.red_len > kTileC) ? g.red_len : kTileC;
    const unsigned span_h = (pool_h - 1) * dil_h + 1;
    g.prefetch = (span_h + stride_h <= kMaxLineBufRows);
    return g;
}

// ---------------------------------------------------------------------------
// Per-chunk geometry — one (ni, ct, owt) chunk = one channel tile × one
// W-tile.  Derived identically by every stage.
// ---------------------------------------------------------------------------
struct ChunkGeom {
    unsigned c_off;
    unsigned c_valid;
    unsigned ow_lo;
    unsigned ow_hi;
    unsigned ow_span;
    unsigned n_groups;
    unsigned iw_lo;      // first input column loaded (>= 0)
    unsigned run_len;    // input columns loaded per row
};

static inline ChunkGeom chunk_geom(
    unsigned ct, unsigned owt,
    unsigned channels, unsigned in_w, unsigned out_w,
    unsigned pool_w, unsigned stride_w, unsigned pad_left, unsigned dil_w,
    const PoolGeometry& geom
) {
    #pragma HLS INLINE
    ChunkGeom c;
    c.c_off   = ct * kTileC;
    c.c_valid = std::min(kTileC, channels - c.c_off);
    c.ow_lo   = owt * geom.ow_tile;
    c.ow_hi   = std::min(c.ow_lo + geom.ow_tile, out_w);
    c.ow_span = c.ow_hi - c.ow_lo;
    c.n_groups = (geom.gw == 1u) ? c.ow_span
               : ((c.ow_span + kOwParallel - 1u) / kOwParallel);
    const int iw_start = (int)(c.ow_lo * stride_w) - (int)pad_left;
    const int iw_end   = (int)((c.ow_hi - 1) * stride_w + (pool_w - 1) * dil_w) - (int)pad_left;
    const int iw_load_lo = (iw_start < 0) ? 0 : iw_start;
    const int iw_load_hi = (iw_end >= (int)in_w) ? (int)in_w - 1 : iw_end;
    c.iw_lo   = (unsigned)iw_load_lo;
    c.run_len = (unsigned)(iw_load_hi - iw_load_lo + 1);
    return c;
}

// ---------------------------------------------------------------------------
// RunCursor — position in a chunk's (ih, c_l) run sequence.  Each run is
// the contiguous element range [run_off, run_off + run_len) of one input
// row of one channel; advancing is adds only (no multiply in the loop-
// carried path of the II=1 loops that step it).
// ---------------------------------------------------------------------------
struct RunCursor {
    unsigned            ih;
    unsigned            c_l;
    unsigned            row_off;    // element offset of (ih, c_l = 0)
    unsigned            run_off;    // element offset of (ih, c_l)
    ap_uint<kPoolPortElems == 8 ? 3 : (kPoolPortElems == 4 ? 2 : 1)> row_shift;  // row_off % kLanes
    ap_uint<kPoolPortElems == 8 ? 3 : (kPoolPortElems == 4 ? 2 : 1)> run_shift;  // run_off % kLanes
};

// The lane shift is tracked separately in kLaneBits-bit arithmetic: the
// run's word count (next iteration's loop bound) depends only on it, and
// deriving it from the 32-bit offset add put the whole adder on the II=1
// loop-carried chain.
static inline RunCursor run_cursor_start(unsigned chunk_base) {
    #pragma HLS INLINE
    RunCursor c;
    c.ih = 0; c.c_l = 0; c.row_off = chunk_base; c.run_off = chunk_base;
    c.row_shift = chunk_base % kPoolPortElems;
    c.run_shift = c.row_shift;
    return c;
}

static inline void run_cursor_advance(RunCursor& c, unsigned c_valid,
                                      unsigned in_w, unsigned in_hw) {
    #pragma HLS INLINE
    if (c.c_l + 1 == c_valid) {
        c.c_l = 0;
        c.ih++;
        c.row_off += in_w;
        c.run_off = c.row_off;
        c.row_shift = c.row_shift + (in_w % kPoolPortElems);
        c.run_shift = c.row_shift;
    } else {
        c.c_l++;
        c.run_off += in_hw;
        c.run_shift = c.run_shift + (in_hw % kPoolPortElems);
    }
}

// ---------------------------------------------------------------------------
// Narrow index types.  Every counter that is structurally bounded by a
// compile-time constant is declared at that width so the II=1 loops' adders,
// comparators and muxes are sized to it (a 32-bit `unsigned` costs ~4× the
// LUT of a 7-bit one and lengthens the loop-carried chain).
// ---------------------------------------------------------------------------
// Bits needed to hold 0 .. v-1 (non-recursive: HLS rejects recursive
// constexpr functions even when only evaluated at compile time).
constexpr unsigned pool_clog2(unsigned v) {
    unsigned b = 0;
    while ((1u << b) < v) b++;
    return b;
}

static constexpr unsigned kLaneBits = pool_clog2(kLanes);                       // lane / bank index
static constexpr unsigned kWordBits = pool_clog2(kLbWords + 2u);                // words per run: 0 .. kLbWords + 1
static constexpr unsigned kColBits  = pool_clog2(kMaxLineBufCols + 1u);         // local column: 0 .. kMaxLineBufCols
static constexpr unsigned kSlotBits = pool_clog2(kMaxLineBufRows);
static constexpr unsigned kChBits   = pool_clog2(kTileC + 1u);
static constexpr unsigned kTapBits  = pool_clog2((kMaxPoolH > kMaxPoolW ? kMaxPoolH : kMaxPoolW) + 1u);
static constexpr unsigned kRbBits   = pool_clog2(kTileC * kLbWords);            // row-buffer entry

typedef ap_uint<kPoolDataBits> Lane;
typedef ap_uint<kLaneBits>     LaneIdx;
typedef ap_uint<kWordBits>     WordIdx;
typedef ap_uint<kColBits>      ColIdx;
typedef ap_uint<kSlotBits>     SlotIdx;
typedef ap_uint<kChBits>       ChIdx;
typedef ap_uint<kTapBits>      TapIdx;

// Words covering `count` elements whose first element sits at lane `shift`
// of its word: (shift + count + kLanes - 1) / kLanes.  Same value as
// pool_words_for(off, count) with shift = off % kLanes, but computed on
// (kLaneBits + kColBits)-bit operands — this sits on the loop-carried chain
// of the loader and emitter loops.
static inline WordIdx pool_run_words(LaneIdx shift, ColIdx count) {
    #pragma HLS INLINE
    const ap_uint<kColBits + kLaneBits + 1> s = (ap_uint<kColBits + kLaneBits + 1>)shift
                                              + (ap_uint<kColBits + kLaneBits + 1>)count
                                              + (ap_uint<kColBits + kLaneBits + 1>)(kLanes - 1);
    return (WordIdx)(s >> kLaneBits);
}

// A word is always split with CONSTANT ranges (a word.range() with a runtime
// lane index is a barrel shifter per use); lanes are then picked by indexing
// the partitioned register array, which HLS maps to a mux.
static inline void pool_split_lanes(PoolWord w, Lane lanes[kLanes]) {
    #pragma HLS INLINE
    for (unsigned l = 0; l < kLanes; l++) {
        #pragma HLS UNROLL
        lanes[l] = w.range(kPoolDataBits * (l + 1) - 1, kPoolDataBits * l);
    }
}

// ---------------------------------------------------------------------------
// row_loader — DATAFLOW source (§2.14).
//
// Pure DDR reader.  Per (ni, ct, owt) chunk it walks the run sequence
// (ih = 0 .. rows_per_chunk - 1, c_l = 0 .. c_valid - 1) and pushes every
// WORD of every run onto row_word_pipe (one word per cycle, one flattened
// II=1 loop per chunk).  Requests run ahead of the drain: a prologue issues
// the first kReadAhead runs' requests, then one further request is issued
// each time a run has been drained, so up to kReadAhead runs (≤ kReadAhead ×
// (kLbWords + 1) words < the adapter's num_read_outstanding ×
// max_read_burst_length buffer) are in flight and the DDR latency is paid
// once per chunk rather than once per row.  The window_emitter derives the
// same run sequence (shift / word count per run) itself, so no descriptor
// travels with the words.
// ---------------------------------------------------------------------------
static constexpr unsigned kReadAhead = 12;
static_assert(kReadAhead <= 16, "kReadAhead must not exceed num_read_outstanding");

static void row_loader(
    hls::burst_maxi<PoolWord> x,
    hls::stream<PoolWord>&    row_word_pipe,
    unsigned             batch,
    unsigned             channels,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_w,
    unsigned             pool_w,
    unsigned             stride_w,
    unsigned             pad_left,
    unsigned             dil_w,
    const PoolGeometry&  geom
) {
    const unsigned c_tiles    = geom.c_tiles;
    const unsigned in_hw      = in_h * in_w;
    const unsigned ow_tiles_w = geom.ow_tiles_w;
    const unsigned n_rows     = geom.rows_per_chunk;

#ifdef DEBUG_LOAD_DATA_CACHING
    PoolAddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const ChunkGeom cg = chunk_geom(ct, owt, channels, in_w, out_w,
                                                pool_w, stride_w, pad_left, dil_w, geom);
                const unsigned chunk_base = (ni * channels + cg.c_off) * in_hw + cg.iw_lo;
                const unsigned total_runs = n_rows * cg.c_valid;
                const ColIdx   run_len    = (ColIdx)cg.run_len;
                if (total_runs == 0) continue;

                // Prologue: the first min(kReadAhead, total_runs) requests.
                RunCursor rq = run_cursor_start(chunk_base);
                const unsigned n_pro = (total_runs < kReadAhead) ? total_runs : kReadAhead;
                for (unsigned k = 0; k < n_pro; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=12
                    x.read_request(rq.run_off / kLanes,
                                   pool_run_words((LaneIdx)rq.run_shift, run_len));
                    run_cursor_advance(rq, cg.c_valid, in_w, in_hw);
                }
                unsigned issued = n_pro;

                // Drain: one word per cycle over every run of the chunk.
                RunCursor rd = run_cursor_start(chunk_base);
                WordIdx  q         = 0;
                WordIdx  nw        = pool_run_words((LaneIdx)rd.run_shift, run_len);
                unsigned runs_done = 0;
                for (;;) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
                    const PoolWord word = x.read();
                    row_word_pipe.write(word);
#ifdef DEBUG_LOAD_DATA_CACHING
                    {
                        const unsigned shift = rd.run_off % kLanes;
                        for (unsigned l = 0; l < kLanes; l++) {
                            const int e = (int)((unsigned)q * kLanes + l) - (int)shift;
                            if (e >= 0 && e < (int)cg.run_len) {
                                PoolReadCounters c_rc;
                                c_rc.ni  = ni;
                                c_rc.ct  = ct;
                                c_rc.c_l = rd.c_l;
                                c_rc.oh  = 0;
                                c_rc.ow  = owt;
                                c_rc.khi = rd.ih;
                                c_rc.kwi = cg.iw_lo + (unsigned)e;
                                read_addresses[(std::size_t)rd.run_off + (unsigned)e].push_back(c_rc);
                            }
                        }
                    }
#endif
                    if (q + 1 == nw) {
                        q = 0;
                        runs_done++;
                        if (issued < total_runs) {
                            x.read_request(rq.run_off / kLanes,
                                           pool_run_words((LaneIdx)rq.run_shift, run_len));
                            run_cursor_advance(rq, cg.c_valid, in_w, in_hw);
                            issued++;
                        }
                        run_cursor_advance(rd, cg.c_valid, in_w, in_hw);
                        nw = pool_run_words((LaneIdx)rd.run_shift, run_len);
                        if (runs_done == total_runs) break;
                    } else {
                        q++;
                    }
                }
            }
        }
    }

#ifdef DEBUG_LOAD_DATA_CACHING
    for (const auto& it : read_addresses) {
        if (it.second.size() > 1) {
            ++g_pool_debug_duplicate_reads;
            std::cout << it.first << " --> " << std::endl;
            for (const auto& l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}

// ---------------------------------------------------------------------------
// window_emitter — DATAFLOW stage (§2.14).
//
// Owns line_buf, re-banked as [channel][column bank][row slot · kLbWords +
// column word]: kTileC × kLanes LUTRAMs (1 write + 1 read port each).  A
// row word from row_word_pipe is written whole in one cycle — its lanes are
// rotated by the run's word alignment so bank b takes the lane whose local
// column ≡ b (mod kLanes), lanes outside the run are dropped by a per-bank
// enable, and the channel is selected by an unrolled compare so every RAM
// sees exactly one conditional store.  A window read fetches, for each of
// the kOwParallel positions, one column of all kTileC channels: the column
// picks the bank and the address, and the position's value is muxed out of
// the bank vector (columns spaced by stride_w hit distinct banks — see
// PoolGeometry::gw).
//
// Per (ni, ct, owt) chunk, per output row: ONE flattened II=1 loop that
// (a) drains the words of the rows the NEXT output row needs into line_buf
// (prefetch mode — the slots are disjoint from the window in use) and
// (b) emits the pool_h·pool_w MultiWindows of every ow-group of the current
// row plus one MultiDenom per group.  When the vertical span leaves no slot
// headroom (PoolGeometry::prefetch == false) the same loop loads the current
// row's rows first and starts emitting kSeqGap iterations after the last
// word landed.  Both sides finish at their own pace; the loop exits when
// both are done, so no cycle is spent on either side's padding.
//
// Denominators: the in-bounds tap count is separable (§2.12), kh(oh) ×
// kw(ow).  kw(ow) is tallied once per chunk into a kMaxLineBufCols-entry
// LUTRAM by a serial (position, tap) pre-pass — one bounds test per cycle
// instead of kOwParallel × kMaxPoolW of them in the hot loop — and kh(oh)
// once per row.
// ---------------------------------------------------------------------------
static constexpr unsigned kSeqGap = 8;   // iterations between the last load and the first emit (sequential mode)

static void window_emitter(
    hls::stream<PoolWord>&     row_word_pipe,
    hls::stream<MultiWindow>&  window_pipe,
    hls::stream<MultiDenom>&   denom_pipe,
    unsigned                  batch,
    unsigned                  channels,
    unsigned                  in_h,
    unsigned                  in_w,
    unsigned                  out_h,
    unsigned                  out_w,
    unsigned                  pool_h,
    unsigned                  pool_w,
    unsigned                  stride_h,
    unsigned                  stride_w,
    unsigned                  pad_top,
    unsigned                  pad_left,
    unsigned                  dil_h,
    unsigned                  dil_w,
    unsigned                  pool_type,
    unsigned                  count_include_pad,
    const PoolGeometry&       geom
) {
    // kTileC × kLanes column banks of kMaxLineBufRows × kLbWords lanes each
    // (8 × 8 × 128 × 16 b = 128 Kb of LUTRAM at the defaults).  One write
    // (the incoming word's rotated lane) and one read (a window column) per
    // bank per cycle.  Loads and window reads inside one loop instance
    // touch different row slots (prefetch) or are kSeqGap iterations apart
    // (sequential), so no intra-loop dependence needs guarding.
    static Lane line_buf[kTileC][kLanes][kMaxLineBufRows * kLbWords];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_s2p impl=lutram
    #pragma HLS DEPENDENCE variable=line_buf inter false

    // kw(ow) for the chunk's positions, filled by the per-chunk pre-pass.
    // kOwParallel banks (position mod kOwParallel): a group's consecutive
    // positions are read in one cycle and each bank has one read port, so
    // every bank is read exactly once at an explicitly computed address and
    // the position's value is muxed out afterwards (a runtime bank index
    // would make HLS read every bank per position: II=2).
    static TapIdx kw_lut[kOwParallel][kMaxLineBufCols / kOwParallel];
    #pragma HLS ARRAY_PARTITION variable=kw_lut complete dim=1
    #pragma HLS BIND_STORAGE variable=kw_lut type=ram_s2p impl=lutram
    #pragma HLS DEPENDENCE variable=kw_lut inter false

    const unsigned c_tiles    = geom.c_tiles;
    const unsigned ow_tiles_w = geom.ow_tiles_w;
    const unsigned in_hw      = in_h * in_w;
    const unsigned gw         = geom.gw;
    const unsigned red_len    = geom.red_len;
    const bool     prefetch   = geom.prefetch;
    const unsigned denom_all  = pool_h * pool_w;
    const unsigned gw_stride  = gw * stride_w;
    const TapIdx   pool_h_t   = (TapIdx)pool_h;
    const TapIdx   pool_w_t   = (TapIdx)pool_w;

    const Lane pad_lane = pool_data_to_lane((pool_type == kPoolMax)
        ? Data_t(kDataMin)
        : Data_t(0));

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const ChunkGeom cg = chunk_geom(ct, owt, channels, in_w, out_w,
                                                pool_w, stride_w, pad_left, dil_w, geom);
                const unsigned chunk_base = (ni * channels + cg.c_off) * in_hw + cg.iw_lo;
                const unsigned n_emit     = cg.n_groups * red_len;
                const ColIdx   run_len    = (ColIdx)cg.run_len;
                const ChIdx    c_valid    = (ChIdx)cg.c_valid;
                // Local column of output position ow_lo for tap kwi = 0.
                const int col_base = (int)(cg.ow_lo * stride_w) - (int)pad_left - (int)cg.iw_lo;

                // ------------------------------------------------------
                // Per-chunk pre-pass: kw(ow) = |{kwi : 0 <= iw(ow, kwi) < in_w}|
                // for every position of the tile, one (position, tap) per
                // cycle.
                // ------------------------------------------------------
                {
                    const unsigned n_pre = cg.ow_span * pool_w;
                    ColIdx   pos = 0;
                    TapIdx   k   = 0;
                    TapIdx   cnt = 0;
                    int      iw_pos = (int)(cg.ow_lo * stride_w) - (int)pad_left;   // tap 0 of position pos
                    int      iw_v   = iw_pos;
                    for (unsigned t = 0; t < n_pre; t++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=448
                        const TapIdx c_next = (iw_v >= 0 && (unsigned)iw_v < in_w) ? (TapIdx)(cnt + 1) : cnt;
                        if (k + 1 == pool_w_t) {
                            for (unsigned b = 0; b < kOwParallel; b++) {
                                #pragma HLS UNROLL
                                if ((unsigned)(pos & (kOwParallel - 1)) == b)
                                    kw_lut[b][pos >> pool_clog2(kOwParallel)] = c_next;
                            }
                            pos++;
                            k = 0;
                            cnt = 0;
                            iw_pos += (int)stride_w;
                            iw_v = iw_pos;
                        } else {
                            k++;
                            cnt = c_next;
                            iw_v += (int)dil_w;
                        }
                    }
                }

                // Load cursor — continues across the output rows of the chunk.
                RunCursor lc = run_cursor_start(chunk_base);
                WordIdx   lq  = 0;
                WordIdx   lnw = pool_run_words((LaneIdx)lc.run_shift, run_len);
                unsigned  loaded_rows = 0;

                for (int oh = -1; oh < (int)out_h; oh++) {
                    // Rows to load in this loop instance: those output row
                    // oh_t needs beyond the ones already resident.
                    const int oh_t = prefetch ? oh + 1 : oh;
                    unsigned n_load_runs = 0;
                    if (oh_t >= 0 && oh_t < (int)out_h) {
                        int load_hi = (int)((unsigned)oh_t * stride_h + (pool_h - 1) * dil_h) - (int)pad_top;
                        if (load_hi >= (int)in_h) load_hi = (int)in_h - 1;
                        if (load_hi + 1 > (int)loaded_rows) {
                            n_load_runs = ((unsigned)(load_hi + 1) - loaded_rows) * cg.c_valid;
                            loaded_rows = (unsigned)(load_hi + 1);
                        }
                    }
                    const bool emit = (oh >= 0);

                    // Per-row window constants: first input row of the window
                    // and kh(oh).
                    const int ih0 = (int)((unsigned)(oh < 0 ? 0 : oh) * stride_h) - (int)pad_top;
                    unsigned num_valid_kh = 0;
                    {
                        int ih_v = ih0;
                        for (unsigned khi = 0; khi < kMaxPoolH; khi++) {
                            #pragma HLS UNROLL
                            if (khi < pool_h && ih_v >= 0 && (unsigned)ih_v < in_h) num_valid_kh++;
                            ih_v += (int)dil_h;
                        }
                    }

                    // Loop state.
                    unsigned ld_runs = 0;
                    bool     ld_done = (n_load_runs == 0);
                    ap_uint<4> gap   = 0;
                    unsigned em_i    = 0;
                    unsigned g       = 0;
                    TapIdx   khi = 0, kwi = 0;
                    unsigned ow_g    = cg.ow_lo;
                    ColIdx   gpos    = 0;             // ow_g - ow_lo
                    int      gcol    = col_base;      // local column of (ow_g, kwi = 0)
                    int      tap_col = 0;             // kwi * dil_w
                    int      ih      = ih0;           // ih0 + khi * dil_h
                    bool     em_done = !emit || (n_emit == 0);

                    for (;;) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=4096
                        const bool ld_now = !ld_done;
                        const bool em_now = !em_done && (prefetch || (ld_done && gap >= kSeqGap));

                        // ------------------------------------------------
                        // (a) one row word → line_buf
                        // ------------------------------------------------
                        if (ld_now) {
                            const PoolWord word  = row_word_pipe.read();
                            const LaneIdx  shift = (LaneIdx)lc.run_shift;
                            const SlotIdx  slot  = (SlotIdx)lc.ih;
                            const ChIdx    ch    = (ChIdx)lc.c_l;
                            Lane lanes[kLanes];
                            #pragma HLS ARRAY_PARTITION variable=lanes complete
                            pool_split_lanes(word, lanes);
                            for (unsigned b = 0; b < kLanes; b++) {
                                #pragma HLS UNROLL
                                // Bank b receives lane (b + shift) mod kLanes;
                                // that lane's local column is lq·kLanes + b,
                                // minus kLanes when the lane index wrapped.
                                const ap_uint<kLaneBits + 1> bs   = (ap_uint<kLaneBits + 1>)b + (ap_uint<kLaneBits + 1>)shift;
                                const bool    wrap = bs[kLaneBits];
                                const LaneIdx src  = (LaneIdx)bs;
                                const ap_int<kColBits + kLaneBits + 2> col =
                                    (ap_int<kColBits + kLaneBits + 2>)(((unsigned)lq << kLaneBits) + b)
                                  - (wrap ? (ap_int<kColBits + kLaneBits + 2>)kLanes : (ap_int<kColBits + kLaneBits + 2>)0);
                                const bool ok = (col >= 0) && (col < (ap_int<kColBits + kLaneBits + 2>)run_len);
                                const WordIdx w = (wrap && lq > 0) ? (WordIdx)(lq - 1) : (wrap ? (WordIdx)0 : lq);
                                const unsigned addr = ((unsigned)slot << pool_clog2(kLbWords)) + (unsigned)w;
                                const Lane v = lanes[src];
                                for (unsigned c = 0; c < kTileC; c++) {
                                    #pragma HLS UNROLL
                                    if (ok && ch == c) line_buf[c][b][addr] = v;
                                }
                            }
                            if (lq + 1 == lnw) {
                                lq = 0;
                                ld_runs++;
                                run_cursor_advance(lc, cg.c_valid, in_w, in_hw);
                                lnw = pool_run_words((LaneIdx)lc.run_shift, run_len);
                                if (ld_runs == n_load_runs) ld_done = true;
                            } else {
                                lq++;
                            }
                        } else if (!em_done && gap < kSeqGap) {
                            gap++;
                        }

                        // ------------------------------------------------
                        // (b) one (khi, kwi) tap of the current ow-group
                        // ------------------------------------------------
                        if (em_now) {
                            if (khi == 0 && kwi == 0) {
                                // Bank b holds position gpos + ((b - gpos) mod kOwParallel).
                                TapIdx kwv[kOwParallel];
                                #pragma HLS ARRAY_PARTITION variable=kwv complete
                                for (unsigned b = 0; b < kOwParallel; b++) {
                                    #pragma HLS UNROLL
                                    const ColIdx pos_b = (ColIdx)(gpos + (unsigned)((b - (unsigned)gpos) & (kOwParallel - 1)));
                                    kwv[b] = kw_lut[b][pos_b >> pool_clog2(kOwParallel)];
                                }
                                MultiDenom md;
                                for (unsigned p = 0; p < kOwParallel; p++) {
                                    #pragma HLS UNROLL
                                    const ap_uint<pool_clog2(kOwParallel) == 0 ? 1 : pool_clog2(kOwParallel)> bank =
                                        (unsigned)(gpos + p) & (kOwParallel - 1);
                                    const unsigned kw = (unsigned)kwv[bank];
                                    md.d[p] = count_include_pad ? denom_all : (num_valid_kh * kw);
                                }
                                denom_pipe.write(md);
                            }

                            const bool    ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const SlotIdx slot  = ih_ok ? (SlotIdx)ih : (SlotIdx)0;
                            const int     col0  = gcol + tap_col;          // local column of position 0

                            LaneIdx  bank_p[kOwParallel];
                            unsigned addr_p[kOwParallel];
                            bool     ok_p[kOwParallel];
                            #pragma HLS ARRAY_PARTITION variable=bank_p complete
                            #pragma HLS ARRAY_PARTITION variable=addr_p complete
                            #pragma HLS ARRAY_PARTITION variable=ok_p complete
                            for (unsigned p = 0; p < kOwParallel; p++) {
                                #pragma HLS UNROLL
                                const int col = col0 + (int)(p * stride_w);
                                const bool spatial_ok = ih_ok && ((unsigned)col < cg.run_len)
                                                     && (ow_g + p < cg.ow_hi) && (p < gw);
                                const ColIdx ucol = spatial_ok ? (ColIdx)col : (ColIdx)0;
                                bank_p[p] = (LaneIdx)ucol;
                                addr_p[p] = ((unsigned)slot << pool_clog2(kLbWords)) + (unsigned)(ucol >> kLaneBits);
                                ok_p[p]   = spatial_ok;
                            }

                            // One read per bank: the address of whichever
                            // position maps to it (position 0 wins on a
                            // collision — only possible when p ≥ 1 is padded).
                            Lane rv[kTileC][kLanes];
                            #pragma HLS ARRAY_PARTITION variable=rv complete dim=0
                            for (unsigned b = 0; b < kLanes; b++) {
                                #pragma HLS UNROLL
                                unsigned addr = addr_p[kOwParallel - 1];
                                for (int p = (int)kOwParallel - 2; p >= 0; p--) {
                                    #pragma HLS UNROLL
                                    if (bank_p[p] == b) addr = addr_p[p];
                                }
                                for (unsigned c = 0; c < kTileC; c++) {
                                    #pragma HLS UNROLL
                                    rv[c][b] = line_buf[c][b][addr];
                                }
                            }

                            MultiWindow mw;
                            for (unsigned p = 0; p < kOwParallel; p++) {
                                #pragma HLS UNROLL
                                for (unsigned c = 0; c < kTileC; c++) {
                                    #pragma HLS UNROLL
                                    const Lane v = rv[c][bank_p[p]];
                                    const bool valid = ok_p[p] && (c < cg.c_valid);
                                    mw.lanes[p].lanes[c] = pool_lane_to_data(valid ? v : pad_lane);
                                }
                            }
                            window_pipe.write(mw);

                            // Advance (kwi, khi, g) and the incremental columns / row.
                            em_i++;
                            if (kwi + 1 == pool_w_t) {
                                kwi = 0;
                                tap_col = 0;
                                if (khi + 1 == pool_h_t) {
                                    khi = 0;
                                    ih = ih0;
                                    g++;
                                    ow_g += gw;
                                    gpos += (ColIdx)gw;
                                    gcol += (int)gw_stride;
                                } else {
                                    khi++;
                                    ih += (int)dil_h;
                                }
                            } else {
                                kwi++;
                                tap_col += (int)dil_w;
                            }
                            if (em_i == n_emit) em_done = true;
                        }

                        if (ld_done && em_done) break;
                    }
                } // oh loop (-1 = prologue)
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// process_pool_kernel_tile — DATAFLOW processor (§2.14).
//
// Per (ni, ct, owt, oh) row: ONE flattened II=1 loop of n_groups × slot_len
// iterations (plus one extra slot after the very last row).  In slot g,
// iterations i < red_len read one MultiWindow and update all kOwParallel ×
// kTileC accumulators (the §2.9 reduce); iterations i < kTileC finalise
// channel i of the PREVIOUS group's snapshot — kOwParallel lanes per cycle
// (AVG: × inv_denom, LP-2: poly_sqrt), saturate to Data_t and push one
// FinBundle to acc_stream.  At the end of the slot the accumulators and the
// group's reciprocals are copied into the snapshot registers.  slot_len =
// max(red_len, kTileC), so the finalise never trails by more than one slot
// and the reduce never waits for it; the finalise hardware is kOwParallel
// lanes wide instead of the §6.2.3 kOwParallel × kTileC.
//
// acc_stream order: (ni, ct, owt, oh, group, c1) with kOwParallel adjacent
// ow positions of channel c1 per bundle — exactly the writer's row-buffer
// fill order.
// ---------------------------------------------------------------------------
typedef ap_uint<kOwParallel * kPoolDataBits> FinBundle;
static constexpr unsigned kSlotCntBits = pool_clog2(kMaxPoolH * kMaxPoolW + kTileC + 1u);
typedef ap_uint<kSlotCntBits> SlotCnt;

static void process_pool_kernel_tile(
    hls::stream<MultiWindow>& window_pipe,
    hls::stream<MultiDenom>&  denom_pipe,
    hls::stream<FinBundle>&   acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                in_w,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                pool_w,
    unsigned                stride_w,
    unsigned                pad_left,
    unsigned                dil_w,
    unsigned                pool_type,
    unsigned                lp_order,
    const PoolGeometry&     geom
) {
    AccData_t acc[kOwParallel][kTileC];
    AccData_t acc_done[kOwParallel][kTileC];
    InvDenom_t inv[kOwParallel];
    InvDenom_t inv_done[kOwParallel];
    #pragma HLS ARRAY_PARTITION variable=acc      complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc_done complete dim=0
    #pragma HLS ARRAY_PARTITION variable=inv      complete
    #pragma HLS ARRAY_PARTITION variable=inv_done complete

    const unsigned c_tiles    = geom.c_tiles;
    const unsigned ow_tiles_w = geom.ow_tiles_w;
    const SlotCnt  red_len    = (SlotCnt)geom.red_len;
    const SlotCnt  slot_len   = (SlotCnt)geom.slot_len;
    const bool     is_max     = (pool_type == kPoolMax);
    const bool     is_avg     = (pool_type == kPoolAvg);
    const bool     lp1        = (lp_order == 1u);
    const AccData_t init_val  = is_max ? AccData_t(kAccMin) : AccData_t(0);

    for (unsigned p = 0; p < kOwParallel; p++) {
        #pragma HLS UNROLL
        inv[p] = inv_denom_lookup(1u);
        inv_done[p] = inv[p];
        for (unsigned c1 = 0; c1 < kTileC; c1++) {
            #pragma HLS UNROLL
            acc[p][c1] = AccData_t(0);
            acc_done[p][c1] = AccData_t(0);
        }
    }
    bool fin_pending = false;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const ChunkGeom cg = chunk_geom(ct, owt, channels, in_w, out_w,
                                                pool_w, stride_w, pad_left, dil_w, geom);
                const bool last_chunk = (ni + 1 == batch) && (ct + 1 == c_tiles)
                                     && (owt + 1 == ow_tiles_w);

                for (unsigned oh = 0; oh < out_h; oh++) {
                    const bool     last_row = last_chunk && (oh + 1 == out_h);
                    const unsigned n_slots  = cg.n_groups + (last_row ? 1u : 0u);
                    const unsigned trip     = n_slots * (unsigned)slot_len;
                    unsigned g = 0;
                    SlotCnt  i = 0;

                    for (unsigned t = 0; t < trip; t++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=4096
                        const bool red   = (g < cg.n_groups) && (i < red_len);
                        const bool first = (i == 0);

                        // ---- reduce step i of group g ----
                        if (red) {
                            if (first) {
                                const MultiDenom md = denom_pipe.read();
                                for (unsigned p = 0; p < kOwParallel; p++) {
                                    #pragma HLS UNROLL
                                    inv[p] = inv_denom_lookup(md.d[p]);
                                }
                            }
                            const MultiWindow mw = window_pipe.read();
                            for (unsigned p = 0; p < kOwParallel; p++) {
                                #pragma HLS UNROLL
                                for (unsigned c1 = 0; c1 < kTileC; c1++) {
                                    #pragma HLS UNROLL
                                    const AccData_t val = AccData_t(mw.lanes[p].lanes[c1]);
                                    const AccData_t cur = first ? init_val : acc[p][c1];
                                    AccData_t nxt;
                                    if (is_max) {
                                        nxt = (val > cur) ? val : cur;
                                    } else {
                                        const AccData_t contrib = is_avg ? val
                                            : (lp1 ? (val < AccData_t(0) ? AccData_t(-val) : val)
                                                   : AccData_t(val * val));
                                        nxt = cur + contrib;
                                    }
                                    acc[p][c1] = nxt;
                                }
                            }
                        }

                        // ---- finalise channel i of the previous group ----
                        if (fin_pending && i < kTileC) {
                            const ChIdx ci = (ChIdx)i;
                            FinBundle fb = 0;
                            for (unsigned p = 0; p < kOwParallel; p++) {
                                #pragma HLS UNROLL
                                const AccData_t a = acc_done[p][ci];
                                AccData_t r;
                                if (is_max) {
                                    r = a;
                                } else if (is_avg) {
                                    r = AccData_t(a * inv_done[p]);
                                } else {
                                    r = lp1 ? a : poly_sqrt(a);
                                }
                                const Data_t d = saturate_cast<Data_t>(r);
                                fb.range(kPoolDataBits * (p + 1) - 1, kPoolDataBits * p) =
                                    pool_data_to_lane(d);
                            }
                            acc_stream.write(fb);
                        }

                        // ---- slot bookkeeping ----
                        if (i + 1 == slot_len) {
                            if (g < cg.n_groups) {
                                for (unsigned p = 0; p < kOwParallel; p++) {
                                    #pragma HLS UNROLL
                                    inv_done[p] = inv[p];
                                    for (unsigned c1 = 0; c1 < kTileC; c1++) {
                                        #pragma HLS UNROLL
                                        acc_done[p][c1] = acc[p][c1];
                                    }
                                }
                                fin_pending = true;
                            } else {
                                fin_pending = false;
                            }
                            i = 0;
                            g++;
                        } else {
                            i++;
                        }
                    }
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// write_output_tile — DATAFLOW sink (§2.14).
//
// Transposes the consumer's (group, channel) bundles into per-channel
// output-row runs and writes them through the 128-bit y port.  Two row
// buffers (ping-pong), each kLanes column banks × (kTileC · kLbWords)
// entries of LUTRAM: a bundle's kOwParallel adjacent columns of channel c1
// go to banks (col mod kLanes) at entry c1·kLbWords + col / kLanes; a DDR
// word of channel c's run is the kLanes banks read at entry c·kLbWords + k
// (or k − 1 for the lanes that wrap), rotated by the run's alignment.
//
// Per (ni, ct, owt, oh) row r: ONE flattened II=1 loop of max(fill, drain)
// iterations that fills buffer r&1 with row r's bundles (masking the
// identity-padded lanes past ow_hi and channels ≥ c_valid) while draining
// row r−1 from the other buffer: per channel one write_request of
// pool_words_for(start, len) ≤ kLbWords + 1 words, and the run's first /
// last words written with byte strobes covering only the run's own lanes
// (masked lanes are also zeroed in WDATA).  A final row-loop iteration
// drains the last row.  At most kWriteInFlight requests are un-acknowledged
// (< num_write_outstanding, the §2.30 rule).
// ---------------------------------------------------------------------------
static constexpr unsigned kWriteInFlight = 4;
static constexpr unsigned kRbEntries     = kTileC * kLbWords;
static constexpr unsigned kBeBits        = kLanes * (kPoolDataBits / 8);
typedef ap_uint<kBeBits> ByteEn;

struct RowDesc {
    bool     valid;
    ChIdx    c_valid;
    ColIdx   len;        // ow_span
    WordIdx  nw_max;     // drain iterations per channel: words_for worst case
    unsigned e_base;     // element index of (channel c_off, ow_lo)
};

static void write_output_tile(
    hls::burst_maxi<PoolWord> y,
    hls::stream<FinBundle>&   acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                in_w,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                pool_w,
    unsigned                stride_w,
    unsigned                pad_left,
    unsigned                dil_w,
    const PoolGeometry&     geom
) {
    static Lane rbA[kLanes][kRbEntries];
    static Lane rbB[kLanes][kRbEntries];
    #pragma HLS ARRAY_PARTITION variable=rbA complete dim=1
    #pragma HLS ARRAY_PARTITION variable=rbB complete dim=1
    #pragma HLS BIND_STORAGE variable=rbA type=ram_s2p impl=lutram
    #pragma HLS BIND_STORAGE variable=rbB type=ram_s2p impl=lutram
    #pragma HLS DEPENDENCE variable=rbA inter false
    #pragma HLS DEPENDENCE variable=rbB inter false

    const unsigned c_tiles    = geom.c_tiles;
    const unsigned ow_tiles_w = geom.ow_tiles_w;
    const unsigned hw         = out_h * out_w;
    const ColIdx   gw         = (ColIdx)geom.gw;
    const unsigned n_rows     = batch * c_tiles * ow_tiles_w * out_h;

    unsigned ni = 0, ct = 0, owt = 0, oh = 0;
    RowDesc  prev;
    prev.valid = false; prev.c_valid = 0; prev.len = 0; prev.nw_max = 0; prev.e_base = 0;
    unsigned pending = 0;

    for (unsigned r = 0; r <= n_rows; r++) {
        // Row r's descriptor (fill side).
        const ChunkGeom cg = chunk_geom(ct, owt, channels, in_w, out_w,
                                        pool_w, stride_w, pad_left, dil_w, geom);
        RowDesc cur;
        cur.valid   = (r < n_rows);
        cur.c_valid = (ChIdx)cg.c_valid;
        cur.len     = (ColIdx)cg.ow_span;
        cur.nw_max  = (WordIdx)((cg.ow_span + kLanes - 1) / kLanes + 1);
        cur.e_base  = ((ni * channels + cg.c_off) * out_h + oh) * out_w + cg.ow_lo;

        const unsigned fill_len  = cur.valid  ? cg.n_groups * kTileC : 0u;
        const unsigned drain_len = prev.valid ? (unsigned)prev.c_valid * (unsigned)prev.nw_max : 0u;
        const unsigned trip      = (fill_len > drain_len) ? fill_len : drain_len;
        const bool     pp        = (r & 1u) != 0u;   // fill rbB / drain rbA when set

        ChIdx    fc = 0;                             // fill: channel
        ColIdx   fcol = 0;                           // fill: first column of the group
        ChIdx    dc = 0;                             // drain: channel
        WordIdx  dk = 0;                             // drain: word
        unsigned de = prev.e_base;                   // drain: run start element of channel dc
        WordIdx  d_nout = 0;                         // words of the current drain run

        for (unsigned t = 0; t < trip; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=4096

            // ---- fill: one bundle = kOwParallel columns of channel fc ----
            if (t < fill_len) {
                const FinBundle fb = acc_stream.read();
                Lane    fv[kOwParallel];
                LaneIdx fbank[kOwParallel];
                bool    fok[kOwParallel];
                #pragma HLS ARRAY_PARTITION variable=fv    complete
                #pragma HLS ARRAY_PARTITION variable=fbank complete
                #pragma HLS ARRAY_PARTITION variable=fok   complete
                for (unsigned p = 0; p < kOwParallel; p++) {
                    #pragma HLS UNROLL
                    const ColIdx col = (ColIdx)(fcol + p);
                    fv[p]    = fb.range(kPoolDataBits * (p + 1) - 1, kPoolDataBits * p);
                    fbank[p] = (LaneIdx)col;
                    fok[p]   = (p < gw) && (col < cur.len) && (fc < cur.c_valid);
                }
                // All kOwParallel columns share the word index (fcol is a
                // multiple of gw and gw divides kLanes).
                const unsigned faddr = ((unsigned)fc << pool_clog2(kLbWords)) + (unsigned)(fcol >> kLaneBits);
                for (unsigned b = 0; b < kLanes; b++) {
                    #pragma HLS UNROLL
                    Lane v  = 0;
                    bool en = false;
                    for (unsigned p = 0; p < kOwParallel; p++) {
                        #pragma HLS UNROLL
                        if (fok[p] && fbank[p] == b) { v = fv[p]; en = true; }
                    }
                    if (en) {
                        if (pp) rbB[b][faddr] = v;
                        else    rbA[b][faddr] = v;
                    }
                }
                if (fc + 1 == kTileC) { fc = 0; fcol += gw; }
                else                  { fc++; }
            }

            // ---- drain: word dk of channel dc's run (previous row) ----
            if (t < drain_len) {
                const LaneIdx shift = (LaneIdx)(de % kLanes);
                if (dk == 0) {
                    d_nout = pool_run_words(shift, prev.len);
                    y.write_request(de / kLanes, (unsigned)d_nout);
                }
                if (dk < d_nout) {
                    Lane rv[kLanes];
                    #pragma HLS ARRAY_PARTITION variable=rv complete
                    for (unsigned b = 0; b < kLanes; b++) {
                        #pragma HLS UNROLL
                        const ap_uint<kLaneBits + 1> bs = (ap_uint<kLaneBits + 1>)b + (ap_uint<kLaneBits + 1>)shift;
                        const bool wrap = bs[kLaneBits];
                        // Word 0's wrapped lanes lie before the run (masked
                        // below); clamp their address so the read stays in
                        // bounds.
                        const WordIdx w = (wrap && dk > 0) ? (WordIdx)(dk - 1) : (wrap ? (WordIdx)0 : dk);
                        const unsigned addr = ((unsigned)dc << pool_clog2(kLbWords)) + (unsigned)w;
                        const Lane a  = rbA[b][addr];
                        const Lane bb = rbB[b][addr];
                        rv[b] = pp ? a : bb;
                    }
                    PoolWord out = 0;
                    ByteEn   be  = 0;
                    for (unsigned l = 0; l < kLanes; l++) {
                        #pragma HLS UNROLL
                        // Lane l of DDR word dk holds local column dk·kLanes + l − shift.
                        const ap_int<kColBits + kLaneBits + 2> col =
                            (ap_int<kColBits + kLaneBits + 2>)(((unsigned)dk << kLaneBits) + l)
                          - (ap_int<kColBits + kLaneBits + 2>)shift;
                        const bool ok = (col >= 0) && (col < (ap_int<kColBits + kLaneBits + 2>)prev.len);
                        const LaneIdx src = (LaneIdx)((ap_uint<kLaneBits + 1>)l - (ap_uint<kLaneBits + 1>)shift);
                        const Lane v  = ok ? rv[src] : Lane(0);
                        out.range(kPoolDataBits * (l + 1) - 1, kPoolDataBits * l) = v;
                        for (unsigned q = 0; q < kPoolDataBits / 8; q++) {
                            #pragma HLS UNROLL
                            be[l * (kPoolDataBits / 8) + q] = ok;
                        }
                    }
                    y.write(out, (ap_int<kBeBits>)be);
                }
                if (dk + 1 == prev.nw_max) {
                    dk = 0;
                    dc++;
                    de += hw;
                    if (pending == kWriteInFlight - 1) y.write_response();
                    else                               pending++;
                } else {
                    dk++;
                }
            }
        }

        prev = cur;
        // Advance (ni, ct, owt, oh).
        if (oh + 1 == out_h) {
            oh = 0;
            if (owt + 1 == ow_tiles_w) {
                owt = 0;
                if (ct + 1 == c_tiles) { ct = 0; ni++; }
                else                   { ct++; }
            } else {
                owt++;
            }
        } else {
            oh++;
        }
    }

    while (pending > 0) {
        y.write_response();
        pending--;
    }
}

void PoolingKernel(
    hls::burst_maxi<PoolWord> x,
    hls::burst_maxi<PoolWord> y,
    unsigned      batch,
    unsigned      channels,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      pool_h,
    unsigned      pool_w,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      dil_h,
    unsigned      dil_w,
    unsigned      pool_type,
    unsigned      lp_order,
    unsigned      count_include_pad
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    //
    // Two 128-bit hls::burst_maxi ports (PoolingKernel.h): gmem0 for the
    // read-only input, gmem1 for the write-only output.  All scalar
    // arguments go into the s_axilite ctrl register file.
    //
    // x: one request per (row, channel) run of <= kMaxLineBufCols elements
    //    (<= kLbWords + 1 words), up to kReadAhead of them in flight.
    // y: one request per (output row, channel) run of <= kLbWords + 1 words,
    //    at most kWriteInFlight un-acknowledged.
    //
    // depth=<N> is a C/RTL co-simulation hint only (POOL_COSIM_DEPTH_* in
    // PoolingKernel.h, shared with test/TestPoolingSim.cpp's cosim buffers).
    // -----------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi port=x  offset=slave bundle=gmem0 depth=POOL_COSIM_DEPTH_X_WORDS max_read_burst_length=16 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=y  offset=slave bundle=gmem1 depth=POOL_COSIM_DEPTH_Y_WORDS max_write_burst_length=16 num_write_outstanding=8
    #pragma HLS INTERFACE s_axilite port=x                 bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=y                 bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=channels          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_h             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_w             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_h            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_w            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_h          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_w          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_top           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_left          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dil_h             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dil_w             bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pool_type         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=lp_order          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=count_include_pad bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return            bundle=ctrl

    // -----------------------------------------------------------------------
    // STABLE: read-only m_axi base pointer x and every s_axilite scalar are
    // latched at ap_start and never written during the DATAFLOW region's
    // execution.  `y` is intentionally NOT listed because write_output_tile
    // writes through it during the dataflow region.
    // -----------------------------------------------------------------------
    #pragma HLS STABLE variable=x
    #pragma HLS STABLE variable=batch
    #pragma HLS STABLE variable=channels
    #pragma HLS STABLE variable=in_h
    #pragma HLS STABLE variable=in_w
    #pragma HLS STABLE variable=out_h
    #pragma HLS STABLE variable=out_w
    #pragma HLS STABLE variable=pool_h
    #pragma HLS STABLE variable=pool_w
    #pragma HLS STABLE variable=stride_h
    #pragma HLS STABLE variable=stride_w
    #pragma HLS STABLE variable=pad_top
    #pragma HLS STABLE variable=pad_left
    #pragma HLS STABLE variable=dil_h
    #pragma HLS STABLE variable=dil_w
    #pragma HLS STABLE variable=pool_type
    #pragma HLS STABLE variable=lp_order
    #pragma HLS STABLE variable=count_include_pad

    // -----------------------------------------------------------------------
    // Top-level DATAFLOW region.  geom is computed ONCE here (the runtime-
    // divisor divisions land in one shared divider set) and passed to all
    // four stages, which iterate (ni, ct, owt, oh, group) in lockstep.
    // -----------------------------------------------------------------------
    const PoolGeometry geom = compute_pool_geometry(
        channels, in_h, out_h, out_w, pool_h, pool_w,
        stride_h, stride_w, pad_top, dil_h, dil_w);

    #pragma HLS DATAFLOW

    // row_word_pipe: whole 128-bit row words, loader → emitter.  Deep enough
    // for the loader to run a few runs ahead of the emitter's row change.
    // LUTRAM: a 128-bit-wide FIFO in block RAM costs 4 BRAM18 for any depth
    // up to 512 (width-limited packing).
    hls::stream<PoolWord> row_word_pipe;
    #pragma HLS STREAM variable=row_word_pipe depth=128
    #pragma HLS BIND_STORAGE variable=row_word_pipe type=fifo impl=lutram

    // window_pipe: one MultiWindow per (khi, kwi) tap; a full window block
    // of slack lets the emitter stage the next group while the consumer
    // reduces the current one.
    hls::stream<MultiWindow> window_pipe;
    #pragma HLS STREAM variable=window_pipe depth=64
    #pragma HLS BIND_STORAGE variable=window_pipe type=fifo impl=lutram

    // denom_pipe: one MultiDenom per ow-group.
    hls::stream<MultiDenom> denom_pipe;
    #pragma HLS STREAM variable=denom_pipe depth=8

    // acc_stream: kTileC FinBundles (kOwParallel saturated outputs each)
    // per group; a few groups of slack between the consumer and the writer.
    hls::stream<FinBundle> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=64
    #pragma HLS BIND_STORAGE variable=acc_stream type=fifo impl=lutram

    row_loader(
        x, row_word_pipe,
        batch, channels, in_h, in_w, out_w,
        pool_w, stride_w, pad_left, dil_w,
        geom);

    window_emitter(
        row_word_pipe, window_pipe, denom_pipe,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w,
        pad_top, pad_left, dil_h, dil_w,
        pool_type, count_include_pad, geom);

    process_pool_kernel_tile(
        window_pipe, denom_pipe, acc_stream,
        batch, channels, in_w, out_h, out_w,
        pool_w, stride_w, pad_left, dil_w,
        pool_type, lp_order, geom);

    write_output_tile(
        y, acc_stream,
        batch, channels, in_w, out_h, out_w,
        pool_w, stride_w, pad_left, dil_w,
        geom);
}
