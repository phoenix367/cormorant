// ---------------------------------------------------------------------------
// PoolingKernel.cpp — 2-D pooling kernel.
//
// Implements the ONNX MaxPool / AveragePool / LpPool operators (and their
// Global variants) in a channel-tiled structure that maps cleanly to Vitis
// HLS synthesis.
//
// Top-level dataflow (HLS DATAFLOW):
//
//   PoolingKernel
//     input_window_producer  ──window_pipe──►  process_pool_kernel_tile
//                            ──denom_pipe ──►          │
//                                                      └──acc_stream──► write_output_tile
//                                                                              │
//     x (DDR gmem0)                                                           ▼
//                                                                       y (DDR gmem1)
//
// Stage responsibilities (post-§2.9 — kOwParallel-wide reduce):
//   * row_loader              — pure DDR reader, emits raw input pixels onto
//                              row_data_pipe (no on-chip buffer).
//   * window_emitter          — owns line_buf.  For each ow-group of size
//                              kOwParallel, gathers kOwParallel × kTileC
//                              pixels per (khi, kwi) into a MultiWindow and
//                              pushes it to window_pipe; pushes one
//                              MultiDenom (kOwParallel denom values) per
//                              group to denom_pipe.  Out-of-bounds spatial
//                              positions and channel-padding lanes are
//                              filled with the pool-type identity so the
//                              consumer's reduce needs no bounds check.
//                              Residual ow positions (when (ow_hi - ow_lo)
//                              is not a multiple of kOwParallel) are
//                              identity-padded — producer always emits
//                              exactly ceil((ow_hi-ow_lo)/kOwParallel) groups
//                              per (oh).
//   * process_pool_kernel_tile — drains pool_h*pool_w MultiWindows per
//                              ow-group into a parallel reducer running on
//                              acc[kOwParallel][kTileC]; one MultiDenom per
//                              group feeds kOwParallel inv_denom_lookup
//                              calls.  After reduction emits kOwParallel ×
//                              c_valid AccData_t results to acc_stream.
//   * write_output_tile      — saturates AccData_t → Data_t and writes to y
//                              with channel-major addressing.  Skips writes
//                              for ow ≥ ow_hi so the producer's identity
//                              padding never reaches DDR.
//
// Vectorised reduction (process_pool_kernel_tile):
//   The reduce loop runs ri ∈ [0, pool_h*pool_w) at II=1.  Each cycle reads
//   one MultiWindow (kOwParallel × kTileC pixels), and the inner update is
//   fully unrolled: acc[p][c1] receives one new value per cycle for every
//   (p, c1) ∈ [0, kOwParallel) × [0, kTileC).  Per-lane RAW distance is 1
//   cycle, covered by ap_fixed<32,16> compare/add/mul latency at 300 MHz.
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
// Per-invocation tile geometry.
//
// Both the W-tile width (compute_ow_tile divides by the runtime stride_w) and
// the W-tile count (ow_tiles_w divides by the runtime ow_tile) need an integer
// division by a RUNTIME divisor, which HLS synthesises as a multi-cycle
// sequential divider.  The geometry is invariant for a whole kernel
// invocation, yet each dataflow stage used to recompute ow_tiles_w (and
// c_tiles) itself — so the same divider was instantiated once per stage (four
// copies across the kernel).
//
// PoolingKernel now computes the geometry ONCE and passes this struct to every
// stage, collapsing the divider count to a single shared set.  The struct
// crosses the DATAFLOW process boundaries as one stable scalar channel.
// ---------------------------------------------------------------------------
struct PoolGeometry {
    unsigned c_tiles;     // (channels + kTileC - 1) / kTileC
    unsigned ow_tile;     // W-tile chunk width (compute_ow_tile)
    unsigned ow_tiles_w;  // number of W-tiles spanning out_w
};

static inline PoolGeometry compute_pool_geometry(
    unsigned channels,
    unsigned out_w,
    unsigned pool_w,
    unsigned stride_w,
    unsigned dil_w
) {
    PoolGeometry g;
    g.c_tiles    = (channels + kTileC - 1) / kTileC;
    g.ow_tile    = compute_ow_tile(out_w, pool_w, stride_w, dil_w);
    g.ow_tiles_w = (g.ow_tile > 0)
        ? ((out_w + g.ow_tile - 1) / g.ow_tile)
        : 1u;
    return g;
}

// ---------------------------------------------------------------------------
// row_loader — DATAFLOW source (Phase 1).
//
// Reads input rows from DDR and pushes them onto row_data_pipe in the order
// the window_emitter consumes them.  Owns no on-chip buffer: it's a thin DDR
// reader that lets the window_emitter run concurrently with row fetching
// (Phase 1 of oh+1 overlaps Phase 2 of oh).
//
// Loop nest is (ni, ct, owt, oh, ih, c_l, iw) with the SAME load_start /
// load_end / iw_load_lo / iw_load_hi schedule the window_emitter mirrors.
// Both functions derive these values from the geometry parameters
// independently — no metadata stream is needed because per-oh row counts
// are deterministic.
// ---------------------------------------------------------------------------
static void row_loader(
    hls::burst_maxi<PoolWord> x,
    hls::stream<Data_t>& row_data_pipe,
    unsigned             batch,
    unsigned             channels,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             pool_h,
    unsigned             pool_w,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             pad_top,
    unsigned             pad_left,
    unsigned             dil_h,
    unsigned             dil_w,
    const PoolGeometry&  geom
) {
    const unsigned c_tiles    = geom.c_tiles;
    const unsigned in_hw      = in_h * in_w;
    const unsigned ow_tile    = geom.ow_tile;
    const unsigned ow_tiles_w = geom.ow_tiles_w;

#ifdef DEBUG_LOAD_DATA_CACHING
    // Local to one kernel invocation — collects every DDR cell read.
    // Walked at function exit; any address with more than one record bumps
    // the global duplicate counter.  When in_w > kMaxLineBufCols, W-tile
    // boundary columns appear multiple times — the documented relaxation.
    PoolAddressMap_t read_addresses;
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                const int iw_start =
                    (int)(ow_lo * stride_w) - (int)pad_left;
                const int iw_end =
                    (int)((ow_hi - 1) * stride_w + (pool_w - 1) * dil_w)
                  - (int)pad_left;
                const int iw_load_lo = (iw_start < 0) ? 0 : iw_start;
                const int iw_load_hi = (iw_end >= (int)in_w)
                                     ? (int)in_w - 1 : iw_end;

                int last_loaded_row = -1;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    const int ih_window_max = (int)(oh * stride_h)
                                            - (int)pad_top
                                            + (int)((pool_h - 1) * dil_h);
                    int load_start = last_loaded_row + 1;
                    if (load_start < 0) load_start = 0;
                    int load_end = ih_window_max;
                    if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                    // Each (ih, c_l) is one contiguous element run
                    // [run_off, run_off + run_len).  All c_valid runs of
                    // the row are requested first (c_valid <= kTileC <=
                    // num_read_outstanding) so their DDR latency overlaps,
                    // then each run's words are drained and their in-range
                    // lanes pushed one per cycle — the rate window_emitter
                    // consumes them at.
                    const unsigned run_len = (unsigned)(iw_load_hi - iw_load_lo + 1);
                    for (int ih = load_start; ih <= load_end; ih++) {
                        for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned run_off =
                                (ni * channels + c_off + c_l) * in_hw
                              + (unsigned)ih * in_w + (unsigned)iw_load_lo;
                            x.read_request(run_off / kPoolPortElems,
                                           pool_words_for(run_off, run_len));
                        }
                        for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                            const unsigned run_off =
                                (ni * channels + c_off + c_l) * in_hw
                              + (unsigned)ih * in_w + (unsigned)iw_load_lo;
                            const unsigned shift   = run_off % kPoolPortElems;
                            const unsigned n_words = pool_words_for(run_off, run_len);
                            // Element (relative to the run) carried by lane 0
                            // of the current word; advances by kPoolPortElems
                            // per word.
                            int e0 = -(int)shift;
                            PoolWord word = 0;
                            const unsigned n_lanes = n_words * kPoolPortElems;
                            for (unsigned q = 0; q < n_lanes; q++) {
                                #pragma HLS PIPELINE II=1
                                const unsigned l = q % kPoolPortElems;
                                if (l == 0) word = x.read();
                                const int e = e0 + (int)l;
                                if (e >= 0 && e < (int)run_len) {
                                    row_data_pipe.write(pool_lane_to_data(word.range(
                                        kPoolDataBits * (l + 1) - 1, kPoolDataBits * l)));
#ifdef DEBUG_LOAD_DATA_CACHING
                                    PoolReadCounters c_rc;
                                    c_rc.ni  = ni;
                                    c_rc.ct  = ct;
                                    c_rc.c_l = c_l;
                                    c_rc.oh  = oh;
                                    c_rc.ow  = owt;
                                    c_rc.khi = (unsigned)ih;
                                    c_rc.kwi = (unsigned)(iw_load_lo + e);
                                    read_addresses[(std::size_t)run_off + (unsigned)e].push_back(c_rc);
#endif /* DEBUG_LOAD_DATA_CACHING */
                                }
                                if (l == kPoolPortElems - 1) e0 += (int)kPoolPortElems;
                            }
                        }
                    }
                    if (load_end > last_loaded_row) {
                        last_loaded_row = load_end;
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
// window_emitter — DATAFLOW stage (Phase 2).
//
// Owns line_buf.  Drains row_data_pipe into line_buf, then for each ow-group
// of size kOwParallel emits one MultiWindow vector per (khi, kwi) into
// window_pipe and one MultiDenom (kOwParallel denom values) into denom_pipe.
// Mirrors the row_loader's (ni, ct, owt, oh) schedule so stream consumption
// matches production.
//
// MultiDenom (kOwParallel valid_counts) is computed in closed form: a
// window position is in-bounds iff its row AND its column are in-bounds,
// and the two tests are independent, so the valid set is the rectangle
// product {valid khi} × {valid kwi} and valid_count = num_valid_kh *
// num_valid_kw.  num_valid_kh is counted once per group (it does not
// depend on the ow position); num_valid_kw is counted per p.
//
// Residual handling — when (ow_hi - ow_lo) is not a multiple of kOwParallel,
// the last group covers padded positions ow ≥ ow_hi.  Padded lanes get
// pool-type identity values so the consumer's reduce produces correct
// results for the in-range positions; the writer skips writes for ow ≥ ow_hi
// so the padded results never reach DDR.
//
// Constraints (validated by the inference scheduler):
//   (pool_w - 1) * dil_w + 1 <= kMaxLineBufCols   (single window fits)
//   (pool_h - 1) * dil_h + 1 <= kMaxLineBufRows   (vertical span fits)
//   kMaxLineBufRows is a power of two (slot = ih & (kMaxLineBufRows-1)).
//   kOwParallel is a power of two (group rounding compiles to bitwise AND).
// ---------------------------------------------------------------------------
static void window_emitter(
    hls::stream<Data_t>&       row_data_pipe,
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
    const unsigned c_tiles    = geom.c_tiles;
    const unsigned ow_tile    = geom.ow_tile;
    const unsigned ow_tiles_w = geom.ow_tiles_w;

    // line_buf:  kTileC * kMaxLineBufRows * kMaxLineBufCols * sizeof(Data_t)
    //         =      8  *       16        *       64        *      2     =  16 KB
    //
    // Partitioning has two axes (both required for kOwParallel reads/cycle):
    //
    //   * ARRAY_PARTITION dim=1 complete → kTileC independent BRAMs of
    //     [kMaxLineBufRows][kMaxLineBufCols], one per channel lane.  Lets
    //     the consumer's reduce update kTileC accumulators in parallel.
    //
    //   * ARRAY_PARTITION dim=3 cyclic factor=kOwParallel → each per-
    //     channel bank is column-cyclic-split into kOwParallel sub-banks,
    //     so kOwParallel adjacent ow positions land in different banks.
    //     Required at kOwParallel ≥ 4: ram_t2p alone (2 ports) cannot
    //     service 4 column reads from a single bank when stride_w=1.
    //
    //   * BIND_STORAGE type=ram_t2p → each sub-bank is a true-dual-port
    //     BRAM, so when stride_w=2 makes kOwParallel cols collide on
    //     fewer cyclic banks (gcd(stride_w, factor) > 1), the dual ports
    //     still deliver 2 reads per bank — together cyclic+ram_t2p
    //     handles every stride_w with gcd(stride_w, kOwParallel) ≤ 2.
    //     For stride_w=1: kOwParallel cols → kOwParallel distinct banks,
    //     1 port each.  For stride_w=2 at kOwParallel=4: 4 cols → 2
    //     banks × 2 ports = 4 reads.
    //
    // Resource cost at kOwParallel=4: kTileC × kOwParallel = 32 sub-
    // banks, each [kMaxLineBufRows][kMaxLineBufCols / kOwParallel] = 16×16
    // × 16 b ≈ 4 Kb → fits in 1 BRAM18 (or LUT-RAM) per sub-bank.
    static Data_t line_buf[kTileC][kMaxLineBufRows][kMaxLineBufCols];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=kOwParallel dim=3
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_t2p

    const Data_t pad_val = (pool_type == kPoolMax)
        ? Data_t(kDataMin)
        : Data_t(0);

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);

                const int iw_start =
                    (int)(ow_lo * stride_w) - (int)pad_left;
                const int iw_end =
                    (int)((ow_hi - 1) * stride_w + (pool_w - 1) * dil_w)
                  - (int)pad_left;
                const int iw_load_lo = (iw_start < 0) ? 0 : iw_start;
                const int iw_load_hi = (iw_end >= (int)in_w)
                                     ? (int)in_w - 1 : iw_end;

                int last_loaded_row = -1;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    const int ih_window_max = (int)(oh * stride_h)
                                            - (int)pad_top
                                            + (int)((pool_h - 1) * dil_h);

                    // -----------------------------------------------
                    // Phase 1 drain: pull row pixels from row_data_pipe
                    // into line_buf using the same (ih, c_l, iw) order
                    // the row_loader emits them.
                    // -----------------------------------------------
                    int load_start = last_loaded_row + 1;
                    if (load_start < 0) load_start = 0;
                    int load_end = ih_window_max;
                    if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                    for (int ih = load_start; ih <= load_end; ih++) {
                        const unsigned slot = (unsigned)ih & (kMaxLineBufRows - 1);
                        for (unsigned c_l = 0; c_l < c_valid; c_l++) {
                            for (int iw = iw_load_lo; iw <= iw_load_hi; iw++) {
                                #pragma HLS PIPELINE II=1
                                const unsigned local_iw =
                                    (unsigned)(iw - iw_load_lo);
                                line_buf[c_l][slot][local_iw] =
                                    row_data_pipe.read();
                            }
                        }
                    }
                    if (load_end > last_loaded_row) {
                        last_loaded_row = load_end;
                    }

                    // -----------------------------------------------
                    // Phase 2 emit (§2.9 — kOwParallel-wide):
                    //
                    // Iterate the ow range as ceil((ow_hi - ow_lo) /
                    // kOwParallel) groups; each group emits one
                    // MultiDenom and pool_h * pool_w MultiWindows.
                    //
                    // n_groups is derived rather than predicated on
                    // (ow < ow_hi) so the inner pipeline stays II=1.
                    // For the residual last group (when ow_hi - ow_lo
                    // is not a multiple of kOwParallel) the padded
                    // lanes (ow_g + p ≥ ow_hi) get pool-type identity
                    // values for the pixel data; the writer skips
                    // writes on those lanes so DDR never sees them.
                    // -----------------------------------------------
                    const unsigned ow_span  = ow_hi - ow_lo;
                    const unsigned n_groups =
                        (ow_span + kOwParallel - 1u) / kOwParallel;

                    for (unsigned g = 0; g < n_groups; g++) {
                        const unsigned ow_g = ow_lo + g * kOwParallel;

                        // -----------------------------------------------
                        // Per-position valid_count, closed form.
                        //
                        // A window position (khi, kwi) is in-bounds iff
                        // its row is in-bounds AND its column is — the row
                        // test depends only on khi, the column test only
                        // on kwi.  So the in-bounds set is the rectangle
                        // product {valid khi} × {valid kwi} and
                        //   valid_count = num_valid_kh * num_valid_kw.
                        // num_valid_kh does not depend on the ow position,
                        // so it is counted once per group; num_valid_kw is
                        // counted per p.  This replaces the former
                        // kOwParallel × kMaxPoolH × kMaxPoolW (98-lane)
                        // bounds-count with kMaxPoolH + kOwParallel ×
                        // kMaxPoolW (21-lane) counting plus kOwParallel
                        // multiplies.  Padded positions (ow_g + p ≥ ow_hi)
                        // compute a nominal count; the writer drops them.
                        // -----------------------------------------------
                        unsigned num_valid_kh = 0;
                        for (unsigned khi = 0; khi < kMaxPoolH; khi++) {
                            #pragma HLS UNROLL
                            if (khi < pool_h) {
                                const int ih_v = (int)(oh * stride_h + khi * dil_h)
                                               - (int)pad_top;
                                if (ih_v >= 0 && (unsigned)ih_v < in_h)
                                    num_valid_kh++;
                            }
                        }

                        MultiDenom md;
                        for (unsigned p = 0; p < kOwParallel; p++) {
                            #pragma HLS UNROLL
                            const unsigned ow_p = ow_g + p;
                            unsigned num_valid_kw = 0;
                            for (unsigned kwi = 0; kwi < kMaxPoolW; kwi++) {
                                #pragma HLS UNROLL
                                if (kwi < pool_w) {
                                    const int iw_v = (int)(ow_p * stride_w + kwi * dil_w)
                                                   - (int)pad_left;
                                    if (iw_v >= 0 && (unsigned)iw_v < in_w)
                                        num_valid_kw++;
                                }
                            }
                            md.d[p] = count_include_pad
                                ? (pool_h * pool_w)
                                : (num_valid_kh * num_valid_kw);
                        }
                        denom_pipe.write(md);

                        // -----------------------------------------------
                        // pool_h * pool_w MultiWindow emits per group.
                        // Each MultiWindow carries kOwParallel × kTileC
                        // pixels in parallel — line_buf reads execute on
                        // both ports of the per-channel ram_t2p banks
                        // (kOwParallel = 2 fits in two BRAM ports).
                        // -----------------------------------------------
                        for (unsigned khi = 0; khi < pool_h; khi++) {
                            const int ih =
                                (int)(oh * stride_h + khi * dil_h) - (int)pad_top;
                            const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                            const unsigned slot = ih_ok
                                ? ((unsigned)ih & (kMaxLineBufRows - 1))
                                : 0u;

                            for (unsigned kwi = 0; kwi < pool_w; kwi++) {
                                #pragma HLS PIPELINE II=1
                                MultiWindow mw;
                                for (unsigned p = 0; p < kOwParallel; p++) {
                                    #pragma HLS UNROLL
                                    const unsigned ow_p = ow_g + p;
                                    const int iw =
                                        (int)(ow_p * stride_w + kwi * dil_w)
                                      - (int)pad_left;
                                    const bool ow_ok = (ow_p < ow_hi);
                                    const bool spatial_ok =
                                        ih_ok && iw >= 0 &&
                                        (unsigned)iw < in_w && ow_ok;
                                    const unsigned local_iw = spatial_ok
                                        ? (unsigned)(iw - iw_load_lo)
                                        : 0u;

                                    for (unsigned c_l = 0; c_l < kTileC; c_l++) {
                                        #pragma HLS UNROLL
                                        const bool valid = spatial_ok && (c_l < c_valid);
                                        mw.lanes[p].lanes[c_l] = valid
                                            ? line_buf[c_l][slot][local_iw]
                                            : pad_val;
                                    }
                                }
                                window_pipe.write(mw);
                            }
                        }
                    } // ow_group loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// process_pool_kernel_tile — DATAFLOW processor (post-§2.9 — kOwParallel).
//
// Loop nest matches the producer: (ni, ct, owt, oh, ow_group) with ct OUTER
// of oh AND a W-tile dimension owt OUTER of oh.  ow_tile = out_w yields a
// single tile; ow_tile < out_w processes ow chunks in turn so the producer's
// line buffer stays bounded.
//
// Per (ni, ct, owt, oh, ow_group):
//   * Read MultiDenom (kOwParallel denom values) from denom_pipe; one
//     inv_denom_lookup per lane → kOwParallel ap_ufixed<24,1> reciprocals
//     ready before the reduce starts.
//   * Reduce: pool_h*pool_w iterations, each reading one MultiWindow from
//     window_pipe and updating ALL kOwParallel × kTileC accumulators in
//     parallel (fully unrolled inner double loop).  acc[p][c1] has 1-cycle
//     RAW distance per lane so HLS schedules at II=1 for MAX and at II≈L
//     for AVG/LP (L = ap_fixed<32,16> add latency).  Total reduce cost
//     drops from `pool_h * pool_w` cycles per output to `pool_h * pool_w`
//     per kOwParallel outputs.
//   * Finalise: per-lane multiply by inv_denom for AVG, poly_sqrt for LP-2,
//     identity otherwise; push kOwParallel × c_valid lanes to acc_stream as
//     AccData_t in (p, c1) order.  Writer saturates AccData_t → Data_t at
//     the boundary and skips writes for out-of-range positions (ow ≥ ow_hi).
// ---------------------------------------------------------------------------
static void process_pool_kernel_tile(
    hls::stream<MultiWindow>& window_pipe,
    hls::stream<MultiDenom>&  denom_pipe,
    hls::stream<AccData_t>&   acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                pool_h,
    unsigned                pool_w,
    unsigned                pool_type,
    unsigned                lp_order,
    const PoolGeometry&     geom
) {
    // acc[kOwParallel][kTileC] — both dimensions fully partitioned so the
    // unrolled reduce can update all kOwParallel * kTileC lanes in one
    // cycle.  dim=0 (= "all dims") gives the maximally-partitioned layout.
    AccData_t acc[kOwParallel][kTileC];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    const unsigned c_tiles    = geom.c_tiles;
    const unsigned ow_tile    = geom.ow_tile;
    const unsigned ow_tiles_w = geom.ow_tiles_w;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);
                const unsigned ow_span  = ow_hi - ow_lo;
                const unsigned n_groups =
                    (ow_span + kOwParallel - 1u) / kOwParallel;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned g = 0; g < n_groups; g++) {

                    // -----------------------------------------------
                    // Per-group reciprocal LUT lookups for kOwParallel
                    // positions in parallel.  inv_denom values feed the
                    // AVG finalize multiply (other pool types ignore).
                    // -----------------------------------------------
                    const MultiDenom md = denom_pipe.read();
                    InvDenom_t inv_denom[kOwParallel];
                    #pragma HLS ARRAY_PARTITION variable=inv_denom complete
                    for (unsigned p = 0; p < kOwParallel; p++) {
                        #pragma HLS UNROLL
                        inv_denom[p] = inv_denom_lookup(md.d[p]);
                    }

                    // -----------------------------------------------
                    // Initialise kOwParallel × kTileC accumulators
                    // (1 cycle, fully unrolled).
                    //   MAX: kAccMin sentinel (any valid input beats
                    //        it on the first comparison).
                    //   AVG / LP: 0 (sum starts at zero).
                    // -----------------------------------------------
                    for (unsigned p = 0; p < kOwParallel; p++) {
                        #pragma HLS UNROLL
                        for (unsigned c1 = 0; c1 < kTileC; c1++) {
                            #pragma HLS UNROLL
                            acc[p][c1] = (pool_type == kPoolMax)
                                ? AccData_t(kAccMin)
                                : AccData_t(0);
                        }
                    }

                    // -----------------------------------------------
                    // Vectorised reduce: one MultiWindow per cycle
                    // updates ALL kOwParallel × kTileC accumulators.
                    //
                    // ri runs 0 .. pool_h*pool_w - 1 (no kOwParallel
                    // factor — that's exactly what the second
                    // vectorisation axis buys).  All lane updates
                    // happen on the same cycle, fully unrolled.
                    //
                    // Per-lane RAW distance on acc[p][c1] is 1 cycle,
                    // so HLS schedules the reduce loop at II=1 for
                    // MAX and II≈L for AVG/LP (L = ap_fixed<32,16>
                    // add latency).  Cycle count per output drops by
                    // ~kOwParallel vs the §2.5 single-position reduce.
                    // -----------------------------------------------
                    const unsigned ri_bound = pool_h * pool_w;
                    for (unsigned ri = 0; ri < ri_bound; ri++) {
                        #pragma HLS PIPELINE II=1
                        const MultiWindow mw = window_pipe.read();
                        for (unsigned p = 0; p < kOwParallel; p++) {
                            #pragma HLS UNROLL
                            for (unsigned c1 = 0; c1 < kTileC; c1++) {
                                #pragma HLS UNROLL
                                const AccData_t val =
                                    AccData_t(mw.lanes[p].lanes[c1]);

                                if (pool_type == kPoolMax) {
                                    if (val > acc[p][c1]) acc[p][c1] = val;
                                } else if (pool_type == kPoolAvg) {
                                    acc[p][c1] += val;
                                } else {
                                    // LP: p_order=1 → |val|, p_order=2 → val²
                                    const AccData_t contrib = (lp_order == 1u)
                                        ? (val < AccData_t(0)
                                            ? AccData_t(-val) : val)
                                        : AccData_t(val * val);
                                    acc[p][c1] += contrib;
                                }
                            }
                        }
                    }

                    // -----------------------------------------------
                    // Finalise and push kOwParallel × c_valid lanes
                    // to acc_stream in (p outer, c1 inner) order so
                    // the writer drains them naturally per group.
                    //
                    //   MAX: identity — acc[p][c1] is the max.
                    //   AVG: multiply by precomputed ap_ufixed<24,1>
                    //        reciprocal of denom for lane p.
                    //   LP p_order=1: identity (Σ|x_i|).
                    //   LP p_order=2: poly_sqrt — fixed-point sqrt.
                    // -----------------------------------------------
                    for (unsigned p = 0; p < kOwParallel; p++) {
                        for (unsigned c1 = 0; c1 < c_valid; c1++) {
                            #pragma HLS PIPELINE II=1
                            AccData_t result;
                            if (pool_type == kPoolMax) {
                                result = acc[p][c1];
                            } else if (pool_type == kPoolAvg) {
                                result = AccData_t(acc[p][c1] * inv_denom[p]);
                            } else {
                                result = (lp_order == 1u)
                                    ? acc[p][c1]
                                    : poly_sqrt(acc[p][c1]);
                            }
                            acc_stream.write(result);
                        }
                    }

                    } // ow_group loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

// ---------------------------------------------------------------------------
// write_output_tile — DATAFLOW sink (post-§2.9 — kOwParallel groups).
//
// Drains acc_stream in the consumer's emit order — (ni, ct, owt, oh,
// ow_group, p, c1) where p ∈ [0, kOwParallel), c1 ∈ [0, c_valid).  Each
// ow-group covers kOwParallel adjacent ow positions.  Output addresses are
// non-contiguous in C (stride = out_h * out_w per channel); y_addr is
// advanced by a counter to avoid a multiplier inside the pipeline.
//
// Residual handling — when (ow_hi - ow_lo) is not a multiple of kOwParallel,
// the last group has padded lanes (ow ≥ ow_hi).  Their values flow through
// acc_stream in the same shape as in-range positions but we MUST NOT write
// them to DDR — masking the write with `ow < ow_hi` keeps the read off
// acc_stream un-conditional (so II=1 holds) while the conditional write
// merely gates the m_axi store.
//
// Saturates AccData_t → Data_t at the boundary.
// ---------------------------------------------------------------------------
static void write_output_tile(
    Data_t*                 y,
    hls::stream<AccData_t>& acc_stream,
    unsigned                batch,
    unsigned                channels,
    unsigned                out_h,
    unsigned                out_w,
    const PoolGeometry&     geom
) {
    const unsigned c_tiles    = geom.c_tiles;
    const unsigned hw_stride  = out_h * out_w;
    const unsigned ow_tile    = geom.ow_tile;
    const unsigned ow_tiles_w = geom.ow_tiles_w;

    for (unsigned ni = 0; ni < batch; ni++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const unsigned c_off   = ct * kTileC;
            const unsigned c_valid = std::min(kTileC, channels - c_off);
            const unsigned y_base  = (ni * channels + c_off) * hw_stride;

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const unsigned ow_lo = owt * ow_tile;
                const unsigned ow_hi = std::min(ow_lo + ow_tile, out_w);
                const unsigned ow_span  = ow_hi - ow_lo;
                const unsigned n_groups =
                    (ow_span + kOwParallel - 1u) / kOwParallel;

                for (unsigned oh = 0; oh < out_h; oh++) {
                    for (unsigned g = 0; g < n_groups; g++) {
                        const unsigned ow_g = ow_lo + g * kOwParallel;
                        for (unsigned p = 0; p < kOwParallel; p++) {
                            const unsigned ow      = ow_g + p;
                            const bool     ow_ok   = (ow < ow_hi);
                            unsigned y_addr = y_base + oh * out_w + ow;
                            for (unsigned c1 = 0; c1 < c_valid; c1++) {
                                #pragma HLS PIPELINE II=1
                                const Data_t v = saturate_cast<Data_t>(
                                    acc_stream.read());
                                if (ow_ok) y[y_addr] = v;
                                y_addr += hw_stride;
                            }
                        }
                    } // ow_group loop (within W-tile)
                } // oh loop
            } // owt loop
        } // c_tile loop
    } // batch loop
}

void PoolingKernel(
    hls::burst_maxi<PoolWord> x,
    Data_t*       y,
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
    // Two m_axi ports: gmem0 for the read-only input, gmem1 for the write-
    // only output.  All scalar arguments go into the s_axilite ctrl register
    // file accessed by the PS driver.
    //
    // depth=<N> is a C/RTL co-simulation hint only — it sizes the verification
    // adapter FIFO cosim builds for each m_axi port.  It does NOT constrain the
    // synthesised AXI master (runtime addresses) or the exported IP.  The
    // POOL_COSIM_DEPTH_* macros (PoolingKernel.h) are the single source of
    // truth shared with the test/TestPoolingSim.cpp cosim buffers.  cosim of an
    // m_axi kernel aborts without depth ("a depth specification is required for
    // interface port 'x'").
    // -----------------------------------------------------------------------
    // x: 128-bit hls::burst_maxi port (PoolingKernel.h); one request per
    // (row, channel) run of <= kMaxLineBufCols elements (<= 9 words), up to
    // kTileC of them in flight.
    #pragma HLS INTERFACE m_axi port=x  offset=slave bundle=gmem0 depth=POOL_COSIM_DEPTH_X_WORDS max_read_burst_length=16 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=y  offset=slave bundle=gmem1 depth=POOL_COSIM_DEPTH_Y
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
    // execution.  Marking them STABLE tells HLS not to insert auto-generated
    // synchronization stages or fan-out FIFOs into the producers — see the
    // ConvKernel.cpp note for the full rationale.  `y` is intentionally NOT
    // listed because write_output_tile writes through it during the dataflow
    // region.
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
    // Top-level DATAFLOW region.
    //
    //   window_pipe  carries kTileC * pool_h * pool_w pixels per output
    //                position; depth covers one full window block so the
    //                producer can stage the next tile while the consumer
    //                is reducing.
    //   denom_pipe   one entry per output position — small FIFO is enough.
    //   acc_stream   c_valid AccData_t entries per (ni, ct, owt, oh, ow);
    //                depth kTileC matches the writer's per-tile drain burst.
    //
    // geom bundles the per-invocation tile geometry (c_tiles, ow_tile,
    // ow_tiles_w) — computed ONCE here so the runtime-divisor divisions land
    // in a single shared divider set instead of one per dataflow stage.  It is
    // the single source of truth, passed to all four stages so they iterate
    // (ni, ct, owt, oh, ow) in lockstep.  When in_w <= kMaxLineBufCols the
    // formula yields ow_tile = out_w and the owt loop runs once: the kernel
    // matches the zero-duplication path.  When in_w > kMaxLineBufCols,
    // ow_tile < out_w and boundary input columns are re-read once per W-tile
    // transition — the documented relaxation.
    // -----------------------------------------------------------------------
    const PoolGeometry geom = compute_pool_geometry(
        channels, out_w, pool_w, stride_w, dil_w);

    #pragma HLS DATAFLOW

    // row_data_pipe carries raw input pixels from row_loader (DDR) to
    // window_emitter (line_buf cache).  Depth holds enough rows for the
    // window_emitter to lag a full Phase-1 row load behind the row_loader,
    // so Phase 1 of oh+1 overlaps Phase 2 of oh.  kTileC * kMaxLineBufRows
    // * kMaxLineBufCols (= 16 KB / 8 lanes * 64 cols ≈ 8192 entries at
    // depth, but FIFO stores a single Data_t per slot so HLS will use
    // BRAM/LUTRAM as appropriate).
    hls::stream<Data_t> row_data_pipe;
    #pragma HLS STREAM variable=row_data_pipe depth=kTileC*kMaxLineBufCols*4

    // window_pipe carries one MultiWindow struct (kOwParallel × kTileC
    // scalar lanes packed) per (khi, kwi) — pool_h*pool_w writes per
    // ow-group of kOwParallel adjacent output positions.  Depth holds a
    // full window block so the producer can stage the next ow-group while
    // the consumer is still reducing the current one.
    hls::stream<MultiWindow> window_pipe;
    #pragma HLS STREAM variable=window_pipe depth=kMaxPoolH*kMaxPoolW

    // denom_pipe carries one MultiDenom (kOwParallel valid_count values)
    // per ow-group — small FIFO is enough.
    hls::stream<MultiDenom> denom_pipe;
    #pragma HLS STREAM variable=denom_pipe depth=4

    // acc_stream carries kOwParallel × c_valid AccData_t entries per
    // (ni, ct, owt, oh, ow_group) — depth covers a full group's drain
    // burst so the consumer's finalise loop never stalls on writer back-
    // pressure.
    hls::stream<AccData_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kOwParallel*kTileC

    row_loader(
        x, row_data_pipe,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w,
        pad_top, pad_left, dil_h, dil_w,
        geom);

    window_emitter(
        row_data_pipe, window_pipe, denom_pipe,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w,
        pad_top, pad_left, dil_h, dil_w,
        pool_type, count_include_pad, geom);

    process_pool_kernel_tile(
        window_pipe, denom_pipe, acc_stream,
        batch, channels, out_h, out_w,
        pool_h, pool_w, pool_type, lp_order, geom);

    write_output_tile(
        y, acc_stream,
        batch, channels, out_h, out_w, geom);
}
