#pragma once
// ---------------------------------------------------------------------------
// ConvMacGrid.h — the ConvKernel MAC array, isolated from all geometry.
//
// These two functions are the ONLY place standard / depthwise conv
// multiplies are instantiated.  They see a patch tile, a weight tile and
// an accumulator vector — never ni / oh / ow / chunk / mt / ict.  Every
// index that decides WHICH tile is computed lives in the producers and
// the consumer's outer nest in ConvKernel.cpp; every index in here is a
// compile-time lane number or a kernel-window counter.
//
// That boundary is what CONV_2D_GRID_PLAN.md §6.1 relies on: the 2-D grid
// (§4 of the plan) is developed and unit-tested (test/TestConvGrid.cpp)
// against a scalar reference on this interface alone, so grid work cannot
// introduce a geometry bug and geometry work cannot introduce a grid bug.
//
// Both functions are `inline` with `#pragma HLS INLINE` so HLS still
// flattens them into the consumer's pipelined loops exactly as when they
// were static functions in ConvKernel.cpp (§2.22 extraction was verified
// bit-exact and synthesis-neutral).
// ---------------------------------------------------------------------------

#include "Config.h"

// ---------------------------------------------------------------------------
// WeightVec — one kernel position's kTileIC input-channel lanes.  It is the
// weight stream beat (§2.32) AND the weight cache word: w_cache stores one
// WeightVec per (bank, tile, m1, khi, kwi), AGGREGATEd into a 256-bit RAM
// word, so a cache write is always one full-word store (§2.35: 16 separate
// lane stores into a reshaped word were inferred as a partial write and
// cost II=2).
// ---------------------------------------------------------------------------
struct WeightVec {
    Data_t lane[kTileIC];
};

// ---------------------------------------------------------------------------
// w_cache geometry (§2.35, §2.40).  The cache is kTileM columns (one RAM per
// m1, ARRAY_PARTITION complete on dim 1) of kWCacheWords WeightVecs; the
// bank, tile and kernel position are folded into ONE flat word address.
// Keeping bank / tile / (khi, kwi) as separate array dimensions let HLS split
// the bank dimension into a second set of RAMs (2x BRAM) and lower the
// runtime-indexed prefetch store as two stores per RAM (II=2 on the sweep
// loop).
//
// §2.40: the sweep reads all kTileM columns every cycle — kTileM x 256 bits
// = 4096 bits at kTileM = 16, and every BRAM36 / URAM port is 72 bits wide,
// so the column count, not the depth, sets the RAM count (4 BRAM36 or 4
// URAM per column, whatever the depth).  Doubling the columns in BRAM would
// have cost +32 BRAM36 on a design at 119/144; all 16 columns in URAM would
// be 64 = the whole pool.  The cache is therefore SPLIT into two arrays of
// identical shape: columns [0, kWCacheBramCols) stay in BRAM (the same 64
// BRAM18 as the 8-column cache) and columns [kWCacheBramCols, kTileM) live
// in URAM (4 blocks per column).  Both are addressed identically; the
// stores select the column with an explicit unrolled compare across the two
// arrays (w_cache_store), exactly as the one-array form did.
// ---------------------------------------------------------------------------
//
// Strides are powers of two so the address is a bit concatenation: with the
// natural kMaxKH * kMaxKW = 49 stride HLS built the address with a 3-cycle
// DSP multiply in front of the RAM read and the sweep loop grew from 7 to 9
// pipeline stages (+2 cycles per sweep drain, +4 % on 1x1 / small cases).
// The depth is unchanged for BRAM (392 -> 512 words rounds to the same
// 512-deep BRAM18 columns).
// ---------------------------------------------------------------------------
static constexpr unsigned kWCacheRowStride = 8;                            // >= kMaxKW
static constexpr unsigned kWCachePos       = 64;                           // >= kMaxKH * kWCacheRowStride
static constexpr unsigned kWCacheBanks     = 2;                            // ping-pong
static constexpr unsigned kWCacheWords     = kWCacheBanks * kMaxMperGroup * kWCachePos;
static constexpr unsigned kWCacheBramCols  = kTileM / 2;                   // columns [0, kWCacheBramCols) in BRAM
static constexpr unsigned kWCacheUramCols  = kTileM - kWCacheBramCols;     // the rest in URAM
static_assert(kMaxKW <= kWCacheRowStride,            "w_cache row stride");
static_assert(kMaxKH * kWCacheRowStride <= kWCachePos, "w_cache tile stride");
static_assert(kWCacheBramCols >= 1 && kWCacheUramCols >= 1, "w_cache column split");

inline unsigned w_cache_addr(unsigned bank, unsigned t, unsigned khi, unsigned kwi)
{
    #pragma HLS INLINE
    return (bank * kMaxMperGroup + t) * kWCachePos + khi * kWCacheRowStride + kwi;
}

// One full-word store into column m1 at word address wa.  Every RAM column
// sees exactly ONE conditional store (§2.35's rule: never a runtime index
// into the partitioned dimension).
inline void w_cache_store(
    WeightVec w_lo[kWCacheBramCols][kWCacheWords],
    WeightVec w_hi[kWCacheUramCols][kWCacheWords],
    unsigned m1, unsigned wa, const WeightVec& wv)
{
    #pragma HLS INLINE
    for (unsigned c = 0; c < kWCacheBramCols; c++) {
        #pragma HLS UNROLL
        if (c == m1) w_lo[c][wa] = wv;
    }
    for (unsigned c = 0; c < kWCacheUramCols; c++) {
        #pragma HLS UNROLL
        if (c + kWCacheBramCols == m1) w_hi[c][wa] = wv;
    }
}

// ---------------------------------------------------------------------------
// mac_grid_step — ONE kernel position on the kTileIC × kTileM grid (§2.24,
// §2.29).
//
// Fires all kTileIC × kTileM products for the patch column p[] against the
// weights of position (khi, kwi) of one m-tile: p[ic_l] is broadcast across
// the kTileM output-channel columns, each column reduces its kTileIC
// products through a private adder tree and adds the result into its own
// accumulator acc[m1].  Both grid axes are spatial.
//
// The caller pipelines over kernel positions (and, since §2.29, over the
// m-tiles of a group as well) at II=1: the only loop-carried recurrence is
// acc[m1] += tree — a lone AccData_t add at distance 1 (the depthwise step
// closes exactly the same recurrence).  The multiply + tree in front of it
// is feed-forward.
//
// Bit-exactness: AccData_t is wrap-around fixed point, so the per-column
// tree order gives results identical to any serial order.
//
// X-propagation guard (an RTL-only failure mode, C-sim would see zeros):
// columns m1 >= m_valid of a partial last M tile are never written by the
// fill, so their RAM words are 'X' in RTL.  The column's weight lanes are
// MUXed to 0 on the way into the multipliers (m1 is a compile-time lane
// number, m_valid a per-tile scalar: one 16-way AND per column that HLS
// folds into the DSP input registers) so acc[m1] keeps its defined value;
// those lanes are padding in the §2.23 word layout and are never drained.
// Masking the column's TREE instead (a 32-bit mux in front of the
// accumulator add) cost one more pipeline stage on the fused sweep loop
// (iteration latency 6 -> 7 = +1 cycle per pixel ramp) — §2.40 trap.
// Input-channel pad lanes need no mask (§2.40): every w_cache word that is
// ever read was written with all kTileIC lanes defined — the producer
// zero-initialises the vector and the packed DDR layout carries zeros in
// the lanes past in_ch (ConvKernel.h) — and the patch producer zero-pads
// its lanes >= ch_valid, so those products are 0·0.  The per-lane mask the
// 8-column grid carried cost 16 LUT per lane (4 k LUT at 256 lanes).
// ---------------------------------------------------------------------------
inline void mac_grid_step(
    const Data_t    p[kTileIC],
    const WeightVec w_lo[kWCacheBramCols][kWCacheWords],   // columns [0, kWCacheBramCols)
    const WeightVec w_hi[kWCacheUramCols][kWCacheWords],   // columns [kWCacheBramCols, kTileM)
    unsigned        w_addr,                        // w_cache_addr(bank, t, khi, kwi)
    AccData_t       acc[kTileM],
    unsigned        ic_valid,                      // unused since §2.40 (see above)
    unsigned        m_valid
) {
    #pragma HLS INLINE
    (void)ic_valid;
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        // One word per m1 RAM column; the array is a compile-time choice.
        WeightVec w;
        if (m1 < kWCacheBramCols) w = w_lo[m1][w_addr];
        else                      w = w_hi[m1 - kWCacheBramCols][w_addr];
        const bool col_ok = (m1 < m_valid);
        AccData_t tree = 0;
        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
            #pragma HLS UNROLL
            const Data_t w_val = col_ok ? w.lane[ic_l] : Data_t(0);
            // 16×16 Data_t multiply (one DSP48); the ap_fixed product of
            // two ap_fixed<16,8> is exactly AccData_t.
            tree += p[ic_l] * w_val;
        }
        acc[m1] += tree;
    }
}

// ---------------------------------------------------------------------------
// accumulate_standard — the whole kh × kw window for one m-tile from a
// patch buffer.  Kept as the unit-test surface (TestConvGrid.cpp) for the
// grid; the consumer itself uses mac_grid_step inside its fused
// (tile, khi, kwi) loop since §2.29.
// ---------------------------------------------------------------------------
inline void accumulate_standard(
    const Data_t    patch[kTileIC][kMaxKH][kMaxKW],
    const WeightVec w_lo[kWCacheBramCols][kWCacheWords],   // bank 0, tile 0 used
    const WeightVec w_hi[kWCacheUramCols][kWCacheWords],
    AccData_t    acc[kTileM],
    unsigned     ic_valid,
    unsigned     m_valid,
    unsigned     kh,
    unsigned     kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0;
    const unsigned ri_bound = kh * kw;
    for (unsigned ri = 0; ri < ri_bound; ri++) {
        #pragma HLS PIPELINE II=1
        Data_t p[kTileIC];
        #pragma HLS ARRAY_PARTITION variable=p complete dim=0
        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
            #pragma HLS UNROLL
            p[ic_l] = patch[ic_l][khi_cnt][kwi_cnt];
        }
        mac_grid_step(p, w_lo, w_hi, w_cache_addr(0, 0, khi_cnt, kwi_cnt),
                      acc, ic_valid, m_valid);
        if (++kwi_cnt == kw) {
            kwi_cnt = 0;
            if (++khi_cnt == kh) {
                khi_cnt = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mac_dw_step — ONE kernel position of the depthwise grid: kTileM
// independent lanes, acc[m1] += p[m1] * w[m1][pos], pos = khi*kw + kwi.
// The depthwise weight buffer is FLAT over the kernel window (§2.32: it is
// filled kWeightPortElems positions per beat straight from the packed DDR
// layout), so the caller keeps a running pos counter.  Per-lane RAW
// distance is 1 cycle; the ap_fixed<32,16> adder closes it at II=1.
// ---------------------------------------------------------------------------
static constexpr unsigned kMaxKPos = kMaxKH * kMaxKW;

inline void mac_dw_step(
    const Data_t p[kTileIC],
    const Data_t w_buf[kTileM][kMaxKPos],
    unsigned     pos,
    AccData_t    acc[kTileM]
) {
    #pragma HLS INLINE
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        acc[m1] += p[m1] * w_buf[m1][pos];
    }
}

// ---------------------------------------------------------------------------
// accumulate_depthwise — whole window from a patch buffer (unit-test
// surface; the consumer consumes PatchVec beats directly since §2.29).
// ---------------------------------------------------------------------------
inline void accumulate_depthwise(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kMaxKPos],
    AccData_t    acc[kTileM],
    unsigned     kh,
    unsigned     kw
) {
    #pragma HLS INLINE

    unsigned kwi_cnt = 0, khi_cnt = 0;
    const unsigned ri_bound_dw = kh * kw;
    for (unsigned ri = 0; ri < ri_bound_dw; ri++) {
        #pragma HLS PIPELINE II=1
        Data_t p[kTileIC];
        #pragma HLS ARRAY_PARTITION variable=p complete dim=0
        for (unsigned m1 = 0; m1 < kTileIC; m1++) {
            #pragma HLS UNROLL
            p[m1] = patch[m1][khi_cnt][kwi_cnt];
        }
        mac_dw_step(p, w_buf, ri, acc);
        if (++kwi_cnt == kw) {
            kwi_cnt = 0;
            ++khi_cnt;
        }
    }
}
