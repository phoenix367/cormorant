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
// w_cache geometry (§2.35).  The cache is kTileM columns (one RAM per m1,
// ARRAY_PARTITION complete on dim 1) of kWCacheWords WeightVecs; the bank,
// tile and kernel position are folded into ONE flat word address.  Keeping
// bank / tile / (khi, kwi) as separate array dimensions let HLS split the
// bank dimension into a second set of RAMs (2x BRAM) and lower the runtime-
// indexed prefetch store as two stores per RAM (II=2 on the sweep loop).
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
static_assert(kMaxKW <= kWCacheRowStride,            "w_cache row stride");
static_assert(kMaxKH * kWCacheRowStride <= kWCachePos, "w_cache tile stride");

inline unsigned w_cache_addr(unsigned bank, unsigned t, unsigned khi, unsigned kwi)
{
    #pragma HLS INLINE
    return (bank * kMaxMperGroup + t) * kWCachePos + khi * kWCacheRowStride + kwi;
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
// X-propagation guards (RTL-only failure modes, C-sim would see zeros):
//   * ic_l >= ic_valid — w_cache cells never written by the fill loop
//     (partial IC tile).  MUX the weight to 0 so the product is 0·0.
//   * m1   >= m_valid  — same for the last partial M tile.  Those acc
//     lanes are padding in the §2.23 word layout and are never drained,
//     but masking keeps 'X' out of the URAM word entirely.
// Both masks are one LUT per lane on the weight input; no DSP cost.
// ---------------------------------------------------------------------------
inline void mac_grid_step(
    const Data_t    p[kTileIC],
    const WeightVec w_cache[kTileM][kWCacheWords],
    unsigned        w_addr,                        // w_cache_addr(bank, t, khi, kwi)
    AccData_t       acc[kTileM],
    unsigned        ic_valid,
    unsigned        m_valid
) {
    #pragma HLS INLINE
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        const WeightVec w = w_cache[m1][w_addr];      // one word per m1 RAM
        AccData_t tree = 0;
        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
            #pragma HLS UNROLL
            const bool   ok    = (ic_l < ic_valid) && (m1 < m_valid);
            const Data_t w_val = ok ? w.lane[ic_l] : Data_t(0);
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
    const WeightVec w_buf[kTileM][kWCacheWords],   // bank 0, tile 0 used
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
        mac_grid_step(p, w_buf, w_cache_addr(0, 0, khi_cnt, kwi_cnt),
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
