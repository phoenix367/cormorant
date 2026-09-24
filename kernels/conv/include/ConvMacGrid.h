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
    const Data_t p[kTileIC],
    const Data_t w_tile[kTileM][kTileIC][kMaxKH][kMaxKW],
    unsigned     khi,
    unsigned     kwi,
    AccData_t    acc[kTileM],
    unsigned     ic_valid,
    unsigned     m_valid
) {
    #pragma HLS INLINE
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        AccData_t tree = 0;
        for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
            #pragma HLS UNROLL
            const bool   ok    = (ic_l < ic_valid) && (m1 < m_valid);
            const Data_t w_val = ok ? w_tile[m1][ic_l][khi][kwi] : Data_t(0);
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
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kTileIC][kMaxKH][kMaxKW],
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
        mac_grid_step(p, w_buf, khi_cnt, kwi_cnt, acc, ic_valid, m_valid);
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
// independent lanes, acc[m1] += p[m1] * w[m1][khi][kwi].  Per-lane RAW
// distance is 1 cycle; the ap_fixed<32,16> adder closes it at II=1.
// ---------------------------------------------------------------------------
inline void mac_dw_step(
    const Data_t p[kTileIC],
    const Data_t w_buf[kTileM][kMaxKH][kMaxKW],
    unsigned     khi,
    unsigned     kwi,
    AccData_t    acc[kTileM]
) {
    #pragma HLS INLINE
    for (unsigned m1 = 0; m1 < kTileM; m1++) {
        #pragma HLS UNROLL
        acc[m1] += p[m1] * w_buf[m1][khi][kwi];
    }
}

// ---------------------------------------------------------------------------
// accumulate_depthwise — whole window from a patch buffer (unit-test
// surface; the consumer consumes PatchVec beats directly since §2.29).
// ---------------------------------------------------------------------------
inline void accumulate_depthwise(
    const Data_t patch[kTileIC][kMaxKH][kMaxKW],
    const Data_t w_buf[kTileM][kMaxKH][kMaxKW],
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
        mac_dw_step(p, w_buf, khi_cnt, kwi_cnt, acc);
        if (++kwi_cnt == kw) {
            kwi_cnt = 0;
            ++khi_cnt;
        }
    }
}
