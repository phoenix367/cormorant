// ---------------------------------------------------------------------------
// MatmulKernel.cpp — tiled matrix multiplication kernel.
//
// This file contains the reference C++ implementation that mirrors the future
// Vitis HLS kernel structure exactly.  In Phase 2 the #pragma HLS annotations
// (shown as comments below) will be uncommented and the function will be
// synthesised for KV260.
//
// Loop structure (see doc/MATMUL_PLAN.md §2.5):
//
//   batch loop        — iterates over batch; advances a/b/c pointers by stride
//     n_tile loop     — tiles the N (output-row) dimension
//       load a_buf    — TILE_N burst reads of k elements (one per row)
//       m_tile loop   — tiles the M (output-col) dimension
//         clear acc   — zero TILE_N × TILE_M accumulators
//         k_tile loop — tiles the K (inner) dimension
//           load b_tile — TILE_K burst reads of TILE_M elements
//           ki loop   — II=1 K-reduction (TILE_N interleaved lanes)
//         write C     — saturate_cast acc → TILE_N burst writes of TILE_M elems
//
// II=1 strategy (§2.4):
//   The inner ki loop iterates k_valid × TILE_N times. Rotating the accumulator
//   lane (n1 = ki % TILE_N) ensures that the same acc[n1] register is only
//   written every TILE_N cycles — breaking the read-after-write hazard that
//   would otherwise prevent II=1.
// ---------------------------------------------------------------------------

#include <algorithm>
#include "MatmulKernel.h"

void MatmulKernel(
    hls::burst_maxi<MatmulWord> a,
    hls::burst_maxi<MatmulWord> b,
    Data_t*       c,
    unsigned      n,
    unsigned      k,
    unsigned      m,
    unsigned      batch,
    unsigned      a_batch_stride,
    unsigned      b_batch_stride,
    unsigned      c_batch_stride
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    //
    // Three m_axi ports keep A, B, and C reads/writes on separate AXI buses
    // so the tool can issue them concurrently.  All scalar arguments go into
    // the s_axilite ctrl register file accessed by the PS driver.
    //
    // depth=<N> is a C/RTL co-simulation hint only — it sizes the cosim
    // verification adapter FIFO per m_axi port and does NOT constrain the
    // synthesised AXI master or the exported IP.  The MATMUL_COSIM_DEPTH_*
    // macros (MatmulKernel.h) are the single source of truth; cosim of an
    // m_axi kernel aborts without a depth specification.
    // -----------------------------------------------------------------------
    // a / b: 128-bit hls::burst_maxi ports (MatmulKernel.h).  A rows are
    // requested as whole word ranges (up to kMaxK / lanes + 1 words, split
    // into <= 256-word requests); B tile rows are <= 3 words each and are
    // requested kBReqAhead rows ahead so their DDR latency overlaps.
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 depth=MATMUL_COSIM_DEPTH_A_WORDS max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 depth=MATMUL_COSIM_DEPTH_B_WORDS max_read_burst_length=16  num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 depth=MATMUL_COSIM_DEPTH_C
    #pragma HLS INTERFACE s_axilite port=a              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=n              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=k              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=m              bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch          bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=a_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c_batch_stride bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return         bundle=ctrl

    // -----------------------------------------------------------------------
    // On-chip buffers (BRAM in HLS).
    //
    // Declared static so that HLS infers BRAM rather than registers.
    // For C++ simulation the static storage persists across calls; this is safe
    // because every element is written before being read within each call.
    //
    // a_buf  — holds TILE_N complete rows of A (all k elements) for the
    //          current n_tile.  Loaded once; reused across all m_tiles and
    //          k_tiles.
    //
    //          ARRAY_PARTITION complete dim=1 → kTileN independent BRAMs,
    //          each kMaxK deep.  All TILE_N rows can be read in the same
    //          cycle (needed by the II=1 ki loop, which reads a_buf[n1][…]
    //          for a different n1 each iteration).
    //
    // b_tile — holds one TILE_K × TILE_M block of B for the current k_tile.
    //          Reloaded from DDR for each (m_tile, k_tile) pair.
    //
    //          ARRAY_PARTITION complete dim=2 → kTileM independent BRAMs,
    //          each kTileK deep.  All TILE_M columns can be read in the
    //          same cycle (the m1 unrolled loop reads kTileM elements per
    //          ki iteration).
    //
    // acc    — TILE_N × TILE_M accumulators.  Cleared at each m_tile; hold
    //          the partial dot products over the K dimension.
    //
    //          ARRAY_PARTITION complete dim=0 → all 64 elements as
    //          registers.  The m1-unrolled loop writes to kTileM of them
    //          per ki cycle; the n1 rotation means no two consecutive
    //          iterations share an acc element.
    // -----------------------------------------------------------------------
    // a_buf is [row][word][lane] so the kMatmulPortElems lanes of one A word
    // land in kMatmulPortElems different banks in one cycle (dims 1 and 3
    // partitioned complete); element ki of row n1 is a_buf[n1][ki / E][ki % E].
    static constexpr unsigned E = kMatmulPortElems;
    static Data_t    a_buf [kTileN][kMaxK / E][E];
    static Data_t    b_tile[kTileK][kTileM];
    static AccData_t acc   [kTileN][kTileM];

    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=b_tile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=acc    complete dim=0

    static constexpr unsigned kAReqWords  = 256;   // max_read_burst_length of a
    static constexpr unsigned kBReqAhead  = 16;    // num_read_outstanding of b

    // -----------------------------------------------------------------------
    // Batch loop — stride=0 on a or b means that pointer stays fixed (broadcasts).
    // -----------------------------------------------------------------------
    for (unsigned bi = 0; bi < batch; bi++) {
        const unsigned a_base = bi * a_batch_stride;   // element offsets
        const unsigned b_base = bi * b_batch_stride;
        Data_t*        c_ptr  = c + bi * c_batch_stride;

        // -------------------------------------------------------------------
        // N-tile loop — process TILE_N output rows per iteration.
        // -------------------------------------------------------------------
        const unsigned n_tiles = (n + kTileN - 1) / kTileN;
        for (unsigned n_tile = 0; n_tile < n_tiles; n_tile++) {
            const unsigned n_off   = n_tile * kTileN;
            const unsigned n_valid = std::min(kTileN, n - n_off);

            // ---------------------------------------------------------------
            // Load a_buf: TILE_N rows × k columns from A.
            // Each n1 iteration issues one burst read of k elements from DDR;
            // the inner ki loop pipelines at II=1 for back-to-back AXI beats.
            // Unused rows (n1 >= n_valid) are left with stale data — they
            // accumulate into acc lanes that are never written to C.
            // ---------------------------------------------------------------
            // Each row is one contiguous element run [row_off, row_off + k):
            // request its covering word range (in <= kAReqWords pieces) and
            // scatter every word's lanes into a_buf.  Lane l of word w holds
            // element (w_lo + w) * E + l - row_off; lanes outside [0, k)
            // (the partial first / last word) are dropped.  The bank of a
            // lane is (l - shift) mod E with shift = row_off mod E, so the
            // unrolled bank loop reads lane (j + shift) mod E for bank j.
            for (unsigned n1 = 0; n1 < n_valid; n1++) {
                const unsigned row_off = a_base + (n_off + n1) * k;
                const unsigned w_lo    = row_off / E;
                const unsigned shift   = row_off % E;
                const unsigned n_words = matmul_words_for(row_off, k);
                for (unsigned w0 = 0; w0 < n_words; w0 += kAReqWords) {
                    const unsigned cnt = std::min(kAReqWords, n_words - w0);
                    a.read_request(w_lo + w0, cnt);
                }
                for (unsigned w = 0; w < n_words; w++) {
                    #pragma HLS PIPELINE II=1
                    const MatmulWord word = a.read();
                    // Element index of lane 0 of this word, relative to the row.
                    const int e0 = (int)(w * E) - (int)shift;
                    for (unsigned j = 0; j < E; j++) {
                        #pragma HLS UNROLL
                        const unsigned l = (j + shift) % E;           // lane feeding bank j
                        const int      e = e0 + (int)l;               // element index, e % E == j
                        if (e >= 0 && e < (int)k) {
                            a_buf[n1][(unsigned)e / E][j] = matmul_lane_to_data(
                                word.range(kMatmulDataBits * (l + 1) - 1, kMatmulDataBits * l));
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // M-tile loop — process TILE_M output columns per iteration.
            // ---------------------------------------------------------------
            const unsigned m_tiles = (m + kTileM - 1) / kTileM;
            for (unsigned m_tile = 0; m_tile < m_tiles; m_tile++) {
                const unsigned m_off   = m_tile * kTileM;
                const unsigned m_valid = std::min(kTileM, m - m_off);

                // Clear accumulators for this (n_tile, m_tile) output block.
                // Both bounds are compile-time constants and acc is fully
                // partitioned — full unroll writes all 64 registers in 1 cycle.
                for (unsigned n1 = 0; n1 < kTileN; n1++) {
                    #pragma HLS UNROLL
                    for (unsigned m1 = 0; m1 < kTileM; m1++) {
                        #pragma HLS UNROLL
                        acc[n1][m1] = AccData_t(0);
                    }
                }

                // -----------------------------------------------------------
                // K-tile loop — accumulate one TILE_K slice of K per pass.
                // -----------------------------------------------------------
                const unsigned k_tiles = (k + kTileK - 1) / kTileK;
                for (unsigned k_tile = 0; k_tile < k_tiles; k_tile++) {
                    const unsigned k_off   = k_tile * kTileK;
                    const unsigned k_valid = std::min(kTileK, k - k_off);

                    // -------------------------------------------------------
                    // Load b_tile: k_valid rows × m_valid columns from B.
                    // Each k1 iteration issues one burst read of m_valid
                    // elements; the inner m1 loop pipelines at II=1.
                    // Partial last M-tile: only m_valid columns are loaded;
                    // remaining b_tile columns are stale (never read for C).
                    // -------------------------------------------------------
                    // Each tile row is one element run of m_valid <= kTileM
                    // elements at row_off = (k_off + k1) * m + m_off, i.e. at
                    // most ceil((E - 1 + kTileM) / E) words.  Requests for
                    // kBReqAhead rows are issued before their words are
                    // drained so consecutive rows' DDR latency overlaps.
                    for (unsigned r0 = 0; r0 < k_valid; r0 += kBReqAhead) {
                        const unsigned rn = std::min(kBReqAhead, k_valid - r0);
                        for (unsigned r = 0; r < rn; r++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned row_off = b_base + (k_off + r0 + r) * m + m_off;
                            b.read_request(row_off / E, matmul_words_for(row_off, m_valid));
                        }
                        for (unsigned r = 0; r < rn; r++) {
                            const unsigned k1      = r0 + r;
                            const unsigned row_off = b_base + (k_off + k1) * m + m_off;
                            const unsigned shift   = row_off % E;
                            const unsigned n_words = matmul_words_for(row_off, m_valid);
                            for (unsigned w = 0; w < n_words; w++) {
                                #pragma HLS PIPELINE II=1
                                const MatmulWord word = b.read();
                                // Lane l of word w is tile column (w * E + l - shift).
                                const int c0 = (int)(w * E) - (int)shift;
                                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                    #pragma HLS UNROLL
                                    const int l = (int)m1 - c0;
                                    if (l >= 0 && l < (int)E && m1 < m_valid) {
                                        b_tile[k1][m1] = matmul_lane_to_data(word.range(
                                            kMatmulDataBits * ((unsigned)l + 1) - 1,
                                            kMatmulDataBits * (unsigned)l));
                                    }
                                }
                            }
                        }
                    }

                    // -------------------------------------------------------
                    // K-reduction: II=1 pipelined loop.
                    //
                    // Iterates k_valid × TILE_N times.  Each group of TILE_N
                    // consecutive iterations processes one K element across
                    // all TILE_N row lanes (n1 = 0, 1, …, TILE_N-1).
                    //
                    //   n1 = ki % kTileN  — which row lane (rotates 0..TILE_N-1)
                    //   kk = ki / kTileN  — K index local to this K-tile
                    //
                    // The same acc[n1][m1] register is written every kTileN
                    // cycles (distance = kTileN ≥ MAC latency ≈ 3), breaking
                    // the RAW hazard that would otherwise prevent II=1.
                    //
                    // Since kTileN is a power of two, ki%kTileN is a bitwise
                    // AND and ki/kTileN is a right shift — no dividers in RTL.
                    //
                    // The inner m1 loop is fully unrolled: kTileM MAC units
                    // operate in parallel each cycle, one per output column.
                    // -------------------------------------------------------
                    const unsigned ki_bound = k_valid * kTileN;
                    for (unsigned ki = 0; ki < ki_bound; ki++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned n1  = ki % kTileN;
                        const unsigned kk  = ki / kTileN;
                        const unsigned kidx  = k_off + kk;
                        const Data_t   a_val = a_buf[n1][kidx / E][kidx % E];
                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            #pragma HLS UNROLL
                            acc[n1][m1] += AccData_t(a_val) * AccData_t(b_tile[kk][m1]);
                        }
                    }
                }

                // -----------------------------------------------------------
                // Write output block: saturate_cast acc → C.
                // n_valid sequential burst writes of m_valid elements each;
                // the inner m1 loop pipelines at II=1 for burst AXI writes.
                // -----------------------------------------------------------
                for (unsigned n1 = 0; n1 < n_valid; n1++) {
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        c_ptr[(n_off + n1) * m + (m_off + m1)] =
                            saturate_cast<Data_t>(acc[n1][m1]);
                    }
                }
            }
        }
    }
}
