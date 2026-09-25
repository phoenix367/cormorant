// ---------------------------------------------------------------------------
// MatmulKernel.cpp — tiled matrix multiplication kernel.
//
// Loop structure (see doc/MATMUL_KERNEL.md §4):
//
//   batch loop        — iterates over batch; advances a/b/c offsets by stride
//     n_tile loop     — tiles the N (output-row) dimension
//       load a_buf    — n_valid rows of k elements, requests batched
//       m_tile loop   — tiles the M (output-col) dimension
//         clear acc   — zero TILE_N × TILE_M accumulators
//         k_tile loop — tiles the K (inner) dimension
//           load b_tile — one (m_tile, k_tile) block; when B is a single
//                         block (m_tiles == k_tiles == 1) only once per
//                         batch slice (once per call if b_batch_stride == 0)
//           ki loop   — II=1 K-reduction (TILE_N interleaved lanes)
//         write C     — saturate_cast acc → TILE_N burst writes of TILE_M elems
//
// II=1 strategy (§5 of the kernel doc):
//   The inner ki loop iterates k_valid × TILE_N times. Rotating the accumulator
//   lane (n1 = ki % TILE_N) ensures that the same acc[n1] register is only
//   written every TILE_N cycles — breaking the read-after-write hazard that
//   would otherwise prevent II=1.
// ---------------------------------------------------------------------------

#include <algorithm>
#include "MatmulKernel.h"

namespace {

constexpr unsigned E = kMatmulPortElems;

constexpr unsigned kAReqWords       = 256;  // max_read_burst_length of a
constexpr unsigned kAReqOutstanding = 4;    // num_read_outstanding of a
constexpr unsigned kBReqAhead       = 16;   // num_read_outstanding of b
constexpr unsigned kBReqWords       = 64;   // max_read_burst_length of b

// A row needs at most ceil((kMaxK / E + 1) / kAReqWords) requests; the
// batching below must always be able to admit at least one row.
static_assert((kMaxK / E + 1 + kAReqWords - 1) / kAReqWords <= kAReqOutstanding,
              "one A row must fit in the outstanding request window");

// ---------------------------------------------------------------------------
// load_b_tile — fetch the (m_tile, k_tile) block of B into b_tile.
//
// Packed (tile-major) B: the block is one contiguous run of k_valid * kTileM
// elements (matmul_packed_index); it is requested in <= kBReqWords pieces
// (all in flight at once: <= 512 / 64 = 8 <= num_read_outstanding) and the
// words are streamed straight in: word w is row w / kMatmulWordsPerTileRow,
// columns (w % kMatmulWordsPerTileRow) * E ...
//
// Row-major B: each tile row is one element run of m_valid <= kTileM
// elements at row_off = (k_off + k1) * m + m_off, i.e. at most
// ceil((E - 1 + kTileM) / E) words.  Requests for kBReqAhead rows are
// issued before their words are drained so consecutive rows' DDR latency
// overlaps.  Lane l of word w is tile column (w * E + l - shift).
// ---------------------------------------------------------------------------
void load_b_tile(hls::burst_maxi<MatmulWord>& b,
                 Data_t b_tile[kTileK][kTileM],
                 unsigned b_base, unsigned m_tile, unsigned k_tile,
                 unsigned k, unsigned m, unsigned b_packed)
{
    #pragma HLS INLINE
    const unsigned m_off   = m_tile * kTileM;
    const unsigned m_valid = std::min(kTileM, m - m_off);
    const unsigned k_off   = k_tile * kTileK;
    const unsigned k_valid = std::min(kTileK, k - k_off);

    if (b_packed) {
        const unsigned blk_off = b_base + (m_tile * k + k_off) * kTileM;
        const unsigned n_words = k_valid * kMatmulWordsPerTileRow;
        for (unsigned w0 = 0; w0 < n_words; w0 += kBReqWords) {
            #pragma HLS PIPELINE II=1
            b.read_request(blk_off / E + w0, std::min(kBReqWords, n_words - w0));
        }
        for (unsigned w = 0; w < n_words; w++) {
            #pragma HLS PIPELINE II=1
            const MatmulWord word = b.read();
            const unsigned   k1   = w / kMatmulWordsPerTileRow;
            const unsigned   part = w % kMatmulWordsPerTileRow;
            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                #pragma HLS UNROLL
                if (m1 / E == part) {
                    const unsigned l = m1 % E;
                    b_tile[k1][m1] = matmul_lane_to_data(word.range(
                        kMatmulDataBits * (l + 1) - 1, kMatmulDataBits * l));
                }
            }
        }
    } else {
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
    }
}

} // namespace

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
    unsigned      c_batch_stride,
    unsigned      b_packed
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
    // into <= 256-word requests, at most num_read_outstanding requests in
    // flight); B tile rows are <= 3 words each and are requested kBReqAhead
    // rows ahead so their DDR latency overlaps.
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 depth=MATMUL_COSIM_DEPTH_A_WORDS max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 depth=MATMUL_COSIM_DEPTH_B_WORDS max_read_burst_length=64  num_read_outstanding=16
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
    #pragma HLS INTERFACE s_axilite port=b_packed       bundle=ctrl
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
    //          k_tiles.  Laid out [row][word][lane] so the E lanes of one A
    //          word land in E different banks in one cycle (dims 1 and 3
    //          partitioned complete); element ki of row n1 is
    //          a_buf[n1][ki / E][ki % E].
    //
    // b_tile — holds one TILE_K × TILE_M block of B for the current k_tile.
    //          Reloaded from DDR for each (m_tile, k_tile) pair, or once per
    //          batch slice when the whole of B is one tile (fast path).
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
    static Data_t    a_buf [kTileN][kMaxK / E][E];
    static Data_t    b_tile[kTileK][kTileM];
    static AccData_t acc   [kTileN][kTileM];

    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=b_tile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=acc    complete dim=0

    const unsigned n_tiles = (n + kTileN - 1) / kTileN;
    const unsigned m_tiles = (m + kTileM - 1) / kTileM;
    const unsigned k_tiles = (k + kTileK - 1) / kTileK;

    // B-resident fast path: when B is a single (m_tile, k_tile) block it is
    // loaded once per batch slice (at that slice's first n_tile) instead of
    // once per (n_tile, m_tile, k_tile); with b_batch_stride == 0 (B
    // broadcasts) it is loaded once per call.
    const bool b_resident = (m_tiles == 1) && (k_tiles == 1);

    // -----------------------------------------------------------------------
    // Batch loop — stride=0 on a or b means that pointer stays fixed (broadcasts).
    // -----------------------------------------------------------------------
    for (unsigned bi = 0; bi < batch; bi++) {
        const unsigned a_base = bi * a_batch_stride;   // element offsets
        const unsigned b_base = bi * b_batch_stride;
        Data_t*        c_ptr  = c + bi * c_batch_stride;

        // B-resident fast path: the single block is (re)loaded only at the
        // first n_tile of a batch slice whose B differs from the previous
        // one (see load_b below); every other (n_tile, m_tile, k_tile)
        // reuses it.
        const bool load_b_this_slice = (bi == 0) || (b_batch_stride != 0);

        // -------------------------------------------------------------------
        // N-tile loop — process TILE_N output rows per iteration.
        // -------------------------------------------------------------------
        for (unsigned n_tile = 0; n_tile < n_tiles; n_tile++) {
            const unsigned n_off   = n_tile * kTileN;
            const unsigned n_valid = std::min(kTileN, n - n_off);

            // K-split across lanes (MATMUL_OPTIMISATION.md §5).  When the
            // n_tile has fewer than kTileN valid rows, the idle row lanes
            // take K-segments of the valid rows instead: lane n1 works on
            // row n1 / lpr, K-segment n1 % lpr, with lpr = lanes per row the
            // largest power of two with lpr * n_valid <= kTileN (4 / 2 / 1
            // for n_valid = 1 / 2 / 3-4).  The segment lanes of a row are
            // summed after the k_tile loop (exact: fixed-point adds are
            // modular, so the order of the K-sum does not change the bits).
            unsigned lpr_log = 0;
            for (unsigned c = 1; (1u << c) <= kTileN; c++) {
                #pragma HLS UNROLL
                if ((n_valid << c) <= kTileN) lpr_log = c;
            }
            const unsigned lpr = 1u << lpr_log;

            // ---------------------------------------------------------------
            // Load a_buf: n_valid rows × k columns from A.
            //
            // Each row is one contiguous element run [row_off, row_off + k):
            // its covering word range is requested in <= kAReqWords pieces.
            // The requests of as many rows as fit in the kAReqOutstanding
            // window are issued BEFORE any of them is drained, so the rows'
            // DDR latencies overlap (issuing more than the window holds
            // would stall the adapter until words are drained — a deadlock
            // here, since draining starts only after issuing).
            //
            // Draining scatters every word's lanes into a_buf.  Lane l of
            // word w holds element (w_lo + w) * E + l - row_off; lanes
            // outside [0, k) (the partial first / last word) are dropped.
            // The bank of a lane is (l - shift) mod E with
            // shift = row_off mod E, so the unrolled bank loop reads lane
            // (j + shift) mod E for bank j.
            // Unused rows (n1 >= n_valid) are left with stale data — they
            // accumulate into acc lanes that are never written to C.
            // ---------------------------------------------------------------
            for (unsigned r0 = 0; r0 < n_valid; ) {
                unsigned reqs = 0;
                unsigned r1   = r0;
                for (; r1 < n_valid; r1++) {
                    const unsigned row_off = a_base + (n_off + r1) * k;
                    const unsigned n_words = matmul_words_for(row_off, k);
                    const unsigned n_reqs  = (n_words + kAReqWords - 1) / kAReqWords;
                    if (reqs + n_reqs > kAReqOutstanding) break;
                    for (unsigned w0 = 0; w0 < n_words; w0 += kAReqWords) {
                        #pragma HLS PIPELINE II=1
                        a.read_request(row_off / E + w0, std::min(kAReqWords, n_words - w0));
                    }
                    reqs += n_reqs;
                }
                for (unsigned n1 = r0; n1 < r1; n1++) {
                    const unsigned row_off = a_base + (n_off + n1) * k;
                    const unsigned shift   = row_off % E;
                    const unsigned n_words = matmul_words_for(row_off, k);
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
                r0 = r1;
            }

            // ---------------------------------------------------------------
            // M-tile loop — process TILE_M output columns per iteration.
            // ---------------------------------------------------------------
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
                for (unsigned k_tile = 0; k_tile < k_tiles; k_tile++) {
                    const unsigned k_off   = k_tile * kTileK;
                    const unsigned k_valid = std::min(kTileK, k - k_off);

                    // ONE call site: load_b_tile is inlined and its
                    // row-major scatter is the largest LUT consumer of the
                    // kernel (~18 k); a second call site for the fast path
                    // duplicated it (44 k -> 64 k LUT).
                    const bool load_b = !b_resident ||
                                        (n_tile == 0 && load_b_this_slice);
                    if (load_b)
                        load_b_tile(b, b_tile, b_base, m_tile, k_tile, k, m, b_packed);

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
                    // The multiply is Data_t × Data_t: for ap_fixed<16,8>
                    // operands the product type IS ap_fixed<32,16> (exact,
                    // no rounding), so accumulating it into AccData_t is
                    // bit-identical to widening both operands first — and
                    // costs one 16×16 DSP instead of a 32×32 multiply.
                    // -------------------------------------------------------
                    // With the K-split, lane n1 = (row n1 >> lpr_log,
                    // segment n1 & (lpr - 1)) covers the local K range
                    // [seg * seg_len, seg * seg_len + seg_len) of this tile,
                    // seg_len = ceil(k_valid / lpr); the loop runs seg_len
                    // × kTileN iterations and the guard `kl < k_valid` idles
                    // the lanes whose segment overruns the tile (the last
                    // segment when lpr does not divide k_valid).  Each
                    // iteration still reads ONE a_buf element and ONE
                    // b_tile row (the same lane and RAM ports as without
                    // the split), so II=1 is unchanged.
                    const unsigned seg_len  = (k_valid + lpr - 1) >> lpr_log;
                    unsigned seg_base[kTileN];
                    #pragma HLS ARRAY_PARTITION variable=seg_base complete dim=0
                    for (unsigned s = 0; s < kTileN; s++) {
                        #pragma HLS UNROLL
                        seg_base[s] = s * seg_len;
                    }
                    const unsigned ki_bound = seg_len * kTileN;
                    for (unsigned ki = 0; ki < ki_bound; ki++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned n1  = ki % kTileN;
                        const unsigned kk  = ki / kTileN;
                        const unsigned row = n1 >> lpr_log;
                        const unsigned seg = n1 & (lpr - 1);
                        const unsigned kl  = seg_base[seg] + kk;   // K index local to this tile
                        if (kl < k_valid) {
                            const unsigned kidx  = k_off + kl;
                            const Data_t   a_val = a_buf[row][kidx / E][kidx % E];
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[n1][m1] += a_val * b_tile[kl][m1];
                            }
                        }
                    }
                }

                // -----------------------------------------------------------
                // K-split reduction: fold the segment lanes of every row into
                // lane row * lpr — a log2(kTileN)-stage tree on the acc
                // registers, each stage active only when lpr > stride
                // (all in one cycle, fully unrolled).
                // -----------------------------------------------------------
                for (unsigned stride = 1; stride < kTileN; stride *= 2) {
                    #pragma HLS UNROLL
                    if (lpr > stride) {
                        for (unsigned n1 = 0; n1 < kTileN; n1 += 2 * stride) {
                            #pragma HLS UNROLL
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[n1][m1] += acc[n1 + stride][m1];
                            }
                        }
                    }
                }

                // -----------------------------------------------------------
                // Write output block: saturate_cast acc → C.
                // n_valid sequential burst writes of m_valid elements each;
                // the inner m1 loop pipelines at II=1 for burst AXI writes.
                // Row n1's sum sits in lane n1 * lpr.
                // -----------------------------------------------------------
                for (unsigned n1 = 0; n1 < n_valid; n1++) {
                    const unsigned lane = n1 << lpr_log;
                    for (unsigned m1 = 0; m1 < m_valid; m1++) {
                        #pragma HLS PIPELINE II=1
                        c_ptr[(n_off + n1) * m + (m_off + m1)] =
                            saturate_cast<Data_t>(acc[lane][m1]);
                    }
                }
            }
        }
    }
}
