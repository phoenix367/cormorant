// ---------------------------------------------------------------------------
// MatmulKernel.cpp — tiled matrix multiplication kernel.
//
// Loop structure (see doc/kernels/MATMUL_KERNEL.md §4):
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
//
// gemv_kw != 0 takes the GEMV streaming path instead (gemv_run below,
// MatmulKernel.h "GEMV streaming mode", MATMUL_OPTIMISATION.md §8b).
// ---------------------------------------------------------------------------

#include <algorithm>
#ifndef __SYNTHESIS__
#include <cassert>
#endif
#include "hls_stream.h"
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
// A packed B block (kTileK rows of kMatmulWordsPerTileRow words) is requested
// all at once in kBReqWords pieces; they must all fit in flight.
static_assert(kTileK * kMatmulWordsPerTileRow <= kBReqAhead * kBReqWords,
              "a packed B block must fit in the outstanding request window");

// ---------------------------------------------------------------------------
// B block fetch cursor — b_tile ping-pong (MATMUL_OPTIMISATION.md §7).
//
// b_tile has two banks; while the K-loop multiplies out of bank `cur`, the
// block needed by the NEXT (bi, n_tile, m_tile, k_tile) iteration that loads
// B is streamed into bank !cur, one word per K-loop iteration.  The cursor
// below is that stream's state.  Rows are the unit of progress for both
// layouts: a row-major block is k_valid DDR rows of <= kMatmulMaxRowWords each (one
// read request per row, <= kBReqAhead rows in flight); a packed block is
// one contiguous run whose requests are all issued up front, drained as
// k_valid "rows" of kMatmulWordsPerTileRow words.  A word's lanes are
// scattered into the column RAMs after ONE rotate by the row's lane shift
// (0 for packed), with fixed lane -> column wiring (§6).
// ---------------------------------------------------------------------------
// The counters are bounded ap_uint types: with plain `unsigned` HLS keeps
// 32-bit compares / adds and pipeline registers for each of them in the
// K-loop (CONV_OPTIMISATION.md §2.35 "narrow prefetch cursors").
typedef ap_uint<matmul_bits_for(kTileK)>             BRowCnt;   // 0 .. kTileK
typedef ap_uint<matmul_bits_for(kTileM)>             BColCnt;   // 0 .. kTileM
typedef ap_uint<matmul_bits_for(kMatmulMaxRowWords)> BWordCnt;  // 0 .. words per row
typedef ap_uint<matmul_bits_for(E - 1)>              BShift;    // 0 .. E - 1
struct BFetch {
    unsigned bi, nt, mt, kt;     // iteration whose block is being fetched (C-sim check only)
    BRowCnt  k_valid;            // rows in the block
    BColCnt  m_valid;            // valid columns in the block
    BRowCnt  rows_req;           // rows requested so far (packed: k_valid at once)
    BRowCnt  rows_done;          // rows fully drained into b_tile
    unsigned req_off;            // element offset of the next row to request
    unsigned drain_off;          // element offset of the row being drained
    unsigned row_stride;         // elements between consecutive rows
    BWordCnt w, rwords;          // word cursor within the row being drained / its word count
    BShift   shift;              // lane shift of the row being drained (0 for packed)
};

// Point the cursor at block (bi, mt, kt) and, for the packed layout, issue
// all of its read requests (kTileK * kMatmulWordsPerTileRow words in
// <= kBReqAhead pieces of kBReqWords — 1024 words in 16 pieces at kTileM = 32).
inline void b_fetch_start(hls::burst_maxi<MatmulWord>& b, BFetch& f,
                          unsigned bi, unsigned nt, unsigned mt, unsigned kt,
                          unsigned k, unsigned m, unsigned b_batch_stride,
                          unsigned b_packed)
{
    #pragma HLS INLINE
    f.bi = bi; f.nt = nt; f.mt = mt; f.kt = kt;
    const unsigned b_base = bi * b_batch_stride;
    const unsigned k_off  = kt * kTileK;
    const unsigned m_off  = mt * kTileM;
    f.k_valid   = std::min(kTileK, k - k_off);
    f.m_valid   = std::min(kTileM, m - m_off);
    f.rows_done = 0;
    f.w         = 0;
    if (b_packed) {
        // Tile-major B: the block is one contiguous run of k_valid * kTileM
        // elements (matmul_packed_index); word w is row w / kMatmulWordsPerTileRow.
        const unsigned blk_off = b_base + (mt * k + k_off) * kTileM;
        const unsigned n_words = (unsigned)f.k_valid * kMatmulWordsPerTileRow;
        for (unsigned w0 = 0; w0 < n_words; w0 += kBReqWords) {
            #pragma HLS PIPELINE II=1
            b.read_request(blk_off / E + w0, std::min(kBReqWords, n_words - w0));
        }
        f.rows_req   = f.k_valid;
        f.req_off    = blk_off;
        f.drain_off  = blk_off;
        f.row_stride = kTileM;
        f.rwords     = kMatmulWordsPerTileRow;
        f.shift      = 0;
    } else {
        // Row-major B: row k1 of the block is the element run
        // [off + k1 * m, + m_valid), requested row by row in b_fetch_step.
        const unsigned off = b_base + k_off * m + m_off;
        f.rows_req   = 0;
        f.req_off    = off;
        f.drain_off  = off;
        f.row_stride = m;
        f.rwords     = matmul_words_for(off, (unsigned)f.m_valid);
        f.shift      = off % E;
    }
}

// One step of the fetch: (row-major) request the next row if fewer than
// kBReqAhead rows are in flight, then drain ONE word of the current row
// into bank `bank` of b_tile.  Called once per K-loop iteration; b.read()
// blocks only when the word has not arrived yet.
inline void b_fetch_step(hls::burst_maxi<MatmulWord>& b, BFetch& f,
                         Data_t b_tile[kTileM][2 * kTileK], ap_uint<1> bank,
                         unsigned b_packed)
{
    #pragma HLS INLINE
    if (!b_packed && f.rows_req < f.k_valid &&
        (unsigned)(f.rows_req - f.rows_done) < kBReqAhead) {
        b.read_request(f.req_off / E, matmul_words_for(f.req_off, (unsigned)f.m_valid));
        f.rows_req++;
        f.req_off += f.row_stride;
    }
    if (f.rows_done < f.rows_req) {
        // Lane l of word w is tile column w * E + l - shift.  After the
        // rotate, column m1 (j = m1 % E, p = m1 / E) always takes lane j;
        // it belongs to THIS word iff p == w and j + shift < E, or
        // p == w - 1 and j + shift >= E.  One conditional store per column
        // RAM per step, at the flat (bank, row) address.
        const MatmulWord rot  = matmul_rotate_lanes(b.read(), f.shift);
        const unsigned   addr = (unsigned)bank * kTileK + (unsigned)f.rows_done;
        const unsigned   fw   = f.w;
        const unsigned   fsh  = f.shift;
        const unsigned   fmv  = f.m_valid;
        for (unsigned m1 = 0; m1 < kTileM; m1++) {
            #pragma HLS UNROLL
            const unsigned j    = m1 % E;
            const unsigned p    = m1 / E;
            const bool     wrap = (j + fsh) >= E;
            const bool     hit  = wrap ? (p + 1 == fw) : (p == fw);
            if (hit && m1 < fmv) b_tile[m1][addr] = matmul_word_lane(rot, j);
        }
        if (++f.w == f.rwords) {
            f.w = 0;
            f.rows_done++;
            f.drain_off += f.row_stride;
            f.rwords = b_packed ? kMatmulWordsPerTileRow
                                : matmul_words_for(f.drain_off, (unsigned)f.m_valid);
            f.shift  = b_packed ? 0u : f.drain_off % E;
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV streaming path (gemv_kw != 0) — MATMUL_OPTIMISATION.md §8b.
//
// One DATAFLOW region of five processes.  Each loops over the same jobs —
// (batch slice bi, A row r, column chunk ch) — and exchanges words through
// FIFOs:
//
//   gemv_read_a   port a: the A row (x) -> xs0 and xs1, then the planes
//                 [0, split) of B's image -> ws0 (port a reaches B at
//                 a_to_b bytes from its base: the host's b - a)
//   gemv_read_b   port b: the planes [split, planes) -> ws1
//   gemv_mac<S>   per B word E products (the kw x taps of the word's plane
//                 times its lanes), summed in groups of kw lanes into E / kw
//                 output columns, accumulated on chip over the stream's
//                 planes; then the chunk's partial sums -> ps<S>
//   gemv_write    port c: ps0 + ps1, saturated, row-major
//
// B's image is [plane c][column p][tap j] (MatmulKernel.h): a word holds E
// consecutive (p, j) of ONE plane (m % E == 0), i.e. E / kw columns times kw
// taps.  With one column chunk a stream's planes are one contiguous run of
// DDR; with several, one run of mc * kw elements per plane.  The two
// partial sums are exact (modular fixed-point adds), so C is bit-identical
// to the tiled path's.
// ---------------------------------------------------------------------------
constexpr unsigned kGemvLogE = matmul_bits_for(E) - 1;   // log2(E)
constexpr unsigned kGemvTile = 16;   // ConvKernel's input-channel tile: the image's lane groups
// Accumulator words: one per word of a plane — a plane of a chunk spans
// mc * kw / E <= kGemvMaxM words (kw <= E).
constexpr unsigned kGemvAccWords = kGemvMaxM > 0 ? kGemvMaxM : 1;
static_assert((1u << kGemvLogE) == E, "lanes per word must be a power of two");
static_assert(kGemvMaxM % E == 0, "kGemvMaxM must be a multiple of the lanes per word");
static_assert(kGemvMinPlane / E >= E,
              "a plane must span >= E words: the next plane's taps are fetched over its "
              "first kw words, and the accumulator RMW needs the distance");

struct GemvAcc { AccData_t v[E]; };   // one accumulator word

// Stream n_runs runs of run_words words (run r starts at word base + r *
// stride) from `port` into `ws`: requests of <= ReqWords words, at most Out
// in flight.  Issuing and draining share one II=1 loop — issuing past the
// window before draining would block the adapter for good.  The next
// request (address, length) is always ready in registers, so read_request
// takes no arithmetic in its cycle.  A request ends at a multiple of
// ReqWords words into its run or at the run's end, which the drain side
// recognises to retire it.
template<unsigned ReqWords, unsigned Out>
void gemv_stream_runs(hls::burst_maxi<MatmulWord>& port, hls::stream<MatmulWord>& ws,
                      unsigned long long base, unsigned stride, unsigned run_words,
                      unsigned n_runs)
{
    #pragma HLS INLINE
    static_assert((ReqWords & (ReqWords - 1)) == 0, "ReqWords must be a power of two");
    const unsigned total = run_words * n_runs;
    unsigned long long q_addr = base, run_addr = base;    // the next request
    unsigned q_len  = std::min(ReqWords, run_words);
    unsigned q_left = run_words - q_len;                   // words of its run after it
    unsigned q_runs = n_runs;                              // runs not fully requested
    unsigned pend = 0, inflight = 0;       // words requested, not read / requests not retired
    unsigned d_off = 0;                    // read offset within its run
    for (unsigned t = 0; t < total; ) {
        #pragma HLS PIPELINE II=1
        const bool     issue = q_runs != 0 && inflight < Out;
        const bool     drain = pend != 0;
        const unsigned len   = q_len;
        if (issue) {
            port.read_request(q_addr, len);
            if (q_left != 0) {
                const unsigned nl = std::min(ReqWords, q_left);
                q_addr += len;
                q_len   = nl;
                q_left -= nl;
            } else {
                q_runs--;
                run_addr += stride;
                q_addr = run_addr;
                q_len  = std::min(ReqWords, run_words);
                q_left = run_words - q_len;
            }
        }
        bool retire = false;
        if (drain) {
            ws.write(port.read());
            t++;
            const bool run_end = d_off + 1 == run_words;
            retire = run_end || ((d_off + 1) & (ReqWords - 1)) == 0;
            d_off  = run_end ? 0u : d_off + 1;
        }
        pend     = pend + (issue ? len : 0u) - (drain ? 1u : 0u);
        inflight = inflight + (issue ? 1u : 0u) - (retire ? 1u : 0u);
    }
}

// The runs of planes [c_lo, c_hi) of batch slice b_base (element offset)
// for column chunk [p0, p0 + mcc) — see the section comment.  `to_b` is the
// word offset of B from the port's base (a_to_b / 16 on port a, 0 on b).
template<unsigned ReqWords, unsigned Out>
void gemv_stream_planes(hls::burst_maxi<MatmulWord>& port, hls::stream<MatmulWord>& ws,
                        long long to_b, unsigned b_base, unsigned m, unsigned lk,
                        unsigned c_lo, unsigned c_hi, unsigned chunks, unsigned p0,
                        unsigned mcc)
{
    #pragma HLS INLINE
    const unsigned plane = m << lk;                 // elements per plane
    const bool     one   = chunks == 1;
    const unsigned np    = c_hi - c_lo;
    const unsigned long long base =
        (unsigned long long)(to_b + (b_base + c_lo * plane + (p0 << lk)) / E);
    const unsigned run_w = one ? (np * plane) / E : (mcc << lk) / E;
    gemv_stream_runs<ReqWords, Out>(port, ws, base, plane / E, run_w, one ? 1u : np);
}

static void gemv_read_a(hls::burst_maxi<MatmulWord>& a, hls::stream<MatmulWord>& xs0,
                        hls::stream<MatmulWord>& xs1, hls::stream<MatmulWord>& ws0,
                        long long a_to_b, unsigned n, unsigned k, unsigned m, unsigned batch,
                        unsigned a_stride, unsigned b_stride,
                        unsigned lk, unsigned split, unsigned chunks, unsigned mc)
{
    const long long to_b = a_to_b >> 4;             // B's word offset on port a
    const unsigned  xw   = k / E;
    for (unsigned bi = 0; bi < batch; bi++) {
        for (unsigned r = 0; r < n; r++) {
            for (unsigned ch = 0; ch < chunks; ch++) {
                // The A row: <= kMaxK / E words, within the request window
                // (static_assert at the top), requested before draining.
                const unsigned x_base = (bi * a_stride + r * k) / E;
                for (unsigned w0 = 0; w0 < xw; w0 += kAReqWords) {
                    #pragma HLS PIPELINE II=1
                    a.read_request(x_base + w0, std::min(kAReqWords, xw - w0));
                }
                for (unsigned w = 0; w < xw; w++) {
                    #pragma HLS PIPELINE II=1
                    const MatmulWord v = a.read();
                    xs0.write(v);
                    xs1.write(v);
                }
                const unsigned p0 = ch * mc;
                gemv_stream_planes<kAReqWords, kAReqOutstanding>(
                    a, ws0, to_b, bi * b_stride, m, lk, 0, split, chunks, p0,
                    std::min(mc, m - p0));
            }
        }
    }
}

static void gemv_read_b(hls::burst_maxi<MatmulWord>& b, hls::stream<MatmulWord>& ws1,
                        unsigned n, unsigned m, unsigned batch, unsigned b_stride,
                        unsigned lk, unsigned split, unsigned planes, unsigned chunks, unsigned mc)
{
    for (unsigned bi = 0; bi < batch; bi++) {
        for (unsigned r = 0; r < n; r++) {
            for (unsigned ch = 0; ch < chunks; ch++) {
                const unsigned p0 = ch * mc;
                gemv_stream_planes<kBReqWords, kBReqAhead>(
                    b, ws1, 0, bi * b_stride, m, lk, split, planes, chunks, p0,
                    std::min(mc, m - p0));
            }
        }
    }
}

// Element e of the A row held in natural order, one word per E elements.
inline Data_t gemv_x_at(const MatmulWord xn[kMaxK / E], unsigned e)
{
    #pragma HLS INLINE
    const MatmulWord w = xn[e / E];
    Data_t lanes[E];
    #pragma HLS ARRAY_PARTITION variable=lanes complete
    for (unsigned j = 0; j < E; j++) {
        #pragma HLS UNROLL
        lanes[j] = matmul_word_lane(w, j);
    }
    return lanes[e % E];
}

// One read stream's MACs: planes [c_lo, c_hi) of every job.
template<int S>
void gemv_mac(hls::stream<MatmulWord>& xs, hls::stream<MatmulWord>& ws,
              hls::stream<AccData_t>& ps,
              unsigned n, unsigned k, unsigned m, unsigned batch,
              unsigned lk, unsigned c_lo, unsigned c_hi, unsigned chunks, unsigned mc)
{
    // xn:  the A row in natural order, one port word per E elements.
    // acc: word q of a plane accumulates its E / kw columns q * E / kw + g
    //      in lanes g — consecutive words never share an address, and word
    //      q is next touched one plane (>= E words) later.
    MatmulWord xn [kMaxK / E];
    GemvAcc    acc[kGemvAccWords];
    #pragma HLS AGGREGATE    variable=acc
    #pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=URAM
    #pragma HLS DEPENDENCE   variable=acc type=inter false

    const unsigned kw  = 1u << lk;
    const unsigned glg = kGemvLogE - lk;            // log2(columns per word)
    for (unsigned bi = 0; bi < batch; bi++) {
        for (unsigned r = 0; r < n; r++) {
            for (unsigned ch = 0; ch < chunks; ch++) {
                for (unsigned w = 0; w < k / E; w++) {
                    #pragma HLS PIPELINE II=1
                    xn[w] = xs.read();
                }

                const unsigned p0    = ch * mc;
                const unsigned mcc   = std::min(mc, m - p0);
                const unsigned wpp   = (mcc << lk) / E;       // words per plane in this chunk
                const unsigned total = (c_hi - c_lo) * wpp;

                // tap[j]: x at tap j of the current plane; ntap: the next
                // plane's, fetched over the current plane's first kw words.
                Data_t tap[E], ntap[E];
                #pragma HLS ARRAY_PARTITION variable=tap  complete
                #pragma HLS ARRAY_PARTITION variable=ntap complete
                for (unsigned j = 0; j < kw; j++) {
                    #pragma HLS PIPELINE II=1
                    tap[j] = gemv_x_at(xn, matmul_gemv_k(c_lo, j, kw));
                }

                unsigned c = c_lo, q = 0;
                for (unsigned t = 0; t < total; t++) {
                    #pragma HLS PIPELINE II=1
                    const MatmulWord w = ws.read();
                    if (q < kw && c + 1 < c_hi)
                        ntap[q] = gemv_x_at(xn, matmul_gemv_k(c + 1, q, kw));
                    // Lane products and the group-sum tree: lvl[d][g] sums
                    // lanes [g << d, (g + 1) << d); level lk is the columns.
                    AccData_t lvl[kGemvLogE + 1][E];
                    #pragma HLS ARRAY_PARTITION variable=lvl complete dim=0
                    for (unsigned l = 0; l < E; l++) {
                        #pragma HLS UNROLL
                        lvl[0][l] = tap[l & (kw - 1)] * matmul_word_lane(w, l);
                    }
                    for (unsigned d = 1; d <= kGemvLogE; d++) {
                        #pragma HLS UNROLL
                        for (unsigned g = 0; g < (E >> d); g++) {
                            #pragma HLS UNROLL
                            lvl[d][g] = lvl[d - 1][2 * g] + lvl[d - 1][2 * g + 1];
                        }
                    }
                    // The stream's first plane overwrites (no clear pass);
                    // lanes >= E / kw of the word are unused.
                    const bool first = c == c_lo;
                    GemvAcc    word  = acc[q];
                    for (unsigned g = 0; g < E; g++) {
                        #pragma HLS UNROLL
                        AccData_t v = lvl[0][g];
                        for (unsigned d = 1; d <= kGemvLogE; d++) {
                            #pragma HLS UNROLL
                            if (lk == d) v = lvl[d][g & ((E >> d) - 1)];
                        }
                        word.v[g] = first ? v : AccData_t(word.v[g] + v);
                    }
                    acc[q] = word;
                    if (++q == wpp) {
                        q = 0;
                        c++;
                        for (unsigned j = 0; j < E; j++) {
                            #pragma HLS UNROLL
                            tap[j] = ntap[j];
                        }
                    }
                }

                // Column p of the chunk: word p / (E / kw), lane p % (E / kw).
                for (unsigned p = 0; p < mcc; p++) {
                    #pragma HLS PIPELINE II=1
                    AccData_t v = AccData_t(0);
                    if (c_hi > c_lo) v = acc[p >> glg].v[p & ((1u << glg) - 1)];
                    ps.write(v);
                }
            }
        }
    }
}

static void gemv_write(Data_t* c, hls::stream<AccData_t>& ps0, hls::stream<AccData_t>& ps1,
                       unsigned n, unsigned m, unsigned batch, unsigned c_stride,
                       unsigned chunks, unsigned mc)
{
    for (unsigned bi = 0; bi < batch; bi++) {
        for (unsigned r = 0; r < n; r++) {
            for (unsigned ch = 0; ch < chunks; ch++) {
                const unsigned p0  = ch * mc;
                const unsigned mcc = std::min(mc, m - p0);
                const unsigned off = bi * c_stride + r * m + p0;
                for (unsigned p = 0; p < mcc; p++) {
                    #pragma HLS PIPELINE II=1
                    const AccData_t s = ps0.read() + ps1.read();   // wraps like the tiled acc
                    c[off + p] = saturate_cast<Data_t>(s);
                }
            }
        }
    }
}

static void gemv_dataflow(hls::burst_maxi<MatmulWord>& a, hls::burst_maxi<MatmulWord>& b,
                          Data_t* c, long long a_to_b,
                          unsigned n, unsigned k, unsigned m, unsigned batch,
                          unsigned a_stride, unsigned b_stride, unsigned c_stride,
                          unsigned lk, unsigned planes, unsigned split,
                          unsigned chunks, unsigned mc)
{
    #pragma HLS DATAFLOW
    hls::stream<MatmulWord> xs0("xs0"), xs1("xs1"), ws0("ws0"), ws1("ws1");
    hls::stream<AccData_t>  ps0("ps0"), ps1("ps1");
    #pragma HLS STREAM       variable=xs0 depth=4
    #pragma HLS STREAM       variable=xs1 depth=4
    #pragma HLS STREAM       variable=ws0 depth=32
    #pragma HLS STREAM       variable=ws1 depth=32
    #pragma HLS STREAM       variable=ps0 depth=16
    #pragma HLS STREAM       variable=ps1 depth=16
    #pragma HLS BIND_STORAGE variable=ws0 type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=ws1 type=fifo impl=lutram
    gemv_read_a(a, xs0, xs1, ws0, a_to_b, n, k, m, batch, a_stride, b_stride,
                lk, split, chunks, mc);
    gemv_read_b(b, ws1, n, m, batch, b_stride, lk, split, planes, chunks, mc);
    gemv_mac<0>(xs0, ws0, ps0, n, k, m, batch, lk, 0, split, chunks, mc);
    gemv_mac<1>(xs1, ws1, ps1, n, k, m, batch, lk, split, planes, chunks, mc);
    gemv_write(c, ps0, ps1, n, m, batch, c_stride, chunks, mc);
}

// Geometry of a GEMV call, then the dataflow region.
void gemv_run(hls::burst_maxi<MatmulWord>& a, hls::burst_maxi<MatmulWord>& b, Data_t* c,
              long long a_to_b, unsigned n, unsigned k, unsigned m, unsigned batch,
              unsigned a_stride, unsigned b_stride, unsigned c_stride, unsigned kw)
{
    #pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert((kw == 1 || kw == 2 || kw == 4 || kw == 8) && kw <= E && "gemv_kw");
    assert(k % E == 0 && k <= kMaxK && (kw == 1 || k % (kGemvTile * kw) == 0) && "gemv k");
    assert(m % E == 0 && (m == 0 || m * kw >= kGemvMinPlane) && "gemv m");
    assert(a_stride % E == 0 && b_stride % E == 0 && "gemv strides");
    assert(a_to_b % 16 == 0 && "gemv a_to_b: 16-byte aligned a and b");
#endif
    const unsigned lk     = (kw >= 2) + (kw >= 4) + (kw >= 8);
    const unsigned planes = k >> lk;
    const unsigned split  = (planes + 1) / 2;
    const unsigned chunks = (m + kGemvMaxM - 1) / kGemvMaxM;
    // Equal chunks (a narrow last chunk would break the plane-length rule).
    const unsigned per    = chunks > 1 ? (m + chunks - 1) / chunks : m;
    const unsigned mc     = (per + E - 1) / E * E;
    gemv_dataflow(a, b, c, a_to_b, n, k, m, batch, a_stride, b_stride, c_stride,
                  lk, planes, split, chunks, mc);
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
    unsigned      b_packed,
    unsigned      gemv_kw,
    long long     a_to_b
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
    // flight); B tile rows are <= kMatmulMaxRowWords each and are requested kBReqAhead
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
    #pragma HLS INTERFACE s_axilite port=gemv_kw        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=a_to_b         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return         bundle=ctrl

    // GEMV streaming path (MatmulKernel.h "GEMV streaming mode").
    if (kGemvMaxM > 0 && gemv_kw != 0) {
        gemv_run(a, b, c, a_to_b, n, k, m, batch, a_batch_stride, b_batch_stride,
                 c_batch_stride, gemv_kw);
        return;
    }

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
    // b_tile — two banks of one TILE_K × TILE_M block of B: the K-loop
    //          multiplies out of bank cur_bank while the next block is
    //          prefetched into the other (ping-pong, §7).  Reloaded from DDR
    //          for each (m_tile, k_tile) pair, or once per batch slice when
    //          the whole of B is one tile (fast path).
    //
    //          ARRAY_PARTITION complete dim=1 → kTileM independent BRAMs,
    //          each 2 × kTileK deep (flat (bank, k1) address).  All TILE_M
    //          columns can be read in the same cycle (the m1 unrolled loop
    //          reads kTileM elements per ki iteration).
    //
    // acc    — TILE_N × TILE_M accumulators.  Cleared at each m_tile; hold
    //          the partial dot products over the K dimension.
    //
    //          ARRAY_PARTITION complete dim=0 → all kTileN × kTileM elements
    //          (4 × 32 = 128 on the KV260) as registers.  The m1-unrolled loop writes to kTileM of them
    //          per ki cycle; the n1 rotation means no two consecutive
    //          iterations share an acc element.
    // -----------------------------------------------------------------------
    static Data_t    a_buf [kTileN][kMaxK / E][E];
    static Data_t    b_tile[kTileM][2 * kTileK];
    static AccData_t acc   [kTileN][kTileM];

    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=a_buf  complete dim=3
    // One RAM column per m1 with a flat (bank, k1) address: the K-loop reads
    // bank cur_bank and the prefetch writes bank !cur_bank in the same
    // iteration — one read + one write port per RAM (CONV_OPTIMISATION.md
    // §2.35: never a runtime index into the partitioned dimension).
    #pragma HLS ARRAY_PARTITION variable=b_tile complete dim=1
    #pragma HLS BIND_STORAGE    variable=b_tile type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=acc    complete dim=0
    static_assert((kTileK & (kTileK - 1)) == 0, "kTileK must be a power of two (flat bank address)");

    const unsigned n_tiles = (n + kTileN - 1) / kTileN;
    const unsigned m_tiles = (m + kTileM - 1) / kTileM;
    const unsigned k_tiles = (k + kTileK - 1) / kTileK;

    // B-resident fast path: when B is a single (m_tile, k_tile) block it is
    // loaded once per batch slice (at that slice's first n_tile) instead of
    // once per (n_tile, m_tile, k_tile); with b_batch_stride == 0 (B
    // broadcasts) it is loaded once per call.
    const bool b_resident = (m_tiles == 1) && (k_tiles == 1);

    // b_tile ping-pong state (§7): the bank holding the block of the
    // current iteration, whether any block has been loaded yet in this call,
    // and the fetch cursor of the next block (active while one is pending).
    ap_uint<1> cur_bank   = 0;
    bool       have_block = false;
    bool       f_active   = false;
    BFetch     f;

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
                        // Rotating the word by `shift` lanes once puts the
                        // element feeding bank j into lane j (fixed wiring
                        // from there); that element is the row's element
                        // w * E + j when j + shift < E, else (w - 1) * E + j.
                        const MatmulWord rot = matmul_rotate_lanes(a.read(), shift);
                        for (unsigned j = 0; j < E; j++) {
                            #pragma HLS UNROLL
                            const bool     wrap = (j + shift) >= E;
                            const unsigned widx = wrap ? w - 1 : w;   // word index within the row
                            const unsigned e    = widx * E + j;       // element index, e % E == j
                            if ((!wrap || w > 0) && e < k) {
                                a_buf[n1][widx][j] = matmul_word_lane(rot, j);
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

                    const bool load_b = !b_resident ||
                                        (n_tile == 0 && load_b_this_slice);
                    if (load_b) {
                        if (!have_block) {
                            // Very first block of the call: fetch it into
                            // cur_bank, blocking (nothing to overlap with).
                            b_fetch_start(b, f, bi, n_tile, m_tile, k_tile,
                                          k, m, b_batch_stride, b_packed);
                            while (f.rows_done < f.k_valid) {
                                #pragma HLS PIPELINE II=1
                                b_fetch_step(b, f, b_tile, cur_bank, b_packed);
                            }
                            have_block = true;
                        } else {
                            // This iteration's block was prefetched into
                            // !cur_bank by the previous K-loop, which does
                            // not exit before its prefetch is complete.
#ifndef __SYNTHESIS__
                            assert(f_active && f.bi == bi && f.mt == m_tile &&
                                   f.kt == k_tile && f.rows_done == f.k_valid &&
                                   "b_tile prefetch cursor out of step");
#endif
                            cur_bank = ~cur_bank;
                        }
                        // Point the cursor at the next block that will be
                        // loaded, in consumption order (k_tile fastest, then
                        // m_tile, n_tile, batch; the B-resident fast path
                        // only reloads at the next batch slice, never when
                        // B broadcasts), and issue its requests (packed).
                        unsigned nb = bi, nn = n_tile, nm = m_tile, nk = k_tile;
                        if (b_resident) {
                            nn = 0; nm = 0; nk = 0;
                            nb = (b_batch_stride == 0) ? batch : bi + 1;
                        } else if (++nk == k_tiles) {
                            nk = 0;
                            if (++nm == m_tiles) {
                                nm = 0;
                                if (++nn == n_tiles) { nn = 0; nb++; }
                            }
                        }
                        f_active = nb < batch;
                        if (f_active)
                            b_fetch_start(b, f, nb, nn, nm, nk, k, m, b_batch_stride, b_packed);
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
                    //
                    // The loop also carries the prefetch of the next B
                    // block (§7): one b_fetch_step per iteration into bank
                    // !cur_bank, and it keeps iterating (MACs idle) until
                    // that block is complete, so the next iteration that
                    // loads B can simply swap banks.
                    const unsigned ki_bound = seg_len * kTileN;
                    const unsigned b_row0   = (unsigned)cur_bank * kTileK;
                    const ap_uint<1> nbank  = ~cur_bank;
                    for (unsigned ki = 0; ; ki++) {
                        #pragma HLS PIPELINE II=1
                        const bool mac  = ki < ki_bound;
                        const bool pend = f_active && f.rows_done < f.k_valid;
                        if (!mac && !pend) break;
                        // The accumulate itself is unconditional: an idle
                        // lane (drain-only iteration, or a K-segment tail
                        // past k_valid) multiplies a zero A operand, which
                        // leaves acc bit-identical.  Predicating the acc
                        // update instead made HLS wrap the lane mux in
                        // 33-bit compare / select logic (+5 k LUT).
                        {
                            const unsigned n1  = ki % kTileN;
                            const unsigned kk  = ki / kTileN;
                            const unsigned row = n1 >> lpr_log;
                            const unsigned seg = n1 & (lpr - 1);
                            const unsigned kl  = seg_base[seg] + kk;   // K index local to this tile
                            const bool     ok  = mac && kl < k_valid;
                            const unsigned kr  = ok ? kl : 0u;         // clamp: kl may reach kTileK + lpr - 1
                            const unsigned kidx  = k_off + kr;
                            const Data_t   a_val = ok ? a_buf[row][kidx / E][kidx % E] : Data_t(0);
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[n1][m1] += a_val * b_tile[m1][b_row0 + kr];
                            }
                        }
                        if (pend) b_fetch_step(b, f, b_tile, nbank, b_packed);
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
