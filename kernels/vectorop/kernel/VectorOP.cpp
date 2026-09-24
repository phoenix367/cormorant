#include "VectorOP.h"
#include "hls_stream.h"

// ---------------------------------------------------------------------------
// Scalar compute helpers (inlined into the compute stage, one per lane)
// ---------------------------------------------------------------------------

static inline Data_t sub_add(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a + b);
}

static inline Data_t sub_sub(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a - b);
}

// Full-precision multiply; saturate_cast clips the 2W-bit product to Data_t.
static inline Data_t sub_mul(Data_t a, Data_t b) {
    return saturate_cast<Data_t>(a * b);
}

// Iterative fixed-point divider (one instance, see compute_div).
static inline Data_t sub_div(Data_t a, Data_t b) {
    if (b == Data_t(0)) return Data_t(0);
    return saturate_cast<Data_t>(a / b);
}

static inline Data_t sub_relu(Data_t a) {
    return (a < Data_t(0)) ? Data_t(0) : a;
}

static inline Data_t sub_relu6(Data_t a) {
    if (a < Data_t(0)) return Data_t(0);
    if (a > Data_t(6)) return Data_t(6);
    return a;
}

// Fused activation (AXI-Lite 'act' register) applied to the op result.
static inline Data_t apply_act(Data_t v, unsigned act) {
    switch (act) {
        case ACT_RELU:  return sub_relu(v);
        case ACT_RELU6: return sub_relu6(v);
        default:        return v;
    }
}

static inline Data_t op_lane(Data_t av, Data_t bv, unsigned op) {
    switch (op) {
        case OP_ADD:   return sub_add (av, bv);
        case OP_SUB:   return sub_sub (av, bv);
        case OP_MUL:   return sub_mul (av, bv);
        case OP_RELU:  return sub_relu (av);
        case OP_RELU6: return sub_relu6(av);
        default:       return av;           // OP_DIV is handled by compute_div
    }
}

// ---------------------------------------------------------------------------
// Port / request geometry (VECTOROP_OPTIMISATION.md §2).
//
// Every stage works on 128-bit words of E = kVecLanes elements.  A run
// (one outer iteration's `size` elements) starts on a word boundary — the
// alignment contract in VectorOP.h — so run o of an operand with element
// stride `inc` is the word range [o * inc / E, o * inc / E + words(size)).
//
// Read requests are at most kReadReqWords words (max_read_burst_length of
// a / b) and kReadAhead of them are kept in flight, both below the port's
// num_read_outstanding = 16; the adapter's read buffer holds
// num_read_outstanding * max_read_burst_length = 1024 words, so the
// (kReadAhead + 1) * kReadReqWords <= 576 words that can be requested but
// unread never fill it and a read_request() can never block ahead of the
// read() that would drain it.  Write requests are at most kWriteReqWords
// words (one max-length burst) and at most kWriteInFlight of them are left
// without a write_response() (CONV_OPTIMISATION.md §2.30 deadlock).
// ---------------------------------------------------------------------------
static constexpr unsigned E              = kVecLanes;
static constexpr unsigned kReadReqWords  = 64;    // max_read_burst_length of a / b
static constexpr unsigned kReadAhead     = 8;     // read requests kept in flight
static constexpr unsigned kWriteReqWords = 256;   // max_write_burst_length of c
static constexpr unsigned kWriteInFlight = 8;     // unacknowledged write requests
static constexpr unsigned kRepWords      = 256;   // stride-0 replay buffer (2048 elements)

// The runs of one operand as the request issuer sees them.
struct RunGeom {
    unsigned n_runs;      // separately addressed runs
    unsigned run_words;   // words per run
    unsigned stride_w;    // word stride between consecutive runs
};

// outer == 1, or runs that abut (inc == size on a whole number of words):
// one contiguous range.  Otherwise every outer iteration is its own run;
// inc == 0 gives stride 0 (the operand is re-read every iteration).
static inline RunGeom run_geometry(unsigned outer, unsigned size, unsigned inc) {
    const unsigned n_words = vec_words_for(size);
    RunGeom g;
    if (outer == 1 || (inc == size && size % E == 0)) {
        g.n_runs    = 1;
        g.run_words = outer * n_words;
        g.stride_w  = 0;
    } else {
        g.n_runs    = outer;
        g.run_words = n_words;
        g.stride_w  = inc / E;
    }
    return g;
}

// Zero the lanes at or above `tail_lanes` (the part of a run's last word
// that lies past `size`).
static inline VecWord mask_tail(VecWord w, unsigned tail_lanes) {
    VecWord out = w;
    for (unsigned l = 0; l < E; ++l) {
        #pragma HLS UNROLL
        if (l >= tail_lanes) out.range(kDataBits * (l + 1) - 1, kDataBits * l) = 0;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Load stage: DDR word ranges -> word stream.
//
// One flattened II=1 loop over every word of every run; the request cursor
// runs kReadAhead pieces (<= kReadReqWords words each) ahead of the read
// cursor.  Lanes past the end of a run are masked to zero so the tail word
// of c holds op(0, 0) and every op sees defined operands.
// ---------------------------------------------------------------------------
static void stream_runs(
    hls::burst_maxi<VecWord> src,
    hls::stream<VecWord>&    dst,
    RunGeom                  g,
    unsigned                 tail_lanes
) {
    const unsigned pieces_per_run   = (g.run_words + kReadReqWords - 1) / kReadReqWords;
    const unsigned last_piece_words = g.run_words - (pieces_per_run - 1) * kReadReqWords;
    const unsigned total_pieces     = g.n_runs * pieces_per_run;
    const unsigned total_words      = g.n_runs * g.run_words;

    // Request cursor.
    unsigned issued   = 0;
    unsigned iss_p    = 0;          // piece within the run being issued
    unsigned iss_base = 0;          // word offset of that run
    unsigned iss_off  = 0;          // word offset of the next piece

    // Prologue: the first kReadAhead pieces.
    const unsigned n_pro = (total_pieces < kReadAhead) ? total_pieces : kReadAhead;
    for (unsigned k = 0; k < n_pro; ++k) {
        #pragma HLS PIPELINE II=1
        const bool     last = (iss_p == pieces_per_run - 1);
        const unsigned len  = last ? last_piece_words : kReadReqWords;
        src.read_request(iss_off, len);
        if (last) { iss_p = 0; iss_base += g.stride_w; iss_off = iss_base; }
        else      { iss_p++;   iss_off += kReadReqWords; }
    }
    issued = n_pro;

    // Read cursor.
    unsigned cur_w   = 0;                                   // word within the piece
    unsigned cur_p   = 0;                                   // piece within the run
    unsigned cur_wr  = 0;                                   // word within the run
    unsigned cur_len = (pieces_per_run == 1) ? last_piece_words : kReadReqWords;

    for (unsigned i = 0; i < total_words; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
        if (cur_w == 0 && issued < total_pieces) {
            const bool     last = (iss_p == pieces_per_run - 1);
            const unsigned len  = last ? last_piece_words : kReadReqWords;
            src.read_request(iss_off, len);
            if (last) { iss_p = 0; iss_base += g.stride_w; iss_off = iss_base; }
            else      { iss_p++;   iss_off += kReadReqWords; }
            issued++;
        }

        VecWord x = src.read();
        if (cur_wr == g.run_words - 1 && tail_lanes != E) x = mask_tail(x, tail_lanes);
        dst.write(x);

        const bool piece_end = (cur_w + 1 == cur_len);
        const bool run_end   = (cur_wr + 1 == g.run_words);
        cur_wr = run_end ? 0u : cur_wr + 1;
        if (piece_end) {
            cur_w   = 0;
            cur_p   = (cur_p + 1 == pieces_per_run) ? 0u : cur_p + 1;
            cur_len = (cur_p == pieces_per_run - 1) ? last_piece_words : kReadReqWords;
        } else {
            cur_w++;
        }
    }
}

static void load_words(
    hls::burst_maxi<VecWord> src,
    hls::stream<VecWord>&    dst,
    unsigned                 outer,
    unsigned                 size,
    unsigned                 inc,
    bool                     enabled
) {
    #pragma HLS INLINE off
    if (!enabled || size == 0 || outer == 0) return;

    const unsigned n_words    = vec_words_for(size);
    const unsigned tail_lanes = (size % E == 0) ? E : size % E;

    if (outer > 1 && inc == 0 && n_words <= kRepWords) {
        // Stride-0 operand: read it once, replay it outer times.
        VecWord rep_buf[kRepWords];
        for (unsigned w0 = 0; w0 < n_words; w0 += kReadReqWords) {
            const unsigned rem = n_words - w0;
            src.read_request(w0, (rem < kReadReqWords) ? rem : kReadReqWords);
        }
        for (unsigned w = 0; w < n_words; ++w) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=256
            VecWord x = src.read();
            if (w == n_words - 1 && tail_lanes != E) x = mask_tail(x, tail_lanes);
            rep_buf[w] = x;
        }
        const unsigned total = outer * n_words;
        unsigned k = 0;
        for (unsigned i = 0; i < total; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
            dst.write(rep_buf[k]);
            k = (k + 1 == n_words) ? 0u : k + 1;
        }
        return;
    }

    stream_runs(src, dst, run_geometry(outer, size, inc), tail_lanes);
}

// ---------------------------------------------------------------------------
// Compute stage: E lanes per cycle for every op except OP_DIV, which runs
// one lane per cycle through a single pipelined divider (compute_div).
// For unary ops b_s is never read (load_words(b) pushed nothing).
// ---------------------------------------------------------------------------
static void compute_div(
    hls::stream<VecWord>& a_s,
    hls::stream<VecWord>& b_s,
    hls::stream<VecWord>& c_s,
    unsigned              total_words,
    unsigned              act
) {
    const unsigned total = total_words * E;
    VecWord aw = 0, bw = 0;
    ap_uint<kDataBits> res[E];
    #pragma HLS ARRAY_PARTITION variable=res complete
    unsigned l = 0;
    for (unsigned i = 0; i < total; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8 max=524288
        if (l == 0) { aw = a_s.read(); bw = b_s.read(); }
        const Data_t av = vec_lane_to_data(aw.range(kDataBits - 1, 0));
        const Data_t bv = vec_lane_to_data(bw.range(kDataBits - 1, 0));
        aw >>= kDataBits;
        bw >>= kDataBits;
        const ap_uint<kDataBits> r = vec_data_to_lane(apply_act(sub_div(av, bv), act));
        for (unsigned j = 0; j < E; ++j) {
            #pragma HLS UNROLL
            if (j == l) res[j] = r;
        }
        if (l == E - 1) {
            VecWord cw;
            for (unsigned j = 0; j < E; ++j) {
                #pragma HLS UNROLL
                cw.range(kDataBits * (j + 1) - 1, kDataBits * j) = res[j];
            }
            c_s.write(cw);
            l = 0;
        } else {
            l++;
        }
    }
}

static void compute_words(
    hls::stream<VecWord>& a_s,
    hls::stream<VecWord>& b_s,
    hls::stream<VecWord>& c_s,
    unsigned              total_words,
    unsigned              op,
    unsigned              act
) {
    #pragma HLS INLINE off
    if (op == OP_DIV) {
        compute_div(a_s, b_s, c_s, total_words, act);
        return;
    }
    const bool binary = (op < OP_RELU);
    for (unsigned i = 0; i < total_words; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
        const VecWord aw = a_s.read();
        const VecWord bw = binary ? b_s.read() : VecWord(0);
        VecWord cw;
        for (unsigned l = 0; l < E; ++l) {
            #pragma HLS UNROLL
            const Data_t av = vec_lane_to_data(aw.range(kDataBits * (l + 1) - 1, kDataBits * l));
            const Data_t bv = vec_lane_to_data(bw.range(kDataBits * (l + 1) - 1, kDataBits * l));
            const Data_t r  = apply_act(op_lane(av, bv, op), act);
            cw.range(kDataBits * (l + 1) - 1, kDataBits * l) = vec_data_to_lane(r);
        }
        c_s.write(cw);
    }
}

// ---------------------------------------------------------------------------
// Store stage: word stream -> DDR.  Whole words only: the last word of a
// run carries op(0, 0) in the lanes past `size` (alignment contract).
// One flattened II=1 loop; a write_request per piece of <= kWriteReqWords
// words, responses collected in a sliding window of kWriteInFlight.
// ---------------------------------------------------------------------------
static void store_words(
    hls::burst_maxi<VecWord> dst,
    hls::stream<VecWord>&    src,
    unsigned                 outer,
    unsigned                 size,
    unsigned                 inc
) {
    #pragma HLS INLINE off
    if (size == 0 || outer == 0) return;

    const RunGeom  g                = run_geometry(outer, size, inc);
    const unsigned pieces_per_run   = (g.run_words + kWriteReqWords - 1) / kWriteReqWords;
    const unsigned last_piece_words = g.run_words - (pieces_per_run - 1) * kWriteReqWords;
    const unsigned total_words      = g.n_runs * g.run_words;

    unsigned pending  = 0;
    unsigned cur_w    = 0;
    unsigned cur_p    = 0;
    unsigned cur_base = 0;
    unsigned cur_off  = 0;
    unsigned cur_len  = (pieces_per_run == 1) ? last_piece_words : kWriteReqWords;

    for (unsigned i = 0; i < total_words; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=65536
        if (cur_w == 0) dst.write_request(cur_off, cur_len);
        dst.write(src.read());
        if (cur_w + 1 == cur_len) {
            if (pending == kWriteInFlight - 1) dst.write_response();
            else                               pending++;
            cur_w = 0;
            if (cur_p + 1 == pieces_per_run) {
                cur_p = 0; cur_base += g.stride_w; cur_off = cur_base;
            } else {
                cur_p++;   cur_off += kWriteReqWords;
            }
            cur_len = (cur_p == pieces_per_run - 1) ? last_piece_words : kWriteReqWords;
        } else {
            cur_w++;
        }
    }
    while (pending > 0) {
        dst.write_response();
        pending--;
    }
}

// ---------------------------------------------------------------------------
// Top kernel
// ---------------------------------------------------------------------------
void VectorOPKernel(
    hls::burst_maxi<VecWord> a,
    hls::burst_maxi<VecWord> b,
    hls::burst_maxi<VecWord> c,
    unsigned      size,
    unsigned      op,
    unsigned      outer,
    unsigned      a_inc,
    unsigned      b_inc,
    unsigned      act
) {
    // 128-bit burst_maxi ports (VecWord).  Read requests are <= 64 words
    // with up to 16 outstanding (adapter buffer 1024 words); write requests
    // <= 256 words with up to 16 outstanding.  See the geometry note above
    // for how the stages stay below those limits.
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 \
        num_read_outstanding=16  max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 \
        num_read_outstanding=16  max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 \
        num_write_outstanding=16 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=a      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=c      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=size   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=op     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=outer  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=a_inc  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=b_inc  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=act    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // Word streams between the stages; depth 64 lets the loaders run a
    // burst ahead of compute / store.  LUTRAM: a 64 x 128-bit FIFO in
    // block RAM costs 8 BRAM18 (width-limited packing) for 8 Kbit.
    static hls::stream<VecWord> a_s("a_s");
    static hls::stream<VecWord> b_s("b_s");
    static hls::stream<VecWord> c_s("c_s");
    #pragma HLS stream variable=a_s depth=64
    #pragma HLS stream variable=b_s depth=64
    #pragma HLS stream variable=c_s depth=64
    #pragma HLS bind_storage variable=a_s type=fifo impl=lutram
    #pragma HLS bind_storage variable=b_s type=fifo impl=lutram
    #pragma HLS bind_storage variable=c_s type=fifo impl=lutram

    // c_inc == 0 when outer==1 (a_inc==b_inc==0) — writes c[i] directly.
    // Unary ops (OP_RELU, OP_RELU6) never read b.
    const unsigned c_inc       = a_inc + b_inc;
    const unsigned total_words = outer * vec_words_for(size);
    const bool     b_enabled   = (op < OP_RELU);

    #pragma HLS dataflow
    load_words(a, a_s, outer, size, a_inc, true);
    load_words(b, b_s, outer, size, b_inc, b_enabled);
    compute_words(a_s, b_s, c_s, total_words, op, act);
    store_words(c, c_s, outer, size, c_inc);
}
