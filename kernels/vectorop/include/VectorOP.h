#pragma once

#include "Config.h"
#include "ap_fixed.h"  // always present in the Vitis HLS environment
#include "ap_int.h"
#include "hls_burst_maxi.h"

#include <climits>
#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts value v to type T, applying saturation if T is an ap_fixed type.
//
//   ap_fixed<W,I,Q,O,N>  — converts v through ap_fixed<W,I,AP_TRN,AP_SAT>,
//                           which clips to the representable range
//                           [-2^(I-1), 2^(I-1) - 2^-(W-I)] before narrowing.
//   float / double / int  — returns T(v) unchanged (pass-through).
//
// Works identically in C simulation and HLS synthesis; no #ifdef needed.
// The partial specialisation is never instantiated when Data_t is float,
// so there is no runtime cost for non-fixed-point types.
// ---------------------------------------------------------------------------

// Primary template: no saturation (float, double, integer types, …)
template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

// Partial specialisation for ap_fixed<W,I,Q,O,N>:
// clips via AP_SAT, then re-wraps in the original overflow mode.
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct saturate_to<ap_fixed<W, I, Q, O, N>> {
    template<typename From>
    static ap_fixed<W, I, Q, O, N> cast(From v) {
        // ap_fixed<W,I,AP_TRN,AP_SAT> applies saturation on the narrowing
        // step; the bits it produces are always within the valid range of the
        // destination type, so the outer ap_fixed<W,I,Q,O,N> cast is lossless.
        return ap_fixed<W, I, Q, O, N>(ap_fixed<W, I, AP_TRN, AP_SAT>(v));
    }
};

template<typename T, typename From>
inline T saturate_cast(From v) {
    return saturate_to<T>::template cast<From>(v);
}

// ---------------------------------------------------------------------------
// round_cast<T>(v)
//
// Like saturate_cast, but rounds to the nearest representable value, ties
// to even (ap_fixed AP_RND_CONV), instead of truncating: the activation
// functions' rounding (the host ops' round-half-even write-back).  Plain
// conversion for non-ap_fixed types.
// ---------------------------------------------------------------------------
template<typename T>
struct round_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct round_to<ap_fixed<W, I, Q, O, N>> {
    template<typename From>
    static ap_fixed<W, I, Q, O, N> cast(From v) {
        return ap_fixed<W, I, Q, O, N>(ap_fixed<W, I, AP_RND_CONV, AP_SAT>(v));
    }
};

template<typename T, typename From>
inline T round_cast(From v) {
    return round_to<T>::template cast<From>(v);
}

// ---------------------------------------------------------------------------
// Operation codes for VectorOPKernel.
// Passed at runtime via the AXI-Lite 'op' register.
// ---------------------------------------------------------------------------
enum Op : unsigned {
    OP_ADD   = 0,  // c[i] = saturate_cast<Data_t>(a[i] + b[i])
    OP_SUB   = 1,  // c[i] = saturate_cast<Data_t>(a[i] - b[i])
    OP_MUL   = 2,  // c[i] = saturate_cast<Data_t>(a[i] * b[i])
    OP_DIV   = 3,  // c[i] = saturate_cast<Data_t>(a[i] / b[i])  (b[i] ≠ 0)
    OP_RELU  = 4,  // c[i] = max(a[i], 0)          — unary, b[] not read
    OP_RELU6 = 5,  // c[i] = min(max(a[i], 0), 6)  — unary, b[] not read
    // The activation ops: c[i] = the activation of a[i] (Act codes 3..6, the
    // op code minus 3), unary; the act register is not applied after them.
    OP_LEAKY_RELU = 6,  // a >= 0 ? a : alpha * a
    OP_SILU       = 7,  // a * sigmoid(a)
    OP_GELU       = 8,  // a * Phi(a) = a / 2 * (1 + erf(a / sqrt(2)))
    OP_GELU_TANH  = 9,  // a / 2 * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 a^3)))
    // Softmax (doc/plans/SOFTMAX_PLAN.md; smx_vector below), unary: b[] not
    // read; c advances by b_inc per vector (not a_inc + b_inc).
    OP_SOFTMAX    = 10, // row mode: outer rows of size elements (input stride
                        // a_inc, output stride b_inc), softmax over each row
    OP_SOFTMAX_T  = 11, // column mode: input s[size keys][outer queries] (row
                        // stride a_inc), output P[outer][size] (row stride
                        // b_inc): softmax over the keys of each query column
};

// ---------------------------------------------------------------------------
// Fused activation applied to the op result (AXI-Lite 'act' register).
// ACT_NONE leaves the op result unchanged; ACT_RELU / ACT_RELU6 clip it
// exactly like OP_RELU / OP_RELU6 would in a second pass, so the scheduler
// can fuse Add -> Relu (or -> Clip(0,6)) into one kernel invocation; the
// other activations likewise (Add -> Gelu, ...).
//
// ACT_LEAKY_RELU .. ACT_GELU_TANH (and the activation ops) are the exact
// function of the Q8.8 input rounded to the nearest Q8.8 value, ties to
// even (round_cast; the host ops' write-back).  LeakyReLU's slope is the
// 'alpha' register: alpha[15:0] / 65536 (0 <= alpha < 1; bits 31:16 are
// ignored).  The RTL kernel computes SiLU and both GELUs from a table of
// f(-|x|) — for these odd-symmetric x * F(x), f(x) = max(x, 0) + f(-|x|) —
// built by kernels/vectorop_rtl/scripts/gen_act_rom.py; IEEE double with
// the formulas of act_fn() in VectorOP.cpp rounds every Q8.8 input the same
// way as the exact value (the nearest tie is 1.6e-5 LSB away).
// ---------------------------------------------------------------------------
enum Act : unsigned {
    ACT_NONE       = 0,
    ACT_RELU       = 1,
    ACT_RELU6      = 2,
    ACT_LEAKY_RELU = 3,
    ACT_SILU       = 4,
    ACT_GELU       = 5,
    ACT_GELU_TANH  = 6,
};

// The activation an op / act pair applies after the op: an activation op
// (OP_LEAKY_RELU .. OP_GELU_TANH) its own, any other op the act register's.
inline unsigned job_act(unsigned op, unsigned act) {
    return (op >= OP_LEAKY_RELU && op <= OP_GELU_TANH) ? op - (OP_LEAKY_RELU - ACT_LEAKY_RELU)
                                                        : act;
}

// ---------------------------------------------------------------------------
// Softmax (OP_SOFTMAX / OP_SOFTMAX_T; doc/plans/SOFTMAX_PLAN.md §2.1) — integer
// arithmetic on the raw int16 lanes, the specification of
// inference-scheduler/src/vectorop_smx.py.  Per vector of n inputs x with v
// valid (registers smx_cm = Cm [23:0], smx_cfg = Cs [5:0] | f_p [12:8], smx_mask =
// valid0 [15:0] | period [31:16]; v = min(n, valid0 + (q mod period)) for
// vector q, period 0: valid0):
//   m = max_{j<v} x_j;  d_j = m - x_j;  y_j = (d_j * Cm) >> Cs;
//   e_j = TAB[y_j mod 2^12] >> (y_j div 2^12),  TAB[k] = round(2^16 * 2^(-k / 4096));
//   S = sum_{j<v} e_j;  R = floor(2^40 / S);
//   P_j = min((e_j * R + 2^(39 - f_p)) >> (40 - f_p), 32767);  P_j = 0 for j >= v.
// Rows of at most kSmxMaxRow elements (row mode), at most kSmxMaxKeys keys
// and outer a multiple of 16 (column mode: blocks of 16 query columns; the
// kernel processes outer & ~15 of them).
// ---------------------------------------------------------------------------
static constexpr unsigned kSmxF       = 12;     // table index bits
static constexpr unsigned kSmxE       = 16;     // e = 1.0 at 2^16
static constexpr unsigned kSmxRB      = 40;     // R = floor(2^40 / S)
static constexpr unsigned kSmxMaxRow  = 2048;
static constexpr unsigned kSmxMaxKeys = 1024;

// TAB[k] = round_half_even(2^16 * 2^(-k / 4096)); no entry lies within 1e-6
// of a rounding tie (kernels/vectorop_rtl/scripts/gen_smx_rom.py checks), so
// any accurate exp2 gives the same table.
inline const int64_t* smx_table() {
    static int64_t tab[1u << kSmxF];
    static bool    init = false;
    if (!init) {
        for (unsigned k = 0; k < (1u << kSmxF); ++k)
            tab[k] = (int64_t)std::nearbyint(std::ldexp(std::exp2(-(double)k / (1u << kSmxF)), kSmxE));
        init = true;
    }
    return tab;
}

inline unsigned smx_valid(unsigned q, unsigned n, unsigned mask) {
    const unsigned valid0 = mask & 0xFFFFu, period = mask >> 16;
    const unsigned v = valid0 + (period ? q % period : 0u);
    return v < n ? v : n;
}

// One softmax vector: x[n] raw int16 -> p[n] raw int16 at 2^-f_p.
inline void smx_vector(const int16_t* x, unsigned n, unsigned v, unsigned cm, unsigned cfg,
                       int16_t* p) {
    const int64_t* tab = smx_table();
    const unsigned cs = cfg & 63u, fp = (cfg >> 8) & 31u;
    cm &= 0xFFFFFFu;
    if (v > n) v = n;
    int64_t m = INT64_MIN;
    for (unsigned j = 0; j < v; ++j) m = x[j] > m ? x[j] : m;
    int64_t S = 0;
    static int64_t e[kSmxMaxRow > kSmxMaxKeys ? kSmxMaxRow : kSmxMaxKeys];
    for (unsigned j = 0; j < v; ++j) {
        const uint64_t y  = ((uint64_t)(m - x[j]) * cm) >> cs;
        const uint64_t sh = y >> kSmxF;
        e[j] = sh >= 63 ? 0 : tab[y & ((1u << kSmxF) - 1)] >> sh;
        S += e[j];
    }
    const uint64_t R = S > 0 ? ((uint64_t)1 << kSmxRB) / (uint64_t)S : 0;
    for (unsigned j = 0; j < n; ++j) {
        if (j >= v) { p[j] = 0; continue; }
        const uint64_t q = ((uint64_t)e[j] * R + ((uint64_t)1 << (kSmxRB - fp - 1))) >> (kSmxRB - fp);
        p[j] = (int16_t)(q > 32767 ? 32767 : q);
    }
}

// ---------------------------------------------------------------------------
// Port width — 128-bit words (VECTOROP_OPTIMISATION.md §2).
//
// a, b and c are hls::burst_maxi<VecWord> ports carrying kVecLanes elements
// per beat (8 × ap_fixed<16,8>).  The DDR layout is unchanged (plain
// element arrays); the kernel requests whole word ranges and extracts /
// packs the lanes itself.
//
// Alignment contract (the scheduler guarantees it, TestSimulation asserts
// it):
//   * the base address of every run of a, b and c is 16-byte aligned:
//     the a / b / c registers hold 16-byte-aligned addresses and
//     a_inc / b_inc are 0 or a multiple of kVecLanes elements;
//   * the run LENGTH (size) is arbitrary.  Input lanes past the end of a
//     run are read (the bytes must be mappable) and masked to zero;
//   * c is written in whole words: the last word of every run is written
//     up to the next 16-byte boundary, i.e. up to kVecLanes - 1 elements
//     past size.  Those tail elements receive op(0, 0) (= 0 for every op
//     and activation).  The caller's buffer / stride gap must cover them
//     (the scheduler's CHUNK_STRIDE is a multiple of kVecLanes and every
//     buffer is allocated in 64-byte multiples).
// ---------------------------------------------------------------------------
template<typename T> struct VecDataBits { static constexpr unsigned value = 8 * sizeof(T); };
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct VecDataBits<ap_fixed<W, I, Q, O, N>> { static constexpr unsigned value = W; };

static constexpr unsigned kDataBits    = VecDataBits<Data_t>::value;
static constexpr unsigned kVecPortBits = 128;
static constexpr unsigned kVecLanes    = kVecPortBits / kDataBits;
static_assert(kVecPortBits % kDataBits == 0, "Data_t must divide the 128-bit port");
typedef ap_uint<kVecPortBits> VecWord;

// Data_t <-> raw lane bits (the byte image the port carries).
template<typename T>
struct VecLane {
    static T from_bits(ap_uint<kDataBits> bits) {
        // Non-ap_fixed types (float / double / half / integers): reinterpret
        // the lane's byte image.  C-simulation only for non-fixed builds.
        T v;
        unsigned char* p = reinterpret_cast<unsigned char*>(&v);
        for (unsigned i = 0; i < sizeof(T); ++i)
            p[i] = (unsigned char)bits.range(8 * i + 7, 8 * i).to_uint();
        return v;
    }
    static ap_uint<kDataBits> to_bits(T v) {
        ap_uint<kDataBits> bits = 0;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
        for (unsigned i = 0; i < sizeof(T); ++i)
            bits.range(8 * i + 7, 8 * i) = p[i];
        return bits;
    }
};
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct VecLane<ap_fixed<W, I, Q, O, N>> {
    typedef ap_fixed<W, I, Q, O, N> T;
    static T from_bits(ap_uint<kDataBits> bits) { T v; v.range(W - 1, 0) = bits; return v; }
    static ap_uint<kDataBits> to_bits(T v)      { return v.range(W - 1, 0); }
};

inline Data_t vec_lane_to_data(ap_uint<kDataBits> bits) { return VecLane<Data_t>::from_bits(bits); }
inline ap_uint<kDataBits> vec_data_to_lane(Data_t v)    { return VecLane<Data_t>::to_bits(v); }

// Number of words that cover `count` elements starting at a word boundary.
inline unsigned vec_words_for(unsigned count) {
    return (count + kVecLanes - 1) / kVecLanes;
}

// ---------------------------------------------------------------------------
// VectorOPKernel — element-wise vector operation with saturating output.
//
// Interface:
//   a, b    — AXI4 master (m_axi) 128-bit read ports; base addresses via AXI-Lite
//   c       — AXI4 master (m_axi) 128-bit write port;  base address  via AXI-Lite
//   size    — AXI-Lite register: elements per inner chunk (run)
//   op      — AXI-Lite register: operation selector (Op enum)
//   outer   — AXI-Lite register: number of outer broadcasting iterations
//             (1 = normal non-broadcast operation)
//   a_inc   — AXI-Lite register: element stride for a per outer iteration
//             (0 = a repeats every outer iteration; size = a advances)
//   b_inc   — AXI-Lite register: element stride for b per outer iteration
//             (0 = b repeats every outer iteration; size = b advances)
//   act     — AXI-Lite register: fused activation (Act enum, 0 = none);
//             appended after b_inc so the earlier register offsets are
//             unchanged (0x5C in the generated driver).
//   alpha   — AXI-Lite register: LeakyReLU slope, alpha[15:0] / 65536 (0x64).
//   smx_cm, smx_cfg, smx_mask — AXI-Lite registers of the softmax ops (0x6C,
//             0x74, 0x7C; see smx_vector), appended last.
//   return  — AXI-Lite control: ap_ctrl_hs (start/done/idle/ready)
//
// The kernel processes outer × size elements:
//   c[o * (a_inc+b_inc) + i] = job_act(op, act)(op(a[o*a_inc + i], b[o*b_inc + i]))
//   for o in 0..outer-1, i in 0..size-1
//
// For non-broadcast use set outer=1, a_inc=0, b_inc=0.
// For broadcast with a advancing: outer=N, a_inc=aligned_chunk, b_inc=0.
// For broadcast with b advancing: outer=N, a_inc=0, b_inc=aligned_chunk.
//
// For unary operations (op >= OP_RELU: OP_RELU, OP_RELU6 and the
// activation ops) only a[] is read; no AXI transactions are issued on the
// gmem1 port and b_addr is ignored.
//
// Data paths (all three stages run at one word = kVecLanes elements per
// cycle):
//   * outer == 1, or a_inc == size (== b_inc == size): the runs are one
//     contiguous word range — a single stream of <= 64-word requests;
//   * a stride-0 operand of <= kRepWords words (2048 elements) is read
//     once into an on-chip replay buffer and re-streamed outer times;
//   * otherwise every run is requested separately (<= 64-word pieces,
//     kReadAhead pieces in flight).
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
    unsigned      act,
    unsigned      alpha,
    unsigned      smx_cm   = 0,
    unsigned      smx_cfg  = 0,
    unsigned      smx_mask = 0
);
