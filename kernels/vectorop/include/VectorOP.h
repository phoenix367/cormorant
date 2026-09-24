#pragma once

#include "Config.h"
#include "ap_fixed.h"  // always present in the Vitis HLS environment
#include "ap_int.h"
#include "hls_burst_maxi.h"

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
};

// ---------------------------------------------------------------------------
// Fused activation applied to the op result (AXI-Lite 'act' register).
// ACT_NONE leaves the op result unchanged; ACT_RELU / ACT_RELU6 clip it
// exactly like OP_RELU / OP_RELU6 would in a second pass, so the scheduler
// can fuse Add -> Relu (or -> Clip(0,6)) into one kernel invocation.
// ---------------------------------------------------------------------------
enum Act : unsigned {
    ACT_NONE  = 0,
    ACT_RELU  = 1,
    ACT_RELU6 = 2,
};

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
//             appended LAST so the earlier register offsets are unchanged
//             (0x5C in the generated driver).
//   return  — AXI-Lite control: ap_ctrl_hs (start/done/idle/ready)
//
// The kernel processes outer × size elements:
//   c[o * (a_inc+b_inc) + i] = act(op(a[o*a_inc + i], b[o*b_inc + i]))
//   for o in 0..outer-1, i in 0..size-1
//
// For non-broadcast use set outer=1, a_inc=0, b_inc=0.
// For broadcast with a advancing: outer=N, a_inc=aligned_chunk, b_inc=0.
// For broadcast with b advancing: outer=N, a_inc=0, b_inc=aligned_chunk.
//
// For unary operations (OP_RELU, OP_RELU6) only a[] is read; no AXI
// transactions are issued on the gmem1 port and b_addr is ignored.
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
    unsigned      act
);
