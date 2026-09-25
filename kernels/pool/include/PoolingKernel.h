#pragma once

#include "Config.h"
#include "ap_int.h"
#include "hls_burst_maxi.h"

// ---------------------------------------------------------------------------
// saturate_cast<T>(v)
//
// Converts v to type T with saturation when T is an ap_fixed type; passes
// through unchanged for float / double / integer types.
//
// Identical in semantics to the version in conv/include/ConvKernel.h;
// duplicated here so the pool/ subdirectory is self-contained.
// ---------------------------------------------------------------------------

template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef POOL_HAVE_APFIXED
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct saturate_to<ap_fixed<W, I, Q, O, N>> {
    template<typename From>
    static ap_fixed<W, I, Q, O, N> cast(From v) {
        return ap_fixed<W, I, Q, O, N>(ap_fixed<W, I, AP_TRN, AP_SAT>(v));
    }
};
#endif

template<typename T, typename From>
inline T saturate_cast(From v) {
    return saturate_to<T>::template cast<From>(v);
}

// ---------------------------------------------------------------------------
// C/RTL co-simulation transfer depths — single source of truth.
//
// cosim of an m_axi kernel needs a fixed transfer depth per pointer port.
// These macros feed BOTH sides of the cosim contract:
//   * the depth=<N> hints on PoolingKernel.cpp's m_axi pragmas (size of the
//     cosim memory model — must be >= the kernel's largest access), and
//   * the fixed global buffers in test/TestPoolingSim.cpp's POOL_COSIM path
//     (must be >= depth, or wrapc reads past the array end and SIGSEGVs).
// They are cosim-only: depth does NOT constrain the synthesised AXI master
// (runtime addresses) or the exported IP.  TestPoolingSim.cpp skips any case
// whose tensors exceed these bounds under cosim (large pools are left to
// plain C-sim); bump a port's value here to pull a larger case into cosim.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 128-bit x and y ports (POOL_OPTIMIZATION.md §2.13 / §2.14).
//
// Both ports are hls::burst_maxi<PoolWord> carrying kPoolPortElems elements
// per beat.  The NCHW layout is unchanged.
//
//   x: the row_loader requests, per (input row, channel) run, the aligned
//      word range that covers it and drops the lanes outside the run.  The
//      x BASE address must be 16-byte aligned (the scheduler aligns every
//      buffer); the last word of a run may extend up to kPoolPortElems - 1
//      elements past the tensor end (bytes must be mappable — the
//      scheduler pads buffers).
//   y: the writer issues one burst per (output row, channel) run and writes
//      the run's first / last words with BYTE STROBES covering only the
//      run's own lanes (hls::burst_maxi::write(word, byte_enable)), so a
//      neighbouring channel's lanes in the same word and the lanes past the
//      tensor end are never modified.  The y BASE address must be 16-byte
//      aligned; no tail padding of the y buffer is required.
// ---------------------------------------------------------------------------
template<typename T> struct PoolDataBits { static constexpr unsigned value = 8 * sizeof(T); };
#ifdef POOL_HAVE_APFIXED
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct PoolDataBits<ap_fixed<W, I, Q, O, N>> { static constexpr unsigned value = W; };
#endif
static constexpr unsigned kPoolDataBits  = PoolDataBits<Data_t>::value;
static constexpr unsigned kPoolPortBits  = 128;
static constexpr unsigned kPoolPortElems = kPoolPortBits / kPoolDataBits;
static_assert(kPoolPortBits % kPoolDataBits == 0, "Data_t must divide the 128-bit port");
typedef ap_uint<kPoolPortBits> PoolWord;

#ifdef POOL_HAVE_APFIXED
inline Data_t pool_lane_to_data(ap_uint<kPoolDataBits> bits) {
    Data_t v; v.range(kPoolDataBits - 1, 0) = bits; return v;
}
inline ap_uint<kPoolDataBits> pool_data_to_lane(Data_t v) {
    return v.range(kPoolDataBits - 1, 0);
}
#else
inline Data_t pool_lane_to_data(ap_uint<kPoolDataBits> bits) {
    union { unsigned u; float f; } c; c.u = bits.to_uint(); return c.f;
}
inline ap_uint<kPoolDataBits> pool_data_to_lane(Data_t v) {
    union { unsigned u; float f; } c; c.f = v; return ap_uint<kPoolDataBits>(c.u);
}
#endif

// Number of words that cover `count` elements starting at element `off`.
inline unsigned pool_words_for(unsigned off, unsigned count) {
    return (off + count - 1) / kPoolPortElems - off / kPoolPortElems + 1;
}

#define POOL_COSIM_DEPTH_X  8192
#define POOL_COSIM_DEPTH_X_WORDS  (POOL_COSIM_DEPTH_X / kPoolPortElems + 1)
#define POOL_COSIM_DEPTH_Y  8192
#define POOL_COSIM_DEPTH_Y_WORDS  (POOL_COSIM_DEPTH_Y / kPoolPortElems + 1)

// ---------------------------------------------------------------------------
// PoolingKernel — 2-D pooling following ONNX semantics.
//
// Supports MaxPool, AveragePool, LpPool and their Global variants.  Global
// variants are handled by the caller passing pool_h=in_h, pool_w=in_w,
// stride_h=stride_w=1, pad_top=pad_left=0.
//
// pool_type         : 0=kPoolMax  1=kPoolAvg  2=kPoolLp
// lp_order          : 1 or 2 (used when pool_type=kPoolLp)
// count_include_pad : 0 = exclude padding pixels from average denominator
//                     1 = include  (used when pool_type=kPoolAvg only)
//
// Output dimensions are precomputed by the caller:
//   out_h = floor((in_h + pad_top + pad_bottom - dil_h*(pool_h-1) - 1) / stride_h) + 1
//   out_w = floor((in_w + pad_left + pad_right - dil_w*(pool_w-1) - 1) / stride_w) + 1
// Symmetric padding is assumed: pad_bottom = pad_top, pad_right = pad_left.
// Asymmetric trailing padding is handled implicitly by the bounds check.
//
// Memory layout (NCHW, row-major):
//   x[batch][channels][in_h ][in_w ]
//   y[batch][channels][out_h][out_w]
//
// AXI interface (in PoolingKernel.cpp):
//   x → m_axi gmem0  (read,  hls::burst_maxi<PoolWord>)
//   y → m_axi gmem1  (write, hls::burst_maxi<PoolWord>, byte-strobed tails)
//   all scalars → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
void PoolingKernel(
    hls::burst_maxi<PoolWord> x,
    hls::burst_maxi<PoolWord> y,
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
);
