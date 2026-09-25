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
// Identical in semantics to the version in include/VectorOP.h; duplicated
// here so the matmul/ subdirectory is self-contained and does not depend on
// the VectorOPKernel headers.
// ---------------------------------------------------------------------------

// Primary template: no saturation (float, double, integer types).
template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef MATMUL_HAVE_APFIXED
// Partial specialisation for ap_fixed<W,I,Q,O,N>:
// rounds the value through AP_TRN, AP_SAT before narrowing to the target type.
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
// A / B port width — 128-bit words (§2.4 in MATMUL_OPTIMISATION.md).
//
// A and B are read through hls::burst_maxi<MatmulWord> ports carrying
// kMatmulPortElems elements per beat.  The DDR layout is UNCHANGED (plain
// row-major, as the scheduler and the host emit it): the kernel computes,
// per matrix row segment, the aligned word range that covers it and
// extracts the lanes, so any element offset (batch strides, m-tile column
// offsets, k-tile row offsets) is handled in-kernel.  The only requirement
// is that the a / b BASE addresses handed to the kernel are 16-byte
// aligned (the scheduler aligns every buffer to INFERENCE_ALIGN_BYTES).
// The last word of a row segment may extend up to kMatmulPortElems - 1
// elements past the matrix end; those lanes are discarded, but the bytes
// must be mappable (the scheduler pads buffers to the alignment).
// ---------------------------------------------------------------------------
template<typename T> struct MatmulDataBits { static constexpr unsigned value = 8 * sizeof(T); };
#ifdef MATMUL_HAVE_APFIXED
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct MatmulDataBits<ap_fixed<W, I, Q, O, N>> { static constexpr unsigned value = W; };
#endif
static constexpr unsigned kMatmulDataBits  = MatmulDataBits<Data_t>::value;
static constexpr unsigned kMatmulPortBits  = 128;
static constexpr unsigned kMatmulPortElems = kMatmulPortBits / kMatmulDataBits;
static_assert(kMatmulPortBits % kMatmulDataBits == 0, "Data_t must divide the 128-bit port");
static_assert(kMaxK % kMatmulPortElems == 0, "kMaxK must be a multiple of the lanes per word");
typedef ap_uint<kMatmulPortBits> MatmulWord;

// Data_t <-> raw lane bits (the byte image the port carries).
#ifdef MATMUL_HAVE_APFIXED
inline Data_t matmul_lane_to_data(ap_uint<kMatmulDataBits> bits) {
    Data_t v; v.range(kMatmulDataBits - 1, 0) = bits; return v;
}
inline ap_uint<kMatmulDataBits> matmul_data_to_lane(Data_t v) {
    return v.range(kMatmulDataBits - 1, 0);
}
#else
inline Data_t matmul_lane_to_data(ap_uint<kMatmulDataBits> bits) {
    union { unsigned u; float f; } c; c.u = bits.to_uint(); return c.f;
}
inline ap_uint<kMatmulDataBits> matmul_data_to_lane(Data_t v) {
    union { unsigned u; float f; } c; c.f = v; return ap_uint<kMatmulDataBits>(c.u);
}
#endif

// ---------------------------------------------------------------------------
// Packed (tile-major) B layout — MATMUL_OPTIMISATION.md §3b.
//
// With b_packed = 1 the kernel expects B stored as
//
//     B_packed[(mt * k + kk) * kTileM + m1]   =  B[kk][mt * kTileM + m1]
//
// i.e. one contiguous [k][kTileM] block per m-tile, m padded with zeros to
// matmul_packed_m(m) = ceil(m / kTileM) * kTileM.  A (m_tile, k_tile) block
// is then ONE contiguous run of k_valid * kTileM elements
// (k_valid * kMatmulWordsPerTileRow words) instead of k_valid separate
// ≤ kMatmulMaxRowWords-word row segments — the scheduler emits constant
// weights this way.
// Batch slices are k * packed_m elements apart; b_batch_stride /
// b_outer offsets are element counts in the PACKED image.
// ---------------------------------------------------------------------------
static constexpr unsigned kMatmulWordsPerTileRow = kTileM / kMatmulPortElems;
static_assert(kTileM % kMatmulPortElems == 0, "kTileM must be a multiple of the lanes per word");
inline unsigned matmul_packed_m(unsigned m) {
    return ((m + kTileM - 1) / kTileM) * kTileM;
}
inline unsigned matmul_packed_index(unsigned kk, unsigned mm, unsigned k) {
    return ((mm / kTileM) * k + kk) * kTileM + (mm % kTileM);
}

// Rotate the lanes of a port word right by `shift` lanes: lane j of the
// result is lane (j + shift) mod kMatmulPortElems of the input.  Written as
// a chain of constant rotates selected by `shift` so HLS builds ONE
// kMatmulPortElems-way mux of the whole word; a per-lane / per-column
// variable part-select (`word.range(16 * (l + 1) - 1, 16 * l)` with a
// runtime l) costs a full 128-bit barrel shifter per destination
// (MATMUL_OPTIMISATION.md §6: 17.9 k + 9 k LUT in the two scatter loops).
inline MatmulWord matmul_rotate_lanes(MatmulWord w, unsigned shift) {
    #pragma HLS INLINE
    MatmulWord r = w;
    for (unsigned s = 1; s < kMatmulPortElems; s++) {
        #pragma HLS UNROLL
        if (shift == s)
            r = (w >> (kMatmulDataBits * s)) |
                (w << (kMatmulPortBits - kMatmulDataBits * s));
    }
    return r;
}

// Lane j (a compile-time constant at every use) of a port word.
inline Data_t matmul_word_lane(MatmulWord w, unsigned j) {
    #pragma HLS INLINE
    return matmul_lane_to_data(w.range(kMatmulDataBits * (j + 1) - 1, kMatmulDataBits * j));
}

// Bits needed to hold the value v (e.g. 9 for 256): narrow HLS counters.
constexpr unsigned matmul_bits_for(unsigned v) { return v == 0 ? 0 : 1 + matmul_bits_for(v >> 1); }
// Most words a run of kTileM elements at any lane offset can span.
static constexpr unsigned kMatmulMaxRowWords = (kTileM + kMatmulPortElems - 1) / kMatmulPortElems + 1;

// Number of words that cover `count` elements starting at element `off`.
inline unsigned matmul_words_for(unsigned off, unsigned count) {
    const unsigned w_lo = off / kMatmulPortElems;
    const unsigned w_hi = (off + count - 1) / kMatmulPortElems;
    return w_hi - w_lo + 1;
}

// ---------------------------------------------------------------------------
// C/RTL co-simulation transfer depths — single source of truth.
//
// cosim of an m_axi kernel needs a fixed transfer depth per pointer port.
// These macros feed BOTH sides of the cosim contract:
//   * the depth=<N> hints on MatmulKernel.cpp's m_axi pragmas (size of the
//     cosim memory model — must be >= the kernel's largest access), and
//   * the fixed global buffers in test/TestMatmulSim.cpp's MATMUL_COSIM path
//     (must be >= depth, or wrapc reads past the array end and SIGSEGVs).
// They are cosim-only: depth does NOT constrain the synthesised AXI master
// (runtime addresses) or the exported IP.  TestMatmulSim.cpp skips any case
// whose matrices exceed these bounds under cosim (large matmuls are left to
// plain C-sim); bump a port's value here to pull a larger case into cosim.
// A and B are in ELEMENTS; their *_WORDS variants (one spare word for the
// over-read of a row's last word) size the 128-bit ports.
// ---------------------------------------------------------------------------
#define MATMUL_COSIM_DEPTH_A  8192
#define MATMUL_COSIM_DEPTH_B  16384
#define MATMUL_COSIM_DEPTH_C  4096
#define MATMUL_COSIM_DEPTH_A_WORDS  (MATMUL_COSIM_DEPTH_A / kMatmulPortElems + 1)
#define MATMUL_COSIM_DEPTH_B_WORDS  (MATMUL_COSIM_DEPTH_B / kMatmulPortElems + 1)

// ---------------------------------------------------------------------------
// MatmulKernel — tiled matrix multiplication.
//
// Computes C = A × B for a batch of 2-D matrix products.  Batch broadcasting
// is encoded via stride=0: set a_batch_stride=0 to reuse A for every batch
// iteration (A broadcasts), b_batch_stride=0 to reuse B.
//
// Parameters:
//   a, b, c         Pointers to row-major matrices in DDR.
//   n               Rows of A (rows of C).
//   k               Inner dimension (cols of A = rows of B).
//                   Must be ≤ kMaxK (compile-time limit, enforced by caller).
//   m               Cols of B (cols of C).
//   batch           Total number of 2-D products to compute.
//   a_batch_stride  Elements to advance 'a' per batch step (0 = broadcasts).
//   b_batch_stride  Elements to advance 'b' per batch step (0 = broadcasts).
//   c_batch_stride  Elements to advance 'c' per batch step.
//   b_packed        0: B is row-major [k][m]; 1: B is in the tile-major
//                   packed layout (matmul_packed_index), b_batch_stride in
//                   packed elements.
//
// Memory layout (row-major):
//   A[n][k] : a[row*k + col]
//   B[k][m] : b[row*m + col]
//   C[n][m] : c[row*m + col]
//
// AXI interface (added in HLS kernel — see MatmulKernel.cpp pragma comments):
//   a, b  → m_axi, gmem0 / gmem1  (128-bit burst_maxi read ports)
//   c     → m_axi, gmem2          (write port)
//   all scalars → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
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
);
