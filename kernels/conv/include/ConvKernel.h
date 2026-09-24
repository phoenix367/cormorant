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
// Identical in semantics to the version in matmul/include/MatmulKernel.h;
// duplicated here so the conv/ subdirectory is self-contained.
// ---------------------------------------------------------------------------

template<typename T>
struct saturate_to {
    template<typename From>
    static T cast(From v) { return T(v); }
};

#ifdef CONV_HAVE_APFIXED
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
// Weight / bias port width and DDR layout (§2.32).
//
// The weight and bias ports are hls::burst_maxi<WeightWord> with
// WeightWord = ap_uint<kWeightPortBits> (128 bits = kWeightPortElems
// Data_t lanes per beat, lane 0 in the lowest-addressed bytes).  The
// consumer's weight cache stores one 256-bit word per (m-tile, m1, khi,
// kwi) holding all kTileIC input-channel lanes, so the DDR layout keeps
// those lanes ADJACENT:
//
//   standard (is_depthwise=0), tile-major:
//     weight[out_ch][ic_tiles][kh][kw][lanes(ict)]
//       lanes(ict) = kTileIC, except the LAST tile is a half tile of
//                    kWeightPortElems lanes when its valid lane count is
//                    <= kWeightPortElems (conv_last_tile_lanes) — §2.34:
//                    a 3-channel stem then moves 8 lanes per kernel
//                    position instead of 16.
//       elem((m, ict, khi, kwi, ic_l)) =
//         m*conv_weight_per_m(in_ch,kh,kw) + ict*kh*kw*kTileIC
//         + (khi*kw + kwi)*lanes(ict) + ic_l
//     ic_tiles = ceil(in_ch / kTileIC); lanes ic_l >= in_ch - ict*kTileIC of
//     the last tile are zero (the kernel masks them anyway).  One
//     (m, ict) slab is kh*kw*lanes(ict) contiguous elements = one burst,
//     and every slab starts on a port-word boundary.
//   depthwise (is_depthwise=1):
//     weight[out_ch][conv_dw_stride(kh, kw)]
//       elem((m, khi, kwi)) = m*conv_dw_stride + khi*kw + kwi
//     conv_dw_stride = kh*kw rounded up to kWeightPortElems so every
//     channel starts on a port-word boundary.
//   bias[conv_bias_numel(out_ch)] — out_ch rounded up to kWeightPortElems
//     (the kernel reads whole words; the pad lanes are ignored).
//
// The inference scheduler emits exactly this layout
// (inference-scheduler/src/nodes.py, ConvNode packing) and the C-sim /
// RTL fixtures are packed by TestConvSim.cpp's pack_conv_weights().
// The buffers handed to the kernel must be 16-byte aligned.
// ---------------------------------------------------------------------------
template<typename T> struct ConvDataBits;   // undefined for unsupported Data_t
template<> struct ConvDataBits<float> { static constexpr unsigned value = 32; };
#ifdef CONV_HAVE_APFIXED
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct ConvDataBits<ap_fixed<W, I, Q, O, N>> { static constexpr unsigned value = W; };
#endif

static constexpr unsigned kDataBits        = ConvDataBits<Data_t>::value;
static constexpr unsigned kWeightPortBits  = 128;
static constexpr unsigned kWeightPortElems = kWeightPortBits / kDataBits;
typedef ap_uint<kWeightPortBits> WeightWord;

static_assert(kWeightPortBits % kDataBits == 0,
              "weight port width must be a whole number of Data_t lanes");
static_assert((kTileIC * kDataBits) % kWeightPortBits == 0,
              "one kTileIC-lane weight word must be a whole number of port beats");
static_assert(kWeightPortElems <= kTileM && kTileM % kWeightPortElems == 0,
              "bias_buf banking assumes a port beat covers at most one m-tile");

inline unsigned conv_round_up(unsigned v, unsigned q) { return ((v + q - 1) / q) * q; }
inline unsigned conv_ic_tiles(unsigned in_ch)          { return (in_ch + kTileIC - 1) / kTileIC; }
inline unsigned conv_dw_stride(unsigned kh, unsigned kw){ return conv_round_up(kh * kw, kWeightPortElems); }
inline unsigned conv_bias_numel(unsigned out_ch)       { return conv_round_up(out_ch, kWeightPortElems); }
// Lanes stored per kernel position in the LAST ic-tile (§2.34 half tile).
inline unsigned conv_last_tile_lanes(unsigned in_ch) {
    const unsigned rem = in_ch - (conv_ic_tiles(in_ch) - 1) * kTileIC;   // 1..kTileIC
    return (rem <= kWeightPortElems) ? kWeightPortElems : kTileIC;
}
inline unsigned conv_tile_lanes(unsigned in_ch, unsigned ict) {
    return (ict + 1 == conv_ic_tiles(in_ch)) ? conv_last_tile_lanes(in_ch) : kTileIC;
}
// Elements per output channel m in the packed standard layout.
inline unsigned conv_weight_per_m(unsigned in_ch, unsigned kh, unsigned kw) {
    return kh * kw * ((conv_ic_tiles(in_ch) - 1) * kTileIC + conv_last_tile_lanes(in_ch));
}
inline unsigned conv_weight_numel(unsigned out_ch, unsigned in_ch,
                                  unsigned kh, unsigned kw, bool is_depthwise) {
    return is_depthwise ? out_ch * conv_dw_stride(kh, kw)
                        : out_ch * conv_weight_per_m(in_ch, kh, kw);
}
inline unsigned conv_weight_index(unsigned m, unsigned ict, unsigned khi, unsigned kwi,
                                  unsigned ic_l, unsigned in_ch, unsigned kh, unsigned kw) {
    return m * conv_weight_per_m(in_ch, kh, kw) + ict * kh * kw * kTileIC
         + (khi * kw + kwi) * conv_tile_lanes(in_ch, ict) + ic_l;
}

// Data_t <-> raw lane bits (the byte image the port carries).
#ifdef CONV_HAVE_APFIXED
inline Data_t conv_lane_to_data(ap_uint<kDataBits> bits) {
    Data_t v; v.range(kDataBits - 1, 0) = bits; return v;
}
inline ap_uint<kDataBits> conv_data_to_lane(Data_t v) {
    return v.range(kDataBits - 1, 0);
}
#else
inline Data_t conv_lane_to_data(ap_uint<kDataBits> bits) {
    union { unsigned u; float f; } c; c.u = bits.to_uint(); return c.f;
}
inline ap_uint<kDataBits> conv_data_to_lane(Data_t v) {
    union { unsigned u; float f; } c; c.f = v; return ap_uint<kDataBits>(c.u);
}
#endif

// ---------------------------------------------------------------------------
// C/RTL co-simulation transfer depths — single source of truth.
//
// cosim of an m_axi kernel needs a fixed transfer depth per pointer port.
// These macros feed BOTH sides of the cosim contract:
//   * the depth=<N> hints on ConvKernel.cpp's m_axi pragmas (size of the
//     cosim memory model — must be >= the kernel's largest access), and
//   * the fixed global buffers in test/TestConvSim.cpp's CONV_COSIM path
//     (must be >= depth, or wrapc reads past the array end and SIGSEGVs).
// They are cosim-only: depth does NOT constrain the synthesised AXI master
// (runtime addresses) or the exported IP.  TestConvSim.cpp skips any case
// whose tensors exceed these bounds under cosim (large convs are left to
// plain C-sim); bump a port's value here to pull a larger case into cosim.
// ---------------------------------------------------------------------------
#define CONV_COSIM_DEPTH_X       8192
#define CONV_COSIM_DEPTH_WEIGHT  16384   /* Data_t elements */
#define CONV_COSIM_DEPTH_BIAS    256     /* Data_t elements */
#define CONV_COSIM_DEPTH_Y       8192
/* The weight / bias ports are WeightWord-wide, so their depth= hints (and
 * the cosim buffers) are in words: elements / kWeightPortElems. */
#define CONV_COSIM_DEPTH_WEIGHT_WORDS (CONV_COSIM_DEPTH_WEIGHT / 8)
#define CONV_COSIM_DEPTH_BIAS_WORDS   (CONV_COSIM_DEPTH_BIAS / 8)

// ---------------------------------------------------------------------------
// ConvKernel — 2-D convolution following ONNX Conv semantics.
//
// Supports standard convolution (group=1) and depthwise convolution
// (group=in_ch, i.e. is_depthwise=1).
//
// Computes Y = conv(X, weight) + bias for a batch of 2-D feature maps in
// NCHW layout.  Padding is applied implicitly: any input index outside
// [0, in_h) × [0, in_w) is treated as zero.
//
// Parameters:
//   x, weight, bias, y   DDR pointers (NCHW row-major; see layout below).
//   batch                N: number of input images.
//   in_ch                C: input channels.
//   in_h, in_w           H, W: input spatial dimensions.
//   out_ch               M: output channels.
//   out_h, out_w         oH, oW: output spatial dimensions (precomputed by caller).
//   kh, kw               Kernel height/width.  Must satisfy kh ≤ kMaxKH and
//                        kw ≤ kMaxKW (compile-time limits); enforced by caller.
//   stride_h, stride_w   Convolution stride (≥ 1).
//   dilation_h, dilation_w  Kernel dilation (1 = standard convolution).
//   pad_top, pad_left    Zero-padding rows/columns before the input.
//                        pad_bottom and pad_right are implicit: out-of-bounds
//                        input accesses are zero-padded by the bounds check.
//   has_bias             0 = do not read bias pointer; 1 = add bias[m] to y.
//   is_depthwise         0 = standard conv (group=1);
//                        1 = depthwise conv (group=in_ch).
//                            weight layout changes to [out_ch][1][kh][kw].
//
// Memory layout (row-major, NCHW):
//   x     [batch][in_ch][in_h ][in_w ]
//   weight, bias: tile-major packed layout — see "Weight / bias port width
//                 and DDR layout" above (conv_weight_index / conv_dw_stride /
//                 conv_bias_numel).
//   y     [batch][out_ch][out_h][out_w]
//
// AXI interface (in ConvKernel.cpp):
//   x, weight, bias → m_axi gmem0/1/2  (read ports; all hls::burst_maxi —
//                     x Data_t-wide, weight/bias WeightWord-wide)
//   y               → m_axi gmem3      (write port, hls::burst_maxi)
//   all scalars     → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
// All four DDR ports are hls::burst_maxi<> (§2.27/§2.28/§2.32): explicit
// read_requests ahead of the data (x per row, weight per (m, ic-tile) slab,
// bias once) and one write_request per contiguous output run, instead of
// burst inference.  A plain pointer converts implicitly (hls_burst_maxi.h's
// pointer constructor); the weight / bias pointers must point at
// WeightWord-packed buffers (see the layout note above).
void ConvKernel(
    hls::burst_maxi<Data_t>     x,
    hls::burst_maxi<WeightWord> weight,
    hls::burst_maxi<WeightWord> bias,
    hls::burst_maxi<Data_t>     y,
    unsigned      batch,
    unsigned      in_ch,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_ch,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      has_bias,
    unsigned      is_depthwise
);
