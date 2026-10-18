// Copyright 2025 GradeBuilder SL
// Modifications copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

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
// Identical in semantics to the version in MatmulKernel.h (the MatMul model);
// duplicated here so this model is self-contained.
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
//     the last tile MUST be zero (§2.40: the MAC grid multiplies them
//     unmasked against the producer's zero patch lanes).  One
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
// RTL fixtures are packed by TestConvModel.cpp's pack_conv_weights().
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

// ---------------------------------------------------------------------------
// Output port (§2.38).  y is an hls::burst_maxi<YWord> port of the same
// 128-bit width: 8 Data_t lanes per beat, lane 0 in the lowest-addressed
// bytes.  The NCHW layout of y is UNCHANGED — the writer re-aligns each
// channel run (which starts at an arbitrary element index m*out_h*out_w)
// onto DDR words and writes the run's first / last words with byte
// strobes, so the lanes of a word that belong to a neighbouring channel
// (or lie past the tensor's end, inside the 16-byte tail pad) are never
// touched.  The buffer handed to the kernel must be 16-byte aligned and
// padded to a whole word (the scheduler aligns every buffer to 64 bytes).
// ---------------------------------------------------------------------------
static constexpr unsigned kYPortBits  = kWeightPortBits;
static constexpr unsigned kYPortElems = kYPortBits / kDataBits;
typedef ap_uint<kYPortBits> YWord;
// Number of kYPortElems-lane words touched by the element run
// [off, off + count) — the run may start and end mid-word.
inline unsigned conv_y_words_for(unsigned off, unsigned count) {
    return (off + count - 1) / kYPortElems - off / kYPortElems + 1;
}

// ---------------------------------------------------------------------------
// Input port (§2.39).  x is an hls::burst_maxi<XWord> port of the same
// 128-bit width (8 Data_t lanes per beat, lane 0 lowest-addressed).  The
// NCHW layout is unchanged: the row loader requests, per (input row,
// channel), the whole words that cover the row segment it needs — the
// first / last word may carry lanes outside the segment, which the line
// buffer fill drops.  The buffer must be 16-byte aligned.
// ---------------------------------------------------------------------------
static constexpr unsigned kXPortBits  = kWeightPortBits;
static constexpr unsigned kXPortElems = kXPortBits / kDataBits;
typedef ap_uint<kXPortBits> XWord;
inline unsigned conv_x_words_for(unsigned off, unsigned count) {
    return (off + count - 1) / kXPortElems - off / kXPortElems + 1;
}

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
//   x, weight, bias → m_axi gmem0/1/2  (read ports; all hls::burst_maxi,
//                     all 128-bit: XWord / WeightWord, §2.32 / §2.39)
//   y               → m_axi gmem3      (write port, hls::burst_maxi<YWord>,
//                     128-bit beats with byte strobes at run ends, §2.38)
//   all scalars     → s_axilite, bundle=ctrl
// ---------------------------------------------------------------------------
// All four DDR ports are hls::burst_maxi<> (§2.27/§2.28/§2.32): explicit
// read_requests ahead of the data (x per row, weight per (m, ic-tile) slab,
// bias once) and one write_request per contiguous output run, instead of
// burst inference.  A plain pointer converts implicitly (hls_burst_maxi.h's
// pointer constructor); the weight / bias pointers must point at
// WeightWord-packed buffers (see the layout note above), x and y at
// 16-byte-aligned, whole-word-padded buffers (see "Input port" /
// "Output port").
void ConvKernel(
    hls::burst_maxi<XWord>      x,
    hls::burst_maxi<WeightWord> weight,
    hls::burst_maxi<WeightWord> bias,
    hls::burst_maxi<YWord>      y,
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
    unsigned      is_depthwise,
    // The y channel pattern (RPU_PLAN §13.2, registers 0xC8 / 0xD0 / 0xD8):
    // channel c's plane starts at (c mod y_cn) * y_cs + (c div y_cn) * y_gs
    // elements — y_cs = 0: out_h * out_w (dense), y_cn = 0: one group.
    // Contract: y_cn a multiple of kTileM and <= kMaxOutCh, batch == 1 when
    // y_cs or y_cn is set (else nothing is written, as the RTL).
    unsigned      y_cs = 0,
    unsigned      y_cn = 0,
    unsigned      y_gs = 0
);
