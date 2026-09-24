// ---------------------------------------------------------------------------
// ConvKernel.cpp — 2-D convolution kernel.
//
// Implements the ONNX Conv operator (group=1 or group=in_ch) in a tiled
// structure that maps cleanly to Vitis HLS synthesis.
// See doc/CONV_PLAN.md for a full explanation of the architecture, tiling
// strategy, and II=1 rationale.
//
// Top-level dataflow (HLS DATAFLOW):
//
//   ConvKernel
//     bias_producer           ──bias_stream────►  process_conv_kernel_tile
//     input_patch_producer    ──patch_stream───►                              │
//     stream_load_weights     ──weight_stream──►                              │
//                                                              └──acc_stream──► write_output_tile
//                                                                                      │
//     x/weight/bias (DDR gmem0/1/2)                                                    ▼
//                                                                               y (DDR gmem3)
//
// mt-hoist (the m_tile loop is INSIDE the spatial nest):
//   * input_patch_producer (one assembler shared by both modes, §2.14)
//     reads each unique x[] pixel from DDR exactly once per (ni, oh, ow),
//     buffers the patch on-chip, then broadcasts it m_tiles times into
//     patch_stream — eliminating the m_tiles× DDR re-read of x.
//   * bias_producer loads the entire bias vector once into bias_buf and
//     streams one kTileM-lane BiasVec per (r, mt) — one padded accumulator
//     word per beat (§2.25).
//   * process_conv_kernel_tile iterates (ni, oh, ow, mt) — same patch is
//     consumed by every output-channel tile back-to-back.
//   * write_output_tile drains acc_stream in (ni, chunk, mt, m1, oh, ow)
//     order — channel-major so each channel's chunk rows are one
//     contiguous DDR run (write bursts, §2.22).
//
// Loop structure — standard conv (is_depthwise=0):
//
//   batch / oh / ow loops    — iterate over output spatial positions
//     m_tile loop            — tiles the output channel dimension (TILE_M)
//       compute tile (producer):
//         [acc init]   — zero acc[], overlay m_valid lanes from bias_stream
//         ic_tile loop — tiles the input channel dimension (TILE_IC)
//           load patch  — current (oh,ow) input patch to on-chip BRAM
//           load w_buf  — weight tile for (m_tile, ic_tile) to on-chip BRAM
//           accumulate  — II=1 K-reduction over ic×kH×kW with TILE_M lanes
//         drain        — push m_valid lanes of acc[] to acc_stream
//       write output (consumer):
//         copy acc_stream → DDR y  (already saturated in Phase 3)
//
// Loop structure — depthwise conv (is_depthwise=1): same outer nest, the
// per-tile compute swaps in compute_depthwise_conv_tile (no ic_tile loop,
// per-lane patch and weight loads, kH×kW reduction).
//
// II=1 strategy (§2.4):
//   The flat counter ri runs 0 .. ic_valid*kh*kw*TILE_M-1 (standard) or
//   0 .. kh*kw*TILE_M-1 (depthwise).
//   m1 = ri & (TILE_M - 1)  — compile-time bitmask, no divider.
//   acc[m1] is written every TILE_M cycles, satisfying the dependency distance
//   requirement (TILE_M ≥ ap_fixed<16,8> MAC latency ≈ 3 cycles).
//
// Depthwise weight layout: weight[out_ch][1][kh][kw] (no in_ch dimension).
//   Offset for channel m: m * kh * kw + khi * kw + kwi
// ---------------------------------------------------------------------------

#include <algorithm>
#include "hls_stream.h"
#include "hls_task.h"

#include "ConvKernel.h"
#include "ConvMacGrid.h"

#ifndef __SYNTHESIS__
#define DEBUG_LOAD_DATA_CACHING
#endif

// ---------------------------------------------------------------------------
// PatchVec — channel-packed patch stream element.
//
// The patch path (input_patch_producer → consumer) previously carried
// one Data_t per stream beat, so the consumer's patch
// drain ran at kTileIC·kh·kw cycles per (oh, ow).  PatchVec packs a full
// channel column — kTileIC lanes — into a single beat, so the producer
// emits and the consumer drains one beat per (khi, kwi): the patch drain
// drops to kh·kw cycles.
//
// The standard path fills all kTileIC lanes; the depthwise path fills the
// first kTileM lanes (its parallel axis) and zero-pads the rest — every
// lane is written so no 'X' reaches RTL.  At kTileIC=16, ap_fixed<16,8>
// this is a 256-bit FIFO element.
// ---------------------------------------------------------------------------
struct PatchVec {
    Data_t lane[kTileIC];
};

// ---------------------------------------------------------------------------
// BiasVec — one m-tile of initial accumulator values per stream beat (§2.25).
//
// bias_producer used to push one AccData_t per (pixel, mt, m1); the
// consumer's Phase 1 then spent m_valid cycles per (pixel, mt) filling the
// padded accumulator word lane by lane.  With the §2.23 word layout Phase 1
// wants exactly one kTileM-lane word per (pixel, mt), so the stream now
// carries it pre-packed: lanes >= m_valid are zero (the padding lanes).
// ---------------------------------------------------------------------------
struct BiasVec {
    AccData_t lane[kTileM];
};

// WeightVec (the weight stream beat and the weight cache word) lives in
// ConvMacGrid.h since §2.35.
static constexpr unsigned kWordsPerWeightVec = kTileIC / kWeightPortElems;

#ifdef DEBUG_LOAD_DATA_CACHING
#include <cassert>
#include <cstdlib>
#include <map>
#include <list>
#include <iostream>

struct CycleCounters {
    unsigned mt;
    unsigned ni;
    unsigned oh;
    unsigned ow;
    unsigned ict;
    unsigned ic_l;
    unsigned khi;
    unsigned kwi;
};

inline std::ostream& operator<<(std::ostream& os, const CycleCounters& c) {
    os << "{mt=" << c.mt
       << " ni=" << c.ni
       << " oh=" << c.oh
       << " ow=" << c.ow
       << " ict=" << c.ict
       << " ic_l=" << c.ic_l
       << " khi=" << c.khi
       << " kwi=" << c.kwi
       << "}";
    return os;
}

typedef std::map<size_t, std::list<CycleCounters>> AddressMap_t;

#endif /* DEBUG_LOAD_DATA_CACHING */

// ---------------------------------------------------------------------------
// oh-chunking helper.
//
// The Option-A consumer keeps a persistent per-output accumulator in
// partial_outputs[kMaxAccPersistEntries] that survives across the ic-tile
// (standard) or mt-tile (depthwise) reduction.  When out_h·out_w·out_ch
// exceeds kMaxAccPersistEntries, the full output no longer fits on-chip.
//
// Rather than degrading to a no-persistent-acc mode (which would re-read
// every input pixel ic_tiles · m_tiles times per (ni, oh, ow) — bandwidth
// catastrophe), the kernel splits the output along the oh axis into chunks
// whose footprint fits the buffer:
//
//     oh_per_chunk = max(1, kMaxAccPersistEntries / (out_w · out_ch_padded))
//     num_chunks   = ceil(out_h / oh_per_chunk)
//
// where out_ch_padded = ceil(out_ch / kTileM)·kTileM is the accumulator
// buffer's channel stride (§2.23): partial_outputs is laid out
// [pixel][m_tile][kTileM] so every m-tile's 8 lanes form one aligned,
// reshaped 256-bit word and the consumer loads / stores a whole tile's
// accumulators in ONE cycle instead of m_valid cycles.
//
// Each chunk runs the full Option-A pipeline for its oh sub-range; between
// chunks the patch producer's line_buf is invalidated so the kh-row sliding
// window for the next chunk's first oh is reloaded from DDR (duplicate read
// of ~kh rows per (ni, c, chunk transition)).  Weights are re-streamed per
// chunk too — for the standard path this is just stream replay (the inline
// load loop already replays per (oh, ow), so per-chunk replay is the same
// bandwidth); for the depthwise path the small weight slice is reloaded
// num_chunks times per (ni, mt).
//
// Relaxed constraint: out_w · ceil(out_ch/kTileM)·kTileM  ≤  kMaxAccPersistEntries
// (was: out_h · out_w · out_ch  ≤  kMaxAccPersistEntries; the kTileM
// padding is the §2.23 layout and is mirrored by the scheduler validator).
//
// Second bound — line_buf residency under M-grouping (standard path only):
// when num_m_groups > 1 the patch producer replays a chunk's whole
// (oh, ow) sweep from line_buf once per M-group WITHOUT re-reading DDR.
// line_buf holds only kMaxLineBufRows rows (circular slot = ih & mask),
// so every input row the chunk touches must still be resident when the
// second group restarts at the chunk's first oh:
//
//     (oh_per_chunk - 1)·stride_h + (kh-1)·dilation_h + 1  ≤  kMaxLineBufRows
//
// compute_conv_geometry() clamps oh_per_chunk to that bound whenever more
// than one M-group is in play.  The cost is more chunks (weights and bias
// are re-streamed per chunk); the alternative — reloading x per M-group —
// would give up the read-x-once property the grouping exists for.
//
// When out_w · out_ch > kMaxAccPersistEntries the buffer cannot hold even
// one output row.  oh_per_chunk is clamped to 1 in that case so the kernel
// still synthesises and (for small enough total) still works in C-sim, but
// large layers with that property will overflow partial_outputs.  Catching
// it remains the scheduler validator's responsibility.
// ---------------------------------------------------------------------------
static inline void compute_oh_chunking(
    unsigned out_h, unsigned out_w, unsigned out_ch,
    unsigned& oh_per_chunk, unsigned& num_chunks
) {
    const unsigned m_tiles  = (out_ch + kTileM - 1) / kTileM;
    const unsigned row_size = out_w * (m_tiles * kTileM);   // padded stride
    unsigned per = (row_size > 0) ? (kMaxAccPersistEntries / row_size) : out_h;
    if (per == 0)        per = 1;
    if (per > out_h)     per = out_h;
    oh_per_chunk = per;
    num_chunks   = (out_h + per - 1) / per;
}

// ---------------------------------------------------------------------------
// M-grouping helper (standard path only).
//
// Eliminates per-(oh, ow) weight DDR replay by caching one (ict, M-group)
// weight slab on-chip and reusing it across the chunk's spatial sweep.
// When m_tiles ≤ kMaxMperGroup the cache holds ALL m_tiles' weights for
// the current ict and weights are read from DDR exactly once per
// (ni, chunk, ict).  When m_tiles > kMaxMperGroup the kernel processes
// M in groups of mt_per_group mt-tiles; weights are reloaded num_m_groups
// times per (ni, chunk, ict), still much cheaper than the pre-caching
// out_h·out_w replay.
//
// Patches are re-emitted by the input patch producer per m_group so the
// consumer can stream-read kTileIC·kh·kw patch values once per
// (m_group, oh, ow) and reuse across mt_in_group with cached weights.
// No DDR re-read for patches — line_buf is retained across the entire
// (m_group, oh, ow) sweep within one (chunk, ict).
// ---------------------------------------------------------------------------
static inline void compute_m_grouping(
    unsigned out_ch,
    unsigned& mt_per_group, unsigned& num_m_groups
) {
    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;
    unsigned mtg = kMaxMperGroup;
    if (mtg == 0)         mtg = 1;
    if (mtg > m_tiles)    mtg = m_tiles;
    mt_per_group = mtg;
    num_m_groups = (m_tiles + mtg - 1) / mtg;
}

// ---------------------------------------------------------------------------
// ow-tiling helper.
//
// Lifts the previous hard cap "in_w ≤ kMaxInW" by splitting the output
// column axis into tiles whose input-column window fits the compile-time
// kMaxLineBufCols bound on line_buf's column dim.  Within an ow-tile the
// patch producer uses circular column indexing (iw_slot = iw &
// (kMaxLineBufCols-1)) so the same line_buf slots are reused across rows.
// Between ow-tiles, line_buf is invalidated and the next tile's iw range
// is reloaded from DDR — the (kw-1)·dilation_w-col overlap is re-fetched.
//
// Tile geometry (per ow-tile spanning ows [ow_start, ow_start+ow_per_tile)):
//   first iw = ow_start * stride_w - pad_left
//   last  iw = (ow_start + ow_per_tile - 1) * stride_w + (kw-1)*dilation_w - pad_left
//   iw range width = (ow_per_tile-1)*stride_w + (kw-1)*dilation_w + 1
//
// We need iw_range_width ≤ kMaxLineBufCols, so:
//   ow_per_tile = (kMaxLineBufCols - ((kw-1)*dilation_w + 1)) / stride_w + 1
//
// When (kw-1)*dilation_w + 1 > kMaxLineBufCols the kernel-width window
// doesn't fit even one ow — falls back to ow_per_tile=1 (the
// kMaxLineBufCols static_assert in Config.h.in plus the per-kernel-size
// validation in the inference scheduler should make this unreachable for
// well-formed inputs).
// ---------------------------------------------------------------------------
static inline void compute_ow_tiling(
    unsigned out_w, unsigned kw,
    unsigned stride_w, unsigned dilation_w,
    unsigned& ow_per_tile, unsigned& num_ow_tiles
) {
    const unsigned window_w = (kw - 1) * dilation_w + 1;
    unsigned per;
    if (window_w >= kMaxLineBufCols) {
        // Window doesn't leave room for any additional output; clamp.
        per = 1;
    } else {
        per = (kMaxLineBufCols - window_w) / (stride_w > 0 ? stride_w : 1) + 1;
        if (per == 0) per = 1;
    }
    if (per > out_w) per = out_w;
    ow_per_tile = per;
    num_ow_tiles = (out_w + per - 1) / per;
}

// ---------------------------------------------------------------------------
// Per-invocation tile geometry (§2.19).
//
// oh-chunking, M-grouping and ow-tiling each need an integer division by a
// RUNTIME divisor (out_w·out_ch, mt_per_group, stride_w, …), which HLS
// synthesises as a multi-cycle sequential divider.  The geometry is
// invariant for a whole kernel invocation, yet each dataflow stage used to
// call compute_*() itself — so the same ~5 dividers were instantiated once
// per stage (16 dividers across the kernel, ≈6.3k FF / 3.8k LUT).
//
// ConvKernel now computes the geometry ONCE and passes this struct to every
// stage, collapsing the divider count to a single shared set.  The struct
// crosses the DATAFLOW process boundaries as one stable scalar channel.
// ---------------------------------------------------------------------------
struct ConvGeometry {
    unsigned oh_per_chunk;
    unsigned num_chunks;
    unsigned mt_per_group;
    unsigned num_m_groups;
    unsigned ow_per_tile;
    unsigned num_ow_tiles;
};

static inline ConvGeometry compute_conv_geometry(
    unsigned out_h, unsigned out_w, unsigned out_ch,
    unsigned kh, unsigned kw,
    unsigned stride_h, unsigned stride_w,
    unsigned dilation_h, unsigned dilation_w,
    unsigned is_depthwise
) {
    // Runs once per invocation (~129 cycles).  A #pragma HLS DATAFLOW here
    // to overlap the three independent divider chains was tried and dropped:
    // the canonical form (struct returned, fields written by 3 processes)
    // segfaults Vitis HLS 2025.2's scalar-propagation pass, and the
    // non-canonical form draws "region may not be handled correctly"
    // warnings — not worth it to shave ~64 one-time cycles (0.02 % of runtime).
    ConvGeometry g;
    compute_oh_chunking(out_h, out_w, out_ch, g.oh_per_chunk, g.num_chunks);
    compute_m_grouping (out_ch, g.mt_per_group, g.num_m_groups);
    compute_ow_tiling  (out_w, kw, stride_w, dilation_w,
                        g.ow_per_tile, g.num_ow_tiles);

    // line_buf residency bound (see compute_oh_chunking header).  Only the
    // standard path replays patches per M-group; depthwise runs one group
    // and reloads rows as it sweeps, so it needs no cap.
    if (!is_depthwise && g.num_m_groups > 1) {
        const unsigned window_h = (kh - 1) * dilation_h + 1;
        unsigned cap = 1;
        if (window_h < kMaxLineBufRows) {
            cap = (kMaxLineBufRows - window_h)
                / (stride_h > 0 ? stride_h : 1) + 1;
        }
        if (cap < g.oh_per_chunk) {
            g.oh_per_chunk = cap;
            g.num_chunks   = (out_h + cap - 1) / cap;
        }
    }
    return g;
}

// ---------------------------------------------------------------------------
// Write outputs to DDR (§2.22: channel-major runs; §2.27: explicit
// hls::burst_maxi write bursts).
//
// Loop nest matches the consumer's Phase-3 drain order —
// (ni, chunk, mt, m1, oh_local, ow).  For each (ni, chunk, channel) the
// chunk_oh_count·out_w outputs are CONTIGUOUS in y (NCHW: a channel's
// rows are adjacent), so the inner pipelined loop walks a single
// sequential run and HLS infers an AXI write burst.  The previous
// pixel-major order strode by out_h·out_w between consecutive stores, so
// every 16-bit output was a lone single-beat transaction — measured at
// 11.7 cycles each and ~45 % of a 64-channel layer's runtime (see
// CONV_2D_GRID_PLAN.md §2).
//
// acc_stream already carries saturated Data_t — the AccData_t→Data_t
// saturate_cast was hoisted into process_conv_kernel_tile's Phase-3
// drain (so the inter-stage FIFO is Data_t-wide, not AccData_t-wide).
// This stage is therefore a pure stream→DDR copy.
// ---------------------------------------------------------------------------
// Output write requests are issued in pieces of at most kWriteReqElems
// elements — one max-length AXI burst each (max_write_burst_length=256
// beats; the burst_maxi port is 16 bits wide, one element per beat) — and
// at most kWriteInFlight requests are left without a write_response().
// Both limits keep the number of unacknowledged bursts below the port's
// num_write_outstanding=16.  A single write_request for a whole channel
// run (up to chunk_oh_count*out_w elements — 4032 on MobileNet v2's first
// 1x1 projection, 16 bursts) deadlocked the m_axi adapter on the board:
// its response FIFO filled before the run's data was written and the
// writer, still inside the run, never reached write_response().
// Reproduced in the RTL test stand by TestConvSim case 28e.
static constexpr unsigned kWriteReqElems = 256;
static constexpr unsigned kWriteInFlight = 8;

static void write_output_tile(
    hls::burst_maxi<Data_t> y,
    hls::stream<Data_t>&    acc_stream,
    unsigned                out_ch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                batch,
    ConvGeometry            geom
) {
    const unsigned m_tiles      = (out_ch + kTileM - 1) / kTileM;
    const unsigned oh_per_chunk = geom.oh_per_chunk;
    const unsigned num_chunks   = geom.num_chunks;
    // write_request()s whose write_response() is still owed.  Responses are
    // collected in a sliding window of kWriteInFlight so several runs' bursts
    // stay in flight (num_write_outstanding=16 on the port); blocking on the
    // previous run's response right after each run was traced to serialise
    // drain → transfer → response per run (1.77 cycles/element).
    unsigned pending = 0;

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start       = chunk * oh_per_chunk;
        const unsigned oh_end         = std::min(out_h, oh_start + oh_per_chunk);
        const unsigned run_len        = (oh_end - oh_start) * out_w;

        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                // First element of this channel's rows [oh_start, oh_end).
                const unsigned base =
                    ((ni * out_ch + m_off + m1) * out_h + oh_start) * out_w;
                // §2.27 explicit bursts (hls::burst_maxi).  Burst
                // inference on a plain pointer was traced in RTL and never
                // overlapped draining with transmitting: with long bursts
                // the adapter buffered a whole burst at the 1-element/cycle
                // drain rate before sending it; with 16-beat bursts it
                // deferred every run's last burst to an end-of-kernel flush
                // and paused ~21 cycles between bursts.  write_request()
                // issues the address up front, write() streams the data as
                // it arrives from acc_stream, and the response is collected
                // in a sliding window so up to kWriteInFlight requests'
                // bursts are outstanding at once.  Each request covers at
                // most kWriteReqElems elements (one burst); a run longer
                // than that is split into consecutive requests.
                for (unsigned off = 0; off < run_len; off += kWriteReqElems) {
                    const unsigned rem = run_len - off;
                    const unsigned len = (rem < kWriteReqElems) ? rem : kWriteReqElems;
                    y.write_request(base + off, len);
                    for (unsigned i = 0; i < len; i++) {
                        #pragma HLS PIPELINE II=1
                        y.write(acc_stream.read());
                    }
                    pending++;
                    if (pending == kWriteInFlight) {
                        y.write_response();
                        pending--;
                    }
                }
            }
        } // m_tile loop
      } // chunk loop
    } // batch loop
    while (pending > 0) {
        y.write_response();
        pending--;
    }
}

// ---------------------------------------------------------------------------
// Bias producer (DATAFLOW source).
//
// Streams bias values to process_conv_kernel_tile in the order the consumer
// reads them after the mt-hoist optimisation:
//
//     for (r in batch*out_h*out_w):
//         for (mt in m_tiles): write(BiasVec{bias[m_off..m_off+kTileM)})
//
// Because the consumer's mt loop lives INSIDE the spatial nest, the
// producer needs all m_tiles' bias slices available simultaneously per
// `r` iteration.  The entire bias vector is therefore loaded once into
// bias_buf[kMaxOutCh] and replayed `reps` times.  When has_bias=0 no DDR
// transactions are issued and the producer pushes zero vectors directly.
//
// §2.37 depthwise: the consumer's flat sweep initialises every pixel's
// accumulator word from a per-TILE register, so it reads ONE BiasVec per
// (ni, chunk, mt) — reps = batch * num_chunks (set by ConvKernel).
// ---------------------------------------------------------------------------
static void bias_producer(
    hls::burst_maxi<WeightWord> bias,
    hls::stream<BiasVec>&   bias_stream,
    unsigned                out_ch,
    unsigned                reps,
    unsigned                has_bias
) {
    Data_t bias_buf[kMaxOutCh];
    // §2.25: one BiasVec (kTileM lanes) per beat — the gather below reads
    // lanes m_off..m_off+kTileM-1 in one cycle, so bank the buffer by the
    // lane index (m_off is a multiple of kTileM → lane m1 lives in bank m1).
    #pragma HLS ARRAY_PARTITION variable=bias_buf cyclic factor=kTileM dim=1

    if (has_bias) {
        // §2.32: the bias buffer in DDR is conv_bias_numel(out_ch) elements
        // (out_ch rounded up to a port word); read it as whole words,
        // kWeightPortElems lanes per cycle.
        const unsigned n_words = conv_bias_numel(out_ch) / kWeightPortElems;
        bias.read_request(0, n_words);
        for (unsigned w = 0; w < n_words; w++) {
            #pragma HLS PIPELINE II=1
            const WeightWord word = bias.read();
            for (unsigned j = 0; j < kWeightPortElems; j++) {
                #pragma HLS UNROLL
                const unsigned idx = w * kWeightPortElems + j;
                if (idx < kMaxOutCh) {
                    bias_buf[idx] = conv_lane_to_data(
                        word.range(kDataBits * (j + 1) - 1, kDataBits * j));
                }
            }
        }
    }

    const unsigned m_tiles = (out_ch + kTileM - 1) / kTileM;

    for (unsigned r = 0; r < reps; r++) {
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            #pragma HLS PIPELINE II=1
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            BiasVec v;
            #pragma HLS aggregate variable=v compact=byte
            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                #pragma HLS UNROLL
                v.lane[m1] = (has_bias && m1 < m_valid)
                           ? AccData_t(bias_buf[m_off + m1])
                           : AccData_t(0);
            }
            bias_stream.write(v);
        }
    }
}

// ---------------------------------------------------------------------------
// input_patch_producer — unified input patch ASSEMBLER (DATAFLOW source).
//
// Standard and depthwise convolution share one producer (§2.14): their
// patch assembly differs only in the channel-parallelism axis and the
// M-group replay, both folded into runtime values here.
//
//                         standard (is_depthwise=0)  depthwise (is_depthwise=1)
//   channel tile width    kTileIC input channels     kTileM channels
//   tile count ct_tiles   ceil(in_ch  / kTileIC)     ceil(out_ch / kTileM)
//   group replay          num_m_groups (§2.10)       1 (no weight caching)
//
// A single kTileIC-wide line_buf serves both modes; depthwise uses only
// banks [0, kTileM) and the PatchVec gather masks the rest to 0 — the
// same X-clean mask the standard partial-IC tail already relies on.
// Merging the former input_patch_producer_standard / _depthwise pair
// reclaims the depthwise producer's duplicate line_buf BRAM and control
// logic.
//
// Tiled-IC/M (Option-A) + M-grouping + ow-tiling.  Loop nest:
//
//     for ni: for chunk: for ct: for ow_tile: for grp:
//         for oh_in_chunk: for ow_in_tile: for ch_l, kh, kw
//
// ow_tile is OUTER of grp so line_buf's column window for the tile is
// loaded once per (ct, ow_tile) and reused across grp's patch
// re-emissions.  line_buf indexes both row and column dims circularly:
//
//     row_slot = ih & (kMaxLineBufRows - 1)
//     col_slot = iw & (kMaxLineBufCols - 1)
//
// kMaxLineBufCols is a compile-time power-of-2 bound on the column
// window; in_w is NOT capped, only the per-tile iw extent.  Wider inputs
// produce more ow_tiles, with the (kw-1)·dilation_w-col overlap
// re-fetched from DDR at each tile transition and the (kh-1)·stride_h
// row overlap re-fetched at each chunk transition.
//
// Per (ni, chunk, ct, ow_tile, grp, oh):
//   Phase 1 — load new rows × this ow_tile's iw range from DDR
//             (ch_valid channels at ch_off).  Only grp=0 loads;
//             grp>0 re-streams the cached rows from line_buf.
// Per (ni, chunk, ct, ow_tile, grp, oh, ow):
//   Phase 2 — stream kh × kw channel-packed PatchVecs into patch_stream.
//
// Constraints: (kh-1)*dilation_h + 1 <= kMaxLineBufRows,
//              (kw-1)*dilation_w + 1 <= kMaxLineBufCols.
// ---------------------------------------------------------------------------
static void input_patch_producer(
    hls::burst_maxi<Data_t> x,
    hls::stream<PatchVec>&  patch_stream,
    unsigned             batch,
    unsigned             in_ch,
    unsigned             in_h,
    unsigned             in_w,
    unsigned             out_ch,
    unsigned             out_h,
    unsigned             out_w,
    unsigned             kh,
    unsigned             kw,
    unsigned             stride_h,
    unsigned             stride_w,
    unsigned             dilation_h,
    unsigned             dilation_w,
    unsigned             pad_top,
    unsigned             pad_left,
    unsigned             is_depthwise,
    ConvGeometry         geom
) {
    const unsigned in_hw    = in_h * in_w;

    // Channel-parallelism axis: standard tiles in_ch by kTileIC, depthwise
    // tiles out_ch by kTileM (in_ch == out_ch in depthwise mode).
    const unsigned ct_width = is_depthwise ? kTileM : kTileIC;
    const unsigned total_ch = is_depthwise ? out_ch : in_ch;
    const unsigned ct_tiles = (total_ch + ct_width - 1) / ct_width;

    // Tile geometry computed once by ConvKernel (§2.19).
    const unsigned oh_per_chunk = geom.oh_per_chunk;
    const unsigned num_chunks   = geom.num_chunks;

    // Depthwise caches its weight slice once per (chunk, mt) — no M-group
    // replay — so it runs a single group.
    const unsigned num_groups   = is_depthwise ? 1u : geom.num_m_groups;

    const unsigned ow_per_tile  = geom.ow_per_tile;
    const unsigned num_ow_tiles = geom.num_ow_tiles;

    // One kTileIC-wide line buffer for both modes; depthwise uses banks
    // [0, kTileM).  dim=1 partitioned complete → kTileIC independent
    // banks so Phase 2 can gather a full PatchVec per cycle.
    Data_t line_buf[kTileIC][kMaxLineBufRows][kMaxLineBufCols];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

#ifdef DEBUG_LOAD_DATA_CACHING
    // Optional duplicate-read report (CONV_DEBUG_READS=1 in the env).
    const bool debug_reads = std::getenv("CONV_DEBUG_READS") != nullptr;
    AddressMap_t read_addresses;
    // C-sim residency invariant: which absolute (ih, iw) each line_buf
    // cell currently holds.  Phase 2 asserts the cell it reads holds the
    // pixel it thinks it does — a circular-slot overwrite (the §2.21 bug
    // class) then fails loudly instead of silently corrupting outputs.
    static long line_tag[kTileIC][kMaxLineBufRows][kMaxLineBufCols];
    for (unsigned a = 0; a < kTileIC; a++)
        for (unsigned r = 0; r < kMaxLineBufRows; r++)
            for (unsigned c = 0; c < kMaxLineBufCols; c++)
                line_tag[a][r][c] = -1;
    auto tag_of = [&](unsigned ch_l_, int ih_, int iw_) -> long {
        return ((long)ch_l_ << 40) | ((long)ih_ << 20) | (long)iw_;
    };
#endif

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start = chunk * oh_per_chunk;
        const unsigned oh_end   = std::min(out_h, oh_start + oh_per_chunk);

        for (unsigned ct = 0; ct < ct_tiles; ct++) {
            const unsigned ch_off   = ct * ct_width;
            const unsigned ch_valid = std::min(ct_width, total_ch - ch_off);

          for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
            const unsigned ow_start = owt * ow_per_tile;
            const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile);

            // Input column range needed for this ow_tile.
            const int iw_load_start = (int)(ow_start * stride_w) - (int)pad_left;
            const int iw_load_last  = (int)((ow_end - 1) * stride_w
                                            + (kw - 1) * dilation_w)
                                      - (int)pad_left;

            // Highest absolute input row currently resident in line_buf for
            // THIS (chunk, ct, ow_tile).  Initialise so the first oh in this
            // chunk triggers a fresh kh-row load.  line_buf is reloaded once
            // per ow_tile (different iw range); within an ow_tile rows are
            // retained across groups (grp>0 sees last_loaded_row already at
            // the chunk's max and Phase 1 loads nothing).  That replay is
            // only valid because compute_conv_geometry() caps oh_per_chunk
            // so the chunk's full input-row span fits kMaxLineBufRows when
            // num_groups > 1 — otherwise circular slot reuse would have
            // overwritten the first oh's rows before grp 1 re-reads them.
            int last_loaded_row =
                (int)(oh_start * stride_h) - (int)pad_top - 1;

          for (unsigned grp = 0; grp < num_groups; grp++) {
            for (unsigned oh = oh_start; oh < oh_end; oh++) {
                const int ih_window_max = (int)(oh * stride_h)
                                        - (int)pad_top
                                        + (int)((kh - 1) * dilation_h);

                // -------------------------------------------------------
                // Phase 1: load any rows the current (oh, ct) window
                // needs that are not yet in line_buf.  Only ch_valid
                // channels are loaded here (ch_off..ch_off+ch_valid-1),
                // and only this ow_tile's iw range (clamped to [0,in_w)).
                // -------------------------------------------------------
                int load_start = last_loaded_row + 1;
                if (load_start < 0) load_start = 0;
                int load_end = ih_window_max;
                if (load_end >= (int)in_h) load_end = (int)in_h - 1;

                // Clip the ow_tile's iw range to valid input columns.
                int iw_clipped_start = iw_load_start;
                if (iw_clipped_start < 0) iw_clipped_start = 0;
                int iw_clipped_last  = iw_load_last;
                if (iw_clipped_last >= (int)in_w)
                    iw_clipped_last = (int)in_w - 1;

                // Number of input columns this ow_tile needs from each row
                // (0 when the whole window is in the padding).
                const int iw_count_i = iw_clipped_last - iw_clipped_start + 1;
                const unsigned iw_count = (iw_count_i > 0) ? (unsigned)iw_count_i : 0u;

                for (int ih = load_start; ih <= load_end; ih++) {
                    const unsigned slot =
                        (unsigned)ih & (kMaxLineBufRows - 1);
                    // §2.28 explicit read bursts: request every channel's
                    // run of this row FIRST, then drain the data.  With a
                    // plain pointer each (row, channel) run was one inferred
                    // burst issued only after the previous one completed —
                    // the RTL trace showed 69 cycles per 16-element burst
                    // with a single read in flight, which made depthwise
                    // layers read-bound at ~4.3 cycles/element.  Up to
                    // ch_valid (<= kTileIC = num_read_outstanding) requests
                    // are now in flight while the first one's data streams
                    // in.
                    if (iw_count > 0) {
                        for (unsigned ch_l = 0; ch_l < ch_valid; ch_l++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned c     = ch_off + ch_l;
                            const unsigned x_row = (ni * in_ch + c) * in_hw
                                                 + (unsigned)ih * in_w;
                            x.read_request(x_row + (unsigned)iw_clipped_start,
                                           iw_count);
                        }
                    }
                    for (unsigned ch_l = 0; ch_l < ch_valid; ch_l++) {
                        const unsigned c     = ch_off + ch_l;
                        const unsigned x_row = (ni * in_ch + c) * in_hw
                                             + (unsigned)ih * in_w;
                        for (int iw = iw_clipped_start;
                             iw <= iw_clipped_last; iw++) {
                            #pragma HLS PIPELINE II=1
                            const size_t addr = x_row + (unsigned)iw;
                            const unsigned col_slot =
                                (unsigned)iw & (kMaxLineBufCols - 1);
                            line_buf[ch_l][slot][col_slot] = x.read();

#ifdef DEBUG_LOAD_DATA_CACHING
                            line_tag[ch_l][slot][col_slot] = tag_of(ch_l, ih, iw);
                            if (debug_reads) {
                            CycleCounters counters;
                            counters.mt   = is_depthwise ? ct : 0u;
                            counters.ni   = ni;
                            counters.ict  = is_depthwise ? (unsigned)-1 : ct;
                            counters.ic_l = ch_l;
                            counters.oh   = oh;
                            counters.ow   = owt;
                            counters.khi  = (unsigned)ih;
                            counters.kwi  = (unsigned)iw;
                            read_addresses[addr].push_back(counters);
                            }
#endif /* DEBUG_LOAD_DATA_CACHING */
                        }
                    }
                }
                if (load_end > last_loaded_row) {
                    last_loaded_row = load_end;
                }

                for (unsigned ow = ow_start; ow < ow_end; ow++) {

                    // ---------------------------------------------------
                    // Phase 2: stream a kh × kw block of PatchVecs into
                    // patch_stream — one beat per (khi, kwi), each beat
                    // packing all kTileIC lanes.  Lanes ic_l >= ch_valid
                    // are zero-padded (the partial-IC tail for standard,
                    // the kTileM..kTileIC-1 tail for depthwise); the
                    // consumer's accumulate ignores the padding.
                    // ---------------------------------------------------
                    for (unsigned khi = 0; khi < kh; khi++) {
                        const int ih = (int)(oh * stride_h + khi * dilation_h)
                                    - (int)pad_top;
                        const bool ih_ok = (ih >= 0 && (unsigned)ih < in_h);
                        const unsigned slot = ih_ok
                            ? ((unsigned)ih & (kMaxLineBufRows - 1))
                            : 0u;
                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                            #pragma HLS PIPELINE II=1
                            const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                        - (int)pad_left;
                            const bool iw_ok = (iw >= 0 && (unsigned)iw < in_w);
                            const unsigned col_slot = iw_ok
                                ? ((unsigned)iw & (kMaxLineBufCols - 1))
                                : 0u;

                            PatchVec v;
                            #pragma HLS aggregate variable=v compact=byte

                            for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                                #pragma HLS UNROLL
                                const bool ch_ok = (ic_l < ch_valid);
                                v.lane[ic_l] = (ch_ok && ih_ok && iw_ok)
                                    ? line_buf[ic_l][slot][col_slot]
                                    : Data_t(0);
#ifdef DEBUG_LOAD_DATA_CACHING
                                if (ch_ok && ih_ok && iw_ok &&
                                    line_tag[ic_l][slot][col_slot] != tag_of(ic_l, ih, iw)) {
                                    std::cerr << "line_buf residency violated: ch_l=" << ic_l
                                              << " wants (ih=" << ih << ", iw=" << iw
                                              << ") slot(" << slot << "," << col_slot
                                              << ") holds tag " << line_tag[ic_l][slot][col_slot]
                                              << " at ni=" << ni << " chunk=" << chunk
                                              << " ct=" << ct << " owt=" << owt << " grp=" << grp
                                              << " oh=" << oh << " ow=" << ow << std::endl;
                                    assert(!"line_buf residency invariant");
                                }
#endif
                            }
                            patch_stream.write(v);
                        }
                    }
                } // ow loop
            } // oh loop
          } // group loop
          } // ow_tile loop
        } // channel-tile loop
      } // chunk loop
    } // batch loop

#ifdef DEBUG_LOAD_DATA_CACHING
    if (debug_reads) for (auto it : read_addresses) {
        if (it.second.size() > 1) {
            std::cout << it.first << " --> " << std::endl;

            for (auto l_item : it.second) {
                std::cout << "\t" << l_item << std::endl;
            }
        }
    }
#endif /* DEBUG_LOAD_DATA_CACHING */
}

// ---------------------------------------------------------------------------
// stream_load_weights — DATAFLOW source for the weight m_axi port.
//
// Streams weights from DDR (gmem1) to process_conv_kernel_tile via
// weight_stream in the exact order the consumer reads them.  The win is
// not fewer DDR transactions but overlap: the weight loads now run
// concurrently with bias_producer, input_patch_producer, and the
// consumer's accumulate / partial_outputs passes, instead of
// serialising inside the (oh, ow, mt) inner nest.
//
// Standard path (is_depthwise=0):
//   Iteration order matches process_conv_kernel_tile's Phase 2a nest with
//   M-grouping + ow-tiling — (ni, chunk, ict, ow_tile, mg, mt_in_group) —
//   so per (ict, ow_tile, mg) the producer emits the full
//   (mt_per_group_actual × m_valid × ic_valid × kh × kw) slab of weights
//   in (mt_in_group, m1, ic_l, khi, kwi) order (kwi fastest).  Each m1
//   stripe is read from a contiguous DDR region (m_axi infers bursts).
//   Weights are emitted ONCE per (ni, chunk, ict, ow_tile, mg) — no
//   spatial replay across (oh, ow_in_tile).  When out_w fits one ow_tile
//   AND m_tiles ≤ kMaxMperGroup each weight is read from DDR exactly once
//   per (ni, chunk); otherwise replay multiplies by num_ow_tiles ×
//   num_m_groups (still tiny compared to the pre-caching out_h·out_w
//   replay).
//
// Depthwise path (is_depthwise=1):
//   Iteration order matches Phase 2b's once-per-(chunk, mt) hoist —
//   (ni, chunk, mt) — so per (ni, chunk, mt) the producer emits m_valid *
//   kh * kw values in (m1, khi, kwi) order.  Weights are re-fetched from
//   DDR num_chunks times per (ni, mt); the per-mt slice is tiny
//   (kTileM·kh·kw values) so this overhead is negligible.
// ---------------------------------------------------------------------------
static void stream_load_weights(
    hls::burst_maxi<WeightWord> weight,
    hls::stream<WeightVec>& weight_stream,
    unsigned ic_tiles,
    unsigned m_tiles,
    unsigned in_ch,
    unsigned out_ch,
    unsigned out_w,
    unsigned out_h,
    unsigned kw,
    unsigned kh,
    unsigned stride_w,
    unsigned dilation_w,
    unsigned batch,
    unsigned is_depthwise,
    ConvGeometry geom
)
{
    // Tile geometry computed once by ConvKernel (§2.19).  Weight emission
    // depends only on the chunk / group / tile COUNTS, not the per-* extents.
    const unsigned num_chunks   = geom.num_chunks;
    const unsigned mt_per_group = geom.mt_per_group;
    const unsigned num_m_groups = geom.num_m_groups;
    const unsigned num_ow_tiles = geom.num_ow_tiles;

    // §2.32 packed layout (ConvKernel.h): one (m, ict) slab is
    // kh*kw*lanes contiguous elements = slab_words port beats, and every
    // lanes/kWeightPortElems beats form one WeightVec (all ic-lanes of one
    // kernel position; §2.34: the last ic-tile may be a HALF tile of
    // kWeightPortElems lanes = one beat per position, lanes above it zero).
    // Depthwise: one channel is dw_words beats of kernel positions, emitted
    // as dw_vecs WeightVecs (the last one padded).
    const unsigned per_m_words = conv_weight_per_m(in_ch, kh, kw) / kWeightPortElems;
    const unsigned last_lanes  = conv_last_tile_lanes(in_ch);
    const unsigned dw_words    = conv_dw_stride(kh, kw) / kWeightPortElems;
    const unsigned dw_vecs     = (dw_words + kWordsPerWeightVec - 1) / kWordsPerWeightVec;

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        if (!is_depthwise) {
            // ---- Standard: once per (ni, chunk, ict, ow_tile, mg) ----
            for (unsigned ict = 0; ict < ic_tiles; ict++) {
                // Half tile (§2.34): one beat per kernel position.
                const unsigned lanes         = (ict + 1 == ic_tiles) ? last_lanes : kTileIC;
                const unsigned words_per_pos = lanes / kWeightPortElems;      // 1 or 2
                const unsigned slab_words    = kh * kw * words_per_pos;
                const unsigned tile_word_off = ict * kh * kw * kTileIC / kWeightPortElems;

              for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
                for (unsigned mg = 0; mg < num_m_groups; mg++) {
                    const unsigned mt_base = mg * mt_per_group;
                    const unsigned mt_in_group_count =
                        (mt_base + mt_per_group <= m_tiles)
                            ? mt_per_group
                            : (m_tiles - mt_base);

                    for (unsigned mt_in_group = 0;
                         mt_in_group < mt_in_group_count; mt_in_group++) {
                        const unsigned mt     = mt_base + mt_in_group;
                        const unsigned m_off  = mt * kTileM;
                        const unsigned m_valid =
                            std::min(kTileM, out_ch - m_off);

                        // Request every m1's slab of this tile up front
                        // (m_valid <= kTileM <= num_read_outstanding), then
                        // drain them in the same order.
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned word_off =
                                (m_off + m1) * per_m_words + tile_word_off;
                            weight.read_request(word_off, slab_words);
                        }
                        for (unsigned m1 = 0; m1 < m_valid; m1++) {
                            WeightVec v;
                            #pragma HLS aggregate variable=v compact=byte
                            for (unsigned l = 0; l < kTileIC; l++) {
                                #pragma HLS UNROLL
                                v.lane[l] = Data_t(0);   // half tile: upper lanes stay 0
                            }
                            unsigned part = 0;
                            for (unsigned w = 0; w < slab_words; w++) {
                                #pragma HLS PIPELINE II=1
                                const WeightWord word = weight.read();
                                for (unsigned j = 0; j < kWeightPortElems; j++) {
                                    #pragma HLS UNROLL
                                    // Lane (part*kWeightPortElems + j) of the
                                    // vector comes from beat `part`; only that
                                    // beat's lanes are updated each cycle.
                                    for (unsigned q = 0; q < kWordsPerWeightVec; q++) {
                                        #pragma HLS UNROLL
                                        if (q == part) {
                                            v.lane[q * kWeightPortElems + j] =
                                                conv_lane_to_data(word.range(
                                                    kDataBits * (j + 1) - 1, kDataBits * j));
                                        }
                                    }
                                }
                                if (part + 1 == words_per_pos) {
                                    weight_stream.write(v);
                                    part = 0;
                                } else {
                                    part++;
                                }
                            }
                        }
                    }
                } // mg
              } // ow_tile
            } // ict
        } else {
            // -------- Depthwise: once per (ni, chunk, mt) --------
            for (unsigned mt = 0; mt < m_tiles; mt++) {
                const unsigned m_off   = mt * kTileM;
                const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                    #pragma HLS PIPELINE II=1
                    weight.read_request((m_off + m1) * dw_words, dw_words);
                }
                for (unsigned m1 = 0; m1 < m_valid; m1++) {
                    WeightVec v;
                    #pragma HLS aggregate variable=v compact=byte
                    for (unsigned w = 0; w < dw_vecs * kWordsPerWeightVec; w++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned part = w % kWordsPerWeightVec;
                        WeightWord word = 0;
                        if (w < dw_words) word = weight.read();   // pad beats stay 0
                        for (unsigned j = 0; j < kWeightPortElems; j++) {
                            #pragma HLS UNROLL
                            for (unsigned q = 0; q < kWordsPerWeightVec; q++) {
                                #pragma HLS UNROLL
                                if (q == part) {
                                    v.lane[q * kWeightPortElems + j] =
                                        conv_lane_to_data(word.range(
                                            kDataBits * (j + 1) - 1, kDataBits * j));
                                }
                            }
                        }
                        if (part == kWordsPerWeightVec - 1) {
                            weight_stream.write(v);
                        }
                    }
                }
            } // mt
        } // depthwise
      } // chunk
    } // batch
}

// ---------------------------------------------------------------------------
// process_conv_kernel_tile — DATAFLOW consumer.
//
// Output is processed in chunks along the oh axis (see compute_oh_chunking).
// Each chunk holds chunk_oh_count·out_w·out_ch accumulators in
// partial_outputs[] and runs the full Option-A three-phase pipeline for its
// sub-range:
//
//   Per (ni, chunk):
//     Phase 1 (init):  standard only — drain chunk_oh_count*out_w*m_tiles
//                      BiasVec words into partial_outputs[] (URAM-resident),
//                      indexed by oh_local = oh - oh_start.
//     Phase 2 (accum): standard   — for ict OUTER, (oh_in_chunk, ow, mt)
//                                   inner; load patch[kTileIC][kh][kw],
//                                   weight[kTileM][kTileIC][kh][kw],
//                                   reduce ic_valid*kh*kw*kTileM at II=1.
//                      depthwise  — for mt OUTER, one flat II=1 loop over
//                                   (oh_in_chunk, ow_in_tile, khi*kw+kwi)
//                                   per ow_tile (§2.37); w_buf[kTileM][kh*kw]
//                                   loaded ONCE per (chunk, mt); the word is
//                                   seeded from the tile's BiasVec and
//                                   stored write-only at the window's end.
//     Phase 3 (drain): saturate_cast partial_outputs to Data_t and push
//                      to acc_stream in (mt, m1, oh_in_chunk, ow) order —
//                      channel-major, so write_output_tile's matching nest
//                      writes each channel's chunk rows as one contiguous
//                      burst (§2.22).
//
//   Both producers read each x pixel from DDR once per (ni, c, chunk):
//   line_buf is retained across oh WITHIN a (chunk, ict|mt) but reloads
//   the (kh-1)-row overlap at chunk boundaries.
//
// partial_outputs layout (§2.23): [pixel][m_tile][kTileM] — flat index
//     word = (oh_local·out_w + ow)·m_tiles + mt,  entry = word·kTileM + m1
// so each m-tile's kTileM accumulators are one aligned word of the
// cyclic-factor-kTileM RESHAPEd array and Phase 2's acc load / store is a
// single wide access.  Lanes m1 ≥ m_valid of the last tile are padding:
// written (zero at init, whatever the grid leaves there after) but never
// drained.
//
// Memory constraint: out_w * m_tiles*kTileM <= kMaxAccPersistEntries
// (one padded row fits).
// ---------------------------------------------------------------------------
static void process_conv_kernel_tile(
    hls::stream<PatchVec>&  patch_stream,
    hls::stream<WeightVec>& weight_stream,
    hls::stream<BiasVec>&   bias_stream,
    hls::stream<Data_t>&    acc_stream,
    unsigned                batch,
    unsigned                in_ch,
    unsigned                in_h,
    unsigned                in_w,
    unsigned                out_ch,
    unsigned                out_h,
    unsigned                out_w,
    unsigned                kh,
    unsigned                kw,
    unsigned                stride_h,
    unsigned                stride_w,
    unsigned                dilation_h,
    unsigned                dilation_w,
    unsigned                pad_top,
    unsigned                pad_left,
    unsigned                is_depthwise,
    ConvGeometry            geom
) {
    const unsigned m_tiles  = (out_ch + kTileM  - 1) / kTileM;
    const unsigned ic_tiles = (in_ch  + kTileIC - 1) / kTileIC;

    // Tile geometry computed once by ConvKernel (§2.19).
    const unsigned oh_per_chunk = geom.oh_per_chunk;
    const unsigned num_chunks   = geom.num_chunks;

    // §2.35 w_cache ping-pong: two weight banks in ONE array, the bank being
    // the slowest address bit (NOT partitioned).  Each of the kTileM m1-RAMs
    // (§2.24 banking: m1 partitioned, ic reshaped into 256-bit words) is
    // only kMaxMperGroup*kMaxKH*kMaxKW = 196 words deep, so doubling the
    // depth to 392 still fits the same 512-deep BRAM36 columns — the second
    // bank costs no BRAM, whereas two separate arrays cost a full second set
    // (and, with the bank MUX needing both read every cycle, HLS duplicated
    // them again: 334 BRAM18 = 115 % of the device).  The sweep reads bank
    // `wbank` while the prefetch writes bank !wbank: one read + one write
    // port per RAM.  Bank state persists across (ni, chunk, ict, owt, mg)
    // because the prefetch crosses all of those boundaries.
    // Layout: one RAM column per m1, flat (bank, tile, khi, kwi) word address
    // (w_cache_addr in ConvMacGrid.h).  The prefetch store selects its column
    // with an explicit unrolled compare so each RAM sees exactly ONE
    // conditional store per iteration — the 5-D form with a runtime m1 index
    // into the partitioned dimension made HLS emit two stores per RAM and
    // split the bank dimension into a second RAM set (II=2, 2x BRAM).
    WeightVec w_cache[kTileM][kWCacheWords];
    #pragma HLS ARRAY_PARTITION variable=w_cache complete dim=1
    #pragma HLS AGGREGATE       variable=w_cache compact=bit
    #pragma HLS BIND_STORAGE    variable=w_cache type=RAM_2P impl=BRAM
    // 1-bit bank select and narrow prefetch cursors: with plain `unsigned`
    // indices HLS could not bound them and lowered the prefetch stores as
    // masked PARTIAL writes (read-modify-write on the word).
    ap_uint<1> wbank        = 0;       // bank holding the CURRENT slab
    bool       w_prefetched = false;   // current slab already loaded by the previous sweep

    AccData_t partial_outputs[kMaxAccPersistEntries];
    // Bound to URAM: this is by far the largest on-chip buffer and the
    // design is BRAM-bound, while the XCK26's 64 URAM blocks (288 Kbit
    // each, 4096 AccData_t entries) are otherwise unused.  Relocating it
    // frees ~16 BRAM and lets kMaxAccPersistEntries grow into the idle
    // URAM pool.  RAM_2P — Phase 1/3 touch a single port (write-only /
    // read-only) and Phase 2a's read and write run in separate II=1
    // sub-loops, so two ports suffice and there is no tight RAW
    // recurrence that URAM's extra read latency could stall.
    #pragma HLS bind_storage variable=partial_outputs type=RAM_2P impl=URAM
    // §2.23: kTileM consecutive entries → one 256-bit word, so an aligned
    // m-tile is loaded / stored per cycle (UG1399: reshape "combines
    // elements into wider words instead of creating separate arrays").
    #pragma HLS ARRAY_RESHAPE variable=partial_outputs cyclic factor=kTileM dim=1

    for (unsigned ni = 0; ni < batch; ni++) {
      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        const unsigned oh_start        = chunk * oh_per_chunk;
        const unsigned oh_end          = std::min(out_h,
                                                  oh_start + oh_per_chunk);
        const unsigned chunk_oh_count  = oh_end - oh_start;

        // -------- Phase 1: init partial_outputs from bias_stream --------
        // One BiasVec beat = one padded accumulator word per (pixel, mt):
        // the producer already zeroed the padding lanes (§2.25).
        // Standard path only: the depthwise sweep (§2.37) touches each
        // (pixel, mt) word exactly once, so it seeds the accumulator from a
        // per-tile bias register and stores the word write-only.
        if (!is_depthwise)
        for (unsigned oh_local = 0; oh_local < chunk_oh_count; oh_local++) {
            for (unsigned ow = 0; ow < out_w; ow++) {
                for (unsigned mt = 0; mt < m_tiles; mt++) {
                    #pragma HLS PIPELINE II=1
                    const unsigned word = (oh_local * out_w + ow) * m_tiles + mt;
#ifdef DEBUG_LOAD_DATA_CACHING
                    assert(word * kTileM + kTileM <= kMaxAccPersistEntries &&
                           "partial_outputs capacity (padded row) exceeded");
#endif
                    const BiasVec v = bias_stream.read();
                    for (unsigned m1 = 0; m1 < kTileM; m1++) {
                        #pragma HLS UNROLL
                        partial_outputs[word * kTileM + m1] = v.lane[m1];
                    }
                }
            }
        }

        // -------- Phase 2a: standard accumulate (ict OUTER, ow_tile, m_group) --------
        // Weight caching: per (ict, ow_tile, m_group) the mt_per_group_actual
        // mt-tiles' worth of weights are streamed into w_cache ONCE and reused
        // across the ow_tile's (oh_local, ow_in_tile) sweep — no per-(oh, ow)
        // replay.  Patches are streamed once per (ow_tile, m_group, oh_local,
        // ow_in_tile); within the m_group's mt_in_group loop the same patch
        // is reused with cached weights for each output-channel tile.
        if (!is_depthwise) {
            const unsigned mt_per_group = geom.mt_per_group;
            const unsigned num_m_groups = geom.num_m_groups;
            const unsigned ow_per_tile  = geom.ow_per_tile;
            const unsigned num_ow_tiles = geom.num_ow_tiles;

            for (unsigned ict = 0; ict < ic_tiles; ict++) {
                const unsigned ic_off   = ict * kTileIC;
                const unsigned ic_valid = std::min(kTileIC, in_ch - ic_off);

              for (unsigned owt = 0; owt < num_ow_tiles; owt++) {
                const unsigned ow_start = owt * ow_per_tile;
                const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile);

              for (unsigned mg = 0; mg < num_m_groups; mg++) {
                const unsigned mt_base = mg * mt_per_group;
                const unsigned mt_in_group_count =
                    (mt_base + mt_per_group <= m_tiles)
                        ? mt_per_group
                        : (m_tiles - mt_base);

                // ---- §2.35 w_cache ping-pong ----
                // Slabs arrive on weight_stream in exactly this loop order
                // (ni, chunk, ict, owt, mg).  The slab for THIS group is
                // already in bank `wbank` unless this is the very first slab
                // of the invocation (then it is loaded here, blocking).  The
                // NEXT slab is prefetched into bank wbank^1 while this
                // group's spatial sweep runs: one non-blocking WeightVec
                // read per sweep iteration, and a blocking tail after the
                // sweep for whatever is still missing.  The prefetch is
                // non-blocking so a slow weight producer never stalls the
                // MAC pipeline.
                if (!w_prefetched) {
                    for (unsigned t = 0; t < mt_in_group_count; t++) {
                        const unsigned mv =
                            std::min(kTileM, out_ch - (mt_base + t) * kTileM);
                        for (unsigned m1 = 0; m1 < mv; m1++) {
                            for (unsigned khi = 0; khi < kh; khi++) {
                                for (unsigned kwi = 0; kwi < kw; kwi++) {
                                    #pragma HLS PIPELINE II=1
                                    const WeightVec wv = weight_stream.read();
                                    const unsigned  wa = w_cache_addr(wbank, t, khi, kwi);
                                    for (unsigned c = 0; c < kTileM; c++) {
                                        #pragma HLS UNROLL
                                        if (c == m1) w_cache[c][wa] = wv;
                                    }
                                }
                            }
                        }
                    }
                }

                // Identify the next slab (same order the producer emits).
                const bool last_slab =
                    (ni + 1 == batch) && (chunk + 1 == num_chunks) &&
                    (ict + 1 == ic_tiles) && (owt + 1 == num_ow_tiles) &&
                    (mg + 1 == num_m_groups);
                const unsigned mg_next      = (mg + 1 < num_m_groups) ? mg + 1 : 0;
                const unsigned mt_base_next = mg_next * mt_per_group;
                const unsigned G_next       =
                    (mt_base_next + mt_per_group <= m_tiles)
                        ? mt_per_group : (m_tiles - mt_base_next);
                unsigned f_total = 0;                          // WeightVecs in the next slab
                for (unsigned t = 0; t < kMaxMperGroup; t++) {
                    #pragma HLS UNROLL
                    if (t < G_next)
                        f_total += std::min(kTileM, out_ch - (mt_base_next + t) * kTileM) * kh * kw;
                }
                if (last_slab) f_total = 0;
                // Prefetch write cursor: (tile, m1, khi, kwi) in emission order.
                unsigned   f_pos = 0;
                ap_uint<3> f_t = 0;                 // < kMaxMperGroup
                ap_uint<4> f_m1 = 0, f_khi = 0, f_kwi = 0;   // < kTileM / kMaxKH / kMaxKW
                unsigned f_mv = std::min(kTileM, out_ch - mt_base_next * kTileM);
                const ap_uint<1> nbank = wbank ^ 1;   // bank being prefetched

                // ---- Spatial sweep (§2.29): per (oh, ow_in_tile) ONE fused
                //      II=1 loop over (tile-in-group, khi, kwi) ----
                // Tile 0 consumes each PatchVec beat straight from the
                // stream and parks it in `patch`; tiles 1..G-1 replay it
                // from `patch`.  All G tiles' accumulators live in registers
                // for the whole pixel.  Each iteration also advances the
                // §2.35 prefetch of the next slab into the other bank.
                for (unsigned oh_local = 0; oh_local < chunk_oh_count;
                     oh_local++) {
                    for (unsigned ow = ow_start; ow < ow_end; ow++) {
                        Data_t patch[kTileIC][kMaxKH][kMaxKW];
                        #pragma HLS ARRAY_PARTITION variable=patch complete dim=1
                        #pragma HLS BIND_STORAGE variable=patch type=RAM_2P impl=lutram

                        AccData_t acc[kMaxMperGroup][kTileM];
                        #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

                        const unsigned pix_word = (oh_local * out_w + ow) * m_tiles;

                        // Load every tile's padded accumulator word (§2.23).
                        for (unsigned g = 0; g < mt_in_group_count; g++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned word = pix_word + mt_base + g;
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                acc[g][m1] = partial_outputs[word * kTileM + m1];
                            }
                        }

                        // Fused (g, khi, kwi) sweep + prefetch.
                        const unsigned n_steps = mt_in_group_count * kh * kw;
                        unsigned g = 0, khi = 0, kwi = 0;
                        for (unsigned ri = 0; ri < n_steps; ri++) {
                            #pragma HLS PIPELINE II=1
                            // The sweep reads bank wbank and the prefetch writes
                            // bank !wbank, so an iteration never reads what a
                            // previous iteration wrote — without this HLS assumes
                            // a RAW hazard through w_cache and schedules II=2
                            // (UG1399 "pragma HLS dependence", inter /
                            // dependent=false).
                            #pragma HLS DEPENDENCE variable=w_cache type=inter dependent=false
                            Data_t p[kTileIC];
                            #pragma HLS ARRAY_PARTITION variable=p complete dim=0
                            if (g == 0) {
                                const PatchVec v = patch_stream.read();
                                for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                                    #pragma HLS UNROLL
                                    p[ic_l]              = v.lane[ic_l];
                                    patch[ic_l][khi][kwi] = v.lane[ic_l];
                                }
                            } else {
                                for (unsigned ic_l = 0; ic_l < kTileIC; ic_l++) {
                                    #pragma HLS UNROLL
                                    p[ic_l] = patch[ic_l][khi][kwi];
                                }
                            }
                            const unsigned m_valid_g =
                                std::min(kTileM, out_ch - (mt_base + g) * kTileM);
                            mac_grid_step(p, w_cache, w_cache_addr(wbank, g, khi, kwi),
                                          acc[g], ic_valid, m_valid_g);

                            // §2.35 prefetch: one WeightVec of the NEXT slab
                            // into the other bank, if one is available.
                            if (f_pos < f_total) {
                                WeightVec wv;
                                if (weight_stream.read_nb(wv)) {
                                    const unsigned wa = w_cache_addr(nbank, f_t, f_khi, f_kwi);
                                    for (unsigned c = 0; c < kTileM; c++) {
                                        #pragma HLS UNROLL
                                        if (c == f_m1) w_cache[c][wa] = wv;
                                    }
                                    f_pos++;
                                    if (++f_kwi == kw) {
                                        f_kwi = 0;
                                        if (++f_khi == kh) {
                                            f_khi = 0;
                                            if (++f_m1 == f_mv) {
                                                f_m1 = 0;
                                                f_t++;
                                                f_mv = std::min(kTileM,
                                                    out_ch - (mt_base_next + (unsigned)f_t) * kTileM);
                                            }
                                        }
                                    }
                                }
                            }

                            if (++kwi == kw) {
                                kwi = 0;
                                if (++khi == kh) {
                                    khi = 0;
                                    g++;
                                }
                            }
                        }

                        // Store every tile's word back.
                        for (unsigned g2 = 0; g2 < mt_in_group_count; g2++) {
                            #pragma HLS PIPELINE II=1
                            const unsigned word = pix_word + mt_base + g2;
                            for (unsigned m1 = 0; m1 < kTileM; m1++) {
                                #pragma HLS UNROLL
                                partial_outputs[word * kTileM + m1] = acc[g2][m1];
                            }
                        }
                    }
                }

                // §2.35 tail: whatever of the next slab the sweep did not
                // absorb (short sweeps, slow producer) is read blocking here.
                for (; f_pos < f_total; f_pos++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS DEPENDENCE variable=w_cache type=inter dependent=false
                    const WeightVec wv = weight_stream.read();
                    const unsigned  wa = w_cache_addr(nbank, f_t, f_khi, f_kwi);
                    for (unsigned c = 0; c < kTileM; c++) {
                        #pragma HLS UNROLL
                        if (c == f_m1) w_cache[c][wa] = wv;
                    }
                    if (++f_kwi == kw) {
                        f_kwi = 0;
                        if (++f_khi == kh) {
                            f_khi = 0;
                            if (++f_m1 == f_mv) {
                                f_m1 = 0;
                                f_t++;
                                f_mv = std::min(kTileM,
                                    out_ch - (mt_base_next + (unsigned)f_t) * kTileM);
                            }
                        }
                    }
                }
                wbank        = nbank;           // the prefetched slab becomes current
                w_prefetched = !last_slab;
              } // mg
              } // ow_tile
            } // ict
        } else {
            // -------- Phase 2b: depthwise accumulate (mt OUTER, ow_tile) --------
            // §2.37 flat sweep: per (mt, ow_tile) ONE II=1 loop over
            // (oh_local, ow_in_tile, ri).  Each (pixel, mt) word is visited
            // exactly once in depthwise mode (no ic-tile reduction), so the
            // accumulator is seeded from the tile's bias register at ri == 0
            // and the full kTileM-lane word is stored write-only at
            // ri == kh*kw-1 — no partial_outputs load, no Phase 1, and the
            // pipeline ramp is paid once per (mt, ow_tile) instead of once
            // per pixel (the per-pixel load / kh*kw-loop / store used to cost
            // ~15 cycles for 9 MACs).
            const unsigned ow_per_tile_dw  = geom.ow_per_tile;
            const unsigned num_ow_tiles_dw = geom.num_ow_tiles;
            const unsigned n_pos           = kh * kw;
            const unsigned row_words       = out_w * m_tiles;

            for (unsigned mt = 0; mt < m_tiles; mt++) {
                const unsigned m_off   = mt * kTileM;
                const unsigned m_valid = std::min(kTileM, out_ch - m_off);

                // One BiasVec per (ni, chunk, mt): the tile's initial
                // accumulator word (padding lanes already zero, §2.25).
                const BiasVec bv = bias_stream.read();
                AccData_t bias_reg[kTileM];
                #pragma HLS ARRAY_PARTITION variable=bias_reg complete dim=0
                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                    #pragma HLS UNROLL
                    bias_reg[m1] = bv.lane[m1];
                }

                // Load weights ONCE per (chunk, mt) (held in BRAM across all
                // ow_tiles and the (oh, ow_in_tile) sweep).  ow-tiling here
                // doesn't add weight DDR replay — weight slice is small and
                // shared across the full ow sweep.
                // Flat over the kernel window (pos = khi*kw + kwi, §2.32).
                // dim 1 complete → kTileM lanes read per cycle by the grid;
                // dim 2 cyclic kTileIC → the kTileIC positions of one
                // WeightVec beat land in distinct banks and are written in
                // one cycle.  Lanes m1 >= m_valid (padding of the last tile)
                // are filled with zeros so no 'X' reaches the accumulator
                // word in RTL (its padding lanes are never drained, but
                // §2.38's Phase 3 reads whole words).
                Data_t w_buf[kTileM][kMaxKPos];
                #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
                #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=kTileIC dim=2
                const unsigned dw_words = conv_dw_stride(kh, kw) / kWeightPortElems;
                const unsigned dw_vecs  = (dw_words + kWordsPerWeightVec - 1) / kWordsPerWeightVec;
                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                    for (unsigned beat = 0; beat < dw_vecs; beat++) {
                        #pragma HLS PIPELINE II=1
                        WeightVec v;
                        if (m1 < m_valid) {
                            v = weight_stream.read();
                        } else {
                            for (unsigned j = 0; j < kTileIC; j++) {
                                #pragma HLS UNROLL
                                v.lane[j] = Data_t(0);
                            }
                        }
                        for (unsigned j = 0; j < kTileIC; j++) {
                            #pragma HLS UNROLL
                            const unsigned pos = beat * kTileIC + j;
                            if (pos < kMaxKPos) w_buf[m1][pos] = v.lane[j];
                        }
                    }
                }

              for (unsigned owt = 0; owt < num_ow_tiles_dw; owt++) {
                const unsigned ow_start = owt * ow_per_tile_dw;
                const unsigned ow_end   = std::min(out_w, ow_start + ow_per_tile_dw);
                const unsigned tw       = ow_end - ow_start;
                const unsigned n_iter   = chunk_oh_count * tw * n_pos;

                // Running partial_outputs word cursor (§2.23 layout:
                // word = (oh_local*out_w + ow)*m_tiles + mt) — incremented
                // per pixel / per row so the loop carries no multiply.
                unsigned word_row = ow_start * m_tiles + mt;
                unsigned word     = word_row;
                unsigned ri = 0, ow_l = 0;

                AccData_t acc[kTileM];
                #pragma HLS ARRAY_PARTITION variable=acc complete dim=0
                for (unsigned m1 = 0; m1 < kTileM; m1++) {
                    #pragma HLS UNROLL
                    acc[m1] = AccData_t(0);
                }

                for (unsigned it = 0; it < n_iter; it++) {
                    #pragma HLS PIPELINE II=1
                    // §2.29: depthwise has a single tile per pixel, so the
                    // PatchVec beats feed the lanes straight from the stream.
                    const PatchVec v = patch_stream.read();
                    const bool first = (ri == 0);
                    const bool last  = (ri + 1 == n_pos);
                    for (unsigned m1 = 0; m1 < kTileM; m1++) {
                        #pragma HLS UNROLL
                        acc[m1] = first ? bias_reg[m1] : acc[m1];
                    }
                    mac_dw_step(v.lane, w_buf, ri, acc);   // ri == khi*kw + kwi
                    if (last) {
                        for (unsigned m1 = 0; m1 < kTileM; m1++) {
                            #pragma HLS UNROLL
                            partial_outputs[word * kTileM + m1] = acc[m1];
                        }
                        ri = 0;
                        if (++ow_l == tw) {
                            ow_l      = 0;
                            word_row += row_words;
                            word      = word_row;
                        } else {
                            word += m_tiles;
                        }
                    } else {
                        ri++;
                    }
                }
              } // ow_tile
            } // mt
        } // depthwise

        // -------- Phase 3: drain partial_outputs to acc_stream --------
        // saturate_cast AccData_t→Data_t here (hoisted out of
        // write_output_tile) so acc_stream is a Data_t-wide FIFO.
        // Order is CHANNEL-major — (mt, m1, oh_local, ow) — so that
        // write_output_tile sees one contiguous run of
        // chunk_oh_count·out_w elements per channel and can burst (§2.22).
        // The partial_outputs reads become out_ch-strided, which is free
        // on the on-chip URAM.
        for (unsigned mt = 0; mt < m_tiles; mt++) {
            const unsigned m_off   = mt * kTileM;
            const unsigned m_valid = std::min(kTileM, out_ch - m_off);
            for (unsigned m1 = 0; m1 < m_valid; m1++) {
                for (unsigned oh_local = 0; oh_local < chunk_oh_count; oh_local++) {
                    for (unsigned ow = 0; ow < out_w; ow++) {
                        #pragma HLS PIPELINE II=1
                        const unsigned idx =
                            ((oh_local * out_w + ow) * m_tiles + mt) * kTileM + m1;
                        acc_stream.write(
                            saturate_cast<Data_t>(partial_outputs[idx]));
                    }
                }
            }
        }
      } // chunk
    } // ni
}

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
) {
    // -----------------------------------------------------------------------
    // HLS AXI interface pragmas.
    //
    // Four m_axi ports allow the tool to issue input, weight, bias, and output
    // transactions on separate AXI buses.  All scalar arguments go into the
    // s_axilite ctrl register file accessed by the PS driver.
    //
    // Per-port burst and outstanding-transaction tuning:
    //   max_{read,write}_burst_length=256  — maximum AXI burst beats (AMBA
    //                                        allows 256), so a fully widened
    //                                        bus moves up to one 4 KB page per
    //                                        burst, amortising address-channel
    //                                        latency.
    //   num_{read,write}_outstanding=8     — the m_axi adapter can have up to
    //                                        8 in-flight bursts before
    //                                        stalling, hiding DDR round-trip
    //                                        latency under DATAFLOW.
    //   m_axi_max_widen_bitwidth is set globally in scripts/Synthesis.tcl.in
    //   via AXI_BUS_WIDTH (lets users dial it back to match a 128-bit block
    //   design); not duplicated per-port so the global stays authoritative.
    //
    //   depth=<N> is a C/RTL co-simulation hint only — it sizes the
    //   verification adapter FIFO cosim builds for each m_axi port.  It does
    //   NOT constrain the synthesised AXI master (runtime addresses) or the
    //   exported IP.  The CONV_COSIM_DEPTH_* macros (ConvKernel.h) are the
    //   single source of truth shared with the test/TestConvSim.cpp cosim
    //   buffers.  (UG1399 documents an argument-expression form,
    //   depth=batch*in_ch*in_h*in_w, but it triggered an HLS 200-1715
    //   source-synthesis error here, so constants are used.)  cosim of an
    //   m_axi kernel aborts without depth ("a depth specification is
    //   required for interface port 'x'").
    // -----------------------------------------------------------------------
    // §2.27: the burst-length / outstanding options described above were
    // documented but never actually on the pragmas (csynth showed the
    // defaults).  Same settings as VectorOP.cpp; y is an hls::burst_maxi
    // port driven by explicit write_request/write/write_response (see
    // write_output_tile).
    #pragma HLS INTERFACE m_axi port=x       offset=slave bundle=gmem0 depth=CONV_COSIM_DEPTH_X      max_read_burst_length=256  num_read_outstanding=16
    // weight / bias are 128-bit ports (§2.32).  Their adapter buffers scale
    // with burst_length × outstanding × 16 B, so they are sized to what the
    // producers actually issue: a weight slab request is <= 2*kMaxKH*kMaxKW
    // = 98 beats with at most kTileM = 8 in flight; the bias is one request
    // of <= kMaxOutCh/8 = 160 beats.
    #pragma HLS INTERFACE m_axi port=weight  offset=slave bundle=gmem1 depth=CONV_COSIM_DEPTH_WEIGHT_WORDS max_read_burst_length=128 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=gmem2 depth=CONV_COSIM_DEPTH_BIAS_WORDS   max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=y       offset=slave bundle=gmem3 depth=CONV_COSIM_DEPTH_Y      max_write_burst_length=256 num_write_outstanding=16

    #pragma HLS INTERFACE s_axilite port=x            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=weight       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=bias         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=y            bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=batch        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_ch        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w         bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_ch       bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_h        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_w        bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kh           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=kw           bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_h     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride_w     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_h   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=dilation_w   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_top      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad_left     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=has_bias     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=is_depthwise bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return       bundle=ctrl

    // §2.20: every scalar argument (and the read-only input pointers) is
    // invariant for the whole kernel invocation, so mark them STABLE.  HLS
    // then forwards each as a stable signal shared across the DATAFLOW
    // processes instead of synchronising it through a per-consumer depth-2
    // FIFO — the pre-§2.20 build spent ~60 scalar channel FIFOs (~5.9k FF /
    // ~4.1k LUT) purely on argument plumbing (`out_ch` alone was replicated
    // into 5 FIFOs).  `y` is the WRITE port and is deliberately excluded.
    #pragma HLS STABLE variable=x
    #pragma HLS STABLE variable=weight
    #pragma HLS STABLE variable=bias
    #pragma HLS STABLE variable=batch
    #pragma HLS STABLE variable=in_ch
    #pragma HLS STABLE variable=in_h
    #pragma HLS STABLE variable=in_w
    #pragma HLS STABLE variable=out_ch
    #pragma HLS STABLE variable=out_h
    #pragma HLS STABLE variable=out_w
    #pragma HLS STABLE variable=kh
    #pragma HLS STABLE variable=kw
    #pragma HLS STABLE variable=stride_h
    #pragma HLS STABLE variable=stride_w
    #pragma HLS STABLE variable=dilation_h
    #pragma HLS STABLE variable=dilation_w
    #pragma HLS STABLE variable=pad_top
    #pragma HLS STABLE variable=pad_left
    #pragma HLS STABLE variable=has_bias
    #pragma HLS STABLE variable=is_depthwise

    static_assert(kTileM <= kTileIC,
                  "depthwise mode reuses patch[kTileIC] for TILE_M lanes: "
                  "kTileM must be <= kTileIC");

    // -----------------------------------------------------------------------
    // Top-level DATAFLOW region.
    //
    // bias_producer streams initial-accumulator values to
    // process_conv_kernel_tile through bias_stream; both processes run
    // concurrently.  The producer's bias DDR fetch for tile mt+1 overlaps
    // the tail of the consumer's compute for tile mt, hiding the (small)
    // bias load latency entirely after the first tile.
    //
    // bias_stream depth = kTileM is enough to hold one full m_valid push
    // batch, so the producer can stage the next inner iteration's bias
    // while the consumer is still in the previous iteration's compute.
    // -----------------------------------------------------------------------
    #pragma HLS DATAFLOW
    const unsigned ic_tiles         = (in_ch  + kTileIC - 1) / kTileIC;
    const unsigned m_tiles          = (out_ch + kTileM  - 1) / kTileM;

    // Tile geometry — computed ONCE here so the runtime-divisor divisions
    // (oh-chunking / M-grouping / ow-tiling) synthesise a single shared
    // divider set instead of one per dataflow stage (§2.19).
    const ConvGeometry geom = compute_conv_geometry(
        out_h, out_w, out_ch, kh, kw, stride_h, stride_w,
        dilation_h, dilation_w, is_depthwise);

    // Bias replay: standard Phase 1 wants one BiasVec per (pixel, mt);
    // the depthwise flat sweep (§2.37) one per (ni, chunk, mt).
    const unsigned bias_rep_count   = is_depthwise
                                    ? batch * geom.num_chunks
                                    : batch * out_h * out_w;

    // bias_stream carries one BiasVec (a whole m-tile of initial
    // accumulators) per beat (§2.25); depth covers one pixel's m-tiles.
    hls_thread_local hls::stream<BiasVec> bias_stream;
    #pragma HLS STREAM variable=bias_stream depth=kTileM

    // patch_stream carries the producer's channel-packed PatchVec
    // emissions straight to the consumer (no intermediate stage since
    // §2.15).  Each beat is one kTileIC-lane column; depth is one
    // kernel window's worth of beats (kMaxKH*kMaxKW) so the consumer
    // drains it as the assembler fills it under DATAFLOW.
    hls_thread_local hls::stream<PatchVec> patch_stream;
    #pragma HLS STREAM variable=patch_stream depth=kMaxKH*kMaxKW

    // acc_stream carries already-saturated Data_t — process_conv_kernel_tile
    // applies saturate_cast in its Phase-3 drain, so this inter-stage FIFO
    // is Data_t-wide (not AccData_t-wide) and write_output_tile is a plain
    // stream→DDR copy.
    //
    // §2.26 write overlap: the FIFO is one full chunk deep and lives in the
    // otherwise idle URAM (UG1399 bind_storage: type=fifo, impl=uram), so
    // Phase 3 drains a finished chunk at 1 element/cycle without ever
    // blocking on the writer, and the consumer moves on to the next
    // chunk's Phase 1/2 while write_output_tile is still bursting the
    // previous chunk to DDR.  Measured after §2.22 the writer costs ~2.3
    // cycles/element; with the 8-deep FIFO that time was fully serial
    // with compute.  Single-chunk layers gain nothing here (there is no
    // next chunk to overlap with) — see the min-chunks note in
    // CONV_OPTIMISATION.md §2.26.
    hls_thread_local hls::stream<Data_t> acc_stream;
    #pragma HLS STREAM variable=acc_stream depth=kMaxAccPersistEntries
    #pragma HLS BIND_STORAGE variable=acc_stream type=fifo impl=uram

    // weight_stream carries one Data_t per cycle from stream_load_weights
    // to process_conv_kernel_tile.  Depth is one full max-tile (kTileM *
    // kTileIC * kMaxKH * kMaxKW = 6272 at defaults) so the producer can
    // pre-fetch the next iteration's weight slice while the consumer is
    // still in accumulate — full producer/consumer overlap.
    // §2.32: WeightVec beats (kTileIC lanes each); depth = one full m-tile's
    // worth of vectors — the same bytes the former element stream held.
    hls_thread_local hls::stream<WeightVec> weight_stream;
    #pragma HLS STREAM variable=weight_stream depth=kTileM*kMaxKH*kMaxKW

    bias_producer(bias, bias_stream,
                  out_ch, bias_rep_count, has_bias);

    input_patch_producer(x, patch_stream, batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w, kh, kw, stride_h, stride_w, dilation_h,
        dilation_w, pad_top, pad_left, is_depthwise, geom
    );

    stream_load_weights(weight, weight_stream,
                        ic_tiles, m_tiles, in_ch, out_ch, out_w, out_h,
                        kw, kh, stride_w, dilation_w, batch, is_depthwise,
                        geom);

    process_conv_kernel_tile(
        patch_stream, weight_stream, bias_stream, acc_stream,
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        kh, kw, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, is_depthwise, geom);

    write_output_tile(y, acc_stream, out_ch, out_h, out_w, batch, geom);

#ifdef DEBUG_LOAD_DATA_CACHING
    // C-sim token accounting.  The four DATAFLOW stages run sequentially in
    // C-sim, so any producer/consumer beat-count mismatch shows up here as
    // leftover tokens (or earlier as a read-on-empty inside the consumer)
    // instead of as a DATAFLOW hang in RTL.
    if (!patch_stream.empty() || !weight_stream.empty() ||
        !bias_stream.empty()  || !acc_stream.empty()) {
        std::cerr << "ConvKernel: leftover stream tokens after run — patch="
                  << patch_stream.size() << " weight=" << weight_stream.size()
                  << " bias=" << bias_stream.size() << " acc=" << acc_stream.size()
                  << std::endl;
        assert(!"stream token accounting");
    }
#endif
}
