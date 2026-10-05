// ---------------------------------------------------------------------------
// pl_pkg — constants, descriptors and helpers of the RTL PoolingKernel.
//
// The bounds are the KV260 platform's kernels.pool block (platforms/kv260.json:
// tile_c 8, max_kh / max_kw 7, max_line_buf_rows 16, max_line_buf_cols 64,
// ow_parallel 2), which the HLS kernel was built with; the scheduler checks
// models against the same file.
// ---------------------------------------------------------------------------
package pl_pkg;

  localparam int E    = 8;             // Q8.8 lanes per 128-bit word
  localparam int EW   = 16;
  localparam int BW   = E * EW;
  localparam int TC   = 8;             // channels per tile
  localparam int OWP  = 2;             // output positions per window beat
  localparam int MAXK = 7;             // pool_h, pool_w <= MAXK
  localparam int LBR  = 16;            // line-buffer rows (power of two)
  localparam int LBC  = 64;            // line-buffer columns
  localparam int LBW  = LBC / E;       // words per line-buffer row

  // AXI engines.  The burst lengths and outstanding counts are declared on the
  // IP's m_axi interfaces (syn/package_ip.tcl, the HLS export's values); the
  // block design sizes each crossbar slot's acceptance from them, so change
  // both together (fact rtl.axi_masters).
  localparam int RD_BURST  = 16;       // max beats per AR burst (a run is <= 9)
  localparam int RD_OUTS   = 16;       // AR bursts awaiting their data
  localparam int RD_FIFO_D = 256;      // beats buffered for the line buffer
  localparam int WR_BURST  = 16;       // max beats per AW burst (a run is <= 9)
  localparam int WR_OUTS   = 8;        // AW bursts awaiting their B

  localparam logic [15:0] DATA_MIN = 16'h8000;   // -128.0: MAX's identity

  // Pool modes (pool_type / lp_order decoded once per job).
  typedef enum logic [1:0] {M_MAX = 2'd0, M_AVG = 2'd1, M_LP1 = 2'd2, M_LP2 = 2'd3} mode_t;

  // Per-job constants, fixed while the job runs (pl_core's configuration).
  typedef struct packed {
    logic [59:0] x_w, y_w;             // word addresses of x and y
    logic [31:0] in_h, in_w, out_h, out_w, in_hw, out_hw;
    logic [31:0] stride_h, stride_w, dil_h, dil_w, pad_top;
    logic [31:0] reach_h;              // (pool_h - 1) * dil_h
    logic [31:0] rows;                 // input rows read per chunk
    logic [2:0]  pool_h, pool_w;       // 1 .. 7
    logic [5:0]  red_len;              // pool_h * pool_w
    logic [5:0]  denom_all;            // the same, for count_include_pad
    logic        gw2;                  // two output positions per group
    logic        prefetch;             // load row oh+1 while emitting row oh
    logic        cip;                  // count_include_pad
    mode_t       mode;
  } job_t;

  // One (batch, channel tile, W-tile) chunk.
  typedef struct packed {
    logic [3:0]  c_valid;              // 1 .. 8 channels
    logic [6:0]  ow_span;              // 1 .. 64 output columns
    logic [6:0]  n_groups;             // window groups per output row
    logic [6:0]  run_len;              // input columns read per row (<= 64)
    logic [31:0] col_base;             // local input column of (ow_lo, tap 0), signed
    logic [31:0] lo_lim, hi_lim;       // in-bounds local columns: [lo_lim, hi_lim), signed
    logic [31:0] base_in;              // element of (plane, row 0, iw_lo)
    logic [31:0] base_out;             // element of (plane, out row 0, ow_lo)
  } chunk_t;

  // Line-buffer run descriptor (loader -> emitter), one per (row, channel).
  typedef struct packed {
    logic [3:0] slot;                  // input row mod LBR
    logic [2:0] ch;
    logic [2:0] shift;                 // first element's lane in its word
    logic [3:0] nw;                    // words of the run
    logic       row_last;              // the row's last channel
  } lbd_t;

  // Window beat (emitter -> reducer): OWP positions x TC channels.
  typedef struct packed {
    logic [OWP*TC*EW-1:0] v;           // lane (p, c) at [(p*TC + c)*EW +: EW]
    logic                 first, last; // the group's first / last tap
    logic [5:0]           d1, d0;      // AVG denominators (valid with last)
  } beat_t;

  // Floor a Q16.16 accumulator to Q8.8 and saturate (saturate_cast<Data_t>).
  function automatic logic [15:0] sat16(input logic [31:0] r);
    logic [23:0] s;
    s = 24'(signed'(r) >>> 8);
    if (!s[23] && (s[22:15] != '0)) return 16'h7FFF;
    if ( s[23] && (s[22:15] != '1)) return 16'h8000;
    return s[15:0];
  endfunction

  // AVG reciprocal: round(2^23 / d) as ap_ufixed<24,1> raw bits (0 for d = 0).
  function automatic logic [23:0] inv_of(input logic [5:0] d);
    int unsigned q;
    q = (d == 0) ? 0 : (((1 << 23) + int'(d) / 2) / int'(d));
    return 24'(q);
  endfunction

endpackage
