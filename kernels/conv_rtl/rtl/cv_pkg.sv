// ---------------------------------------------------------------------------
// cv_pkg — constants and types of the RTL ConvKernel.
//
// The tile constants are the platform's kernels.conv bounds (platforms/kv260.json;
// CMakeLists.txt checks them at configure time).  The AXI limits are the HLS
// export's m_axi bus parameters, which syn/package_ip.tcl declares on the IP.
// ---------------------------------------------------------------------------
package cv_pkg;

  localparam int E    = 8;             // Q8.8 lanes per 128-bit word
  localparam int EW   = 16;
  localparam int BW   = E * EW;

  localparam int TM    = 16;           // tile_m: output channels per tile (grid columns)
  localparam int TIC   = 16;           // tile_ic: input channels per tile (grid rows)
  localparam int MAXK  = 7;            // max_kh = max_kw
  localparam int LBR   = 16;           // max_line_buf_rows (power of two)
  localparam int LBC   = 64;           // max_line_buf_cols (power of two)
  localparam int ACCN  = 65536;        // max_acc_persist_entries
  localparam int MPG   = 4;            // max_m_per_group
  localparam int MAXIC = 1024;         // max_in_ch
  localparam int MAXOC = 1280;         // max_out_ch
  localparam int AWORDS = ACCN / TM;   // accumulator words (TM lanes of 32 bits)
  localparam int MAXMT  = MAXOC / TM;  // m-tiles (80)

  // AXI masters: burst beats and outstanding bursts (the HLS bus parameters)
  localparam int X_BURST = 16,  X_OUTS = 16;
  localparam int W_BURST = 128, W_OUTS = 8;
  localparam int B_BURST = 256, B_OUTS = 2;
  localparam int Y_BURST = 64,  Y_OUTS = 8;

  // weight cache: per column, 2 banks x MPG tiles x WC_POS positions
  localparam int WC_ROW   = 8;         // >= MAXK
  localparam int WC_POS   = 64;        // >= MAXK * WC_ROW
  localparam int WC_WORDS = 2 * MPG * WC_POS;

  localparam int DSEG = 256;           // drain segment (pixels per channel run)

  // Job constants (stable from the first sweep on).
  typedef struct packed {
    logic [59:0] x_w, w_w, b_w, y_w;   // 16-byte word addresses
    logic [31:0] batch, in_ch, in_h, in_w, out_ch, out_h, out_w;
    logic [31:0] sh, sw, dh, dw, pt, pl;
    logic [2:0]  kh, kw;               // 1..MAXK
    logic        has_bias, dwm;        // depthwise
    logic [31:0] in_hw, out_hw;        // in_h * in_w, out_h * out_w
    logic [31:0] ch_out;               // out_ch * out_hw (one image of y)
    logic [6:0]  m_tiles;              // ceil(out_ch / TM)
    logic [6:0]  ic_tiles;             // ceil(in_ch / TIC)
    logic [5:0]  n_pos;                // kh * kw
    logic [5:0]  n_win;                // max(n_pos, 2)
    logic [3:0]  reach_h;              // (kh - 1) * dh  (< LBR)
    logic [12:0] row_words;            // out_w * m_tiles accumulator words per output row
    logic [12:0] per_m_words;          // packed weight words per output channel (standard)
    logic [3:0]  dw_words;             // ceil(kh * kw / 8) words per channel (depthwise)
  } job_t;

  // One sweep: (ni, chunk, channel tile, ow-tile, m-group) — the work unit every
  // part of the kernel walks in the same order.
  typedef struct packed {
    logic        load;                 // first m-group of (chunk, tile, ow-tile): input rows are loaded
    logic        seed_bias;            // first input tile: seeds come from the bias
    logic        chunk_first;          // first sweep of (ni, chunk)
    logic        chunk_last;           // last sweep of (ni, chunk)
    logic        par;                  // accumulator buffer of the chunk
    logic        half;                 // standard: the input tile is a half tile (8 weight lanes)
    logic [31:0] x_cbase;              // element offset of channel tile ct of image ni in x
    logic [4:0]  ch_valid;             // channels of the tile (1..16)
    logic [12:0] coh;                  // output rows of the chunk
    logic [31:0] ih0;                  // oh_start * stride_h - pad_top (signed)
    logic [12:0] ow_start, ow_end;
    logic [31:0] iw_lo;                // first input column the tile loads (clipped)
    logic [6:0]  iw_cnt;               // input columns the tile loads (0..LBC)
    logic [12:0] pair_base;            // even first column of the tile's pairs
    logic [12:0] n_pairs;
    logic [31:0] iwb0;                 // pair_base * stride_w - pad_left (signed)
    logic [6:0]  mt_base;              // first m-tile of the group
    logic [2:0]  g_n;                  // m-tiles in the group (1..MPG)
    logic [12:0] wrow0;                // accumulator word of (row 0, pair_base, mt_base)
    logic [31:0] w_off;                // weight word offset of the slab's first run
    logic [31:0] y_cbase;              // element offset of (ni, oh_start) in y (+ m * out_hw)
    logic [12:0] run_len;              // pixels per channel in the chunk (coh * out_w)
  } sweep_t;

  // A finished chunk for the drain: its accumulator buffer and where it goes in y.
  typedef struct packed {
    logic        par;
    logic [31:0] y_cbase;              // element offset of (ni, oh_start) in y
    logic [12:0] run_len;              // pixels per channel
  } dreq_t;

  localparam int PIX2 = 2;             // output pixels per grid instant (a pixel pair)
  localparam int PATCH_W = PIX2 * TIC * EW;   // one patch beat: two pixels x TIC lanes

  // Q16.16 accumulator -> Q8.8: floor and saturate
  function automatic logic [15:0] sat16(input logic [31:0] a);
    logic signed [23:0] q;
    q = 24'($signed(a) >>> 8);
    if (q > 24'sd32767)       return 16'h7FFF;
    else if (q < -24'sd32768) return 16'h8000;
    else                      return q[15:0];
  endfunction

endpackage
