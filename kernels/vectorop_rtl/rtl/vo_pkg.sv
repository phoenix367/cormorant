// ---------------------------------------------------------------------------
// vo_pkg — shared parameters and types of the VectorOPKernel RTL.
// ---------------------------------------------------------------------------
package vo_pkg;
  localparam int E         = 8;         // Q8.8 lanes per 128-bit AXI word
  localparam int EW        = 16;        // element width
  localparam int BW        = E * EW;    // word width (128)

  // Burst lengths and outstanding counts are the HLS interface's; the IP
  // declares them on its m_axi interfaces (syn/package_ip.tcl), and the
  // block design's crossbar sizes its acceptance from them.
  localparam int RD_BURST  = 64;        // max beats per AR burst (HLS max_read_burst_length)
  localparam int RD_OUTS   = 16;        // AR bursts awaiting their data (num_read_outstanding)
  localparam int RD_FIFO_D = 512;       // beats buffered per read port
  localparam int WR_BURST  = 256;       // max beats per AW burst (HLS max_write_burst_length)
  localparam int WR_FIFO_D = 512;       // beats buffered before an AW is issued
  localparam int WR_OUTS   = 16;        // AW bursts awaiting W / B (num_write_outstanding)
  localparam int REP_D     = 256;       // stride-0 replay buffer, words (2048 elements)
  localparam int OUT_D     = 8;         // read port output FIFO
  localparam int CF_D      = 16;        // compute output FIFO

  // Op / act codes (VectorOP.h).
  localparam logic [31:0] OP_ADD = 32'd0, OP_SUB = 32'd1, OP_MUL = 32'd2, OP_DIV = 32'd3,
                          OP_RELU = 32'd4, OP_RELU6 = 32'd5;
  localparam logic [31:0] ACT_RELU = 32'd1, ACT_RELU6 = 32'd2;
  localparam logic [15:0] SIX = 16'h0600;   // 6.0 in Q8.8

  // How one operand (or the output) walks DDR during a job: n_runs runs of
  // run_words words, run r at word base + r * stride_w; the last word of every
  // run has `tail` valid lanes (1..8).  replay: one run, streamed `reps` times.
  typedef struct packed {
    logic        en;
    logic [59:0] base_w;      // word address of the first run
    logic [31:0] n_runs;
    logic [31:0] run_words;
    logic [31:0] stride_w;
    logic [3:0]  tail;
    logic        replay;
    logic [31:0] reps;
  } geom_t;

  // Saturate a 17-bit signed sum / difference to int16.
  function automatic logic [15:0] sat17(input logic [16:0] x);
    if (x[16] != x[15]) return x[16] ? 16'h8000 : 16'h7FFF;
    return x[15:0];
  endfunction

  function automatic logic [15:0] relu(input logic [15:0] x);
    return x[15] ? 16'h0 : x;
  endfunction

  function automatic logic [15:0] relu6(input logic [15:0] x);
    if (x[15]) return 16'h0;
    return (x > SIX) ? SIX : x;
  endfunction

  // The fused activation (act register) on an op result.
  function automatic logic [15:0] activate(input logic [15:0] x, input logic [1:0] act);
    case (act)
      2'd1:    return relu(x);
      2'd2:    return relu6(x);
      default: return x;
    endcase
  endfunction
endpackage
