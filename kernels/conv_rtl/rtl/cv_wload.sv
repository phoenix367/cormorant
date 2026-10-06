// ---------------------------------------------------------------------------
// cv_wload — the weight reader (gmem1, the HLS stream_load_weights).
//
// Per sweep, one run per output channel of the slab (the m-group's channels,
// in order): standard, the channel's n_pos kernel positions of the input
// tile (two 8-lane beats per position, one for a half tile); depthwise, the
// channel's dw_words beats of 8 positions.  Each run is one burst (two where
// it crosses 4 KiB); at most W_OUTS bursts await their data and a run is only
// requested when the vector FIFO has room for all of it, so RREADY stays
// high.  Out come TIC-lane vectors (standard) or 8-position beats in the low
// half (depthwise).
// ---------------------------------------------------------------------------
module cv_wload
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          wq_valid,
  output logic          wq_ready,
  input  sweep_t        wq,

  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          wv_valid,
  input  logic          wv_ready,
  output logic [TIC*EW-1:0] wv_data,

  output logic          idle
);
  localparam int FD = 512;                 // vector FIFO depth
  localparam int FW = $clog2(FD) + 1;
  localparam int OW = $clog2(W_OUTS) + 1;

  // a standard run (two beats per kernel position) is one burst unless it crosses 4 KiB
  if (W_BURST < 2 * MAXK * MAXK) begin : g_burst_check
    $error("cv_wload: W_BURST must hold a run of 2 * MAXK * MAXK words");
  end

  // ---- runs -------------------------------------------------------------------------
  logic        s_act;
  logic [10:0] s_nruns, s_r;               // runs of the sweep (<= MPG * TM), run index
  logic [59:0] s_waddr;                    // word address of run s_r
  logic [12:0] s_stride;                   // words between consecutive channels
  logic [7:0]  s_len;                      // beats per run (<= 2 * MAXK * MAXK)
  logic [6:0]  s_ent;                      // FIFO entries per run
  logic        s_pair;                     // two beats per entry
  logic        s_second;                   // the run's second burst (4 KiB split) is next
  logic [FW-1:0] credit;                   // FIFO entries not reserved
  logic [OW-1:0] outs;

  assign wq_ready = !s_act;

  logic [31:0] nrem;
  assign nrem = j.out_ch - 32'({wq.mt_base, 4'b0});

  logic [8:0] to4k;
  logic [7:0] b_len;
  logic       b_last;
  always_comb begin
    to4k = 9'd256 - {1'b0, s_waddr[7:0]};
    if (!s_second) begin
      b_len  = (9'(s_len) > to4k) ? to4k[7:0] : s_len;
      b_last = (9'(s_len) <= to4k);
    end else begin
      b_len  = s_len - to4k[7:0];
      b_last = 1'b1;
    end
  end

  logic rq_in_ready, issue;
  assign issue = s_act && (!arvalid || arready) && (outs != OW'(W_OUTS)) &&
                 (s_second || (rq_in_ready && credit >= FW'(s_ent)));

  logic fifo_pop;
  always_ff @(posedge clk) begin
    if (rst) begin
      s_act    <= 1'b0;
      arvalid  <= 1'b0;
      outs     <= '0;
      credit   <= FW'(FD);
      s_second <= 1'b0;
    end else begin
      if (arvalid && arready) arvalid <= 1'b0;
      if (!s_act && wq_valid) begin
        s_act    <= 1'b1;
        s_r      <= '0;
        s_nruns  <= (nrem >= {25'b0, wq.g_n, 4'b0}) ? {4'b0, wq.g_n, 4'b0} : nrem[10:0];
        s_waddr  <= j.w_w + 60'(wq.w_off);
        s_stride <= j.dwm ? {9'b0, j.dw_words} : j.per_m_words;
        s_len    <= j.dwm ? {4'b0, j.dw_words}
                          : (wq.half ? {2'b0, j.n_pos} : {1'b0, j.n_pos, 1'b0});
        s_ent    <= j.dwm ? {3'b0, j.dw_words} : {1'b0, j.n_pos};
        s_pair   <= !j.dwm && !wq.half;
      end
      if (issue) begin
        arvalid  <= 1'b1;
        araddr   <= {(s_second ? s_waddr + 60'(to4k) : s_waddr), 4'b0};
        arlen    <= b_len - 8'd1;
        s_second <= !b_last;
        if (b_last) begin
          s_waddr <= s_waddr + 60'(s_stride);
          if (s_r + 11'd1 == s_nruns) s_act <= 1'b0;
          s_r <= s_r + 11'd1;
        end
      end
      credit <= credit - ((issue && !s_second) ? FW'(s_ent) : '0) + (fifo_pop ? FW'(1) : '0);
      outs   <= outs + (issue ? OW'(1) : '0) - ((rvalid && rlast) ? OW'(1) : '0);
    end
  end

  // run queue: how the arriving beats of each run form entries
  typedef struct packed {
    logic       pair;
    logic [7:0] len;
  } run_t;
  run_t run_in, run_out;
  logic run_valid, run_pop;
  assign run_in = '{pair: s_pair, len: s_len};
  logic [4:0] run_n;
  cv_fifo #(.W($bits(run_t)), .D(16), .BRAM(1'b0)) u_run (
    .clk, .rst, .in_valid (issue && !s_second), .in_ready (rq_in_ready),
    .in_data (run_in),
    .out_valid (run_valid), .out_ready (run_pop), .out_data (run_out), .count (run_n)
  );

  // ---- data: beats to entries ----------------------------------------------------------
  logic [7:0]    d_b;                      // beat within the run
  logic [BW-1:0] d_lo;                     // first beat of a pair
  logic          push;
  logic [TIC*EW-1:0] push_data;
  assign rready  = 1'b1;
  assign run_pop = rvalid && run_valid && (d_b + 8'd1 == run_out.len);
  assign push    = rvalid && (!run_out.pair || d_b[0]);
  assign push_data = run_out.pair ? {rdata, d_lo} : {{BW{1'b0}}, rdata};

  always_ff @(posedge clk) begin
    if (rst) d_b <= '0;
    else if (rvalid) d_b <= run_pop ? 8'd0 : d_b + 8'd1;
    if (rvalid) d_lo <= rdata;
  end

  // the entry through a register into the FIFO (the run queue's pair bit
  // selects 256 bits): the credits reserved its room when the run was requested
  logic          push_q;
  logic [TIC*EW-1:0] push_data_q;
  always_ff @(posedge clk) begin
    push_q      <= push && !rst;
    push_data_q <= push_data;
  end

  logic fifo_in_ready;
  logic [FW-1:0] fifo_n;
  cv_fifo #(.W(TIC * EW), .D(FD), .BRAM(1'b1)) u_vf (
    .clk, .rst, .in_valid (push_q), .in_ready (fifo_in_ready), .in_data (push_data_q),
    .out_valid (wv_valid), .out_ready (wv_ready), .out_data (wv_data), .count (fifo_n)
  );
  assign fifo_pop = wv_valid && wv_ready;

  assign idle = !s_act && !arvalid && (outs == '0) && !push_q && !wv_valid;

  logic unused;
  assign unused = fifo_in_ready ^ ^fifo_n ^ ^run_n ^ ^j ^ ^wq;

endmodule
