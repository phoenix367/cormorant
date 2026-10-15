// ---------------------------------------------------------------------------
// ro_rd_port — one operand (a on gmem0, b on gmem1): DDR runs -> word stream.
//
//   burstgen ─► AR ─► (R) block-RAM FIFO ─► tail mask ─┬────────────► out FIFO ─► out
//                                                      └► replay RAM ─┘
//
// A burst's AR is only issued when the FIFO can take all of its beats, so
// RREADY stays high and the port never back-pressures the interconnect; at
// most RD_OUTS bursts are awaiting their data.
// The lanes of a run's last word past `size` are zeroed (g.tail valid lanes).
// Runs outside the geometry's window (g.win_lo / win_hi) are not read: the
// burst generator flags their bursts, which take no AR and no FIFO space and
// are delivered as zero words in order (a descriptor queue keeps the issue
// order of real and zero bursts).
// Replay (a stride-0 operand of <= REP_D words): the single run is passed on
// and stored as it arrives, then streamed g.reps - 1 more times from the RAM.
// External runs (ext_en, FOLD): the geometry is not walked; each run offered
// on ext_* (a word address and a count >= 1) restarts the burst generator
// once the previous run's bursts are issued — the bursts, the credits and the
// output stage as for the geometry's runs (g sets the output stage: tail 8, no
// replay).
// The AXI side is registered both ways: AR leaves through a register slice
// (ARREADY only enables it) and R is registered before the FIFO write.
// ---------------------------------------------------------------------------
module ro_rd_port
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  geom_t         g,
  input  logic          ext_en,           // runs from ext_* (stable for the job)
  input  logic          ext_valid,
  output logic          ext_ready,
  input  logic [59:0]   ext_waddr,
  input  logic [31:0]   ext_words,

  output logic          arvalid,
  input  logic          arready,
  output logic [63:0]   araddr,
  output logic [7:0]    arlen,
  input  logic          rvalid,
  output logic          rready,
  input  logic [BW-1:0] rdata,
  input  logic          rlast,

  output logic          out_valid,
  input  logic          out_ready,
  output logic [BW-1:0] out_data,

  output logic          idle
);
  localparam int SW = $clog2(RD_FIFO_D) + 1;
  localparam int OW = $clog2(OUT_D);
  localparam int PW = $clog2(REP_D);

  // Bursts and AR -------------------------------------------------------------------
  logic        bg_valid, bg_ready, bg_done, bg_zero;
  logic [59:0] bg_waddr;
  logic [8:0]  bg_len;

  // External runs: the run taken in one cycle starts the generator in the next.
  geom_t gx, g_bg;
  logic  kick;
  assign ext_ready = ext_en && bg_done && !kick;
  assign g_bg      = ext_en ? gx : g;
  always_ff @(posedge clk) begin
    if (rst || start) kick <= 1'b0;
    else              kick <= ext_valid && ext_ready;
    if (ext_valid && ext_ready) begin
      gx           <= '0;
      gx.en        <= 1'b1;
      gx.base_w    <= ext_waddr;
      gx.n_blk     <= 32'd1;
      gx.n_tiles   <= 32'd1;
      gx.n_runs    <= 32'd1;
      gx.run_words <= ext_words;
      gx.tail      <= 4'd8;
      gx.reps      <= 32'd1;
      gx.win_hi    <= 32'hFFFFFFFF;
    end
  end

  ro_burstgen #(.MAXB(RD_BURST)) u_bg (
    .clk, .rst, .start ((start && !ext_en) || kick), .g (g_bg),
    .b_valid (bg_valid), .b_ready (bg_ready), .b_waddr (bg_waddr), .b_len (bg_len), .b_zero (bg_zero),
    .done    (bg_done)
  );

  // Issued bursts in order: {zero, len}; the output stage walks them.
  logic       dq_in_ready, dq_valid, dq_zero, dq_pop;
  logic [8:0] dq_len, bcnt;
  logic [5:0] dq_count;
  ro_fifo #(.W(10), .D(32), .BRAM(1'b0)) u_dq (
    .clk, .rst,
    .in_valid (bg_valid && bg_ready), .in_ready (dq_in_ready), .in_data ({bg_zero, bg_len}),
    .out_valid (dq_valid), .out_ready (dq_pop), .out_data ({dq_zero, dq_len}),
    .count (dq_count)
  );

  localparam int NW = $clog2(RD_OUTS) + 1;
  logic [SW-1:0] space;                     // FIFO entries not reserved by a burst
  logic [NW-1:0] outs;                      // bursts issued, last beat not yet received
  logic          rf_pop, issue, ar_in_ready, ar_issue;
  assign issue    = bg_valid && dq_in_ready &&
                    (bg_zero || (ar_in_ready && (space >= SW'(bg_len)) && (outs != NW'(RD_OUTS))));
  assign ar_issue = issue && !bg_zero;
  assign bg_ready = issue;

  // R registered before the FIFO (a beat in rv_q is already reserved).
  logic          rv_q, rl_q;
  logic [BW-1:0] rd_q;
  always_ff @(posedge clk) begin
    if (rst) rv_q <= 1'b0;
    else     rv_q <= rvalid;
    rl_q <= rlast;
    rd_q <= rdata;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      space   <= SW'(RD_FIFO_D);
      outs    <= '0;
    end else begin
      space <= space - (ar_issue ? SW'(bg_len) : '0) + (rf_pop ? SW'(1) : '0);
      outs  <= outs + (ar_issue ? NW'(1) : '0) - ((rv_q && rl_q) ? NW'(1) : '0);
    end
  end

  ro_rs #(.W(72)) u_ar (
    .clk, .rst,
    .in_valid  (ar_issue), .in_ready (ar_in_ready), .in_data ({bg_waddr, 4'b0, 8'(bg_len - 9'd1)}),
    .out_valid (arvalid), .out_ready (arready),     .out_data ({araddr, arlen}),
    .head_data ()
  );

  assign rready = 1'b1;

  logic          rf_valid, rf_in_ready;
  logic [BW-1:0] rf_data;
  logic [SW-1:0] rf_count;
  ro_fifo #(.W(BW), .D(RD_FIFO_D), .BRAM(1'b1)) u_rf (
    .clk, .rst,
    .in_valid (rv_q),     .in_ready (rf_in_ready), .in_data (rd_q),
    .out_valid(rf_valid), .out_ready(rf_pop),      .out_data(rf_data),
    .count    (rf_count)
  );

  // Output stage ------------------------------------------------------------------------
  logic          of_in_valid, of_in_ready;
  logic [BW-1:0] of_in_data;
  logic [OW:0]   of_count;
  ro_fifo #(.W(BW), .D(OUT_D), .BRAM(1'b0)) u_of (
    .clk, .rst,
    .in_valid (of_in_valid), .in_ready (of_in_ready), .in_data (of_in_data),
    .out_valid, .out_ready, .out_data,
    .count    (of_count)
  );

  typedef enum logic [1:0] {P_PASS, P_REPLAY, P_DONE} phase_t;
  phase_t      phase;
  logic [31:0] wr_left;                     // words left in the current run
  logic [PW-1:0] rp_idx;                    // replay RAM index
  logic [31:0] reps_left;                   // replays still to stream
  logic        rp_rd, rp_v;                 // RAM read issued / data valid

  (* ram_style = "distributed" *) logic [BW-1:0] rep [REP_D];
  logic [BW-1:0] rp_q;

  // credits: the out FIFO must hold every word in flight
  logic room;
  assign room = ({1'b0, of_count} + (OW+2)'(rp_v)) < (OW+2)'(OUT_D);

  // The tail mask reads registers only: last_w (wr_left == 1) is kept as a
  // flag beside wr_left, and the lanes past g.tail as a per-lane mask set at
  // start, so the masked word is one LUT after the FIFO's output.
  logic          last_w;                    // the popped word ends its run
  logic          rw1;                       // g.run_words == 1
  logic [E-1:0]  tmask;                     // lanes zeroed in a run's last word
  logic [BW-1:0] masked;
  always_comb begin
    masked = rf_data;
    for (int l = 0; l < E; l++)
      if (last_w && tmask[l]) masked[EW*l +: EW] = '0;
  end

  logic zero_emit, word_out;
  assign rf_pop    = (phase == P_PASS) && dq_valid && !dq_zero && rf_valid && room;
  assign zero_emit = (phase == P_PASS) && dq_valid && dq_zero && room;
  assign word_out  = rf_pop || zero_emit;
  assign dq_pop    = word_out && (bcnt + 9'd1 == dq_len);
  assign rp_rd     = (phase == P_REPLAY) && room;

  assign of_in_valid = word_out || rp_v;
  assign of_in_data  = rp_v ? rp_q : zero_emit ? '0 : masked;

  always_ff @(posedge clk) begin
    if (rst || start) bcnt <= '0;
    else if (word_out) bcnt <= dq_pop ? 9'd0 : bcnt + 9'd1;
  end

  always_ff @(posedge clk) begin
    if (rp_rd) rp_q <= rep[rp_idx];
    if (rf_pop && g.replay) rep[rp_idx] <= masked;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      phase <= P_DONE;
      rp_v  <= 1'b0;
    end else if (start) begin
      phase     <= P_PASS;
      wr_left   <= g.run_words;
      last_w    <= (g.run_words == 32'd1);
      rw1       <= (g.run_words == 32'd1);
      for (int l = 0; l < E; l++) tmask[l] <= (l >= int'(g.tail));
      rp_idx    <= '0;
      reps_left <= g.reps - 32'd1;
      rp_v      <= 1'b0;
    end else begin
      rp_v <= rp_rd;
      if (word_out) begin
        wr_left <= last_w ? g.run_words : wr_left - 32'd1;
        last_w  <= last_w ? rw1 : (wr_left == 32'd2);
        if (g.replay && rf_pop) begin
          rp_idx <= last_w ? '0 : rp_idx + PW'(1);
          if (last_w) phase <= (reps_left != '0) ? P_REPLAY : P_DONE;
        end
      end
      if (rp_rd) begin
        if (32'(rp_idx) + 32'd1 == g.run_words) begin
          rp_idx    <= '0;
          reps_left <= reps_left - 32'd1;
          if (reps_left == 32'd1) phase <= P_DONE;
        end else begin
          rp_idx <= rp_idx + PW'(1);
        end
      end
    end
  end

  assign idle = bg_done && !kick && !arvalid && ar_in_ready && (space == SW'(RD_FIFO_D)) && (phase != P_REPLAY) &&
                !rp_v && (of_count == '0) && !dq_valid;

  // The FIFO never fills because every burst reserved its space.
  logic unused;
  assign unused = rf_in_ready ^ ^rf_count ^ of_in_ready ^ ^dq_count;

endmodule
