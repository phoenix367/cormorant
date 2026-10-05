// ---------------------------------------------------------------------------
// vo_rd_port — one operand (a on gmem0, b on gmem1): DDR runs -> word stream.
//
//   burstgen ─► AR ─► (R) block-RAM FIFO ─► tail mask ─┬────────────► out FIFO ─► out
//                                                      └► replay RAM ─┘
//
// A burst's AR is only issued when the FIFO can take all of its beats, so
// RREADY stays high and the port never back-pressures the interconnect; at
// most RD_OUTS bursts are awaiting their data.
// The lanes of a run's last word past `size` are zeroed (g.tail valid lanes).
// Replay (a stride-0 operand of <= REP_D words): the single run is passed on
// and stored as it arrives, then streamed g.reps - 1 more times from the RAM.
// ---------------------------------------------------------------------------
module vo_rd_port
  import vo_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  geom_t         g,

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
  logic        bg_valid, bg_ready, bg_done;
  logic [59:0] bg_waddr;
  logic [8:0]  bg_len;

  vo_burstgen #(.MAXB(RD_BURST)) u_bg (
    .clk, .rst, .start, .g,
    .b_valid (bg_valid), .b_ready (bg_ready), .b_waddr (bg_waddr), .b_len (bg_len),
    .done    (bg_done)
  );

  localparam int NW = $clog2(RD_OUTS) + 1;
  logic [SW-1:0] space;                     // FIFO entries not reserved by a burst
  logic [NW-1:0] outs;                      // bursts issued, last beat not yet received
  logic          rf_pop, issue;
  assign issue    = bg_valid && (!arvalid || arready) && (space >= SW'(bg_len)) &&
                    (outs != NW'(RD_OUTS));
  assign bg_ready = issue;

  always_ff @(posedge clk) begin
    if (rst) begin
      arvalid <= 1'b0;
      space   <= SW'(RD_FIFO_D);
      outs    <= '0;
    end else begin
      if (arvalid && arready) arvalid <= 1'b0;
      if (issue) begin
        arvalid <= 1'b1;
        araddr  <= {bg_waddr, 4'b0};
        arlen   <= 8'(bg_len - 9'd1);
      end
      space <= space - (issue ? SW'(bg_len) : '0) + (rf_pop ? SW'(1) : '0);
      outs  <= outs + (issue ? NW'(1) : '0) - ((rvalid && rlast) ? NW'(1) : '0);
    end
  end

  assign rready = 1'b1;

  logic          rf_valid, rf_in_ready;
  logic [BW-1:0] rf_data;
  logic [SW-1:0] rf_count;
  vo_fifo #(.W(BW), .D(RD_FIFO_D), .BRAM(1'b1)) u_rf (
    .clk, .rst,
    .in_valid (rvalid),   .in_ready (rf_in_ready), .in_data (rdata),
    .out_valid(rf_valid), .out_ready(rf_pop),      .out_data(rf_data),
    .count    (rf_count)
  );

  // Output stage ------------------------------------------------------------------------
  logic          of_in_valid, of_in_ready;
  logic [BW-1:0] of_in_data;
  logic [OW:0]   of_count;
  vo_fifo #(.W(BW), .D(OUT_D), .BRAM(1'b0)) u_of (
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

  (* ram_style = "block" *) logic [BW-1:0] rep [REP_D];
  logic [BW-1:0] rp_q;

  // credits: the out FIFO must hold every word in flight
  logic room;
  assign room = ({1'b0, of_count} + (OW+2)'(rp_v)) < (OW+2)'(OUT_D);

  logic          last_w;                    // the popped word ends its run
  logic [BW-1:0] masked;
  assign last_w = (wr_left == 32'd1);
  always_comb begin
    masked = rf_data;
    if (last_w)
      for (int l = 0; l < E; l++)
        if (l >= int'(g.tail)) masked[EW*l +: EW] = '0;
  end

  assign rf_pop = (phase == P_PASS) && rf_valid && room;
  assign rp_rd  = (phase == P_REPLAY) && room;

  assign of_in_valid = rf_pop || rp_v;
  assign of_in_data  = rp_v ? rp_q : masked;

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
      rp_idx    <= '0;
      reps_left <= g.reps - 32'd1;
      rp_v      <= 1'b0;
    end else begin
      rp_v <= rp_rd;
      if (rf_pop) begin
        wr_left <= last_w ? g.run_words : wr_left - 32'd1;
        if (g.replay) begin
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

  assign idle = bg_done && !arvalid && (space == SW'(RD_FIFO_D)) && (phase != P_REPLAY) &&
                !rp_v && (of_count == '0);

  // The FIFO never fills because every burst reserved its space.
  logic unused;
  assign unused = rf_in_ready ^ ^rf_count ^ of_in_ready;

endmodule
