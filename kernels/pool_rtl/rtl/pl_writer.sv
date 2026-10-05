// ---------------------------------------------------------------------------
// pl_writer — the output writer (the HLS write_output_tile, gmem1).
//
// Fill: an output row's bundles (OWP adjacent columns of one channel, TC
// channels per group) go into one of two row buffers, E column banks of
// 2 x TC x LBW entries (LUTRAM); positions past the W-tile and channels past
// c_valid are dropped.  Drain: the other buffer's previous row, as one run per
// channel — word k of a run holds local columns k*E + l - shift — written with
// byte strobes for the run's own lanes only (other lanes zero, strobe off).
// A run is one AXI burst, split where it crosses 4 KiB; at most WR_OUTS
// bursts await their B.  The job is done when every B has arrived.
// ---------------------------------------------------------------------------
module pl_writer
  import pl_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          cq_valid,
  output logic          cq_ready,
  input  chunk_t        cq,

  input  logic          fb_valid,
  output logic          fb_ready,
  input  logic [OWP*EW-1:0] fb,

  output logic          awvalid,
  input  logic          awready,
  output logic [63:0]   awaddr,
  output logic [7:0]    awlen,
  output logic          wvalid,
  input  logic          wready,
  output logic [BW-1:0] wdata,
  output logic [E*2-1:0] wstrb,
  output logic          wlast,
  input  logic          bvalid,
  output logic          bready,

  output logic          idle
);
  // Fill ------------------------------------------------------------------------------------
  typedef struct packed {
    logic        buf_i;                    // row buffer
    logic [3:0]  c_valid;
    logic [6:0]  span;
    logic [31:0] e;                        // element of (channel 0, column 0)
  } row_t;

  logic        f_act;                      // a chunk is being filled
  chunk_t      fck;
  logic [31:0] f_oh, f_e;
  logic [6:0]  f_g, f_gpos;
  logic [2:0]  f_c;
  logic        f_buf;
  logic [1:0]  full;                       // row buffer holds a row not yet drained
  logic        drq_in_ready, drq_valid, drq_ready, f_take, f_row_end;
  row_t        drq;
  logic [1:0]  drq_count;

  assign cq_ready  = !f_act;
  assign f_take    = f_act && fb_valid && !full[f_buf] && drq_in_ready;
  assign fb_ready  = f_take;
  assign f_row_end = f_take && (f_c == 3'd7) && (f_g + 7'd1 == fck.n_groups);

  // row-buffer write ports: bank (column mod E), entry {buffer, channel, column / E}
  logic          rb_we [E];
  logic [6:0]    rb_wa [E];
  logic [EW-1:0] rb_wd [E];
  always_comb begin
    for (int b = 0; b < E; b++) begin
      rb_we[b] = 1'b0;
      rb_wa[b] = {f_buf, f_c, f_gpos[5:3]};
      rb_wd[b] = '0;
    end
    for (int p = 0; p < OWP; p++) begin
      logic [6:0] col;
      logic       ok;
      col = f_gpos + 7'(p);
      ok  = f_take && (p == 0 || j.gw2) && (col < fck.ow_span) && (4'(f_c) < fck.c_valid);
      if (ok) begin
        rb_we[col[2:0]] = 1'b1;
        rb_wd[col[2:0]] = fb[p * EW +: EW];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      f_act <= 1'b0;
      f_buf <= 1'b0;
    end else if (cq_valid && cq_ready) begin
      fck   <= cq;
      f_act <= (cq.n_groups != '0);
      f_oh  <= '0;
      f_e   <= cq.base_out;
      f_g   <= '0;
      f_gpos <= '0;
      f_c   <= '0;
    end else if (f_take) begin
      if (f_c == 3'd7) begin
        f_c <= '0;
        if (f_g + 7'd1 == fck.n_groups) begin           // the row is complete
          f_g    <= '0;
          f_gpos <= '0;
          f_buf  <= !f_buf;
          f_oh   <= f_oh + 32'd1;
          f_e    <= f_e + j.out_w;
          if (f_oh + 32'd1 == j.out_h) f_act <= 1'b0;
        end else begin
          f_g    <= f_g + 7'd1;
          f_gpos <= f_gpos + (j.gw2 ? 7'd2 : 7'd1);
        end
      end else begin
        f_c <= f_c + 3'd1;
      end
    end
  end

  row_t drq_in;
  assign drq_in = '{buf_i: f_buf, c_valid: fck.c_valid, span: fck.ow_span, e: f_e};
  pl_fifo #(.W($bits(row_t)), .D(2), .BRAM(1'b0)) u_drq (
    .clk, .rst,
    .in_valid (f_row_end), .in_ready (drq_in_ready),
    .in_data  (drq_in),
    .out_valid(drq_valid), .out_ready(drq_ready), .out_data(drq),
    .count    (drq_count)
  );

  // Drain -----------------------------------------------------------------------------------
  // d0: the word's sequencing; d1: bank reads; d2: rotate, mask -> W FIFO.
  localparam int WD = 32;                  // W FIFO depth
  localparam int OW = $clog2(WR_OUTS) + 1;
  logic        d_act;
  row_t        dr;
  logic [3:0]  d_c, d_k, d_nw, d_bend;     // channel, word, run words, burst end word
  logic [31:0] d_e;                        // run start element
  logic        d_bstart;                   // word d_k starts a burst
  logic [OW-1:0] outs;                     // bursts queued on AW, B not yet received
  logic [5:0]  wf_count;
  logic        aw_in_ready, d_go, d1_v, d2_v, d_end;
  logic [59:0] d_waddr;
  logic [8:0]  d_to4k;

  // an output run (<= LBC columns) is at most LBW + 1 words
  if (WR_BURST < LBW + 1) begin : g_burst_check
    $error("pl_writer: WR_BURST must hold a run of LBW + 1 words");
  end

  // the word's address; the burst splits where it crosses 4 KiB
  assign d_waddr = j.y_w + 60'(d_e[31:3]) + 60'(d_k);
  assign d_to4k  = 9'd256 - {1'b0, d_waddr[7:0]};

  assign d_go = d_act && (32'(wf_count) + 32'(d1_v) + 32'(d2_v) < WD) &&
                (!d_bstart || (aw_in_ready && (outs != OW'(WR_OUTS))));
  assign d_end   = d_go && (d_k + 4'd1 == d_nw) && (4'(d_c) + 4'd1 == dr.c_valid);
  assign drq_ready = !d_act;

  // burst queued at its first word: length = up to the run end or the 4 KiB line
  logic [3:0] b_len;
  assign b_len = (9'(4'(d_nw - d_k)) > d_to4k) ? d_to4k[3:0] : 4'(d_nw - d_k);

  always_ff @(posedge clk) begin
    if (rst) begin
      d_act <= 1'b0;
      full  <= '0;
    end else begin
      if (f_row_end) full[f_buf] <= 1'b1;
      if (drq_valid && drq_ready) begin
        dr       <= drq;
        d_act    <= 1'b1;
        d_c      <= '0;
        d_k      <= '0;
        d_e      <= drq.e;
        d_bstart <= 1'b1;
        begin
          logic [9:0] ns;
          ns   = 10'(drq.e[2:0]) + 10'(drq.span) + 10'd7;
          d_nw <= ns[6:3];
        end
      end else if (d_go) begin
        if (d_bstart) d_bend <= d_k + b_len - 4'd1;
        d_bstart <= (d_bstart ? (b_len == 4'd1) : (d_k == d_bend)) && (d_k + 4'd1 != d_nw);
        if (d_k + 4'd1 == d_nw) begin
          d_k      <= '0;
          d_bstart <= 1'b1;
          d_c      <= d_c + 3'd1;
          d_e      <= d_e + j.out_hw;
          begin
            logic [31:0] en;
            logic [9:0]  ns;
            en   = d_e + j.out_hw;
            ns   = 10'(en[2:0]) + 10'(dr.span) + 10'd7;
            d_nw <= ns[6:3];
          end
          if (d_end) d_act <= 1'b0;
        end else begin
          d_k <= d_k + 4'd1;
        end
      end
      if (d2_v && d2_end) full[d2_buf] <= 1'b0;
    end
  end

  // d1: read the banks for word d_k; bank b holds lane (b + shift) mod E of
  // the run's word, column word d_k (or d_k - 1 where the lane index wrapped)
  logic [6:0]    rb_ra [E];
  logic [EW-1:0] rb_rd [E];
  always_comb
    for (int b = 0; b < E; b++) begin
      logic [3:0] bs;
      logic [3:0] w;
      bs = 4'(b) + 4'(d_e[2:0]);
      w  = bs[3] ? ((d_k == 0) ? 4'd0 : d_k - 4'd1) : d_k;
      rb_ra[b] = {dr.buf_i, d_c[2:0], w[2:0]};
    end

  for (genvar b = 0; b < E; b++) begin : g_rb
    pl_lutram #(.W(EW), .D(2 * TC * LBW)) u_rb (
      .clk, .we (rb_we[b]), .waddr (rb_wa[b]), .wdata (rb_wd[b]),
      .raddr (rb_ra[b]), .rdata (rb_rd[b])
    );
  end

  logic [EW-1:0] d1_rv [E];
  logic [3:0]    d1_k;
  logic [2:0]    d1_sh;
  logic [6:0]    d1_span;
  logic          d1_last, d1_end, d1_buf;
  logic [BW-1:0] d2_data;
  logic [2*E-1:0] d2_strb;
  logic          d2_last, d2_end, d2_buf;

  always_ff @(posedge clk) begin
    if (rst) begin
      d1_v <= 1'b0;
      d2_v <= 1'b0;
    end else begin
      d1_v <= d_go;
      d2_v <= d1_v;
    end
    if (d_go) begin
      d1_rv   <= rb_rd;
      d1_k    <= d_k;
      d1_sh   <= d_e[2:0];
      d1_span <= dr.span;
      d1_last <= d_bstart ? (b_len == 4'd1) || (d_k + 4'd1 == d_nw) : (d_k == d_bend) || (d_k + 4'd1 == d_nw);
      d1_end  <= d_end;
      d1_buf  <= dr.buf_i;
    end
    if (d1_v) begin
      for (int l = 0; l < E; l++) begin
        logic [7:0] col;
        logic       ok;
        col = {d1_k, 3'(l)} - 8'(d1_sh);
        ok  = !col[7] && (col < {1'b0, d1_span});
        d2_data[l * EW +: EW] <= ok ? d1_rv[3'(l) - d1_sh] : '0;
        d2_strb[l * 2 +: 2]   <= ok ? 2'b11 : 2'b00;
      end
      d2_last <= d1_last;
      d2_end  <= d1_end;
      d2_buf  <= d1_buf;
    end
  end

  logic wf_in_ready;
  pl_fifo #(.W(BW + 2 * E + 1), .D(WD), .BRAM(1'b0)) u_wf (
    .clk, .rst,
    .in_valid (d2_v),   .in_ready (wf_in_ready), .in_data ({d2_last, d2_strb, d2_data}),
    .out_valid(wvalid), .out_ready(wready),      .out_data({wlast, wstrb, wdata}),
    .count    (wf_count)
  );

  // AW queue: one entry per burst, queued with the burst's first word
  logic [3:0] aw_count;
  logic [67:0] aw_q;
  pl_fifo #(.W(68), .D(8), .BRAM(1'b0)) u_aw (
    .clk, .rst,
    .in_valid (d_go && d_bstart), .in_ready (aw_in_ready),
    .in_data  ({d_waddr, 4'b0, 4'(b_len - 4'd1)}),
    .out_valid(awvalid), .out_ready(awready), .out_data(aw_q),
    .count    (aw_count)
  );
  assign awaddr = aw_q[67:4];
  assign awlen  = {4'b0, aw_q[3:0]};

  assign bready = 1'b1;
  always_ff @(posedge clk) begin
    if (rst) outs <= '0;
    else     outs <= outs + ((d_go && d_bstart) ? OW'(1) : '0) - (bvalid ? OW'(1) : '0);
  end

  assign idle = !f_act && !fb_valid && !drq_valid && !d_act && !d1_v && !d2_v &&
                !wvalid && !awvalid && (outs == '0) && (full == '0);

  logic unused;
  assign unused = wf_in_ready ^ ^aw_count ^ ^drq_count ^ ^j;

endmodule
