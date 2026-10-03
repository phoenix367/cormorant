// ---------------------------------------------------------------------------
// mm_xpf — x prefetcher of lane H.
//
// For every B row (plane) the lane will stream, reads the A values that
// multiply it — one per panel row and GEMV tap — and assembles them into a
// "tap set": tap[l][r] is the A operand of MAC (row r, beat lane l) for the
// whole B row.  Tap j of plane cl (lane-local) is at lane-local K index
//     lk == 0:  cl
//     lk  > 0:  ((cl >> 4) << (4 + lk)) | (j << 4) | (cl & 15)
// and feeds the beat lanes l with l % kw == j.  kw = 1 << lk reads per row.
//
// Tap sets queue in a LUTRAM FIFO (TAP_D entries); a row's reads are only
// started when an entry is free.  xpf_cnt counts the panels whose A reads
// are complete; a panel's first read waits until both A writers finished it.
// ---------------------------------------------------------------------------
module mm_xpf
  import mm_pkg::*;
(
  input  logic            clk,
  input  logic            rst,
  input  cfg_t            cfg,

  input  logic            xq_valid,
  output logic            xq_ready,
  input  x_run_t          xq,

  input  logic [1:0][31:0] awr_cnt,
  output logic [31:0]     xpf_cnt,

  output logic            re,
  output logic [A_AW-1:0] raddr,
  input  logic [R-1:0][BW-1:0] rdata,

  output logic            tap_valid,
  input  logic            tap_pop,
  output logic [E-1:0][R-1:0][EW-1:0] tap_data
);
  // Current run.
  logic        active, marker, plast, pan_ok;
  logic [4:0]  rows_left;
  logic [10:0] cl;
  logic [2:0]  j;
  logic [2:0]  kwm;
  assign kwm = 3'((4'd1 << cfg.lk) - 4'd1);

  logic [TAP_AW:0] reserved;   // tap entries in use or being assembled
  logic            bar_q;
  logic pan_go;
  assign pan_go = pan_ok || bar_q;

  logic row_start, issue, row_done, run_done;
  assign row_start = (j == 3'd0);
  assign issue     = active && !marker && pan_go &&
                     (!row_start || reserved != (TAP_AW+1)'(TAP_D));
  assign row_done  = issue && (j == kwm);
  assign run_done  = (row_done && rows_left == 5'd1) || (active && marker && pan_go);

  assign xq_ready = !active;

  // K index of tap j of plane cl.
  logic [10:0] kl;
  always_comb begin
    if (cfg.lk == 2'd0) kl = cl;
    else kl = 11'(({7'b0, cl[10:4]} << (4'd4 + 4'(cfg.lk))) | {7'b0, j, 4'b0} | {10'b0, cl[3:0]});
  end

  logic tap_fire;
  assign tap_fire = tap_valid && tap_pop;

  // Registered barrier (both A writers are past this lane's panel), evaluated
  // against xpf_cnt's next value: it can only open late, never early.
  logic [31:0] xpf_cnt_n;
  assign xpf_cnt_n = xpf_cnt + {31'b0, run_done && plast};
  always_ff @(posedge clk)
    bar_q <= !rst && (awr_cnt[0] != xpf_cnt_n) && (awr_cnt[1] != xpf_cnt_n);

  always_ff @(posedge clk) begin
    if (rst) begin
      active   <= 1'b0;
      pan_ok   <= 1'b0;
      xpf_cnt  <= '0;
      reserved <= '0;
    end else begin
      if (!active && xq_valid) begin
        active    <= 1'b1;
        marker    <= xq.marker;
        plast     <= xq.panel_last;
        rows_left <= xq.rows;
        cl        <= xq.cl0;
        j         <= 3'd0;
      end
      if (active && pan_go) pan_ok <= 1'b1;
      if (issue) begin
        if (row_done) begin
          j         <= 3'd0;
          cl        <= cl + 11'd1;
          rows_left <= rows_left - 5'd1;
        end else begin
          j <= j + 3'd1;
        end
      end
      if (run_done) begin
        active <= 1'b0;
        if (plast) begin
          xpf_cnt <= xpf_cnt + 32'd1;
          pan_ok  <= 1'b0;
        end
      end
      reserved <= reserved + (TAP_AW+1)'(issue && row_start) - (TAP_AW+1)'(tap_fire);
    end
  end

  assign re    = issue;
  assign raddr = A_AW'(kl >> 3);

  // Read pipeline: RAM (2 cycles) -> element select -> tap FIFO write.
  logic [2:0] sel1, sel2, j1, j2;
  logic       v1, v2, last1, last2;
  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0; v2 <= 1'b0;
    end else begin
      v1 <= issue; v2 <= v1;
    end
    sel1 <= kl[2:0]; j1 <= j; last1 <= row_done;
    sel2 <= sel1;    j2 <= j1; last2 <= last1;
  end

  logic [R-1:0][EW-1:0] tv;
  always_comb
    for (int r = 0; r < R; r++) tv[r] = rdata[r][sel2 * EW +: EW];

  // Tap FIFO: one LUTRAM per beat lane, written by the taps it uses.
  logic [TAP_AW-1:0] wptr, rptr;
  logic [TAP_AW:0]   count;
  for (genvar l = 0; l < E; l++) begin : g_tap
    (* ram_style = "distributed" *) logic [R*EW-1:0] mem [TAP_D];
    always_ff @(posedge clk)
      if (v2 && ((3'(l) & kwm) == j2)) mem[wptr] <= tv;
    assign tap_data[l] = mem[rptr];
  end

  assign tap_valid = (count != '0);

  always_ff @(posedge clk) begin
    if (rst) begin
      wptr <= '0; rptr <= '0; count <= '0;
    end else begin
      if (v2 && last2) wptr <= wptr + 1'b1;
      if (tap_fire)    rptr <= rptr + 1'b1;
      count <= count + (TAP_AW+1)'(v2 && last2) - (TAP_AW+1)'(tap_fire);
    end
  end

endmodule
