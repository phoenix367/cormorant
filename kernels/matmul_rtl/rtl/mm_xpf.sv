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
// started when an entry is free.  A tap set is in the FIFO 6 cycles after its
// row's last read issues; tap_data is the FIFO head of the previous cycle.  xpf_cnt counts the panels whose A reads
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

  // Write slot of the row being read: rows are written in issue order.
  logic [TAP_AW-1:0] ws;
  always_ff @(posedge clk) begin
    if (rst) ws <= '0;
    else if (row_done) ws <= ws + 1'b1;
  end

  // Read pipeline.  The A buffer's rows sit in several block-RAM columns and
  // the tap FIFO next to the MAC lane's DSP columns (one LUTRAM per beat
  // lane, next to that beat lane's 8 DSPs), so both ends are registered and
  // the tap set is copied per beat lane before it is written:
  //   t    issue (re, K index)
  //   t+1  re / raddr registered -> RAM read (the RAM's latch)
  //   t+2  RAM output register (mm_abuf)
  //   t+3  per row: element select into tv (next to the row's RAM)
  //   t+4  per beat lane: the tap set, write enable and slot copied (tvg)
  //   t+5  tap FIFO write; the entry is visible from t+6
  logic            re_q;
  logic [A_AW-1:0] raddr_q;
  always_ff @(posedge clk) begin
    if (rst) re_q <= 1'b0;
    else     re_q <= issue;
    raddr_q <= A_AW'(kl >> 3);
  end
  assign re    = re_q;
  assign raddr = raddr_q;

  logic [2:0]        sel1, sel2, j1, j2, j3;
  logic              v1, v2, v3, v4, v5, last1, last2, last3, last4, last5;
  logic [TAP_AW-1:0] ws1, ws2, ws3;
  (* keep = "true" *) logic [R-1:0][2:0] sel3;      // per A-buffer row
  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0;
    end else begin
      v1 <= issue; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4;
    end
    sel1 <= kl[2:0]; j1 <= j; last1 <= row_done; ws1 <= ws;
    sel2 <= sel1;    j2 <= j1; last2 <= last1;   ws2 <= ws1;
    for (int r = 0; r < R; r++) sel3[r] <= sel2;
    j3 <= j2; last3 <= last2; ws3 <= ws2;
    last4 <= last3;
    last5 <= last4;
  end

  // t+3: per row, the selected element
  logic [R-1:0][EW-1:0] tv;
  always_ff @(posedge clk)
    for (int r = 0; r < R; r++) tv[r] <= rdata[r][sel3[r] * EW +: EW];

  // Tap FIFO: one LUTRAM per beat lane, written by the taps it uses.  The
  // read pointer is copied per beat lane, one cycle late: tap_data[l] is the
  // head of the previous cycle (what mm_lane expects).
  logic [TAP_AW-1:0] rptr;
  logic [TAP_AW:0]   count;
  logic              we4_c [E];
  logic [TAP_AW-1:0] ws4_c;
  always_ff @(posedge clk) begin
    for (int l = 0; l < E; l++) we4_c[l] <= v3 && ((3'(l) & kwm) == j3);
    ws4_c <= ws3;
  end

  for (genvar l = 0; l < E; l++) begin : g_tap
    (* keep = "true" *) logic              we5;
    (* keep = "true" *) logic [TAP_AW-1:0] ws5, rptr_d;
    (* keep = "true" *) logic [R*EW-1:0]   tvg;
    (* ram_style = "distributed" *) logic [R*EW-1:0] mem [TAP_D];
    always_ff @(posedge clk) begin
      we5    <= we4_c[l];
      ws5    <= ws4_c;
      tvg    <= tv;
      rptr_d <= rptr;
      if (we5) mem[ws5] <= tvg;
    end
    assign tap_data[l] = mem[rptr_d];
  end

  assign tap_valid = (count != '0);

  always_ff @(posedge clk) begin
    if (rst) begin
      rptr <= '0; count <= '0;
    end else begin
      if (tap_fire) rptr <= rptr + 1'b1;
      count <= count + (TAP_AW+1)'(v5 && last5) - (TAP_AW+1)'(tap_fire);
    end
  end

endmodule
