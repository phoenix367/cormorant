// ---------------------------------------------------------------------------
// mm_lane — one MAC lane: R x E DSP slices (panel row r, beat lane l) fed by
// the B beats of one read port and the tap sets of the lane's x prefetcher.
//
// MAC (r, l) accumulates  acc[r][l][w] += tap[l][r] * beat[l]  where w is
// the beat's accumulator word (run word base + beat index in the row).  For
// row-major / packed B the taps of a row are all A[r][kk]; for the GEMV image
// lane l holds tap l % kw of its column, and the drain later sums the kw
// lanes of a column.  The accumulators are 32-bit LUTRAMs, ACC_D deep, one
// per MAC:  LUTRAM read -> CREG (or 0 on the run's first row) -> PREG = C + M
// -> LUTRAM write, a loop of DMIN cycles.  Rows shorter than DMIN beats get
// bubbles so no accumulator is read before its previous update lands.
//
// tap_data[l] is the tap FIFO's head of the previous cycle (mm_xpf keeps a
// copy of the read pointer per beat lane), so the B operand of a row whose
// first beat issues at t is tap_data at t+1.
//
// Step handshake: step s (= cmp_cnt) may only start when the drain has
// emptied the accumulators of step s - 1 (drn_cnt == cmp_cnt).  After the
// step's last beat the pipeline flushes, then cmp_cnt increments and
// `active` tells the drain whether this lane contributed.
// ---------------------------------------------------------------------------
module mm_lane
  import mm_pkg::*;
(
  input  logic            clk,
  input  logic            rst,

  input  logic            in_valid,
  output logic            in_ready,
  input  logic [BW-1:0]   in_data,
  input  logic            in_marker,
  input  logic            in_row_first,
  input  logic            in_row_last,
  input  logic            in_run_last,
  input  logic [8:0]      in_beat,
  input  logic [4:0]      in_row,
  input  gb_run_t         in_meta,

  input  logic            tap_valid,
  output logic            tap_pop,
  input  logic [E-1:0][R-1:0][EW-1:0] tap_data,

  input  logic [31:0]     drn_cnt,
  output logic [31:0]     cmp_cnt,
  output logic            active,

  input  logic            drd_en,
  input  logic [ACC_AW-1:0] drd_addr,
  input  logic [2:0]      drd_row,
  output logic [E-1:0][31:0] drd_data      // word of the request made 4 cycles ago
);

  // Issue control ----------------------------------------------------------------
  logic       step_open, flushing, issued_any;
  logic [3:0] fl_cnt;
  logic [2:0] gap;          // cycles since the last issue (saturating)
  logic [1:0] rowbeats;     // beats of the current row so far (saturating)
  logic [1:0] prev_len;     // beats of the previous row (saturating at DMIN)

  logic step_ok, step_q, gap_ok, d_ok, m_ok, d_fire, m_fire, end_step;
  assign step_ok = step_open || step_q;
  assign gap_ok  = !in_row_first || ({1'b0, gap} + {2'b0, prev_len} >= 4'(DMIN + 1));
  assign d_ok    = !flushing && step_ok && gap_ok && (!in_row_first || tap_valid);
  assign m_ok    = !flushing && step_ok;
  assign in_ready = in_marker ? m_ok : d_ok;
  assign d_fire  = in_valid && !in_marker && d_ok;
  assign m_fire  = in_valid &&  in_marker && m_ok;
  assign end_step = (d_fire && in_run_last && in_meta.step_last) ||
                    (m_fire && in_meta.step_last);
  assign tap_pop = d_fire && in_row_first;

  // Registered "drain has caught up" (drn_cnt == cmp_cnt), evaluated against
  // cmp_cnt's next value: it can only open late (drn_cnt only grows).
  logic        cmp_inc;
  logic [31:0] cmp_cnt_n;
  assign cmp_inc   = (end_step && m_fire) || (!end_step && flushing && fl_cnt == 4'd0);
  assign cmp_cnt_n = cmp_cnt + {31'b0, cmp_inc};
  always_ff @(posedge clk) step_q <= !rst && (drn_cnt == cmp_cnt_n);

  logic [1:0] row_len_now;
  assign row_len_now = in_row_first ? 2'd1 :
                       (rowbeats == 2'(DMIN) ? 2'(DMIN) : rowbeats + 2'd1);

  always_ff @(posedge clk) begin
    if (rst) begin
      step_open  <= 1'b0;
      flushing   <= 1'b0;
      issued_any <= 1'b0;
      cmp_cnt    <= '0;
      active     <= 1'b0;
      gap        <= 3'd7;
      rowbeats   <= '0;
      prev_len   <= 2'(DMIN);
    end else begin
      if (d_fire || m_fire) step_open <= 1'b1;
      if (d_fire) issued_any <= 1'b1;

      if (d_fire) gap <= 3'd1;
      else if (gap != 3'd7) gap <= gap + 3'd1;
      if (d_fire) begin
        rowbeats <= row_len_now;
        if (in_row_last) prev_len <= row_len_now;
      end

      if (end_step) begin
        prev_len <= 2'(DMIN);
        if (m_fire) begin
          cmp_cnt    <= cmp_cnt + 32'd1;
          active     <= issued_any;
          issued_any <= 1'b0;
          step_open  <= 1'b0;
        end else begin
          flushing <= 1'b1;
          fl_cnt   <= 4'(FLUSH);
        end
      end else if (flushing) begin
        if (fl_cnt == 4'd0) begin
          flushing   <= 1'b0;
          cmp_cnt    <= cmp_cnt + 32'd1;
          active     <= 1'b1;
          issued_any <= 1'b0;
          step_open  <= 1'b0;
        end else begin
          fl_cnt <= fl_cnt - 4'd1;
        end
      end
    end
  end

  // Pipeline ------------------------------------------------------------------------
  // The MAC array spans many DSP columns, so every signal that reaches it
  // crosses a long distance.  The issue decision (cycle t) is registered once
  // here (stage 1, x1 / v1 / p1 / ...), copied once per beat lane l (stage 2:
  // the 8 MACs of a beat lane sit together, they share the beat element), and
  // the operands enter the DSP input registers (mm_mac: AREG = BREG = 2):
  //
  //   t     decision; tap FIFO pop (rptr advances)
  //   t+1   x1 -> A1          tap_data (the head at t) -> B1   v2 p2 f2 ra2
  //   t+2   A1 -> A2          B1 -> B2 (p2)                    ra3 f3 (per half)
  //   t+3   M = A2 * B2       C = f3 ? 0 : acc[ra3]
  //   t+4   P = C + M
  //   t+5   acc[w5] <= P
  //
  // The accumulator loop (read at t+3, write at the end of t+5) is still
  // DMIN = 3 cycles.  A drain read requested at T reads at T+3 (ra3) and its
  // row-selected word is in drd_data at T+4.
  logic [ACC_AW-1:0] w0;
  logic              f0;
  assign w0 = in_meta.wb + in_beat[ACC_AW-1:0];
  assign f0 = in_meta.init && (in_row == 5'd0);

  // stage 1
  logic [BW-1:0]     x1;
  logic              v1, p1, f1, de1;
  logic [ACC_AW-1:0] wa1, da1;
  logic [2:0]        dr1;
  always_ff @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0; p1 <= 1'b0; de1 <= 1'b0;
    end else begin
      v1 <= d_fire; p1 <= tap_pop; de1 <= drd_en;
    end
    x1  <= in_data;
    wa1 <= w0;
    f1  <= f0;
    da1 <= drd_addr;
    dr1 <= drd_row;
  end

  // MAC array + accumulators, one group per beat lane ---------------------------------
  localparam int RHF = R / 2;      // rows per accumulator-address copy

  for (genvar l = 0; l < E; l++) begin : g_l
    // stage 2 (per beat lane)
    (* keep = "true" *) logic              v2, p2, f2;
    (* keep = "true" *) logic [ACC_AW-1:0] ra2;
    (* keep = "true" *) logic [2:0]        dr2;
    // stage 3 .. 5 (accumulator addresses per half of the rows)
    (* keep = "true" *) logic [1:0][ACC_AW-1:0] ra3, w4, w5;
    (* keep = "true" *) logic [1:0]        f3;
    logic              v3, v4, v5;
    logic [2:0]        dr3;
    always_ff @(posedge clk) begin
      if (rst) begin
        v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; p2 <= 1'b0;
      end else begin
        v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; p2 <= p1;
      end
      f2  <= f1;
      ra2 <= de1 ? da1 : wa1;
      dr2 <= dr1;
      dr3 <= dr2;
      for (int h = 0; h < 2; h++) begin
        ra3[h] <= ra2;
        f3[h]  <= f2;
        w4[h]  <= ra3[h];
        w5[h]  <= w4[h];
      end
    end

    logic [R-1:0][31:0] acc_rd, p;
    for (genvar r = 0; r < R; r++) begin : g_r
      (* ram_style = "distributed" *) logic [31:0] acc [ACC_D];
      always_ff @(posedge clk) if (v5) acc[w5[r / RHF]] <= p[r];
      assign acc_rd[r] = acc[ra3[r / RHF]];

      mm_mac u_mac (
        .clk,
        .a     (x1[l*EW +: EW]),
        .b     (tap_data[l][r]),
        .b_ce  (p2),
        .c     (acc_rd[r]),
        .c_rst (f3[r / RHF]),
        .p     (p[r])
      );
    end

    // drain: the word of row dr3, registered next to the group
    always_ff @(posedge clk) drd_data[l] <= acc_rd[dr3];
  end

endmodule
