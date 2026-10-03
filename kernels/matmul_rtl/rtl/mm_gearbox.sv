// ---------------------------------------------------------------------------
// mm_gearbox — turns the raw 128-bit words of one read port into row-aligned
// beats.
//
// Each run descriptor (mm_pkg::gb_run_t) covers `rows` consecutive rows of
// `len` elements starting at lane `s` of its first word; the run occupies
// exactly `nw` words of the input stream.  Every output beat carries up to 8
// elements of ONE row, element i of the beat being row element
// 8 * out_beat + i (lanes past the row end are don't-care).  A marker run
// (rows == 0) consumes no words and yields one beat with `out_marker` set.
//
// Data path: a two-word window {w1, w0} and one 8-way lane rotate by the
// element offset of the next output element.  At most one input word is
// consumed per output beat, so long rows stream at one beat per cycle.  The
// next descriptor is held in a lookahead register and its first words enter
// the window behind the current run's last word, so back-to-back runs do not
// leave bubbles.
// ---------------------------------------------------------------------------
module mm_gearbox
  import mm_pkg::*;
(
  input  logic          clk,
  input  logic          rst,

  input  logic          run_valid,
  output logic          run_ready,
  input  gb_run_t       run,

  input  logic          in_valid,
  output logic          in_ready,
  input  logic [BW-1:0] in_data,

  output logic          out_valid,
  input  logic          out_ready,
  output logic [BW-1:0] out_data,
  output logic          out_marker,
  output logic          out_row_first,
  output logic          out_row_last,
  output logic          out_run_first,
  output logic          out_run_last,
  output logic [8:0]    out_beat,      // beat index within the row
  output logic [4:0]    out_row,       // row index within the run
  output gb_run_t       out_meta
);

  // Current run and lookahead descriptor ---------------------------------------
  logic          active, nxt_v;
  gb_run_t       cur, nxt;
  logic [2:0]    off;         // element offset in w0 of the next output element
  logic [12:0]   row_rem;     // elements left in the current row
  logic [4:0]    rows_left;   // rows left including the current one
  logic [11:0]   words_in;    // words of `cur` taken into the window
  logic [11:0]   words_nx;    // words of `nxt` taken into the window
  logic [8:0]    beat_i;
  logic [4:0]    row_i;
  logic          first_beat;

  logic [BW-1:0] w0, w1;
  logic          v0, v1;

  // Output register stage ------------------------------------------------------
  logic          o_valid;
  logic          adv;
  assign adv = !o_valid || out_ready;

  // Beat production ------------------------------------------------------------
  logic [3:0]  need;
  logic [4:0]  off_end;
  logic        data_ok, fire, fire_mk, pop0, row_end, run_end;

  assign need    = (row_rem >= 13'd8) ? 4'd8 : row_rem[3:0];
  assign off_end = {2'b0, off} + {1'b0, need};
  assign data_ok = v0 && (off_end <= 5'd8 || v1);
  assign fire_mk = active && cur.marker && adv;
  assign fire    = active && !cur.marker && data_ok && adv;
  assign pop0    = fire && (off_end >= 5'd8);
  assign row_end = fire && (row_rem == {9'b0, need});
  assign run_end = fire_mk || (row_end && rows_left == 5'd1);

  logic [2*BW-1:0] win;
  logic [BW-1:0]   rot;
  assign win = {w1, w0};
  assign rot = win[off * EW +: BW];

  // Window after this cycle's consumption.
  logic          v0_c, v1_c, sh_c;   // sh_c: w0 <= w1
  always_comb begin
    v0_c = v0; v1_c = v1; sh_c = 1'b0;
    if (fire && run_end) begin
      // Drop the finished run's words: w0 always, w1 too when the last beat
      // reached into it; otherwise w1 (if valid) is the next run's first word.
      if (off_end > 5'd8) begin
        v0_c = 1'b0; v1_c = 1'b0;
      end else begin
        v0_c = v1; v1_c = 1'b0; sh_c = 1'b1;
      end
    end else if (pop0) begin
      v0_c = v1; v1_c = 1'b0; sh_c = 1'b1;
    end
  end

  // Which run the next input word belongs to.
  logic cur_wants, nxt_wants, room, take, take_nx;
  assign cur_wants = active && !cur.marker && (words_in != cur.nw);
  assign nxt_wants = nxt_v && !nxt.marker && (words_nx != nxt.nw);
  assign room      = !(v0_c && v1_c);
  assign in_ready  = room && (cur_wants || nxt_wants);
  assign take      = in_ready && in_valid;
  assign take_nx   = take && !cur_wants;

  // Promote the lookahead descriptor.
  logic promote;
  assign promote   = nxt_v && (!active || run_end);
  assign run_ready = !nxt_v;

  always_ff @(posedge clk) begin
    if (rst) begin
      active  <= 1'b0;
      nxt_v   <= 1'b0;
      v0      <= 1'b0;
      v1      <= 1'b0;
      o_valid <= 1'b0;
    end else begin
      // Lookahead descriptor.
      if (run_valid && run_ready) begin
        nxt      <= run;
        nxt_v    <= 1'b1;
        words_nx <= '0;
      end else if (promote) begin
        nxt_v    <= 1'b0;
      end
      if (take_nx && !promote) words_nx <= words_nx + 12'd1;

      // Current run.
      if (promote) begin
        active     <= 1'b1;
        cur        <= nxt;
        off        <= nxt.s;
        row_rem    <= nxt.len;
        rows_left  <= nxt.rows;
        words_in   <= words_nx + (take_nx ? 12'd1 : 12'd0);
        beat_i     <= '0;
        row_i      <= '0;
        first_beat <= 1'b1;
      end else begin
        if (run_end) active <= 1'b0;
        if (take && !take_nx) words_in <= words_in + 12'd1;
        if (fire || fire_mk) first_beat <= 1'b0;
        if (fire) begin
          off <= off_end[2:0];
          if (row_end) begin
            row_rem   <= cur.len;
            rows_left <= rows_left - 5'd1;
            beat_i    <= '0;
            row_i     <= row_i + 5'd1;
          end else begin
            row_rem   <= row_rem - {9'b0, need};
            beat_i    <= beat_i + 9'd1;
          end
        end
      end

      // Window.
      if (sh_c) w0 <= w1;
      v0 <= v0_c;
      v1 <= v1_c;
      if (take) begin
        if (!v0_c) begin w0 <= in_data; v0 <= 1'b1; end
        else       begin w1 <= in_data; v1 <= 1'b1; end
      end

      // Output register.
      if (adv) begin
        o_valid <= fire || fire_mk;
        if (fire || fire_mk) begin
          out_data      <= rot;
          out_marker    <= cur.marker;
          out_row_first <= fire && (beat_i == '0);
          out_row_last  <= fire && row_end;
          out_run_first <= first_beat;
          out_run_last  <= run_end;
          out_beat      <= beat_i;
          out_row       <= row_i;
          out_meta      <= cur;
        end
      end
    end
  end

  assign out_valid = o_valid;

endmodule
