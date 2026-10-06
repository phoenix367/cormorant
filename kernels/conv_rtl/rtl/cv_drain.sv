// ---------------------------------------------------------------------------
// cv_drain — a finished chunk's accumulators to channel-major output words
// (the HLS Phase 3), while the engine computes the next chunk in the other
// accumulator buffer.
//
// Per (m-tile, 256-pixel segment): the fill reads the segment's accumulator
// words (one per pixel, TM channels), floors and saturates them and scatters
// them into one of two transposer buffers — 8 channels per cycle, so a tile
// with more than 8 valid channels takes two cycles per pixel; the emitter
// reads the other buffer as runs of 8 consecutive pixels of one channel, one
// output word per cycle, and hands the writer one run per (channel,
// segment).  Pixel p of channel m1 lives in bank (m1 + p) mod 8 at address
// m1 * 32 + p / 8, so the fill's 8 channels and the emitter's 8 pixels are
// each spread over the 8 banks.
// ---------------------------------------------------------------------------
module cv_drain
  import cv_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  job_t          j,

  input  logic          dr_valid,
  output logic          dr_ready,
  input  dreq_t         dr,

  output logic          d_ren,             // accumulator reads (latency 3)
  output logic          d_par,
  output logic [11:0]   d_addr,
  input  logic [TM*32-1:0] d_rdata,
  output logic          d_free,            // the chunk's buffer is read to the end
  output logic          d_free_par,

  output logic          run_valid,         // one (channel, segment) run
  input  logic          run_ready,
  output logic [31:0]   run_e,             // element offset in y
  output logic [8:0]    run_len,           // 1 .. DSEG
  output logic          yw_valid,          // its words, ceil(len / 8) of them
  input  logic          yw_ready,
  output logic [BW-1:0] yw_data,

  output logic          idle
);
  localparam int SW = DSEG / E;            // words per (channel, segment): 32

  // ---- segment descriptors between the fill and the emitter ----------------------
  typedef struct packed {
    logic        buf_i;
    logic [4:0]  mv;                       // valid channels
    logic [8:0]  len;                      // pixels
    logic [31:0] e0;                       // element offset of channel 0 of the tile, segment start
  } seg_t;

  // ---- fill ---------------------------------------------------------------------------
  logic        f_act, f_par;
  logic [31:0] f_ycb;
  logic [12:0] f_runlen;
  logic [6:0]  f_mt;                       // m-tile
  logic [12:0] f_seg0;                     // first pixel of the segment
  logic [8:0]  f_len, f_p;                 // segment length, pixels read
  logic [4:0]  f_mv;
  logic [11:0] f_word;                     // accumulator word of pixel f_p
  logic        f_two;                      // two fill cycles per pixel (mv > 8)
  logic        f_half;                     // read pacing: second cycle of a two-cycle pixel
  logic        f_buf;                      // transposer buffer being filled
  logic [1:0]  full;                       // transposer buffers holding a segment (or being scattered)
  logic [31:0] f_mbase;                    // element offset of the tile's channel 0
  logic        sq_in_ready;

  // the fill can start a segment when its buffer is free and the queue has room
  logic f_issue, f_seg_last_rd, f_chunk_last_rd;

  assign dr_ready = !f_act;

  function automatic logic [4:0] mvalid(input logic [31:0] rem);
    return (rem >= 32'(TM)) ? 5'(TM) : rem[4:0];
  endfunction

  // one accumulator read per pixel, every cycle (mv <= 8) or every other
  assign f_issue = f_act && (f_p != f_len) && !full[f_buf] && !f_half;
  assign d_ren   = f_issue;
  assign d_par   = f_par;
  assign d_addr  = f_word;

  assign f_seg_last_rd   = f_issue && (f_p + 9'd1 == f_len);
  assign f_chunk_last_rd = f_seg_last_rd && (f_mt + 7'd1 == j.m_tiles) &&
                           (f_seg0 + 13'(DSEG) >= f_runlen);

  always_ff @(posedge clk) begin
    if (rst) begin
      f_act <= 1'b0;
      f_buf <= 1'b0;
    end else if (!f_act) begin
      if (dr_valid) begin
        f_act    <= 1'b1;
        f_par    <= dr.par;
        f_ycb    <= dr.y_cbase;
        f_runlen <= dr.run_len;
        f_mt     <= '0;
        f_seg0   <= '0;
        f_len    <= (dr.run_len >= 13'(DSEG)) ? 9'(DSEG) : dr.run_len[8:0];
        f_p      <= '0;
        f_mv     <= mvalid(j.out_ch);
        f_two    <= (j.out_ch > 32'd8);
        f_half   <= 1'b0;
        f_word   <= '0;
        f_mbase  <= dr.y_cbase;
      end
    end else begin
      if (f_issue) begin
        f_p    <= f_p + 9'd1;
        f_word <= f_word + {5'b0, j.m_tiles};
        f_half <= f_two;
      end else if (f_half) begin
        f_half <= 1'b0;
      end
      // the segment's last read has gone out: next segment / tile / done
      if (f_seg_last_rd) begin
        f_buf <= !f_buf;
        f_p   <= '0;
        if (f_seg0 + 13'(DSEG) < f_runlen) begin
          f_seg0 <= f_seg0 + 13'(DSEG);
          f_len  <= (f_runlen - f_seg0 - 13'(DSEG) >= 13'(DSEG)) ? 9'(DSEG)
                                                               : 9'(f_runlen - f_seg0 - 13'(DSEG));
          // f_word has advanced one pixel (m_tiles words) per read: it is at the
          // next segment's first pixel already
        end else begin
          f_seg0  <= '0;
          f_len   <= (f_runlen >= 13'(DSEG)) ? 9'(DSEG) : f_runlen[8:0];
          f_mt    <= f_mt + 7'd1;
          f_word  <= {5'b0, f_mt + 7'd1};
          f_mbase <= f_mbase + {j.out_hw[27:0], 4'b0};
          f_mv    <= mvalid(j.out_ch - {21'b0, f_mt + 7'd1, 4'b0});
          f_two   <= (j.out_ch - {21'b0, f_mt + 7'd1, 4'b0} > 32'd8);
          if (f_mt + 7'd1 == j.m_tiles) f_act <= 1'b0;
        end
      end
    end
  end

  assign d_free     = f_chunk_last_rd_d[3];
  assign d_free_par = f_par_d[3];
  logic [3:0] f_chunk_last_rd_d;
  logic [3:0] f_par_d;
  always_ff @(posedge clk) begin
    if (rst) f_chunk_last_rd_d <= '0;
    else     f_chunk_last_rd_d <= {f_chunk_last_rd_d[2:0], f_chunk_last_rd};
    f_par_d <= {f_par_d[2:0], f_par};
  end

  // read data returns 3 cycles after the address (fr_d[3]); it is registered
  // (fr_d[4]) and saturated into a register (fr_d[5]), then the pixel's 16
  // channels are scattered in one or two cycles (channels 0-7, then 8-15)
  typedef struct packed {
    logic        v;
    logic        two;
    logic        buf_i;
    logic [8:0]  p;
    logic        seg_last;
    logic [4:0]  mv;
    logic [8:0]  len;
    logic [31:0] e0;
  } fr_t;
  fr_t fr0, fr_d [1:5];
  assign fr0 = '{v: f_issue, two: f_two, buf_i: f_buf, p: f_p, seg_last: f_seg_last_rd,
                 mv: f_mv, len: f_len, e0: f_mbase + {19'b0, f_seg0}};
  always_ff @(posedge clk) begin
    if (rst) begin
      fr_d[1] <= '0; fr_d[2] <= '0; fr_d[3] <= '0; fr_d[4] <= '0; fr_d[5] <= '0;
    end else begin
      fr_d[1] <= fr0; fr_d[2] <= fr_d[1]; fr_d[3] <= fr_d[2]; fr_d[4] <= fr_d[3]; fr_d[5] <= fr_d[4];
    end
  end

  // register the arriving word, saturate it into a register; keep channels
  // 8-15 for the second cycle
  logic [TM*32-1:0] d_rq;
  logic [E*EW-1:0] sat_lo, sat_hi, hi_q;
  logic            hi_pend, hi_buf, hi_last;
  logic [8:0]      hi_p;
  fr_t             hi_fr;
  always_ff @(posedge clk) begin
    d_rq <= d_rdata;
    for (int l = 0; l < E; l++) begin
      sat_lo[l * EW +: EW] <= sat16(d_rq[l * 32 +: 32]);
      sat_hi[l * EW +: EW] <= sat16(d_rq[(E + l) * 32 +: 32]);
    end
  end

  // transposer: 8 banks x (2 buffers x TM channels x SW words)
  logic            t_we;
  logic            t_buf;
  logic [3:0]      t_h;                    // channel group base (0 or 8)
  logic [8:0]      t_p;
  logic [E*EW-1:0] t_lanes;                // channel t_h + j in lane j
  always_comb begin
    if (hi_pend) begin
      t_we = 1'b1; t_buf = hi_buf; t_h = 4'd8; t_p = hi_p; t_lanes = hi_q;
    end else begin
      t_we = fr_d[5].v; t_buf = fr_d[5].buf_i; t_h = 4'd0; t_p = fr_d[5].p; t_lanes = sat_lo;
    end
  end

  logic seg_done;                          // a segment's last scatter
  always_ff @(posedge clk) begin
    if (rst) begin
      hi_pend <= 1'b0;
    end else begin
      hi_pend <= fr_d[5].v && fr_d[5].two;
    end
    hi_q    <= sat_hi;
    hi_buf  <= fr_d[5].buf_i;
    hi_p    <= fr_d[5].p;
    hi_last <= fr_d[5].seg_last;
    hi_fr   <= fr_d[5];
  end
  assign seg_done = (fr_d[5].v && fr_d[5].seg_last && !fr_d[5].two) || (hi_pend && hi_last);

  // the filled segment's descriptor
  seg_t sq_in, sq;
  logic sq_valid, sq_pop;
  logic [2:0] sq_n;
  always_comb begin
    if (hi_pend && hi_last) sq_in = '{buf_i: hi_fr.buf_i, mv: hi_fr.mv, len: hi_fr.len, e0: hi_fr.e0};
    else                    sq_in = '{buf_i: fr_d[5].buf_i, mv: fr_d[5].mv, len: fr_d[5].len, e0: fr_d[5].e0};
  end
  cv_fifo #(.W($bits(seg_t)), .D(4), .BRAM(1'b0)) u_sq (
    .clk, .rst, .in_valid (seg_done), .in_ready (sq_in_ready), .in_data (sq_in),
    .out_valid (sq_valid), .out_ready (sq_pop), .out_data (sq), .count (sq_n)
  );

  // read latency 2 (the BRAM's output register): the data of the address
  // given with e_go arrives with rd_v2
  logic [E*EW-1:0] t_rdata;
  logic [9:0]      t_raddr;
  for (genvar b = 0; b < E; b++) begin : g_tb
    (* ram_style = "block" *) logic [EW-1:0] mem [2 * TM * SW];
    logic [2:0]  jl;                       // the lane (channel t_h + jl) bank b takes
    logic [9:0]  wa;
    logic [EW-1:0] rd;
    assign jl = 3'(b) - t_p[2:0];
    assign wa = {t_buf, t_h[3] ? 1'b1 : 1'b0, jl, t_p[7:3]};   // buf, channel (t_h + jl), word p / 8
    always_ff @(posedge clk) begin
      if (t_we) mem[wa] <= t_lanes[jl * EW +: EW];
      rd <= mem[t_raddr];
      t_rdata[b * EW +: EW] <= rd;
    end
  end

  // ---- emitter --------------------------------------------------------------------
  logic        e_act;
  seg_t        es;
  logic [4:0]  e_m1;
  logic [5:0]  e_j, e_nw;                  // word within the channel's run
  logic [31:0] e_e;
  logic        e_go, e_runpush;
  logic        rd_v1, rd_v2;
  logic [2:0]  rd_rot1, rd_rot2;
  logic        yq_in_ready, rq_in_ready;
  logic [5:0]  yq_n;
  logic [2:0]  rq_n;

  // output word queue (credit: words in the RAM read pipeline count)
  assign e_go     = e_act && (32'(yq_n) + 32'(rd_v1) + 32'(rd_v2) < 30) && (e_j != 0 || rq_in_ready);
  assign e_runpush = e_go && (e_j == 6'd0);
  assign t_raddr  = {es.buf_i, e_m1[3:0], e_j[4:0]};
  assign sq_pop   = !e_act && sq_valid;

  always_ff @(posedge clk) begin
    if (rst) begin
      e_act <= 1'b0;
      full  <= '0;
      rd_v1 <= 1'b0;
      rd_v2 <= 1'b0;
    end else begin
      if (sq_pop) begin
        e_act <= 1'b1;
        es    <= sq;
        e_m1  <= '0;
        e_j   <= '0;
        e_nw  <= 6'((sq.len + 9'd7) >> 3);
        e_e   <= sq.e0;
      end else if (e_go) begin
        if (e_j + 6'd1 == e_nw) begin
          e_j <= '0;
          e_e <= e_e + j.out_hw;
          if (e_m1 + 5'd1 == es.mv) begin
            e_act <= 1'b0;
            full[es.buf_i] <= 1'b0;
          end else begin
            e_m1 <= e_m1 + 5'd1;
          end
        end else begin
          e_j <= e_j + 6'd1;
        end
      end
      // a buffer is taken from its segment's last read on (its scatter is
      // still under way), so the fill cannot start the segment after next in it
      if (f_seg_last_rd) full[f_buf] <= 1'b1;
      rd_v1 <= e_go;
      rd_v2 <= rd_v1;
    end
    rd_rot1 <= e_m1[2:0];
    rd_rot2 <= rd_rot1;
  end

  // the word: pixel q of the 8 is in bank (m1 + q) mod 8
  logic [BW-1:0] yw_in;
  always_comb
    for (int q = 0; q < E; q++)
      yw_in[q * EW +: EW] = t_rdata[3'(rd_rot2 + 3'(q)) * EW +: EW];

  cv_fifo #(.W(BW), .D(32), .BRAM(1'b0)) u_yq (
    .clk, .rst, .in_valid (rd_v2), .in_ready (yq_in_ready), .in_data (yw_in),
    .out_valid (yw_valid), .out_ready (yw_ready), .out_data (yw_data), .count (yq_n)
  );
  cv_fifo #(.W(41), .D(4), .BRAM(1'b0)) u_rq (
    .clk, .rst, .in_valid (e_runpush), .in_ready (rq_in_ready), .in_data ({e_e, es.len}),
    .out_valid (run_valid), .out_ready (run_ready), .out_data ({run_e, run_len}), .count (rq_n)
  );

  assign idle = !f_act && !fr_d[1].v && !fr_d[2].v && !fr_d[3].v && !fr_d[4].v && !fr_d[5].v &&
                !hi_pend && !sq_valid &&
                !e_act && !rd_v1 && !rd_v2 && (full == '0);

  logic unused;
  assign unused = sq_in_ready ^ yq_in_ready ^ ^sq_n ^ ^rq_n ^ ^f_ycb ^ ^j ^ ^hi_fr;

endmodule
