// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_xpose — the TRANSPOSE op of the RowOpsKernel (ROWOPS_RTL_KERNEL.md §1.4):
// tiles of R (<= 8) rows x L (<= 1024) int16 elements become L granules of R
// elements, out[d][r] = in[r][d].
//
//   LOAD  the tile's R rows (A words, rows one after the other: the A geometry
//         walks blocks / tiles / rows) into eight row banks
//   EMIT  one granule per cycle: element d of every row bank
//     contiguous (g_stride == 2R): the granules form one run of L*R elements
//         per tile, packed 8 per word for the write port's regular geometry
//         (tiles at c_s1 / c_s2, the last word's lanes past L*R zero)
//     scattered: granule d at byte address tile + d * g_stride, written with
//         byte strobes as one or two beats (when it straddles a word) through
//         the write port's external burst input (one burst per granule)
// One tile at a time (the next tile loads after the emit; the A port keeps
// prefetching its words).
// ---------------------------------------------------------------------------
module ro_xpose
  import ro_pkg::*;
(
  input  logic          clk,
  input  logic          rst,
  input  logic          start,
  input  logic          en,
  input  logic [31:0]   L,                // elements per row
  input  logic [3:0]    R,                // rows per tile
  input  logic [31:0]   c1,               // tiles per block
  input  logic [31:0]   c2,               // blocks
  input  logic [63:0]   c_base,
  input  logic [31:0]   c_s1,
  input  logic [31:0]   c_s2,
  input  logic [31:0]   g_stride,         // bytes between granules (b_s2)

  input  logic          a_valid,
  output logic          a_ready,
  input  logic [BW-1:0] a_data,

  output logic          c_valid,          // beats, in burst order
  input  logic          c_ready,
  output logic [BW-1:0] c_data,
  output logic [BW/8-1:0] c_strb,

  output logic          ext_en,           // scattered mode: bursts from here
  output logic          ext_valid,
  input  logic          ext_ready,
  output logic [59:0]   ext_waddr,
  output logic [8:0]    ext_len,
  output logic          ext_done,

  output logic          idle
);
  localparam int TW = 128;                // words per row bank (L <= 1024)

  // Job constants ---------------------------------------------------------------------------
  logic [31:0] k_L, k_c1, k_c2, k_cs1, k_cs2, k_gs;
  logic [63:0] k_cb;
  logic [3:0]  k_R;
  logic [7:0]  k_rw;                      // words per row (ceil(L / 8))
  logic        k_contig;
  always_ff @(posedge clk)
    if (start) begin
      k_L <= L; k_R <= R; k_c1 <= c1; k_c2 <= c2; k_cb <= c_base; k_cs1 <= c_s1; k_cs2 <= c_s2; k_gs <= g_stride;
      k_rw <= 8'((L + 32'd7) >> 3);
      k_contig <= (g_stride == {27'b0, R, 1'b0});
      k_c1m1 <= c1 - 32'd1;
      k_mask <= (32'd1 << {R, 1'b0}) - 32'd1;       // the granule's 2R bytes
    end
  logic [31:0] k_mask;
  // the step from a block's last tile to the next block: c_s2 - c_s1 * (c1 - 1), in
  // registered stages after start (c1 - 1 < 2^16; the first tile takes longer than that)
  logic [31:0] k_c1m1, sp_m0;
  logic [31:0] sp_m1, sp_m2;
  logic [47:0] k_span;
  logic [63:0] k_delta;
  always_ff @(posedge clk) begin
    sp_m0   <= k_cs1[15:0] * k_c1m1[15:0];
    sp_m1   <= k_cs1[31:16] * k_c1m1[15:0];
    sp_m2   <= k_cs1[15:0] * k_c1m1[31:16];
    k_span  <= 48'(sp_m0) + (48'(sp_m1) << 16) + (48'(sp_m2) << 16);
    k_delta <= 64'(k_cs2) - 64'(k_span);
  end
  assign ext_en = !k_contig;

  // Tile buffer: row r's word w in RAM r (distributed: 8 asynchronous read ports, one per row)
  logic [BW-1:0] tb_q [R_MAX];             // row r's word d[9:3]
  (* max_fanout = 64 *) logic          tb_we;
  (* max_fanout = 64 *) logic [3:0]    lr;  // row being loaded
  (* max_fanout = 64 *) logic [7:0]    lw;  // word being loaded
  logic [31:0]   d;                         // granule being emitted
  for (genvar r = 0; r < R_MAX; r++) begin : g_tb
    (* ram_style = "distributed" *) logic [BW-1:0] mem [TW];
    always_ff @(posedge clk) if (tb_we && lr[2:0] == 3'(r)) mem[lw[6:0]] <= a_data;
    assign tb_q[r] = mem[d[9:3]];
  end

  // Tile sequencing ------------------------------------------------------------------------
  typedef enum logic [1:0] {X_IDLE, X_LOAD, X_EMIT, X_FLUSH} xst_t;
  xst_t        xs;
  logic [31:0] i1, i2;                    // the tile within its block / the block
  logic [63:0] tile_addr, gran_addr;      // scattered: the tile's / the granule's byte address
  logic        i1_last, i2_last;          // the tile is its block's last / the block the job's last
  logic        last_tile;
  logic        issue, room, g0_v;
  assign last_tile = i1_last && i2_last;
  assign a_ready   = (xs == X_LOAD);
  assign tb_we     = (xs == X_LOAD) && a_valid;
  // scattered granules may take two beats: one granule every other cycle there
  assign issue     = (xs == X_EMIT) && room && (k_contig || !g0_v);

  always_ff @(posedge clk) begin
    if (rst) begin
      xs <= X_IDLE;
    end else if (start) begin
      xs <= (en && L != '0 && R != '0 && c1 != '0 && c2 != '0) ? X_LOAD : X_IDLE;
      i1 <= '0; i2 <= '0; lr <= '0; lw <= '0; d <= '0;
      i1_last <= (c1 == 32'd1);
      i2_last <= (c2 == 32'd1);
      tile_addr <= c_base;
      gran_addr <= c_base;
    end else begin
      case (xs)
        X_LOAD: if (a_valid && a_ready) begin
          if (lw + 8'd1 == k_rw) begin
            lw <= '0;
            if (lr + 4'd1 == k_R) begin
              lr <= '0;
              xs <= X_EMIT;
              d  <= '0;
              gran_addr <= tile_addr;
            end else begin
              lr <= lr + 4'd1;
            end
          end else begin
            lw <= lw + 8'd1;
          end
        end
        X_EMIT: if (issue) begin
          gran_addr <= gran_addr + 64'(k_gs);
          if (d + 32'd1 == k_L) begin
            d  <= '0;
            xs <= X_FLUSH;
          end else begin
            d <= d + 32'd1;
          end
        end
        X_FLUSH: begin                              // the next tile's address; stop after the last
          tile_addr <= tile_addr + (i1_last ? k_delta : 64'(k_cs1));
          if (i1_last) begin
            i1      <= '0;
            i1_last <= (k_c1 == 32'd1);
            i2      <= i2 + 32'd1;
            i2_last <= (i2 + 32'd2 == k_c2);
          end else begin
            i1      <= i1 + 32'd1;
            i1_last <= (i1 + 32'd2 == k_c1);
          end
          xs <= last_tile ? X_IDLE : X_LOAD;
        end
        default: xs <= X_IDLE;
      endcase
    end
  end

  // Granule read (stage 0) ------------------------------------------------------------------
  logic          g0_last;                           // the tile's last granule
  logic [127:0]  g0;                                // the R elements (lanes r), the rest 0
  logic [63:0]   g0_addr;
  always_ff @(posedge clk) begin
    if (rst || start) g0_v <= 1'b0;
    else g0_v <= issue;
    g0_last <= (d + 32'd1 == k_L);
    g0_addr <= gran_addr;
    for (int r = 0; r < R_MAX; r++)
      g0[EW*r +: EW] <= (r < int'(k_R)) ? tb_q[r][EW*d[2:0] +: EW] : 16'h0;
  end

  // Contiguous packer: R elements per granule into words of 8 ------------------------------
  logic [255:0]  acc;                               // up to 15 pending elements
  logic [3:0]    acc_n;
  logic          pk_v;
  logic [BW-1:0] pk_d;
  always_ff @(posedge clk) begin
    if (rst || start) begin
      acc   <= '0;                                 // a job starts with an empty accumulator
      acc_n <= '0;
      pk_v  <= 1'b0;
    end else begin
      pk_v <= 1'b0;
      if (g0_v && k_contig) begin
        logic [255:0] a;
        logic [4:0]   n;
        a = acc | (256'(g0[127:0]) << (16 * acc_n));
        n = {1'b0, acc_n} + {1'b0, k_R};
        if (n >= 5'd8) begin
          pk_v <= 1'b1;
          pk_d <= a[127:0];
          acc  <= a >> 128;
          acc_n <= 4'(n - 5'd8);
          if (g0_last && (n - 5'd8) != '0) begin  // the tile's last granule: flush follows
            acc_n <= '0;
            acc   <= '0;
          end
        end else if (g0_last) begin                // a partial word ends the tile
          pk_v <= 1'b1;
          pk_d <= a[127:0];
          acc  <= '0;
          acc_n <= '0;
        end else begin
          acc  <= a;
          acc_n <= n[3:0];
        end
      end
    end
  end
  // a tile whose last granule overflowed a word: the overflow goes out one cycle
  // after that word (no granule follows within a cycle: the next tile loads first)
  logic          fl_p, fl_v;
  logic [BW-1:0] fl_d;
  always_ff @(posedge clk) begin
    if (rst || start) begin
      fl_p <= 1'b0;
      fl_v <= 1'b0;
    end else begin
      fl_p <= 1'b0;
      fl_v <= fl_p;
      if (g0_v && k_contig && g0_last) begin
        logic [255:0] a;
        logic [4:0]   n;
        a = acc | (256'(g0[127:0]) << (16 * acc_n));
        n = {1'b0, acc_n} + {1'b0, k_R};
        if (n > 5'd8) begin
          fl_p <= 1'b1;
          fl_d <= a[255:128];
        end
      end
    end
  end

  // Scattered beats: one or two per granule, then the burst descriptor --------------------
  // s1: the beats of the granule g0 (data / strobes), s2.. a small queue to the port
  logic          sc_v, sc_two, sc_split;             // split: the two beats straddle 4 KiB
  logic [BW-1:0] sc_d0, sc_d1;
  logic [15:0]   sc_s0, sc_s1;
  logic [59:0]   sc_wa;
  always_ff @(posedge clk) begin
    if (rst || start) sc_v <= 1'b0;
    else begin
      sc_v <= g0_v && !k_contig;
      begin
        logic [4:0]   o;                            // byte offset in the word
        logic [5:0]   nb;                           // bytes of the granule
        logic [255:0] dd;
        logic [31:0]  ss;
        o  = {1'b0, g0_addr[3:0]};
        nb = {1'b0, k_R, 1'b0};
        dd = 256'(g0) << (8 * o);
        ss = k_mask << o;
        sc_d0  <= dd[127:0];
        sc_d1  <= dd[255:128];
        sc_s0  <= ss[15:0];
        sc_s1  <= ss[31:16];
        sc_two   <= (o + nb) > 6'd16;
        sc_split <= ((o + nb) > 6'd16) && (g0_addr[11:4] == 8'hFF);
        sc_wa    <= g0_addr[63:4];
      end
    end
  end

  // Output queues: beats (data + strobe) and descriptors; credits bound the groups in flight
  localparam int Q_D = 32;
  localparam int QW  = $clog2(Q_D);
  logic          bq_in_valid, bq_in_ready, dq_in_valid, dq_in_ready;
  logic [BW+15:0] bq_in;
  logic [68:0]   dq_in;
  logic [QW:0]   bq_count, dq_count, inflight;

  // the beat queue takes one beat per cycle: the second beat of a straddling granule
  // goes in the next cycle (sc_pend)
  logic          sc_pend, sc_pend_split;
  logic [BW-1:0] sc_pd;
  logic [15:0]   sc_ps;
  logic [59:0]   sc_pwa;
  always_ff @(posedge clk) begin
    if (rst || start) sc_pend <= 1'b0;
    else if (sc_v && sc_two) begin
      sc_pend <= 1'b1; sc_pd <= sc_d1; sc_ps <= sc_s1; sc_pend_split <= sc_split; sc_pwa <= sc_wa + 60'd1;
    end else begin
      sc_pend <= 1'b0;
    end
  end
  assign bq_in_valid = k_contig ? (pk_v || fl_v) : (sc_v || sc_pend);
  assign bq_in       = k_contig ? {16'hFFFF, fl_v ? fl_d : pk_d}
                                : (sc_pend ? {sc_ps, sc_pd} : {sc_s0, sc_d0});
  // one descriptor per granule (len 1 or 2), or two of len 1 when it straddles 4 KiB
  assign dq_in_valid = sc_v || (sc_pend && sc_pend_split);
  assign dq_in       = sc_v ? {sc_wa, 8'd0, sc_two && !sc_split} : {sc_pwa, 8'd0, 1'b0};

  ro_fifo #(.W(BW + 16), .D(Q_D), .BRAM(1'b0)) u_bq (
    .clk, .rst (rst || start),
    .in_valid (bq_in_valid), .in_ready (bq_in_ready), .in_data (bq_in),
    .out_valid (c_valid), .out_ready (c_ready), .out_data ({c_strb, c_data}),
    .count (bq_count)
  );
  logic [68:0] dq_out;
  ro_fifo #(.W(69), .D(Q_D), .BRAM(1'b0)) u_dq (
    .clk, .rst (rst || start),
    .in_valid (dq_in_valid), .in_ready (dq_in_ready), .in_data (dq_in),
    .out_valid (ext_valid), .out_ready (ext_ready), .out_data (dq_out),
    .count (dq_count)
  );
  assign ext_waddr = dq_out[68:9];
  assign ext_len   = dq_out[0] ? 9'd2 : 9'd1;

  // credits: a granule may occupy up to two beat entries and one descriptor entry
  always_ff @(posedge clk) begin
    if (rst || start) inflight <= '0;
    else inflight <= inflight + (QW+1)'(issue) - (QW+1)'(g0_v);
  end
  assign room = ({1'b0, bq_count} + {1'b0, inflight} + {1'b0, inflight}) < (QW+2)'(Q_D - 6) &&
                ({1'b0, dq_count} + {1'b0, inflight}) < (QW+2)'(Q_D - 4);

  assign ext_done = (xs == X_IDLE) && !g0_v && !sc_v && !sc_pend && (dq_count == '0);
  assign idle     = (xs == X_IDLE) && !g0_v && !pk_v && !fl_p && !fl_v && !sc_v && !sc_pend &&
                    (bq_count == '0) && (dq_count == '0) && (acc_n == '0);

  logic unused;
  assign unused = ^{bq_in_ready, dq_in_ready, k_cb, d[31:10], lr[3], lw[7]};

endmodule
