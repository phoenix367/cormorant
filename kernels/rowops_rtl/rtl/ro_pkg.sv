// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// ro_pkg — shared parameters and types of the RowOpsKernel RTL
// (doc/kernels/ROWOPS_RTL_KERNEL.md).  The port modules (ro_fifo, ro_rs,
// ro_burstgen, ro_rd_port, ro_wr_port) are the VectorOPKernel's, so the
// geometry type and the burst constants keep their names.
// ---------------------------------------------------------------------------
package ro_pkg;
  localparam int E         = 8;         // int16 lanes per 128-bit AXI word
  localparam int EW        = 16;        // element width (int16); int32 / f32 elements take two lanes
  localparam int BW        = E * EW;    // word width (128)
  localparam int WA        = 44;        // word-address bits of the address arithmetic: byte addresses
                                         // below 2^48 (the PS ports' 40-bit physical space, SMMU IOVAs)
  localparam int E32       = 4;         // int32 elements per word

  // Burst lengths and outstanding counts the IP declares (syn/package_ip.tcl).
  localparam int RD_BURST  = 64;        // max beats per AR burst
  localparam int RD_OUTS   = 16;        // AR bursts awaiting their data, per read port
  localparam int RD_FIFO_D = 256;       // beats buffered per read port (one BRAM36)
  localparam int WR_BURST  = 64;        // max beats per AW burst
  localparam int WR_FIFO_D = 256;       // beats buffered before an AW is issued
  localparam int WR_OUTS   = 16;        // AW bursts awaiting W / B
  localparam int REP_D     = 16;        // the read port's replay RAM (never enabled here)
  localparam int OUT_D     = 8;         // read port output FIFO

  localparam int N_MAX     = 4096;      // elements per row (the parameter RAM, the norm ring)
  localparam int PB        = 4;         // arithmetic lanes = parameter banks (column c in bank c % 4)
  localparam int PIW       = 10;        // parameter index width: N_MAX / PB entries per bank
  // The table image of OP_TLOAD (ROWOPS_RTL_KERNEL.md §1.4): 4096 words — 2048 of two
  // sigmoid entries {D, T} (49 bits in each 64-bit half), then 2048 of four rsqrt entries.
  localparam int TBL_WORDS = 4096;
  localparam int SIG_W     = 49;
  localparam int HD_MAX    = 128;       // ROPE head segment
  localparam int R_MAX     = 8;         // TRANSPOSE rows per tile
  // Descriptor chains (ROWOPS_RTL_KERNEL.md §1.6): chain_n jobs (at most CHAIN_MAX; more
  // are clamped) read from DDR at `chain` before the first; a descriptor is DESC_BEATS
  // words of 16 bytes, of which the first DESC_USED hold the 24 argument registers
  // 0x10 .. 0x6C in register order (the rest is ignored).
  localparam int CHAIN_MAX  = 16;
  localparam int DESC_BEATS = 8;
  localparam int DESC_USED  = 6;

  // Op codes (register op[3:0]) and kinds (op[5:4] A, [7:6] B, [9:8] C).
  localparam logic [3:0] OP_COPY = 4'd1, OP_ADDX = 4'd2, OP_DEQUANT = 4'd3, OP_MULACT = 4'd4,
                         OP_NORM = 4'd5, OP_ROPE = 4'd6, OP_XPOSE = 4'd7, OP_TLOAD = 4'd8,
                         OP_FOLD = 4'd9, OP_GATE = 4'd10;
  localparam logic [1:0] K_I16 = 2'd0, K_I32 = 2'd1, K_F32 = 2'd2;
  // Flags (register op bits).
  localparam int F_B_EN = 12, F_P_BCAST = 13, F_P_KEEP = 14, F_RELU = 15, F_NORM_LN = 16, F_SCALE = 17;

  // How one stream walks DDR during a job (the VectorOPKernel's geometry with
  // one more level): n_blk blocks, block b at word base + b * blk_w, each of
  // n_tiles tiles at + t * tile_w (1 tile, tile_w 0 for every op but
  // TRANSPOSE), each of n_runs runs of run_words words, run r at the tile's
  // base + r * stride_w; the last word
  // of every run has `tail` valid lanes (1..8).  replay / reps are the
  // VectorOPKernel's stride-0 replay, never enabled by this kernel.  Runs
  // (rows) with an index below win_lo or at / above win_hi are not read: the
  // port delivers their words as zeros (COPY / ADDX's window, §1.4).
  typedef struct packed {
    logic        en;
    logic [59:0] base_w;      // word address of the first run
    logic [31:0] n_blk;
    logic [31:0] blk_w;
    logic [31:0] n_tiles;
    logic [31:0] tile_w;
    logic [31:0] n_runs;
    logic [31:0] run_words;
    logic [31:0] stride_w;
    logic [3:0]  tail;
    logic        replay;
    logic [31:0] reps;
    logic [31:0] win_lo;
    logic [31:0] win_hi;
  } geom_t;

  // One column's parameters (ROWOPS_RTL_KERNEL.md §1.3): 72 bits in the RAM.
  typedef struct packed {
    logic [5:0]         sh_c;
    logic [5:0]         sh_b;
    logic [5:0]         sh_a;
    logic signed [21:0] bq;       // Bq'
    logic signed [31:0] g;        // G
  } param_t;
  localparam int PRM_W = $bits(param_t);   // 72

  function automatic logic [15:0] sat16(input logic signed [63:0] x);
    if (x > 64'sd32767)  return 16'h7FFF;
    if (x < -64'sd32768) return 16'h8000;
    return x[15:0];
  endfunction

  function automatic logic [31:0] sat32(input logic signed [63:0] x);
    if (x > 64'sd2147483647)  return 32'h7FFFFFFF;
    if (x < -64'sd2147483648) return 32'h80000000;
    return x[31:0];
  endfunction

  // round half to even of x / 2^k, k in 0..63 (x signed 64-bit), in two
  // pipeline stages: rne_a shifts and keeps the round bit and the sticky
  // bit (any lower bit set), rne_b adds the rounding.  Equal to rne_shr.
  typedef struct packed {
    logic signed [63:0] q;        // floor(x / 2^k)
    logic               rb;       // bit k - 1 of x
    logic               st;       // bits k - 2 .. 0 of x not all zero
  } rne_t;

  // rne_a shifts z = 2x right by k in three 4:1 steps (16 k[5:4], 4 k[3:2],
  // k[1:0]): y = floor(2x / 2^k), so y[64:1] = floor(x / 2^k) and y[0] = bit
  // k - 1 of x (0 for k = 0); the bits the steps drop are {x[k-2:0], 0}, the
  // sticky bit — no mask arithmetic, no bypass for k = 0.
  function automatic rne_t rne_a(input logic signed [63:0] x, input logic [5:0] k);
    rne_t r;
    logic signed [64:0] z0, z1, z2, z3;
    logic               s1, s2, s3;
    z0 = {x, 1'b0};
    case (k[5:4])
      2'd0:    z1 = z0;
      2'd1:    z1 = z0 >>> 16;
      2'd2:    z1 = z0 >>> 32;
      default: z1 = z0 >>> 48;
    endcase
    s1 = |(z0[47:0] & {{16{k[5:4] == 2'd3}}, {16{k[5]}}, {16{k[5:4] != 2'd0}}});
    case (k[3:2])
      2'd0:    z2 = z1;
      2'd1:    z2 = z1 >>> 4;
      2'd2:    z2 = z1 >>> 8;
      default: z2 = z1 >>> 12;
    endcase
    s2 = |(z1[11:0] & {{4{k[3:2] == 2'd3}}, {4{k[3]}}, {4{k[3:2] != 2'd0}}});
    case (k[1:0])
      2'd0:    z3 = z2;
      2'd1:    z3 = z2 >>> 1;
      2'd2:    z3 = z2 >>> 2;
      default: z3 = z2 >>> 3;
    endcase
    s3 = |(z2[2:0] & {k[1:0] == 2'd3, k[1], k[1:0] != 2'd0});
    r.q  = z3[64:1];
    r.rb = z3[0];
    r.st = s1 | s2 | s3;
    return r;
  endfunction

  function automatic logic signed [63:0] rne_b(input rne_t r);
    return r.q + (64'(r.rb && (r.st || r.q[0])));
  endfunction

  // round half to even of x / 2^k, k in 0..63 (x signed 64-bit); the result
  // of a right shift of a two's complement value is the floor.
  function automatic logic signed [63:0] rne_shr(input logic signed [63:0] x, input logic [5:0] k);
    logic signed [63:0] q;
    logic [63:0]        r, half;
    logic               up;
    if (k == 6'd0) return x;
    q    = x >>> k;
    r    = x - (q <<< k);
    half = 64'd1 << (k - 6'd1);
    up   = (r > half) || ((r == half) && q[0]);
    return q + (up ? 64'sd1 : 64'sd0);
  endfunction

endpackage
