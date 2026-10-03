// ---------------------------------------------------------------------------
// mm_rungen — expands step descriptors into the run descriptors of one read
// port / MAC lane (P = 0 or 1).
//
// Per step:
//   * first chunk of a panel: one A run — panel rows [P*RH, P*RH + RH) as a
//     single contiguous element range (rows of k) — or an A marker when this
//     port has no rows, the A panel is reused, or k == 0;
//   * the B runs of lane P for the chunk's columns, over the lane's K blocks
//     (blocks of 16 planes, block b belongs to lane b % 2):
//       packed B        one run per (tile of 32 columns, block): 16 rows x 32
//       image, 1 chunk  one run per block: 16 planes x (m << lk) elements
//       image, chunked  one run per plane: 1 row of (mcc << lk) elements
//     or one B marker when the lane has no K blocks.
//
// Every run goes to the gearbox queue; data runs also to the read engine,
// B runs and markers also to the x prefetcher.
// ---------------------------------------------------------------------------
module mm_rungen
  import mm_pkg::*;
#(
  parameter int P = 0
) (
  input  logic     clk,
  input  logic     rst,
  input  logic     start,
  input  cfg_t     cfg,

  input  logic     step_valid,
  output logic     step_ready,
  input  step_t    step,

  output logic     gb_valid,
  input  logic     gb_ready,
  output gb_run_t  gb_run,

  output logic     rd_valid,
  input  logic     rd_ready,
  output rd_run_t  rd_run,

  output logic     x_valid,
  input  logic     x_ready,
  output x_run_t   x_run,

  output logic     idle
);

  typedef enum logic [2:0] {S_IDLE, S_A, S_BINIT, S_BINIT2, S_B, S_BM} st_t;
  st_t   state;
  step_t st;

  logic [8:0]  blk;          // current K block
  logic [4:0]  j;            // plane within the block (chunked image mode)
  logic [4:0]  t, tiles_n;   // tile within the chunk (packed mode)
  logic [63:0] addr_run, addr_tile, addr_blk, pk_base, base_q;

  // Planes of the current block.
  logic [12:0] blk_left;
  logic [4:0]  blk_rows;
  logic        last_blk;
  assign blk_left = cfg.planes - {blk, 4'b0};
  assign blk_rows = (blk_left >= 13'd16) ? 5'd16 : blk_left[4:0];
  assign last_blk = ({1'b0, blk} + 10'd2 >= {1'b0, cfg.nblk});

  // A run geometry of this port.
  localparam logic [3:0] R0 = 4'(P * RH);
  logic [3:0]  a_rows;
  logic        a_marker;
  logic [63:0] a_addr;
  always_comb begin
    if (st.n_valid <= R0)            a_rows = 4'd0;
    else if (st.n_valid - R0 >= 4'(RH)) a_rows = 4'(RH);
    else                             a_rows = st.n_valid - R0;
    a_marker = !st.a_load || (a_rows == 4'd0) || (cfg.k == '0);
    // rows P*RH.. start P * RH * k elements into the panel
    a_addr   = st.a_addr + ({51'b0, cfg.k} << 1) * 64'(P * RH);
  end

  // Candidate run -------------------------------------------------------------------
  logic        cand;
  logic        c_dest_b, c_marker, c_init, c_last, c_plast;
  logic [63:0] c_addr;
  logic [12:0] c_len;
  logic [4:0]  c_rows;
  logic [5:0]  c_wb;
  logic [10:0] c_cl0;
  logic        c_bdone;       // this is the step's last B run

  always_comb begin
    cand     = 1'b0;
    c_dest_b = 1'b1;
    c_marker = 1'b0;
    c_addr   = addr_run;
    c_len    = 13'd32;
    c_rows   = blk_rows;
    c_wb     = 6'd0;
    c_cl0    = {blk[7:1], 4'b0};
    c_init   = (blk == 9'(P));
    c_bdone  = last_blk;
    case (state)
      S_A: begin
        cand     = 1'b1;
        c_dest_b = 1'b0;
        c_marker = a_marker;
        c_addr   = a_addr;
        c_len    = cfg.k;
        c_rows   = {1'b0, a_rows};
        c_bdone  = 1'b0;
      end
      S_BM: begin
        cand     = 1'b1;
        c_marker = 1'b1;
        c_rows   = 5'd0;
        c_bdone  = 1'b1;
      end
      S_B: begin
        cand = 1'b1;
        if (cfg.pk) begin
          c_len   = 13'd32;
          c_wb    = {t[3:0], 2'b00};
          c_bdone = last_blk && (t + 5'd1 == tiles_n);
        end else if (cfg.contig) begin
          c_len   = cfg.lfull[12:0];
        end else begin
          c_len   = 13'({st.mcc, 3'b0} >> (2'd3 - cfg.lk));   // mcc << lk
          c_rows  = 5'd1;
          c_cl0   = {blk[7:1], 4'b0} + {6'b0, j};
          c_init  = (blk == 9'(P)) && (j == 5'd0);
          c_bdone = last_blk && (j + 5'd1 == blk_rows);
        end
      end
      default: ;
    endcase
    c_last  = c_bdone;
    c_plast = c_bdone && st.panel_last;
  end

  // Elements of the run, without a general multiplier: A runs have <= 4
  // rows, B runs 16 rows except the last block (elems_last, set in S_BINIT).
  logic [17:0] elems_last;
  logic [17:0] c_elems;
  always_comb begin
    case (state)
      S_A:  c_elems = ({3'b0, cfg.k, 2'b0} & {18{a_rows[2]}}) +
                      ({4'b0, cfg.k, 1'b0} & {18{a_rows[1]}}) +
                      ({5'b0, cfg.k}       & {18{a_rows[0]}});
      S_B:  if (cfg.pk)          c_elems = {8'b0, blk_rows, 5'b0};
            else if (cfg.contig) c_elems = (blk_rows == 5'd16) ? {1'b0, cfg.lfull[12:0], 4'b0}
                                                               : elems_last;
            else                 c_elems = {5'b0, c_len};
      default: c_elems = '0;
    endcase
  end

  // Two register stages: s1 = candidate, o = derived queue descriptors.
  logic        s1_v, s1_dest_b, s1_marker, s1_init, s1_last, s1_plast;
  logic [63:0] s1_addr;
  logic [12:0] s1_len;
  logic [4:0]  s1_rows;
  logic [5:0]  s1_wb;
  logic [10:0] s1_cl0;
  logic [17:0] s1_elems;

  logic [11:0] s1_nw;
  assign s1_nw = 12'((s1_elems + 18'(s1_addr[3:1]) + 18'd7) >> 3);

  logic    o_valid, o_rd, o_x;
  gb_run_t o_gb;
  rd_run_t o_rdr;
  x_run_t  o_xr;
  logic    push, adv, adv1;

  assign push = o_valid && gb_ready && (!o_rd || rd_ready) && (!o_x || x_ready);
  assign adv  = !o_valid || push;
  assign adv1 = !s1_v || adv;

  assign gb_valid = o_valid && (!o_rd || rd_ready) && (!o_x || x_ready);
  assign rd_valid = o_valid && o_rd && gb_ready && (!o_x || x_ready);
  assign x_valid  = o_valid && o_x && gb_ready && (!o_rd || rd_ready);
  assign gb_run   = o_gb;
  assign rd_run   = o_rdr;
  assign x_run    = o_xr;

  logic emit;
  assign emit       = cand && adv1;
  assign step_ready = (state == S_IDLE);

  always_ff @(posedge clk) begin
    if (rst || start) begin
      state   <= S_IDLE;
      s1_v    <= 1'b0;
      o_valid <= 1'b0;
    end else begin
      if (adv) begin
        o_valid <= s1_v;
        if (s1_v) begin
          o_gb.dest_b     <= s1_dest_b;
          o_gb.marker     <= s1_marker;
          o_gb.s          <= s1_addr[3:1];
          o_gb.len        <= s1_len;
          o_gb.rows       <= s1_marker ? 5'd0 : s1_rows;
          o_gb.nw         <= s1_marker ? 12'd0 : s1_nw;
          o_gb.wb         <= s1_wb;
          o_gb.init       <= s1_init;
          o_gb.step_last  <= s1_last;
          o_gb.panel_last <= s1_plast;
          o_rd            <= !s1_marker;
          o_rdr.waddr     <= s1_addr[63:4];
          o_rdr.nw        <= s1_nw;
          o_x             <= s1_dest_b;
          o_xr.marker     <= s1_marker;
          o_xr.rows       <= s1_marker ? 5'd0 : s1_rows;
          o_xr.cl0        <= s1_cl0;
          o_xr.panel_last <= s1_plast;
        end
      end
      if (adv1) begin
        s1_v <= cand;
        if (cand) begin
          s1_dest_b <= c_dest_b;
          s1_marker <= c_marker;
          s1_addr   <= c_addr;
          s1_len    <= c_len;
          s1_rows   <= c_rows;
          s1_wb     <= c_wb;
          s1_cl0    <= c_cl0;
          s1_init   <= c_init;
          s1_last   <= c_last;
          s1_plast  <= c_plast;
          s1_elems  <= c_elems;
        end
      end

      case (state)
        S_IDLE: if (step_valid) begin
          st    <= step;
          state <= step.panel_first ? S_A : S_BINIT;
        end
        S_A: if (emit) state <= S_BINIT;
        S_BINIT: begin
          blk     <= 9'(P);
          j       <= '0;
          t       <= '0;
          tiles_n <= 5'((11'(st.mcc) + 11'd31) >> 5);
          elems_last <= 18'(cfg.planes[3:0]) * 18'(cfg.lfull[9:0]);
          // Base of the chunk in B.  Packed: chunk c0 starts at tile c0 / 32
          // = 16 * chunk index, 16 tiles of 32 x k elements per chunk.  Image:
          // column c0 of plane 0 (c0 == 0 for a single chunk).
          if (cfg.pk) base_q <= (st.c0 == '0) ? st.b_addr : pk_base + ({51'b0, cfg.k} << 10);
          else        base_q <= st.b_addr + ({32'b0, st.c0} << ({2'b0, cfg.lk} + 4'd1));
          state <= S_BINIT2;
        end
        S_BINIT2: begin
          // plus this lane's first K block: block P (16 rows of 32 or 16 planes)
          logic [63:0] first;
          first = base_q + (cfg.pk ? 64'(P * 1024)
                                   : ((P != 0) ? {23'b0, cfg.lfull, 5'b0} : 64'd0));
          if (cfg.pk) pk_base <= base_q;
          addr_tile <= first;
          addr_blk  <= first;
          addr_run  <= first;
          state     <= (cfg.nblk <= 9'(P)) ? S_BM : S_B;
        end
        S_BM: if (emit) state <= S_IDLE;
        S_B: if (emit) begin
          if (c_bdone) state <= S_IDLE;
          if (cfg.pk) begin
            if (!last_blk) begin
              blk      <= blk + 9'd2;
              addr_run <= addr_run + 64'd2048;          // two blocks of 16 x 32
            end else begin
              blk       <= 9'(P);
              t         <= t + 5'd1;
              addr_tile <= addr_tile + ({51'b0, cfg.k} << 6);   // next tile: 32 x k elements
              addr_run  <= addr_tile + ({51'b0, cfg.k} << 6);
            end
          end else if (cfg.contig) begin
            blk      <= blk + 9'd2;
            addr_run <= addr_run + {22'b0, cfg.lfull, 6'b0};   // 32 planes
          end else begin
            if (j + 5'd1 != blk_rows) begin
              j        <= j + 5'd1;
              addr_run <= addr_run + {27'b0, cfg.lfull, 1'b0};
            end else begin
              j        <= '0;
              blk      <= blk + 9'd2;
              addr_blk <= addr_blk + {22'b0, cfg.lfull, 6'b0};
              addr_run <= addr_blk + {22'b0, cfg.lfull, 6'b0};
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  assign idle = (state == S_IDLE) && !s1_v && !o_valid;

endmodule
