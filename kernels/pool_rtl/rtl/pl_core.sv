// ---------------------------------------------------------------------------
// pl_core — 2-D pooling (MaxPool / AveragePool / LpPool, their Global
// variants) for the Kria KV260, in RTL.  The IP top level is the Verilog
// wrapper PoolingKernel.v (generated from this header by
// scripts/gen_top_wrapper.py); all logic lives here.
//
// Drop-in replacement for the Vitis-HLS PoolingKernel (kernels/pool): same
// module / port names, AXI-Lite register map, DDR access pattern and
// bit-identical results.  See doc/plans/POOL_RTL_PLAN.md.
//
//   ap_start ─► config ─► chunk sequencer ─┬► loader (gmem0) ─► emitter (line buffer) ─► reducer ─┐
//                                          └────────────────────────────────────────► writer (gmem1)
// ---------------------------------------------------------------------------
module pl_core
  import pl_pkg::*;
#(
  parameter int C_S_AXI_CTRL_DATA_WIDTH    = 32,
  parameter int C_S_AXI_CTRL_ADDR_WIDTH    = 8,
  parameter int C_M_AXI_GMEM0_ID_WIDTH     = 1,
  parameter int C_M_AXI_GMEM0_ADDR_WIDTH   = 64,
  parameter int C_M_AXI_GMEM0_DATA_WIDTH   = 128,
  parameter int C_M_AXI_GMEM0_AWUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_ARUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM0_WUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_RUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_BUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM0_USER_VALUE   = 0,
  parameter int C_M_AXI_GMEM0_PROT_VALUE   = 0,
  parameter int C_M_AXI_GMEM0_CACHE_VALUE  = 3,
  parameter int C_M_AXI_GMEM1_ID_WIDTH     = 1,
  parameter int C_M_AXI_GMEM1_ADDR_WIDTH   = 64,
  parameter int C_M_AXI_GMEM1_DATA_WIDTH   = 128,
  parameter int C_M_AXI_GMEM1_AWUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_ARUSER_WIDTH = 1,
  parameter int C_M_AXI_GMEM1_WUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_RUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_BUSER_WIDTH  = 1,
  parameter int C_M_AXI_GMEM1_USER_VALUE   = 0,
  parameter int C_M_AXI_GMEM1_PROT_VALUE   = 0,
  parameter int C_M_AXI_GMEM1_CACHE_VALUE  = 3
) (
  input  logic ap_clk,
  input  logic ap_rst_n,

  // gmem0: x (read only)
  output logic                                  m_axi_gmem0_AWVALID,
  input  logic                                  m_axi_gmem0_AWREADY,
  output logic [C_M_AXI_GMEM0_ADDR_WIDTH-1:0]   m_axi_gmem0_AWADDR,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_AWID,
  output logic [7:0]                            m_axi_gmem0_AWLEN,
  output logic [2:0]                            m_axi_gmem0_AWSIZE,
  output logic [1:0]                            m_axi_gmem0_AWBURST,
  output logic [1:0]                            m_axi_gmem0_AWLOCK,
  output logic [3:0]                            m_axi_gmem0_AWCACHE,
  output logic [2:0]                            m_axi_gmem0_AWPROT,
  output logic [3:0]                            m_axi_gmem0_AWQOS,
  output logic [3:0]                            m_axi_gmem0_AWREGION,
  output logic [C_M_AXI_GMEM0_AWUSER_WIDTH-1:0] m_axi_gmem0_AWUSER,
  output logic                                  m_axi_gmem0_WVALID,
  input  logic                                  m_axi_gmem0_WREADY,
  output logic [C_M_AXI_GMEM0_DATA_WIDTH-1:0]   m_axi_gmem0_WDATA,
  output logic [C_M_AXI_GMEM0_DATA_WIDTH/8-1:0] m_axi_gmem0_WSTRB,
  output logic                                  m_axi_gmem0_WLAST,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_WID,
  output logic [C_M_AXI_GMEM0_WUSER_WIDTH-1:0]  m_axi_gmem0_WUSER,
  output logic                                  m_axi_gmem0_ARVALID,
  input  logic                                  m_axi_gmem0_ARREADY,
  output logic [C_M_AXI_GMEM0_ADDR_WIDTH-1:0]   m_axi_gmem0_ARADDR,
  output logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_ARID,
  output logic [7:0]                            m_axi_gmem0_ARLEN,
  output logic [2:0]                            m_axi_gmem0_ARSIZE,
  output logic [1:0]                            m_axi_gmem0_ARBURST,
  output logic [1:0]                            m_axi_gmem0_ARLOCK,
  output logic [3:0]                            m_axi_gmem0_ARCACHE,
  output logic [2:0]                            m_axi_gmem0_ARPROT,
  output logic [3:0]                            m_axi_gmem0_ARQOS,
  output logic [3:0]                            m_axi_gmem0_ARREGION,
  output logic [C_M_AXI_GMEM0_ARUSER_WIDTH-1:0] m_axi_gmem0_ARUSER,
  input  logic                                  m_axi_gmem0_RVALID,
  output logic                                  m_axi_gmem0_RREADY,
  input  logic [C_M_AXI_GMEM0_DATA_WIDTH-1:0]   m_axi_gmem0_RDATA,
  input  logic                                  m_axi_gmem0_RLAST,
  input  logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_RID,
  input  logic [C_M_AXI_GMEM0_RUSER_WIDTH-1:0]  m_axi_gmem0_RUSER,
  input  logic [1:0]                            m_axi_gmem0_RRESP,
  input  logic                                  m_axi_gmem0_BVALID,
  output logic                                  m_axi_gmem0_BREADY,
  input  logic [1:0]                            m_axi_gmem0_BRESP,
  input  logic [C_M_AXI_GMEM0_ID_WIDTH-1:0]     m_axi_gmem0_BID,
  input  logic [C_M_AXI_GMEM0_BUSER_WIDTH-1:0]  m_axi_gmem0_BUSER,

  // gmem1: y (write only)
  output logic                                  m_axi_gmem1_AWVALID,
  input  logic                                  m_axi_gmem1_AWREADY,
  output logic [C_M_AXI_GMEM1_ADDR_WIDTH-1:0]   m_axi_gmem1_AWADDR,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_AWID,
  output logic [7:0]                            m_axi_gmem1_AWLEN,
  output logic [2:0]                            m_axi_gmem1_AWSIZE,
  output logic [1:0]                            m_axi_gmem1_AWBURST,
  output logic [1:0]                            m_axi_gmem1_AWLOCK,
  output logic [3:0]                            m_axi_gmem1_AWCACHE,
  output logic [2:0]                            m_axi_gmem1_AWPROT,
  output logic [3:0]                            m_axi_gmem1_AWQOS,
  output logic [3:0]                            m_axi_gmem1_AWREGION,
  output logic [C_M_AXI_GMEM1_AWUSER_WIDTH-1:0] m_axi_gmem1_AWUSER,
  output logic                                  m_axi_gmem1_WVALID,
  input  logic                                  m_axi_gmem1_WREADY,
  output logic [C_M_AXI_GMEM1_DATA_WIDTH-1:0]   m_axi_gmem1_WDATA,
  output logic [C_M_AXI_GMEM1_DATA_WIDTH/8-1:0] m_axi_gmem1_WSTRB,
  output logic                                  m_axi_gmem1_WLAST,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_WID,
  output logic [C_M_AXI_GMEM1_WUSER_WIDTH-1:0]  m_axi_gmem1_WUSER,
  output logic                                  m_axi_gmem1_ARVALID,
  input  logic                                  m_axi_gmem1_ARREADY,
  output logic [C_M_AXI_GMEM1_ADDR_WIDTH-1:0]   m_axi_gmem1_ARADDR,
  output logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_ARID,
  output logic [7:0]                            m_axi_gmem1_ARLEN,
  output logic [2:0]                            m_axi_gmem1_ARSIZE,
  output logic [1:0]                            m_axi_gmem1_ARBURST,
  output logic [1:0]                            m_axi_gmem1_ARLOCK,
  output logic [3:0]                            m_axi_gmem1_ARCACHE,
  output logic [2:0]                            m_axi_gmem1_ARPROT,
  output logic [3:0]                            m_axi_gmem1_ARQOS,
  output logic [3:0]                            m_axi_gmem1_ARREGION,
  output logic [C_M_AXI_GMEM1_ARUSER_WIDTH-1:0] m_axi_gmem1_ARUSER,
  input  logic                                  m_axi_gmem1_RVALID,
  output logic                                  m_axi_gmem1_RREADY,
  input  logic [C_M_AXI_GMEM1_DATA_WIDTH-1:0]   m_axi_gmem1_RDATA,
  input  logic                                  m_axi_gmem1_RLAST,
  input  logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_RID,
  input  logic [C_M_AXI_GMEM1_RUSER_WIDTH-1:0]  m_axi_gmem1_RUSER,
  input  logic [1:0]                            m_axi_gmem1_RRESP,
  input  logic                                  m_axi_gmem1_BVALID,
  output logic                                  m_axi_gmem1_BREADY,
  input  logic [1:0]                            m_axi_gmem1_BRESP,
  input  logic [C_M_AXI_GMEM1_ID_WIDTH-1:0]     m_axi_gmem1_BID,
  input  logic [C_M_AXI_GMEM1_BUSER_WIDTH-1:0]  m_axi_gmem1_BUSER,

  // AXI4-Lite control
  input  logic                                  s_axi_ctrl_AWVALID,
  output logic                                  s_axi_ctrl_AWREADY,
  input  logic [C_S_AXI_CTRL_ADDR_WIDTH-1:0]    s_axi_ctrl_AWADDR,
  input  logic                                  s_axi_ctrl_WVALID,
  output logic                                  s_axi_ctrl_WREADY,
  input  logic [C_S_AXI_CTRL_DATA_WIDTH-1:0]    s_axi_ctrl_WDATA,
  input  logic [C_S_AXI_CTRL_DATA_WIDTH/8-1:0]  s_axi_ctrl_WSTRB,
  input  logic                                  s_axi_ctrl_ARVALID,
  output logic                                  s_axi_ctrl_ARREADY,
  input  logic [C_S_AXI_CTRL_ADDR_WIDTH-1:0]    s_axi_ctrl_ARADDR,
  output logic                                  s_axi_ctrl_RVALID,
  input  logic                                  s_axi_ctrl_RREADY,
  output logic [C_S_AXI_CTRL_DATA_WIDTH-1:0]    s_axi_ctrl_RDATA,
  output logic [1:0]                            s_axi_ctrl_RRESP,
  output logic                                  s_axi_ctrl_BVALID,
  input  logic                                  s_axi_ctrl_BREADY,
  output logic [1:0]                            s_axi_ctrl_BRESP,
  output logic                                  interrupt
);
  // The reset, registered: one copy for the control slave, one for the rest.
  // keep: the four kernels' copies are equivalent registers, and the block
  // design's global synthesis would otherwise merge them into one that
  // drives all four kernels across the device (FMAX_250_PLAN).
  logic clk;
  (* keep = "true" *) logic rst;
  (* keep = "true", max_fanout = 128 *) logic rst_c;
  assign clk = ap_clk;
  always_ff @(posedge clk) begin
    rst   <= !ap_rst_n;
    rst_c <= !ap_rst_n;
  end

  // Control slave ------------------------------------------------------------------
  logic        ap_start, ap_done, ap_idle;
  logic [63:0] r_x, r_y;
  logic [31:0] r_batch, r_ch, r_in_h, r_in_w, r_out_h, r_out_w, r_ph, r_pw, r_sh, r_sw,
               r_pt, r_pl, r_dh, r_dw, r_type, r_lp, r_cip;

  pl_ctrl_s_axi u_ctrl (
    .clk, .rst (rst_c),
    .awvalid (s_axi_ctrl_AWVALID), .awready (s_axi_ctrl_AWREADY), .awaddr (s_axi_ctrl_AWADDR),
    .wvalid  (s_axi_ctrl_WVALID),  .wready  (s_axi_ctrl_WREADY),  .wdata  (s_axi_ctrl_WDATA),
    .wstrb   (s_axi_ctrl_WSTRB),
    .bvalid  (s_axi_ctrl_BVALID),  .bready  (s_axi_ctrl_BREADY),  .bresp  (s_axi_ctrl_BRESP),
    .arvalid (s_axi_ctrl_ARVALID), .arready (s_axi_ctrl_ARREADY), .araddr (s_axi_ctrl_ARADDR),
    .rvalid  (s_axi_ctrl_RVALID),  .rready  (s_axi_ctrl_RREADY),  .rdata  (s_axi_ctrl_RDATA),
    .rresp   (s_axi_ctrl_RRESP),
    .interrupt,
    .ap_start, .ap_done, .ap_ready (ap_done), .ap_idle,
    .x (r_x), .y (r_y), .batch (r_batch), .channels (r_ch), .in_h (r_in_h), .in_w (r_in_w),
    .out_h (r_out_h), .out_w (r_out_w), .pool_h (r_ph), .pool_w (r_pw), .stride_h (r_sh),
    .stride_w (r_sw), .pad_top (r_pt), .pad_left (r_pl), .dil_h (r_dh), .dil_w (r_dw),
    .pool_type (r_type), .lp_order (r_lp), .count_include_pad (r_cip)
  );

  // Job configuration ----------------------------------------------------------------
  // PoolingKernel.cpp's compute_pool_geometry: the contract (pool_h, pool_w in
  // 1 .. MAXK, a dilated window of at most LBR rows and LBC columns), the W-tile
  // ow_tile = (LBC - span) / stride_w + 1 (capped to out_w), groups of two
  // positions unless stride_w is a multiple of 8, the rows each chunk reads, and
  // prefetch iff span_h + stride_h <= LBR.  Every product the chunks need is a
  // job constant (one pipelined multiplier: a product issued in cycle i is in
  // mp3 in cycle i + 4).
  typedef enum logic [2:0] {T_IDLE, T_PA, T_DIV, T_PB, T_SEQ, T_WAIT, T_DONE} tst_t;
  tst_t        tstate;
  logic [31:0] q_batch, q_ch, q_in_h, q_in_w, q_out_h, q_out_w, q_ph, q_pw, q_sh, q_sw,
               q_pt, q_pl, q_dh, q_dw, q_type, q_lp, q_cip;
  logic [63:0] q_x, q_y;
  logic [3:0]  ci;
  logic [31:0] ma, mb, ma_q, mb_q, mp1, mp2, mp3;
  logic [31:0] reach_h, reach_w, in_hw, out_hw, ohs, lastw, denom, ch_in, ch_out, owt_sw, owt1_sw;
  logic [5:0]  dv_n, dv_rem, dv_q;          // (LBC - span) / stride_w, restoring
  logic [2:0]  dv_i;
  logic [6:0]  ow_tile;
  logic        job_start, start_q, seq_done, all_idle, ok_job;
  // The job reset, registered and copied per unit group.
  (* keep = "true" *) logic urst, urst_ld, urst_em, urst_rd, urst_wr;
  job_t        j;

  // operands registered (the DSP's input registers), then three product stages:
  // the product of the operands of cycle ci is mp3 in cycle ci + 4
  always_ff @(posedge clk) begin
    ma_q <= ma;
    mb_q <= mb;
    mp1  <= ma_q * mb_q;
    mp2  <= mp1;
    mp3  <= mp2;
  end

  // products, phase A (ci = 0..6, results at ci + 4) and B (ci = 0..3)
  always_comb begin
    ma = '0; mb = '0;
    if (tstate == T_PA)
      case (ci)
        4'd0: begin ma = q_ph - 32'd1;   mb = q_dh;    end
        4'd1: begin ma = q_pw - 32'd1;   mb = q_dw;    end
        4'd2: begin ma = q_in_h;         mb = q_in_w;  end
        4'd3: begin ma = q_out_h;        mb = q_out_w; end
        4'd4: begin ma = q_out_h - 32'd1; mb = q_sh;   end
        4'd5: begin ma = q_out_w - 32'd1; mb = q_sw;   end
        4'd6: begin ma = q_ph;           mb = q_pw;    end
        default: ;
      endcase
    else if (tstate == T_PB)
      case (ci)
        4'd0: begin ma = q_ch;                 mb = in_hw;  end
        4'd1: begin ma = q_ch;                 mb = out_hw; end
        4'd2: begin ma = 32'(ow_tile);         mb = q_sw;   end
        4'd3: begin ma = 32'(ow_tile) - 32'd1; mb = q_sw;   end
        default: ;
      endcase
  end

  // the contract
  logic contract;
  assign contract = (q_ph >= 32'd1) && (q_ph <= 32'(MAXK)) && (q_pw >= 32'd1) && (q_pw <= 32'(MAXK)) &&
                    ((q_ph == 32'd1) || (q_dh < 32'(LBR))) && ((q_pw == 32'd1) || (q_dw < 32'(LBC))) &&
                    (reach_h < 32'(LBR)) && (reach_w < 32'(LBC));

  // Job FSM.  Only the state and job_start are reset; the latched arguments
  // and the derived values below are loaded before every use, so the reset
  // does not reach their enables.
  always_ff @(posedge clk) begin
    if (rst) begin
      tstate    <= T_IDLE;
      job_start <= 1'b0;
    end else begin
      job_start <= 1'b0;
      case (tstate)
        T_IDLE: if (ap_start) tstate <= T_PA;
        T_PA:   if (ci == 4'd10) tstate <= T_DIV;
        T_DIV:  if (dv_i == 3'd0) tstate <= T_PB;
        T_PB:   if (ci == 4'd7) begin
          tstate    <= ok_job ? T_SEQ : T_DONE;
          job_start <= ok_job;
        end
        T_SEQ:  if (seq_done && !job_start) tstate <= T_WAIT;   // (the last job's flag clears now)
        T_WAIT: if (all_idle) tstate <= T_DONE;
        T_DONE: tstate <= T_IDLE;
        default: tstate <= T_IDLE;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    case (tstate)
      T_IDLE: if (ap_start) begin
        {q_x, q_y} <= {r_x, r_y};
        {q_batch, q_ch, q_in_h, q_in_w, q_out_h, q_out_w} <= {r_batch, r_ch, r_in_h, r_in_w, r_out_h, r_out_w};
        {q_ph, q_pw, q_sh, q_sw, q_pt, q_pl, q_dh, q_dw}   <= {r_ph, r_pw, r_sh, r_sw, r_pt, r_pl, r_dh, r_dw};
        {q_type, q_lp, q_cip}                              <= {r_type, r_lp, r_cip};
        ci <= '0;
      end
      T_PA: begin
        ci <= ci + 4'd1;
        case (ci)
          4'd4:  reach_h <= mp3;
          4'd5:  reach_w <= mp3;
          4'd6:  in_hw   <= mp3;
          4'd7:  out_hw  <= mp3;
          4'd8:  ohs     <= mp3;
          4'd9:  lastw   <= mp3;
          4'd10: begin
            denom  <= mp3;
            // the W-tile: (LBC - span) / stride_w + 1 when span <= LBC and stride_w > 0
            dv_n   <= 6'(32'(LBC) - (reach_w + 32'd1));
            dv_rem <= '0;
            dv_q   <= '0;
            dv_i   <= 3'd5;
          end
          default: ;
        endcase
      end
      T_DIV: begin
        logic [6:0] r;
        r = {dv_rem, dv_n[dv_i]};
        if (q_sw <= 32'd63 && 32'(r) >= q_sw) begin
          dv_rem <= 6'(32'(r) - q_sw);
          dv_q[dv_i] <= 1'b1;
        end else begin
          dv_rem <= r[5:0];
        end
        if (dv_i == 3'd0) ci <= '0;
        dv_i <= dv_i - 3'd1;
      end
      T_PB: begin
        if (ci == 4'd0) begin
          // ow_tile: the quotient + 1 (0 -> 1), at most out_w
          logic [31:0] t;
          t = (reach_w + 32'd1 <= 32'(LBC) && q_sw != 32'd0) ? 32'(dv_q) + 32'd1 : 32'd1;
          ow_tile <= 7'((t > q_out_w) ? q_out_w : t);
        end
        ci <= ci + 4'd1;
        case (ci)
          4'd4: ch_in   <= mp3;
          4'd5: ch_out  <= mp3;
          4'd6: owt_sw  <= mp3;
          4'd7: owt1_sw <= mp3;
          default: ;
        endcase
      end
      default: ;
    endcase
  end

  // ow_tile is set in T_PB's first cycle, used by its products 2 and 3 (ci 2, 3).
  assign ok_job = contract && (q_batch != '0) && (q_ch != '0) && (q_out_h != '0) && (q_out_w != '0);

  // job constants (stable from T_SEQ on; reach_h ... ohs are phase A's)
  always_ff @(posedge clk) begin
    if (tstate == T_PB && ci == 4'd7) begin
      logic [31:0] t;
      t = ohs + reach_h - q_pt;            // the last input row any output row needs (signed)
      j.x_w      <= q_x[63:4];
      j.y_w      <= q_y[63:4];
      j.in_h     <= q_in_h;
      j.in_w     <= q_in_w;
      j.out_h    <= q_out_h;
      j.out_w    <= q_out_w;
      j.in_hw    <= in_hw;
      j.out_hw   <= out_hw;
      j.stride_h <= q_sh;
      j.stride_w <= q_sw;
      j.dil_h    <= q_dh;
      j.dil_w    <= q_dw;
      j.pad_top  <= q_pt;
      j.reach_h  <= reach_h;
      j.rows     <= t[31] ? 32'd0 : ((t >= q_in_h) ? q_in_h : t + 32'd1);
      j.pool_h   <= q_ph[2:0];
      j.pool_w   <= q_pw[2:0];
      j.red_len  <= denom[5:0];
      j.denom_all <= denom[5:0];
      j.gw2      <= (q_sw[2:0] != 3'd0);
      j.prefetch <= (reach_h + 32'd1 + q_sh <= 32'(LBR));
      j.cip      <= (q_cip != '0);
      j.mode     <= (q_type == 32'd0) ? M_MAX : (q_type == 32'd1) ? M_AVG :
                    (q_lp == 32'd1) ? M_LP1 : M_LP2;
    end
  end

  always_ff @(posedge clk) start_q <= job_start;
  always_ff @(posedge clk) begin
    urst    <= rst || job_start;      // sequencer and chunk FIFOs
    urst_ld <= rst || job_start;
    urst_em <= rst || job_start;
    urst_rd <= rst || job_start;
    urst_wr <= rst || job_start;
  end

  // Chunk sequencer ----------------------------------------------------------------------
  // (batch ni, channel tile, W-tile) in the HLS order; every quantity by
  // increments: the input / output plane offsets of the tile, ow_lo and
  // ow_lo * stride_w.
  typedef enum logic [2:0] {Q_OFF, Q_K1, Q_KE, Q_K2, Q_PUSH} qst_t;
  qst_t        qs;
  logic [31:0] s_ni, s_cleft, s_bin_n, s_bout_n, s_bin_c, s_bout_c, s_owlo, s_awlo;
  logic        k_last;                     // the W-tile is the row's last
  logic [6:0]  k_span;
  logic [3:0]  k_cv;
  logic [31:0] k_iws, k_iwe;               // first / last input column of the tile (signed)
  logic [31:0] k_mid, k_rwp;               // Q_K1 -> Q_KE: ow_lo * stride_w + (ow_tile - 1) * stride_w, reach_w - pad_left
  chunk_t      k;
  logic        cl_ready, ce_ready, cw_ready, c_push;

  assign c_push = (qs == Q_PUSH) && cl_ready && ce_ready && cw_ready;

  always_ff @(posedge clk) begin
    if (urst) begin
      qs       <= start_q ? Q_K1 : Q_OFF;
      seq_done <= 1'b0;
      s_ni     <= '0;
      s_cleft  <= q_ch;
      {s_bin_n, s_bout_n, s_bin_c, s_bout_c, s_owlo, s_awlo} <= '0;
    end else if (job_start) begin
      seq_done <= 1'b0;
    end else if (start_q) begin
      qs <= Q_K1;
    end else begin
      case (qs)
        Q_K1: begin
          logic [31:0] left;
          left   = j.out_w - s_owlo;
          k_last <= (left <= 32'(ow_tile));
          k_span <= (left <= 32'(ow_tile)) ? 7'(left) : ow_tile;
          k_cv   <= (s_cleft >= 32'(TC)) ? 4'(TC) : 4'(s_cleft);
          k_iws  <= s_awlo - q_pl;
          k_mid  <= s_awlo + owt1_sw;
          k_rwp  <= reach_w - q_pl;
          qs     <= Q_KE;
        end
        Q_KE: begin
          k_iwe  <= (k_last ? lastw : k_mid) + k_rwp;
          qs     <= Q_K2;
        end
        Q_K2: begin
          logic [31:0] lo, hi;
          lo = k_iws[31] ? 32'd0 : k_iws;
          hi = ($signed(k_iwe) >= $signed(j.in_w)) ? j.in_w - 32'd1 : k_iwe;
          k.c_valid  <= k_cv;
          k.ow_span  <= k_span;
          k.n_groups <= j.gw2 ? 7'((8'(k_span) + 8'd1) >> 1) : k_span;
          k.run_len  <= 7'(hi - lo + 32'd1);
          k.col_base <= k_iws - lo;
          k.lo_lim   <= 32'd0 - lo;
          k.hi_lim   <= j.in_w - lo;
          k.base_in  <= s_bin_c + lo;
          k.base_out <= s_bout_c + s_owlo;
          qs <= Q_PUSH;
        end
        Q_PUSH: if (c_push) begin
          qs <= Q_K1;
          if (k_last) begin
            s_owlo <= '0;
            s_awlo <= '0;
            if (s_cleft > 32'(TC)) begin
              s_cleft  <= s_cleft - 32'(TC);
              s_bin_c  <= s_bin_c + {in_hw[28:0], 3'b0};
              s_bout_c <= s_bout_c + {out_hw[28:0], 3'b0};
            end else if (s_ni + 32'd1 == q_batch) begin
              seq_done <= 1'b1;
              qs       <= Q_OFF;
            end else begin
              s_ni     <= s_ni + 32'd1;
              s_cleft  <= q_ch;
              s_bin_n  <= s_bin_n + ch_in;
              s_bin_c  <= s_bin_n + ch_in;
              s_bout_n <= s_bout_n + ch_out;
              s_bout_c <= s_bout_n + ch_out;
            end
          end else begin
            s_owlo <= s_owlo + 32'(ow_tile);
            s_awlo <= s_awlo + owt_sw;
          end
        end
        default: ;
      endcase
    end
  end

  // one copy of each chunk per consumer
  logic   cl_valid, ce_valid, cw_valid, cl_pop, ce_pop, cw_pop;
  chunk_t cl, ce, cw;
  logic [2:0] cl_n, ce_n, cw_n;
  pl_fifo #(.W($bits(chunk_t)), .D(4), .BRAM(1'b0)) u_cl (
    .clk, .rst (urst), .in_valid (c_push), .in_ready (cl_ready), .in_data (k),
    .out_valid (cl_valid), .out_ready (cl_pop), .out_data (cl), .count (cl_n));
  pl_fifo #(.W($bits(chunk_t)), .D(4), .BRAM(1'b0)) u_ce (
    .clk, .rst (urst), .in_valid (c_push), .in_ready (ce_ready), .in_data (k),
    .out_valid (ce_valid), .out_ready (ce_pop), .out_data (ce), .count (ce_n));
  pl_fifo #(.W($bits(chunk_t)), .D(4), .BRAM(1'b0)) u_cw (
    .clk, .rst (urst), .in_valid (c_push), .in_ready (cw_ready), .in_data (k),
    .out_valid (cw_valid), .out_ready (cw_pop), .out_data (cw), .count (cw_n));

  // Units -----------------------------------------------------------------------------------
  logic          dq_valid, dq_ready, wq_valid, wq_ready, bt_valid, bt_ready, fb_valid, fb_ready;
  lbd_t          dq;
  logic [BW-1:0] wq;
  beat_t         bt;
  logic [OWP*EW-1:0] fb;
  logic          ld_idle, em_idle, rd_idle, wr_idle, cl_rdy, ce_rdy, cw_rdy;

  pl_loader u_ld (
    .clk, .rst (urst_ld), .j,
    .cq_valid (cl_valid), .cq_ready (cl_rdy), .cq (cl),
    .arvalid (m_axi_gmem0_ARVALID), .arready (m_axi_gmem0_ARREADY),
    .araddr  (m_axi_gmem0_ARADDR),  .arlen   (m_axi_gmem0_ARLEN),
    .rvalid  (m_axi_gmem0_RVALID),  .rready  (m_axi_gmem0_RREADY),
    .rdata   (m_axi_gmem0_RDATA),   .rlast   (m_axi_gmem0_RLAST),
    .dq_valid, .dq_ready, .dq_data (dq),
    .wq_valid, .wq_ready, .wq_data (wq),
    .idle (ld_idle)
  );
  assign cl_pop = cl_valid && cl_rdy;

  pl_emit u_em (
    .clk, .rst (urst_em), .j,
    .cq_valid (ce_valid), .cq_ready (ce_rdy), .cq (ce),
    .dq_valid, .dq_ready, .dq_data (dq),
    .wq_valid, .wq_ready, .wq_data (wq),
    .bt_valid, .bt_ready, .bt,
    .idle (em_idle)
  );
  assign ce_pop = ce_valid && ce_rdy;

  // The window beats leave the emitter's beat FIFO through a register slice:
  // the reducer spreads over several DSP columns, so its stall (bt_ready)
  // and the beat's lanes cross a long distance between two registers.
  logic  bs_valid, bs_ready;
  beat_t bs;
  pl_rs #(.W($bits(beat_t))) u_bs (
    .clk, .rst (urst_rd),
    .in_valid  (bt_valid), .in_ready  (bt_ready), .in_data (bt),
    .out_valid (bs_valid), .out_ready (bs_ready), .out_data (bs)
  );

  pl_reduce u_rd (
    .clk, .rst (urst_rd), .j,
    .bt_valid (bs_valid), .bt_ready (bs_ready), .bt (bs),
    .fb_valid, .fb_ready, .fb,
    .idle (rd_idle)
  );

  pl_writer u_wr (
    .clk, .rst (urst_wr), .j,
    .cq_valid (cw_valid), .cq_ready (cw_rdy), .cq (cw),
    .fb_valid, .fb_ready, .fb,
    .awvalid (m_axi_gmem1_AWVALID), .awready (m_axi_gmem1_AWREADY),
    .awaddr  (m_axi_gmem1_AWADDR),  .awlen   (m_axi_gmem1_AWLEN),
    .wvalid  (m_axi_gmem1_WVALID),  .wready  (m_axi_gmem1_WREADY),
    .wdata   (m_axi_gmem1_WDATA),   .wstrb   (m_axi_gmem1_WSTRB),
    .wlast   (m_axi_gmem1_WLAST),
    .bvalid  (m_axi_gmem1_BVALID),  .bready  (m_axi_gmem1_BREADY),
    .idle (wr_idle)
  );
  assign cw_pop = cw_valid && cw_rdy;

  assign all_idle = !cl_valid && !ce_valid && !cw_valid && ld_idle && em_idle && !bs_valid && bt_ready &&
                    rd_idle && wr_idle;
  assign ap_done  = (tstate == T_DONE);
  assign ap_idle  = (tstate == T_IDLE);

  // AXI constants and unused channels --------------------------------------------------------
  assign m_axi_gmem0_ARID     = '0;
  assign m_axi_gmem0_ARSIZE   = 3'b100;
  assign m_axi_gmem0_ARBURST  = 2'b01;
  assign m_axi_gmem0_ARLOCK   = 2'b00;
  assign m_axi_gmem0_ARCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_ARPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_ARQOS    = 4'b0;
  assign m_axi_gmem0_ARREGION = 4'b0;
  assign m_axi_gmem0_ARUSER   = C_M_AXI_GMEM0_ARUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_AWVALID  = 1'b0;
  assign m_axi_gmem0_AWADDR   = '0;
  assign m_axi_gmem0_AWID     = '0;
  assign m_axi_gmem0_AWLEN    = '0;
  assign m_axi_gmem0_AWSIZE   = 3'b100;
  assign m_axi_gmem0_AWBURST  = 2'b01;
  assign m_axi_gmem0_AWLOCK   = 2'b00;
  assign m_axi_gmem0_AWCACHE  = 4'(C_M_AXI_GMEM0_CACHE_VALUE);
  assign m_axi_gmem0_AWPROT   = 3'(C_M_AXI_GMEM0_PROT_VALUE);
  assign m_axi_gmem0_AWQOS    = 4'b0;
  assign m_axi_gmem0_AWREGION = 4'b0;
  assign m_axi_gmem0_AWUSER   = C_M_AXI_GMEM0_AWUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_WVALID   = 1'b0;
  assign m_axi_gmem0_WDATA    = '0;
  assign m_axi_gmem0_WSTRB    = '0;
  assign m_axi_gmem0_WLAST    = 1'b0;
  assign m_axi_gmem0_WID      = '0;
  assign m_axi_gmem0_WUSER    = C_M_AXI_GMEM0_WUSER_WIDTH'(C_M_AXI_GMEM0_USER_VALUE);
  assign m_axi_gmem0_BREADY   = 1'b1;

  assign m_axi_gmem1_AWID     = '0;
  assign m_axi_gmem1_AWSIZE   = 3'b100;
  assign m_axi_gmem1_AWBURST  = 2'b01;
  assign m_axi_gmem1_AWLOCK   = 2'b00;
  assign m_axi_gmem1_AWCACHE  = 4'(C_M_AXI_GMEM1_CACHE_VALUE);
  assign m_axi_gmem1_AWPROT   = 3'(C_M_AXI_GMEM1_PROT_VALUE);
  assign m_axi_gmem1_AWQOS    = 4'b0;
  assign m_axi_gmem1_AWREGION = 4'b0;
  assign m_axi_gmem1_AWUSER   = C_M_AXI_GMEM1_AWUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_WID      = '0;
  assign m_axi_gmem1_WUSER    = C_M_AXI_GMEM1_WUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_ARVALID  = 1'b0;
  assign m_axi_gmem1_ARADDR   = '0;
  assign m_axi_gmem1_ARID     = '0;
  assign m_axi_gmem1_ARLEN    = '0;
  assign m_axi_gmem1_ARSIZE   = 3'b100;
  assign m_axi_gmem1_ARBURST  = 2'b01;
  assign m_axi_gmem1_ARLOCK   = 2'b00;
  assign m_axi_gmem1_ARCACHE  = 4'(C_M_AXI_GMEM1_CACHE_VALUE);
  assign m_axi_gmem1_ARPROT   = 3'(C_M_AXI_GMEM1_PROT_VALUE);
  assign m_axi_gmem1_ARQOS    = 4'b0;
  assign m_axi_gmem1_ARREGION = 4'b0;
  assign m_axi_gmem1_ARUSER   = C_M_AXI_GMEM1_ARUSER_WIDTH'(C_M_AXI_GMEM1_USER_VALUE);
  assign m_axi_gmem1_RREADY   = 1'b1;

  logic unused_top;
  assign unused_top = ^{m_axi_gmem0_AWREADY, m_axi_gmem0_WREADY, m_axi_gmem0_RID,
                        m_axi_gmem0_RUSER, m_axi_gmem0_RRESP, m_axi_gmem0_BVALID,
                        m_axi_gmem0_BRESP, m_axi_gmem0_BID, m_axi_gmem0_BUSER,
                        m_axi_gmem1_ARREADY, m_axi_gmem1_RVALID, m_axi_gmem1_RDATA,
                        m_axi_gmem1_RLAST, m_axi_gmem1_RID, m_axi_gmem1_RUSER,
                        m_axi_gmem1_RRESP, m_axi_gmem1_BRESP, m_axi_gmem1_BID,
                        m_axi_gmem1_BUSER, cl_n, ce_n, cw_n, s_ni, q_type, q_lp,
                        q_pt, q_dh, q_dw, q_in_w, q_in_h, q_out_h, ohs, j};

endmodule
