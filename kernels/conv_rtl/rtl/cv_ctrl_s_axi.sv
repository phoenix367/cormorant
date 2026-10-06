// ---------------------------------------------------------------------------
// cv_ctrl_s_axi — AXI4-Lite control slave with the exact register map and
// ap_ctrl_hs behaviour of the Vitis-HLS ConvKernel (xconvkernel_hw.h):
//
//   0x00 ctrl  b0 ap_start (R/W, cleared on ap_ready unless auto_restart)
//              b1 ap_done (clear on read)  b2 ap_idle  b3 ap_ready (COR)
//              b7 auto_restart  b9 interrupt
//   0x04 GIE   0x08 IER (b0 done, b1 ready)   0x0C ISR (toggle on write)
//   0x10/14 x  0x1C/20 weight  0x28/2C bias  0x34/38 y  0x40 batch  0x48 in_ch
//   0x50 in_h  0x58 in_w  0x60 out_ch  0x68 out_h  0x70 out_w  0x78 kh  0x80 kw
//   0x88 stride_h  0x90 stride_w  0x98 dilation_h  0xA0 dilation_w
//   0xA8 pad_top  0xB0 pad_left  0xB8 has_bias  0xC0 is_depthwise
//
// Like the HLS slave, AW and W are taken one after the other and every
// access gets an OKAY response.  (Derived from the RTL PoolingKernel's
// pl_ctrl_s_axi.)
// ---------------------------------------------------------------------------
module cv_ctrl_s_axi (
  input  logic        clk,
  input  logic        rst,

  input  logic        awvalid,
  output logic        awready,
  input  logic [7:0]  awaddr,
  input  logic        wvalid,
  output logic        wready,
  input  logic [31:0] wdata,
  input  logic [3:0]  wstrb,
  output logic        bvalid,
  input  logic        bready,
  output logic [1:0]  bresp,
  input  logic        arvalid,
  output logic        arready,
  input  logic [7:0]  araddr,
  output logic        rvalid,
  input  logic        rready,
  output logic [31:0] rdata,
  output logic [1:0]  rresp,

  output logic        interrupt,

  // Kernel handshake.
  output logic        ap_start,
  input  logic        ap_done,
  input  logic        ap_ready,
  input  logic        ap_idle,

  // Arguments.
  output logic [63:0] x,
  output logic [63:0] weight,
  output logic [63:0] bias,
  output logic [63:0] y,
  output logic [31:0] batch,
  output logic [31:0] in_ch,
  output logic [31:0] in_h,
  output logic [31:0] in_w,
  output logic [31:0] out_ch,
  output logic [31:0] out_h,
  output logic [31:0] out_w,
  output logic [31:0] kh,
  output logic [31:0] kw,
  output logic [31:0] stride_h,
  output logic [31:0] stride_w,
  output logic [31:0] dilation_h,
  output logic [31:0] dilation_w,
  output logic [31:0] pad_top,
  output logic [31:0] pad_left,
  output logic [31:0] has_bias,
  output logic [31:0] is_depthwise
);
  localparam logic [7:0] A_CTRL = 8'h00, A_GIE = 8'h04, A_IER = 8'h08, A_ISR = 8'h0C,
                          A_X0 = 8'h10, A_X1 = 8'h14, A_WEIGHT0 = 8'h1C, A_WEIGHT1 = 8'h20,
                          A_BIAS0 = 8'h28, A_BIAS1 = 8'h2C, A_Y0 = 8'h34, A_Y1 = 8'h38,
                          A_BATCH = 8'h40, A_IN_CH = 8'h48, A_IN_H = 8'h50, A_IN_W = 8'h58,
                          A_OUT_CH = 8'h60, A_OUT_H = 8'h68, A_OUT_W = 8'h70, A_KH = 8'h78,
                          A_KW = 8'h80, A_STRIDE_H = 8'h88, A_STRIDE_W = 8'h90,
                          A_DIL_H = 8'h98, A_DIL_W = 8'hA0, A_PAD_TOP = 8'hA8,
                          A_PAD_LEFT = 8'hB0, A_HAS_BIAS = 8'hB8, A_IS_DW = 8'hC0;

  // Write channel ------------------------------------------------------------------
  typedef enum logic [1:0] {WR_IDLE, WR_DATA, WR_RESP} wst_t;
  wst_t       wst;
  logic [7:0] waddr;
  logic       w_hs;

  assign awready = (wst == WR_IDLE);
  assign wready  = (wst == WR_DATA);
  assign bvalid  = (wst == WR_RESP);
  assign bresp   = 2'b00;
  assign w_hs    = wvalid && wready;

  always_ff @(posedge clk) begin
    if (rst) wst <= WR_IDLE;
    else case (wst)
      WR_IDLE: if (awvalid) wst <= WR_DATA;
      WR_DATA: if (wvalid)  wst <= WR_RESP;
      WR_RESP: if (bready)  wst <= WR_IDLE;
      default: wst <= WR_IDLE;
    endcase
    if (awvalid && awready) waddr <= {awaddr[7:2], 2'b00};
  end

  logic [31:0] wmask;
  assign wmask = {{8{wstrb[3]}}, {8{wstrb[2]}}, {8{wstrb[1]}}, {8{wstrb[0]}}};

  function automatic logic [31:0] merge(input logic [31:0] old);
    return (wdata & wmask) | (old & ~wmask);
  endfunction

  // Control / status -------------------------------------------------------------
  logic       int_ap_start, int_task_done, int_ap_idle, int_ap_ready;
  logic       int_auto_restart, auto_restart_status, int_gie, int_irq;
  logic [1:0] int_ier, int_isr;
  logic       ar_hs;
  logic [7:0] raddr;
  logic       task_ap_done, task_ap_ready, auto_restart_done;

  assign ar_hs             = arvalid && arready;
  assign raddr             = araddr;
  assign auto_restart_done = auto_restart_status && ap_idle && !int_ap_idle;
  assign task_ap_done      = (ap_done && !auto_restart_status) || auto_restart_done;
  assign task_ap_ready     = ap_ready && !int_auto_restart;

  assign ap_start  = int_ap_start;
  assign interrupt = int_irq;

  always_ff @(posedge clk) begin
    if (rst) begin
      int_ap_start        <= 1'b0;
      int_task_done       <= 1'b0;
      int_ap_idle         <= 1'b0;
      int_ap_ready        <= 1'b0;
      int_auto_restart    <= 1'b0;
      auto_restart_status <= 1'b0;
      int_gie             <= 1'b0;
      int_ier             <= 2'b0;
      int_isr             <= 2'b0;
      int_irq             <= 1'b0;
      x <= '0;
      weight <= '0;
      bias <= '0;
      y <= '0;
      batch <= '0;
      in_ch <= '0;
      in_h <= '0;
      in_w <= '0;
      out_ch <= '0;
      out_h <= '0;
      out_w <= '0;
      kh <= '0;
      kw <= '0;
      stride_h <= '0;
      stride_w <= '0;
      dilation_h <= '0;
      dilation_w <= '0;
      pad_top <= '0;
      pad_left <= '0;
      has_bias <= '0;
      is_depthwise <= '0;
    end else begin
      int_irq <= int_gie && (|int_isr);

      if (w_hs && waddr == A_CTRL && wstrb[0] && wdata[0]) int_ap_start <= 1'b1;
      else if (ap_ready)                                  int_ap_start <= int_auto_restart;

      if (task_ap_done)                   int_task_done <= 1'b1;
      else if (ar_hs && raddr == A_CTRL)  int_task_done <= 1'b0;

      int_ap_idle <= ap_idle;

      if (task_ap_ready)                  int_ap_ready <= 1'b1;
      else if (ar_hs && raddr == A_CTRL)  int_ap_ready <= 1'b0;

      if (w_hs && waddr == A_CTRL && wstrb[0]) int_auto_restart <= wdata[7];

      if (int_auto_restart) auto_restart_status <= 1'b1;
      else if (ap_idle)     auto_restart_status <= 1'b0;

      if (w_hs && waddr == A_GIE && wstrb[0]) int_gie <= wdata[0];
      if (w_hs && waddr == A_IER && wstrb[0]) int_ier <= wdata[1:0];

      if (int_ier[0] && ap_done)                    int_isr[0] <= 1'b1;
      else if (w_hs && waddr == A_ISR && wstrb[0])  int_isr[0] <= int_isr[0] ^ wdata[0];
      if (int_ier[1] && ap_ready)                   int_isr[1] <= 1'b1;
      else if (w_hs && waddr == A_ISR && wstrb[0])  int_isr[1] <= int_isr[1] ^ wdata[1];

      if (w_hs) begin
        case (waddr)
          A_X0: x[31:0] <= merge(x[31:0]);
          A_X1: x[63:32] <= merge(x[63:32]);
          A_WEIGHT0: weight[31:0] <= merge(weight[31:0]);
          A_WEIGHT1: weight[63:32] <= merge(weight[63:32]);
          A_BIAS0: bias[31:0] <= merge(bias[31:0]);
          A_BIAS1: bias[63:32] <= merge(bias[63:32]);
          A_Y0: y[31:0] <= merge(y[31:0]);
          A_Y1: y[63:32] <= merge(y[63:32]);
          A_BATCH: batch <= merge(batch);
          A_IN_CH: in_ch <= merge(in_ch);
          A_IN_H: in_h <= merge(in_h);
          A_IN_W: in_w <= merge(in_w);
          A_OUT_CH: out_ch <= merge(out_ch);
          A_OUT_H: out_h <= merge(out_h);
          A_OUT_W: out_w <= merge(out_w);
          A_KH: kh <= merge(kh);
          A_KW: kw <= merge(kw);
          A_STRIDE_H: stride_h <= merge(stride_h);
          A_STRIDE_W: stride_w <= merge(stride_w);
          A_DIL_H: dilation_h <= merge(dilation_h);
          A_DIL_W: dilation_w <= merge(dilation_w);
          A_PAD_TOP: pad_top <= merge(pad_top);
          A_PAD_LEFT: pad_left <= merge(pad_left);
          A_HAS_BIAS: has_bias <= merge(has_bias);
          A_IS_DW: is_depthwise <= merge(is_depthwise);
          default: ;
        endcase
      end
    end
  end

  // Read channel -------------------------------------------------------------------
  typedef enum logic {RD_IDLE, RD_DATA} rst_t;
  rst_t rst_q;

  assign arready = (rst_q == RD_IDLE);
  assign rvalid  = (rst_q == RD_DATA);
  assign rresp   = 2'b00;

  always_ff @(posedge clk) begin
    if (rst) rst_q <= RD_IDLE;
    else case (rst_q)
      RD_IDLE: if (arvalid) rst_q <= RD_DATA;
      RD_DATA: if (rready)  rst_q <= RD_IDLE;
      default: rst_q <= RD_IDLE;
    endcase
    if (ar_hs) begin
      rdata <= '0;
      case (raddr)
        A_CTRL:  begin
                   rdata[0] <= int_ap_start;
                   rdata[1] <= int_task_done;
                   rdata[2] <= int_ap_idle;
                   rdata[3] <= int_ap_ready;
                   rdata[7] <= int_auto_restart;
                   rdata[9] <= int_irq;
                 end
        A_GIE:   rdata <= {31'b0, int_gie};
        A_IER:   rdata <= {30'b0, int_ier};
        A_ISR:   rdata <= {30'b0, int_isr};
        A_X0: rdata <= x[31:0];
        A_X1: rdata <= x[63:32];
        A_WEIGHT0: rdata <= weight[31:0];
        A_WEIGHT1: rdata <= weight[63:32];
        A_BIAS0: rdata <= bias[31:0];
        A_BIAS1: rdata <= bias[63:32];
        A_Y0: rdata <= y[31:0];
        A_Y1: rdata <= y[63:32];
        A_BATCH: rdata <= batch;
        A_IN_CH: rdata <= in_ch;
        A_IN_H: rdata <= in_h;
        A_IN_W: rdata <= in_w;
        A_OUT_CH: rdata <= out_ch;
        A_OUT_H: rdata <= out_h;
        A_OUT_W: rdata <= out_w;
        A_KH: rdata <= kh;
        A_KW: rdata <= kw;
        A_STRIDE_H: rdata <= stride_h;
        A_STRIDE_W: rdata <= stride_w;
        A_DIL_H: rdata <= dilation_h;
        A_DIL_W: rdata <= dilation_w;
        A_PAD_TOP: rdata <= pad_top;
        A_PAD_LEFT: rdata <= pad_left;
        A_HAS_BIAS: rdata <= has_bias;
        A_IS_DW: rdata <= is_depthwise;
        default: ;
      endcase
    end
  end

endmodule
