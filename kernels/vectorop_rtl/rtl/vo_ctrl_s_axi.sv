// ---------------------------------------------------------------------------
// vo_ctrl_s_axi — AXI4-Lite control slave with the exact register map and
// ap_ctrl_hs behaviour of the Vitis-HLS VectorOPKernel (xvectoropkernel_hw.h):
//
//   0x00 ctrl  b0 ap_start (R/W, cleared on ap_ready unless auto_restart)
//              b1 ap_done (clear on read)  b2 ap_idle  b3 ap_ready (COR)
//              b7 auto_restart  b9 interrupt
//   0x04 GIE   0x08 IER (b0 done, b1 ready)   0x0C ISR (toggle on write)
//   0x10/14 a  0x1C/20 b  0x28/2C c  0x34 size  0x3C op  0x44 outer
//   0x4C a_inc  0x54 b_inc  0x5C act
//
// Like the HLS slave, AW and W are taken one after the other and every
// access gets an OKAY response.  (Derived from the RTL MatmulKernel's
// mm_ctrl_s_axi.)
// ---------------------------------------------------------------------------
module vo_ctrl_s_axi (
  input  logic        clk,
  input  logic        rst,

  input  logic        awvalid,
  output logic        awready,
  input  logic [6:0]  awaddr,
  input  logic        wvalid,
  output logic        wready,
  input  logic [31:0] wdata,
  input  logic [3:0]  wstrb,
  output logic        bvalid,
  input  logic        bready,
  output logic [1:0]  bresp,
  input  logic        arvalid,
  output logic        arready,
  input  logic [6:0]  araddr,
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
  output logic [63:0] a,
  output logic [63:0] b,
  output logic [63:0] c,
  output logic [31:0] size,
  output logic [31:0] op,
  output logic [31:0] outer,
  output logic [31:0] a_inc,
  output logic [31:0] b_inc,
  output logic [31:0] act
);
  localparam logic [6:0] A_CTRL = 7'h00, A_GIE = 7'h04, A_IER = 7'h08, A_ISR = 7'h0C,
                         A_A0 = 7'h10, A_A1 = 7'h14, A_B0 = 7'h1C, A_B1 = 7'h20,
                         A_C0 = 7'h28, A_C1 = 7'h2C, A_SIZE = 7'h34, A_OP = 7'h3C,
                         A_OUTER = 7'h44, A_AINC = 7'h4C, A_BINC = 7'h54, A_ACT = 7'h5C;

  // Write channel ------------------------------------------------------------------
  typedef enum logic [1:0] {WR_IDLE, WR_DATA, WR_RESP} wst_t;
  wst_t       wst;
  logic [6:0] waddr;
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
    if (awvalid && awready) waddr <= {awaddr[6:2], 2'b00};
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
  logic [6:0] raddr;
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
      a <= '0; b <= '0; c <= '0; size <= '0; op <= '0; outer <= '0;
      a_inc <= '0; b_inc <= '0; act <= '0;
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
          A_A0:    a[31:0]        <= merge(a[31:0]);
          A_A1:    a[63:32]       <= merge(a[63:32]);
          A_B0:    b[31:0]        <= merge(b[31:0]);
          A_B1:    b[63:32]       <= merge(b[63:32]);
          A_C0:    c[31:0]        <= merge(c[31:0]);
          A_C1:    c[63:32]       <= merge(c[63:32]);
          A_SIZE:  size           <= merge(size);
          A_OP:    op             <= merge(op);
          A_OUTER: outer          <= merge(outer);
          A_AINC:  a_inc          <= merge(a_inc);
          A_BINC:  b_inc          <= merge(b_inc);
          A_ACT:   act            <= merge(act);
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
        A_A0:    rdata <= a[31:0];
        A_A1:    rdata <= a[63:32];
        A_B0:    rdata <= b[31:0];
        A_B1:    rdata <= b[63:32];
        A_C0:    rdata <= c[31:0];
        A_C1:    rdata <= c[63:32];
        A_SIZE:  rdata <= size;
        A_OP:    rdata <= op;
        A_OUTER: rdata <= outer;
        A_AINC:  rdata <= a_inc;
        A_BINC:  rdata <= b_inc;
        A_ACT:   rdata <= act;
        default: ;
      endcase
    end
  end

endmodule
