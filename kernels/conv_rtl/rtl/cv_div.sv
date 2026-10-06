// ---------------------------------------------------------------------------
// cv_div — unsigned restoring division of an N-bit dividend by a 32-bit
// divisor, one quotient bit per cycle (N cycles from start to done).  The
// job configuration's three divisions run on three of these at once.
// ---------------------------------------------------------------------------
module cv_div #(
  parameter int N = 17
) (
  input  logic          clk,
  input  logic          start,
  input  logic [N-1:0]  dividend,
  input  logic [31:0]   divisor,
  output logic [N-1:0]  quotient,
  output logic          done
);
  logic [N-1:0]  num;
  logic [32:0]   rem;
  logic [$clog2(N+1)-1:0] i;
  logic          busy;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (start) begin
      num      <= dividend;
      rem      <= '0;
      quotient <= '0;
      i        <= $clog2(N+1)'(N);
      busy     <= 1'b1;
    end else if (busy) begin
      logic [32:0] r;
      r = {rem[31:0], num[N-1]};
      num <= num << 1;
      if (r >= {1'b0, divisor}) begin
        rem      <= r - {1'b0, divisor};
        quotient <= {quotient[N-2:0], 1'b1};
      end else begin
        rem      <= r;
        quotient <= {quotient[N-2:0], 1'b0};
      end
      i <= i - 1'b1;
      if (i == 1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule
