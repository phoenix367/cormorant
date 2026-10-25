// Copyright 2026 Ivan Gubochkin
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// cv_div — unsigned restoring division of an N-bit dividend by a DW-bit
// divisor, one quotient bit per cycle (N cycles from start to done).  The
// job configuration's three divisions run on two of these.
// ---------------------------------------------------------------------------
module cv_div #(
  parameter int N  = 17,
  parameter int DW = 32
) (
  input  logic          clk,
  input  logic          start,
  input  logic [N-1:0]  dividend,
  input  logic [DW-1:0] divisor,          // held from start to done
  output logic [N-1:0]  quotient,
  output logic          done
);
  logic [N-1:0]  num;
  logic [DW:0]   rem;
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
      logic [DW:0] r;
      r = {rem[DW-1:0], num[N-1]};
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
