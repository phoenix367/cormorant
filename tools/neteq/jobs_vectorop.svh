    // VectorOP: a, b, c, size, op, outer, a_inc, b_inc, act
    for (int k = 0; k < 24; k++) begin
      int size, op, outer, ainc, binc, act;
      longint unsigned a, b, c;
      size  = (k % 3 == 0) ? 1 + $urandom % 40 : 1 + $urandom % 3000;
      op    = k % 8; act = $urandom % 3;
      outer = (k % 4 == 0) ? 1 + $urandom % 30 : 1;
      ainc  = 8 * ((size + 7) / 8) + 8 * ($urandom % 2);
      binc  = (k % 5 == 0) ? 0 : ainc;
      a = 64'h1_0000_0000 + k * 64'h100_0000; b = a + 64'h40_0000; c = a + 64'h80_0000;
      wr(8'h10, a[31:0]); wr(8'h14, a[63:32]); wr(8'h1C, b[31:0]); wr(8'h20, b[63:32]);
      wr(8'h28, c[31:0]); wr(8'h2C, c[63:32]); wr(8'h34, size); wr(8'h3C, op); wr(8'h44, outer);
      wr(8'h4C, ainc); wr(8'h54, binc); wr(8'h5C, act);
      start_and_wait($sformatf("%0d size %0d op %0d outer %0d", k, size, op, outer));
    end
