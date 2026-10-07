    // VectorOP: a, b, c, size, op, outer, a_inc, b_inc, act, alpha (ops 0..9:
    // the activation ops 6..9 among them; acts 0..6; alpha's high bits ignored)
    for (int k = 0; k < 30; k++) begin
      int size, op, outer, ainc, binc, act, alpha;
      longint unsigned a, b, c;
      size  = (k % 3 == 0) ? 1 + $urandom % 40 : 1 + $urandom % 3000;
      op    = k % 10; act = $urandom % 7; alpha = $urandom;
      outer = (k % 4 == 0) ? 1 + $urandom % 30 : 1;
      ainc  = 8 * ((size + 7) / 8) + 8 * ($urandom % 2);
      binc  = (k % 5 == 0) ? 0 : ainc;
      a = 64'h1_0000_0000 + k * 64'h100_0000; b = a + 64'h40_0000; c = a + 64'h80_0000;
      wr(8'h10, a[31:0]); wr(8'h14, a[63:32]); wr(8'h1C, b[31:0]); wr(8'h20, b[63:32]);
      wr(8'h28, c[31:0]); wr(8'h2C, c[63:32]); wr(8'h34, size); wr(8'h3C, op); wr(8'h44, outer);
      wr(8'h4C, ainc); wr(8'h54, binc); wr(8'h5C, act); wr(8'h64, alpha);
      start_and_wait($sformatf("%0d size %0d op %0d act %0d outer %0d", k, size, op, act, outer));
    end
    // the softmax (ops 10 / 11, doc/plans/SOFTMAX_PLAN.md): smx_cm 0x6C, smx_cfg 0x74,
    // smx_mask 0x7C; row mode rows at a_inc -> b_inc, column mode size key rows of
    // outer queries -> outer & ~15 rows; masks with and without a period
    for (int k = 0; k < 12; k++) begin
      int size, op, outer, ainc, binc, cm, cfg, mask;
      longint unsigned a, c;
      op    = 10 + k % 2;
      size  = (op == 10) ? 1 + $urandom % 300 : 1 + $urandom % 100;
      outer = (op == 10) ? 1 + $urandom % 6 : 16 * (1 + $urandom % 3) + ((k % 4 == 1) ? 5 : 0);
      ainc  = (op == 10) ? 8 * ((size + 7) / 8) : 8 * ((outer + 7) / 8) + 8 * ($urandom % 2);
      binc  = 8 * ((size + 7) / 8) + 8 * ($urandom % 2);
      cm    = (1 << 23) + $urandom % (1 << 23);
      cfg   = (12 + $urandom % 14) | ((8 + $urandom % 8) << 8);
      mask  = (k % 3 == 0) ? (1 + $urandom % size) | ((1 + $urandom % 20) << 16)
                           : ((k % 3 == 1) ? $urandom % (size + 2) : size);
      a = 64'h2_0000_0000 + k * 64'h100_0000; c = a + 64'h80_0000;
      wr(8'h10, a[31:0]); wr(8'h14, a[63:32]); wr(8'h28, c[31:0]); wr(8'h2C, c[63:32]);
      wr(8'h34, size); wr(8'h3C, op); wr(8'h44, outer); wr(8'h4C, ainc); wr(8'h54, binc);
      wr(8'h6C, cm); wr(8'h74, cfg); wr(8'h7C, mask);
      start_and_wait($sformatf("smx %0d size %0d op %0d outer %0d mask %0h", k, size, op, outer, mask));
    end
