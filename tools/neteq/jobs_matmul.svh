    // Matmul: a, b, c, n, k, m, batch, strides, b_packed, gemv_kw
    for (int t = 0; t < 14; t++) begin
      int n, k, m, bt, pk, kw;
      longint unsigned a, b, c;
      case (t % 7)
        0: begin n = 1;  k = 64;  m = 128; kw = 0; pk = 0; end
        1: begin n = 16; k = 48;  m = 40;  kw = 0; pk = 1; end
        2: begin n = 1;  k = 128; m = 64;  kw = 4; pk = 0; end
        3: begin n = 9;  k = 144; m = 96;  kw = 0; pk = 0; end
        4: begin n = 3;  k = 24;  m = 8;   kw = 1; pk = 0; end
        5: begin n = 20; k = 72;  m = 600; kw = 0; pk = 0; end
        default: begin n = 1; k = 256; m = 64; kw = 8; pk = 0; end
      endcase
      bt = 1 + (t / 7);
      a = 64'h1_0000_0000 + t * 64'h100_0000; b = a + 64'h40_0000; c = a + 64'h80_0000;
      wr(8'h10, a[31:0]); wr(8'h14, a[63:32]); wr(8'h1C, b[31:0]); wr(8'h20, b[63:32]);
      wr(8'h28, c[31:0]); wr(8'h2C, c[63:32]); wr(8'h34, n); wr(8'h3C, k); wr(8'h44, m);
      wr(8'h4C, bt); wr(8'h54, n * k); wr(8'h5C, k * m + 64); wr(8'h64, n * m);
      wr(8'h6C, pk); wr(8'h74, kw); wr(8'h7C, 0); wr(8'h80, 0);
      start_and_wait($sformatf("%0d n %0d k %0d m %0d batch %0d pk %0d kw %0d", t, n, k, m, bt, pk, kw));
    end
