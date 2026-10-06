// @tasks
  task automatic job(input int n, c, h, w, oh, ow, ph, pw, sh, sw, pt, pl, dh, dw, ty, lp, cip, input int k);
    longint unsigned x, y;
    x = 64'h1_0000_0000 + k * 64'h40_0000 + 16 * ($urandom % 64);
    y = 64'h2_0000_0000 + k * 64'h40_0000 + 16 * ($urandom % 64);
    wr(8'h10, x[31:0]); wr(8'h14, x[63:32]); wr(8'h1C, y[31:0]); wr(8'h20, y[63:32]);
    wr(8'h28, n); wr(8'h30, c); wr(8'h38, h); wr(8'h40, w); wr(8'h48, oh); wr(8'h50, ow);
    wr(8'h58, ph); wr(8'h60, pw); wr(8'h68, sh); wr(8'h70, sw); wr(8'h78, pt); wr(8'h80, pl);
    wr(8'h88, dh); wr(8'h90, dw); wr(8'h98, ty); wr(8'hA0, lp); wr(8'hA8, cip);
    start_and_wait($sformatf("%0d: %0d %0d %0dx%0d -> %0dx%0d k%0dx%0d s%0dx%0d p%0d,%0d t%0d", k, n, c, h, w, oh, ow, ph, pw, sh, sw, pt, pl, ty));
  endtask
// @jobs
    // PoolingKernel: the board's pool models (pool_* of run_remote_tests) and a few more
    job(1, 4, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0, 0);       // pool_maxpool_simple
    job(1, 4, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 0, 1);       // pool_avgpool_simple
    job(1, 1024, 7, 7, 1, 1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2);    // global average 7x7
    job(1, 4, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0, 3);
    job(1, 4, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 0, 4);       // pool_maxpool_padded
    job(1, 2, 6, 6, 3, 3, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 5);       // count_include_pad
    job(1, 4, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0, 6);       // lp p2
    job(1, 4, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 0, 7);       // lp p1
    job(1, 4, 9, 13, 4, 6, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0, 8);      // w13
    job(1, 8, 20, 77, 10, 39, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 9);   // w77 avg tiled
    job(1, 24, 17, 17, 17, 17, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 0, 10); // c24 w17
    job(3, 5, 7, 30, 3, 15, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0, 11);    // batch3
    job(1, 64, 28, 28, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 0, 1, 0, 12); // resnet-like
