/* bench_rowops.c — RowOpsKernel (the row-ops unit) latency / GB/s benchmark
 *
 * args: instance label op n rows blocks a_s1 a_s2 b_s1 b_s2 c_s1 c_s2 aux win_lo win_hi iters [warmup [layout]]
 *
 * The registers are src/perf_calls.py FIELDS["RowOpsKernel"] in order (eps
 * and bq_sh, which do not change the timing, are 0).  The buffers are sized
 * as perf_calibrate.buffer_sizes does: A, B (when the op word's B_EN is set),
 * C from the strides and the element widths the op word's kind fields give,
 * P one 16-byte entry per column (one entry with P_BCAST).  The tables are
 * not loaded (the timing does not depend on their contents); a generated
 * project loads them once in inference_init().
 *
 * layout "OFF_A,OFF_B,OFF_C,OFF_P": byte offsets inside ONE buffer (default:
 * four buffers wherever the allocator puts them).
 * output (stdout): one JSON line
 */
#include "inference.h"
#include "xrowopskernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RO_B_EN    (1u << 12)
#define RO_P_BCAST (1u << 13)

static unsigned kind_bytes(unsigned k) { return k == 0u ? 2u : 4u; }

static void run_once(XRowopskernel *k, uint64_t ap, uint64_t bp, uint64_t cp, uint64_t pp,
                     const unsigned *r)
{
    XRowopskernel_Set_op(k, r[0]);
    XRowopskernel_Set_n(k, r[1]);
    XRowopskernel_Set_rows(k, r[2]);
    XRowopskernel_Set_blocks(k, r[3]);
    XRowopskernel_Set_a(k, ap);
    XRowopskernel_Set_a_s1(k, r[4]);
    XRowopskernel_Set_a_s2(k, r[5]);
    XRowopskernel_Set_b(k, bp);
    XRowopskernel_Set_b_s1(k, r[6]);
    XRowopskernel_Set_b_s2(k, r[7]);
    XRowopskernel_Set_c(k, cp);
    XRowopskernel_Set_c_s1(k, r[8]);
    XRowopskernel_Set_c_s2(k, r[9]);
    XRowopskernel_Set_p(k, pp);
    XRowopskernel_Set_eps(k, 0u);
    XRowopskernel_Set_aux(k, r[10]);
    XRowopskernel_Set_bq_sh(k, 0u);
    XRowopskernel_Set_win_lo(k, r[11]);
    XRowopskernel_Set_win_hi(k, r[12]);
#ifdef XROWOPSKERNEL_CTRL_ADDR_CHAIN_N_DATA   /* one job, not a chain (RPU_PLAN §13.3): the registers
                                               * keep what the last program wrote */
    XRowopskernel_Set_chain(k, (u64)0);
    XRowopskernel_Set_chain_n(k, 0u);
#endif
    XRowopskernel_Start(k);
    while (!XRowopskernel_IsDone(k)) {}
}

int main(int argc, char **argv)
{
    unsigned r[13], i;
    if (argc < 17) {
        fprintf(stderr,
            "usage: %s instance label op n rows blocks a_s1 a_s2 b_s1 b_s2 c_s1 c_s2 aux win_lo win_hi "
            "iters [warmup [layout]]\n", argv[0]);
        return 1;
    }
    const char *inst  = argv[1];
    const char *label = argv[2];
    for (i = 0; i < 13u; i++) r[i] = (unsigned)strtoul(argv[3 + i], NULL, 0);
    unsigned iters  = (unsigned)strtoul(argv[16], NULL, 0);
    unsigned warmup = (argc > 17) ? (unsigned)strtoul(argv[17], NULL, 0) : 10u;
    const char *layout = (argc > 18) ? argv[18] : NULL;

    const unsigned op = r[0], n = r[1], rows = r[2], blocks = r[3];
    const unsigned ea = kind_bytes((op >> 4) & 3u), eb = kind_bytes((op >> 6) & 3u), ec = kind_bytes((op >> 8) & 3u);
    const unsigned rm1 = rows ? rows - 1u : 0u, bm1 = blocks ? blocks - 1u : 0u;
    const unsigned code = op & 15u, R = r[10] & 15u, HD = r[10] & 0xFFFu;
    /* the extents (ROWOPS_RTL_KERNEL.md §1; perf_calibrate.buffer_sizes): a TRANSPOSE
     * tile's R rows are b_s1 apart and column d's granule sits at d * b_s2; ROPE's B
     * row is the position's 2 HD int32 cos / sin */
    unsigned a_bytes = code == 7u ? rm1 * r[4] + bm1 * r[5] + (R ? R - 1u : 0u) * r[6] + n * 2u
                                  : rm1 * r[4] + bm1 * r[5] + n * ea;
    unsigned b_bytes = code == 6u ? rm1 * r[6] + bm1 * r[7] + 8u * HD
                     : (op & RO_B_EN) ? rm1 * r[6] + bm1 * r[7] + n * eb : 16u;
    unsigned c_bytes = code == 7u ? rm1 * r[8] + bm1 * r[9] + (n ? n - 1u : 0u) * r[7] + 2u * R
                                  : rm1 * r[8] + bm1 * r[9] + n * ec;
    unsigned p_bytes = (op & RO_P_BCAST) ? 16u : 16u * n;
    unsigned counts[4] = { (a_bytes + 1u) / 2u + 64u, (b_bytes + 1u) / 2u + 64u,
                           (c_bytes + 1u) / 2u + 64u, (p_bytes + 1u) / 2u + 64u };

    if (inference_buf_pool_init() != 0) return 1;

    XRowopskernel k;
    if (XRowopskernel_Initialize(&k, inst) != 0) {
        fprintf(stderr, "bench_rowops: init '%s' failed\n", inst); return 1;
    }

    inference_buf_t *bufs[4], *pool;
    if (inference_buf_alloc_layout(layout, 4, counts, bufs, &pool) != 0) {
        fprintf(stderr, "bench_rowops: alloc failed\n"); return 1;
    }
    for (i = 0; i < 4u; i++) {
        memset(inference_buf_ptr(bufs[i]), i == 3u ? 0 : 1, (size_t)counts[i] * INFERENCE_BYTES_PER_ELEM);
        inference_buf_sync_to_device(bufs[i]);
    }
    uint64_t ap = inference_buf_phys(bufs[0]), bp = inference_buf_phys(bufs[1]);
    uint64_t cp = inference_buf_phys(bufs[2]), pp = inference_buf_phys(bufs[3]);

    for (i = 0; i < warmup; i++) {
        run_once(&k, ap, bp, cp, pp, r);
        inference_buf_sync_from_device(bufs[2]);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < iters; i++) {
        run_once(&k, ap, bp, cp, pp, r);
        inference_buf_sync_from_device(bufs[2]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (double)(t1.tv_sec  - t0.tv_sec ) * 1e3
               + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double lat = ms / (double)iters;
    double moved = (double)a_bytes + (double)((op & RO_B_EN) ? b_bytes : 0u) + (double)c_bytes
                 + (double)(code == 7u ? 0u : p_bytes);
    double gbs = moved / (lat * 1e-3) / 1e9;

    printf("{\"kernel\":\"RowOpsKernel\",\"label\":\"%s\",\"op\":%u,\"n\":%u,\"rows\":%u,\"blocks\":%u,"
           "\"a_bytes\":%u,\"b_bytes\":%u,\"c_bytes\":%u,\"p_bytes\":%u,\"iters\":%u,"
           "\"lat_ms\":%.4f,\"gbs\":%.3f,\"phys\":%s}\n",
           label, op, n, rows, blocks, a_bytes, (op & RO_B_EN) ? b_bytes : 0u, c_bytes, p_bytes, iters,
           lat, gbs, inference_buf_phys_json(4, bufs));

    inference_buf_free_layout(4, bufs, pool);
    return 0;
}
