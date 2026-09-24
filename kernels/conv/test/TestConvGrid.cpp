// ---------------------------------------------------------------------------
// TestConvGrid.cpp — unit test for the ConvKernel MAC array in isolation.
//
// Exercises accumulate_standard / accumulate_depthwise (include/ConvMacGrid.h)
// on random tiles for every (kh, kw) up to the compile-time maxima, every
// partial ic_valid, and random starting accumulators, against a scalar
// triple loop.  Exact match required: AccData_t is wrap-around fixed point,
// so any reduction order is bit-identical (CONV_2D_GRID_PLAN.md §4).
//
// This is the gate for changes to the grid itself (the 1-D → 2-D widening):
// it runs in well under a second and needs no geometry at all.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <random>
#include "ConvKernel.h"
#include "ConvMacGrid.h"

static std::mt19937 rng(kSeed);

static Data_t rnd(float scale)
{
    std::uniform_real_distribution<float> d(-scale, scale);
    return Data_t(d(rng));
}

static int test_standard(unsigned kh, unsigned kw, unsigned ic_valid, unsigned m_valid)
{
    Data_t    patch[kTileIC][kMaxKH][kMaxKW];
    Data_t    w_buf[kTileM][kTileIC][kMaxKH][kMaxKW];
    AccData_t acc[kTileM], ref[kTileM];

    for (unsigned c = 0; c < kTileIC; c++)
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++) {
                // Lanes >= ic_valid carry the producer's zero pad in the real
                // kernel; here give them garbage so the ic_valid guard is tested.
                patch[c][i][j] = (c < ic_valid) ? rnd(2.0f) : rnd(50.0f);
                for (unsigned m = 0; m < kTileM; m++)
                    w_buf[m][c][i][j] = rnd(1.0f);
            }
    for (unsigned m = 0; m < kTileM; m++) {
        acc[m] = AccData_t(rnd(8.0f));
        ref[m] = acc[m];
    }

    // Scalar reference: same wrap-around accumulator type, any order.
    // Lanes m >= m_valid must come back untouched (masked weights).
    for (unsigned m = 0; m < m_valid; m++)
        for (unsigned c = 0; c < ic_valid; c++)
            for (unsigned i = 0; i < kh; i++)
                for (unsigned j = 0; j < kw; j++)
                    ref[m] += patch[c][i][j] * w_buf[m][c][i][j];

    accumulate_standard(patch, w_buf, acc, ic_valid, m_valid, kh, kw);

    int bad = 0;
    for (unsigned m = 0; m < kTileM; m++)
        if (acc[m] != ref[m]) {
            if (bad < 3)
                std::printf("  STD kh=%u kw=%u ic_valid=%u m_valid=%u lane %u: got %f ref %f\n",
                            kh, kw, ic_valid, m_valid, m, (double)acc[m], (double)ref[m]);
            bad++;
        }
    return bad;
}

static int test_depthwise(unsigned kh, unsigned kw)
{
    Data_t    patch[kTileIC][kMaxKH][kMaxKW];
    Data_t    w_buf[kTileM][kMaxKH][kMaxKW];
    AccData_t acc[kTileM], ref[kTileM];

    for (unsigned c = 0; c < kTileIC; c++)
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++)
                patch[c][i][j] = rnd(2.0f);
    for (unsigned m = 0; m < kTileM; m++) {
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++)
                w_buf[m][i][j] = rnd(1.0f);
        acc[m] = AccData_t(rnd(8.0f));
        ref[m] = acc[m];
    }
    for (unsigned m = 0; m < kTileM; m++)
        for (unsigned i = 0; i < kh; i++)
            for (unsigned j = 0; j < kw; j++)
                ref[m] += patch[m][i][j] * w_buf[m][i][j];

    accumulate_depthwise(patch, w_buf, acc, kh, kw);

    int bad = 0;
    for (unsigned m = 0; m < kTileM; m++)
        if (acc[m] != ref[m]) {
            if (bad < 3)
                std::printf("  DW kh=%u kw=%u lane %u: got %f ref %f\n",
                            kh, kw, m, (double)acc[m], (double)ref[m]);
            bad++;
        }
    return bad;
}

int main(int argc, char** argv)
{
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 3;
    int failures = 0, cases = 0;
    std::printf("ConvKernel MAC-grid unit test  (TILE_M=%u TILE_IC=%u MAX_KH=%u MAX_KW=%u, %d rep(s))\n",
                kTileM, kTileIC, kMaxKH, kMaxKW, reps);
    for (int r = 0; r < reps; r++) {
        for (unsigned kh = 1; kh <= kMaxKH; kh++)
            for (unsigned kw = 1; kw <= kMaxKW; kw++) {
                for (unsigned icv = 1; icv <= kTileIC; icv++)
                    for (unsigned mv = 1; mv <= kTileM; mv++) {
                        failures += test_standard(kh, kw, icv, mv); cases++;
                    }
                failures += test_depthwise(kh, kw); cases++;
            }
    }
    std::printf("%d grid cases, %d lane mismatch(es)\n", cases, failures);
    std::printf(failures == 0 ? "ALL TESTS PASSED\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
