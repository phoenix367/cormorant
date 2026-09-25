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
    // One 16-lane word per (m, khi, kwi); the cache is two column arrays
    // (BRAM / URAM halves, §2.40) with identical addressing.
    WeightVec w_lo[kWCacheBramCols][kWCacheWords];
    WeightVec w_hi[kWCacheUramCols][kWCacheWords];
    auto wcol = [&](unsigned m, unsigned a) -> WeightVec& {
        return (m < kWCacheBramCols) ? w_lo[m][a] : w_hi[m - kWCacheBramCols][a];
    };
    AccData_t acc[kTileM], ref[kTileM];

    for (unsigned c = 0; c < kTileIC; c++)
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++) {
                // Lanes >= ic_valid: the producer zero-pads the PATCH lanes
                // and the packed layout / weight producer zero the WEIGHT
                // lanes (§2.40: the grid multiplies them unmasked).  Give the
                // patch garbage and the weight the contract's zero so the
                // product is 0 exactly as in the kernel.
                patch[c][i][j] = (c < ic_valid) ? rnd(2.0f) : rnd(50.0f);
                for (unsigned m = 0; m < kTileM; m++)
                    wcol(m, w_cache_addr(0, 0, i, j)).lane[c] =
                        (c < ic_valid) ? rnd(1.0f) : Data_t(0);
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
                    ref[m] += patch[c][i][j] * wcol(m, w_cache_addr(0, 0, i, j)).lane[c];

    accumulate_standard(patch, w_lo, w_hi, acc, ic_valid, m_valid, kh, kw);

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

// §2.42: two pixels against one weight read, with the accumulator seeds
// entering the adder trees at DIFFERENT positions (pixel 0 at position 0,
// pixel 1 at position 1).  The result must equal the plain seeded sweep.
static int test_pair(unsigned kh, unsigned kw, unsigned m_valid)
{
    Data_t    patch[2][kTileIC][kMaxKH][kMaxKW];
    WeightVec w_lo[kWCacheBramCols][kWCacheWords];
    WeightVec w_hi[kWCacheUramCols][kWCacheWords];
    auto wcol = [&](unsigned m, unsigned a) -> WeightVec& {
        return (m < kWCacheBramCols) ? w_lo[m][a] : w_hi[m - kWCacheBramCols][a];
    };
    AccData_t seed[2][kTileM], acc[2][kTileM], ref[2][kTileM];

    for (unsigned c = 0; c < kTileIC; c++)
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++) {
                patch[0][c][i][j] = rnd(2.0f);
                patch[1][c][i][j] = rnd(2.0f);
                for (unsigned m = 0; m < kTileM; m++)
                    wcol(m, w_cache_addr(0, 0, i, j)).lane[c] = rnd(1.0f);
            }
    for (unsigned h = 0; h < 2; h++)
        for (unsigned m = 0; m < kTileM; m++) {
            seed[h][m] = AccData_t(rnd(8.0f));
            // Lanes >= m_valid: the masked columns add nothing, so the
            // accumulator keeps the seed (as the kernel's word does).
            ref[h][m]  = seed[h][m];
            acc[h][m]  = 0;
        }
    for (unsigned h = 0; h < 2; h++)
        for (unsigned m = 0; m < m_valid; m++)
            for (unsigned c = 0; c < kTileIC; c++)
                for (unsigned i = 0; i < kh; i++)
                    for (unsigned j = 0; j < kw; j++)
                        ref[h][m] += patch[h][c][i][j] * wcol(m, w_cache_addr(0, 0, i, j)).lane[c];

    // The kernel's pair window: max(kh*kw, 2) positions, seeds at 0 / 1.
    const unsigned n_pos = kh * kw, n_win = (n_pos < 2) ? 2 : n_pos;
    unsigned khi = 0, kwi = 0;
    for (unsigned pos = 0; pos < n_win; pos++) {
        const bool at_pos = pos < n_pos;
        Data_t p0[kTileIC], p1[kTileIC];
        for (unsigned c = 0; c < kTileIC; c++) {
            p0[c] = at_pos ? patch[0][c][khi][kwi] : Data_t(0);
            p1[c] = at_pos ? patch[1][c][khi][kwi] : Data_t(0);
        }
        WeightVec w[kTileM];
        w_cache_read(w_lo, w_hi, w_cache_addr(0, 0, khi, kwi), m_valid, w);
        mac_grid_column_step(p0, w, seed[0], pos == 0, acc[0]);
        mac_grid_column_step(p1, w, seed[1], pos == 1, acc[1]);
        if (pos + 1 < n_pos && ++kwi == kw) { kwi = 0; khi++; }
    }

    int bad = 0;
    for (unsigned h = 0; h < 2; h++)
        for (unsigned m = 0; m < kTileM; m++)
            if (acc[h][m] != ref[h][m]) {
                if (bad < 3)
                    std::printf("  PAIR kh=%u kw=%u m_valid=%u px %u lane %u: got %f ref %f\n",
                                kh, kw, m_valid, h, m, (double)acc[h][m], (double)ref[h][m]);
                bad++;
            }
    return bad;
}

static int test_depthwise(unsigned kh, unsigned kw)
{
    Data_t    patch[kTileIC][kMaxKH][kMaxKW];
    Data_t    w_buf[kTileM][kMaxKPos];     // flat over the window: pos = i*kw + j
    AccData_t acc[kTileM], ref[kTileM];

    for (unsigned c = 0; c < kTileIC; c++)
        for (unsigned i = 0; i < kMaxKH; i++)
            for (unsigned j = 0; j < kMaxKW; j++)
                patch[c][i][j] = rnd(2.0f);
    for (unsigned m = 0; m < kTileM; m++) {
        for (unsigned q = 0; q < kMaxKPos; q++)
            w_buf[m][q] = rnd(1.0f);
        acc[m] = AccData_t(rnd(8.0f));
        ref[m] = acc[m];
    }
    for (unsigned m = 0; m < kTileM; m++)
        for (unsigned i = 0; i < kh; i++)
            for (unsigned j = 0; j < kw; j++)
                ref[m] += patch[m][i][j] * w_buf[m][i * kw + j];

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
                for (unsigned mv = 1; mv <= kTileM; mv++) {
                    failures += test_pair(kh, kw, mv); cases++;
                }
            }
    }
    std::printf("%d grid cases, %d lane mismatch(es)\n", cases, failures);
    std::printf(failures == 0 ? "ALL TESTS PASSED\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
