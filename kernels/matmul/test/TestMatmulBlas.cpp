// ---------------------------------------------------------------------------
// TestMatmulBlas.cpp — validates MatmulKernel (the configured Data_t build,
// ap_fixed<16,8> by default) against cblas_sgemm, bit-exactly.
//
// Exactness argument.  Every input is an integer multiple of 2^-8 with
// magnitude <= kInputMax = 64 * 2^-8 = 0.25, so
//
//   * each product a*b is a multiple of 2^-16 with |a*b| <= 2^-4,
//   * every partial sum over K <= kMaxK (2048) terms is a multiple of 2^-16
//     with magnitude <= K * 2^-4 (256 at K = 4096), i.e. an integer of
//     magnitude <= 2^24 in units of 2^-16, which float holds exactly, so cblas_sgemm's result is exact whatever its summation
//     order or FMA usage,
//   * the kernel's AccData_t (ap_fixed<32,16>) holds the same values
//     exactly, so its accumulation is exact too.
//
// The only inexact step is the kernel's final narrowing to Data_t
// (saturate_cast: AP_TRN / AP_SAT).  The reference applies the same
// saturate_cast to the exact float sum, after which the two results must be
// bit-identical.  Two dedicated cases drive the sum to exactly -128 (the
// most negative Data_t value, stored as is) and +128 (saturated to the
// largest positive value) to cover the saturation path.
//
// With Data_t = float the same bound makes the kernel's own accumulation
// exact, so the comparison stays bit-exact for that configuration as well.
//
// cblas_sgemm call convention (row-major, no transpose):
//   C[n×m] = alpha * A[n×k] * B[k×m] + beta * C[n×m]
//   α=1, β=0  →  C = A × B
//   lda = k,  ldb = m,  ldc = m
//
// Test matrix:
//   2D shapes  : 1×1×1, tile-exact, partial-tile (N/M/K independently and
//                all together), arbitrary small, unit N, unit M, multi-tile
//   Large K    : N=8, K=512, M=32; K=kMaxK/2; K=kMaxK
//   Saturation : constant inputs, K=kMaxK, sums of exactly -128 and +128
//   Batch      : no broadcast, A broadcasts, B broadcasts
//   Packed B   : tile-major constant-weight layout (b_packed=1)
// ---------------------------------------------------------------------------

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "MatmulKernel.h"

// Every partial sum is an integer of magnitude <= K * 64 * 64 in units of 2^-16;
// float holds every integer up to 2^24 exactly.
static_assert(kMaxK * 64u * 64u <= (1u << 24), "exactness bound: K * 64 * 64 must not exceed 2^24");

// Inputs are i * 2^-8 with |i| <= kInputMaxUnits.
static constexpr int kInputMaxUnits = 64;
static constexpr float kInputUnit    = 1.0f / 256.0f;

enum class Fill { Random, ConstPos, ConstNeg };

// ---------------------------------------------------------------------------
// Element helpers.
// ---------------------------------------------------------------------------
static void fill_inputs(std::vector<float>& v, Fill fill, std::default_random_engine& rng)
{
    std::uniform_int_distribution<int> dist(-kInputMaxUnits, kInputMaxUnits);
    for (auto& x : v) {
        switch (fill) {
            case Fill::Random:   x = static_cast<float>(dist(rng)) * kInputUnit; break;
            case Fill::ConstPos: x = static_cast<float>(kInputMaxUnits) * kInputUnit; break;
            case Fill::ConstNeg: x = -static_cast<float>(kInputMaxUnits) * kInputUnit; break;
        }
    }
}

static std::vector<Data_t> to_data(const std::vector<float>& v)
{
    std::vector<Data_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = Data_t(v[i]);
    return out;
}

// The kernel's exact accumulator value, narrowed exactly as the kernel does.
static Data_t quantise_like_kernel(float exact_sum)
{
    return saturate_cast<Data_t>(AccData_t(exact_sum));
}

// A / B are 128-bit word ports: pack the element vectors into MatmulWord
// arrays (row-major layout unchanged; one spare word so the kernel's last
// partial-word read of a row stays inside the buffer).
static std::vector<MatmulWord> to_words(const std::vector<Data_t>& e)
{
    std::vector<MatmulWord> out(e.size() / kMatmulPortElems + 1);
    for (auto& wd : out) wd = 0;
    for (size_t i = 0; i < e.size(); i++) {
        const unsigned lane = (unsigned)(i % kMatmulPortElems);
        out[i / kMatmulPortElems].range(kMatmulDataBits * (lane + 1) - 1, kMatmulDataBits * lane)
            = matmul_data_to_lane(e[i]);
    }
    return out;
}

// Tile-major packed image of B (MatmulKernel.h "Packed (tile-major) B
// layout"): every batch slice of k*m becomes k*packed_m elements.
static std::vector<Data_t> pack_b_tile_major(const std::vector<Data_t>& B,
                                             unsigned k, unsigned m,
                                             unsigned batch, unsigned b_stride,
                                             unsigned& packed_stride)
{
    const unsigned pm     = matmul_packed_m(m);
    const unsigned slices = (b_stride == 0) ? 1u : batch;
    packed_stride         = (b_stride == 0) ? 0u : k * pm;
    std::vector<Data_t> out((size_t)slices * k * pm, Data_t(0));
    for (unsigned s = 0; s < slices; s++)
        for (unsigned kk = 0; kk < k; kk++)
            for (unsigned mm = 0; mm < m; mm++)
                out[(size_t)s * k * pm + matmul_packed_index(kk, mm, k)] =
                    B[(size_t)s * b_stride + (size_t)kk * m + mm];
    return out;
}

// ---------------------------------------------------------------------------
// Bit-exact comparison; reports the first mismatch.
// ---------------------------------------------------------------------------
static bool compare_outputs(const std::vector<Data_t>& ref,
                            const std::vector<Data_t>& got,
                            const char* label)
{
    unsigned mismatches = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (matmul_data_to_lane(ref[i]) != matmul_data_to_lane(got[i])) {
            if (mismatches == 0) {
                printf("  FAIL  [%zu] blas=%.8f  kernel=%.8f\n",
                       i, static_cast<double>(ref[i]), static_cast<double>(got[i]));
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("  PASS  %s\n", label);
        return true;
    }
    printf("  FAIL  %s  (%u/%zu elements differ)\n", label, mismatches, ref.size());
    return false;
}

// ---------------------------------------------------------------------------
// RunTest — batched product via cblas_sgemm (looped) vs MatmulKernel.
// a_stride / b_stride of 0 broadcast that operand across the batch.
// ---------------------------------------------------------------------------
static bool RunTest(const char* label,
                    unsigned n, unsigned k, unsigned m,
                    unsigned batch, unsigned a_stride, unsigned b_stride,
                    unsigned b_packed = 0,
                    Fill fill = Fill::Random,
                    unsigned seed = kSeed)
{
    const unsigned a_total  = (a_stride == 0) ? n * k : batch * a_stride;
    const unsigned b_total  = (b_stride == 0) ? k * m : batch * b_stride;
    const unsigned c_stride = n * m;

    std::vector<float> A(a_total), B(b_total);
    std::vector<float> C_blas(batch * c_stride, 0.0f);

    std::default_random_engine rng(seed);
    // Constant fills: A is +0.25 everywhere, B is +0.25 (ConstPos) or -0.25
    // (ConstNeg), so every sum is +(K/16) or -(K/16).
    fill_inputs(A, fill == Fill::Random ? Fill::Random : Fill::ConstPos, rng);
    fill_inputs(B, fill, rng);

    // BLAS reference: one call per batch element (cblas_sgemm is not batched).
    for (unsigned bi = 0; bi < batch; bi++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                    1.0f,
                    A.data() + bi * a_stride, static_cast<int>(k),
                    B.data() + bi * b_stride, static_cast<int>(m),
                    0.0f,
                    C_blas.data() + bi * c_stride, static_cast<int>(m));
    }
    std::vector<Data_t> C_ref(C_blas.size());
    for (size_t i = 0; i < C_blas.size(); i++) C_ref[i] = quantise_like_kernel(C_blas[i]);

    // Kernel: element vectors → 128-bit words, optional tile-major B.
    std::vector<Data_t> A_d = to_data(A), B_d = to_data(B);
    unsigned b_stride_eff = b_stride;
    if (b_packed) B_d = pack_b_tile_major(B_d, k, m, batch, b_stride, b_stride_eff);
    std::vector<MatmulWord> aw = to_words(A_d), bw = to_words(B_d);
    std::vector<Data_t> C_kern(batch * c_stride, Data_t(0));

    MatmulKernel(aw.data(), bw.data(), C_kern.data(),
                 n, k, m, batch, a_stride, b_stride_eff, c_stride, b_packed);

    return compare_outputs(C_ref, C_kern, label);
}

static bool RunTest2D(const char* label, unsigned n, unsigned k, unsigned m,
                      unsigned b_packed = 0, Fill fill = Fill::Random)
{
    return RunTest(label, n, k, m, /*batch=*/1,
                   /*a_stride=*/n * k, /*b_stride=*/k * m, b_packed, fill);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    bool all_ok = true;
    int  total  = 0;
    int  passed = 0;

    printf("MatmulKernel BLAS comparison tests (bit-exact, inputs i * 2^-8, |i| <= %d)\n",
           kInputMaxUnits);
    printf("  Data_t=%u bits  kMatmulPortElems=%u\n", kMatmulDataBits, kMatmulPortElems);
    printf("  kTileN=%u  kTileM=%u  kTileK=%u  kMaxK=%u\n\n",
           kTileN, kTileM, kTileK, kMaxK);

    auto run = [&](bool ok) { total++; if (ok) passed++; else all_ok = false; };

    // -----------------------------------------------------------------------
    // 2-D shape tests
    // -----------------------------------------------------------------------
    printf("--- 2D shapes ---\n");

    run(RunTest2D("1x1x1",           1,           1,      1));
    run(RunTest2D("TileN x TileK x TileM",
                  kTileN,       kTileK,      kTileM));
    run(RunTest2D("(TileN+2) x TileK x TileM  [partial N]",
                  kTileN + 2,   kTileK,      kTileM));
    run(RunTest2D("TileN x (TileK+5) x TileM  [partial K]",
                  kTileN,       kTileK + 5,  kTileM));
    run(RunTest2D("TileN x TileK x (TileM+3)  [partial M]",
                  kTileN,       kTileK,      kTileM + 3));
    run(RunTest2D("(TileN+2) x (TileK+5) x (TileM+3)  [all partial]",
                  kTileN + 2,   kTileK + 5,  kTileM + 3));
    run(RunTest2D("7 x 13 x 5  [arbitrary small]",
                  7,            13,          5));
    run(RunTest2D("1 x K x M  [N=1 row vector]",
                  1,            kTileK,      kTileM));
    run(RunTest2D("N x K x 1  [M=1 column vector]",
                  kTileN,       kTileK,      1));
    run(RunTest2D("3*TileN x (2*TileK+7) x (2*TileM+1)  [multi-tile all]",
                  kTileN * 3,   kTileK * 2 + 7,  kTileM * 2 + 1));

    // -----------------------------------------------------------------------
    // Large-K tests — long accumulations, still exact (see header).
    // -----------------------------------------------------------------------
    printf("\n--- Large K ---\n");

    run(RunTest2D("8 x 512 x 32  [large K]",    8,  512, 32));
    run(RunTest2D("16 x kMaxK/2 x 16",          16, kMaxK / 2, 16));
    run(RunTest2D("4 x kMaxK x 40  [K=kMaxK]",  4,  kMaxK, 40));

    // -----------------------------------------------------------------------
    // Saturation — constant inputs +-0.25, K=kMaxK: every sum is
    // +-(kMaxK / 16).  With kMaxK = 2048 that is -128 (representable, stored
    // as is) and +128 (saturated to the largest Data_t).
    // -----------------------------------------------------------------------
    printf("\n--- Saturation ---\n");

    // K = 2048 puts the sums exactly on the Data_t limits (-128 stored,
    // +128 saturated); K = kMaxK (> 2048) saturates both ways.
    const unsigned KSAT = kMaxK < 2048u ? kMaxK : 2048u;
    run(RunTest2D("3 x 2048 x 5  [sum = +128, saturates]",   3, KSAT, 5, 0, Fill::ConstPos));
    run(RunTest2D("3 x 2048 x 5  [sum = -128, stored]",      3, KSAT, 5, 0, Fill::ConstNeg));
    run(RunTest2D("3 x kMaxK x 5  [sum = -kMaxK/16]",         3, kMaxK, 5, 0, Fill::ConstNeg));

    // -----------------------------------------------------------------------
    // Batch tests
    // -----------------------------------------------------------------------
    printf("\n--- Batch ---\n");

    const unsigned BN = kTileN + 1, BK = kTileK / 4, BM = kTileM + 3;

    run(RunTest("batch=3, no broadcast",
                BN, BK, BM, /*batch=*/3,
                /*a_stride=*/BN * BK, /*b_stride=*/BK * BM));
    run(RunTest("batch=4, A broadcasts (a_stride=0)",
                BN, BK, BM, /*batch=*/4,
                /*a_stride=*/0, /*b_stride=*/BK * BM));
    run(RunTest("batch=4, B broadcasts (b_stride=0)",
                BN, BK, BM, /*batch=*/4,
                /*a_stride=*/BN * BK, /*b_stride=*/0));

    // -----------------------------------------------------------------------
    // Packed (tile-major) B
    // -----------------------------------------------------------------------
    printf("\n--- Packed B ---\n");

    run(RunTest2D("TileN x TileK x TileM  [packed]",
                  kTileN, kTileK, kTileM, 1));
    run(RunTest2D("(TileN+2) x (TileK+5) x (TileM+3)  [packed, all partial]",
                  kTileN + 2, kTileK + 5, kTileM + 3, 1));
    run(RunTest2D("1 x kMaxK x 1001  [packed, FC-style]",
                  1, kMaxK, 1001, 1));
    run(RunTest("batch=3, packed, no broadcast",
                BN, BK, BM, /*batch=*/3,
                /*a_stride=*/BN * BK, /*b_stride=*/BK * BM, 1));
    run(RunTest("batch=4, packed, B broadcasts (b_stride=0)",
                BN, BK, BM, /*batch=*/4,
                /*a_stride=*/BN * BK, /*b_stride=*/0, 1));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%d/%d tests passed\n", passed, total);
    if (!all_ok) {
        printf("TestMatmulBlas FAILED\n");
        return 1;
    }
    printf("TestMatmulBlas PASSED\n");
    return 0;
}
