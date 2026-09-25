// ---------------------------------------------------------------------------
// TestMatmulSim.cpp — reference tests for MatmulKernel.
//
// Each test computes the same matrix product with two independent implementations:
//
//   ref_matmul_2d()  — naive triple-nested loop, the ground-truth oracle.
//                      Uses AccData_t for accumulation and saturate_cast<Data_t>
//                      for output, exactly matching the kernel's arithmetic.
//
//   MatmulKernel()   — the tiled reference that mimics the future HLS structure.
//
// Outputs are compared element-by-element with zero tolerance for fixed-point
// types (both implementations produce bitwise-identical results when they share
// the same accumulator type and saturation policy).  For float/double Data_t the
// comparison allows a 1 ULP tolerance since floating-point addition is not
// strictly associative and the tile ordering may reorder operations.
//
// Test matrix:
//   2D shapes:
//     1×1×1              degenerate minimum
//     TILE_N×TILE_K×TILE_M  exactly fills one tile in every dimension
//     (TILE_N+2)×TILE_K×TILE_M  partial last N-tile
//     TILE_N×TILE_K×(TILE_M+3)  partial last M-tile
//     TILE_N×(TILE_K+5)×TILE_M  partial last K-tile (spans 2 K-tiles)
//     (TILE_N+2)×(TILE_K+5)×(TILE_M+3)  all dimensions span 2 tiles
//     7×13×5             arbitrary small dims (all below tile sizes)
//     1×K×M, N×K×1       unit N / unit M (degenerate row/column)
//     3×TILE_N×(TILE_K*2+7)×(TILE_M*2+1)  multiple tiles in all dimensions
//   Batch:
//     batch=3, no broadcast
//     batch=4, a_batch_stride=0 (A broadcasts)
//     batch=4, b_batch_stride=0 (B broadcasts)
//   Saturation:
//     positive: all a=100, all b=100, K=3  → sum=30000 → saturate to AP_MAX
//     negative: all a=100, all b=-100, K=3 → sum=-30000 → saturate to AP_MIN
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "MatmulKernel.h"

// ---------------------------------------------------------------------------
// --dump-data <dir> mode: instead of running MatmulKernel and comparing,
// dump the per-test A/B/C_ref tensors to hex files (one 16-bit value per
// line, suitable for $readmemh).  A manifest.txt indexes every test with
// its geometry so an HDL testbench can load the same fixtures.  RNG state
// is shared with verify mode (same seed, same draw order).
// ---------------------------------------------------------------------------
static std::string g_dump_dir;     // empty → verify mode (default)
static int         g_test_idx = 0;
static FILE*       g_manifest = nullptr;

// ---------------------------------------------------------------------------
// Scalar limits for Data_t (derived via saturate_cast, same trick as
// TestSimulation.cpp — avoids std::numeric_limits which fails for ap_fixed
// under regular GCC).
// ---------------------------------------------------------------------------
static const double kSatMax = static_cast<double>(saturate_cast<Data_t>( 1e30));
static const double kSatMin = static_cast<double>(saturate_cast<Data_t>(-1e30));

// ---------------------------------------------------------------------------
// Naive reference: 2D matmul, ground-truth oracle.
//
// Uses AccData_t for the K-reduction sum and saturate_cast<Data_t> for the
// output — identical arithmetic policy to the tiled kernel.  The naive loop
// order (n, m, k) differs from the tiled kernel's (tile-n, tile-m, tile-k, ki)
// but produces the same result because both use the same accumulator type and
// saturate only once at the very end of the K-loop, not during accumulation.
// ---------------------------------------------------------------------------
static void ref_matmul_2d(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      n,
    unsigned      k,
    unsigned      m)
{
    for (unsigned ni = 0; ni < n; ni++) {
        for (unsigned mi = 0; mi < m; mi++) {
            AccData_t sum = AccData_t(0);
            for (unsigned ki = 0; ki < k; ki++) {
                sum += AccData_t(a[ni * k + ki]) * AccData_t(b[ki * m + mi]);
            }
            c[ni * m + mi] = saturate_cast<Data_t>(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch wrapper for the naive reference.
// ---------------------------------------------------------------------------
static void ref_matmul_batch(
    const Data_t* a,
    const Data_t* b,
    Data_t*       c,
    unsigned      n,
    unsigned      k,
    unsigned      m,
    unsigned      batch,
    unsigned      a_stride,
    unsigned      b_stride,
    unsigned      c_stride)
{
    for (unsigned bi = 0; bi < batch; bi++) {
        ref_matmul_2d(a + bi * a_stride, b + bi * b_stride, c + bi * c_stride,
                      n, k, m);
    }
}

// ---------------------------------------------------------------------------
// Dump-mode helpers (only meaningful for fixed-point builds — the HDL
// testbench reads 16-bit hex values one per line).
// ---------------------------------------------------------------------------
#ifdef MATMUL_HAVE_APFIXED
static uint16_t data_to_raw16(const Data_t& v) {
    return static_cast<uint16_t>(v.range().to_uint());
}

static void write_hex_file(const std::string& path,
                           const std::vector<Data_t>& vec) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        std::exit(1);
    }
    for (const auto& v : vec)
        std::fprintf(f, "%04x\n", data_to_raw16(v));
    std::fclose(f);
}
#endif

static std::string sanitize_label(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-') out.push_back(c);
        else                                          out.push_back('_');
    }
    return out;
}

// Write A/B/C_ref hex files and a manifest row for one test case.
static void dump_one_case(const char* label,
                          unsigned n, unsigned k, unsigned m,
                          unsigned batch,
                          unsigned a_stride, unsigned b_stride,
                          const std::vector<Data_t>& a,
                          const std::vector<Data_t>& b,
                          const std::vector<Data_t>& c_ref,
                          unsigned b_packed = 0) {
#ifndef MATMUL_HAVE_APFIXED
    (void)label; (void)n; (void)k; (void)m; (void)b_packed;
    (void)batch; (void)a_stride; (void)b_stride;
    (void)a; (void)b; (void)c_ref;
    std::fprintf(stderr, "--dump-data requires MATMUL_HAVE_APFIXED build\n");
    std::exit(1);
#else
    const int idx = g_test_idx++;
    char idx_buf[16];
    std::snprintf(idx_buf, sizeof(idx_buf), "%02d", idx);
    const std::string prefix = g_dump_dir + "/test_" + idx_buf + "_";

    write_hex_file(prefix + "a.hex", a);
    write_hex_file(prefix + "b.hex", b);
    write_hex_file(prefix + "c.hex", c_ref);

    std::fprintf(g_manifest, "%d %u %u %u %u %u %u %u %s\n",
                 idx, n, k, m, batch, a_stride, b_stride, b_packed,
                 sanitize_label(label).c_str());
    std::printf("[DUMP] test_%02d  %-50s  N=%u K=%u M=%u batch=%u\n",
                idx, label, n, k, m, batch);
#endif
}

// ---------------------------------------------------------------------------
// Element comparison.
//
// For fixed-point types both implementations are bitwise-identical → zero tol.
// For float/double, associativity differences can shift the last bit → 1-ULP tol.
// ---------------------------------------------------------------------------
static bool is_close(Data_t ref_val, Data_t got_val) {
#ifdef MATMUL_HAVE_APFIXED
    return ref_val == got_val;
#else
    // Allow 1 ULP for floating-point types.
    const double r = static_cast<double>(ref_val);
    const double g = static_cast<double>(got_val);
    const double scale = std::max(std::abs(r), std::abs(g));
    if (scale < 1e-12) return std::abs(r - g) < 1e-12;
    return std::abs(r - g) / scale < 1e-5;
#endif
}

// ---------------------------------------------------------------------------
// Compare two output arrays, print first mismatch, return pass/fail.
// ---------------------------------------------------------------------------
static bool compare_outputs(
    const Data_t* ref,
    const Data_t* got,
    unsigned      count,
    const char*   label)
{
    unsigned mismatches = 0;
    for (unsigned i = 0; i < count; i++) {
        if (!is_close(ref[i], got[i])) {
            if (mismatches == 0) {
                printf("  FAIL  [%u] ref=%.6f  got=%.6f\n",
                       i,
                       static_cast<double>(ref[i]),
                       static_cast<double>(got[i]));
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("  PASS  %s\n", label);
        return true;
    }
    printf("  FAIL  %s  (%u/%u elements wrong)\n", label, mismatches, count);
    return false;
}

// ---------------------------------------------------------------------------
// invoke_matmul — call MatmulKernel, routing through the cosim fixed buffers.
//
// Plain C-sim passes the per-test std::vector storage straight to the kernel.
// The cosim build (-DMATMUL_COSIM, set by scripts/Cosim.tcl.in) instead copies
// each case into the fixed MATMUL_COSIM_DEPTH_*-sized globals: cosim's wrapc
// adapter sizes its RTL memory model from the kernel's m_axi depth= hint and
// transfers the whole buffer, so a smaller per-test allocation would be read
// past its end.  Cases whose matrices exceed the cosim buffers are skipped and
// left to plain C-sim.
//
// Returns true if the kernel ran (caller then compares C_got); false if the
// case was skipped (the caller should report it as a pass).
// ---------------------------------------------------------------------------
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
// Returns the packed vector and the packed batch stride.
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

#ifdef MATMUL_COSIM
static MatmulWord g_a[MATMUL_COSIM_DEPTH_A_WORDS];
static MatmulWord g_b[MATMUL_COSIM_DEPTH_B_WORDS];
static Data_t     g_c[MATMUL_COSIM_DEPTH_C];
#endif

static bool invoke_matmul(const char* label,
                          const std::vector<Data_t>& A,
                          const std::vector<Data_t>& B,
                          std::vector<Data_t>&       C_got,
                          unsigned n, unsigned k, unsigned m, unsigned batch,
                          unsigned a_stride, unsigned b_stride,
                          unsigned c_stride, unsigned b_packed = 0)
{
    (void)label;  // used only by the MATMUL_COSIM skip message below
#ifdef MATMUL_COSIM
    if (A.size() > MATMUL_COSIM_DEPTH_A ||
        B.size() > MATMUL_COSIM_DEPTH_B ||
        C_got.size() > MATMUL_COSIM_DEPTH_C) {
        printf("  SKIP  %s  (exceeds cosim buffers)\n", label);
        return false;
    }
    {
        const std::vector<MatmulWord> aw = to_words(A), bw = to_words(B);
        std::copy(aw.begin(), aw.end(), g_a);
        std::copy(bw.begin(), bw.end(), g_b);
    }
    MatmulKernel(g_a, g_b, g_c, n, k, m, batch, a_stride, b_stride, c_stride, b_packed);
    std::copy(g_c, g_c + C_got.size(), C_got.begin());
#else
    std::vector<MatmulWord> aw = to_words(A), bw = to_words(B);
    MatmulKernel(aw.data(), bw.data(), C_got.data(),
                 n, k, m, batch, a_stride, b_stride, c_stride, b_packed);
#endif
    return true;
}

// ---------------------------------------------------------------------------
// RunTest2D — fill A, B with deterministic random values, compare ref vs kernel.
//
// Inputs are drawn from [-1, 1] to keep accumulator sums well within
// AccData_t range regardless of K.
// ---------------------------------------------------------------------------
static bool RunTest2D(const char* label, unsigned n, unsigned k, unsigned m,
                      unsigned seed = kSeed, unsigned b_packed = 0)
{
    std::vector<Data_t> A(n * k), B(k * m);
    std::vector<Data_t> C_ref(n * m, Data_t(0));
    std::vector<Data_t> C_got(n * m, Data_t(0));

    std::default_random_engine rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : A) v = Data_t(dist(rng));
    for (auto& v : B) v = Data_t(dist(rng));

    ref_matmul_2d   (A.data(), B.data(), C_ref.data(), n, k, m);

    unsigned b_stride = k * m;
    if (b_packed) B = pack_b_tile_major(B, k, m, 1u, k * m, b_stride);

    if (!g_dump_dir.empty()) {
        dump_one_case(label, n, k, m, /*batch*/1u,
                      /*a_stride*/ n * k, b_stride,
                      A, B, C_ref, b_packed);
        return true;
    }

    if (!invoke_matmul(label, A, B, C_got,
                       n, k, m, /*batch=*/1,
                       /*a_stride=*/n * k, b_stride,
                       /*c_stride=*/n * m, b_packed))
        return true;  // skipped — case exceeds the cosim buffers

    return compare_outputs(C_ref.data(), C_got.data(), n * m, label);
}

// ---------------------------------------------------------------------------
// RunTestBatch — fill A, B with random values and test batch + broadcasting.
// ---------------------------------------------------------------------------
static bool RunTestBatch(const char* label,
                         unsigned n, unsigned k, unsigned m,
                         unsigned batch,
                         unsigned a_stride, unsigned b_stride,
                         unsigned seed = kSeed, unsigned b_packed = 0)
{
    const unsigned a_total = (a_stride == 0) ? n * k : batch * a_stride;
    const unsigned b_total = (b_stride == 0) ? k * m : batch * b_stride;
    const unsigned c_stride = n * m;

    std::vector<Data_t> A(a_total), B(b_total);
    std::vector<Data_t> C_ref(batch * c_stride, Data_t(0));
    std::vector<Data_t> C_got(batch * c_stride, Data_t(0));

    std::default_random_engine rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : A) v = Data_t(dist(rng));
    for (auto& v : B) v = Data_t(dist(rng));

    ref_matmul_batch(A.data(), B.data(), C_ref.data(),
                     n, k, m, batch, a_stride, b_stride, c_stride);

    unsigned b_stride_eff = b_stride;
    if (b_packed) B = pack_b_tile_major(B, k, m, batch, b_stride, b_stride_eff);

    if (!g_dump_dir.empty()) {
        dump_one_case(label, n, k, m, batch, a_stride, b_stride_eff,
                      A, B, C_ref, b_packed);
        return true;
    }

    if (!invoke_matmul(label, A, B, C_got,
                       n, k, m, batch, a_stride, b_stride_eff, c_stride, b_packed))
        return true;  // skipped — case exceeds the cosim buffers

    return compare_outputs(C_ref.data(), C_got.data(), batch * c_stride, label);
}

// ---------------------------------------------------------------------------
// RunTestSaturation — use constant inputs large enough to force output saturation.
//
// a[*] = a_val, b[*] = b_val, K = k_sat.
// Output = saturate_cast<Data_t>(k_sat × a_val × b_val) for every element.
//
// The expected saturated value is derived by running the naive ref; then we
// additionally verify that it equals kSatMax / kSatMin as appropriate.
// ---------------------------------------------------------------------------
static bool RunTestSaturation(const char* label,
                               double a_val, double b_val, unsigned k_sat,
                               double expected_sat)
{
    const unsigned N = kTileN, M = kTileM;

    std::vector<Data_t> A(N * k_sat, Data_t(a_val));
    std::vector<Data_t> B(k_sat * M, Data_t(b_val));
    std::vector<Data_t> C_ref(N * M, Data_t(0));
    std::vector<Data_t> C_got(N * M, Data_t(0));

    ref_matmul_2d(A.data(), B.data(), C_ref.data(), N, k_sat, M);

    if (!g_dump_dir.empty()) {
        dump_one_case(label, N, k_sat, M, /*batch*/1u,
                      /*a_stride*/ N * k_sat, /*b_stride*/ k_sat * M,
                      A, B, C_ref);
        return true;
    }

    if (!invoke_matmul(label, A, B, C_got,
                       N, k_sat, M, /*batch=*/1,
                       /*a_stride=*/N * k_sat, /*b_stride=*/k_sat * M,
                       /*c_stride=*/N * M))
        return true;  // skipped — case exceeds the cosim buffers

    bool ok = compare_outputs(C_ref.data(), C_got.data(), N * M, label);

    // Verify that the reference actually saturated (not a data-generation mistake).
    const double sat_tol = 1.0 / 256.0;
    for (unsigned i = 0; i < N * M; i++) {
        const double got = static_cast<double>(C_ref[i]);
        if (std::abs(got - expected_sat) > sat_tol) {
            printf("  WARN  %s: element %u = %.6f, expected saturation at %.6f "
                   "(check Data_t range or K size)\n",
                   label, i, got, expected_sat);
            break;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Optional --dump-data <dir>: write per-test A/B/C_ref hex files plus a
    // manifest, then exit (no kernel run).  Otherwise: original verify mode.
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if ((a == "--dump-data" || a == "-d") && i + 1 < argc) {
            g_dump_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--dump-data <dir>]\n", argv[0]);
            return 0;
        }
    }

    if (!g_dump_dir.empty()) {
        const std::string manifest_path = g_dump_dir + "/manifest.txt";
        g_manifest = std::fopen(manifest_path.c_str(), "w");
        if (!g_manifest) {
            std::fprintf(stderr, "Failed to open %s for writing\n",
                         manifest_path.c_str());
            return 1;
        }
        std::fprintf(g_manifest,
            "# MatmulKernel test fixture manifest\n"
            "# idx n k m batch a_stride b_stride b_packed label\n");
    }

    bool all_ok = true;
    int  total  = 0;
    int  passed = 0;

    // Print type information.
    printf("MatmulKernel reference tests\n");
    printf("  Data_t    = %s\n", typeid(Data_t).name());
    printf("  AccData_t = %s\n", typeid(AccData_t).name());
    printf("  kTileN=%u  kTileM=%u  kTileK=%u  kMaxK=%u\n\n",
           kTileN, kTileM, kTileK, kMaxK);

    auto run = [&](bool ok) { total++; if (ok) passed++; else all_ok = false; };

    // -----------------------------------------------------------------------
    // 2-D shape tests
    // -----------------------------------------------------------------------
    printf("--- 2D shapes ---\n");

    // Absolute minimum: 1 element
    run(RunTest2D("1x1x1",                1, 1, 1));

    // Exactly fills one tile in all dimensions
    run(RunTest2D("TileN x TileK x TileM",
                  kTileN, kTileK, kTileM));

    // Each dimension independently spans exactly 2 tiles (no partial)
    run(RunTest2D("2*TileN x TileK x TileM",
                  kTileN * 2, kTileK, kTileM));
    run(RunTest2D("TileN x 2*TileK x TileM",
                  kTileN, kTileK * 2, kTileM));
    run(RunTest2D("TileN x TileK x 2*TileM",
                  kTileN, kTileK, kTileM * 2));

    // Partial last tile in each dimension individually
    run(RunTest2D("(TileN+2) x TileK x TileM  [partial N]",
                  kTileN + 2, kTileK, kTileM));
    run(RunTest2D("TileN x (TileK+5) x TileM  [partial K]",
                  kTileN, kTileK + 5, kTileM));
    run(RunTest2D("TileN x TileK x (TileM+3)  [partial M]",
                  kTileN, kTileK, kTileM + 3));

    // All dimensions simultaneously have a partial tile
    run(RunTest2D("(TileN+2) x (TileK+5) x (TileM+3)  [all partial]",
                  kTileN + 2, kTileK + 5, kTileM + 3));

    // Arbitrary small non-power-of-two dimensions (all fit within a single tile)
    run(RunTest2D("7 x 13 x 5  [arbitrary small]",
                  7, 13, 5));

    // K=1: degenerate inner dimension (outer product)
    run(RunTest2D("N x 1 x M  [K=1 outer product]",
                  kTileN + 1, 1, kTileM + 1));

    // Unit output row (N=1) and unit output column (M=1)
    run(RunTest2D("1 x K x M  [N=1 row vector]",
                  1, kTileK, kTileM));
    run(RunTest2D("N x K x 1  [M=1 column vector]",
                  kTileN, kTileK, 1));

    // Multiple tiles in all dimensions simultaneously
    run(RunTest2D("3*TileN x (2*TileK+7) x (2*TileM+1)  [multi-tile all]",
                  kTileN * 3, kTileK * 2 + 7, kTileM * 2 + 1));

    // -----------------------------------------------------------------------
    // Batch tests
    // -----------------------------------------------------------------------
    printf("\n--- Batch ---\n");

    const unsigned BN = kTileN + 1, BK = kTileK / 4, BM = kTileM + 3;

    // batch=3, both A and B advance (no broadcasting)
    run(RunTestBatch("batch=3, no broadcast",
                     BN, BK, BM,
                     /*batch=*/3,
                     /*a_stride=*/BN * BK, /*b_stride=*/BK * BM));

    // batch=4, A repeats every iteration (stride=0 → A broadcasts)
    run(RunTestBatch("batch=4, A broadcasts (a_stride=0)",
                     BN, BK, BM,
                     /*batch=*/4,
                     /*a_stride=*/0, /*b_stride=*/BK * BM));

    // batch=4, B repeats every iteration (stride=0 → B broadcasts)
    run(RunTestBatch("batch=4, B broadcasts (b_stride=0)",
                     BN, BK, BM,
                     /*batch=*/4,
                     /*a_stride=*/BN * BK, /*b_stride=*/0));

    // batch=6, multi-dim batch flattened (verifies correct pointer arithmetic)
    run(RunTestBatch("batch=6, both strided (multi-dim flat)",
                     BN, BK, BM,
                     /*batch=*/6,
                     /*a_stride=*/BN * BK, /*b_stride=*/BK * BM));

    // -----------------------------------------------------------------------
    // Saturation tests
    //
    // Input values: a=100, b=±100, K=3.
    //   sum = 3 × 100 × 100 = 30 000
    //   ap_fixed<32,16> max ≈ 32 767  → 30 000 fits (no AccData_t overflow)
    //   ap_fixed<16,8>  max ≈ 127.996 → 30 000 >> max → saturates to AP_MAX
    //
    // For float Data_t: no saturation occurs (kSatMax ≈ FLT_MAX), but
    // correctness is still verified (kernel matches naive reference).
    // -----------------------------------------------------------------------
    printf("\n--- Saturation ---\n");

    // Tile-major packed B (b_packed = 1) — the layout the scheduler emits
    // for constant weights.  Same geometries as above, packed in the test.
    run(RunTest2D("TileN x TileK x TileM  [B packed]",           kTileN, kTileK, kTileM, kSeed, 1u));
    run(RunTest2D("(TileN+2) x (TileK+5) x (TileM+3)  [B packed, all partial]",
                  kTileN + 2, kTileK + 5, kTileM + 3, kSeed, 1u));
    run(RunTest2D("7 x 13 x 5  [B packed, arbitrary small]",     7, 13, 5, kSeed, 1u));
    run(RunTest2D("1 x K x M  [B packed, N=1 row vector]",       1, kTileK, kTileM, kSeed, 1u));
    run(RunTest2D("N x K x 1  [B packed, M=1]",                  kTileN, kTileK, 1, kSeed, 1u));
    run(RunTest2D("3*TileN x (2*TileK+7) x (2*TileM+1)  [B packed, multi-tile]",
                  3 * kTileN, 2 * kTileK + 7, 2 * kTileM + 1, kSeed, 1u));
    run(RunTest2D("1 x 2*TileK x 4*TileM  [B packed, FC-like]",  1, 2 * kTileK, 4 * kTileM, kSeed, 1u));
    run(RunTestBatch("batch=3, no broadcast  [B packed]",
                     5, 64, 19, 3, 5 * 64, 64 * 19, kSeed, 1u));
    run(RunTestBatch("batch=4, B broadcasts  [B packed, b_stride=0]",
                     5, 64, 19, 4, 5 * 64, 0, kSeed, 1u));

    // K-split across lanes (MATMUL_OPTIMISATION.md §5): n_valid < kTileN
    // splits K over 4 / 2 lanes per row with a tail guard.  Geometries chosen
    // so k_valid is not a multiple of lanes_per_row and K spans two tiles.
    run(RunTest2D("1 x 261 x 19  [K-split n=1]",                 1, 261, 19));
    run(RunTest2D("1 x 261 x 19  [B packed, K-split n=1]",       1, 261, 19, kSeed, 1u));
    run(RunTest2D("2 x 13 x 5  [K-split n=2]",                   2, 13, 5));
    run(RunTest2D("2 x 13 x 5  [B packed, K-split n=2]",         2, 13, 5, kSeed, 1u));
    run(RunTest2D("3 x 517 x 33  [K-split n=3, multi-tile]",     3, 517, 33));
    run(RunTest2D("3 x 517 x 33  [B packed, K-split n=3]",       3, 517, 33, kSeed, 1u));

    // b_tile prefetch (MATMUL_OPTIMISATION.md §7): the next block's cursor
    // crosses k_tile, m_tile, n_tile and batch boundaries (k_tiles = 3,
    // m_tiles = 3, n_tiles = 2, batch = 3), with and without B broadcast.
    run(RunTestBatch("batch=3, 6 x 517 x 35  [prefetch crosses tiles]",
                     6, 517, 35, 3, 6 * 517, 517 * 35));
    run(RunTestBatch("batch=3, 6 x 517 x 35  [B packed, prefetch crosses tiles]",
                     6, 517, 35, 3, 6 * 517, 517 * 35, kSeed, 1u));
    run(RunTestBatch("batch=3, 6 x 517 x 35, B broadcasts (b_stride=0)",
                     6, 517, 35, 3, 6 * 517, 0));
    run(RunTestBatch("batch=3, 6 x 517 x 35, B broadcasts  [B packed, b_stride=0]",
                     6, 517, 35, 3, 6 * 517, 0, kSeed, 1u));

    run(RunTestSaturation("sat_pos: a=100, b=100, K=3  → AP_MAX",
                           100.0, 100.0, 3, kSatMax));
    run(RunTestSaturation("sat_neg: a=100, b=-100, K=3 → AP_MIN",
                           100.0, -100.0, 3, kSatMin));

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (!g_dump_dir.empty()) {
        if (g_manifest) {
            std::fclose(g_manifest);
            g_manifest = nullptr;
        }
        printf("\nDumped %d test(s) to %s\n", g_test_idx, g_dump_dir.c_str());
        return 0;
    }

    printf("\n%d/%d tests passed\n", passed, total);
    if (!all_ok) {
        printf("TestMatmulSim FAILED\n");
        return 1;
    }
    printf("TestMatmulSim PASSED\n");
    return 0;
}
