// ---------------------------------------------------------------------------
// TestPoolingSim.cpp — C++ simulation tests for PoolingKernel.
//
// Verifies all three pool types (MAX, AVG, LP) against a pure-float reference
// implementation for a range of geometries, including padding, dilation,
// batched inputs, multi-tile channel counts, and Global* variants.
//
// Build and run via CMake:
//   cmake pool/   &&  make TestPoolingSim  &&  ./TestPoolingSim
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "PoolingKernel.h"
#include "PoolingKernelDebug.h"

// ---------------------------------------------------------------------------
// --dump-data <dir> mode: instead of running PoolingKernel and comparing,
// dump the per-test x/y_ref tensors to hex files (one 16-bit value per
// line, suitable for $readmemh).  A manifest.txt indexes every test with
// its geometry so an HDL testbench can load the same fixtures.
// ---------------------------------------------------------------------------
static std::string g_dump_dir;     // empty → verify mode (default)
static int         g_test_idx = 0;
static FILE*       g_manifest = nullptr;

// Tolerance: ~5 LSBs for ap_fixed<16,8> (1 LSB = 1/256 ≈ 0.0039).
// LP p=2 uses the same poly_sqrt approximation as the kernel (see
// ref_poly_sqrt below), so the residual error here is just the kernel's
// fixed-point quantization noise — well under 1 LSB.
static constexpr float kTol = 0.02f;

static float to_float(Data_t v)    { return float(v); }
static Data_t from_float(float v)  { return Data_t(v); }

// ---------------------------------------------------------------------------
// ref_poly_sqrt — bit-accurate float64 mirror of poly_sqrt() in
// PoolingKernel.cpp.  Uses the SAME range-reduction and the SAME 3rd-order
// polynomial coefficients, AND simulates the kernel's intermediate ap_fixed
// truncations so the dump-mode hex fixtures match the kernel's RTL output
// without any 1-LSB drift.
//
// AP_TRN = truncate toward −∞ = floor.  Quantize to N fractional bits via:
//     v_q = floor(v · 2^N) / 2^N
// ---------------------------------------------------------------------------
static double quantize_trn(double v, int frac_bits)
{
    const double scale = std::ldexp(1.0, frac_bits);
    return std::floor(v * scale) / scale;
}

static double ref_poly_sqrt(double acc)
{
    if (acc <= 0.0) return 0.0;

    // Range-reduce: find k such that m = acc / 4^k ∈ [1, 4).
    double m = acc;
    int k = 0;
    while (m >= 4.0) { m *= 0.25; ++k; }
    while (m <  1.0) { m *= 4.0;  --k; }

    // Mirror the kernel's ap_ufixed<18,2> on m (16 frac bits).
    m = quantize_trn(m, 16);

    // Coefficients in ap_fixed<16,1> (15 frac bits).
    const double c0 = quantize_trn( 0.4434, 15);
    const double c1 = quantize_trn( 0.6432, 15);
    const double c2 = quantize_trn(-0.0943, 15);
    const double c3 = quantize_trn( 0.0077, 15);

    // Horner's scheme — each intermediate truncated to ap_fixed<24,4>
    // (20 frac bits) to mirror the kernel exactly.
    const double t1     = quantize_trn(c3 * m + c2, 20);
    const double t2     = quantize_trn(t1 * m + c1, 20);
    const double sqrt_m = quantize_trn(t2 * m + c0, 20);

    // Apply 2^k scaling (exact in fixed-point shift), then quantize to
    // AccData_t = ap_fixed<32,16> (16 frac bits).
    return quantize_trn(std::ldexp(sqrt_m, k), 16);
}

// ---------------------------------------------------------------------------
// ref_avg_pool_fixed — bit-accurate float64 mirror of the kernel's
// fixed-point AVG-pool finalize (PoolingKernel.cpp::inv_denom_lookup +
// process_pool_kernel_tile's AVG branch).
//
// Pipeline mirrored:
//   raw      = ((1<<23) + denom/2) / denom        (LUT entry; +d/2 is the
//                                                  integer round-to-nearest)
//   inv_q    = raw / 2^23                         (ap_ufixed<24,1> grid)
//   prod_56  = acc * inv_q                        (ap_fixed<56,17> exact)
//   acc_data = floor(prod_56 * 2^16) / 2^16       (AP_TRN  → ap_fixed<32,16>)
//   data_t   = clamp(acc_data, [-128, 127.996])   (AP_SAT)
//              then floor(... * 2^8) / 2^8        (AP_TRN  → ap_fixed<16,8>)
//
// The two-step truncation collapses to a single floor-at-LSB for any value
// that fits AccData_t without wrapping, which is always the case for AVG
// (|acc/denom| ≤ max input magnitude ≤ 128).  We still write it as two
// explicit steps so any future change to AccData_t width is caught here.
// ---------------------------------------------------------------------------
static double ref_avg_pool_fixed(double acc_exact, unsigned denom)
{
    if (denom == 0u) return 0.0;
    const uint64_t raw_inv =
        ((uint64_t)1 << 23) + (uint64_t)(denom / 2u);
    const uint64_t raw     = raw_inv / (uint64_t)denom;
    const double inv_q = (double)raw / (double)((uint64_t)1 << 23);
    const double prod  = acc_exact * inv_q;

    // AP_TRN to ap_fixed<32,16>.
    const double acc_data_t = std::floor(prod * 65536.0) / 65536.0;
    // AP_SAT, then AP_TRN to ap_fixed<16,8>.
    const double clamped =
        std::max(-128.0, std::min(127.99609375, acc_data_t));
    return std::floor(clamped * 256.0) / 256.0;
}

// ---------------------------------------------------------------------------
// Reference pooling — bit-accurate mirror of PoolingKernel's fixed-point
// arithmetic.  MAX is exact in float64 (no divide).  LP p=2 reuses
// ref_poly_sqrt to mirror the kernel's polynomial sqrt.  AVG uses
// ref_avg_pool_fixed to mirror the inv_denom_lookup ROM divide — the
// kernel's float-domain reciprocal was replaced by an ap_ufixed<24,1>
// LUT, and on ~2.5% of inputs that produces a 1-LSB drift versus a true
// `acc / denom`.  The HDL behavior test compares the kernel's RTL output
// against the y.hex fixtures generated by --dump-data with strict
// equality, so this reference must match the kernel exactly — otherwise
// legitimate kernel cells get flagged as mismatches (which is what
// happened on AvgPool_*_no_include / _include_pad / _wide_W_96 cases
// after the kernel switch).
// ---------------------------------------------------------------------------
static float ref_pool_elem(const std::vector<Data_t>& x,
                            int C, int H, int W,
                            int n, int c, int oh, int ow,
                            int pool_h, int pool_w,
                            int stride_h, int stride_w,
                            int pad_top,  int pad_left,
                            int dil_h,    int dil_w,
                            int pool_type, int lp_order,
                            int count_include_pad)
{
    double acc = 0.0;
    int valid_count = 0;

    if (pool_type == 0) acc = -1.0e30; // MAX sentinel

    for (int khi = 0; khi < pool_h; khi++) {
        for (int kwi = 0; kwi < pool_w; kwi++) {
            int ih = oh * stride_h + khi * dil_h - pad_top;
            int iw = ow * stride_w + kwi * dil_w - pad_left;
            bool valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
            if (valid) {
                double v = to_float(x[(n * C + c) * H * W + ih * W + iw]);
                if (pool_type == 0) {          // MAX
                    acc = std::max(acc, v);
                } else if (pool_type == 1) {   // AVG
                    acc += v;
                } else {                        // LP
                    acc += (lp_order == 1) ? std::abs(v) : v * v;
                }
                valid_count++;
            }
        }
    }

    if (pool_type == 1) {
        // AVG: mirror inv_denom_lookup() bit-for-bit.  ref_avg_pool_fixed
        // already applies the AccData_t AP_TRN, AP_SAT clamp, and AP_TRN
        // to the Data_t grid, so its return value is the exact ap_fixed
        // <16,8> the kernel will write.
        const unsigned denom_u = count_include_pad
            ? (unsigned)(pool_h * pool_w)
            : (unsigned)valid_count;
        return (float)ref_avg_pool_fixed(acc, denom_u);
    }

    double result;
    if (pool_type == 0) {
        result = acc;
    } else {
        result = (lp_order == 1) ? acc : ref_poly_sqrt(acc);
    }

    // Clamp to ap_fixed<16,8> range [-128, 127.996] to match saturate_cast.
    return (float)std::max(-128.0, std::min(127.996, result));
}

// ---------------------------------------------------------------------------
// Dump-mode helpers (only meaningful for fixed-point builds — the HDL
// testbench reads 16-bit hex values one per line).
// ---------------------------------------------------------------------------
#ifdef POOL_HAVE_APFIXED
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

// ---------------------------------------------------------------------------
// Run one test case.
// ---------------------------------------------------------------------------
struct TC {
    const char* name;
    int N, C, H, W;
    int out_h, out_w;
    int pool_h, pool_w;
    int stride_h, stride_w;
    int pad_top,  pad_left;
    int dil_h,    dil_w;
    int pool_type;        // 0=MAX  1=AVG  2=LP
    int lp_order;         // 1 or 2 (LP only)
    int count_include_pad; // 0 or 1 (AVG only)
};

// ---------------------------------------------------------------------------
// cosim m_axi buffers (only the cosim build, -DPOOL_COSIM).
//
// Every pointer handed to PoolingKernel must be a fixed allocation >= the
// kernel's m_axi depth= hint, or cosim's wrapc adapter reads past the buffer
// and SIGSEGVs.  std::vector sized to the exact tensor is fine for plain
// C-sim but not for cosim, so run_test() / run_avg_pool_strict_test() copy
// each case into these globals.  Sizes come from the POOL_COSIM_DEPTH_*
// macros in PoolingKernel.h — the same single source of truth that feeds the
// depth= hints on PoolingKernel.cpp.
// ---------------------------------------------------------------------------
#ifdef POOL_COSIM
static PoolWord g_pool_x[POOL_COSIM_DEPTH_X_WORDS];
static Data_t   g_pool_y[POOL_COSIM_DEPTH_Y];
#endif

// x is a 128-bit word port: pack the element vector into PoolWord words
// (NCHW layout unchanged; one spare word so the kernel's last partial-word
// read of a row segment stays inside the buffer).
static std::vector<PoolWord> to_pool_words(const std::vector<Data_t>& e)
{
    std::vector<PoolWord> out(e.size() / kPoolPortElems + 1);
    for (auto& wd : out) wd = 0;
    for (size_t i = 0; i < e.size(); i++) {
        const unsigned lane = (unsigned)(i % kPoolPortElems);
        out[i / kPoolPortElems].range(kPoolDataBits * (lane + 1) - 1, kPoolDataBits * lane)
            = pool_data_to_lane(e[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// compute_ow_tile_ref — mirror of compute_ow_tile() inside PoolingKernel.cpp.
//
// Duplicated here (instead of exposed in a header) because it's a tiny
// arithmetic helper.  Must stay byte-identical to the kernel's version so
// the dup-read predictor below tracks the kernel's actual W-tile policy.
// ---------------------------------------------------------------------------
static unsigned compute_ow_tile_ref(int out_w, int pool_w, int stride_w, int dil_w)
{
    const int win_w_span = (pool_w - 1) * dil_w + 1;
    int ow_tile = 0;
    if (win_w_span <= (int)kMaxLineBufCols && stride_w > 0) {
        ow_tile = ((int)kMaxLineBufCols - win_w_span) / stride_w + 1;
    }
    if (ow_tile == 0) ow_tile = 1;
    if (ow_tile > out_w) ow_tile = out_w;
    return (unsigned)ow_tile;
}

// ---------------------------------------------------------------------------
// expected_dup_reads_for(tc) — predict pool_debug_duplicate_read_count() for
// the current kernel build.  Cache-aware: simulates the same Phase-1 load
// schedule the kernel runs (line buffer cached across the oh sweep within a
// (ni, ct, owt) chunk; W-tiling kicks in when in_w > kMaxLineBufCols), so
// the prediction adjusts automatically when kMaxLineBufCols / kTileC change.
//
// Pass condition (in run_test): `actual <= expected`.
//   * If the kernel matches our model, actual == expected and the message
//     reads e.g. `dup_reads=16/16` — clear evidence the cache is behaving
//     as designed for the current geometry and config.
//   * If the kernel does even better than our model, actual < expected
//     (no failure — but worth investigating: maybe the model is stale).
//   * If actual > expected the kernel is fetching more than the policy
//     allows — a regression flagged with "UNEXPECTED duplicate DDR read".
//
// The simulation does NOT model line_buf eviction explicitly: as long as
// kMaxLineBufRows >= (pool_h-1)*dil_h + 1 (a precondition the inference
// scheduler enforces), the kernel never re-fetches a row that was already
// loaded within the same (ni, ct, owt) chunk.  Re-fetches happen only
// across owt transitions on overlapping boundary columns — captured here
// by resetting the "loaded so far" tracker per (ni, ct, owt).
// ---------------------------------------------------------------------------
static unsigned expected_dup_reads_for(const TC& tc)
{
    std::map<std::size_t, unsigned> reads;

    const unsigned c_tiles = ((unsigned)tc.C + kTileC - 1) / kTileC;
    const unsigned ow_tile = compute_ow_tile_ref(
        tc.out_w, tc.pool_w, tc.stride_w, tc.dil_w);
    const unsigned ow_tiles_w =
        (ow_tile > 0) ? ((tc.out_w + ow_tile - 1) / ow_tile) : 1u;

    for (int n = 0; n < tc.N; n++) {
        for (unsigned ct = 0; ct < c_tiles; ct++) {
            const int c_off   = (int)(ct * kTileC);
            const int c_valid = std::min((int)kTileC, tc.C - c_off);

            for (unsigned owt = 0; owt < ow_tiles_w; owt++) {
                const int ow_lo = (int)(owt * ow_tile);
                const int ow_hi =
                    std::min((int)(ow_lo + (int)ow_tile), tc.out_w);

                const int iw_start =
                    ow_lo * tc.stride_w - tc.pad_left;
                const int iw_end =
                    (ow_hi - 1) * tc.stride_w
                  + (tc.pool_w - 1) * tc.dil_w
                  - tc.pad_left;
                const int iw_load_lo = std::max(iw_start, 0);
                const int iw_load_hi = std::min(iw_end, tc.W - 1);

                int last_loaded_row = -1;

                for (int oh = 0; oh < tc.out_h; oh++) {
                    const int ih_window_max = oh * tc.stride_h
                                            - tc.pad_top
                                            + (tc.pool_h - 1) * tc.dil_h;

                    int load_start = last_loaded_row + 1;
                    if (load_start < 0) load_start = 0;
                    int load_end = ih_window_max;
                    if (load_end >= tc.H) load_end = tc.H - 1;

                    for (int ih = load_start; ih <= load_end; ih++) {
                        for (int c_l = 0; c_l < c_valid; c_l++) {
                            const int c = c_off + c_l;
                            for (int iw = iw_load_lo; iw <= iw_load_hi; iw++) {
                                const std::size_t addr =
                                    ((std::size_t)n * tc.C + c) * tc.H * tc.W
                                  + (std::size_t)ih * tc.W
                                  + (std::size_t)iw;
                                reads[addr]++;
                            }
                        }
                    }
                    if (load_end > last_loaded_row) {
                        last_loaded_row = load_end;
                    }
                }
            }
        }
    }

    unsigned dup = 0;
    for (const auto& kv : reads) {
        if (kv.second > 1) dup++;
    }
    return dup;
}

static bool run_test(const TC& tc)
{
    const int in_size  = tc.N * tc.C * tc.H * tc.W;
    const int out_size = tc.N * tc.C * tc.out_h * tc.out_w;

    std::vector<Data_t> x(in_size);

    // Deterministic fill: values in [-4, 4] in steps of 0.1
    for (int i = 0; i < in_size; i++) {
        int v = (int)((unsigned)(kSeed * 1103515245u + (unsigned)i * 12345u) >> 16u) % 81;
        x[i] = from_float((v - 40) * 0.1f);
    }

    // Dump mode: compute y_ref via the naive oracle, write x/y hex files,
    // and append a manifest line.  No kernel run, no comparison.
    if (!g_dump_dir.empty()) {
#ifndef POOL_HAVE_APFIXED
        std::fprintf(stderr, "--dump-data requires POOL_HAVE_APFIXED build\n");
        std::exit(1);
#else
        std::vector<Data_t> y_ref(out_size, Data_t(0));
        for (int n = 0; n < tc.N; n++) {
            for (int c = 0; c < tc.C; c++) {
                for (int oh = 0; oh < tc.out_h; oh++) {
                    for (int ow = 0; ow < tc.out_w; ow++) {
                        float r = ref_pool_elem(
                            x, tc.C, tc.H, tc.W,
                            n, c, oh, ow,
                            tc.pool_h, tc.pool_w,
                            tc.stride_h, tc.stride_w,
                            tc.pad_top, tc.pad_left,
                            tc.dil_h, tc.dil_w,
                            tc.pool_type, tc.lp_order,
                            tc.count_include_pad);
                        y_ref[(n * tc.C + c) * tc.out_h * tc.out_w
                              + oh * tc.out_w + ow] = from_float(r);
                    }
                }
            }
        }

        const int idx = g_test_idx++;
        char idx_buf[16];
        std::snprintf(idx_buf, sizeof(idx_buf), "%02d", idx);
        const std::string prefix = g_dump_dir + "/test_" + idx_buf + "_";
        write_hex_file(prefix + "x.hex", x);
        write_hex_file(prefix + "y.hex", y_ref);

        std::fprintf(g_manifest,
                     "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %s\n",
                     idx,
                     tc.N, tc.C, tc.H, tc.W,
                     tc.out_h, tc.out_w,
                     tc.pool_h, tc.pool_w,
                     tc.stride_h, tc.stride_w,
                     tc.pad_top, tc.pad_left,
                     tc.dil_h, tc.dil_w,
                     tc.pool_type, tc.lp_order, tc.count_include_pad,
                     sanitize_label(tc.name).c_str());
        std::printf("[DUMP] test_%02d  %-45s  in=%d out=%d\n",
                    idx, tc.name, in_size, out_size);
        return true;
#endif
    }

    pool_debug_reset_duplicate_reads();

    // cosim needs fixed allocations >= the kernel's m_axi depth (the g_pool_*
    // globals); plain C-sim passes the per-test std::vector storage directly.
#ifdef POOL_COSIM
    if (in_size > (int)POOL_COSIM_DEPTH_X ||
        out_size > (int)POOL_COSIM_DEPTH_Y) {
        printf("  [SKIP] %-45s  (exceeds cosim buffers)\n", tc.name);
        return true;
    }
    {
        const std::vector<PoolWord> xw = to_pool_words(x);
        std::copy(xw.begin(), xw.end(), g_pool_x);
    }
    std::fill(g_pool_y, g_pool_y + out_size, Data_t(0));
    PoolWord* x_ptr = g_pool_x;
    Data_t*   y_ptr = g_pool_y;
#else
    std::vector<Data_t> y(out_size, Data_t(0));
    std::vector<PoolWord> xw = to_pool_words(x);
    PoolWord* x_ptr = xw.data();
    Data_t*   y_ptr = y.data();
#endif

    PoolingKernel(
        x_ptr, y_ptr,
        (unsigned)tc.N,    (unsigned)tc.C,
        (unsigned)tc.H,    (unsigned)tc.W,
        (unsigned)tc.out_h,(unsigned)tc.out_w,
        (unsigned)tc.pool_h,(unsigned)tc.pool_w,
        (unsigned)tc.stride_h,(unsigned)tc.stride_w,
        (unsigned)tc.pad_top, (unsigned)tc.pad_left,
        (unsigned)tc.dil_h,   (unsigned)tc.dil_w,
        (unsigned)tc.pool_type,
        (unsigned)tc.lp_order,
        (unsigned)tc.count_include_pad
    );

    const unsigned dup_reads     = pool_debug_duplicate_read_count();
    const unsigned expected_dups = expected_dup_reads_for(tc);

    int failures = 0;
    for (int n = 0; n < tc.N; n++) {
        for (int c = 0; c < tc.C; c++) {
            for (int oh = 0; oh < tc.out_h; oh++) {
                for (int ow = 0; ow < tc.out_w; ow++) {
                    float ref = ref_pool_elem(
                        x, tc.C, tc.H, tc.W,
                        n, c, oh, ow,
                        tc.pool_h, tc.pool_w,
                        tc.stride_h, tc.stride_w,
                        tc.pad_top, tc.pad_left,
                        tc.dil_h, tc.dil_w,
                        tc.pool_type, tc.lp_order,
                        tc.count_include_pad);
                    float got = to_float(y_ptr[(n * tc.C + c) * tc.out_h * tc.out_w
                                               + oh * tc.out_w + ow]);
                    if (std::abs(ref - got) > kTol) {
                        if (failures < 4) {
                            printf("    FAIL [n=%d,c=%d,oh=%d,ow=%d]: "
                                   "ref=%.4f  got=%.4f  diff=%.4f\n",
                                   n, c, oh, ow, ref, got, std::abs(ref - got));
                        }
                        failures++;
                    }
                }
            }
        }
    }

    const unsigned unexpected_dups =
        (dup_reads > expected_dups) ? (dup_reads - expected_dups) : 0u;
    const bool ok = (failures == 0) && (unexpected_dups == 0);
    const char* status = ok ? "PASS" : "FAIL";
    printf("  [%s] %-45s  failures=%d/%d  dup_reads=%u/%u",
           status, tc.name, failures, out_size, dup_reads, expected_dups);
    if (unexpected_dups > 0) {
        printf("  (%u UNEXPECTED duplicate DDR read(s))", unexpected_dups);
    }
    printf("\n");
    return ok;
}

// ---------------------------------------------------------------------------
// run_avg_pool_strict_test — strict-equality verification of the kernel's
// fixed-point AVG-pool reciprocal divide (the inv_denom_lookup ROM path in
// PoolingKernel.cpp).
//
// Why a separate test from run_test() — the existing AVG cases compare
// kernel output to a `acc / denom` float64 reference with kTol = 0.02
// (≈ 5 Data_t LSBs).  That tolerance comfortably hides the kernel's
// fixed-point reciprocal quantisation: ap_ufixed<24,1>(round(2^23/d)/2^23)
// rounds at different boundaries than IEEE float, so on ~2.5% of typical
// inputs the kernel's Data_t output differs from a float-domain divide by
// exactly 1 LSB.
//
// The codegen simulator (inference-scheduler/_pool2d_ref) was updated to
// mirror the kernel's exact fixed-point arithmetic so generated
// test_inference.c expected fixtures match the on-device kernel output
// byte-for-byte (test_inference.c uses strict `!=` element comparison, no
// tolerance).  This test pins the contract from the kernel side: if a
// future change re-introduced a float reciprocal — or shifted the LUT
// rounding mode, or skipped the AccData_t truncation — the strict
// equality check below would fail on the same ~2.5% of cells the
// simulator is now relying on.
//
// Coverage — one geometry, two count_include_pad variants:
//   * 5×5 stride=1 pad=2 yields six distinct denominators per kernel run
//     (corner=9, edge_e=12, edge_c=15, side=20, near-centre=20, centre=25
//     — count_include_pad=0 case), which exercises six different LUT
//     entries in a single invocation.  count_include_pad=1 fixes denom=25
//     and exercises the constant-denominator path.
//   * Mixed-magnitude inputs on the ap_fixed<16,8> grid: a deterministic
//     ramp combined with sign flips produces accumulator magnitudes from
//     ~0 up to ~25*128 = 3200.  At magnitudes > 256 the prior float-cast
//     path lost bits below 2^-15 — a regression there would show up as
//     1-LSB diffs at this strict tolerance.
// ---------------------------------------------------------------------------
static bool run_avg_pool_strict_test()
{
    // Geometry — picked so a single kernel run hits multiple LUT entries.
    constexpr int N=1, C=4, H=8, W=8;
    constexpr int pool_h=5, pool_w=5;
    constexpr int stride=1, pad=2;
    constexpr int out_h=H, out_w=W;       // SAME-style padding

    std::vector<Data_t> x(N*C*H*W);

    // Deterministic input — a small LCG so the test is reproducible
    // without depending on libc rand() ordering.  Each value is snapped to
    // the ap_fixed<16,8> grid (LSB = 1/256) so summation is exact.  Range
    // straddles zero with wide magnitudes, including near-extreme values
    // that drove the prior float-cast precision loss.
    uint32_t seed = 0xA15Du;
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;     // Numerical Recipes LCG
        const int q = (int)(seed >> 16) & 0x3FF;  // [0, 1023]
        const float v = ((float)q - 511.5f) * 0.25f;  // [-127.875, +127.875]
        x[i] = from_float(v);
    }

    int total_failures = 0;
    int total_cells    = 0;

    for (int c_inc = 0; c_inc < 2; ++c_inc) {
        pool_debug_reset_duplicate_reads();

        // cosim needs fixed allocations >= the kernel's m_axi depth; this
        // geometry is compile-time constant and small, so a static_assert
        // pins it under the bound rather than a runtime skip.
#ifdef POOL_COSIM
        static_assert(N*C*H*W         <= POOL_COSIM_DEPTH_X,
                      "strict-test input exceeds cosim buffer");
        static_assert(N*C*out_h*out_w <= POOL_COSIM_DEPTH_Y,
                      "strict-test output exceeds cosim buffer");
        {
            const std::vector<PoolWord> xw = to_pool_words(x);
            std::copy(xw.begin(), xw.end(), g_pool_x);
        }
        std::fill(g_pool_y, g_pool_y + (N*C*out_h*out_w), Data_t(0));
        PoolWord* x_ptr = g_pool_x;
        Data_t*   y_ptr = g_pool_y;
#else
        std::vector<Data_t> y(N*C*out_h*out_w, Data_t(0));
        std::vector<PoolWord> xw = to_pool_words(x);
        PoolWord* x_ptr = xw.data();
        Data_t*   y_ptr = y.data();
#endif

        PoolingKernel(
            x_ptr, y_ptr,
            (unsigned)N, (unsigned)C, (unsigned)H, (unsigned)W,
            (unsigned)out_h, (unsigned)out_w,
            (unsigned)pool_h, (unsigned)pool_w,
            (unsigned)stride, (unsigned)stride,
            (unsigned)pad, (unsigned)pad,
            1u, 1u,
            /*pool_type=*/1u,                  // AVG
            /*lp_order=*/0u,
            (unsigned)c_inc
        );

        int failures = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        // Exact accumulator.  Each input is on the
                        // ap_fixed<16,8> grid so the float64 sum is exact —
                        // no quantisation noise enters before the divide.
                        double acc = 0.0;
                        int valid_count = 0;
                        for (int kh = 0; kh < pool_h; kh++) {
                            for (int kw = 0; kw < pool_w; kw++) {
                                const int ih = oh * stride + kh - pad;
                                const int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < H &&
                                    iw >= 0 && iw < W) {
                                    acc += (double)to_float(
                                        x[(n*C + c) * H*W + ih*W + iw]);
                                    valid_count++;
                                }
                            }
                        }
                        const unsigned denom = c_inc
                            ? (unsigned)(pool_h * pool_w)
                            : (unsigned)valid_count;
                        const double ref_d = ref_avg_pool_fixed(acc, denom);
                        const double got_d = (double)to_float(
                            y_ptr[(n*C + c) * out_h*out_w + oh*out_w + ow]);

                        // Strict equality on the ap_fixed<16,8> grid: any
                        // 1-LSB drift indicates the kernel deviated from
                        // its specified fixed-point reciprocal arithmetic.
                        if (ref_d != got_d) {
                            if (failures < 4) {
                                printf("    FAIL [c_inc=%d n=%d c=%d "
                                       "oh=%d ow=%d]: denom=%u acc=%.6f  "
                                       "ref=%.6f  got=%.6f  diff=%.6f\n",
                                       c_inc, n, c, oh, ow, denom, acc,
                                       ref_d, got_d, std::abs(ref_d - got_d));
                            }
                            failures++;
                        }
                    }
                }
            }
        }

        const int cells = N*C*out_h*out_w;
        total_failures += failures;
        total_cells    += cells;
        printf("  [%s] AvgPool 5x5 pad=2 c_inc=%d strict-eq failures=%d/%d\n",
               failures == 0 ? "PASS" : "FAIL",
               c_inc, failures, cells);
    }

    return total_failures == 0;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Optional --dump-data <dir>: write per-test x/y_ref hex files plus a
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
            "# PoolingKernel test fixture manifest\n"
            "# idx N C H W out_h out_w pool_h pool_w stride_h stride_w "
            "pad_top pad_left dil_h dil_w pool_type lp_order "
            "count_include_pad label\n");
    }

    const TC tests[] = {
        // --- MaxPool ---
        // {name, N,C,H,W, out_h,out_w, pool_h,pool_w, stride_h,stride_w,
        //  pad_top,pad_left, dil_h,dil_w, pool_type, lp_order, count_include_pad}
        {"MaxPool 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool 3x3 stride1 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 0,0,0},
        {"MaxPool 2x2 stride2 rect 6x10",
                1,8,6,10, 3,5, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool batch=3 2x2 stride2",
                3,4,6,6, 3,3, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool channels>kTileC (C=16)",
                1,16,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        {"MaxPool dilation=2 pool2x2",
                1,4,8,8, 6,6, 2,2, 1,1, 0,0, 2,2, 0,0,0},
        // Global MaxPool: pool_h=in_h, pool_w=in_w, stride=1, pad=0
        {"GlobalMaxPool 4x4",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 0,0,0},
        {"GlobalMaxPool batch=2 C=12 6x6",
                2,12,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 0,0,0},

        // --- AveragePool ---
        {"AvgPool 2x2 stride2 no_pad",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 1,0,0},
        {"AvgPool 3x3 stride1 pad1 no_include",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        {"AvgPool 3x3 stride1 pad1 include_pad",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 1,1,0},
        {"AvgPool channels>kTileC (C=12)",
                1,12,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 1,0,0},
        // Global AveragePool
        {"GlobalAvgPool 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 1,0,0},
        {"GlobalAvgPool batch=2 C=16 6x6",
                2,16,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 1,0,0},
        // AvgPool asymmetric spatial (H≠W)
        {"AvgPool rect 6x10 2x2 stride2",
                1,4,6,10, 3,5, 2,2, 2,2, 0,0, 1,1, 1,0,0},

        // --- LpPool ---
        {"LpPool p=1 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 2,1,0},
        {"LpPool p=2 2x2 stride2",
                1,4,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 2,2,0},
        {"LpPool p=1 3x3 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 2,1,0},
        {"LpPool p=2 3x3 pad1",
                1,4,8,8, 8,8, 3,3, 1,1, 1,1, 1,1, 2,2,0},
        // Global LpPool
        {"GlobalLpPool p=1 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 2,1,0},
        {"GlobalLpPool p=2 4x4 C=8",
                1,8,4,4, 1,1, 4,4, 1,1, 0,0, 1,1, 2,2,0},
        {"GlobalLpPool p=2 batch=2 C=16 6x6",
                2,16,6,6, 1,1, 6,6, 1,1, 0,0, 1,1, 2,2,0},

        // --- Edge cases ---
        // 1x1 output (global-like, single output per channel)
        {"MaxPool 1x1 pool full 5x5",
                1,4,5,5, 1,1, 5,5, 1,1, 0,0, 1,1, 0,0,0},
        // All-padded corner: 3x3 pool, pad=1, on a 2x2 input → corner pixel
        // sees only 1 valid neighbour
        {"AvgPool corner padding 3x3 pad1 on 2x2",
                1,2,2,2, 2,2, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        // Large channel count to stress tiling (C=32 = 4 tiles of kTileC=8)
        {"MaxPool C=32 2x2 stride2",
                1,32,8,8, 4,4, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        // ---------------------------------------------------------------
        // Wide W: in_w > kMaxLineBufCols (=64 by default).  The producer
        // falls back to W-tiling — out_w is split into chunks whose
        // input-column span fits in line_buf, with boundary columns re-
        // read at tile transitions for overlapping windows.
        // ---------------------------------------------------------------
        // Overlapping windows: 3x3 stride1 pad1 → adjacent W-tiles share
        // 2 boundary input columns; expect a small (bounded) dup_reads
        // count, well below the no-cache baseline.
        {"MaxPool wide W=128 3x3 stride1 pad1",
                1,2,2,128, 2,128, 3,3, 1,1, 1,1, 1,1, 0,0,0},
        // Same geometry, AvgPool variant — sanity-check denom_pipe under
        // multi-tile iteration order (denom is emitted per (ni, ct, owt,
        // oh, ow), consumed in the same order).
        {"AvgPool wide W=96 3x3 stride1 pad1",
                1,2,4,96, 4,96, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        // Non-overlapping windows: 2x2 stride2 → W-tile boundaries land
        // on stride boundaries, no shared columns, dup_reads must be 0.
        {"MaxPool wide W=128 2x2 stride2",
                1,4,4,128, 2,64, 2,2, 2,2, 0,0, 1,1, 0,0,0},
        // Batch=2 variants of the wide-W cases — verify the (ni, ct, owt,
        // oh, ow) iteration order is correct across multiple batch elements
        // (each ni resets last_loaded_row inside its (ct, owt) chunk).
        {"MaxPool wide W=128 3x3 stride1 pad1 batch=2",
                2,2,2,128, 2,128, 3,3, 1,1, 1,1, 1,1, 0,0,0},
        {"AvgPool wide W=96 3x3 stride1 pad1 batch=2",
                2,2,4,96, 4,96, 3,3, 1,1, 1,1, 1,1, 1,0,0},
        {"MaxPool wide W=128 2x2 stride2 batch=2",
                2,4,4,128, 2,64, 2,2, 2,2, 0,0, 1,1, 0,0,0},
    };

    const int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    printf("PoolingKernel simulation tests\n");
    printf("  Data_t    = %s\n", sizeof(Data_t) == 4 ? "float" : "ap_fixed<16,8>");
    printf("  kTileC    = %u\n", kTileC);
    printf("  kMaxPoolH = %u\n", kMaxPoolH);
    printf("  kMaxPoolW = %u\n", kMaxPoolW);
    printf("  tolerance = %.4f\n\n", kTol);

    for (int i = 0; i < n_tests; i++) {
        if (run_test(tests[i])) passed++;
    }

    if (!g_dump_dir.empty()) {
        if (g_manifest) {
            std::fclose(g_manifest);
            g_manifest = nullptr;
        }
        printf("\nDumped %d test(s) to %s\n", g_test_idx, g_dump_dir.c_str());
        return 0;
    }

    // Strict-equality verification of the fixed-point AVG reciprocal path.
    // Adds two sub-cases (count_include_pad ∈ {0,1}) to the pass count.
    const int strict_subtests = 2;
    const int total_tests = n_tests + strict_subtests;
    if (run_avg_pool_strict_test()) passed += strict_subtests;

    printf("\n%d / %d tests passed.\n", passed, total_tests);
    return (passed == total_tests) ? 0 : 1;
}
