// ---------------------------------------------------------------------------
// TestConvSim.cpp — reference tests for ConvKernel.
//
// This single bench serves BOTH verification flows:
//   * plain C simulation  — the CMake TestConvRef target (host compiler);
//   * C/RTL co-simulation — the cosim_conv_<platform> target, which compiles
//     this file with -DCONV_COSIM (set by scripts/Cosim.tcl.in).
//
// Each test computes the same convolution with two independent implementations:
//
//   ref_conv()          — naive 7-nested-loop ground-truth oracle.
//   ref_depthwise_conv()— ground truth for depthwise (group=in_ch).
//   ConvKernel()        — tiled reference that mirrors the HLS kernel structure.
//
// Outputs are compared element-by-element with zero tolerance for fixed-point
// types (both share the same accumulator type and saturation policy), and
// relative 1e-5 tolerance for float.
//
// cosim note: cosim of an m_axi kernel needs every pointer argument backed by
// a fixed allocation at least as large as the kernel's depth= hint, so under
// -DCONV_COSIM run_test() copies each case into the CONV_COSIM_DEPTH_*-sized
// globals below.  Cases whose tensors exceed those bounds are skipped — cosim
// is full RTL simulation, so large convs are left to plain C-sim.
//
// Test matrix (standard conv):
//   1×1 kernel, 3×3 various pads/strides, 5×5 kernel, large spatial,
//   partial TILE_IC, partial TILE_M, dilation=2, batch>1, bias,
//   exact tile multiples, asymmetric stride, asymmetric dilation,
//   1×1 output, horizontal filter, ResNet-style strided block,
//   oh-chunking, M-grouping, M-grouping with chunks taller than the
//   line buffer (residency cap), ow-tiling, saturation (ap_fixed only).
//
// Test matrix (depthwise conv, is_depthwise=1):
//   3×3 depthwise, 3×3 depthwise+bias, partial TILE_M, dilation=2,
//   stride=2, batch>1, exact tile multiple, 5×5 kernel,
//   asymmetric stride, saturation (ap_fixed only).
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

#include "ConvKernel.h"

// ---------------------------------------------------------------------------
// --dump-data <dir> mode: instead of running ConvKernel and comparing, dump
// the per-test input/weight/bias tensors plus the naive-reference expected
// y to hex files (one 16-bit value per line, suitable for $readmemh).  A
// manifest.txt indexes every test with its geometry so an HDL testbench can
// load the same fixtures.  RNG state is shared with verify mode (same seed,
// same draw order), so the data is reproducible.
// ---------------------------------------------------------------------------
static std::string g_dump_dir;     // empty → verify mode (default)
static int         g_test_idx = 0; // increments per call to run_test
static FILE*       g_manifest = nullptr;
static int         g_sweep_n  = 0; // --sweep N: randomised-geometry mode
static unsigned    g_sweep_seed = 1234u;

// ---------------------------------------------------------------------------
// Scalar limits derived via saturate_cast — works for both ap_fixed and float.
// ---------------------------------------------------------------------------
static const double kSatMax = static_cast<double>(saturate_cast<Data_t>( 1e30));
static const double kSatMin = static_cast<double>(saturate_cast<Data_t>(-1e30));

// ---------------------------------------------------------------------------
// Naive reference — standard conv ground truth (group=1).
// ---------------------------------------------------------------------------
static void ref_conv(
    const Data_t* x,
    const Data_t* weight,
    const Data_t* bias,
    Data_t*       y,
    unsigned      batch,
    unsigned      in_ch,
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_ch,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      has_bias)
{
    for (unsigned n = 0; n < batch; n++) {
        for (unsigned m = 0; m < out_ch; m++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
                    AccData_t sum = has_bias ? AccData_t(bias[m]) : AccData_t(0);
                    for (unsigned c = 0; c < in_ch; c++) {
                        for (unsigned khi = 0; khi < kh; khi++) {
                            const int ih = (int)(oh * stride_h + khi * dilation_h)
                                         - (int)pad_top;
                            for (unsigned kwi = 0; kwi < kw; kwi++) {
                                const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                             - (int)pad_left;
                                if (ih >= 0 && (unsigned)ih < in_h &&
                                    iw >= 0 && (unsigned)iw < in_w)
                                {
                                    sum += AccData_t(x[(n*in_ch+c)*in_h*in_w + ih*in_w + iw])
                                         * AccData_t(weight[(m*in_ch+c)*kh*kw + khi*kw + kwi]);
                                }
                            }
                        }
                    }
                    y[(n*out_ch+m)*out_h*out_w + oh*out_w + ow] = saturate_cast<Data_t>(sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Naive reference — depthwise conv ground truth (group=in_ch).
//
// Weight layout: [ch][1][kh][kw] → offset m*kh*kw + khi*kw + kwi
// out_ch == in_ch == ch.
// ---------------------------------------------------------------------------
static void ref_depthwise_conv(
    const Data_t* x,
    const Data_t* weight,
    const Data_t* bias,
    Data_t*       y,
    unsigned      batch,
    unsigned      ch,      // in_ch == out_ch
    unsigned      in_h,
    unsigned      in_w,
    unsigned      out_h,
    unsigned      out_w,
    unsigned      kh,
    unsigned      kw,
    unsigned      stride_h,
    unsigned      stride_w,
    unsigned      dilation_h,
    unsigned      dilation_w,
    unsigned      pad_top,
    unsigned      pad_left,
    unsigned      has_bias)
{
    for (unsigned n = 0; n < batch; n++) {
        for (unsigned m = 0; m < ch; m++) {
            for (unsigned oh = 0; oh < out_h; oh++) {
                for (unsigned ow = 0; ow < out_w; ow++) {
                    AccData_t sum = has_bias ? AccData_t(bias[m]) : AccData_t(0);
                    for (unsigned khi = 0; khi < kh; khi++) {
                        const int ih = (int)(oh * stride_h + khi * dilation_h)
                                     - (int)pad_top;
                        for (unsigned kwi = 0; kwi < kw; kwi++) {
                            const int iw = (int)(ow * stride_w + kwi * dilation_w)
                                         - (int)pad_left;
                            if (ih >= 0 && (unsigned)ih < in_h &&
                                iw >= 0 && (unsigned)iw < in_w)
                            {
                                sum += AccData_t(x[(n*ch+m)*in_h*in_w + ih*in_w + iw])
                                     * AccData_t(weight[m*kh*kw + khi*kw + kwi]);
                            }
                        }
                    }
                    y[(n*ch+m)*out_h*out_w + oh*out_w + ow] = saturate_cast<Data_t>(sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output spatial size helper.
// ---------------------------------------------------------------------------
static unsigned out_size(unsigned in_sz, unsigned k, unsigned stride,
                         unsigned dilation, unsigned pad_begin, unsigned pad_end)
{
    const unsigned eff_k = dilation * (k - 1) + 1;
    return (in_sz + pad_begin + pad_end - eff_k) / stride + 1;
}

// ---------------------------------------------------------------------------
// Per-element comparison.
// ---------------------------------------------------------------------------
static bool vals_close(double ref, double got)
{
#ifdef CONV_HAVE_APFIXED
    return ref == got;  // fixed-point: exact match required
#else
    if (ref == got) return true;
    const double scale = std::max({std::abs(ref), std::abs(got), 1e-6});
    return std::abs(ref - got) <= 1e-5 * scale;
#endif
}

// ---------------------------------------------------------------------------
// Run one test case.  Returns number of mismatches.
// ---------------------------------------------------------------------------
struct ConvParams {
    unsigned batch, in_ch, in_h, in_w;
    unsigned out_ch;
    unsigned kh, kw;
    unsigned stride_h, stride_w;
    unsigned dilation_h, dilation_w;
    unsigned pad_top, pad_left, pad_bottom, pad_right;
    bool     has_bias;
    bool     is_depthwise;
};

// ---------------------------------------------------------------------------
// cosim m_axi buffers (only the cosim build, -DCONV_COSIM).
//
// Every pointer handed to ConvKernel must be a fixed allocation >= the
// kernel's m_axi depth= hint, or cosim's wrapc adapter reads past the buffer
// and SIGSEGVs.  std::vector sized to the exact tensor is fine for plain
// C-sim but not for cosim, so run_test() copies each case into these globals.
// Sizes come from the CONV_COSIM_DEPTH_* macros in ConvKernel.h — the same
// single source of truth that feeds the depth= hints on ConvKernel.cpp.
// ---------------------------------------------------------------------------
#ifdef CONV_COSIM
static Data_t g_x[CONV_COSIM_DEPTH_X];
static Data_t g_w[CONV_COSIM_DEPTH_WEIGHT];
static Data_t g_b[CONV_COSIM_DEPTH_BIAS];
static Data_t g_y[CONV_COSIM_DEPTH_Y];
#endif

// ---------------------------------------------------------------------------
// Dump-mode helpers (only meaningful for fixed-point builds — the HDL
// testbench reads 16-bit hex values one per line).
// ---------------------------------------------------------------------------
#ifdef CONV_HAVE_APFIXED
static uint16_t data_to_raw16(const Data_t& v)
{
    // ap_fixed<16,8>::range() returns the underlying int as an ap_int.
    return static_cast<uint16_t>(v.range().to_uint());
}

static void write_hex_file(const std::string&         path,
                           const std::vector<Data_t>& vec)
{
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

static std::string sanitize_label(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-')
            out.push_back(c);
        else
            out.push_back('_');
    }
    return out;
}

// Compute y_ref via the naive oracle, then write x/w/b/y_ref hex files and
// append a manifest line for one test case.
static void dump_test_data(const std::string&         dir,
                           int                        idx,
                           const char*                label,
                           const ConvParams&          p,
                           const std::vector<Data_t>& x,
                           const std::vector<Data_t>& w,
                           const std::vector<Data_t>& b)
{
#ifndef CONV_HAVE_APFIXED
    (void)dir; (void)idx; (void)label;
    (void)p; (void)x; (void)w; (void)b;
    std::fprintf(stderr, "--dump-data requires CONV_HAVE_APFIXED build\n");
    std::exit(1);
#else
    const unsigned out_h = out_size(p.in_h, p.kh, p.stride_h, p.dilation_h,
                                    p.pad_top,  p.pad_bottom);
    const unsigned out_w = out_size(p.in_w, p.kw, p.stride_w, p.dilation_w,
                                    p.pad_left, p.pad_right);
    const unsigned y_size = p.batch * p.out_ch * out_h * out_w;

    std::vector<Data_t> y_ref(y_size, Data_t(0));
    if (p.is_depthwise) {
        ref_depthwise_conv(x.data(), w.data(),
                           p.has_bias ? b.data() : nullptr,
                           y_ref.data(),
                           p.batch, p.in_ch, p.in_h, p.in_w,
                           out_h, out_w,
                           p.kh, p.kw,
                           p.stride_h, p.stride_w,
                           p.dilation_h, p.dilation_w,
                           p.pad_top, p.pad_left,
                           p.has_bias ? 1u : 0u);
    } else {
        ref_conv(x.data(), w.data(),
                 p.has_bias ? b.data() : nullptr,
                 y_ref.data(),
                 p.batch, p.in_ch, p.in_h, p.in_w,
                 p.out_ch, out_h, out_w,
                 p.kh, p.kw,
                 p.stride_h, p.stride_w,
                 p.dilation_h, p.dilation_w,
                 p.pad_top, p.pad_left,
                 p.has_bias ? 1u : 0u);
    }

    char idx_buf[16];
    std::snprintf(idx_buf, sizeof(idx_buf), "%02d", idx);
    const std::string prefix = dir + "/test_" + idx_buf + "_";
    write_hex_file(prefix + "x.hex", x);
    write_hex_file(prefix + "w.hex", w);
    write_hex_file(prefix + "b.hex", b);
    write_hex_file(prefix + "y.hex", y_ref);

    std::fprintf(g_manifest,
                 "%d %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %s\n",
                 idx,
                 p.batch, p.in_ch, p.in_h, p.in_w,
                 p.out_ch, out_h, out_w,
                 p.kh, p.kw,
                 p.stride_h, p.stride_w,
                 p.dilation_h, p.dilation_w,
                 p.pad_top, p.pad_left,
                 p.has_bias ? 1u : 0u,
                 p.is_depthwise ? 1u : 0u,
                 sanitize_label(label).c_str());

    std::printf("[DUMP] test_%02d  %-50s  x=%zu w=%zu b=%zu y=%u\n",
                idx, label,
                x.size(), w.size(), b.size(), y_size);
#endif
}

static int run_test(const char* name, const ConvParams& p,
                    const std::vector<Data_t>& x_data,
                    const std::vector<Data_t>& w_data,
                    const std::vector<Data_t>& b_data)
{
    if (!g_dump_dir.empty()) {
        dump_test_data(g_dump_dir, g_test_idx++, name, p,
                       x_data, w_data, b_data);
        return 0;
    }

    const unsigned out_h = out_size(p.in_h, p.kh, p.stride_h, p.dilation_h,
                                    p.pad_top,  p.pad_bottom);
    const unsigned out_w = out_size(p.in_w, p.kw, p.stride_w, p.dilation_w,
                                    p.pad_left, p.pad_right);

    const unsigned y_size = p.batch * p.out_ch * out_h * out_w;
    std::vector<Data_t> y_ref(y_size, Data_t(0));

    if (p.is_depthwise) {
        ref_depthwise_conv(x_data.data(), w_data.data(),
                           p.has_bias ? b_data.data() : nullptr,
                           y_ref.data(),
                           p.batch, p.in_ch, p.in_h, p.in_w,
                           out_h, out_w,
                           p.kh, p.kw,
                           p.stride_h, p.stride_w,
                           p.dilation_h, p.dilation_w,
                           p.pad_top, p.pad_left,
                           p.has_bias ? 1u : 0u);
    } else {
        ref_conv(x_data.data(), w_data.data(),
                 p.has_bias ? b_data.data() : nullptr,
                 y_ref.data(),
                 p.batch, p.in_ch, p.in_h, p.in_w,
                 p.out_ch, out_h, out_w,
                 p.kh, p.kw,
                 p.stride_h, p.stride_w,
                 p.dilation_h, p.dilation_w,
                 p.pad_top, p.pad_left,
                 p.has_bias ? 1u : 0u);
    }

    // -----------------------------------------------------------------------
    // Pick the buffers handed to ConvKernel.  cosim needs fixed allocations
    // >= the kernel's m_axi depth (the g_* globals); plain C-sim passes the
    // per-test std::vector storage directly.  b_ptr is always a valid pointer
    // (the kernel guards bias reads by has_bias).
    // -----------------------------------------------------------------------
#ifdef CONV_COSIM
    if (x_data.size() > CONV_COSIM_DEPTH_X ||
        w_data.size() > CONV_COSIM_DEPTH_WEIGHT ||
        b_data.size() > CONV_COSIM_DEPTH_BIAS ||
        y_size        > CONV_COSIM_DEPTH_Y) {
        printf("%-55s SKIP (exceeds cosim buffers)\n", name);
        return 0;
    }
    std::copy(x_data.begin(), x_data.end(), g_x);
    std::copy(w_data.begin(), w_data.end(), g_w);
    std::copy(b_data.begin(), b_data.end(), g_b);
    const Data_t* x_ptr = g_x;
    const Data_t* w_ptr = g_w;
    const Data_t* b_ptr = g_b;
    Data_t*       y_ptr = g_y;
#else
    std::vector<Data_t> y_got(y_size, Data_t(0));
    const Data_t* x_ptr = x_data.data();
    const Data_t* w_ptr = w_data.data();
    const Data_t* b_ptr = b_data.data();
    Data_t*       y_ptr = y_got.data();
#endif

    // x / y are hls::burst_maxi<Data_t> ports; the pointer constructor
    // takes a non-const Data_t* (the kernel only ever reads through x).
    ConvKernel(const_cast<Data_t*>(x_ptr), w_ptr, b_ptr, y_ptr,
               p.batch, p.in_ch, p.in_h, p.in_w,
               p.out_ch, out_h, out_w,
               p.kh, p.kw,
               p.stride_h, p.stride_w,
               p.dilation_h, p.dilation_w,
               p.pad_top, p.pad_left,
               p.has_bias ? 1u : 0u,
               p.is_depthwise ? 1u : 0u);

    int mismatches = 0;
    for (unsigned i = 0; i < y_size; i++) {
        const double r = static_cast<double>(y_ref[i]);
        const double g = static_cast<double>(y_ptr[i]);
        if (!vals_close(r, g)) {
            if (mismatches < 5) {
                printf("  [%u] ref=%.6f got=%.6f\n", i, r, g);
            }
            mismatches++;
        }
    }

    const char* status = (mismatches == 0) ? "PASS" : "FAIL";
    printf("%-55s %s", name, status);
    if (mismatches > 0) printf("  (%d mismatches)", mismatches);
    printf("  [%s batch=%u C=%u H=%u W=%u M=%u kH=%u kW=%u s=%u,%u d=%u,%u p=%u,%u out=%ux%u]\n",
           p.is_depthwise ? "DW" : "STD",
           p.batch, p.in_ch, p.in_h, p.in_w, p.out_ch,
           p.kh, p.kw, p.stride_h, p.stride_w,
           p.dilation_h, p.dilation_w, p.pad_top, p.pad_left,
           out_h, out_w);
    return mismatches;
}

// ---------------------------------------------------------------------------
// Fill vector with random values in [-scale, +scale].
// ---------------------------------------------------------------------------
template<typename T>
static std::vector<T> rand_vec(unsigned n, float scale, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<T> v(n);
    for (auto& e : v) e = T(dist(rng));
    return v;
}

// ---------------------------------------------------------------------------
// --sweep N [--seed S]: randomised-geometry sweep.
//
// Draws N geometries uniformly from the space the inference scheduler
// admits (kernel window fits the line buffer, one output row fits the
// persistent accumulator, depthwise ⇒ in_ch == out_ch) and checks each
// against the naive oracle, bit-exact.  Sizes are capped so a 300-case
// sweep runs in about a minute of C-sim.  The named tests above cover the
// mechanisms one at a time; the sweep covers their INTERACTIONS — the
// §2.21 stale-line-buffer bug (out_ch > 32 with a chunk taller than the
// line buffer) is exactly the kind of corner it exists for, and ~20 % of
// this space would have tripped it.  Any failure is printed with its full
// geometry so it can be re-added above as a named regression case.
// ---------------------------------------------------------------------------
static int run_sweep(int n, unsigned seed)
{
    std::mt19937 rng(seed);
    auto U = [&](unsigned lo, unsigned hi) {
        return std::uniform_int_distribution<unsigned>(lo, hi)(rng);
    };
    int failures = 0, cases = 0, attempts = 0;
    printf("Randomised geometry sweep: %d cases, seed %u\n", n, seed);
    printf("------------------------------------------------------------------\n");
    while (cases < n && attempts < n * 50) {
        attempts++;
        ConvParams p{};
        p.is_depthwise = (U(0, 3) == 0);
        p.batch      = U(1, 3);
        p.in_ch      = U(1, 40);
        p.out_ch     = p.is_depthwise ? p.in_ch : U(1, 72);
        p.in_h       = U(1, 70);
        p.in_w       = U(1, 70);
        p.kh         = U(1, kMaxKH);
        p.kw         = U(1, kMaxKW);
        p.stride_h   = U(1, 3);
        p.stride_w   = U(1, 3);
        p.dilation_h = U(1, 3);
        p.dilation_w = U(1, 3);
        p.pad_top    = U(0, 3);  p.pad_bottom = U(0, 3);
        p.pad_left   = U(0, 3);  p.pad_right  = U(0, 3);
        p.has_bias   = (U(0, 1) == 1);

        // Scheduler-side admissibility (nodes.py validation).
        if ((p.kh - 1) * p.dilation_h + 1 > kMaxLineBufRows) continue;
        if ((p.kw - 1) * p.dilation_w + 1 > kMaxLineBufCols) continue;
        const unsigned eff_h = p.dilation_h * (p.kh - 1) + 1;
        const unsigned eff_w = p.dilation_w * (p.kw - 1) + 1;
        if (p.in_h + p.pad_top  + p.pad_bottom < eff_h) continue;
        if (p.in_w + p.pad_left + p.pad_right  < eff_w) continue;
        const unsigned out_h = out_size(p.in_h, p.kh, p.stride_h, p.dilation_h,
                                        p.pad_top, p.pad_bottom);
        const unsigned out_w = out_size(p.in_w, p.kw, p.stride_w, p.dilation_w,
                                        p.pad_left, p.pad_right);
        const unsigned out_ch_padded = ((p.out_ch + kTileM - 1) / kTileM) * kTileM;
        if (out_w * out_ch_padded > kMaxAccPersistEntries) continue;
        // C-sim time cap: bound the MAC count per case.
        const unsigned long macs = (unsigned long)p.batch * p.out_ch * out_h * out_w
                                 * (p.is_depthwise ? 1u : p.in_ch) * p.kh * p.kw;
        if (macs > 40000000ul) continue;

        const unsigned n_w = p.is_depthwise ? p.out_ch * p.kh * p.kw
                                            : p.out_ch * p.in_ch * p.kh * p.kw;
        auto x = rand_vec<Data_t>(p.batch * p.in_ch * p.in_h * p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(n_w, 0.1f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        char name[64];
        std::snprintf(name, sizeof(name), "sweep #%d", cases);
        failures += run_test(name, p, x, w, b);
        cases++;
    }
    printf("------------------------------------------------------------------\n");
    printf("Sweep: %d cases run (%d attempts), %d element mismatch(es)\n",
           cases, attempts, failures);
    return failures;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Optional --dump-data <dir>: write per-test x/w/b/y_ref hex files plus
    // a manifest, then exit (no kernel run).  --sweep N [--seed S]: run only
    // the randomised-geometry sweep.  Otherwise: original verify mode.
    for (int i = 1; i < argc; i++) {
        const std::string a(argv[i]);
        if ((a == "--dump-data" || a == "-d") && i + 1 < argc) {
            g_dump_dir = argv[++i];
        } else if (a == "--sweep" && i + 1 < argc) {
            g_sweep_n = std::atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            g_sweep_seed = (unsigned)std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--dump-data <dir>] [--sweep N [--seed S]]\n", argv[0]);
            return 0;
        }
    }

    if (g_sweep_n > 0) {
        const int f = run_sweep(g_sweep_n, g_sweep_seed);
        printf(f == 0 ? "ALL TESTS PASSED\n" : "FAILED: %d element mismatch(es)\n", f);
        return f == 0 ? 0 : 1;
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
            "# ConvKernel test fixture manifest\n"
            "# idx batch in_ch in_h in_w out_ch out_h out_w kh kw "
            "stride_h stride_w dilation_h dilation_w pad_top pad_left "
            "has_bias is_depthwise label\n");
    }

    std::mt19937 rng(kSeed);
    int total_failures = 0;

    if (g_dump_dir.empty()) {
        printf("ConvKernel simulation tests\n");
        printf("Data_t    = %s\n", sizeof(Data_t) == 2 ? "ap_fixed<16,8>" : "float");
        printf("TILE_M=%u  TILE_IC=%u  MAX_KH=%u  MAX_KW=%u\n",
               kTileM, kTileIC, kMaxKH, kMaxKW);
    } else {
        printf("ConvKernel test data dump → %s\n", g_dump_dir.c_str());
        printf("Data_t    = %s  (CONV_HAVE_APFIXED required)\n",
               sizeof(Data_t) == 2 ? "ap_fixed<16,8>" : "float");
    }
    printf("------------------------------------------------------------------\n");

    // -----------------------------------------------------------------------
    // Test 1: 1×1 kernel, single channel, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 2.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    2.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x1 kernel, 1ch, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 1: 1×1 kernel, single channel, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 2.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    2.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x1 kernel, 1ch, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 2: 3×3 kernel, no padding, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, no pad, no bias → 3x3 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 3: 3×3 kernel, pad=1 (same-size output)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, pad=1 (same) → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 4: 3×3 kernel, stride=2, no padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=6; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, stride=2 → 2x2 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 5: 3×3 kernel with bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=2;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("3x3, pad=1, has_bias, 2 out_ch", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 6: Batch = 2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=2; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("batch=2, 3x3, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 7: Partial input channel tile (in_ch = TILE_IC + 5)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC+5; p.in_h=5; p.in_w=5; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("partial IC tile (in_ch=TILE_IC+5)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 8: Partial output channel tile (out_ch = TILE_M + 3)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=5; p.out_ch=kTileM+3;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("partial M tile (out_ch=TILE_M+3)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 9: Dilation = 2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=9; p.in_w=9; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=2; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3 dilation=2 → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 10: 5×5 kernel
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=7; p.out_ch=1;
        p.kh=5; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("5x5 kernel → 3x3 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 11: Multiple tiles in all dimensions (14×14 feature map)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC*2+3; p.in_h=14; p.in_w=14; p.out_ch=kTileM*2+1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.25f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.25f, rng);
        total_failures += run_test("14x14, multi-tile M and IC, bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 12: Non-square input and kernel
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=8; p.out_ch=2;
        p.kh=3; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("non-square: 6x8 input, 3x5 kernel", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 13: Mixed stride and padding (asymmetric)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=7; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("7x7, stride=2, asymmetric pad [1,1,0,0]", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 21: 1×1 kernel, exact IC and M tile multiples (no partial tile)
    // Pure channel projection; exercises the IC reduction with two full tiles.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC*2; p.in_h=5; p.in_w=5; p.out_ch=kTileM*2;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.1f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("1x1, IC=TILE_IC*2 M=TILE_M*2 bias (exact tiles)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 22: Asymmetric strides (stride_h=2, stride_w=1) → 2×4 output
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=6; p.in_w=6; p.out_ch=2;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, asymmetric stride h=2 w=1 → 2x4 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 23: Asymmetric dilation (dilation_h=1, dilation_w=2) → 5×5 output
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=7; p.in_w=9; p.out_ch=1;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("3x3, asymmetric dilation h=1 w=2 → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 24: Single-pixel output (3×3 input, 3×3 kernel, no padding)
    // Exercises the oh/ow loops iterating exactly once.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileIC; p.in_h=3; p.in_w=3; p.out_ch=kTileM;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f,  rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("3x3 input 3x3 kernel → 1x1 out, C=TILE_IC M=TILE_M", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 25: 1×5 horizontal filter, same-width padding (pad_left=pad_right=2)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=5; p.in_w=8; p.out_ch=2;
        p.kh=1; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=2; p.pad_bottom=0; p.pad_right=2;
        p.has_bias=false; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 1.0f, rng);
        total_failures += run_test("1x5 horizontal filter, pad_left=pad_right=2", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 26: batch=3, C=TILE_IC, M=TILE_M, stride=2, pad=1, bias
    // Models a ResNet-style strided block; verifies batch iteration across tiles.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=3; p.in_ch=kTileIC; p.in_h=7; p.in_w=7; p.out_ch=kTileM;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f,  rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("batch=3, C=TILE_IC M=TILE_M stride=2 (ResNet-style)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 27: oh-chunking standard.  out_h*out_w*out_ch = 32*32*32 = 32768
    // exceeds kMaxAccPersistEntries (16384); out_w*out_ch = 1024 fits, so
    // the kernel splits oh into 2 chunks (16 rows each).  This exercises the
    // line-buffer reload at chunk boundaries and oh_local indexing in the
    // consumer.  Small values keep ap_fixed<32,16> well clear of saturation.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=32; p.in_w=32; p.out_ch=32;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("oh-chunking standard (out=32x32x32, 2 chunks)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28: M-grouping standard.  out_ch=64 → m_tiles=8.  With default
    // kMaxMperGroup=4 this triggers num_m_groups=2: weights for tiles
    // 0..3 are cached together, processed across the entire (oh, ow)
    // sweep, then the cache is refilled for tiles 4..7.  out_h*out_w*out_ch
    // = 16*16*64 = 16384 = kMaxAccPersistEntries, so num_chunks=1 (this
    // test isolates the M-grouping behaviour from chunking).
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=16; p.in_w=16; p.out_ch=64;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("M-grouping standard (out_ch=64, 2 M-groups)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28b: M-grouping with a chunk taller than the line buffer.
    // Regression for the stale-row bug: the patch producer replays each
    // chunk's (oh, ow) sweep from line_buf once per M-group without
    // re-reading DDR, but line_buf only holds kMaxLineBufRows (16) rows.
    // Test 28 above uses in_h=16, exactly the buffer depth, so it could
    // never see rows being overwritten.  Here in_h=17 with 3x3/pad=1 needs
    // input rows -1..17 in one chunk (out_h*out_w*out_ch = 17*17*64 fits
    // kMaxAccPersistEntries in a single chunk if uncapped), so without the
    // residency cap in compute_conv_geometry() group 1 reads rows 0..1
    // after rows 16..17 have taken their slots — mismatches start at
    // m=32, oh=0.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=17; p.in_w=17; p.out_ch=64;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("M-grouping, in_h=17 > line_buf rows (residency cap)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28c: ResNet-style stem geometry — 7x7 stride 2 pad 3, 3→40 ch on
    // a 32x32 input (out 16x16).  40 ch = 5 m-tiles → two M-groups; the
    // stride-2 sweep covers 2 input rows per output row so the uncapped
    // chunk (16 rows) spans ~37 input rows.  The cap must shrink chunks to
    // (16 - 7) / 2 + 1 = 5 output rows.  Also exercises the multi-chunk
    // path that tests 27/31 lost when kMaxAccPersistEntries grew to 65536.
    // (Kept small on purpose — this case is also an RTL fixture, and with
    // only 3 input channels the 16-lane MAC tree runs at 3/16 utilisation,
    // so simulated time scales badly with output size.)
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=3; p.in_h=32; p.in_w=32; p.out_ch=40;
        p.kh=7; p.kw=7; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=3; p.pad_left=3; p.pad_bottom=3; p.pad_right=3;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("M-grouping, 7x7 s2 stem (ResNet-style conv0)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28d: M-grouping + dilation=2 + stride 2 — the cap's window term
    // uses (kh-1)*dilation_h, so a dilated kernel must also survive replay.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=2; p.in_ch=8; p.in_h=24; p.in_w=16; p.out_ch=40;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=1;
        p.dilation_h=2; p.dilation_w=2;
        p.pad_top=2; p.pad_left=2; p.pad_bottom=2; p.pad_right=2;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("M-grouping, batch=2 dil=2 s_h=2 (residency cap)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28e: long contiguous output runs.  1x1, 32 -> 16 ch on 40x64:
    // out_w*out_ch_padded = 1024 so one chunk holds 40 rows and each
    // channel's Phase-3 run is 40*64 = 2560 elements — more than 16
    // max-length AXI write bursts per write_request.  Regression for the
    // on-board hang of MobileNet v2's first 1x1 projection (32 -> 16 on
    // 112x112: 4032-element runs), where the burst_maxi writer's
    // sliding-window response collection deadlocked the m_axi adapter.
    // Bit-exact in C-sim regardless; the RTL fixture is what catches it.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=32; p.in_h=40; p.in_w=64; p.out_ch=16;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.1f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("1x1 32->16 on 40x64: 2560-element output runs", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 29: wide-input ow-tiling.  in_w=128 > kMaxLineBufCols (64), so
    // the producers split the output column axis into multiple ow_tiles
    // (3 at default kw=3, stride=1: ow_per_tile = 64 - 3 + 1 = 62; 128/62
    // = 3 tiles).  out_h*out_w*out_ch = 8*128*8 = 8192 fits one chunk.
    // This validates the line_buf circular column indexing and per-tile
    // (kw-1)*dilation_w-col DDR overlap re-fetch.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=8; p.in_w=128; p.out_ch=8;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=false;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.2f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*p.in_ch*p.kh*p.kw,    0.05f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.05f, rng);
        total_failures += run_test("wide input ow-tiling (in_w=128, 3 ow-tiles)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Depthwise tests (is_depthwise=1).
    // Weight layout: [ch][1][kh][kw]  (no in_ch dimension in weight)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Test 14 (DW): 3×3 depthwise, 4 channels, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=8; p.in_w=8; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           1.0f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3, 4ch, no pad, no bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 15 (DW): 3×3 depthwise + bias, same padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=8; p.in_w=8; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3, 4ch, pad=1, has_bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 16 (DW): Partial TILE_M — channels = TILE_M + 3
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileM+3; p.in_h=6; p.in_w=6; p.out_ch=kTileM+3;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW partial M tile (ch=TILE_M+3)", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 17 (DW): Depthwise with dilation=2
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=9; p.in_w=9; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=2; p.dilation_w=2;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW 3x3 dilation=2, 4ch", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 18 (DW): Stride=2 depthwise
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=8; p.in_w=8; p.out_ch=8;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=2;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW stride=2, 8ch, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 27 (DW): batch=2, 3×3 depthwise, same padding, no bias
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=2; p.in_ch=4; p.in_h=6; p.in_w=6; p.out_ch=4;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 1.0f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW batch=2, 4ch, pad=1", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 28 (DW): ch=TILE_M*2 — exact tile multiple, no partial last tile
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=kTileM*2; p.in_h=6; p.in_w=6; p.out_ch=kTileM*2;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW ch=TILE_M*2 (exact tile), bias", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 29 (DW): 5×5 depthwise kernel, 4 channels, no padding
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=4; p.in_h=9; p.in_w=9; p.out_ch=4;
        p.kh=5; p.kw=5; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.25f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.25f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.25f, rng);
        total_failures += run_test("DW 5x5 kernel, 4ch → 5x5 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 30 (DW): Asymmetric stride (stride_h=2, stride_w=1), 8 channels
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=8; p.in_h=8; p.in_w=8; p.out_ch=8;
        p.kh=3; p.kw=3; p.stride_h=2; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=false; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.5f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.5f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.5f, rng);
        total_failures += run_test("DW asymmetric stride h=2 w=1, 8ch → 4x8 out", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 31 (DW): oh-chunking depthwise.  32 channels, 32x32 output.
    // out_h*out_w*out_ch = 32768 > kMaxAccPersistEntries; out_w*out_ch = 1024
    // fits → 2 chunks of 16 rows each.  Exercises chunk-aware weight reload
    // in the depthwise consumer (w_buf re-streamed per (chunk, mt)).
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=32; p.in_h=32; p.in_w=32; p.out_ch=32;
        p.kh=3; p.kw=3; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=1; p.pad_left=1; p.pad_bottom=1; p.pad_right=1;
        p.has_bias=true; p.is_depthwise=true;
        auto x = rand_vec<Data_t>(p.batch*p.in_ch*p.in_h*p.in_w, 0.3f, rng);
        auto w = rand_vec<Data_t>(p.out_ch*1*p.kh*p.kw,           0.3f, rng);
        auto b = rand_vec<Data_t>(p.out_ch, 0.1f, rng);
        total_failures += run_test("DW oh-chunking (32ch, 32x32 out, 2 chunks)", p, x, w, b);
    }

#ifdef CONV_HAVE_APFIXED
    // -----------------------------------------------------------------------
    // Test 19: Saturation — positive overflow
    // All weights = +1, all inputs = kSatMax; kernel_size=1, C=1
    // Bias = kSatMax; output must saturate at kSatMax.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        const unsigned n_w = p.out_ch*p.in_ch*p.kh*p.kw;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(n_w, Data_t(1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMax));
        total_failures += run_test("saturation: positive overflow → AP_MAX", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 20: Saturation — negative overflow
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=false;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        const unsigned n_w = p.out_ch*p.in_ch*p.kh*p.kw;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(n_w, Data_t(-1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMin));
        total_failures += run_test("saturation: negative overflow → AP_MIN", p, x, w, b);
    }

    // -----------------------------------------------------------------------
    // Test 31: Depthwise saturation — positive overflow
    // weight=+1, input=kSatMax, bias=kSatMax → output must clamp at kSatMax.
    // -----------------------------------------------------------------------
    {
        ConvParams p{};
        p.batch=1; p.in_ch=1; p.in_h=3; p.in_w=3; p.out_ch=1;
        p.kh=1; p.kw=1; p.stride_h=1; p.stride_w=1;
        p.dilation_h=1; p.dilation_w=1;
        p.pad_top=0; p.pad_left=0; p.pad_bottom=0; p.pad_right=0;
        p.has_bias=true; p.is_depthwise=true;
        const unsigned n_x = p.batch*p.in_ch*p.in_h*p.in_w;
        std::vector<Data_t> x(n_x, Data_t(kSatMax));
        std::vector<Data_t> w(p.out_ch*1*p.kh*p.kw, Data_t(1));
        std::vector<Data_t> b(p.out_ch, Data_t(kSatMax));
        total_failures += run_test("DW saturation: positive overflow → AP_MAX", p, x, w, b);
    }
#endif

    printf("------------------------------------------------------------------\n");
    if (!g_dump_dir.empty()) {
        if (g_manifest) {
            std::fclose(g_manifest);
            g_manifest = nullptr;
        }
        printf("Dumped %d test(s) to %s\n", g_test_idx, g_dump_dir.c_str());
        return 0;
    }
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILED: %d element mismatch(es) across all tests\n", total_failures);
    }
    return total_failures == 0 ? 0 : 1;
}
