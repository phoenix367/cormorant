#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "VectorOP.h"

// ---------------------------------------------------------------------------
// --dump-data <dir> mode: instead of running VectorOPKernel and comparing,
// dump the per-test a/b/c_ref tensors to hex files (one 16-bit value per
// line, suitable for $readmemh).  A manifest.txt indexes every test with
// its geometry so an HDL testbench can load the same fixtures.  RNG state
// is shared with verify mode (same seed, same draw order).
//
// Fixture extents follow the kernel's geometry: a holds (outer-1)*a_inc +
// size elements, b (outer-1)*b_inc + size, c (outer-1)*(a_inc+b_inc) +
// size.  c_ref carries the reference value at every valid position; gap /
// tail positions hold 0 and the testbench only compares valid positions.
// ---------------------------------------------------------------------------
static std::string g_dump_dir;     // empty → verify mode (default)
static int         g_test_idx = 0;
static FILE*       g_manifest = nullptr;

// ---------------------------------------------------------------------------
// 16-byte-aligned element buffers.
//
// The kernel's ports are 128-bit words: every run start must be 16-byte
// aligned, input words past the end of a run are read (and masked), and
// the last word of every output run is written whole.  Buffers are
// therefore allocated in whole words on a 64-byte boundary, exactly like
// the scheduler's DMA buffers (VectorOP.h alignment contract).
// ---------------------------------------------------------------------------
struct AlignedBuf {
    Data_t*  p     = nullptr;
    unsigned n     = 0;        // logical element count
    unsigned words = 0;        // allocated words (>= ceil(n / kVecLanes))

    explicit AlignedBuf(unsigned n_elems, Data_t fill = Data_t(0)) { reset(n_elems, fill); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    ~AlignedBuf() { std::free(p); }

    void reset(unsigned n_elems, Data_t fill) {
        std::free(p);
        n     = n_elems;
        words = (n_elems + kVecLanes - 1) / kVecLanes;
        if (words == 0) words = 1;
        const size_t bytes = ((size_t)words * 16 + 63) & ~(size_t)63;
        void* raw = nullptr;
        if (posix_memalign(&raw, 64, bytes) != 0 || raw == nullptr) {
            std::fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
            std::exit(1);
        }
        std::memset(raw, 0, bytes);
        p = static_cast<Data_t*>(raw);
        for (unsigned i = 0; i < words * kVecLanes; ++i) p[i] = fill;
    }
    Data_t&       operator[](unsigned i)       { return p[i]; }
    const Data_t& operator[](unsigned i) const { return p[i]; }
    unsigned capacity() const { return words * kVecLanes; }
};

// Wrap the kernel call: asserts the alignment contract the scheduler
// guarantees (16-byte-aligned bases, strides in whole words) and builds
// the burst_maxi port objects.
static void run_kernel(AlignedBuf& a, AlignedBuf* b, AlignedBuf& c,
                       unsigned size, unsigned op, unsigned outer,
                       unsigned a_inc, unsigned b_inc, unsigned act) {
    assert(a_inc % kVecLanes == 0 && "a_inc must be a multiple of kVecLanes");
    assert(b_inc % kVecLanes == 0 && "b_inc must be a multiple of kVecLanes");
    assert(reinterpret_cast<uintptr_t>(a.p) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(c.p) % 16 == 0);
    const unsigned n_words = (size + kVecLanes - 1) / kVecLanes;
    const unsigned c_inc   = a_inc + b_inc;
    assert((outer - 1) * (a_inc / kVecLanes) + n_words <= a.words && "a buffer too small");
    assert((outer - 1) * (c_inc / kVecLanes) + n_words <= c.words && "c buffer too small");
    Data_t* bp = a.p;   // unary ops never read b; hand them a valid address anyway
    if (b != nullptr) {
        assert(reinterpret_cast<uintptr_t>(b->p) % 16 == 0);
        assert((outer - 1) * (b_inc / kVecLanes) + n_words <= b->words && "b buffer too small");
        bp = b->p;
    }
    hls::burst_maxi<VecWord> pa(reinterpret_cast<VecWord*>(a.p));
    hls::burst_maxi<VecWord> pb(reinterpret_cast<VecWord*>(bp));
    hls::burst_maxi<VecWord> pc(reinterpret_cast<VecWord*>(c.p));
    VectorOPKernel(pa, pb, pc, size, op, outer, a_inc, b_inc, act);
}

// ---------------------------------------------------------------------------
// Reference model
//
// ref_sat() derives its clamp range from saturate_cast<Data_t> so that it
// matches the kernel for every supported type:
//
//   ap_fixed<W,I>  → AP_SAT clips to the representable extreme.
//   float / double → identity cast, value stays ~1e38 (no practical clamp).
// ---------------------------------------------------------------------------
static const double kSatMax = static_cast<double>(saturate_cast<Data_t>( 1e38));
static const double kSatMin = static_cast<double>(saturate_cast<Data_t>(-1e38));

static double ref_sat(double v) {
    return std::max(kSatMin, std::min(kSatMax, v));
}

static double ref_act(double v, unsigned act) {
    switch (act) {
        case ACT_RELU:  return std::max(0.0, v);
        case ACT_RELU6: return std::min(std::max(0.0, v), 6.0);
        default:        return v;
    }
}

static double ref_op(unsigned op, double a, double b, unsigned act = ACT_NONE) {
    double r;
    switch (op) {
        case OP_ADD:   r = ref_sat(a + b); break;
        case OP_SUB:   r = ref_sat(a - b); break;
        case OP_MUL:   r = ref_sat(a * b); break;
        case OP_DIV:   r = (b == 0.0) ? 0.0 : ref_sat(a / b); break;
        case OP_RELU:  r = std::max(0.0, a); break;
        case OP_RELU6: r = std::min(std::max(0.0, a), 6.0); break;
        default:       r = a; break;
    }
    return ref_act(r, act);
}

static bool is_unary(unsigned op) { return op >= OP_RELU; }

// ---------------------------------------------------------------------------
// Dump-mode helpers (only meaningful for fixed-point builds — the HDL
// testbench reads 16-bit hex values one per line).
// ---------------------------------------------------------------------------
#ifdef VA_HAVE_APFIXED
static uint16_t data_to_raw16(const Data_t& v) {
    return static_cast<uint16_t>(v.range().to_uint());
}

static void write_hex_file(const std::string& path, const Data_t* vec, unsigned n) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        std::exit(1);
    }
    for (unsigned i = 0; i < n; ++i)
        std::fprintf(f, "%04x\n", data_to_raw16(vec[i]));
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

// Extents of the a / b / c arrays for a geometry (elements).
static unsigned extent(unsigned size, unsigned outer, unsigned inc) {
    return (outer - 1) * inc + size;
}

// Write a/b/c hex files plus a manifest row.  c_ref holds one double per
// c position (gaps = 0).
static void dump_one_case(const char* label,
                          unsigned size, unsigned op_code,
                          unsigned outer, unsigned a_inc, unsigned b_inc, unsigned act,
                          const AlignedBuf& a, const AlignedBuf& b,
                          const std::vector<double>& c_ref_d) {
#ifndef VA_HAVE_APFIXED
    (void)label; (void)size; (void)op_code;
    (void)outer; (void)a_inc; (void)b_inc; (void)act;
    (void)a; (void)b; (void)c_ref_d;
    std::fprintf(stderr, "--dump-data requires VA_HAVE_APFIXED build\n");
    std::exit(1);
#else
    const int idx = g_test_idx++;
    char idx_buf[16];
    std::snprintf(idx_buf, sizeof(idx_buf), "%02d", idx);
    const std::string prefix = g_dump_dir + "/test_" + idx_buf + "_";

    std::vector<Data_t> c_ref(c_ref_d.size());
    for (size_t i = 0; i < c_ref_d.size(); ++i)
        c_ref[i] = saturate_cast<Data_t>(c_ref_d[i]);

    const unsigned a_n = extent(size, outer, a_inc);
    const unsigned b_n = extent(size, outer, b_inc);
    write_hex_file(prefix + "a.hex", a.p, a_n);
    write_hex_file(prefix + "b.hex", b.p, b_n);
    write_hex_file(prefix + "c.hex", c_ref.data(), (unsigned)c_ref.size());

    std::fprintf(g_manifest, "%d %u %u %u %u %u %u %s\n",
                 idx, size, op_code, outer, a_inc, b_inc, act,
                 sanitize_label(label).c_str());
    std::printf("[DUMP] test_%02d  %-34s  size=%u op=%u outer=%u a_inc=%u b_inc=%u act=%u\n",
                idx, label, size, op_code, outer, a_inc, b_inc, act);
#endif
}

// ---------------------------------------------------------------------------
// Generic case runner.
//
// Fills a / b with random values for the op, computes the reference over
// the full c extent (gaps marked don't-care), runs the kernel and checks:
//   * every valid position matches the reference (1 LSB for ap_fixed,
//     relative 1e-5 for float);
//   * the tail lanes of every run's last word (positions [size, ceil8(size))
//     within the run) read 0 — the contract's op(0, 0);
//   * every other gap position still holds the pre-fill poison (never
//     written).
// ---------------------------------------------------------------------------
struct Range { double lo, hi; };

static Range input_range(unsigned op) {
    switch (op) {
        case OP_ADD:
        case OP_SUB:  return { -80.0,  80.0 };   // sums/diffs can exceed ±128
        case OP_MUL:  return { -15.0,  15.0 };   // products can exceed ±128
        case OP_DIV:  return {   1.0,  10.0 };   // b always positive, non-zero
        default:      return { -10.0,  10.0 };
    }
}

static const Data_t kPoison = Data_t(-77.5);

static bool RunCase(const char* label, unsigned op, unsigned size, unsigned outer,
                    unsigned a_inc, unsigned b_inc, unsigned act, unsigned seed,
                    bool verbose = true) {
    std::default_random_engine rng(seed);
    Range ra = input_range(op);
    Range rb = (op == OP_DIV) ? Range{1.0, 10.0} : input_range(op);
    std::uniform_real_distribution<double> distA(ra.lo, ra.hi);
    std::uniform_real_distribution<double> distB(rb.lo, rb.hi);

    const unsigned c_inc = a_inc + b_inc;
    const unsigned a_n   = extent(size, outer, a_inc);
    const unsigned b_n   = extent(size, outer, b_inc);
    const unsigned c_n   = extent(size, outer, c_inc);
    const unsigned n_w   = (size + kVecLanes - 1) / kVecLanes;

    // Inputs: data at every position of their extent (gaps included, so an
    // operand that is read past `size` sees non-zero bytes there).
    AlignedBuf a(a_n), b(b_n), c(c_n, kPoison);
    for (unsigned i = 0; i < a.capacity(); ++i) a[i] = Data_t(distA(rng));
    for (unsigned i = 0; i < b.capacity(); ++i) b[i] = Data_t(distB(rng));

    std::vector<double> c_ref(c_n, 0.0);
    std::vector<char>   kind(c.capacity(), 'g');   // 'v' valid, 't' tail word, 'g' gap
    for (unsigned o = 0; o < outer; ++o) {
        for (unsigned i = 0; i < size; ++i) {
            const double av = static_cast<double>(a[o * a_inc + i]);
            const double bv = static_cast<double>(b[o * b_inc + i]);
            c_ref[o * c_inc + i] = ref_op(op, av, bv, act);
            kind [o * c_inc + i] = 'v';
        }
        for (unsigned i = size; i < n_w * kVecLanes; ++i) {
            const unsigned pos = o * c_inc + i;
            if (pos < c.capacity() && kind[pos] != 'v') kind[pos] = 't';
        }
    }

    if (!g_dump_dir.empty()) {
        dump_one_case(label, size, op, outer, a_inc, b_inc, act, a, b, c_ref);
        return true;
    }

    run_kernel(a, is_unary(op) ? nullptr : &b, c, size, op, outer, a_inc, b_inc, act);

    const bool   isFloat = std::is_floating_point<Data_t>::value;
    const double absTol  = 1.0 / 256.0;
    const double relTol  = 1e-5;

    unsigned mismatches = 0;
    for (unsigned i = 0; i < c.capacity(); ++i) {
        const double got = static_cast<double>(c[i]);
        bool bad = false;
        double ref = 0.0;
        if (kind[i] == 'v') {
            ref = c_ref[i];
            const double diff = std::abs(got - ref);
            if (isFloat) {
                const double absRef = std::abs(ref);
                bad = (absRef > 1e-9) ? (diff / absRef > relTol) : (diff > relTol);
            } else {
                bad = (diff > absTol);
            }
        } else if (kind[i] == 't') {
            bad = (got != 0.0);                     // op(0, 0) == 0 for every op
        } else {
            ref = static_cast<double>(kPoison);
            bad = (got != ref);                     // gap never written
        }
        if (bad) {
            ++mismatches;
            if (mismatches <= 3 && verbose)
                std::cerr << "  [" << label << "] MISMATCH at [" << i << "] (" << kind[i] << "): "
                          << "got=" << got << "  ref=" << ref << "\n";
        }
    }
    return mismatches == 0;
}

// ---------------------------------------------------------------------------
// RunSatTest — saturation boundary verification with known inputs.
//
// Uses a fixed-size vector so every element receives the same constant values.
// Comparison tolerance is zero for fixed-point types (ap_fixed results are
// exact when converted to double) and 1 LSB for float.
// ---------------------------------------------------------------------------
static bool RunSatTest(unsigned op, double a_val, double b_val, const char* desc) {
    static const unsigned kSize = 8;
    static const double   kTol  = std::is_floating_point<Data_t>::value
                                  ? 1.0 / 256.0
                                  : 0.0;

    AlignedBuf a(kSize, Data_t(a_val));
    AlignedBuf b(kSize, Data_t(b_val));
    AlignedBuf c(kSize, Data_t(0));

    // Reference uses the same saturate-cast semantics as the kernel.
    const double expected = ref_op(op,
                                   static_cast<double>(Data_t(a_val)),
                                   static_cast<double>(Data_t(b_val)));

    if (!g_dump_dir.empty()) {
        std::vector<double> c_ref(kSize, expected);
        dump_one_case(desc, kSize, op, 1u, 0u, 0u, ACT_NONE, a, b, c_ref);
        return true;
    }

    run_kernel(a, is_unary(op) ? nullptr : &b, c, kSize, op, 1u, 0u, 0u, ACT_NONE);

    bool ok = true;
    for (unsigned i = 0; i < kSize; ++i) {
        const double got  = static_cast<double>(c[i]);
        const double diff = std::abs(got - expected);
        if (diff > kTol) {
            if (ok)  // print header only once
                std::cerr << "  FAIL — " << desc << "\n"
                          << "    a=" << a_val << "  b=" << b_val
                          << "  expected=" << expected
                          << "  (sat_max=" << kSatMax
                          << "  sat_min=" << kSatMin << ")\n";
            std::cerr << "    c[" << i << "]=" << got
                      << "  diff=" << diff << "\n";
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Saturation test table
//
// One LSB of ap_fixed<16,8> = 1/256 = 0.00390625.
// MAX =  127.99609375 = 0x7FFF.
// MIN = -128.0        = 0x8000.
// ---------------------------------------------------------------------------
struct SatEntry {
    unsigned    op;
    double      a, b;
    const char* desc;
};

static const SatEntry kSatTests[] = {
    // ── ADD ─────────────────────────────────────────────────────────────────
    { OP_ADD,  100.0,          100.0,         "ADD  100+100=200    → sat_max" },
    { OP_ADD,   64.0,           64.0,         "ADD   64+64=128     → sat_max" },
    { OP_ADD,  127.99609375,     0.00390625,  "ADD  max+1LSB=128   → sat_max" },
    { OP_ADD, -100.0,          -100.0,        "ADD -100-100=-200   → sat_min" },
    { OP_ADD, -128.0,            -0.00390625, "ADD  min-1LSB       → sat_min" },
    { OP_ADD,   64.0,           63.99609375,  "ADD   64+63.996=max (no clip)" },
    { OP_ADD,  -64.0,          -64.0,         "ADD  -64-64=-128=min (no clip)" },

    // ── SUB ─────────────────────────────────────────────────────────────────
    { OP_SUB,  100.0,          -100.0,        "SUB  100-(-100)=200 → sat_max" },
    { OP_SUB,  127.99609375,    -0.00390625,  "SUB  max-(-1LSB)    → sat_max" },
    { OP_SUB, -100.0,           100.0,        "SUB -100-100=-200   → sat_min" },
    { OP_SUB, -128.0,             0.00390625, "SUB  min-1LSB       → sat_min" },

    // ── MUL ─────────────────────────────────────────────────────────────────
    { OP_MUL,   16.0,           16.0,         "MUL  16×16=256      → sat_max" },
    { OP_MUL,   12.0,           12.0,         "MUL  12×12=144      → sat_max" },
    { OP_MUL,  -16.0,          -16.0,         "MUL -16×-16=256     → sat_max" },
    { OP_MUL,  -16.0,           16.0,         "MUL -16×16=-256     → sat_min" },
    { OP_MUL,   11.0,           11.0,         "MUL  11×11=121 (no clip)"      },

    // ── RELU  (clipping at 0) ────────────────────────────────────────────────
    { OP_RELU,  -0.00390625,     0.0,         "RELU -1LSB → 0"                },
    { OP_RELU,  -1.0,            0.0,         "RELU -1.0  → 0"                },
    { OP_RELU,   0.0,            0.0,         "RELU  0    → 0"                },
    { OP_RELU,   0.00390625,     0.0,         "RELU +1LSB → +1LSB (no clip)"  },
    { OP_RELU,   3.5,            0.0,         "RELU  3.5  → 3.5 (no clip)"    },

    // ── RELU6 (clipping at 0 and 6) ─────────────────────────────────────────
    { OP_RELU6, -1.0,            0.0,         "RELU6 -1    → 0"               },
    { OP_RELU6, -0.00390625,     0.0,         "RELU6 -1LSB → 0"               },
    { OP_RELU6,  0.0,            0.0,         "RELU6  0    → 0 (no clip)"     },
    { OP_RELU6,  0.00390625,     0.0,         "RELU6 +1LSB → +1LSB (no clip)" },
    { OP_RELU6,  3.0,            0.0,         "RELU6  3.0  → 3.0 (no clip)"   },
    { OP_RELU6,  5.99609375,     0.0,         "RELU6  6-1LSB → 6-1LSB (no clip)" },
    { OP_RELU6,  6.0,            0.0,         "RELU6  6.0  → 6.0 (no clip)"   },
    { OP_RELU6,  6.00390625,     0.0,         "RELU6  6+1LSB → 6"             },
    { OP_RELU6,  8.0,            0.0,         "RELU6  8    → 6"               },
    { OP_RELU6, 20.0,            0.0,         "RELU6 20    → 6"               },
};

static bool RunSatTests() {
    const unsigned n = sizeof(kSatTests) / sizeof(kSatTests[0]);
    bool allPassed = true;

    std::cout << "\n--- Saturation boundary tests (" << n << ") ---\n";
    for (unsigned i = 0; i < n; ++i) {
        const SatEntry& t = kSatTests[i];
        std::cout << "[" << (i + 1) << "/" << n << "] " << t.desc
                  << " ... " << std::flush;
        const bool ok = RunSatTest(t.op, t.a, t.b, t.desc);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        allPassed &= ok;
    }
    return allPassed;
}

// ---------------------------------------------------------------------------
// Geometry / activation table — broadcast, stride-0, flattened-loop and
// act cases (all dumped as RTL fixtures too).
//
//   inc == size on whole words → one contiguous stream (fast path);
//   inc == 0, <= 2048 elements → replay buffer; > 2048 → re-read per run;
//   otherwise separately requested runs with the tail lanes masked.
// ---------------------------------------------------------------------------
struct GeomEntry {
    unsigned    op;
    unsigned    size, outer, a_inc, b_inc, act;
    const char* desc;
};

static const GeomEntry kGeomTests[] = {
    // Broadcast: chunk 12 at stride 16 (masked tail word, 1 piece per run)
    { OP_ADD,   12,    5,  16,    0, ACT_NONE,  "bcast_b chunk12 stride16 ADD"          },
    { OP_MUL,   12,    5,   0,   16, ACT_NONE,  "bcast_a chunk12 stride16 MUL"          },
    { OP_SUB,   13,    9,  16,    0, ACT_NONE,  "bcast_b chunk13 stride16 SUB"          },
    { OP_DIV,    9,    4,   0,   16, ACT_NONE,  "bcast_a chunk9 stride16 DIV"           },
    // Contiguous outer loop (inc == size): 1000 x 16 (the dw chunk-16 pattern)
    { OP_MUL,   16, 1000,  16,    0, ACT_NONE,  "outer1000 x size16 MUL"                },
    { OP_ADD,   16, 1000,   0,   16, ACT_NONE,  "outer1000 x size16 ADD a-bcast"        },
    // Stride-0 operand at / past the replay-buffer bound
    { OP_ADD, 2048,    3,   0, 2048, ACT_NONE,  "stride0 a 2048 (replay max)"           },
    { OP_SUB, 2100,    3,   0, 2104, ACT_NONE,  "stride0 a 2100 (> replay, re-read)"    },
    { OP_MUL, 2100,    3, 2104,   0, ACT_NONE,  "stride0 b 2100 (> replay, re-read)"    },
    // Multi-piece runs (> 64 read words / > 256 write words per run)
    { OP_ADD, 1000,    3, 1008,   0, ACT_NONE,  "size1000 stride1008 (2 read pieces)"   },
    { OP_SUB, 3000,    2,   0, 3008, ACT_NONE,  "size3000 stride3008 (2 write pieces)"  },
    // Unary with an outer loop (c_inc == a_inc)
    { OP_RELU,  12,    4,  16,    0, ACT_NONE,  "unary bcast-shaped RELU"               },
    { OP_RELU6, 20,    3,  24,    0, ACT_NONE,  "unary outer3 size20 RELU6"             },
    // Long contiguous runs: > 16 x 256 words (§2.30 bound), 40000 elements
    { OP_ADD, 40000,   1,   0,    0, ACT_NONE,  "run 5000 words ADD"                    },
    { OP_RELU,40000,   1,   0,    0, ACT_NONE,  "run 5000 words RELU"                   },
    // Fused activation
    { OP_ADD,  255,    1,   0,    0, ACT_RELU,  "ADD + act RELU"                        },
    { OP_ADD, 1023,    1,   0,    0, ACT_RELU6, "ADD + act RELU6"                       },
    { OP_SUB,   64,    1,   0,    0, ACT_RELU,  "SUB + act RELU"                        },
    { OP_MUL,   12,    5,  16,    0, ACT_RELU6, "MUL bcast + act RELU6"                 },
    { OP_DIV,  100,    1,   0,    0, ACT_RELU6, "DIV + act RELU6"                       },
    { OP_RELU,  33,    1,   0,    0, ACT_RELU6, "RELU + act RELU6"                      },
    { OP_ADD,   16, 1000,   0,   16, ACT_RELU,  "outer1000 x size16 ADD + act RELU"     },
};

static bool RunGeomTests(unsigned seed_base) {
    const unsigned n = sizeof(kGeomTests) / sizeof(kGeomTests[0]);
    bool allPassed = true;

    std::cout << "\n--- Geometry / activation tests (" << n << ") ---\n";
    for (unsigned i = 0; i < n; ++i) {
        const GeomEntry& t = kGeomTests[i];
        std::cout << "[" << (i + 1) << "/" << n << "] " << t.desc
                  << " ... " << std::flush;
        const bool ok = RunCase(t.desc, t.op, t.size, t.outer,
                                t.a_inc, t.b_inc, t.act, seed_base + i);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        allPassed &= ok;
    }
    return allPassed;
}

// ---------------------------------------------------------------------------
// Ops table (used by RunAllTests and single-run mode)
// ---------------------------------------------------------------------------
struct OpEntry { unsigned op; const char* name; };

static const OpEntry kOps[] = {
    { OP_ADD,   "ADD"   },
    { OP_SUB,   "SUB"   },
    { OP_MUL,   "MUL"   },
    { OP_DIV,   "DIV"   },
    { OP_RELU,  "RELU"  },
    { OP_RELU6, "RELU6" },
};

// ---------------------------------------------------------------------------
// Full test suite
// ---------------------------------------------------------------------------
static bool RunAllTests() {
    const unsigned sizes[] = { 1, 3, 8, 9, 13, 64, 255, 256, 1023, 1024, 4097 };
    const unsigned nSizes  = sizeof(sizes) / sizeof(sizes[0]);
    const unsigned nOps    = sizeof(kOps)  / sizeof(kOps[0]);
    const unsigned nRandom = nSizes * nOps;

    bool allPassed = true;
    unsigned idx = 0;

    std::cout << "--- Random-value tests (" << nRandom << ") ---\n";
    for (unsigned o = 0; o < nOps; ++o) {
        for (unsigned s = 0; s < nSizes; ++s) {
            ++idx;
            std::cout << "[" << idx << "/" << nRandom << "] "
                      << kOps[o].name << "  size=" << sizes[s]
                      << " ... " << std::flush;
            const bool ok = RunCase(kOps[o].name, kOps[o].op, sizes[s],
                                    1u, 0u, 0u, ACT_NONE, kSeed + idx);
            std::cout << (ok ? "PASS" : "FAIL") << "\n";
            allPassed &= ok;
        }
    }

    allPassed &= RunSatTests();
    allPassed &= RunGeomTests(kSeed + 1000);

    const unsigned nTotal = nRandom
                          + sizeof(kSatTests)  / sizeof(kSatTests[0])
                          + sizeof(kGeomTests) / sizeof(kGeomTests[0]);
    std::cout << "\n"
              << (allPassed ? "All " : "FAILED — ")
              << nTotal << " tests"
              << (allPassed ? " passed.\n" : " had errors.\n");
    return allPassed;
}

int main(int argc, char** argv) {
    // Optional --dump-data <dir>: emit per-test a/b/c_ref hex files plus a
    // manifest, then exit (no kernel run).  Otherwise: original verify mode.
    std::vector<const char*> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if ((a == "--dump-data" || a == "-d") && i + 1 < argc) {
            g_dump_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--dump-data <dir>] [op size]\n"
                      << "  No arguments: full verify suite.\n"
                      << "  --dump-data <dir>: write hex fixtures + manifest "
                         "and exit.\n"
                      << "  op size: single random test "
                         "(op: 0=ADD 1=SUB 2=MUL 3=DIV 4=RELU 5=RELU6).\n";
            return 0;
        } else {
            rest.push_back(argv[i]);
        }
    }

    if (!g_dump_dir.empty()) {
        const std::string manifest_path = g_dump_dir + "/manifest.txt";
        g_manifest = std::fopen(manifest_path.c_str(), "w");
        if (!g_manifest) {
            std::cerr << "Failed to open " << manifest_path << " for writing\n";
            return 1;
        }
        std::fprintf(g_manifest,
            "# VectorOPKernel test fixture manifest\n"
            "# idx size op outer a_inc b_inc act label\n");

        std::cout << "VectorOP test data dump → " << g_dump_dir << "\n";
        const bool ok = RunAllTests();
        std::fclose(g_manifest);
        g_manifest = nullptr;
        std::cout << "Dumped " << g_test_idx << " test(s) to " << g_dump_dir << "\n";
        return ok ? 0 : 1;
    }

    if (rest.size() == 2) {
        const unsigned opCode = std::stoul(rest[0]);
        const unsigned size   = std::stoul(rest[1]);
        const unsigned nOps   = sizeof(kOps) / sizeof(kOps[0]);
        if (opCode >= nOps) {
            std::cerr << "op must be 0–" << (nOps - 1) << "\n";
            return 1;
        }
        std::cout << "Single test: op=" << kOps[opCode].name
                  << "  size=" << size << "\n";
        const bool ok = RunCase(kOps[opCode].name, kOps[opCode].op, size,
                                1u, 0u, 0u, ACT_NONE, kSeed);
        std::cout << (ok ? "PASS\n" : "FAIL\n");
        return ok ? 0 : 1;
    }
    if (!rest.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " [--dump-data <dir>] [op size]\n";
        return 1;
    }
    return RunAllTests() ? 0 : 1;
}
