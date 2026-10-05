# Testing

Cormorant has five distinct testing layers, each independent of the
next. You can validate everything except final hardware on a host
machine without an FPGA.

| Layer | Needs | What it validates |
|-------|-------|-------------------|
| 1. **Python unit tests** | nothing | Inference scheduler correctness — codegen, DAG, layout, simulation, host ops, Llama / ViT ops, planning (1648 tests); the chat app (188 tests) |
| 2. **HLS C-sim** | gcc/g++, CMake (Verilator for the RTL MatmulKernel, VectorOPKernel and PoolingKernel) | Each kernel's C++ reference against per-test golden vectors, and the SystemVerilog MatmulKernel, VectorOPKernel and PoolingKernel in Verilator against the same fixtures (`ctest`) |
| 3. **RTL behavioural sim** | Vitis, Vivado | Per-kernel test-stand testbenches and the block-design testbench in xsim (no board) |
| 4. **On-device correctness** | KV260 over SSH, bitstream loaded | End-to-end model output checked against Python-simulated ground truth |
| 5. **On-device performance** | KV260 over SSH, bitstream loaded | Raw kernel throughput / latency benchmarks |

Layers 1–3 run on the host. Layers 4–5 run on the KV260 over SSH and
require the Cormorant bitstream to be loaded first
(see the [Quick start](../../README.md#quick-start) in the README
or [`inference-scheduler/doc/REMOTE_TESTING.md`](../../inference-scheduler/doc/REMOTE_TESTING.md#bitstream-upload-upload_bitstreampy)).

### Running the host layers in one go

`.claude/agents/run-tests/run_tests.py` runs layers 1 and 2, plus the
Piper library's host check, and prints one JSON report.  With Claude Code,
ask for the `run-tests` subagent (`.claude/agents/run-tests.md`): it runs the
helper and reads the report.

```bash
python3 .claude/agents/run-tests/run_tests.py --suite default    # scheduler + chat pytest, ruff, facts (~5 min)
python3 .claude/agents/run-tests/run_tests.py --suite all        # + csim (make + ctest), tts-host (~9 min)
python3 .claude/agents/run-tests/run_tests.py --suite scheduler --tests test/test_piper.py
python3 .claude/agents/run-tests/run_tests.py --list             # suites and baselines
```

- **What the report holds.**  For every failure: file:line, the assertion
  lines and failed subtests.  Also every skip, xfail and warning (grouped;
  marked project or third-party, known or new).
- **Anomalies against the baselines** in
  `.claude/agents/run-tests/baselines.json`: fewer tests than the baseline,
  a skip not listed there, a run under 40 % of its usual time, no tests
  collected, an odd exit code.
- **The `facts` suite** checks `facts.yaml`, the facts that code and docs
  repeat (test counts, register maps, supported ops, CLI and script flags,
  HTTP routes, ctypes bindings, config keys, pool sizes, board results), with
  `tools/facts/facts.py`
  ([README](../../tools/facts/README.md)).  Run it alone with
  `python3 tools/facts/facts.py check`; `fix` rewrites stale counts.
  `python3 tools/facts/facts.py install-hook` makes every `git commit`
  check the facts its staged files touch (a pre-commit hook).
- **Where things go.**  Logs, JUnit XML and `report.json` are written to
  `$CLAUDE_JOB_DIR/tmp/run-tests-<time>/` (or `/tmp/run-tests-<uid>-<time>/`).
- **The baselines.**  After adding tests,
  `--suite <s> --record-baseline` re-records one from a clean run.
- **Not included.**  The RTL behaviour tests (`--suite rtl`) run only on
  request, and the board layers are never run.

---

## 1. Python unit tests (no hardware)

The full Python test suite runs entirely on the host — no FPGA needed:

```bash
cd inference-scheduler

# Generate all test models first (one-time step)
.venv/bin/python test/gen_all_models.py

# Run all 1648 tests in 75 modules (all pass, none skipped; the first run
# downloads the 435 MB bertsquad-12 model for test_bert_base.py)
.venv/bin/python -m pytest test/ -q

# Run a specific module
.venv/bin/python -m pytest test/test_pool_alloc.py -v
```

`gen_all_models.py` runs every generator, including `gen_bert_models.py`
(tiny BERT-like fixtures in bertsquad-12's exact node arrangement, plus an
erf-GELU and a native-op opset-20 variant) and `gen_llama_models.py` (a
tiny random Llama: decode / prefill / head entries, which imports
`demo/chat/scripts/llm_study.py`).

Some tests run generated C on the host: `test/host_emu.py` compiles a
project's `inference.c` + `test_inference.c` unchanged against software
models of the VectorOP / Matmul / Conv drivers (register semantics of the
kernels, executed at Start; no Pool model) and a malloc-backed buffer pool,
runs it, and requires `test_inference PASSED` — the generated host-op code
(in place on cacheable buffers and staged on non-cacheable ones, lookup
tables, 1 / 3 / 4 host threads) and the kernel call parameters must
reproduce the scheduler simulation bit for bit (`test_bert_tiny.py`,
`test_split_int.py`, `test_matmul_on_conv.py`, `test_matmul_gemv.py`,
`test_numeric.py`, `test_llm_ops.py`, `test_llama.py`, `test_vit.py`,
`test_planning.py`, `test_conv_exp.py`, `test_piper.py`).  With `incoherent=True` it also gives
every buffer a separate "DDR" copy, so a missing cache sync changes the
output.
`test_cache_coherency.py` walks the emitted `inference_run()` of every model
with a per-buffer cache-state model and fails on any missing flush /
invalidate at a CPU ↔ kernel hand-off (the DMA buffers are mapped cacheable
on the board).  `test_bert_base.py` does the
same for BERT-base and checks the simulation against the BERT study's
independent emulation.  It runs on the real model (~40 s, ~6 GB RAM):
- **The files.**
  - `demo/bert_squad/assets/models/bertsquad-12-simplified.onnx`, plus
    `vocab.txt` and `dev-v1.1.json` from `demo/bert_squad/assets/`.
  - On the first run it downloads whatever is missing with
    `demo/bert_squad/scripts/fetch_assets.py`: 435 MB from Google Drive,
    md5-checked.
- **Other locations.** `BERT_SQUAD_MODEL` / `BERT_SQUAD_ASSETS` point it
  elsewhere.
- **No download.** `BERT_SQUAD_DOWNLOAD=0` skips it instead of downloading.
  A failed download skips it too, with the error as the reason.

```bash
.venv/bin/python ../demo/bert_squad/scripts/fetch_assets.py   # optional: fetch ahead of time
.venv/bin/python -m pytest test/test_bert_base.py -v
```

**CI.**  `.github/workflows/inference-scheduler-tests.yml` runs this layer
on GitHub's hosted Ubuntu runners on every push to `main` / pull request that touches
`inference-scheduler/`, `platforms/` or `demo/bert_squad/scripts/`.  It runs:
1. `ruff check .`;
2. `test/gen_all_models.py`;
3. `demo/bert_squad/scripts/fetch_assets.py`, for `test_bert_base.py`.  Its
   435 MB download is cached under a key named after the model's md5, and a
   failed download only makes those tests skip;
4. pytest with coverage.

The fixtures generated by `gen_all_models.py` include hardware-bound
boundary models derived from `platforms/<AXI_PLATFORM>.json`
(see [`PLATFORM_CONFIGURATION.md`](PLATFORM_CONFIGURATION.md) → *Editing
an existing platform's bounds*) — re-run the generators after any
`max_*` change in a platform JSON.

### Chat app tests

The chat server's tests ([`demo/chat/doc/DEVELOPMENT.md`](../../demo/chat/doc/DEVELOPMENT.md#tests))
run with the same venv, from the repo root:

```bash
inference-scheduler/.venv/bin/python -m pytest demo/chat/tests -q   # 188 tests
```

About 60 of them skip in a fresh clone, until the assets they read are
present: the SmolLM2 tokenizer (`demo/chat/scripts/llm_calibrate.py fetch`),
the SmolVLM tokenizer (`demo/chat/scripts/vlm_study.py fetch`), BERT's
`vocab.txt` (`demo/bert_squad/scripts/fetch_assets.py vocab`)
and Pillow.

---

## 2. HLS C-sim (no hardware)

Each kernel ships with a C++ reference test executed by `ctest`.  The
kernel headers include the Vitis HLS headers, so source Vitis first:

```bash
source <Xilinx>/2025.2/Vitis/settings64.sh
cd build && cmake ..

make -j8              # every C-simulation executable
ctest                 # run all registered tests
```

Plain `make` builds every test executable `ctest` registers: `TestSimulation`
(VectorOPKernel), `TestConvRef` and `TestConvGrid` (ConvKernel, MAC-grid unit
test), `TestMatmulRef` (MatmulKernel), `TestPoolingSim` (PoolingKernel) and,
when a BLAS is found at configure time, `TestMatmulBlas`.  Or build them one
by one: `make TestSimulation TestConvRef TestConvGrid TestMatmulRef
TestPoolingSim` (+ `TestMatmulBlas`); a registered test whose executable is
missing shows as *Not Run* and fails `ctest`.  `ctest` also runs
`TestConvSweep` (`TestConvRef --sweep 300`, randomised geometries, ~1 min).

These tests exercise the kernel C++ source directly without HLS
synthesis, so they catch logic regressions in seconds.

The SystemVerilog MatmulKernel (`kernels/matmul_rtl/`,
[MATMUL_RTL_KERNEL](../kernels/MATMUL_RTL_KERNEL.md)) has two tests in the
same `ctest` run when Verilator 5.x is installed (`sudo apt install
verilator`): `TestMatmulRtl` builds the RTL with Verilator (~30 s, in plain
`make`) and runs the 50 checked-in MatmulKernel fixtures — the ones
`TestMatmulRef --dump-data` writes from `ref_matmul_2d` — plus 200 random
cases against randomised AXI timing (~50 s); `MatmulRtlDriver` checks the C
driver's register table against the RTL.  `make lint_matmul_rtl` is the
Verilator lint; the testbench's own options (one case, a waveform, `--perf`)
are in the kernel's reference.

The SystemVerilog VectorOPKernel (`kernels/vectorop_rtl/`,
[VECTOROP_RTL_KERNEL](../kernels/VECTOROP_RTL_KERNEL.md)) likewise:
`TestVectorOpRtl` (in plain `make`) runs the 119 checked-in VectorOP
fixtures — the ones `TestSimulation --dump-data` writes — plus 300 random
jobs checked against the HLS kernel's C++ under randomised AXI timing;
`VectorOpRtlDriver` checks the C driver's register table against the RTL;
`make lint_vectorop_rtl` is its lint.

The SystemVerilog PoolingKernel (`kernels/pool_rtl/`,
[POOL_RTL_KERNEL](../kernels/POOL_RTL_KERNEL.md)) likewise: `TestPoolRtl`
(in plain `make`) runs the 45 checked-in PoolingKernel fixtures — the ones
`TestPoolingSim --dump-data` writes — plus 200 random jobs checked against
the HLS kernel's C++ under randomised AXI timing; `PoolRtlDriver` checks the
C driver's register table against the RTL; `make lint_pool_rtl` is its lint.

---

## 3. Hardware simulation (Vivado, no board)

RTL behavioural simulation in Vivado xsim — no board required, but the
per-kernel IP archives (ConvKernel's HLS synthesis; the RTL MatmulKernel's, VectorOPKernel's and PoolingKernel's packaging) must exist first.

```bash
cd build

# Prerequisite: the four kernel IPs (ConvKernel's HLS synthesis + the RTL
# MatmulKernel's, VectorOPKernel's and PoolingKernel's packaging, ~5–10 min); Vitis's
# settings64.sh puts vitis-run, vivado and xclbinutil on PATH
source <Xilinx>/2025.2/Vitis/settings64.sh
make synthesize_kv260

# Per-kernel RTL behaviour tests (hw/cormorant_test_stand submodule):
# checked-in golden fixtures under hw/test_data/<kernel>_test_data/
make behavior_test_vectorop   # also: behavior_test_conv / _matmul / _pool
make behavior_test            # all four in sequence

# Block-design behavioural sim against the SystemVerilog testbench
# (hw/cormorant_hw_128 submodule; ~3 min, all four kernels through the PS VIP)
make sim_hw_kv260
```

Each `behavior_test_<k>` depends on its kernel's IP target —
`synthesize_conv_kv260`, `package_matmul_rtl`, `package_vectorop_rtl`,
`package_pool_rtl` — (so it rebuilds its kernel's IP and driver directory) and fails when
the scoreboard report records any mismatch (see
[`BUILD_TARGETS.md`](BUILD_TARGETS.md) §RTL behavior tests).  The fixture
manifests currently hold 119 VectorOP, 63 Conv, 50 Matmul (11 of them GEMV) and 45 Pool
cases, and all pass (`VectorOP Test Summary: 119 / 119 passed`, …): each
kernel alone on the C-simulation fixtures.  They modify tracked
files of the `hw/cormorant_test_stand` submodule (`.bd` / `.xci` / `.xpr`);
do not commit them.

**`sim_hw_kv260`** runs the whole block design (the four kernels, the
interconnects, the PS VIP's DDR model) against constant-fill test cases
in `hw/cormorant_hw_128/cormorant_hw_128.srcs/sim_1/new/`.  It ends with:

```
##########################################################
##  CORMORANT TESTBENCH — OVERALL RESULTS
##########################################################
##        VectorOPKernel   22 /  22  (0 failed)
##            ConvKernel   17 /  17  (0 failed)
##          MatmulKernel   10 /  10  (0 failed)
##         PoolingKernel   19 /  19  (0 failed)
##########################################################
##  TOTAL: 68 / 68 passed
##  ALL TESTS PASSED
##########################################################
```

`scripts/sim.tcl` exits 1 unless `simulate.log` contains `ALL TESTS
PASSED` (a missing log or an early stop, e.g. an AXI protocol-checker
fatal, is a failure).  The testbench writes every buffer the way the
kernels read it: whole 16-byte words, row starts aligned (VectorOP
`a_inc` / `b_inc` are 0 or multiples of 8 elements) and ConvKernel
weights in the packed tile-major layout (`tb_functions.svh`
`conv_const_weights`); the LpPool p=2 reference mirrors the kernel's
fixed-point `poly_sqrt` bit for bit.

For PS-VIP / xsim quirks observed during simulation development see
[`SIMULATION_ISSUES.md`](SIMULATION_ISSUES.md).

---

## 4. On-device correctness tests (KV260)

End-to-end correctness testing over SSH. `run_remote_tests.py`
generates a C inference project per ONNX model, uploads it, builds on
the board, executes the test binary, and compares every output element
against Python-simulated ground truth.

**Prerequisite:** the Cormorant bitstream must be loaded on the board.
Use `upload_bitstream.py` (see the README Quick start or
[`REMOTE_TESTING.md`](../../inference-scheduler/doc/REMOTE_TESTING.md#bitstream-upload-upload_bitstreampy)).

The tracked template `inference-scheduler/remote_config.json.example`
lists every on-board test model (148; narrow the list, or pass
`--models`, to test a subset of kernels). Copy it and fill in your board
details:

```bash
cd inference-scheduler

cp remote_config.json.example remote_config.json
$EDITOR remote_config.json
```

The two fields you must set are:

- **`ssh.host`** — IP address or hostname of the KV260 (`"kv260.local"` in the example)
- **`local.driver_dirs`** — paths on your host machine to the
  driver sources for each kernel (Vitis HLS generates ConvKernel's;
  `make driver_matmul_rtl` / `driver_vectorop_rtl` / `driver_pool_rtl`
  write MatmulKernel's, VectorOPKernel's and PoolingKernel's; the example's
  `../build/…` paths, relative to `inference-scheduler/`, fit a build in
  `<repo>/build`), e.g.:

  ```
  "VectorOPKernel": "<repo>/build/kernels/vectorop_rtl/driver/VectorOPKernel_v1_0/src"
  "ConvKernel":     "<repo>/build/kernels/conv/kv260/conv_kv260/hls/impl/ip/drivers/ConvKernel_v1_0/src"
  "PoolKernel":     "<repo>/build/kernels/pool_rtl/driver/PoolingKernel_v1_0/src"
  ```

The `remote.uio_devices` map must list the UIO sysfs name for every
kernel in the config. After loading the `design_cormorant.dtbo`
overlay all four names are `fabric_vecop`, `fabric_matmul`,
`fabric_conv`, and `fabric_pool`.  `run_remote_tests.py` keys the map by
the scheduler's kernel names — `VectorOPKernel`, `MatmulKernel`,
`ConvKernel`, **`PoolKernel`** (not `PoolingKernel`; the example uses
these keys).

```bash
# Verify board prerequisites before running
.venv/bin/python run_remote_tests.py --config remote_config.json --check-only

# Run all models in the config
.venv/bin/python run_remote_tests.py --config remote_config.json

# Run a specific model
.venv/bin/python run_remote_tests.py --config remote_config.json \
    --models test/models/single_add.onnx
```

For full SSH setup, config reference, and debugging guide see
[`REMOTE_TESTING.md`](../../inference-scheduler/doc/REMOTE_TESTING.md).

---

### 4.x On-board correctness set — coverage added 2026-09-25

The on-board set had **144 models** on 2026-09-25 (`remote_config.json.example`
now lists 148: those plus the four tiny-BERT fixtures `bert_tiny_*`).  The
18 added with the 128-bit port and packed-B work target what those changes
touch:

- **MatMul packed B** (`mm_packed_*`): the MNIST Gemm (256×10), the
  ResNet-18 (512×1000) and MobileNet v2 (1280×1001) classifier shapes,
  batched constant B (stride rescale), the 4D×3D outer loop over a packed
  constant, two packed layers with `m` not a multiple of 16, and
  `mm_packed_then_activation` — a packed MatMul followed by a row-major one
  in the same inference, which fails if `b_packed` is not rewritten on
  every call (the stale-register failure of 2026-09-25).
- **Row-major guard**: `mm_shared_const_row_major` (a constant read by a
  MatMul and an Add must stay row-major) and `mm_unaligned_rows` /
  `mm_relu_then_packed_odd` (13- and 5-element rows so the 128-bit A/B
  ports extract lanes at every shift; A from an intermediate tensor).
- **Pool 128-bit x**: 13-, 17- and 77-column rows (the last forces
  ow-tiling), three channel tiles, batch slices at odd element offsets, a
  7×7 global pool over two channel tiles.
- **Conv weight path at scale**: `conv_fc_7x7_64to256` (LeNet conv3-style
  one-pixel layer, 1.6 MB packed weights), `conv_1x1_classifier_1024`
  (8 ic-tiles × 32 M-groups) and `conv_mgroups_prefetch` (half-tile last
  ic-tile × 3 M-groups for the §2.35 prefetch).

## 5. Performance benchmarking (KV260)

`run_remote_perf.py` measures raw kernel throughput and latency.
Unlike the correctness runner it does not check output values — it
only times how fast each kernel runs for a configurable set of
parameter cases.

The script uploads a single self-contained C benchmark project,
builds all four kernel binaries in one pass, then runs each case
and reports results. Kernels whose driver files are absent are not
built (their cases fail), so benchmark only what is currently
deployed with `--kernels` or `"enabled": false`.

Copy the example config and fill in your board details before the
first run:

```bash
cd inference-scheduler
cp perf_config.json.example perf_config.json
$EDITOR perf_config.json   # set ssh.host; the example's remote.uio_devices are the
                           # fabric_* names and its local.driver_dirs point at <repo>/build
```

The copy (`perf_config*.json`, like the other per-user configs) is not
tracked.

**Config file:** `perf_config.json` — extends the same SSH schema as
the correctness configs with an additional `benchmarks` section.

```json
"benchmarks": {
  "VectorOPKernel": { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "MatmulKernel":   { "enabled": true, "warmup": 10, "cases": [ ... ] },
  "ConvKernel":     { "enabled": true,  "warmup": 10, "cases": [ ... ] },
  "PoolingKernel":  { "enabled": true,  "warmup": 10, "cases": [ ... ] }
}
```

`perf_config.json.example` ships with **60 benchmark cases**
across the four kernels (15 VectorOPKernel, 24 MatmulKernel — including
row-major / packed-B twins of the FC and classifier shapes, `b_packed`
case field, MATMUL_OPTIMISATION §3b, and GEMV twins, `gemv_kw` case
field, §8b — 10 ConvKernel, 11 PoolingKernel).

VectorOPKernel `op` values outside the supported range (0..5) are
rejected when the cases are loaded, before any case runs (right
after the config, before it connects to the board) — see the *Case fields*
reference in `REMOTE_TESTING.md`.

```bash
cd inference-scheduler

# Full benchmark run (all enabled kernels)
.venv/bin/python run_remote_perf.py --config perf_config.json

# Specific kernels only
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --kernels VectorOPKernel MatmulKernel

# Override iteration and warmup counts for a quick spot-check
.venv/bin/python run_remote_perf.py --config perf_config.json \
    --iters 20 --warmup 5

# Preflight check — verify board is ready without running benchmarks
.venv/bin/python run_remote_perf.py --config perf_config.json --check-only

# Also write the results as JSON (one entry per case: kernel, label,
# case fields, ok, lat_ms, metric)
.venv/bin/python run_remote_perf.py --config perf_config.json --json /tmp/perf.json
```

`perf_calibrate.py run --config <this format>` measures the kernel calls
behind the performance models of the scheduler's `--plan` mode (`cases` /
`run` / `fit`; `host` / `simulate` work from board profiles); its data
lives in `inference-scheduler/perf_models/kv260/` (see
`perf_models/README.md` and TACTICS_PLAN.md §9).

**Sample output** (2026-09-29, hw_128 d7ce129 at 100 MHz, the 60-case
`perf_config.json.example` set; abbreviated — the full table is in
[REMOTE_TESTING.md](../../inference-scheduler/doc/REMOTE_TESTING.md#reading-the-report)):

```
  VectorOPKernel
  ───────────────────────────────────────────────────────────────────────────────────
  Label                    Parameters                        Lat(ms)      GB/s
  ───────────────────────────────────────────────────────────────────────────────────
  ADD-1K                   ADD    size=1024    outer=1        0.0087     0.703
  ADD-4K                   ADD    size=4096    outer=1        0.0175     1.406
  ADD-16K                  ADD    size=16384   outer=1        0.0483     2.034
  ADD-64K                  ADD    size=65536   outer=1        0.1711     2.298
  ADD-256K                 ADD    size=262144  outer=1        0.6627     2.373
  ...
  ───────────────────────────────────────────────────────────────────────────────────
                                                 peak GB/s                4.604
                                               min latency     0.0087
  15/15 OK

  MatmulKernel
  ────────────────────────────────────────────────────────────────────────────────────
  Label                     Parameters                        Lat(ms)    GOps/s
  ────────────────────────────────────────────────────────────────────────────────────
  8x8x8                     N=8    K=8    M=8    batch=1       0.0114     0.090
  16x16x16                  N=16   K=16   M=16   batch=1       0.0209     0.391
  32x32x32                  N=32   K=32   M=32   batch=1       0.0499     1.313
  ...
  FC-1x1536x576-gemv-kw4    N=1    K=1536 M=576  batch=1 GEMV kw=4     0.5675     3.118
  ────────────────────────────────────────────────────────────────────────────────────
                                                peak GOps/s                4.679
                                                min latency     0.0114
  24/24 OK

  ConvKernel
  ─────────────────────────────────────────────────────────────────────────────────────
  Label                      Parameters                        Lat(ms)    GOps/s
  ─────────────────────────────────────────────────────────────────────────────────────
  3x3-1ch-28x28-32out        1ch 28x28→32ch 3x3k                0.1490     3.031
  3x3-1ch-28x28-32out-b16    1ch 28x28→32ch 3x3k                2.0737     3.484
  3x3-64ch-56x56             64ch 56x56→64ch 3x3k               2.6830    86.175
  ...
  dw-3x3-64ch-56x56          64ch 56x56→64ch 3x3k               1.0129     3.567
  ─────────────────────────────────────────────────────────────────────────────────────
                                                 peak GOps/s               86.175
                                                 min latency     0.1490
  10/10 OK

  PoolingKernel
  ────────────────────────────────────────────────────────────────────────────────────
  Label                     Parameters                        Lat(ms)      GB/s
  ────────────────────────────────────────────────────────────────────────────────────
  MaxPool-2x2-56x56         MaxPool 2x2 64ch 56x56             0.3352     1.497
  MaxPool-2x2-28x28         MaxPool 2x2 64ch 28x28             0.1161     1.080
  MaxPool-3x3-56x56         MaxPool 3x3 64ch 56x56             0.3385     1.482
  ...
  ────────────────────────────────────────────────────────────────────────────────────
                                                  peak GB/s                1.498
                                                min latency     0.0254
  11/11 OK
  ── OVERALL: All 60 cases passed ──
```

| Metric | Meaning |
|--------|---------|
| `Lat(ms)` | Mean kernel wall-clock time per call (after warmup) |
| `GB/s` | Memory bandwidth — VectorOPKernel and PoolingKernel |
| `GOps/s` | Arithmetic throughput — MatmulKernel and ConvKernel |
| `peak` | Best metric across all passing cases in the group |

For the full config reference, per-kernel case field definitions, and
debugging guide see the *Performance Benchmarking* section of
[`REMOTE_TESTING.md`](../../inference-scheduler/doc/REMOTE_TESTING.md).

---

## UIO device names

After loading the `design_cormorant.dtbo` overlay the four kernels
appear as:

| Kernel | UIO sysfs name |
|--------|----------------|
| `VectorOPKernel` | `fabric_vecop` |
| `MatmulKernel` | `fabric_matmul` |
| `ConvKernel` | `fabric_conv` |
| `PoolingKernel` | `fabric_pool` |

Verify on the board: `cat /sys/class/uio/uio*/name` (the Kria image lists
four `axi-pmon` devices first, uio0–3; the kernels follow as uio4–7).
