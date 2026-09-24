---
description: Trace ONE ConvKernel RTL case with the testbench's +VERBOSE AXI / FIFO probes and analyse the log — burst lengths, AW→AW and W-beat spacing, bursts in flight, read-request spacing, acc_stream push/pop rate, per-process done times. Use when a conv timing delta is unexplained, a layer is slower than the cycle model predicts, or a kernel hangs on the board; run on a scaled-down fixture, never the whole suite.
allowed-tools: Bash Read
---

# conv-rtl-trace

Every non-obvious ConvKernel finding after the 2-D grid (adapter buffering
whole write bursts, the deferred-tail flush, one read burst in flight, the
burst_maxi response deadlock, a "stall" that was chunk 1's compute) came
from ONE verbose RTL case plus a per-beat analysis of its log.  This skill
packages that.  A run takes ~2–4 min for a fixture with < 20 k outputs.

## 1. Pick or build the fixture

Fixtures are the packed hex files `TestConvRef --dump-data` writes
(x / w / b / y + manifest, one 16-bit value per line).  Use an existing
index from `build/conv_test_data/manifest.txt`, or add a named case to
`kernels/conv/test/TestConvSim.cpp` first (keep it SMALL — xsim is
CPU-bound at ~20 µs of simulated time per wall second; a 64-ch 16×16 3×3
case is ~0.5 ms simulated ≈ 2 min).

```bash
# from the repo root; BUILD_DIR as in conv-verify (default build/)
make -C build gen_conv_test_data                     # refresh dumps
scripts/make_fixture.sh build 21 /tmp/trace_fixture  # one case → dir
```

`make_fixture.sh <build-dir> <test-index> <out-dir>` copies that case's
four hex files and a 2-line-header manifest into `<out-dir>`.

## 2. Run the verbose testbench

```bash
scripts/run_trace.sh build /tmp/trace_fixture /tmp/trace.log
```

This runs `make -C hw/cormorant_test_stand tb-conv` with `TS_VERBOSE=1`
(the probes are compiled in but silent without it) against the IP repo
in `<build-dir>/kernels/conv/kv260/conv_kv260/hls/impl/ip` — i.e. the
LAST synthesised kernel; re-run `make synthesize_conv_kv260` first if the
source changed.  The report goes next to the log.

## 3. Analyse

```bash
python3 scripts/analyze_trace.py /tmp/trace.log
```

Prints, per test in the log:

- **gmem3 (y)**: write bursts (LEN histogram), AW→AW spacing, W-beat
  spacing split into intra-burst vs inter-burst gaps, bursts in flight at
  each AW, AW→B latency, gaps > 1 µs.  Healthy after §2.30: intra-burst
  10 ns, ≤ 8 in flight, no gap longer than a chunk's compute.
- **gmem0 (x)**: AR bursts and AR→AR spacing.  Healthy after §2.28: ~30 ns
  between a row's channel requests; 690 ns means one burst in flight.
- **acc_stream**: pushes / pops per 1024, occupancy, and any push gap
  > 1 µs with the pushes/pops at that moment.  A gap at a multiple of
  `chunk_rows·out_w·out_ch` pushes is the next chunk's compute, not a stall.
- **process done times** (weight producer, patch producer, consumer,
  writer) to see which stage finishes last.

## Reading the numbers

Compare against the cycle model (`conv-cycle-model` skill) for the same
geometry: compute ≈ pixels · groups · (G·kh·kw + 2G + 6), fill ≈ weights
/ 8 per (chunk, ict, owt, mg), drain = outputs at 1/cycle, writer ≈ drain
unless bursts are shorter than 16 beats.  Anything the model does not
explain shows up here as a spacing or an in-flight count.

## Board hangs

If the kernel hung ON THE BOARD, first read its registers
(`board-deploy` skill, `read_kernel_regs.sh`) to get the layer geometry,
scale it down to a fixture that keeps the suspect structure (run
length, chunk count, tile count) and trace that.  The AXI VIP's
forward-progress watchdog turns a deadlock into a `$finish` with
"Pending AW ... no progress" and no report — that is the reproduction.
