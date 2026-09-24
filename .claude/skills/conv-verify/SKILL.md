---
description: Verify a conv-kernel optimisation end-to-end on the KV260 platform — build and run C-sim, synthesise the HLS IP, run the RTL behavior test, then diff per-test timing against the most recent run. Use after any edit to kernels/conv/, the conv CMakeLists, the synthesis TCL template, or the platform JSON.
allowed-tools: Bash Read
---

# conv-verify

Four sequential gates. **If any gate fails, stop immediately and report the failure** — do not run later gates on a broken build.

## Locate the build directory

The user may have placed `build/` inside the repo or anywhere else (out-of-source builds are common). Before running any gate, locate it once:

```bash
# Prefer the conventional in-repo location; fall back to a shallow find.
if [ -f build/CMakeCache.txt ]; then
    BUILD_DIR=$(pwd)/build
else
    BUILD_DIR=$(find . -maxdepth 4 -name CMakeCache.txt -path '*/build/CMakeCache.txt' | head -n 1 | xargs -r dirname)
fi
echo "BUILD_DIR=$BUILD_DIR"
```

If `BUILD_DIR` ends up empty, ask the user where the build tree is and stop. Otherwise `cd "$BUILD_DIR"` and run all `make` commands from there.

Note `BUILD_DIR` in your scratch state for Gate 4's report-path argument.

## Gate 1 — C-simulation (three nets)

```bash
cd "$BUILD_DIR"
make TestConvRef TestConvGrid
ctest -R TestConv --output-on-failure
```

`ctest -R TestConv` runs all three C-sim nets: `TestConvRef` (the named
cases, bit-exact vs the naive oracle), `TestConvGrid` (the MAC array in
isolation — every (kh, kw, ic_valid, m_valid) tile against a scalar loop)
and `TestConvSweep` (`TestConvRef --sweep 300`: random scheduler-admissible
geometries; this is the net that catches interactions such as the §2.21
stale-line-buffer bug).  All three must pass.  If the sweep fails, re-run
it with `--seed <n>` for a second sample and add the failing geometry to
TestConvSim.cpp as a named case before fixing anything.

- Every named line must end in `PASS`; **any `FAIL` line or `FAILED: N
  element mismatch(es)` is a regression**, even if the final line looks OK.
- A `line_buf residency` or `stream token accounting` assert (C-sim only
  invariants) is a regression too.

## Gate 1b — fixtures (only when the bench or a DDR layout changed)

The RTL run reads the CHECKED-IN fixtures under `hw/test_data/conv_test_data/`.
They must be regenerated when a named case was added / changed, or when
the packed weight / bias layout changed (ConvKernel.h "Weight / bias port
width and DDR layout"):

```bash
make gen_conv_test_data
rm -f ../hw/test_data/conv_test_data/test_*.hex ../hw/test_data/conv_test_data/manifest.txt
cp conv_test_data/* ../hw/test_data/conv_test_data/
```

Keep new fixtures SMALL (xsim ≈ 20 µs simulated per wall-clock second;
the 40-case set runs in ~12 min).  If the layout changed, the testbench's
element-count formulas in `hw/cormorant_test_stand/.../conv_tb.sv` must
match `conv_weight_numel` / `conv_bias_numel`.

## Gate 2 — HLS synthesis

```bash
make synthesize_conv_kv260
```

- Wait for it (~60–90 s on this machine, sometimes longer — conv has more loops than pool). The final line must be `[100%] Built target synthesize_conv_kv260`.
- The bash exit code must be 0. Any `ERROR:` line in the tool output is a hard failure.
- After it succeeds, glance at the synthesis summary for new violations (path is relative to the build directory):

  ```bash
  sed -n '15,80p' kernels/conv/kv260/conv_kv260/hls/syn/report/csynth.rpt
  ```

  (The conv build uses the Vitis unified component flow, so reports live
  under `<component>/hls/syn/report/`, not the legacy `solution1/syn/report/`.)

  Report any of these against the prior run (baseline as of §2.34:
  top-level slack **0.00 ns**, **no** II violations, ports
  `gmem0 16 -> 16`, `gmem1 128 -> 128`, `gmem2 128 -> 128`, `gmem3 16 -> 16`,
  BRAM 158 (54 %), DSP 248, FF ~34.9 k, LUT ~47.3 k (40 %), URAM 24):
  - **Top-level slack** going negative, or any sub-block slack that worsened.
  - **Any** `II Violation Information` entry — the design has none; the
    grid loop (`ConvMacGrid.h`, iteration latency 6) and the fused consumer
    loops are all II=1.
  - `m_axi_gmem0..3` data-width column changes.  `gmem0/1/2` are READ_ONLY
    (x, w, b), `gmem3` is WRITE_ONLY (y); x and y are 16-bit burst_maxi
    ports, weight and bias 128-bit burst_maxi ports — a change here means
    the port type or the layout contract moved.
  - Resource jumps (BRAM / DSP / FF / LUT % columns on the top-level
    `ConvKernel` row) — flag anything >10 % of the previous value.  BRAM is
    the tight resource on the full 128-bit design (56 % placed).

  Don't fail the gate on these — the user wants to see them in the report — but list any change clearly.

## Gate 3 — RTL behavior test

```bash
make behavior_test_conv
```

- Takes ~2–3 min (Vivado xsim). The final two lines must be of the form:

  ```
  [ts] kernel=ConvKernel  total=N  passed=N  failed=0  all_passed=True
  [ck] ConvKernel: PASS  (N/N)  …/conv_test_report.json
  ```

- `failed=0` and `all_passed=True` are mandatory. **If anything else, stop here.**

## Gate 4 — Timing comparison vs most-recent run

The report and baseline live under `$BUILD_DIR/kernels/conv/kv260/`. Pass `--report` so the script doesn't assume the conventional in-repo build location:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/compare_conv_timing.py" \
    --report   "$BUILD_DIR/kernels/conv/kv260/conv_test_report.json" \
    --baseline "$BUILD_DIR/kernels/conv/kv260/conv_timing_last.json"
```

The script reads the freshly-written `conv_test_report.json`, compares per-test `duration_ns` against `conv_timing_last.json` (the previous run's snapshot), prints a delta table sorted by absolute movement, and overwrites `conv_timing_last.json` with the current run so the next invocation has a fresh baseline.

- First-ever run: there's no baseline yet, the script prints absolute values and saves a snapshot. Note this in your reply so the user knows the next run will produce a real diff.
- Add `--no-save` if you want a one-off comparison without overwriting the baseline (e.g. to keep a known-good reference while testing a speculative change). Use this when the user explicitly says "don't update the baseline".

## When a timing delta is unexplained

Run the `conv-cycle-model` skill on the moved cases first (it predicts the
RTL within ~10 %); if the model disagrees with the measurement, trace ONE
scaled case with the `conv-rtl-trace` skill before touching the kernel —
the write path, the read path and a chunk-boundary "stall" all looked
like compute problems until traced.

## Reporting back to the user

After all four gates pass, summarise in this order:

1. **Pass/fail status** of each gate (one line each).
2. **Total wall-clock delta** vs the previous run — both ns and % (taken straight from the comparison script's TOTAL row).
3. **Top movers** — the 3-5 tests with the largest absolute deltas, listed as "Test name: prev → now (Δns, ±X%)".
4. **Synthesis-report changes** — any new timing/II violations, data-width column changes, or notable resource jumps, from Gate 2.

Keep the report tight: one short paragraph per section, no extra prose.

## When `make` reports an unknown target

If any gate prints `make: *** No rule to make target '<X>'. Stop.` (e.g. `behavior_test_conv` is missing), the build tree was configured against a different branch — its cached files predate a CMakeLists addition/removal. From inside `$BUILD_DIR`, refresh once and retry the same `make` command:

```bash
cmake .
```

`cmake .` re-runs configure in place using the existing cache; targets that exist on the current branch get registered. Don't fall back to deleting the build tree — that loses the synthesis cache and forces a full ~60+ s HLS re-run.

## What to skip

- Don't rebuild dependencies the user hasn't touched (other kernels, the inference scheduler, etc.).
- Don't proactively re-run cmake unless a CMakeLists or `platforms/*.json` was edited — the tcl is auto-regenerated on those changes via `CMAKE_CONFIGURE_DEPENDS`, so `make` alone picks it up. (Exception: the unknown-target case above, where the cmake refresh is the targeted fix.)
- Don't read the full `csynth.rpt` (it's >500 KB for conv) — slice with `sed -n` or grep for what you need.
