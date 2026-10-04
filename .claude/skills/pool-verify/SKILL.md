---
description: Verify a pool-kernel optimisation end-to-end on the KV260 platform — build and run C-sim, synthesise the HLS IP, run the RTL behavior test, then diff per-test timing against the most recent run. Use after any edit to kernels/pool/, the pool CMakeLists, the synthesis TCL template, or the platform JSON.
allowed-tools: Bash Read
---

# pool-verify

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

## Gate 1 — C-simulation

```bash
cd "$BUILD_DIR"
make TestPoolingSim
./kernels/pool/TestPoolingSim
```

- The build must finish with `[100%] Built target TestPoolingSim`.
- The test run must end with a line like `49 / 49 tests passed.` (the count grows with the suite; it must equal the total) — every test must pass. **Any line of the form `[FAIL] …` is a regression**, even if the summary line still reports the right pass count for the rest. Scan the output for `[FAIL]`/`failures=N/M` with N>0 before continuing.

## Gate 2 — HLS synthesis

```bash
make synthesize_pool_kv260
```

- Wait for it (~45 s on this machine, sometimes longer). The final line must be `[100%] Built target synthesize_pool_kv260`.
- The bash exit code must be 0. Any `ERROR:` line in the tool output is a hard failure.
- After it succeeds, glance at the synthesis summary for new violations (path is relative to the build directory):

  ```bash
  sed -n '15,70p' kernels/pool/kv260/pool_kv260/hls/syn/report/csynth.rpt
  ```

  (The pool build uses the Vitis unified component flow, so reports live
  under `<component>/hls/syn/report/`, not the legacy `solution1/syn/report/`.)

  Report any of these against the prior run (baseline as of POOL_OPTIMISATION §2.14: top-level slack 0.00 ns, II=1 and `Pipelined = yes` on every pipelined loop, no `SCHED 204-65`, both ports `128 -> 128`):
  - **New** `Issue Type | Timing` rows (negative-slack stages) or a negative top-level slack.
  - **New** `II Violation Information` entries (II larger than 1 on previously II=1 loops).
  - `m_axi_gmem0` (x read) / `m_axi_gmem1` (y write) data-width column changes (both `128 -> 128`).

  Don't fail the gate on these — the user wants to see them in the report — but list any change clearly.

## Gate 3 — RTL behavior test

```bash
make behavior_test_pool
```

- Takes ~2 min (Vivado xsim). The final two lines must be of the form:

  ```
  [ts] kernel=PoolingKernel  total=43  passed=43  failed=0  all_passed=True
  [ck] PoolingKernel: PASS  (43/43)  …/pooling_test_report.json
  ```

- `failed=0` and `all_passed=True` are mandatory. **If anything else, stop here.**

## Gate 4 — Timing comparison vs most-recent run

The report and baseline live under `$BUILD_DIR/kernels/pool/kv260/`. Pass `--report` so the script doesn't assume the conventional in-repo build location:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/compare_pool_timing.py" \
    --report   "$BUILD_DIR/kernels/pool/kv260/pooling_test_report.json" \
    --baseline "$BUILD_DIR/kernels/pool/kv260/pool_timing_last.json"
```

The script reads the freshly-written `pooling_test_report.json`, compares per-test `duration_ns` against `pool_timing_last.json` (the previous run's snapshot), prints a delta table sorted by absolute movement, and overwrites `pool_timing_last.json` with the current run so the next invocation has a fresh baseline.

- First-ever run: there's no baseline yet, the script prints absolute values and saves a snapshot. Note this in your reply so the user knows the next run will produce a real diff.
- Add `--no-save` if you want a one-off comparison without overwriting the baseline (e.g. to keep a known-good reference while testing a speculative change). Use this when the user explicitly says "don't update the baseline".

## Reporting back to the user

After all four gates pass, summarise in this order:

1. **Pass/fail status** of each gate (one line each).
2. **Total wall-clock delta** vs the previous run — both ns and % (taken straight from the comparison script's TOTAL row).
3. **Top movers** — the 3-5 tests with the largest absolute deltas, listed as "Test name: prev → now (Δns, ±X%)".
4. **Synthesis-report changes** — any new timing/II violations, or data-width column changes, from Gate 2.

Keep the report tight: one short paragraph per section, no extra prose.

## When `make` reports an unknown target

If any gate prints `make: *** No rule to make target '<X>'. Stop.` (e.g. `behavior_test_pool` is missing), the build tree was configured against a different branch — its cached files predate a CMakeLists addition/removal. From inside `$BUILD_DIR`, refresh once and retry the same `make` command:

```bash
cmake .
```

`cmake .` re-runs configure in place using the existing cache; targets that exist on the current branch get registered. Don't fall back to deleting the build tree — that loses the synthesis cache and forces a full ~45 s HLS re-run.

## What to skip

- Don't rebuild dependencies the user hasn't touched (other kernels, the inference scheduler, etc.).
- Don't proactively re-run cmake unless a CMakeLists or `platforms/*.json` was edited — the tcl is auto-regenerated on those changes via `CMAKE_CONFIGURE_DEPENDS`, so `make` alone picks it up. (Exception: the unknown-target case above, where the cmake refresh is the targeted fix.)
- Don't read the full `csynth.rpt` (it's >450 KB) — slice with `sed -n` or grep for what you need.
