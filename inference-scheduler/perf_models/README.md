# Performance models

The measurements and models the scheduler's optional planning mode
(`--plan`) prices its choices with: see
[`doc/plans/TACTICS_PLAN.md`](../../doc/plans/TACTICS_PLAN.md) and
`doc/scheduler/INFERENCE_SCHEDULER.md` §"Planning".  One folder per
platform.

| file | written by | holds |
|---|---|---|
| `<bitstream-id>.cases.json` | `perf_calibrate.py cases [--refine]` | the calls to measure: every kernel call of the shipped models and of their MatMul tactics, the space-filling set, the refinement calls |
| `<bitstream-id>.calib.json` | `perf_calibrate.py run` | the board measurements, two passes (mean / min / max / sd µs per call) |
| `<bitstream-id>.json` | `perf_calibrate.py fit` | the kernel model: exact µs per measured call, a fitted model per kernel family with its calibrated ranges and held-out error, the determinism check |
| `host.json` | `perf_calibrate.py host [--merge]` | the host-op model: measured µs per op signature from board profiles, a fit per op kind (independent of the bitstream); `--merge` adds a new model's kinds and keeps the others |

`<bitstream-id>` is the first 12 hex digits of the SHA-256 of the flat
bitstream the board loads (`/lib/firmware/pl.bin`), computed locally from
the Vivado `.bit` (`src/perf_calls.py`).  A new bitstream needs a new
campaign; planning refuses a model made for another one.

`kv260/caa67f49a5a3` is hw_128 d7ce129 (2026-09-28; Piper added 2026-10-01): 1505 calls,
repeat spread median 0.019 %.

Beside it, the bitstream with the SystemVerilog MatmulKernel (hw_128 7d8eefe
built with `AXI_MATMUL_IMPL=rtl`, 2026-10-04; doc/plans/MATMUL_RTL_PLAN.md
phase 3): `kv260/1d28630fbfa4`, 1383 calls (the shipped models with the RTL
engine choices, one refinement round), repeat spread median 0.023 %.  Its
MatmulKernel families use the RTL job-walk terms (`rtl_*`).
