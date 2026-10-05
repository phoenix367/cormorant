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

`kv260/68665fc1833a` is hw_128 7d8eefe (2026-10-05; the SystemVerilog MatmulKernel and
VectorOPKernel and the PoolingKernel out-of-contract guard, doc/plans/VECTOROP_RTL_PLAN.md
phase 3): 1581 calls, repeat spread median 0.017 %.  Its case list started from the converged
one of `bbb9a37f73f8` (the same scheduler); refinement rounds added 1 and 0 calls.  Against that
campaign the Conv, Matmul and Pool calls agree to a median 0.000 %; the 95 VectorOP calls
are a median 1.1 % faster (up to 38 % for jobs of many 2-word runs; four 0.5–1.2 % slower).
Its MatmulKernel families use the RTL job-walk terms (`rtl_*`).

Beside it, `kv260/bbb9a37f73f8`, the same design with the Vitis HLS VectorOPKernel
(2026-10-04), with 1580 calls, repeat spread median 0.018 % (four refinement rounds to
convergence: 122 + 31 + 8 + 3 calls); and `kv260/1d28630fbfa4`, without the pool guard
either (MATMUL_RTL_PLAN phases 3–4), with 1383 calls, repeat spread median 0.023 % (one
refinement round).  The 1251 calls both of those campaigns measured agree to a median
0.002 %.

And the last bitstream with the Vitis HLS MatmulKernel: `kv260/caa67f49a5a3`,
hw_128 d7ce129 (2026-09-28; Piper added 2026-10-01), 1505 calls, repeat spread
median 0.019 % — for projects generated with `AXI_MATMUL_IMPL=hls` for that
bitstream.
