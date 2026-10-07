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

`kv260/588d721997cb` is the production design (2026-10-07, SOFTMAX_PLAN: `6436623029f7` plus
VectorOPKernel's softmax unit): 1858 calls, `clock_mhz` 250, repeat spread median 0.057 %.  A
fresh case list (1443 cases: BERT's, SmolVLM's and the prefills' softmaxes on VectorOP), three
refinement rounds (139 + 126 + 78 calls; a fourth would add 27), then the prefill softmax at
every runtime key count (`LlmAttnSoftmaxVopNode` is priced at `keys_of(sn)` keys, as the
attention convs are: 72 calls).  The simulator against the board: the CNNs, BERT, SmolLM2-135M
and SmolVLM within ±2 %; SmolLM2-360M's prefill 16 −12 %, 256 +4 % (its host ops priced by the
kinds' fits).  `host.json` unchanged.

Before it, `kv260/6436623029f7` — the production design from 2026-10-06 to 2026-10-07
(ACTIVATIONS_PLAN: `986cef4866a0` plus VectorOPKernel's activation unit) — had 1930 calls, `clock_mhz` 250, repeat spread median
0.078 %.  First a fresh case list (BERT's 12 GELUs now ride on VectorOP Adds; 1402 cases) and three
refinement rounds (157 + 103 + 67 calls) to 1729 calls; topped up on 2026-10-07 (OFFLOAD_PLAN:
SmolVLM's vision GELU and Piper's decoder sums on VectorOP, the engine cost model refitted) with a
fresh case list (1406 cases) and three refinement rounds (125 + 29 + 5 calls) to 1852, then
for the chat libraries' decode attention on ConvKernel (KV_DECODE_PLAN) a fresh case list
(1438 cases) and two rounds (118 + 26 calls); `host.json` unchanged.

Before it, `kv260/986cef4866a0` — the 250 MHz design (2026-10-06, FMAX_250_PLAN: the kernels on
an MMCM at 250 MHz, register slices, the kernels' register trees) — had 2011 calls, `clock_mhz` 250, repeat
spread median 0.099 % (the calls are 2.5× shorter; 268 above 0.5 %).  Its campaign started from
the converged case list of `c2b2a6e5e50e` (the shipped calls are the same: the engine choices
do not depend on the clock), then five refinement rounds: 108 + 59 + 33 + 69 + 13 calls.
`host.json` unchanged.  The family fits are loose at 250 MHz — conv / conv-dw / conv-mm held-out
p90 31 / 50 / 64 %, mm-gemv 18.5 %, mm-tiled 7.1 %: their features come from the engine cost
model, whose board terms (DDR latency, host overhead in cycles) were fitted at 100 MHz.  The
planner trusts no family and prices from the exact calls until that model is refitted
(FMAX_250_PLAN §5).

Before it, `kv260/c2b2a6e5e50e`, hw_128 850cc88 (2026-10-06; all four kernels in SystemVerilog at
100 MHz — CONV_RTL_PLAN phase 3), 2343 calls, repeat spread median 0.017 %.  Its campaign started
from the converged case list of `dbb320fb7297` (one pass, a refinement round of 136 calls),
then the list was rebuilt with the scheduler's RTL ConvKernel cost model, whose engine and
geometry choices change the shipped ConvKernel calls (BERT's attention P·V moves onto
ConvKernel), and refined to convergence: 181 + 93 + 32 + 16 + 5 calls.  Against
`dbb320fb7297` the 1042 ConvKernel calls both campaigns measured are a median 28.4 % faster
(up to 73.5 %; one depthwise grid case 8.4 % slower), the MatmulKernel, PoolingKernel and
VectorOPKernel calls agree to a median 0.00 %.  The ConvKernel families use the RTL
kernel's walk terms (`rtl_total`, `rtl_fill`, `rtl_loads` from `cost_model.rtl_conv_walk`):
conv / conv-dw held-out p90 9.4 / 12.1 % (from 39.4 / 31.6 % with the HLS terms),
conv-mm 42.8 %; mm-gemv / mm-tiled 1.44 / 3.14 %.

Before that, `kv260/dbb320fb7297`, hw_128 7d8eefe (2026-10-05; the SystemVerilog MatmulKernel,
VectorOPKernel and PoolingKernel — POOL_RTL_PLAN phase 3 — with the ConvKernel the one Vitis
HLS kernel; `AXI_CONV_IMPL=hls` for projects on it), 1581 calls, repeat spread median 0.016 %.  Its case list started from the
converged one of `b3309f424562`; the first refinement round added 0 calls.  Against that
campaign the 30 PoolingKernel calls are a median 19.5 % faster (2.7–36.1 %); the Conv,
Matmul and VectorOP calls agree to a median 0.00 % (within ±2.7 %).  mm-gemv / mm-tiled
held-out p90 0.79 / 3.22 %.

Before it, `kv260/b3309f424562`, the same design with the Vitis HLS PoolingKernel (and its
out-of-contract guard; 2026-10-05, the MatmulKernel IP declaring its m_axi bus parameters —
MATMUL_RTL_PLAN phase 5), 1581 calls, repeat spread median 0.015 %.  Its case list started
from the converged one of `68665fc1833a`; the first refinement round added 0 calls.  Against
that campaign the 414 MatmulKernel calls are unchanged at the median, 90 faster by more than
1 % (up to 15 %), 3 slower by 1–2.2 %; mm-gemv / mm-tiled held-out p90 0.89 / 3.13 % (from
2.91 / 4.20 %).  The MatmulKernel families of both use the RTL job-walk terms (`rtl_*`).

Further back, `kv260/68665fc1833a`, the same design with every MatmulKernel crossbar slot at
2 outstanding bursts (2026-10-05, VECTOROP_RTL_PLAN phase 3), with 1581 calls, repeat
spread median 0.017 %: its case list started from the converged one of `bbb9a37f73f8`
(refinement rounds of 1 and 0 calls); against that campaign the Conv, Matmul and Pool calls
agree to a median 0.000 %, the 95 VectorOP calls are a median 1.1 % faster.

Before that, `kv260/bbb9a37f73f8`, the same design with the Vitis HLS VectorOPKernel
(2026-10-04), with 1580 calls, repeat spread median 0.018 % (four refinement rounds to
convergence: 122 + 31 + 8 + 3 calls); and `kv260/1d28630fbfa4`, without the pool guard
either (MATMUL_RTL_PLAN phases 3–4), with 1383 calls, repeat spread median 0.023 % (one
refinement round).  The 1251 calls both of those campaigns measured agree to a median
0.002 %.

And the last bitstream with the Vitis HLS MatmulKernel: `kv260/caa67f49a5a3`,
hw_128 d7ce129 (2026-09-28; Piper added 2026-10-01), 1505 calls, repeat spread
median 0.019 % — for projects generated with `AXI_MATMUL_IMPL=hls` for that
bitstream.
