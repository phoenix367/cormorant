---
description: Analytical cycle model of ConvKernel — the SystemVerilog kernel's pipeline recurrence (default, the scheduler's cost_model.rtl_conv_walk) or the retired HLS kernel's walk (--arch 37 … 42) — per-layer cycles split into MAC sweep, weight fill, drain and input loads, for an ONNX model or a single geometry, from the platform JSON's kernels.conv bounds. Use before choosing the next conv optimisation, to explain a timing delta, or to size a fixture; re-validate after any kernel change.
allowed-tools: Bash Read
---

# conv-cycle-model

**The SystemVerilog ConvKernel (default, `--arch rtl`).**  The script runs
the scheduler's own model, `cost_model.rtl_conv_walk` with the
ideal-memory parameters `RTL_CONV_SIM`: the kernel's sweeps as a pipeline
recurrence — the weight loader running ahead of the two cache banks, the
patch producer at the pace of the line buffer or the x loader, the drain of
chunk n beside chunk n + 1 (no bias pass: `bias` is always 0 %) — tuned to
the Verilator testbench on the 1 042 ConvKernel calls of the calibration
case list (median error 0.65 %, p90 5.3 %; `vl/Vtb --case-file F
--no-oracle` gives the cycle counts).  `cost_model.conv_board_cycles` uses
the board's parameters (`RTL_CONV_BOARD`, the c2b2a6e5e50e campaign:
median 3.1 %, p90 14.6 %); the misses are narrow, tall MatMul-on-ConvKernel
jobs whose short x runs wait on the DDR.  To change the model, change it
there — `test_matmul_on_conv.py` pins its anchors.

**The retired HLS kernel (`--arch 37 … 42`).**  The script's own walk
reproduces the HLS ConvKernel's loop structure as of CONV_OPTIMISATION §2.42 — oh-chunking with the M-group residency cap,
ow-tiling (even tile widths), M-grouping, the flat sweep over output-pixel
PAIRS (`G · max(kh·kw, 2)` cycles per pair), one-word-per-cycle weight
fill from the 128-bit port with half tiles (ping-pong overlapped), the
8-lane Phase-3 drain, the `x_row_loader` row loads — and adds up ideal-II
cycles plus the measured pipeline ramps.  `--arch 37 … 41` models the
earlier steps (to re-validate their reports).  It tracked
the RTL behavior test within 6 % at every step of the 2-D grid plan
(CONV_2D_GRID_PLAN.md §7a) and is what located the write path (§2.22),
the fill (§2.32) and the read path (§2.28) before any hardware ran.

```bash
# whole models (needs the scheduler venv for onnx)
inference-scheduler/.venv/bin/python .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py \
    demo/image_classification/assets/models/resnet18-simplified-fused.onnx [more.onnx ...]

# one geometry:  C M H W kh kw [sh sw dh dw pad_top pad_left pad_bottom pad_right dw]
python3 .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py --case 8 64 16 16 3 3 1 1 1 1 1 1 1 1 0

# validate against the last behaviour test (prints model vs measured per named case;
# the test stand's DDR model adds a few % to the ideal-memory model)
python3 .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py --validate build/kernels/conv/kv260/conv_test_report.json
```

Output per layer: total cycles and ms at the board's kernel clock (250 MHz), and the
share of MAC sweep / weight fill / bias init / drain+write / input
loads, plus per model the totals and the fill-heaviest layers.  Shares
above ~30 % in one bucket are where the next step is.

## Accuracy

The RTL kernel: see above; against the test stand's 63 cases (2026-10-06)
5.7 % mean |error| above 20 k cycles.

The HLS kernel, latest (`--arch 42`, CONV_OPTIMISATION §2.42, 57 RTL cases): **5.0 %**
mean |error| on the cases above 20 k cycles, 22.6 % over all (the
sub-2 k-cycle stubs are dominated by the fixed invocation overhead).
Validate against a report from the current kernel — an older
`conv_test_report.json` in the build tree gives meaningless errors.

Validated 2026-09-24 against the 40-case RTL report (§2.34 kernel):

Mean |error| 6.5 % over all cases, 7.9 % over cases above 20 k cycles;
the 3×3 / 7×7 M-grouped cases are within 5 %, the DW chunking case
-6 %.  Known biases: 1×1 layers with long output runs are OVER-estimated
(~+20 %) because the model treats input row loads as fully serial with
compute and the 1×1 producer overlaps part of them; the 7×7 stem and the
ow-tiling case are UNDER-estimated (-12…-15 %, per-tile / per-request
latencies not modelled).  A fixed 1 000-cycle invocation overhead
covers the tiny cases.

**On the board** (962 calls measured on bitstream 1d28630fbfa4,
2026-10-04) the model is the RTL simulation's: median error 20.5 %, and
it UNDER-estimates calls whose weight slabs are bound by their requests
— one per output channel of `kh·kw·16` elements, 2 words for a 1×1
kernel, 8 in flight — where the simulation's DDR model answers fast: a
1×1 kernel on ≤ 256 pixels and > 64 output channels is 2–6× short
(BERT's per-head P·V: 0.23 ms modelled, 0.61 ms on the board).  The
scheduler's `cost_model.conv_board_cycles` adds that (12 cycles per
request, the sweep hiding one word per iteration; median error 18.7 %,
p90 37 %) for its engine choices; this script and `conv_cycles` stay the
RTL simulation's model.

## Rules

- **Re-validate after every kernel change** (`--validate`).  If a case
  is off by more than ~10 %, fix the model (a new loop, a changed
  ramp) BEFORE using it to plan — the model's value is that it has been
  right so far.
- Numbers are per kernel invocation at the platform clock; the demo's
  end-to-end latency also contains matmul / pool / vectorop and host
  time, so a 20 % conv saving is less end to end.
- The drain/write bucket assumes a single chunk's drain is serial with
  compute (true for single-chunk layers; §2.26 overlaps it only when a
  next chunk exists).
- Fill is bounded by the 128-bit port (8 elements / cycle) and by
  `stream_load_weights`' one-request-per-m1 issue.
