---
description: Analytical cycle model of ConvKernel (post-§2.34 architecture) — per-layer cycles split into MAC sweep, weight fill, bias init, drain/write and input loads, for an ONNX model or a single geometry, from the platform JSON's kernels.conv bounds. Use before choosing the next conv optimisation, to explain an RTL timing delta, or to size a fixture; re-validate against one RTL case after any kernel change.
allowed-tools: Bash Read
---

# conv-cycle-model

`scripts/conv_cycle_model.py` reproduces ConvKernel's loop structure —
oh-chunking with the M-group residency cap, ow-tiling, M-grouping, the
fused (tile, khi, kwi) sweep, one-word-per-cycle weight fill from the
128-bit port with half tiles, one-element-per-cycle Phase-3 drain — and
adds up ideal-II cycles plus the measured pipeline ramps.  It tracked
the RTL behavior test within 6 % at every step of the 2-D grid plan
(CONV_2D_GRID_PLAN.md §7a) and is what located the write path (§2.22),
the fill (§2.32) and the read path (§2.28) before any hardware ran.

```bash
# whole models (needs the scheduler venv for onnx)
inference-scheduler/.venv/bin/python .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py \
    demo/image_classification/assets/models/resnet18-simplified-fused.onnx [more.onnx ...]

# one geometry:  C M H W kh kw [sh sw dh dw pad_top pad_left pad_bottom pad_right dw]
python3 .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py --case 8 64 16 16 3 3 1 1 1 1 1 1 1 1 0

# validate against the last RTL run (prints model vs measured per named case)
python3 .claude/skills/conv-cycle-model/scripts/conv_cycle_model.py --validate build/kernels/conv/kv260/conv_test_report.json
```

Output per layer: total cycles and ms at the platform clock, and the
share of MAC sweep / weight fill / bias init / drain+write / input
loads, plus per model the totals and the fill-heaviest layers.  Shares
above ~30 % in one bucket are where the next step is.

## Accuracy (validated 2026-09-24 against the 40-case RTL report, §2.34 kernel)

Mean |error| 6.5 % over all cases, 7.9 % over cases above 20 k cycles;
the 3×3 / 7×7 M-grouped cases are within 5 %, the DW chunking case
-6 %.  Known biases: 1×1 layers with long output runs are OVER-estimated
(~+20 %) because the model treats input row loads as fully serial with
compute and the 1×1 producer overlaps part of them; the 7×7 stem and the
ow-tiling case are UNDER-estimated (-12…-15 %, per-tile / per-request
latencies not modelled).  A fixed 1 000-cycle invocation overhead
covers the tiny cases.

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
