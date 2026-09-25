"""Generate a human-readable markdown report describing what
inference-scheduler produced for a given ONNX model.

The report covers, at a glance:

  * model identity (path, hash, generation time, dtype),
  * graph inputs and outputs,
  * parameter (weight) statistics — total elements, bytes, inline vs
    external .dat split,
  * activation memory — pool slot count after event-timeline interval
    coloring, saving vs the naive sequential layout,
  * hardware lanes used and node counts per lane,
  * applied transformations — Gemm decomposition, Reshape folding,
    buffer-pool reuse, cross-lane parallelism,
  * a per-layer table covering every scheduled node, and
  * the list of files written to the output directory.

Pure read access — never mutates `OnnxGraph` or `CodeGenerator`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from typing import List, Optional

import numpy as np

from .dtype import ApFixed
from .nodes import (ConvNode, MatmulNode, PoolNode, ReshapeNode,
                    SpaceToDepthNode, ScheduledNode, OP_NAMES)
from .host_nodes import HostNode, SliceNode
from .codegen._simulate import _residual_stats


def _human_bytes(n: int) -> str:
    """Format a byte count as 'N B' / 'N KiB' / 'N MiB' / 'N GiB'."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n_f = n / 1024.0
        if n_f < 1024 or unit == "TiB":
            return f"{n_f:.2f} {unit}"
        n = n_f
    return f"{n:.2f} TiB"


def _grouped(n: int) -> str:
    """Thousand-grouped integer ('1 234 567')."""
    return f"{n:,}".replace(",", " ")


def _fmt_pct(x: float) -> str:
    """Render a ratio as a percentage with adaptive precision.

    Quantisation NRMSEs are typically ≪ 1%, so we want decimals that
    don't render as `0.00%`.  Below 0.01% we drop into scientific
    notation; above 1% we round to two decimals.
    """
    if x != x:                    # NaN
        return "n/a"
    if not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    pct = x * 100.0
    if pct >= 1.0:
        return f"{pct:.2f}%"
    if pct >= 0.01:
        return f"{pct:.3f}%"
    if pct == 0.0:
        return "0.000%"
    return f"{pct:.2e}%"


def _fmt_db(x: float) -> str:
    """Render an SQNR figure in dB.  Infinity → `∞`."""
    if x != x:
        return "n/a"
    if not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.1f} dB"


def _shape_str(shape) -> str:
    """`[1, 3, 224, 224]` → `[1,3,224,224]` (compact)."""
    return "[" + ",".join(str(d) for d in shape) + "]"


def _dt_now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _file_sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------------- #
# Per-node "notes" — short string describing op-specific parameters       #
# ----------------------------------------------------------------------- #

def _node_notes(sn) -> str:
    if isinstance(sn, ScheduledNode):
        bits = [OP_NAMES.get(sn.op_code, "?")]
        if sn.outer_count > 1:
            bits.append(f"broadcast×{sn.outer_count}")
        for fused in sn.fused_nodes:
            bits.append(f"+ {fused.op_type} (act)")
        return " · ".join(bits)
    if isinstance(sn, MatmulNode):
        bits = [f"{sn.n}×{sn.k}·{sn.k}×{sn.m}"]
        if sn.batch > 1:
            bits.append(f"batch={sn.batch}")
        if sn.outer_count > 1:
            bits.append(f"outer-loop×{sn.outer_count}")
        return " · ".join(bits)
    if isinstance(sn, ConvNode):
        bits = [
            f"k={sn.kh}×{sn.kw}",
            f"s={sn.stride_h}×{sn.stride_w}",
            f"p={sn.pad_top},{sn.pad_left}",
        ]
        if sn.has_bias:
            bits.append("+bias")
        if sn.is_depthwise:
            bits.append("depthwise")
        if sn.dilation_h != 1 or sn.dilation_w != 1:
            bits.append(f"dil={sn.dilation_h}×{sn.dilation_w}")
        return " · ".join(bits)
    if isinstance(sn, PoolNode):
        from .nodes import _GLOBAL_POOL_OP_TYPES, _POOL_NAMES
        op_type = sn.onnx_node.op_type
        bits = [_POOL_NAMES.get(sn.pool_type, "?")]
        if op_type in _GLOBAL_POOL_OP_TYPES:
            bits.append("global")
        else:
            bits.append(f"k={sn.pool_h}×{sn.pool_w}")
            bits.append(f"s={sn.stride_h}×{sn.stride_w}")
            bits.append(f"p={sn.pad_top},{sn.pad_left}")
        if sn.pool_type == 2:   # POOL_LP
            bits.append(f"lp={sn.lp_order}")
        if sn.pool_type == 1 and sn.count_include_pad:
            bits.append("count_pad")
        return " · ".join(bits)
    if isinstance(sn, ReshapeNode):
        return "buffer alias · no kernel call"
    if isinstance(sn, SpaceToDepthNode):
        return f"host CPU reorder · blocksize={sn.blocksize} · no kernel call"
    if isinstance(sn, SliceNode) and sn.is_view:
        return f"sub-buffer view @{sn.offset} · no copy"
    if isinstance(sn, HostNode):
        det = sn.describe()
        return f"host CPU{' · ' + det if det else ''} · no kernel call"
    return ""


def _node_inputs_compact(sn) -> str:
    parts = []
    for t in sn.inputs:
        tag = "W" if t.is_weight else t.onnx_name
        parts.append(f"{tag}{_shape_str(t.shape)}")
    return " · ".join(parts) if parts else "(none)"


# ----------------------------------------------------------------------- #
# Parallelism analysis — count starts that overlap a lane already busy    #
# ----------------------------------------------------------------------- #

def _count_overlapping_starts(events: list) -> int:
    """Number of starts that fire while at least one other node is
    already in flight on a different lane — i.e. genuine cross-lane
    overlap.

    The event tuple carries enough information to maintain the
    in-flight set without knowing each node's lane: every `start` adds
    a node, every `wait` / `drain` removes one.  `start_sync` is a
    synchronous helper that drains its own lane internally, so it
    never contributes to in-flight but still counts as overlapping if
    *other* lanes are busy when it fires.
    """
    in_flight: set = set()
    overlapping = 0
    for ev in events:
        kind = ev[0]
        if kind == "start":
            if in_flight:
                overlapping += 1
            in_flight.add(ev[1])
        elif kind in ("start_sync", "cpu"):
            if in_flight:
                overlapping += 1
            # the synchronous helper / host op drains itself — never in-flight
        elif kind in ("wait", "drain"):
            in_flight.discard(ev[2])
    return overlapping


# ----------------------------------------------------------------------- #
# Public API                                                               #
# ----------------------------------------------------------------------- #


class ReportGenerator:
    """Compose a markdown report describing the generated project.

    Construction is read-only against the supplied :class:`OnnxGraph`
    and :class:`CodeGenerator`; no codegen state is mutated.
    """

    def __init__(self, *, graph, codegen, model_path: str,
                 out_dir: str, generated_files: List[str]):
        self.graph    = graph
        self.codegen  = codegen
        self.model_path = model_path
        self.out_dir    = out_dir
        self.generated_files = list(generated_files)

        # Quantization is type-specific.  ApFixed truncates to a fixed
        # grid; Float32 has no quantization.  Cache the figures so the
        # markdown sections don't re-run the simulator multiple times.
        self._is_quantized = isinstance(self.codegen._dtype, ApFixed)
        if self._is_quantized:
            self._weight_q_errors = self._compute_weight_q_errors()
            self._layer_q_errors  = self.codegen.compute_quant_errors()
        else:
            self._weight_q_errors = {}
            self._layer_q_errors  = {}

    # ----- quantization-error helpers ------------------------------ #

    def _compute_weight_q_errors(self) -> dict:
        """``{onnx_name: (abs_max, rel_max)}`` for each constant initializer
        whose float source we can compare against its quantized form."""
        dtype = self.codegen._dtype
        out: dict = {}
        for t in self.graph.weight_tensors:
            if t.data is None:
                continue
            full  = t.data.reshape(t.shape).astype(np.float64)
            quant = dtype.quantize(full)
            out[t.onnx_name] = _residual_stats(full, quant)
        return out

    @staticmethod
    def _input_role(sn, position: int) -> str:
        """Human-readable role of `sn.inputs[position]`.

        For Conv this distinguishes weight vs bias; for binary VectorOP
        ops we just say `operand`; MatMul uses ONNX naming (`a` / `b`).
        """
        if isinstance(sn, ConvNode):
            return ("x", "weight", "bias")[position] if position < 3 else "?"
        if isinstance(sn, MatmulNode):
            return ("a", "b")[position] if position < 2 else "?"
        if isinstance(sn, ScheduledNode):
            return ("a", "b")[position] if position < 2 else "?"
        if isinstance(sn, PoolNode):
            return "x"
        return "input"

    def _weight_consumers(self, weight_name: str) -> str:
        """Return a compact string listing every layer that reads this
        weight, with the per-layer role.  Most weights have exactly one
        consumer; tied weights (rare in CV models) appear comma-separated.
        """
        parts = []
        for sn in self.graph.nodes:
            for i, t in enumerate(sn.inputs):
                if t.onnx_name == weight_name:
                    role = self._input_role(sn, i)
                    parts.append(
                        f"[{sn.index}] {sn.onnx_node.op_type} ({role})"
                    )
        return ", ".join(parts) if parts else "—"

    # ----- top-level entry point ------------------------------------ #

    def render_markdown(self) -> str:
        sections = [
            self._header(),
            self._inputs_outputs(),
            self._parameters(),
            self._activation_memory(),
            self._hardware_lanes(),
            self._transformations(),
            self._layers_table(),
            self._generated_files(),
        ]
        return "\n\n".join(s for s in sections if s) + "\n"

    # ----- §1 header ------------------------------------------------- #

    def _header(self) -> str:
        sha   = _file_sha256(self.model_path)
        dtype = self.codegen._dtype
        active_lanes = ", ".join(kd.name for kd in self.codegen._active_kernels) or "(none)"

        rows = [
            ("Model",             f"`{os.path.basename(self.model_path)}`"),
            ("Path",              f"`{os.path.abspath(self.model_path)}`"),
            ("SHA-256",           f"`{sha}`" if sha else None),
            ("Generated",         _dt_now_utc()),
            ("Output directory",  f"`{self.out_dir}`"),
            ("Data type",         f"`{dtype.name}` ({dtype.bytes_per_elem} byte/elem)"),
            ("Hardware lanes used", active_lanes),
        ]
        lines = ["# Inference scheduler report",
                 "",
                 "| Field | Value |",
                 "|-------|-------|"]
        for label, value in rows:
            if value is None:
                continue
            lines.append(f"| {label} | {value} |")
        return "\n".join(lines)

    # ----- §2 inputs / outputs --------------------------------------- #

    def _inputs_outputs(self) -> str:
        bpe = self.codegen._dtype.bytes_per_elem
        rows = ["## Inputs and outputs",
                "",
                "| Direction | Tensor | Shape | Elements | Bytes |",
                "|-----------|--------|-------|---------:|------:|"]
        for t in self.graph.input_tensors:
            rows.append(
                f"| Input  | `{t.onnx_name}` | {_shape_str(t.shape)} "
                f"| {_grouped(t.numel)} | {_grouped(t.numel * bpe)} |"
            )
        for t in self.graph.output_tensors:
            rows.append(
                f"| Output | `{t.onnx_name}` | {_shape_str(t.shape)} "
                f"| {_grouped(t.numel)} | {_grouped(t.numel * bpe)} |"
            )
        return "\n".join(rows)

    # ----- §3 parameters --------------------------------------------- #

    def _parameters(self) -> str:
        weights = self.graph.weight_tensors
        bpe     = self.codegen._dtype.bytes_per_elem
        total_e = sum(t.numel for t in weights)
        total_b = total_e * bpe
        large   = self.codegen.large_weight_tensors
        large_e = sum(t.numel for t in large)
        small   = [t for t in weights if t not in large]
        small_e = sum(t.numel for t in small)

        rows = ["## Parameters",
                "",
                "| Statistic | Value |",
                "|-----------|------:|",
                f"| Weight tensors | {_grouped(len(weights))} |",
                f"| Total parameters | {_grouped(total_e)} |",
                f"| Total bytes | {_grouped(total_b)} ({_human_bytes(total_b)}) |",
                f"| Embedded inline (≤ {self._weight_threshold()} elem) "
                f"| {_grouped(len(small))} tensors · {_grouped(small_e)} elem |",
                f"| External `.dat` files (> {self._weight_threshold()} elem) "
                f"| {_grouped(len(large))} tensors · {_grouped(large_e)} elem |"]

        if self._is_quantized and self._weight_q_errors:
            rows.append("")
            rows.append("### Weight quantization error (vs original float32)")
            rows.append("")
            rows.append(f"Active dtype: `{self.codegen._dtype.name}`. "
                        f"Each row reports the residual between the source "
                        f"float32 weights and the value the kernel actually "
                        f"reads, `dtype.quantize(w)` (round-to-nearest).")
            rows.append("")
            rows.append(
                "**Legend.** Per tensor, with residual `r = w − dtype.quantize(w)`:\n\n"
                "- `Max |abs|` — `max |r|` over all elements; tightest bound "
                "on any single value. Round-to-nearest gives `Max |abs| ≤ ½ "
                "LSB = 2^(I−W−1)`.\n"
                "- `NRMSE` — normalised RMS error, `‖r‖₂ / ‖w‖₂` (as a "
                "percentage). Treats the residual as noise and the original "
                "as signal, weighted by magnitude — so a single near-zero "
                "element cannot inflate it the way it does for max-elementwise "
                "relative error. Typical CV weights see NRMSE well below "
                "`0.1%` with `ap_fixed<16,8>`.\n"
                "- `SQNR (dB)` — signal-to-quantisation-noise ratio, "
                "`20·log₁₀(‖w‖₂ / ‖r‖₂)`. Higher is better; round-to-nearest "
                "16-bit on roughly-Gaussian weights typically yields ≥ 60 dB."
            )
            rows.append("")
            # Escape the literal `|` inside `|abs|` so it isn't parsed as
            # a column separator.
            rows.append("| Tensor | Shape | Elements | Used by "
                        "| Max \\|abs\\| | NRMSE | SQNR (dB) |")
            rows.append("|--------|-------|---------:|---------"
                        "|-----------:|------:|---------:|")
            # Aggregate row: worst-case across the set.
            all_abs = [s.abs_max for s in self._weight_q_errors.values()]
            all_nrmse = [s.nrmse for s in self._weight_q_errors.values()]
            all_sqnr = [s.sqnr_db for s in self._weight_q_errors.values()]
            agg_abs   = max(all_abs)   if all_abs   else 0.0
            agg_nrmse = max(all_nrmse) if all_nrmse else 0.0
            agg_sqnr  = min(all_sqnr)  if all_sqnr  else float("inf")
            rows.append(
                f"| **All weights (worst)** | — | — | — "
                f"| `{agg_abs:.3e}` | `{_fmt_pct(agg_nrmse)}` "
                f"| `{_fmt_db(agg_sqnr)}` |"
            )
            for t in weights:
                stats = self._weight_q_errors.get(t.onnx_name)
                if stats is None:
                    continue
                rows.append(
                    f"| `{t.onnx_name}` | {_shape_str(t.shape)} "
                    f"| {_grouped(t.numel)} | {self._weight_consumers(t.onnx_name)} "
                    f"| `{stats.abs_max:.3e}` "
                    f"| `{_fmt_pct(stats.nrmse)}` "
                    f"| `{_fmt_db(stats.sqnr_db)}` |"
                )
        return "\n".join(rows)

    @staticmethod
    def _weight_threshold() -> str:
        from .tensor import LARGE_WEIGHT_THRESHOLD
        return _grouped(LARGE_WEIGHT_THRESHOLD)

    # ----- §4 activation memory ------------------------------------- #

    def _activation_memory(self) -> str:
        bpe = self.codegen._dtype.bytes_per_elem
        layout, total_elems = self.codegen._compute_pool_layout()

        # Subtract weights — they sit at the start of the pool but are not
        # candidates for reuse.  The "intermediates region" is the part the
        # interval coloring affects.
        weight_names = {t.onnx_name for t in self.graph.weight_tensors}
        # Intermediates region size = total minus weights' raw allocs,
        # rounded for alignment within the pool.  Approximating from the
        # layout entries:
        interm_entries = [(n, off, a) for n, off, a in layout
                          if n not in weight_names]
        # Pool offsets are continuous; the intermediates region is the
        # tail of the pool after the last weight's slot.
        if interm_entries:
            interm_start = min(off for _, off, _ in interm_entries)
            interm_end   = max(off + a for _, off, a in interm_entries)
            # Slot count = number of distinct offsets across the
            # intermediates region (shared-slot tenants share an offset).
            distinct_offsets = {off for _, off, _ in interm_entries}
            slot_count = len(distinct_offsets)
        else:
            interm_start = total_elems
            interm_end   = total_elems
            slot_count   = 0

        # Naive baseline = each intermediate gets its own aligned slot.
        # We approximate this by summing 64-byte-aligned alloc per entry.
        align_to = 64 // bpe
        def _align_up(n): return (n + align_to - 1) & ~(align_to - 1)
        naive_elems = sum(_align_up(a) for _, _, a in interm_entries)

        actual_elems = interm_end - interm_start
        saving_elems = max(0, naive_elems - actual_elems)
        saving_pct   = (saving_elems / naive_elems * 100.0) if naive_elems else 0.0

        reshape_count = len(self.codegen._reshape_aliases)

        rows = ["## Activation memory",
                "",
                "| Statistic | Value |",
                "|-----------|------:|",
                f"| Intermediate tensors | {_grouped(len(self.graph.intermediate_tensors))} |",
                f"| ReshapeNode aliases (zero-cost) | {_grouped(reshape_count)} |",
                f"| Pool slots after coloring | {_grouped(slot_count)} |",
                f"| Pool size (intermediates) "
                f"| {_grouped(actual_elems)} elem ({_human_bytes(actual_elems * bpe)}) |",
                f"| Pool size (total incl. weights) "
                f"| {_grouped(total_elems)} elem ({_human_bytes(total_elems * bpe)}) |",
                f"| Naive baseline (no reuse) "
                f"| {_grouped(naive_elems)} elem ({_human_bytes(naive_elems * bpe)}) |",
                f"| Saving "
                f"| {_grouped(saving_elems)} elem ({_human_bytes(saving_elems * bpe)}, {saving_pct:.1f}%) |"]
        return "\n".join(rows)

    # ----- §5 hardware lanes ---------------------------------------- #

    def _hardware_lanes(self) -> str:
        from .kernels import KERNEL_REGISTRY
        active = {kd.name for kd in self.codegen._active_kernels}

        # Count nodes per lane.
        per_lane: dict = {kn: 0 for kn in KERNEL_REGISTRY}
        per_lane["(no lane)"] = 0
        for sn in self.graph.nodes:
            kn = getattr(type(sn), "kernel_name", "")
            per_lane[kn or "(no lane)"] = per_lane.get(kn or "(no lane)", 0) + 1

        rows = ["## Hardware lanes",
                "",
                "| Lane | Used? | Nodes |",
                "|------|:-----:|------:|"]
        for kn in KERNEL_REGISTRY:
            tick = "✓" if kn in active else "–"
            rows.append(f"| `{kn}` | {tick} | {_grouped(per_lane.get(kn, 0))} |")
        if per_lane["(no lane)"]:
            rows.append(f"| Reshape / host op (no lane) | – | {_grouped(per_lane['(no lane)'])} |")
        return "\n".join(rows)

    # ----- §6 transformations --------------------------------------- #

    def _transformations(self) -> str:
        events = self.codegen._compute_event_stream()
        overlapping = _count_overlapping_starts(events)
        starts = sum(1 for ev in events if ev[0] in ("start", "start_sync"))

        gemm = getattr(self.graph, "gemm_decomposed_count", 0)
        act_fused = getattr(self.graph, "act_fused_count", 0)
        s2d = getattr(self.graph, "s2d_stem_count", 0)
        reshape_count = sum(1 for sn in self.graph.nodes if isinstance(sn, ReshapeNode))

        # Pool reuse summary: re-derive the saving figure already shown in §4
        # (kept short here — §4 has the full breakdown).
        layout, _ = self.codegen._compute_pool_layout()
        weight_names = {t.onnx_name for t in self.graph.weight_tensors}
        interm = [(n, off, a) for n, off, a in layout if n not in weight_names]
        distinct_offsets = {off for _, off, _ in interm}
        merged = len(interm) - len(distinct_offsets)

        bullets = []
        if gemm:
            bullets.append(
                f"- **Gemm decomposition** — {gemm} `Gemm` "
                f"node{'s' if gemm != 1 else ''} rewritten to `MatMul` + `Add` "
                f"at load time."
            )
        else:
            bullets.append(
                "- **Gemm decomposition** — none (no `Gemm` nodes in the source graph)."
            )
        if act_fused:
            bullets.append(
                f"- **Activation fusion** — {act_fused} `Relu` / `Clip(0,6)` "
                f"node{'s' if act_fused != 1 else ''} folded into the producing "
                f"VectorOP call (kernel `act` register)."
            )
        if s2d:
            bullets.append(
                f"- **Space-to-depth stem** — {s2d} stride-2 `Conv`"
                f"{'s' if s2d != 1 else ''} rewritten as a host-side "
                f"`SpaceToDepth(2)` + stride-1 `Conv` over 4× the input "
                f"channels (ConvKernel IC-lane utilisation)."
            )
        n_host = sum(1 for sn in self.graph.nodes
                     if isinstance(sn, HostNode) and not (isinstance(sn, SliceNode) and sn.is_view))
        if n_host:
            bullets.append(
                f"- **Host-CPU ops** — {n_host} node{'s' if n_host != 1 else ''} run on "
                f"the CPU (double precision, round-half-even write-back, staged "
                f"through cached memory)."
            )
        if reshape_count:
            bullets.append(
                f"- **Reshape folding** — {reshape_count} `ReshapeNode` "
                f"output{'s' if reshape_count != 1 else ''} aliased to "
                f"the source buffer (no kernel call, no allocation)."
            )
        else:
            bullets.append(
                "- **Reshape folding** — no NOP-class layers in the graph."
            )
        if interm:
            bullets.append(
                f"- **Buffer-pool reuse** — {len(interm)} intermediate "
                f"tensor{'s' if len(interm) != 1 else ''} packed into "
                f"{len(distinct_offsets)} pool slot"
                f"{'s' if len(distinct_offsets) != 1 else ''} via "
                f"event-stream interval coloring (`{merged}` shared-slot pair"
                f"{'s' if merged != 1 else ''})."
            )
        if starts:
            bullets.append(
                f"- **Cross-lane parallelism** — {overlapping} of {starts} "
                f"kernel start{'s' if starts != 1 else ''} dispatched while "
                f"another lane was still in flight (overlap windows)."
            )

        return "## Applied transformations\n\n" + "\n".join(bullets)

    # ----- §7 layers ----------------------------------------------- #

    def _layers_table(self) -> str:
        rows = [f"## Layers ({_grouped(len(self.graph.nodes))})", ""]
        if self._is_quantized:
            rows.append("Quantization columns show the per-layer truncation "
                        "residual at the kernel's output: the difference "
                        "between the full-precision (float64) result and the "
                        "fixed-point value the hardware actually writes back. "
                        "ReshapeNode rows have no truncation and report `—`.")
            rows.append("")
            rows.append(
                "**Legend.** Per layer, with residual "
                "`r = y_full − dtype.truncate(y_full)` "
                "(`truncate_div` for `Div`):\n\n"
                "- `Out max |abs|` — `max |r|` over the output tensor; "
                "absolute upper bound on any single element. Floor-truncation "
                "gives `Max |abs| ≤ 1 LSB = 2^(I−W)`. `Relu`, `MaxPool`, "
                "integer `Add`/`Sub` preserve the grid and report `0`.\n"
                "- `NRMSE` — normalised RMS error of the output, "
                "`‖r‖₂ / ‖y_full‖₂` (as a percentage). Magnitude-weighted, "
                "robust to small-value outliers — this is the metric to "
                "watch for end-to-end model quality.\n"
                "- `SQNR (dB)` — signal-to-quantisation-noise ratio, "
                "`20·log₁₀(‖y_full‖₂ / ‖r‖₂)`. Higher = cleaner. Layers "
                "that don't truncate report `∞`."
            )
            rows.append("")
            rows.append("| # | Op | Lane | Inputs | Output | Notes "
                        "| Out max \\|abs\\| | NRMSE | SQNR (dB) |")
            rows.append("|--:|----|------|--------|--------|-------"
                        "|-----------:|------:|---------:|")
        else:
            rows.append("| # | Op | Lane | Inputs | Output | Notes |")
            rows.append("|--:|----|------|--------|--------|-------|")

        for sn in self.graph.nodes:
            lane = getattr(type(sn), "kernel_name", "") or "—"
            base = (
                f"| {sn.index} "
                f"| `{sn.onnx_node.op_type}` "
                f"| `{lane}` "
                f"| {_node_inputs_compact(sn)} "
                f"| `{sn.output.onnx_name}`{_shape_str(sn.output.shape)} "
                f"| {_node_notes(sn)} "
            )
            if self._is_quantized:
                stats = self._layer_q_errors.get(sn.output.onnx_name)
                if stats is None:
                    abs_c, nrmse_c, sqnr_c = "—", "—", "—"
                else:
                    abs_c   = f"`{stats.abs_max:.3e}`"
                    nrmse_c = f"`{_fmt_pct(stats.nrmse)}`"
                    sqnr_c  = f"`{_fmt_db(stats.sqnr_db)}`"
                base += f"| {abs_c} | {nrmse_c} | {sqnr_c} "
            rows.append(base + "|")
        return "\n".join(rows)

    # ----- §8 generated files --------------------------------------- #

    def _generated_files(self) -> str:
        rows = ["## Generated artifacts", ""]
        for rel in self.generated_files:
            rows.append(f"- `{rel}`")
        return "\n".join(rows)
