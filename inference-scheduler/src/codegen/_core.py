"""
C code generator for the inference scheduler.

CodeGenerator produces three files for an inference project:

  generate_header()  →  include/inference.h
      Public API: Data_t typedef, array-size macros,
      inference_init() and inference_run() declarations.

  generate_source()  →  src/inference.c
      Implementation: weight arrays, intermediate buffers,
      static run_op() helper, inference_init(), inference_run().
      Includes "inference.h" for the shared type and declarations.

  generate_cmake()   →  CMakeLists.txt
      Builds a static library 'inference' from src/inference.c
      and the XVectoropkernel driver sources in driver/.

The generated code targets the Xilinx KV260 (bare-metal and Linux):
  - XVectoropkernel driver API  (driver/xvectoropkernel.h)
  - Xil cache maintenance API   (xil_cache.h, from BSP; no-op on Linux)
  - DMA-capable buffers via inference_buf_t (src/inference_buf.c)
  - ap_fixed<16,8> element type

Physical-address model
----------------------
VectorOPKernel's AXI master ports read/write DDR using physical addresses
stored in AXI-Lite registers (Set_a / Set_b / Set_c).  On Linux, virtual
pointers from malloc/stack are NOT valid DDR addresses.

inference_buf_t abstracts this:
  - Linux:      dma-proxy pool; physical address from /proc/self/pagemap
                (requires root; dma_alloc_coherent memory is contiguous).
  - Bare-metal: malloc (virtual == physical on Xilinx standalone).

inference_buf_phys() returns the physical address to program into the kernel
registers; inference_buf_ptr() returns the virtual pointer for CPU access.
"""

from __future__ import annotations
from typing import List, Optional

from ..graph   import OnnxGraph
from ..nodes    import (ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode,
                        SpaceToDepthNode, SchedulerError)
from ..host_nodes import HostNode, SliceNode
from ..kernels  import KernelDesc, KERNEL_REGISTRY
from ..schedule import Dag
from ..tensor  import TensorInfo
from ..dtype   import DataType, AP_FIXED_16_8
from ..layout  import TensorLayout


class _CoreMixin:
    """Core state and shared helpers for CodeGenerator."""

    def __init__(self, graph: OnnxGraph, model_path: str,
                 embed_large_weights: bool = False,
                 embed_large_expected: bool = False,
                 dtype: Optional[DataType] = None) -> None:
        self._graph                = graph
        self._model_path           = model_path
        self._embed_large_weights  = embed_large_weights
        self._embed_large_expected = embed_large_expected
        # Data type descriptor — controls encoding, quantization, and C type
        # declarations.  Defaults to ap_fixed<16,8> (the standard KV260 type).
        # Pass dtype=FLOAT32 (or any DataType subclass) for other types.
        self._dtype               = dtype if dtype is not None else AP_FIXED_16_8
        # Unified layout map: {onnx_name: TensorLayout} for every tensor.
        # Accounts for broadcast alignment gaps and propagated padding.
        # _alloc_sizes is derived for backward-compatibility with callers that
        # only need the flat {name: alloc_elements} view.
        self._layouts             = self._compute_tensor_layouts()
        self._alloc_sizes         = {k: v.alloc for k, v in self._layouts.items()}
        self._demote_strided_views()

    # ------------------------------------------------------------------
    # Buffer aliases: Reshape-family nodes and Slice views
    # ------------------------------------------------------------------

    @staticmethod
    def _is_view(sn) -> bool:
        """A Slice piece emitted as a zero-cost sub-buffer view."""
        return isinstance(sn, SliceNode) and sn.is_view

    @classmethod
    def _is_alias_node(cls, sn) -> bool:
        """Nodes that do no work at run time: their output shares (part of)
        the source's buffer (ReshapeNode, Slice view)."""
        return isinstance(sn, ReshapeNode) or cls._is_view(sn)

    def _alias_source_map(self) -> dict:
        """{output onnx_name: source onnx_name} for every alias node."""
        return {sn.output.onnx_name: sn.inputs[0].onnx_name
                for sn in self._graph.nodes if self._is_alias_node(sn)}

    def _alias_root(self, name: str) -> str:
        """Follow Reshape aliases and Slice views down to the buffer that
        owns the memory."""
        src = self._alias_source_map()
        seen = set()
        while name in src and name not in seen:
            seen.add(name)
            name = src[name]
        return name

    def _reshape_root(self, name: str) -> str:
        """Follow ReshapeNode aliases only (same bytes, same layout)."""
        src = {sn.output.onnx_name: sn.inputs[0].onnx_name
               for sn in self._graph.nodes if isinstance(sn, ReshapeNode)}
        seen = set()
        while name in src and name not in seen:
            seen.add(name)
            name = src[name]
        return name

    @property
    def _view_aliases(self) -> dict:
        """{view onnx_name: (root c_name, element offset, numel)} for every
        Slice view.  The source is resolved through Reshape aliases to the
        pool buffer that owns the memory (OnnxGraph guarantees one exists)."""
        result: dict = {}
        tensors = {t.onnx_name: t for t in self._graph.intermediate_tensors}
        for sn in self._graph.nodes:
            if self._is_view(sn):
                root = self._reshape_root(sn.inputs[0].onnx_name)
                result[sn.output.onnx_name] = (tensors[root].c_name, sn.offset,
                                               sn.output.numel)
        return result

    def _demote_strided_views(self) -> None:
        """A view can only share memory with a flat (gap-free) buffer and be
        flat itself; if a broadcast consumer gave either a strided layout the
        piece is copied on the host instead."""
        def flat(name):
            lay = self._layouts.get(name)
            return lay is None or (lay.n_chunks == 1 and lay.alloc == lay.numel)
        for sn in self._graph.nodes:
            if self._is_view(sn):
                root = self._reshape_root(sn.inputs[0].onnx_name)
                if not (flat(root) and flat(sn.output.onnx_name)):
                    sn.is_view = False

    def _compute_tensor_layouts(self) -> dict:
        """
        Return {onnx_name: TensorLayout} for every tensor in the graph.

        Three-phase computation:

          Phase 1 — Seed all tensors as flat (alloc == numel).

          Phase 2 — VectorOP broadcast nodes (outer_count > 1, not MatmulNode):
            advancing inputs and the output get advancing-strided layouts
            (n_chunks == outer_count, stride == aligned_chunk_size).
            Repeating inputs (bias at offset 0 each iteration) get a single
            padded block (n_chunks == 1, alloc == aligned_chunk_size).
            MatmulNode is excluded: its outer loop uses physical-address offsets
            rather than alignment-padded chunks.

          Phase 3 — Non-broadcast non-MatmulNode propagation (topological order):
            When an input has a larger alloc than the output, the output
            inherits the alloc and (if the dominant input is strided) also
            inherits n_chunks and stride.  Co-inputs whose alloc is smaller
            than the output are raised to match, acquiring the same layout
            structure so that a single run_op() call with size=alloc is valid
            for every buffer in the operation.
            MatmulNode is excluded: it reads A/Y with per-row strides derived
            from its inputs' layouts at emit time, not via alloc propagation.
        """
        all_tensors = (
            self._graph.weight_tensors
            + self._graph.intermediate_tensors
            + self._graph.input_tensors
            + self._graph.output_tensors
        )

        # Phase 1: seed all tensors as flat.
        layouts: dict = {t.onnx_name: TensorLayout.flat(t.numel)
                         for t in all_tensors}

        # Phase 2: broadcast VectorOP nodes.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode, SpaceToDepthNode,
                               HostNode)):
                continue

            n      = sn.outer_count          # number of loop iterations
            stride = sn.aligned_chunk_size   # buffer elements per iteration (>= chunk_size)
            alloc  = n * stride

            def _should_update_advancing(cur: TensorLayout, alloc=alloc) -> bool:
                # Update if switching from flat to advancing (n_chunks==1 → >1),
                # or if a later broadcast node needs a larger allocation.
                # When stride == chunk_size (no gap), alloc == numel so the
                # pure-alloc comparison would wrongly skip the flat→advancing
                # transition.  `alloc` is captured as a default arg so the
                # closure binds the loop iteration's value (B023-safe).
                return cur.n_chunks == 1 or alloc > cur.alloc

            # Output: advancing-strided.
            out_name = sn.output.onnx_name
            if _should_update_advancing(layouts[out_name]):
                layouts[out_name] = TensorLayout.advancing(
                    sn.output.numel, n, stride
                )

            # Input a: advancing or repeating.
            a_name = sn.inputs[0].onnx_name
            if sn.a_advances:
                if _should_update_advancing(layouts[a_name]):
                    layouts[a_name] = TensorLayout.advancing(
                        sn.inputs[0].numel, n, stride
                    )
            else:
                new_alloc = max(layouts[a_name].alloc, stride)
                if new_alloc > layouts[a_name].alloc:
                    layouts[a_name] = TensorLayout.repeating(
                        sn.inputs[0].numel, new_alloc
                    )

            # Input b: advancing or repeating (binary ops only).
            if sn.arity == 2:
                b_name = sn.inputs[1].onnx_name
                if sn.b_advances:
                    if _should_update_advancing(layouts[b_name]):
                        layouts[b_name] = TensorLayout.advancing(
                            sn.inputs[1].numel, n, stride
                        )
                else:
                    new_alloc = max(layouts[b_name].alloc, stride)
                    if new_alloc > layouts[b_name].alloc:
                        layouts[b_name] = TensorLayout.repeating(
                            sn.inputs[1].numel, new_alloc
                        )

        # Phase 3: propagate layout through non-broadcast non-MatmulNode chains.
        # Nodes are in topological order; one forward pass is sufficient.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode, SpaceToDepthNode,
                               HostNode)):
                continue

            input_layouts = [layouts[inp.onnx_name] for inp in sn.inputs]
            max_input_alloc = max(lay.alloc for lay in input_layouts)

            out_name = sn.output.onnx_name
            out_lay  = layouts[out_name]

            # Find the dominant strided input (highest alloc with n_chunks > 1).
            dominant = None
            for lay in input_layouts:
                if lay.n_chunks > 1:
                    if dominant is None or lay.alloc > dominant.alloc:
                        dominant = lay

            # Update output layout if any input has a larger alloc.
            if max_input_alloc > out_lay.alloc:
                if dominant is not None:
                    layouts[out_name] = TensorLayout.advancing(
                        out_lay.numel, dominant.n_chunks, dominant.stride
                    )
                else:
                    layouts[out_name] = TensorLayout(
                        numel=out_lay.numel, alloc=max_input_alloc,
                        n_chunks=1, chunk=out_lay.numel, stride=max_input_alloc
                    )

            # Raise every co-input whose alloc is smaller than the output's.
            # The dominant layout for inheritance is the (possibly updated) output.
            out_lay  = layouts[out_name]
            out_dom  = out_lay if out_lay.n_chunks > 1 else None

            for inp in sn.inputs:
                cur = layouts[inp.onnx_name]
                if cur.alloc >= out_lay.alloc:
                    continue
                if out_dom is not None:
                    layouts[inp.onnx_name] = TensorLayout.advancing(
                        cur.numel, out_dom.n_chunks, out_dom.stride
                    )
                else:
                    layouts[inp.onnx_name] = TensorLayout(
                        numel=cur.numel, alloc=out_lay.alloc,
                        n_chunks=1, chunk=cur.numel, stride=out_lay.alloc
                    )

        # Post-computation validation: ConvKernel / PoolingKernel write flat NCHW
        # output.  If the output ended up with n_chunks > 1 a broadcast VectorOP
        # downstream tried to assign an advancing layout — the kernel only populates
        # the first numel elements, so the gaps would be uninitialised.
        for sn in self._graph.nodes:
            if isinstance(sn, ConvNode):
                lay = layouts.get(sn.output.onnx_name)
                if lay is not None and lay.n_chunks > 1:
                    raise SchedulerError(
                        f"Conv node [{sn.index}] output '{sn.output.onnx_name}' "
                        f"(shape={sn.output.shape}) feeds a broadcast VectorOP node, "
                        f"which requires an advancing-strided buffer layout "
                        f"(n_chunks={lay.n_chunks}). "
                        f"ConvKernel writes a flat NCHW output, so adding a "
                        f"per-channel bias via a separate broadcast Add after Conv "
                        f"is not supported. Use the Conv operator's built-in bias "
                        f"input (3rd Conv input) instead."
                    )
            elif isinstance(sn, PoolNode):
                lay = layouts.get(sn.output.onnx_name)
                if lay is not None and lay.n_chunks > 1:
                    raise SchedulerError(
                        f"Pool node [{sn.index}] output '{sn.output.onnx_name}' "
                        f"(shape={sn.output.shape}) feeds a broadcast VectorOP node, "
                        f"which requires an advancing-strided buffer layout "
                        f"(n_chunks={lay.n_chunks}). "
                        f"PoolingKernel writes a flat NCHW output; insert a "
                        f"non-broadcast node between the pool and the broadcast op."
                    )

        return layouts

    def _strided_weight_params(self) -> dict:
        """
        Return {onnx_name: (n_chunks, stride)} for weight tensors whose DMA
        buffer has alignment gaps (n_chunks > 1 and stride > chunk).

        These weights are used as co-inputs to non-broadcast nodes alongside
        advancing-strided intermediates.  Their ROM arrays are emitted in
        strided layout — data elements interleaved with zero padding in the
        gap slots — so that a single run_op() call with size=alloc is valid
        for every buffer in the operation.
        """
        result: dict = {}
        for t in self._graph.weight_tensors:
            lay = self._layouts.get(t.onnx_name)
            if lay is None or lay.n_chunks <= 1 or not lay.is_strided:
                continue
            result[t.onnx_name] = (lay.n_chunks, lay.stride)
        return result

    # Map a kernel registry name → kernel_id_t enum literal used in the
    # generated C.  Mirrors _SourceMixin._KERNEL_ID_ENUM but lives in core
    # so the schedule walker can use it without depending on the source
    # mixin.  Update both sites if the enum naming ever changes.
    _KERNEL_ID_ENUM = {
        "VectorOPKernel": "KERNEL_VECTOROP",
        "MatmulKernel":   "KERNEL_MATMUL",
        "ConvKernel":     "KERNEL_CONV",
        "PoolKernel":     "KERNEL_POOL",
    }

    def _kernel_id_of(self, sched) -> Optional[str]:
        """Return the kernel_id_t enum literal for sched's lane, or None
        if sched does not occupy a hardware lane (ReshapeNode,
        SpaceToDepthNode, HostNode)."""
        kn = getattr(type(sched), "kernel_name", "")
        return self._KERNEL_ID_ENUM.get(kn) if kn else None

    @staticmethod
    def _is_synchronous_node(sched) -> bool:
        """A node whose helper drains its lane internally — currently only
        the MatmulNode 4D×3D outer loop, which calls run_matmul_at() in a
        for-loop; each iteration polls IsDone() before returning."""
        return isinstance(sched, MatmulNode) and sched.outer_count > 1

    def _compute_event_stream(self) -> list:
        """Build the linear event sequence inference_run() will emit.

        Each event is one of:

          ('comment', node_idx)             — node header (always present)
          ('wait',    kid, drained_idx)     — kernel_wait(KERNEL_*) call
          ('start',   node_idx)             — non-blocking Start
          ('start_sync', node_idx)          — synchronous Start (drains
                                               its own lane internally)
          ('drain',   kid, drained_idx)     — final kernel_wait before
                                               output cache sync
          ('reshape', node_idx)             — ReshapeNode / Slice view: no
                                               work (buffer alias)
          ('cpu',     node_idx)             — SpaceToDepthNode / HostNode:
                                               host code, synchronous,
                                               occupies no lane

        ReshapeNodes are emitted as no-ops, but predecessor-wait analysis
        walks *through* them: a consumer of a Reshape alias must wait on
        the real producing kernel of the underlying source.

        A 'cpu' node waits on its producers like a kernel start would, then
        runs to completion inline, so its consumers never wait on it (it
        has no lane) and the tensors it produced are complete at the event.
        """
        graph = self._graph
        dag   = Dag.from_graph(graph)

        def effective_preds(idx: int) -> set:
            """Predecessors with their kernel_name set, traversing through
            any chain of ReshapeNodes to the real producing op."""
            seen: set = set()
            out: set  = set()
            stack = list(dag.predecessors(idx))
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                p_sched = dag.by_index[p].sched
                if self._is_alias_node(p_sched):
                    stack.extend(dag.predecessors(p))
                else:
                    out.add(p)
            return out

        events: list   = []
        pending: dict  = {}   # kid -> node index of last started, not yet waited

        for sn in graph.nodes:
            events.append(('comment', sn.index))

            if self._is_alias_node(sn):
                events.append(('reshape', sn.index))
                continue

            target = self._kernel_id_of(sn)
            is_cpu = isinstance(sn, (SpaceToDepthNode, HostNode))

            # 1. Wait on each effective predecessor whose lane is still in flight.
            waits: list = []
            seen_kid: set = set()
            for p in sorted(effective_preds(sn.index)):
                p_kid = self._kernel_id_of(dag.by_index[p].sched)
                if p_kid is None:
                    continue
                if pending.get(p_kid) == p and p_kid not in seen_kid:
                    waits.append((p_kid, p))
                    seen_kid.add(p_kid)

            # 2. Wait on target lane if a different op is still in flight there.
            if target is not None and target in pending and target not in seen_kid:
                waits.append((target, pending[target]))
                seen_kid.add(target)

            for k, drained in waits:
                events.append(('wait', k, drained))
                pending.pop(k, None)

            if is_cpu:
                events.append(('cpu', sn.index))
            elif self._is_synchronous_node(sn):
                events.append(('start_sync', sn.index))
                if target is not None:
                    pending.pop(target, None)
            else:
                events.append(('start', sn.index))
                if target is not None:
                    pending[target] = sn.index

        # Final drain — wait on every still-pending lane.
        for k in sorted(pending):
            events.append(('drain', k, pending[k]))

        return events

    def _compute_live_intervals(self) -> dict:
        """Return {onnx_name: (start_event_idx, end_event_idx)} for every
        non-reshape-alias intermediate tensor.

        Intervals are expressed in event-stream index space (not node-index
        space) so that pool-slot colouring respects the actual execution
        timeline emitted by inference_run() under the parallel-wait scheme:
        a tensor is live from its producer's Start event until the latest
        kernel_wait that drains a consumer's lane.

        Weights and graph inputs/outputs are excluded — they live for the
        entire inference call and are never candidates for buffer reuse.
        """
        alias_src = self._alias_source_map()      # Reshape aliases + Slice views
        intermediates = {
            t.onnx_name
            for t in self._graph.intermediate_tensors
            if t.onnx_name not in alias_src
        }

        # Build alias-resolution map: tensor_name → underlying intermediate.
        # A consumer of a Reshape alias (or of a Slice view) contributes to
        # the source tensor's live interval, since both share the same DMA
        # buffer.
        def resolve(name: str) -> str:
            seen = set()
            while name in alias_src and name not in seen:
                seen.add(name)
                name = alias_src[name]
            return name

        events = self._compute_event_stream()

        # Map node_idx → event index of its Start (or start_sync / cpu).
        start_event: dict = {}
        # Map node_idx → event index of the wait that drained it.  For a
        # synchronous node the helper drains the lane itself (a 'cpu' node
        # runs inline); we record the same event index as the Start so the
        # interval doesn't extend past the call.
        drain_event: dict = {}
        for ei, ev in enumerate(events):
            if ev[0] in ('start', 'start_sync', 'cpu'):
                start_event[ev[1]] = ei
                if ev[0] in ('start_sync', 'cpu'):
                    drain_event[ev[1]] = ei
            elif ev[0] in ('wait', 'drain'):
                drain_event[ev[2]] = ei

        producer_of: dict = {}
        consumers_of: dict = {n: [] for n in intermediates}
        for sn in self._graph.nodes:
            if self._is_alias_node(sn):
                continue
            out = sn.output.onnx_name
            if out in intermediates:
                producer_of[out] = sn.index
            for inp in sn.inputs:
                src = resolve(inp.onnx_name)
                if src in intermediates and src != out:
                    consumers_of[src].append(sn.index)

        intervals: dict = {}
        for name in intermediates:
            p = producer_of.get(name)
            if p is None:
                continue
            start_ei = start_event.get(p)
            if start_ei is None:
                continue
            ends = []
            for c in consumers_of.get(name, []):
                # The buffer is held until the consumer's lane drains.
                de = drain_event.get(c)
                if de is None:
                    de = start_event.get(c, start_ei)
                ends.append(de)
            # No consumer recorded → buffer is at least live across its
            # own producer's drain (covers final outputs that flow only
            # via a Reshape chain to a graph output).
            end_ei = max(ends) if ends else drain_event.get(p, start_ei)
            intervals[name] = (start_ei, end_ei)
        return intervals

    def _compute_pool_layout(self):
        """
        Compute per-buffer offsets for the single contiguous pool allocation.

        Returns (layout, total_elems) where:
          layout     = list of (onnx_name, offset_in_elems, alloc_in_elems)
                       for all weights + non-reshape-alias intermediates, in
                       declaration order (weights first, then intermediates)
          total_elems = total pool size in elements (sum of aligned alloc sizes)

        Each slot is padded to a 64-byte boundary to ensure DMA cache-line
        alignment between sub-buffers.  Intermediate tensors with non-overlapping
        live intervals are packed into the same slot to minimise pool size.
        """
        bpe       = self._dtype.bytes_per_elem
        align_to  = 64 // bpe   # elements per 64-byte boundary
        def align_up(n):
            return (n + align_to - 1) & ~(align_to - 1)

        reshape_aliases = self._reshape_aliases
        layout  = []
        offset  = 0

        # Weights: live for the entire call — always sequential, never shared.
        for t in self._graph.weight_tensors:
            alloc = self._alloc_sizes[t.onnx_name]
            layout.append((t.onnx_name, offset, alloc))
            offset += align_up(alloc)

        # Intermediates: greedy interval-graph colouring.
        # Two tensors may share a slot only when their live intervals are disjoint.
        intervals   = self._compute_live_intervals()
        alloc_sizes = self._alloc_sizes

        # Preserve graph order for tensors without an interval entry (fallback).
        # Slice views own no memory either (they point into their root).
        views = self._view_aliases
        cand_names = [
            t.onnx_name for t in self._graph.intermediate_tensors
            if t.onnx_name not in reshape_aliases and t.onnx_name not in views
        ]

        # Sort candidates by start time; break ties largest-first so that the
        # biggest buffer claims the slot and smaller ones only reuse it.
        with_intervals = sorted(
            [(n, intervals[n]) for n in cand_names if n in intervals],
            key=lambda kv: (kv[1][0], -alloc_sizes.get(kv[0], 0)),
        )
        no_intervals = [n for n in cand_names if n not in intervals]

        # slots: [(end_time, slot_alloc_elems, [onnx_name, ...])]
        slots: list = []
        for name, (start, end) in with_intervals:
            a = alloc_sizes.get(name, 0)
            placed = False
            for slot in slots:
                if slot[0] < start:      # slot is free before this tensor starts
                    slot[0] = end
                    slot[1] = max(slot[1], align_up(a))
                    slot[2].append(name)
                    placed = True
                    break
            if not placed:
                slots.append([end, align_up(a), [name]])

        for _slot_end, slot_alloc, names in slots:
            for name in names:
                layout.append((name, offset, alloc_sizes[name]))
            offset += slot_alloc

        # Tensors with no interval data (should not occur in a well-formed
        # graph but handled defensively) get their own sequential slot.
        for name in no_intervals:
            alloc = alloc_sizes[name]
            layout.append((name, offset, alloc))
            offset += align_up(alloc)

        return layout, offset

    def _compute_pool_bytes(self) -> int:
        """
        Total DMA memory needed for all model buffers, with 64-byte alignment
        per buffer, rounded up to a 4 KiB page boundary.

        Covers: weight tensors + intermediate tensors + model inputs + outputs.
        Uses _layouts (which accounts for broadcast alignment padding) and
        dtype.bytes_per_elem so the result is correct regardless of the data type.
        Used to size the u-dma-buf pool and advertised as
        INFERENCE_BUF_POOL_SIZE_BYTES in the generated header.
        """
        def _align64(n: int) -> int:
            return (n + 63) & ~63

        bpe   = self._dtype.bytes_per_elem
        total = sum(
            _align64(lay.alloc * bpe)
            for lay in self._layouts.values()
        )
        page = 4096
        return (total + page - 1) & ~(page - 1)

    @property
    def _active_kernels(self) -> List[KernelDesc]:
        """Ordered list of KernelDesc for each kernel type present in the graph.

        Order follows KERNEL_REGISTRY insertion order (VectorOPKernel before
        MatmulKernel before any future kernels), not the graph's node order.
        This makes the generated inference_init() parameter order deterministic
        regardless of which node appears first in the model.
        """
        present = {sn.kernel_name for sn in self._graph.nodes}
        return [kd for kd in KERNEL_REGISTRY.values() if kd.name in present]

    # ------------------------------------------------------------------
    # Backward-compat helpers (kept for existing code and tests)
    # ------------------------------------------------------------------

    @property
    def _has_matmul_nodes(self) -> bool:
        """True when the graph contains at least one MatmulNode."""
        return any(isinstance(sn, MatmulNode) for sn in self._graph.nodes)

    @property
    def _has_conv_nodes(self) -> bool:
        """True when the graph contains at least one ConvNode."""
        return any(isinstance(sn, ConvNode) for sn in self._graph.nodes)

    @property
    def _has_pool_nodes(self) -> bool:
        """True when the graph contains at least one PoolNode."""
        return any(isinstance(sn, PoolNode) for sn in self._graph.nodes)

    @property
    def _reshape_aliases(self) -> dict:
        """Return {output_onnx_name: source_c_name} for all ReshapeNode outputs.

        The reshape output shares the same DMA buffer as its source — no
        allocation or free is needed.  Used by the pool allocator to skip
        alias tensors and by inference_init() / inference_deinit() to manage
        the pointer.  When the source is a graph input the actual pointer
        assignment is deferred to inference_run() (see _run_reshape_aliases).
        """
        result: dict = {}
        for sn in self._graph.nodes:
            if isinstance(sn, ReshapeNode):
                result[sn.output.onnx_name] = sn.inputs[0].c_name
        return result

    @property
    def _run_reshape_aliases(self) -> dict:
        """Subset of _reshape_aliases whose source is a graph input tensor.

        Graph inputs only exist as inference_run() parameters, so their aliases
        cannot be resolved in inference_init().  Instead, the pointer assignment
        (e.g. ``Z = X;``) is emitted at the top of inference_run() and cleared
        (``Z = NULL;``) at the bottom.
        """
        input_onnx = {t.onnx_name for t in self._graph.input_tensors}
        result: dict = {}
        for sn in self._graph.nodes:
            if isinstance(sn, ReshapeNode) and sn.inputs[0].onnx_name in input_onnx:
                result[sn.output.onnx_name] = sn.inputs[0].c_name
        return result

    @property
    def _input_output_reshape_copies(self) -> list:
        """Return [(out_t, in_c_name)] for graph outputs that are reshape
        aliases of graph inputs (possibly through a chain of reshape-only
        intermediate tensors).

        When no hardware kernel sits between a graph input and a graph output,
        the pointer-alias trick cannot be used (the caller supplied both
        buffers independently).  inference_run() must copy the data explicitly:
        ``memcpy(out, in, N * BYTES_PER_ELEM)`` followed by
        ``inference_buf_sync_to_device(out)`` so the subsequent
        ``inference_buf_sync_from_device(out)`` sees fresh DDR contents.
        """
        reshape_onnx: dict = {}
        for sn in self._graph.nodes:
            if isinstance(sn, ReshapeNode):
                reshape_onnx[sn.output.onnx_name] = sn.inputs[0].onnx_name

        input_map = {t.onnx_name: t.c_name for t in self._graph.input_tensors}
        intermediate_onnx = {t.onnx_name for t in self._graph.intermediate_tensors}

        result = []
        for out_t in self._graph.output_tensors:
            cur = out_t.onnx_name
            while cur in reshape_onnx:
                src = reshape_onnx[cur]
                if src in input_map:
                    result.append((out_t, input_map[src]))
                    break
                if src not in intermediate_onnx:
                    break
                cur = src
        return result

    @property
    def _output_aliases(self) -> dict:
        """Return {out_c_name: [chain_c_name, ...]} for graph outputs that are
        the terminal node of a reshape alias chain.

        A graph output is a "terminal alias" when it is produced by a ReshapeNode
        (Reshape / Squeeze / Unsqueeze / Dropout) whose source is an intermediate
        buffer (a kernel output).  In that case the kernel never writes directly
        to the caller-supplied output buffer — the codegen must redirect the
        internal buffer pointer so the kernel writes to the correct destination.

        The returned list contains the c_names of every intermediate buffer in
        the alias chain, ordered from the immediate source of the graph-output
        ReshapeNode to the root buffer (the one the hardware kernel actually
        writes to).  All entries are guaranteed to be intermediate tensors.
        """
        # Build onnx_name → onnx_name alias map from all ReshapeNodes
        reshape_onnx: dict = {}
        for sn in self._graph.nodes:
            if isinstance(sn, ReshapeNode):
                reshape_onnx[sn.output.onnx_name] = sn.inputs[0].onnx_name

        # Only redirect intermediate tensors (never weights or graph inputs)
        intermediate_c = {
            t.onnx_name: t.c_name
            for t in self._graph.intermediate_tensors
        }

        result: dict = {}
        for out_t in self._graph.output_tensors:
            if out_t.onnx_name not in reshape_onnx:
                continue
            chain: list = []
            cur = reshape_onnx[out_t.onnx_name]
            while cur is not None and cur in intermediate_c:
                chain.append(intermediate_c[cur])
                cur = reshape_onnx.get(cur)
            if chain:
                result[out_t.c_name] = chain

        return result

    @property
    def _has_vectorop_nodes(self) -> bool:
        """True when the graph contains at least one VectorOP ScheduledNode."""
        return any(isinstance(sn, ScheduledNode) for sn in self._graph.nodes)

    def _layer_display_names(self) -> List[str]:
        """One human-readable name per scheduled node, suitable for the
        per-layer profiler.

        Prefers the ONNX node name when present; falls back to ``op_type_index``
        when empty.  After that, any duplicates are disambiguated by suffixing
        every collider with ``_<index>`` so each name is unique and stable.
        """
        nodes = self._graph.nodes
        candidates: List[str] = []
        for sn in nodes:
            raw = (sn.onnx_node.name or "").strip()
            if not raw:
                raw = f"{sn.onnx_node.op_type}_{sn.index}"
            candidates.append(raw)

        counts: dict = {}
        for n in candidates:
            counts[n] = counts.get(n, 0) + 1

        return [
            (f"{n}_{sn.index}" if counts[n] > 1 else n)
            for sn, n in zip(nodes, candidates, strict=True)
        ]

    @property
    def _driver_prefix(self) -> str:
        """Driver prefix for the first active kernel — kept for backward compat."""
        ak = self._active_kernels
        return ak[0].driver_prefix if ak else "xvectoropkernel"

    @property
    def large_weight_tensors(self) -> List[TensorInfo]:
        """Weight tensors stored as external .dat files.
        Empty when embed_large_weights=True (all weights inlined)."""
        if self._embed_large_weights:
            return []
        return [t for t in self._graph.weight_tensors if t.is_large_weight]

    def generate_weight_dat(self, tensor: TensorInfo) -> bytes:
        """
        Return the raw binary content for weights/<c_name>.dat.
        Encoded using the active dtype (little-endian), suitable for
        fread() at runtime.
        """
        return tensor.to_dat_bytes(self._dtype)

    def _broadcast_io_map(self) -> dict:
        """
        Returns {onnx_name → (outer_count, chunk_macro, stride_macro)} for
        every tensor whose DMA buffer has alignment-gap padding and therefore
        needs strided fill/print logic.

        The integer values (outer_count) come from self._layouts.  The C macro
        name strings are derived from the broadcast node whose output sets the
        stride (pass 1), propagated through non-broadcast non-MatmulNode chains
        (pass 2 — Fix A: MatmulNode excluded to prevent its output inheriting
        a stride descriptor it doesn't use).
        """
        canonical: dict = {}   # onnx_name → c_up prefix string for C macros

        # Pass 1 — advancing inputs + output of each broadcast VectorOP node.
        for sn in self._graph.nodes:
            if sn.outer_count <= 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode, SpaceToDepthNode,
                               HostNode)):
                continue
            c_up = sn.output.c_name.upper()
            canonical[sn.output.onnx_name] = c_up
            if sn.a_advances:
                canonical[sn.inputs[0].onnx_name] = c_up
            if sn.arity == 2 and sn.b_advances:
                canonical[sn.inputs[1].onnx_name] = c_up

        # Pass 2 — propagate through non-broadcast non-special-node chains.
        for sn in self._graph.nodes:
            if sn.outer_count > 1:
                continue
            if isinstance(sn, (MatmulNode, ConvNode, PoolNode, ReshapeNode, SpaceToDepthNode,
                               HostNode)):
                continue
            if sn.output.onnx_name in canonical:
                continue
            for inp in sn.inputs:
                if inp.onnx_name in canonical:
                    canonical[sn.output.onnx_name] = canonical[inp.onnx_name]
                    break

        # Build result: combine canonical string prefixes with integer values
        # from _layouts.  Only include tensors that actually have n_chunks > 1
        # (advancing-strided buffers) — repeating/flat tensors are excluded.
        result: dict = {}
        for name, c_up in canonical.items():
            lay = self._layouts.get(name)
            if lay and lay.n_chunks > 1:
                result[name] = (
                    lay.n_chunks,
                    f"INFERENCE_{c_up}_CHUNK",
                    f"INFERENCE_{c_up}_CHUNK_STRIDE",
                )
        return result
