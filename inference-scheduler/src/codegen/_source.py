"""Mixin that generates src/inference.c."""

from __future__ import annotations
from typing import List

from ..nodes    import (ACT_NAMES, OP_NAMES, MatmulNode, ScheduledNode, SchedulerError,
                        SpaceToDepthNode)
from ..host_nodes import (HOST_C_COMMON, HOST_C_HELPER_ORDER, HOST_C_POOL, HostNode,
                          SliceNode, host_c_helper)
from ._banners  import _banner, _file_banner


class _SourceMixin:
    """Generates src/inference.c."""

    def generate_source(self) -> str:
        """Content for src/inference.c."""
        parts = [
            _file_banner("inference.c", self._graph, self._model_path),
            self._source_includes(),
        ]
        op_defines = self._source_op_defines()
        if op_defines:
            parts.append(op_defines)
        parts += [
            self._weight_arrays(),
            self._buffer_declarations(),
            self._kernel_instance(),
            self._kernel_wait_helper(),
            self._layer_names_table(),
            self._run_op_helper(),
        ]
        host_section = self._host_ops_section()
        if host_section:
            parts.append(host_section)
        parts += [
            self._init_function(),
            self._inference_function(),
        ]
        return "\n".join(parts) + "\n"

    def _source_includes(self) -> str:
        stdio = '#include <stdio.h>    /* fopen, fread, snprintf, fprintf */\n' \
                if (self.large_weight_tensors or self._host_nodes) else ''
        stdlib = '#include <stdlib.h>   /* malloc, free (host staging buffers) */\n' \
                if (self._host_op_nodes or self._host_nodes) else ''
        mathh = '#include <math.h>     /* host ops: exp, sqrt, tanh, erf, nearbyint */\n' \
                if self._host_nodes else ''
        kernel_headers = "".join(
            f'#include "{kd.driver_prefix}.h"\n'
            for kd in self._active_kernels
        )
        return (
            _banner("Includes") +
            '#include "inference.h"\n'
            '#include "inference_prof.h"  /* INFERENCE_PROF_BEGIN/END (no-ops unless'
            ' INFERENCE_PROFILING is set) */\n'
            f'{kernel_headers}'
            '#include <string.h>    /* memcpy */\n'
            f'{stdio}'
            f'{stdlib}'
            f'{mathh}'
            '\n'
            '/*\n'
            ' * Cache coherency between the CPU and the AXI DMA master is handled by\n'
            ' * inference_buf_sync_to_device() and inference_buf_sync_from_device(),\n'
            ' * both implemented in inference_buf.c:\n'
            ' *\n'
            ' *   Linux:      xclSyncBO (XRT) — XCL_BO_SYNC_BO_TO/FROM_DEVICE\n'
            ' *   Bare-metal: Xil_DCacheFlushRange / Xil_DCacheInvalidateRange\n'
            ' */\n'
            '\n'
            '/* Forward declarations — defined in inference_buf.c */\n'
            'void inference_buf_sync_to_device(inference_buf_t *buf);\n'
            'void inference_buf_sync_from_device(inference_buf_t *buf);\n'
        )

    def _source_op_defines(self) -> str:
        """Emit VectorOPKernel op-code #defines.  Empty for MatMul-only models."""
        if not self._has_vectorop_nodes:
            return ""
        lines = [_banner("VectorOPKernel operation codes (must match VectorOP.h)")]
        for code, name in sorted(OP_NAMES.items()):
            lines.append(f"#define {name:<20} {code}u")
        lines.append("")
        lines.append("/* Fused activation applied after the op (VectorOP.h Act enum) */")
        for code, name in sorted(ACT_NAMES.items()):
            lines.append(f"#define {name:<20} {code}u")
        return "\n".join(lines)

    def _weight_arrays(self) -> str:
        weights = self._graph.weight_tensors
        if not weights:
            return ""
        external = {t.onnx_name for t in self.large_weight_tensors}
        strided  = self._strided_weight_params()
        parts = [_banner("Constant weight arrays (ONNX initializers)")]
        for t in weights:
            if t.onnx_name in external:
                parts.append(t.emit_large_weight_ptr_decl())
            elif t.onnx_name in strided:
                outer_count, aligned_chunk = strided[t.onnx_name]
                parts.append(t.emit_weight_decl_strided(outer_count, aligned_chunk, self._dtype))
            else:
                parts.append(t.emit_weight_decl(self._dtype))
            parts.append("")
        return "\n".join(parts)

    @property
    def _host_op_nodes(self) -> List[SpaceToDepthNode]:
        """Host-side ops that need a cached staging block (SpaceToDepthNode)."""
        return [sn for sn in self._graph.nodes if isinstance(sn, SpaceToDepthNode)]

    @property
    def _host_nodes(self) -> List[HostNode]:
        """Host-CPU nodes that run code (all HostNodes except Slice views);
        they share the cached staging arena s_host_stage."""
        return [sn for sn in self._graph.nodes
                if isinstance(sn, HostNode) and not self._is_view(sn)]

    # ------------------------------------------------------------------ #
    # Host-CPU ops (Softmax, LayerNorm, GELU, Transpose, Gather, ...)       #
    # ------------------------------------------------------------------ #

    def _host_io_layout(self, t) -> tuple:
        """(n_chunks, chunk, stride) of the buffer holding ``t``: the layout
        of the Reshape-alias root (a reshape never changes the bytes), flat
        unless a broadcast VectorOP consumer / producer made it strided."""
        lay = self._layouts.get(self._reshape_root(t.onnx_name))
        if lay is None or lay.n_chunks <= 1 or lay.stride == lay.chunk:
            return (1, t.numel, t.numel)
        if lay.n_chunks * lay.chunk != t.numel:
            raise SchedulerError(
                f"host op: tensor '{t.onnx_name}' layout {lay} does not cover its "
                f"{t.numel} elements.")
        return (lay.n_chunks, lay.chunk, lay.stride)

    @staticmethod
    def _staged_count(io: tuple) -> int:
        nc, ch, st = io
        return nc * st if nc > 1 else ch

    def _host_stage_plan(self, sn) -> tuple:
        """Offsets (Data_t elements, 16-byte aligned) of a host node's staged
        inputs, output and scratch inside s_host_stage.
        Returns (inputs=[(tensor, off, count, io)], (off, count, io),
        scratch_off | None, total)."""
        bpe = self._dtype.bytes_per_elem
        a8 = lambda n: (n + 7) & ~7  # noqa: E731
        off, ins = 0, []
        for t in sn.staged_inputs():
            io = self._host_io_layout(t)
            cnt = self._staged_count(io)
            ins.append((t, off, cnt, io))
            off += a8(cnt)
        io = self._host_io_layout(sn.output)
        cnt = self._staged_count(io)
        out = (off, cnt, io)
        off += a8(cnt)
        scratch = None
        if sn.scratch_bytes():
            scratch = off
            off += a8(-(-sn.scratch_bytes() // bpe))
        return ins, out, scratch, off

    @property
    def _host_stage_elems(self) -> int:
        return max((self._host_stage_plan(sn)[3] for sn in self._host_nodes), default=0)

    def _written_by_kernel(self, t) -> bool:
        """True when the buffer behind ``t`` (through aliases / views) is
        written by a PL kernel, i.e. the CPU must invalidate before reading."""
        root = self._alias_root(t.onnx_name)
        for sn in self._graph.nodes:
            if sn.output.onnx_name == root:
                return bool(getattr(type(sn), "kernel_name", ""))
        return False

    def _host_luts(self) -> List[tuple]:
        """Distinct lookup tables of the host nodes: [(c_name, init_call)]."""
        luts: dict = {}
        for sn in self._host_nodes:
            for name, init in sn.c_luts(self._dtype):
                luts.setdefault(name, init)
        return list(luts.items())

    def _host_ops_section(self) -> str:
        host = self._host_nodes
        if not host:
            return ""
        lut = bool(self._dtype.host_lut_bits)
        used = set()
        for sn in host:
            used.update(sn.c_helpers())
        if lut and used & {"gelu_tanh", "gelu_erf"}:
            used.add("lut_map")
        parts = [
            _banner("Host-CPU ops (no hardware kernel)"),
            "/*\n"
            " * Ops the PL kernels cannot run execute here, inline in inference_run():\n"
            " * each computes in double precision straight from / into the DMA buffers\n"
            " * (cacheable mapping; else via the cached staging arena s_host_stage),\n"
            " * rounds half to even + saturates on write-back, and flushes its output\n"
            " * for the consuming kernel.  GELU and Softmax's exp() are lookup tables\n"
            " * filled at init by the same code path.  The scheduler's simulator (_simulate.py /\n"
            " * host_nodes.py) implements the same operations in the same order, so\n"
            " * test_inference.c expects bit-identical results.\n"
            " *\n"
            " * No FMA contraction (GCC's GNU modes default to -ffp-contract=fast and\n"
            " * would fuse a*b + c on the A53): */\n"
            "#if defined(__clang__)\n"
            "#  pragma clang fp contract(off)\n"
            "#elif defined(__GNUC__)\n"
            "#  pragma GCC optimize (\"fp-contract=off\")\n"
            "#endif\n",
            self._dtype.c_host_conversions(),
        ]
        if lut:
            parts.append(self._dtype.c_host_lut_defs())
        parts += [HOST_C_COMMON, HOST_C_POOL]
        for kind in HOST_C_HELPER_ORDER:
            if kind in used:
                parts.append(host_c_helper(kind, lut))
        consts = []
        for sn in host:
            consts.extend(sn.c_file_consts(self._dtype))
        if consts:
            parts.append("/* Per-node host-op constants */\n" + "\n".join(consts) + "\n")
        parts.append(self._host_runtime_functions())
        return "\n".join(parts)

    def _host_runtime_functions(self) -> str:
        """host_runtime_init() / host_runtime_deinit(): the lookup tables
        (called by inference_init / inference_deinit)."""
        luts = self._host_luts()
        decl = [f"static Data_t *{name} = NULL;" for name, _ in luts
                if name != "s_host_exp_lut"]          # declared with its helper
        init = [f"    if ({call} != 0) return -1;" for _, call in luts]
        free = [f"    free({name}); {name} = NULL;" for name, _ in luts]
        return (
            ("/* Lookup tables (filled in host_runtime_init) */\n" + "\n".join(decl) + "\n\n"
             if decl else "") +
            "/* Host-op runtime: the lookup tables.  Called by inference_init() /\n"
            " * inference_deinit(). */\n"
            "static int host_runtime_init(void)\n"
            "{\n" +
            "".join(ln + "\n" for ln in init) +
            "    return 0;\n"
            "}\n"
            "\n"
            "static void host_runtime_deinit(void)\n"
            "{\n" +
            "".join(ln + "\n" for ln in free) +
            "}\n"
        )

    def _emit_host_block(self, sn) -> str:
        """One host op inside inference_run(): invalidate kernel-written
        inputs, get flat input / output pointers (the BO itself when the
        mapping is cacheable and the layout flat, else the staging arena),
        compute, and hand the output back (store if staged + flush)."""
        ins, (o_off, o_cnt, o_io), scratch, _ = self._host_stage_plan(sn)
        lines = ["    {"]
        for i, (t, off, cnt, _io) in enumerate(ins):
            lines.append(f"        const Data_t *in{i};  /* '{t.onnx_name}': {cnt} elem"
                         f" (stage +{off}) */")
        lines.append(f"        Data_t       *out;  /* '{sn.output.onnx_name}': {o_cnt} elem"
                     f" (stage +{o_off}) */")
        if scratch is not None:
            lines.append(f"        void         *tmp = s_host_stage + {scratch}u;"
                         f"  /* {sn.scratch_bytes()} B scratch */")
        seen = set()
        for t in list(sn.staged_inputs()) + list(sn.direct_inputs()):
            if t.c_name not in seen and self._written_by_kernel(t):
                seen.add(t.c_name)
                lines.append(f"        inference_buf_sync_from_device({t.c_name});"
                             f"  /* written by a kernel */")
        for t in sn.direct_inputs():
            lay = self._layouts.get(self._reshape_root(t.onnx_name))
            if lay is not None and (lay.n_chunks > 1 or lay.alloc != lay.numel):
                raise SchedulerError(
                    f"host op [{sn.index}]: '{t.onnx_name}' must have a flat layout.")
        for i, (t, off, _cnt, (nc, ch, st)) in enumerate(ins):
            lines.append(f"        in{i} = host_in({t.c_name}, s_host_stage + {off}u,"
                         f" {nc}u, {ch}u, {st}u);")
        nc, ch, st = o_io
        lines.append(f"        out = host_out({sn.output.c_name}, s_host_stage + {o_off}u,"
                     f" {nc}u, {ch}u, {st}u);")
        for ln in sn.c_call([f"in{i}" for i in range(len(ins))], "out", "tmp",
                            [t.c_name for t in sn.direct_inputs()], self._dtype):
            lines.append("        " + ln)
        lines.append(f"        host_out_done({sn.output.c_name}, out, {nc}u, {ch}u, {st}u);")
        lines.append("    }")
        return "\n".join(lines)

    def _buffer_declarations(self) -> str:
        lines = [_banner("Mutable intermediate buffers")]

        for t in self._graph.input_tensors:
            lines.append(f"/* INPUT  '{t.onnx_name}'  shape={t.shape} — caller-supplied */")
        for t in self._graph.output_tensors:
            lines.append(f"/* OUTPUT '{t.onnx_name}'  shape={t.shape} — caller-supplied */")
        lines.append("")

        intermediates = self._graph.intermediate_tensors
        if intermediates:
            for t in intermediates:
                lines.append(t.emit_buffer_decl())
        else:
            lines.append("/* (no intermediate buffers) */")

        # Pool pointer and static view-struct backing storage
        weights       = self._graph.weight_tensors
        reshape_aliases = self._reshape_aliases
        pool_tensors  = (
            list(weights) +
            [t for t in intermediates if t.onnx_name not in reshape_aliases]
        )
        if pool_tensors:
            lines.append("")
            lines.append(
                "/* Pool allocation that owns all weight + intermediate DMA memory */"
            )
            lines.append("static inference_buf_t *s_alloc_pool = NULL;")
            lines.append("")
            lines.append(
                "/* Static view-struct backing storage"
                " (no heap allocation for sub-buffer metadata) */"
            )
            for t in pool_tensors:
                lines.append(f"static inference_buf_t _s_buf_{t.c_name};")

        host_ops = self._host_op_nodes
        if host_ops:
            lines.append("")
            lines.append(
                "/* Cached host staging blocks for the SpaceToDepth reorders (2 x source\n"
                " * numel each: stage_in | stage_out), used when a DMA buffer is mapped\n"
                " * non-cacheable (INFERENCE_BUF_CACHEABLE=0): the reorder then runs on a\n"
                " * plain malloc'd copy and only sequential memcpys touch the DMA memory.\n"
                " * malloc'd in inference_init(). */"
            )
            for sn in host_ops:
                lines.append(
                    f"static Data_t *{sn.stage_c_name} = NULL;"
                    f"  /* [{sn.index}] {sn.numel} elem x 2 */"
                )

        if self._host_nodes:
            n = self._host_stage_elems
            lines.append("")
            lines.append(
                "/* Cached staging arena shared by the host-CPU ops (they run one at a\n"
                " * time): largest op's inputs + output + scratch, "
                f"{n} elem ({n * self._dtype.bytes_per_elem} B).\n"
                " * Only touched for a non-cacheable DMA buffer or a strided layout.\n"
                " * malloc'd in inference_init(). */"
            )
            lines.append("static Data_t *s_host_stage = NULL;")

        return "\n".join(lines)

    def _kernel_instance(self) -> str:
        lines = [_banner("Kernel driver instances (one per hardware IP)")]
        for kd in self._active_kernels:
            lines.append(f"static {kd.c_type} {kd.c_var};")
        return "\n".join(lines) + "\n"

    # Stable map from a kernel's registry name to its kernel_id_t enum literal.
    # Both _kernel_wait_helper() and the inference_run() body emitter use this.
    _KERNEL_ID_ENUM = {
        "VectorOPKernel": "KERNEL_VECTOROP",
        "MatmulKernel":   "KERNEL_MATMUL",
        "ConvKernel":     "KERNEL_CONV",
        "PoolKernel":     "KERNEL_POOL",
    }

    def _kernel_wait_helper(self) -> str:
        """Emit kernel_id_t and a weak kernel_wait() polling default.

        Only enumerates kernels the model actually uses, so single-kernel
        projects don't pull in references to undeclared driver instances.
        The function is __weak so deployments that prefer interrupt-driven
        waiting (UIO/IRQ on Linux, GIC under bare-metal) can override it at
        link time without touching the generated source.
        """
        active = list(self._active_kernels)
        if not active:
            return ""

        enum_lines = ["typedef enum {"]
        for i, kd in enumerate(active):
            enum_lines.append(f"    {self._KERNEL_ID_ENUM[kd.name]} = {i}u,")
        enum_lines.append(f"    KERNEL_COUNT = {len(active)}u")
        enum_lines.append("} kernel_id_t;")

        case_lines = []
        for kd in active:
            case_lines.append(
                f"        case {self._KERNEL_ID_ENUM[kd.name]}:\n"
                f"            while (!{kd.c_type}_IsDone(&{kd.c_var})) {{}}\n"
                f"            break;"
            )

        return (
            _banner("Kernel synchronization (split start / wait)") +
            "/*\n"
            " * kernel_wait(k) — block until the most-recent Start on lane k completes.\n"
            " *\n"
            " * The codegen emits NON-BLOCKING run_*() helpers; inference_run() calls\n"
            " * kernel_wait() lazily — only when the next op needs the result of an\n"
            " * earlier op, or when a lane must be reused.  This lets different lanes\n"
            " * (Conv ‖ Pool ‖ VectorOP ‖ Matmul) overlap without threading.\n"
            " *\n"
            " * The default implementation polls IsDone() in a tight loop.  It is\n"
            " * declared __weak so deployments that prefer interrupt-driven waiting\n"
            " * can override it at link time:\n"
            " *\n"
            " *     void kernel_wait(kernel_id_t k) { ... wait-for-IRQ ... }\n"
            " */\n"
            + "\n".join(enum_lines) + "\n\n"
            "__attribute__((weak))\n"
            "void kernel_wait(kernel_id_t k)\n"
            "{\n"
            "    switch (k) {\n"
            + "\n".join(case_lines) + "\n"
            "        default: break;\n"
            "    }\n"
            "}\n"
        )

    @staticmethod
    def _c_string_escape(s: str) -> str:
        """Escape a Python string for use as a C string literal body."""
        out = []
        for ch in s:
            o = ord(ch)
            if ch == '\\':
                out.append('\\\\')
            elif ch == '"':
                out.append('\\"')
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\r':
                out.append('\\r')
            elif ch == '\t':
                out.append('\\t')
            elif o < 0x20 or o == 0x7f:
                out.append(f"\\x{o:02x}")
            else:
                out.append(ch)
        return "".join(out)

    def _layer_names_table(self) -> str:
        """Emit the static name table consumed by inference_prof and the
        accessors declared in inference.h.  Always emitted, even when
        profiling is off — the table is tiny and gives host code a stable
        way to introspect the graph."""
        names = self._layer_display_names()
        n     = len(names)

        parts = [_banner("Per-layer name table (used by inference_prof)")]
        if n > 0:
            parts.append(
                f"static const char *const inference_layer_names[{n}] = {{"
            )
            for i, name in enumerate(names):
                parts.append(f'    "{self._c_string_escape(name)}",'
                             f"  /* [{i}] */")
            parts.append("};")
        else:
            parts.append("/* (no scheduled nodes — layer-name table is empty) */")
        parts.append("")
        parts.append("unsigned inference_num_layers(void)")
        parts.append("{")
        parts.append(f"    return {n}u;")
        parts.append("}")
        parts.append("")
        parts.append("const char *const *inference_layer_names_ptr(void)")
        parts.append("{")
        if n > 0:
            parts.append("    return inference_layer_names;")
        else:
            parts.append("    return (const char *const *)0;")
        parts.append("}")
        parts.append("")
        return "\n".join(parts)

    def _run_op_helper(self) -> str:
        nodes          = self._graph.nodes
        # All VectorOP ScheduledNodes use run_op() (broadcast via outer/inc params)
        need_run_op    = any(isinstance(sn, ScheduledNode) for sn in nodes)
        need_run_op_act = any(
            isinstance(sn, ScheduledNode) and sn.act != 0 for sn in nodes
        )
        need_run_matmul = self._has_matmul_nodes
        need_run_matmul_at = any(
            isinstance(sn, MatmulNode) and sn.outer_count > 1
            for sn in nodes
        )
        need_run_conv = self._has_conv_nodes
        need_run_pool = self._has_pool_nodes

        titles = []
        if need_run_op and need_run_op_act:
            titles.append("run_op() / run_op_act() — VectorOPKernel dispatch helpers")
        elif need_run_op:
            titles.append("run_op() — VectorOPKernel dispatch helper")
        if need_run_matmul and need_run_matmul_at:
            titles.append("run_matmul() / run_matmul_at() — MatmulKernel dispatch helpers")
        elif need_run_matmul_at:
            titles.append("run_matmul_at() — MatmulKernel dispatch helper")
        elif need_run_matmul:
            titles.append("run_matmul() — MatmulKernel dispatch helper")
        if need_run_conv:
            titles.append("run_conv() — ConvKernel dispatch helper")
        if need_run_pool:
            titles.append("run_pool() — PoolingKernel dispatch helper")
        title = " / ".join(titles) if titles else "Kernel dispatch helpers"

        parts = [_banner(title)]

        # Look up the VectorOP, Matmul, Conv, and Pool kernel descriptors (if active)
        from ..kernels import KERNEL_REGISTRY
        vop_kd  = KERNEL_REGISTRY.get("VectorOPKernel")
        mm_kd   = KERNEL_REGISTRY.get("MatmulKernel")
        conv_kd = KERNEL_REGISTRY.get("ConvKernel")
        pool_kd = KERNEL_REGISTRY.get("PoolKernel")
        vop_var  = vop_kd.c_var  if vop_kd  else "s_vectoropkernel"
        mm_var   = mm_kd.c_var   if mm_kd   else "s_matmulkernel"
        conv_var = conv_kd.c_var if conv_kd else "s_convkernel"
        pool_var = pool_kd.c_var if pool_kd else "s_poolkernel"

        if need_run_op:
            parts.append(
                "/*\n"
                " * run_op() — program AXI-Lite registers and start the kernel.\n"
                " *\n"
                " * NON-BLOCKING: returns immediately after Start.  The caller must\n"
                " * drain the kernel via kernel_wait(KERNEL_VECTOROP) before reading\n"
                " * the output buffer or reusing the lane for another op.  Waits are\n"
                " * interleaved by inference_run() based on the data-flow DAG so\n"
                " * different-lane kernels can overlap.\n"
                " *\n"
                " * The kernel's AXI master reads/writes DDR using PHYSICAL addresses;\n"
                " * inference_buf_phys() provides the address to program into the registers.\n"
                " *\n"
                " * Cache sync is entirely the CALLER'S responsibility:\n"
                " *   - sync_to_device for inputs: done once per inference_run() call\n"
                " *     for user inputs; done once in inference_init() for weights.\n"
                " *   - sync_from_device for outputs: done once at the end of\n"
                " *     inference_run() for graph outputs only.  Internal buffers that\n"
                " *     flow kernel-to-kernel never touch the CPU, so they need no sync.\n"
                " *\n"
                " *   a      input buffer A  (always required)\n"
                " *   b      input buffer B  (NULL for unary ops: RELU, RELU6)\n"
                " *   c      output buffer C (must not alias a or b)\n"
                " *   size   elements per inner chunk\n"
                " *   op     VECTOROP_* constant\n"
                " *   outer  number of outer broadcasting iterations (1 = no broadcast)\n"
                " *   a_inc  element stride for a per outer step (0 = a repeats)\n"
                " *   b_inc  element stride for b per outer step (0 = b repeats)\n"
                " *\n"
                " * run_op writes act = 0 (no fused activation; the register keeps its\n"
                " * last value across runs, so it is written every call).  A node with\n"
                " * a fused Relu / Clip(0,6) uses run_op_act, which takes the act code.\n"
                " *\n"
                " * Alignment contract (VectorOP.h): every run start of a, b and c is\n"
                " * 16-byte aligned (buffer bases are 64-byte aligned, a_inc / b_inc\n"
                " * are 0 or a multiple of INFERENCE_ALIGN_ELEMS); the kernel reads\n"
                " * whole 16-byte words and writes the last word of every run whole,\n"
                " * i.e. c may be written up to the next 16-byte boundary past size\n"
                " * (into the stride gap / the buffer's 64-byte allocation padding).\n"
                " */\n"
                "static void run_op(\n"
                "    inference_buf_t *a,\n"
                "    inference_buf_t *b,\n"
                "    inference_buf_t *c,\n"
                "    unsigned         size,\n"
                "    unsigned         op,\n"
                "    unsigned         outer,\n"
                "    unsigned         a_inc,\n"
                "    unsigned         b_inc)\n"
                "{\n"
                f"    XVectoropkernel_Set_a(&{vop_var}, inference_buf_phys(a));\n"
                f"    XVectoropkernel_Set_b(&{vop_var}, b ? inference_buf_phys(b) : (u64)0);\n"
                f"    XVectoropkernel_Set_c(&{vop_var}, inference_buf_phys(c));\n"
                f"    XVectoropkernel_Set_size(&{vop_var}, size);\n"
                f"    XVectoropkernel_Set_op(&{vop_var}, op);\n"
                f"    XVectoropkernel_Set_outer(&{vop_var}, outer);\n"
                f"    XVectoropkernel_Set_a_inc(&{vop_var}, a_inc);\n"
                f"    XVectoropkernel_Set_b_inc(&{vop_var}, b_inc);\n"
                f"    XVectoropkernel_Set_act(&{vop_var}, VECTOROP_ACT_NONE);\n"
                f"    XVectoropkernel_Start(&{vop_var});\n"
                "}\n"
            )

        if need_run_op_act:
            parts.append(
                "/*\n"
                " * run_op_act() — run_op() with a fused activation: the kernel applies\n"
                " * VECTOROP_ACT_RELU / VECTOROP_ACT_RELU6 to the op result before\n"
                " * writing c, so Add -> Relu is one pass over the data.\n"
                " */\n"
                "static void run_op_act(\n"
                "    inference_buf_t *a,\n"
                "    inference_buf_t *b,\n"
                "    inference_buf_t *c,\n"
                "    unsigned         size,\n"
                "    unsigned         op,\n"
                "    unsigned         outer,\n"
                "    unsigned         a_inc,\n"
                "    unsigned         b_inc,\n"
                "    unsigned         act)\n"
                "{\n"
                f"    XVectoropkernel_Set_a(&{vop_var}, inference_buf_phys(a));\n"
                f"    XVectoropkernel_Set_b(&{vop_var}, b ? inference_buf_phys(b) : (u64)0);\n"
                f"    XVectoropkernel_Set_c(&{vop_var}, inference_buf_phys(c));\n"
                f"    XVectoropkernel_Set_size(&{vop_var}, size);\n"
                f"    XVectoropkernel_Set_op(&{vop_var}, op);\n"
                f"    XVectoropkernel_Set_outer(&{vop_var}, outer);\n"
                f"    XVectoropkernel_Set_a_inc(&{vop_var}, a_inc);\n"
                f"    XVectoropkernel_Set_b_inc(&{vop_var}, b_inc);\n"
                f"    XVectoropkernel_Set_act(&{vop_var}, act);\n"
                f"    XVectoropkernel_Start(&{vop_var});\n"
                "}\n"
            )

        if need_run_matmul:
            parts.append(
                "/*\n"
                " * run_matmul() — program XMatmulkernel AXI-Lite registers and start.\n"
                " *\n"
                " * NON-BLOCKING: caller must kernel_wait(KERNEL_MATMUL) before\n"
                " * reading c or reusing the Matmul lane.\n"
                " *\n"
                " *   a / b / c   DMA buffer pointers (physical addresses via\n"
                " *                inference_buf_phys())\n"
                " *   n           output rows  (A.rows)\n"
                " *   k           inner dim    (A.cols == B.rows)\n"
                " *   m           output cols  (B.cols)\n"
                " *   batch       number of independent matrix multiplications\n"
                " *   a_stride    elements between consecutive A batch slices\n"
                " *                (0 when batch == 1)\n"
                " *   b_stride    elements between consecutive B batch slices\n"
                " *                (0 when B broadcasts across batch)\n"
                " *   c_stride    elements between consecutive Y batch slices\n"
                " *                (0 when batch == 1)\n"
                " *   b_packed    1 when b holds the kernel's tile-major packed\n"
                " *                weight image (constant weights, emitted by the\n"
                " *                scheduler); 0 for a row-major [k][m] matrix\n"
                " */\n"
                "static void run_matmul(\n"
                "    inference_buf_t *a,\n"
                "    inference_buf_t *b,\n"
                "    inference_buf_t *c,\n"
                "    uint32_t n, uint32_t k, uint32_t m, uint32_t batch,\n"
                "    uint32_t a_stride, uint32_t b_stride, uint32_t c_stride,\n"
                "    uint32_t b_packed)\n"
                "{\n"
                f"    XMatmulkernel_Set_a(&{mm_var}, inference_buf_phys(a));\n"
                f"    XMatmulkernel_Set_b(&{mm_var}, inference_buf_phys(b));\n"
                f"    XMatmulkernel_Set_c(&{mm_var}, inference_buf_phys(c));\n"
                f"    XMatmulkernel_Set_n(&{mm_var}, n);\n"
                f"    XMatmulkernel_Set_k(&{mm_var}, k);\n"
                f"    XMatmulkernel_Set_m(&{mm_var}, m);\n"
                f"    XMatmulkernel_Set_batch(&{mm_var}, batch);\n"
                f"    XMatmulkernel_Set_a_batch_stride(&{mm_var}, a_stride);\n"
                f"    XMatmulkernel_Set_b_batch_stride(&{mm_var}, b_stride);\n"
                f"    XMatmulkernel_Set_c_batch_stride(&{mm_var}, c_stride);\n"
                f"    XMatmulkernel_Set_b_packed(&{mm_var}, b_packed);\n"
                f"    XMatmulkernel_Start(&{mm_var});\n"
                "}\n"
            )

        if need_run_matmul_at:
            parts.append(
                "/*\n"
                " * run_matmul_at() — offset-based dispatch for outer-loop broadcasting.\n"
                " *\n"
                " * BLOCKING: each iteration of the calling outer loop reuses the same\n"
                " * Matmul lane registers, so this helper polls IsDone() before returning.\n"
                " * Nodes that emit run_matmul_at therefore drain the Matmul lane themselves;\n"
                " * inference_run() treats them as synchronous on the Matmul lane.\n"
                " *\n"
                " * Used when one input has a leading batch dimension absent from the\n"
                " * other.  The outer loop advances the larger-rank operand by one\n"
                " * batch-block per step while the smaller-rank operand repeats at\n"
                " * offset 0.  Element offsets are converted to byte offsets using\n"
                " * INFERENCE_BYTES_PER_ELEM.\n"
                " *\n"
                " *   a_off / b_off / c_off   element offset into each buffer\n"
                " */\n"
                "static void run_matmul_at(\n"
                "    inference_buf_t *a, unsigned a_off,\n"
                "    inference_buf_t *b, unsigned b_off,\n"
                "    inference_buf_t *c, unsigned c_off,\n"
                "    uint32_t n, uint32_t k, uint32_t m, uint32_t batch,\n"
                "    uint32_t a_stride, uint32_t b_stride, uint32_t c_stride,\n"
                "    uint32_t b_packed)\n"
                "{\n"
                f"    XMatmulkernel_Set_a(&{mm_var},\n"
                "        inference_buf_phys(a)"
                " + (uint64_t)a_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XMatmulkernel_Set_b(&{mm_var},\n"
                "        inference_buf_phys(b)"
                " + (uint64_t)b_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XMatmulkernel_Set_c(&{mm_var},\n"
                "        inference_buf_phys(c)"
                " + (uint64_t)c_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XMatmulkernel_Set_n(&{mm_var}, n);\n"
                f"    XMatmulkernel_Set_k(&{mm_var}, k);\n"
                f"    XMatmulkernel_Set_m(&{mm_var}, m);\n"
                f"    XMatmulkernel_Set_batch(&{mm_var}, batch);\n"
                f"    XMatmulkernel_Set_a_batch_stride(&{mm_var}, a_stride);\n"
                f"    XMatmulkernel_Set_b_batch_stride(&{mm_var}, b_stride);\n"
                f"    XMatmulkernel_Set_c_batch_stride(&{mm_var}, c_stride);\n"
                f"    XMatmulkernel_Set_b_packed(&{mm_var}, b_packed);\n"
                f"    XMatmulkernel_Start(&{mm_var});\n"
                f"    while (!XMatmulkernel_IsDone(&{mm_var})) {{}}\n"
                "}\n"
            )

        if need_run_conv:
            parts.append(
                "/*\n"
                " * run_conv() — program XConvkernel AXI-Lite registers and start.\n"
                " *\n"
                " * NON-BLOCKING: caller must kernel_wait(KERNEL_CONV) before\n"
                " * reading y or reusing the Conv lane.\n"
                " *\n"
                " *   x / weight / bias / y   DMA buffer pointers\n"
                " *                            (physical addresses via inference_buf_phys()).\n"
                " *                            bias may be NULL when has_bias == 0;\n"
                " *                            ConvKernel never reads gmem2 in that case.\n"
                " *   batch        input batch size (N)\n"
                " *   in_ch        input channels (C)\n"
                " *   in_h/w       input spatial dimensions\n"
                " *   out_ch       output channels (M)\n"
                " *   out_h/w      output spatial dimensions\n"
                " *   kh/kw        filter kernel size\n"
                " *   stride_h/w   convolution stride\n"
                " *   dilation_h/w convolution dilation\n"
                " *   pad_top/left padding (top row / left column)\n"
                " *   has_bias     1 = add per-channel bias; 0 = skip bias\n"
                " *   is_depthwise 0 = standard (group=1); 1 = depthwise (group=in_ch)\n"
                " */\n"
                "static void run_conv(\n"
                "    inference_buf_t *x,\n"
                "    inference_buf_t *weight,\n"
                "    inference_buf_t *bias,\n"
                "    inference_buf_t *y,\n"
                "    unsigned batch,\n"
                "    unsigned in_ch,  unsigned in_h,  unsigned in_w,\n"
                "    unsigned out_ch, unsigned out_h, unsigned out_w,\n"
                "    unsigned kh,     unsigned kw,\n"
                "    unsigned stride_h, unsigned stride_w,\n"
                "    unsigned dilation_h, unsigned dilation_w,\n"
                "    unsigned pad_top, unsigned pad_left,\n"
                "    unsigned has_bias, unsigned is_depthwise)\n"
                "{\n"
                f"    XConvkernel_Set_x           (&{conv_var}, inference_buf_phys(x));\n"
                f"    XConvkernel_Set_weight      (&{conv_var}, inference_buf_phys(weight));\n"
                f"    XConvkernel_Set_bias        (&{conv_var},"
                " bias ? inference_buf_phys(bias) : (u64)0);\n"
                f"    XConvkernel_Set_y           (&{conv_var}, inference_buf_phys(y));\n"
                f"    XConvkernel_Set_batch       (&{conv_var}, batch);\n"
                f"    XConvkernel_Set_in_ch       (&{conv_var}, in_ch);\n"
                f"    XConvkernel_Set_in_h        (&{conv_var}, in_h);\n"
                f"    XConvkernel_Set_in_w        (&{conv_var}, in_w);\n"
                f"    XConvkernel_Set_out_ch      (&{conv_var}, out_ch);\n"
                f"    XConvkernel_Set_out_h       (&{conv_var}, out_h);\n"
                f"    XConvkernel_Set_out_w       (&{conv_var}, out_w);\n"
                f"    XConvkernel_Set_kh          (&{conv_var}, kh);\n"
                f"    XConvkernel_Set_kw          (&{conv_var}, kw);\n"
                f"    XConvkernel_Set_stride_h    (&{conv_var}, stride_h);\n"
                f"    XConvkernel_Set_stride_w    (&{conv_var}, stride_w);\n"
                f"    XConvkernel_Set_dilation_h  (&{conv_var}, dilation_h);\n"
                f"    XConvkernel_Set_dilation_w  (&{conv_var}, dilation_w);\n"
                f"    XConvkernel_Set_pad_top     (&{conv_var}, pad_top);\n"
                f"    XConvkernel_Set_pad_left    (&{conv_var}, pad_left);\n"
                f"    XConvkernel_Set_has_bias    (&{conv_var}, has_bias);\n"
                f"    XConvkernel_Set_is_depthwise(&{conv_var}, is_depthwise);\n"
                f"    XConvkernel_Start(&{conv_var});\n"
                "}\n"
            )

        if need_run_pool:
            parts.append(
                "/*\n"
                " * run_pool() — program XPoolingkernel AXI-Lite registers and start.\n"
                " *\n"
                " * NON-BLOCKING: caller must kernel_wait(KERNEL_POOL) before\n"
                " * reading y or reusing the Pool lane.\n"
                " *\n"
                " *   x / y       DMA buffer pointers (physical addresses via\n"
                " *                inference_buf_phys())\n"
                " *   batch       input batch size (N)\n"
                " *   channels    input/output channels (C)\n"
                " *   in_h/w      input spatial dimensions\n"
                " *   out_h/w     output spatial dimensions\n"
                " *   pool_h/w    pool window size\n"
                " *   stride_h/w  pooling stride\n"
                " *   pad_top/left  zero-padding (top row / left column)\n"
                " *   dil_h/w     dilation (1 = no dilation)\n"
                " *   pool_type   0=MaxPool  1=AveragePool  2=LpPool\n"
                " *   lp_order    Lp norm order (1 or 2; ignored for Max/Avg)\n"
                " *   count_include_pad  1 = include padding in AVG denominator\n"
                " */\n"
                "static void run_pool(\n"
                "    inference_buf_t *x,\n"
                "    inference_buf_t *y,\n"
                "    unsigned batch,\n"
                "    unsigned channels,\n"
                "    unsigned in_h,     unsigned in_w,\n"
                "    unsigned out_h,    unsigned out_w,\n"
                "    unsigned pool_h,   unsigned pool_w,\n"
                "    unsigned stride_h, unsigned stride_w,\n"
                "    unsigned pad_top,  unsigned pad_left,\n"
                "    unsigned dil_h,    unsigned dil_w,\n"
                "    unsigned pool_type, unsigned lp_order,\n"
                "    unsigned count_include_pad)\n"
                "{\n"
                f"    XPoolingkernel_Set_x                 (&{pool_var}, inference_buf_phys(x));\n"
                f"    XPoolingkernel_Set_y                 (&{pool_var}, inference_buf_phys(y));\n"
                f"    XPoolingkernel_Set_batch             (&{pool_var}, batch);\n"
                f"    XPoolingkernel_Set_channels          (&{pool_var}, channels);\n"
                f"    XPoolingkernel_Set_in_h              (&{pool_var}, in_h);\n"
                f"    XPoolingkernel_Set_in_w              (&{pool_var}, in_w);\n"
                f"    XPoolingkernel_Set_out_h             (&{pool_var}, out_h);\n"
                f"    XPoolingkernel_Set_out_w             (&{pool_var}, out_w);\n"
                f"    XPoolingkernel_Set_pool_h            (&{pool_var}, pool_h);\n"
                f"    XPoolingkernel_Set_pool_w            (&{pool_var}, pool_w);\n"
                f"    XPoolingkernel_Set_stride_h          (&{pool_var}, stride_h);\n"
                f"    XPoolingkernel_Set_stride_w          (&{pool_var}, stride_w);\n"
                f"    XPoolingkernel_Set_pad_top           (&{pool_var}, pad_top);\n"
                f"    XPoolingkernel_Set_pad_left          (&{pool_var}, pad_left);\n"
                f"    XPoolingkernel_Set_dil_h             (&{pool_var}, dil_h);\n"
                f"    XPoolingkernel_Set_dil_w             (&{pool_var}, dil_w);\n"
                f"    XPoolingkernel_Set_pool_type         (&{pool_var}, pool_type);\n"
                f"    XPoolingkernel_Set_lp_order          (&{pool_var}, lp_order);\n"
                f"    XPoolingkernel_Set_count_include_pad (&{pool_var}, count_include_pad);\n"
                f"    XPoolingkernel_Start(&{pool_var});\n"
                "}\n"
            )

        return "".join(parts)

    def _load_weight_helper(self) -> str:
        """
        Emit the _load_weight() helper used by inference_init() to read a .dat
        file into a DMA buffer.  Only included when the model has large weights.
        """
        return (
            _banner("_load_weight() — read an external weight .dat file") +
            "/*\n"
            " * _load_weight() — open weights/<name>.dat and fread() its contents\n"
            " * directly into the DMA buffer.  The .dat file contains raw little-\n"
            " * endian uint16 values in the same ap_fixed<16,8> encoding used by\n"
            " * the inline ROM arrays.\n"
            " *\n"
            " * INFERENCE_WEIGHTS_DIR is set by CMake (default \".\").\n"
            " * Override at configure time: cmake -DINFERENCE_WEIGHTS_DIR=/path/to/weights\n"
            " */\n"
            "#ifndef INFERENCE_WEIGHTS_DIR\n"
            "#  define INFERENCE_WEIGHTS_DIR  \".\"\n"
            "#endif\n"
            "\n"
            "static int _load_weight(inference_buf_t *buf,\n"
            "                        const char *name, unsigned n_elem)\n"
            "{\n"
            "    char path[512];\n"
            "    FILE *f;\n"
            "    size_t n_read;\n"
            "\n"
            "    snprintf(path, sizeof(path),\n"
            "             INFERENCE_WEIGHTS_DIR \"/weights/%s.dat\", name);\n"
            "    f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            '        fprintf(stderr,\n'
            '                "inference: cannot open weight file \'%s\'\\n", path);\n'
            "        return -1;\n"
            "    }\n"
            "    n_read = fread(inference_buf_ptr(buf),\n"
            "                  INFERENCE_BYTES_PER_ELEM, n_elem, f);\n"
            "    fclose(f);\n"
            "    if (n_read != (size_t)n_elem) {\n"
            '        fprintf(stderr,\n'
            '                "inference: short read from \'%s\':"\n'
            '                " got %zu, expected %u\\n",\n'
            '                path, n_read, n_elem);\n'
            "        return -1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )

    def _init_function(self) -> str:
        graph         = self._graph
        weights       = graph.weight_tensors
        intermediates = graph.intermediate_tensors

        reshape_aliases     = self._reshape_aliases      # {out_name: src_c_name}
        run_reshape_aliases = self._run_reshape_aliases  # subset: source is graph input
        external            = {t.onnx_name for t in self.large_weight_tensors}

        # Compute pool layout: list of (onnx_name, offset, alloc) + total_elems
        pool_layout, total_pool_elems = self._compute_pool_layout()
        # Build lookup: onnx_name -> (offset, alloc)
        pool_map = {name: (off, alloc) for (name, off, alloc) in pool_layout}

        # Determine if a pool is needed (at least one weight or non-alias intermediate)
        need_pool = bool(pool_layout)

        alloc_lines: List[str] = []

        if need_pool:
            alloc_lines.append(
                "    /* One contiguous DMA allocation covers all weights and intermediate buffers.\n"
                "     * Each sub-buffer is an aligned view; 64-byte gaps between slots ensure\n"
                "     * cache-line alignment at every physical base address. */"
            )
            alloc_lines.append(
                f"    s_alloc_pool = inference_buf_alloc({total_pool_elems}u);"
            )
            alloc_lines.append(
                "    if (!s_alloc_pool) { rc = -1; goto fail; }"
            )
            alloc_lines.append("")

        if weights:
            alloc_lines.append("    /* Weights */")
            for t in weights:
                off, alloc = pool_map[t.onnx_name]
                alloc_lines.append(
                    f"    inference_buf_init_view(&_s_buf_{t.c_name},"
                    f" s_alloc_pool, {off}u, {alloc}u);"
                )
                alloc_lines.append(
                    f"    {t.c_name} = &_s_buf_{t.c_name};"
                )
                if t.onnx_name in external:
                    alloc_lines.append(
                        f"    if (_load_weight({t.c_name},"
                        f" \"{t.c_name}\", {t.numel}u) != 0) {{ rc = -1; goto fail; }}"
                    )
                else:
                    alloc_lines.append(
                        f"    memcpy(inference_buf_ptr({t.c_name}),"
                        f" _rom_{t.c_name}, sizeof(_rom_{t.c_name}));"
                    )
            alloc_lines.append("")
            # Single sync for all weights
            alloc_lines.append(
                "    /* Flush all weights to device in one shot — intermediates get zeroed memory\n"
                "     * (harmless; the hardware overwrites them before any CPU read). */"
            )
            alloc_lines.append("    inference_buf_sync_to_device(s_alloc_pool);")
            alloc_lines.append("")

        views = self._view_aliases                       # Slice pieces (sub-buffers)
        if intermediates:
            # Emit reshape aliases last (they reference other intermediate/weight c_name pointers)
            non_alias = [t for t in intermediates
                         if t.onnx_name not in reshape_aliases and t.onnx_name not in views]
            alias_tensors = [t for t in intermediates if t.onnx_name in reshape_aliases]

            if non_alias:
                alloc_lines.append("    /* Intermediate buffers */")
                for t in non_alias:
                    off, alloc = pool_map[t.onnx_name]
                    alloc_lines.append(
                        f"    inference_buf_init_view(&_s_buf_{t.c_name},"
                        f" s_alloc_pool, {off}u, {alloc}u);"
                    )
                    alloc_lines.append(
                        f"    {t.c_name} = &_s_buf_{t.c_name};"
                    )
                alloc_lines.append("")

            view_tensors = [t for t in intermediates if t.onnx_name in views]
            if view_tensors:
                alloc_lines.append("    /* Slice views: contiguous pieces of a pool buffer (no copy) */")
                for t in view_tensors:
                    root_c, off, count = views[t.onnx_name]
                    alloc_lines.append(
                        f"    inference_buf_init_view(&_s_buf_{t.c_name}, {root_c},"
                        f" {off}u, {count}u);"
                    )
                    alloc_lines.append(f"    {t.c_name} = &_s_buf_{t.c_name};")
                alloc_lines.append("")

            if alias_tensors:
                alloc_lines.append("    /* Reshape aliases */")
                for t in alias_tensors:
                    if t.onnx_name in run_reshape_aliases:
                        # Source is a graph input — assigned in inference_run(), not here.
                        alloc_lines.append(
                            f"    /* {t.c_name}: input reshape alias — assigned in"
                            f" inference_run() */"
                        )
                        continue
                    src_c = reshape_aliases[t.onnx_name]
                    alloc_lines.append(
                        f"    {t.c_name} = {src_c};  /* reshape alias */"
                    )
                alloc_lines.append("")

        host_ops = self._host_op_nodes
        if host_ops:
            alloc_lines.append("    /* Cached host staging for the SpaceToDepth reorders */")
            for sn in host_ops:
                alloc_lines.append(
                    f"    {sn.stage_c_name} = (Data_t *)malloc(2u * {sn.numel}u"
                    f" * INFERENCE_BYTES_PER_ELEM);"
                )
                alloc_lines.append(
                    f"    if (!{sn.stage_c_name}) {{ rc = -1; goto fail; }}"
                )
            alloc_lines.append("")

        if self._host_nodes:
            alloc_lines.append("    /* Cached staging arena for the host-CPU ops */")
            alloc_lines.append(
                f"    s_host_stage = (Data_t *)malloc({self._host_stage_elems}u"
                f" * INFERENCE_BYTES_PER_ELEM);"
            )
            alloc_lines.append("    if (!s_host_stage) { rc = -1; goto fail; }")
            alloc_lines.append("")
            alloc_lines.append("    /* Host-op lookup tables */")
            alloc_lines.append("    if (host_runtime_init() != 0) { rc = -1; goto fail; }")
            alloc_lines.append("")

        alloc_str = ("\n".join(alloc_lines) + "\n") if alloc_lines else ""

        # inference_deinit(): null all weight and intermediate pointers,
        # free the single pool, then close pool.
        deinit_free: List[str] = []
        for sn in host_ops:
            deinit_free.append(
                f"    free({sn.stage_c_name}); {sn.stage_c_name} = NULL;"
            )
        if self._host_nodes:
            deinit_free.append("    host_runtime_deinit();")
            deinit_free.append("    free(s_host_stage); s_host_stage = NULL;")
        for t in weights + intermediates:
            if t.onnx_name in reshape_aliases:
                deinit_free.append(
                    f"    {t.c_name} = NULL;  /* reshape alias — not owned */"
                )
            else:
                deinit_free.append(f"    {t.c_name} = NULL;")
        if need_pool:
            deinit_free.append(
                "    inference_buf_free(s_alloc_pool); s_alloc_pool = NULL;"
            )
        deinit_body = ("\n".join(deinit_free) + "\n") if deinit_free else ""

        load_helper = self._load_weight_helper() if self.large_weight_tensors else ""

        # Build inference_init() signature: one const char* param per active kernel
        active = self._active_kernels
        if len(active) == 1:
            params_decl = f"    const char *{active[0].init_param}"
        else:
            params_decl = ",\n".join(
                f"    const char *{kd.init_param}" for kd in active
            )

        # One Initialize call per active kernel
        init_calls = []
        for kd in active:
            init_calls.append(
                f"    /* Initialise {kd.name} driver (UIO: {kd.uio_default}) */\n"
                f"    rc = {kd.c_type}_Initialize(&{kd.c_var}, {kd.init_param});\n"
                "    if (rc != 0) goto fail;\n"
            )
        init_calls_str = "\n".join(init_calls)

        return (
            load_helper +
            _banner("inference_init() / inference_deinit()") +
            "/* Internal: pool lifecycle — defined in inference_buf.c */\n"
            "int  inference_buf_pool_init(void);\n"
            "void inference_buf_pool_deinit(void);\n"
            "\n"
            "int inference_init(\n"
            f"{params_decl})\n"
            "{\n"
            "    int rc;\n"
            "\n"
            "    /* Initialise DMA buffer pool */\n"
            "    rc = inference_buf_pool_init();\n"
            "    if (rc != 0) return rc;\n"
            "\n"
            f"{alloc_str}"
            f"{init_calls_str}"
            "    return 0;\n"
            "\n"
            "fail:\n"
            "    inference_deinit();\n"
            "    return rc;\n"
            "}\n"
            "\n"
            "void inference_deinit(void)\n"
            "{\n"
            f"{deinit_body}"
            "    inference_buf_pool_deinit();\n"
            "}\n"
        )

    def _inference_function(self) -> str:
        graph   = self._graph
        inputs  = graph.input_tensors
        outputs = graph.output_tensors

        params = []
        for t in inputs:
            params.append(f"    inference_buf_t *{t.c_name}")
        for t in outputs:
            params.append(f"    inference_buf_t *{t.c_name}")
        param_str = ",\n".join(params)

        # ---- Cache sync strategy ------------------------------------------ #
        # The FPGA kernels access DDR directly (non-coherent); the CPU mapping
        # of the buffers is cacheable by default (inference_buf.c).  Every
        # CPU <-> kernel hand-off is therefore bracketed (Linux DMA-API rules):
        #   1. Flush user inputs (graph inputs, CPU-written) to DDR once before
        #      the first kernel invocation.  Weights are already flushed in
        #      inference_init() and never change.
        #   2. Clean graph outputs before any kernel writes them: a caller that
        #      wrote into an output buffer (e.g. a memset) leaves dirty cache
        #      lines whose eviction would overwrite the kernel's result.
        #   3. Internal buffers are only written by the CPU inside a host op
        #      (SpaceToDepthNode / HostNode), which invalidates a kernel-written
        #      source before reading it and flushes its own output, at its
        #      position in the schedule (after the producer's lane drained).
        #   4. Invalidate graph outputs once after all ops complete, so the
        #      caller reads the results through the CPU virtual address.
        #      Internal (kernel-to-kernel) buffers are skipped entirely.
        # -------------------------------------------------------------------
        sync_in_lines = ([
            "    /* Flush user inputs to DDR (CPU → FPGA) */",
        ] + [
            f"    inference_buf_sync_to_device({t.c_name});"
            for t in inputs
        ]) if inputs else []
        if outputs:
            sync_in_lines += [
                "    /* Clean graph outputs: no CPU-dirty cache line may be evicted over a",
                "     * kernel's result (the caller may have written these buffers). */",
            ] + [
                f"    inference_buf_sync_to_device({t.c_name});"
                for t in outputs
            ]

        sync_out_lines = [
            "    /* Invalidate graph outputs in CPU cache (FPGA → CPU) */",
        ] + [
            f"    inference_buf_sync_from_device({t.c_name});"
            for t in outputs
        ]

        # ---- Interleaved start / wait emission --------------------------- #
        # The schedule is built once by _compute_event_stream() (in _core)
        # and consumed verbatim here.  The same stream drives _compute_
        # live_intervals(), so the buffer-slot colouring and the emitted
        # kernel_wait() calls agree on which kernels are in flight at any
        # point — buffers cannot be reused while another lane is still
        # reading or writing them.
        # ----------------------------------------------------------------- #
        nodes_by_idx = {sn.index: sn for sn in graph.nodes}
        body_lines: list = []
        drain_started = False

        for ev in self._compute_event_stream():
            kind = ev[0]
            if kind == 'comment':
                sn = nodes_by_idx[ev[1]]
                body_lines.append(sn.emit_comment())

            elif kind == 'reshape':
                sn = nodes_by_idx[ev[1]]
                # ReshapeNode emits an empty call — keep blank for readability.
                # A Slice view was set up once in inference_init().
                if not isinstance(sn, SliceNode):
                    body_lines.append(sn.emit_call(self._layouts))
                body_lines.append("")

            elif kind == 'wait':
                _, kid, drained = ev
                body_lines.append(f"    kernel_wait({kid});")
                body_lines.append(f"    INFERENCE_PROF_END({drained}u);")

            elif kind == 'start':
                sn = nodes_by_idx[ev[1]]
                body_lines.append(f"    INFERENCE_PROF_BEGIN({sn.index}u);")
                body_lines.append(sn.emit_call(self._layouts))
                body_lines.append("")

            elif kind == 'start_sync':
                sn = nodes_by_idx[ev[1]]
                body_lines.append(f"    INFERENCE_PROF_BEGIN({sn.index}u);")
                body_lines.append(sn.emit_call(self._layouts))
                body_lines.append(f"    INFERENCE_PROF_END({sn.index}u);")
                body_lines.append("")

            elif kind == 'cpu':
                # Host-side op (SpaceToDepthNode / HostNode): runs inline on
                # the CPU and syncs its own buffers; profiled like a
                # synchronous node.
                sn = nodes_by_idx[ev[1]]
                body_lines.append(f"    INFERENCE_PROF_BEGIN({sn.index}u);")
                if isinstance(sn, HostNode):
                    body_lines.append(self._emit_host_block(sn))
                else:
                    body_lines.append(sn.emit_call(self._layouts))
                body_lines.append(f"    INFERENCE_PROF_END({sn.index}u);")
                body_lines.append("")

            elif kind == 'drain':
                _, kid, drained = ev
                if not drain_started:
                    body_lines.append(
                        "    /* Drain remaining in-flight kernels before output sync. */"
                    )
                    drain_started = True
                body_lines.append(f"    kernel_wait({kid});")
                body_lines.append(f"    INFERENCE_PROF_END({drained}u);")

        if body_lines and body_lines[-1] == "":
            body_lines.pop()

        # ---- Run-time input reshape aliases ----------------------------------- #
        # Intermediate tensors that are reshape aliases of graph inputs cannot be
        # resolved in inference_init() because graph inputs only exist as
        # inference_run() parameters.  Assign them here at the top of the
        # function body and clear them at the bottom.
        # -------------------------------------------------------------------
        run_reshape_aliases = self._run_reshape_aliases  # {onnx_name: src_c_name}
        intermediates_map = {t.onnx_name: t for t in graph.intermediate_tensors}

        run_alias_assign_lines: list = []
        run_alias_clear_lines:  list = []
        for onnx_name, src_c in run_reshape_aliases.items():
            t = intermediates_map.get(onnx_name)
            if t is None:
                continue  # graph-output alias — handled by _input_output_reshape_copies
            run_alias_assign_lines.append(
                f"    {t.c_name} = {src_c};  /* input reshape alias */"
            )
            run_alias_clear_lines.append(
                f"    {t.c_name} = NULL;    /* input reshape alias */"
            )

        # ---- Input-to-output reshape copies ----------------------------------- #
        # When a graph output is the terminus of a reshape chain rooted at a graph
        # input (no kernel in between), the data never passes through the FPGA.
        # A CPU memcpy followed by sync_to_device ensures the subsequent
        # sync_from_device reads the correct data from DDR.
        # -------------------------------------------------------------------
        io_reshape_copies = self._input_output_reshape_copies  # [(out_t, in_c_name)]

        io_reshape_copy_lines: list = []
        for out_t, in_c in io_reshape_copies:
            io_reshape_copy_lines += [
                f"    /* '{out_t.onnx_name}' is a reshape alias of input"
                f" '{in_c}' — copy data (no kernel involved). */",
                f"    memcpy(inference_buf_ptr({out_t.c_name}),"
                f" inference_buf_ptr({in_c}),"
                f" {out_t.numel}u * INFERENCE_BYTES_PER_ELEM);",
                f"    inference_buf_sync_to_device({out_t.c_name});",
            ]

        # ---- Output alias redirect ----------------------------------------- #
        # When a graph output is the terminal node of a reshape alias chain
        # (e.g. Conv → Reshape → Squeeze → graph_output), the hardware kernel
        # writes to an internal intermediate buffer, not to the caller-supplied
        # output buffer.  We fix this by redirecting the internal buffer
        # pointer to the caller's buffer before the kernels run, then restoring
        # it after.  inference_buf_retain/release guard against a premature free
        # of the caller's buffer while the internal pointer borrows it.
        # -------------------------------------------------------------------
        output_aliases = self._output_aliases  # {out_c_name: [chain_c_name, ...]}

        alias_redirect_lines: list = []
        alias_restore_lines: list  = []

        for out_t in outputs:
            chain = output_aliases.get(out_t.c_name)
            if not chain:
                continue
            out_c = out_t.c_name
            alias_redirect_lines += [
                f"    /* '{out_t.onnx_name}' is a terminal reshape alias — redirect the",
                "     * kernel output chain to write directly into the caller's buffer. */",
                f"    inference_buf_retain({out_c});",
            ]
            for c in chain:
                alias_redirect_lines.append(f"    inference_buf_t *_prev_{c} = {c};")
            for c in chain:
                alias_redirect_lines.append(f"    {c} = {out_c};")

            alias_restore_lines += [
                f"    /* Restore internal buffer pointers for '{out_t.onnx_name}'. */",
            ]
            for c in reversed(chain):
                alias_restore_lines.append(f"    {c} = _prev_{c};")
            alias_restore_lines.append(f"    inference_buf_release({out_c});")

        sections = []
        if run_alias_assign_lines:
            sections.append("\n".join(run_alias_assign_lines))
        if alias_redirect_lines:
            sections.append("\n".join(alias_redirect_lines))
        if inputs or outputs:
            sections.append("\n".join(sync_in_lines))
        if body_lines:
            sections.append("\n".join(body_lines))
        if io_reshape_copy_lines:
            sections.append("\n".join(io_reshape_copy_lines))
        if outputs:
            sections.append("\n".join(sync_out_lines))
        if alias_restore_lines:
            sections.append("\n".join(alias_restore_lines))
        if run_alias_clear_lines:
            sections.append("\n".join(run_alias_clear_lines))
        body = "\n\n".join(sections) if sections else "    /* (empty graph) */"

        return (
            _banner("inference_run()") +
            "void inference_run(\n"
            f"{param_str})\n"
            "{\n"
            f"{body}\n"
            "}\n"
        )
