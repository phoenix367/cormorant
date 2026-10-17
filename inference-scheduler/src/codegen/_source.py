"""Mixin that generates src/inference.c."""

from __future__ import annotations
from typing import List

import numpy as np

from ..nodes    import (ACT_NAMES, OP_NAMES, OP_SOFTMAX, ConvNode, MatmulConvNode, MatmulNode,
                        ScheduledNode, SchedulerError, SpaceToDepthNode)
from ..host_nodes import (HOST_C_COMMON, HOST_C_HELPER_ORDER, HOST_C_POOL, HostNode,
                          SliceNode, host_c_helper)
from ..llm_nodes import (LLM_C_DMA, RUNTIME_GROUPS, LlmAttnConvNode, LlmKernelNode, LlmNode,
                         llm_c_helpers)
from ..vit_nodes import vit_c_helpers
from ..tts_nodes import tts_c_helpers
from ..stereo_nodes import stereo_c_helpers
from ..asr_nodes import asr_c_helpers
from ._banners  import _banner, _file_banner
from .._matmul_hw_config import MATMUL_GEMV_MAX_M


# A group-major state [G][R][D] (TensorInfo.group_layout, rows interleaved by
# K = group_kw) from its logical init image [rows][G*D] (the non-zero prefix,
# whole rows).
_STATE_SCATTER_C = """/* A group-major state [G][R][D] from its logical initial image [rows][G*D];
 * K > 1: each group's rows interleaved as a 1 x K conv input image, row r,
 * element d at ((r / 16K) * 16 + r % 16) * D*K + d*K + (r / 16) % K. */
static void _state_scatter(int16_t *dst, const int16_t *init, unsigned rows,
                           unsigned R, unsigned G, unsigned D, unsigned K)
{
    unsigned r, g, d;
    for (r = 0u; r < rows; r++)
        for (g = 0u; g < G; g++) {
            int16_t *o = dst + (size_t)g * R * D
                       + ((size_t)(r / (16u * K)) * 16u + r % 16u) * D * K + (r / 16u) % K;
            for (d = 0u; d < D; d++)
                o[(size_t)d * K] = init[((size_t)r * G + g) * D + d];
        }
}
"""


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
                if (self.large_weight_tensors or self._host_nodes
                    or self._uses_activation_unit or self._uses_softmax_unit) else ''
        stdlib = '#include <stdlib.h>   /* malloc, free (host staging buffers) */\n' \
                if (self._host_op_nodes or self._host_nodes) else ''
        mathh = '#include <math.h>     /* host ops: exp, sqrt, tanh, erf, nearbyint */\n' \
                if self._host_nodes else ''
        kernel_headers = "".join(
            f'#include "{kd.driver_prefix}.h"\n'
            for kd in self._active_kernels
        )
        err = ('/* Diagnostics: stderr on Linux; the bare-metal firmware reports through its\n'
               ' * control block (rpu/rpu_main.c) and has no console. */\n'
               '#ifdef __linux__\n'
               '#  define INFERENCE_ERR(...) fprintf(stderr, __VA_ARGS__)\n'
               '#else\n'
               '#  define INFERENCE_ERR(...) ((void)0)\n'
               '#endif\n')
        from ._source import _SourceMixin as _M
        starts = "".join(
            f"#define {kd.c_type}_Start(inst)  (inference_rpu_active() ? "
            f"inference_rpu_start({_M._KERNEL_ID_ENUM[kd.name]}) : "
            f"(inference_irq_start({_M._KERNEL_ID_ENUM[kd.name]}), ({kd.c_type}_Start)(inst)))\n"
            for kd in self._active_kernels)
        rpu = ('/* Driving the kernels from the Cortex-R5 (inference_rpu.h, RPU_PLAN B1): while\n'
               ' * attached, a Start records the call for the firmware (the registers went to a\n'
               ' * shadow page), a kernel_wait() records a wait, and INFERENCE_RPU_FLUSH() hands\n'
               ' * the batch over where the host needs the results.  While the A53 drives, a\n'
               ' * Start arms the kernel\'s interrupt for kernel_wait() (inference_irq.h), and\n'
               ' * INFERENCE_KERNEL_EXPECT(k, us) before a node\'s Start gives its calls\' predicted\n'
               ' * length (µs; the performance model or the cost model), so the wait sleeps until\n'
               ' * just before the end instead of waking on the interrupt after it. */\n'
               '#ifdef __linux__\n'
               '#include "inference_rpu.h"\n'
               '#include "inference_irq.h"\n'
               f'{starts}'
               '#define INFERENCE_RPU_FLUSH()  inference_rpu_flush()\n'
               '#define INFERENCE_RPU_ENTER(r) inference_rpu_enter(r)\n'
               '#define INFERENCE_KERNEL_EXPECT(k, us)  inference_irq_expect(k, us)\n'
               '#else\n'
               '#define INFERENCE_RPU_FLUSH()  ((void)0)\n'
               '#define INFERENCE_RPU_ENTER(r) ((void)0)\n'
               '#define INFERENCE_KERNEL_EXPECT(k, us)  ((void)0)\n'
               '#endif\n')
        return (
            _banner("Includes") +
            '#include "inference.h"\n'
            '#include "inference_prof.h"  /* INFERENCE_PROF_BEGIN/END (no-ops unless'
            ' INFERENCE_PROFILING is set) */\n'
            f'{kernel_headers}'
            f'{rpu}'
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
            + err
        )

    def _source_op_defines(self) -> str:
        """Emit VectorOPKernel op-code #defines.  Empty for MatMul-only models."""
        if not self._has_vectorop_nodes:
            return ""
        lines = [_banner("VectorOPKernel operation codes (must match VectorOP.h)")]
        for code, name in sorted(OP_NAMES.items()):
            if code >= OP_SOFTMAX and not self._uses_softmax_unit:
                continue                       # the softmax ops: only where a node uses them
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
                if isinstance(sn, HostNode) and not self._is_view(sn)
                and not self._is_inplace_merge(sn)]

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
        if sn.output.is_host:
            out = (off, 0, None)                  # host-memory output: no staging
        else:
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
                if self._is_inplace_merge(sn):        # its P.V calls wrote it
                    return True
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
            " * for the consuming kernel.  Rows / elements are split over the host\n"
            " * thread pool, GELU and Softmax's exp() are lookup tables filled at init\n"
            " * by the same code path.  The scheduler's simulator (_simulate.py /\n"
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
        if "llm" in used:
            parts.append(llm_c_helpers())
            if self._dma_states:
                parts.append(LLM_C_DMA)
        if "vit" in used:
            parts.append(vit_c_helpers())
        if "tts" in used:
            parts.append(tts_c_helpers())
        if "stereo" in used:
            parts.append(stereo_c_helpers())
        if "asr" in used:
            parts.append(asr_c_helpers())
        consts = []
        for sn in host:
            consts.extend(sn.c_file_consts(self._dtype))
        if consts:
            parts.append("/* Per-node host-op constants */\n" + "\n".join(consts) + "\n")
        items = self._llm_runtime_items()
        if items:
            groups = self._llm_runtime_groups(items)
            parts.append("/* LLM host-op runtime objects: exponent scales, tables (built in\n"
                         " * host_runtime_init) */\n"
                         + "\n".join(it.decl for it in items if it.decl) + "\n")
            for g, rows in groups.items():
                struct, table, ctype, _i, _f = RUNTIME_GROUPS[g]
                parts.append(f"{struct}\nstatic const {ctype} {table}[{len(rows)}] = {{\n"
                             + ",\n".join("    " + r for r in rows) + "\n};\n")
        parts.append(self._host_runtime_functions())
        return "\n".join(parts)

    def _llm_runtime_items(self) -> list:
        """Distinct runtime items (llm_nodes.RuntimeItem) of the LLM host ops,
        in first-use order."""
        seen: dict = {}
        for sn in self._host_nodes:
            if isinstance(sn, LlmNode):
                for it in sn.c_runtime():
                    if it.key in seen:
                        if seen[it.key] != it:
                            raise SchedulerError(f"host runtime item {it.key}: conflicting "
                                                 f"definitions")
                        continue
                    seen[it.key] = it
        return list(seen.values())

    @staticmethod
    def _llm_runtime_groups(items) -> dict:
        """{group: [row, ...]} of the grouped runtime items (RUNTIME_GROUPS order)."""
        out: dict = {}
        for g in RUNTIME_GROUPS:
            rows = [it.row for it in items if it.group == g]
            if rows:
                out[g] = rows
        return out

    def host_table_files(self) -> list:
        """[(weights/<name>.dat, bytes)] of the LLM host tables that are
        loaded from files at init (llm_nodes.HostTable.is_file)."""
        out, seen = [], set()
        for sn in self._host_nodes:
            if isinstance(sn, LlmNode):
                for tb in sn.tables():
                    if tb.is_file and tb.name not in seen:
                        seen.add(tb.name)
                        out.append((tb.name, tb))
        return out

    def _host_runtime_functions(self) -> str:
        """host_runtime_init() / host_runtime_deinit(): the thread pool and
        the lookup tables (called by inference_init / inference_deinit)."""
        luts = self._host_luts()
        decl = [f"static Data_t *{name} = NULL;" for name, _ in luts
                if name != "s_host_exp_lut"]          # declared with its helper
        init = [f"    if ({call} != 0) return -1;" for _, call in luts]
        free = [f"    free({name}); {name} = NULL;" for name, _ in luts]
        items = self._llm_runtime_items()
        for it in items:
            if it.init:
                init.append(it.init)
            if it.free:
                free.append(it.free)
        for g, rows in self._llm_runtime_groups(items).items():
            _s, table, ctype, gi, gf = RUNTIME_GROUPS[g]
            loop = (f"    {{\n        unsigned _i;\n        for (_i = 0u; _i < {len(rows)}u; _i++) {{\n"
                    f"            const {ctype} *T = &{table}[_i];\n")
            init.append(loop + f"            {gi}\n        }}\n    }}")
            free.append(loop + f"            {gf}\n        }}\n    }}")
        return (
            ("/* Lookup tables (filled in host_runtime_init) */\n" + "\n".join(decl) + "\n\n"
             if decl else "") +
            "/* Host-op runtime: worker threads, then the lookup tables (filled by\n"
            " * the pool).  Called by inference_init() / inference_deinit(). */\n"
            "static int host_runtime_init(void)\n"
            "{\n"
            "    if (host_pool_init() != 0) return -1;\n" +
            "".join(ln + "\n" for ln in init) +
            "    return 0;\n"
            "}\n"
            "\n"
            "static void host_runtime_deinit(void)\n"
            "{\n" +
            "".join(ln + "\n" for ln in free) +
            "    host_pool_deinit();\n"
            "}\n"
        )

    def _emit_llm_block(self, sn) -> str:
        """An LLM host op (llm_nodes.py): its DMA inputs / output go through
        host_in / host_out / host_out_done like every host op (invalidate a
        kernel-written input first); host-memory tensors and states are
        plain pointers passed by name."""
        ins, (o_off, o_cnt, o_io), _scratch, _ = self._host_stage_plan(sn)
        staged = {t.onnx_name: (i, off, cnt, io) for i, (t, off, cnt, io) in enumerate(ins)}
        lines = ["    {"]
        for i, (t, off, cnt, _io) in enumerate(ins):
            lines.append(f"        const Data_t *in{i};  /* '{t.onnx_name}': {cnt} elem"
                         f" (stage +{off}) */")
        odma = sn.output.dma is not None            # an int32 / float32 DMA tensor, written in place
        if not (sn.output.is_host or odma):
            lines.append(f"        Data_t       *out;  /* '{sn.output.onnx_name}': {o_cnt} elem"
                         f" (stage +{o_off}) */")
        seen = set()
        for t in sn.dma_inputs() + [t for t in sn.inputs if t.dma is not None and not t.is_state]:
            if t.c_name not in seen and self._written_by_kernel(t):
                seen.add(t.c_name)
                lines.append(f"        inference_buf_sync_from_device({t.c_name});"
                             f"  /* written by a kernel */")
        for t in sn.inputs:                         # a float32 DMA state: a kernel of another entry wrote it
            if t.is_state and t.dma == "f32" and t.c_name not in seen:
                seen.add(t.c_name)
                lines.append(f"        inference_buf_sync_from_device({t.c_name});"
                             f"  /* a kernel-written state */")
        for i, (t, off, _cnt, (nc, ch, st)) in enumerate(ins):
            lines.append(f"        in{i} = host_in({t.c_name}, s_host_stage + {off}u,"
                         f" {nc}u, {ch}u, {st}u);")
        if not (sn.output.is_host or odma):
            nc, ch, st = o_io
            lines.append(f"        out = host_out({sn.output.c_name}, s_host_stage + {o_off}u,"
                         f" {nc}u, {ch}u, {st}u);")
        ctype = {"i32": "int32_t", "f32": "float"}
        args = []
        for t in sn.inputs:
            if t.onnx_name in staged:
                args.append(f"in{staged[t.onnx_name][0]}")
            elif t.dma is not None:                 # an int32 / float32 DMA tensor, read in place
                args.append(f"({ctype[t.dma]} *)inference_buf_ptr({t.c_name})")
            elif t.is_state and not t.is_host:      # a DMA state, read / updated in place
                args.append(f"(int16_t *)inference_buf_ptr({t.c_name})")
            else:
                args.append(t.c_name)
        out = "out" if not (sn.output.is_host or odma) else (
            f"({ctype[sn.output.dma]} *)inference_buf_ptr({sn.output.c_name})" if odma else sn.output.c_name)
        for ln in sn.c_call(args, out, "NULL", [], self._dtype):
            lines.append("        " + ln)
        if odma:                                    # written in place: flush before a kernel reads it
            lines.append(f"        inference_buf_sync_to_device({sn.output.c_name});")
        elif not sn.output.is_host:
            nc, ch, st = o_io
            lines.append(f"        host_out_done({sn.output.c_name}, out, {nc}u, {ch}u, {st}u);")
        lines.append("    }")
        return "\n".join(lines)

    def _emit_host_block(self, sn) -> str:
        """One host op inside inference_run(): invalidate kernel-written
        inputs, get flat input / output pointers (the BO itself when the
        mapping is cacheable and the layout flat, else the staging arena),
        compute, and hand the output back (store if staged + flush)."""
        if isinstance(sn, LlmNode):
            return self._emit_llm_block(sn)
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

        dma_states = self._dma_states
        if dma_states:
            lines.append("")
            lines.append("/* DMA states (src/numeric.py): persistent buffers in the pool that host\n"
                         " * ops write (flushing the rows a kernel reads) and kernels read */")
            for t in dma_states:
                lay = (f" group-major [{t.group_layout[0]}][{t.shape[0]}][{t.group_layout[1]}]"
                       + (f" rows interleaved x{t.group_kw}" if t.group_kw > 1 else "")
                       if t.group_layout else "")
                lines.append(f"static inference_buf_t *{t.c_name} = NULL;"
                             f"  /* STATE '{t.onnx_name}' {t.shape}{lay} */")

        # Pool pointer and static view-struct backing storage
        weights       = self._graph.weight_tensors
        reshape_aliases = self._reshape_aliases
        pool_tensors  = (
            list(weights) + list(dma_states) +
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

        host_t = self._graph.host_tensors
        states = self._graph.state_tensors
        host_states = [t for t in states if t.is_host]
        if host_t or host_states:
            lines.append("")
            lines.append("/* Host-memory tensors (src/numeric.py): never in a DMA buffer, only the\n"
                         " * host ops touch them.  Intermediates live in s_host_arena (slots reused\n"
                         " * by liveness); states persist across inference_run() calls. */")
            if host_t:
                lines.append("static unsigned char *s_host_arena = NULL;")
            for t in host_t:
                lines.append(f"static {self._host_c_type(t)} *{t.c_name} = NULL;"
                             f"  /* '{t.onnx_name}' {t.host} {t.shape} */")
            for t in host_states:
                lay = (f" group-major [{t.group_layout[0]}][{t.shape[0]}][{t.group_layout[1]}]"
                       + (f" rows interleaved x{t.group_kw}" if t.group_kw > 1 else "")
                       if t.group_layout else "")
                lines.append(f"static {self._host_c_type(t)} *{t.c_name} = NULL;"
                             f"  /* STATE '{t.onnx_name}' {t.host} {t.shape}{lay} */")
        if states:
            for t in states:
                pre = self._state_init_prefix(t)
                if pre is not None:
                    lines.append(pre[0])
            # (the compact init loop references it for every group-major state)
            if any(t.group_layout for t in states):
                lines.append(_STATE_SCATTER_C)

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

    def _state_init_prefix(self, t):
        """(C declaration, element count) of the raw image of a state's
        non-zero prefix (its initial value, LOGICAL order — whole rows for a
        group-major state, which init scatters), or None when it starts at 0."""
        if t.init_data is None:
            return None
        v = np.asarray(t.init_data, np.float64).reshape(-1)
        nz = np.nonzero(v)[0]
        if nz.size == 0:
            return None
        n = int(nz[-1]) + 1
        if t.group_layout:
            row = int(t.shape[1])
            n = -(-n // row) * row
        if t.dma == "f32":
            raise SchedulerError(f"state '{t.onnx_name}': a float32 DMA state starts at zero")
        if t.dma == "i32":
            raw = np.clip(np.rint(v[:n] * 2.0 ** t.exp_full(self._dtype.frac_bits).reshape(-1)[:n]),
                          -2147483648.0, 2147483647.0).astype(np.int32).view(np.int16)
            lits = [str(int(x)) for x in raw]
            n = int(raw.size)
        elif t.host == "i16" or not t.is_host:
            raw = self._dtype.exp_to_storage(
                v[:n], t.exp_full(self._dtype.frac_bits).reshape(-1)[:n]).view(np.int16)
            lits = [str(int(x)) for x in raw]
        elif t.host == "i32":
            lits = [str(int(x)) for x in v[:n]]
        else:
            from ..host_nodes import _c_float
            lits = [_c_float(float(x)) for x in v[:n]]
        rows = ",\n".join("    " + ", ".join(lits[i:i + 12]) for i in range(0, n, 12))
        ctype = self._host_c_type(t) if t.is_host else "int16_t"
        return (f"static const {ctype} _state_init_{t.c_name}[{n}] = {{\n"
                f"{rows}\n}};  /* initial value of '{t.onnx_name}' (non-zero prefix) */", n)

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
        "RowOpsKernel":   "KERNEL_ROWOPS",
    }

    def _kernel_wait_helper(self) -> str:
        """Emit kernel_id_t and the weak kernel_wait(): the R5's recorded wait
        while it drives, the kernel's interrupt where the lane's Start armed it
        (Linux, inference_irq.h), else IsDone() polling.

        Only enumerates kernels the model actually uses, so single-kernel
        projects don't pull in references to undeclared driver instances.
        """
        active = list(self._active_kernels)
        if not active:
            return (
                "#ifdef __linux__\n"
                "/* No kernel in this project: nothing for the Cortex-R5 recorder (inference_rpu.h). */\n"
                "const inference_rpu_kernel_t *inference_rpu_kernels(unsigned *count)\n"
                "{\n"
                "    *count = 0u;\n"
                "    return (const inference_rpu_kernel_t *)0;\n"
                "}\n"
                "#endif\n")

        enum_lines = ["/* kernel_id_t: include/inference.h */"]

        case_lines = []
        for kd in active:
            case_lines.append(
                f"        case {self._KERNEL_ID_ENUM[kd.name]}:\n"
                f"            while (!{kd.c_type}_IsDone(&{kd.c_var})) {{}}\n"
                f"            break;"
            )

        table = "".join(
            f"    {{ {kd.rpu_lane}u, {kd.rpu_reg_words}u, &{kd.c_var}.Ctrl_BaseAddress }},"
            f"   /* {self._KERNEL_ID_ENUM[kd.name]} */\n" for kd in active)
        rpu_table = (
            "#ifdef __linux__\n"
            "/* The kernels for the Cortex-R5 recorder (inference_rpu.h), by kernel_id_t:\n"
            " * the firmware's lane, the argument registers to copy, the instance. */\n"
            "const inference_rpu_kernel_t *inference_rpu_kernels(unsigned *count)\n"
            "{\n"
            "    static const inference_rpu_kernel_t k[] = {\n"
            f"{table}"
            "    };\n"
            "    *count = (unsigned)(sizeof k / sizeof k[0]);\n"
            "    return k;\n"
            "}\n"
            "#endif\n\n")
        return (
            _banner("Kernel synchronization (split start / wait)") +
            rpu_table +
            "/*\n"
            " * kernel_wait(k) — block until the most-recent Start on lane k completes.\n"
            " *\n"
            " * The codegen emits NON-BLOCKING run_*() helpers; inference_run() calls\n"
            " * kernel_wait() lazily — only when the next op needs the result of an\n"
            " * earlier op, or when a lane must be reused.  This lets different lanes\n"
            " * (Conv ‖ Pool ‖ VectorOP ‖ Matmul) overlap without threading.\n"
            " *\n"
            " * Who waits, in this order:\n"
            " *   - the Cortex-R5, while it drives the kernels (inference_rpu.h): the\n"
            " *     wait is recorded for the firmware;\n"
            " *   - the kernel's interrupt (Linux, inference_irq.h): when the lane's\n"
            " *     Start armed it — INFERENCE_WAIT=irq, the default, and a UIO\n"
            " *     device with the kernel's line — the A53 spins briefly, then sleeps\n"
            " *     until the line fires;\n"
            " *   - otherwise IsDone() in a tight loop (bare metal, INFERENCE_WAIT=poll,\n"
            " *     the host emulation).\n"
            " * It is declared __weak so a deployment can still replace it at link time.\n"
            " */\n"
            + "\n".join(enum_lines) + "\n\n"
            + ("/* The row-ops node collecting a descriptor chain (run_rowops, RPU_PLAN §13.3),\n"
               " * or NULL: the waits between its jobs are the unit's, not the CPU's. */\n"
               "typedef struct ro_chain ro_chain_t;\n"
               "static ro_chain_t *s_ro_chain = NULL;\n\n" if self._ro_chains else "") +
            "__attribute__((weak))\n"
            "void kernel_wait(kernel_id_t k)\n"
            "{\n"
            + ("    if (k == KERNEL_ROWOPS && s_ro_chain) return;   /* a chained node: the unit orders its jobs */\n"
               if self._ro_chains else "") +
            "#ifdef __linux__\n"
            "    if (inference_rpu_active()) { inference_rpu_wait(k); return; }   /* the R5 waits */\n"
            "    if (inference_irq_wait(k)) return;                 /* the kernel's interrupt */\n"
            "#endif\n"
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

    def _ro_chain_helpers(self, ro_var: str) -> str:
        """The descriptor chains' C (RPU_PLAN §13.3, ROWOPS_RTL_KERNEL.md §1.6):
        the chain state, ro_chain_begin / _add / _end and one ro_chain_t per
        chained node (its region is set up by inference_init)."""
        decls = "".join(f"static ro_chain_t s_ro_ch_{i};   /* node [{i}]: {n} descriptors */\n"
                        for i, n in sorted(self._ro_chains.items()))
        return (
            "/*\n"
            " * Descriptor chains (RPU_PLAN §13.3, doc/kernels/ROWOPS_RTL_KERNEL.md §1.6).  A\n"
            " * row-ops node of several jobs runs between ro_chain_begin() and ro_chain_end():\n"
            " * run_rowops() writes each job as a 128-byte descriptor — words 0..23 the\n"
            " * argument registers 0x10..0x6C in order — into the node's region at the end of\n"
            " * the pool, kernel_wait(KERNEL_ROWOPS) inside the node returns at once, and the\n"
            " * jobs start as chains of at most 16 with one register block each: the unit\n"
            " * runs a chain's jobs in order (each sees the previous one's writes) and raises\n"
            " * ap_done once.  A node's region holds every job it can issue\n"
            " * (RowOpsNode.chain_jobs), so no descriptor is rewritten before its chain ran —\n"
            " * also when the Cortex-R5 replays the Starts later (inference_rpu.h).\n"
            " */\n"
            "#ifndef XROWOPSKERNEL_CTRL_ADDR_CHAIN_N_DATA\n"
            "#error \"chained row-ops nodes need the driver of a unit with descriptor chains "
            "(kernels.rowops.chains)\"\n"
            "#endif\n"
            "struct ro_chain {\n"
            "    inference_buf_t view;   /* the node's descriptors: a view of the pool */\n"
            "    unsigned        cap;    /* descriptors it holds */\n"
            "    unsigned        n;      /* written */\n"
            "    unsigned        first;  /* the first not started yet */\n"
            "    int             busy;   /* a chain of the node is in flight */\n"
            "};\n"
            + decls +
            "\n"
            "static void ro_chain_start(ro_chain_t *c)\n"
            "{\n"
            "    if (c->n == c->first) return;\n"
            "    if (c->busy) {                         /* the node's previous chain first */\n"
            "        s_ro_chain = NULL;\n"
            "        kernel_wait(KERNEL_ROWOPS);\n"
            "        s_ro_chain = c;\n"
            "    }\n"
            "    inference_buf_sync_to_device(&c->view);\n"
            f"    XRowopskernel_Set_chain(&{ro_var}, inference_buf_phys(&c->view) + 128u * (u64)c->first);\n"
            f"    XRowopskernel_Set_chain_n(&{ro_var}, c->n - c->first);\n"
            f"    XRowopskernel_Start(&{ro_var});\n"
            "    c->busy = 1;\n"
            "    c->first = c->n;\n"
            "}\n"
            "\n"
            "static void ro_chain_add(const uint32_t w[24])\n"
            "{\n"
            "    ro_chain_t *c = s_ro_chain;\n"
            "    uint32_t   *d;\n"
            "    unsigned    i;\n"
            "    if (c->n == c->cap) {                  /* past chain_jobs (never): start over */\n"
            "        ro_chain_start(c);\n"
            "        s_ro_chain = NULL;\n"
            "        kernel_wait(KERNEL_ROWOPS);\n"
            "        s_ro_chain = c;\n"
            "        c->n = c->first = 0u;\n"
            "        c->busy = 0;\n"
            "    }\n"
            "    d = (uint32_t *)(void *)inference_buf_ptr(&c->view) + 32u * c->n;\n"
            "    for (i = 0u; i < 24u; i++) d[i] = w[i];\n"
            "    c->n++;\n"
            "    if (c->n - c->first == 16u) ro_chain_start(c);   /* the unit runs at most 16 */\n"
            "}\n"
            "\n"
            "static void ro_chain_begin(ro_chain_t *c)\n"
            "{\n"
            "    c->n = c->first = 0u;\n"
            "    c->busy = 0;\n"
            "    s_ro_chain = c;\n"
            "}\n"
            "\n"
            "static void ro_chain_end(void)\n"
            "{\n"
            "    ro_chain_start(s_ro_chain);\n"
            "    s_ro_chain = NULL;\n"
            "}\n"
        )

    @staticmethod
    def _run_conv_at_fn(conv_var: str, pattern: bool) -> str:
        """run_conv_at() (dense y) or run_conv_at_y() (the y channel pattern's
        three registers as arguments, RPU_PLAN §13.2)."""
        if pattern:
            head = (
                "/*\n"
                " * run_conv_at_y() — run_conv_at() writing y through ConvKernel's y\n"
                " * channel pattern (RPU_PLAN §13.2): channel c lands at y_off +\n"
                " * (c mod y_cn) * y_cs + (c div y_cn) * y_gs elements, its out_h * out_w\n"
                " * pixels contiguous (y_cn = 0: one group).  Batch 1, y_cn a multiple\n"
                " * of 16.  The attention P.V calls write the merged tensor this way.\n"
                " */\n"
                "#ifndef XCONVKERNEL_CTRL_ADDR_Y_CS_DATA\n"
                "#error \"run_conv_at_y needs the driver of a ConvKernel with the y channel pattern "
                "(kernels.conv.y_pattern)\"\n"
                "#endif\n"
                "static void run_conv_at_y(\n")
            tail_args = ("    unsigned has_bias, unsigned is_depthwise,\n"
                         "    unsigned y_cs, unsigned y_cn, unsigned y_gs)\n")
            pat = (f"    XConvkernel_Set_y_cs        (&{conv_var}, y_cs);\n"
                   f"    XConvkernel_Set_y_cn        (&{conv_var}, y_cn);\n"
                   f"    XConvkernel_Set_y_gs        (&{conv_var}, y_gs);\n")
        else:
            head = (
                "/*\n"
                " * run_conv_at() — run_conv() on sub-buffers: x / weight / y start at\n"
                " * element offsets x_off / w_off / y_off of their buffers (no bias).\n"
                " * Used by MatMuls lowered onto ConvKernel whose batch items each\n"
                " * need their own weights (attention: one call per head); every\n"
                " * offset is a multiple of INFERENCE_ALIGN_ELEMS (16-byte aligned).\n"
                " *\n"
                " * NON-BLOCKING like run_conv(): the caller waits on KERNEL_CONV\n"
                " * before the next call reprograms the registers.\n"
                " */\n"
                "static void run_conv_at(\n")
            tail_args = "    unsigned has_bias, unsigned is_depthwise)\n"
            pat = (f"#ifdef XCONVKERNEL_CTRL_ADDR_Y_CS_DATA   /* the y pattern (RPU_PLAN §13.2): dense */\n"
                   f"    XConvkernel_Set_y_cs        (&{conv_var}, 0u);\n"
                   f"    XConvkernel_Set_y_cn        (&{conv_var}, 0u);\n"
                   f"    XConvkernel_Set_y_gs        (&{conv_var}, 0u);\n"
                   "#endif\n")
        return (
            head +
            "    inference_buf_t *x,      unsigned x_off,\n"
            "    inference_buf_t *weight, unsigned w_off,\n"
            "    inference_buf_t *y,      unsigned y_off,\n"
            "    unsigned batch,\n"
            "    unsigned in_ch,  unsigned in_h,  unsigned in_w,\n"
            "    unsigned out_ch, unsigned out_h, unsigned out_w,\n"
            "    unsigned kh,     unsigned kw,\n"
            "    unsigned stride_h, unsigned stride_w,\n"
            "    unsigned dilation_h, unsigned dilation_w,\n"
            "    unsigned pad_top, unsigned pad_left,\n"
            + tail_args +
            "{\n"
            f"    XConvkernel_Set_x           (&{conv_var},\n"
            "        inference_buf_phys(x) + (uint64_t)x_off * INFERENCE_BYTES_PER_ELEM);\n"
            f"    XConvkernel_Set_weight      (&{conv_var},\n"
            "        inference_buf_phys(weight) + (uint64_t)w_off * INFERENCE_BYTES_PER_ELEM);\n"
            f"    XConvkernel_Set_bias        (&{conv_var}, (u64)0);\n"
            f"    XConvkernel_Set_y           (&{conv_var},\n"
            "        inference_buf_phys(y) + (uint64_t)y_off * INFERENCE_BYTES_PER_ELEM);\n"
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
            + pat +
            f"    XConvkernel_Start(&{conv_var});\n"
            "}\n"
        )

    def _run_op_helper(self) -> str:
        nodes          = self._graph.nodes
        # All VectorOP ScheduledNodes use run_op() (broadcast via outer/inc params)
        need_run_op    = self._has_vectorop_nodes
        # run_op_act(): fused activations, and the LLM nodes' activation calls
        need_run_op_act = any(
            (isinstance(sn, ScheduledNode) and not sn.uses_run_op)
            or (isinstance(sn, LlmKernelNode) and sn.uses_activation_unit) for sn in nodes
        )
        need_act_unit = self._uses_activation_unit
        need_run_matmul = self._has_matmul_nodes
        need_run_matmul_at = any(
            isinstance(sn, MatmulNode) and sn.outer_count > 1
            for sn in nodes
        )
        # run_conv: ConvNodes and single-call MatMuls on ConvKernel;
        # run_conv_at: MatMuls on ConvKernel issued as one call per batch item.
        need_run_conv = self._has_conv_nodes and any(
            not (isinstance(sn, MatmulConvNode) and sn.calls > 1)
            and not isinstance(sn, LlmAttnConvNode)
            and not (isinstance(sn, ConvNode) and sn.sliced)
            for sn in nodes if sn.kernel_name == "ConvKernel"
        )
        # run_conv_dw_at: depthwise Convs issued as channel-slice calls
        need_run_conv_dw = any(isinstance(sn, ConvNode) and sn.sliced for sn in nodes)
        need_run_conv_at = any(
            (isinstance(sn, MatmulConvNode) and sn.calls > 1)
            or (isinstance(sn, LlmAttnConvNode) and sn.y_pat is None) for sn in nodes
        )
        # run_conv_at_y: the P.V calls that write the merged tensor through the
        # y channel pattern (CodeGenerator._merge_in_place, RPU_PLAN §13.2)
        need_run_conv_at_y = any(isinstance(sn, LlmAttnConvNode) and sn.y_pat is not None
                                 for sn in nodes)
        need_run_pool = self._has_pool_nodes
        need_run_rowops = self._has_rowops_nodes

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
        conv_helpers = [n for n, need in (("run_conv()", need_run_conv), ("run_conv_at()", need_run_conv_at),
                                          ("run_conv_at_y()", need_run_conv_at_y),
                                          ("run_conv_dw_at()", need_run_conv_dw)) if need]
        if conv_helpers:
            titles.append(" / ".join(conv_helpers) + " — ConvKernel dispatch helper"
                          + ("s" if len(conv_helpers) > 1 else ""))
        if need_run_pool:
            titles.append("run_pool() — PoolingKernel dispatch helper")
        if need_run_rowops:
            titles.append("run_rowops() — RowOpsKernel dispatch helper")
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

        # The alpha register (LeakyReLU's slope) exists only in IPs with the
        # activation unit; every program writes it (a register keeps the last
        # program's value).  A program that uses the unit writes it plainly —
        # with an older driver it does not compile, and inference_init()
        # checks the IP — the others guarded, so the older driver compiles.
        alpha_reg = ("#ifdef XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA\n"
                     f"    XVectoropkernel_Set_alpha(&{vop_var}, 0u);\n"
                     "#endif\n")
        alpha_arg = (f"    XVectoropkernel_Set_alpha(&{vop_var}, alpha);\n" if need_act_unit
                     else "#ifdef XVECTOROPKERNEL_CTRL_ADDR_ALPHA_DATA\n"
                          f"    XVectoropkernel_Set_alpha(&{vop_var}, alpha);\n"
                          "#else\n"
                          "    (void)alpha;\n"
                          "#endif\n")
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
                " *   b      input buffer B  (NULL for unary ops: op >= RELU)\n"
                " *   c      output buffer C (must not alias a or b)\n"
                " *   size   elements per inner chunk\n"
                " *   op     VECTOROP_* constant\n"
                " *   outer  number of outer broadcasting iterations (1 = no broadcast)\n"
                " *   a_inc  element stride for a per outer step (0 = a repeats)\n"
                " *   b_inc  element stride for b per outer step (0 = b repeats)\n"
                " *\n"
                " * run_op writes act = 0 and alpha = 0 (no fused activation; a register\n"
                " * keeps its last value across runs, so they are written every call).\n"
                " * A node with a fused activation or a LeakyReLU slope uses run_op_act,\n"
                " * which takes the act code and alpha.\n"
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
                + alpha_reg +
                f"    XVectoropkernel_Start(&{vop_var});\n"
                "}\n"
            )

        if need_run_op_act:
            parts.append(
                "/*\n"
                " * run_op_act() — run_op() with a fused activation: the kernel applies\n"
                " * act (VECTOROP_ACT_*) to the op result before writing c, so\n"
                " * Add -> Relu / Gelu is one pass over the data; alpha is LeakyReLU's\n"
                " * slope (alpha / 65536) for VECTOROP_LEAKY_RELU / VECTOROP_ACT_LEAKY_RELU.\n"
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
                "    unsigned         act,\n"
                "    unsigned         alpha)\n"
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
                + alpha_arg +
                f"    XVectoropkernel_Start(&{vop_var});\n"
                "}\n"
            )

        if self._uses_softmax_unit:
            parts.append(
                "/*\n"
                " * run_softmax() — VectorOPKernel's softmax unit (doc/plans/SOFTMAX_PLAN.md):\n"
                " * op VECTOROP_SOFTMAX (row mode: outer rows of size elements, input row\n"
                " * stride a_inc, output row stride b_inc) or VECTOROP_SOFTMAX_T (column\n"
                " * mode: input s[size keys][outer queries] at row stride a_inc -> P[outer]\n"
                " * [size] at row stride b_inc; outer a multiple of 16), the integer\n"
                " * specification of src/vectorop_smx.py with smx_cm (Cm), smx_cfg (Cs [5:0],\n"
                " * f_p [12:8]) and smx_mask (vector q valid over min(size, valid0 + q mod\n"
                " * period) elements: valid0 [15:0], period [31:16]).  a_off / c_off are\n"
                " * element offsets into a / c (16-byte aligned).  NON-BLOCKING, as run_op.\n"
                " */\n"
                "static void run_softmax(\n"
                "    inference_buf_t *a,\n"
                "    unsigned         a_off,\n"
                "    inference_buf_t *c,\n"
                "    unsigned         c_off,\n"
                "    unsigned         size,\n"
                "    unsigned         op,\n"
                "    unsigned         outer,\n"
                "    unsigned         a_inc,\n"
                "    unsigned         b_inc,\n"
                "    unsigned         cm,\n"
                "    unsigned         cfg,\n"
                "    unsigned         mask)\n"
                "{\n"
                f"    XVectoropkernel_Set_a(&{vop_var}, inference_buf_phys(a) + (u64)a_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XVectoropkernel_Set_b(&{vop_var}, (u64)0);\n"
                f"    XVectoropkernel_Set_c(&{vop_var}, inference_buf_phys(c) + (u64)c_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XVectoropkernel_Set_size(&{vop_var}, size);\n"
                f"    XVectoropkernel_Set_op(&{vop_var}, op);\n"
                f"    XVectoropkernel_Set_outer(&{vop_var}, outer);\n"
                f"    XVectoropkernel_Set_a_inc(&{vop_var}, a_inc);\n"
                f"    XVectoropkernel_Set_b_inc(&{vop_var}, b_inc);\n"
                f"    XVectoropkernel_Set_smx_cm(&{vop_var}, cm);\n"
                f"    XVectoropkernel_Set_smx_cfg(&{vop_var}, cfg);\n"
                f"    XVectoropkernel_Set_smx_mask(&{vop_var}, mask);\n"
                f"    XVectoropkernel_Start(&{vop_var});\n"
                "}\n"
            )

        if need_run_rowops:
            from ..kernels import KERNEL_REGISTRY
            ro_var = KERNEL_REGISTRY["RowOpsKernel"].c_var
            if self._ro_chains:
                parts.append(self._ro_chain_helpers(ro_var))
            parts.append(
                "/*\n"
                " * run_rowops() — the row-ops unit (doc/kernels/ROWOPS_RTL_KERNEL.md): every\n"
                " * register of one job — the op word (op [3:0], the kinds of A / B / C, the\n"
                " * flags), rows x blocks rows of n elements, the byte strides, the parameter\n"
                " * block p, eps, aux (ROPE's HD / TRANSPOSE's R), bq_sh and COPY / ADDX's row\n"
                " * window.  a_off / b_off / c_off are BYTE offsets; a NULL buffer programs\n"
                " * address 0 (a stream the op does not read).  NON-BLOCKING, as run_op.\n"
                " */\n"
                "static void run_rowops(\n"
                "    inference_buf_t *a, unsigned a_off,\n"
                "    inference_buf_t *b, unsigned b_off,\n"
                "    inference_buf_t *c, unsigned c_off,\n"
                "    inference_buf_t *p,\n"
                "    unsigned op, unsigned n, unsigned rows, unsigned blocks,\n"
                "    unsigned a_s1, unsigned a_s2, unsigned b_s1, unsigned b_s2,\n"
                "    unsigned c_s1, unsigned c_s2,\n"
                "    uint64_t eps, unsigned aux, unsigned bq_sh, unsigned win_lo, unsigned win_hi)\n"
                "{\n"
                + ("    if (s_ro_chain) {                  /* a chained node: one descriptor (ro_chain_add) */\n"
                   "        const u64 aa = a ? inference_buf_phys(a) + (u64)a_off : (u64)0;\n"
                   "        const u64 ba = b ? inference_buf_phys(b) + (u64)b_off : (u64)0;\n"
                   "        const u64 ca = c ? inference_buf_phys(c) + (u64)c_off : (u64)0;\n"
                   "        const u64 pa = p ? inference_buf_phys(p) : (u64)0;\n"
                   "        const uint32_t w[24] = {\n"
                   "            op, n, rows, blocks,\n"
                   "            (uint32_t)aa, (uint32_t)(aa >> 32), a_s1, a_s2,\n"
                   "            (uint32_t)ba, (uint32_t)(ba >> 32), b_s1, b_s2,\n"
                   "            (uint32_t)ca, (uint32_t)(ca >> 32), c_s1, c_s2,\n"
                   "            (uint32_t)pa, (uint32_t)(pa >> 32), (uint32_t)eps, (uint32_t)(eps >> 32),\n"
                   "            aux, bq_sh, win_lo, win_hi };\n"
                   "        ro_chain_add(w);\n"
                   "        return;\n"
                   "    }\n" if self._ro_chains else "") +
                f"    XRowopskernel_Set_op(&{ro_var}, op);\n"
                f"    XRowopskernel_Set_n(&{ro_var}, n);\n"
                f"    XRowopskernel_Set_rows(&{ro_var}, rows);\n"
                f"    XRowopskernel_Set_blocks(&{ro_var}, blocks);\n"
                f"    XRowopskernel_Set_a(&{ro_var}, a ? inference_buf_phys(a) + (u64)a_off : (u64)0);\n"
                f"    XRowopskernel_Set_a_s1(&{ro_var}, a_s1);\n"
                f"    XRowopskernel_Set_a_s2(&{ro_var}, a_s2);\n"
                f"    XRowopskernel_Set_b(&{ro_var}, b ? inference_buf_phys(b) + (u64)b_off : (u64)0);\n"
                f"    XRowopskernel_Set_b_s1(&{ro_var}, b_s1);\n"
                f"    XRowopskernel_Set_b_s2(&{ro_var}, b_s2);\n"
                f"    XRowopskernel_Set_c(&{ro_var}, c ? inference_buf_phys(c) + (u64)c_off : (u64)0);\n"
                f"    XRowopskernel_Set_c_s1(&{ro_var}, c_s1);\n"
                f"    XRowopskernel_Set_c_s2(&{ro_var}, c_s2);\n"
                f"    XRowopskernel_Set_p(&{ro_var}, p ? inference_buf_phys(p) : (u64)0);\n"
                f"    XRowopskernel_Set_eps(&{ro_var}, eps);\n"
                f"    XRowopskernel_Set_aux(&{ro_var}, aux);\n"
                f"    XRowopskernel_Set_bq_sh(&{ro_var}, bq_sh);\n"
                f"    XRowopskernel_Set_win_lo(&{ro_var}, win_lo);\n"
                f"    XRowopskernel_Set_win_hi(&{ro_var}, win_hi);\n"
                "#ifdef XROWOPSKERNEL_CTRL_ADDR_CHAIN_N_DATA   /* one job, not a chain (RPU_PLAN §13.3) */\n"
                f"    XRowopskernel_Set_chain(&{ro_var}, (u64)0);\n"
                f"    XRowopskernel_Set_chain_n(&{ro_var}, 0u);\n"
                "#endif\n"
                f"    XRowopskernel_Start(&{ro_var});\n"
                "}\n"
            )
            from ..rowops_nodes import ATTN_PREP_C, AttnPrepRoNode
            if any(isinstance(sn, AttnPrepRoNode) for sn in nodes):
                parts.append(ATTN_PREP_C)

        # GEMV registers (MatmulKernel.h "GEMV streaming mode"): written on
        # every call — they persist across calls — when the platform's
        # kernel has them; a_to_b lets port a reach its half of B.
        def gemv_regs(a_addr: str, b_addr: str) -> str:
            if MATMUL_GEMV_MAX_M <= 0:
                return "    (void)gemv_kw;\n"
            return (f"    XMatmulkernel_Set_gemv_kw(&{mm_var}, gemv_kw);\n"
                    f"    XMatmulkernel_Set_a_to_b(&{mm_var},\n"
                    f"        (uint64_t)(({b_addr}) - ({a_addr})));\n")

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
                " *   gemv_kw     0: the tiled path; 1 / 2 / 4 / 8: the GEMV streaming\n"
                " *                path, b in the ConvKernel x image of that kernel\n"
                " *                width (1: row-major [k][m])\n"
                " */\n"
                "static void run_matmul(\n"
                "    inference_buf_t *a,\n"
                "    inference_buf_t *b,\n"
                "    inference_buf_t *c,\n"
                "    uint32_t n, uint32_t k, uint32_t m, uint32_t batch,\n"
                "    uint32_t a_stride, uint32_t b_stride, uint32_t c_stride,\n"
                "    uint32_t b_packed, uint32_t gemv_kw)\n"
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
                + gemv_regs("inference_buf_phys(a)", "inference_buf_phys(b)")
                + f"    XMatmulkernel_Start(&{mm_var});\n"
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
                "    uint32_t b_packed, uint32_t gemv_kw)\n"
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
                + gemv_regs("inference_buf_phys(a) + (uint64_t)a_off * INFERENCE_BYTES_PER_ELEM",
                            "inference_buf_phys(b) + (uint64_t)b_off * INFERENCE_BYTES_PER_ELEM")
                + f"    XMatmulkernel_Start(&{mm_var});\n"
                f"    kernel_wait(KERNEL_MATMUL);\n"
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
                f"#ifdef XCONVKERNEL_CTRL_ADDR_Y_CS_DATA   /* the y pattern (RPU_PLAN §13.2): dense */\n"
                f"    XConvkernel_Set_y_cs        (&{conv_var}, 0u);\n"
                f"    XConvkernel_Set_y_cn        (&{conv_var}, 0u);\n"
                f"    XConvkernel_Set_y_gs        (&{conv_var}, 0u);\n"
                "#endif\n"
                f"    XConvkernel_Start(&{conv_var});\n"
                "}\n"
            )

        if need_run_conv_at:
            parts.append(self._run_conv_at_fn(conv_var, False))
        if need_run_conv_at_y:
            parts.append(self._run_conv_at_fn(conv_var, True))

        if need_run_conv_dw:
            parts.append(
                "/*\n"
                " * run_conv_dw_at() — one channel slice of a depthwise conv (batch 1):\n"
                " * x / weight / bias / y start at element offsets x_off / w_off / b_off /\n"
                " * y_off of their buffers (NCHW: each slice of x and y is contiguous;\n"
                " * the weights are [C][roundup(kh*kw, 8)], the bias [roundup(C, 8)]),\n"
                " * ch channels.  A depthwise job of fewer channels gets taller output-row\n"
                " * chunks (cost_model.dw_slice); every offset is 16-byte aligned.\n"
                " *\n"
                " * NON-BLOCKING like run_conv(): the caller waits on KERNEL_CONV\n"
                " * before the next call reprograms the registers.\n"
                " */\n"
                "static void run_conv_dw_at(\n"
                "    inference_buf_t *x,      unsigned x_off,\n"
                "    inference_buf_t *weight, unsigned w_off,\n"
                "    inference_buf_t *bias,   unsigned b_off,\n"
                "    inference_buf_t *y,      unsigned y_off,\n"
                "    unsigned ch,\n"
                "    unsigned in_h,  unsigned in_w,  unsigned out_h, unsigned out_w,\n"
                "    unsigned kh,    unsigned kw,\n"
                "    unsigned stride_h, unsigned stride_w,\n"
                "    unsigned dilation_h, unsigned dilation_w,\n"
                "    unsigned pad_top, unsigned pad_left,\n"
                "    unsigned has_bias)\n"
                "{\n"
                f"    XConvkernel_Set_x           (&{conv_var},\n"
                "        inference_buf_phys(x) + (uint64_t)x_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XConvkernel_Set_weight      (&{conv_var},\n"
                "        inference_buf_phys(weight) + (uint64_t)w_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XConvkernel_Set_bias        (&{conv_var}, bias ?\n"
                "        inference_buf_phys(bias) + (uint64_t)b_off * INFERENCE_BYTES_PER_ELEM : (u64)0);\n"
                f"    XConvkernel_Set_y           (&{conv_var},\n"
                "        inference_buf_phys(y) + (uint64_t)y_off * INFERENCE_BYTES_PER_ELEM);\n"
                f"    XConvkernel_Set_batch       (&{conv_var}, 1u);\n"
                f"    XConvkernel_Set_in_ch       (&{conv_var}, ch);\n"
                f"    XConvkernel_Set_in_h        (&{conv_var}, in_h);\n"
                f"    XConvkernel_Set_in_w        (&{conv_var}, in_w);\n"
                f"    XConvkernel_Set_out_ch      (&{conv_var}, ch);\n"
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
                f"    XConvkernel_Set_is_depthwise(&{conv_var}, 1u);\n"
                f"#ifdef XCONVKERNEL_CTRL_ADDR_Y_CS_DATA   /* the y pattern (RPU_PLAN §13.2): dense */\n"
                f"    XConvkernel_Set_y_cs        (&{conv_var}, 0u);\n"
                f"    XConvkernel_Set_y_cn        (&{conv_var}, 0u);\n"
                f"    XConvkernel_Set_y_gs        (&{conv_var}, 0u);\n"
                "#endif\n"
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
        compact = self._use_compact_init()
        slot_tables: List[str] = []            # file-scope descriptor tables (compact form)

        if need_pool:
            alloc_lines.append(
                "    /* One contiguous DMA allocation covers all weights and intermediate buffers.\n"
                "     * Each sub-buffer is an aligned view; 64-byte gaps between slots ensure\n"
                "     * cache-line alignment at every physical base address. */"
            )
            desc_lines = []
            if self._ro_chains:                  # the descriptor area after the laid-out pool
                bpe = self._dtype.bytes_per_elem
                off = -(-total_pool_elems * bpe // 128) * 128 // bpe
                desc_lines.append("    /* RowOpsKernel descriptor chains (RPU_PLAN §13.3): each chained node's\n"
                                  "     * region at the end of the pool, 128 bytes per descriptor */")
                for i, n in sorted(self._ro_chains.items()):
                    desc_lines.append(f"    inference_buf_init_view(&s_ro_ch_{i}.view, s_alloc_pool, {off}u, "
                                      f"{n * 128 // bpe}u);")
                    desc_lines.append(f"    s_ro_ch_{i}.cap = {n}u;")
                    off += n * 128 // bpe
                total_pool_elems = off
                desc_lines.append("")
            alloc_lines.append(
                f"    s_alloc_pool = inference_buf_alloc({total_pool_elems}u);"
            )
            alloc_lines.append(
                "    if (!s_alloc_pool) { rc = -1; goto fail; }"
            )
            alloc_lines.append("")
            alloc_lines += desc_lines

        if weights and compact:
            rows = []
            for t in weights:
                off, alloc = pool_map[t.onnx_name]
                if t.onnx_name in external:
                    rows.append(f"    {{ &_s_buf_{t.c_name}, &{t.c_name}, {off}u, {alloc}u, NULL, 0u,"
                                f" \"{t.c_name}\", {t.numel}u }},")
                else:
                    rows.append(f"    {{ &_s_buf_{t.c_name}, &{t.c_name}, {off}u, {alloc}u,"
                                f" _rom_{t.c_name}, (unsigned)sizeof(_rom_{t.c_name}), NULL, 0u }},")
            slot_tables.append(f"static const _pool_slot_t _s_weight_slots[{len(rows)}] = {{\n"
                               + "\n".join(rows) + "\n};")
            load = ("            else if (_load_weight(s->view, s->name, s->n_elem) != 0) "
                    "{ rc = -1; goto fail; }\n" if external else "")
            alloc_lines.append(
                "    /* Weights: one table-driven loop (thousands of straight-line views make\n"
                "     * the compiler's register allocator explode on these projects) */\n"
                "    {\n"
                "        unsigned _i;\n"
                f"        for (_i = 0u; _i < {len(rows)}u; _i++) {{\n"
                "            const _pool_slot_t *s = &_s_weight_slots[_i];\n"
                "            inference_buf_init_view(s->view, s_alloc_pool, s->off, s->count);\n"
                "            *s->var = s->view;\n"
                "            if (s->rom)\n"
                "                memcpy(inference_buf_ptr(s->view), s->rom, s->rom_bytes);\n"
                + load +
                "        }\n"
                "    }")
            alloc_lines.append("")
            alloc_lines.append("    inference_buf_sync_to_device(s_alloc_pool);")
            alloc_lines.append("")
        elif weights:
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

            if non_alias and compact:
                rows = []
                for t in non_alias:
                    off, alloc = pool_map[t.onnx_name]
                    rows.append(f"    {{ &_s_buf_{t.c_name}, &{t.c_name}, {off}u, {alloc}u,"
                                f" NULL, 0u, NULL, 0u }},")
                slot_tables.append(f"static const _pool_slot_t _s_inter_slots[{len(rows)}] = {{\n"
                                   + "\n".join(rows) + "\n};")
                alloc_lines.append(
                    "    /* Intermediate buffers (table-driven) */\n"
                    "    {\n"
                    "        unsigned _i;\n"
                    f"        for (_i = 0u; _i < {len(rows)}u; _i++) {{\n"
                    "            const _pool_slot_t *s = &_s_inter_slots[_i];\n"
                    "            inference_buf_init_view(s->view, s_alloc_pool, s->off, s->count);\n"
                    "            *s->var = s->view;\n"
                    "        }\n"
                    "    }")
                alloc_lines.append("")
            elif non_alias:
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

        host_layout, host_bytes = self._compute_host_layout()
        if host_layout:
            alloc_lines.append(f"    /* Host-memory intermediates ({host_bytes} B, slots reused by"
                               f" liveness) */")
            alloc_lines.append(f"    s_host_arena = (unsigned char *)malloc({host_bytes}u);")
            alloc_lines.append("    if (!s_host_arena) { rc = -1; goto fail; }")
            by_name = {t.onnx_name: t for t in self._graph.host_tensors}
            for name, off, _nb in host_layout:
                t = by_name[name]
                alloc_lines.append(f"    {t.c_name} = ({self._host_c_type(t)} *)"
                                   f"(s_host_arena + {off}u);")
            alloc_lines.append("")
        states = [t for t in self._graph.state_tensors if t.is_host]
        if states:
            alloc_lines.append("    /* Persistent states: zeroed, then their initial non-zero prefix */")
            for t in states:
                ct = self._host_c_type(t)
                alloc_lines.append(f"    {t.c_name} = ({ct} *)calloc({t.numel}u, sizeof({ct}));")
                alloc_lines.append(f"    if (!{t.c_name}) {{ rc = -1; goto fail; }}")
                pre = self._state_init_prefix(t)
                if pre is not None and t.group_layout:
                    G, D = t.group_layout
                    alloc_lines.append(f"    _state_scatter({t.c_name}, _state_init_{t.c_name},"
                                       f" {pre[1] // (G * D)}u, {t.shape[0]}u, {G}u, {D}u,"
                                       f" {t.group_kw}u);")
                elif pre is not None:
                    alloc_lines.append(f"    memcpy({t.c_name}, _state_init_{t.c_name},"
                                       f" sizeof _state_init_{t.c_name});")
            alloc_lines.append("")

        if self._host_nodes:
            alloc_lines.append("    /* Cached staging arena for the host-CPU ops */")
            alloc_lines.append(
                f"    s_host_stage = (Data_t *)malloc({self._host_stage_elems}u"
                f" * INFERENCE_BYTES_PER_ELEM);"
            )
            alloc_lines.append("    if (!s_host_stage) { rc = -1; goto fail; }")
            alloc_lines.append("")
            alloc_lines.append("    /* Host-op worker threads + lookup tables */")
            alloc_lines.append("    if (host_runtime_init() != 0) { rc = -1; goto fail; }")
            alloc_lines.append("")

        dma_states = self._dma_states
        if dma_states:
            rows = []
            for t in dma_states:
                off, alloc = pool_map[t.onnx_name]
                pre = self._state_init_prefix(t)
                G, D = t.group_layout or (0, 0)
                init = f"_state_init_{t.c_name}" if pre is not None else "NULL"
                n_init = (pre[1] // (G * D) if G else pre[1]) if pre is not None else 0
                rows.append(f"    {{ &_s_buf_{t.c_name}, &{t.c_name}, {off}u, {alloc}u, {init},"
                            f" {n_init}u, {t.shape[0]}u, {G}u, {D}u, {t.group_kw}u }},")
            slot_tables.append(
                "typedef struct {\n"
                "    inference_buf_t  *view;\n"
                "    inference_buf_t **var;\n"
                "    unsigned          off, count;\n"
                "    const int16_t    *init;        /* logical non-zero prefix, or NULL */\n"
                "    unsigned          n_init;      /* its rows (group-major) / elements */\n"
                "    unsigned          R, G, D, K;  /* group-major [G][R][D] (rows x K); G = 0: flat */\n"
                "} _state_slot_t;\n\n"
                f"static const _state_slot_t _s_state_slots[{len(rows)}] = {{\n"
                + "\n".join(rows) + "\n};")
            state_lines = [
                "    /* DMA states: zeroed, then their initial value (the KV caches' sink row) */",
                "    {",
                "        unsigned _i;",
                f"        for (_i = 0u; _i < {len(rows)}u; _i++) {{",
                "            const _state_slot_t *s = &_s_state_slots[_i];",
                "            Data_t *p;",
                "            inference_buf_init_view(s->view, s_alloc_pool, s->off, s->count);",
                "            *s->var = s->view;",
                "            p = inference_buf_ptr(s->view);",
                "            memset(p, 0, (size_t)s->count * INFERENCE_BYTES_PER_ELEM);",
                *(["            if (s->init && s->G)",
                   "                _state_scatter((int16_t *)p, s->init, s->n_init, s->R, s->G, s->D, s->K);",
                   "            else if (s->init)"]
                  if any(t.group_layout for t in dma_states) else
                  ["            if (s->init)"]),
                "                memcpy(p, s->init, (size_t)s->n_init * INFERENCE_BYTES_PER_ELEM);",
                "        }",
                "    }",
                ""]
            sync = "    inference_buf_sync_to_device(s_alloc_pool);"
            if sync in alloc_lines:
                i = alloc_lines.index(sync)
                alloc_lines[i:i] = state_lines
            else:
                i = alloc_lines.index("    if (!s_alloc_pool) { rc = -1; goto fail; }") + 1
                alloc_lines[i:i] = [""] + state_lines + [sync]
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
        for t in self._graph.state_tensors:
            if t.is_host:
                deinit_free.append(f"    free({t.c_name}); {t.c_name} = NULL;")
        if self._dma_states:
            deinit_free.append(
                "    { unsigned _i; for (_i = 0u; _i < sizeof _s_state_slots / sizeof _s_state_slots[0];"
                " _i++) *_s_state_slots[_i].var = NULL; }")
        if self._graph.host_tensors:
            for t in self._graph.host_tensors:
                deinit_free.append(f"    {t.c_name} = NULL;")
            deinit_free.append("    free(s_host_arena); s_host_arena = NULL;")
        tabled = set()
        if compact:
            for tab in ("_s_weight_slots", "_s_inter_slots"):
                if any(tab in t for t in slot_tables):
                    deinit_free.append(
                        f"    {{ unsigned _i; for (_i = 0u; _i < sizeof {tab} / sizeof {tab}[0];"
                        f" _i++) *{tab}[_i].var = NULL; }}")
            tabled = ({t.onnx_name for t in weights}
                      | {t.onnx_name for t in intermediates
                         if t.onnx_name not in reshape_aliases and t.onnx_name not in views})
        for t in weights + intermediates:
            if t.onnx_name in tabled:
                continue
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
        # (Linux: the UIO names); on bare metal the AXI-Lite base addresses
        active = self._active_kernels
        params_decl = ",\n".join(f"    const char *{kd.init_param}" for kd in active)
        params_decl_bm = ",\n".join(f"    UINTPTR {kd.init_param}" for kd in active)
        cfg_inits = "".join(
            f"static int _inference_cfg_init_{kd.driver_prefix}({kd.c_type} *inst, UINTPTR base)\n"
            "{\n"
            f"    static {kd.c_type}_Config cfg;\n"
            "    cfg.Ctrl_BaseAddress = base;\n"
            f"    return {kd.c_type}_CfgInitialize(inst, &cfg);\n"
            "}\n\n" for kd in active)

        # One Initialize call per active kernel
        init_calls = []
        for kd in active:
            probe = ""
            if kd.c_type == "XVectoropkernel" and self._uses_softmax_unit:
                # An IP without the softmax unit has no smx registers (they read
                # 0) and would pass ops 10 / 11 through as copies.
                probe += (
                    "    /* The softmax runs in VectorOPKernel's softmax unit\n"
                    "     * (doc/plans/SOFTMAX_PLAN.md): an IP without it has no smx_cm\n"
                    "     * register and would copy instead. */\n"
                    f"    XVectoropkernel_Set_smx_cm(&{kd.c_var}, 0x5A5A5Au);\n"
                    f"    if (XVectoropkernel_Get_smx_cm(&{kd.c_var}) != 0x5A5A5Au) {{\n"
                    "        INFERENCE_ERR(\"inference_init: the VectorOPKernel has no softmax \"\n"
                    "                \"unit (the bitstream predates it)\\n\");\n"
                    "        rc = -1;\n"
                    "        goto fail;\n"
                    "    }\n"
                )
            if kd.c_type == "XVectoropkernel" and self._uses_activation_unit:
                # An IP without the activation unit has no alpha register (it
                # reads 0) and would pass the activation ops through unchanged.
                probe += (
                    "    /* LeakyReLU / SiLU / GELU run in VectorOPKernel's activation unit\n"
                    "     * (doc/plans/ACTIVATIONS_PLAN.md): an IP without it has no alpha\n"
                    "     * register and would pass these ops through unchanged. */\n"
                    f"    XVectoropkernel_Set_alpha(&{kd.c_var}, 0x5A5Au);\n"
                    f"    if (XVectoropkernel_Get_alpha(&{kd.c_var}) != 0x5A5Au) {{\n"
                    "        INFERENCE_ERR(\"inference_init: the VectorOPKernel has no activation \"\n"
                    "                \"unit (the bitstream predates it)\\n\");\n"
                    "        rc = -1;\n"
                    "        goto fail;\n"
                    "    }\n"
                )
            init_calls.append(
                f"    /* Initialise {kd.name} driver (UIO: {kd.uio_default}) */\n"
                "#ifdef __linux__\n"
                f"    rc = {kd.c_type}_Initialize(&{kd.c_var}, {kd.init_param});\n"
                "#else\n"
                f"    rc = _inference_cfg_init_{kd.driver_prefix}(&{kd.c_var}, {kd.init_param});\n"
                "#endif\n"
                "    if (rc != 0) goto fail;\n" + probe +
                "#ifdef __linux__\n"
                "    /* its interrupt, for kernel_wait() (inference_irq.h; without one it polls) */\n"
                f"    (void)inference_irq_open({self._KERNEL_ID_ENUM[kd.name]}, {kd.init_param},\n"
                f"                             (void *)(uintptr_t){kd.c_var}.Ctrl_BaseAddress);\n"
                "#endif\n"
            )
        if self._has_rowops_nodes:
            from ..rowops_nodes import TABLES_NAME
            from ..rowops_spec import OP_TLOAD, K_I32, opword
            tb = self._graph._tensors[TABLES_NAME].c_name
            init_calls.append(
                "    /* RowOpsKernel: its sigmoid / rsqrt tables live in UltraRAM, which has no\n"
                "     * initial contents — load the 64 KiB image once (OP_TLOAD, ROWOPS_RTL_KERNEL.md\n"
                "     * §1.4) before any job. */\n"
                f"    run_rowops({tb}, 0u, NULL, 0u, NULL, 0u, NULL,\n"
                f"               0x{opword(OP_TLOAD, K_I32, K_I32, K_I32):05X}u, 4096u, 4u, 1u, "
                "16384u, 65536u, 0u, 0u, 0u, 0u, 0ull, 0u, 0u, 0u, 0xFFFFFFFFu);\n"
                "    kernel_wait(KERNEL_ROWOPS);\n")
        init_calls_str = "\n".join(init_calls)

        tables = ""
        if slot_tables:
            tables = (
                "/* Pool views of the weights and intermediates: one descriptor per buffer,\n"
                " * set up by a loop in inference_init() (compact form, see\n"
                " * _use_compact_init). */\n"
                "typedef struct {\n"
                "    inference_buf_t  *view;\n"
                "    inference_buf_t **var;\n"
                "    unsigned          off, count;\n"
                "    const void       *rom;         /* inline weight image, or NULL */\n"
                "    unsigned          rom_bytes;\n"
                "    const char       *name;        /* weights/<name>.dat when rom == NULL */\n"
                "    unsigned          n_elem;\n"
                "} _pool_slot_t;\n\n" + "\n\n".join(slot_tables) + "\n")
        return (
            load_helper +
            _banner("inference_init() / inference_deinit()") +
            tables +
            "/* Internal: pool lifecycle — defined in inference_buf.c */\n"
            "int  inference_buf_pool_init(void);\n"
            "void inference_buf_pool_deinit(void);\n"
            "\n"
            "#ifndef __linux__\n"
            "/* Bare metal: the drivers' CfgInitialize with the base address (no\n"
            " * configuration table, no device ids). */\n"
            f"{cfg_inits}"
            "#endif\n"
            "\n"
            "#ifdef __linux__\n"
            "int inference_init(\n"
            f"{params_decl})\n"
            "#else\n"
            "int inference_init(\n"
            f"{params_decl_bm})\n"
            "#endif\n"
            "{\n"
            "    int rc;\n"
            "\n"
            "    /* Initialise DMA buffer pool */\n"
            "    rc = inference_buf_pool_init();\n"
            "    if (rc != 0) return rc;\n"
            "\n"
            f"{alloc_str}"
            f"{init_calls_str}"
            "#ifdef __linux__\n"
            "    /* The Cortex-R5 firmware, when it runs, drives the kernels of the entries that\n"
            "     * recommend it (INFERENCE_RPU_RECOMMENDED*; INFERENCE_RPU=0 / 1 overrides). */\n"
            "    inference_rpu_select(0u);\n"
            "#endif\n"
            "    return 0;\n"
            "\n"
            "fail:\n"
            "    inference_deinit();\n"
            "    return rc;\n"
            "}\n"
            "\n"
            "void inference_deinit(void)\n"
            "{\n"
            "#ifdef __linux__\n"
            "    inference_rpu_detach();   /* the drivers' real bases back; the ring is a pool buffer */\n"
            "    inference_irq_close();\n"
            "#endif\n"
            f"{deinit_body}"
            "    inference_buf_pool_deinit();\n"
            "}\n"
        )

    def _use_compact_init(self) -> bool:
        """Table-driven pool views in inference_init() / deinit() (LLM and
        multi-entry projects): a straight-line view per buffer — thousands
        of them — sends GCC's integrated register allocator past 2.5 GB
        (aarch64 gcc 13 -O1; gcc 11 -O2 on the KV260).  Existing models keep
        the straight-line form (byte-identical projects)."""
        from ..numeric import is_active
        return bool(getattr(self, "_compact_init", False)
                    or is_active(getattr(self._graph, "numeric", None) or
                                 {"exp": {}, "host": {}, "state": []}))

    def _kernel_expect_us(self) -> dict:
        """{node index: the predicted µs of the node's shortest kernel call} —
        the A53's early wake (runtime/inference_irq.h): the bitstream's
        performance model where it prices the call, else the analytic cost
        model, as the R5 policy prices them (src/rpu_policy.py).  The shortest
        call, so a node of unequal calls wakes early rather than late (late
        costs a wake-up, early only spin); a chained row-ops node's shortest
        chain (its jobs' mean times the last chain's length, + the chain term).
        A node whose calls cannot be listed (runtime-keyed) is absent."""
        cached = getattr(self, "_expect_us_cache", None)
        if cached is not None:
            return cached
        from ..rpu_policy import _calls_of, _default_perf_model, estimate_us
        pm = _default_perf_model()
        out = {}
        for sn in self._graph.nodes:
            if not getattr(sn, "kernel_name", ""):
                continue
            cs = _calls_of(sn, self._layouts)
            if not cs or any(c is None for c in cs):
                continue
            us = []
            for c in cs:
                p = pm.predict(c) if pm is not None else None
                us.append(p[0] if p is not None else estimate_us(c))
            if sn.index in self._ro_chains:              # one Start per chain of <= 16 jobs
                jobs = sum(c.count for c in cs)
                last = jobs - 16 * ((jobs - 1) // 16)
                mean = sum(u * c.count for u, c in zip(us, cs, strict=True)) / jobs
                adj = pm.chain_adjust_us(last) if pm is not None else 0.0
                out[sn.index] = max(0, int(mean * last + adj))
                continue
            out[sn.index] = max(0, int(min(us)))
        self._expect_us_cache = out
        return out

    def _emit_kernel_call(self, sn) -> str:
        """A kernel node's call: a chained row-ops node's between
        ro_chain_begin() and ro_chain_end() (_ro_chain_helpers)."""
        text = sn.emit_call(self._layouts)
        if sn.index not in self._ro_chains:
            return text
        return (f"    ro_chain_begin(&s_ro_ch_{sn.index});   /* its jobs as descriptor chains */\n"
                + text + "\n    ro_chain_end();")

    def _expect_line(self, sn) -> str:
        """INFERENCE_KERNEL_EXPECT before a node's Start: its calls' predicted
        length for the early wake (0 when unpriced: the lane keeps no stale
        prediction)."""
        kid = self._KERNEL_ID_ENUM.get(sn.kernel_name)
        if kid is None:
            return ""
        us = self._kernel_expect_us().get(sn.index, 0)
        return f"    INFERENCE_KERNEL_EXPECT({kid}, {us}u);"

    def _inference_function(self) -> str:
        graph   = self._graph
        inputs  = graph.input_tensors
        outputs = graph.output_tensors

        param_str = ",\n".join(self._run_params())
        inputs  = [t for t in inputs if not t.is_host]      # DMA I/O: the cache syncs
        outputs = [t for t in outputs if not t.is_host]

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
            "    INFERENCE_RPU_FLUSH();   /* the R5's batch, when it drives the kernels */",
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
                # A Slice view was set up once in inference_init(); an in-place
                # merge's P.V calls already wrote its output in head order.
                if self._is_inplace_merge(sn):
                    body_lines.append("    /* merged in place: the P.V calls wrote "
                                      f"{sn.output.c_name} (RPU_PLAN §13) */")
                elif not isinstance(sn, SliceNode):
                    body_lines.append(sn.emit_call(self._layouts))
                body_lines.append("")

            elif kind == 'wait':
                _, kid, drained = ev
                body_lines.append(f"    kernel_wait({kid});")
                body_lines.append(f"    INFERENCE_PROF_END({drained}u);")

            elif kind == 'start':
                sn = nodes_by_idx[ev[1]]
                body_lines.append(self._expect_line(sn))
                body_lines.append(f"    INFERENCE_PROF_BEGIN({sn.index}u);")
                body_lines.append(self._emit_kernel_call(sn))
                body_lines.append("")

            elif kind == 'start_sync':
                sn = nodes_by_idx[ev[1]]
                body_lines.append(self._expect_line(sn))
                body_lines.append(f"    INFERENCE_PROF_BEGIN({sn.index}u);")
                body_lines.append(self._emit_kernel_call(sn))
                body_lines.append(f"    INFERENCE_PROF_END({sn.index}u);")
                body_lines.append("")

            elif kind == 'cpu':
                # Host-side op (SpaceToDepthNode / HostNode): runs inline on
                # the CPU and syncs its own buffers; profiled like a
                # synchronous node.  With the R5 driving the kernels, the
                # recorded batch must have run before the host reads.
                sn = nodes_by_idx[ev[1]]
                body_lines.append("    INFERENCE_RPU_FLUSH();")
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
        parts_src = ""
        if self._use_compact_init():
            body_lines, parts_src = self._split_run_body(body_lines)

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

        suffix = self._run_name[len("inference_run"):].upper()
        sections = [f"    INFERENCE_RPU_ENTER(INFERENCE_RPU_RECOMMENDED{suffix});   "
                    "/* the R5 or the A53 drives this entry's kernels (src/rpu_policy.py) */"]
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
            parts_src +
            f"void {self._run_name}(\n"
            f"{param_str or '    void'})\n"
            "{\n"
            f"{body}\n"
            "}\n"
        )

    _RUN_PART_NODES = 40

    def _split_run_body(self, body_lines: List[str]):
        """Split a long run body at node boundaries into noinline static
        functions of about _RUN_PART_NODES nodes (compact-form projects:
        thousand-line straight-line functions are what makes the compiler's
        memory explode).  Returns (body calling the parts, their source)."""
        chunks, cur, nodes = [], [], 0
        for ln in body_lines:
            if ln.startswith("    /* [") and nodes >= self._RUN_PART_NODES:
                chunks.append(cur)
                cur, nodes = [], 0
            if ln.startswith("    /* ["):
                nodes += 1
            cur.append(ln)
        if cur:
            chunks.append(cur)
        if len(chunks) <= 1:
            return body_lines, ""
        params = self._run_params()
        names = [p.split("*")[-1].strip() for p in params]
        src, calls = [], []
        for k, ch in enumerate(chunks):
            fn = f"{self._run_name}_part{k}"
            while ch and ch[-1] == "":
                ch = ch[:-1]
            src.append(f"static void __attribute__((noinline)) {fn}(\n"
                       + (",\n".join(params) or "    void") + ")\n{\n"
                       + "".join(f"    (void){n};\n" for n in names)
                       + "\n".join(ch) + "\n}\n\n")
            calls.append(f"    {fn}({', '.join(names)});")
        return calls, "".join(src)

    @property
    def _run_name(self) -> str:
        return getattr(self, "_run_fn_name", "inference_run")

    def _run_params(self) -> List[str]:
        """inference_run() parameters: DMA buffers, and plain pointers for
        host-memory graph inputs (const) / outputs."""
        out = []
        for t in self._graph.input_tensors:
            out.append(f"    const {self._host_c_type(t)} *{t.c_name}" if t.is_host
                       else f"    inference_buf_t *{t.c_name}")
        for t in self._graph.output_tensors:
            out.append(f"    {self._host_c_type(t)} *{t.c_name}" if t.is_host
                       else f"    inference_buf_t *{t.c_name}")
        return out
