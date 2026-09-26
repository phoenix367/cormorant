"""Mixin that generates include/inference.h."""

from __future__ import annotations

import numpy as np

from ..nodes  import _ALIGN_BYTES, MatmulNode, ReshapeNode
from ._banners import _banner


class _HeaderMixin:
    """Generates include/inference.h."""

    def generate_header(self) -> str:
        """Content for include/inference.h."""
        from ._banners import _file_banner
        parts = [
            _file_banner("inference.h", self._graph, self._model_path),
            self._header_guard_open(),
            self._header_api(),
            self._header_guard_close(),
        ]
        return "\n".join(parts) + "\n"

    def _header_guard_open(self) -> str:
        return (
            "\n#pragma once\n"
            "\n#ifdef __cplusplus\n"
            "extern \"C\" {\n"
            "#endif\n"
            "\n#include <stdint.h>\n"
        )

    def _header_guard_close(self) -> str:
        return "\n#ifdef __cplusplus\n}\n#endif\n"

    def _header_api(self) -> str:
        lines = (self._header_types(self._graph.input_tensors + self._graph.output_tensors)
                 + self._header_sizes()
                 + self._header_pool(self._compute_pool_bytes())
                 + self._header_buf_api()
                 + self._header_uio()
                 + self._header_init()
                 + self._header_layers(len(self._layer_display_names()))
                 + self._header_run())
        return "\n".join(lines)

    def _header_types(self, io_tensors) -> list:
        """Data type section (+ the integer-tensor note for ``io_tensors``)."""
        dtype = self._dtype
        lines = [_banner("Data type")]
        lines.append(dtype.c_typedef_comment())
        lines.append(f"typedef {dtype.c_type} Data_t;")
        lines.append(f"#define INFERENCE_BYTES_PER_ELEM  {dtype.bytes_per_elem}u")
        lines.append("")
        lines.append(
            "/* AXI burst alignment — every broadcast chunk stride is a multiple\n"
            " * of INFERENCE_ALIGN_BYTES so each kernel invocation starts at an\n"
            " * aligned physical address.  Derived from the two macros above;\n"
            " * never hardcode the element count directly. */"
        )
        lines.append(f"#define INFERENCE_ALIGN_BYTES  {_ALIGN_BYTES}u")
        lines.append(
            "#define INFERENCE_ALIGN_ELEMS  "
            "(INFERENCE_ALIGN_BYTES / INFERENCE_BYTES_PER_ELEM)"
        )
        lines.append(
            "#define INFERENCE_ALIGN_UP(n)  \\\n"
            "    (((n) + INFERENCE_ALIGN_ELEMS - 1u) & ~(INFERENCE_ALIGN_ELEMS - 1u))"
        )
        lines.append("")
        int_io = [t for t in io_tensors if t.is_int and not t.is_host]
        if int_io:
            int_t = f"int{8 * dtype.bytes_per_elem}_t"
            lines.append(
                "/* Integer tensors (token ids, segment ids, masks, ...) are NOT stored\n"
                f" * in the {dtype.name} encoding: every Data_t element holds the raw\n"
                f" * signed integer value ({int_t}), e.g.\n"
                f" *     inference_buf_ptr(buf)[i] = (Data_t)({int_t})token_id;\n"
                f" * Values must fit in {int_t} (a 30522-entry vocabulary does).  Here:\n"
                + "".join(f" *   {t.c_name} ({t.dtype} {t.shape})\n" for t in int_io)
                + " */"
            )
            lines.append("")
        return lines

    def _header_sizes(self, banner: bool = True) -> list:
        """Broadcast chunk macros and one SIZE macro per graph input / output."""
        graph   = self._graph
        inputs  = graph.input_tensors
        outputs = graph.output_tensors
        lines: list = []

        # ---- collect broadcast info for the SIZE-macro section ----
        # Only VectorOP ScheduledNodes produce CHUNK/STRIDE alignment macros.
        # MatmulNode with outer_count > 1 uses physical-address offsets instead,
        # so it is excluded from broadcast_nodes and from bcast_map.
        bcast_map       = self._broadcast_io_map()   # onnx_name → (n, chunk_macro, stride_macro)
        broadcast_nodes = [
            sn for sn in graph.nodes
            if sn.outer_count > 1 and not isinstance(sn, (MatmulNode, ReshapeNode))
        ]

        # ---- Array-size macros ----
        # Emit CHUNK / CHUNK_STRIDE macros first (referenced by SIZE macros below).
        if banner:
            lines.append(_banner("Array size constants"))
        if broadcast_nodes:
            lines.append("/* Broadcast chunk macros (one set per broadcast op).\n"
                         " * CHUNK        = data elements per kernel call\n"
                         " * CHUNK_STRIDE = INFERENCE_ALIGN_UP(CHUNK) — padded stride;\n"
                         " *               the kernel's 128-bit ports read every run as\n"
                         " *               whole 16-byte words and write the last word\n"
                         " *               of a run whole, so the gap up to the next\n"
                         " *               16-byte boundary receives 0 and the rest of\n"
                         " *               the gap is never touched. */")
            for sn in broadcast_nodes:
                c_up = sn.output.c_name.upper()
                chunk_macro  = f"INFERENCE_{c_up}_CHUNK"
                stride_macro = f"INFERENCE_{c_up}_CHUNK_STRIDE"
                lines.append(
                    f"#define {chunk_macro:<44} {sn.chunk_size}u"
                    f"  /* shape={sn.output.shape}, outer_count={sn.outer_count} */"
                )
                lines.append(
                    f"#define {stride_macro:<44} "
                    f"INFERENCE_ALIGN_UP({chunk_macro})"
                )
            lines.append("")

        for t in inputs:
            macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
            if t.onnx_name in bcast_map:
                n, _chunk_macro, stride_macro = bcast_map[t.onnx_name]
                size_expr = f"({n}u * {stride_macro})"
                lines.append(
                    f"#define {macro:<40} {size_expr}"
                    f"  /* shape={t.shape} */"
                )
            else:
                lines.append(
                    f"#define {macro:<40} {t.numel}u  /* shape={t.shape} */"
                )
        for t in outputs:
            macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
            if t.onnx_name in bcast_map:
                n, _chunk_macro, stride_macro = bcast_map[t.onnx_name]
                size_expr = f"({n}u * {stride_macro})"
                lines.append(
                    f"#define {macro:<40} {size_expr}"
                    f"  /* shape={t.shape} */"
                )
            else:
                lines.append(
                    f"#define {macro:<40} {t.numel}u  /* shape={t.shape} */"
                )
        return lines

    def _header_pool(self, pool_bytes: int) -> list:
        lines = []
        # DMA buffer pool size
        lines.append(_banner("DMA buffer pool"))
        lines.append(
            "/*\n"
            " * Minimum contiguous DMA pool required for this model.\n"
            " *\n"
            " * Linux:      run scripts/check_inference_setup.sh before the\n"
            " *             application to verify root access and pagemap\n"
            " *             availability (required for physical address lookup).\n"
            " * Bare-metal: pool is not used; each buffer is allocated from\n"
            " *             the heap individually.\n"
            " */"
        )
        lines.append(
            f"#define INFERENCE_BUF_POOL_SIZE_BYTES  {pool_bytes}u"
        )
        lines.append("")
        return lines

    def _header_buf_api(self) -> list:
        lines = []
        # DMA buffer API
        lines.append(_banner("DMA-capable buffer API"))
        lines.append(
            "/* DMA-capable buffer handle.\n"
            " * Allocate owners with inference_buf_alloc().\n"
            " * Initialise static sub-buffer views with inference_buf_init_view().\n"
            " * Physical address (for kernel DMA registers) and virtual address (for\n"
            " * CPU access) are exposed via inference_buf_phys() / inference_buf_ptr().\n"
            " *\n"
            " * The struct is defined here (not opaque) so that inference.c can\n"
            " * declare static view instances without heap allocation. */"
        )
        lines.append(
            "struct inference_buf {\n"
            "    void     *virt;       /* CPU-accessible virtual address */\n"
            "    uint64_t  phys;       /* physical DDR address for AXI DMA registers */\n"
            "    unsigned  count;      /* number of Data_t elements allocated */\n"
            "    unsigned  refcount;   /* reference count; 0 for views, >=1 for owners */\n"
            "    uint8_t   is_owner;   /* 1 = owns the allocation; 0 = view (alias) */\n"
            "    uint8_t   cached;     /* 1 = CPU mapping is cacheable (sync_* do real cache\n"
            "                           * maintenance); 0 = non-cacheable mapping */\n"
            "#ifdef __linux__\n"
            "    unsigned  bo;         /* XRT buffer object handle */\n"
            "    uint64_t  bo_offset;  /* byte offset within the BO (0 for owners) */\n"
            "#endif\n"
            "};\n"
            "typedef struct inference_buf inference_buf_t;"
        )
        lines.append("")
        lines.append(
            "/* Allocate a buffer for n_elem Data_t elements from the DMA pool.\n"
            " * Returns NULL on failure.  On Linux the bump allocator never frees\n"
            " * individual sub-allocations; call inference_deinit() to release all. */"
        )
        lines.append("inference_buf_t *inference_buf_alloc(unsigned n_elem);")
        lines.append("")
        lines.append(
            "/* Increment the reference count.  No-op for views (is_owner == 0).\n"
            " * Call before sharing a buffer with an alias that may outlive the\n"
            " * original pointer (e.g. redirecting an internal buffer to a\n"
            " * caller-supplied output in inference_run()). */"
        )
        lines.append("void  inference_buf_retain(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Decrement the reference count and free when it reaches zero.\n"
            " * Equivalent to inference_buf_free() for a freshly allocated buffer. */"
        )
        lines.append("void  inference_buf_release(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Release a buffer — equivalent to inference_buf_release().\n"
            " * Kept for backward compatibility; respects the reference count. */"
        )
        lines.append("void  inference_buf_free(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* CPU-accessible virtual pointer to the buffer data. */"
        )
        lines.append("Data_t  *inference_buf_ptr(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Physical DDR address — write this value into AXI-Lite DMA registers. */"
        )
        lines.append("uint64_t inference_buf_phys(const inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Number of Data_t elements that were allocated. */"
        )
        lines.append("unsigned inference_buf_count(const inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* 1 when the CPU mapping of buf is cacheable (fast CPU access; the\n"
            " * sync_* calls below then write back / drop the CPU cache lines), 0 for\n"
            " * a non-cacheable (write-combine) mapping.  Linux: XRT buffer objects\n"
            " * are cacheable unless built with -DINFERENCE_BUF_CACHEABLE=OFF or run\n"
            " * with the environment variable INFERENCE_BUF_CACHEABLE=0. */"
        )
        lines.append(
            "static inline int inference_buf_is_cached(const inference_buf_t *buf)\n"
            "{\n"
            "    return buf->cached != 0u;\n"
            "}"
        )
        lines.append("")
        lines.append(
            "/* Initialise *view as a sub-buffer of *base starting at offset_elems\n"
            " * elements.  The view shares the same physical memory as base; it does NOT\n"
            " * own any allocation and must not be passed to inference_buf_free().\n"
            " * The backing storage for *view must have static or longer lifetime. */"
        )
        lines.append(
            "void inference_buf_init_view(inference_buf_t *view,\n"
            "                             inference_buf_t *base,\n"
            "                             unsigned offset_elems,\n"
            "                             unsigned count_elems);"
        )
        lines.append("")
        lines.append(
            "/* Sync buffer to device — write back (clean) the CPU cache lines of\n"
            " * the buffer's range before a PL kernel reads OR writes it.\n"
            " * Called automatically by inference_run(); exposed for manual control. */"
        )
        lines.append("void inference_buf_sync_to_device(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Sync buffer from device — drop (invalidate) the CPU cache lines of the\n"
            " * buffer's range AFTER a PL kernel wrote it, before the CPU reads it.\n"
            " * Called automatically by inference_run(); exposed for manual control. */"
        )
        lines.append("void inference_buf_sync_from_device(inference_buf_t *buf);")
        lines.append("")
        lines.append(
            "/* Fill n elements of buf from a float array.\n"
            " * Each value is cast to Data_t using C's built-in conversion.\n"
            " * n must be <= inference_buf_count(buf). */"
        )
        lines.append(
            "void inference_buf_fill_float(inference_buf_t *buf,"
            " const float *src, unsigned n);"
        )
        lines.append("")
        lines.append(
            "/* Read n elements from buf into a float array.\n"
            " * Each Data_t element is cast to float using C's built-in conversion.\n"
            " * n must be <= inference_buf_count(buf). */"
        )
        lines.append(
            "void inference_buf_read_float(const inference_buf_t *buf,"
            " float *dst, unsigned n);"
        )
        lines.append("")
        return lines

    def _header_uio(self) -> list:
        lines = []
        # UIO device name defaults — one macro per active kernel
        lines.append(_banner("UIO device name defaults"))
        lines.append(
            "/*\n"
            " * Default UIO sysfs device names for the hardware kernels used by\n"
            " * this model.  These match the DT node labels in the DTBO overlay\n"
            " * (check: cat /sys/class/uio/uio0/name on the board).\n"
            " *\n"
            " * Override at CMake configure time:\n"
            " *   cmake -DINFERENCE_VECTOROPKERNEL_INSTANCE=my_vop ...\n"
            " * or define the macro before including this header.\n"
            " */"
        )
        for kd in self._active_kernels:
            lines.append(
                f"#ifndef {kd.instance_macro}\n"
                f"#  define {kd.instance_macro}  \"{kd.uio_default}\"\n"
                f"#endif"
            )
        lines.append("")
        return lines

    def _header_init(self) -> list:
        lines = []
        # inference_init()
        lines.append(_banner("Inference API"))
        # Build init signature
        active = self._active_kernels
        if len(active) == 1:
            init_params = f"const char *{active[0].init_param}"
            param_doc = (
                f" * {active[0].init_param}  "
                f"UIO sysfs name (Linux) or xparameters.h name (bare-metal).\n"
                f" *              Default: \"{active[0].uio_default}\"\n"
            )
        else:
            param_doc_lines = []
            for kd in active:
                param_doc_lines.append(
                    f" * {kd.init_param}  "
                    f"UIO name for {kd.name}. Default: \"{kd.uio_default}\""
                )
            param_doc = "\n".join(param_doc_lines) + "\n"
            init_params = ",\n    ".join(
                f"const char *{kd.init_param}" for kd in active
            )
        lines.append(
            "/*\n"
            " * inference_init() — initialise the DMA pool, weight buffers,\n"
            " *                    and all hardware kernel drivers.\n"
            " *\n"
            f"{param_doc}"
            " * Returns 0 on success, non-zero on failure.\n"
            " */"
        )
        lines.append(f"int  inference_init({init_params});")
        lines.append("")
        lines.append(
            "/*\n"
            " * inference_deinit() — release all DMA buffers and the pool.\n"
            " * Call once when the inference library is no longer needed.\n"
            " */"
        )
        lines.append("void inference_deinit(void);")
        lines.append("")
        return lines

    def _header_layers(self, n_layers: int) -> list:
        lines = []
        # Per-layer profiling introspection — exposes the layer-name table
        # baked into inference.c.  The host benchmark feeds these into
        # inference_prof_init() when INFERENCE_PROFILING is enabled.
        lines.append(_banner("Per-layer profiling introspection"))
        lines.append(
            "/*\n"
            " * Number of profileable layers in this model — one entry per\n"
            " * scheduled ONNX node (ReshapeNode aliases included; they record\n"
            " * count=0 because they emit no kernel call).  Use the accessors\n"
            " * below to feed inference_prof_init() at runtime.\n"
            " */"
        )
        lines.append(f"#define INFERENCE_NUM_LAYERS  {n_layers}u")
        lines.append("")
        lines.append("unsigned             inference_num_layers(void);")
        lines.append("const char *const   *inference_layer_names_ptr(void);")
        lines.append("")
        return lines

    def _header_run(self) -> list:
        lines = []
        inputs  = self._graph.input_tensors
        outputs = self._graph.output_tensors
        # inference_run()
        params = self._run_params()

        doc_lines = [
            "/*",
            f" * {self._run_name}() — execute the full inference graph.",
            " *",
            " *   Buffers must be allocated with inference_buf_alloc().",
            " *   Fill input buffers via inference_buf_ptr() before calling.",
            " *",
        ]
        def _kind(t):
            if t.is_host:
                return f", host {self._host_c_type(t)}"
            if t.exp is not None:
                e = np.asarray(t.exp)
                return (f", raw int16 at 2^-{int(e)}" if e.ndim == 0
                        else ", raw int16 at per-channel 2^-f")
            return f", raw {t.dtype} values" if t.is_int else ""
        for t in inputs:
            ptype = f"const {self._host_c_type(t)}*" if t.is_host else "inference_buf_t*"
            doc_lines.append(
                f" *   {t.c_name:<22} [in]   {ptype:<18}"
                f"({t.numel} elem, shape={t.shape}{_kind(t)})"
            )
        for t in outputs:
            ptype = f"{self._host_c_type(t)}*" if t.is_host else "inference_buf_t*"
            doc_lines.append(
                f" *   {t.c_name:<22} [out]  {ptype:<18}"
                f"({t.numel} elem, shape={t.shape}{_kind(t)})"
            )
        doc_lines.append(" */")

        lines.append("\n".join(doc_lines))
        lines.append(
            f"void {self._run_name}(\n"
            + (",\n".join(params) or "    void")
            + ");"
        )
        return lines
