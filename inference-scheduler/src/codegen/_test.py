"""Mixin that generates test/test_inference.c."""

from __future__ import annotations

from ._banners import _file_banner
from ._simulate import LARGE_EXPECTED_THRESHOLD


class _TestMixin:
    """Generates test/test_inference.c."""

    def generate_test(self) -> str:
        """Content for test/test_inference.c — on-device smoke + GT check.

        Allocates input/output DMA buffers, fills inputs with a ramp pattern,
        runs inference, compares every output element against expected values
        computed by the Python fixed-point simulator, and prints PASSED/FAILED.

        The ramp fill, display casts, and expected-array types are all driven
        by self._dtype so the file is correct for any supported element type.

        For broadcast models, inputs are filled chunk-by-chunk (skipping
        alignment gap elements), and both the output print and the GT
        comparison iterate chunk-by-chunk so gap elements — which
        VectorOPKernel never writes — are not included in either check.

        Small expected arrays (≤ LARGE_EXPECTED_THRESHOLD elements) are
        embedded as static C arrays.  Large arrays are loaded at runtime
        from expected/<c_name>.dat via _load_expected(), following the same
        pattern as large weight tensors and weights/<c_name>.dat.
        """
        dtype     = self._dtype
        graph     = self._graph
        inputs    = graph.input_tensors
        outputs   = graph.output_tensors
        bcast_map = self._broadcast_io_map()  # onnx_name → (n, chunk_macro, stride_macro)

        # --- Run fixed-point simulation; classify outputs by size --------- #
        sim_arrays   = self._simulate()
        large_names  = {t.onnx_name for t in self.large_expected_tensors}
        has_large    = bool(large_names)

        # Expected declarations at file scope
        expected_decls = []
        for t in outputs:
            storage_buf = self._expected_storage(t.onnx_name, sim_arrays[t.onnx_name])
            alloc_size  = len(storage_buf)
            if t.onnx_name in large_names:
                # Heap pointer — loaded from file at runtime
                expected_decls.append(
                    f"static {dtype.c_array_type} *expected_{t.c_name} = NULL;"
                    f"  /* {alloc_size} elem — loaded from expected/{t.c_name}.dat */"
                )
            else:
                expected_decls.append(self._emit_expected_c(t.c_name, storage_buf))
        expected_str = "\n".join(expected_decls)

        # --- Variable declarations at top of main() ---------------------- #
        decl_lines = []
        for t in inputs + outputs:
            decl_lines.append(f"    inference_buf_t *{t.c_name} = NULL;")

        # --- Buffer allocations with goto-cleanup error handling ---------- #
        alloc_lines = []
        for t in inputs + outputs:
            macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
            alloc_lines.append(f"    {t.c_name} = inference_buf_alloc({macro});")
            alloc_lines.append(f"    if (!{t.c_name}) {{")
            alloc_lines.append(
                f"        fprintf(stderr, "
                f"\"inference_buf_alloc failed for '{t.c_name}'\\n\");"
            )
            alloc_lines.append("        rc = 1; goto cleanup;")
            alloc_lines.append("    }")

        # --- Fill each input with the ramp ------------------------------- #
        # For broadcasting inputs, iterate chunk-by-chunk so gap elements
        # (alignment padding between data blocks) are left untouched.
        fill_lines = []
        for t in inputs:
            if t.is_int:
                # Integer tensor (token ids, masks): raw integer values, a
                # pattern that is a valid index for every Gather / OneHot
                # reading it (the simulator uses the same values).
                r     = self._int_fill_range(t)
                macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
                fill_lines += [
                    f"    {{  /* '{t.c_name}' — integer tensor: p[i] = i % {r} */",
                    f"        Data_t *p = inference_buf_ptr({t.c_name});",
                    f"        for (i = 0u; i < {macro}; i++)",
                    f"            p[i] = (Data_t)(i % {r}u);",
                    "    }",
                ]
                continue
            if t.onnx_name in bcast_map:
                n, chunk_macro, stride_macro = bcast_map[t.onnx_name]
                rhs = dtype.c_fill_rhs("_off + i")
                fill_lines += [
                    f"    {{  /* '{t.c_name}' — broadcast input, fill data chunks only */",
                    f"        Data_t   *p = inference_buf_ptr({t.c_name});",
                    "        unsigned  _chunk, _off;",
                    f"        for (_chunk = 0u; _chunk < {n}u; _chunk++) {{",
                    f"            _off = _chunk * {stride_macro};",
                    f"            for (i = 0u; i < {chunk_macro}; i++)",
                    f"                p[_off + i] = {rhs};",
                    "        }",
                    "    }",
                ]
            else:
                macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
                rhs   = dtype.c_fill_rhs("i")
                fill_lines += [
                    "    {",
                    f"        Data_t *p = inference_buf_ptr({t.c_name});",
                    f"        for (i = 0u; i < {macro}; i++)",
                    f"            p[i] = {rhs};",
                    "    }",
                ]

        # --- inference_run() call ---------------------------------------- #
        run_args = ", ".join(t.c_name for t in inputs + outputs)

        # --- Print first ≤8 elements of each output ---------------------- #
        print_lines = []
        for t in outputs:
            display = dtype.c_display("p", "_off + i") if t.onnx_name in bcast_map \
                      else dtype.c_display("p", "i")
            if t.is_int:
                macro = f"INFERENCE_{t.c_name.upper()}_SIZE"
                print_lines += [
                    "    {",
                    f"        Data_t   *p   = inference_buf_ptr({t.c_name});",
                    f"        unsigned  lim = ({macro} < 8u) ? {macro} : 8u;",
                    f"        printf(\"Output '{t.onnx_name}'"
                    f" (%u elem, integer, first %u):\\n\", (unsigned){macro}, lim);",
                    "        for (i = 0u; i < lim; i++) {",
                    f"            printf(\"  [%u] %d\\n\", i, {dtype.c_int_display('p', 'i')});",
                    "        }",
                    "    }",
                ]
                continue
            if t.onnx_name in bcast_map:
                n, chunk_macro, stride_macro = bcast_map[t.onnx_name]
                print_lines += [
                    f"    {{  /* '{t.c_name}' — broadcast output, print data chunks only */",
                    f"        Data_t   *p     = inference_buf_ptr({t.c_name});",
                    f"        unsigned  numel = {n}u * {chunk_macro};",
                    "        unsigned  lim   = (numel < 8u) ? numel : 8u;",
                    "        unsigned  shown = 0u, _chunk, _off;",
                    f"        printf(\"Output '{t.onnx_name}' (%u elem, first %u):\\n\","
                    f" numel, lim);",
                    f"        for (_chunk = 0u; _chunk < {n}u && shown < lim; _chunk++) {{",
                    f"            _off = _chunk * {stride_macro};",
                    f"            for (i = 0u; i < {chunk_macro} && shown < lim;"
                    f" i++, shown++) {{",
                    "                printf(\"  [%u] (%.4f)\\n\",",
                    f"                       shown, {display});",
                    "            }",
                    "        }",
                    "    }",
                ]
            else:
                macro   = f"INFERENCE_{t.c_name.upper()}_SIZE"
                display = dtype.c_display("p", "i")
                print_lines += [
                    "    {",
                    f"        Data_t   *p   = inference_buf_ptr({t.c_name});",
                    f"        unsigned  lim = ({macro} < 8u) ? {macro} : 8u;",
                    f"        printf(\"Output '{t.onnx_name}'"
                    f" (%u elem, first %u):\\n\", (unsigned){macro}, lim);",
                    "        for (i = 0u; i < lim; i++) {",
                    f"            printf(\"  [%u] (%.4f)\\n\", i, {display});",
                    "        }",
                    "    }",
                ]

        # --- GT comparison: output vs expected_<name>[] ------------------ #
        # Broadcast outputs: compare data positions only (gap elements are
        # never written by the kernel, so their DMA-buffer content is
        # uninitialised and must not be checked).
        # Non-broadcast outputs: compare every element (alloc == numel).
        verify_lines = []
        for t in outputs:
            if t.onnx_name in bcast_map:
                n, chunk_macro, stride_macro = bcast_map[t.onnx_name]
                got_disp = dtype.c_display("p", "_off + j")
                exp_disp = dtype.c_display(f"expected_{t.c_name}", "_off + j")
                verify_lines += [
                    f"    {{  /* [GT] '{t.c_name}' — broadcast output, data chunks only */",
                    f"        const Data_t *p = (const Data_t *)inference_buf_ptr({t.c_name});",
                    "        unsigned _chunk, j, _off;",
                    f"        for (_chunk = 0u; _chunk < {n}u; _chunk++) {{",
                    f"            _off = _chunk * {stride_macro};",
                    f"            for (j = 0u; j < {chunk_macro}; j++) {{",
                    f"                if (p[_off + j] != expected_{t.c_name}[_off + j]) {{",
                    "                    fprintf(stderr,",
                    f"                            \"FAIL {t.onnx_name}[chunk=%u,j=%u]: \"",
                    "                            \"got %.4f expected %.4f\\n\",",
                    "                            _chunk, j,",
                    f"                            {got_disp},",
                    f"                            {exp_disp});",
                    "                    rc = 1;",
                    "                }",
                    "            }",
                    "        }",
                    "    }",
                ]
            elif t.is_int:
                alloc_size = self._alloc_sizes[t.onnx_name]
                verify_lines += [
                    f"    {{  /* [GT] '{t.c_name}' — integer tensor, exact */",
                    f"        const Data_t *p = (const Data_t *)inference_buf_ptr({t.c_name});",
                    "        unsigned k;",
                    f"        for (k = 0u; k < {alloc_size}u; k++) {{",
                    f"            if (p[k] != expected_{t.c_name}[k]) {{",
                    "                fprintf(stderr,",
                    f"                        \"FAIL {t.onnx_name}[%u]: got %d expected %d\\n\",",
                    f"                        k, {dtype.c_int_display('p', 'k')},",
                    f"                        {dtype.c_int_display(f'expected_{t.c_name}', 'k')});",
                    "                rc = 1;",
                    "            }",
                    "        }",
                    "    }",
                ]
            else:
                alloc_size = self._alloc_sizes[t.onnx_name]
                got_disp   = dtype.c_display("p", "k")
                exp_disp   = dtype.c_display(f"expected_{t.c_name}", "k")
                verify_lines += [
                    f"    {{  /* [GT] '{t.c_name}' */",
                    f"        const Data_t *p = (const Data_t *)inference_buf_ptr({t.c_name});",
                    "        unsigned k;",
                    f"        for (k = 0u; k < {alloc_size}u; k++) {{",
                    f"            if (p[k] != expected_{t.c_name}[k]) {{",
                    "                fprintf(stderr,",
                    f"                        \"FAIL {t.onnx_name}[%u]: \"",
                    "                        \"got %.4f expected %.4f\\n\",",
                    "                        k,",
                    f"                        {got_disp},",
                    f"                        {exp_disp});",
                    "                rc = 1;",
                    "            }",
                    "        }",
                    "    }",
                ]

        # --- Large expected: load step before GT check ------------------- #
        load_lines = []
        if has_large:
            for t in outputs:
                if t.onnx_name not in large_names:
                    continue
                alloc_size = self._alloc_sizes[t.onnx_name]
                load_lines += [
                    f"    if (_load_expected(&expected_{t.c_name},"
                    f" \"{t.c_name}\", {alloc_size}u) != 0)"
                    f" {{ rc = 1; goto cleanup; }}",
                ]

        # --- Cleanup: free I/O buffers + heap expected arrays ------------ #
        cleanup_lines = []
        for t in inputs + outputs:
            cleanup_lines.append(f"    inference_buf_free({t.c_name});")
        for t in outputs:
            if t.onnx_name in large_names:
                cleanup_lines.append(f"    free(expected_{t.c_name});")

        decl_str    = "\n".join(decl_lines)
        alloc_str   = "\n".join(alloc_lines)
        fill_str    = "\n".join(fill_lines) if fill_lines else "    /* (no inputs) */"
        print_str   = "\n".join(print_lines)
        load_str    = "\n".join(load_lines)
        verify_str  = "\n".join(verify_lines)
        cleanup_str = "\n".join(cleanup_lines)

        # --- _load_expected() helper (only when there are large expected) - #
        load_expected_helper = ""
        if has_large:
            load_expected_helper = (
                "\n"
                "#ifndef INFERENCE_EXPECTED_DIR\n"
                "#  define INFERENCE_EXPECTED_DIR  \".\"\n"
                "#endif\n"
                "\n"
                "static int _load_expected("
                f"{dtype.c_array_type} **out, const char *name, unsigned n_elem)\n"
                "{\n"
                "    char   path[512];\n"
                "    FILE  *f;\n"
                "    size_t n_read;\n"
                "    snprintf(path, sizeof(path),\n"
                "             INFERENCE_EXPECTED_DIR \"/expected/%s.dat\", name);\n"
                "    f = fopen(path, \"rb\");\n"
                "    if (!f) {\n"
                "        fprintf(stderr,\n"
                "                \"GT: cannot open expected file '%s'\\n\", path);\n"
                "        return -1;\n"
                "    }\n"
                f"    *out = ({dtype.c_array_type} *)"
                "malloc((size_t)n_elem * INFERENCE_BYTES_PER_ELEM);\n"
                "    if (!*out) { fclose(f); return -1; }\n"
                "    n_read = fread(*out, INFERENCE_BYTES_PER_ELEM, n_elem, f);\n"
                "    fclose(f);\n"
                "    if (n_read != (size_t)n_elem) {\n"
                "        fprintf(stderr,\n"
                "                \"GT: short read from '%s':"
                " got %zu expected %u\\n\",\n"
                "                path, n_read, n_elem);\n"
                "        free(*out); *out = NULL;\n"
                "        return -1;\n"
                "    }\n"
                "    return 0;\n"
                "}\n"
            )

        stdlib_include = "#include <stdlib.h>\n" if has_large else ""

        gt_load_block = (
            "\n"
            "    /* 5b. Load large expected arrays from file */\n"
            f"{load_str}\n"
        ) if has_large else ""

        return (
            _file_banner("test_inference.c", graph, self._model_path) +
            "\n"
            "#include <stdio.h>\n"
            "#include <stdint.h>\n"
            f"{stdlib_include}"
            "#include \"inference.h\"\n"
            "\n"
            "/*\n"
            " * Default UIO device names for each hardware kernel.\n"
            " *\n"
            " * On Linux the driver scans /sys/class/uio/uioN/name and opens the\n"
            " * first device whose sysfs name matches the string below.  The names\n"
            " * are the DT node labels from the DTBO overlay (check with:\n"
            " *   cat /sys/class/uio/uio0/name  on the board).\n"
            " *\n"
            " * On bare-metal the name must match the IP instance in xparameters.h\n"
            " * (e.g. XPAR_VECTOROPKERNEL_0_DEVICE_ID).\n"
            " *\n"
            " * Override at CMake configure time, e.g.:\n"
            + "".join(
                f" *   cmake -D{kd.instance_macro}={kd.uio_default} ...\n"
                for kd in self._active_kernels
            ) +
            " */\n"
            + "".join(
                f"/* {kd.name} — AXI base 0x{kd.axi_base:08X} */\n"
                f"#ifndef {kd.instance_macro}\n"
                f"#  define {kd.instance_macro}  \"{kd.uio_default}\"\n"
                f"#endif\n"
                for kd in self._active_kernels
            ) +
            f"{load_expected_helper}"
            "\n"
            "/* Expected output values — computed by the Python fixed-point simulator.\n"
            f" * Data type  : {dtype.name}  ({dtype.bytes_per_elem} bytes/elem)\n"
            " * Inputs     : ramp (element at buffer position p cast to Data_t).\n"
            " * Broadcast tensors use strided layout; gap slots are zero.\n"
            f" * Arrays > {LARGE_EXPECTED_THRESHOLD} elem are loaded from"
            " expected/<name>.dat at runtime. */\n"
            f"{expected_str}"
            "\n"
            "int main(int argc, char **argv)\n"
            "{\n"
            "    (void)argc; (void)argv;\n"
            f"{decl_str}\n"
            "    unsigned i;\n"
            "    int      rc = 0;\n"
            "\n"
            "    /* 1. Initialise: open UIO devices, map DMA pool */\n"
            + (
                "    rc = inference_init(\n"
                + ",\n".join(
                    f"        {kd.instance_macro}"
                    for kd in self._active_kernels
                )
                + ");\n"
            ) +
            "    if (rc != 0) {\n"
            "        fprintf(stderr, \"inference_init() failed: %d\\n\", rc);\n"
            "        return 1;\n"
            "    }\n"
            "    printf(\"inference_init OK\\n\");\n"
            "\n"
            "    /* 2. Allocate I/O buffers from the DMA pool */\n"
            f"{alloc_str}\n"
            "\n"
            "    /* 3. Fill inputs with ramp pattern */\n"
            f"{fill_str}\n"
            "\n"
            "    /* 4. Run inference */\n"
            f"    inference_run({run_args});\n"
            "\n"
            "    /* 5. Print first 8 output elements */\n"
            f"{print_str}\n"
            f"{gt_load_block}"
            "\n"
            "    /* 6. Compare against fixed-point simulation (GT check) */\n"
            f"{verify_str}\n"
            "\n"
            "    if (rc == 0)\n"
            "        printf(\"test_inference PASSED\\n\");\n"
            "    else\n"
            "        fprintf(stderr, \"test_inference FAILED\\n\");\n"
            "\n"
            "    /* 7. Teardown — reached on both success and alloc failure */\n"
            "cleanup:\n"
            f"{cleanup_str}\n"
            "    inference_deinit();\n"
            "    return rc;\n"
            "}\n"
        )
