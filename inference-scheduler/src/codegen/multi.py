"""
Multi-entry projects (doc/CHAT_PLAN.md §3.2 B2): one generated library with
one ``inference_run_<entry>()`` per graph — e.g. an LLM's decode step, its
prefill buckets and the head — over ONE weight pool.

  * Weights are deduplicated by name AND emitted image: every entry that
    reads the initializer ``w`` in the same layout shares one DMA buffer; an
    entry that needs another layout (a MatMul lowered onto ConvKernel reads
    its B in the conv image, MatmulKernel in the packed image) gets its own
    copy, renamed ``w@<k>``.
  * States (src/numeric.py) are shared by name: a KV cache written by one
    entry is read by the next, ``h_last`` hands a prefill's last row to the
    head entry.
  * Entries never run concurrently, so the intermediates of all entries
    overlap in one region of the pool (each entry keeps its own
    liveness-coloured slots inside it), and likewise the host arena and the
    host-op staging arena.
  * Node indices are global (entry after entry), so the per-layer profiler
    and the layer-name table cover every entry.
  * inference_deinit() also releases the kernel drivers (UIO mappings) on
    Linux, so inference_init() / inference_deinit() can cycle.

The union-level parts of inference.c (weights, kernel instances, run_*
helpers, host-op helpers and tables, init / deinit) come from a
CodeGenerator over ``CombinedGraph`` — the entries' node lists concatenated —
and each run function from the entry's own CodeGenerator (its event stream,
waits, liveness and cache maintenance are exactly those of a single-entry
project of that graph).
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nodes import SchedulerError
from . import CodeGenerator
from ._banners import _banner

__all__ = ("CombinedGraph", "MultiEntryGenerator")


class CombinedGraph:
    """OnnxGraph facade over the entries of a multi-entry project."""

    def __init__(self, entries: Sequence[Tuple[str, object]]):
        self.entries = list(entries)
        self._nodes = [sn for _, g in self.entries for sn in g.nodes]
        self._tensors: Dict[str, object] = {}
        for _, g in self.entries:
            for k, t in g._tensors.items():
                self._tensors.setdefault(k, t)
        self.numeric = {"exp": {}, "host": {}, "state": [], "test_fill": {}}
        for _, g in self.entries:
            for k in ("exp", "host", "test_fill"):
                self.numeric[k].update(g.numeric.get(k, {}))
            for s in g.numeric.get("state", []):
                if s not in self.numeric["state"]:
                    self.numeric["state"].append(s)
        self.opset = max(g.opset for _, g in self.entries)
        self.weights_saturated = {}
        for _, g in self.entries:
            self.weights_saturated.update(getattr(g, "weights_saturated", {}))

    @property
    def nodes(self):
        return self._nodes

    def _dedup(self, attr):
        seen, out = set(), []
        for _, g in self.entries:
            for t in getattr(g, attr):
                if t.c_name not in seen:
                    seen.add(t.c_name)
                    out.append(t)
        return out

    @property
    def weight_tensors(self):
        return self._dedup("weight_tensors")

    @property
    def state_tensors(self):
        return self._dedup("state_tensors")

    @property
    def intermediate_tensors(self):
        return [t for _, g in self.entries for t in g.intermediate_tensors]

    @property
    def host_tensors(self):
        return [t for _, g in self.entries for t in g.host_tensors]

    @property
    def input_tensors(self):
        return [t for _, g in self.entries for t in g.input_tensors]

    @property
    def output_tensors(self):
        return [t for _, g in self.entries for t in g.output_tensors]

    def get_tensor(self, name):
        return self._tensors[name]


def _image_digest(t, dtype) -> str:
    return hashlib.sha1(dtype.dat_bytes(np.asarray(t.emit_data))).hexdigest()


class _CombinedCG(CodeGenerator):
    """CodeGenerator over a CombinedGraph with the multi-entry pool."""

    def __init__(self, multi: "MultiEntryGenerator", **kw):
        self._multi = multi
        self._compact_init = True
        super().__init__(multi.combined, model_path=multi.model_name + ".onnx", **kw)

    def _compute_pool_layout(self):
        return self._multi.pool_layout()

    def _compute_host_layout(self):
        return self._multi.host_layout()

    def _compute_pool_bytes(self) -> int:
        _layout, total = self._multi.pool_layout()
        b = total * self._dtype.bytes_per_elem
        return (b + 4095) & ~4095

    def _init_function(self) -> str:
        """The single-entry init / deinit, plus: every kernel driver that was
        initialised is released in inference_deinit() (UIO munmap + close on
        Linux), so the library can be closed and opened again."""
        src = super()._init_function()
        flags, rel = [], []
        for kd in self._active_kernels:
            call = (f"    rc = {kd.c_type}_Initialize(&{kd.c_var}, {kd.init_param});\n"
                    "    if (rc != 0) goto fail;\n")
            if src.count(call) != 1:
                raise SchedulerError(f"internal: {kd.name} initialisation not found")
            flag = f"s_{kd.name.lower()}_open"
            src = src.replace(call, call + f"    {flag} = 1;\n")
            flags.append(f"static int {flag} = 0;   /* {kd.name} driver initialised */")
            rel.append(f"    if ({flag}) {kd.c_type}_Release(&{kd.c_var});\n"
                       f"    {flag} = 0;\n")
        marker = "    inference_buf_pool_deinit();\n}\n"
        if src.count(marker) != 1:
            raise SchedulerError("internal: inference_deinit() tail not found")
        src = src.replace(marker, "#ifdef __linux__\n" + "".join(rel) + "#endif\n" + marker)
        return "\n".join(flags) + "\n" + src


class MultiEntryGenerator:
    """Generator of one project with several entry graphs.

    entries   [(name, OnnxGraph)] — names become inference_run_<name>();
              tensor names other than shared weights / states must be
              unique across entries (the Llama frontend prefixes them)."""

    def __init__(self, entries: Sequence[Tuple[str, object]], model_name: str,
                 embed_large_weights: bool = False, dtype=None):
        if not entries:
            raise SchedulerError("multi-entry project without entries")
        names = [n for n, _ in entries]
        if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
            raise SchedulerError(f"entry names must be unique C identifiers: {names}")
        self.model_name = model_name
        self.entries = list(entries)
        self._renumber()
        self.renamed = self._dedup_weights(dtype)
        self._check_names()
        self._check_states()
        self.combined = CombinedGraph(self.entries)
        self.cgs: Dict[str, CodeGenerator] = {}
        for name, g in self.entries:
            cg = CodeGenerator(g, model_path=f"{model_name}_{name}.onnx",
                               embed_large_weights=embed_large_weights, dtype=dtype)
            cg._run_fn_name = f"inference_run_{name}"
            self.cgs[name] = cg
        self.cg = _CombinedCG(self, embed_large_weights=embed_large_weights, dtype=dtype)
        self._dtype = self.cg._dtype
        self._pool = None
        self._host = None

    # ---- construction ---------------------------------------------------- #
    def _renumber(self):
        base = 0
        for _, g in self.entries:
            for i, sn in enumerate(g.nodes):
                sn.index = base + i
            base += len(g.nodes)

    def _dedup_weights(self, dtype):
        """Same name + same image -> one buffer; same name + another image
        -> the later entry's tensor is renamed ``<name>@<k>``."""
        from ..dtype import AP_FIXED_16_8
        dtype = dtype or AP_FIXED_16_8
        images: Dict[str, List[str]] = {}          # name -> digests seen, in order
        renamed = {}
        for ename, g in self.entries:
            for t in g.weight_tensors:
                d = _image_digest(t, dtype)
                seen = images.setdefault(t.onnx_name, [])
                if d not in seen:
                    seen.append(d)
                k = seen.index(d)
                if k:
                    old = t.onnx_name
                    t.onnx_name = f"{old}@{k}"
                    renamed.setdefault(old, set()).add(t.onnx_name)
        return {k: sorted(v) for k, v in renamed.items()}

    def _check_names(self):
        owner: Dict[str, str] = {}
        shared = set()
        for _, g in self.entries:
            shared.update(t.c_name for t in g.weight_tensors)
            shared.update(t.c_name for t in g.state_tensors)
        for ename, g in self.entries:
            ts = g.intermediate_tensors + g.host_tensors + g.input_tensors + g.output_tensors
            for t in ts:
                if t.c_name in shared:
                    raise SchedulerError(f"entry {ename}: '{t.onnx_name}' clashes with a "
                                         f"shared weight / state")
                if owner.setdefault(t.c_name, ename) != ename:
                    raise SchedulerError(f"tensor C name '{t.c_name}' used by entries "
                                         f"{owner[t.c_name]} and {ename}")

    def _check_states(self):
        first = {}
        for ename, g in self.entries:
            for t in g.state_tensors:
                f = first.setdefault(t.onnx_name, t)
                if f is t:
                    continue
                if (list(f.shape) != list(t.shape) or f.host != t.host
                        or not np.array_equal(f.exp_full(8), t.exp_full(8))):
                    raise SchedulerError(f"state '{t.onnx_name}' differs between entries")
                if t.init_data is not None:
                    if f.init_data is None:
                        f.init_data = t.init_data
                    elif not np.array_equal(f.init_data, t.init_data):
                        raise SchedulerError(f"state '{t.onnx_name}': conflicting initial values")

    # ---- layouts ----------------------------------------------------------- #
    def pool_layout(self):
        """Weights (deduplicated, sequential), then ONE intermediates region
        in which every entry's own slots start at the same base."""
        if self._pool is None:
            bpe = self._dtype.bytes_per_elem
            a = 64 // bpe
            up = lambda n: (n + a - 1) & ~(a - 1)  # noqa: E731
            layout, off = [], 0
            for t in self.combined.weight_tensors:
                alloc = self.cg._alloc_sizes[t.onnx_name]
                layout.append((t.onnx_name, off, alloc))
                off += up(alloc)
            region = 0
            for name, _g in self.entries:
                inter, total = self.cgs[name]._compute_intermediate_layout()
                layout.extend((n, off + o, al) for n, o, al in inter)
                region = max(region, total)
            self._pool = (layout, off + region)
            self.weights_elems = off
            self.region_elems = region
        return self._pool

    def host_layout(self):
        if self._host is None:
            layout, total = [], 0
            for name, _g in self.entries:
                lay, tot = self.cgs[name]._compute_host_layout()
                layout.extend(lay)
                total = max(total, tot)
            self._host = (layout, total)
        return self._host

    # ---- files ------------------------------------------------------------- #
    def generate_header(self) -> str:
        from ._banners import _file_banner
        cg = self.cg
        lines = cg._header_types(self.combined.input_tensors + self.combined.output_tensors)
        lines.append(_banner("Array size constants"))
        for name, _g in self.entries:
            sz = self.cgs[name]._header_sizes(banner=False)
            if sz:
                lines.append(f"/* entry '{name}' */")
                lines += sz
        lines += cg._header_pool(cg._compute_pool_bytes())
        lines += cg._header_buf_api()
        lines += cg._header_uio()
        lines += cg._header_init()
        lines += cg._header_layers(len(cg._layer_display_names()))
        lines.append(_banner("Entries"))
        lines.append(f"#define INFERENCE_NUM_ENTRIES  {len(self.entries)}u")
        base = 0
        for name, g in self.entries:
            lines.append(f"#define INFERENCE_ENTRY_{name.upper()}_FIRST_LAYER  {base}u")
            base += len(g.nodes)
        lines.append("")
        for name, _g in self.entries:
            lines += self.cgs[name]._header_run()
            lines.append("")
        return "\n".join([_file_banner("inference.h", self.combined, cg._model_path),
                          cg._header_guard_open(), "\n".join(lines),
                          cg._header_guard_close()]) + "\n"

    def generate_source(self) -> str:
        from ._banners import _file_banner
        cg = self.cg
        parts = [_file_banner("inference.c", self.combined, cg._model_path),
                 cg._source_includes()]
        op_defines = cg._source_op_defines()
        if op_defines:
            parts.append(op_defines)
        parts += [cg._weight_arrays(), cg._buffer_declarations(), cg._kernel_instance(),
                  cg._kernel_wait_helper(), cg._layer_names_table(), cg._run_op_helper()]
        host = cg._host_ops_section()
        if host:
            parts.append(host)
        parts.append(cg._init_function())
        for name, _g in self.entries:
            parts.append(self.cgs[name]._inference_function())
        return "\n".join(parts) + "\n"

    def generate_buf_impl(self) -> str:
        return self.cg.generate_buf_impl()

    def generate_cmake(self) -> str:
        return self.cg.generate_cmake()

    def generate_setup_script(self) -> str:
        return self.cg.generate_setup_script()

    @property
    def large_weight_tensors(self):
        return self.cg.large_weight_tensors

    def generate_weight_dat(self, t) -> bytes:
        return self.cg.generate_weight_dat(t)

    def host_table_files(self):
        return self.cg.host_table_files()

    @property
    def _active_kernels(self):
        return self.cg._active_kernels

    # ---- simulation + test harness ---------------------------------------- #
    def initial_states(self) -> Dict[str, np.ndarray]:
        out = {}
        for t in self.combined.state_tensors:
            init = t.init_data
            out[t.onnx_name] = (np.zeros(t.shape) if init is None
                                else np.array(init, np.float64).reshape(t.shape))
        return out

    def simulate_sequence(self, calls: Optional[Sequence[str]] = None) -> Dict[str, dict]:
        """Run the entries in order (default: declaration order), each with the
        test harness's inputs, over shared states.  Returns {entry: arrays}."""
        states = self.initial_states()
        out = {}
        for name in (calls or [n for n, _ in self.entries]):
            cg = self.cgs[name]
            out[name] = cg._forward_pass(cg._build_ramp_inputs(), states=states)
        return out

    def generate_test(self) -> str:
        """test/test_inference.c: inference_init(), every entry once in order
        with the ramp / test_fill inputs, outputs compared bit for bit with
        simulate_sequence(), inference_deinit(); then init / run / deinit
        again (the re-open path) and the first entry's outputs compared."""
        from ._banners import _file_banner
        cg, dtype = self.cg, self._dtype
        sims = self.simulate_sequence()
        decls, body, verify_fns = [], [], []
        first_name = self.entries[0][0]
        for name, g in self.entries:
            ecg = self.cgs[name]
            sim = sims[name]
            args = []
            for t in g.input_tensors + g.output_tensors:
                if t.is_host:
                    decls.append(f"static {ecg._host_c_type(t)} io_{t.c_name}[{t.numel}];")
                    args.append(f"io_{t.c_name}")
                else:
                    decls.append(f"static inference_buf_t *{t.c_name};")
                    args.append(t.c_name)
            chk = [f"static int check_{name}(void)", "{", "    unsigned i;", "    int bad = 0;"]
            for t in g.output_tensors:
                st = ecg._expected_storage(t.onnx_name, sim[t.onnx_name])
                if t.is_host:
                    decls.append(ecg._emit_expected_host_c(t, st).rstrip())
                    chk += [f"    for (i = 0u; i < {t.numel}u; i++)",
                            f"        if (memcmp(&io_{t.c_name}[i], &expected_{t.c_name}[i],"
                            f" sizeof io_{t.c_name}[i]) != 0) {{",
                            f"            if (bad++ < 8) fprintf(stderr, \"FAIL {t.onnx_name}[%u]:"
                            f" got %.9g expected %.9g\\n\", i, (double)io_{t.c_name}[i],"
                            f" (double)expected_{t.c_name}[i]);",
                            "        }"]
                else:
                    if ecg._layouts[t.onnx_name].n_chunks > 1:
                        raise SchedulerError("multi-entry test: strided outputs not supported")
                    decls.append(ecg._emit_expected_c(t.c_name, st).rstrip())
                    chk += [f"    for (i = 0u; i < {t.numel}u; i++)",
                            f"        if (inference_buf_ptr({t.c_name})[i] != expected_{t.c_name}[i]) {{",
                            f"            if (bad++ < 8) fprintf(stderr, \"FAIL {t.onnx_name}[%u]\\n\", i);",
                            "        }"]
            chk += ["    (void)i;", "    return bad;", "}"]
            verify_fns.append("\n".join(chk))
            fill = []
            for t in g.input_tensors:
                if t.is_host:
                    fill += ecg._host_fill_c(t, f"io_{t.c_name}")
                else:
                    rhs = "(Data_t)(i & 0xFFFFu)" if (t.exp is not None or t.is_int is False) \
                        else "(Data_t)i"
                    if t.is_int:
                        rhs = f"(Data_t)(i % {ecg._int_fill_range(t)}u)"
                    elif t.exp is None:
                        rhs = dtype.c_fill_rhs("i")
                    fill.append(f"    for (i = 0u; i < {t.numel}u; i++) "
                                f"inference_buf_ptr({t.c_name})[i] = {rhs};")
            body.append((name, fill, f"    inference_run_{name}({', '.join(args)});"))
        allocs, frees = [], []
        for _name, g in self.entries:
            for t in g.input_tensors + g.output_tensors:
                if not t.is_host:
                    allocs += [f"    {t.c_name} = inference_buf_alloc({t.numel}u);",
                               f"    if (!{t.c_name}) {{ fprintf(stderr, \"alloc failed\\n\"); rc = 1; goto done; }}"]
                    frees.append(f"    inference_buf_free({t.c_name}); {t.c_name} = NULL;")
        init_args = ", ".join(kd.instance_macro for kd in cg._active_kernels)
        run_lines = []
        for name, fill, call in body:
            run_lines += [f"    /* entry '{name}' */"] + fill + [call,
                          f"    if (check_{name}() != 0) {{ fprintf(stderr, \"entry {name}: FAIL\\n\");"
                          f" rc = 1; }} else printf(\"entry {name}: ok\\n\");"]
        first_fill, first_call = body[0][1], body[0][2]
        return (
            _file_banner("test_inference.c", self.combined, cg._model_path) +
            "\n#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\n"
            "#include \"inference.h\"\n\n"
            + "".join(f"#ifndef {kd.instance_macro}\n#  define {kd.instance_macro}  "
                      f"\"{kd.uio_default}\"\n#endif\n" for kd in cg._active_kernels)
            + "\n/* I/O and expected outputs (multi-entry simulation: every entry once,\n"
              " * in declaration order, over shared states) */\n"
            + "\n".join(decls) + "\n\n" + "\n\n".join(verify_fns) + "\n\n"
            "int main(void)\n{\n    unsigned i;\n    int rc = 0, pass;\n\n"
            "    for (pass = 0; pass < 2 && rc == 0; pass++) {\n"
            f"    if (inference_init({init_args}) != 0) {{\n"
            "        fprintf(stderr, \"inference_init() failed\\n\");\n        return 1;\n    }\n"
            + "\n".join(allocs) + "\n"
            "    if (pass == 0) {\n"
            + "\n".join("    " + ln for ln in run_lines) + "\n"
            "    } else {   /* re-open: the first entry again on fresh states */\n"
            + "\n".join("    " + ln for ln in first_fill) + "\n    " + first_call + "\n"
            f"        if (check_{first_name}() != 0) {{ fprintf(stderr, \"re-open: FAIL\\n\"); rc = 1; }}\n"
            "        else printf(\"re-open: ok\\n\");\n"
            "    }\n"
            "    if (rc != 0) goto done;\n"
            "done:\n"
            + "\n".join(frees) + "\n"
            "    inference_deinit();\n    }\n"
            "    printf(rc == 0 ? \"test_inference PASSED\\n\" : \"test_inference FAILED\\n\");\n"
            "    return rc;\n}\n"
        )

    # host_emu.build_and_run() compatibility
    @property
    def large_expected_tensors(self):
        return []

    def write_project(self, out_dir: str, runtime_dir: Optional[str] = None,
                      weights: bool = True) -> dict:
        """Write CMakeLists.txt, include/, src/, test/, scripts/ and (unless
        weights=False) weights/*.dat.  Returns a summary."""
        import shutil

        def w(rel, text):
            p = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
        w("CMakeLists.txt", self.generate_cmake())
        w("include/inference.h", self.generate_header())
        w("src/inference.c", self.generate_source())
        w("src/inference_buf.c", self.generate_buf_impl())
        w("scripts/check_inference_setup.sh", self.generate_setup_script())
        w("test/test_inference.c", self.generate_test())
        rt = runtime_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "runtime")
        for src_name, rel in (("inference_prof.h", "include/inference_prof.h"),
                              ("inference_prof.c", "src/inference_prof.c"),
                              ("inference_ddr.h", "include/inference_ddr.h"),
                              ("inference_ddr.c", "src/inference_ddr.c"),
                              ("inference_ddr_backend.h", "src/inference_ddr_backend.h"),
                              ("ddr/zuplus_apm.c", "src/ddr/zuplus_apm.c")):
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(rt, src_name), dst)
        nbytes = 0
        if weights:
            wd = os.path.join(out_dir, "weights")
            os.makedirs(wd, exist_ok=True)
            for t in self.large_weight_tensors:
                data = self.generate_weight_dat(t)
                nbytes += len(data)
                with open(os.path.join(wd, f"{t.c_name}.dat"), "wb") as f:
                    f.write(data)
            for name, tb in self.host_table_files():
                data = tb.dat_bytes()
                nbytes += len(data)
                with open(os.path.join(wd, f"{name}.dat"), "wb") as f:
                    f.write(data)
        return self.summary(weight_file_bytes=nbytes)

    def summary(self, weight_file_bytes: int = 0) -> dict:
        layout, total = self.pool_layout()
        bpe = self._dtype.bytes_per_elem
        host_layout, host_bytes = self.host_layout()
        tables = sum(tb.nbytes for _n, tb in self.host_table_files())
        return {
            "entries": [n for n, _ in self.entries],
            "nodes": {n: len(g.nodes) for n, g in self.entries},
            "pool_bytes": total * bpe,
            "weights_bytes": self.weights_elems * bpe,
            "intermediate_region_bytes": self.region_elems * bpe,
            "weights": len(self.combined.weight_tensors),
            "renamed_weights": self.renamed,
            "host_arena_bytes": host_bytes,
            "host_table_bytes": tables,
            "state_bytes": sum(t.numel * self.cg._HOST_ELEM_BYTES[t.host]
                               for t in self.combined.state_tensors if t.is_host),
            "weight_file_bytes": weight_file_bytes,
            "active_kernels": [kd.name for kd in self._active_kernels],
        }
