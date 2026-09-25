"""Cache-coherency audit of the emitted inference_run() (BERT_PLAN phase 2B).

The XRT buffer objects are mapped CACHEABLE on the board and the PL kernels'
AXI masters do not snoop the CPU caches, so every CPU <-> kernel hand-off
must be bracketed by the right cache maintenance (Linux DMA-API rules):

  * a range the CPU wrote must be flushed (inference_buf_sync_to_device)
    before a kernel READS it — else the kernel reads stale DDR;
  * ... and before a kernel WRITES it — else a dirty line evicted later
    overwrites the kernel's result;
  * a range a kernel wrote must be invalidated (inference_buf_sync_from_device)
    AFTER the kernel's lane drained and before the CPU reads it — else the
    CPU reads stale (possibly speculatively prefetched) lines;
  * the CPU never touches a buffer a kernel in flight reads or writes.

``CoherencyChecker`` walks the generated C of inference_run() in order —
kernel starts (``run_*`` argument roles), ``kernel_wait``, host-op blocks
(``host_in`` / ``host_out_done`` / raw ``inference_buf_ptr`` accesses of the
node's inputs and output), the input-to-output reshape copies, pointer
redirects of output aliases, and every sync call — with a per-buffer
dirty / stale model (Slice views are sub-ranges of their root), starting
from "the caller wrote every input and output buffer", and reports every
violated rule.  It runs over every generated test model, the tiny BERT
fixtures, the SpaceToDepth stems and a purpose-built kernel -> host ->
kernel -> output model; mutated sources (one sync removed) must fail.
"""

import glob
import os
import re
import tempfile
import unittest

import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
import numpy as np
from onnx import TensorProto

import gen_bert_models as gbm
from src.codegen import CodeGenerator
from src.graph import OnnxGraph
from src.host_nodes import HostNode
from src.nodes import SpaceToDepthNode

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")

# run_* helper -> (lane, read-arg positions, write-arg positions, synchronous)
_KERNEL_CALLS = {
    "run_op":        ("KERNEL_VECTOROP", (0, 1), (2,), False),
    "run_op_act":    ("KERNEL_VECTOROP", (0, 1), (2,), False),
    "run_matmul":    ("KERNEL_MATMUL",   (0, 1), (2,), False),
    "run_matmul_at": ("KERNEL_MATMUL",   (0, 2), (4,), True),
    "run_conv":      ("KERNEL_CONV",     (0, 1, 2), (3,), False),
    # MatMul on ConvKernel, one call per batch item (x, x_off, weight, w_off,
    # y, y_off, ...); asynchronous like run_conv — the next call waits.
    "run_conv_at":   ("KERNEL_CONV",     (0, 2), (4,), False),
    "run_pool":      ("KERNEL_POOL",     (0,), (1,), False),
}

_TOKEN = re.compile(
    r"(?P<flush>inference_buf_sync_to_device)\(\s*(?P<fb>\w+)\s*\)"
    r"|(?P<inval>inference_buf_sync_from_device)\(\s*(?P<ib>\w+)\s*\)"
    # per-batch-item loops (MatMul on ConvKernel): `if (_i) kernel_wait(L);`
    # only runs from the second iteration on, i.e. after the loop's own
    # previous call — a no-op when nothing of that lane is in flight yet.
    r"|if\s*\(\s*_i\s*\)\s*kernel_wait\((?P<cwait>KERNEL_\w+)\)"
    r"|kernel_wait\((?P<wait>KERNEL_\w+)\)"
    r"|\b(?P<run>run_(?:op_act|op|matmul_at|matmul|conv_at|conv|pool))\((?P<args>[^;]*)\);"
    r"|INFERENCE_PROF_BEGIN\((?P<begin>\d+)u\)"
    r"|INFERENCE_PROF_END\((?P<end>\d+)u\)"
    r"|\bhost_in\(\s*(?P<hin>\w+)\s*,"
    r"|\bhost_out\(\s*(?P<hout>\w+)\s*,"
    r"|\bhost_out_done\(\s*(?P<hdone>\w+)\s*,"
    r"|memcpy\(inference_buf_ptr\((?P<cpy_dst>\w+)\),\s*inference_buf_ptr\((?P<cpy_src>\w+)\)"
    r"|inference_buf_ptr\((?P<ptr>\w+)\)"
    r"|^\s*(?:inference_buf_t\s*\*\s*)?(?P<lhs>\w+)\s*=\s*(?P<rhs>\w+)\s*;",
    re.M | re.S)


def _split_args(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def run_body(src):
    """Text of the inference_run() function body."""
    i = src.index("\nvoid inference_run(")
    j = src.index("\n{\n", i)
    k = src.index("\n}\n", j)
    return src[j + 3:k]


class CoherencyChecker:
    """Per-buffer cache-state model over the emitted inference_run()."""

    def __init__(self, cg, src=None):
        self.cg = cg
        self.src = src if src is not None else cg.generate_source()
        g = cg._graph
        self.nodes = {sn.index: sn for sn in g.nodes}
        tensors = {t.onnx_name: t for t in (g.input_tensors + g.output_tensors
                                            + g.weight_tensors + g.intermediate_tensors)}
        c_of = {n: t.c_name for n, t in tensors.items()}
        # pointer variable -> buffer id; views -> parent buffer id
        self.ptr = {t.c_name: t.c_name for t in tensors.values()}
        self.parent = {}
        for onnx_name, (root_c, _off, _cnt) in cg._view_aliases.items():
            self.parent[c_of[onnx_name]] = root_c
        aliases = {c_of[n]: src_c for n, src_c in cg._reshape_aliases.items()
                   if n not in cg._run_reshape_aliases}
        for c in aliases:
            seen, cur = set(), c
            while cur in aliases and cur not in seen:
                seen.add(cur)
                cur = aliases[cur]
            self.ptr[c] = cur
        self.inputs = [t.c_name for t in g.input_tensors]
        self.outputs = [t.c_name for t in g.output_tensors]
        self.cpu_nodes = {sn.index for sn in g.nodes
                          if isinstance(sn, (HostNode, SpaceToDepthNode))}

    # --- buffer relations ------------------------------------------------ #
    def _root(self, b):
        return self.parent.get(b, b)

    def _family(self, b):
        """b plus every buffer overlapping it (root + its views)."""
        r = self._root(b)
        return {r} | {v for v, p in self.parent.items() if p == r}

    def _overlaps(self, a, b):
        return a == b or self._root(a) == b or self._root(b) == a

    # --- the walk -------------------------------------------------------- #
    def check(self):
        errors = []
        body = run_body(self.src)
        dirty = set()           # CPU-written, not flushed since
        stale = set()           # kernel-written, not invalidated (after drain) since
        in_flight = {}          # lane -> (reads, writes)
        for b in self.inputs + self.outputs:
            dirty.add(self.ptr[b])          # the caller wrote them (fill / memset)
        cur_node = None

        def buf(name):
            return self.ptr.get(name, name)

        def busy(b, writes_only):
            for lane, (rd, wr) in in_flight.items():
                for x in wr + ([] if writes_only else rd):
                    if self._overlaps(b, x):
                        return lane
            return None

        def cpu_read(b, where):
            if b in stale or self._root(b) in stale:
                errors.append(f"{where}: CPU reads '{b}' written by a kernel without "
                              f"an invalidate after the kernel drained")
            lane = busy(b, writes_only=True)
            if lane:
                errors.append(f"{where}: CPU reads '{b}' while {lane} still writes it")

        def cpu_write(b, where):
            lane = busy(b, writes_only=False)
            if lane:
                errors.append(f"{where}: CPU writes '{b}' while {lane} still uses it")
            dirty.update(self._family(b))

        def flush(b):
            fam = self._family(b) if b == self._root(b) else {b}
            dirty.difference_update(fam)

        def inval(b, where):
            lane = busy(b, writes_only=True)
            if lane:
                errors.append(f"{where}: invalidate of '{b}' while {lane} still writes "
                              f"it (does not count)")
                return
            fam = self._family(b) if b == self._root(b) else {b}
            stale.difference_update(fam)

        for m in _TOKEN.finditer(body):
            line = body.count("\n", 0, m.start()) + 1
            where = f"line {line}" + (f" [node {cur_node}]" if cur_node is not None else "")
            if m.group("begin"):
                cur_node = int(m.group("begin"))
            elif m.group("end"):
                cur_node = None
            elif m.group("flush"):
                flush(buf(m.group("fb")))
            elif m.group("inval"):
                inval(buf(m.group("ib")), where)
            elif m.group("cwait"):
                in_flight.pop(m.group("cwait"), None)
            elif m.group("wait"):
                lane = m.group("wait")
                if lane not in in_flight:
                    errors.append(f"{where}: kernel_wait({lane}) with nothing in flight")
                in_flight.pop(lane, None)
            elif m.group("run"):
                lane, rpos, wpos, sync = _KERNEL_CALLS[m.group("run")]
                args = _split_args(m.group("args"))
                rd = [buf(args[i]) for i in rpos if re.fullmatch(r"\w+", args[i])
                      and args[i] != "NULL"]
                wr = [buf(args[i]) for i in wpos]
                if lane in in_flight:
                    errors.append(f"{where}: {m.group('run')} on busy lane {lane}")
                for b in rd:
                    if self._family(b) & dirty:
                        errors.append(f"{where}: {lane} reads '{b}' with CPU-dirty lines "
                                      f"(no flush after the CPU wrote it)")
                for b in wr:
                    if self._family(b) & dirty:
                        errors.append(f"{where}: {lane} writes '{b}' over CPU-dirty lines "
                                      f"(a later eviction would overwrite the result)")
                    if busy(b, writes_only=False):
                        errors.append(f"{where}: {lane} writes '{b}' still used by "
                                      f"{busy(b, writes_only=False)}")
                    stale.update(self._family(b))
                if not sync:
                    in_flight[lane] = (rd, wr)
            elif m.group("hin"):
                cpu_read(buf(m.group("hin")), where)
            elif m.group("hout"):
                cpu_write(buf(m.group("hout")), where)     # the helper writes through it
            elif m.group("hdone"):
                flush(buf(m.group("hdone")))  # host_out_done flushes (unit-tested)
            elif m.group("cpy_dst"):
                cpu_read(buf(m.group("cpy_src")), where)
                cpu_write(buf(m.group("cpy_dst")), where)
            elif m.group("ptr"):
                name = m.group("ptr")
                sn = self.nodes.get(cur_node)
                if sn is None or cur_node not in self.cpu_nodes:
                    errors.append(f"{where}: CPU access to '{name}' outside a host op")
                    continue
                if name in {t.c_name for t in sn.inputs}:
                    cpu_read(buf(name), where)
                elif name == sn.output.c_name:
                    cpu_write(buf(name), where)
                else:
                    errors.append(f"{where}: host op touches unrelated buffer '{name}'")
            elif m.group("lhs"):
                lhs, rhs = m.group("lhs"), m.group("rhs")
                if rhs != "NULL" and rhs in self.ptr:
                    self.ptr[lhs] = self.ptr[rhs]
        for lane in in_flight:
            errors.append(f"end: {lane} still in flight at return")
        for o in self.outputs:
            b = buf(o)
            if b in stale or self._root(b) in stale:
                errors.append(f"end: output '{o}' not invalidated after its kernel wrote it")
        return errors


def _save(path, nodes, inputs, outputs, inits=(), opset=13):
    g = oh.make_graph(nodes, "m", inputs, outputs, initializer=list(inits))
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def _mixed_model(path):
    """X -> LayerNorm (host, reads a caller input) -> MatMul (kernel reads a
    host output) -> Softmax (host reads a kernel output) -> Add (kernel) ->
    Z (graph output written by a kernel); Transpose(Add) -> T (host reads a
    kernel-written graph output, writes a graph output); Relu(Softmax) -> R
    (kernel, second consumer of the host output)."""
    rng = np.random.default_rng(0)
    vi = oh.make_tensor_value_info
    inits = [nph.from_array(rng.normal(0, .3, (16, 32)).astype(np.float32), "W"),
             nph.from_array(rng.normal(0, .3, (32,)).astype(np.float32), "B"),
             nph.from_array(np.ones(16, np.float32), "g"),
             nph.from_array(np.zeros(16, np.float32), "b")]
    nodes = [oh.make_node("LayerNormalization", ["X", "g", "b"], ["L"], axis=-1),
             oh.make_node("MatMul", ["L", "W"], ["M"]),
             oh.make_node("Softmax", ["M"], ["S"], axis=-1),
             oh.make_node("Add", ["S", "B"], ["Z"]),
             oh.make_node("Transpose", ["Z"], ["T"], perm=[1, 0]),
             oh.make_node("Relu", ["S"], ["R"])]
    return _save(path, nodes, [vi("X", TensorProto.FLOAT, [8, 16])],
                 [vi("Z", TensorProto.FLOAT, [8, 32]), vi("T", TensorProto.FLOAT, [32, 8]),
                  vi("R", TensorProto.FLOAT, [8, 32])], inits, opset=17)


def _gen(path, **kw):
    kw.setdefault("fuse_act", True)
    kw.setdefault("s2d_stem", True)
    g = OnnxGraph(path, **kw)
    return CodeGenerator(g, model_path=path)


class TestMixedModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cg = _gen(_mixed_model(os.path.join(cls._tmp.name, "mixed.onnx")))
        cls.src = cls.cg.generate_source()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_partition(self):
        kinds = [type(sn).__name__ for sn in self.cg._graph.nodes]
        self.assertEqual(kinds, ["LayerNormNode", "MatmulNode", "SoftmaxNode",
                                 "ScheduledNode", "TransposeNode", "ScheduledNode"])

    def test_sync_sequence_is_coherent(self):
        self.assertEqual(CoherencyChecker(self.cg, self.src).check(), [])

    def test_emitted_sequence(self):
        """The hand-offs in order: inputs flushed and outputs cleaned first;
        the host op reading the MatMul output invalidates it after the Matmul
        lane drained; host outputs go through host_out_done (flush); the
        outputs are invalidated after the final drain."""
        body = run_body(self.src)
        seq = [ln.strip() for ln in body.splitlines()
               if re.search(r"sync_|kernel_wait|run_\w+\(|host_in\(|host_out_done\(", ln)]
        seq = [re.sub(r",.*", "", s) for s in seq]
        self.assertEqual(seq[:4], ["inference_buf_sync_to_device(X);",
                                   "inference_buf_sync_to_device(Z);",
                                   "inference_buf_sync_to_device(T);",
                                   "inference_buf_sync_to_device(R);"])
        i_mm = seq.index("run_matmul(L")
        self.assertLess(seq.index("host_out_done(L"), i_mm)
        self.assertLess(seq.index("in0 = host_in(X"), seq.index("host_out_done(L"))
        i_wait = seq.index("kernel_wait(KERNEL_MATMUL);")
        i_inv = seq.index("inference_buf_sync_from_device(M);  /* written by a kernel */")
        self.assertLess(i_mm, i_wait)
        self.assertLess(i_wait, i_inv)
        self.assertLess(i_inv, seq.index("in0 = host_in(M"))
        # Transpose reads Z (a kernel-written graph output) after the VectorOP drain
        i_z = seq.index("inference_buf_sync_from_device(Z);  /* written by a kernel */")
        self.assertLess(seq.index("kernel_wait(KERNEL_VECTOROP);"), i_z)
        self.assertLess(i_z, seq.index("in0 = host_in(Z"))
        tail = seq[-3:]
        self.assertEqual(sorted(tail), ["inference_buf_sync_from_device(R);",
                                        "inference_buf_sync_from_device(T);",
                                        "inference_buf_sync_from_device(Z);"])

    def _mutant(self, old, new=""):
        self.assertIn(old, self.src)
        return CoherencyChecker(self.cg, self.src.replace(old, new, 1)).check()

    def test_checker_catches_missing_invalidate(self):
        errs = self._mutant("        inference_buf_sync_from_device(M);  /* written by a kernel */\n")
        self.assertTrue(any("CPU reads 'M'" in e for e in errs), errs)

    def test_checker_catches_missing_output_clean(self):
        errs = self._mutant("    inference_buf_sync_to_device(R);\n")
        self.assertTrue(any("writes 'R' over CPU-dirty lines" in e for e in errs), errs)

    def test_checker_catches_missing_output_invalidate(self):
        errs = self._mutant("    inference_buf_sync_from_device(R);\n")
        self.assertTrue(any("output 'R' not invalidated" in e for e in errs), errs)
        # Z was already invalidated (after the VectorOP drain) by the Transpose
        # that reads it, so its final invalidate is redundant, not required
        self.assertEqual(self._mutant("    inference_buf_sync_from_device(Z);\n"), [])

    def test_checker_catches_missing_input_flush(self):
        errs = self._mutant("    inference_buf_sync_to_device(X);\n")
        # X is read by the CPU only (LayerNorm) -> still coherent; the flush of
        # the host op's output is what the kernel needs:
        self.assertEqual(errs, [])
        line = re.search(r"\n *host_out_done\(L,[^\n]*", self.src).group(0)
        errs = self._mutant(line)
        self.assertTrue(any("reads 'L' with CPU-dirty lines" in e for e in errs), errs)

    def test_checker_catches_early_invalidate(self):
        # invalidate M before the Matmul lane drained
        body = run_body(self.src)
        i = body.index("kernel_wait(KERNEL_MATMUL);")
        inv = "        inference_buf_sync_from_device(M);  /* written by a kernel */\n"
        moved = body.replace(inv, "")
        moved = moved[:i] + inv.strip() + "\n    " + moved[i:]
        errs = CoherencyChecker(self.cg, self.src.replace(body, moved)).check()
        self.assertTrue(any("does not count" in e for e in errs), errs)


class TestAllModels(unittest.TestCase):
    """Every generated test model, the tiny BERT fixtures and the stems."""

    def _check(self, path, **kw):
        cg = _gen(path, **kw)
        errs = CoherencyChecker(cg).check()
        self.assertEqual(errs, [], os.path.basename(path))

    def test_generated_models(self):
        paths = sorted(glob.glob(os.path.join(_MODELS, "*.onnx")))
        if not paths:
            self.skipTest("run test/gen_all_models.py first")
        n = 0
        for p in paths:
            try:
                cg = _gen(p)
            except Exception:      # models that exist to test scheduler errors
                continue
            with self.subTest(model=os.path.basename(p)):
                self.assertEqual(CoherencyChecker(cg).check(), [])
            n += 1
        self.assertGreater(n, 100)

    def test_bert_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            for i, (fname, h, nh, nl, s, gelu, style, opset) in enumerate(gbm.VARIANTS):
                p = gbm.make_bert(os.path.join(td, fname), h, nh, nl, s, gelu, style, opset,
                                  seed=i)
                with self.subTest(model=fname):
                    self._check(p)

    def test_s2d_stems(self):
        """SpaceToDepth host op reading a caller input (stem7) and a
        kernel-written source (Relu -> SpaceToDepth, stem7_pre; native op)."""
        import test_s2d_stem as t
        with tempfile.TemporaryDirectory() as td:
            j = lambda n: os.path.join(td, n)  # noqa: E731
            paths = [t._stem_model(j("stem7.onnx"), "stem7", 3, 16, 16, 7, 3),
                     t._stem_model(j("stem7_pre.onnx"), "stem7_pre", 3, 16, 16, 7, 3,
                                   pre_relu=True),
                     t._save([oh.make_node("Relu", ["X"], ["r"]),
                              oh.make_node("SpaceToDepth", ["r"], ["s"], blocksize=2),
                              oh.make_node("Relu", ["s"], ["Y"])],
                             [t._f32("X", [1, 2, 6, 4])], [t._f32("Y", [1, 8, 3, 2])], [],
                             "native_s2d", j("native.onnx"))]
            for p in paths:
                cg = _gen(p)
                self.assertTrue(any(isinstance(sn, SpaceToDepthNode) for sn in cg._graph.nodes))
                with self.subTest(model=os.path.basename(p)):
                    self.assertEqual(CoherencyChecker(cg).check(), [])
            # the kernel-written source must be invalidated: remove it -> caught
            cg = _gen(paths[1])
            src = cg.generate_source()
            line = re.search(r"\n *inference_buf_sync_from_device\(relu_X\);", src).group(0)
            errs = CoherencyChecker(cg, src.replace(line, "", 1)).check()
            self.assertTrue(any("CPU reads 'relu_X'" in e for e in errs), errs)

if __name__ == "__main__":
    unittest.main()
