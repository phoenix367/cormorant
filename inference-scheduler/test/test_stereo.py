"""The LightStereo-S frontend (src/stereo.py, doc/plans/STEREO_PLAN.md): the
torch-free checkpoint reader, the stripe split, the polyphase deconvs, the
exponent solver, and the whole network with random weights of the real
shapes compiled on the host and compared bit for bit with the simulation
(host_emu).  No downloaded asset is needed: the formats are the checked-in
demo/stereo_depth/lightstereo_s_formats.json."""

import collections
import io
import os
import pickle
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

import numpy as np

import host_emu
from src.graph import OnnxGraph
from src.codegen import CodeGenerator
from src.stereo import (LightStereoFrontend, StereoError, load_checkpoint, load_formats, polyphase,
                        stripe_pieces)

FORMATS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "demo", "stereo_depth", "lightstereo_s_formats.json")


def lightstereo_s_shapes():
    """key -> shape of LightStereo-S's state dict (BatchNorm: weight, bias, running_mean, running_var)."""
    s = {}

    def bn(p, c):
        for k in ("weight", "bias", "running_mean", "running_var"):
            s[f"{p}.{k}"] = (c,)

    def conv(p, shape, bias=False):
        s[p + ".weight"] = shape
        if bias:
            s[p + ".bias"] = (shape[0],)
    conv("backbone.conv_stem", (32, 3, 3, 3))
    bn("backbone.bn1", 32)
    conv("backbone.block0.0.conv_dw", (32, 1, 3, 3))
    bn("backbone.block0.0.bn1", 32)
    conv("backbone.block0.0.conv_pw", (16, 32, 1, 1))
    bn("backbone.block0.0.bn2", 16)
    c = 16
    stages = [(1, 2, 6, 24), (2, 3, 6, 32), (3, 4, 6, 64), (4, 3, 6, 96), (5, 3, 6, 160)]
    for st, nb, e, oc in stages:
        for j in range(nb):
            p = (f"backbone.block{st}.{j}" if st <= 2 else f"backbone.block3.{st}.{j}" if st in (3, 4)
                 else f"backbone.block4.{j}")
            mid = c * e
            conv(p + ".conv_pw", (mid, c, 1, 1))
            bn(p + ".bn1", mid)
            conv(p + ".conv_dw", (mid, 1, 3, 3))
            bn(p + ".bn2", mid)
            conv(p + ".conv_pwl", (oc, mid, 1, 1))
            bn(p + ".bn3", oc)
            c = oc
    for name, cl, ch in (("backbone.fpn_layer4", 160, 96), ("backbone.fpn_layer3", 96, 32),
                         ("backbone.fpn_layer2", 32, 24), ("refine_2", 24, 16)):
        conv(name + ".deconv.block.0", (cl, ch, 4, 4))
        bn(name + ".deconv.block.1", ch)
        conv(name + ".conv.block.0", (ch, 2 * ch, 3, 3))
        bn(name + ".conv.block.1", ch)
    conv("backbone.out_conv.block.0", (24, 24, 3, 3))

    def mv2(p, i, o):
        h = i * 4
        conv(p + ".pwconv.0", (h, i, 1, 1))
        bn(p + ".pwconv.1", h)
        conv(p + ".dwconv.0", (h, 1, 3, 3))
        bn(p + ".dwconv.1", h)
        conv(p + ".pwliner.0", (o, h, 1, 1))
        bn(p + ".pwliner.1", o)
    for p, i, o in (("conv0.0", 48, 48), ("conv1", 48, 96), ("conv2.0", 96, 96), ("conv3", 96, 192),
                    ("conv4.0", 192, 192), ("conv4.1", 192, 192), ("conv4.2", 192, 192),
                    ("redir1", 48, 48), ("redir2", 96, 96)):
        mv2("cost_agg." + p, i, o)
    conv("cost_agg.conv5.0", (192, 96, 3, 3))
    bn("cost_agg.conv5.1", 96)
    conv("cost_agg.conv6.0", (96, 48, 3, 3))
    bn("cost_agg.conv6.1", 48)
    for a, dim, img in (("att0", 48, 24), ("att2", 96, 32), ("att4", 192, 96)):
        p = "cost_agg." + a
        conv(p + ".conv0", (dim, img, 1, 1), True)
        for i, k in enumerate((7, 11, 21)):
            conv(f"{p}.conv{i}_1", (dim, 1, 1, k), True)
            conv(f"{p}.conv{i}_2", (dim, 1, k, 1), True)
        conv(p + ".conv3", (dim, dim, 1, 1), True)
    conv("refine_1.0.block.0", (24, 24, 3, 3))
    conv("refine_1.1.block.0", (24, 24, 3, 3))
    conv("stem_2.0.block.0", (16, 3, 3, 3))
    bn("stem_2.0.block.1", 16)
    conv("stem_2.1.block.0", (16, 16, 3, 3))
    bn("stem_2.1.block.1", 16)
    conv("refine_3.block.0", (16, 9, 4, 4))
    return s


def random_state(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in lightstereo_s_shapes().items():
        if k.endswith("running_var"):
            out[k] = rng.uniform(0.5, 2.0, s).astype(np.float32)
        elif k.endswith(("running_mean", "bias")):
            out[k] = (rng.standard_normal(s) * 0.1).astype(np.float32)
        elif len(s) == 1:
            out[k] = rng.uniform(0.5, 1.5, s).astype(np.float32)
        else:
            fan = int(np.prod(s[1:]))
            out[k] = (rng.standard_normal(s) / np.sqrt(fan)).astype(np.float32)
    return out


def conv_transpose(x, w, stride, pad, out_pad):
    """numpy ConvTranspose2d: x [Cin][H][W], w [Cin][Cout][k][k]."""
    cin, h, wd = x.shape
    k = w.shape[2]
    oh = (h - 1) * stride - 2 * pad + k + out_pad
    full = np.zeros((w.shape[1], (h - 1) * stride + k + out_pad, (wd - 1) * stride + k + out_pad))
    for i in range(h):
        for j in range(wd):
            full[:, i * stride:i * stride + k, j * stride:j * stride + k] += np.einsum("c,cokl->okl", x[:, i, j], w)
    return full[:, pad:pad + oh, pad:pad + oh * wd // h]


def conv2d(x, w, pads):
    """numpy Conv2d, stride 1: x [C][H][W], w [O][C][kh][kw], pads (top, left, bottom, right)."""
    xp = np.pad(x, ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    kh, kw = w.shape[2:]
    oh, ow = xp.shape[1] - kh + 1, xp.shape[2] - kw + 1
    y = np.zeros((w.shape[0], oh, ow))
    for a in range(kh):
        for b in range(kw):
            y += np.einsum("chw,oc->ohw", xp[:, a:a + oh, b:b + ow], w[:, :, a, b])
    return y


class TestStripePieces(unittest.TestCase):
    def test_cover(self):
        for k in (7, 11, 21):
            for span in (16, 64):
                pieces = stripe_pieces(k, span)
                taps = sorted(t0 + d * a for t0, d, n in pieces for a in range(n))
                self.assertEqual(taps, list(range(k)))
                for t0, d, n in pieces:
                    self.assertLessEqual(n, 7)
                    self.assertLessEqual(t0, k // 2)
                    self.assertGreaterEqual(t0 + d * (n - 1), k // 2)
                    self.assertLessEqual(d * (n - 1) + 1, span)
        self.assertEqual(len(stripe_pieces(11, 16)), 2)
        self.assertEqual(len(stripe_pieces(21, 64)), 3)
        self.assertEqual(len(stripe_pieces(21, 16)), 6)

    def test_pieces_sum_to_the_stripe(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(40)
        for k in (11, 21):
            w = rng.standard_normal(k)
            want = np.convolve(np.pad(x, k // 2), w[::-1], "valid")
            got = np.zeros(40)
            for t0, d, n in stripe_pieces(k, 16):
                for a in range(n):
                    t = t0 + d * a
                    got += w[t] * np.pad(x, k // 2)[t:t + 40]
            np.testing.assert_allclose(got, want, atol=1e-12)


class TestPolyphase(unittest.TestCase):
    def check(self, k, pad, out_pad):
        rng = np.random.default_rng(k)
        x = rng.standard_normal((5, 6, 7))
        wt = rng.standard_normal((5, 3, k, k))
        want = conv_transpose(x, wt, 2, pad, out_pad)
        wp, pads = polyphase(wt, k, pad, out_pad)
        y = conv2d(x, wp, pads)                                        # [4*Cout][H][W]
        h, w = x.shape[1:]
        got = y.reshape(2, 2, 3, h, w).transpose(2, 3, 0, 4, 1).reshape(3, 2 * h, 2 * w)
        np.testing.assert_allclose(got, want, atol=1e-10)

    def test_k4(self):
        self.check(4, 1, 0)

    def test_k3_output_padding(self):
        self.check(3, 1, 1)


class TestCheckpoint(unittest.TestCase):
    def test_torch_zip_without_torch(self):
        """A torch.save zip (data.pkl + data/<key>) read with zipfile + pickle."""
        arrs = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
                "a.bias": np.array([1.5, -2.0], np.float32),
                "bn.num_batches_tracked": np.array(7, np.int64)}
        mods = {}
        for name in ("torch", "torch._utils"):
            mods[name] = sys.modules.get(name)
            sys.modules[name] = types.ModuleType(name)

        def _rebuild_tensor_v2(*a):
            return None
        sys.modules["torch._utils"]._rebuild_tensor_v2 = _rebuild_tensor_v2
        FloatStorage = type("FloatStorage", (), {"__module__": "torch"})
        LongStorage = type("LongStorage", (), {"__module__": "torch"})
        sys.modules["torch"].FloatStorage = FloatStorage
        sys.modules["torch"].LongStorage = LongStorage
        _rebuild_tensor_v2.__module__ = "torch._utils"
        _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
        try:
            class T:
                def __init__(self, key, a):
                    self.key, self.a = key, a

                def __reduce__(self):
                    a = self.a
                    st = tuple(s // a.itemsize for s in a.strides)
                    return (_rebuild_tensor_v2, (("storage", self.key, a), 0, a.shape, st, False,
                                                 collections.OrderedDict()))

            class P(pickle.Pickler):
                def persistent_id(self, obj):
                    if isinstance(obj, tuple) and len(obj) == 3 and obj and obj[0] == "storage":
                        _, key, a = obj
                        return ("storage", FloatStorage if a.dtype == np.float32 else LongStorage, key, "cpu",
                                a.size)
                    return None
            buf = io.BytesIO()
            state = collections.OrderedDict((k, T(str(i), a)) for i, (k, a) in enumerate(arrs.items()))
            P(buf, protocol=2).dump({"model_state": state})
            d = tempfile.mkdtemp()
            path = os.path.join(d, "ck.pt")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("archive/data.pkl", buf.getvalue())
                for i, a in enumerate(arrs.values()):
                    z.writestr(f"archive/data/{i}", a.tobytes())
        finally:
            for name, m in mods.items():
                if m is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = m
        sd = load_checkpoint(path)
        shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(sorted(sd), ["a.bias", "a.weight"])
        np.testing.assert_array_equal(sd["a.weight"], arrs["a.weight"])
        np.testing.assert_array_equal(sd["a.bias"], arrs["a.bias"])


class TestFrontend(unittest.TestCase):
    H, W = 64, 256
    PER_CHANNEL = None

    @classmethod
    def setUpClass(cls):
        cls.fe = LightStereoFrontend(random_state(), load_formats(FORMATS), cls.H, cls.W,
                                     per_channel=cls.PER_CHANNEL)
        cls.model = cls.fe.build()
        cls.g = OnnxGraph(cls.model, fuse_act=False, s2d_stem=False, fc_conv="off", matmul_on_conv="off")

    def test_partition(self):
        kinds = collections.Counter(type(sn).__name__ for sn in self.g.nodes)
        self.assertEqual(kinds["StereoCorrelationNode"], 1)
        self.assertEqual(kinds["StereoSoftmaxNode"], 2)
        self.assertEqual(kinds["StereoInstanceNormNode"], 4)
        self.assertEqual(kinds["StereoUpsampleNode"], 1)
        self.assertEqual(kinds["MatmulNode"], 0)              # the regression runs in StereoUpsample
        self.assertGreater(kinds["ConvNode"], 150)
        self.assertGreater(kinds["StereoVopNode"], 100)
        self.assertEqual(self.g.output_tensors[0].host, "f32")

    def test_exponent_rules(self):
        e = self.fe.exponents
        self.assertEqual(e["left"], e["right"])
        for sn in self.g.nodes:
            if type(sn).__name__ == "StereoVopNode" and sn.op == 5:
                self.assertEqual(int(sn.output.exp), 8)

    def test_sizes(self):
        with self.assertRaises(StereoError):
            LightStereoFrontend(random_state(), load_formats(FORMATS), 60, 256)

    def test_host_emu_bit_exact(self):
        if host_emu.which_cc() is None:
            self.fail("no C compiler")
        cg = CodeGenerator(self.g, model_path="lightstereo.onnx")
        d = tempfile.mkdtemp(prefix="stereo_fe_emu_")
        try:
            rc, out = host_emu.build_and_run(cg, d, timeout=900)
            self.assertEqual(rc, 0, out[-3000:])
            self.assertEqual(host_emu.failures(out), [], out[-3000:])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestFrontendPerChannel(TestFrontend):
    """Per-channel weight exponents on every conv that gains (chexp + rescale)."""
    PER_CHANNEL = 0.0

    def test_rescales(self):
        self.assertGreater(len(self.fe.chexp), 20)
        names = {sn.onnx_node.name for sn in self.g.nodes}
        for t in self.fe.chexp:
            self.assertIn(t[:-len("_pc")] + "_rescale", names)


if __name__ == "__main__":
    unittest.main()
