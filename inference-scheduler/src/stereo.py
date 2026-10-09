"""
LightStereo-S frontend (doc/plans/STEREO_PLAN.md): a rectified stereo pair ->
a disparity map, every layer on this repo's kernels or host ops.

The checkpoint (OpenStereo's LightStereo, a PyTorch state dict) and the
study's calibration (``demo/stereo_depth/scripts/stereo_study.py calibrate``:
the largest |value| of every site) become one ONNX graph with
``axi.numeric`` exponents, which OnnxGraph / CodeGenerator compile like any
model.  The arithmetic is the study's emulation (``pow2``):

  * every conv on ConvKernel — BatchNorm folded, the weight at f_y + 8 - f_x
    (numeric.encode_conv_weights); a ConvTranspose (4 x 4 or 3 x 3, stride 2)
    as its polyphase conv (the four output phases as 4C channels of one 3 x 3
    / 2 x 2 conv) and a pixel shuffle (Reshape / Transpose / Reshape, a host
    copy); a depthwise stripe longer than 7 taps as the fewest arithmetic
    progressions of taps that straddle its centre (only top / left padding
    registers) and fit the line buffer (``stripe_pieces``), summed by adds;
  * ReLU6, LeakyReLU, ReLU, the residual and stripe adds and the attention
    product on VectorOPKernel (``StereoVop``); the softmax over the 48
    disparities and over the 9 upsampling weights (padded to 16 channels,
    9 valid, one call per output phase of the last polyphase deconv) on its
    softmax unit's column mode (``StereoSoftmax``, P at 2^-15);
  * the instance norms, the replicate pad, the correlation volume and the
    disparity regression with the context upsampling as host ops
    (src/stereo_nodes.py).

Exponents: a site's calibrated range with one bit of headroom,
f = floor(log2(16383 / max)) within [0, 15]; tensors an add, a concat, a
copy or an activation ties together share the smallest of their sites'; a
ReLU6 input and output stay at 2^-8 (the kernel clamps at raw 6.0 in Q8.8);
the attention product sits at f_a + f_cost - 8, so its conv3 takes the
exponent that puts the product at its group's.

``load_checkpoint`` reads a torch zip checkpoint with the standard library
(no torch).
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as nph

from . import numeric
from ._conv_hw_config import CONV_MAX_KW, CONV_MAX_LINE_BUF_COLS, CONV_MAX_LINE_BUF_ROWS
from .llm_nodes import LLM_DOMAIN
from .nodes import ACT_RELU, OP_ADD, OP_LEAKY_RELU, OP_MUL, OP_RELU, OP_RELU6

F = 8                       # the kernels' output shift
MAX_DISP = 192
P_EXP = 15                  # the softmax unit's P
UP_KEYS = 16                # the 9 upsampling weights, padded for the unit (keys % 8 == 0)

# timm mobilenetv2_100, blocks 0 .. 5: (blocks, expansion, out channels, stride)
MOBILENETV2 = [(1, 1, 16, 1), (2, 6, 24, 2), (3, 6, 32, 2), (4, 6, 64, 2), (3, 6, 96, 1), (3, 6, 160, 2)]


class StereoError(ValueError):
    pass


# ------------------------------------------------------------------ #
# The checkpoint                                                       #
# ------------------------------------------------------------------ #

_STORAGE = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
            "LongStorage": np.int64, "IntStorage": np.int32, "BFloat16Storage": None,
            "ByteStorage": np.uint8, "BoolStorage": np.bool_}


def load_checkpoint(path) -> Dict[str, np.ndarray]:
    """``model_state`` of a torch zip checkpoint as float32 / int64 arrays,
    read with zipfile + pickle (no torch)."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    prefix = pkl[:-len("data.pkl")]

    def rebuild(storage, offset, size, stride, *_args):
        if not size:
            return storage[offset:offset + 1].reshape(()).copy()
        st = [s * storage.itemsize for s in stride]
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=st).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, *_a: data
            if mod == "torch" and name in _STORAGE:
                return name
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            dt = _STORAGE.get(stype if isinstance(stype, str) else getattr(stype, "__name__", ""))
            if dt is None:
                raise StereoError(f"{path}: storage {stype} not supported")
            return np.frombuffer(zf.read(f"{prefix}data/{key}"), dt)

    obj = Unpickler(zf.open(pkl)).load()
    sd = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    return {k: np.asarray(v) for k, v in sd.items() if "num_batches_tracked" not in k}


def fold(sd: Dict[str, np.ndarray], conv: str, bn: Optional[str], transposed: bool = False):
    """Weight and bias (float64) with the BatchNorm folded in (eps 1e-5)."""
    w = np.asarray(sd[conv + ".weight"], np.float64)
    oc = w.shape[1] if transposed else w.shape[0]
    b = np.asarray(sd.get(conv + ".bias", np.zeros(oc)), np.float64)
    if bn is not None:
        s = sd[bn + ".weight"].astype(np.float64) / np.sqrt(sd[bn + ".running_var"].astype(np.float64) + 1e-5)
        w = w * (s.reshape(1, -1, 1, 1) if transposed else s.reshape(-1, 1, 1, 1))
        b = (b - sd[bn + ".running_mean"]) * s + sd[bn + ".bias"]
    return w, b


# ------------------------------------------------------------------ #
# Stripes and deconvs                                                  #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=None)
def stripe_pieces(k: int, span: int, max_taps: int = CONV_MAX_KW) -> Tuple[Tuple[int, int, int], ...]:
    """The fewest pieces (t0, dilation, taps) of a centred k-tap stripe:
    arithmetic progressions t0 + d*a (a < taps <= max_taps) that partition
    0 .. k-1, each straddling the centre k//2 (its padding before is then
    k//2 - t0 >= 0 and after >= 0: ConvKernel programs only top / left) with a
    dilated window d*(taps-1)+1 <= span (the line buffer)."""
    c = k // 2
    best: List = []

    def dfs(left, pieces):
        if best and len(pieces) >= len(best[0]):
            return
        if not left:
            best[:] = [list(pieces)]
            return
        u = min(left)
        for d in range(1, max(2, k)):
            for n in range(max_taps, 0, -1):
                taps = [u + d * a for a in range(n)]
                if (taps[-1] < k and all(t in left for t in taps) and u <= c <= taps[-1]
                        and d * (n - 1) + 1 <= span):
                    dfs(left - set(taps), pieces + [(u, d, n)])
    dfs(frozenset(range(k)), [])
    if not best:
        raise StereoError(f"no split of a {k}-tap stripe into pieces of <= {max_taps} taps within {span}")
    return tuple(best[0])


def polyphase(wt: np.ndarray, k: int, pad: int, out_pad: int) -> Tuple[np.ndarray, List[int]]:
    """A stride-2 ConvTranspose weight [Cin][Cout][k][k] -> the conv weight
    [4*Cout][Cin][kp][kp] of its four output phases (phase-major channels,
    q = 2*qy + qx) and that conv's ONNX pads.  out[2m + q] = sum over taps i
    of x[m + off] w[k_tap] with 2(m + off) + k_tap - pad = 2m + q."""
    cin, cout = wt.shape[:2]
    taps = {}                                    # (q, off) -> k_tap
    for q in (0, 1):
        for kt in range(k):
            num = q + pad - kt
            if num % 2 == 0:
                taps[(q, num // 2)] = kt
    offs = sorted({o for (_q, o) in taps})
    lo, hi = offs[0], offs[-1]
    kp = hi - lo + 1
    w = np.zeros((4 * cout, cin, kp, kp))
    for qy in (0, 1):
        for qx in (0, 1):
            ph = (2 * qy + qx) * cout
            for (q1, oy), ky in taps.items():
                if q1 != qy:
                    continue
                for (q2, ox), kx in taps.items():
                    if q2 != qx:
                        continue
                    w[ph:ph + cout, :, oy - lo, ox - lo] = wt[:, :, ky, kx].T
    del out_pad                                   # the output is 2 x the input either way
    return w, [-lo, -lo, hi, hi]


# ------------------------------------------------------------------ #
# The graph builder                                                    #
# ------------------------------------------------------------------ #

class _Builder:
    """ONNX nodes, initializers, shapes and the exponent constraints."""

    def __init__(self, sd, site_exp: Dict[str, int]):
        self.sd, self.site_exp = sd, site_exp
        self.nodes: List = []
        self.inits: Dict[str, np.ndarray] = {}
        self.shape: Dict[str, Tuple[int, ...]] = {}
        self.site: Dict[str, str] = {}           # tensor -> calibration site
        self.fixed: Dict[str, int] = {}          # tensor -> required exponent
        self.parent: Dict[str, str] = {}         # union-find of tensors sharing an exponent
        self.muls: List[Tuple[str, str, str]] = []
        self.n = 0
        self.float_out: set = set()               # host float32 tensors (no exponent)
        self.convs: List[dict] = []               # standard convs, for the per-channel weights

    # ---- bookkeeping ---- #
    def name(self, base: str) -> str:
        self.n += 1
        return f"{base}_{self.n}"

    def find(self, t):
        while self.parent.get(t, t) != t:
            t = self.parent[t]
        return t

    def tie(self, *ts):
        r = self.find(ts[0])
        for t in ts[1:]:
            q = self.find(t)
            if q != r:
                self.parent[q] = r

    def out(self, base, shape, site=None):
        y = self.name(base)
        self.shape[y] = tuple(int(s) for s in shape)
        if site is not None:
            self.site[y] = site
        return y

    def init(self, name, arr, dtype=np.float32):
        if name not in self.inits:
            self.inits[name] = np.asarray(arr, dtype)
        return name

    def llm(self, op_type, ins, y, **attrs):
        self.nodes.append(oh.make_node(op_type, ins, [y], domain=LLM_DOMAIN, name=y, **attrs))
        return y

    # ---- ops ---- #
    def conv(self, x, w, b, site, stride=1, pads=(0, 0, 0, 0), group=1, dil=(1, 1), wname=None):
        cin_g, kh, kw = w.shape[1:]
        _, c, h, wd = self.shape[x]
        oh_ = (h + pads[0] + pads[2] - dil[0] * (kh - 1) - 1) // stride + 1
        ow_ = (wd + pads[1] + pads[3] - dil[1] * (kw - 1) - 1) // stride + 1
        y = self.out(site, (1, w.shape[0], oh_, ow_), site)
        wn = self.init(wname or (site + ".w"), w)
        ins = [x, wn]
        if b is not None:
            ins.append(self.init((wname or site) + ".b", b))
        node = oh.make_node("Conv", ins, [y], name=y, kernel_shape=[kh, kw], strides=[stride, stride],
                            pads=list(pads), group=group, dilations=list(dil))
        self.nodes.append(node)
        if group == 1:
            self.convs.append({"node": node, "x": x, "y": y, "w": w, "site": site, "wname": wname or site})
        return y

    def vop(self, op, ins, site=None, act=0, alpha=None):
        y = self.out(site or "vop", self.shape[ins[0]], site)
        attrs = {"op": op}
        if act:
            attrs["act"] = act
        if alpha is not None:
            attrs["alpha"] = repr(float(alpha))
        self.llm("StereoVop", ins, y, **attrs)
        if op == OP_ADD:
            self.tie(ins[0], ins[1], y)
        elif op == OP_MUL:
            self.muls.append((ins[0], ins[1], y))
        elif op == OP_RELU6:
            self.tie(ins[0], y)
            self.fixed[ins[0]] = F
        else:
            self.tie(ins[0], y)
        return y

    def act(self, x, act):
        if act == "relu6":
            return self.vop(OP_RELU6, [x])
        if act == "relu":
            return self.vop(OP_RELU, [x])
        if act and act.startswith("leaky"):
            return self.vop(OP_LEAKY_RELU, [x], alpha=float(act[5:] or 0.01))
        return x

    def shuffle(self, x):
        """[1][4C][H][W] (phase-major) -> [1][C][2H][2W]: Reshape / Transpose / Reshape."""
        _, c4, h, w = self.shape[x]
        c = c4 // 4
        s5 = self.init(f"shape5_{c}_{h}_{w}", [2, 2, c, h, w], np.int64)
        s4 = self.init(f"shape4_{c}_{h}_{w}", [1, c, 2 * h, 2 * w], np.int64)
        a = self.out("sh5", (2, 2, c, h, w))
        t = self.out("sh5t", (c, h, 2, w, 2))
        y = self.out("shuffle", (1, c, 2 * h, 2 * w))
        self.nodes += [oh.make_node("Reshape", [x, s5], [a], name=a),
                       oh.make_node("Transpose", [a], [t], name=t, perm=[2, 3, 0, 4, 1]),
                       oh.make_node("Reshape", [t, s4], [y], name=y)]
        self.tie(x, a, t, y)
        return y

    def deconv(self, x, conv, bn, k, pad, out_pad=0, pad_out_to=None, shuffle=True):
        w, b = fold(self.sd, conv, bn, transposed=True)
        if pad_out_to is not None and w.shape[1] < pad_out_to:            # zero channels up to pad_out_to
            w = np.concatenate([w, np.zeros((w.shape[0], pad_out_to - w.shape[1], k, k))], axis=1)
            b = np.concatenate([b, np.zeros(pad_out_to - b.shape[0])])
        wp, pads = polyphase(w, k, pad, out_pad)
        bp = np.tile(b, 4)
        y = self.conv(x, wp, bp, conv, pads=pads)
        return self.shuffle(y) if shuffle else y

    def stripe(self, x, conv, horizontal):
        w = np.asarray(self.sd[conv + ".weight"], np.float64)
        b = np.asarray(self.sd[conv + ".bias"], np.float64)
        k = w.shape[3] if horizontal else w.shape[2]
        c = k // 2
        taps = w[:, :, 0, :] if horizontal else w[:, :, :, 0]                # [C][1][k]
        span = CONV_MAX_LINE_BUF_COLS if horizontal else CONV_MAX_LINE_BUF_ROWS
        out = None
        for i, (t0, d, n) in enumerate(stripe_pieces(k, span)):
            wp = taps[:, :, t0:t0 + d * (n - 1) + 1:d]                       # [C][1][n]
            wp = wp[:, :, None, :] if horizontal else wp[:, :, :, None]
            before, after = c - t0, d * (n - 1) - (c - t0)
            pads = (0, before, 0, after) if horizontal else (before, 0, after, 0)
            dil = (1, d) if horizontal else (d, 1)
            y = self.conv(x, wp, b if i == 0 else None, conv, pads=pads, group=w.shape[0], dil=dil,
                          wname=f"{conv}.p{i}")
            out = y if out is None else self.vop(OP_ADD, [out, y], conv + ".sum")
        return out

    def host(self, op_type, ins, shape, site, **attrs):
        y = self.out(site, shape, site)
        return self.llm(op_type, ins, y, **attrs)

    # ---- exponents ---- #
    def cal(self, site: str) -> int:
        if site not in self.site_exp:
            raise StereoError(f"no calibrated exponent for site '{site}'")
        return self.site_exp[site]

    def per_channel(self, exps: Dict[str, int], chan: Dict[str, List[int]], threshold: float) -> Dict[str, list]:
        """Per-output-channel weight exponents for every standard conv whose
        per-tensor weight (at f_y + 8 - f_x) loses more than ``threshold`` of
        its norm to rounding: channel c is written at f_c = max(f_y, min(the
        finest exponent its weights fit, its calibrated exponent, f_y + 8)),
        and an identity depthwise 1 x 1 conv floors it to f_y (weight 1.0 at
        f_y + 8 - f_c: 2^(8 - k) raw).  Returns {per-channel tensor: [f_c]}."""
        out: Dict[str, list] = {}
        for cv in self.convs:
            w, fx, fy = cv["w"], exps[cv["x"]], exps[cv["y"]]
            fw = fy + F - fx
            q = np.clip(np.round(w * 2.0 ** fw), -32768, 32767) * 2.0 ** -fw
            err = float(np.linalg.norm(q - w) / max(np.linalg.norm(w), 1e-30))
            if err <= threshold:
                continue
            m = w.shape[0]
            cal = chan.get(cv["site"])
            if cal is None:
                continue
            cal = np.asarray(cal, np.int64)
            if cal.size != m:                       # a polyphase deconv: the channels per phase
                cal = np.tile(np.pad(cal, (0, m // 4 - cal.size), constant_values=fy), 4)
            wmax = np.abs(w.reshape(m, -1)).max(axis=1)
            fit = np.where(wmax > 0, np.floor(np.log2(32767.0 / np.maximum(wmax, 1e-30))), 30).astype(np.int64)
            fc = np.maximum(fy, np.minimum(np.minimum(fit + fx - F, cal), fy + F))
            if (fc == fy).all():
                continue
            y, yp = cv["y"], cv["y"] + "_pc"
            self.shape[yp] = self.shape[y]
            cv["node"].output[0] = yp
            ones = self.init(cv["wname"] + ".rescale", np.ones((m, 1, 1, 1)))
            i = self.nodes.index(cv["node"])
            self.nodes.insert(i + 1, oh.make_node("Conv", [yp, ones], [y], name=y + "_rescale", kernel_shape=[1, 1],
                                                  group=m))
            out[yp] = [int(v) for v in fc]
        return out

    def exponents(self) -> Dict[str, int]:
        groups: Dict[str, List[str]] = collections.defaultdict(list)
        for t in self.shape:
            if t not in self.float_out:
                groups[self.find(t)].append(t)
        gexp: Dict[str, int] = {}
        for r, ts in groups.items():
            fixed = {self.fixed[t] for t in ts if t in self.fixed}
            if len(fixed) > 1:
                raise StereoError(f"tensors {ts} need exponents {sorted(fixed)}")
            cal = [self.cal(self.site[t]) for t in ts if t in self.site]
            gexp[r] = fixed.pop() if fixed else (min(cal) if cal else None)
        # a product a * b sits at f_a + f_b - 8: a (a conv output only the
        # product reads) takes the exponent that puts it at its group's
        for _ in range(4):
            changed = False
            for a, b, y in self.muls:
                ra, rb, ry = self.find(a), self.find(b), self.find(y)
                want = gexp[ry] - gexp[rb] + F
                if want > gexp[ra]:                     # a would saturate: lower the product's group
                    gexp[ry] = gexp[ra] + gexp[rb] - F
                    changed = True
                elif want < gexp[ra]:
                    gexp[ra] = want
                    changed = True
            if not changed:
                break
        out = {}
        for t in self.shape:
            if t in self.float_out:
                continue
            e = gexp[self.find(t)]
            if e is None:
                raise StereoError(f"tensor '{t}' has no exponent")
            out[t] = int(e)
        for a, b, y in self.muls:
            if out[y] != out[a] + out[b] - F:
                raise StereoError(f"product '{y}': exponents {out[a]} + {out[b]} - {F} != {out[y]}")
        return out


class LightStereoFrontend:
    """LightStereo-S at ``height`` x ``width`` (multiples of 32): inputs
    ``left`` / ``right`` [1][3][H][W] (ImageNet-normalised RGB), output
    ``disparity`` [H][W] (float32, pixels of the input resolution)."""

    def __init__(self, state: Dict[str, np.ndarray], formats: dict, height: int, width: int,
                 max_disp: int = MAX_DISP, per_channel: Optional[float] = None):
        if height % 32 or width % 32:
            raise StereoError(f"{width} x {height}: both must be multiples of 32")
        self.sd = state
        self.site_exp = site_exponents(formats)
        self.H, self.W, self.D = height, width, max_disp // 4
        # per-channel weight exponents where per-tensor rounding loses more than
        # this share of a conv's weights (None: off; 0: every conv that gains)
        self.per_channel = per_channel
        self.channel_exp = {k: v for k, v in formats.get("channel_exponents", {}).items()}

    # ---- blocks ---- #
    def _mbv2_block(self, g, x, p, e, stride, skip):
        if e == 1:
            h = g.conv(x, *fold(self.sd, p + ".conv_dw", p + ".bn1"), p + ".conv_dw", stride, (1, 1, 1, 1),
                       group=g.shape[x][1])
            h = g.act(h, "relu6")
            h = g.conv(h, *fold(self.sd, p + ".conv_pw", p + ".bn2"), p + ".conv_pw")
        else:
            h = g.act(g.conv(x, *fold(self.sd, p + ".conv_pw", p + ".bn1"), p + ".conv_pw"), "relu6")
            h = g.conv(h, *fold(self.sd, p + ".conv_dw", p + ".bn2"), p + ".conv_dw", stride, (1, 1, 1, 1),
                       group=g.shape[h][1])
            h = g.act(h, "relu6")
            h = g.conv(h, *fold(self.sd, p + ".conv_pwl", p + ".bn3"), p + ".conv_pwl")
        return g.vop(OP_ADD, [x, h], p + ".add") if skip else h

    @staticmethod
    def _bp(stage, j):
        if stage <= 2:
            return f"backbone.block{stage}.{j}"
        if stage in (3, 4):                     # nn.Sequential(blocks[3:5]) keeps timm's indices
            return f"backbone.block3.{stage}.{j}"
        return f"backbone.block4.{j}"

    def _fpn(self, g, low, high, p):
        x = g.act(g.deconv(low, p + ".deconv.block.0", p + ".deconv.block.1", 4, 1), "leaky0.2")
        cat = g.out("concat", (1, g.shape[high][1] + g.shape[x][1]) + g.shape[x][2:])
        g.nodes.append(oh.make_node("Concat", [high, x], [cat], name=cat, axis=1))
        g.tie(high, x, cat)
        return g.act(g.conv(cat, *fold(self.sd, p + ".conv.block.0", p + ".conv.block.1"), p + ".conv.block.0",
                            1, (1, 1, 1, 1)), "leaky0.2")

    def _backbone(self, g, img):
        x = g.act(g.conv(img, *fold(self.sd, "backbone.conv_stem", "backbone.bn1"), "backbone.conv_stem",
                         2, (1, 1, 1, 1)), "relu6")
        c, feats = 32, []
        for stage, (nb, e, oc, s) in enumerate(MOBILENETV2):
            for j in range(nb):
                st = s if j == 0 else 1
                x = self._mbv2_block(g, x, self._bp(stage, j), e, st, st == 1 and c == oc)
                c = oc
            feats.append(x)
        c2, c3, c4, c5 = feats[1], feats[2], feats[4], feats[5]
        p4 = self._fpn(g, c5, c4, "backbone.fpn_layer4")
        p3 = self._fpn(g, p4, c3, "backbone.fpn_layer3")
        p2 = self._fpn(g, p3, c2, "backbone.fpn_layer2")
        _, ch, h, w = g.shape[p2]
        pp = g.host("StereoPadEdge", [p2], (1, ch, h + 2, w + 2), "backbone.pad", pad=1)
        g.tie(p2, pp)
        g.site.pop(pp, None)
        p2 = g.conv(pp, fold(self.sd, "backbone.out_conv.block.0", None)[0], None, "backbone.out_conv.block.0")
        p2 = g.host("StereoInstanceNorm", [p2], g.shape[p2], "backbone.out_conv.in")
        return [p2, p3, p4, c5]

    def _mv2res(self, g, x, p, stride, skip):
        h = g.act(g.conv(x, *fold(self.sd, p + ".pwconv.0", p + ".pwconv.1"), p + ".pwconv.0"), "relu6")
        h = g.act(g.conv(h, *fold(self.sd, p + ".dwconv.0", p + ".dwconv.1"), p + ".dwconv.0", stride,
                         (1, 1, 1, 1), group=g.shape[h][1]), "relu6")
        h = g.conv(h, *fold(self.sd, p + ".pwliner.0", p + ".pwliner.1"), p + ".pwliner.0")
        return g.vop(OP_ADD, [x, h], p + ".add") if skip else h

    def _attention(self, g, cost, feat, p):
        a = g.conv(feat, *fold(self.sd, p + ".conv0", None), p + ".conv0")
        parts = []
        for i in range(3):
            s = g.stripe(a, f"{p}.conv{i}_1", True)
            parts.append(g.stripe(s, f"{p}.conv{i}_2", False))
        a = g.vop(OP_ADD, [a, parts[0]], p + ".sum0")
        a = g.vop(OP_ADD, [a, parts[1]], p + ".sum1")
        a = g.vop(OP_ADD, [a, parts[2]], p + ".sum2")
        a = g.conv(a, *fold(self.sd, p + ".conv3", None), p + ".conv3")
        return g.vop(OP_MUL, [a, cost], p + ".mul")

    def _aggregation(self, g, x, fl):
        p = "cost_agg"
        x = self._mv2res(g, x, p + ".conv0.0", 1, True)
        x = self._attention(g, x, fl[0], p + ".att0")
        c1 = self._mv2res(g, x, p + ".conv1", 2, False)
        c2 = self._mv2res(g, c1, p + ".conv2.0", 1, True)
        c2 = self._attention(g, c2, fl[1], p + ".att2")
        c3 = self._mv2res(g, c2, p + ".conv3", 2, False)
        c4 = c3
        for j in range(3):
            c4 = self._mv2res(g, c4, f"{p}.conv4.{j}", 1, True)
        c4 = self._attention(g, c4, fl[2], p + ".att4")
        c5 = g.deconv(c4, p + ".conv5.0", p + ".conv5.1", 3, 1, out_pad=1)
        c5 = g.vop(OP_ADD, [c5, self._mv2res(g, c2, p + ".redir2", 1, True)], p + ".c5add", act=ACT_RELU)
        c6 = g.deconv(c5, p + ".conv6.0", p + ".conv6.1", 3, 1, out_pad=1)
        return g.vop(OP_ADD, [c6, self._mv2res(g, x, p + ".redir1", 1, True)], p + ".c6add", act=ACT_RELU)

    # ---- the graph ---- #
    def build(self) -> onnx.ModelProto:
        H, W, D = self.H, self.W, self.D
        g = _Builder(self.sd, self.site_exp)
        left, right = "left", "right"
        for t in (left, right):
            g.shape[t] = (1, 3, H, W)
            g.site[t] = "input"
        fl = self._backbone(g, left)
        fr = self._backbone(g, right)[0]
        h4, w4 = H // 4, W // 4
        vol = g.host("StereoCorrelation", [fl[0], fr], (1, D, h4, w4), "correlation", disp=D)
        agg = self._aggregation(g, vol, fl)
        prob = g.out("prob", (h4 * w4, D))
        g.fixed[prob] = P_EXP
        g.llm("StereoSoftmax", [agg], prob, p_exp=P_EXP)
        # refinement: the 9 upsampling weights per pixel
        xs = g.conv(fl[0], fold(self.sd, "refine_1.0.block.0", None)[0], None, "refine_1.0.block.0",
                    pads=(1, 1, 1, 1))
        xs = g.act(g.host("StereoInstanceNorm", [xs], g.shape[xs], "refine_1.0.in"), "leaky")
        xs = g.conv(xs, fold(self.sd, "refine_1.1.block.0", None)[0], None, "refine_1.1.block.0", pads=(1, 1, 1, 1))
        xs = g.act(g.host("StereoInstanceNorm", [xs], g.shape[xs], "refine_1.1.in"), "relu")
        st = g.act(g.conv(left, *fold(self.sd, "stem_2.0.block.0", "stem_2.0.block.1"), "stem_2.0.block.0", 2,
                          (1, 1, 1, 1)), "leaky")
        st = g.act(g.conv(st, *fold(self.sd, "stem_2.1.block.0", "stem_2.1.block.1"), "stem_2.1.block.0", 1,
                          (1, 1, 1, 1)), "relu")
        xs = self._fpn(g, xs, st, "refine_2")
        # the 9 weights per pixel as the polyphase conv's four phases (no pixel
        # shuffle): one softmax per phase, read by phase by the upsampling
        xs = g.deconv(xs, "refine_3.block.0", None, 4, 1, pad_out_to=UP_KEYS, shuffle=False)
        spx = g.out("spx", (H * W, UP_KEYS))
        g.fixed[spx] = P_EXP
        g.llm("StereoSoftmax", [xs], spx, p_exp=P_EXP, valid=9, groups=4)
        disp = "disparity"
        g.shape[disp] = (H, W)
        g.float_out.add(disp)
        g.llm("StereoUpsample", [prob, spx], disp, h=h4, w=w4, scale=4, phases=2)
        exps = g.exponents()
        chexp = g.per_channel(exps, self.channel_exp, self.per_channel) if self.per_channel is not None else {}
        outputs = [oh.make_tensor_value_info(disp, TensorProto.FLOAT, [H, W])]
        inputs = [oh.make_tensor_value_info(t, TensorProto.FLOAT, [1, 3, H, W]) for t in (left, right)]
        names_out = {o.name for o in outputs}
        value_info = [oh.make_tensor_value_info(t, TensorProto.FLOAT, list(s)) for t, s in g.shape.items()
                      if t not in (left, right) and t not in names_out]
        graph = oh.make_graph(g.nodes, f"lightstereo_s_{W}x{H}", inputs, outputs,
                              initializer=[nph.from_array(v, k) for k, v in g.inits.items()],
                              value_info=value_info)
        m = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)])
        m.ir_version = 8
        meta = numeric.empty()
        meta["exp"] = exps
        meta["chexp"] = chexp
        meta["host"] = {disp: "f32"}
        p = m.metadata_props.add()
        p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(meta)
        self.exponents = exps
        self.chexp = chexp
        return m


def load_formats(path) -> dict:
    return json.loads(Path(path).read_text())


def site_exponent(m: float) -> int:
    """A calibrated largest |value| -> its exponent with one bit of headroom, within [0, 15]."""
    return 15 if m <= 0 else max(0, min(15, math.floor(math.log2(16383.0 / m))))


def site_exponents(formats: dict) -> Dict[str, int]:
    """Per-site exponents of a formats file: the compact one (``exponents``,
    demo/stereo_depth/lightstereo_s_formats.json) or the study's calibration
    (``max_abs``)."""
    if "exponents" in formats and all(not k.endswith(":c") for k in formats["exponents"]):
        return {k: int(v) for k, v in formats["exponents"].items()}
    return {k: site_exponent(float(v)) for k, v in formats["max_abs"].items() if not k.endswith(":c")}


__all__ = ("LightStereoFrontend", "StereoError", "load_checkpoint", "load_formats", "site_exponents", "fold",
           "polyphase",
           "stripe_pieces", "MAX_DISP", "P_EXP", "UP_KEYS")
