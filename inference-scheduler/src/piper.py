"""
Piper (VITS) text-to-speech frontend (doc/plans/TTS_PLAN.md §4): the Piper
ONNX export's weights + calibrated exponents -> the ``chunk`` entry graph
of the library, the flow and the HiFi-GAN decoder of one chunk:

  inputs   zp [192][FLOW_FRAMES] float32 host (z_p of the chunk: zero outside
           the utterance), lo / hi int32 (the utterance's frames in chunk
           coordinates)
  output   pcm [DEC_FRAMES * 256] int16 host (the central OUT_FRAMES * 256
           samples valid)

The computation is demo/tts/scripts/piper_vits.py ``chunk_forward`` (the
specification, bit for bit): every 1-D conv is a ConvKernel call on its
input folded into rows (TtsPrep writes [C][rows][w0 + halo]: one padded
output row of out_w * out_ch <= 65 536 accumulators), transposed convs are
polyphase kernel-3 convs + TtsInterleave, the k7 dilation-12 convs run as
two tap groups (one call spans <= 64 columns); every DMA tensor has one
power-of-two exponent, each conv input the searched one ("#in").
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from . import numeric
from .llm_nodes import LLM_DOMAIN

FLOW_FRAMES, DEC_FRAMES, OUT_FRAMES = 256, 192, 128
DEC_OFF = (FLOW_FRAMES - DEC_FRAMES) // 2
HOP = 256
UPS = ((8, 16), (8, 16), (4, 8))                       # (stride, kernel)
RB_KERNELS = (3, 5, 7)
RB_DIL = ((1, 2), (2, 6), (3, 12))
MAX_SPAN = 64                                          # kernels.conv.max_line_buf_cols
ACC_ENTRIES = 65536                                    # kernels.conv.max_acc_persist_entries

__all__ = ("load_weights", "polyphase_weight", "fold", "exponent_keys", "PiperChunkFrontend", "entry_info",
           "FLOW_FRAMES", "DEC_FRAMES", "OUT_FRAMES", "DEC_OFF", "HOP")


def load_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """{VITS module name: float64 array} from Piper's ONNX export.  The
    flow's weight-normed convs are exported as ``onnx::Conv_*``: named after
    the bias input of their node; the duration predictor's ElementwiseAffine
    exp(-logs) is folded into the graph: evaluated once with onnxruntime."""
    m = onnx.load(onnx_path)
    raw = {i.name: nph.to_array(i).astype(np.float64) for i in m.graph.initializer}
    W = {k: v for k, v in raw.items() if not k.startswith(("onnx::", "/")) and not k[0].isdigit()}
    W["emb"] = W.pop("sid")                              # the phoneme embedding [256][192]
    for n in m.graph.node:
        if n.op_type == "Conv" and n.input[1].startswith("onnx::") and len(n.input) > 2:
            W[n.input[2][: -len(".bias")] + ".weight"] = raw[n.input[1]]
    W["dp.flows.0.exp_neg_logs"] = _eval(m, "/dp/flows.0/Exp_output_0")
    return W


def _eval(model: onnx.ModelProto, name: str) -> np.ndarray:
    import onnxruntime as ort
    g = onnx.ModelProto()
    g.CopyFrom(model)
    g.graph.output.append(oh.make_empty_tensor_value_info(name))
    s = ort.InferenceSession(g.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {"input": np.array([[1, 20, 2]], np.int64), "input_lengths": np.array([3], np.int64),
             "scales": np.array([0.0, 1.0, 0.0], np.float32)}
    return np.asarray(s.run([name], feeds)[0], np.float64).reshape(-1)


def polyphase_weight(w: np.ndarray, b: np.ndarray, s: int) -> Tuple[np.ndarray, np.ndarray]:
    """ConvTranspose1d (C -> O, kernel 2s, stride s, padding s/2) as a 'same'
    kernel-3 conv with s * O outputs (phase-major) + an interleave:
    y[o][m*s + r] = yp[r*O + o][m]; the bias repeats per phase."""
    C, n_out, K = w.shape
    P = (K - s) // 2
    w3 = np.zeros((s * n_out, C, 3))
    for r in range(s):
        q = r + P
        for d in (-1, 0, 1):
            j = q - d * s
            if 0 <= j < K:
                w3[r * n_out:(r + 1) * n_out, :, d + 1] = w[:, :, j].T
    return w3, np.tile(b, s)


def fold(L: int, out_ch: int) -> Tuple[int, int]:
    """(rows, w0) of a conv over L samples: the largest power-of-two row
    that divides L with w0 * roundup(out_ch, 16) <= 65 536."""
    o16 = -(-out_ch // 16) * 16
    w0 = 1
    while L % (w0 * 2) == 0 and w0 * 2 * o16 <= ACC_ENTRIES:
        w0 *= 2
    return L // w0, w0


def exponent_keys() -> List[str]:
    """Every exponent the chunk entry needs (piper_study.py calibrate writes
    them: conv inputs "<conv>#in", conv outputs, the residual chains)."""
    keys = []
    for fi in (6, 4, 2, 0):
        p, e = f"flow.flows.{fi}", f"flow.flows.{fi}.enc"
        keys += [f"{p}.pre#in", f"{p}.pre", f"{p}.post#in", f"{p}.post"]
        for i in range(4):
            keys += [f"{e}.in.{i}#in", f"{e}.in.{i}", f"{e}.rs.{i}#in", f"{e}.rs.{i}"]
            if i < 3:
                keys.append(f"{e}.h.{i + 1}")
    keys += ["dec.pre#in", "dec.pre", "dec.post#in", "dec.post"]
    for i in range(3):
        keys += [f"dec.up{i}#in", f"dec.up{i}", f"dec.st{i}"]
        for j in range(3):
            for c in range(2):
                keys += [f"dec.rb{i}{j}.c{c}#in", f"dec.rb{i}{j}.c{c}", f"dec.rb{i}{j}.y{c}"]
    return keys


class PiperChunkFrontend:
    """Builds the ``chunk`` entry.  ``W``: load_weights(); ``E``: the
    exponents (demo/tts/scripts/piper_study.py calibrate: tensor name ->
    f, conv inputs "<conv>#in")."""

    def __init__(self, W: Dict[str, np.ndarray], E: Dict[str, int], name: str = "piper"):
        missing = [k for k in exponent_keys() if k not in E]
        if missing:
            raise ValueError(f"exponents missing for {len(missing)} tensors, e.g. {missing[:3]}")
        self.W, self.E, self.name = W, {k: int(v) for k, v in E.items()}, name

    # ---- builder helpers ------------------------------------------------ #
    def _new(self):
        self.nodes: List[onnx.NodeProto] = []
        self.vi: Dict[str, onnx.ValueInfoProto] = {}
        self.inits: List[onnx.TensorProto] = []
        self.meta = numeric.empty()
        self.n = 0

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, host=None):
        self.vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if exp is not None:
            self.meta["exp"][name] = int(exp)
        if host is not None:
            self.meta["host"][name] = host
        return name

    def _uid(self, stem):
        self.n += 1
        return f"{stem}#{self.n}"

    def _prep(self, src, L_src, key, *, L, k=1, dil=1, taps=None, out_ch, base=0, ch0=0,
              nch=None, reverse=0, alpha=1.0, rate=1, frame_off=0, window=None):
        """TtsPrep writing the input of conv ``key`` over L output samples;
        ``base`` = the source index of output sample 0; taps (t0, t1) = a
        tap group of a split conv.  Returns (tensor, rows, w0, kernel)."""
        t_a, t_b = taps or (0, k)
        pad = (k - 1) * dil // 2
        o = base - pad + t_a * dil                     # source index of row 0, column 0
        span = (t_b - t_a - 1) * dil
        hl = max(0, -o)
        rows, w0 = fold(L, out_ch)
        nch = nch or 0
        y = self._t(self._uid(f"{key}#in"), [1, nch, rows, w0 + span], exp=self.E[f"{key}#in"])
        wlo, whi = window or (0, L_src)
        self.nodes.append(oh.make_node(
            "TtsPrep", [src, "lo", "hi"], [y], name=self._uid(f"{key}.prep"), domain=LLM_DOMAIN,
            ch0=ch0, nch=nch, reverse=reverse, t0=o + hl, rows=rows, w0=w0, hl=hl, hr=span - hl,
            rate=rate, frame_off=frame_off, alpha=repr(float(alpha)), wlo=wlo, whi=whi))
        return y, rows, w0, t_b - t_a

    def _conv(self, key, x, rows, w0, w, b, dil=1, out_key=None):
        """Conv on a folded input: weight [O][C][k] -> [O][C][1][k], valid
        along the row; output [1][O][rows][w0] at exponent E[key]."""
        n_out, C, k = w.shape
        wn, bn = self._uid(f"{key}.w"), self._uid(f"{key}.b")
        self.inits.append(nph.from_array(np.asarray(w, np.float32).reshape(n_out, C, 1, k), wn))
        ins = [x, wn]
        if b is not None:
            self.inits.append(nph.from_array(np.asarray(b, np.float32), bn))
            ins.append(bn)
        y = self._t(self._uid(out_key or key), [1, n_out, rows, w0], exp=self.E[key])
        self.nodes.append(oh.make_node("Conv", ins, [y], name=self._uid(key), kernel_shape=[1, k],
                                       dilations=[1, dil], pads=[0, 0, 0, 0]))
        return y

    def _sum(self, xs, ch0s, nch, L, key, div=1.0, shape=None, masked=None):
        y = self._t(self._uid(key), shape or [nch, L], exp=self.E[key])
        attrs = dict(ch0s=list(ch0s), nch=nch, div=float(div))
        ins = list(xs)
        if masked:
            rate, off = masked
            ins += ["lo", "hi"]
            attrs.update(masked=1, rate=rate, frame_off=off)
        self.nodes.append(oh.make_node("TtsSum", ins, [y], name=self._uid(f"{key}.sum"),
                                       domain=LLM_DOMAIN, **attrs))
        return y

    # ---- the entry --------------------------------------------------------- #
    def entry(self) -> onnx.ModelProto:
        W, E = self.W, self.E
        self._new()
        F = FLOW_FRAMES
        z = self._t("zp", [192, F], host="f32")
        self._t("lo", [1], TensorProto.INT32, host="i32")
        self._t("hi", [1], TensorProto.INT32, host="i32")
        self.meta["test_fill"]["lo"] = 16
        self.meta["test_fill"]["hi"] = F - 16
        for fi in (6, 4, 2, 0):
            p, e = f"flow.flows.{fi}", f"flow.flows.{fi}.enc"
            x0, r, w0, _ = self._prep(z, F, f"{p}.pre", L=F, out_ch=192, ch0=96, nch=96, reverse=1)
            h = self._conv(f"{p}.pre", x0, r, w0, W[f"{p}.pre.weight"], W[f"{p}.pre.bias"])
            rs_all = []
            for i in range(4):
                xi, r, w0, _ = self._prep(h, F, f"{e}.in.{i}", L=F, k=5, out_ch=384, nch=192)
                xin = self._conv(f"{e}.in.{i}", xi, r, w0, W[f"{e}.in_layers.{i}.weight"],
                                 W[f"{e}.in_layers.{i}.bias"])
                o_rs = 384 if i < 3 else 192
                r2, w2 = fold(F, o_rs)
                acts = self._t(self._uid(f"{e}.rs.{i}#in"), [1, 192, r2, w2], exp=E[f"{e}.rs.{i}#in"])
                self.nodes.append(oh.make_node("TtsGate", [xin], [acts], name=self._uid(f"{e}.gate.{i}"),
                                               domain=LLM_DOMAIN))
                rs = self._conv(f"{e}.rs.{i}", acts, r2, w2, W[f"{e}.res_skip_layers.{i}.weight"],
                                W[f"{e}.res_skip_layers.{i}.bias"])
                rs_all.append(rs)
                if i < 3:
                    h = self._sum([h, rs], [0, 0], 192, F, f"{e}.h.{i + 1}")
            r3, w3 = fold(F, 96)
            wn = self._sum(rs_all, [192, 192, 192, 0], 192, F, f"{p}.post#in",
                           shape=[1, 192, r3, w3], masked=(1, 0))
            m = self._conv(f"{p}.post", wn, r3, w3, W[f"{p}.post.weight"], W[f"{p}.post.bias"])
            zn = self._t(self._uid(f"{p}.z"), [192, F], host="f32")
            self.nodes.append(oh.make_node("TtsFlowOut", [z, m, "lo", "hi"], [zn],
                                           name=self._uid(f"{p}.out"), domain=LLM_DOMAIN))
            z = zn
        # decoder on the central DEC_FRAMES frames
        L = DEC_FRAMES
        xi, r, w0, _ = self._prep(z, F, "dec.pre", L=L, k=7, out_ch=256, nch=192, base=DEC_OFF,
                                  window=(DEC_OFF, DEC_OFF + L))
        x = self._conv("dec.pre", xi, r, w0, W["dec.conv_pre.weight"], W["dec.conv_pre.bias"])
        C, rate = 256, 1
        for i, (s, _k) in enumerate(UPS):
            n_out = C // 2
            w3, b3 = polyphase_weight(W[f"dec.ups.{i}.weight"], W[f"dec.ups.{i}.bias"], s)
            a, r, w0, _ = self._prep(x, L, f"dec.up{i}", L=L, k=3, out_ch=s * n_out, nch=C, alpha=0.1,
                                     rate=rate, frame_off=DEC_OFF)
            yp = self._conv(f"dec.up{i}", a, r, w0, w3, b3)
            L, rate = L * s, rate * s
            xu = self._t(self._uid(f"dec.up{i}.x"), [n_out, L], exp=E[f"dec.up{i}"])
            self.nodes.append(oh.make_node("TtsInterleave", [yp], [xu], name=self._uid(f"dec.up{i}.il"),
                                           domain=LLM_DOMAIN, s=s))
            ys = []
            for j in range(3):
                rb = f"dec.rb{i}{j}"
                kk = RB_KERNELS[j]
                y = xu
                for c in range(2):
                    d = RB_DIL[j][c]
                    wt = W[f"dec.resblocks.{3 * i + j}.convs.{c}.weight"]
                    bt = W[f"dec.resblocks.{3 * i + j}.convs.{c}.bias"]
                    groups = [(0, kk)] if (kk - 1) * d + 1 <= MAX_SPAN else [(0, 4), (4, kk)]
                    parts = []
                    for (ta, tb) in groups:
                        a, r, w0, _ = self._prep(y, L, f"{rb}.c{c}", L=L, k=kk, dil=d, taps=(ta, tb),
                                                 out_ch=n_out, nch=n_out, alpha=0.1, rate=rate,
                                                 frame_off=DEC_OFF)
                        parts.append(self._conv(f"{rb}.c{c}", a, r, w0, wt[:, :, ta:tb],
                                                bt if ta == 0 else None, dil=d))
                    y = self._sum(parts + [y], [0] * (len(parts) + 1), n_out, L, f"{rb}.y{c}")
                ys.append(y)
            x = self._sum(ys, [0, 0, 0], n_out, L, f"dec.st{i}", div=3.0)
            C = n_out
        a, r, w0, _ = self._prep(x, L, "dec.post", L=L, k=7, out_ch=1, nch=C, alpha=0.01, rate=rate,
                                 frame_off=DEC_OFF)
        yo = self._conv("dec.post", a, r, w0, W["dec.conv_post.weight"], None)
        pcm = self._t("pcm", [L], TensorProto.INT16, exp=0, host="i16")
        self.nodes.append(oh.make_node("TtsPcm", [yo], [pcm], name="dec.pcm", domain=LLM_DOMAIN))
        return self._model(["zp", "lo", "hi"], ["pcm"])

    def _model(self, inputs, outputs):
        g = oh.make_graph(self.nodes, f"{self.name}_chunk", [self.vi[n] for n in inputs],
                          [self.vi[n] for n in outputs], initializer=self.inits,
                          value_info=[v for n, v in self.vi.items() if n not in inputs and n not in outputs])
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/piper.py")
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key, p.value = numeric.METADATA_KEY, numeric.to_metadata(self.meta)
        p = m.metadata_props.add()
        p.key = "axi.tts.entry"
        p.value = json.dumps({"entry": "chunk", "model": self.name, "flow_frames": FLOW_FRAMES,
                              "dec_frames": DEC_FRAMES, "out_frames": OUT_FRAMES, "dec_off": DEC_OFF,
                              "hop": HOP, "sample_rate": 22050})
        return m


def entry_info(model: onnx.ModelProto) -> Optional[dict]:
    for p in model.metadata_props:
        if p.key == "axi.tts.entry":
            return json.loads(p.value)
    return None
