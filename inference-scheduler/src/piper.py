"""
Piper (VITS) text-to-speech frontend (doc/plans/TTS_PLAN.md §4): the Piper
ONNX export's weights + calibrated exponents -> the ``chunk`` entry graph
of the library, the flow and the HiFi-GAN decoder of one chunk:

  inputs   zp [192][FLOW_FRAMES] float32 host (z_p of the chunk: zero outside
           the utterance), lo / hi int32 (the utterance's frames in chunk
           coordinates)
  output   pcm [DEC_FRAMES * 256] int16 host (the central OUT_FRAMES * 256
           samples valid)

``PiperEncoderFrontend`` builds the text encoder as ``encode_<T>`` entries
(ids padded to a bucket of T rows, n valid; TTS_PLAN §6): every projection
and FFN conv a MatMul, the attention's q.K^T / P.V on ConvKernel (the ViT's
static-key path), the rest host ops (src/tts_nodes.py); the specification is
piper_vits.py ``encoder_forward``.

The chunk's computation is demo/tts/scripts/piper_vits.py ``chunk_forward`` (the
specification, bit for bit): every 1-D conv is a ConvKernel call on its
input folded into rows (TtsPrep writes [C][rows][w0 + halo]: one padded
output row of out_w * out_ch <= 65 536 accumulators), transposed convs are
polyphase kernel-3 convs + TtsInterleave, the k7 dilation-12 convs run as
two tap groups (one call spans <= 64 columns); every DMA tensor has one
power-of-two exponent, each conv input the searched one ("#in").
"""

from __future__ import annotations

import json
import math
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

ENC_BUCKETS = (32, 64, 128, 256, 400)                 # encode_<T> entries: ids padded to T rows
ENC_LAYERS, ENC_D, ENC_H, ENC_HD, ENC_WIN = 6, 192, 2, 96, 4

__all__ = ("load_weights", "polyphase_weight", "fold", "exponent_keys", "PiperChunkFrontend", "entry_info",
           "encoder_weights", "encoder_exponent_keys", "PiperEncoderFrontend", "ENC_BUCKETS",
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


# ---- the text encoder: encode_<T> entries ------------------------------------ #

def encoder_weights(W: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The encoder's weights as the library stores them (float32; as
    piper_vits.encoder_weights): row-layout MatMul weights [in][out], q's
    weight and bias scaled by 1 / sqrt(96), the kernel-3 convs unrolled
    tap-major into [3C][O]."""
    f32 = lambda a: np.asarray(a, np.float32)                   # noqa: E731
    k3 = lambda w: np.concatenate([w[:, :, t].T for t in range(3)], axis=0)   # noqa: E731
    sq = 1.0 / math.sqrt(ENC_HD)
    EW = {"emb": f32(W["emb"]), "wp": f32(W["enc_p.proj.weight"][:, :, 0].T), "bp": f32(W["enc_p.proj.bias"])}
    for i in range(ENC_LAYERS):
        p, f = f"enc_p.encoder.attn_layers.{i}", f"enc_p.encoder.ffn_layers.{i}"
        EW[f"l{i}.wq"] = f32(W[f"{p}.conv_q.weight"][:, :, 0].T * sq)
        EW[f"l{i}.bq"] = f32(W[f"{p}.conv_q.bias"] * sq)
        for t in ("k", "v", "o"):
            EW[f"l{i}.w{t}"] = f32(W[f"{p}.conv_{t}.weight"][:, :, 0].T)
            EW[f"l{i}.b{t}"] = f32(W[f"{p}.conv_{t}.bias"])
        EW[f"l{i}.ek"] = f32(W[f"{p}.emb_rel_k"][0])
        EW[f"l{i}.ev"] = f32(W[f"{p}.emb_rel_v"][0])
        EW[f"l{i}.w1"], EW[f"l{i}.b1"] = f32(k3(W[f"{f}.conv_1.weight"])), f32(W[f"{f}.conv_1.bias"])
        EW[f"l{i}.w2"], EW[f"l{i}.b2"] = f32(k3(W[f"{f}.conv_2.weight"])), f32(W[f"{f}.conv_2.bias"])
        for j in (1, 2):
            EW[f"l{i}.g{j}"] = f32(W[f"enc_p.encoder.norm_layers_{j}.{i}.gamma"])
            EW[f"l{i}.be{j}"] = f32(W[f"enc_p.encoder.norm_layers_{j}.{i}.beta"])
    return EW


def encoder_exponent_keys() -> List[str]:
    """The exponents of the encode entries (piper_study.py encoder writes them
    under "encoder" in exponents.json)."""
    keys = ["enc.p", "enc.x0", "enc.st#in", "enc.st"]
    for i in range(ENC_LAYERS):
        keys += [f"enc.l{i}.{t}" for t in ("xin", "q0", "k0", "v0", "q", "k", "v", "s", "o", "att", "y",
                                           "x1", "c1", "f1", "h", "f2", "x2")]
    return keys


class PiperEncoderFrontend:
    """Builds the ``encode_<T>`` entry: inputs ids [T] (int32, padded) and n
    [1] (the valid ids); outputs x [T][192] and stats [T][384] (float32 host,
    rows >= n zero): the text encoder's output and its projection (m_p,
    logs_p).  ``E``: exponents.json "encoder"."""

    def __init__(self, W: Dict[str, np.ndarray], E: Dict[str, int], T: int, name: str = "piper"):
        missing = [k for k in encoder_exponent_keys() if k not in E]
        if missing:
            raise ValueError(f"encoder exponents missing for {len(missing)} tensors, e.g. {missing[:3]}")
        if T % 16:
            raise ValueError(f"bucket {T}: a multiple of 16 rows")
        self.EW, self.E, self.T, self.name = encoder_weights(W), {k: int(v) for k, v in E.items()}, int(T), name
        self.pv_kw = next(k for k in (4, 2, 1) if T % (16 * k) == 0)

    def choose_qk_kw(self) -> int:
        """The q image's kernel width for q.K^T: the cheapest by the conv cost model."""
        from .cost_model import conv_cycles
        best = None
        for kw in (1, 2, 4):
            if ENC_HD % (16 * kw):
                continue
            for ow in [d for d in range(8, 65) if self.T % d == 0] or [self.T]:
                cyc = conv_cycles(in_ch=ENC_HD // kw, out_ch=self.T, in_h=self.T // ow, in_w=kw * ow,
                                  oh=self.T // ow, ow=ow, kh=1, kw=kw, sw=kw)["total"]
                if best is None or cyc < best[0]:
                    best = (cyc, kw)
        return best[1]

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, host=None, state=False):
        self.vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if exp is not None:
            self.meta["exp"][name] = int(exp)
        if host is not None:
            self.meta["host"][name] = host
        if state:
            self.meta["state"].append(name)
        return name

    def _init(self, name, arr):
        if name not in self.init_names:
            self.init_names.add(name)
            self.inits.append(nph.from_array(np.asarray(arr, np.float32), name))
        return name

    def _node(self, op, ins, outs, name, domain=LLM_DOMAIN, **attrs):
        self.nodes.append(oh.make_node(op, ins, outs, name=name, domain=domain, **attrs))

    def _matmul(self, x, wkey, y, cols, exp):
        wname = self._init(f"w.enc.{wkey}", self.EW[wkey])
        self._t(y, [self.T, cols], exp=exp)
        self._node("MatMul", [x, wname], [y], y, domain="")
        return y

    def _rowprep(self, x, y, cols, exp, taps=1, bias=None, relu=0):
        self._t(y, [self.T, taps * cols], exp=exp)
        ins = [x, self.n] + ([self._init(f"v.enc.{bias}", self.EW[bias])] if bias else [])
        self._node("TtsRowPrep", ins, [y], y, taps=taps, relu=relu)
        return y

    def entry(self) -> onnx.ModelProto:
        T, E, EW, D, H, HD, Wn = self.T, self.E, self.EW, ENC_D, ENC_H, ENC_HD, ENC_WIN
        p = f"enc{T}"
        self.nodes, self.vi, self.inits, self.init_names = [], {}, [], set()
        self.meta = numeric.empty()
        ids = self._t(f"{p}.ids", [T], TensorProto.INT32, host="i32")
        self.n = self._t(f"{p}.n", [1], TensorProto.INT32, host="i32")
        self.meta["test_fill"][self.n] = max(1, T - 5)          # the generated test: padding rows too
        x = self._t(f"{p}.x0", [T, D], exp=E["enc.x0"])
        self._node("TtsEmbed", [ids, self._init("v.enc.emb", EW["emb"]), self.n], [x], f"{p}.embed",
                   scale=repr(math.sqrt(D)))
        kc = self._t(f"{p}.kc", [T, D], exp=0, state=True)
        vc = self._t(f"{p}.vc", [T, D], exp=0, state=True)
        self.meta["layout"][kc] = [H, HD]
        self.meta["layout"][vc] = [H, HD] + ([self.pv_kw] if self.pv_kw > 1 else [])
        self._init(kc, np.zeros((T, D), np.float32))
        self._init(vc, np.zeros((T, D), np.float32))
        kw = self.choose_qk_kw()
        common = dict(num_heads=H, num_kv_heads=H, head_dim=HD, key_quantum=16 * self.pv_kw)
        fp = E["enc.p"]
        for i in range(ENC_LAYERS):
            e, te = f"enc.l{i}", f"{p}.l{i}"
            xin = self._rowprep(x, f"{te}.xin", D, E[f"{e}.xin"])
            q0 = self._matmul(xin, f"l{i}.wq", f"{te}.q0", D, E[f"{e}.q0"])
            k0 = self._matmul(xin, f"l{i}.wk", f"{te}.k0", D, E[f"{e}.k0"])
            v0 = self._matmul(xin, f"l{i}.wv", f"{te}.v0", D, E[f"{e}.v0"])
            qx = self._t(f"{te}.qx", [H, HD, T], exp=0)
            bq = self._init(f"v.enc.l{i}.bq", EW[f"l{i}.bq"])
            self._node("VitAttnPrep", [q0, k0, v0, kc, vc, bq, self._init(f"v.enc.l{i}.bk", EW[f"l{i}.bk"]),
                                       self._init(f"v.enc.l{i}.bv", EW[f"l{i}.bv"])],
                       [qx], f"{te}.attn_prep", num_heads=H, head_dim=HD, qk_kw=kw,
                       q_exp=[E[f"{e}.q"]] * H, k_exp=[E[f"{e}.k"]] * H, v_exp=[E[f"{e}.v"]] * D)
            ek = self._init(f"v.enc.l{i}.ek", EW[f"l{i}.ek"].reshape(-1))
            s = [self._t(f"{te}.s{g}", [T, T], exp=0) for g in range(H)]
            pr = [self._t(f"{te}.p{g}", [T, T], exp=0) for g in range(H)]
            o = [self._t(f"{te}.o{g}", [T, HD], exp=0) for g in range(H)]
            for g in range(H):
                self._node("LlmAttnScores", [kc, qx], [s[g]], f"{te}.qk{g}", group=g, qk_kw=kw, **common)
            for g in range(H):
                self._node("TtsAttnSoftmax", [s[g], q0, bq, ek, self.n], [pr[g]], f"{te}.softmax{g}", head=g,
                           head_dim=HD, window=Wn, s_exp=[E[f"{e}.s"]], q_exp=[E[f"{e}.q"]], p_exp=[fp])
                self._node("LlmAttnPV", [pr[g], vc], [o[g]], f"{te}.pv{g}", group=g, **common)
            att = self._t(f"{te}.att", [T, D], exp=E[f"{e}.att"])
            self._node("TtsAttnMerge", o + pr + [self._init(f"v.enc.l{i}.ev", EW[f"l{i}.ev"].reshape(-1)), self.n],
                       [att], f"{te}.attn_merge", num_heads=H, head_dim=HD, window=Wn,
                       o_exp=[E[f"{e}.o"]] * H, p_exp=[fp])
            y = self._matmul(att, f"l{i}.wo", f"{te}.y", D, E[f"{e}.y"])
            x1 = self._t(f"{te}.x1", [T, D], exp=E[f"{e}.x1"])
            self._node("TtsResNorm", [x, y, self._init(f"v.enc.l{i}.bo", EW[f"l{i}.bo"]),
                                      self._init(f"v.enc.l{i}.g1", EW[f"l{i}.g1"]),
                                      self._init(f"v.enc.l{i}.be1", EW[f"l{i}.be1"]), self.n],
                       [x1], f"{te}.norm1", eps=repr(1e-5))
            c1 = self._rowprep(x1, f"{te}.c1", D, E[f"{e}.c1"], taps=3)
            f1 = self._matmul(c1, f"l{i}.w1", f"{te}.f1", EW[f"l{i}.w1"].shape[1], E[f"{e}.f1"])
            h = self._rowprep(f1, f"{te}.h", EW[f"l{i}.w1"].shape[1], E[f"{e}.h"], taps=3, bias=f"l{i}.b1",
                              relu=1)
            f2 = self._matmul(h, f"l{i}.w2", f"{te}.f2", D, E[f"{e}.f2"])
            x = self._t(f"{te}.x2", [T, D], exp=E[f"{e}.x2"])
            self._node("TtsResNorm", [x1, f2, self._init(f"v.enc.l{i}.b2", EW[f"l{i}.b2"]),
                                      self._init(f"v.enc.l{i}.g2", EW[f"l{i}.g2"]),
                                      self._init(f"v.enc.l{i}.be2", EW[f"l{i}.be2"]), self.n],
                       [x], f"{te}.norm2", eps=repr(1e-5))
        sti = self._rowprep(x, f"{p}.st_in", D, E["enc.st#in"])
        st = self._matmul(sti, "wp", f"{p}.st", 2 * D, E["enc.st"])
        xo = self._t(f"{p}.x", [T, D], host="f32")
        so = self._t(f"{p}.stats", [T, 2 * D], host="f32")
        self._node("TtsEncOut", [x, self.n], [xo], f"{p}.x_out")
        self._node("TtsEncOut", [st, self.n, self._init("v.enc.bp", EW["bp"])], [so], f"{p}.stats_out")
        g = oh.make_graph(self.nodes, f"{self.name}_encode_{T}", [self.vi[ids], self.vi[self.n]],
                          [self.vi[xo], self.vi[so]], initializer=self.inits,
                          value_info=[v for k, v in self.vi.items() if k not in (ids, self.n, xo, so)])
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/piper.py")
        m.ir_version = 8
        md = m.metadata_props.add()
        md.key, md.value = numeric.METADATA_KEY, numeric.to_metadata(self.meta)
        md = m.metadata_props.add()
        md.key = "axi.tts.entry"
        md.value = json.dumps({"entry": f"encode_{T}", "model": self.name, "rows": T, "hidden": D,
                               "heads": H, "head_dim": HD, "qk_kw": kw, "pv_kw": self.pv_kw})
        return m
