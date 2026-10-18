# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
Moonshine speech-to-text frontend (doc/plans/ASR_PLAN.md §3): config.json +
the checkpoint (safetensors) + the calibrated formats (demo/asr/scripts
asr_study.py formats) -> the entries of one multi-entry project; the
specification is demo/asr/scripts/moonshine_q.py (QM with the formats), which
every entry's simulation equals bit for bit.

  encode_<T>   pcm[384 T + 511] (int16 at 2^-15 in host memory), n (the
               valid samples, i32), pos (0, i32: the attention's start; an input,
               not a constant) -> the cross-attention K / V of every decoder
               layer (DMA states, read by decode); T = the bucket's frames
               (a multiple of 64, at most 1024: the softmax unit's keys)
  decode       ids[1], pos[1] (the token's position), xpos[1] (the encoder's valid
               frames - 1) -> id[1] (i32, the argmax)

Encoder (the bucket's T rows; frames past the valid ones are computed but never
read — the attention masks them):

  img  = AsrFrames(pcm, n)                  conv1's polyphase image [64][rows][w + 1]
  c1   = Conv 1 x 2 (127 taps + a zero)     per-channel exponents (chexp) from here
  gn   = AsrStemNorm(c1, n)                 tanh + GroupNorm -> conv2's folded input
  c2   = Conv 1 x 7, stride 3, bias         g2 = AsrStemGelu(c2) -> conv3's folded input
  c3   = Conv 1 x 3, stride 2, bias         h0 = AsrStemGelu(c3, residual) [T][d] int32
  nf   = AsrFrameCount(n)                   the valid frames (i32)
  per layer:
    x = VitLayerNorm(h)                     row-ops NORM on the int32 residual
    q0 / k0 / v = x . W                     heads padded to HDp (a multiple of 16), q / k
                                            permuted into the unit's rotate-half order
    qx = LlmAttnPrep(rowops)                RoPE of q / k (the first rot dims), the K / V
                                            caches, the q image; pos = 0, n = nf
    per head: LlmAttnScores, LlmAttnSoftmax (VectorOPKernel's unit, causal = 0,
              logit_scale hd^-1/2), LlmAttnPV; LlmAttnMerge -> pv; o = pv . Wo
    h1 = LlmResAdd(h, o); x2 = VitLayerNorm(h1); f = x2 . W1
    a  = VitGelu(f, b1, erf) (VectorOPKernel: ADD, MUL + act GELU); d = a . W2
    h  = VitResAdd(h1, d, b2)
  xf = VitLayerNorm(h); per decoder layer: AsrCrossPrep(xf . Wk, xf . Wv) -> the caches

Decoder (one token):
  h0 = LlmEmbed(ids)                        the float32 table -> the int32 residual
  per layer: self attention as Llama's decode step on the unit (LlmAttnPrep rowops,
  one row, RoPE at pos; the host p12 softmax over pos + 1 keys); cross attention:
  cq = y2 . Wq, AsrCrossQ -> the q image, LlmAttnScores / LlmAttnSoftmax / LlmAttnPV
  over the cross caches at "pos" xpos (keys = xpos + 1 valid); AsrSwiGLU(y3 . W1)
  (the fc1 bias added on the unit) . W2 + b2
  id = AsrArgmax(yf . E^T)                  the tied head, one exponent
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto

from . import numeric
from .llama import graph_with_initializers
from .llm_nodes import LLM_DOMAIN

F = 8                                   # the kernels' output shift
RES_BITS = 15                           # the int32 residual's exponent
PHASES = 64                             # conv1's stride
KEY_Q = 64                              # the attention's key quantum (16 x the V interleave 4)
PV_KW = 4                               # the V caches' interleave
ENC_C = 1024                            # the encoder's (and the cross caches') rows: the softmax unit's keys
BUCKETS = (128, 256, 512, 1024)         # encoder frames per entry (3.1 / 6.2 / 12.3 / 24.6 s)
ACC = 65536                             # ConvKernel's accumulator entries per output row


def samples(T: int) -> int:
    """The samples of a bucket of T encoder frames: 384 per frame + 511."""
    return 384 * T + 511


def stem_frames(n: int):
    """(t1, t2, T) valid frames after conv1, conv2, conv3 of n samples."""
    from .asr_nodes import stem_frames as sf
    return sf(n)


@dataclass
class MoonshineConfig:
    d: int                  # hidden size
    H: int                  # heads (encoder = decoder)
    hd: int                 # head dim
    HDp: int                # the padded head dim (a multiple of 16)
    rot: int                # rotary dims
    ff: int                 # intermediate size
    L: int                  # encoder layers
    Ld: int                 # decoder layers
    V: int                  # vocabulary
    theta: float
    bos: int
    eos: int
    max_pos: int

    @classmethod
    def from_dict(cls, c: dict) -> "MoonshineConfig":
        d, H = int(c["hidden_size"]), int(c["encoder_num_attention_heads"])
        if int(c["decoder_num_attention_heads"]) != H or int(c["encoder_num_key_value_heads"]) != H \
                or int(c["decoder_num_key_value_heads"]) != H:
            raise ValueError("Moonshine: one head count, no grouped K / V")
        hd = d // H
        return cls(d=d, H=H, hd=hd, HDp=-(-hd // 16) * 16, rot=int(hd * float(c["partial_rotary_factor"])),
                   ff=int(c["intermediate_size"]), L=int(c["encoder_num_hidden_layers"]),
                   Ld=int(c["decoder_num_hidden_layers"]), V=int(c["vocab_size"]),
                   theta=float(c["rope_theta"]), bos=int(c["decoder_start_token_id"]),
                   eos=int(c["eos_token_id"]), max_pos=int(c["max_position_embeddings"]))

    @classmethod
    def from_file(cls, path: str) -> "MoonshineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def DP(self) -> int:
        """The attention width with padded heads."""
        return self.H * self.HDp

    @property
    def dec_C(self) -> int:
        """The decoder's self-attention cache rows: the positions, rounded up to KEY_Q."""
        return -(-self.max_pos // KEY_Q) * KEY_Q


class MoonshineFormats:
    """Exponents per ``class@<part><layer>`` (part e / d / s / h: encoder,
    decoder, stem, head) — asr_study.py formats; the unpadded model's channels."""

    def __init__(self, d: dict, cfg: MoonshineConfig):
        self.cfg = cfg
        self.exp = dict(d["exponents"])
        self.policy = d.get("policy", "")

    @classmethod
    def from_file(cls, path: str, cfg: MoonshineConfig) -> "MoonshineFormats":
        with open(path) as f:
            return cls(json.load(f), cfg)

    def get(self, cls_: str, part: str, layer: int, n: int) -> np.ndarray:
        """moonshine_q.QM.E: a scalar broadcast, a per-head vector repeated over the heads'
        channels where n = heads * hd."""
        v = np.asarray(self.exp.get(f"{cls_}@{part}{layer}", F), np.int64).reshape(-1)
        if v.size == 1:
            return np.full(n, int(v[0]), np.int64)
        if v.size == n:
            return v.copy()
        if n % v.size:
            raise ValueError(f"formats: {cls_}@{part}{layer} has {v.size} values, {n} wanted")
        return np.repeat(v, n // v.size)

    def heads(self, cls_: str, part: str, layer: int) -> np.ndarray:
        return self.get(cls_, part, layer, self.cfg.H)


def _compact(e):
    e = np.asarray(e, np.int64).reshape(-1)
    return int(e[0]) if (e == e[0]).all() else [int(v) for v in e]


# ------------------------------------------------------------------ #
# Padded / permuted heads                                              #
# ------------------------------------------------------------------ #

def rope_perm(hd: int, rot: int) -> np.ndarray:
    """moonshine_q.rope_perm: the head dims in rotate-half order (pair j of the
    interleaved RoPE at (j, j + hd / 2), the pass-through dims paired)."""
    half, npair = hd // 2, rot // 2
    order = np.empty(hd, np.int64)
    order[:npair] = np.arange(0, rot, 2)
    order[half: half + npair] = np.arange(1, rot, 2)
    rest = np.arange(rot, hd)
    order[npair:half] = rest[: half - npair]
    order[half + npair:] = rest[half - npair:]
    return order


def head_map(cfg: MoonshineConfig, rope: bool) -> np.ndarray:
    """[H * HDp] -> the source channel of the unpadded [H * hd] (-1: a zero channel):
    rope — the rotate-half order with each half padded (pairs (j, j + HDp / 2)), else
    the head's channels then zeros."""
    hd, P = cfg.hd, cfg.HDp
    m = np.full((cfg.H, P), -1, np.int64)
    for h in range(cfg.H):
        if rope:
            order = rope_perm(hd, cfg.rot)
            m[h, :hd // 2] = h * hd + order[:hd // 2]
            m[h, P // 2:P // 2 + hd // 2] = h * hd + order[hd // 2:]
        else:
            m[h, :hd] = h * hd + np.arange(hd)
    return m.reshape(-1)


def map_cols(W: np.ndarray, m: np.ndarray) -> np.ndarray:
    """W [K][N] -> [K][len(m)]: column j = W[:, m[j]] (0 where m[j] < 0)."""
    out = np.zeros((W.shape[0], m.size), np.float32)
    ok = m >= 0
    out[:, ok] = W[:, m[ok]]
    return out


def map_rows(W: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.zeros((m.size, W.shape[1]), np.float32)
    ok = m >= 0
    out[ok] = W[m[ok]]
    return out


def map_exp(e: np.ndarray, m: np.ndarray, P: int) -> np.ndarray:
    """Exponents over the padded channels: a zero channel takes its head's first
    real channel's (any value is right for zeros; this keeps per-head vectors
    per head)."""
    out = np.empty(m.size, np.int64)
    for i, s in enumerate(m):
        out[i] = e[s] if s >= 0 else e[m[(i // P) * P]]
    return out


def rope_half_tables(cfg: MoonshineConfig, n: int):
    """float32 cos / sin [n][HDp / 2] of the padded rotate-half pairs:
    moonshine_q.rope_half_tables' values for the first hd / 2 pairs, cos 1 / sin 0
    for the pass-through and the zero pairs."""
    th, rot, P = cfg.theta, cfg.rot, cfg.HDp
    inv = np.array([1.0 / (th ** (2 * j / rot)) for j in range(rot // 2)], np.float64).astype(np.float32)
    ang = (np.arange(n, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    cs = np.vectorize(math.cos, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    sn = np.vectorize(math.sin, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    c = np.ones((n, P // 2), np.float32)
    s = np.zeros((n, P // 2), np.float32)
    c[:, :rot // 2], s[:, :rot // 2] = cs, sn
    return c, s


def fold_dims(n_out: int, out_ch: int, stride: int, k: int):
    """(rows, w, width) of a 1-D conv's folded input: w outputs per row with
    w * roundup(out_ch, 16) <= ACC, rows * w >= n_out; each input row width =
    stride * w + k - stride (the halo)."""
    o16 = -(-out_ch // 16) * 16
    w = max(1, min(ACC // o16, n_out))
    rows = -(-n_out // w)
    w = -(-n_out // rows)                       # balance the rows
    return rows, w, stride * w + k - stride


class MoonshineFrontend:
    """Builds the entries of one model.  ``weights``: the checkpoint's tensors by
    Hugging Face name (float32); ``formats``: MoonshineFormats."""

    def __init__(self, cfg: MoonshineConfig, weights: Dict[str, np.ndarray], formats: MoonshineFormats,
                 name: str = "moonshine"):
        from . import rowops_nodes, smx_nodes, vectorop_act
        if not rowops_nodes.enabled():
            raise ValueError("Moonshine needs the row-ops unit (kernels.rowops.unit)")
        if not smx_nodes.enabled() or not vectorop_act.enabled():
            raise ValueError("Moonshine needs VectorOPKernel's softmax and activation units")
        if cfg.HDp > 128 or cfg.HDp % 16:
            raise ValueError(f"head dim {cfg.hd} -> {cfg.HDp}: the unit's ROPE takes 16 .. 128")
        self.cfg, self.W, self.fmt, self.name = cfg, weights, formats, name
        self.mq = head_map(cfg, rope=True)            # q / k of the self attentions
        self.mp = head_map(cfg, rope=False)           # v, the cross q / k / v
        for li in range(cfg.L):
            fa = formats.get("a", "e", li, cfg.ff)
            if not (fa == 8).all():
                raise ValueError(f"formats: the encoder's GELU output of layer {li} must be at 2^-8 "
                                 f"(VectorOPKernel's activation unit)")

    # ---- builder helpers (src/llama.py's conventions) ------------------------ #
    def _new(self, ename):
        self._nodes: List[onnx.NodeProto] = []
        self._vi: Dict[str, onnx.ValueInfoProto] = {}
        self._inits: Dict[str, np.ndarray] = {}
        self._meta = numeric.empty()
        self._e = ename

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, chexp=None, host=None, state=False, dma=None):
        if name not in self._vi:
            self._vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if dma is not None:
            self._meta["dma"][name] = dma
        if exp is not None:
            self._meta["exp"][name] = _compact(exp)
        if chexp is not None:
            self._meta["chexp"][name] = [int(v) for v in np.asarray(chexp).reshape(-1)]
        if host is not None:
            self._meta["host"][name] = host
        if state and name not in self._meta["state"]:
            self._meta["state"].append(name)
        return name

    def _init(self, name, arr):
        if name not in self._inits:
            self._inits[name] = np.asarray(arr, np.float32)
        return name

    def _node(self, op, ins, outs, name, domain=LLM_DOMAIN, **attrs):
        self._nodes.append(oh.make_node(op, ins, outs, name=name, domain=domain, **attrs))

    def _matmul(self, x, wname, W, T, y, exp, name):
        self._init(wname, W)
        self._t(y, [T, W.shape[1]], exp=exp)
        self._node("MatMul", [x, wname], [y], name, domain="")
        return y

    def _res(self, name, T):
        return self._t(name, [T, self.cfg.d], exp=RES_BITS, dma="i32")

    def _ln(self, h, gname, T, out, exp, name):
        g = self._init(f"g.{gname}", self.W[gname])
        b = self._init("ln.zero", np.zeros(self.cfg.d, np.float32))
        self._t(out, [T, self.cfg.d], exp=exp)
        self._node("VitLayerNorm", [h, g, b], [out], name, axi_eps="1e-05")
        return out

    def _w(self, name) -> np.ndarray:
        return np.asarray(self.W[name], np.float32)

    # ---- the encoder --------------------------------------------------------- #
    def encode_entry(self, T: int, output_xf: bool = False) -> onnx.ModelProto:
        """``encode_<T>``; ``output_xf``: also the encoder output xf as a graph output
        (tests)."""
        c, fm = self.cfg, self.fmt
        if T % KEY_Q or not 0 < T <= ENC_C:
            raise ValueError(f"bucket {T} frames: a multiple of {KEY_Q}, at most {ENC_C}")
        e = f"enc{T}"
        self._new(e)
        d, DP, P = c.d, c.DP, c.HDp
        N = samples(T)
        t2n = 2 * T + 1
        t1n = 3 * t2n + 4
        pcm = self._t(f"{e}.pcm", [N], exp=15, host="i16")
        n = self._t(f"{e}.n", [1], TensorProto.INT32, host="i32")
        r1, w1, _ = fold_dims(t1n, d, 1, 2)
        r2, w2, W2 = fold_dims(t2n, 2 * d, 3, 7)
        r3, w3, W3 = fold_dims(T, d, 2, 3)
        img = self._t(f"{e}.img", [1, PHASES, r1, w1 + 1], exp=15)
        self._node("AsrFrames", [pcm, n], [img], f"{e}.frames")
        w1c = self._w("model.encoder.conv1.weight")[:, 0, :]                     # [d][127]
        w1c = np.concatenate([w1c, np.zeros((d, 1), np.float32)], 1)             # 128 taps
        wconv1 = w1c.reshape(d, 2, PHASES).transpose(0, 2, 1)[:, :, None, :]      # [d][64][1][2]
        S = ("s", 0)
        c1 = self._t(f"{e}.c1", [1, d, r1, w1], chexp=fm.get("c1", *S, d))
        self._init("w.enc.conv1", wconv1)
        self._node("Conv", [img, "w.enc.conv1"], [c1], f"{e}.conv1", domain="", kernel_shape=[1, 2],
                   strides=[1, 1], pads=[0, 0, 0, 0])
        gn = self._t(f"{e}.gn", [1, d, r2, W2], chexp=fm.get("gn", *S, d))
        self._node("AsrStemNorm", [c1, n, self._init("enc.gn.g", self.W["model.encoder.groupnorm.weight"]),
                                   self._init("enc.gn.b", self.W["model.encoder.groupnorm.bias"])],
                   [gn], f"{e}.stem_norm", step=3 * w2, axi_eps="1e-05")
        c2 = self._t(f"{e}.c2", [1, 2 * d, r2, w2], chexp=fm.get("c2", *S, 2 * d))
        self._init("w.enc.conv2", self._w("model.encoder.conv2.weight")[:, :, None, :])
        self._init("b.enc.conv2", self.W["model.encoder.conv2.bias"])
        self._node("Conv", [gn, "w.enc.conv2", "b.enc.conv2"], [c2], f"{e}.conv2", domain="",
                   kernel_shape=[1, 7], strides=[1, 3], pads=[0, 0, 0, 0])
        g2 = self._t(f"{e}.g2", [1, 2 * d, r3, W3], chexp=fm.get("g2", *S, 2 * d))
        self._node("AsrStemGelu", [c2, n], [g2], f"{e}.gelu2", stage=2, step=2 * w3)
        c3 = self._t(f"{e}.c3", [1, d, r3, w3], chexp=fm.get("c3", *S, d))
        self._init("w.enc.conv3", self._w("model.encoder.conv3.weight")[:, :, None, :])
        self._init("b.enc.conv3", self.W["model.encoder.conv3.bias"])
        self._node("Conv", [g2, "w.enc.conv3", "b.enc.conv3"], [c3], f"{e}.conv3", domain="",
                   kernel_shape=[1, 3], strides=[1, 2], pads=[0, 0, 0, 0])
        h = self._res(f"{e}.h0", T)
        self._node("AsrStemGelu", [c3, n], [h], f"{e}.gelu3", stage=3, residual=1)
        nf = self._t(f"{e}.nf", [1], TensorProto.INT32, host="i32")
        self._node("AsrFrameCount", [n], [nf], f"{e}.frame_count", cap=T)
        zero = self._t(f"{e}.pos", [1], TensorProto.INT32, host="i32")     # the attention's start: 0
        # the RoPE table (the K / V caches per layer: their exponents are the layer's)
        cos, sin = rope_half_tables(c, ENC_C)
        self._init("asr.enc.cos", cos)
        self._init("asr.enc.sin", sin)
        qs = self._t(f"{e}.ro.qs", [T, DP], exp=0, state=True)
        ks = self._t(f"{e}.ro.ks", [T, DP], exp=0, state=True)
        vs = self._t(f"{e}.ro.vs", [T, DP], exp=0, state=True)
        for t in (qs, ks, vs):
            self._init(t, np.zeros((T, DP), np.float32))
        common = dict(num_heads=c.H, num_kv_heads=c.H, head_dim=P, key_quantum=KEY_Q)
        scale = repr(float(c.hd) ** -0.5)
        for li in range(c.L):
            lw = f"model.encoder.layers.{li}."
            el = f"{e}.l{li}"
            sc = ("e", li)
            x = self._ln(h, lw + "input_layernorm.weight", T, f"{el}.x", fm.get("x", *sc, d), f"{el}.ln1")
            q0 = self._matmul(x, f"w.enc.l{li}.q", map_cols(self._w(lw + "self_attn.q_proj.weight").T, self.mq), T,
                              f"{el}.q0", map_exp(fm.get("q0", *sc, d), self.mq, P), f"{el}.q_proj")
            k0 = self._matmul(x, f"w.enc.l{li}.k", map_cols(self._w(lw + "self_attn.k_proj.weight").T, self.mq), T,
                              f"{el}.k0", map_exp(fm.get("k0", *sc, d), self.mq, P), f"{el}.k_proj")
            v0 = self._matmul(x, f"w.enc.l{li}.v", map_cols(self._w(lw + "self_attn.v_proj.weight").T, self.mp), T,
                              f"{el}.v", map_exp(fm.get("v", *sc, d), self.mp, P), f"{el}.v_proj")
            fq, fk = fm.heads("q", *sc), fm.heads("k", *sc)
            fvc = map_exp(fm.get("vc", *sc, d), self.mp, P)
            kc = self._t(f"asr.enc.k{li}", [ENC_C, DP], exp=np.repeat(fk, P), state=True)
            vc = self._t(f"asr.enc.v{li}", [ENC_C, DP], exp=fvc, state=True)
            self._meta["layout"][kc] = [c.H, P]
            self._meta["layout"][vc] = [c.H, P, PV_KW]
            self._init(kc, np.zeros((ENC_C, DP), np.float32))
            self._init(vc, np.zeros((ENC_C, DP), np.float32))
            qx = self._t(f"{el}.qx", [c.H, P, T], exp=0)
            self._node("LlmAttnPrep", [q0, k0, v0, zero, nf, kc, vc, "asr.enc.cos", "asr.enc.sin", qs, ks, vs],
                       [qx], f"{el}.attn_prep", qk_kw=1, q_exp=[int(v) for v in fq], rowops=1,
                       first=int(li == 0), **common)
            fs, fp = fq + fk - F, fm.heads("p", *sc)
            o = []
            for g in range(c.H):
                s_ = self._t(f"{el}.s{g}", [ENC_C, T], exp=0)
                p_ = self._t(f"{el}.p{g}", [T, ENC_C], exp=0)
                o_ = self._t(f"{el}.o{g}", [T, P], exp=0)
                self._node("LlmAttnScores", [kc, qx, zero, nf], [s_], f"{el}.qk{g}", group=g, qk_kw=1, **common)
                self._node("LlmAttnSoftmax", [s_, zero, nf], [p_], f"{el}.softmax{g}", group=g,
                           s_exp=[int(fs[g])], p_exp=[int(fp[g])], vsmx=1, causal=0, logit_scale=scale, **common)
                self._node("LlmAttnPV", [p_, vc, zero, nf], [o_], f"{el}.pv{g}", group=g, **common)
                o.append(o_)
            pv = self._t(f"{el}.pv", [T, DP], exp=np.repeat(fp, P) + fvc - F)
            self._node("LlmAttnMerge", o, [pv], f"{el}.attn_merge", num_heads=c.H, num_kv_heads=c.H, head_dim=P)
            ov = self._matmul(pv, f"w.enc.l{li}.o", map_rows(self._w(lw + "self_attn.o_proj.weight").T, self.mp), T,
                              f"{el}.o", fm.get("o", *sc, d), f"{el}.o_proj")
            h1 = self._res(f"{el}.h1", T)
            self._node("LlmResAdd", [h, ov], [h1], f"{el}.res_attn")
            x2 = self._ln(h1, lw + "post_attention_layernorm.weight", T, f"{el}.x2", fm.get("x2", *sc, d),
                          f"{el}.ln2")
            fv = self._matmul(x2, f"w.enc.l{li}.f1", self._w(lw + "mlp.fc1.weight").T, T, f"{el}.f",
                              fm.get("f", *sc, c.ff), f"{el}.fc1")
            a = self._t(f"{el}.a", [T, c.ff], exp=8)
            from .vit import _gelu_vop_rows
            b1 = self._init(f"b.enc.l{li}.f1", self.W[lw + "mlp.fc1.bias"])
            vop = _gelu_vop_rows(self._inits[b1], fm.get("f", *sc, c.ff), np.full(c.ff, 8), c.ff)
            if vop is None:
                raise ValueError(f"layer {li}: fc1's exponents do not fit VectorOPKernel's GELU rows")
            self._node("VitGelu", [fv, b1, self._init(f"enc.l{li}.gelu_ba", vop[0]),
                                   self._init(f"enc.l{li}.gelu_sc", vop[1])], [a], f"{el}.gelu",
                       approximate="none")
            dd = self._matmul(a, f"w.enc.l{li}.f2", self._w(lw + "mlp.fc2.weight").T, T, f"{el}.d",
                              fm.get("d", *sc, d), f"{el}.fc2")
            h2 = self._res(f"{el}.h2", T)
            self._node("VitResAdd", [h1, dd, self._init(f"b.enc.l{li}.f2", self.W[lw + "mlp.fc2.bias"])], [h2],
                       f"{el}.res_mlp")
            h = h2
        xf = self._ln(h, "model.encoder.layer_norm.weight", T, f"{e}.xf", fm.get("xf", "e", c.L, d), f"{e}.ln_f")
        for li in range(c.Ld):
            lw = f"model.decoder.layers.{li}."
            sc = ("d", li)
            fck = map_exp(fm.get("ck", *sc, d), self.mp, P)
            fcv = map_exp(fm.get("cv", *sc, d), self.mp, P)
            k0 = self._matmul(xf, f"w.dec.l{li}.ck", map_cols(self._w(lw + "encoder_attn.k_proj.weight").T, self.mp),
                              T, f"{e}.ck{li}", fck, f"{e}.cross_k{li}")
            v0 = self._matmul(xf, f"w.dec.l{li}.cv", map_cols(self._w(lw + "encoder_attn.v_proj.weight").T, self.mp),
                              T, f"{e}.cv{li}", fcv, f"{e}.cross_v{li}")
            ckc, cvc = self._cross_caches(li, written=True)
            self._node("AsrCrossPrep", [k0, v0, cvc], [ckc], f"{e}.cross_prep{li}", num_heads=c.H, head_dim=P)
        outputs = []
        if output_xf:
            xo = self._t(f"{e}.xf_out", [T, d], host="f32")
            self._node("LlmDequant", [xf], [xo], f"{e}.xf_dequant")
            outputs.append(xo)
        self._meta["test_fill"][n] = samples(T) - 384 * 3
        self._meta["test_fill"][zero] = 0
        return self._model(e, [pcm, n, zero], outputs, {"entry": e, "frames": T, "samples": N})

    def _cross_caches(self, li, written: bool = False):
        """The cross K / V caches of decoder layer li; ``written``: the encode entry,
        whose AsrCrossPrep outputs the K cache (a node output, not an initializer)."""
        c, fm, P = self.cfg, self.fmt, self.cfg.HDp
        sc = ("d", li)
        ckc = self._t(f"asr.cross.k{li}", [ENC_C, c.DP], state=True,
                      exp=map_exp(fm.get("ck", *sc, c.d), self.mp, P))
        cvc = self._t(f"asr.cross.v{li}", [ENC_C, c.DP], state=True,
                      exp=map_exp(fm.get("cv", *sc, c.d), self.mp, P))
        self._meta["layout"][ckc] = [c.H, P]
        self._meta["layout"][cvc] = [c.H, P, PV_KW]
        if not written:
            self._init(ckc, np.zeros((ENC_C, c.DP), np.float32))
        self._init(cvc, np.zeros((ENC_C, c.DP), np.float32))
        return ckc, cvc

    # ---- the decoder --------------------------------------------------------- #
    def decode_entry(self) -> onnx.ModelProto:
        c, fm = self.cfg, self.fmt
        e = "decode"
        self._new(e)
        d, DP, P, C = c.d, c.DP, c.HDp, c.dec_C
        ids = self._t(f"{e}.ids", [1], TensorProto.INT32, host="i32")
        pos = self._t(f"{e}.pos", [1], TensorProto.INT32, host="i32")
        xpos = self._t(f"{e}.xpos", [1], TensorProto.INT32, host="i32")
        self._init("emb", self.W["model.decoder.embed_tokens.weight"])
        h = self._res(f"{e}.h0", 1)
        self._node("LlmEmbed", [ids, "emb"], [h], f"{e}.embed")
        cos, sin = rope_half_tables(c, C)
        self._init("asr.dec.cos", cos)
        self._init("asr.dec.sin", sin)
        common = dict(num_heads=c.H, num_kv_heads=c.H, head_dim=P, key_quantum=KEY_Q)
        scale = repr(float(c.hd) ** -0.5)
        for li in range(c.Ld):
            lw = f"model.decoder.layers.{li}."
            el = f"{e}.l{li}"
            sc = ("d", li)
            y = self._ln(h, lw + "input_layernorm.weight", 1, f"{el}.y", fm.get("y", *sc, d), f"{el}.ln1")
            q0 = self._matmul(y, f"w.dec.l{li}.q", map_cols(self._w(lw + "self_attn.q_proj.weight").T, self.mq), 1,
                              f"{el}.q0", map_exp(fm.get("q0", *sc, d), self.mq, P), f"{el}.q_proj")
            k0 = self._matmul(y, f"w.dec.l{li}.k", map_cols(self._w(lw + "self_attn.k_proj.weight").T, self.mq), 1,
                              f"{el}.k0", map_exp(fm.get("k0", *sc, d), self.mq, P), f"{el}.k_proj")
            v0 = self._matmul(y, f"w.dec.l{li}.v", map_cols(self._w(lw + "self_attn.v_proj.weight").T, self.mp), 1,
                              f"{el}.v", map_exp(fm.get("v", *sc, d), self.mp, P), f"{el}.v_proj")
            fq, fk = fm.heads("q", *sc), fm.heads("k", *sc)
            fvc = map_exp(fm.get("vc", *sc, d), self.mp, P)
            kc = self._t(f"asr.dec.k{li}", [C, DP], exp=np.repeat(fk, P), state=True)
            vc = self._t(f"asr.dec.v{li}", [C, DP], exp=fvc, state=True)
            self._meta["layout"][kc] = [c.H, P]
            self._meta["layout"][vc] = [c.H, P, PV_KW]
            self._init(kc, np.zeros((C, DP), np.float32))
            self._init(vc, np.zeros((C, DP), np.float32))
            qs = self._t(f"{e}.ro.qs", [1, DP], exp=0, state=True)
            ks = self._t(f"{e}.ro.ks", [1, DP], exp=0, state=True)
            vs = self._t(f"{e}.ro.vs{li}", [1, DP], exp=0, state=True)
            for t in (qs, ks, vs):
                self._init(t, np.zeros((1, DP), np.float32))
            qx = self._t(f"{el}.qx", [c.H, P, 1], exp=0)
            self._node("LlmAttnPrep", [q0, k0, v0, pos, "", kc, vc, "asr.dec.cos", "asr.dec.sin", qs, ks, vs],
                       [qx], f"{el}.attn_prep", qk_kw=1, q_exp=[int(v) for v in fq], rowops=1, **common)
            fs, fp = fq + fk - F, fm.heads("p", *sc)
            o = []
            for g in range(c.H):
                s_ = self._t(f"{el}.s{g}", [C, 1], exp=0)
                p_ = self._t(f"{el}.p{g}", [1, C], exp=0)
                o_ = self._t(f"{el}.o{g}", [1, P], exp=0)
                self._node("LlmAttnScores", [kc, qx, pos, ""], [s_], f"{el}.qk{g}", group=g, qk_kw=1, **common)
                self._node("LlmAttnSoftmax", [s_, pos, ""], [p_], f"{el}.softmax{g}", group=g,
                           s_exp=[int(fs[g])], p_exp=[int(fp[g])], logit_scale=scale, **common)
                self._node("LlmAttnPV", [p_, vc, pos, ""], [o_], f"{el}.pv{g}", group=g, **common)
                o.append(o_)
            pv = self._t(f"{el}.pv", [1, DP], exp=np.repeat(fp, P) + fvc - F)
            self._node("LlmAttnMerge", o, [pv], f"{el}.attn_merge", num_heads=c.H, num_kv_heads=c.H, head_dim=P)
            ov = self._matmul(pv, f"w.dec.l{li}.o", map_rows(self._w(lw + "self_attn.o_proj.weight").T, self.mp), 1,
                              f"{el}.o", fm.get("o", *sc, d), f"{el}.o_proj")
            h1 = self._res(f"{el}.h1", 1)
            self._node("LlmResAdd", [h, ov], [h1], f"{el}.res_attn")
            # cross attention over the encoder's caches (keys = xpos + 1 valid)
            y2 = self._ln(h1, lw + "post_attention_layernorm.weight", 1, f"{el}.y2", fm.get("y2", *sc, d),
                          f"{el}.ln2")
            fcq, fck = fm.heads("cq", *sc), fm.heads("ck", *sc)
            cq = self._matmul(y2, f"w.dec.l{li}.cq", map_cols(self._w(lw + "encoder_attn.q_proj.weight").T, self.mp),
                              1, f"{el}.cq", np.repeat(fcq, P), f"{el}.cross_q")
            cqx = self._t(f"{el}.cqx", [c.H, P, 1], exp=0)
            self._node("AsrCrossQ", [cq], [cqx], f"{el}.cross_q_image")
            ckc, cvc = self._cross_caches(li)
            fcs, fcp = fcq + fck - F, fm.heads("cp", *sc)
            fcv = map_exp(fm.get("cv", *sc, d), self.mp, P)
            co = []
            for g in range(c.H):
                s_ = self._t(f"{el}.cs{g}", [ENC_C, 1], exp=0)
                p_ = self._t(f"{el}.cp{g}", [1, ENC_C], exp=0)
                o_ = self._t(f"{el}.co{g}", [1, P], exp=0)
                self._node("LlmAttnScores", [ckc, cqx, xpos, ""], [s_], f"{el}.cqk{g}", group=g, qk_kw=1, **common)
                self._node("LlmAttnSoftmax", [s_, xpos, ""], [p_], f"{el}.csoftmax{g}", group=g,
                           s_exp=[int(fcs[g])], p_exp=[int(fcp[g])], logit_scale=scale, **common)
                self._node("LlmAttnPV", [p_, cvc, xpos, ""], [o_], f"{el}.cpv{g}", group=g, **common)
                co.append(o_)
            cpv = self._t(f"{el}.cpv", [1, DP], exp=np.repeat(fcp, P) + fcv - F)
            self._node("LlmAttnMerge", co, [cpv], f"{el}.cross_merge", num_heads=c.H, num_kv_heads=c.H, head_dim=P)
            cov = self._matmul(cpv, f"w.dec.l{li}.co",
                               map_rows(self._w(lw + "encoder_attn.o_proj.weight").T, self.mp), 1, f"{el}.co",
                               fm.get("co", *sc, d), f"{el}.cross_o")
            h2 = self._res(f"{el}.h2", 1)
            self._node("LlmResAdd", [h1, cov], [h2], f"{el}.res_cross")
            # SwiGLU
            y3 = self._ln(h2, lw + "final_layernorm.weight", 1, f"{el}.y3", fm.get("y3", *sc, d), f"{el}.ln3")
            u = self._matmul(y3, f"w.dec.l{li}.f1", self._w(lw + "mlp.fc1.weight").T, 1, f"{el}.u",
                             fm.get("u", *sc, 2 * c.ff), f"{el}.fc1")
            ub = self._t(f"{e}.ro.ub{li}", [1, c.ff], exp=fm.get("ub", *sc, c.ff), state=True)
            gb = self._t(f"{e}.ro.gb{li}", [1, c.ff], exp=fm.get("gb", *sc, c.ff), state=True)
            self._init(ub, np.zeros((1, c.ff), np.float32))
            self._init(gb, np.zeros((1, c.ff), np.float32))
            a = self._t(f"{el}.a", [1, c.ff], exp=fm.get("a", *sc, c.ff))
            self._node("AsrSwiGLU", [u, self._init(f"b.dec.l{li}.f1", self.W[lw + "mlp.fc1.bias"]), ub, gb], [a],
                       f"{el}.swiglu")
            dd = self._matmul(a, f"w.dec.l{li}.f2", self._w(lw + "mlp.fc2.weight").T, 1, f"{el}.d",
                              fm.get("d", *sc, d), f"{el}.fc2")
            h3 = self._res(f"{el}.h3", 1)
            self._node("VitResAdd", [h2, dd, self._init(f"b.dec.l{li}.f2", self.W[lw + "mlp.fc2.bias"])], [h3],
                       f"{el}.res_mlp")
            h = h3
        yf = self._ln(h, "model.decoder.norm.weight", 1, f"{e}.yf", fm.get("yf", "h", 0, d), f"{e}.ln_f")
        lq = self._matmul(yf, "w.lm_head", self._w("model.decoder.embed_tokens.weight").T, 1, f"{e}.logits",
                          fm.get("logits", "h", 0, 1)[0], f"{e}.lm_head")
        tid = self._t(f"{e}.id", [1], TensorProto.INT32, host="i32")
        self._node("AsrArgmax", [lq], [tid], f"{e}.argmax")
        self._meta["test_fill"][pos] = 3
        self._meta["test_fill"][xpos] = 100
        return self._model(e, [ids, pos, xpos], [tid], {"entry": e})

    def entries(self, buckets=BUCKETS) -> Dict[str, onnx.ModelProto]:
        out = {f"encode_{T}": self.encode_entry(T) for T in buckets}
        out["decode"] = self.decode_entry()
        return out

    def _model(self, ename, inputs, outputs, info):
        g = graph_with_initializers(
            self._nodes, f"{self.name}_{ename}",
            [self._vi[n] for n in inputs], [self._vi[n] for n in outputs], self._inits,
            value_info=[vi for n, vi in self._vi.items() if n not in inputs and n not in outputs])
        self._inits = {}
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/moonshine.py")
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key = numeric.METADATA_KEY
        p.value = numeric.to_metadata(self._meta)
        p = m.metadata_props.add()
        p.key = "axi.llm.entry"
        c = self.cfg
        p.value = json.dumps(dict(info, model=self.name, hidden=c.d, heads=c.H, head_dim=c.hd, head_pad=c.HDp,
                                  enc_layers=c.L, dec_layers=c.Ld, vocab=c.V, bos=c.bos, eos=c.eos,
                                  max_pos=c.max_pos, cache_rows=ENC_C, dec_cache_rows=c.dec_C))
        return m


__all__ = ("MoonshineConfig", "MoonshineFormats", "MoonshineFrontend", "BUCKETS", "samples", "stem_frames",
           "rope_perm", "head_map", "rope_half_tables", "fold_dims")
