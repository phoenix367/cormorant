"""Piper (VITS) inference in numpy, with the fixed-point emulation of the
study's KV260 partition (doc/plans/TTS_PLAN.md §3) and the bit-exact
specifications of libpiper_tts.so: chunk_forward (§4), encoder_forward (§6)
and duration_predictor_seq (§7).

The float path (``Q = None``) is the reference: float64, the same
computation as Piper's ONNX export (``piper_study.py validate`` checks it
against onnxruntime).  With a quantizer ``Q`` the flow and the decoder run
as the board would: every Conv / ConvTranspose on ConvKernel (raw int16
operands at power-of-two exponents, the sum exact in ap_fixed<32,16>,
floor(acc / 2^8) saturated to int16), every other op on the host in double
with round-half-even + saturate where it writes an int16 tensor.  In that
emulation the text encoder, the duration predictor and the alignment stay
in float64, so every policy sees the same durations and noise and the
waveforms align sample by sample.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

SR = 22050
HOP = 256


# ---- weights -------------------------------------------------------------- #

def load_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """{VITS module name: float64 array} from the ONNX export.  The flow's
    weight-normed convs are exported as ``onnx::Conv_*``; their names come
    from the bias input of the same node.  The duration predictor's
    ElementwiseAffine ``exp(-logs)`` is folded into the graph: evaluated
    once with onnxruntime."""
    import onnx
    from onnx import numpy_helper as nph
    m = onnx.load(onnx_path)
    raw = {i.name: nph.to_array(i).astype(np.float64) for i in m.graph.initializer}
    W: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if not k.startswith("onnx::") and not k[0].isdigit():
            W[k] = v
    W["emb"] = W.pop("sid")                    # the phoneme embedding [256][192]
    for n in m.graph.node:
        if n.op_type == "Conv" and n.input[1].startswith("onnx::") and len(n.input) > 2:
            W[n.input[2][: -len(".bias")] + ".weight"] = raw[n.input[1]]
    W["dp.flows.0.exp_neg_logs"] = _eval(m, "/dp/flows.0/Exp_output_0")
    return W


def _eval(model, name: str) -> np.ndarray:
    import onnx
    import onnxruntime as ort
    from onnx import helper as oh
    g = onnx.ModelProto()
    g.CopyFrom(model)
    g.graph.output.append(oh.make_empty_tensor_value_info(name))
    s = ort.InferenceSession(g.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {"input": np.array([[1, 20, 2]], np.int64), "input_lengths": np.array([3], np.int64),
             "scales": np.array([0.0, 1.0, 0.0], np.float32)}
    out = s.run([name], feeds)[0]
    return np.asarray(out, np.float64).reshape(-1)


# ---- float building blocks ------------------------------------------------- #

def conv1d(x, w, b=None, dil=1, pad=None):
    """x [C, T], w [O, C/g, k] (groups = C when w.shape[1] == 1 and C > 1)."""
    n_out, Cg, k = w.shape
    pad = (k * dil - dil) // 2 if pad is None else pad
    C, T = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L = T + 2 * pad - dil * (k - 1)
    y = np.zeros((n_out, L))
    if Cg == 1 and C > 1:                      # depthwise
        for t in range(k):
            y += w[:, 0, t][:, None] * xp[:, t * dil: t * dil + L]
    else:
        for t in range(k):                     # contiguous: BLAS (a strided tap is numpy's
            y += np.ascontiguousarray(w[:, :, t]) @ xp[:, t * dil: t * dil + L]   # naive loop)
    if b is not None:
        y += b[:, None]
    return y


def conv_transpose1d(x, w, b, stride, pad):
    """x [C, T], w [C, O, k]; output length (T - 1) * stride - 2 * pad + k."""
    C, T = x.shape
    _, n_out, k = w.shape
    full = np.zeros((n_out, (T - 1) * stride + k))
    for t in range(k):
        full[:, t: t + (T - 1) * stride + 1: stride] += np.ascontiguousarray(w[:, :, t].T) @ x
    y = full[:, pad: full.shape[1] - pad]
    return y + b[:, None]


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(0, keepdims=True)
    v = ((x - m) ** 2).mean(0, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g[:, None] + b[:, None]


_ERF = np.vectorize(math.erf, otypes=[np.float64])


def erf_fast(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorized: the erf of
    gelu(fast=True), which _gelu_fast computes in place, and of the C
    duration predictor (tts_dp.c); math.erf element by element costs
    seconds on the A53."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return np.sign(x) * (1.0 - p * np.exp(-a * a))


def _gelu_fast(x):
    """0.5 * x * (1 + erf_fast(x / sqrt 2)) with in-place ufuncs, the same
    operations in the same order (bit-identical): 5 temporaries instead of
    ~14 (the A53 front end spent half the duration predictor in them)."""
    u = x / math.sqrt(2.0)
    a = np.abs(u)
    t = a * 0.3275911
    t += 1.0
    np.divide(1.0, t, out=t)
    p = t * 1.061405429
    p -= 1.453152027
    p *= t
    p += 1.421413741
    p *= t
    p -= 0.284496736
    p *= t
    p += 0.254829592
    p *= t
    np.negative(a, out=t)
    t *= a
    np.exp(t, out=t)
    p *= t
    np.subtract(1.0, p, out=p)
    p *= np.sign(u)
    p += 1.0
    h = x * 0.5
    h *= p
    return h


def gelu(x, fast=False):
    if fast:
        return _gelu_fast(x)
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def leaky(x, a):
    return np.where(x > 0, x, a * x)


# ---- text encoder ---------------------------------------------------------- #

def text_encoder(W, ids):
    x = W["emb"][ids].T * math.sqrt(192)                     # [192, T]
    T = x.shape[1]
    for i in range(6):
        p = f"enc_p.encoder.attn_layers.{i}"
        q = conv1d(x, W[f"{p}.conv_q.weight"], W[f"{p}.conv_q.bias"])
        k = conv1d(x, W[f"{p}.conv_k.weight"], W[f"{p}.conv_k.bias"])
        v = conv1d(x, W[f"{p}.conv_v.weight"], W[f"{p}.conv_v.bias"])
        ek, ev = W[f"{p}.emb_rel_k"][0], W[f"{p}.emb_rel_v"][0]   # [9][96], offsets -4..4
        out = np.zeros_like(q)
        for h in range(2):
            qh, kh, vh = q[h * 96:(h + 1) * 96].T / math.sqrt(96), k[h * 96:(h + 1) * 96].T, v[h * 96:(h + 1) * 96].T
            s = qh @ kh.T                                    # [T, T]
            for d in range(-4, 5):
                idx = np.arange(max(0, -d), min(T, T - d))
                s[idx, idx + d] += qh[idx] @ ek[d + 4]
            pa = softmax(s, -1)
            o = pa @ vh
            for d in range(-4, 5):
                idx = np.arange(max(0, -d), min(T, T - d))
                o[idx] += pa[idx, idx + d][:, None] * ev[d + 4][None, :]
            out[h * 96:(h + 1) * 96] = o.T
        y = conv1d(out, W[f"{p}.conv_o.weight"], W[f"{p}.conv_o.bias"])
        x = layer_norm(x + y, W[f"enc_p.encoder.norm_layers_1.{i}.gamma"], W[f"enc_p.encoder.norm_layers_1.{i}.beta"])
        f = f"enc_p.encoder.ffn_layers.{i}"
        y = conv1d(x, W[f"{f}.conv_1.weight"], W[f"{f}.conv_1.bias"])
        y = conv1d(np.maximum(y, 0), W[f"{f}.conv_2.weight"], W[f"{f}.conv_2.bias"])
        x = layer_norm(x + y, W[f"enc_p.encoder.norm_layers_2.{i}.gamma"], W[f"enc_p.encoder.norm_layers_2.{i}.beta"])
    stats = conv1d(x, W["enc_p.proj.weight"], W["enc_p.proj.bias"])
    return x, stats[:192], stats[192:]


# ---- stochastic duration predictor (reverse) -------------------------------- #

def dds_conv(W, p, x, g=None, fast=False):
    if g is not None:
        x = x + g
    for i in range(3):
        y = conv1d(x, W[f"{p}.convs_sep.{i}.weight"], W[f"{p}.convs_sep.{i}.bias"], dil=3 ** i)
        y = gelu(layer_norm(y, W[f"{p}.norms_1.{i}.gamma"], W[f"{p}.norms_1.{i}.beta"]), fast)
        y = conv1d(y, W[f"{p}.convs_1x1.{i}.weight"], W[f"{p}.convs_1x1.{i}.bias"])
        y = gelu(layer_norm(y, W[f"{p}.norms_2.{i}.gamma"], W[f"{p}.norms_2.{i}.beta"]), fast)
        x = x + y
    return x


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def rq_spline_inverse(x, uw, uh, ud, tb=5.0, mbw=1e-3, mbh=1e-3, md=1e-3):
    """Inverse rational-quadratic spline with linear tails (nflows / VITS)."""
    nb = uw.shape[-1]
    inside = (x >= -tb) & (x <= tb)
    ud = np.pad(ud, ((0, 0), (1, 1)))
    const = math.log(math.exp(1 - md) - 1)
    ud[:, 0] = const
    ud[:, -1] = const
    widths = mbw + (1 - mbw * nb) * softmax(uw, -1)
    cw = np.pad(np.cumsum(widths, -1), ((0, 0), (1, 0)))
    cw = 2 * tb * cw - tb
    cw[:, 0], cw[:, -1] = -tb, tb
    widths = cw[:, 1:] - cw[:, :-1]
    der = md + _softplus(ud)
    heights = mbh + (1 - mbh * nb) * softmax(uh, -1)
    ch = np.pad(np.cumsum(heights, -1), ((0, 0), (1, 0)))
    ch = 2 * tb * ch - tb
    ch[:, 0], ch[:, -1] = -tb, tb
    heights = ch[:, 1:] - ch[:, :-1]
    loc = ch.copy()
    loc[:, -1] += 1e-6
    bi = np.clip((x[:, None] >= loc).sum(-1) - 1, 0, nb - 1)
    r = np.arange(len(x))
    icw, ibw, ich, ih = cw[r, bi], widths[r, bi], ch[r, bi], heights[r, bi]
    delta = heights / widths
    idl, idr, ide = der[r, bi], der[r, bi + 1], delta[r, bi]
    a = (x - ich) * (idl + idr - 2 * ide) + ih * (ide - idl)
    b = ih * idl - (x - ich) * (idl + idr - 2 * ide)
    c = -ide * (x - ich)
    root = (2 * c) / (-b - np.sqrt(b * b - 4 * a * c))
    return np.where(inside, root * ibw + icw, x)


def duration_predictor(W, x_enc, noise_w, rng, fast=False):
    x = conv1d(x_enc, W["dp.pre.weight"], W["dp.pre.bias"])
    x = dds_conv(W, "dp.convs", x, fast=fast)
    x = conv1d(x, W["dp.proj.weight"], W["dp.proj.bias"])
    T = x.shape[1]
    z = rng.standard_normal((2, T)) * noise_w
    for fi in (7, 5, 3):
        z = z[::-1]                                          # Flip
        p = f"dp.flows.{fi}"
        x0, x1 = z[:1], z[1]
        h = conv1d(x0, W[f"{p}.pre.weight"], W[f"{p}.pre.bias"])
        h = dds_conv(W, f"{p}.convs", h, g=x, fast=fast)
        h = conv1d(h, W[f"{p}.proj.weight"], W[f"{p}.proj.bias"])   # [29, T]
        hh = h.T
        uw, uh, ud = hh[:, :10] / math.sqrt(192), hh[:, 10:20] / math.sqrt(192), hh[:, 20:]
        z = np.stack([x0[0], rq_spline_inverse(x1, uw, uh, ud)])
    z = z[::-1]                                              # the last Flip
    z = (z - W["dp.flows.0.m"]) * W["dp.flows.0.exp_neg_logs"][:, None]
    return z[0]                                              # logw [T]


# ---- fixed-point helpers ---------------------------------------------------- #

class Quant:
    """Exponent table + counters for one emulated run.  ``exp`` maps a tensor
    name to its power-of-two exponent f (value = raw * 2^-f); ``weights``:
    "int16" (rank-1 encoding at f_w = f_y + 8 - f_x) or "float" (the +fw
    bound).  ``bf16`` rounds tensors and weights to bfloat16 instead (the
    yardstick)."""

    def __init__(self, exp: Dict[str, int], weights: str = "int16", bf16: bool = False,
                 default: Optional[int] = None, host_chain: bool = False, dec_chain: bool = False):
        self.exp, self.weights, self.bf16, self.default = exp, weights, bf16, default
        self.host_chain = host_chain          # the flow's WN residual chain in float on the host
        self.dec_chain = dec_chain            # the decoder's residual chain in float on the host
        self.sat: Dict[str, int] = {}
        self.acc_peak = 0.0
        self.record: Optional[Dict[str, float]] = None        # calibration: name -> max |x|
        self.convs = None                                     # calibration: the convs' inputs

    def f(self, name):
        if name in self.exp:
            return self.exp[name]
        if self.default is not None:
            return self.default
        raise KeyError(f"no exponent for {name}")

    def q(self, name, x):
        """A host op writes int16 at the tensor's exponent: round-half-even + saturate."""
        if self.record is not None:
            self.record[name] = max(self.record.get(name, 0.0), float(np.abs(x).max()))
            return x
        if self.bf16:
            return _bf16(x)
        f = self.f(name)
        r = np.round(x * 2.0 ** f)
        n = int(((r < -32768) | (r > 32767)).sum())
        if n:
            self.sat[name] = self.sat.get(name, 0) + n
        return np.clip(r, -32768, 32767) * 2.0 ** -f

    def weight(self, w, f_w):
        if self.bf16:
            return _bf16(w)
        if self.weights == "float":
            return w
        r = np.round(w * 2.0 ** f_w)
        n = int(((r < -32768) | (r > 32767)).sum())
        if n:
            self.sat["weights"] = self.sat.get("weights", 0) + n
        return np.clip(r, -32768, 32767) * 2.0 ** -f_w

    def kernel_out(self, name, full):
        """ConvKernel write-back: floor(acc / 2^8) at the output exponent."""
        if self.record is not None:
            self.record[name] = max(self.record.get(name, 0.0), float(np.abs(full).max()))
            return full
        if self.bf16:
            return _bf16(full)
        f = self.f(name)
        acc = full * 2.0 ** (f + 8)
        self.acc_peak = max(self.acc_peak, float(np.abs(acc).max()))
        r = np.floor(full * 2.0 ** f)
        n = int(((r < -32768) | (r > 32767)).sum())
        if n:
            self.sat[name] = self.sat.get(name, 0) + n
        return np.clip(r, -32768, 32767) * 2.0 ** -f


def _bf16(x):
    a = np.asarray(x, np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16                   # round to nearest even
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def kconv(Q, name, x, xname, w, b, dil=1, pad=None, transpose=None):
    """A Conv / ConvTranspose on ConvKernel: x already int16 at xname's
    exponent; the weight encoded at f_w = f_y + 8 - f_x; the bias at f_y (in
    the accumulator); floor at the output exponent."""
    if Q is None:
        return (conv_transpose1d(x, w, b, *transpose) if transpose
                else conv1d(x, w, b, dil=dil, pad=pad))
    if Q.record is not None:
        y = (conv_transpose1d(x, w, b, *transpose) if transpose else conv1d(x, w, b, dil=dil, pad=pad))
        if Q.convs is not None:                              # calibration: keep a slice of the input
            Q.convs.append((name, x[:, :4096].copy(), w, b, dil, pad, transpose))
        return Q.kernel_out(name, y)
    if Q.bf16:
        wq, bq = _bf16(w), None if b is None else _bf16(b)
    else:
        if name + "#in" in Q.exp:                            # the host writes this conv's input
            xname = name + "#in"
            x = Q.q(xname, x)
        fx, fy = Q.f(xname), Q.f(name)
        wq = Q.weight(w, fy + 8 - fx)
        bq = None if b is None else np.clip(np.round(b * 2.0 ** fy), -32768, 32767) * 2.0 ** -fy
    y = conv_transpose1d(x, wq, bq, *transpose) if transpose else conv1d(x, wq, bq, dil=dil, pad=pad)
    return Q.kernel_out(name, y)


# ---- flow (reverse) and decoder --------------------------------------------- #

def flow_reverse(W, z, Q=None):
    q = (lambda n, x: x) if Q is None else Q.q
    for fi in (6, 4, 2, 0):
        z = z[::-1]                                          # Flip
        p = f"flow.flows.{fi}"
        x0, x1 = z[:96], z[96:]
        x0q = q(f"{p}.x0", x0)
        h = kconv(Q, f"{p}.pre", x0q, f"{p}.x0", W[f"{p}.pre.weight"], W[f"{p}.pre.bias"])
        out = 0.0
        for i in range(4):
            e = f"{p}.enc"
            xin = kconv(Q, f"{e}.in.{i}", h, f"{e}.h.{i}" if i else f"{p}.pre",
                        W[f"{e}.in_layers.{i}.weight"], W[f"{e}.in_layers.{i}.bias"])
            acts = q(f"{e}.acts.{i}", np.tanh(xin[:192]) * (1.0 / (1.0 + np.exp(-xin[192:]))))
            rs = kconv(Q, f"{e}.rs.{i}", acts, f"{e}.acts.{i}",
                       W[f"{e}.res_skip_layers.{i}.weight"], W[f"{e}.res_skip_layers.{i}.bias"])
            if i < 3:
                h = h + rs[:192] if Q is not None and Q.host_chain else q(f"{e}.h.{i + 1}", h + rs[:192])
                out = out + rs[192:]
            else:
                out = out + rs
        out = q(f"{p}.wn_out", out)
        m = kconv(Q, f"{p}.post", out, f"{p}.wn_out", W[f"{p}.post.weight"], W[f"{p}.post.bias"])
        z = np.concatenate([x0, x1 - m])                    # host, float
    return z


DEC_DIL = ((1, 2), (2, 6), (3, 12))


def decoder(W, z, Q=None):
    q = (lambda n, x: x) if Q is None else Q.q
    if Q is not None and Q.dec_chain:
        # the residual chain in float on the host; each conv's input copy is
        # written at its own exponent (kconv, "#in")
        q = lambda n, x: x                                   # noqa: E731
    x = q("dec.in", z)
    x = kconv(Q, "dec.pre", x, "dec.in", W["dec.conv_pre.weight"], W["dec.conv_pre.bias"])
    for i, (s, k) in enumerate(((8, 16), (8, 16), (4, 8))):
        a = q(f"dec.up{i}.in", leaky(x, 0.1))
        x = kconv(Q, f"dec.up{i}", a, f"dec.up{i}.in", W[f"dec.ups.{i}.weight"], W[f"dec.ups.{i}.bias"],
                  transpose=(s, (k - s) // 2))
        xs = 0.0
        for j in range(3):
            r = f"dec.rb{i}{j}"
            y = x
            for c in range(2):
                a = q(f"{r}.a{c}", leaky(y, 0.1))
                t = kconv(Q, f"{r}.c{c}", a, f"{r}.a{c}", W[f"dec.resblocks.{3 * i + j}.convs.{c}.weight"],
                          W[f"dec.resblocks.{3 * i + j}.convs.{c}.bias"], dil=DEC_DIL[j][c])
                y = q(f"{r}.y{c}", t + y)
            xs = xs + y
        x = q(f"dec.st{i}", xs / 3.0)
    a = q("dec.post.in", leaky(x, 0.01))
    y = kconv(Q, "dec.post", a, "dec.post.in", W["dec.conv_post.weight"], None)
    return np.tanh(y[0])


# ---- the whole synthesis ----------------------------------------------------- #

def synthesize(W, ids, noise_scale=0.667, length_scale=1.0, noise_w=0.8, seed=0,
               Q_flow=None, Q_dec=None, return_parts=False, z_seed=None):
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    x, m_p, logs_p = text_encoder(W, ids)
    logw = duration_predictor(W, x, noise_w, rng)
    w_ceil = np.ceil(np.exp(logw) * length_scale).astype(int)
    rep = np.repeat(np.arange(len(ids)), np.maximum(w_ceil, 0))
    mp, lp = m_p[:, rep], logs_p[:, rep]
    zr = rng if z_seed is None else np.random.default_rng(z_seed)
    z_p = mp + zr.standard_normal(mp.shape) * np.exp(lp) * noise_scale
    z = flow_reverse(W, z_p, Q_flow)
    wav = decoder(W, z, Q_dec)
    if return_parts:
        return wav, {"m_p": m_p, "logs_p": logs_p, "logw": logw, "w_ceil": w_ceil, "z": z}
    return wav


# ---- the library's chunk: the specification (TTS_PLAN §4) ------------------- #
#
# One call of the library computes 128 frames (32 768 samples, 1.49 s) of
# audio: the flow on FLOW_FRAMES frames of z_p, the decoder on the central
# DEC_FRAMES of them, the output's central OUT_FRAMES valid.  Frames of the
# chunk outside [lo, hi) are outside the utterance: every conv input (and
# the flow's state) is zero there, as in the full-utterance computation.
# The halos (flow 32 frames each side, decoder 32) exceed the receptive
# fields (flow 32, decoder ~13), so stitched chunks equal one long chunk.
# Rounding points: every conv input is written once by the host at its
# searched exponent (exp[name + "#in"]); kernel outputs are floored at the
# output exponent; residual sums are rounded once at the chain's exponent;
# the flow's state is float32; the k7 dilation-12 convs run as two tap
# groups (taps 0-3, 4-6: one call spans <= 64 columns), each floored.

FLOW_FRAMES, DEC_FRAMES, OUT_FRAMES = 256, 192, 128
DEC_OFF = (FLOW_FRAMES - DEC_FRAMES) // 2                       # 32
OUT_OFF = (FLOW_FRAMES - OUT_FRAMES) // 2                       # 64 (flow frames)
SPLIT_TAPS = 4                                                   # k7 d12: taps [0, 4) + [4, 7)


def _libm(fn, x):
    """fn from Python's math module (libm) element-wise, once per distinct
    value: the C host ops call the same libm (numpy's own tanh / exp may
    differ in the last bit)."""
    x = np.asarray(x, np.float64)
    u, inv = np.unique(x, return_inverse=True)
    return np.array([fn(float(v)) for v in u])[inv].reshape(x.shape)


def _sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def _rq(x, f):
    """Host write: round-half-even at 2^-f, saturate to int16 (value domain)."""
    return np.clip(np.round(x * 2.0 ** f), -32768, 32767) * 2.0 ** -f


def _mask(x, lo, hi):
    """Zero the columns outside [lo, hi)."""
    y = np.zeros_like(x)
    y[:, max(lo, 0):max(min(hi, x.shape[1]), 0)] = x[:, max(lo, 0):max(min(hi, x.shape[1]), 0)]
    return y


def _kconv(x, fx, fy, w, b, dil=1, taps=None):
    """ConvKernel on a 'same' conv over the chunk (zero padding at the chunk
    ends): x on its 2^-fx grid; weight at f_w = fy + 8 - fx, bias at fy;
    floor at fy, saturate.  ``taps`` = (t0, t1) computes that tap group only
    (no bias on the second group)."""
    fw = fy + 8 - fx
    w = np.asarray(w, np.float32).astype(np.float64)          # the library stores float32 weights
    b = None if b is None else np.asarray(b, np.float32).astype(np.float64)
    wq = np.clip(np.round(w * 2.0 ** fw), -32768, 32767) * 2.0 ** -fw
    k = w.shape[2]
    pad = (k - 1) * dil // 2
    t0, t1 = taps if taps else (0, k)
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L = x.shape[1]
    y = np.zeros((w.shape[0], L))
    for t in range(t0, t1):
        y += wq[:, :, t] @ xp[:, t * dil: t * dil + L]
    if b is not None and t0 == 0:
        y += (np.clip(np.round(b * 2.0 ** fy), -32768, 32767) * 2.0 ** -fy)[:, None]
    acc = y * 2.0 ** (fy + 8)                                  # ap_fixed<32,16>: an int32 that wraps
    big = np.abs(acc) >= 2.0 ** 31
    if big.any():
        acc = np.where(big, np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31, acc)
    return np.clip(np.floor(acc / 256.0), -32768, 32767) * 2.0 ** -fy


def polyphase_weight(w, b, s):
    """ConvTranspose1d (C -> O, kernel 2s, stride s, padding s/2) as a 'same'
    kernel-3 conv with s * O outputs (phase-major) + an interleave:
    y[o, m*s + r] = yp[r*O + o, m]; the bias repeats per phase."""
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


def vop_exponents(E):
    """The exponents of the decoder's residual sums on VectorOPKernel
    (doc/plans/OFFLOAD_PLAN.md §2.2): one exponent per decoder stage — the
    upsampler's output, every resblock conv's output and every residual
    y — the smallest of the calibrated ones, so ``y = rq(t + y)`` adds two
    raw tensors at one exponent (an exact sum: the rounding is the identity,
    VectorOP ADD's saturation is rq's)."""
    E2 = dict(E)
    for i in range(3):
        keys = [f"dec.up{i}"] + [f"dec.rb{i}{j}.{k}{c}" for j in range(3) for k in "cy" for c in range(2)]
        e = min(E[k] for k in keys)
        E2.update({k: e for k in keys})
    return E2


def chunk_forward(W, E, zp, lo, hi, dec_off=DEC_OFF, trace=None, vop_sums=True):
    """The library's chunk.  zp [192][FLOW_FRAMES] float32 (z_p, zero
    outside the utterance), [lo, hi) the utterance's frames in chunk
    coordinates, E the exponents (calibrate).  Returns int16 PCM
    [DEC_FRAMES * 256] (the central OUT_FRAMES * 256 valid).  The decoder
    runs on frames [dec_off, frames - dec_off) (tests: longer chunks).
    ``vop_sums`` (the library since doc/plans/OFFLOAD_PLAN.md §2.2): the
    decoder at vop_exponents(E), a split conv's two tap groups summed and
    saturated first — every residual sum one VectorOP ADD; False: the
    calibrated exponents, one host sum of three (the library before)."""
    tr = trace if trace is not None else {}
    if vop_sums:
        E = vop_exponents(E)
    z = np.asarray(zp, np.float32)
    for fi in (6, 4, 2, 0):
        p, e = f"flow.flows.{fi}", f"flow.flows.{fi}.enc"
        zf = z[::-1]                                           # Flip
        x0 = _rq(_mask(zf[:96].astype(np.float64), lo, hi), E[f"{p}.pre#in"])
        h = _kconv(x0, E[f"{p}.pre#in"], E[f"{p}.pre"], W[f"{p}.pre.weight"], W[f"{p}.pre.bias"])
        tr[f"{p}.pre#in"], tr[f"{p}.pre"] = x0, h
        rs_all = []
        for i in range(4):
            xin_in = _rq(_mask(h, lo, hi), E[f"{e}.in.{i}#in"])
            xin = _kconv(xin_in, E[f"{e}.in.{i}#in"], E[f"{e}.in.{i}"],
                         W[f"{e}.in_layers.{i}.weight"], W[f"{e}.in_layers.{i}.bias"])
            acts = _rq(_libm(math.tanh, xin[:192]) * _libm(_sigmoid, xin[192:]), E[f"{e}.rs.{i}#in"])
            rs = _kconv(acts, E[f"{e}.rs.{i}#in"], E[f"{e}.rs.{i}"],
                        W[f"{e}.res_skip_layers.{i}.weight"], W[f"{e}.res_skip_layers.{i}.bias"])
            rs_all.append(rs)
            tr[f"{e}.in.{i}"], tr[f"{e}.rs.{i}#in"], tr[f"{e}.rs.{i}"] = xin, acts, rs
            if i < 3:
                h = _rq(h + rs[:192], E[f"{e}.h.{i + 1}"])
                tr[f"{e}.h.{i + 1}"] = h
        wn = rs_all[0][192:] + rs_all[1][192:] + rs_all[2][192:] + rs_all[3]
        wn_in = _rq(_mask(wn, lo, hi), E[f"{p}.post#in"])
        m = _kconv(wn_in, E[f"{p}.post#in"], E[f"{p}.post"], W[f"{p}.post.weight"], W[f"{p}.post.bias"])
        tr[f"{p}.post#in"], tr[f"{p}.post"] = wn_in, m
        x1 = _mask((zf[96:].astype(np.float64) - m), lo, hi).astype(np.float32)
        z = np.concatenate([zf[:96], x1]).astype(np.float32)
        z[:96] = _mask(z[:96], lo, hi)
        tr[f"{p}.z"] = z.copy()
    # decoder on the central DEC_FRAMES frames
    dlo, dhi = lo - dec_off, hi - dec_off
    zd = z[:, dec_off:z.shape[1] - dec_off].astype(np.float64)
    xi = _rq(_mask(zd, dlo, dhi), E["dec.pre#in"])
    x = _kconv(xi, E["dec.pre#in"], E["dec.pre"], W["dec.conv_pre.weight"], W["dec.conv_pre.bias"])
    tr["dec.pre"] = x
    rate = 1
    for i, s in enumerate((8, 8, 4)):
        a = _rq(_mask(leaky(x, 0.1), dlo * rate, dhi * rate), E[f"dec.up{i}#in"])
        w3, b3 = polyphase_weight(W[f"dec.ups.{i}.weight"], W[f"dec.ups.{i}.bias"], s)
        yp = _kconv(a, E[f"dec.up{i}#in"], E[f"dec.up{i}"], w3, b3)
        n_out = w3.shape[0] // s
        xu = yp.reshape(s, n_out, -1).transpose(1, 2, 0).reshape(n_out, -1)   # interleave
        tr[f"dec.up{i}"], tr[f"dec.up{i}.x"] = yp, xu
        rate *= s
        ys = []
        for j in range(3):
            r = f"dec.rb{i}{j}"
            y = xu
            for c in range(2):
                dil = DEC_DIL[j][c]
                wt, bt = W[f"dec.resblocks.{3 * i + j}.convs.{c}.weight"], W[f"dec.resblocks.{3 * i + j}.convs.{c}.bias"]
                ain = _rq(_mask(leaky(y, 0.1), dlo * rate, dhi * rate), E[f"{r}.c{c}#in"])
                fx, fy = E[f"{r}.c{c}#in"], E[f"{r}.c{c}"]
                if (wt.shape[2] - 1) * dil + 1 > 64:
                    t = _kconv(ain, fx, fy, wt, bt, dil, taps=(0, SPLIT_TAPS)) + \
                        _kconv(ain, fx, fy, wt, bt, dil, taps=(SPLIT_TAPS, wt.shape[2]))
                    if vop_sums:
                        t = _rq(t, fy)
                else:
                    t = _kconv(ain, fx, fy, wt, bt, dil)
                y = _rq(t + y, E[f"{r}.y{c}"])
                tr[f"{r}.y{c}"] = y
            ys.append(y)
        x = _rq((ys[0] + ys[1] + ys[2]) / 3.0, E[f"dec.st{i}"])
        tr[f"dec.st{i}"] = x
    a = _rq(_mask(leaky(x, 0.01), dlo * rate, dhi * rate), E["dec.post#in"])
    yo = _kconv(a, E["dec.post#in"], E["dec.post"], W["dec.conv_post.weight"], None)
    return np.clip(np.round(_libm(math.tanh, yo[0]) * 32767.0), -32768, 32767).astype(np.int16)


def front_end(W, ids, noise_scale=0.667, length_scale=1.0, noise_w=0.8, seed=0, fast_erf=False,
              encoder=None, duration=None):
    """Host front end (float64): text encoder, durations, alignment, noise ->
    z_p [192][frames] (float32, as handed to the library).  fast_erf: the
    duration predictor's GELU on erf_fast.
    ``encoder``: ids -> (x, m_p, logs_p) [192][n] in place of text_encoder
    (the library's int16 encoder); ``duration``: (x, z) -> logw in place of
    duration_predictor (the library's C one; z = the noise * noise_w, drawn
    where duration_predictor draws it).  With both given, W is not read (the
    chat server passes None)."""
    rng = np.random.default_rng(seed)
    x, m_p, logs_p = (encoder or (lambda i: text_encoder(W, np.asarray(i))))(ids)
    if duration is None:
        logw = duration_predictor(W, x, noise_w, rng, fast_erf)
    else:
        logw = duration(x, rng.standard_normal((2, x.shape[1])) * noise_w)
    w_ceil = np.ceil(np.exp(logw) * length_scale).astype(int)
    rep = np.repeat(np.arange(len(ids)), np.maximum(w_ceil, 0))
    mp, lp = m_p[:, rep], logs_p[:, rep]
    return (mp + rng.standard_normal(mp.shape) * np.exp(lp) * noise_scale).astype(np.float32)


def synthesize_chunked(W, E, zp, vop_sums=True):
    """Stitch the chunks of an utterance: chunk k covers utterance frames
    [128k - 64, 128k + 192), its output frames [128k, 128k + 128)."""
    T = zp.shape[1]
    out = []
    for c0 in range(0, T, OUT_FRAMES):
        start = c0 - OUT_OFF
        chunk = np.zeros((192, FLOW_FRAMES), np.float32)
        a, b = max(start, 0), min(start + FLOW_FRAMES, T)
        chunk[:, a - start:b - start] = zp[:, a:b]
        pcm = chunk_forward(W, E, chunk, -start, T - start, vop_sums=vop_sums)
        o = (OUT_OFF - DEC_OFF) * HOP
        out.append(pcm[o:o + OUT_FRAMES * HOP])
    return np.concatenate(out)[:T * HOP]


# ---- the library's text encoder: the specification (TTS_PLAN §6) ------------ #
#
# The text encoder in row layout [n][192] on the FPGA: every projection and
# the FFN's kernel-3 convs are ConvKernel MatMuls (raw int16 operands,
# weights at f_w = f_y + 8 - f_x, exact sums, floor(acc / 2^8), saturate);
# q.K^T and P.V run on ConvKernel as well (the ViT's static-key attention).
# The host ops compute in double with sequential sums (as the C loops) and
# write int16 with round-half-even + saturation: the q / k / v prep (bias,
# per-head exponents), the softmax (relative keys, window 4), the merge
# (relative values), residual + LayerNorm, the kernel-3 conv inputs
# (im2col, the FFN's bias + ReLU), the float32 outputs.  Every MatMul input
# is written by a host op at its own searched exponent (the layer input for
# q / k / v and the projection's input as separate copies, the residual
# stream stays at its natural exponent): a coarser input puts the weight on
# a finer grid (f_w = f_y + 8 - f_x), as the chunk's conv inputs (#in).  The library pads
# the ids to a bucket of T >= n rows and masks: rows and keys >= n never
# reach rows < n, so the result does not depend on the bucket.

ENC_LAYERS, ENC_D, ENC_H, ENC_HD, ENC_FF, ENC_WIN, ENC_EPS = 6, 192, 2, 96, 768, 4, 1e-5
ENC_P = 15                                                      # the softmax output P at 2^-15


def _seqsum(a, axis=-1):
    """Sequential (left-to-right) sum, as the C loops add."""
    return np.cumsum(a, axis=axis).take(-1, axis=axis)


def _raw(v, f):
    """A host write in raw integers: round-half-even at 2^-f, saturate (+ 0.0:
    an int16 has no negative zero)."""
    return np.clip(np.round(np.asarray(v, np.float64) * 2.0 ** f), -32768, 32767) + 0.0


def _kmm(x_raw, w_raw):
    """ConvKernel MatMul on raw int16 operands: the exact sum (float64 holds
    it: |sum| < 2^53), the int32 wrap, floor(acc / 2^8), saturate -> raw."""
    acc = np.asarray(x_raw, np.float64) @ np.asarray(w_raw, np.float64)
    big = np.abs(acc) >= 2.0 ** 31
    if big.any():
        acc = np.where(big, np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31, acc)
    return np.clip(np.floor(acc / 256.0), -32768, 32767) + 0.0


def _k3(w):
    """Conv weight [O][C][3] -> the unrolled MatMul weight [3C][O] (tap-major)."""
    return np.concatenate([w[:, :, t].T for t in range(3)], axis=0)


def _im2col3(x):
    """[n][C] -> [n][3C]: row i holds rows i-1, i, i+1 ('same' padding, zeros outside)."""
    p = np.pad(x, ((1, 1), (0, 0)))
    return np.concatenate([p[:-2], p[1:-1], p[2:]], axis=1)


def encoder_weights(W):
    """The encoder's weights as the library stores them (float32): row-layout
    MatMul weights [in][out], q's weight and bias scaled by 1 / sqrt(96),
    the kernel-3 convs unrolled."""
    f32 = lambda a: np.asarray(a, np.float32)                   # noqa: E731
    sq = 1.0 / math.sqrt(ENC_HD)
    EW = {"emb": f32(W["emb"]), "wp": f32(W["enc_p.proj.weight"][:, :, 0].T), "bp": f32(W["enc_p.proj.bias"])}
    for i in range(ENC_LAYERS):
        p, f = f"enc_p.encoder.attn_layers.{i}", f"enc_p.encoder.ffn_layers.{i}"
        EW[f"l{i}.wq"] = f32(W[f"{p}.conv_q.weight"][:, :, 0].T * sq)
        EW[f"l{i}.bq"] = f32(W[f"{p}.conv_q.bias"] * sq)
        for t in ("k", "v", "o"):
            EW[f"l{i}.w{t}"] = f32(W[f"{p}.conv_{t}.weight"][:, :, 0].T)
            EW[f"l{i}.b{t}"] = f32(W[f"{p}.conv_{t}.bias"])
        EW[f"l{i}.ek"] = f32(W[f"{p}.emb_rel_k"][0])            # [9][96], offsets -4 .. 4
        EW[f"l{i}.ev"] = f32(W[f"{p}.emb_rel_v"][0])
        EW[f"l{i}.w1"], EW[f"l{i}.b1"] = f32(_k3(W[f"{f}.conv_1.weight"])), f32(W[f"{f}.conv_1.bias"])
        EW[f"l{i}.w2"], EW[f"l{i}.b2"] = f32(_k3(W[f"{f}.conv_2.weight"])), f32(W[f"{f}.conv_2.bias"])
        for j in (1, 2):
            EW[f"l{i}.g{j}"] = f32(W[f"enc_p.encoder.norm_layers_{j}.{i}.gamma"])
            EW[f"l{i}.be{j}"] = f32(W[f"enc_p.encoder.norm_layers_{j}.{i}.beta"])
    return {k: v.astype(np.float64) for k, v in EW.items()}


def _enc_ln(s, g, b):
    """LayerNorm over the channels of every row, sequential sums (the C loop)."""
    mean = _seqsum(s) / s.shape[1]
    d = s - mean[:, None]
    var = _seqsum(d * d) / s.shape[1]
    return d / np.sqrt(var + ENC_EPS)[:, None] * g + b


def _enc_rel(qh, ek):
    """The relative-key terms: rel[i][d + 4] = sum_c q[i][c] * ek[d + 4][c], sequential."""
    return _seqsum(qh[:, None, :] * ek[None, :, :])             # [n][9]


def encoder_float(EW, ids, record=None):
    """The encoder in float64 with the library's structure (row layout,
    unrolled convs, biases added where the host ops add them): the
    calibration run (``record``: tensor -> max |x|) and the float yardstick.
    Returns x [n][192] and stats [n][384]."""
    rec = record if record is not None else {}

    def r(name, v):
        rec[name] = max(rec.get(name, 0.0), float(np.abs(v).max()) if v.size else 0.0)
        return v

    def inp(name, v):                     # a MatMul input: kept for the exponent search
        rec.setdefault("__in", {}).setdefault(name, []).append(v)
        return r(name, v)
    n = len(ids)
    x = r("enc.x0", EW["emb"][np.asarray(ids)] * math.sqrt(ENC_D))
    for i in range(ENC_LAYERS):
        e = f"enc.l{i}"
        inp(f"{e}.xin", x)
        q0 = r(f"{e}.q0", x @ EW[f"l{i}.wq"])
        k0 = r(f"{e}.k0", x @ EW[f"l{i}.wk"])
        v0 = r(f"{e}.v0", x @ EW[f"l{i}.wv"])
        q, k, v = r(f"{e}.q", q0 + EW[f"l{i}.bq"]), r(f"{e}.k", k0 + EW[f"l{i}.bk"]), r(f"{e}.v", v0 + EW[f"l{i}.bv"])
        heads = []
        for h in range(ENC_H):
            sl = slice(h * ENC_HD, (h + 1) * ENC_HD)
            s = r(f"{e}.s", q[:, sl] @ k[:, sl].T)
            rel = _enc_rel(q[:, sl], EW[f"l{i}.ek"])
            val = s.copy()
            for d in range(-ENC_WIN, ENC_WIN + 1):
                idx = np.arange(max(0, -d), min(n, n - d))
                val[idx, idx + d] += rel[idx, d + ENC_WIN]
            ex = np.exp(val - val.max(1, keepdims=True))
            p = ex / ex.sum(1, keepdims=True)
            o = r(f"{e}.o", p @ v[:, sl])
            for d in range(-ENC_WIN, ENC_WIN + 1):
                idx = np.arange(max(0, -d), min(n, n - d))
                o[idx] += p[idx, idx + d][:, None] * EW[f"l{i}.ev"][d + ENC_WIN][None, :]
            heads.append(o)
        att = inp(f"{e}.att", np.concatenate(heads, axis=1))
        y = r(f"{e}.y", att @ EW[f"l{i}.wo"])
        x1 = r(f"{e}.x1", _enc_ln(x + (y + EW[f"l{i}.bo"]), EW[f"l{i}.g1"], EW[f"l{i}.be1"]))
        f1 = r(f"{e}.f1", inp(f"{e}.c1", _im2col3(x1)) @ EW[f"l{i}.w1"])
        hh = np.maximum(f1 + EW[f"l{i}.b1"], 0.0)
        f2 = r(f"{e}.f2", inp(f"{e}.h", _im2col3(hh)) @ EW[f"l{i}.w2"])
        x = r(f"{e}.x2", _enc_ln(x1 + (f2 + EW[f"l{i}.b2"]), EW[f"l{i}.g2"], EW[f"l{i}.be2"]))
    st = r("enc.st", inp("enc.st#in", x) @ EW["wp"])
    return x, st + EW["bp"]


def _nat(v, headroom=1):
    """The finest exponent that leaves ``headroom`` bits over max |x| = v."""
    return 24 if v <= 0 else int(math.floor(math.log2(2.0 ** (15 - headroom) / v)))


def _search_input(E, rec, key, consumers, span=12):
    """The exponent of MatMul input ``key`` that minimises its consumers'
    relative output error on the calibration inputs (weights at
    f_y + 8 - f_x must fit int16); ``consumers``: [(weight [in][out], output key)]."""
    xs = rec["__in"][key]
    nat = _nat(rec[key])
    best, best_err = None, None
    for fx in range(nat - span, nat + 1):
        if any(float(np.abs(w).max()) * 2.0 ** (E[y] + 8 - fx) > 32767 for w, y in consumers):
            continue
        err = 0.0
        for w, y in consumers:
            wr = _raw(w, E[y] + 8 - fx)
            for x in xs:
                ref = x @ w
                got = _kmm(_raw(x, fx), wr) * 2.0 ** -E[y]
                err += float(np.mean((got - ref) ** 2) / max(np.mean(ref ** 2), 1e-30))
        if best_err is None or err < best_err:
            best, best_err = fx, err
    if best is None:                      # no input exponent fits every weight: coarser outputs
        for w, y in consumers:
            while float(np.abs(w).max()) * 2.0 ** (E[y] + 8 - nat) > 32767:
                E[y] -= 1
        best = nat
    E[key] = best


def encoder_exponents(EW, rec, p_exp=ENC_P, search=True):
    """Exponents of the int16 encoder from a calibration record (encoder_float):
    every tensor at its natural exponent (one bit of headroom); k at
    f_s + 8 - f_q and v at f_o + 8 - f_p must fit int16 (else f_s / f_o
    coarser); every MatMul input at the searched exponent (``search``, else
    the natural one with coarser outputs where a weight would not fit)."""
    E = {"enc.p": int(p_exp)}
    E["enc.x0"] = _nat(rec["enc.x0"])
    for k in rec:
        if k != "__in" and k not in ("enc.x0",):
            E[k] = _nat(rec[k])
    for i in range(ENC_LAYERS):
        e = f"enc.l{i}"
        fq = E[f"{e}.q"]
        fs = E[f"{e}.s"]
        while rec[f"{e}.k"] * 2.0 ** (fs + 8 - fq) > 32767:
            fs -= 1
        E[f"{e}.s"], E[f"{e}.k"] = fs, fs + 8 - fq
        fo = E[f"{e}.o"]
        while rec[f"{e}.v"] * 2.0 ** (fo + 8 - p_exp) > 32767:
            fo -= 1
        E[f"{e}.o"], E[f"{e}.v"] = fo, fo + 8 - p_exp
    groups = [("enc.st#in", [(EW["wp"], "enc.st")])]
    for i in range(ENC_LAYERS):
        e = f"enc.l{i}"
        groups += [(f"{e}.xin", [(EW[f"l{i}.w{t}"], f"{e}.{t}0") for t in "qkv"]),
                   (f"{e}.att", [(EW[f"l{i}.wo"], f"{e}.y")]),
                   (f"{e}.c1", [(EW[f"l{i}.w1"], f"{e}.f1")]),
                   (f"{e}.h", [(EW[f"l{i}.w2"], f"{e}.f2")])]
    for key, consumers in groups:
        _search_input(E, rec, key, consumers, span=12 if search else 0)
    return E


def encoder_forward(EW, E, ids, trace=None):
    """The library's encoder, bit for bit: ids -> x [n][192], stats [n][384]
    (float32, as the library returns them)."""
    tr = trace if trace is not None else {}
    n = len(ids)
    L2 = lambda f: 2.0 ** -f                                     # noqa: E731
    fx = E["enc.x0"]
    xr = tr["enc.x0"] = _raw(EW["emb"][np.asarray(ids)] * math.sqrt(ENC_D), fx)
    fp = E["enc.p"]
    for i in range(ENC_LAYERS):
        e = f"enc.l{i}"
        # q / k / v projections (kernel), then the prep: + bias at the per-head exponents
        pr = {}
        fxi = E[f"{e}.xin"]
        xin = tr[f"{e}.xin"] = _raw(xr * L2(fx), fxi)                    # the q / k / v input copy
        for t, ft in (("q", E[f"{e}.q"]), ("k", E[f"{e}.k"]), ("v", E[f"{e}.v"])):
            fy = E[f"{e}.{t}0"]
            y0 = _kmm(xin, _raw(EW[f"l{i}.w{t}"], fy + 8 - fxi))
            tr[f"{e}.{t}0"] = y0
            pr[t] = tr[f"{e}.{t}"] = _raw(y0 * L2(fy) + EW[f"l{i}.b{t}"], ft)
        fq, fs, fo = E[f"{e}.q"], E[f"{e}.s"], E[f"{e}.o"]
        heads = []
        for h in range(ENC_H):
            sl = slice(h * ENC_HD, (h + 1) * ENC_HD)
            s_raw = _kmm(pr["q"][:, sl], pr["k"][:, sl].T)           # [queries][keys]
            rel = _enc_rel(pr["q"][:, sl] * L2(fq), EW[f"l{i}.ek"])
            val = s_raw * L2(fs)
            for d in range(-ENC_WIN, ENC_WIN + 1):
                idx = np.arange(max(0, -d), min(n, n - d))
                val[idx, idx + d] = val[idx, idx + d] + rel[idx, d + ENC_WIN]
            ex = _libm(math.exp, val - val.max(1, keepdims=True))
            p_raw = np.clip(np.round(ex / _seqsum(ex)[:, None] * 2.0 ** fp), 0, 32767)
            o_raw = _kmm(p_raw, pr["v"][:, sl])
            o = o_raw * L2(fo)
            for d in range(-ENC_WIN, ENC_WIN + 1):
                idx = np.arange(max(0, -d), min(n, n - d))
                o[idx] = o[idx] + (p_raw[idx, idx + d] * L2(fp))[:, None] * EW[f"l{i}.ev"][d + ENC_WIN][None, :]
            tr[f"{e}.s{h}"], tr[f"{e}.p{h}"], tr[f"{e}.o{h}"] = s_raw, p_raw, o_raw
            heads.append(o)
        fa = E[f"{e}.att"]
        att = tr[f"{e}.att"] = _raw(np.concatenate(heads, axis=1), fa)
        fy = E[f"{e}.y"]
        y = tr[f"{e}.y"] = _kmm(att, _raw(EW[f"l{i}.wo"], fy + 8 - fa))
        f1x = E[f"{e}.x1"]
        x1 = tr[f"{e}.x1"] = _raw(_enc_ln(xr * L2(fx) + (y * L2(fy) + EW[f"l{i}.bo"]),
                                          EW[f"l{i}.g1"], EW[f"l{i}.be1"]), f1x)
        ff1, fc1 = E[f"{e}.f1"], E[f"{e}.c1"]
        c1 = tr[f"{e}.c1"] = _im2col3(_raw(x1 * L2(f1x), fc1))
        f1 = tr[f"{e}.f1"] = _kmm(c1, _raw(EW[f"l{i}.w1"], ff1 + 8 - fc1))
        fh = E[f"{e}.h"]
        hh = tr[f"{e}.h"] = _raw(np.maximum(f1 * L2(ff1) + EW[f"l{i}.b1"], 0.0), fh)
        ff2 = E[f"{e}.f2"]
        f2 = tr[f"{e}.f2"] = _kmm(_im2col3(hh), _raw(EW[f"l{i}.w2"], ff2 + 8 - fh))
        fx = E[f"{e}.x2"]
        xr = tr[f"{e}.x2"] = _raw(_enc_ln(x1 * L2(f1x) + (f2 * L2(ff2) + EW[f"l{i}.b2"]),
                                          EW[f"l{i}.g2"], EW[f"l{i}.be2"]), fx)
    fst, fsi = E["enc.st"], E["enc.st#in"]
    st = tr["enc.st"] = _kmm(_raw(xr * L2(fx), fsi), _raw(EW["wp"], fst + 8 - fsi))
    return (xr * L2(fx)).astype(np.float32), (st * L2(fst) + EW["bp"]).astype(np.float32)


def library_encoder(EW, E):
    """ids -> (x, m_p, logs_p) [192][n] float64 from encoder_forward: the
    front end's ``encoder`` for the library's int16 encoder."""
    def enc(ids):
        x, st = encoder_forward(EW, E, ids)
        x, st = x.astype(np.float64).T, st.astype(np.float64).T
        return x, st[:ENC_D], st[ENC_D:]
    return enc


# ---- the library's duration predictor (C, float64): the specification (TTS_PLAN §7) ---- #
#
# The stochastic duration predictor in float64 as demo/tts/src/tts_dp.c
# computes it: duration_predictor(fast=True) with every sum left to right
# (1x1 convs, LayerNorm, the spline's softmax) and exp / log1p from libm;
# LayerNorm multiplies by 1 / sd and GELU's erf argument is x * (1 / sqrt 2)
# (a multiply instead of a divide per element: the A53 divides slowly).
# Inputs: x [192][n] (the encoder's output, float32 values) and z [2][n]
# (the noise, already * noise_w); output logw [n].

DP_FLOWS = (7, 5, 3)


def dp_tensors():
    """(name, shape) of the duration predictor's weights in dp.dat's order
    (float32, the shapes as tts_dp.c reads them)."""
    C = 192

    def dds(p):
        r = []
        for i in range(3):
            r += [(f"{p}.convs_sep.{i}.weight", (C, 3)), (f"{p}.convs_sep.{i}.bias", (C,)),
                  (f"{p}.norms_1.{i}.gamma", (C,)), (f"{p}.norms_1.{i}.beta", (C,)),
                  (f"{p}.convs_1x1.{i}.weight", (C, C)), (f"{p}.convs_1x1.{i}.bias", (C,)),
                  (f"{p}.norms_2.{i}.gamma", (C,)), (f"{p}.norms_2.{i}.beta", (C,))]
        return r
    out = [("dp.pre.weight", (C, C)), ("dp.pre.bias", (C,))] + dds("dp.convs") + \
        [("dp.proj.weight", (C, C)), ("dp.proj.bias", (C,))]
    for fi in DP_FLOWS:
        p = f"dp.flows.{fi}"
        out += [(f"{p}.pre.weight", (C,)), (f"{p}.pre.bias", (C,))] + dds(f"{p}.convs") + \
            [(f"{p}.proj.weight", (29, C)), (f"{p}.proj.bias", (29,))]
    return out + [("dp.flows.0.m", (2,)), ("dp.flows.0.exp_neg_logs", (2,))]


def dp_flat(W):
    """dp.dat: the weights of dp_tensors(), float32, in that order."""
    parts = []
    for name, shape in dp_tensors():
        a = np.asarray(W[name], np.float32)
        if a.size != int(np.prod(shape)):
            raise ValueError(f"{name}: {a.shape} is not {shape}")
        parts.append(a.reshape(-1))
    return np.concatenate(parts)


def _dp_w(W):
    """The weights as tts_dp.c holds them: float32 values in float64, its shapes."""
    return {n: np.asarray(W[n], np.float32).astype(np.float64).reshape(s) for n, s in dp_tensors()}


def _conv1x1_seq(x, w, b):
    """y[o][t] = (sum_c w[o][c] * x[c][t], left to right) + b[o]."""
    return _seqsum(w[:, :, None] * x[None, :, :], axis=1) + b[:, None]


def _dwconv_seq(x, w, b, dil):
    """Depthwise kernel 3 'same' (zero padding): ((0 + w0 x[t-d]) + w1 x[t]) + w2 x[t+d] + b."""
    xp = np.pad(x, ((0, 0), (dil, dil)))
    n = x.shape[1]
    y = np.zeros_like(x)
    for k in range(3):
        y = y + w[:, k][:, None] * xp[:, k * dil: k * dil + n]
    return y + b[:, None]


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _ln_seq(x, g, b):
    """LayerNorm over the channels of every column, sums left to right, x 1 / sd."""
    C = x.shape[0]
    mean = _seqsum(x, axis=0) / C
    d = x - mean[None, :]
    var = _seqsum(d * d, axis=0) / C
    inv = 1.0 / np.sqrt(var + 1e-5)
    return d * inv[None, :] * g[:, None] + b[:, None]


def _gelu_seq(x):
    """_gelu_fast's operations with erf's argument x * (1 / sqrt 2), exp from libm."""
    u = x * _INV_SQRT2
    a = np.abs(u)
    t = 1.0 / (a * 0.3275911 + 1.0)
    p = ((((t * 1.061405429 - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    p = 1.0 - p * _libm(math.exp, (-a) * a)
    return (x * 0.5) * (p * np.sign(u) + 1.0)


def _dds_seq(D, p, x, g=None):
    if g is not None:
        x = x + g
    for i in range(3):
        y = _dwconv_seq(x, D[f"{p}.convs_sep.{i}.weight"], D[f"{p}.convs_sep.{i}.bias"], 3 ** i)
        y = _gelu_seq(_ln_seq(y, D[f"{p}.norms_1.{i}.gamma"], D[f"{p}.norms_1.{i}.beta"]))
        y = _conv1x1_seq(y, D[f"{p}.convs_1x1.{i}.weight"], D[f"{p}.convs_1x1.{i}.bias"])
        y = _gelu_seq(_ln_seq(y, D[f"{p}.norms_2.{i}.gamma"], D[f"{p}.norms_2.{i}.beta"]))
        x = x + y
    return x


def _spline_seq(x, uw, uh, ud, tb=5.0, mbw=1e-3, mbh=1e-3, md=1e-3):
    """rq_spline_inverse, sums left to right, exp / log1p from libm."""
    nb = uw.shape[-1]
    inside = (x >= -tb) & (x <= tb)
    ud = np.pad(ud, ((0, 0), (1, 1)))
    const = math.log(math.exp(1 - md) - 1)
    ud[:, 0] = const
    ud[:, -1] = const

    def smax(u):
        e = _libm(math.exp, u - u.max(-1, keepdims=True))
        return e / _seqsum(e)[:, None]
    widths = mbw + (1 - mbw * nb) * smax(uw)
    cw = np.pad(np.cumsum(widths, -1), ((0, 0), (1, 0)))
    cw = 2 * tb * cw - tb
    cw[:, 0], cw[:, -1] = -tb, tb
    widths = cw[:, 1:] - cw[:, :-1]
    der = md + (_libm(math.log1p, _libm(math.exp, -np.abs(ud))) + np.maximum(ud, 0))
    heights = mbh + (1 - mbh * nb) * smax(uh)
    ch = np.pad(np.cumsum(heights, -1), ((0, 0), (1, 0)))
    ch = 2 * tb * ch - tb
    ch[:, 0], ch[:, -1] = -tb, tb
    heights = ch[:, 1:] - ch[:, :-1]
    loc = ch.copy()
    loc[:, -1] += 1e-6
    bi = np.clip((x[:, None] >= loc).sum(-1) - 1, 0, nb - 1)
    r = np.arange(len(x))
    icw, ibw, ich, ih = cw[r, bi], widths[r, bi], ch[r, bi], heights[r, bi]
    delta = heights / widths
    idl, idr, ide = der[r, bi], der[r, bi + 1], delta[r, bi]
    a = (x - ich) * (idl + idr - 2 * ide) + ih * (ide - idl)
    b = ih * idl - (x - ich) * (idl + idr - 2 * ide)
    c = -ide * (x - ich)
    root = (2 * c) / (-b - np.sqrt(b * b - 4 * a * c))
    return np.where(inside, root * ibw + icw, x)


def duration_predictor_seq(W, x, z):
    """logw [n] of the library's duration predictor: x [192][n] (float32
    values), z [2][n] (the noise * noise_w)."""
    D = _dp_w(W)
    x = np.asarray(x, np.float32).astype(np.float64)
    z0, z1 = (np.asarray(v, np.float64).copy() for v in z)
    h = _conv1x1_seq(x, D["dp.pre.weight"], D["dp.pre.bias"])
    h = _dds_seq(D, "dp.convs", h)
    g = _conv1x1_seq(h, D["dp.proj.weight"], D["dp.proj.bias"])
    for fi in DP_FLOWS:
        p = f"dp.flows.{fi}"
        z0, z1 = z1, z0                                          # Flip
        hf = D[f"{p}.pre.weight"][:, None] * z0[None, :] + D[f"{p}.pre.bias"][:, None]
        hf = _dds_seq(D, f"{p}.convs", hf, g=g)
        hh = _conv1x1_seq(hf, D[f"{p}.proj.weight"], D[f"{p}.proj.bias"]).T  # [n][29]
        z1 = _spline_seq(z1, hh[:, :10] / math.sqrt(192), hh[:, 10:20] / math.sqrt(192), hh[:, 20:])
    z0, z1 = z1, z0                                              # the last Flip
    return (z0 - D["dp.flows.0.m"][0]) * D["dp.flows.0.exp_neg_logs"][0]
