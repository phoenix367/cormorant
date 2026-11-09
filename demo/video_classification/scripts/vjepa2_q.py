#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vjepa2_q.py — V-JEPA 2 ViT-L with its Something-Something v2 classifier
(facebook/vjepa2-vitl-fpc16-256-ssv2) in numpy: the float64 reference and the
bit-level emulation of the planned KV260 partition (model-study skill, route C;
doc/plans/VJEPA2_PLAN.md).  pol None = the float64 reference.

The model (transformers 5.17 VJEPA2ForVideoClassification):
  clip       16 RGB frames, resized (shortest edge 292, bilinear) and center-cropped
             to 256 x 256, x / 255, normalised per channel (ImageNet mean / std)
  patches    Conv3d 2 x 16 x 16, stride 2 x 16 x 16: 8 x 16 x 16 = 2048 tokens of 1024
  encoder    24 pre-LN layers (LayerNorm eps 1e-6): q / k / v / proj with biases, 16 heads
             of 64, 3-D RoPE on q and k, GELU (erf) MLP 4096; a final LayerNorm
  RoPE       per head, dims 0-19 rotate by the token's frame, 20-39 by its row, 40-59 by
             its column (token n: frame n // 256, row n // 16 mod 16, column n mod 16);
             interleaved pairs (2i, 2i + 1): y[2i] = x[2i] c - x[2i+1] s, y[2i+1] =
             x[2i+1] c' + x[2i] s', and element j of a 20-dim segment turns by
             pos * 10000^-((j mod 10) / 10) (HF / Meta's repeat(..., 2) expansion: a pair's
             two elements turn by different angles); dims 60-63 pass through
  pooler     3 self-attention layers over the 2048 tokens (the same block without RoPE),
             then a cross-attention layer: a learned query (1 x 1024) attends LN1(h)
             through q / k / v projections (no output projection), + the query, then
             LN2 and the MLP; the classifier 1024 -> 174 on that row

Emulated datapath (llm_study / vlm_study conventions — kernel MatMuls on raw int16,
exact sums in ap_fixed<32,16> (float64 holds them exactly), floor(acc / 2^8),
saturate; host regions in double, round half to even + saturate on write):
  pixels     the uint8 values of the resized and cropped clip (exact integers,
             exponent 0); the normalisation folded into the patch embedding:
             W / (255 std_c) and b - sum W mean_c / std_c (left to right), float32
  patch emb  kernel MatMul [2048, 1536] x [1536, 1024] (a patch as one row: channel,
             frame, row, column), + the folded bias into the residual
  layers     vlm_study.VisionModel's primitives and policies (SmolVLM's / SmolVLA's
             vision encoder: LayerNorm, q / k / v with their biases, scores, softmax,
             P.V, GELU on VectorOPKernel's activation unit, the residual stream)
  RoPE       host (one region): rope(q0 + b) written once at f_q / f_k per head; or,
             policy irope, the row-ops unit: its COPY with the bias at f_qb / f_kb
             (iqkv, rowops_spec.bias_round), then its ROPE (rowops_spec.rope) on the
             head dims permuted into rotate-half order (PERM), cos / sin per token as
             int24 tables of the double values, at f_q / f_k
  GELU       erf form: host table per exponent (libm erf), or VectorOPKernel's ADD +
             MUL + act GELU (vlm_study.vop_gelu_b's operands, the erf table at 2^-8)
  pooler     the same layers; the cross layer's q = q_proj(query) + b is a constant
             (float64, written at f_q per head); its residual = the query + P.V
  classifier the residual row written at f_xc, a kernel MatMul, + the bias (host)
Layer indices of the tensor classes: encoder 0 .. 23, its final LayerNorm 24, the
pooler's self-attention layers 25 .. 27, the cross layer 28, the classifier 29.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "chat", "scripts")))
import llm_study as ls   # noqa: E402
import vlm_study as vs   # noqa: E402

p2, p2v = ls.p2, vs.p2v
I16_LO, I16_HI = ls.I16_LO, ls.I16_HI

ENC = "vjepa2.encoder."
POOL = "pooler."
# key: name under the layer's prefix (the classes come from vlm_study.VLIN)
ENC_LIN = {"q": "attention.query", "k": "attention.key", "v": "attention.value", "o": "attention.proj",
           "f1": "mlp.fc1", "f2": "mlp.fc2"}
POOL_LIN = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.out_proj",
            "f1": "mlp.fc1", "f2": "mlp.fc2"}
CROSS_LIN = {"k": "cross_attn.k_proj", "v": "cross_attn.v_proj", "f1": "mlp.fc1", "f2": "mlp.fc2"}


# ------------------------------------------------------------------ configuration
class VCfg:
    """config.json and video_preprocessor_config.json of the checkpoint."""

    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        pp = json.load(open(os.path.join(path, "video_preprocessor_config.json")))
        assert c["hidden_act"] == "gelu" and c["qkv_bias"] and not c["use_SiLU"] and c["in_chans"] == 3, c
        self.D, self.L, self.H = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
        self.KV, self.HD = self.H, self.D // self.H
        self.F = int(self.D * c["mlp_ratio"])
        self.eps = float(c["layer_norm_eps"])
        self.P, self.tub, self.frames = c["patch_size"], c["tubelet_size"], c["frames_per_clip"]
        self.S = c["crop_size"]
        self.grid = self.S // self.P
        self.T = self.frames // self.tub
        self.N = self.T * self.grid * self.grid
        self.rot = 2 * ((self.HD // 3) // 2)          # 20: the frame / row / column segments
        self.n_pool = c["num_pooler_layers"]
        self.labels = [c["id2label"][str(i)] for i in range(len(c["id2label"]))]
        self.C = len(self.labels)
        self.short = pp["size"]["shortest_edge"]
        assert pp["crop_size"] == {"height": self.S, "width": self.S} and pp["do_center_crop"], pp
        assert pp["resample"] == 2 and pp["do_rescale"] and pp["do_normalize"], pp
        self.mean = np.array(pp["image_mean"], np.float64)
        self.std = np.array(pp["image_std"], np.float64)
        self.lf = self.L                              # the encoder's final LayerNorm
        self.lp = [self.L + 1 + i for i in range(self.n_pool)]
        self.lc = self.L + 1 + self.n_pool            # the cross-attention layer
        self.lk = self.lc + 1                         # the classifier
        self.NL = self.lk

    def attn_layers(self):
        return list(range(self.L)) + self.lp + [self.lc]

    def prefix(self, l):
        if l < self.L:
            return f"{ENC}layer.{l}.", ("norm1", "norm2"), ENC_LIN
        if l in self.lp:
            return f"{POOL}self_attention_layers.{l - self.L - 1}.", ("layer_norm1", "layer_norm2"), POOL_LIN
        assert l == self.lc
        return f"{POOL}cross_attention_layer.", ("layer_norm1", "layer_norm2"), CROSS_LIN


# ------------------------------------------------------------------ the clip
def sample_frames(n, fpc=16, step=4):
    """The frames of one clip — Meta's evaluation sampler (src/datasets/video_dataset.py
    loadvideo_decord) with one segment and clip overlap allowed: 16 frames over the
    first 64 at step ~4 of a longer video, else every 4th frame of the video padded
    with its last frame."""
    clip_len = fpc * step
    if n > clip_len:
        idx = np.linspace(0, clip_len, num=fpc)
        return np.clip(idx, 0, clip_len - 1).astype(np.int64)
    sample_len = min(clip_len, n) - 1
    idx = np.linspace(0, sample_len, num=sample_len // step)
    idx = np.concatenate((idx, np.ones(fpc - sample_len // step) * sample_len))
    return np.clip(idx, 0, sample_len - 1).astype(np.int64)


def resample_taps(n_in, n_out):
    """PIL's bilinear resample (the triangle filter, its support widened by the downscale
    factor: antialiased) as taps: (lo [n_out], W [n_out][taps]) — output i is
    sum_j W[i][j] x[lo[i] + j], the weights normalised by their sum taken left to right,
    zero past the filter.  Every operation is the C library's (vjepa2_api.c), in order."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = fs
    lo, ws = np.zeros(n_out, np.int64), []
    for i in range(n_out):
        c = (i + 0.5) * scale
        a = max(int(c - support + 0.5), 0)
        b = min(int(c + support + 0.5), n_in)
        w = np.maximum(0.0, 1.0 - np.abs((np.arange(a, b) - c + 0.5) / fs))
        tot = 0.0
        for v in w:
            tot += float(v)
        lo[i] = a
        ws.append(w / tot)
    W = np.zeros((n_out, max(len(w) for w in ws)))
    for i, w in enumerate(ws):
        W[i, :len(w)] = w
    return lo, W


def _resample(x, n_out, axis):
    """One pass along ``axis`` of float64 x: the taps summed left to right (the C
    library's order — a BLAS product's is not portable), floor(v + 0.5), clipped to [0, 255]."""
    lo, W = resample_taps(x.shape[axis], n_out)
    xm = np.moveaxis(x, axis, 0)
    idx = np.minimum(lo[:, None] + np.arange(W.shape[1])[None, :], xm.shape[0] - 1)   # zero weights past the end
    shp = (n_out,) + (1,) * (xm.ndim - 1)
    acc = W[:, 0].reshape(shp) * xm[idx[:, 0]]
    for j in range(1, W.shape[1]):
        acc = acc + W[:, j].reshape(shp) * xm[idx[:, j]]
    return np.moveaxis(np.clip(np.floor(acc + 0.5), 0, 255), 0, axis)


def resize_short(img, short):
    """uint8 [h, w, 3] -> uint8 with its shorter side `short` (the longer one
    int(short * long / short_side), transformers' get_resize_output_image_size):
    bilinear, horizontal pass then vertical, each rounded to uint8."""
    h, w = img.shape[:2]
    if h <= w:
        oh, ow = short, int(short * w / h)
    else:
        oh, ow = int(short * h / w), short
    x = img.astype(np.float64)
    if ow != w:
        x = _resample(x, ow, 1)                                     # [h, ow, 3]
    if oh != h:
        x = _resample(x, oh, 0)
    return x.astype(np.uint8)


def center_crop(img, size):
    """The centre size x size, its offsets rounded down (the video processor's)."""
    h, w = img.shape[:2]
    t, l = (h - size) // 2, (w - size) // 2
    return img[t:t + size, l:l + size]


def make_clip(frames, cfg):
    """uint8 frames [n, h, w, 3] of a video -> the clip uint8 [16, 256, 256, 3]."""
    idx = sample_frames(len(frames), cfg.frames)
    return np.stack([center_crop(resize_short(frames[i], cfg.short), cfg.S) for i in idx]), idx


def patches(clip, cfg):
    """uint8 [16, S, S, 3] -> [2048, 1536]: token (frame pair, row, column), a patch as
    (channel, frame, row, column) — Conv3d's weight order."""
    T, g, P, tb = cfg.T, cfg.grid, cfg.P, cfg.tub
    x = clip.reshape(T, tb, g, P, g, P, 3).transpose(0, 2, 4, 6, 1, 3, 5)
    return x.reshape(T * g * g, 3 * tb * P * P)


# ------------------------------------------------------------------ weights
def patch_weight(W, cfg, exact=False):
    """The patch embedding with the pixel normalisation folded in: [1536, 1024]
    W / (255 std_c) and the bias b - sum W mean_c / std_c (the sum left to right),
    both float32 (as the frontend's ONNX initializers hold them); exact: unrounded
    doubles (the float64 reference's: the same function of the pixels as normalising
    them first)."""
    Wp = W[ENC + "embeddings.patch_embeddings.proj.weight"].astype(np.float64).reshape(cfg.D, 3, -1)
    Wf = np.ascontiguousarray((Wp / (255.0 * cfg.std)[None, :, None]).reshape(cfg.D, -1).T)
    b = W[ENC + "embeddings.patch_embeddings.proj.bias"].astype(np.float64) \
        - np.cumsum((Wp * (cfg.mean / cfg.std)[None, :, None]).reshape(cfg.D, -1), axis=1)[:, -1]
    return (Wf, b) if exact else (Wf.astype(np.float32), b.astype(np.float32))


def kernel_weights(W, cfg):
    """Every kernel MatMul's B operand [K, N], float32: the layers' linears, the
    folded patch embedding (0, "pe") and the classifier (lk, "cls")."""
    KW = {}
    for l in cfg.attn_layers():
        pre, _, lin = cfg.prefix(l)
        for key, name in lin.items():
            KW[(l, key)] = np.ascontiguousarray(W[pre + name + ".weight"].T)
    KW[(0, "pe")] = patch_weight(W, cfg)[0]
    KW[(cfg.lk, "cls")] = np.ascontiguousarray(W["classifier.weight"].T)
    return KW


def cross_query(W, cfg):
    """The cross layer's q = query . Wq^T + bq in double [D]: each output a correctly
    rounded sum of the exact products (math.fsum — the same on every machine, unlike a
    BLAS dot product; src/vjepa2.py cross_query) plus the bias."""
    query = W[POOL + "query_tokens"].reshape(-1).astype(np.float64)
    cq = POOL + "cross_attention_layer.cross_attn.q_proj."
    Wq, bq = W[cq + "weight"].astype(np.float64), W[cq + "bias"].astype(np.float64)
    return np.array([math.fsum(query * Wq[j]) for j in range(Wq.shape[0])]) + bq


# ------------------------------------------------------------------ RoPE
def rope_perm(HD):
    """The head dims in rotate-half order: pair (2j, 2j + 1) at (j, j + HD / 2)."""
    return np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])


def rope_tables(cfg):
    """cos / sin [N, HD] of every token in double (rotate_queries_or_keys: omega =
    1 / 10000^(k / 10), freq = pos * omega, the 10 values repeated twice over a
    20-dim segment); dims past 3 x 20: cos 1, sin 0."""
    n = np.arange(cfg.N)
    g2 = cfg.grid * cfg.grid
    pos = (n // g2, (n % g2) // cfg.grid, n % cfg.grid)
    half = cfg.rot // 2
    omega = 1.0 / np.power(10000.0, np.arange(half) / half)
    C = np.ones((cfg.N, cfg.HD))
    S = np.zeros((cfg.N, cfg.HD))
    for seg in range(3):
        freq = pos[seg].astype(np.float64)[:, None] * omega[None, :]
        sl = slice(seg * cfg.rot, (seg + 1) * cfg.rot)
        C[:, sl] = np.tile(np.cos(freq), 2)
        S[:, sl] = np.tile(np.sin(freq), 2)
    return C, S


def rope_apply(x, C, S, H):
    """x [N, H * HD] -> RoPE'd: y = x * cos + rot(x) * sin, rot(x)[2i] = -x[2i+1],
    rot(x)[2i+1] = x[2i] (in x's dtype)."""
    N = x.shape[0]
    xs = x.reshape(N, H, -1)
    rot = np.empty_like(xs)
    rot[..., 0::2] = -xs[..., 1::2]
    rot[..., 1::2] = xs[..., 0::2]
    return (xs * C[:, None, :] + rot * S[:, None, :]).reshape(N, -1)


# ------------------------------------------------------------------ GELU (erf)
def gelu_ref(x):
    """torch's exact GELU (nn.functional.gelu, approximate none) in x's dtype —
    the float reference's (transformers' ACT2FN["gelu"])."""
    try:
        import torch
        return torch.nn.functional.gelu(torch.from_numpy(np.ascontiguousarray(x))).numpy()
    except ImportError:
        e = np.vectorize(math.erf, otypes=[np.float64])(np.asarray(x, np.float64) / math.sqrt(2.0))
        return (x * (0.5 * (1.0 + e))).astype(x.dtype)


_ERF = np.vectorize(math.erf, otypes=[np.float64])
_GELU_ERF = {}


def gelu_erf_table(f):
    """GELU (erf) of every raw int16 at 2^-f (index raw + 32768) through libm, in the
    operation order of vectorop_act.act_values (the activation unit's specification);
    f = 8: the unit's input grid."""
    if f not in _GELU_ERF:
        x = (np.arange(65536) - 32768) / 2.0 ** int(f)
        _GELU_ERF[f] = x * (0.5 * (1.0 + _ERF(x / math.sqrt(2.0))))
    return _GELU_ERF[f]


# ------------------------------------------------------------------ the model
class VJEPA2(vs.VisionModel):
    """Encoder, attentive pooler and classifier.  pol None: the float64 reference
    (every tensor, weight, table and the residual in double); otherwise the
    emulation (exact integer sums in float64) or the bf16 yardstick."""

    def __init__(self, cfg, W, pol=None, fmt=None, record_ch=False, KW=None):
        self.cfg = cfg
        self.pol = pol or {}
        self.q = pol is not None
        self.fa_diag = bool(self.pol.get("float_acts"))
        self.qclasses = self.pol.get("qclasses")
        self.bf = bool(self.pol.get("bf16"))
        self.floor_fix = False
        self.hattn = False
        self.xattn_mode = False
        self.phase = "prefill"
        if self.bf:
            self.q = False
        self.fmt = fmt or {}
        if self.pol.get("res_bits") and ("res", cfg.NL) in self.fmt:
            self.pol = dict(self.pol, res_bits=int(self.fmt[("res", cfg.NL)]))   # calibrated
        self.ref = not self.q and not self.bf
        self.record_ch = record_ch
        self.grp = np.arange(cfg.H)
        self.stats = ls.Stats(cfg.NL)
        self.wsat = 0
        g = lambda n: W[n].astype(np.float64)
        self.ln1, self.ln2, self.b = {}, {}, {}
        for l in cfg.attn_layers():
            pre, (n1, n2), lin = cfg.prefix(l)
            self.ln1[l] = (g(pre + n1 + ".weight"), g(pre + n1 + ".bias"))
            self.ln2[l] = (g(pre + n2 + ".weight"), g(pre + n2 + ".bias"))
            for key, name in lin.items():
                self.b[(l, key)] = g(pre + name + ".bias")
        self.lnf = (g(ENC + "layernorm.weight"), g(ENC + "layernorm.bias"))
        self.b[(cfg.lk, "cls")] = g("classifier.bias")
        self.query = W[POOL + "query_tokens"].reshape(1, cfg.D).astype(np.float64)
        # the cross layer's query projection: a constant of the model (the frontend's)
        self.qc = cross_query(W, cfg)[None, :]
        self.b0 = patch_weight(W, cfg, exact=not self.q)[1].astype(np.float64)
        KW = kernel_weights(W, cfg) if KW is None else KW
        self.w = {k: self._enc(Wk, k[0], k[1], *self._io(k[1])) for k, Wk in KW.items()}
        if not self.q:                                  # the unrounded fold (see patch_weight)
            self.w[(0, "pe")] = self._enc(patch_weight(W, cfg, exact=True)[0], 0, "pe", "pix", "pe")
        self.C, self.S = rope_tables(cfg)
        self.perm = rope_perm(cfg.HD)
        if self.q:
            self.CR, self.SR = ls.rowops().rope_tables_raw(self.C[:, self.perm], self.S[:, self.perm])

    @staticmethod
    def _io(key):
        return {"pe": ("pix", "pe"), "cls": ("xc", "logits")}.get(key) or vs.VLIN[key][1:]

    def _enc(self, Wt, l, key, ci, co):
        if not self.q:
            Wt = np.ascontiguousarray(Wt, np.float64)
            return ls.bf16(Wt) if self.bf else Wt       # bf16: the weights too
        return super()._enc(Wt, l, key, ci, co)

    def resadd(self, h, d, l):
        if self.ref:                                    # the double residual
            y = h + d
            self.stats.add("h", l, float(np.abs(y).max()), int((np.abs(y) * 256 > 32767.5).sum()), y.size)
            if self.record_ch and y.size:
                self.stats.add_ch("h", 0, np.abs(y).max(0))
            return y
        return super().resadd(h, d, l)

    # ---- GELU (erf)
    def gelu(self, f, l):
        b = self.b[(l, "f1")]
        if not self.q or self.exact("f"):
            if self.record_ch:                          # f's exponent must hold f + b too
                self.stats.add_ch("f", l, np.abs(f + b).max(0))
            return self.host(gelu_ref(f + b), "a", l, self.E("a", l))
        ff = self.Ev("f", l, f.shape[1])
        if self.pol.get("vop_gelu") and vs.vop_gelu_ok(ff, self.Ev("a", l, f.shape[1]), b):
            # VectorOPKernel: ADD (bias + half an LSB), MUL (2^(8 - ff)), act GELU (erf)
            ba, sc = vs.vop_gelu_b(b, ff)
            s = np.clip(np.rint(f * p2v(ff)) + ba[None, :], I16_LO, I16_HI)
            qv = np.clip(np.floor(s * sc[None, :] / 256.0), I16_LO, I16_HI).astype(np.int64)
            return self.host(gelu_erf_table(8)[qv + 32768], "a", l, self.E("a", l))
        raw = np.clip(np.rint(f * p2v(ff)) + vs.bias_raw(b, ff)[None, :], I16_LO, I16_HI).astype(np.int64)
        y = np.empty(f.shape)
        for e in np.unique(ff):
            cols = ff == e
            y[:, cols] = gelu_erf_table(int(e))[raw[:, cols] + 32768]
        return self.host(y, "a", l, self.E("a", l))

    # ---- attention
    def qk_rope(self, x0, b, cls, l):
        """q or k of an encoder layer: bias, then RoPE (see the module docstring)."""
        cfg = self.cfg
        H = cfg.H
        if not self.q or self.exact(cls):
            xb = self.host(x0 + b, cls + "b", l, 0)     # the reference / bf16: records qb / kb
            return self.host(rope_apply(xb, self.C, self.S, H), cls, l, self.heads_e(cls, l))
        if self.pol.get("irope"):
            xb = self.qkv(x0, b, cls + "b", l, self.heads_e(cls + "b", l), cls + "0")
            fx = self.heads_e(cls + "b", l).reshape(H, cfg.HD)[:, self.perm]
            fo = self.Eh(cls, l, H)
            xr = np.rint(xb.reshape(-1, H, cfg.HD)[:, :, self.perm] * p2v(fx)[None]).astype(np.int64)
            y = ls.rowops().rope(xr, fx, self.CR, self.SR, fo)
            out = np.empty(y.shape)
            out[:, :, self.perm] = y / p2v(fo)[None, :, None]
            self.stats.add(cls, l, float(np.abs(out).max()), int(((y == I16_LO) | (y == I16_HI)).sum()), y.size)
            return out.reshape(x0.shape)
        y = rope_apply(x0 + b, self.C, self.S, H)
        return self.host(y, cls, l, self.heads_e(cls, l))

    def attend(self, q, k, v, l):
        """Per head: s = q.k^T (kernel), softmax (host or VectorOPKernel's unit), P.V."""
        cfg = self.cfg
        H, HD = cfg.H, cfg.HD
        mask = np.ones((q.shape[0], k.shape[0]), bool)
        out = np.empty((q.shape[0], cfg.D))
        if self.q:
            fq, fk, fs = self.Eh("q", l, H), self.Eh("k", l, H), self.Eh("s", l, H)
            fp, fv, fpv = self.Eh("p", l, H), self.Ev("vc", l, cfg.D), self.Ev("pv", l, cfg.D)
        for h in range(H):
            sl = slice(h * HD, (h + 1) * HD)
            if not self.q:
                s = self.fmm(q[:, sl], k[:, sl].T, "s", l, rec=False)
                if self.record_ch:
                    self.stats.add_ch("s", l, np.abs(s).max(), h, H)
                p = self.softmax(s, mask, 0, 0, l)
                out[:, sl] = self.fmm(p, v[:, sl], "pv", l, sl, cfg.D)
                continue
            s = self.kmm(q[:, sl] * p2(fq[h]), (k[:, sl] * p2(fk[h])).T, fs[h], 8, "s", l)
            p = self.softmax(s, mask, fs[h], fp[h], l)
            out[:, sl] = self.kmm(p * p2(fp[h]), v[:, sl] * p2v(fv[sl])[None, :], fpv[sl], 8, "pv", l)
        return out

    def self_attention(self, x, l, rope):
        q0, k0, v0 = self.lin(x, l, "q"), self.lin(x, l, "k"), self.lin(x, l, "v")
        if rope:
            q = self.qk_rope(q0, self.b[(l, "q")], "q", l)
            k = self.qk_rope(k0, self.b[(l, "k")], "k", l)
        else:
            q = self.qkv(q0, self.b[(l, "q")], "q", l, self.heads_e("q", l), "q0")
            k = self.qkv(k0, self.b[(l, "k")], "k", l, self.heads_e("k", l), "k0")
        v = self.qkv(v0, self.b[(l, "v")], "vc", l, self.Ev("vc", l, self.cfg.D), "v")
        return self.attend(q, k, v, l)

    def mlp(self, h, l):
        x2 = self.ln(h, self.ln2[l], "x2", l)
        a = self.gelu(self.lin(x2, l, "f1"), l)
        return self.resadd(h, self.lin(a, l, "f2") + self.ibias(self.b[(l, "f2")]), l)

    def block(self, h, l, rope):
        xa = self.ln(h, self.ln1[l], "x", l)
        att = self.self_attention(xa, l, rope)
        h = self.resadd(h, self.lin(att, l, "o") + self.ibias(self.b[(l, "o")]), l)
        return self.mlp(h, l)

    def cross(self, h):
        cfg, l = self.cfg, self.cfg.lc
        x = self.ln(h, self.ln1[l], "x", l)
        k = self.qkv(self.lin(x, l, "k"), self.b[(l, "k")], "k", l, self.heads_e("k", l), "k0")
        v = self.qkv(self.lin(x, l, "v"), self.b[(l, "v")], "vc", l, self.Ev("vc", l, cfg.D), "v")
        q = self.host(self.qc, "q", l, self.heads_e("q", l))
        r = self.resadd(self.query, self.attend(q, k, v, l), l)
        return self.mlp(r, l)

    def forward(self, clip, trace=None):
        """uint8 clip [16, 256, 256, 3] -> the 174 logits."""
        cfg = self.cfg
        x = patches(clip, cfg).astype(np.float64)
        h = self.resadd(np.zeros((cfg.N, cfg.D)), self.lin(x, 0, "pe") + self.ibias(self.b0), 0)
        for l in range(cfg.L):
            h = self.block(h, l, True)
        xf = self.ln(h, self.lnf, "xf", cfg.lf)
        h = self.resadd(np.zeros_like(xf), xf, cfg.lf)
        for l in cfg.lp:
            h = self.block(h, l, False)
        r = self.cross(h)
        xc = self.host(r, "xc", cfg.lk, self.E("xc", cfg.lk))
        logits = (self.lin(xc, cfg.lk, "cls") + self.b[(cfg.lk, "cls")])[0]
        if trace is not None:
            trace.update(xf=xf, pooled=r[0], logits=logits)
        return logits


# ------------------------------------------------------------------ formats
def make_formats(pol, ch, KW, cfg):
    """Exponents from the float calibration maxima (vlm_study.make_vformats' rules,
    one bit of headroom; q / k per head, every other class per channel).  The cross
    layer's one-row tensors (LN2 out, fc1, GELU, fc2, the pooled row, the logits) get
    one exponent per tensor: a calibration run gives them one row a clip, too few
    for per-channel maxima.  Policy ch_floor = k: no channel more than k bits finer
    than its tensor's maximum (its calibration maximum raised to tensor max / 2^k)."""
    if not pol or not pol.get("fmt"):
        return {}
    kind, per_tensor = pol["fmt"], pol.get("per_tensor", False)
    H, HD = cfg.H, cfg.HD
    fb = ls.fbits
    fmt = {}

    row = {(c, cfg.lc) for c in ("x2", "f", "a", "d")} | {("xc", cfg.lk), ("logits", cfg.lk)}
    cf = pol.get("ch_floor")

    def rng(cls, l, cap):
        m = ch[(cls, l)]
        if per_tensor or (cls, l) in row:
            m = np.full_like(m, m.max())
        elif cf is not None:
            m = np.maximum(m, m.max() / 2.0 ** cf)
        f = fb(m)
        return np.minimum(f, cap) if cap is not None else f

    def heads(cls, l, cap):
        m = ch[(cls, l)].reshape(H, -1).max(1)
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fb(m)
        return np.minimum(f, cap) if cap is not None else f

    out_cap = 8 if kind == "fit" else None
    in_cap = pol.get("in_cap", 8)
    fmt[("pe", 0)] = rng("pe", 0, out_cap)
    for l in cfg.attn_layers():
        for cls in ("x", "x2", "a"):
            fmt[(cls, l)] = rng(cls, l, in_cap)
        for cls in ("k0", "v", "f", "d") + (("q0", "o") if l != cfg.lc else ()):
            fmt[(cls, l)] = rng(cls, l, out_cap)
        if l < cfg.L:                                   # the ROPE's inputs
            fmt[("qb", l)], fmt[("kb", l)] = heads("qb", l, None), heads("kb", l, None)
        fq, fk = heads("q", l, in_cap), heads("k", l, in_cap)
        smax = ch[("s", l)] if not per_tensor else np.full(H, ch[("s", l)].max())
        fq = np.minimum(fq, fb(smax) + 8 - fk)
        pvm = ch[("pv", l)] if not per_tensor else np.full(cfg.D, ch[("pv", l)].max())
        fpv_max = fb(pvm).reshape(H, HD)
        fp = np.full(H, pol["p_bits"])
        fvc = np.minimum(rng("vc", l, in_cap), (fpv_max + 8 - fp[:, None]).reshape(-1))
        fmt[("q", l)], fmt[("k", l)], fmt[("p", l)], fmt[("vc", l)] = fq, fk, fp, fvc
    fmt[("xf", cfg.lf)] = rng("xf", cfg.lf, in_cap)
    fmt[("xc", cfg.lk)] = rng("xc", cfg.lk, in_cap)
    fmt[("logits", cfg.lk)] = rng("logits", cfg.lk, out_cap)
    if pol.get("res_bits"):          # the int32 residual: one bit of headroom below 2^31
        hmax = float(ch[("h", 0)].max())
        fmt[("res", cfg.NL)] = int(min(int(pol["res_bits"]), math.floor(math.log2(2.0 ** 31 / (2.0 * hmax)))))
    # every weight must fit int16 at fw[i][j] = f_out[j] + 8 - f_in[i]: lower f_out[j] where not
    for (l, key), Wk in KW.items():
        ci, co = VJEPA2._io(key)
        Wk = Wk.astype(np.float64)
        if ci == "pix":
            fa = np.zeros(Wk.shape[0], np.int64)
        elif ci == "pv":
            fa = (np.asarray(fmt[("p", l)])[:, None] + np.asarray(fmt[("vc", l)]).reshape(H, HD) - 8).reshape(-1)
        else:
            fa = np.broadcast_to(np.asarray(fmt[(ci, l)]), (Wk.shape[0],))
        with np.errstate(divide="ignore"):
            lim = np.floor(np.log2(32767.0 / np.maximum(np.abs(Wk), 1e-30)))
        fo_max = (lim + fa[:, None] - 8).min(0).astype(np.int64)
        fmt[(co, l)] = np.minimum(np.broadcast_to(np.asarray(fmt[(co, l)]), fo_max.shape), fo_max)
    return {k: ls._vec_or_scalar(v) for k, v in fmt.items()}


# ------------------------------------------------------------------ policies
_F = dict(residual="float")
_P = dict(_F, fmt="pow2")
_UNITS = dict(vop_gelu=True, vsmx=True)
# the row-ops unit (doc/plans/HOSTOPS_PLAN.md): int32 residual at 2^-15, its LayerNorms,
# residual adds with the bias rounded once, q / k / v bias adds (COPY), RoPE (ROPE)
_ROW = dict(res_bits=15, inorm=True, ibias=True, iqkv=True, irope=True)
POLICIES = {
    "float":                        None,
    "bf16":                         dict(bf16=True),
    "q88":                          dict(residual="q88"),            # the BERT partition: all Q8.8
    "pow2+p12":                     dict(_P, p_bits=12),
    "pow2+p14":                     dict(_P, p_bits=14),
    "pow2+p12+vgelu+vsmx":          dict(_P, p_bits=12, **_UNITS),
    "pow2+p12+vgelu+vsmx+rowops":   dict(_P, p_bits=12, **_UNITS, **_ROW),     # the planned partition
    "pow2+p14+vgelu+vsmx+rowops":   dict(_P, p_bits=14, **_UNITS, **_ROW),
    "pow2+p12+vgelu+vsmx+rowops+cf6": dict(_P, p_bits=12, ch_floor=6, **_UNITS, **_ROW),
    "pow2+p12+fw":                  dict(_P, p_bits=12, float_weights=True),  # diagnostic: float weights
}


def formats_json(fmt, name, pol):
    def enc(f):
        return {f"{k[0]}@{k[1]}": np.asarray(v).tolist() for k, v in f.items()}
    return {"policy": name, "spec": pol, "margin": ls.MARGIN, "exponents": enc(fmt)}
