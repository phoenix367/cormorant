#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""smolvla_q.py — SmolVLA (lerobot/smolvla_base) in numpy: the float64
reference and the bit-level emulation of the planned KV260 partition
(model-study skill, route C; doc/plans/VLA_PLAN.md).  pol None = float64.

The model (lerobot 0.6.1, policies/smolvla):
  images     each camera: resize_with_pad to 512 x 512 (bilinear, align_corners
             False, the aspect ratio kept, zero padding on the left and top),
             x * 2 - 1, SmolVLM2-500M's SigLIP encoder (12 layers, 768 wide,
             1024 tokens), the connector (pixel shuffle x4 -> 64 tokens of 12288,
             a linear 12288 -> 960), times sqrt(960)
  language   the task + "\\n", tokenized, up to 48 ids (the padding is masked:
             dropped here), embedding rows times sqrt(960)
  state      float32, padded to 32, state_proj (32 -> 960; a double sum left to
             right, + the bias) -> one token
  prefix     [image tokens per camera, language, state]: T tokens, positions
             0 .. T-1, prefix-LM mask (images and language attend each other,
             the state token attends everything, nothing attends the state)
  VLM        SmolVLM2's text model cut to 16 layers (960 wide, 15 / 5 heads of
             64, SwiGLU 2560, RMSNorm 1e-5), RoPE rotate-half with theta 10000
             (lerobot's apply_rope, not the checkpoint's 100000): only the K / V
             caches of every layer are used (the last layer's q, o and MLP feed
             nothing)
  expert     16 layers 720 wide (q 720 -> 960, 15 / 5 heads of 64, SwiGLU 2048);
             the suffix = 50 action tokens; even layers self-attention over
             [prefix K / V; own K / V] (causal among the actions, positions
             T .. T+49), odd layers cross-attention over K' = K . Wk', V' = V . Wv'
             (the VLM layer's post-RoPE keys / its v projection through 320 x 320
             linears, the same for every denoising step: computed once; v
             re-rounded at the capped input exponent f_vin), q at positions 0 .. 49
  flow       x_1 = noise [50, 32]; ten Euler steps t = 1, 0.9, ..., 0.1:
             x += -0.1 * v(x, t); v = action_out_proj(norm(expert(embed(x, t))));
             embed(x, t) = mlp_out(silu(mlp_in([action_in_proj(x), sincos(t)])))
             -> here the fused W_in = A^T . M_a^T (32 -> 720) and a bias per step
             (the time half of mlp_in is a constant of each step)

Emulated datapath (llm_study.py's conventions: kernel MatMuls on raw int16,
exact ap_fixed<32,16> sums, floor(acc / 2^8), saturate; host regions in
double, round half to even + saturate on write):
  pixels     the resized image rounded to 2^-pix_bits pixel units (0: uint8);
             the vision encoder = vlm_study.VisionModel (the SmolVLM-256M study's
             partition and policies), sqrt(960) folded into the connector (the
             product rounded to float32), the features rounded to float32
  residuals  float32 host tensors (or int32 at 2^-res_bits: the row-ops unit)
  VLM        llm_study.Model's primitives (RMSNorm, RoPE, softmax, SiLU*up) with
             the prefix-LM mask; the K / V caches at per-head / per-channel
             exponents
  expert     the same; the self layers write their K / V rows into the VLM's
             caches (rows T .. T+49) at the caches' exponents, the cross layers'
             K' / V' are kernel MatMul outputs (per head / per channel)
  suffix     x written at f_xt (host), W_in kernel MatMul -> a0; the host adds
             the step's bias as an integer at a0's exponent (round half even,
             saturate) and reads SiLU from a 65 536-entry table (llm_study
             silu_table), written at f_at; mlp_out kernel -> + bias in the residual
  output     action_out_proj kernel -> v = float32(raw 2^-f + bias); x = float32(x - 0.1 v)
             (x and v float32 as in torch; the arithmetic in double)
  rowops     (policy flag ibias) the expert's first residual = the kernel output + the
             bias rounded once at 2^-res_bits (rowops_spec.quant_bias: the unit's ADDX)
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

PFX = "model.vlm_with_expert."
VLM_PFX = PFX + "vlm.model."
TXT_PFX = VLM_PFX + "text_model."
EXP_PFX = PFX + "lm_expert."
ROPE_THETA = 10000.0        # lerobot apply_rope's max_wavelength
S32 = float(np.float32(math.sqrt(960.0)))     # replaced per model in VLACfg (float32, as torch scales)
p2 = ls.p2


# ------------------------------------------------------------------ configuration
def intermediate_size(hidden, mult=4, multiple_of=256):
    """lerobot get_intermediate_size: the expert's SwiGLU width."""
    h = int(2 * hidden / 3)
    h = int(mult * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


class DecCfg:
    """One stack of Llama layers (llm_study.Cfg's fields)."""
    def __init__(self, L, D, H, KV, HD, F, eps, theta):
        self.L, self.D, self.H, self.KV, self.HD, self.F = L, D, H, KV, HD, F
        self.eps, self.theta = float(eps), float(theta)
        self.V, self.tied = 0, False


class VLACfg:
    def __init__(self, ckpt, vlm):
        """``ckpt`` / ``vlm``: the policy's / SmolVLM2's checkpoint directory, or their
        config.json as a dict."""
        c = ckpt if isinstance(ckpt, dict) else json.load(open(os.path.join(ckpt, "config.json")))
        v = vlm if isinstance(vlm, dict) else json.load(open(os.path.join(vlm, "config.json")))
        t = v["text_config"]
        if (c["attention_mode"] != "cross_attn" or c["add_image_special_tokens"] or c["prefix_length"] > 0
                or c["num_expert_layers"] > 0 or c["empty_cameras"] or c["adapt_to_pi_aloha"]):
            raise ValueError("only smolvla_base's layout is modelled: cross_attn, no image special tokens, "
                             "no prefix padding, one expert layer per VLM layer, no empty cameras, no aloha")
        self.chunk, self.steps = c["chunk_size"], c["num_steps"]
        self.max_state, self.max_action = c["max_state_dim"], c["max_action_dim"]
        self.min_period, self.max_period = c["min_period"], c["max_period"]
        self.tok_max = c["tokenizer_max_length"]
        self.img_w, self.img_h = c["resize_imgs_with_padding"]   # lerobot: [1] the height, [0] the width
        self.every = c["self_attn_every_n_layers"]
        self.cameras = [k for k, f in c["input_features"].items() if f["type"] == "VISUAL"]
        self.state_dim = c["input_features"]["observation.state"]["shape"][0]
        self.action_dim = c["output_features"]["action"]["shape"][0]
        L = c["num_vlm_layers"]
        hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        self.text = DecCfg(L, t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], hd,
                           t["intermediate_size"], t["rms_norm_eps"], ROPE_THETA)
        de = int(t["hidden_size"] * c["expert_width_multiplier"])
        self.expert = DecCfg(L, de, t["num_attention_heads"], t["num_key_value_heads"], hd,
                             intermediate_size(de), t["rms_norm_eps"], ROPE_THETA)
        vc = dict(num_hidden_layers=12, intermediate_size=3072, layer_norm_eps=1e-6)   # SmolVLMVisionConfig
        vc.update(v["vision_config"])
        self.vision = vs.VCfg(vc)
        self.scale = v["scale_factor"]
        # torch: img_emb * tensor(sqrt(D), dtype=float32); lang_emb * math.sqrt(D) in float32
        self.emb_scale = float(np.float32(math.sqrt(self.text.D)))
        self.img_tokens = self.vision.N // (self.scale * self.scale)

    def self_layer(self, l):
        return self.every > 0 and l % self.every == 0


# ------------------------------------------------------------------ weights
def vision_weights(W):
    return {k[len(VLM_PFX):]: v for k, v in W.items()
            if k.startswith(VLM_PFX + "vision_model.") or k.startswith(VLM_PFX + "connector.")}


def connector_weight(Wv, cfg):
    """[12288][960]: the connector with sqrt(960) folded in, float32 (as the frontend's
    ONNX initializer holds it)."""
    Wc = Wv["connector.modality_projection.proj.weight"].astype(np.float64).T
    return (Wc * cfg.emb_scale).astype(np.float32)


def vision_kernel_weights(Wv, cfg):
    """vlm_study.kernel_weights with sqrt(960) folded into the connector."""
    KW = vs.kernel_weights(Wv, cfg.vision)
    KW[(cfg.vision.L, "c")] = connector_weight(Wv, cfg).astype(np.float64)
    return KW


def time_embedding(t, dim, min_period, max_period):
    """lerobot create_sinusoidal_pos_embedding for one float32 time: [sin, cos] float64."""
    frac = np.linspace(0.0, 1.0, dim // 2)
    period = min_period * (max_period / min_period) ** frac
    x = (1.0 / period * 2 * math.pi) * t
    return np.concatenate([np.sin(x), np.cos(x)])


def step_times(cfg):
    """The Euler steps' times as torch builds them: float32(1 + step * dt), dt = -1 / steps."""
    dt = -1.0 / cfg.steps
    return [float(np.float32(1.0 + s * dt)) for s in range(cfg.steps)], dt


def suffix_weights(W, cfg):
    """(W_in [32, 720], per-step bias [steps, 720], mlp_out [720, 720], its bias,
    action_out [720, 32], its bias), float64: action_in_proj and the action half of
    action_time_mlp_in fused (rounded to float32, as the frontend's initializer holds
    it), the time half folded into one bias per step (float64: the frontend rounds it
    to integers at a0's exponent itself)."""
    g = lambda n: W["model." + n].astype(np.float64)
    A, ab = g("action_in_proj.weight"), g("action_in_proj.bias")             # [720, 32]
    M, mb = g("action_time_mlp_in.weight"), g("action_time_mlp_in.bias")     # [720, 1440]
    De = cfg.expert.D
    Ma, Mt = M[:, :De], M[:, De:]
    w_in = (A.T @ Ma.T).astype(np.float32).astype(np.float64)               # [32, 720]
    times, _ = step_times(cfg)
    b_step = np.stack([ab @ Ma.T + time_embedding(t, De, cfg.min_period, cfg.max_period) @ Mt.T + mb
                       for t in times])
    return (w_in, b_step, g("action_time_mlp_out.weight").T, g("action_time_mlp_out.bias"),
            g("action_out_proj.weight").T, g("action_out_proj.bias"))


# ------------------------------------------------------------------ preprocessing
def bilinear(img, oh, ow):
    """torch F.interpolate(mode="bilinear", align_corners=False, antialias=False)
    of an [H, W, C] float64 image."""
    h, w = img.shape[:2]

    def axis(n_in, n_out):
        sc = n_in / n_out
        src = np.maximum((np.arange(n_out) + 0.5) * sc - 0.5, 0.0)
        i0 = src.astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        lam = src - i0
        return i0, i1, lam

    y0, y1, ly = axis(h, oh)
    x0, x1, lx = axis(w, ow)
    top = img[y0] * (1 - ly)[:, None, None] + img[y1] * ly[:, None, None]
    return top[:, x0] * (1 - lx)[None, :, None] + top[:, x1] * lx[None, :, None]


def resize_with_pad(img, height, width):
    """lerobot resize_with_pad of a uint8 [H, W, 3] frame (as img / 255), in pixel
    units (0 .. 255) float64 [height, width, 3]: the aspect ratio kept, zero padding
    on the left and top."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img.astype(np.float64)
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    out = np.zeros((height, width, img.shape[2]))
    out[height - rh:, width - rw:] = bilinear(img.astype(np.float64), rh, rw)
    return out


def pixels(img, cfg, pix_bits=None):
    """The vision input: the resized frame, rounded to 2^-pix_bits pixel units
    (None: unrounded, the float reference)."""
    x = resize_with_pad(img, cfg.img_h, cfg.img_w)
    if pix_bits is None:
        return x
    return np.round(x * 2.0 ** pix_bits) / 2.0 ** pix_bits


# ------------------------------------------------------------------ the vision encoder
class VModel(vs.VisionModel):
    """vlm_study.VisionModel with the pixel exponent a policy option (pix_bits)."""
    def E(self, cls, l):
        if cls == "pix":
            return int(self.pol.get("pix_bits", 0))
        return super().E(cls, l)


# ------------------------------------------------------------------ the decoders
# key: (weight name under layers.<l>., input class, output class)
LIN = {"q": ("self_attn.q_proj", "x", "q0"), "k": ("self_attn.k_proj", "x", "k0"),
       "v": ("self_attn.v_proj", "x", "v"), "o": ("self_attn.o_proj", "pv", "o"),
       "g": ("mlp.gate_proj", "x2", "g"), "u": ("mlp.up_proj", "x2", "u"), "d": ("mlp.down_proj", "a", "d")}
# the expert's cross layers: K' / V' from the VLM caches (kin / vin = their exponents)
XLIN = {"xk": ("self_attn.k_proj", "kin", "k"), "xv": ("self_attn.v_proj", "vin", "vc")}


class Dec(ls.Model):
    """One stack of Llama layers on llm_study.Model's primitives (pol None: float64).
    layers: {l: (keys of LIN / XLIN)}; head: [(key, W [K, N], input class, output
    class)] — linears at layer index L (the expert's suffix embedding and output)."""

    def __init__(self, cfg, W, prefix, layers, head=(), pol=None, fmt=None, record_ch=False, cos_sin=None):
        self.cfg = cfg
        self.pol = pol or {}
        self.q = pol is not None
        self.fa_diag = bool(self.pol.get("float_acts"))
        self.qclasses = self.pol.get("qclasses")
        self.bf = bool(self.pol.get("bf16"))
        if self.bf:
            self.q = False
        self.floor_fix = False
        self.hattn = bool(self.pol.get("hattn"))
        self.xattn_mode = False
        self.phase = "prefill"
        self.fmt = fmt or {}
        self.sink, self.sink_model, self._sink = False, None, {}
        self.record_ch = record_ch
        self.grp = np.arange(cfg.H) // (cfg.H // cfg.KV)
        self.cos, self.sin = cos_sin
        g = lambda n: W[n].astype(np.float64)
        self.gin = [g(f"{prefix}layers.{l}.input_layernorm.weight") for l in range(cfg.L)]
        self.gpost = [g(f"{prefix}layers.{l}.post_attention_layernorm.weight") for l in range(cfg.L)]
        self.gfin = g(prefix + "norm.weight")
        self.stats = ls.Stats(cfg.L)
        self.wsat = 0
        self.io, self.w = {}, {}
        for l, keys in layers.items():
            for key in keys:
                name, ci, co = {**LIN, **XLIN}[key]
                self.io[(l, key)] = (ci, co)
                self.w[(l, key)] = self._enc(W[f"{prefix}layers.{l}.{name}.weight"].T, l, key, ci, co)
        for key, Wt, ci, co in head:
            self.io[(cfg.L, key)] = (ci, co)
            self.w[(cfg.L, key)] = self._enc(Wt, cfg.L, key, ci, co)

    def Ev(self, cls, l, n):
        """Per channel of n: a per-head exponent repeated over its head's channels."""
        e = self.E(cls, l)
        if np.ndim(e) == 0:
            return np.full(n, e, np.int64)
        e = np.asarray(e, np.int64)
        return e if len(e) == n else np.repeat(e, n // len(e))

    def lin(self, x, l, key):
        ci, co = self.io[(l, key)]
        if not self.q:
            return self.fmm(x, self.w[(l, key)], co, l)
        n = self.w[(l, key)].shape[1]
        return self.kmm(x * p2(self.Ev(ci, l, x.shape[1])), self.w[(l, key)], self.Ev(co, l, n), 8, co, l)

    def res_init(self, h, l):
        """A residual stream's first rows (float32, or int32 at 2^-res_bits)."""
        if self.bf:
            return ls.bf16(h)
        if self.q and self.pol.get("res_bits"):
            return self.res32(h, l)
        if self.q and self.pol.get("residual") == "q88":
            return np.clip(np.round(h * 256.0), ls.I16_LO, ls.I16_HI) / 256.0
        if self.q:
            return h.astype(np.float32).astype(np.float64)
        return h

    def vcache(self, v, l):
        """The V cache's rows: the host re-rounds the v projection at the cache's exponent."""
        if self.q and ("vc", l) in self.fmt:
            return self.host(v, "vc", l, self.Ev("vc", l, v.shape[1]))
        return v

    def attn_mask(self, q, K, V, mask, l):
        """q [T, H, HD] (after RoPE), K / V [KV, S, HD] (values as stored), mask [T, S]
        -> [T, H * HD] (llm_study.Model.attn with an explicit mask)."""
        cfg = self.cfg
        T, HD = q.shape[0], cfg.HD
        out = np.empty((T, cfg.H, HD))
        if self.q and self.hattn:
            for h in range(cfg.H):
                g = self.grp[h]
                x = np.where(mask, (q[:, h] @ K[g].T) * (1.0 / math.sqrt(HD)), -np.inf)
                e = np.exp(x - x.max(-1, keepdims=True))
                out[:, h] = (e / np.cumsum(e, -1)[..., -1:]) @ V[g]
            return self.host(out.reshape(T, -1), "pv", l, self.E("pv", l))
        if self.q:
            fq, fk, fs = self.Eh("q", l, cfg.H), self.Eh("k", l, cfg.KV), self.Eh("s", l, cfg.H)
            fp, fv, fpv = self.Eh("p", l, cfg.H), self.Ev("vc", l, cfg.KV * HD), self.Ev("pv", l, cfg.H * HD)
        for h in range(cfg.H):
            g = self.grp[h]
            if not self.q:
                s = self.fmm(q[:, h], K[g].T, "s", l, rec=False)
                if self.record_ch and s.size:
                    self.stats.add_ch("s", l, np.abs(np.where(mask, s, 0.0)).max(), h, cfg.H)
                p = self.softmax(s, mask, 0, 0, l)
                out[:, h] = self.fmm(p, V[g], "pv", l, slice(h * HD, (h + 1) * HD), cfg.H * HD)
                continue
            s = self.kmm(q[:, h] * p2(fq[h]), (K[g] * p2(fk[g])).T, fs[h], 8, "s", l)
            p = self.softmax(s, mask, fs[h], fp[h], l)
            out[:, h] = self.kmm(p * p2(fp[h]), V[g] * p2(fv[g * HD:(g + 1) * HD])[None, :],
                                 fpv[h * HD:(h + 1) * HD], 8, "pv", l)
        return out.reshape(T, -1)

    def ibias(self, b):
        """A residual add's bias as the row-ops unit adds it (policy ibias): rounded once
        at 2^-res_bits (rowops_spec.quant_bias)."""
        if self.q and self.pol.get("ibias") and self.pol.get("res_bits"):
            return ls.rowops().quant_bias(b, int(self.pol["res_bits"]))
        return b

    def mlp(self, h, l):
        x2 = self.rms(h, self.gpost[l], "x2", l)
        return self.resadd(h, self.lin(self.silu_mul(self.lin(x2, l, "g"), self.lin(x2, l, "u"), l), l, "d"), l)

    def silu_bias(self, a0, b):
        """silu(a0 + b) of the suffix embedding.  Emulation: b added to a0's raw
        value as an integer at its exponent (round half even, saturate), SiLU from
        silu_table, written at f_at."""
        L = self.cfg.L
        if not self.q or self.exact("a0"):
            y = a0 + b[None, :]
            if self.record_ch:                              # a0's exponent must hold a0 + b too
                self.stats.add_ch("a0", L, np.abs(y).max(0))
            return self.host(y / (1.0 + np.exp(-y)), "at", L, self.Ev("at", L, y.shape[1]))
        fa0 = self.Ev("a0", L, a0.shape[1])
        raw = np.clip(np.rint(a0 * p2(fa0)) + np.rint(b * p2(fa0))[None, :], ls.I16_LO, ls.I16_HI).astype(np.int64)
        y = np.empty(a0.shape)
        for f in np.unique(fa0):
            cols = fa0 == f
            y[:, cols] = ls.silu_table(int(f))[raw[:, cols] + 32768]
        return self.host(y, "at", L, self.Ev("at", L, y.shape[1]))


def layer_keys(cfg, model):
    """The linears each layer runs: the VLM's last layer only k / v (its q, o and
    MLP feed nothing); the expert's cross layers xk / xv instead of k / v."""
    L = cfg.text.L
    if model == "text":
        return {l: (("k", "v") if l == L - 1 else tuple(LIN)) for l in range(L)}
    return {l: (tuple(LIN) if cfg.self_layer(l) else ("q", "xk", "xv", "o", "g", "u", "d")) for l in range(L)}


# ------------------------------------------------------------------ the policy
class SmolVLA:
    """The whole policy network under one numeric policy.  pol: {"vision": .., "text":
    .., "expert": ..} (each a policy dict or None = float64), fmt likewise."""

    def __init__(self, cfg, W, pol=None, fmt=None, record_ch=False, KW=None, cos_sin=None):
        self.cfg = cfg
        pol = pol or {}
        fmt = fmt or {}
        self.pol = pol
        vp = pol.get("vision")
        self.pix_bits = vp.get("pix_bits", 0) if vp and not vp.get("bf16") else None   # None: unrounded
        Wv = vision_weights(W)
        self.KW = KW if KW is not None else vision_kernel_weights(Wv, cfg)
        self.vision = VModel(Wv, self.KW, cfg.vision, cfg.scale, pol.get("vision"), fmt.get("vision"),
                             record_ch=record_ch)
        self.vision.conn_k = vs.CONN_K
        cs = cos_sin if cos_sin is not None else ls.rope_tables(cfg.text, 1024)
        self.text = Dec(cfg.text, W, TXT_PFX, layer_keys(cfg, "text"), (), pol.get("text"), fmt.get("text"),
                        record_ch, cs)
        w_in, self.b_step, w_out, self.b_out, w_v, self.b_v = suffix_weights(W, cfg)
        head = [("ain", w_in, "xt", "a0"), ("aout", w_out, "at", "h0"), ("vout", w_v, "xf", "vt")]
        self.expert = Dec(cfg.expert, W, EXP_PFX, layer_keys(cfg, "expert"), head, pol.get("expert"),
                          fmt.get("expert"), record_ch, cs)
        self.emb = W[TXT_PFX + "embed_tokens.weight"]
        self.ws = W["model.state_proj.weight"].astype(np.float64).T
        self.bs = W["model.state_proj.bias"].astype(np.float64)

    # ---- the prefix
    def image_features(self, frames):
        """uint8 frames (one per camera) -> [64][960] each (times sqrt(960)); float32 in
        the emulation (the vision entry's DEQUANT / host sum writes float32 rows)."""
        out = [self.vision.forward(pixels(f, self.cfg, self.pix_bits)) for f in frames]
        if self.vision.q:
            out = [f.astype(np.float32).astype(np.float64) for f in out]
        return out

    def prefix_rows(self, feats, ids, state):
        cfg = self.cfg
        lang = self.emb[ids].astype(np.float64) * cfg.emb_scale
        s = np.zeros(cfg.max_state)
        s[:len(state)] = np.asarray(state, np.float32)
        srow = (np.cumsum(s[:, None] * self.ws, axis=0)[-1] + self.bs)[None, :]   # left to right
        return np.concatenate(list(feats) + [lang, srow])

    def prefix(self, h):
        """The VLM over the prefix rows h [T, 960] -> per layer K, V [KV, T, HD]
        (values as cached) and v0 [T, KV HD] (the v projection, which the cross
        layers' V' reads)."""
        m, cfg = self.text, self.cfg.text
        T, H, KV, HD, L = len(h), cfg.H, cfg.KV, cfg.HD, cfg.L
        pos = np.arange(T)
        mask = np.ones((T, T), bool)
        mask[:-1, -1] = False                               # nothing but the state attends the state
        h = m.res_init(h, 0)
        K, V, V0 = [], [], []
        for l in range(L):
            x = m.rms(h, m.gin[l], "x", l)
            k = m.rope(m.lin(x, l, "k").reshape(T, KV, HD), pos, "k", l)
            v0 = m.lin(x, l, "v")
            v = m.vcache(v0, l)
            K.append(k.transpose(1, 0, 2))
            V.append(v.reshape(T, KV, HD).transpose(1, 0, 2))
            V0.append(v0)
            if l == L - 1:
                break
            q = m.rope(m.lin(x, l, "q").reshape(T, H, HD), pos, "q", l)
            h = m.resadd(h, m.lin(m.attn_mask(q, K[l], V[l], mask, l), l, "o"), l)
            h = m.mlp(h, l)
        return K, V, V0

    def cross_kv(self, K, V0):
        """The cross layers' K' / V' [KV, T, HD] (once per inference): K' from the
        cached (rotated) keys, V' from the v projection re-rounded at f_vin (the
        row-ops unit's COPY: round half even, saturate)."""
        e, cfg = self.expert, self.cfg.expert
        Kx, Vx = {}, {}
        for l in range(cfg.L):
            if self.cfg.self_layer(l):
                continue
            T = K[l].shape[1]
            kin = K[l].transpose(1, 0, 2).reshape(T, -1)
            vin = V0[l]
            if e.q:
                fv = p2(e.Ev("vin", l, vin.shape[1]))[None, :]
                vin = np.clip(np.rint(vin * fv), ls.I16_LO, ls.I16_HI) / fv
            Kx[l] = e.lin(kin, l, "xk").reshape(T, cfg.KV, cfg.HD).transpose(1, 0, 2)
            Vx[l] = e.lin(vin, l, "xv").reshape(T, cfg.KV, cfg.HD).transpose(1, 0, 2)
        return Kx, Vx

    # ---- one denoising step
    def velocity(self, x, step, K, V, Kx, Vx):
        e, cfg = self.expert, self.cfg.expert
        n, T = len(x), K[0].shape[1]
        H, KV, HD, L = cfg.H, cfg.KV, cfg.HD, cfg.L
        xt = e.host(x, "xt", L, e.Ev("xt", L, x.shape[1]))
        a = e.silu_bias(e.lin(xt, L, "ain"), self.b_step[step])
        h = e.res_init(e.lin(a, L, "aout") + e.ibias(self.b_out)[None, :], L)
        e.stats.add("h", L, float(np.abs(h).max()), 0, h.size)
        pos_s, pos_x = T + np.arange(n), np.arange(n)
        mask_s = np.arange(T + n)[None, :] <= (T + np.arange(n))[:, None]
        mask_x = np.ones((n, T), bool)
        for l in range(L):
            xa = e.rms(h, e.gin[l], "x", l)
            if self.cfg.self_layer(l):
                q = e.rope(e.lin(xa, l, "q").reshape(n, H, HD), pos_s, "q", l)
                k = e.rope(e.lin(xa, l, "k").reshape(n, KV, HD), pos_s, "k", l)
                v = e.vcache(e.lin(xa, l, "v"), l).reshape(n, KV, HD)
                Kc = np.concatenate([K[l], k.transpose(1, 0, 2)], 1)
                Vc = np.concatenate([V[l], v.transpose(1, 0, 2)], 1)
                att = e.attn_mask(q, Kc, Vc, mask_s, l)
            else:
                q = e.rope(e.lin(xa, l, "q").reshape(n, H, HD), pos_x, "q", l)
                att = e.attn_mask(q, Kx[l], Vx[l], mask_x, l)
            h = e.resadd(h, e.lin(att, l, "o"), l)
            h = e.mlp(h, l)
        xf = e.rms(h, e.gfin, "xf", L)
        return (e.lin(xf, L, "vout") + self.b_v[None, :]).astype(np.float32).astype(np.float64)

    def sample(self, frames, ids, state, noise, feats=None, trace=None):
        """One action chunk [chunk, max_action] (normalised units) from uint8 frames
        (one per camera, in config order), the task's valid token ids, the
        normalised state and noise [chunk, max_action].  trace: a dict that
        receives the intermediates (features, caches, the first velocity)."""
        if feats is None:
            feats = self.image_features(frames)
        K, V, V0 = self.prefix(self.prefix_rows(feats, ids, state))
        Kx, Vx = self.cross_kv(K, V0)
        _, dt = step_times(self.cfg)
        x = np.asarray(noise, np.float32).astype(np.float64)
        for step in range(self.cfg.steps):
            v = self.velocity(x, step, K, V, Kx, Vx)
            if trace is not None and step == 0:
                trace["v0"] = v
            x = (x + dt * v).astype(np.float32).astype(np.float64)
        if trace is not None:
            trace.update(feats=feats, K=K, V=V)
        return x


# ------------------------------------------------------------------ formats
def make_formats(pol, chv, cht, che, cfg, KW, W):
    """Exponents of one policy from the float calibration maxima (llm_study.make_formats'
    rules per decoder; the expert's self layers share the VLM layer's K / V cache
    exponents; one bit of headroom)."""
    out = {"vision": vs.make_vformats(pol.get("vision"), chv, KW, cfg.vision, cfg.scale)
           if pol.get("vision") else {}}
    tp, ep = pol.get("text"), pol.get("expert")
    if not (tp and tp.get("fmt")) and not (ep and ep.get("fmt")):
        out.update(text={}, expert={})
        return out
    assert tp and ep and tp.get("fmt") and ep.get("fmt") and tp.get("p_bits") and tp.get("p_bits") == ep.get("p_bits"), \
        "the text and expert policies share their caches: both pow2 / fit with the same p_bits"
    assert not tp.get("hattn") and not ep.get("hattn")
    tc, ec = cfg.text, cfg.expert
    L, H, KV, HD = tc.L, tc.H, tc.KV, tc.HD
    grp = np.arange(H) // (H // KV)
    fb = ls.fbits

    def rng(ch, cls, l, cap, per_tensor=False):
        m = ch[(cls, l)]
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fb(m)
        return np.minimum(f, cap) if cap is not None else f

    def heads(ch, cls, l, n, cap):
        f = fb(ch[(cls, l)].reshape(n, -1).max(1))
        return np.minimum(f, cap) if cap is not None else f

    ft, fe = {}, {}
    for f, ch, p, keys in ((ft, cht, tp, layer_keys(cfg, "text")), (fe, che, ep, layer_keys(cfg, "expert"))):
        out_cap = 8 if p["fmt"] == "fit" else None
        in_cap = p.get("in_cap", 8)
        for l in range(L):
            for key in keys[l]:
                if key in LIN:
                    _, ci, co = LIN[key]
                    if ci != "pv":
                        f[(ci, l)] = rng(ch, ci, l, in_cap)
                    f[(co, l)] = rng(ch, co, l, out_cap)
        if f is fe:
            fe[("xt", L)] = rng(che, "xt", L, in_cap)
            fe[("at", L)] = rng(che, "at", L, in_cap)
            fe[("xf", L)] = rng(che, "xf", L, in_cap)
            for cls in ("a0", "h0", "vt"):
                fe[(cls, L)] = rng(che, cls, L, out_cap)
    pb = tp["p_bits"]
    fp = np.full(H, pb)
    for l in range(L):
        own_t = l < L - 1                       # the VLM's last layer does not attend
        selfl = cfg.self_layer(l)
        in_t, in_e = tp.get("in_cap", 8), ep.get("in_cap", 8)

        def pv_lim(ch):
            fpv_max = fb(ch[("pv", l)]).reshape(H, HD)
            return (fpv_max + 8 - fp[:, None]).reshape(KV, H // KV, HD).min(1).reshape(-1)

        fk = heads(cht, "k", l, KV, in_t)
        fvc = np.minimum(fb(cht[("v", l)]), in_t)
        if own_t:
            fvc = np.minimum(fvc, pv_lim(cht))
        if selfl:                               # the expert writes into this cache
            fk = np.minimum(fk, heads(che, "k", l, KV, in_e))
            fvc = np.minimum(np.minimum(fvc, np.minimum(fb(che[("v", l)]), in_e)), pv_lim(che))
        ft[("k", l)], ft[("vc", l)] = fk, fvc
        if own_t:
            ft[("q", l)] = np.minimum(heads(cht, "q", l, H, in_t), fb(cht[("s", l)]) + 8 - fk[grp])
            ft[("p", l)] = fp
        if selfl:
            fe[("k", l)], fe[("vc", l)] = fk, fvc
            fke = fk
        else:                                   # K' / V': kernel outputs read by kernels
            out_cap = 8 if ep["fmt"] == "fit" else None
            fe[("kin", l)], fe[("vin", l)] = np.repeat(fk, HD), np.asarray(ft[("v", l)])
            fke = heads(che, "k", l, KV, out_cap)
            fe[("k", l)] = fke
            fe[("vc", l)] = np.minimum(rng(che, "vc", l, out_cap), pv_lim(che))
        fe[("q", l)] = np.minimum(heads(che, "q", l, H, in_e), fb(che[("s", l)]) + 8 - fke[grp])
        fe[("p", l)] = fp

    # every weight must fit int16 at fw[i][j] = f_out[j] + 8 - f_in[i]: lower f_out[j] where not
    # (per head where the output is per head: K'); the V cache's producers before o reads it
    def fit_weight(f, Wt, l, ci, co):
        def per_ch(cls):
            e = np.asarray(f[(cls, l)], np.int64)
            n = Wt.shape[0] if cls == ci else Wt.shape[1]
            return np.full(n, int(e)) if e.ndim == 0 else (e if len(e) == n else np.repeat(e, n // len(e)))
        if ci == "pv":
            fvc = per_ch_n(f, "vc", l, KV * HD)
            fa = (np.asarray(f[("p", l)])[:, None] + fvc.reshape(KV, HD)[grp] - 8).reshape(-1)
        else:
            fa = per_ch(ci)
        with np.errstate(divide="ignore"):
            lim = np.floor(np.log2(32767.0 / np.maximum(np.abs(Wt), 1e-30)))
        fo_max = (lim + fa[:, None] - 8).min(0).astype(np.int64)
        cur = np.asarray(f[(co, l)], np.int64)
        if cur.ndim == 1 and len(cur) * HD == len(fo_max):         # per head (K')
            f[(co, l)] = np.minimum(cur, fo_max.reshape(len(cur), HD).min(1))
        else:
            f[(co, l)] = np.minimum(np.broadcast_to(cur, fo_max.shape), fo_max)

    def per_ch_n(f, cls, l, n):
        e = np.asarray(f[(cls, l)], np.int64)
        return np.full(n, int(e)) if e.ndim == 0 else e

    order = ("xk", "xv", "q", "k", "v", "o", "g", "u", "d")
    for f, prefix, model in ((ft, TXT_PFX, "text"), (fe, EXP_PFX, "expert")):
        keys = layer_keys(cfg, model)
        if model == "expert":                   # V' reads the v projection re-rounded, capped as an input
            for l in range(L):
                if not cfg.self_layer(l):
                    fe[("vin", l)] = np.minimum(np.asarray(ft[("v", l)]), ep.get("in_cap", 8))
        for l in range(L):
            for key in order:
                if key in keys[l]:
                    name, ci, co = {**LIN, **XLIN}[key]
                    fit_weight(f, W[f"{prefix}layers.{l}.{name}.weight"].T.astype(np.float64), l, ci, co)
    w_in, _, w_out, _, w_v, _ = suffix_weights(W, cfg)
    for key, Wt, ci, co in (("ain", w_in, "xt", "a0"), ("aout", w_out, "at", "h0"), ("vout", w_v, "xf", "vt")):
        fit_weight(fe, Wt, L, ci, co)
    out["text"] = {k: ls._vec_or_scalar(v) for k, v in ft.items()}
    out["expert"] = {k: ls._vec_or_scalar(v) for k, v in fe.items()}
    return out


# ------------------------------------------------------------------ policies
_RF = dict(residual="float")
_P12 = dict(_RF, fmt="pow2", p_bits=12)
_ROW = dict(res_bits=15, inorm=True, isilu=True, irope=True, ibias=True)
POLICIES = {
    "float": None,
    # yardsticks: bf16 at every boundary; the noise yardstick is float with another seed (study)
    "bf16": dict(vision=dict(bf16=True), text=dict(bf16=True), expert=dict(bf16=True)),
    # the BERT partition (Q8.8 everywhere, saturating residual adds)
    "q88": dict(vision=vs.VPOLICIES["q88"], text=dict(residual="q88", fmt="fit", p_bits=8),
                expert=dict(residual="q88", fmt="fit", p_bits=8)),
    # SmolVLM's recipe: per-channel power-of-two exponents, float residuals, P at 2^-12
    "pow2+p12": dict(vision=vs.VPOLICIES["pow2+p12"], text=dict(_P12), expert=dict(_P12)),
    # ... with the VectorOP GELU / softmax units (doc/plans/OFFLOAD_PLAN.md, SOFTMAX_PLAN.md)
    "pow2+p12+vgelu+vsmx": dict(vision=vs.VPOLICIES["pow2+p12+vgelu+vsmx"], text=dict(_P12, vsmx=True),
                                expert=dict(_P12, vsmx=True)),
    # ... and the row-ops unit: int32 residuals, its norms, RoPE, SiLU * up (HOSTOPS_PLAN); the
    # vision entry as SmolVLM ships it (src/vit.py, IEMBED_DEFAULT False)
    "pow2+p12+vgelu+vsmx+rowops": dict(vision=vs.VPOLICIES["pow2+p12+vgelu+vsmx+rowops"],
                                       text=dict(_P12, vsmx=True, **_ROW), expert=dict(_P12, vsmx=True, **_ROW)),
    # diagnostics: float weights (what a finer weight format could buy)
    "pow2+p12+fw": dict(vision=dict(vs.VPOLICIES["pow2+p12"], float_weights=True),
                        text=dict(_P12, float_weights=True), expert=dict(_P12, float_weights=True)),
}


def part_policy(name, part):
    """One part (vision / text / expert) quantised by the policy name, the others float
    (ablation)."""
    p = POLICIES[name]
    return {k: (v if k in part else None) for k, v in p.items()}


def formats_json(fmt, name, pol):
    def enc(f):
        return {f"{k[0]}@{k[1]}": np.asarray(v).tolist() for k, v in f.items()}
    return {"policy": name, "spec": pol, "margin": ls.MARGIN,
            "exponents": {part: enc(f) for part, f in fmt.items()}}


def load_formats(path):
    d = json.load(open(path))

    def dec(f):
        out = {}
        for k, v in f.items():
            cls, l = k.rsplit("@", 1)
            out[(cls, int(l))] = ls._vec_or_scalar(v) if isinstance(v, list) else int(v)
        return out
    return d["policy"], {part: dec(f) for part, f in d["exponents"].items()}
