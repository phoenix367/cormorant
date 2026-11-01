# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLA frontend (lerobot/smolvla_base: SmolVLM2-500M's vision encoder and
first 16 text layers, plus the flow-matching action expert;
doc/plans/VLA_PLAN.md): the policy's config.json + SmolVLM2's config.json +
the checkpoint + the calibrated formats (demo/vla/scripts/vla_study.py
formats) -> the entries of one multi-entry project.  The specification is
demo/vla/scripts/smolvla_q.py (SmolVLA with the formats), which every
entry's simulation equals bit for bit.  Needs the row-ops unit and
VectorOPKernel's softmax and activation units (the policy
pow2+p12+vgelu+vsmx+rowops).

  vision      patches [1024][768] (the resized frame's raw uint8 pixels,
              src/vit.py's input) -> state vla.img [64][960] (float32 DMA): the
              image features, sqrt(960) folded into the connector
  image_<c>   (state vla.img) -> state vla.img<c> [64][960] (float32 host):
              camera c's features (c < the cameras of the config)
  prefix_<C>  ids [L] (the task's ids, the rest any), n (the prefix's valid rows:
              64 C + the ids + 1), state [S] (f32, normalised, zero padded),
              zero (0) -> the VLM's K / V caches (kv.k.l<i> / kv.v.l<i>, rows < n)
              of every layer but the last, the cross caches (vla.xk.l<i> /
              vla.xv.l<i>) of the odd layers
  expert      x [R][A] (f32: x_t), step (0 .. steps - 1), pos (the prefix's n),
              n (R), xpos (pos - R), zero (0) -> v [R][A] (f32: the velocity);
              writes the self layers' rows pos .. pos + R - 1 of the VLM caches

The prefix (T = roundup(64 C + L + 1, 64) rows, n valid; positions = rows):
  h0 = VlaPrefix(ids, n, state, the image states)       int32 residual at 2^-15
  per layer l:
    x = RMSNorm(h); k0 = x . Wk; v0 = x . Wv
    l < L - 1:  q0 = x . Wq; LlmAttnPrep (rowops) -> the caches, the q image;
                per KV group g: s_g = q.K^T (ConvKernel), p_g = LlmAttnSoftmax
                (the unit, causal 2: every row over the keys < n - 1), o_g = P.V,
                st_g = VlaStateAttn (the state row over the keys < n);
                pv = VlaAttnMerge; h += pv . Wo; the SwiGLU MLP
    l odd:      kr = VlaAttnPrep(parts k) (RoPE of k0 at the cache's exponents),
                vr = VlaAttnPrep(parts v) (v0 re-rounded at the capped f_vin);
                AsrCrossPrep(kr . Wk', vr . Wv') -> the cross caches
The expert (T = roundup(R, 16) rows, R valid):
  xq = VlaActIn(x); a0 = xq . W_in (action_in_proj and the action half of
  action_time_mlp_in fused); at = VlaTimeSilu(a0, step); h = VitResAdd(at . W_out,
  0, b_out) (the int32 residual)
  per layer l: x = RMSNorm(h); q0 = x . Wq
    self (l even):  k0, v0; LlmAttnPrep (rowops) at pos -> the VLM caches' rows
                    pos + t; q.K^T / softmax (the unit, causal) / P.V over the
                    keys < pos + 1 + t
    cross (l odd):  VlaAttnPrep(parts q) at positions t; q.K^T / softmax (causal 0:
                    the keys < xpos + n = pos) / P.V over the cross caches
    pv = LlmAttnMerge; h += pv . Wo; the SwiGLU MLP
  v = VlaVelocity(RMSNorm(h) . W_v, b_v)
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
from .llama import LlamaConfig, graph_with_initializers, rope_tables
from .llm_nodes import LLM_DOMAIN
from .vit import VisionFormats, VitConfig, VitFrontend

F = 8                                   # the kernels' output shift
RES_BITS = 15                           # the int32 residual's exponent
KEY_Q = 64                              # the attention's key quantum (16 x the V interleave 4)
PV_KW = 4                               # the V caches' interleave
ROPE_THETA = 10000.0                    # lerobot apply_rope's max_wavelength
PFX = "model.vlm_with_expert."
VLM_PFX = PFX + "vlm.model."
TXT_PFX = VLM_PFX + "text_model."
EXP_PFX = PFX + "lm_expert."
VISION_DEFAULTS = dict(num_hidden_layers=12, intermediate_size=3072, layer_norm_eps=1e-6)   # SmolVLMVisionConfig


def _roundup(x: int, q: int) -> int:
    return -(-int(x) // q) * q


def intermediate_size(hidden, mult=4, multiple_of=256):
    """lerobot get_intermediate_size: the expert's SwiGLU width."""
    h = int(2 * hidden / 3)
    h = int(mult * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


@dataclass
class SmolVLAConfig:
    L: int                  # VLM layers = expert layers
    D: int                  # the VLM's width
    De: int                 # the expert's width
    H: int
    KV: int
    HD: int
    F: int                  # the VLM's MLP
    Fe: int                 # the expert's MLP
    eps: float
    V: int                  # the vocabulary
    every: int              # self-attention every n layers (the others cross)
    chunk: int              # actions per chunk (R)
    steps: int              # Euler steps
    max_state: int
    max_action: int
    tok_max: int            # language ids
    cameras: int            # the config's cameras
    img: int                # image tokens per camera
    image_size: int
    state_dim: int
    action_dim: int
    min_period: float
    max_period: float
    vision: dict            # SmolVLM2's config (src/vit.py VitConfig.from_dict)

    @classmethod
    def from_dicts(cls, c: dict, v: dict) -> "SmolVLAConfig":
        """``c``: the policy's config.json, ``v``: SmolVLM2's."""
        if (c["attention_mode"] != "cross_attn" or c["add_image_special_tokens"] or c["prefix_length"] > 0
                or c["num_expert_layers"] > 0 or c["empty_cameras"] or c["adapt_to_pi_aloha"]):
            raise ValueError("SmolVLA: only smolvla_base's layout (cross_attn, no image special tokens, no "
                             "prefix padding, one expert layer per VLM layer, no empty cameras, no aloha)")
        t = v["text_config"]
        D, H = int(t["hidden_size"]), int(t["num_attention_heads"])
        De = int(D * float(c["expert_width_multiplier"]))
        vis = dict(v)
        vis["vision_config"] = dict(VISION_DEFAULTS, **v["vision_config"])
        vc = VitConfig.from_dict(vis)
        w, h = c["resize_imgs_with_padding"]
        if (w, h) != (vc.S, vc.S):
            raise ValueError(f"resize_imgs_with_padding {w}x{h}: the vision encoder takes {vc.S}x{vc.S}")
        cams = [k for k, f in c["input_features"].items() if f["type"] == "VISUAL"]
        if vc.n_img + 1 < int(c["chunk_size"]):
            # the cross attention's start xpos = n - chunk (the prefix's valid rows n >= one camera's
            # image tokens + the state row) must not be negative
            raise ValueError(f"SmolVLA: {c['chunk_size']} actions per chunk exceed one camera's prefix "
                             f"({vc.n_img} image tokens + the state row)")
        return cls(L=int(c["num_vlm_layers"]), D=D, De=De, H=H, KV=int(t["num_key_value_heads"]),
                   HD=int(t.get("head_dim") or D // H), F=int(t["intermediate_size"]), Fe=intermediate_size(De),
                   eps=float(t["rms_norm_eps"]), V=int(t["vocab_size"]), every=int(c["self_attn_every_n_layers"]),
                   chunk=int(c["chunk_size"]), steps=int(c["num_steps"]), max_state=int(c["max_state_dim"]),
                   max_action=int(c["max_action_dim"]), tok_max=int(c["tokenizer_max_length"]),
                   cameras=len(cams), img=vc.n_img, image_size=vc.S,
                   state_dim=int(c["input_features"]["observation.state"]["shape"][0]),
                   action_dim=int(c["output_features"]["action"]["shape"][0]),
                   min_period=float(c["min_period"]), max_period=float(c["max_period"]), vision=vis)

    @classmethod
    def from_files(cls, policy_config: str, vlm_config: str) -> "SmolVLAConfig":
        with open(policy_config) as f, open(vlm_config) as g:
            return cls.from_dicts(json.load(f), json.load(g))

    def self_layer(self, li: int) -> bool:
        return self.every > 0 and li % self.every == 0

    @property
    def emb_scale(self) -> float:
        """sqrt(D) as float32 (torch scales the image and language embeddings so)."""
        return float(np.float32(math.sqrt(self.D)))

    def prefix_rows(self, cams: int) -> int:
        """T of prefix_<cams>: the image rows, the language ids and the state row, rounded up to KEY_Q."""
        return _roundup(cams * self.img + self.tok_max + 1, KEY_Q)

    @property
    def expert_rows(self) -> int:
        return _roundup(self.chunk, 16)

    @property
    def cache_rows(self) -> int:
        """C: the longest prefix plus the action rows, rounded up to KEY_Q."""
        return _roundup(self.cameras * self.img + self.tok_max + 1 + self.chunk, KEY_Q)

    def llama(self) -> LlamaConfig:
        """The RoPE tables' parameters (head dim, theta 10000)."""
        return LlamaConfig(L=self.L, D=self.D, H=self.H, KV=self.KV, HD=self.HD, FF=self.F, V=self.V,
                           eps=self.eps, theta=ROPE_THETA)


class SmolVLAFormats:
    """Exponents per ``class@layer`` of the parts vision / text / expert
    (vla_study.py formats; smolvla_q.formats_json)."""

    def __init__(self, d: dict, cfg: SmolVLAConfig):
        self.cfg = cfg
        self.exp = {p: dict(v) for p, v in d["exponents"].items()}
        self.policy = d.get("policy", "")
        self.vision = VisionFormats({"exponents": self.exp["vision"], "policy": self.policy},
                                    VitConfig.from_dict(cfg.vision))

    @classmethod
    def from_file(cls, path: str, cfg: SmolVLAConfig) -> "SmolVLAFormats":
        with open(path) as f:
            return cls(json.load(f), cfg)

    def get(self, cls_: str, part: str, layer: int, n: int) -> np.ndarray:
        """A scalar broadcast, a per-head vector repeated over its heads' channels."""
        v = np.asarray(self.exp[part].get(f"{cls_}@{layer}", F), np.int64).reshape(-1)
        if v.size == 1:
            return np.full(n, int(v[0]), np.int64)
        if v.size == n:
            return v.copy()
        if n % v.size:
            raise ValueError(f"formats: {part} {cls_}@{layer} has {v.size} values, {n} wanted")
        return np.repeat(v, n // v.size)


def _compact(e):
    e = np.asarray(e, np.int64).reshape(-1)
    return int(e[0]) if (e == e[0]).all() else [int(v) for v in e]


def time_embedding(t: float, dim: int, min_period: float, max_period: float) -> np.ndarray:
    """lerobot create_sinusoidal_pos_embedding for one (float32) time: [sin, cos] float64."""
    frac = np.linspace(0.0, 1.0, dim // 2)
    period = min_period * (max_period / min_period) ** frac
    x = (1.0 / period * 2 * math.pi) * t
    return np.concatenate([np.sin(x), np.cos(x)])


def suffix_weights(W: Dict[str, np.ndarray], cfg: SmolVLAConfig):
    """smolvla_q.suffix_weights: W_in [A][De] (float32 values), the per-step bias
    [steps][De] (float64), mlp_out [De][De], its bias, action_out [De][A], its bias."""
    g = lambda n: np.asarray(W["model." + n], np.float64)   # noqa: E731
    A, ab = g("action_in_proj.weight"), g("action_in_proj.bias")
    M, mb = g("action_time_mlp_in.weight"), g("action_time_mlp_in.bias")
    Ma, Mt = M[:, :cfg.De], M[:, cfg.De:]
    w_in = (A.T @ Ma.T).astype(np.float32).astype(np.float64)
    dt = -1.0 / cfg.steps
    times = [float(np.float32(1.0 + s * dt)) for s in range(cfg.steps)]
    b_step = np.stack([ab @ Ma.T + time_embedding(t, cfg.De, cfg.min_period, cfg.max_period) @ Mt.T + mb
                       for t in times])
    return (w_in, b_step, g("action_time_mlp_out.weight").T, g("action_time_mlp_out.bias"),
            g("action_out_proj.weight").T, g("action_out_proj.bias"))


def vision_weights(W: Dict[str, np.ndarray], cfg: SmolVLAConfig) -> Dict[str, np.ndarray]:
    """src/vit.py's names (model.vision_model.*, the connector) for the checkpoint's
    vision tensors; sqrt(D) folded into the connector, rounded to float32 (the
    specification's connector_weight)."""
    out = {}
    for k, v in W.items():
        if k.startswith(VLM_PFX + "vision_model."):
            out["model." + k[len(VLM_PFX):]] = v
    from .vit import CONN
    Wc = np.asarray(W[VLM_PFX + "connector.modality_projection.proj.weight"], np.float64)
    out[CONN] = (Wc * cfg.emb_scale).astype(np.float32)
    return out


class SmolVLAFrontend:
    """Builds the entries.  ``weights``: the checkpoint's tensors by name (float32)."""

    def __init__(self, cfg: SmolVLAConfig, weights: Dict[str, np.ndarray], formats: SmolVLAFormats,
                 name: str = "smolvla"):
        from . import rowops_nodes, smx_nodes, vectorop_act
        if not rowops_nodes.enabled():
            raise ValueError("SmolVLA needs the row-ops unit (kernels.rowops.unit)")
        if not smx_nodes.enabled() or not vectorop_act.enabled():
            raise ValueError("SmolVLA needs VectorOPKernel's softmax and activation units")
        self.cfg, self.W, self.fmt, self.name = cfg, weights, formats, name
        C = cfg.cache_rows
        cos, sin = rope_tables(cfg.llama(), C)
        self._cos, self._sin = cos, sin
        self.suffix = suffix_weights(weights, cfg)
        # the expert's self layers write into the VLM's caches: one set of exponents
        for li in range(cfg.L):
            if cfg.self_layer(li):
                for c_ in ("k", "vc"):
                    n = cfg.KV * cfg.HD
                    if not np.array_equal(formats.get(c_, "text", li, n), formats.get(c_, "expert", li, n)):
                        raise ValueError(f"formats: {c_}@{li} of the VLM and the expert differ (shared cache)")
            for part, key in (("text", "o"), ("text", "d"), ("expert", "o"), ("expert", "d")):
                fd = formats.get(key, part, li, cfg.D if part == "text" else cfg.De)
                if (fd > RES_BITS).any() and not (part == "text" and key in ("o", "d") and li == cfg.L - 1):
                    raise ValueError(f"formats: the {part} {key} exponents of layer {li} exceed 2^-{RES_BITS}")

    # ---- builder helpers (src/llama.py's conventions) ------------------------ #
    def _new(self, ename):
        self._nodes: List[onnx.NodeProto] = []
        self._vi: Dict[str, onnx.ValueInfoProto] = {}
        self._inits: Dict[str, np.ndarray] = {}
        self._meta = numeric.empty()
        self._e = ename

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, host=None, state=False, dma=None):
        if name not in self._vi:
            self._vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if dma is not None:
            self._meta["dma"][name] = dma
        if exp is not None:
            self._meta["exp"][name] = _compact(exp)
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

    def _res(self, name, T, D):
        return self._t(name, [T, D], exp=RES_BITS, dma="i32")

    def _rms(self, h, gname, T, D, out, exp, name):
        g = self._init(f"g.{gname}", self.W[gname])
        self._t(out, [T, D], exp=exp)
        self._node("LlmRMSNorm", [h, g], [out], name, axi_eps=repr(float(self.cfg.eps)), rowops=1)
        return out

    def _w(self, name) -> np.ndarray:
        return np.asarray(self.W[name], np.float32)

    def _rope_init(self):
        self._init("vla.cos", self._cos)
        self._init("vla.sin", self._sin)

    def _vlm_caches(self, li):
        c, fm, C = self.cfg, self.fmt, self.cfg.cache_rows
        n = c.KV * c.HD
        kc = self._t(f"kv.k.l{li}", [C, n], exp=fm.get("k", "text", li, n), state=True)
        vc = self._t(f"kv.v.l{li}", [C, n], exp=fm.get("vc", "text", li, n), state=True)
        self._meta["layout"][kc] = [c.KV, c.HD]
        self._meta["layout"][vc] = [c.KV, c.HD, PV_KW]
        self._init(kc, np.zeros((C, n), np.float32))
        self._init(vc, np.zeros((C, n), np.float32))
        return kc, vc

    def _cross_caches(self, li, written: bool = False):
        c, fm, C = self.cfg, self.fmt, self.cfg.cache_rows
        n = c.KV * c.HD
        ck = self._t(f"vla.xk.l{li}", [C, n], exp=fm.get("k", "expert", li, n), state=True)
        cv = self._t(f"vla.xv.l{li}", [C, n], exp=fm.get("vc", "expert", li, n), state=True)
        self._meta["layout"][ck] = [c.KV, c.HD]
        self._meta["layout"][cv] = [c.KV, c.HD, PV_KW]
        if not written:
            self._init(ck, np.zeros((C, n), np.float32))
        self._init(cv, np.zeros((C, n), np.float32))
        return ck, cv

    def _staging(self, T, prefix):
        c = self.cfg
        qs = self._t(f"{prefix}.ro.qs", [T, c.H * c.HD], exp=0, state=True)
        ks = self._t(f"{prefix}.ro.ks", [T, c.KV * c.HD], exp=0, state=True)
        vs = self._t(f"{prefix}.ro.vs", [T, c.KV * c.HD], exp=0, state=True)
        self._init(qs, np.zeros((T, c.H * c.HD), np.float32))
        self._init(ks, np.zeros((T, c.KV * c.HD), np.float32))
        self._init(vs, np.zeros((T, c.KV * c.HD), np.float32))
        return qs, ks, vs

    def _attn(self, el, part, li, T, kc, vc, qx, pos, n, causal, state_row=False):
        """Per KV group: q.K^T, the unit's softmax, P.V (+ the state row); -> pv."""
        c, fm = self.cfg, self.fmt
        G, KV, HD, H = c.H // c.KV, c.KV, c.HD, c.H
        grp = np.arange(H) // G
        fq, fk = fm.get("q", part, li, H), fm.get("k", part, li, KV)
        fs, fp = fq + fk[grp] - F, fm.get("p", part, li, H)
        common = dict(num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=KEY_Q)
        o, st = [], []
        for g in range(KV):
            hs = slice(g * G, (g + 1) * G)
            s_ = self._t(f"{el}.s{g}", [c.cache_rows, G * T], exp=0)
            p_ = self._t(f"{el}.p{g}", [G * T, c.cache_rows], exp=0)
            o_ = self._t(f"{el}.o{g}", [G * T, HD], exp=0)
            self._node("LlmAttnScores", [kc, qx, pos, n], [s_], f"{el}.qk{g}", group=g, qk_kw=1, **common)
            self._node("LlmAttnSoftmax", [s_, pos, n], [p_], f"{el}.softmax{g}", group=g,
                       s_exp=[int(v) for v in fs[hs]], p_exp=[int(v) for v in fp[hs]], vsmx=1, causal=causal,
                       **common)
            self._node("LlmAttnPV", [p_, vc, pos, n], [o_], f"{el}.pv{g}", group=g, **common)
            o.append(o_)
            if state_row:
                st_ = self._t(f"{el}.st{g}", [G, HD], host="i16")
                self._node("VlaStateAttn", [s_, vc, n], [st_], f"{el}.state_attn{g}", group=g,
                           s_exp=[int(v) for v in fs[hs]], p_exp=[int(v) for v in fp[hs]], **common)
                st.append(st_)
        fvc = fm.get("vc", part, li, KV * HD)
        pv = self._t(f"{el}.pv", [T, H * HD], exp=(fp[:, None] + fvc.reshape(KV, HD)[grp] - F).reshape(-1))
        if state_row:
            self._node("VlaAttnMerge", o + st + [n], [pv], f"{el}.attn_merge", num_heads=H, num_kv_heads=KV,
                       head_dim=HD)
        else:
            self._node("LlmAttnMerge", o, [pv], f"{el}.attn_merge", num_heads=H, num_kv_heads=KV, head_dim=HD)
        return pv

    def _mlp(self, el, part, li, h, T, D, FF, wp):
        fm = self.fmt
        h1 = h
        x2 = self._rms(h1, f"{wp}layers.{li}.post_attention_layernorm.weight", T, D, f"{el}.x2",
                       fm.get("x2", part, li, D), f"{el}.norm_post")
        tag = "t" if part == "text" else "e"
        g = self._matmul(x2, f"w.{tag}.l{li}.g", self._w(f"{wp}layers.{li}.mlp.gate_proj.weight").T, T, f"{el}.g",
                         fm.get("g", part, li, FF), f"{el}.gate_proj")
        u = self._matmul(x2, f"w.{tag}.l{li}.u", self._w(f"{wp}layers.{li}.mlp.up_proj.weight").T, T, f"{el}.u",
                         fm.get("u", part, li, FF), f"{el}.up_proj")
        a = self._t(f"{el}.a", [T, FF], exp=fm.get("a", part, li, FF))
        self._node("LlmSiluMul", [g, u], [a], f"{el}.silu_mul", rowops=1)
        d = self._matmul(a, f"w.{tag}.l{li}.d", self._w(f"{wp}layers.{li}.mlp.down_proj.weight").T, T, f"{el}.d",
                         fm.get("d", part, li, D), f"{el}.down_proj")
        h2 = self._res(f"{el}.h2", T, D)
        self._node("LlmResAdd", [h1, d], [h2], f"{el}.res_mlp", rowops=1)
        return h2

    # ---- the vision entry and the image slots --------------------------------- #
    def vision_entry(self) -> onnx.ModelProto:
        vc = VitConfig.from_dict(self.cfg.vision)
        fe = VitFrontend(vc, vision_weights(self.W, self.cfg), self.fmt.vision, name=self.name,
                         image_state="vla.img", conn_k=min(4096, vc.D * vc.scale ** 2), vsmx=True, rowops=True)
        return fe.entry()

    def image_entry(self, cam: int) -> onnx.ModelProto:
        c = self.cfg
        e = f"image_{cam}"
        self._new(e)
        src = self._t("vla.img", [c.img, c.D], dma="f32", state=True)
        self._init(src, np.zeros((c.img, c.D), np.float32))
        dst = self._t(f"vla.img{cam}", [c.img, c.D], host="f32", state=True)
        self._node("LlmToHost", [src], [dst], f"{e}.copy")
        return self._model(e, [], [], {"entry": e, "camera": cam})

    # ---- the prefix ------------------------------------------------------------ #
    def prefix_entry(self, cams: int, output: bool = False) -> onnx.ModelProto:
        """``prefix_<cams>``; ``output``: also the last cross layer's K' (float32) as a
        graph output ``<entry>.kx`` (tests: it depends on every layer)."""
        c, fm = self.cfg, self.fmt
        if not 1 <= cams <= c.cameras:
            raise ValueError(f"prefix: 1 .. {c.cameras} cameras, got {cams}")
        e = f"prefix_{cams}"
        self._new(e)
        T, D, KV, HD, H, L = c.prefix_rows(cams), c.D, c.KV, c.HD, c.H, c.L
        outputs = []
        ids = self._t(f"{e}.ids", [c.tok_max], TensorProto.INT32, host="i32")
        n = self._t(f"{e}.n", [1], TensorProto.INT32, host="i32")
        state = self._t(f"{e}.state", [c.max_state], host="f32")
        zero = self._t(f"{e}.zero", [1], TensorProto.INT32, host="i32")
        imgs = []
        for cam in range(cams):
            t = self._t(f"vla.img{cam}", [c.img, D], host="f32", state=True)
            self._init(t, np.zeros((c.img, D), np.float32))
            imgs.append(t)
        self._init("emb", self.W[TXT_PFX + "embed_tokens.weight"])
        self._init("vla.w_state", np.asarray(self.W["model.state_proj.weight"], np.float32).T)
        self._init("vla.b_state", self.W["model.state_proj.bias"])
        h = self._res(f"{e}.h0", T, D)
        self._node("VlaPrefix", [ids, n, state, "emb", "vla.w_state", "vla.b_state"] + imgs, [h], f"{e}.embed",
                   scale=repr(c.emb_scale))
        self._rope_init()
        qs, ks, vs = self._staging(T, e)
        for li in range(L):
            el = f"{e}.l{li}"
            wp = f"{TXT_PFX}layers.{li}."
            x = self._rms(h, wp + "input_layernorm.weight", T, D, f"{el}.x", fm.get("x", "text", li, D),
                          f"{el}.norm_in")
            k0 = self._matmul(x, f"w.t.l{li}.k", self._w(wp + "self_attn.k_proj.weight").T, T, f"{el}.k0",
                              fm.get("k0", "text", li, KV * HD), f"{el}.k_proj")
            v0 = self._matmul(x, f"w.t.l{li}.v", self._w(wp + "self_attn.v_proj.weight").T, T, f"{el}.v",
                              fm.get("v", "text", li, KV * HD), f"{el}.v_proj")
            if li < L - 1:
                q0 = self._matmul(x, f"w.t.l{li}.q", self._w(wp + "self_attn.q_proj.weight").T, T, f"{el}.q0",
                                  fm.get("q0", "text", li, H * HD), f"{el}.q_proj")
                kc, vc = self._vlm_caches(li)
                qx = self._t(f"{el}.qx", [KV, HD, (H // KV) * T], exp=0)
                fq = fm.get("q", "text", li, H)
                self._node("LlmAttnPrep", [q0, k0, v0, zero, n, kc, vc, "vla.cos", "vla.sin", qs, ks, vs], [qx],
                           f"{el}.attn_prep", num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=KEY_Q,
                           qk_kw=1, q_exp=[int(v) for v in fq], rowops=1, first=int(li == 0))
                pv = self._attn(el, "text", li, T, kc, vc, qx, zero, n, causal=2, state_row=True)
                ov = self._matmul(pv, f"w.t.l{li}.o", self._w(wp + "self_attn.o_proj.weight").T, T, f"{el}.o",
                                  fm.get("o", "text", li, D), f"{el}.o_proj")
                h1 = self._res(f"{el}.h1", T, D)
                self._node("LlmResAdd", [h, ov], [h1], f"{el}.res_attn", rowops=1)
                h = self._mlp(el, "text", li, h1, T, D, c.F, TXT_PFX)
            if not c.self_layer(li):
                # the expert's cross layer li: K' = RoPE(k) . Wk', V' = v0 . Wv' -> the cross caches
                kr = self._t(f"{el}.kr", [T, KV * HD], exp=fm.get("k", "text", li, KV * HD))
                self._node("VlaAttnPrep", [k0, zero, n, "vla.cos", "vla.sin"], [kr], f"{el}.k_rope", parts="k",
                           num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=KEY_Q)
                ew = f"{EXP_PFX}layers.{li}."
                xk = self._matmul(kr, f"w.e.l{li}.xk", self._w(ew + "self_attn.k_proj.weight").T, T,
                                  f"{el}.xk", fm.get("k", "expert", li, KV * HD), f"{el}.cross_k")
                vr = self._t(f"{el}.vr", [T, KV * HD], exp=fm.get("vin", "expert", li, KV * HD))
                self._node("VlaAttnPrep", [v0, zero, n, "vla.cos", "vla.sin"], [vr], f"{el}.v_round", parts="v",
                           num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=KEY_Q)
                xv = self._matmul(vr, f"w.e.l{li}.xv", self._w(ew + "self_attn.v_proj.weight").T, T,
                                  f"{el}.xv", fm.get("vc", "expert", li, KV * HD), f"{el}.cross_v")
                ck, cv = self._cross_caches(li, written=True)
                self._node("AsrCrossPrep", [xk, xv, cv], [ck], f"{el}.cross_prep", num_heads=KV, head_dim=HD)
                if output and li == max(i for i in range(L) if not c.self_layer(i)):
                    xd = self._t(f"{e}.kx_d", [T, KV * HD], dma="f32")
                    xo = self._t(f"{e}.kx", [T, KV * HD], host="f32")
                    self._node("LlmDequant", [xk], [xd], f"{e}.kx_dequant")
                    self._node("LlmToHost", [xd], [xo], f"{e}.kx_out")
                    outputs.append(xo)
        self._meta["test_fill"][n] = cams * c.img + 8
        self._meta["test_fill"][zero] = 0
        return self._model(e, [ids, n, state, zero], outputs, {"entry": e, "cameras": cams, "rows": T})

    # ---- one denoising step --------------------------------------------------- #
    def expert_entry(self) -> onnx.ModelProto:
        c, fm = self.cfg, self.fmt
        e = "expert"
        self._new(e)
        R, T, A, De, KV, HD, H, L = c.chunk, c.expert_rows, c.max_action, c.De, c.KV, c.HD, c.H, c.L
        x = self._t(f"{e}.x", [R, A], host="f32")
        step = self._t(f"{e}.step", [1], TensorProto.INT32, host="i32")
        pos = self._t(f"{e}.pos", [1], TensorProto.INT32, host="i32")
        n = self._t(f"{e}.n", [1], TensorProto.INT32, host="i32")
        xpos = self._t(f"{e}.xpos", [1], TensorProto.INT32, host="i32")
        zero = self._t(f"{e}.zero", [1], TensorProto.INT32, host="i32")
        w_in, b_step, w_out, b_out, w_v, b_v = self.suffix
        Lx = L                                                        # the suffix classes' layer index
        xq = self._t(f"{e}.xq", [T, A], exp=fm.get("xt", "expert", Lx, A))
        self._node("VlaActIn", [x], [xq], f"{e}.act_in")
        a0 = self._matmul(xq, "w.e.in", w_in, T, f"{e}.a0", fm.get("a0", "expert", Lx, De), f"{e}.in_proj")
        fa0 = fm.get("a0", "expert", Lx, De)
        bq = np.rint(b_step * np.power(2.0, fa0.astype(np.float64))[None, :])
        at = self._t(f"{e}.at", [T, De], exp=fm.get("at", "expert", Lx, De))
        self._node("VlaTimeSilu", [a0, step, self._init("vla.time_bq", bq)], [at], f"{e}.time_silu")
        h0 = self._matmul(at, "w.e.out", w_out, T, f"{e}.h0", fm.get("h0", "expert", Lx, De), f"{e}.out_proj")
        h = self._res(f"{e}.h", T, De)
        self._node("VitResAdd", [h0, self._init("vla.zero_rows", np.zeros((T, De), np.float32)),
                                 self._init("vla.b_out", b_out)], [h], f"{e}.res_in")
        self._rope_init()
        qs, ks, vs = self._staging(T, e)
        common = dict(num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=KEY_Q)
        first = True
        for li in range(L):
            el = f"{e}.l{li}"
            wp = f"{EXP_PFX}layers.{li}."
            xa = self._rms(h, wp + "input_layernorm.weight", T, De, f"{el}.x", fm.get("x", "expert", li, De),
                           f"{el}.norm_in")
            q0 = self._matmul(xa, f"w.e.l{li}.q", self._w(wp + "self_attn.q_proj.weight").T, T, f"{el}.q0",
                              fm.get("q0", "expert", li, H * HD), f"{el}.q_proj")
            fq = fm.get("q", "expert", li, H)
            qx = self._t(f"{el}.qx", [KV, HD, (H // KV) * T], exp=0)
            if c.self_layer(li):
                k0 = self._matmul(xa, f"w.e.l{li}.k", self._w(wp + "self_attn.k_proj.weight").T, T, f"{el}.k0",
                                  fm.get("k0", "expert", li, KV * HD), f"{el}.k_proj")
                v0 = self._matmul(xa, f"w.e.l{li}.v", self._w(wp + "self_attn.v_proj.weight").T, T, f"{el}.v",
                                  fm.get("v", "expert", li, KV * HD), f"{el}.v_proj")
                kc, vc = self._vlm_caches(li)
                self._node("LlmAttnPrep", [q0, k0, v0, pos, n, kc, vc, "vla.cos", "vla.sin", qs, ks, vs], [qx],
                           f"{el}.attn_prep", qk_kw=1, q_exp=[int(v) for v in fq], rowops=1, first=int(first),
                           **common)
                pv = self._attn(el, "expert", li, T, kc, vc, qx, pos, n, causal=1)
            else:
                self._node("VlaAttnPrep", [q0, zero, n, "vla.cos", "vla.sin", qs], [qx], f"{el}.q_rope",
                           parts="q", q_exp=[int(v) for v in fq], first=int(first), **common)
                ck, cv = self._cross_caches(li)
                pv = self._attn(el, "expert", li, T, ck, cv, qx, xpos, n, causal=0)
            first = False
            ov = self._matmul(pv, f"w.e.l{li}.o", self._w(wp + "self_attn.o_proj.weight").T, T, f"{el}.o",
                              fm.get("o", "expert", li, De), f"{el}.o_proj")
            h1 = self._res(f"{el}.h1", T, De)
            self._node("LlmResAdd", [h, ov], [h1], f"{el}.res_attn", rowops=1)
            h = self._mlp(el, "expert", li, h1, T, De, c.Fe, EXP_PFX)
        xf = self._rms(h, EXP_PFX + "norm.weight", T, De, f"{e}.xf", fm.get("xf", "expert", Lx, De), f"{e}.norm_f")
        v0 = self._matmul(xf, "w.e.v", w_v, T, f"{e}.v0", fm.get("vt", "expert", Lx, A), f"{e}.out_v")
        v = self._t(f"{e}.v", [R, A], host="f32")
        self._node("VlaVelocity", [v0, self._init("vla.b_v", b_v)], [v], f"{e}.velocity")
        self._meta["test_fill"][step] = 3
        self._meta["test_fill"][pos] = c.img + 8
        self._meta["test_fill"][n] = R
        self._meta["test_fill"][xpos] = c.img + 8 - R
        self._meta["test_fill"][zero] = 0
        return self._model(e, [x, step, pos, n, xpos, zero], [v], {"entry": e, "rows": T, "chunk": R})

    def entries(self, cameras=None) -> Dict[str, onnx.ModelProto]:
        cams = self.cfg.cameras if cameras is None else int(cameras)
        out = {"vision": self.vision_entry()}
        for cam in range(cams):
            out[f"image_{cam}"] = self.image_entry(cam)
        for k in range(1, cams + 1):
            out[f"prefix_{k}"] = self.prefix_entry(k)
        out["expert"] = self.expert_entry()
        return out

    def _model(self, ename, inputs, outputs, info):
        g = graph_with_initializers(
            self._nodes, f"{self.name}_{ename}",
            [self._vi[n] for n in inputs], [self._vi[n] for n in outputs], self._inits,
            value_info=[vi for n, vi in self._vi.items() if n not in inputs and n not in outputs])
        self._inits = {}
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/smolvla.py")
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key = numeric.METADATA_KEY
        p.value = numeric.to_metadata(self._meta)
        p = m.metadata_props.add()
        p.key = "axi.llm.entry"
        c = self.cfg
        p.value = json.dumps(dict(info, model=self.name, layers=c.L, hidden=c.D, expert_hidden=c.De, heads=c.H,
                                  kv_heads=c.KV, head_dim=c.HD, chunk=c.chunk, steps=c.steps,
                                  max_action=c.max_action, max_state=c.max_state, tok_max=c.tok_max,
                                  image_tokens=c.img, cache_rows=c.cache_rows))
        return m


__all__ = ("SmolVLAConfig", "SmolVLAFormats", "SmolVLAFrontend", "suffix_weights", "vision_weights",
           "time_embedding", "intermediate_size", "KEY_Q", "PV_KW")
