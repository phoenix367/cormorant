"""
Vision-encoder frontend (SmolVLM-256M-Instruct: a SigLIP-style ViT + the
Idefics3 connector; doc/plans/CHAT_PLAN.md §22-§23): config.json + the
checkpoint's vision / connector weights + the calibrated vision formats
(demo/chat/scripts/vlm_study.py formats) -> the fixed-shape ``vision`` entry
graph of a multi-entry project (the text model's entries come from
src/llama.py; both share one weight pool).

  vision   patches[N][3*P*P] (raw uint8 pixels as int16, exponent 0)
           -> state img[N / s^2][D_text] (float32): the image features the
              prefill entries' LlmEmbed reads for image ids

The graph (vlm_study.VisionModel's emulation, policy pow2+p12):

  pe  = patches . Wpe               MatMul (the pixel normalisation folded in:
                                     W * 2/255, bias b - sum W)
  h   = VitEmbedAdd(pe, b0)          float32(pe + b0), b0 = bias + position emb.
  per layer:
    x   = VitLayerNorm(h)            q0 / k0 / v = x . W (MatMul)
    qx  = VitAttnPrep(q0, k0, v)     K / V caches (raw DMA states, shared by all
                                     layers), q.K^T input image per head
    per head g: s_g = LlmAttnScores (ConvKernel), P_g = VitAttnSoftmax (host),
                o_g = LlmAttnPV (ConvKernel)          keys = N (static)
    pv  = LlmAttnMerge(o_*)          o = pv . Wo
    h1  = VitResAdd(h, o, bo)        x2 = VitLayerNorm(h1)   f = x2 . W1
    a   = VitGelu(f, b1)             d = a . W2    h = VitResAdd(h1, d, b2)
  xf  = VitLayerNorm(h, post)
  per K chunk k: xs_k = VitPixelShuffle(xf)[:, chunk], img_k = xs_k . Wc[chunk]
  img = VitSumDequant(img_*)         (K = D s^2 exceeds every kernel's bound)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto

from . import numeric
from .llama import graph_with_initializers
from .llm_nodes import LLM_DOMAIN

F = 8                                  # the kernels' output shift
VP = "model.vision_model."
CONN = "model.connector.modality_projection.proj.weight"


@dataclass
class VitConfig:
    L: int                  # layers
    D: int                  # hidden size
    H: int                  # heads
    HD: int                 # head dim
    FF: int                 # MLP intermediate size
    eps: float
    P: int                  # patch size
    S: int                  # image size
    scale: int              # pixel-shuffle factor of the connector
    Dt: int                 # the text model's hidden size (connector output)

    @property
    def side(self) -> int:
        return self.S // self.P

    @property
    def N(self) -> int:
        return self.side ** 2

    @property
    def n_img(self) -> int:
        """Image tokens per tile (N / scale^2)."""
        return self.N // (self.scale ** 2)

    @classmethod
    def from_dict(cls, c: dict) -> "VitConfig":
        v = c["vision_config"]
        D, H = int(v["hidden_size"]), int(v["num_attention_heads"])
        return cls(L=int(v["num_hidden_layers"]), D=D, H=H, HD=D // H,
                   FF=int(v["intermediate_size"]), eps=float(v["layer_norm_eps"]),
                   P=int(v["patch_size"]), S=int(v["image_size"]),
                   scale=int(c["scale_factor"]), Dt=int(c["text_config"]["hidden_size"]))

    @classmethod
    def from_file(cls, path: str) -> "VitConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class VisionFormats:
    """Exponents per ``class@layer`` (vlm_study.py vision formats JSON; layer
    L = the post-LayerNorm and the connector)."""

    def __init__(self, d: dict, cfg: VitConfig):
        self.cfg = cfg
        self.exp = dict(d["exponents"])
        self.policy = d.get("policy", "")
        if int(d.get("pixel_exp", 0)) != 0:
            raise ValueError("vision formats: pixel_exp must be 0 (raw uint8 pixels)")

    @classmethod
    def from_file(cls, path: str, cfg: VitConfig) -> "VisionFormats":
        with open(path) as f:
            return cls(json.load(f), cfg)

    def get(self, cls_: str, layer: int, n: int) -> np.ndarray:
        v = self.exp.get(f"{cls_}@{layer}", F)
        return np.broadcast_to(np.asarray(v, np.int64), (n,)).copy()

    def pv(self, li: int) -> np.ndarray:
        c = self.cfg
        fp = self.get("p", li, c.H)
        return (fp[:, None] + self.get("vc", li, c.D).reshape(c.H, c.HD) - F).reshape(-1)


def _compact(e):
    e = np.asarray(e, np.int64).reshape(-1)
    return int(e[0]) if (e == e[0]).all() else [int(v) for v in e]


def patch_weight(W: Dict[str, np.ndarray], cfg: VitConfig) -> np.ndarray:
    """[3*P*P][D]: the patch embedding as a MatMul B over (channel, row,
    column) of a patch, x 2/255 (the pixel normalisation), float32."""
    Wp = W[VP + "embeddings.patch_embedding.weight"].astype(np.float64).reshape(cfg.D, -1)
    return np.ascontiguousarray((Wp * (2.0 / 255.0)).astype(np.float32).T)


def patch_bias_table(W: Dict[str, np.ndarray]) -> np.ndarray:
    """float32(b - sum_i W[c][i] (left to right) + pos[t][c]) [N][D]."""
    Wp = W[VP + "embeddings.patch_embedding.weight"].astype(np.float64)
    Wp = Wp.reshape(Wp.shape[0], -1)
    b = W[VP + "embeddings.patch_embedding.bias"].astype(np.float64) - np.cumsum(Wp, axis=1)[:, -1]
    pos = W[VP + "embeddings.position_embedding.weight"].astype(np.float64)
    return (b[None, :] + pos).astype(np.float32)


def patches(rgb: np.ndarray, P: int) -> np.ndarray:
    """uint8 [S][S][3] -> [N][3*P*P] (row-major patch grid; channel, row,
    column within a patch) — the vision entry's input (llm_image in C)."""
    S = rgb.shape[0]
    n = S // P
    x = np.asarray(rgb).reshape(n, P, n, P, 3).transpose(0, 2, 4, 1, 3)
    return np.ascontiguousarray(x.reshape(n * n, 3 * P * P))


class VitFrontend:
    """Builds the ``vision`` entry.  ``weights``: the checkpoint's tensors by
    Hugging Face name (float32); ``image_state``: the float32 host state the
    image features go to (read by the text model's prefill LlmEmbed)."""

    def __init__(self, cfg: VitConfig, weights: Dict[str, np.ndarray], formats: VisionFormats,
                 name: str = "vit", image_state: str = "vlm.img", conn_k: int = 4096,
                 qk_kw: Optional[int] = None, pv_kw: Optional[int] = None,
                 attn_split: int = 1):
        """``attn_split`` R > 1 splits every head's softmax and P.V by query
        rows into R parts (TACTICS_PLAN §4.4): the CPU can then start the
        next ConvKernel call between the parts (--plan's order search)."""
        self.cfg = cfg
        if attn_split < 1 or cfg.N % (attn_split * 16):
            raise ValueError(f"attn_split {attn_split}: {cfg.N} query rows not in parts of 16")
        self.attn_split = int(attn_split)
        self.W = weights
        self.fmt = formats
        self.name = name
        self.image_state = image_state
        self.conn_k = int(conn_k)
        C = cfg.N
        if pv_kw is None:
            pv_kw = next(k for k in (4, 2, 1) if C % (16 * k) == 0)
        self.pv_kw = int(pv_kw)
        self.qk_kw = qk_kw
        if (cfg.D * cfg.scale ** 2) % self.conn_k:
            raise ValueError(f"connector K {cfg.D * cfg.scale ** 2} % chunk {self.conn_k}")

    def choose_qk_kw(self) -> int:
        """Kernel width of the q.K^T calls' q image: the cheapest by the conv
        cost model at N keys (head_dim % 16 kw == 0)."""
        if self.qk_kw is not None:
            return int(self.qk_kw)
        from .cost_model import conv_cycles
        c = self.cfg
        M, keys = c.N, c.N
        best = None
        for kw in (1, 2, 4):
            if c.HD % (16 * kw):
                continue
            for ow in [d for d in range(8, 65) if M % d == 0] or [M]:
                cyc = conv_cycles(in_ch=c.HD // kw, out_ch=keys, in_h=M // ow, in_w=kw * ow,
                                  oh=M // ow, ow=ow, kh=1, kw=kw, sw=kw)["total"]
                if best is None or cyc < best[0]:
                    best = (cyc, kw)
        return best[1]

    # ---- builder helpers (src/llama.py's conventions) ------------------------ #
    def _new(self):
        self._nodes: List[onnx.NodeProto] = []
        self._vi: Dict[str, onnx.ValueInfoProto] = {}
        self._inits: Dict[str, np.ndarray] = {}      # converted in _model
        self._meta = numeric.empty()

    def _t(self, name, shape, elem=TensorProto.FLOAT, exp=None, host=None, state=False):
        if name not in self._vi:
            self._vi[name] = oh.make_tensor_value_info(name, elem, list(shape))
        if exp is not None:
            self._meta["exp"][name] = _compact(exp)
        if host is not None:
            self._meta["host"][name] = host
        if state and name not in self._meta["state"]:
            self._meta["state"].append(name)
        return name

    def _init(self, name, arr):
        if name not in self._inits:
            self._inits[name] = arr
        return name

    def _node(self, op, ins, outs, name, domain="", **attrs):
        self._nodes.append(oh.make_node(op, ins, outs, name=name, domain=domain, **attrs))

    def _matmul(self, x, wname, W, T, y, exp, name):
        self._init(wname, W)
        self._t(y, [T, W.shape[1]], exp=exp)
        self._node("MatMul", [x, wname], [y], name)
        return y

    def _vec(self, name, arr):
        return self._init(name, np.asarray(arr, np.float32).reshape(-1))

    def _ln(self, h, li, which, out, exp, gname, bname):
        c = self.cfg
        g = self._vec(f"v.{which}.g" if li is None else f"v.l{li}.{which}.g", self.W[gname])
        b = self._vec(f"v.{which}.b" if li is None else f"v.l{li}.{which}.b", self.W[bname])
        self._t(out, [c.N, c.D], exp=exp)
        self._node("VitLayerNorm", [h, g, b], [out], out.replace("vision.", "vision.ln_"),
                   domain=LLM_DOMAIN, axi_eps=repr(float(c.eps)))
        return out

    # ---- the entry ----------------------------------------------------------- #
    def entry(self, output: bool = False) -> onnx.ModelProto:
        """The vision entry; ``output``: the image features as a graph output
        ``vision.image`` instead of the state (tests, the board's model suite)."""
        c, fm, W = self.cfg, self.fmt, self.W
        N, D, H, HD, L = c.N, c.D, c.H, c.HD, c.L
        self._new()
        pin = self._t("vision.patches", [N, 3 * c.P * c.P], exp=0)
        pe = self._matmul(pin, "w.v.pe", patch_weight(W, c), N, "vision.pe", fm.get("pe", 0, D),
                          "vision.patch_embed")
        self._init("vision.b0", patch_bias_table(W))
        h = self._t("vision.h0", [N, D], host="f32")
        self._node("VitEmbedAdd", [pe, "vision.b0"], [h], "vision.embed", domain=LLM_DOMAIN)
        # the K / V caches: raw integers, one pair for every layer
        kc = self._t("vit.k", [N, D], exp=0, state=True)
        vc = self._t("vit.v", [N, D], exp=0, state=True)
        self._meta["layout"][kc] = [H, HD]
        self._meta["layout"][vc] = [H, HD] + ([self.pv_kw] if self.pv_kw > 1 else [])
        self._init(kc, np.zeros((N, D), np.float32))
        self._init(vc, np.zeros((N, D), np.float32))
        kw = self.choose_qk_kw()
        common = dict(num_heads=H, num_kv_heads=H, head_dim=HD, key_quantum=16 * self.pv_kw)
        for li in range(L):
            lw = f"{VP}encoder.layers.{li}."
            e = f"vision.l{li}"
            x = self._ln(h, li, "ln1", f"{e}.x", fm.get("x", li, D), lw + "layer_norm1.weight",
                         lw + "layer_norm1.bias")
            q0 = self._matmul(x, f"w.v.l{li}.q", W[lw + "self_attn.q_proj.weight"].T, N, f"{e}.q0",
                              fm.get("q0", li, D), f"{e}.q_proj")
            k0 = self._matmul(x, f"w.v.l{li}.k", W[lw + "self_attn.k_proj.weight"].T, N, f"{e}.k0",
                              fm.get("k0", li, D), f"{e}.k_proj")
            v0 = self._matmul(x, f"w.v.l{li}.v", W[lw + "self_attn.v_proj.weight"].T, N, f"{e}.v",
                              fm.get("v", li, D), f"{e}.v_proj")
            fq, fk, fp = fm.get("q", li, H), fm.get("k", li, H), fm.get("p", li, H)
            fs = fq + fk - F
            qx = self._t(f"{e}.qx", [H, HD, N], exp=0)
            self._node("VitAttnPrep", [q0, k0, v0, kc, vc,
                                       self._vec(f"v.l{li}.bq", W[lw + "self_attn.q_proj.bias"]),
                                       self._vec(f"v.l{li}.bk", W[lw + "self_attn.k_proj.bias"]),
                                       self._vec(f"v.l{li}.bv", W[lw + "self_attn.v_proj.bias"])],
                       [qx], f"{e}.attn_prep", domain=LLM_DOMAIN, num_heads=H, head_dim=HD,
                       qk_kw=kw, q_exp=[int(v) for v in fq], k_exp=[int(v) for v in fk],
                       v_exp=[int(v) for v in fm.get("vc", li, D)])
            R = self.attn_split
            s = [self._t(f"{e}.s{g}", [N, N], exp=0) for g in range(H)]
            if R == 1:
                p = [self._t(f"{e}.p{g}", [N, N], exp=0) for g in range(H)]
                o = [self._t(f"{e}.o{g}", [N, HD], exp=0) for g in range(H)]
            else:
                p = o = None
                pr = [[self._t(f"{e}.p{g}r{r}", [N // R, N], exp=0) for r in range(R)]
                      for g in range(H)]
                orr = [[self._t(f"{e}.o{g}r{r}", [N // R, HD], exp=0) for r in range(R)]
                       for g in range(H)]

            def qk(g, qx=qx, s=s, e=e):
                self._node("LlmAttnScores", [kc, qx], [s[g]], f"{e}.qk{g}", domain=LLM_DOMAIN,
                           group=g, qk_kw=kw, **common)

            def softmax(g, s=s, p=p, e=e, fs=fs, fp=fp):
                self._node("VitAttnSoftmax", [s[g]], [p[g]], f"{e}.softmax{g}", domain=LLM_DOMAIN,
                           head_dim=HD, s_exp=[int(fs[g])], p_exp=[int(fp[g])])

            def pv_(g, p=p, o=o, e=e):
                self._node("LlmAttnPV", [p[g], vc], [o[g]], f"{e}.pv{g}", domain=LLM_DOMAIN,
                           group=g, **common)
            # ConvKernel runs qk(g+1) under softmax(g); pv(g) then qk(g+2)
            # follow right away, so the CPU waits only for the short P.V to
            # free the lane (qk(g+1) after softmax(g) waited for the longer
            # qk before the next softmax)
            def softmax_r(g, r, s=s, pr=None if R == 1 else pr, e=e, fs=fs, fp=fp, R=R):
                self._node("VitAttnSoftmax", [s[g]], [pr[g][r]], f"{e}.softmax{g}r{r}",
                           domain=LLM_DOMAIN, head_dim=HD, s_exp=[int(fs[g])],
                           p_exp=[int(fp[g])], cols=[r * (N // R), N // R])

            def pv_r(g, r, pr=None if R == 1 else pr, orr=None if R == 1 else orr, e=e):
                self._node("LlmAttnPV", [pr[g][r], vc], [orr[g][r]], f"{e}.pv{g}r{r}",
                           domain=LLM_DOMAIN, group=g, **common)
            qk(0)
            if H > 1:
                qk(1)
            for g in range(H):
                if R == 1:
                    softmax(g)
                    pv_(g)
                else:
                    for r in range(R):
                        softmax_r(g, r)
                    for r in range(R):
                        pv_r(g, r)
                if g + 2 < H:
                    qk(g + 2)
            pv = self._t(f"{e}.pv", [N, D], exp=fm.pv(li))
            if R == 1:
                self._node("LlmAttnMerge", o, [pv], f"{e}.attn_merge", domain=LLM_DOMAIN,
                           num_heads=H, num_kv_heads=H, head_dim=HD)
            else:
                self._node("LlmAttnMerge", [t for g in range(H) for t in orr[g]], [pv],
                           f"{e}.attn_merge", domain=LLM_DOMAIN, num_heads=H, num_kv_heads=H,
                           head_dim=HD, row_splits=R)
            ov = self._matmul(pv, f"w.v.l{li}.o", W[lw + "self_attn.out_proj.weight"].T, N, f"{e}.o",
                              fm.get("o", li, D), f"{e}.o_proj")
            h1 = self._t(f"{e}.h1", [N, D], host="f32")
            self._node("VitResAdd", [h, ov, self._vec(f"v.l{li}.bo", W[lw + "self_attn.out_proj.bias"])],
                       [h1], f"{e}.res_attn", domain=LLM_DOMAIN)
            x2 = self._ln(h1, li, "ln2", f"{e}.x2", fm.get("x2", li, D), lw + "layer_norm2.weight",
                          lw + "layer_norm2.bias")
            fv = self._matmul(x2, f"w.v.l{li}.f1", W[lw + "mlp.fc1.weight"].T, N, f"{e}.f",
                              fm.get("f", li, c.FF), f"{e}.fc1")
            a = self._t(f"{e}.a", [N, c.FF], exp=fm.get("a", li, c.FF))
            self._node("VitGelu", [fv, self._vec(f"v.l{li}.b1", W[lw + "mlp.fc1.bias"])], [a],
                       f"{e}.gelu", domain=LLM_DOMAIN)
            d = self._matmul(a, f"w.v.l{li}.f2", W[lw + "mlp.fc2.weight"].T, N, f"{e}.d",
                             fm.get("d", li, D), f"{e}.fc2")
            h2 = self._t(f"{e}.h2", [N, D], host="f32")
            self._node("VitResAdd", [h1, d, self._vec(f"v.l{li}.b2", W[lw + "mlp.fc2.bias"])], [h2],
                       f"{e}.res_mlp", domain=LLM_DOMAIN)
            h = h2
        xf = self._ln(h, None, "post", "vision.xf", fm.get("xf", L, D),
                      VP + "post_layernorm.weight", VP + "post_layernorm.bias")
        K = D * c.scale ** 2
        fxs = fm.get("xs", L, K)
        Wc = W[CONN].T                                                   # [K][Dt]
        parts = []
        for i, k0 in enumerate(range(0, K, self.conn_k)):
            xs = self._t(f"vision.xs{i}", [c.n_img, self.conn_k], exp=fxs[k0:k0 + self.conn_k])
            self._node("VitPixelShuffle", [xf], [xs], f"vision.shuffle{i}", domain=LLM_DOMAIN,
                       scale=c.scale, col0=k0)
            parts.append(self._matmul(xs, f"w.v.conn.{i}", Wc[k0:k0 + self.conn_k], c.n_img,
                                      f"vision.img{i}", fm.get("img", L, c.Dt), f"vision.connector{i}"))
        img = (self._t("vision.image", [c.n_img, c.Dt], host="f32") if output else
               self._t(self.image_state, [c.n_img, c.Dt], host="f32", state=True))
        self._node("VitSumDequant", parts, [img], "vision.image_features", domain=LLM_DOMAIN)
        return self._model([pin], [img] if output else [])

    def _model(self, inputs, outputs):
        g = graph_with_initializers(
            self._nodes, f"{self.name}_vision",
            [self._vi[n] for n in inputs], [self._vi[n] for n in outputs], self._inits,
            value_info=[vi for n, vi in self._vi.items() if n not in inputs and n not in outputs])
        self._inits = {}
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/vit.py")
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key = numeric.METADATA_KEY
        p.value = numeric.to_metadata(self._meta)
        p = m.metadata_props.add()
        p.key = "axi.llm.entry"
        c = self.cfg
        p.value = json.dumps({"entry": "vision", "model": self.name, "layers": c.L, "hidden": c.D,
                              "heads": c.H, "head_dim": c.HD, "patch": c.P, "image": c.S,
                              "image_tokens": c.n_img, "pv_kw": self.pv_kw})
        return m


__all__ = ("VitConfig", "VisionFormats", "VitFrontend", "patch_weight", "patch_bias_table", "patches")
