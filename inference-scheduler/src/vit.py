# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

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
    a   = VitGelu(f, b1[, ba, sc])   d = a . W2    h = VitResAdd(h1, d, b2)
                                     (ba, sc where a is at 2^-8 on every channel:
                                     VectorOPKernel's activation unit runs it,
                                     policy pow2+p12+vgelu — _gelu_vop_rows)
  xf  = VitLayerNorm(h, post)
  per K chunk k: xs_k = VitPixelShuffle(xf)[:, chunk], img_k = xs_k . Wc[chunk]
  img = VitSumDequant(img_*)         (K = D s^2 exceeds every kernel's bound)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import onnx

from .graph_builder import (RES_BITS, ExponentTable, GraphBuilder, cheapest_qk_kw, group_order, pv_exp,
                            score_exp)
from .llm_nodes import LLM_DOMAIN

VP = "model.vision_model."
IEMBED_DEFAULT = False  # rowops: the patch-embedding add on the unit too (the position table rounded once at
                        # 2^-RES_BITS as an int32 stream; vlm_study's policy ...+rowops+iembed, RPU_PLAN §3.1)
CONN = "model.connector.modality_projection.proj.weight"

# Row pieces (VitFrontend ``pieces``, doc/plans/VLA_PLAN.md §4): the board's rates on the production
# bitstream b4e63457057a, in µs per MAC (a linear's rows on either kernel) or per element (the row ops,
# the host's embedding add) — only for the issue order of the pieces (_Pipe), never for a result.
PIECE_US_PER_MAC = {"conv": 1 / 123e3, "matmul": 1 / 27e3}     # (MatmulKernel: pieces given explicitly)
PIECE_US_PER_ELEM = {"rowops_ln": 1.0e-3, "rowops_res": 1.5e-3, "vop_gelu": 1.3e-3, "cpu": 2.0e-3,
                     "vop_smx": 1.73e-3}
# the attention's calls (1024 x 1024 x 64 on the board): q.K^T on ConvKernel 1.87 ms, P.V 1.07 ms,
# the row-mode softmax 1.81 ms
ATTN_US_PER_MAC = {"qk_conv": 1 / 35.9e3, "pv_conv": 1 / 62.7e3}


def default_pieces(n: int) -> Optional[tuple]:
    """Row pieces of the encoder's ``n`` tokens: two halves on ConvKernel, in
    multiples of 16 rows (1024 -> 512 + 512); None below 256 rows.  On the
    board two pieces and four take the same time (VLA_PLAN §4: 1314 ms an
    image either way), and a MatmulKernel piece loses: beside ConvKernel its
    calls run ~1.9x slower than alone (1826 ms)."""
    if n < 256 or n % 16:
        return None
    half = n // 32 * 16
    return ((half, "conv"), (n - half, "conv"))


class _Pipe:
    """The issue order of a row-pieced stretch of the graph: each node is added
    with its lane, its estimated time and its input / output tensors, and
    flush() emits them in the order of a greedy list schedule — the earliest
    estimated start first (each lane serial, inputs written), ties in the order
    added.  The generated inference_run() issues in list order and waits in
    order, so every lane gets its next piece as soon as it frees."""

    def __init__(self):
        self.tasks: list = []

    def add(self, lane: str, us: float, ins, outs, emit) -> None:
        self.tasks.append((lane, float(us), list(ins), list(outs), emit))

    def flush(self) -> None:
        here = {o for t in self.tasks for o in t[3]}
        ready: Dict[str, float] = {}
        free: Dict[str, float] = {}
        todo = list(range(len(self.tasks)))
        while todo:
            best = None
            for j in todo:
                lane, _us, ins, _outs, _emit = self.tasks[j]
                if any(x in here and x not in ready for x in ins):
                    continue
                t0 = max([free.get(lane, 0.0)] + [ready.get(x, 0.0) for x in ins])
                if best is None or t0 < best[0]:
                    best = (t0, j)
            t0, j = best
            lane, us, _ins, outs, emit = self.tasks[j]
            free[lane] = t0 + us
            for o in outs:
                ready[o] = t0 + us
            emit()
            todo.remove(j)
        self.tasks = []


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


class VisionFormats(ExponentTable):
    """Exponents per ``class@layer`` (vlm_study.py vision formats JSON; layer
    L = the post-LayerNorm and the connector)."""

    def __init__(self, d: dict, cfg: VitConfig):
        super().__init__(d)
        self.cfg = cfg
        if int(d.get("pixel_exp", 0)) != 0:
            raise ValueError("vision formats: pixel_exp must be 0 (raw uint8 pixels)")

    def get(self, cls_: str, layer: int, n: int) -> np.ndarray:
        return self.lookup(f"{cls_}@{layer}", n)

    def pv(self, li: int) -> np.ndarray:
        c = self.cfg
        return pv_exp(self.get("p", li, c.H), self.get("vc", li, c.D), c.H, c.HD)


def _gelu_vop_rows(b, ef, ea, n):
    """The two constant rows of GELU(f + b) on VectorOPKernel (doc/plans/OFFLOAD_PLAN.md
    §2.1; vlm_study.vop_gelu_b): ba = round_half_even(b * 2^f) + 2^(f - 9) (f > 8: half
    a Q8.8 LSB at f's exponent, so the MUL's floor rounds half up), sc = 2^(16 - f) —
    both raw int16 rows, returned as float32 values raw / 2^8 (the default
    exponent).  None when the output is not at 2^-8 on every channel or a row does
    not fit int16 (the host VitGelu runs it)."""
    ef = np.broadcast_to(np.asarray(ef, np.int64), (n,))
    if not np.all(np.broadcast_to(np.asarray(ea, np.int64), (n,)) == 8):
        return None
    p = np.power(2.0, ef.astype(np.float64))
    ba = np.round(np.asarray(b, np.float64) * p) + np.where(ef > 8, p / 512.0, 0.0)
    sc = np.power(2.0, 16.0 - ef)
    if not (np.all(np.abs(ba) < 32768) and np.all((sc < 32768) & (sc >= 1))):
        return None
    return (ba / 256.0).astype(np.float32), (sc / 256.0).astype(np.float32)


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


class VitFrontend(GraphBuilder):
    """Builds the ``vision`` entry.  ``weights``: the checkpoint's tensors by
    Hugging Face name (float32); ``image_state``: the float32 host state the
    image features go to (read by the text model's prefill LlmEmbed)."""

    PRODUCER = "inference-scheduler/src/vit.py"

    def __init__(self, cfg: VitConfig, weights: Dict[str, np.ndarray], formats: VisionFormats,
                 name: str = "vit", image_state: str = "vlm.img", conn_k: int = 4096,
                 qk_kw: Optional[int] = None, pv_kw: Optional[int] = None,
                 attn_split: int = 1, vsmx: bool = False, rowops: bool = False,
                 iembed: Optional[bool] = None, query_major: Optional[bool] = None,
                 pieces=None):
        """``attn_split`` R > 1 splits every head's softmax and P.V by query
        rows into R parts (TACTICS_PLAN §4.4): the CPU can then start the
        next ConvKernel call between the parts (--plan's order search).
        ``vsmx`` (policy pow2+p12+vgelu+vsmx, doc/plans/SOFTMAX_PLAN.md): the
        softmax on VectorOPKernel's softmax unit (VitAttnSoftmax vsmx = 1).
        ``query_major`` (default: with vsmx and rowops): the scores conv with
        the roles of q and k swapped (VitAttnPrep query_major = 1: q in the
        cache rows, k in the image), s [queries][keys], and the unit's row
        mode over it — the same P, the row mode twice as fast as the column
        mode's two-beat reads (doc/plans/VLA_PLAN.md §4).
        ``pieces`` (default with rowops: default_pieces(N); False: none): the
        rows [(rows, "conv" | "matmul"), ...] the linears, LayerNorms, GELUs
        and residual adds run in, each linear piece pinned to its kernel
        (axi.numeric ``engine``; the pieces of a weight read one image of it)
        and the pieces issued as a pipeline (_Pipe) — MatmulKernel takes a
        share of every linear, and the row ops of one piece run under the
        linears of the next; the attention reads q / k / v whole (in-place
        Concats) and its output by row views, its calls issued as a pipeline
        too.  Same results."""
        self.cfg = cfg
        self.vsmx = bool(vsmx)
        # rowops (policy ...+rowops, doc/plans/HOSTOPS_PLAN.md §6): the residual stream
        # as the int32 DMA tensor at 2^-RES_BITS, the residual adds and LayerNorms on
        # the row-ops unit (src/rowops_nodes.py)
        self.rowops = bool(rowops)
        self.iembed = (self.rowops and IEMBED_DEFAULT) if iembed is None else bool(iembed)
        if self.iembed and not self.rowops:
            raise ValueError("iembed needs rowops")
        self.query_major = (self.vsmx and self.rowops) if query_major is None else bool(query_major)
        if self.query_major and not (self.vsmx and self.rowops):
            raise ValueError("query_major needs vsmx and rowops")
        if pieces is None:
            pieces = default_pieces(cfg.N) if self.rowops else None
        self.pieces = tuple((int(r), str(k)) for r, k in pieces) if pieces else None
        if self.pieces:
            if not self.rowops:
                raise ValueError("pieces need rowops (the residual stream in DMA)")
            if sum(r for r, _k in self.pieces) != cfg.N or any(r % 16 or k not in ("conv", "matmul")
                                                                for r, k in self.pieces):
                raise ValueError(f"pieces {self.pieces}: rows in multiples of 16 adding up to {cfg.N}, "
                                 "on 'conv' or 'matmul'")
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

    def _vsmx_attr(self) -> dict:
        return (dict(vsmx=1) if self.vsmx else {}) | (dict(query_major=1) if self.query_major else {})

    def choose_qk_kw(self) -> int:
        """Kernel width of the q.K^T calls' q image: the cheapest by the conv
        cost model at N keys (head_dim % 16 kw == 0)."""
        if self.qk_kw is not None:
            return int(self.qk_kw)
        return cheapest_qk_kw(self.cfg.HD, self.cfg.N, self.cfg.N)

    # ---- builder helpers (GraphBuilder's and these) ---------------------------- #
    def _res(self, name, rows=None):
        """A residual-stream tensor [N][D] (or ``rows`` of it): with rowops the int32
        DMA tensor, else float32 host."""
        return self._residual(name, self.cfg.N if rows is None else rows, self.cfg.D, self.rowops)

    def _vec(self, name, arr):
        return self._init(name, np.asarray(arr, np.float32).reshape(-1))

    def _ln(self, h, li, which, out, exp, gname, bname, rows=None):
        c = self.cfg
        g = self._vec(f"v.{which}.g" if li is None else f"v.l{li}.{which}.g", self.W[gname])
        b = self._vec(f"v.{which}.b" if li is None else f"v.l{li}.{which}.b", self.W[bname])
        self._t(out, [c.N if rows is None else rows, c.D], exp=exp)
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
        h = None if self.pieces else self._res("vision.h0")
        if self.pieces:
            pipe = _Pipe()
            hs = self._embed_pieces(pipe, pe)
        elif self.iembed:                 # the table as int32 words at 2^-RES_BITS (rounded once), one ADDX
            from .rowops_nodes import EmbedAddRoNode
            b0q = np.clip(np.rint(patch_bias_table(W).astype(np.float64) * 2.0 ** RES_BITS),
                          -2147483648.0, 2147483647.0)
            self._init("vision.b0i", EmbedAddRoNode.table_words(b0q))
            self._node("VitEmbedAdd", [pe, "vision.b0i"], [h], "vision.embed", domain=LLM_DOMAIN, rowops=1)
        else:
            self._init("vision.b0", patch_bias_table(W))
            self._node("VitEmbedAdd", [pe, "vision.b0"], [h], "vision.embed", domain=LLM_DOMAIN)
        if self.rowops:
            for key in ("o", "d"):
                for li in range(L):
                    fd = np.asarray(fm.get(key, li, D), np.int64)
                    if (fd > RES_BITS).any():
                        raise ValueError(f"rowops: the {key} exponents of layer {li} exceed 2^-{RES_BITS} "
                                         f"(max {int(fd.max())}): the residual add would not be exact")
        # the K / V caches: raw integers, one pair for every layer
        kc = self._t("vit.k", [N, D], exp=0, state=True)
        vc = self._t("vit.v", [N, D], exp=0, state=True)
        self._meta["layout"][kc] = [H, HD]
        self._meta["layout"][vc] = [H, HD] + ([self.pv_kw] if self.pv_kw > 1 else [])
        self._init(kc, np.zeros((N, D), np.float32))
        self._init(vc, np.zeros((N, D), np.float32))
        # rowops (HOSTOPS_PLAN §10, phase 4b): the attention prep on the unit, which
        # writes the q image at kw = 1 and stages q / v in DMA states of the entry
        kw = 1 if self.rowops else self.choose_qk_kw()
        prep_extra, prep_attrs = [], {}
        if self.rowops:
            qs = self._t("vision.ro.qs", [N, D], exp=0, state=True)
            vs = self._t("vision.ro.vs", [N, D], exp=0, state=True)
            self._init(qs, np.zeros((N, D), np.float32))
            self._init(vs, np.zeros((N, D), np.float32))
            prep_extra, prep_attrs = [qs, vs], dict(rowops=1)
            if self.query_major:
                prep_attrs["query_major"] = 1
        common = dict(num_heads=H, num_kv_heads=H, head_dim=HD, key_quantum=16 * self.pv_kw)
        for li in range(L if not self.pieces else 0):
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
            pv = self._attention(li, e, lw, q0, k0, v0, kc, vc, kw, prep_extra, prep_attrs,
                                 common)
            ov = self._matmul(pv, f"w.v.l{li}.o", W[lw + "self_attn.out_proj.weight"].T, N, f"{e}.o",
                              fm.get("o", li, D), f"{e}.o_proj")
            h1 = self._res(f"{e}.h1")
            self._node("VitResAdd", [h, ov, self._vec(f"v.l{li}.bo", W[lw + "self_attn.out_proj.bias"])],
                       [h1], f"{e}.res_attn", domain=LLM_DOMAIN)
            x2 = self._ln(h1, li, "ln2", f"{e}.x2", fm.get("x2", li, D), lw + "layer_norm2.weight",
                          lw + "layer_norm2.bias")
            fv = self._matmul(x2, f"w.v.l{li}.f1", W[lw + "mlp.fc1.weight"].T, N, f"{e}.f",
                              fm.get("f", li, c.FF), f"{e}.fc1")
            a = self._t(f"{e}.a", [N, c.FF], exp=fm.get("a", li, c.FF))
            b1 = self._vec(f"v.l{li}.b1", W[lw + "mlp.fc1.bias"])
            vop = _gelu_vop_rows(self._inits[b1], fm.get("f", li, c.FF), fm.get("a", li, c.FF), c.FF)
            extra = [] if vop is None else [self._vec(f"v.l{li}.gelu_ba", vop[0]),
                                            self._vec(f"v.l{li}.gelu_sc", vop[1])]
            self._node("VitGelu", [fv, b1] + extra, [a], f"{e}.gelu", domain=LLM_DOMAIN)
            d = self._matmul(a, f"w.v.l{li}.f2", W[lw + "mlp.fc2.weight"].T, N, f"{e}.d",
                             fm.get("d", li, D), f"{e}.fc2")
            h2 = self._res(f"{e}.h2")
            self._node("VitResAdd", [h1, d, self._vec(f"v.l{li}.b2", W[lw + "mlp.fc2.bias"])], [h2],
                       f"{e}.res_mlp", domain=LLM_DOMAIN)
            h = h2
        if self.pieces:
            xf = self._layers_pieces(pipe, hs, kc, vc, kw, prep_extra, prep_attrs, common)
        else:
            xf = self._ln(h, None, "post", "vision.xf", fm.get("xf", L, D),
                          VP + "post_layernorm.weight", VP + "post_layernorm.bias")
        K = D * c.scale ** 2
        fxs = fm.get("xs", L, K)
        Wc = W[CONN].T                                                   # [K][Dt]
        parts = []
        for i, k0 in enumerate(range(0, K, self.conn_k)):
            xs = self._t(f"vision.xs{i}", [c.n_img, self.conn_k], exp=fxs[k0:k0 + self.conn_k])
            self._node("VitPixelShuffle", [xf], [xs], f"vision.shuffle{i}", domain=LLM_DOMAIN,
                       scale=c.scale, col0=k0, **(dict(rowops=1) if self.rowops else {}))
            parts.append(self._matmul(xs, f"w.v.conn.{i}", Wc[k0:k0 + self.conn_k], c.n_img,
                                      f"vision.img{i}", fm.get("img", L, c.Dt), f"vision.connector{i}"))
        if not self.rowops:
            img = (self._t("vision.image", [c.n_img, c.Dt], host="f32") if output else
                   self._t(self.image_state, [c.n_img, c.Dt], host="f32", state=True))
            self._node("VitSumDequant", parts, [img], "vision.image_features", domain=LLM_DOMAIN)
            return self._model([pin], [img] if output else [])
        # rowops (RPU_PLAN §3.1): the parts summed exactly on the unit as an int32 row at the
        # finest part exponent (ADDX, per-column shifts), then DEQUANT into the float32 DMA
        # state (the text model's LlmEmbed reads it in place) or, as an output, a DMA row the
        # host copies out — the host op's value bit for bit
        fr = int(max(int(np.max(fm.get("img", L, c.Dt))), RES_BITS))
        acc = parts[0]
        for i, part in enumerate(parts[1:]):
            s = self._t(f"vision.sum{i}", [c.n_img, c.Dt], exp=fr, dma="i32")
            if i == 0:
                self._node("VitResAdd", [acc, part, self._vec("v.conn.zero", np.zeros(c.Dt, np.float32))], [s],
                           f"vision.sum{i}", domain=LLM_DOMAIN, rowops=1)
            else:
                self._node("LlmResAdd", [acc, part], [s], f"vision.sum{i}", domain=LLM_DOMAIN)
            acc = s
        if output:
            imgd = self._t("vision.image_d", [c.n_img, c.Dt], dma="f32")
            img = self._t("vision.image", [c.n_img, c.Dt], host="f32")
            self._node("LlmDequant", [acc], [imgd], "vision.image_features", domain=LLM_DOMAIN)
            self._node("LlmToHost", [imgd], [img], "vision.image_out", domain=LLM_DOMAIN)
            return self._model([pin], [img])
        img = self._t(self.image_state, [c.n_img, c.Dt], dma="f32", state=True)
        self._node("LlmDequant", [acc], [img], "vision.image_features", domain=LLM_DOMAIN)
        return self._model([pin], [])

    # ---- row pieces (``pieces``) ------------------------------------------------ #
    def _rows(self):
        """[(piece, first row, rows, engine)]."""
        out, r0 = [], 0
        for i, (n, eng) in enumerate(self.pieces):
            out.append((i, r0, n, eng))
            r0 += n
        return out

    def _slice_rows(self, x, r0, n, y, exp):
        """y = x[r0 : r0 + n] — a view (the scheduler's Slice views)."""
        s, e = f"vision.rows{r0}.s", f"vision.rows{r0}.e"
        self._int_init(s, np.array([r0], np.int64))
        self._int_init(e, np.array([r0 + n], np.int64))
        self._int_init("vision.axis0", np.array([0], np.int64))
        self._t(y, [n, self.cfg.D], exp=exp)
        self._node("Slice", [x, s, e, "vision.axis0"], [y], y.replace("vision.", "vision.slice_"))
        return y

    def _concat_rows(self, parts, y, cols, exp):
        """y = the pieces one after the other (a Concat done in place by the
        pieces' kernels, CodeGenerator._merge_in_place)."""
        self._t(y, [self.cfg.N, cols], exp=exp)
        self._node("Concat", list(parts), [y], y.replace("vision.", "vision.cat_"), axis=0)
        return y

    def _mm_piece(self, pipe, x, wname, W, n, y, exp, name, eng):
        self._init(wname, W)
        self._t(y, [n, W.shape[1]], exp=exp)
        self._meta["engine"][y] = eng
        pipe.add(eng, n * W.shape[0] * W.shape[1] * PIECE_US_PER_MAC[eng], [x], [y],
                 lambda: self._node("MatMul", [x, wname], [y], name))
        return y

    def _ln_piece(self, pipe, h, li, which, out, exp, gname, bname, n):
        pipe.add("rowops", n * self.cfg.D * PIECE_US_PER_ELEM["rowops_ln"], [h], [out],
                 lambda: self._ln(h, li, which, out, exp, gname, bname, rows=n))
        return out

    def _res_piece(self, pipe, h, d, bname, b, y, name, n):
        self._res(y, n)
        bv = self._vec(bname, b)
        pipe.add("rowops", n * self.cfg.D * PIECE_US_PER_ELEM["rowops_res"], [h, d], [y],
                 lambda: self._node("VitResAdd", [h, d, bv], [y], name, domain=LLM_DOMAIN))
        return y

    def _embed_pieces(self, pipe, pe):
        """h_i = VitEmbedAdd(pe[rows of piece i], the table's rows): the residual
        stream in pieces from the start (int32 DMA rows are never sliced)."""
        c, fm, W = self.cfg, self.fmt, self.W
        table = patch_bias_table(W)
        hs = []
        for i, r0, n, _eng in self._rows():
            pe_i = self._slice_rows(pe, r0, n, f"vision.pe.p{i}", fm.get("pe", 0, c.D))
            h = self._res(f"vision.h0.p{i}", n)
            if self.iembed:
                from .rowops_nodes import EmbedAddRoNode
                b0q = np.clip(np.rint(table[r0:r0 + n].astype(np.float64) * 2.0 ** RES_BITS),
                              -2147483648.0, 2147483647.0)
                tab = self._init(f"vision.b0i.p{i}", EmbedAddRoNode.table_words(b0q))
                lane, attrs = "rowops", dict(rowops=1)
            else:
                tab = self._init(f"vision.b0.p{i}", table[r0:r0 + n])
                lane, attrs = "cpu", {}
            pipe.add(lane, n * c.D * PIECE_US_PER_ELEM["rowops_res" if self.iembed else "cpu"], [pe_i], [h],
                     lambda pe_i=pe_i, tab=tab, h=h, i=i, attrs=attrs:
                     self._node("VitEmbedAdd", [pe_i, tab], [h], f"vision.embed.p{i}", domain=LLM_DOMAIN, **attrs))
            hs.append(h)
        return hs

    def _layers_pieces(self, pipe, hs, kc, vc, kw, prep_extra, prep_attrs, common):
        """The encoder layers over the row pieces ``hs`` (``pipe`` holds the
        embedding's), then the post LayerNorm -> xf [N][D] (in place)."""
        c, fm, W = self.cfg, self.fmt, self.W
        D, FF, L = c.D, c.FF, c.L
        rows = self._rows()
        for li in range(L):
            lw = f"{VP}encoder.layers.{li}."
            e = f"vision.l{li}"
            qkv = {"q0": [], "k0": [], "v": []}
            for i, _r0, n, eng in rows:
                x = self._ln_piece(pipe, hs[i], li, "ln1", f"{e}.x.p{i}", fm.get("x", li, D),
                                   lw + "layer_norm1.weight", lw + "layer_norm1.bias", n)
                for key, w, proj in (("q0", "q", "q_proj"), ("k0", "k", "k_proj"), ("v", "v", "v_proj")):
                    qkv[key].append(self._mm_piece(pipe, x, f"w.v.l{li}.{w}", W[lw + f"self_attn.{proj}.weight"].T,
                                                   n, f"{e}.{key}.p{i}", fm.get(key, li, D), f"{e}.{proj}.p{i}",
                                                   eng))
            pipe.flush()
            q0, k0, v0 = (self._concat_rows(qkv[key], f"{e}.{key}", D, fm.get(key, li, D))
                          for key in ("q0", "k0", "v"))
            pv = self._attention(li, e, lw, q0, k0, v0, kc, vc, kw, prep_extra, prep_attrs, common)
            b1 = self._vec(f"v.l{li}.b1", W[lw + "mlp.fc1.bias"])
            vop = _gelu_vop_rows(self._inits[b1], fm.get("f", li, FF), fm.get("a", li, FF), FF)
            extra = [] if vop is None else [self._vec(f"v.l{li}.gelu_ba", vop[0]),
                                            self._vec(f"v.l{li}.gelu_sc", vop[1])]
            for i, r0, n, eng in rows:
                pv_i = self._slice_rows(pv, r0, n, f"{e}.pv.p{i}", fm.pv(li))
                o = self._mm_piece(pipe, pv_i, f"w.v.l{li}.o", W[lw + "self_attn.out_proj.weight"].T, n,
                                   f"{e}.o.p{i}", fm.get("o", li, D), f"{e}.o_proj.p{i}", eng)
                h1 = self._res_piece(pipe, hs[i], o, f"v.l{li}.bo", W[lw + "self_attn.out_proj.bias"],
                                     f"{e}.h1.p{i}", f"{e}.res_attn.p{i}", n)
                x2 = self._ln_piece(pipe, h1, li, "ln2", f"{e}.x2.p{i}", fm.get("x2", li, D),
                                    lw + "layer_norm2.weight", lw + "layer_norm2.bias", n)
                f = self._mm_piece(pipe, x2, f"w.v.l{li}.f1", W[lw + "mlp.fc1.weight"].T, n, f"{e}.f.p{i}",
                                   fm.get("f", li, FF), f"{e}.fc1.p{i}", eng)
                a = self._t(f"{e}.a.p{i}", [n, FF], exp=fm.get("a", li, FF))
                pipe.add("vectorop" if vop is not None else "cpu", n * FF * PIECE_US_PER_ELEM["vop_gelu"], [f], [a],
                         lambda f=f, a=a, i=i, b1=b1, extra=extra, e=e:    # flushed in the next layer
                         self._node("VitGelu", [f, b1] + extra, [a], f"{e}.gelu.p{i}", domain=LLM_DOMAIN))
                d = self._mm_piece(pipe, a, f"w.v.l{li}.f2", W[lw + "mlp.fc2.weight"].T, n, f"{e}.d.p{i}",
                                   fm.get("d", li, D), f"{e}.fc2.p{i}", eng)
                hs[i] = self._res_piece(pipe, h1, d, f"v.l{li}.b2", W[lw + "mlp.fc2.bias"],
                                        f"{e}.h2.p{i}", f"{e}.res_mlp.p{i}", n)
        xs = [self._ln_piece(pipe, hs[i], None, "post", f"vision.xf.p{i}", fm.get("xf", L, D),
                             VP + "post_layernorm.weight", VP + "post_layernorm.bias", n)
              for i, _r0, n, _eng in rows]
        pipe.flush()
        return self._concat_rows(xs, "vision.xf", D, fm.get("xf", L, D))

    def _attention(self, li, e, lw, q0, k0, v0, kc, vc, kw, prep_extra, prep_attrs, common):
        """Layer ``li``'s attention: VitAttnPrep of q0 / k0 / v0 (the K / V caches, the
        q.K^T image), per head q.K^T, the softmax and P.V, the merge -> pv [N][D]."""
        c, fm, W = self.cfg, self.fmt, self.W
        N, D, H, HD = c.N, c.D, c.H, c.HD
        fq, fk, fp = fm.get("q", li, H), fm.get("k", li, H), fm.get("p", li, H)
        fs = score_exp(fq, fk, H)
        qx = self._t(f"{e}.qx", [H, HD, N], exp=0)
        self._node("VitAttnPrep", [q0, k0, v0, kc, vc,
                                   self._vec(f"v.l{li}.bq", W[lw + "self_attn.q_proj.bias"]),
                                   self._vec(f"v.l{li}.bk", W[lw + "self_attn.k_proj.bias"]),
                                   self._vec(f"v.l{li}.bv", W[lw + "self_attn.v_proj.bias"])] + prep_extra,
                   [qx], f"{e}.attn_prep", domain=LLM_DOMAIN, num_heads=H, head_dim=HD,
                   qk_kw=kw, q_exp=[int(v) for v in fq], k_exp=[int(v) for v in fk],
                   v_exp=[int(v) for v in fm.get("vc", li, D)], **prep_attrs)
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
                       head_dim=HD, s_exp=[int(fs[g])], p_exp=[int(fp[g])], **self._vsmx_attr())

        def pv_(g, p=p, o=o, e=e):
            self._node("LlmAttnPV", [p[g], vc], [o[g]], f"{e}.pv{g}", domain=LLM_DOMAIN,
                       group=g, **common)
        def softmax_r(g, r, s=s, pr=None if R == 1 else pr, e=e, fs=fs, fp=fp, R=R):
            self._node("VitAttnSoftmax", [s[g]], [pr[g][r]], f"{e}.softmax{g}r{r}",
                       domain=LLM_DOMAIN, head_dim=HD, s_exp=[int(fs[g])],
                       p_exp=[int(fp[g])], cols=[r * (N // R), N // R], **self._vsmx_attr())

        def pv_r(g, r, pr=None if R == 1 else pr, orr=None if R == 1 else orr, e=e):
            self._node("LlmAttnPV", [pr[g][r], vc], [orr[g][r]], f"{e}.pv{g}r{r}",
                       domain=LLM_DOMAIN, group=g, **common)
        if self.pieces and R == 1:
            # the heads as a pipeline (_Pipe): q.K^T and P.V on ConvKernel, the softmax
            pipe, macs = _Pipe(), N * N * HD
            for g in range(H):
                pipe.add("conv", macs * ATTN_US_PER_MAC["qk_conv"], [qx], [s[g]], lambda g=g: qk(g))
                pipe.add("vectorop", N * N * PIECE_US_PER_ELEM["vop_smx"], [s[g]], [p[g]],
                         lambda g=g: softmax(g))
                pipe.add("conv", macs * ATTN_US_PER_MAC["pv_conv"], [p[g]], [o[g]], lambda g=g: pv_(g))
            pipe.flush()
        else:
            # graph_builder.group_order "ahead2": ConvKernel runs q.K^T g+1 under
            # softmax g, so the CPU waits only for the short P.V to free the lane
            stage = {"qk": qk, "softmax": softmax, "pv": pv_} if R == 1 else \
                {"qk": qk, "softmax": lambda g: [softmax_r(g, r) for r in range(R)],
                 "pv": lambda g: [pv_r(g, r) for r in range(R)]}
            for st, g in group_order(H, "ahead2"):
                stage[st](g)
        pv = self._t(f"{e}.pv", [N, D], exp=fm.pv(li))
        if R == 1:
            self._node("LlmAttnMerge", o, [pv], f"{e}.attn_merge", domain=LLM_DOMAIN,
                       num_heads=H, num_kv_heads=H, head_dim=HD)
        else:
            self._node("LlmAttnMerge", [t for g in range(H) for t in orr[g]], [pv],
                       f"{e}.attn_merge", domain=LLM_DOMAIN, num_heads=H, num_kv_heads=H,
                       head_dim=HD, row_splits=R)
        return pv

    def _model(self, inputs, outputs):
        c = self.cfg
        return self._make_model(f"{self.name}_vision", inputs, outputs,
                                {"entry": "vision", "model": self.name, "layers": c.L, "hidden": c.D,
                                 "heads": c.H, "head_dim": c.HD, "patch": c.P, "image": c.S,
                                 "image_tokens": c.n_img, "pv_kw": self.pv_kw})


__all__ = ("VitConfig", "VisionFormats", "VitFrontend", "default_pieces", "patch_weight", "patch_bias_table",
           "patches")
