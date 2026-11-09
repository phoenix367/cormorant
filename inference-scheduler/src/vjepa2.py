# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
V-JEPA 2 video-classifier frontend (facebook/vjepa2-vitl-fpc16-256-ssv2: the
ViT-L encoder with 3-D RoPE, the attentive pooler and the Something-Something
v2 head; doc/plans/VJEPA2_PLAN.md): config.json + video_preprocessor_config.json
+ the checkpoint + the calibrated formats (demo/video_classification/scripts/
vjepa2_study.py formats) -> the ``clip`` entry.  The specification is
demo/video_classification/scripts/vjepa2_q.py (vjepa2_q.VJEPA2 with the
formats), which the entry's simulation equals bit for bit.  Needs the row-ops
unit and VectorOPKernel's softmax and activation units (the policy
pow2+p12+vgelu+vsmx+rowops).

  clip   patches.p<i> [rows of piece i][3 tub P P] (the clip's raw uint8 pixels
         as int16, exponent 0; token t = (frame pair, row, column), a patch as
         (channel, frame, row, column) — Conv3d's weight order; one input per
         row piece, so no piece is a copy) -> logits [1][classes] (float32)

The graph (vjepa2_q.VJEPA2's emulation; the rows in pieces as src/vit.py's,
pinned to ConvKernel and issued as a pipeline):

  pe = patches . Wpe          MatMul (the pixel normalisation folded: W / (255
                              std_c), b - sum W mean_c / std_c)
  h  = VjResCopy(pe, b0)      the int32 residual at 2^-15 (the bias rounded once)
  per encoder layer (24):
    x = LN1(h); q0 / k0 / v = x . W      (q / k: each head's columns in
                                          rotate-half order, the biases too)
    VitAttnPrep (rope): q0 + bq, k0 + bk at the bias exponents, ROPE (the
      per-token cos / sin rows of the 3-D positions) at f_q / f_k; query-major:
      q -> the "K cache" rows, k -> the image; v -> the V image
    per head, per query part r: s = q.K^T (ConvKernel, rows of part r),
      P = VitAttnSoftmax (VectorOPKernel's row mode), o = P.V
    pv = LlmAttnMerge; h += pv . Wo + bo; x2 = LN2(h); f = x2 . W1;
    a = VitGelu (erf, VectorOPKernel's activation unit); h += a . W2 + b2
  xf = LN(h); h = VjResCopy(xf)            the pooler's residual
  per pooler self-attention layer (3): the same without RoPE
  the cross layer: x = LN1(h); k0 = x . Wk; v0 = x . Wv;
    r = VjCrossAttn(k0, v0)    (the learned query attends; + the query; int32)
    x2 = LN2(r); f = x2 . W1; a = VitGelu; r += a . W2 + b2
  xc = VjResCopy(r) (int16); logits = VjLogits(xc . Wc, bc)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import onnx

from .graph_builder import RES_BITS, ExponentTable, GraphBuilder, pv_exp, score_exp
from .llm_nodes import LLM_DOMAIN
from .vit import ATTN_US_PER_MAC, PIECE_US_PER_ELEM, PIECE_US_PER_MAC, _gelu_vop_rows, _Pipe, default_pieces

ENC = "vjepa2.encoder."
POOL = "pooler."
ENC_LIN = {"q": "attention.query", "k": "attention.key", "v": "attention.value", "o": "attention.proj",
           "f1": "mlp.fc1", "f2": "mlp.fc2"}
POOL_LIN = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.out_proj",
            "f1": "mlp.fc1", "f2": "mlp.fc2"}
CROSS_LIN = {"k": "cross_attn.k_proj", "v": "cross_attn.v_proj", "f1": "mlp.fc1", "f2": "mlp.fc2"}
ROPE_THETA = 10000.0                    # V-JEPA 2's rotate_queries_or_keys
ROPE_BITS = 23                          # the unit's cos / sin: int24 at 2^-23 (rowops_spec.ROPE_BITS)
MAX_PART_ROWS = 1024                    # the attention's query parts: within ConvKernel's out_ch bound


@dataclass
class VJepa2Config:
    D: int                  # hidden size
    L: int                  # encoder layers
    H: int                  # heads
    HD: int                 # head dim
    FF: int                 # MLP size
    eps: float
    P: int                  # patch size
    tub: int                # tubelet (frames per token)
    frames: int             # frames per clip
    S: int                  # crop size
    n_pool: int             # the pooler's self-attention layers
    labels: list
    short: int              # the resize's shortest edge
    mean: tuple
    std: tuple

    @property
    def grid(self) -> int:
        return self.S // self.P

    @property
    def T(self) -> int:
        return self.frames // self.tub

    @property
    def N(self) -> int:
        return self.T * self.grid * self.grid

    @property
    def C(self) -> int:
        return len(self.labels)

    @property
    def rot(self) -> int:
        """Dims per rotated position axis (frame, row, column): 2 floor(HD / 6)."""
        return 2 * ((self.HD // 3) // 2)

    @property
    def patch_k(self) -> int:
        return 3 * self.tub * self.P * self.P

    # the layer index of every tensor class (vjepa2_q.VCfg): encoder 0 .. L-1, its final
    # LayerNorm L, the pooler's self layers L+1 .., the cross layer, the classifier
    @property
    def lf(self) -> int:
        return self.L

    @property
    def lp(self) -> list:
        return [self.L + 1 + i for i in range(self.n_pool)]

    @property
    def lc(self) -> int:
        return self.L + 1 + self.n_pool

    @property
    def lk(self) -> int:
        return self.lc + 1

    def prefix(self, li: int):
        """(weight-name prefix, (LN1, LN2 names), the linears' names) of layer li."""
        if li < self.L:
            return f"{ENC}layer.{li}.", ("norm1", "norm2"), ENC_LIN
        if li in self.lp:
            return f"{POOL}self_attention_layers.{li - self.L - 1}.", ("layer_norm1", "layer_norm2"), POOL_LIN
        if li == self.lc:
            return f"{POOL}cross_attention_layer.", ("layer_norm1", "layer_norm2"), CROSS_LIN
        raise ValueError(f"layer {li}: not an attention layer")

    @classmethod
    def from_dicts(cls, c: dict, pp: dict) -> "VJepa2Config":
        """``c``: config.json, ``pp``: video_preprocessor_config.json."""
        if c.get("hidden_act") != "gelu" or not c.get("qkv_bias") or c.get("use_SiLU") or c.get("in_chans") != 3:
            raise ValueError("V-JEPA 2: only the GELU (erf) MLP, q / k / v biases, 3 input channels")
        if not (pp.get("do_center_crop") and pp.get("do_rescale") and pp.get("do_normalize")
                and pp.get("resample") == 2):
            raise ValueError("V-JEPA 2: the video processor must resize (bilinear), center-crop, rescale, "
                             "normalise")
        S = int(c["crop_size"])
        if pp["crop_size"] != {"height": S, "width": S}:
            raise ValueError(f"V-JEPA 2: the processor's crop {pp['crop_size']} is not the model's {S}")
        D, H = int(c["hidden_size"]), int(c["num_attention_heads"])
        labels = [c["id2label"][str(i)] for i in range(len(c["id2label"]))]
        return cls(D=D, L=int(c["num_hidden_layers"]), H=H, HD=D // H, FF=int(D * float(c["mlp_ratio"])),
                   eps=float(c["layer_norm_eps"]), P=int(c["patch_size"]), tub=int(c["tubelet_size"]),
                   frames=int(c["frames_per_clip"]), S=S, n_pool=int(c["num_pooler_layers"]), labels=labels,
                   short=int(pp["size"]["shortest_edge"]), mean=tuple(float(v) for v in pp["image_mean"]),
                   std=tuple(float(v) for v in pp["image_std"]))

    @classmethod
    def from_dir(cls, path: str) -> "VJepa2Config":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        with open(os.path.join(path, "video_preprocessor_config.json")) as f:
            pp = json.load(f)
        return cls.from_dicts(c, pp)


class VJepa2Formats(ExponentTable):
    """Exponents per ``class@layer`` (vjepa2_study.py formats JSON; layer numbers of
    VJepa2Config: encoder 0 .., xf@L, the pooler, the cross layer, xc / logits / res
    at the classifier's)."""

    def __init__(self, d: dict, cfg: VJepa2Config):
        super().__init__(d)
        self.cfg = cfg

    def get(self, cls_: str, layer: int, n: int) -> np.ndarray:
        return self.lookup(f"{cls_}@{layer}", n)

    def pv(self, li: int) -> np.ndarray:
        c = self.cfg
        return pv_exp(self.get("p", li, c.H), self.get("vc", li, c.D), c.H, c.HD)


# ------------------------------------------------------------------ the model's constants
def patch_weight(W: Dict[str, np.ndarray], cfg: VJepa2Config):
    """The patch embedding with the pixel normalisation folded in (vjepa2_q.patch_weight):
    [3 tub P P][D] W / (255 std_c) and the bias b - sum W mean_c / std_c (left to
    right), both float32."""
    std = np.asarray(cfg.std, np.float64)
    mean = np.asarray(cfg.mean, np.float64)
    Wp = W[ENC + "embeddings.patch_embeddings.proj.weight"].astype(np.float64).reshape(cfg.D, 3, -1)
    Wf = np.ascontiguousarray((Wp / (255.0 * std)[None, :, None]).reshape(cfg.D, -1).T)
    b = W[ENC + "embeddings.patch_embeddings.proj.bias"].astype(np.float64) \
        - np.cumsum((Wp * (mean / std)[None, :, None]).reshape(cfg.D, -1), axis=1)[:, -1]
    return Wf.astype(np.float32), b.astype(np.float32)


def rope_perm(HD: int) -> np.ndarray:
    """The head dims in rotate-half order: pair (2j, 2j + 1) at (j, j + HD / 2)."""
    return np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])


def rope_tables(cfg: VJepa2Config):
    """cos / sin [N][HD] of every token in double (vjepa2_q.rope_tables: the frame /
    row / column of token n rotate dims [0, rot), [rot, 2 rot), [2 rot, 3 rot), the
    rot / 2 frequencies 10000^(-k / (rot / 2)) repeated twice; the rest cos 1, sin 0)."""
    n = np.arange(cfg.N)
    g2 = cfg.grid * cfg.grid
    pos = (n // g2, (n % g2) // cfg.grid, n % cfg.grid)
    half = cfg.rot // 2
    omega = 1.0 / np.power(ROPE_THETA, np.arange(half) / half)
    C = np.ones((cfg.N, cfg.HD))
    S = np.zeros((cfg.N, cfg.HD))
    for seg in range(3):
        freq = pos[seg].astype(np.float64)[:, None] * omega[None, :]
        sl = slice(seg * cfg.rot, (seg + 1) * cfg.rot)
        C[:, sl] = np.tile(np.cos(freq), 2)
        S[:, sl] = np.tile(np.sin(freq), 2)
    return C, S


def rope_raw(cfg: VJepa2Config):
    """The unit's tables: int24 at 2^-23 of the double cos / sin, the head dims in
    rotate-half order (rowops_spec.rope_tables_raw of vjepa2_q's tables)."""
    C, S = rope_tables(cfg)
    p = rope_perm(cfg.HD)
    return (np.rint(C[:, p] * 2.0 ** ROPE_BITS).astype(np.int64),
            np.rint(S[:, p] * 2.0 ** ROPE_BITS).astype(np.int64))


def cross_query(W: Dict[str, np.ndarray], cfg: VJepa2Config) -> np.ndarray:
    """The cross layer's q = query . Wq^T + bq in double, each output a correctly
    rounded sum of the exact products (math.fsum: the same on every machine, as
    vjepa2_q.cross_query) plus the bias."""
    query = W[POOL + "query_tokens"].reshape(-1).astype(np.float64)
    cq = POOL + "cross_attention_layer.cross_attn.q_proj."
    Wq, bq = W[cq + "weight"].astype(np.float64), W[cq + "bias"].astype(np.float64)
    return np.array([math.fsum(query * Wq[j]) for j in range(Wq.shape[0])]) + bq


def patches(clip: np.ndarray, cfg: VJepa2Config) -> np.ndarray:
    """uint8 [frames][S][S][3] -> [N][3 tub P P] (vjepa2_q.patches): token (frame pair,
    row, column), a patch as (channel, frame, row, column)."""
    T, g, P, tb = cfg.T, cfg.grid, cfg.P, cfg.tub
    x = np.asarray(clip).reshape(T, tb, g, P, g, P, 3).transpose(0, 2, 4, 6, 1, 3, 5)
    return np.ascontiguousarray(x.reshape(T * g * g, 3 * tb * P * P))


def attention_parts(N: int) -> int:
    """Query parts of the attention: the fewest whose rows (a multiple of 16) fit
    MAX_PART_ROWS (ConvKernel's out_ch bound for q.K^T and P.V)."""
    for R in range(1, N + 1):
        if N % R == 0 and (N // R) % 16 == 0 and N // R <= MAX_PART_ROWS:
            return R
    raise ValueError(f"{N} tokens: no query parts of at most {MAX_PART_ROWS} rows (multiples of 16)")


# ------------------------------------------------------------------ the frontend
class VJepa2Frontend(GraphBuilder):
    """Builds the ``clip`` entry.  ``weights``: the checkpoint's tensors by Hugging
    Face name (float32).  ``pieces``: the rows of the linears / row ops [(rows,
    "conv" | "matmul"), ...] (default src/vit.py's default_pieces: two halves on
    ConvKernel); ``parts``: the attention's query parts (default
    attention_parts)."""

    PRODUCER = "inference-scheduler/src/vjepa2.py"

    def __init__(self, cfg: VJepa2Config, weights: Dict[str, np.ndarray], formats: VJepa2Formats,
                 name: str = "vjepa2", pieces=None, parts: Optional[int] = None):
        self.cfg = cfg
        self.W = weights
        self.fmt = formats
        self.name = name
        N = cfg.N
        if pieces is None:
            pieces = default_pieces(N)
            if pieces is None:
                raise ValueError(f"{N} tokens: give the row pieces")
        self.pieces = tuple((int(r), str(k)) for r, k in pieces)
        if sum(r for r, _k in self.pieces) != N or any(r % 16 or k not in ("conv", "matmul")
                                                       for r, k in self.pieces):
            raise ValueError(f"pieces {self.pieces}: rows in multiples of 16 adding up to {N}, on 'conv' or "
                             "'matmul'")
        self.parts = attention_parts(N) if parts is None else int(parts)
        if N % self.parts or (N // self.parts) % 16:
            raise ValueError(f"{self.parts} query parts of {N} tokens: rows in multiples of 16")
        self.pv_kw = next(k for k in (4, 2, 1) if N % (16 * k) == 0)
        if cfg.HD % 16 or cfg.HD > 128 or cfg.rot < 2:
            raise ValueError(f"head_dim {cfg.HD}: the unit's ROPE takes 16 .. 128, a multiple of 16")
        self.fr = int(np.asarray(formats.get("res", cfg.lk, 1))[0])
        if self.fr != RES_BITS:
            raise ValueError(f"the formats' residual at 2^-{self.fr}: the row-ops nodes take 2^-{RES_BITS}")
        perm = rope_perm(cfg.HD)
        self.cols = np.concatenate([h * cfg.HD + perm for h in range(cfg.H)])   # q / k columns, rotate-half

    # ---- helpers ------------------------------------------------------------- #
    def _rows(self):
        out, r0 = [], 0
        for i, (n, eng) in enumerate(self.pieces):
            out.append((i, r0, n, eng))
            r0 += n
        return out

    def _vec(self, name, arr):
        return self._init(name, np.asarray(arr, np.float32).reshape(-1))

    def _res(self, name, rows):
        return self._residual(name, rows, self.cfg.D, True, bits=self.fr)

    def _ln(self, h, li, which, out, exp, gname, bname, rows):
        self._t(out, [rows, self.cfg.D], exp=exp)
        self._node("VitLayerNorm", [h, self._vec(f"vj.l{li}.{which}.g", self.W[gname]),
                                    self._vec(f"vj.l{li}.{which}.b", self.W[bname])], [out], f"{out}.ln",
                   domain=LLM_DOMAIN, axi_eps=repr(float(self.cfg.eps)))
        return out

    def _mm(self, pipe, x, wname, W, n, y, exp, name, eng):
        self._init(wname, W)
        self._t(y, [n, W.shape[1]], exp=exp)
        if eng is not None:                          # a row piece: pinned to its kernel
            self._meta["engine"][y] = eng
        if pipe is None:
            self._node("MatMul", [x, wname], [y], name)
        else:
            pipe.add(eng, n * W.shape[0] * W.shape[1] * PIECE_US_PER_MAC[eng], [x], [y],
                     lambda: self._node("MatMul", [x, wname], [y], name))
        return y

    def _ln_piece(self, pipe, h, li, which, out, exp, gname, bname, n):
        pipe.add("rowops", n * self.cfg.D * PIECE_US_PER_ELEM["rowops_ln"], [h], [out],
                 lambda: self._ln(h, li, which, out, exp, gname, bname, n))
        return out

    def _res_piece(self, pipe, h, d, bname, b, y, name, n):
        self._res(y, n)
        bv = self._vec(bname, b)
        pipe.add("rowops", n * self.cfg.D * PIECE_US_PER_ELEM["rowops_res"], [h, d], [y],
                 lambda: self._node("VitResAdd", [h, d, bv], [y], name, domain=LLM_DOMAIN))
        return y

    def _copy(self, pipe, x, bname, b, y, name, n):
        """VjResCopy x -> y (one side the int32 residual)."""
        bv = self._vec(bname, b)
        pipe.add("rowops", n * self.cfg.D * PIECE_US_PER_ELEM["rowops_res"], [x], [y],
                 lambda: self._node("VjResCopy", [x, bv], [y], name, domain=LLM_DOMAIN))
        return y

    def _slice_rows(self, x, r0, n, y, exp, cols):
        s, e = f"vj.rows{r0}.s", f"vj.rows{r0}.e"
        self._int_init(s, np.array([r0], np.int64))
        self._int_init(e, np.array([r0 + n], np.int64))
        self._int_init("vj.axis0", np.array([0], np.int64))
        self._t(y, [n, cols], exp=exp)
        self._node("Slice", [x, s, e, "vj.axis0"], [y], f"{y}.slice")
        return y

    def _concat_rows(self, parts, y, cols, exp):
        self._t(y, [self.cfg.N, cols], exp=exp)
        self._node("Concat", list(parts), [y], f"{y}.cat", axis=0)
        return y

    def _gelu(self, pipe, f, a, b1, extra, li, name, n):
        c = self.cfg
        lane = "vectorop"
        add = lambda: self._node("VitGelu", [f, b1] + extra, [a], name, domain=LLM_DOMAIN,   # noqa: E731
                                 approximate="none")
        if pipe is None:
            add()
        else:
            pipe.add(lane, n * c.FF * PIECE_US_PER_ELEM["vop_gelu"], [f], [a], add)
        return a

    def _gelu_rows(self, li, b1name):
        c, fm = self.cfg, self.fmt
        vop = _gelu_vop_rows(self._inits[b1name], fm.get("f", li, c.FF), fm.get("a", li, c.FF), c.FF)
        if vop is None:
            raise ValueError(f"layer {li}: GELU (erf) runs on VectorOPKernel's activation unit only, and its "
                             "output is not at 2^-8 on every channel (or the operands leave int16)")
        return [self._vec(f"vj.l{li}.gelu_ba", vop[0]), self._vec(f"vj.l{li}.gelu_sc", vop[1])]

    # ---- the entry ----------------------------------------------------------- #
    def entry(self) -> onnx.ModelProto:
        c, fm, W = self.cfg, self.fmt, self.W
        N, D = c.N, c.D
        self._new()
        rows = self._rows()
        for li in list(range(c.L)) + c.lp + [c.lc]:
            for key in ("o", "d"):
                if li != c.lc or key == "d":
                    fd = np.asarray(fm.get(key, li, D), np.int64)
                    if (fd > self.fr).any():
                        raise ValueError(f"the {key} exponents of layer {li} exceed 2^-{self.fr} (max "
                                         f"{int(fd.max())}): the residual add would not be exact")
        # the caches (raw integers, one pair for every layer) and the unit's staging rows
        self.kc = self._t("vj.k", [N, D], exp=0, state=True)
        self.vc = self._t("vj.v", [N, D], exp=0, state=True)
        self._meta["layout"][self.kc] = [c.H, c.HD]
        self._meta["layout"][self.vc] = [c.H, c.HD] + ([self.pv_kw] if self.pv_kw > 1 else [])
        for nm in (self.kc, self.vc, "vj.ro.qs", "vj.ro.vs", "vj.ro.sa"):
            self._t(nm, [N, D], exp=0, state=True)
            self._init(nm, np.zeros((N, D), np.float32))
        cr, sr = rope_raw(c)
        self._init("vj.rope.cos", cr.astype(np.float32))       # int24 values: exact in float32
        self._init("vj.rope.sin", sr.astype(np.float32))
        # the patch embedding into the int32 residual
        Wpe, b0 = patch_weight(W, c)
        pipe = _Pipe()
        ins, hs = [], []
        for i, _r0, n, eng in rows:
            pin = self._t(f"clip.patches.p{i}", [n, c.patch_k], exp=0)
            ins.append(pin)
            pe = self._mm(pipe, pin, "vj.w.pe", Wpe, n, f"vj.pe.p{i}", fm.get("pe", 0, D), f"vj.patch_embed.p{i}",
                          eng)
            h = self._res(f"vj.h0.p{i}", n)
            hs.append(self._copy(pipe, pe, "vj.b0", b0, h, f"vj.embed.p{i}", n))
        for li in range(c.L):
            self._block(pipe, hs, li, rope=True)
        # the encoder's final LayerNorm, the pooler's residual
        for i, _r0, n, _eng in rows:
            xf = self._ln_piece(pipe, hs[i], c.lf, "lnf", f"vj.xf.p{i}", fm.get("xf", c.lf, D),
                                ENC + "layernorm.weight", ENC + "layernorm.bias", n)
            hs[i] = self._copy(pipe, xf, "vj.zero", np.zeros(D), self._res(f"vj.hp.p{i}", n), f"vj.pool_in.p{i}",
                               n)
        for li in c.lp:
            self._block(pipe, hs, li, rope=False)
        logits = self._cross(pipe, hs)
        return self._make_model(f"{self.name}_clip", ins, [logits],
                                {"entry": "clip", "model": self.name, "layers": c.L, "pooler_layers": c.n_pool,
                                 "hidden": D, "heads": c.H, "head_dim": c.HD, "tokens": N, "classes": c.C,
                                 "frames": c.frames, "tubelet": c.tub, "patch": c.P, "crop": c.S,
                                 "short_side": c.short, "mean": list(c.mean), "std": list(c.std),
                                 "pieces": [list(p) for p in self.pieces], "parts": self.parts,
                                 "pv_kw": self.pv_kw})

    def _block(self, pipe, hs, li, rope):
        """A self-attention block over the row pieces ``hs`` (updated in place)."""
        c, fm, W = self.cfg, self.fmt, self.W
        D = c.D
        pre, (n1, _n2), lin = c.prefix(li)
        e = f"vj.l{li}"
        rows = self._rows()
        perm = self.cols if rope else np.arange(D)
        qkv = {"q0": [], "k0": [], "v": []}
        for i, _r0, n, eng in rows:
            x = self._ln_piece(pipe, hs[i], li, "ln1", f"{e}.x.p{i}", fm.get("x", li, D), pre + n1 + ".weight",
                               pre + n1 + ".bias", n)
            for key, wk in (("q0", "q"), ("k0", "k"), ("v", "v")):
                Wt, ex = W[pre + lin[wk] + ".weight"].T, fm.get(key, li, D)
                if wk != "v":
                    Wt, ex = Wt[:, perm], ex[perm]
                qkv[key].append(self._mm(pipe, x, f"vj.w.l{li}.{wk}", np.ascontiguousarray(Wt), n, f"{e}.{key}.p{i}",
                                         ex, f"{e}.{wk}_proj.p{i}", eng))
        pipe.flush()
        cat = {}
        for key in ("q0", "k0", "v"):
            ex = fm.get(key, li, D)
            cat[key] = self._concat_rows(qkv[key], f"{e}.{key}", D, ex[perm] if key != "v" else ex)
        b = {wk: W[pre + lin[wk] + ".bias"] for wk in ("q", "k", "v")}
        pv = self._attention(li, e, cat["q0"], cat["k0"], cat["v"],
                             self._vec(f"vj.l{li}.bq", b["q"][perm]), self._vec(f"vj.l{li}.bk", b["k"][perm]),
                             self._vec(f"vj.l{li}.bv", b["v"]), rope)
        b1 = self._vec(f"vj.l{li}.b1", W[pre + lin["f1"] + ".bias"])
        extra = self._gelu_rows(li, b1)
        for i, r0, n, eng in rows:
            pv_i = self._slice_rows(pv, r0, n, f"{e}.pv.p{i}", fm.pv(li), D)
            o = self._mm(pipe, pv_i, f"vj.w.l{li}.o", W[pre + lin["o"] + ".weight"].T, n, f"{e}.o.p{i}",
                         fm.get("o", li, D), f"{e}.o_proj.p{i}", eng)
            h1 = self._res_piece(pipe, hs[i], o, f"vj.l{li}.bo", W[pre + lin["o"] + ".bias"], f"{e}.h1.p{i}",
                                 f"{e}.res_attn.p{i}", n)
            hs[i] = self._mlp(pipe, h1, li, e, f".p{i}", n, eng, b1, extra)

    def _mlp(self, pipe, h1, li, e, sfx, n, eng, b1, extra):
        c, fm, W = self.cfg, self.fmt, self.W
        D, FF = c.D, c.FF
        pre, (_n1, n2), lin = c.prefix(li)
        if pipe is None:
            x2 = self._ln(h1, li, "ln2", f"{e}.x2{sfx}", fm.get("x2", li, D), pre + n2 + ".weight",
                          pre + n2 + ".bias", n)
        else:
            x2 = self._ln_piece(pipe, h1, li, "ln2", f"{e}.x2{sfx}", fm.get("x2", li, D), pre + n2 + ".weight",
                                pre + n2 + ".bias", n)
        f = self._mm(pipe, x2, f"vj.w.l{li}.f1", W[pre + lin["f1"] + ".weight"].T, n, f"{e}.f{sfx}",
                     fm.get("f", li, FF), f"{e}.fc1{sfx}", eng)
        a = self._t(f"{e}.a{sfx}", [n, FF], exp=fm.get("a", li, FF))
        self._gelu(pipe, f, a, b1, extra, li, f"{e}.gelu{sfx}", n)
        d = self._mm(pipe, a, f"vj.w.l{li}.f2", W[pre + lin["f2"] + ".weight"].T, n, f"{e}.d{sfx}",
                     fm.get("d", li, D), f"{e}.fc2{sfx}", eng)
        y = f"{e}.h2{sfx}"
        if pipe is None:
            self._res(y, n)
            self._node("VitResAdd", [h1, d, self._vec(f"vj.l{li}.b2", W[pre + lin["f2"] + ".bias"])], [y],
                       f"{e}.res_mlp{sfx}", domain=LLM_DOMAIN)
            return y
        return self._res_piece(pipe, h1, d, f"vj.l{li}.b2", W[pre + lin["f2"] + ".bias"], y, f"{e}.res_mlp{sfx}",
                               n)

    def _attention(self, li, e, q0, k0, v0, bq, bk, bv, rope):
        """VitAttnPrep (query-major; rope: q / k rotated), then per head and query
        part q.K^T, the row-mode softmax and P.V as a pipeline, the merge -> pv [N][D]."""
        c, fm = self.cfg, self.fmt
        N, D, H, HD = c.N, c.D, c.H, c.HD
        fq, fk, fp = fm.get("q", li, H), fm.get("k", li, H), fm.get("p", li, H)
        fs = score_exp(fq, fk, H)
        qx = self._t(f"{e}.qx", [H, HD, N], exp=0)
        ins = [q0, k0, v0, self.kc, self.vc, bq, bk, bv, "vj.ro.qs", "vj.ro.vs"]
        attrs = dict(num_heads=H, head_dim=HD, qk_kw=1, q_exp=[int(v) for v in fq], k_exp=[int(v) for v in fk],
                     v_exp=[int(v) for v in fm.get("vc", li, D)], rowops=1, query_major=1)
        if rope:
            ins += ["", "vj.rope.cos", "vj.rope.sin", "vj.ro.sa"]
            attrs.update(rope=1, qb_exp=[int(v) for v in fm.get("qb", li, H)],
                         kb_exp=[int(v) for v in fm.get("kb", li, H)])
        self._node("VitAttnPrep", ins, [qx], f"{e}.attn_prep", domain=LLM_DOMAIN, **attrs)
        R = self.parts
        n = N // R
        common = dict(num_heads=H, num_kv_heads=H, head_dim=HD, key_quantum=16 * self.pv_kw)
        pipe, macs = _Pipe(), n * N * HD
        outs = []
        for g in range(H):
            for r in range(R):
                s = self._t(f"{e}.s{g}r{r}", [n, N], exp=0)
                p = self._t(f"{e}.p{g}r{r}", [n, N], exp=0)
                o = self._t(f"{e}.o{g}r{r}", [n, HD], exp=0)
                outs.append(o)
                pipe.add("conv", macs * ATTN_US_PER_MAC["qk_conv"], [qx], [s],
                         lambda g=g, r=r, s=s: self._node("LlmAttnScores", [self.kc, qx], [s], f"{e}.qk{g}r{r}",
                                                          domain=LLM_DOMAIN, group=g, qk_kw=1, rows=[r * n, n],
                                                          **common))
                pipe.add("vectorop", n * N * PIECE_US_PER_ELEM["vop_smx"], [s], [p],
                         lambda g=g, r=r, s=s, p=p: self._node("VitAttnSoftmax", [s], [p], f"{e}.softmax{g}r{r}",
                                                               domain=LLM_DOMAIN, head_dim=HD, s_exp=[int(fs[g])],
                                                               p_exp=[int(fp[g])], vsmx=1, query_major=1))
                pipe.add("conv", macs * ATTN_US_PER_MAC["pv_conv"], [p], [o],
                         lambda g=g, r=r, p=p, o=o: self._node("LlmAttnPV", [p, self.vc], [o], f"{e}.pv{g}r{r}",
                                                               domain=LLM_DOMAIN, group=g, **common))
        pipe.flush()
        pv = self._t(f"{e}.pv", [N, D], exp=fm.pv(li))
        self._node("LlmAttnMerge", outs, [pv], f"{e}.attn_merge", domain=LLM_DOMAIN, num_heads=H, num_kv_heads=H,
                   head_dim=HD, **(dict(row_splits=R) if R > 1 else {}))
        return pv

    def _cross(self, pipe, hs):
        """The pooler's cross-attention layer and the classifier -> the logits tensor."""
        c, fm, W = self.cfg, self.fmt, self.W
        D, H, HD = c.D, c.H, c.HD
        li = c.lc
        pre, (n1, _n2), lin = c.prefix(li)
        e = f"vj.l{li}"
        ks, vs = [], []
        for i, _r0, n, eng in self._rows():
            x = self._ln_piece(pipe, hs[i], li, "ln1", f"{e}.x.p{i}", fm.get("x", li, D), pre + n1 + ".weight",
                               pre + n1 + ".bias", n)
            ks.append(self._mm(pipe, x, f"vj.w.l{li}.k", W[pre + lin["k"] + ".weight"].T, n, f"{e}.k0.p{i}",
                               fm.get("k0", li, D), f"{e}.k_proj.p{i}", eng))
            vs.append(self._mm(pipe, x, f"vj.w.l{li}.v", W[pre + lin["v"] + ".weight"].T, n, f"{e}.v.p{i}",
                               fm.get("v", li, D), f"{e}.v_proj.p{i}", eng))
        pipe.flush()
        k0 = self._concat_rows(ks, f"{e}.k0", D, fm.get("k0", li, D))
        v0 = self._concat_rows(vs, f"{e}.v", D, fm.get("v", li, D))
        fq = fm.get("q", li, H)
        fqc = np.repeat(fq, HD).astype(np.float64)
        q = np.clip(np.round(cross_query(W, c) * np.power(2.0, fqc)), -32768, 32767) / np.power(2.0, fqc)
        r = self._res(f"{e}.r", 1)
        self._node("VjCrossAttn", [k0, v0, self._vec(f"vj.l{li}.bk", W[pre + lin["k"] + ".bias"]),
                                   self._vec(f"vj.l{li}.bv", W[pre + lin["v"] + ".bias"]),
                                   self._vec(f"vj.l{li}.q", q), self._vec("vj.query", W[POOL + "query_tokens"])],
                   [r], f"{e}.cross_attn", domain=LLM_DOMAIN, num_heads=H, head_dim=HD,
                   k_exp=[int(v) for v in fm.get("k", li, H)], v_exp=[int(v) for v in fm.get("vc", li, D)],
                   q_exp=[int(v) for v in fq], p_exp=[int(v) for v in fm.get("p", li, H)])
        b1 = self._vec(f"vj.l{li}.b1", W[pre + lin["f1"] + ".bias"])
        r2 = self._mlp(None, r, li, e, "", 1, None, b1, self._gelu_rows(li, b1))   # one row: the scheduler's choice
        lk = c.lk
        xc = self._t("vj.xc", [1, D], exp=fm.get("xc", lk, D))
        self._node("VjResCopy", [r2, self._vec("vj.zero", np.zeros(D))], [xc], "vj.pooled", domain=LLM_DOMAIN)
        lg0 = self._t("vj.logits0", [1, c.C], exp=fm.get("logits", lk, c.C))
        self._init("vj.w.cls", np.ascontiguousarray(W["classifier.weight"].T))
        self._node("MatMul", [xc, "vj.w.cls"], [lg0], "vj.classifier")
        logits = self._t("clip.logits", [1, c.C], host="f32")
        self._node("VjLogits", [lg0, self._vec("vj.b.cls", W["classifier.bias"])], [logits], "vj.logits",
                   domain=LLM_DOMAIN)
        return logits


__all__ = ("VJepa2Config", "VJepa2Formats", "VJepa2Frontend", "patch_weight", "rope_perm", "rope_tables", "rope_raw",
           "cross_query", "patches", "attention_parts")
