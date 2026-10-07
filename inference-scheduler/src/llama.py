"""
Llama-family frontend: config.json + safetensors weights + calibrated
formats -> fixed-shape ONNX entry graphs for the scheduler
(doc/plans/CHAT_PLAN.md §3.2 B1 / §10.5; decision and rationale in §12).

Instead of exporting the Hugging Face model with torch and pattern-matching
the result back into RMSNorm / RoPE / SwiGLU / KV-cache structure, the graphs
are written directly from the checkpoint with ``onnx.helper``: one standard
``MatMul`` per linear (per layer and class, weights as initializers shared
by name across the entries) and ``axi.llm`` host ops (src/llm_nodes.py) for
everything else.  Numerics — the power-of-two exponents of every int16
tensor, the host tensors (float32 residual stream, int32 ids / positions,
the int16 KV cache) and the states — ride in the model's ``axi.numeric``
metadata (src/numeric.py).  Nothing is model specific beyond what
config.json says (layers, hidden, heads, KV heads, head_dim, FFN, vocab,
RoPE theta, RMSNorm eps, tied embedding), so a SmolLM2-360M or another
Llama-architecture checkpoint only needs its formats (llm_study.py formats)
and a regeneration.

Entries (T = rows per call, C = context incl. the position-0 sink):

  decode       ids[1], pos[1]      -> logits[V] (float32)      KV cache += 1 row
  prefill_T    ids[T], pos[1], n[1] -> state h_last            KV cache += n rows
  head         (state h_last)      -> logits[V] (float32)

``pos`` is the cache position of the first new token (the number of
positions already filled, >= 1: row 0 is the sink), ``n <= T`` the number of
valid rows of a padded prefill (rows >= n are computed but write nothing
and are never read).  The KV caches (``kv.k.l<i>`` / ``kv.v.l<i>``, logical
[C][KV*HD] raw int16 states stored group-major [KV][C][HD]) start with the
sink row: the float run of the position-0 token, rounded at the cache
exponents (formats JSON).

Attention (``prefill_attn``):
  "fpga" (default, policy pow2+sink+p12+mix, doc/plans/CHAT_PLAN.md §16): decode
         steps run the xattn host region (LlmAttention); prefill runs q.K^T
         and P.V on ConvKernel per KV group with the p12 host softmax
         (LlmAttnPrep / LlmAttnScores / LlmAttnSoftmax / LlmAttnPV /
         LlmAttnMerge) over keys16 = roundup(pos + n, 16) keys — a runtime
         dimension.  The caches are DMA states in the CMA pool.
  "host" (phase 3, pow2+sink+p12+xattn): LlmAttention everywhere, the caches
         host-memory i16 states.
Decode attention (``decode_attn``, with prefill_attn "fpga"):
  "host" (default): the xattn host region above (policy pow2+sink+p12+mix).
  "fpga" (doc/plans/KV_DECODE_PLAN.md, policy pow2+sink+p12): decode steps run
         the prefill path with one row — LlmAttnPrep (no n: the step's own cache
         rows flushed), q.K^T / P.V on ConvKernel over roundup(pos + 1, Q) keys,
         the p12 softmax, LlmAttnMerge.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

from . import numeric
from .llm_nodes import LLM_DOMAIN

# The shift of the kernels' output (floor(acc / 2^F)); ap_fixed<16,8>.
F = 8


# ------------------------------------------------------------------ #
# Config / weights                                                     #
# ------------------------------------------------------------------ #

@dataclass
class LlamaConfig:
    L: int                  # layers
    D: int                  # hidden size
    H: int                  # attention heads
    KV: int                 # key / value heads
    HD: int                 # head dim
    FF: int                 # MLP intermediate size
    V: int                  # vocabulary
    eps: float
    theta: float
    tied: bool = True

    @classmethod
    def from_dict(cls, c: dict) -> "LlamaConfig":
        H = int(c["num_attention_heads"])
        D = int(c["hidden_size"])
        return cls(L=int(c["num_hidden_layers"]), D=D, H=H,
                   KV=int(c.get("num_key_value_heads", H)),
                   HD=int(c.get("head_dim") or D // H),
                   FF=int(c["intermediate_size"]), V=int(c["vocab_size"]),
                   eps=float(c["rms_norm_eps"]), theta=float(c.get("rope_theta", 10000.0)),
                   tied=bool(c.get("tie_word_embeddings", True)))

    @classmethod
    def from_file(cls, path: str) -> "LlamaConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _st_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """A safetensors tensor's bytes -> float32 (bf16 upcast exactly)."""
    if dtype == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return raw.view(np.float32).copy()
    raise ValueError(f"dtype {dtype}")


def _st_header(f) -> Tuple[int, dict]:
    n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n, json.loads(f.read(n))


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """bf16 / f16 / f32 safetensors -> float32 arrays (bf16 upcast exactly)."""
    with open(path, "rb") as f:
        _, hdr = _st_header(f)
        blob = np.frombuffer(f.read(), np.uint8)
    out = {}
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        try:
            arr = _st_float32(blob[a:b], m["dtype"])
        except ValueError:
            raise ValueError(f"{name}: dtype {m['dtype']}") from None
        out[name] = arr.reshape(m["shape"])
    return out


class LazySafetensors(Mapping):
    """``load_safetensors`` without the float32 copy of the checkpoint: the
    file is memory-mapped and a tensor becomes float32 when read — every read
    converts anew and nothing is kept (the same values as load_safetensors).
    For frontends that build one entry at a time (generate_llm_project.py).
    ``select({name: name in the file})`` gives a renamed subset."""

    def __init__(self, path: str, names: Optional[Dict[str, str]] = None):
        with open(path, "rb") as f:
            off, hdr = _st_header(f)
        self.path = path
        self._blob = np.memmap(path, np.uint8, mode="r", offset=off)
        self._meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
        self._names = names if names is not None else {k: k for k in self._meta}

    def select(self, names: Dict[str, str]) -> "LazySafetensors":
        out = object.__new__(LazySafetensors)
        out.path, out._blob, out._meta = self.path, self._blob, self._meta
        out._names = {n: self._names[f] for n, f in names.items()}
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        m = self._meta[self._names[name]]
        a, b = m["data_offsets"]
        try:
            arr = _st_float32(np.asarray(self._blob[a:b]), m["dtype"])
        except ValueError:
            raise ValueError(f"{name}: dtype {m['dtype']}") from None
        return arr.reshape(m["shape"])

    def __contains__(self, name) -> bool:
        return name in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


def rope_tables(cfg: LlamaConfig, n: int):
    """float32 cos / sin [n][HD/2] exactly as llm_study.rope_tables:
    inv_j = float32(1 / theta^(2j/HD)), a = float32(float32(pos) * inv_j),
    cos = float32(cos(double(a))) (libm)."""
    half = cfg.HD // 2
    inv = np.array([1.0 / (cfg.theta ** (2 * j / cfg.HD)) for j in range(half)],
                   np.float64).astype(np.float32)
    ang = (np.arange(n, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    c = np.vectorize(math.cos, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    s = np.vectorize(math.sin, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    return c, s


# The linears: (key, file name, input class, output class) — llm_study.LINEARS
LINEARS = (("q", "self_attn.q_proj", "x", "q0"), ("k", "self_attn.k_proj", "x", "k0"),
           ("v", "self_attn.v_proj", "x", "v"), ("o", "self_attn.o_proj", "pv", "o"),
           ("g", "mlp.gate_proj", "x2", "g"), ("u", "mlp.up_proj", "x2", "u"),
           ("d", "mlp.down_proj", "a", "d"))


# ------------------------------------------------------------------ #
# Formats                                                              #
# ------------------------------------------------------------------ #

class Formats:
    """Exponents per ``class@layer`` (llm_study.py ``formats`` JSON; layer L
    = the final norm / LM head) and the sink rows."""

    def __init__(self, d: dict, cfg: LlamaConfig):
        self.cfg = cfg
        self.exp = {k: v for k, v in d["exponents"].items()}
        self.sink_k = d.get("sink_k_raw")
        self.sink_v = d.get("sink_v_raw")
        self.policy = d.get("policy", "")

    @classmethod
    def from_file(cls, path: str, cfg: LlamaConfig) -> "Formats":
        with open(path) as f:
            return cls(json.load(f), cfg)

    def get(self, cls_: str, layer: int, n: int) -> np.ndarray:
        """Exponent vector of length n (a scalar is broadcast; missing = F)."""
        v = self.exp.get(f"{cls_}@{layer}", F)
        return np.broadcast_to(np.asarray(v, np.int64), (n,)).copy()

    def heads(self, cls_: str, layer: int, n_heads: int) -> np.ndarray:
        return self.get(cls_, layer, n_heads)

    def k_cache(self, li: int) -> np.ndarray:
        """[KV*HD]: the K cache exponent (per KV head) over the channels."""
        c = self.cfg
        return np.repeat(self.heads("k", li, c.KV), c.HD)

    def v_cache(self, li: int) -> np.ndarray:
        c = self.cfg
        key = "vc" if f"vc@{li}" in self.exp else "v"
        return self.get(key, li, c.KV * c.HD)

    def pv(self, li: int) -> np.ndarray:
        """P.V output exponent per channel of [H*HD]: f_p[h] + f_vc[g(h)] - F."""
        c = self.cfg
        grp = np.arange(c.H) // (c.H // c.KV)
        fp = self.heads("p", li, c.H)
        vc = self.v_cache(li).reshape(c.KV, c.HD)
        if f"pv@{li}" in self.exp:
            return self.get("pv", li, c.H * c.HD)
        return (fp[:, None] + vc[grp] - F).reshape(-1)


def graph_with_initializers(nodes, name, inputs, outputs, inits: Dict[str, np.ndarray],
                            value_info) -> onnx.GraphProto:
    """``oh.make_graph`` with ``inits`` ({name: array}) converted to float32
    TensorProtos one at a time, straight into the graph: one copy of each
    weight, not a second set of TensorProtos beside the graph's."""
    g = oh.make_graph(nodes, name, inputs, outputs, value_info=value_info)
    for n, arr in inits.items():
        g.initializer.append(nph.from_array(np.ascontiguousarray(arr, np.float32), n))
    return g


def _compact(e: np.ndarray):
    """An int when every channel has the same exponent, else a list."""
    e = np.asarray(e, np.int64).reshape(-1)
    return int(e[0]) if (e == e[0]).all() else [int(v) for v in e]


# ------------------------------------------------------------------ #
# Graph builder                                                        #
# ------------------------------------------------------------------ #

class LlamaFrontend:
    """Builds the entry graphs of one model.  ``weights``: the checkpoint's
    tensors by Hugging Face name (float32), ``formats``: see Formats."""

    def __init__(self, cfg: LlamaConfig, weights: Dict[str, np.ndarray], formats: Formats,
                 ctx: int = 1024, name: str = "llama", prefill_attn: str = "fpga",
                 decode_attn: str = "host",
                 qk_kw: Optional[int] = None, pv_kw: Optional[int] = None,
                 image_rows: int = 0, image_state: str = "vlm.img"):
        # image_rows > 0 (a VLM's text model, src/vit.py): the prefill entries'
        # LlmEmbed also reads image_state [image_rows][D] (float32 host state the
        # vision entry writes); ids V .. V + image_rows - 1 select its rows
        self.image_rows = int(image_rows)
        self.image_state = image_state
        self.cfg = cfg
        self.W = weights
        self.fmt = formats
        self.C = int(ctx)
        self.name = name
        if prefill_attn not in ("fpga", "host"):
            raise ValueError(f"prefill_attn must be 'fpga' or 'host', got {prefill_attn!r}")
        self.prefill_attn = prefill_attn
        if decode_attn not in ("fpga", "host"):
            raise ValueError(f"decode_attn must be 'fpga' or 'host', got {decode_attn!r}")
        if decode_attn == "fpga" and prefill_attn != "fpga":
            raise ValueError("decode_attn 'fpga' needs prefill_attn 'fpga' (the caches in the pool)")
        self.decode_attn = decode_attn
        self.qk_kw = qk_kw
        # The V cache's interleave K = the P.V call's kernel width: the cache
        # is stored as that conv's input image, so each P row of a weight slab
        # is one 16K-lane request (K = 4: 2.3-2.6x faster P.V on the board than
        # K = 1, CHAT_PLAN §16).  Needs context % 16K == 0; host mode: 1.
        if pv_kw is None:
            pv_kw = next(k for k in (4, 2, 1) if self.C % (16 * k) == 0) \
                if prefill_attn == "fpga" else 1
        if self.C % (16 * int(pv_kw)):
            raise ValueError(f"pv_kw {pv_kw}: context {self.C} must be a multiple of {16 * pv_kw}")
        self.pv_kw = int(pv_kw)
        if prefill_attn == "fpga" and self.C % 16:
            raise ValueError(f"FPGA prefill attention: context {self.C} must be a multiple of 16")
        if formats.sink_k is None or formats.sink_v is None:
            raise ValueError("formats: sink_k_raw / sink_v_raw missing (policy with a sink)")
        self._cos, self._sin = rope_tables(cfg, self.C)
        self._consts: Dict[str, np.ndarray] = {}

    # ---- shared constants ------------------------------------------------ #
    def _w(self, li: int, fname: str) -> np.ndarray:
        return self.W[f"model.layers.{li}.{fname}.weight"]

    def lm_weight(self) -> np.ndarray:
        W = self.W
        return W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]

    def _cache_init(self, li: int, which: str) -> np.ndarray:
        c = self.cfg
        raw = np.asarray((self.fmt.sink_k if which == "k" else self.fmt.sink_v)[li],
                         np.float64).reshape(-1)                      # [KV*HD]
        f = self.fmt.k_cache(li) if which == "k" else self.fmt.v_cache(li)
        init = np.zeros((self.C, c.KV * c.HD), np.float32)
        init[0] = (raw * np.power(2.0, -f.astype(np.float64))).astype(np.float32)
        return init

    # ---- entries ----------------------------------------------------------- #
    def entry(self, kind: str, T: int = 1, with_head: bool = False) -> onnx.ModelProto:
        """``kind`` in ("decode", "prefill", "head"); T = rows of a prefill.
        ``with_head``: a prefill that also returns the logits of row n-1
        (one self-contained graph, e.g. for the board's model suite)."""
        if kind == "decode":
            return self._build("decode", 1, head=True, select=False)
        if kind == "prefill":
            return self._build(f"prefill_{T}" + ("_logits" if with_head else ""), T,
                               head=with_head, select=True)
        if kind == "head":
            return self._build_head()
        raise ValueError(kind)

    def _new(self, ename):
        self._nodes: List[onnx.NodeProto] = []
        self._vi: Dict[str, onnx.ValueInfoProto] = {}
        self._inits: Dict[str, np.ndarray] = {}      # converted in _model
        self._meta = numeric.empty()
        self._e = ename

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

    def _matmul(self, li, key, x, T, out_exp, lm=False):
        c = self.cfg
        if lm:
            wname, W = "w.lm_head", self.lm_weight().T
        else:
            fname = dict((k, f) for k, f, _, _ in LINEARS)[key]
            wname, W = f"w.l{li}.{key}", self._w(li, fname).T
        K, M = W.shape
        self._init(wname, W)
        y = self._t(f"{self._e}.l{li}.{key}" if not lm else f"{self._e}.logits_q", [T, M],
                    exp=out_exp)
        self._node("MatMul", [x, wname], [y], f"{self._e}.l{li}.{key}_proj" if not lm
                   else f"{self._e}.lm_head")
        del c
        return y

    def _rmsnorm(self, h, gamma_name, gamma, T, out, exp, name):
        self._init(gamma_name, gamma)
        y = self._t(out, [T, self.cfg.D], exp=exp)
        self._node("LlmRMSNorm", [h, gamma_name], [y], name, domain=LLM_DOMAIN,
                   axi_eps=repr(float(self.cfg.eps)))
        return y

    def choose_qk_kw(self, T: int) -> int:
        """Kernel width of the q.K^T calls' q image (the frontend writes the
        image, so any kw with head_dim % (16 kw) == 0 is free): the cheapest
        by cost_model.conv_cycles at C / 2 keys."""
        if self.qk_kw is not None:
            return int(self.qk_kw)
        from .cost_model import conv_cycles
        c = self.cfg
        M = (c.H // c.KV) * T
        keys = max(16, (self.C // 2) // 16 * 16)
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

    def _caches(self, li):
        c, fm = self.cfg, self.fmt
        host = "i16" if self.prefill_attn == "host" else None
        kc = self._t(f"kv.k.l{li}", [self.C, c.KV * c.HD], exp=fm.k_cache(li), host=host, state=True)
        vc = self._t(f"kv.v.l{li}", [self.C, c.KV * c.HD], exp=fm.v_cache(li), host=host, state=True)
        self._meta["layout"][kc] = [c.KV, c.HD]
        self._meta["layout"][vc] = [c.KV, c.HD] + ([self.pv_kw] if self.pv_kw > 1 else [])
        self._init(kc, self._cache_init(li, "k"))
        self._init(vc, self._cache_init(li, "v"))
        return kc, vc

    def _fpga_attention(self, li, q0, k0, v, T, pos, n, kc, vc):
        """Per layer: LlmAttnPrep, then per KV group q.K^T (ConvKernel), the
        p12 softmax (host) and P.V (ConvKernel), then LlmAttnMerge.  Node
        order q.K^T 0, q.K^T 1, softmax 0, P.V 0, then per group g >= 1
        softmax g, q.K^T g+1, P.V g: the ConvKernel runs one call at a time
        and the host issues them in order, so each softmax follows a call it
        can hide behind — softmax 0 the short q.K^T 1, every later one the
        long P.V of the previous group."""
        c, fm, e = self.cfg, self.fmt, self._e
        G, KV, HD, H = c.H // c.KV, c.KV, c.HD, c.H
        grp = np.arange(H) // G
        fq = fm.heads("q", li, H)
        fk = fm.heads("k", li, KV)
        fs = (fm.heads("s", li, H) if f"s@{li}" in fm.exp else fq + fk[grp] - F)
        fp = fm.heads("p", li, H)
        kw = self.choose_qk_kw(T)
        common = dict(num_heads=H, num_kv_heads=KV, head_dim=HD, key_quantum=16 * self.pv_kw)
        qx = self._t(f"{e}.l{li}.qx", [KV, HD, G * T], exp=0)
        self._node("LlmAttnPrep", [q0, k0, v, pos, n, kc, vc, "rope.cos", "rope.sin"], [qx],
                   f"{e}.l{li}.attn_prep", domain=LLM_DOMAIN, qk_kw=kw,
                   q_exp=[int(x) for x in fq], **common)
        s = [self._t(f"{e}.l{li}.s{g}", [self.C, G * T], exp=0) for g in range(KV)]
        p = [self._t(f"{e}.l{li}.p{g}", [G * T, self.C], exp=0) for g in range(KV)]
        o = [self._t(f"{e}.l{li}.o{g}", [G * T, HD], exp=0) for g in range(KV)]

        def qk(g):
            self._node("LlmAttnScores", [kc, qx, pos, n], [s[g]], f"{e}.l{li}.qk{g}",
                       domain=LLM_DOMAIN, group=g, qk_kw=kw, **common)
        def softmax(g):
            hs = slice(g * G, (g + 1) * G)
            self._node("LlmAttnSoftmax", [s[g], pos, n], [p[g]], f"{e}.l{li}.softmax{g}",
                       domain=LLM_DOMAIN, group=g, s_exp=[int(x) for x in fs[hs]],
                       p_exp=[int(x) for x in fp[hs]], **common)

        def pv_(g):
            self._node("LlmAttnPV", [p[g], vc, pos, n], [o[g]], f"{e}.l{li}.pv{g}",
                       domain=LLM_DOMAIN, group=g, **common)
        qk(0)
        if KV > 1:
            qk(1)
        softmax(0)
        pv_(0)
        for g in range(1, KV):
            softmax(g)
            if g + 1 < KV:
                qk(g + 1)
            pv_(g)
        pv = self._t(f"{e}.l{li}.pv", [T, H * HD], exp=fm.pv(li))
        self._node("LlmAttnMerge", o, [pv], f"{e}.l{li}.attn_merge", domain=LLM_DOMAIN,
                   num_heads=H, num_kv_heads=KV, head_dim=HD)
        return pv

    def _layers(self, h, T, pos, n):
        c, fm, e = self.cfg, self.fmt, self._e
        self._init("rope.cos", self._cos)
        self._init("rope.sin", self._sin)
        for li in range(c.L):
            x = self._rmsnorm(h, f"g.l{li}.in", self.W[f"model.layers.{li}.input_layernorm.weight"],
                              T, f"{e}.l{li}.x", fm.get("x", li, c.D), f"{e}.l{li}.norm_in")
            q0 = self._matmul(li, "q", x, T, fm.get("q0", li, c.H * c.HD))
            k0 = self._matmul(li, "k", x, T, fm.get("k0", li, c.KV * c.HD))
            v = self._matmul(li, "v", x, T, fm.get("v", li, c.KV * c.HD))
            kc, vc = self._caches(li)
            if self.prefill_attn == "fpga" and (n is not None or self.decode_attn == "fpga"):
                pv = self._fpga_attention(li, q0, k0, v, T, pos, n or "", kc, vc)
            else:
                pv = self._t(f"{e}.l{li}.pv", [T, c.H * c.HD], exp=fm.pv(li))
                self._node("LlmAttention", [q0, k0, v, pos, n or "", kc, vc, "rope.cos",
                                            "rope.sin"],
                           [pv], f"{e}.l{li}.attn", domain=LLM_DOMAIN,
                           num_heads=c.H, num_kv_heads=c.KV, head_dim=c.HD)
            o = self._matmul(li, "o", pv, T, fm.get("o", li, c.D))
            h1 = self._t(f"{e}.l{li}.h1", [T, c.D], host="f32")
            self._node("LlmResAdd", [h, o], [h1], f"{e}.l{li}.res_attn", domain=LLM_DOMAIN)
            x2 = self._rmsnorm(h1, f"g.l{li}.post",
                               self.W[f"model.layers.{li}.post_attention_layernorm.weight"],
                               T, f"{e}.l{li}.x2", fm.get("x2", li, c.D), f"{e}.l{li}.norm_post")
            g = self._matmul(li, "g", x2, T, fm.get("g", li, c.FF))
            u = self._matmul(li, "u", x2, T, fm.get("u", li, c.FF))
            a = self._t(f"{e}.l{li}.a", [T, c.FF], exp=fm.get("a", li, c.FF))
            self._node("LlmSiluMul", [g, u], [a], f"{e}.l{li}.silu_mul", domain=LLM_DOMAIN)
            d = self._matmul(li, "d", a, T, fm.get("d", li, c.D))
            h2 = self._t(f"{e}.l{li}.h2", [T, c.D], host="f32")
            self._node("LlmResAdd", [h1, d], [h2], f"{e}.l{li}.res_mlp", domain=LLM_DOMAIN)
            h = h2
        return h

    def _head(self, h):
        c, fm, e, L = self.cfg, self.fmt, self._e, self.cfg.L
        xf = self._rmsnorm(h, "g.final", self.W["model.norm.weight"], 1, f"{e}.xf",
                           fm.get("xf", L, c.D), f"{e}.norm_final")
        lq = self._matmul(L, "lm", xf, 1, fm.get("logits", L, c.V), lm=True)
        y = self._t(f"{e}.logits", [1, c.V], host="f32")
        self._node("LlmDequant", [lq], [y], f"{e}.dequant", domain=LLM_DOMAIN)
        return y

    def _build(self, ename, T, head, select):
        c = self.cfg
        self._new(ename)
        ids = self._t(f"{ename}.ids", [T], TensorProto.INT32, host="i32")
        pos = self._t(f"{ename}.pos", [1], TensorProto.INT32, host="i32")
        inputs = [ids, pos]
        n = None
        if select:
            n = self._t(f"{ename}.n", [1], TensorProto.INT32, host="i32")
            inputs.append(n)
        self._init("emb", self.W["model.embed_tokens.weight"])
        h0 = self._t(f"{ename}.h0", [T, c.D], host="f32")
        emb_in = [ids, "emb"]
        if select and self.image_rows:
            img = self._t(self.image_state, [self.image_rows, c.D], host="f32", state=True)
            self._init(img, np.zeros((self.image_rows, c.D), np.float32))
            emb_in.append(img)
        self._node("LlmEmbed", emb_in, [h0], f"{ename}.embed", domain=LLM_DOMAIN)
        h = self._layers(h0, T, pos, n)
        outputs = []
        if select:
            hl = self._t("h_last", [1, c.D], host="f32", state=True)
            self._node("LlmSelectRow", [h, n], [hl], f"{ename}.select_last", domain=LLM_DOMAIN)
            h = hl
        if head:
            outputs.append(self._head(h))
        self._meta["test_fill"][pos] = 3
        if n:
            self._meta["test_fill"][n] = max(1, T - 1)
        return self._model(ename, inputs, outputs)

    def _build_head(self):
        c = self.cfg
        self._new("head")
        hl = self._t("h_last", [1, c.D], host="f32", state=True)
        self._init(hl, np.zeros((1, c.D), np.float32))
        y = self._head(hl)
        return self._model("head", [], [y])

    def _model(self, ename, inputs, outputs):
        g = graph_with_initializers(
            self._nodes, f"{self.name}_{ename}",
            [self._vi[n] for n in inputs], [self._vi[n] for n in outputs], self._inits,
            value_info=[vi for n, vi in self._vi.items() if n not in inputs and n not in outputs])
        self._inits = {}
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17),
                                            oh.make_opsetid(LLM_DOMAIN, 1)],
                          producer_name="inference-scheduler/src/llama.py")
        m.ir_version = 8
        p = m.metadata_props.add()
        p.key = numeric.METADATA_KEY
        p.value = numeric.to_metadata(self._meta)
        p = m.metadata_props.add()
        p.key = "axi.llm.entry"
        p.value = json.dumps({"entry": ename, "model": self.name, "context": self.C,
                              "prefill_attn": self.prefill_attn, "decode_attn": self.decode_attn,
                              "pv_kw": self.pv_kw,
                              "layers": self.cfg.L, "hidden": self.cfg.D, "heads": self.cfg.H,
                              "kv_heads": self.cfg.KV, "head_dim": self.cfg.HD,
                              "vocab": self.cfg.V}
                             | ({"image_rows": self.image_rows} if self.image_rows else {}))
        return m


def entry_info(model: onnx.ModelProto) -> Optional[dict]:
    for p in model.metadata_props:
        if p.key == "axi.llm.entry":
            return json.loads(p.value)
    return None


__all__ = ("LlamaConfig", "LlamaFrontend", "Formats", "load_safetensors", "LazySafetensors",
           "graph_with_initializers", "rope_tables", "entry_info", "LINEARS")
