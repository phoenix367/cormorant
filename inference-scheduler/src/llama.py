"""
Llama-family frontend: config.json + safetensors weights + calibrated
formats -> fixed-shape ONNX entry graphs for the scheduler
(doc/CHAT_PLAN.md §3.2 B1 / §10.5; decision and rationale in §12).

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
and are never read).  The KV caches (``kv.k.l<i>`` / ``kv.v.l<i>``,
[C][KV*HD] raw int16 host states) start with the sink row: the float run
of the position-0 token, rounded at the cache exponents (formats JSON).
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """bf16 / f16 / f32 safetensors -> float32 arrays (bf16 upcast exactly)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = np.frombuffer(f.read(), np.uint8)
    out = {}
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        raw = blob[a:b]
        if m["dtype"] == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif m["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif m["dtype"] == "F32":
            arr = raw.view(np.float32).copy()
        else:
            raise ValueError(f"{name}: dtype {m['dtype']}")
        out[name] = arr.reshape(m["shape"])
    return out


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

    def k_cache(self, l: int) -> np.ndarray:
        """[KV*HD]: the K cache exponent (per KV head) over the channels."""
        c = self.cfg
        return np.repeat(self.heads("k", l, c.KV), c.HD)

    def v_cache(self, l: int) -> np.ndarray:
        c = self.cfg
        key = "vc" if f"vc@{l}" in self.exp else "v"
        return self.get(key, l, c.KV * c.HD)

    def pv(self, l: int) -> np.ndarray:
        """P.V output exponent per channel of [H*HD]: f_p[h] + f_vc[g(h)] - F."""
        c = self.cfg
        grp = np.arange(c.H) // (c.H // c.KV)
        fp = self.heads("p", l, c.H)
        vc = self.v_cache(l).reshape(c.KV, c.HD)
        if f"pv@{l}" in self.exp:
            return self.get("pv", l, c.H * c.HD)
        return (fp[:, None] + vc[grp] - F).reshape(-1)


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
                 ctx: int = 1024, name: str = "llama"):
        self.cfg = cfg
        self.W = weights
        self.fmt = formats
        self.C = int(ctx)
        self.name = name
        if formats.sink_k is None or formats.sink_v is None:
            raise ValueError("formats: sink_k_raw / sink_v_raw missing (policy with a sink)")
        self._cos, self._sin = rope_tables(cfg, self.C)
        self._consts: Dict[str, np.ndarray] = {}

    # ---- shared constants ------------------------------------------------ #
    def _w(self, l: int, fname: str) -> np.ndarray:
        return self.W[f"model.layers.{l}.{fname}.weight"]

    def lm_weight(self) -> np.ndarray:
        W = self.W
        return W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]

    def _cache_init(self, l: int, which: str) -> np.ndarray:
        c = self.cfg
        raw = np.asarray((self.fmt.sink_k if which == "k" else self.fmt.sink_v)[l],
                         np.float64).reshape(-1)                      # [KV*HD]
        f = self.fmt.k_cache(l) if which == "k" else self.fmt.v_cache(l)
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
        self._inits: Dict[str, onnx.TensorProto] = {}
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
            self._inits[name] = nph.from_array(np.ascontiguousarray(arr, np.float32), name)
        return name

    def _node(self, op, ins, outs, name, domain="", **attrs):
        self._nodes.append(oh.make_node(op, ins, outs, name=name, domain=domain, **attrs))

    def _matmul(self, l, key, x, T, out_exp, lm=False):
        c = self.cfg
        if lm:
            wname, W = "w.lm_head", self.lm_weight().T
        else:
            fname = dict((k, f) for k, f, _, _ in LINEARS)[key]
            wname, W = f"w.l{l}.{key}", self._w(l, fname).T
        K, M = W.shape
        self._init(wname, W)
        y = self._t(f"{self._e}.l{l}.{key}" if not lm else f"{self._e}.logits_q", [T, M],
                    exp=out_exp)
        self._node("MatMul", [x, wname], [y], f"{self._e}.l{l}.{key}_proj" if not lm
                   else f"{self._e}.lm_head")
        del c
        return y

    def _rmsnorm(self, h, gamma_name, gamma, T, out, exp, name):
        self._init(gamma_name, gamma)
        y = self._t(out, [T, self.cfg.D], exp=exp)
        self._node("LlmRMSNorm", [h, gamma_name], [y], name, domain=LLM_DOMAIN,
                   axi_eps=repr(float(self.cfg.eps)))
        return y

    def _layers(self, h, T, pos, n):
        c, fm, e = self.cfg, self.fmt, self._e
        self._init("rope.cos", self._cos)
        self._init("rope.sin", self._sin)
        for l in range(c.L):
            x = self._rmsnorm(h, f"g.l{l}.in", self.W[f"model.layers.{l}.input_layernorm.weight"],
                              T, f"{e}.l{l}.x", fm.get("x", l, c.D), f"{e}.l{l}.norm_in")
            q0 = self._matmul(l, "q", x, T, fm.get("q0", l, c.H * c.HD))
            k0 = self._matmul(l, "k", x, T, fm.get("k0", l, c.KV * c.HD))
            v = self._matmul(l, "v", x, T, fm.get("v", l, c.KV * c.HD))
            kc = self._t(f"kv.k.l{l}", [self.C, c.KV * c.HD], exp=fm.k_cache(l), host="i16",
                         state=True)
            vc = self._t(f"kv.v.l{l}", [self.C, c.KV * c.HD], exp=fm.v_cache(l), host="i16",
                         state=True)
            self._init(kc, self._cache_init(l, "k"))
            self._init(vc, self._cache_init(l, "v"))
            pv = self._t(f"{e}.l{l}.pv", [T, c.H * c.HD], exp=fm.pv(l))
            self._node("LlmAttention", [q0, k0, v, pos, n or "", kc, vc, "rope.cos", "rope.sin"],
                       [pv], f"{e}.l{l}.attn", domain=LLM_DOMAIN,
                       num_heads=c.H, num_kv_heads=c.KV, head_dim=c.HD)
            o = self._matmul(l, "o", pv, T, fm.get("o", l, c.D))
            h1 = self._t(f"{e}.l{l}.h1", [T, c.D], host="f32")
            self._node("LlmResAdd", [h, o], [h1], f"{e}.l{l}.res_attn", domain=LLM_DOMAIN)
            x2 = self._rmsnorm(h1, f"g.l{l}.post",
                               self.W[f"model.layers.{l}.post_attention_layernorm.weight"],
                               T, f"{e}.l{l}.x2", fm.get("x2", l, c.D), f"{e}.l{l}.norm_post")
            g = self._matmul(l, "g", x2, T, fm.get("g", l, c.FF))
            u = self._matmul(l, "u", x2, T, fm.get("u", l, c.FF))
            a = self._t(f"{e}.l{l}.a", [T, c.FF], exp=fm.get("a", l, c.FF))
            self._node("LlmSiluMul", [g, u], [a], f"{e}.l{l}.silu_mul", domain=LLM_DOMAIN)
            d = self._matmul(l, "d", a, T, fm.get("d", l, c.D))
            h2 = self._t(f"{e}.l{l}.h2", [T, c.D], host="f32")
            self._node("LlmResAdd", [h1, d], [h2], f"{e}.l{l}.res_mlp", domain=LLM_DOMAIN)
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
        self._node("LlmEmbed", [ids, "emb"], [h0], f"{ename}.embed", domain=LLM_DOMAIN)
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
        g = oh.make_graph(
            self._nodes, f"{self.name}_{ename}",
            [self._vi[n] for n in inputs], [self._vi[n] for n in outputs],
            initializer=list(self._inits.values()),
            value_info=[vi for n, vi in self._vi.items() if n not in inputs and n not in outputs])
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
                              "layers": self.cfg.L, "hidden": self.cfg.D, "heads": self.cfg.H,
                              "kv_heads": self.cfg.KV, "head_dim": self.cfg.HD,
                              "vocab": self.cfg.V})
        return m


def entry_info(model: onnx.ModelProto) -> Optional[dict]:
    for p in model.metadata_props:
        if p.key == "axi.llm.entry":
            return json.loads(p.value)
    return None


__all__ = ("LlamaConfig", "LlamaFrontend", "Formats", "load_safetensors", "rope_tables",
           "entry_info", "LINEARS")
