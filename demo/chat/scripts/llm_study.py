#!/usr/bin/env python3
"""B0 numeric go / no-go study: SmolLM2-135M-Instruct on the KV260 Q8.8 datapath.

doc/CHAT_PLAN.md §3.2 B0 (results in §9).  Mirrors the BERT study
(demo/bert_squad/scripts/bert_study.py, doc/BERT_PLAN.md §0): a float
reference, and a bit-level emulation of the planned partition (CHAT_PLAN §3.2
B2) under several numeric policies, compared on

  * top-1 next-token agreement with float at every position of the prompt set
    (chat prompts + float's greedy answers, teacher-forced) and of a held-out
    text (WikiText-2 test, 3 x 1024-token windows = the context length);
  * perplexity and mean KL(float || policy) on the held-out text;
  * greedy generation (128 new tokens) per chat prompt: exact-match length
    with float and the full texts side by side;
  * value ranges per tensor class (max |x| before quantisation, saturated
    fraction), logits range, max |matmul accumulator|.

Model: a numpy float64 Llama forward written from the safetensors weights
(RMSNorm, RoPE rotate-half, GQA 9 / 3 heads, SwiGLU, tied LM head, KV cache),
validated against transformers / torch float32 (`validate`).

Emulated datapath (the semantics the scheduler must implement).  Every DDR
tensor is int16 with a power-of-two exponent: value = raw * 2^-f (f = 8 is
ap_fixed<16,8>).  f is a scalar or a vector over the tensor's last-axis
channels (per head for q, k, P); the kernels never see it:

  kernel MatMul   q/k/v/o, gate/up/down, LM head, attention q.K^T and P.V:
                  raw int16 in, exact products summed in ap_fixed<32,16> (an int32
                  raw sum that wraps at 2^31, i.e. |acc| >= 32768 in Q16.16 units —
                  emulated), out_raw = floor(acc_raw / 2^8) (AP_TRN) saturated to
                  int16 (AP_SAT).  Hence f_out[j] = f_in[i] + f_w[i][j] - 8 for every
                  i: weights are encoded (round half to even, saturate) at the rank-1
                  exponent f_w[i][j] = f_out[j] + 8 - f_in[i].
  kernel VectorOP residual add (policy q88 only): raw add, saturate.
  host regions    IEEE double, reductions left to right, read raw * 2^-f[c],
                  write round_half_even(v * 2^f[c]) saturated to int16 (NaN -> 0):
    Gather        embedding row (bf16 table read exactly, or a Q8.8 table)
    RMSNorm       [h = float32(h + delta) — the residual adds, float residual]
                  ss = sum_i h_i*h_i (left to right); r = 1 / sqrt(ss / 576 + 1e-5);
                  y_i = (h_i * r) * gamma_i          (gamma float32 from bf16)
    RoPE          y = x*cos + rotate_half(x)*sin, rotate_half(x) = [-x[32:], x[:32]];
                  float32 tables: inv_j = float32(1 / 100000^(2j/64)),
                  a = float32(float32(pos) * inv_j), cos = float32(cos(a)), sin likewise
    V cache write (p_bits policies) v re-rounded to the cache's exponent
    Softmax       over the causal row (keys 0..pos): k = raw_max - raw_j (integer),
                  e_j = exp(-k * 2^-f_s * 0.125)  (the 1/sqrt(64) lives here; a
                  65 536-entry table per f_s, filled with libm exp),
                  p_j = e_j / sum(e) (left to right), written at f_p
    SiLU*up       a = silu(g) * u, silu(g) = g / (1 + exp(-g)) (a 65 536-entry
                  double table per f_g, libm exp), written at f_a
    attention     (hattn policies only) q.K^T, softmax, P.V in double, one write-back
  float residual  float32 host tensor (policies other than q88).
  sink            position 0 (<|im_start|>) is not run on the datapath: its K / V
                  rows (a float run of that token, written at the cache exponents)
                  are constants; prefill starts at position 1.
  argmax          first maximum (np.argmax).
Exponents come from a float calibration run over a separate set (WikiText-2
validation + 3 extra prompts), one bit of headroom (MARGIN), see make_formats.
Policies: POLICIES below; recommended pow2+sink+p12 (doc/CHAT_PLAN.md §9).

Usage (.venv-export: torch CPU, transformers, safetensors, numpy):
  PY=/home/ivan/projects/axi_demo/.venv-export/bin/python
  $PY demo/chat/scripts/llm_study.py fetch        # WikiText-2 held-out / calibration text -> assets
  $PY demo/chat/scripts/llm_study.py validate     # numpy float64 vs torch float32 (+ template, greedy)
  $PY demo/chat/scripts/llm_study.py study [--policies q88,...] [--quick] [--nogreedy] [--windows N]
  $PY demo/chat/scripts/llm_study.py ablate [--base fit+sink]   # error per tensor class
  $PY demo/chat/scripts/llm_study.py costs        # decode / prefill bytes and MACs per token

Assets (not in git, see demo/chat/assets/.gitignore): SMOLLM_ASSETS or
demo/chat/assets/smollm2-135m-instruct/ (config.json, model.safetensors,
tokenizer.json, tokenizer_config.json, generation_config.json; `fetch` adds
heldout_wikitext2_test.txt and calib_wikitext2_valid.txt).  From a git
worktree the main checkout's assets are used.  Results: <assets>/../study/.
Deterministic: greedy decoding only, no sampling, fixed data.
"""
import argparse, collections, json, math, os, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REL = os.path.join("demo", "chat", "assets", "smollm2-135m-instruct")


def default_assets():
    if os.environ.get("SMOLLM_ASSETS"):
        return os.environ["SMOLLM_ASSETS"]
    cands = [os.path.join(REPO, REL)]
    try:  # a git worktree shares the main checkout's untracked assets
        common = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--git-common-dir"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        cands.append(os.path.join(os.path.dirname(os.path.abspath(os.path.join(REPO, common))), REL))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return cands[0]


# ------------------------------------------------------------------ data
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
EOS = 2            # <|im_end|>
CTX = 1024         # context length (CHAT_PLAN decision)
MAX_NEW = 128

PARAGRAPH = (
    "The honeybee colony is often described as a superorganism. A single colony can contain "
    "tens of thousands of workers, a few hundred drones and one queen. Workers are sterile "
    "females that change jobs as they age: young bees clean cells and feed larvae, middle-aged "
    "bees build comb and guard the entrance, and the oldest bees fly out to collect nectar, "
    "pollen and water. Foragers share the location of good flowers through the waggle dance, "
    "whose angle and duration encode the direction and distance of the food source. In winter "
    "the colony survives by clustering together and shivering its flight muscles to keep the "
    "centre of the cluster warm.")

PROMPTS = [
    ("factual", [{"role": "user", "content": "What is the capital of France?"}]),
    ("factual-2", [{"role": "user", "content": "Who wrote the novel Pride and Prejudice?"}]),
    ("instruction", [{"role": "user", "content": "Give me three tips for staying focused while studying."}]),
    ("creative", [{"role": "user", "content": "Write a short poem about the ocean."}]),
    ("summarise", [{"role": "user", "content": "Summarize the following paragraph in one sentence.\n\n" + PARAGRAPH}]),
    ("arithmetic", [{"role": "user", "content": "What is 17 + 25?"}]),
    ("word-problem", [{"role": "user", "content": "I have 3 apples, I buy 5 more and then eat 2. How many apples do I have now?"}]),
    ("code", [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]),
    ("explain", [{"role": "user", "content": "Explain why the sky is blue in simple terms."}]),
    ("translate", [{"role": "user", "content": "Translate 'Good morning, how are you?' into French."}]),
    ("system", [{"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
                {"role": "user", "content": "Why do leaves change color in autumn?"}]),
    ("multi-turn", [{"role": "user", "content": "I'm planning a weekend trip to the mountains."},
                    {"role": "assistant", "content": "That sounds wonderful! Mountain trips are a great way to "
                     "relax and enjoy nature. Do you already know where you would like to go, or would you "
                     "like some suggestions?"},
                    {"role": "user", "content": "I know where I'm going. What should I pack?"}]),
]
CALIB_PROMPTS = [
    [{"role": "user", "content": "Name three primary colors."}],
    [{"role": "user", "content": "Write a haiku about winter."}],
    [{"role": "user", "content": "How do I reverse a list in Python?"}],
]


def chatml(messages):
    """SmolLM2's chat template (tokenizer_config.json), add_generation_prompt=True."""
    s = ""
    if messages[0]["role"] != "system":
        s += "<|im_start|>system\n" + DEFAULT_SYSTEM + "<|im_end|>\n"
    for m in messages:
        s += "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


class Tok:
    def __init__(self, assets):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(os.path.join(assets, "tokenizer.json"))

    def encode(self, text):
        return np.array(self.t.encode(text).ids, np.int64)

    def decode(self, ids):
        return self.t.decode([int(i) for i in ids], skip_special_tokens=True)


def wikitext_detok(s):
    """Light WikiText detokenisation (after lm-eval-harness' wikitext_detokenizer)."""
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" : ", ": "), (" ; ", "; "),
                 (" . ", ". "), (" ! ", "! "), (" ? ", "? "), (" , ", ", "), (" 's", "'s"),
                 ("( ", "("), (" )", ")"), ("\" ", "\""), (" n't", "n't")):
        s = s.replace(a, b)
    return s


def cmd_fetch(args):
    import urllib.request
    url = "https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1"
    for split, name, nchars in (("test", "heldout_wikitext2_test.txt", 24000),
                                ("validation", "calib_wikitext2_valid.txt", 8000)):
        text, off = "", 0
        while len(text) < nchars:
            with urllib.request.urlopen(f"{url}&split={split}&offset={off}&length=100") as r:
                rows = json.load(r)["rows"]
            if not rows:
                break
            text += "".join(wikitext_detok(x["row"]["text"]) for x in rows)
            off += 100
        path = os.path.join(args.assets, name)
        open(path, "w", encoding="utf-8").write(text)
        print(f"{path}: {len(text)} chars (WikiText-2 raw {split}, CC BY-SA 3.0)")


# ------------------------------------------------------------------ weights / config
def load_safetensors(path):
    """bf16 / f16 / f32 safetensors -> dict of float32 arrays (bf16 upcast exactly)."""
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
            raise ValueError(m["dtype"])
        out[name] = arr.reshape(m["shape"])
    return out


class Cfg:
    def __init__(self, assets):
        c = json.load(open(os.path.join(assets, "config.json")))
        self.L = c["num_hidden_layers"]; self.D = c["hidden_size"]; self.H = c["num_attention_heads"]
        self.KV = c["num_key_value_heads"]; self.F = c["intermediate_size"]; self.V = c["vocab_size"]
        self.HD = c.get("head_dim") or self.D // self.H
        self.eps = float(c["rms_norm_eps"]); self.theta = float(c["rope_theta"])
        self.tied = c.get("tie_word_embeddings", True)


def rope_tables(cfg, n):
    """float32 cos / sin [n, HD] (the host RoPE tables; HF's float32 math up to 1 ulp)."""
    half = cfg.HD // 2
    inv = np.array([1.0 / (cfg.theta ** (2 * j / cfg.HD)) for j in range(half)], np.float64).astype(np.float32)
    ang = (np.arange(n, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)   # float32 multiply
    c = np.vectorize(math.cos, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    s = np.vectorize(math.sin, otypes=[np.float64])(ang.astype(np.float64)).astype(np.float32)
    return (np.concatenate([c, c], 1).astype(np.float64), np.concatenate([s, s], 1).astype(np.float64))


# ------------------------------------------------------------------ numerics
I16_LO, I16_HI = -32768, 32767
_exp_tab, _silu_tab = {}, {}


def exp_table(f_s):
    """e[k] = exp(-k * 2^-f_s * 0.125), k = raw_max - raw in [0, 65535] (libm exp)."""
    if f_s not in _exp_tab:
        _exp_tab[f_s] = np.array([math.exp(-k / 2.0 ** f_s * 0.125) for k in range(65536)])
    return _exp_tab[f_s]


def silu_table(f_g, off=0.0):
    """silu(g) = g / (1 + exp(-g)) for every raw int16 r, g = (r + off) * 2^-f_g
    (off = 1/2 with the floor-bias fix), libm exp."""
    if (f_g, off) not in _silu_tab:
        t = np.empty(65536)
        for i in range(65536):
            g = (i - 32768 + off) / 2.0 ** f_g
            t[i] = g / (1.0 + math.exp(-g)) if g > -700 else 0.0
        _silu_tab[(f_g, off)] = t
    return _silu_tab[(f_g, off)]


class Stats:
    """Per tensor class and layer: max |x| before quantisation, saturated / total
    elements (kernel accumulators "acc:*": max |acc| in Q16.16 units — ap_fixed<32,16>
    wraps at 32768 — and wrapped elements); optionally per-channel maxima (the
    calibration input of the calibrated policies)."""
    def __init__(self, L):
        self.L = L
        self.d = {}
        self.ch = {}

    def add(self, key, l, mx, sat, n):
        e = self.d.setdefault(key, [np.zeros(self.L + 1), np.zeros(self.L + 1, np.int64), np.zeros(self.L + 1, np.int64)])
        e[0][l] = max(e[0][l], mx); e[1][l] += sat; e[2][l] += n

    def add_ch(self, key, l, v, sl=None, n=None):
        cur = self.ch.get((key, l))
        if sl is not None:
            full = np.zeros(n) if cur is None else cur
            full[sl] = np.maximum(full[sl], v)
            self.ch[(key, l)] = full
        else:
            self.ch[(key, l)] = v.copy() if cur is None else np.maximum(cur, v)

    def summary(self):
        return {k: dict(max=float(v[0].max()), argmax_layer=int(v[0].argmax()), sat=int(v[1].sum()),
                        n=int(v[2].sum()), per_layer_max=[float(x) for x in v[0]])
                for k, v in sorted(self.d.items())}


class Seq:
    """One sequence's KV cache: per layer K, V [KV, cap, HD] (values as stored in DDR)."""
    def __init__(self, cfg, cap):
        self.k = [np.zeros((cfg.KV, cap, cfg.HD)) for _ in range(cfg.L)]
        self.v = [np.zeros((cfg.KV, cap, cfg.HD)) for _ in range(cfg.L)]
        self.n = 0


# weights: (name in the file, input class, output class)
LINEARS = {"q": ("self_attn.q_proj", "x", "q0"), "k": ("self_attn.k_proj", "x", "k0"),
           "v": ("self_attn.v_proj", "x", "v"), "o": ("self_attn.o_proj", "pv", "o"),
           "g": ("mlp.gate_proj", "x2", "g"), "u": ("mlp.up_proj", "x2", "u"), "d": ("mlp.down_proj", "a", "d")}


def p2(e):
    return np.power(2.0, e)


def bf16(x):
    """round to bfloat16 (nearest even), returned as float64."""
    u = np.asarray(x, np.float64).astype(np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32).astype(np.float64)


# kernel outputs the host consumes (RoPE, SiLU*up, the residual adds); softmax and
# argmax are shift-invariant, so the floor-bias fix does not apply to s / logits
HOST_READ = ("q0", "k0", "g", "u", "o", "d")


class Model:
    """numpy Llama forward.

    pol None: the float64 reference.  Otherwise the emulation; fmt holds the
    exponents: fmt[(cls, l)] = f (int, or an int vector over the channels of the
    tensor's last axis; per head for q, k, s, p) — a value x is stored as the
    int16 round/floor(x * 2^f); fmt[("sh", key, l)] = the kernel's output shift
    (8 on today's bitstream; per head for "s" / "pv").  Unset exponents are 8
    (ap_fixed<16,8>).  Weight exponents follow: fw[i][j] = f_out[j] + sh - f_in[i],
    so every kernel call keeps out = floor(sum a_raw * w_raw / 2^sh) unchanged.
    sink: position 0 is not run on the datapath — its K / V (and logits) come
    from a float run of that token (sink_model), written into the cache at the
    cache's exponents (a constant of the application: position 0 is always
    <|im_start|>)."""

    def __init__(self, W, cfg, pol=None, fmt=None, cos_sin=None, sink=None, sink_model=None,
                 record_ch=False, base=None):
        self.cfg = cfg
        self.pol = pol or {}
        self.q = pol is not None
        self.fa_diag = bool(self.pol.get("float_acts"))
        self.qclasses = self.pol.get("qclasses")          # ablation: only these classes are quantised
        self.bf = bool(self.pol.get("bf16"))               # yardstick: float path, bf16 at every boundary
        self.floor_fix = bool(self.pol.get("floor_fix"))
        self.hattn = bool(self.pol.get("hattn"))           # attention as one float host region
        if self.bf:
            self.q = False
        self.fmt = fmt or {}
        self.sink = self.pol.get("sink", False) if sink is None else sink
        self.sink_model = sink_model
        self._sink = {}
        self.record_ch = record_ch
        L, H, KV = cfg.L, cfg.H, cfg.KV
        self.grp = np.arange(H) // (H // KV)
        self.cos, self.sin = cos_sin if cos_sin is not None else rope_tables(cfg, 2 * CTX)
        self.gin = [W[f"model.layers.{l}.input_layernorm.weight"].astype(np.float64) for l in range(L)]
        self.gpost = [W[f"model.layers.{l}.post_attention_layernorm.weight"].astype(np.float64) for l in range(L)]
        self.gfin = W["model.norm.weight"].astype(np.float64)
        E = W["model.embed_tokens.weight"]
        lm = W["lm_head.weight"] if "lm_head.weight" in W else E
        self.stats = Stats(L)
        self.wsat = 0
        if base is not None and not self.q:
            self.w = base.w
        else:
            self.w = {}
            for l in range(L):
                for key, (name, ci, co) in LINEARS.items():
                    self.w[(l, key)] = self._enc(W[f"model.layers.{l}.{name}.weight"].T, l, key, ci, co)
            self.w[(L, "lm")] = self._enc(lm.T, L, "lm", "xf", "logits")
        if not self.q or self.pol.get("emb") == "float":
            self.emb = E                                  # bf16 values as float32 (host table)
        else:
            self.emb = np.clip(np.round(E.astype(np.float64) * 256.0), I16_LO, I16_HI).astype(np.float32) / 256.0

    # ---- exponents
    def E(self, cls, l):
        if (cls, l) in self.fmt:
            return self.fmt[(cls, l)]
        if cls == "s":        # per head: f_q[h] + f_k[g(h)] - sh
            return self.Eh("q", l, self.cfg.H) + self.Eh("k", l, self.cfg.KV)[self.grp] - self.sh("s", l)
        if cls == "vc":       # the V cache: the v projection's output unless the host re-rounds it
            return self.E("v", l)
        if cls == "pv":       # per channel of [T, H*HD]: f_p[h] + f_vc[g(h), c] - sh[h]
            ph = self.Eh("p", l, self.cfg.H) - self.sh("pv", l)
            v = self.Ev("vc", l, self.cfg.KV * self.cfg.HD).reshape(self.cfg.KV, self.cfg.HD)
            return (ph[:, None] + v[self.grp]).reshape(-1)
        return 8

    def Eh(self, cls, l, n):                  # as a per-head / per-channel vector of length n
        e = self.E(cls, l)
        return np.full(n, e, np.int64) if np.ndim(e) == 0 else np.asarray(e, np.int64)

    def Ev(self, cls, l, n):
        return self.Eh(cls, l, n)

    def sh(self, key, l):
        v = self.fmt.get(("sh", key, l), 8)
        if key in ("s", "pv"):
            return np.full(self.cfg.H, v, np.int64) if np.ndim(v) == 0 else np.asarray(v, np.int64)
        return v

    def _enc(self, Wt, l, key, ci, co):
        Wt = np.ascontiguousarray(Wt, np.float64)
        if not self.q:
            return Wt
        K, N = Wt.shape
        fa, fo, sh = self.Ev(ci, l, K), self.Ev(co, l, N), self.sh(key, l)
        assert sh == 8
        sc = p2(fo + sh)[None, :] / p2(fa)[:, None]      # 2^fw, fw = f_out + sh - f_in
        if self.pol.get("float_weights"):
            return Wt * sc                              # unrounded (diagnostic)
        r = np.round(Wt * sc)
        self.wsat += int(((r < I16_LO) | (r > I16_HI)).sum())
        return np.clip(r, I16_LO, I16_HI)

    def exact(self, cls):
        return self.fa_diag or (self.qclasses is not None and cls not in self.qclasses)

    # ---- primitives
    def host(self, v, cls, l, e, record=True):
        """host-region write-back of a 2-D [rows, C] tensor: round half to even +
        saturate at 2^-e (e scalar or per channel)."""
        mx = float(np.abs(v).max()) if v.size else 0.0
        if self.record_ch and record and v.size:
            self.stats.add_ch(cls, l, np.abs(v).max(0))
        if not self.q:
            self.stats.add(cls, l, mx, int((np.abs(v) * 256.0 > 32767.5).sum()), v.size)
            return bf16(v) if self.bf else v
        if self.exact(cls):
            self.stats.add(cls, l, mx, 0, v.size)
            return v
        r = np.round(np.nan_to_num(v, nan=0.0) * p2(e))
        sat = int(((r < I16_LO) | (r > I16_HI)).sum())
        self.stats.add(cls, l, mx, sat, v.size)
        return np.clip(r, I16_LO, I16_HI) / p2(e)

    def kmm(self, a_raw, b_raw, fo, sh, co, l):
        """kernel MatMul on raw int16 operands (exact products / sums, ap_fixed<32,16>
        wrap, floor(acc / 2^sh), saturate); returns the value at exponent fo."""
        acc = a_raw @ b_raw                             # exact: integers < 2^53
        amax = float(np.abs(acc).max()) if acc.size else 0.0
        self.stats.add("acc:" + co, l, amax / 65536.0, int((np.abs(acc) >= 2.0 ** 31).sum()), acc.size)
        y = acc / p2(sh) / p2(fo)
        ymax = float(np.abs(y).max()) if y.size else 0.0
        if self.exact(co):
            self.stats.add(co, l, ymax, 0, y.size)
            return y
        if amax >= 2.0 ** 31:                           # ap_fixed<32,16> wraps
            acc = np.mod(acc + 2.0 ** 31, 2.0 ** 32) - 2.0 ** 31
        r = np.floor(acc / p2(sh))                      # AP_TRN
        sat = int(((r < I16_LO) | (r > I16_HI)).sum())
        self.stats.add(co, l, ymax, sat, r.size)
        r = np.clip(r, I16_LO, I16_HI)                  # AP_SAT
        if self.floor_fix and co in HOST_READ:          # the host reads floor(x) + 1/2 LSB
            r = r + 0.5
        return r / p2(fo)

    def fmm(self, a, b, co, l, sl=None, n=None, rec=True):
        """float reference matmul (+ range statistics)."""
        y = a @ b
        if y.size:
            self.stats.add(co, l, float(np.abs(y).max()), int((np.abs(y) * 256.0 >= 32768).sum()), y.size)
            if self.record_ch and rec:
                self.stats.add_ch(co, l, np.abs(y).max(0), sl, n)
        return bf16(y) if self.bf else y

    def lin(self, x, l, key):
        cfg = self.cfg
        ci, co = ("xf", "logits") if key == "lm" else LINEARS[key][1:]
        if not self.q:
            return self.fmm(x, self.w[(l, key)], co, l)
        return self.kmm(x * p2(self.E(ci, l)), self.w[(l, key)], self.E(co, l), self.sh(key, l), co, l)

    def rms(self, h, gamma, cls, l):
        ss = np.cumsum(h * h, axis=-1)[:, -1]           # left to right
        r = 1.0 / np.sqrt(ss / h.shape[1] + self.cfg.eps)
        return self.host((h * r[:, None]) * gamma, cls, l, self.E(cls, l))

    def rope(self, x, pos, cls, l):                      # x [T, nh, HD]
        T, nh, HD = x.shape
        c, s = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        half = HD // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        e = self.E(cls, l)
        e = np.repeat(np.asarray(e, np.int64), HD) if np.ndim(e) else e
        y = (x * c + rot * s).reshape(T, nh * HD)
        if self.hattn and cls == "q":                   # RoPE(q) stays inside the float attention region
            self.stats.add(cls, l, float(np.abs(y).max()), 0, y.size)
            return y.reshape(T, nh, HD)
        return self.host(y, cls, l, e).reshape(T, nh, HD)

    def resadd(self, h, d, l):
        if self.q and self.pol.get("residual") == "q88":   # VectorOP add, saturating
            y = h + d
            sat = int(((y * 256 < I16_LO) | (y * 256 > I16_HI)).sum())
            self.stats.add("h", l, float(np.abs(y).max()), sat, y.size)
            return np.clip(y * 256, I16_LO, I16_HI) / 256
        if self.bf:
            y = bf16(h + d)                                 # bf16 residual (as torch bf16 inference)
            self.stats.add("h", l, float(np.abs(y).max()), 0, y.size)
            return y
        y = (h + d).astype(np.float32).astype(np.float64)  # float32 host residual
        self.stats.add("h", l, float(np.abs(y).max()), int((np.abs(y) * 256 > 32767.5).sum()), y.size)
        return y

    def softmax(self, s, mask, fs, fp, l):              # one head: s [T, S], mask [T, S]
        if not self.q or self.exact("s"):
            x = np.where(mask, s * 0.125, -np.inf)
            e = np.exp(x - x.max(-1, keepdims=True))
        else:
            raw = np.rint(s * p2(fs)).astype(np.int64)
            m = np.where(mask, raw, -(1 << 20)).max(-1, keepdims=True)
            k = np.where(mask, m - raw, 0)
            e = np.where(mask, exp_table(int(fs))[k], 0.0)
        return self.host(e / np.cumsum(e, -1)[..., -1:], "p", l, fp, record=False)

    def silu_mul(self, g, u, l):
        if not self.q or self.exact("g"):
            return self.host(g / (1.0 + np.exp(-g)) * u, "a", l, self.E("a", l))
        fg = self.E("g", l)
        sil = np.empty_like(g)
        fgv = np.full(g.shape[1], fg, np.int64) if np.ndim(fg) == 0 else np.asarray(fg)
        for f in np.unique(fgv):                        # one table per distinct exponent
            cols = fgv == f
            idx = np.floor(g[:, cols] * p2(f)).astype(np.int64) + 32768     # the raw int16
            sil[:, cols] = silu_table(int(f), 0.5 if self.floor_fix else 0.0)[idx]
        return self.host(sil * u, "a", l, self.E("a", l))

    def attn(self, q, K, V, n0, l):
        """q [T, H, HD] (this sequence's new rows), K / V [KV, S, HD] incl. the new rows."""
        cfg = self.cfg
        T, S, HD = q.shape[0], K.shape[1], cfg.HD
        mask = np.arange(S)[None, :] <= (n0 + np.arange(T))[:, None]
        out = np.empty((T, cfg.H, HD))
        if self.q and self.hattn:
            # host region: s = q.k (exact in double), softmax in double, P.V in double,
            # one round-half-even write-back of the [T, H*HD] output at f_pv
            for h in range(cfg.H):
                g = self.grp[h]
                x = np.where(mask, (q[:, h] @ K[g].T) * 0.125, -np.inf)
                e = np.exp(x - x.max(-1, keepdims=True))
                out[:, h] = (e / np.cumsum(e, -1)[..., -1:]) @ V[g]
            return self.host(out.reshape(T, -1), "pv", l, self.E("pv", l)).reshape(T, cfg.H, HD)
        if self.q:
            fq, fk, fs = self.Eh("q", l, cfg.H), self.Eh("k", l, cfg.KV), self.Eh("s", l, cfg.H)
            fp, fv, fpv = self.Eh("p", l, cfg.H), self.Ev("vc", l, cfg.KV * HD), self.Ev("pv", l, cfg.H * HD)
            shs, shp = self.sh("s", l), self.sh("pv", l)
        for h in range(cfg.H):
            g = self.grp[h]
            if not self.q:
                s = self.fmm(q[:, h], K[g].T, "s", l, rec=False)
                if self.record_ch and s.size:
                    self.stats.add_ch("s", l, np.abs(s).max(), h, cfg.H)
                p = self.softmax(s, mask, 0, 0, l)
                out[:, h] = self.fmm(p, V[g], "pv", l, slice(h * HD, (h + 1) * HD), cfg.H * HD)
                continue
            fvg = fv[g * HD:(g + 1) * HD]
            s = self.kmm(q[:, h] * p2(fq[h]), (K[g] * p2(fk[g])).T, fs[h], shs[h], "s", l)
            p = self.softmax(s, mask, fs[h], fp[h], l)
            out[:, h] = self.kmm(p * p2(fp[h]), V[g] * p2(fvg)[None, :], fpv[h * HD:(h + 1) * HD], shp[h], "pv", l)
        return out

    # ---- position 0 (attention sink) precomputed in float
    def _write_sink(self, s, tok):
        cfg = self.cfg
        if tok not in self._sink:
            ss = Seq(cfg, 1)
            lg = self.sink_model.forward([ss], [np.array([tok])], "all")[0][0]
            self._sink[tok] = ([ss.k[l][:, 0].copy() for l in range(cfg.L)],
                               [ss.v[l][:, 0].copy() for l in range(cfg.L)], lg)
        K0, V0, lg = self._sink[tok]
        for l in range(cfg.L):
            if self.q and not self.exact("k"):
                fk = self.Eh("k", l, cfg.KV)[:, None]
                fv = self.Ev("vc", l, cfg.KV * cfg.HD).reshape(cfg.KV, cfg.HD)
                s.k[l][:, 0] = np.clip(np.round(K0[l] * p2(fk)), I16_LO, I16_HI) / p2(fk)
                s.v[l][:, 0] = np.clip(np.round(V0[l] * p2(fv)), I16_LO, I16_HI) / p2(fv)
            else:
                s.k[l][:, 0], s.v[l][:, 0] = K0[l], V0[l]
        s.n = 1
        return lg

    def forward(self, seqs, toks, want="last"):
        """Run the new tokens toks[i] of every sequence seqs[i] (prefill: one
        sequence, many tokens; batched decode: many sequences, one token each —
        numerically identical, every row is independent).  Returns per sequence
        the logits of its last position or of all its new positions."""
        if not self.sink:
            return self._forward(seqs, toks, want)
        pre, new = [], []
        for s, t in zip(seqs, toks):
            if s.n == 0:
                pre.append(self._write_sink(s, int(t[0])))
                t = t[1:]
            else:
                pre.append(None)
            new.append(t)
        act = [i for i, t in enumerate(new) if len(t)]
        res = [None] * len(seqs)
        if act:
            for i, o in zip(act, self._forward([seqs[i] for i in act], [new[i] for i in act], want)):
                res[i] = o
        for i in range(len(seqs)):
            if pre[i] is not None and (res[i] is None or want == "all"):
                res[i] = pre[i][None] if res[i] is None else np.concatenate([pre[i][None], res[i]])
        return res

    def _forward(self, seqs, toks, want):
        cfg = self.cfg
        L, H, KV, HD = cfg.L, cfg.H, cfg.KV, cfg.HD
        ids = np.concatenate(toks)
        T = len(ids)
        offs = np.cumsum([0] + [len(t) for t in toks])
        pos = np.concatenate([np.arange(s.n, s.n + len(t)) for s, t in zip(seqs, toks)])
        h = self.emb[ids].astype(np.float64)
        if self.q and self.pol.get("residual") == "q88":
            self.stats.add("h", 0, float(np.abs(h).max()), 0, h.size)
        for l in range(L):
            x = self.rms(h, self.gin[l], "x", l)
            q = self.rope(self.lin(x, l, "q").reshape(T, H, HD), pos, "q", l)
            k = self.rope(self.lin(x, l, "k").reshape(T, KV, HD), pos, "k", l)
            v = self.lin(x, l, "v")
            if self.q and ("vc", l) in self.fmt:            # host writes the V cache at its own exponent
                v = self.host(v, "vc", l, self.E("vc", l))
            v = v.reshape(T, KV, HD)
            att = np.empty((T, H, HD))
            for i, s in enumerate(seqs):
                a, b = offs[i], offs[i + 1]
                s.k[l][:, s.n:s.n + b - a] = k[a:b].transpose(1, 0, 2)
                s.v[l][:, s.n:s.n + b - a] = v[a:b].transpose(1, 0, 2)
                n1 = s.n + b - a
                att[a:b] = self.attn(q[a:b], s.k[l][:, :n1], s.v[l][:, :n1], s.n, l)
            h = self.resadd(h, self.lin(att.reshape(T, H * HD), l, "o"), l)
            x2 = self.rms(h, self.gpost[l], "x2", l)
            a_ = self.silu_mul(self.lin(x2, l, "g"), self.lin(x2, l, "u"), l)
            h = self.resadd(h, self.lin(a_, l, "d"), l)
        for s, t in zip(seqs, toks):
            s.n += len(t)
        if want == "last":
            h = h[offs[1:] - 1]
            offs = np.arange(len(seqs) + 1)
        logits = self.lin(self.rms(h, self.gfin, "xf", L), L, "lm")
        return [logits[offs[i]:offs[i + 1]] for i in range(len(seqs))]


def greedy(model, prompts, max_new=MAX_NEW):
    """Greedy decoding, stop at <|im_end|>.  Prefill per prompt, then batched
    one-token decode steps over the still-active prompts."""
    cfg = model.cfg
    seqs = [Seq(cfg, len(p) + max_new) for p in prompts]
    nxt = [int(np.argmax(model.forward([s], [p])[0][-1])) for s, p in zip(seqs, prompts)]
    out = [[] for _ in prompts]
    active = list(range(len(prompts)))
    while active:
        for i in active:
            out[i].append(nxt[i])
        active = [i for i in active if nxt[i] != EOS and len(out[i]) < max_new]
        if not active:
            break
        lg = model.forward([seqs[i] for i in active], [np.array([nxt[i]]) for i in active])
        for i, x in zip(active, lg):
            nxt[i] = int(np.argmax(x[-1]))
    return out


def teacher_forced(model, seq_ids):
    s = Seq(model.cfg, len(seq_ids))
    return model.forward([s], [seq_ids], want="all")[0]


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


# ------------------------------------------------------------------ policies
# residual  "q88" = a DDR tensor with VectorOP saturating adds (the BERT partition);
#           "float" = a float32 host tensor, the adds inside the RMSNorm regions
# emb       "q88" = Gather from a Q8.8 table; "float" = from the bf16 table (exact)
# sink      position 0 (<|im_start|>) precomputed in float: its K / V rows are
#           constants of the application, written once into the cache
# fmt       None = every exponent 8 (Q8.8);
#           "fit"  = Q8.8 unless the calibration range needs a coarser exponent
#                    (per channel, per head for q / k / p);
#           "pow2" = as fit, but kernel outputs read by the host (q0, k0, v, o, gate,
#                    up, down, logits) take the finest exponent their range allows, so
#                    the weights (fw = f_out + 8 - f_in) gain those bits;
#           per_tensor: one exponent per tensor instead of per channel
# p_bits    softmax P written at 2^-p_bits; the host re-rounds the V cache (vc) so
#           that P.V still fits int16
# hattn     attention (q.k, softmax, P.V) as one float host region; P never quantised
# in_cap    cap on the exponents of host-written kernel inputs (default 8)
# diagnostics: float_weights (+fw), float_acts (+fa), qclasses (ablate), floor_fix
#           (+ff: the host adds 1/2 LSB to the floored kernel outputs it reads),
#           bf16 (yardstick: float math, bf16 tensors at the same boundaries)
# The recommended policy (doc/CHAT_PLAN.md §9) is pow2+sink+p12.
_RF = dict(residual="float", emb="float")
POLICIES = {
    "float":                 None,
    "bf16":                  dict(bf16=True),
    "q88":                   dict(residual="q88", emb="q88"),
    "res_float":             dict(_RF),
    "res_float+sink":        dict(_RF, sink=True),
    "fit":                   dict(_RF, fmt="fit"),
    "fit+sink":              dict(_RF, sink=True, fmt="fit"),
    "fit_tensor+sink":       dict(_RF, sink=True, fmt="fit", per_tensor=True),
    "fit+sink+fw":           dict(_RF, sink=True, fmt="fit", float_weights=True),
    "fit+sink+fa":           dict(_RF, sink=True, fmt="fit", float_acts=True),
    "fit+sink+ff":           dict(_RF, sink=True, fmt="fit", floor_fix=True),
    "fit+sink+p11":          dict(_RF, sink=True, fmt="fit", p_bits=11),
    "fit+sink+p12":          dict(_RF, sink=True, fmt="fit", p_bits=12),
    "fit+sink+p13":          dict(_RF, sink=True, fmt="fit", p_bits=13),
    "fit+sink+hattn":        dict(_RF, sink=True, fmt="fit", hattn=True),
    "pow2+sink":             dict(_RF, sink=True, fmt="pow2"),
    "pow2+sink+p12":         dict(_RF, sink=True, fmt="pow2", p_bits=12),
    "pow2+sink+p13":         dict(_RF, sink=True, fmt="pow2", p_bits=13),
    "pow2+p12":              dict(_RF, fmt="pow2", p_bits=12),
    "pow2_tensor+sink+p12":  dict(_RF, sink=True, fmt="pow2", p_bits=12, per_tensor=True),
    "pow2+sink+p12+emb_q88": dict(_RF, emb="q88", sink=True, fmt="pow2", p_bits=12),
    "pow2+sink+p12+ff":      dict(_RF, sink=True, fmt="pow2", p_bits=12, floor_fix=True),
    "pow2+sink+p12+in7":     dict(_RF, sink=True, fmt="pow2", p_bits=12, in_cap=7),
    "pow2+sink+p12+in6":     dict(_RF, sink=True, fmt="pow2", p_bits=12, in_cap=6),
    "pow2+sink+p12+fw":      dict(_RF, sink=True, fmt="pow2", p_bits=12, float_weights=True),
    "pow2+sink+hattn":       dict(_RF, sink=True, fmt="pow2", hattn=True),
}
DEFAULT_POLICIES = ("bf16,q88,res_float,res_float+sink,fit+sink,pow2+sink,pow2+sink+p12,pow2+sink+hattn,"
                    "pow2+p12,pow2_tensor+sink+p12,pow2+sink+p12+emb_q88,pow2+sink+p12+fw")
MARGIN = 2.0          # calibrated formats keep one bit of headroom over the calibration max


def fbits(mx, margin=MARGIN, lo=-8, hi=15):
    """largest f with mx * margin * 2^f <= 32767 (int16 raw), elementwise."""
    mx = np.asarray(mx, np.float64)
    with np.errstate(divide="ignore"):
        f = np.floor(np.log2(32767.0 / np.maximum(mx * margin, 1e-30)))
    return np.clip(f, lo, hi).astype(np.int64)


def _vec_or_scalar(v):
    v = np.asarray(v, np.int64)
    return int(v.flat[0]) if v.size and (v == v.flat[0]).all() else v


def make_formats(pol, ch, W, cfg):
    """Exponents from the float calibration maxima ch[(cls, l)] (per channel;
    "s" per head)."""
    if not pol or not pol.get("fmt"):
        return {}
    kind, per_tensor = pol["fmt"], pol.get("per_tensor", False)
    L, H, KV, HD = cfg.L, cfg.H, cfg.KV, cfg.HD
    grp = np.arange(H) // (H // KV)
    fmt = {}

    def rng(cls, l, cap):
        m = ch[(cls, l)]
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fbits(m)
        return np.minimum(f, cap) if cap is not None else f

    def heads(cls, l, n, cap):
        m = ch[(cls, l)].reshape(n, -1).max(1)
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fbits(m)
        return np.minimum(f, cap) if cap is not None else f

    out_cap = 8 if kind == "fit" else None      # kernel outputs: fit caps at Q8.8, pow2 uses the range
    in_cap = pol.get("in_cap", 8)               # host-written kernel inputs (x, x2, a, xf, q, k, V cache)
    for l in list(range(L)) + [L]:
        if l == L:
            fmt[("xf", L)] = rng("xf", L, in_cap)
            fmt[("logits", L)] = rng("logits", L, out_cap)
            continue
        for cls in ("x", "x2", "a"):
            fmt[(cls, l)] = rng(cls, l, in_cap)
        for cls in ("q0", "k0", "v", "o", "g", "u", "d"):
            fmt[(cls, l)] = rng(cls, l, out_cap)
        fq, fk = heads("q", l, H, in_cap), heads("k", l, KV, in_cap)
        if pol.get("hattn"):
            fmt[("q", l)], fmt[("k", l)] = fq, fk
            fmt[("pv", l)] = rng("pv", l, in_cap)
            continue
        # scores q.k must fit int16 at f_q + f_k - 8: lower f_q (per head)
        smax = ch[("s", l)] if not per_tensor else np.full(H, ch[("s", l)].max())
        fq = np.minimum(fq, fbits(smax) + 8 - fk[grp])
        fv = fmt[("v", l)] if np.ndim(fmt[("v", l)]) else np.full(KV * HD, fmt[("v", l)])
        pvm = ch[("pv", l)] if not per_tensor else np.full(H * HD, ch[("pv", l)].max())
        fpv_max = fbits(pvm).reshape(H, HD)
        if pol.get("p_bits"):
            # P at p_bits; the host re-rounds the V cache so that P.V (f_p + f_vc - 8) still fits
            fp = np.full(H, pol["p_bits"])
            lim = (fpv_max + 8 - fp[:, None]).reshape(KV, H // KV, HD).min(1).reshape(-1)
            fmt[("vc", l)] = np.minimum(np.minimum(fbits(ch[("v", l)]), in_cap), lim)
        else:
            fp = np.minimum(15 if kind == "pow2" else 8, (fpv_max - fv.reshape(KV, HD)[grp] + 8).min(1))
        fmt[("q", l)], fmt[("k", l)], fmt[("p", l)] = fq, fk, fp
    # every weight must fit int16 at fw[i][j] = f_out[j] + 8 - f_in[i]: lower f_out[j] where not
    # (LINEARS order: v is capped before o reads it)
    for l in range(L + 1):
        for key, (name, ci, co) in (LINEARS.items() if l < L else [("lm", (None, "xf", "logits"))]):
            Wt = (W[f"model.layers.{l}.{name}.weight"] if l < L else W["model.embed_tokens.weight"]).T
            if ci == "pv" and ("pv", l) not in fmt:
                fvc = np.asarray(fmt.get(("vc", l), fmt[("v", l)]))
                fvc = np.full(KV * HD, fvc) if fvc.ndim == 0 else fvc
                fa = (np.asarray(fmt[("p", l)])[:, None] + fvc.reshape(KV, HD)[grp] - 8).reshape(-1)
            else:
                fa = np.broadcast_to(np.asarray(fmt[(ci, l)]), (Wt.shape[0],))
            with np.errstate(divide="ignore"):
                lim = np.floor(np.log2(32767.0 / np.maximum(np.abs(Wt), 1e-30)))
            fo_max = (lim + fa[:, None] - 8).min(0).astype(np.int64)
            fmt[(co, l)] = np.minimum(np.broadcast_to(np.asarray(fmt[(co, l)]), fo_max.shape), fo_max)
    return {k: (_vec_or_scalar(v) if k[0] != "sh" else v) for k, v in fmt.items()}


# ------------------------------------------------------------------ study
def load_all(assets):
    cfg = Cfg(assets)
    W = load_safetensors(os.path.join(assets, "model.safetensors"))
    return cfg, W


def build_data(assets, tok, quick):
    """Chat prompts; held-out windows and the calibration text start with
    <|im_start|> (position 0 of every chat sequence, the attention sink) followed
    by plain text, so every sequence has the same position-0 token."""
    prompts = [(name, tok.encode(chatml(m))) for name, m in PROMPTS]
    text = open(os.path.join(assets, "heldout_wikitext2_test.txt"), encoding="utf-8").read()
    ids = tok.encode(text)
    win = 256 if quick else CTX
    nwin = 1 if quick else 3
    assert len(ids) >= (win - 1) * nwin, len(ids)
    held = [np.concatenate([[1], ids[i * (win - 1):(i + 1) * (win - 1)]]) for i in range(nwin)]
    ctext = open(os.path.join(assets, "calib_wikitext2_valid.txt"), encoding="utf-8").read()
    calib = [np.concatenate([[1], tok.encode(ctext)[:CTX - 1]])] + [tok.encode(chatml(m)) for m in CALIB_PROMPTS]
    if quick:
        prompts = prompts[:4]
    return prompts, held, calib


def run_policy(name, model, prompts, held, ref, max_new, log, do_greedy=True):
    """Evaluate one model.  ref None -> this is the float reference: returns it."""
    t0 = time.time()
    res = dict(policy=name)
    # greedy generations
    gens = greedy(model, [p for _, p in prompts], max_new) if do_greedy else None
    res["gens"] = gens
    if do_greedy:
        log(f"  [{name}] greedy {sum(len(g) for g in gens)} tokens  {time.time() - t0:.0f}s")
    # teacher-forced: prompt + float continuation, then held-out windows
    tf_seqs = [np.concatenate([p, np.array((ref or res)["gens"][i], np.int64)]) for i, (_, p) in enumerate(prompts)]
    top1, lps = [], []
    agree = collections.Counter()
    for i, sq in enumerate(tf_seqs):
        lg = teacher_forced(model, sq)
        am = lg.argmax(-1)
        lp = log_softmax(lg).astype(np.float32)
        top1.append(am)
        if ref is not None:
            fam = ref["tf_top1"][i]
            # position 0 (<|im_start|>, the sink) is excluded everywhere
            resp = np.arange(len(sq)) >= len(prompts[i][1]) - 1        # positions that generate the answer
            eq = (am == fam)[1:]
            agree["tf_all"] += int(eq.sum()); agree["tf_all_n"] += len(sq) - 1
            agree["tf_resp"] += int((am == fam)[resp].sum()); agree["tf_resp_n"] += int(resp.sum())
            top5 = np.argpartition(-lg, 5, axis=-1)[1:, :5]
            agree["tf_top5"] += int((top5 == fam[1:, None]).any(-1).sum())
            flp = ref["tf_lp"][i][1:].astype(np.float64)
            agree["tf_kl"] += float((np.exp(flp) * (flp - lp[1:])).sum())
        else:
            lps.append(lp)
    log(f"  [{name}] prompt set teacher-forced  {time.time() - t0:.0f}s")
    h_top1, h_lps, nll, n_nll = [], [], 0.0, 0
    for i, w in enumerate(held):
        lg = teacher_forced(model, w)
        am = lg.argmax(-1)
        lp = log_softmax(lg)
        nll -= float(lp[np.arange(1, len(w) - 1), w[2:]].sum()); n_nll += len(w) - 2   # positions >= 1
        h_top1.append(am)
        if ref is not None:
            fam = ref["h_top1"][i]
            agree["h_all"] += int((am == fam)[1:].sum()); agree["h_all_n"] += len(w) - 1
            top5 = np.argpartition(-lg, 5, axis=-1)[1:, :5]
            agree["h_top5"] += int((top5 == fam[1:, None]).any(-1).sum())
            flp = ref["h_lp"][i][1:].astype(np.float64)
            agree["h_kl"] += float((np.exp(flp) * (flp - lp[1:])).sum())
        else:
            h_lps.append(lp.astype(np.float32))
    log(f"  [{name}] held-out  {time.time() - t0:.0f}s")
    res["ppl"] = math.exp(nll / n_nll)
    if ref is None:
        res.update(tf_top1=top1, tf_lp=lps, h_top1=h_top1, h_lp=h_lps)
    else:
        res["agree"] = dict(agree)
    if ref is not None and gens is not None:
        match = []
        for g, fg in zip(gens, ref["gens"]):
            n = 0
            while n < min(len(g), len(fg)) and g[n] == fg[n]:
                n += 1
            match.append(n)
        res["match"] = match
        res["identical"] = sum(int(g == fg) for g, fg in zip(gens, ref["gens"]))
    res["stats"] = model.stats.summary()
    res["seconds"] = time.time() - t0
    return res


CLASS_DOC = [
    ("h", "residual stream (after each add)"), ("x", "RMSNorm (input) out"), ("q0", "q = x.Wq"),
    ("k0", "k = x.Wk"), ("v", "v = x.Wv (V cache)"), ("q", "RoPE(q)"), ("k", "RoPE(k) (K cache)"),
    ("s", "scores q.k (raw, before 1/8)"), ("p", "softmax P"), ("pv", "P.V"), ("o", "o_proj out"),
    ("x2", "RMSNorm (post-attn) out"), ("g", "gate = x2.Wg"), ("u", "up = x2.Wu"),
    ("a", "silu(gate)*up"), ("d", "down_proj out"), ("xf", "final RMSNorm out"), ("logits", "LM head")]


def cmd_study(args):
    assets = args.assets
    out_dir = args.out or os.path.join(os.path.dirname(assets), "study")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "study.log"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    cfg, W = load_all(assets)
    tok = Tok(assets)
    prompts, held, calib = build_data(assets, tok, args.quick)
    held = held[:args.windows]
    max_new = 32 if args.quick else args.max_new
    cs = rope_tables(cfg, 2 * CTX)
    log(f"study {time.strftime('%Y-%m-%d %H:%M:%S')}  prompts {len(prompts)} "
        f"(tokens {[len(p) for _, p in prompts]}), held-out {len(held)} x {len(held[0])}, max_new {max_new}")
    pols = args.policies.split(",")
    # float reference (always) + calibration
    fm = Model(W, cfg, None, cos_sin=cs)
    fm0 = Model(W, cfg, None, cos_sin=cs, base=fm)          # computes the position-0 sink
    ref = run_policy("float", fm, prompts, held, None, max_new, log)
    ref_stats = ref["stats"]
    cal = {}
    for sink in (False, True):                              # per-channel calibration maxima
        cm = Model(W, cfg, None, cos_sin=cs, base=fm, sink=sink, sink_model=fm0, record_ch=True)
        for c in calib:
            teacher_forced(cm, c)
        cal[sink] = cm.stats.ch
        del cm
    log(f"  calibration {time.time() - t0:.0f}s")
    results = {"float": {k: v for k, v in ref.items() if k not in ("tf_lp", "h_lp", "tf_top1", "h_top1")}}
    for name in pols:
        if name == "float":
            continue
        pol = POLICIES[name]
        fmt = make_formats(pol, cal[bool(pol.get("sink"))], W, cfg)
        m = Model(W, cfg, pol, fmt, cos_sin=cs, sink_model=fm0)
        r = run_policy(name, m, prompts, held, ref, max_new, log, do_greedy=not args.nogreedy)
        r["formats"] = {"@".join(str(x) for x in k): v for k, v in fmt.items()}
        r["weights_saturated"] = m.wsat
        results[name] = r
        del m
        a = r["agree"]
        log(f"  [{name}] top1 tf {a['tf_all'] / a['tf_all_n']:.4f} resp {a['tf_resp'] / a['tf_resp_n']:.4f} "
            f"held {a['h_all'] / a['h_all_n']:.4f}  KL tf {a['tf_kl'] / a['tf_all_n']:.4f} held {a['h_kl'] / a['h_all_n']:.4f}  "
            f"ppl {r['ppl']:.3f} (float {ref['ppl']:.3f})  identical {r.get('identical')}/{len(prompts)}  match {r.get('match')}")
    json.dump(dict(results=results, prompts=[n for n, _ in prompts],
                   prompt_tokens=[int(len(p)) for _, p in prompts],
                   heldout=[len(w) for w in held], max_new=max_new,
                   texts={name: [tok.decode(g) for g in r["gens"]] for name, r in results.items() if r.get("gens")}),
              open(os.path.join(out_dir, "results.json"), "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else int(o))
    report(results, prompts, tok, ref_stats, out_dir, log)
    log(f"total {time.time() - t0:.0f} s")


RANGE_POLICIES = ("q88", "res_float+sink", "pow2+sink+p12")


def report(results, prompts, tok, ref_stats, out_dir, log):
    names = [n for n in results if n != "float"]
    ref = results["float"]
    log("\n== metrics (vs the numpy float64 reference) ==")
    log(f"{'policy':20s} {'top1 all':>8s} {'top1 resp':>9s} {'top1 held':>9s} {'top5 held':>9s} "
        f"{'KL held':>8s} {'ppl':>7s} {'ident':>5s} {'match (of 12 gens)':s}")
    log(f"{'float':20s} {'':8s} {'':9s} {'':9s} {'':9s} {'':8s} {ref['ppl']:7.3f}")
    for n in names:
        r = results[n]; a = r["agree"]
        log(f"{n:20s} {a['tf_all'] / a['tf_all_n']:8.4f} {a['tf_resp'] / a['tf_resp_n']:9.4f} "
            f"{a['h_all'] / a['h_all_n']:9.4f} {a['h_top5'] / a['h_all_n']:9.4f} {a['h_kl'] / a['h_all_n']:8.4f} "
            f"{r['ppl']:7.3f} {str(r.get('identical', '-')):>5s} {r.get('match', '')}")
    log("\n== ranges: max |x| before quantisation (layer of max), saturated fraction ==")
    hdr = f"{'class':8s} {'float max':>16s} {'>=128':>9s}"
    rnames = [n for n in names if n in RANGE_POLICIES] or names[:4]
    for n in rnames:
        hdr += f" | {n[:18]:>18s} max    sat%"
    log(hdr)
    for cls, doc in CLASS_DOC:
        if cls not in ref_stats:
            continue
        s = ref_stats[cls]
        line = f"{cls:8s} {s['max']:10.2f} (L{s['argmax_layer']:2d}) {100 * s['sat'] / max(s['n'], 1):8.4f}%"
        for n in rnames:
            t = results[n]["stats"].get(cls)
            line += f" | {t['max']:18.2f} {100 * t['sat'] / max(t['n'], 1):7.4f}" if t else " |" + " " * 30
        log(line + "   " + doc)
    log("\n== exponents used (f: value = int16 * 2^-f; 8 = Q8.8) ==")
    for n in names:
        fm = results[n].get("formats", {})
        if not fm:
            log(f"  [{n}] all 8"); continue
        byc = collections.defaultdict(collections.Counter)
        for k, v in fm.items():
            byc[k.split("@")[0] + ("@" + k.split("@")[1] if k.startswith("sh@") else "")].update(np.atleast_1d(v).tolist())
        log(f"  [{n}] weights saturated {results[n]['weights_saturated']}; " +
            "; ".join(f"{c}: " + ",".join(f"{e}x{cnt}" for e, cnt in sorted(ct.items())) for c, ct in sorted(byc.items())))
    for n in names:
        accs = {k: v for k, v in results[n]["stats"].items() if k.startswith("acc:")}
        if accs:
            log(f"  [{n}] max |acc| (Q16.16 units, wraps at 32768): " +
                ", ".join(f"{k[4:]} {v['max']:.1f}" for k, v in accs.items()) +
                f"; wrapped elements {sum(v['sat'] for v in accs.values())}")
    log("\n== residual stream max |h| per layer (float) ==")
    log(" ".join(f"{x:.0f}" for x in ref_stats["h"]["per_layer_max"][:-1]))
    # side by side
    with open(os.path.join(out_dir, "generations.txt"), "w") as f:
        for i, (pname, p) in enumerate(prompts):
            f.write(f"\n==== [{i}] {pname}  (prompt {len(p)} tokens)\n")
            f.write(tok.t.decode([int(x) for x in p], skip_special_tokens=False) + "\n")
            f.write(f"-- float ({len(ref['gens'][i])} tokens):\n{tok.decode(ref['gens'][i])}\n")
            for n in names:
                r = results[n]
                if not r.get("gens"):
                    continue
                f.write(f"-- {n} (match {r['match'][i]} / {len(r['gens'][i])} tokens):\n{tok.decode(r['gens'][i])}\n")
    log(f"\nside-by-side generations: {os.path.join(out_dir, 'generations.txt')}")


ABLATE_GROUPS = [("x",), ("q0", "k0"), ("q", "k"), ("s",), ("p",), ("v",), ("pv",), ("o",), ("x2",),
                 ("g", "u"), ("a",), ("d",), ("xf",), ("logits",)]


def cmd_ablate(args):
    """Where does the error come from?  Starting from a policy's formats, quantise
    one tensor class at a time (float weights, every other class exact), then
    the weights alone, the activations alone, and everything; teacher-forced
    metrics on the prompt set and the first held-out window."""
    assets = args.assets
    out_dir = args.out or os.path.join(os.path.dirname(assets), "study")
    os.makedirs(out_dir, exist_ok=True)
    cfg, W = load_all(assets)
    tok = Tok(assets)
    prompts, held, calib = build_data(assets, tok, args.quick)
    held = held[:1]
    cs = rope_tables(cfg, 2 * CTX)
    say = lambda m: print(m, flush=True)
    fm = Model(W, cfg, None, cos_sin=cs)
    fm0 = Model(W, cfg, None, cos_sin=cs, base=fm)
    ref = run_policy("float", fm, prompts, held, None, args.max_new, lambda m: None)
    base = dict(POLICIES[args.base])
    cm = Model(W, cfg, None, cos_sin=cs, base=fm, sink=bool(base.get("sink")), sink_model=fm0, record_ch=True)
    for c in calib:
        teacher_forced(cm, c)
    fmt = make_formats(base, cm.stats.ch, W, cfg)
    del cm
    variants = [("everything", {}), ("weights only", dict(qclasses=())), ("activations only", dict(float_weights=True))]
    variants += [("only " + "+".join(g), dict(qclasses=g, float_weights=True)) for g in ABLATE_GROUPS]
    say(f"ablation of {args.base}: prompt set (teacher-forced on float's answers) + held-out window 1 x {len(held[0])}")
    say(f"{'variant':24s} {'top1 tf':>8s} {'top1 held':>9s} {'KL tf':>8s} {'KL held':>8s} {'ppl':>7s}   (float ppl {ref['ppl']:.3f})")
    for name, extra in variants:
        m = Model(W, cfg, dict(base, **extra), fmt, cos_sin=cs, sink_model=fm0)
        r = run_policy(name, m, prompts, held, ref, args.max_new, lambda m: None, do_greedy=False)
        a = r["agree"]
        say(f"{name:24s} {a['tf_all'] / a['tf_all_n']:8.4f} {a['h_all'] / a['h_all_n']:9.4f} "
            f"{a['tf_kl'] / a['tf_all_n']:8.4f} {a['h_kl'] / a['h_all_n']:8.4f} {r['ppl']:7.3f}")
        del m


# ------------------------------------------------------------------ validation vs torch
def cmd_validate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(os.cpu_count())
    assets = args.assets
    cfg, W = load_all(assets)
    tok = Tok(assets)
    htok = AutoTokenizer.from_pretrained(assets)
    ok = 0
    for name, msgs in PROMPTS + [("calib", m) for m in CALIB_PROMPTS]:
        r = htok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        r = r["input_ids"] if not isinstance(r, list) else r
        ok += int(list(r) == tok.encode(chatml(msgs)).tolist())
    print(f"chat template + tokenizer: {ok}/{len(PROMPTS) + len(CALIB_PROMPTS)} prompts identical to HF apply_chat_template")
    hf = AutoModelForCausalLM.from_pretrained(assets, dtype=torch.float32, attn_implementation="eager").eval()
    fm = Model(W, cfg, None)
    prompts = [tok.encode(chatml(m)) for _, m in PROMPTS]
    text = open(os.path.join(assets, "heldout_wikitext2_test.txt"), encoding="utf-8").read()
    held = tok.encode(text)[:CTX]
    worst = 0.0
    for name, ids in [("prompt0", prompts[0]), ("prompt4", prompts[4]), ("heldout1024", held)]:
        with torch.no_grad():
            tl = hf(torch.tensor(ids[None])).logits[0].double().numpy()
        nl = teacher_forced(fm, ids)
        d = np.abs(tl - nl).max()
        worst = max(worst, d)
        print(f"{name:12s} T={len(ids):4d}  logits max|torch f32 - numpy f64| {d:.2e}  (max|logit| {np.abs(nl).max():.1f})"
              f"  argmax agree {(tl.argmax(-1) == nl.argmax(-1)).mean():.4f}")
    n_new = args.max_new
    t0 = time.time()
    ng = greedy(fm, prompts, n_new)
    same = 0
    for i, ids in enumerate(prompts):
        with torch.no_grad():
            o = hf.generate(torch.tensor(ids[None]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                            max_new_tokens=n_new, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
        tg = o[0, len(ids):].tolist()
        same += int(tg == ng[i])
        if tg != ng[i]:
            k = next((j for j in range(min(len(tg), len(ng[i]))) if tg[j] != ng[i][j]), min(len(tg), len(ng[i])))
            print(f"  greedy differs, prompt {i}: first difference at token {k} of {len(tg)} / {len(ng[i])}")
    print(f"greedy {n_new} tokens: torch == numpy float64 on {same}/{len(prompts)} prompts ({time.time() - t0:.0f} s)")
    print(f"max |logit diff| {worst:.2e}")


# ------------------------------------------------------------------ cost model
def cmd_costs(args):
    cfg = Cfg(args.assets)
    D, F, L, V, KV, HD, H = cfg.D, cfg.F, cfg.L, cfg.V, cfg.KV, cfg.HD, cfg.H
    per_layer = D * D + 2 * D * KV * HD + D * D + 3 * D * F
    lin = L * per_layer
    emb = V * D
    print(f"parameters: layers {lin / 1e6:.2f} M ({per_layer / 1e6:.3f} M/layer), embedding / LM head {emb / 1e6:.2f} M (tied), total {(lin + emb) / 1e6:.2f} M")
    print(f"weights at 16 bit: {2 * (lin + emb) / 1e6:.1f} MB; + a second (packed-B LM head) copy of the embedding {2 * emb / 1e6:.1f} MB")
    kv_tok = 2 * L * KV * HD * 2
    print(f"KV cache: {kv_tok} B/token ({kv_tok / 1024:.1f} KiB), context {CTX}: {kv_tok * CTX / 2**20:.1f} MiB")
    bw = 1.5e9
    for ctx in (64, 256, 512, 1024):
        mac = lin + emb + L * H * HD * ctx * 2
        byt = 2 * (lin + emb) + kv_tok * ctx
        print(f"decode @ ctx {ctx:4d}: {mac / 1e6:7.1f} M MAC, {byt / 1e6:6.1f} MB read -> {1e3 * byt / bw:5.0f} ms at 1.5 GB/s"
              f" (+ calls / host ops)")
    calls = L * (3 + KV * 2 + 1 + 3)   # q,k,v | per KV group QK^T + PV | o | gate, up, down
    print(f"decode kernel calls per token: {calls + 1} (q, k, v, {KV} x QK^T, {KV} x P.V, o, gate, up, down per layer + LM head); "
          f"host ops: {L * 5 + 2} (2 RMSNorm, RoPE, softmax, SiLU*up per layer + final norm + Gather)")
    for P in (64, 128, 256, 512):
        mac = P * lin + L * H * HD * P * P * 2 + emb     # LM head on the last position only
        print(f"prefill P={P:4d}: {mac / 1e9:6.2f} GMAC (linears {P * lin / 1e9:.2f}, attention {L * H * HD * P * P * 2 / 1e9:.2f} full square) "
              f"-> {mac / 40e9:5.2f} s at 40 GMAC/s (ConvKernel), {mac / 2.3e9:5.1f} s at 2.3 GMAC/s (MatmulKernel)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "validate", "study", "ablate", "costs"])
    ap.add_argument("--base", default="fit+sink", help="ablate: the policy whose formats are ablated")
    ap.add_argument("--assets", default=default_assets())
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--quick", action="store_true", help="4 prompts, 32 new tokens, one 256-token held-out window")
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", type=int, default=3, help="held-out windows (1024 tokens each)")
    ap.add_argument("--nogreedy", action="store_true", help="skip the policies' greedy generations")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study, "ablate": cmd_ablate,
     "costs": cmd_costs}[args.cmd](args)


if __name__ == "__main__":
    main()
