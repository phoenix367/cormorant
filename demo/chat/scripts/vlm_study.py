#!/usr/bin/env python3
"""B0 numeric study for SmolVLM-256M-Instruct on the KV260 Q8.8 datapath.

The VLM = a SigLIP-style vision encoder (12 layers, 768 wide, FFN 3072, 12
heads of 64, GELU tanh, LayerNorm; a 512 x 512 image in 16 x 16 patches ->
1024 tokens), a connector (pixel shuffle x4 -> 64 tokens of 12288, one
linear 12288 -> 576) and a Llama text model shaped like SmolLM2-135M (30
layers, 576 wide, 9 / 3 heads, untied LM head, vocab 49280) whose prompt
carries the 64 image features in place of the <image> tokens.  One tile
per image (do_image_splitting off: the image squared to 512 x 512).

A numpy float64 reference of the whole model (validated against transformers
in `validate`) and a bit-level emulation of the planned partition under
several numeric policies, compared with float on image prompts:

  * the image features (connector output): relative L2 error, cosine;
  * teacher-forced next-token agreement / KL on float's greedy answers;
  * greedy answers: exact-match length with float, full texts side by side;
  * value ranges per tensor class and saturation.

Emulated vision datapath (the conventions of llm_study.py — kernel MatMul:
raw int16 operands, exact sums in ap_fixed<32,16>, floor(acc / 2^8) and
saturate; host regions in double, round half to even + saturate on write):

  pixels          the uint8 values after the two LANCZOS resizes (exact integers,
                  exponent 0); the normalisation x = p * 2/255 - 1 is folded into
                  the patch-embedding weights (x 2/255) and bias (b - sum W)
  patch embed     kernel MatMul [1024, 768] x [768, 768] (the 16 x 16 x 3 patch
                  as one row: channel, then kernel row, then column); the host adds
                  float32(folded bias + position embedding) into the residual
  residual h      float32 host tensor (policy q88: Q8.8 VectorOP adds)
  LayerNorm       mean and variance left to right, (h - mu) / sqrt(var + 1e-6)
                  * gamma + beta, written at f_x / f_x2 / f_xf
  q, k, v         kernel MatMuls (q0, k0, v); the host adds the biases and writes
                  q, k at per-head exponents and V ("vc") per channel
  scores          kernel q.k^T per head; softmax on the host over all 1024 keys
                  (no mask) from the raw scores: e(k) = T_hi[k >> 8] * T_lo[k & 255]
                  (k = raw_max - raw; two 256-entry libm exp tables, sexp2_table),
                  P at 2^-p_bits; P.V kernel per head (f_pv = f_p + f_vc - 8)
  out_proj, fc2   kernel; the host adds the bias in the residual add
  fc1             kernel; the host adds the bias as an integer at fc1's exponent
                  (round half even, saturate) and reads GELU (tanh form, via libm
                  exp: y = x - x / (exp(2u) + 1)) from a 65 536-entry table per
                  exponent, written at f_a
  post-LN         host, written at f_xf; pixel shuffle (data movement: channel
                  c' of the 12288 has the exponent of c' mod 768)
  connector       kernel MatMuls [64, 4096] x [4096, 576] over the three K chunks
                  (K = 12288 exceeds every kernel's bound), each floored at f_img;
                  the host sums the three partial values (exact) into the image
                  features, the text model's float residual rows

Text model: llm_study.Model (policies there; the shipped pow2+sink+p12+mix),
image rows injected at the <image> positions.  Exponents come from float
calibration runs over separate images (and the WikiText-2 calibration text
for the text model), MARGIN one bit of headroom.

Usage (.venv-export + Pillow):
  PY=.venv-export/bin/python
  $PY demo/chat/scripts/vlm_study.py fetch       # checkpoint (pinned revision) + COCO images
  $PY demo/chat/scripts/vlm_study.py validate    # preprocessing, prompt ids, features, logits vs transformers
  $PY demo/chat/scripts/vlm_study.py study [--quick] [--combos ...]

Assets (not in git): demo/chat/assets/smolvlm-256m-instruct/ (Hugging Face
checkpoint, plus the WikiText-2 texts of llm_study.py fetch) and
demo/chat/assets/vlm_images/ (COCO val2017 images, CC-licensed by their
Flickr authors).  Results: demo/chat/assets/study/smolvlm-256m-instruct/.
Deterministic: greedy decoding only, fixed data.
"""
import argparse, collections, hashlib, json, math, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_study as ls  # noqa: E402

MODEL = "smolvlm-256m-instruct"
# the pinned inputs: checkpoint revision + SHA-256 per file, COCO val2017 images
INPUTS = json.load(open(os.path.join(HERE, "vlm_study_inputs.json")))
REPO_ID, REVISION = INPUTS["model"]["repo"], INPUTS["model"]["revision"]
FILES = tuple(INPUTS["model"]["files"])
ASSETS = os.path.join(HERE, "..", "assets")
IMAGES = os.path.join(ASSETS, "vlm_images")
COCO = "http://images.cocodataset.org/val2017/{:012d}.jpg"
# calibration and evaluation images are disjoint
CALIB_IDS = tuple(int(i) for i in INPUTS["images"]["calibration"])
EVAL_IDS = tuple(int(i) for i in INPUTS["images"]["evaluation"])
EVAL_PROMPTS = ("Can you describe this image?", "What is in this image? Answer briefly.",
                "What is happening in this image?")
CALIB_PROMPTS = ("Can you describe this image?", "What objects can you see in this image?")

BOS, IMAGE_TOK, FAKE_TOK, GLOBAL_TOK, EOU = 1, 49190, 49189, 49152, 49279
IMG_SEQ, SCALE = 64, 4
MAX_NEW = 96


# ------------------------------------------------------------------ assets
def default_assets():
    return os.path.normpath(os.path.join(ASSETS, MODEL))


def image_path(i):
    return os.path.join(IMAGES, f"{i:012d}.jpg")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch(args):
    """Download what is missing or different; every file checked against vlm_study_inputs.json."""
    import urllib.request
    os.makedirs(args.assets, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    want = [(os.path.join(args.assets, f), f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{f}", h)
            for f, h in INPUTS["model"]["files"].items()]
    want += [(image_path(int(i)), COCO.format(int(i)), h) for part in ("calibration", "evaluation")
             for i, h in INPUTS["images"][part].items()]
    bad = 0
    for p, url, h in want:
        if not os.path.exists(p) or sha256(p) != h:
            urllib.request.urlretrieve(url, p)
        ok = sha256(p) == h
        bad += not ok
        print(f"{os.path.basename(p):26s} {'ok' if ok else 'SHA-256 MISMATCH'}")
    if bad:
        sys.exit(f"{bad} file(s) differ from vlm_study_inputs.json")
    for name in ("heldout_wikitext2_test.txt", "calib_wikitext2_valid.txt"):
        src = os.path.join(ASSETS, ls.DEFAULT_MODEL, name)
        if not os.path.exists(os.path.join(args.assets, name)) and os.path.exists(src):
            shutil.copy2(src, args.assets)


# ------------------------------------------------------------------ preprocessing / prompt
def _max_len_size(h, w, max_len):
    """transformers' _resize_output_size_rescale_to_max_len (even short side)."""
    ar = w / h
    if w >= h:
        w, h = max_len, int(max_len / ar)
        h += h % 2
    else:
        h, w = max_len, int(max_len * ar)
        w += w % 2
    return max(h, 1), max(w, 1)


def load_pixels(path, size=512, longest=2048):
    """The Idefics3 PIL pipeline without image splitting: RGB, LANCZOS resize
    of the longest edge to 2048 (aspect kept, even short side; the 4096 cap
    never binds here), LANCZOS resize to size x size.  uint8 [size, size, 3]."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    h2, w2 = _max_len_size(h, w, longest)
    im = im.resize((w2, h2), Image.LANCZOS).resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.uint8)


def hf_pixel_values(pix):
    """transformers' rescale + normalize: float32((float32(p * 1/255) - 0.5) / 0.5), [3, H, W]."""
    x = (pix.astype(np.float64) * (1 / 255)).astype(np.float32).transpose(2, 0, 1)
    return (x - np.float32(0.5)) / np.float32(0.5)


def prompt_ids(tok, text):
    """The chat template with one image before the text, add_generation_prompt:
    <|im_start|>User:<fake><global-img><image>*64<fake>TEXT<end_of_utterance>\\nAssistant:"""
    s = ("<|im_start|>User:<fake_token_around_image><global-img>" + "<image>" * IMG_SEQ
         + "<fake_token_around_image>" + text + "<end_of_utterance>\nAssistant:")
    return tok.encode(s)


def text_prompt_ids(tok, text):
    return tok.encode("<|im_start|>User: " + text + "<end_of_utterance>\nAssistant:")


# ------------------------------------------------------------------ weights / configs
class TCfg(ls.Cfg):
    def __init__(self, c):                       # text_config
        self.L = c["num_hidden_layers"]; self.D = c["hidden_size"]; self.H = c["num_attention_heads"]
        self.KV = c["num_key_value_heads"]; self.F = c["intermediate_size"]; self.V = c["vocab_size"]
        self.HD = c.get("head_dim") or self.D // self.H
        self.eps = float(c["rms_norm_eps"]); self.theta = float(c["rope_theta"])
        self.tied = False


class VCfg:
    def __init__(self, c):                       # vision_config
        self.L = c["num_hidden_layers"]; self.D = c["hidden_size"]; self.H = c["num_attention_heads"]
        self.KV = self.H; self.HD = self.D // self.H; self.F = c["intermediate_size"]
        self.eps = float(c["layer_norm_eps"]); self.P = c["patch_size"]; self.S = c["image_size"]
        self.side = self.S // self.P; self.N = self.side ** 2


def load_all(assets):
    c = json.load(open(os.path.join(assets, "config.json")))
    W = ls.load_safetensors(os.path.join(assets, "model.safetensors"))
    Wt = {k.replace("model.text_model.", "model."): v for k, v in W.items() if k.startswith("model.text_model.")}
    Wt["lm_head.weight"] = W["lm_head.weight"]
    Wv = {k[len("model."):]: v for k, v in W.items() if not k.startswith("model.text_model.") and k != "lm_head.weight"}
    return TCfg(c["text_config"]), VCfg(c["vision_config"]), c["scale_factor"], Wt, Wv


# ------------------------------------------------------------------ vision model
# key: (weight name under encoder.layers.<l>., input class, output class)
VLIN = {"q": ("self_attn.q_proj", "x", "q0"), "k": ("self_attn.k_proj", "x", "k0"),
        "v": ("self_attn.v_proj", "x", "v"), "o": ("self_attn.out_proj", "pv", "o"),
        "f1": ("mlp.fc1", "x2", "f"), "f2": ("mlp.fc2", "a", "d")}
LV = "vision_model.encoder.layers."


def kernel_weights(Wv, vc):
    """Every vision / connector kernel MatMul's B operand, float64 [K, N] (as the
    kernel sees it: the patch embedding with the pixel normalisation folded in)."""
    K = {}
    for l in range(vc.L):
        for key, (name, _, _) in VLIN.items():
            K[(l, key)] = np.ascontiguousarray(Wv[f"{LV}{l}.{name}.weight"].T, np.float64)
    Wp = Wv["vision_model.embeddings.patch_embedding.weight"].astype(np.float64).reshape(vc.D, -1)
    # float32, as the frontend's ONNX initializer holds it
    K[(0, "pe")] = np.ascontiguousarray((Wp * (2.0 / 255.0)).astype(np.float32).astype(np.float64).T)
    K[(vc.L, "c")] = np.ascontiguousarray(Wv["connector.modality_projection.proj.weight"].T, np.float64)
    return K


def patch_bias_table(Wv):
    """float32(b - sum_i W[c][i] + pos[t][c]): the patch bias with the pixel
    normalisation folded in (the sum left to right) plus the position
    embedding — src/vit.py's table."""
    Wp = Wv["vision_model.embeddings.patch_embedding.weight"].astype(np.float64)
    Wp = Wp.reshape(Wp.shape[0], -1)
    b = Wv["vision_model.embeddings.patch_embedding.bias"].astype(np.float64) - np.cumsum(Wp, axis=1)[:, -1]
    pos = Wv["vision_model.embeddings.position_embedding.weight"].astype(np.float64)
    return (b[None, :] + pos).astype(np.float32).astype(np.float64)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


GELU_C = 0.7978845608028654          # sqrt(2 / pi) as a double literal (the C helper's)


def gelu_exp(x):
    """GELU, tanh form, through libm exp (the C helper's operation order):
    u = C * (x + 0.044715 * ((x * x) * x)), y = x - x / (exp(2u) + 1)
    (= 0.5 x (1 + tanh u); exp overflow -> inf -> y = x)."""
    u = GELU_C * (x + 0.044715 * ((x * x) * x))
    try:
        e = math.exp(2.0 * u)
    except OverflowError:
        e = math.inf
    return x - x / (e + 1.0)


_GELU_TAB = {}


def gelu_table(f):
    """gelu_exp(r * 2^-f) for every raw int16 r (index r + 32768)."""
    if f not in _GELU_TAB:
        d = 2.0 ** int(f)
        _GELU_TAB[f] = np.array([gelu_exp((i - 32768) / d) for i in range(65536)])
    return _GELU_TAB[f]


_SEXP2 = {}


def sexp2_table(f, scale):
    """The vision softmax's exp of k = raw_max - raw (k in [0, 65535]):
    e(k) = T_hi[k >> 8] * T_lo[k & 255] (a double product), T_hi[i] =
    exp(-(i * 256) / 2^f * scale), T_lo[j] = exp(-j / 2^f * scale) (libm exp):
    two 256-entry tables stay in the A53's L1 (the 65 536-entry table of the
    text model's softmax misses it).  Returned expanded to 65 536 entries."""
    key = (int(f), float(scale))
    if key not in _SEXP2:
        d = 2.0 ** int(f)
        hi = np.array([math.exp(-(i * 256) / d * scale) for i in range(256)])
        lo = np.array([math.exp(-j / d * scale) for j in range(256)])
        k = np.arange(65536)
        _SEXP2[key] = hi[k >> 8] * lo[k & 255]
    return _SEXP2[key]


def bias_raw(b, f):
    """A bias as raw integers at exponent(s) f: round half to even."""
    return np.round(np.asarray(b, np.float64) * p2v(f))


def vop_gelu_b(b, ff):
    """The VectorOP form of GELU(fc1 + b1) (doc/plans/OFFLOAD_PLAN.md §2.1), its two
    per-channel operands: the ADD's bias at fc1's exponent ff plus half an LSB of
    Q8.8 (2^(ff-9) for ff > 8: the MUL's floor becomes a round half up), and the
    MUL's scale 2^(8-ff) as a Q8.8 raw value 2^(16-ff)."""
    ff = np.asarray(ff, np.int64)
    half = np.where(ff > 8, np.power(2.0, ff - 9), 0.0)
    return bias_raw(b, ff) + half, np.power(2.0, 16 - ff)


def vop_gelu_ok(ff, fa, b):
    """The VectorOP form applies: GELU written at 2^-8 on every channel, the
    operands int16 integers (the scale 2^(16 - ff): 2 <= ff <= 16)."""
    if not np.all(np.asarray(fa) == 8):
        return False
    ba, sc = vop_gelu_b(b, ff)
    return bool(np.all(np.abs(ba) < 32768) and np.all((sc < 32768) & (sc >= 1)))


def vop_gelu(f, ff, b):
    """VectorOPKernel: s = sat16(raw + ba) (ADD), q = sat16(floor(s * sc / 2^8))
    (MUL), then act GELU_TANH — the exact GELU of q / 256 rounded to 2^-8
    (gelu_table(8) here; the host() write-back rounds it the kernel's way)."""
    ba, sc = vop_gelu_b(b, ff)
    s = np.clip(np.rint(f * p2v(ff)) + ba[None, :], -32768, 32767)
    q = np.clip(np.floor(s * sc[None, :] / 256.0), -32768, 32767).astype(np.int64)
    return gelu_table(8)[q + 32768]


def pixel_shuffle(x, s):
    """Idefics3Connector.pixel_shuffle for one image: [seq, D] -> [seq / s^2, D s^2]."""
    seq, D = x.shape
    n = int(round(seq ** 0.5))
    x = x.reshape(n, n, D).reshape(n, n // s, D * s).transpose(1, 0, 2)
    x = x.reshape(n // s, n // s, D * s * s).transpose(1, 0, 2)
    return x.reshape(seq // (s * s), D * s * s)


class VisionModel(ls.Model):
    """numpy vision encoder + connector (pol None: float64 reference).  Reuses
    llm_study.Model's primitives (E / host / kmm / fmm / softmax / resadd) with
    the vision tensor classes; layer index L holds the post-LN and connector."""

    def __init__(self, Wv, KW, vc, scale, pol=None, fmt=None, record_ch=False):
        self.cfg = vc
        self.scale = scale
        self.pol = pol or {}
        self.q = pol is not None
        self.fa_diag = bool(self.pol.get("float_acts"))
        self.qclasses = self.pol.get("qclasses")
        self.bf = bool(self.pol.get("bf16"))
        self.floor_fix = False
        self.hattn = bool(self.pol.get("hattn"))
        self.xattn_mode = False
        self.phase = "prefill"
        if self.bf:
            self.q = False
        self.fmt = fmt or {}
        self.record_ch = record_ch
        self.grp = np.arange(vc.H)
        self.stats = ls.Stats(vc.L)
        self.wsat = 0
        g = lambda n: Wv[n].astype(np.float64)
        self.ln1 = [(g(f"{LV}{l}.layer_norm1.weight"), g(f"{LV}{l}.layer_norm1.bias")) for l in range(vc.L)]
        self.ln2 = [(g(f"{LV}{l}.layer_norm2.weight"), g(f"{LV}{l}.layer_norm2.bias")) for l in range(vc.L)]
        self.lnf = (g("vision_model.post_layernorm.weight"), g("vision_model.post_layernorm.bias"))
        self.b = {(l, key): g(f"{LV}{l}.{name}.bias") for l in range(vc.L) for key, (name, _, _) in VLIN.items()}
        Wp = g("vision_model.embeddings.patch_embedding.weight").reshape(vc.D, -1)
        # folded normalisation: W (2p/255 - 1) + b = (2/255) W p + (b - W 1); + position embedding
        self.b0 = patch_bias_table(Wv)
        self.w = {k: self._enc(Wk, k[0], k[1], *self._io(k[1])) for k, Wk in KW.items()}
        self.conn_k = CONN_K

    def _enc(self, Wt, l, key, ci, co):
        qw = self.pol.get("qweights")                     # ablation: only these weights rounded
        if qw is None or key in qw or not self.q:
            return super()._enc(Wt, l, key, ci, co)
        K, N = Wt.shape
        return Wt * (p2v(self.Ev(co, l, N) + 8)[None, :] / p2v(self.Ev(ci, l, K))[:, None])

    @staticmethod
    def _io(key):
        return {"pe": ("pix", "pe"), "c": ("xs", "img")}.get(key) or VLIN[key][1:]

    def E(self, cls, l):
        if cls == "pix":
            return 0                                     # uint8 pixels: exact integers
        return super().E(cls, l)

    def lin(self, x, l, key):
        ci, co = self._io(key)
        if not self.q:
            return self.fmm(x, self.w[(l, key)], co, l)
        return self.kmm(x * p2v(self.Ev(ci, l, x.shape[1])), self.w[(l, key)], self.E(co, l), 8, co, l)

    def ln(self, h, gb, cls, l):
        gm, bt = gb
        D = h.shape[1]
        mu = np.cumsum(h, axis=-1)[:, -1] / D
        d = h - mu[:, None]
        var = np.cumsum(d * d, axis=-1)[:, -1] / D
        return self.host(d / np.sqrt(var + self.cfg.eps)[:, None] * gm + bt, cls, l, self.E(cls, l))

    def gelu(self, f, l):
        """GELU(fc1 + b1).  Emulation: the bias added to fc1's raw output as an
        integer at its exponent (round half even, saturate), then a 65 536-entry
        table per exponent (gelu_table) — the C helper, no libm call per value."""
        b = self.b[(l, "f1")]
        if not self.q or self.exact("f"):
            if self.record_ch:                              # f's exponent must hold f + b too
                self.stats.add_ch("f", l, np.abs(f + b).max(0))
            return self.host(gelu_tanh(f + b), "a", l, self.E("a", l))
        ff = self.Ev("f", l, f.shape[1])
        if self.pol.get("vop_gelu") and vop_gelu_ok(ff, self.Ev("a", l, f.shape[1]), b):
            return self.host(vop_gelu(f, ff, b), "a", l, self.E("a", l))
        raw = np.clip(np.rint(f * p2v(ff)) + bias_raw(b, ff)[None, :], -32768, 32767).astype(np.int64)
        y = np.empty(f.shape)
        for e in np.unique(ff):
            cols = ff == e
            y[:, cols] = gelu_table(int(e))[raw[:, cols] + 32768]
        return self.host(y, "a", l, self.E("a", l))

    def softmax(self, s, mask, fs, fp, layer):
        """llm_study.Model.softmax with the vision exp (sexp2_table) in the
        emulation; every key valid."""
        if not self.q or self.exact("s"):
            return super().softmax(s, mask, fs, fp, layer)
        raw = np.rint(s * p2(fs)).astype(np.int64)
        k = raw.max(-1, keepdims=True) - raw
        e = sexp2_table(int(fs), 1.0 / math.sqrt(self.cfg.HD))[k]
        return self.host(e / np.cumsum(e, -1)[..., -1:], "p", layer, fp, record=False)

    def heads_e(self, cls, l):
        return np.repeat(self.Eh(cls, l, self.cfg.H), self.cfg.HD)

    def attn(self, q0, k0, v0, l):
        vc = self.cfg
        N, H, HD = q0.shape[0], vc.H, vc.HD
        mask = np.ones((N, N), bool)
        out = np.empty((N, vc.D))
        if self.q and self.hattn:
            # one float host region (q / k / v unquantised inside), one write-back at f_pv
            q, k, v = q0 + self.b[(l, "q")], k0 + self.b[(l, "k")], v0 + self.b[(l, "v")]
            for h in range(H):
                sl = slice(h * HD, (h + 1) * HD)
                x = (q[:, sl] @ k[:, sl].T) / math.sqrt(HD)
                e = np.exp(x - x.max(-1, keepdims=True))
                out[:, sl] = (e / np.cumsum(e, -1)[..., -1:]) @ v[:, sl]
            return self.host(out, "pv", l, self.E("pv", l))
        q = self.host(q0 + self.b[(l, "q")], "q", l, self.heads_e("q", l))
        k = self.host(k0 + self.b[(l, "k")], "k", l, self.heads_e("k", l))
        v = self.host(v0 + self.b[(l, "v")], "vc", l, self.Ev("vc", l, vc.D))
        if self.q:
            fq, fk, fs = self.Eh("q", l, H), self.Eh("k", l, H), self.Eh("s", l, H)
            fp, fv, fpv = self.Eh("p", l, H), self.Ev("vc", l, vc.D), self.Ev("pv", l, vc.D)
        for h in range(H):
            sl = slice(h * HD, (h + 1) * HD)
            if not self.q:
                s = self.fmm(q[:, sl], k[:, sl].T, "s", l, rec=False)
                if self.record_ch:
                    self.stats.add_ch("s", l, np.abs(s).max(), h, H)
                p = self.softmax(s, mask, 0, 0, l)
                out[:, sl] = self.fmm(p, v[:, sl], "pv", l, sl, vc.D)
                continue
            s = self.kmm(q[:, sl] * p2(fq[h]), (k[:, sl] * p2(fk[h])).T, fs[h], 8, "s", l)
            p = self.softmax(s, mask, fs[h], fp[h], l)
            out[:, sl] = self.kmm(p * p2(fp[h]), v[:, sl] * p2(fv[sl])[None, :], fpv[sl], 8, "pv", l)
        return out

    def forward(self, pix):
        """pix uint8 [S, S, 3] -> image features [64, 576] (values as the text model reads them)."""
        vc = self.cfg
        n, P = vc.side, vc.P
        x = pix.astype(np.float64).reshape(n, P, n, P, 3).transpose(0, 2, 4, 1, 3).reshape(n * n, 3 * P * P)
        h = self.resadd(np.zeros((n * n, vc.D)), self.lin(x, 0, "pe") + self.b0, 0)
        for l in range(vc.L):
            xa = self.ln(h, self.ln1[l], "x", l)
            att = self.attn(self.lin(xa, l, "q"), self.lin(xa, l, "k"), self.lin(xa, l, "v"), l)
            h = self.resadd(h, self.lin(att, l, "o") + self.b[(l, "o")], l)
            x2 = self.ln(h, self.ln2[l], "x2", l)
            a = self.gelu(self.lin(x2, l, "f1"), l)
            h = self.resadd(h, self.lin(a, l, "f2") + self.b[(l, "f2")], l)
        xf = self.ln(h, self.lnf, "xf", vc.L)
        return self.connector(pixel_shuffle(xf, self.scale))

    def connector(self, xs):
        """The connector MatMul in K chunks of CONN_K (K = 12288 exceeds every
        kernel's bound): one kernel call per chunk, each output floored at f_img,
        the partial values summed exactly on the host (which dequantises them
        into the text model's float residual)."""
        L, K = self.cfg.L, xs.shape[1]
        parts = []
        for a in range(0, K, self.conn_k):
            b = min(K, a + self.conn_k)
            if not self.q:
                parts.append(self.fmm(xs[:, a:b], self.w[(L, "c")][a:b], "img", L))
                continue
            fin = self.Ev("xs", L, K)[a:b]
            parts.append(self.kmm(xs[:, a:b] * p2v(fin), self.w[(L, "c")][a:b], self.E("img", L), 8, "img", L))
        y = parts[0]
        for q in parts[1:]:
            y = y + q
        if self.record_ch and not self.q:
            self.stats.add_ch("img", L, np.abs(y).max(0))
        return y


def p2(e):
    return np.power(2.0, e)


def p2v(e):
    return np.power(2.0, np.asarray(e, np.float64))


# ------------------------------------------------------------------ vision formats
def make_vformats(pol, ch, KW, vc, scale):
    """Exponents from the float calibration maxima (llm_study.make_formats' rules
    for the vision classes)."""
    if not pol or not pol.get("fmt"):
        return {}
    kind, per_tensor = pol["fmt"], pol.get("per_tensor", False)
    L, H, HD, D = vc.L, vc.H, vc.HD, vc.D
    fmt = {}
    fbits = ls.fbits

    def rng(cls, l, cap):
        m = ch[(cls, l)]
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fbits(m)
        return np.minimum(f, cap) if cap is not None else f

    def heads(cls, l, cap):
        m = ch[(cls, l)].reshape(H, -1).max(1)
        if per_tensor:
            m = np.full_like(m, m.max())
        f = fbits(m)
        return np.minimum(f, cap) if cap is not None else f

    out_cap = 8 if kind == "fit" else None
    in_cap = pol.get("in_cap", 8)
    fmt[("pe", 0)] = rng("pe", 0, out_cap)
    for l in range(L):
        for cls in ("x", "x2", "a"):
            fmt[(cls, l)] = rng(cls, l, in_cap)
        for cls in ("q0", "k0", "v", "o", "f", "d"):
            fmt[(cls, l)] = rng(cls, l, out_cap)
        if pol.get("hattn"):
            fmt[("pv", l)] = rng("pv", l, in_cap)
            continue
        fq, fk = heads("q", l, in_cap), heads("k", l, in_cap)
        smax = ch[("s", l)] if not per_tensor else np.full(H, ch[("s", l)].max())
        fq = np.minimum(fq, fbits(smax) + 8 - fk)
        pvm = ch[("pv", l)] if not per_tensor else np.full(D, ch[("pv", l)].max())
        fpv_max = fbits(pvm).reshape(H, HD)
        fvc = rng("vc", l, in_cap)
        if pol.get("p_bits"):
            fp = np.full(H, pol["p_bits"])
            fvc = np.minimum(fvc, (fpv_max + 8 - fp[:, None]).reshape(-1))
        else:
            fp = np.minimum(15 if kind == "pow2" else 8, (fpv_max - fvc.reshape(H, HD) + 8).min(1))
        fmt[("q", l)], fmt[("k", l)], fmt[("p", l)], fmt[("vc", l)] = fq, fk, fp, fvc
    fmt[("xf", L)] = rng("xf", L, in_cap)
    fmt[("xs", L)] = np.tile(np.broadcast_to(fmt[("xf", L)], (D,)), scale * scale)
    fmt[("img", L)] = rng("img", L, out_cap)
    # weights must fit int16 at fw[i][j] = f_out[j] + 8 - f_in[i]: lower f_out[j] where not
    for (l, key), Wk in KW.items():
        ci, co = VisionModel._io(key)
        if ci == "pix":
            fa = np.zeros(Wk.shape[0], np.int64)
        elif ci == "pv" and ("pv", l) not in fmt:
            fa = (np.asarray(fmt[("p", l)])[:, None] + np.asarray(fmt[("vc", l)]).reshape(H, HD) - 8).reshape(-1)
        else:
            fa = np.broadcast_to(np.asarray(fmt[(ci, l)]), (Wk.shape[0],))
        with np.errstate(divide="ignore"):
            lim = np.floor(np.log2(32767.0 / np.maximum(np.abs(Wk), 1e-30)))
        fo_max = (lim + fa[:, None] - 8).min(0).astype(np.int64)
        fmt[(co, l)] = np.minimum(np.broadcast_to(np.asarray(fmt[(co, l)]), fo_max.shape), fo_max)
    return {k: ls._vec_or_scalar(v) for k, v in fmt.items()}


# ------------------------------------------------------------------ text model with image rows
class TextModel(ls.Model):
    img = None                                      # features for the sequence being run
    image_tok = IMAGE_TOK

    def embed(self, ids):
        h = self.emb[ids].astype(np.float64)
        m = ids == self.image_tok
        if m.any():
            assert self.img is not None and int(m.sum()) == len(self.img), (int(m.sum()), self.img is None)
            h[m] = self.img
        return h


def greedy(model, prompts, feats, max_new=MAX_NEW):
    """Greedy decoding, stop at <end_of_utterance>: prefill per prompt (its image
    features set), batched one-token decode steps."""
    seqs = [ls.Seq(model.cfg, len(p) + max_new) for p in prompts]
    nxt = []
    for s, p, f in zip(seqs, prompts, feats):
        model.img = f
        nxt.append(int(np.argmax(model.forward([s], [p])[0][-1])))
    model.img = None
    out = [[] for _ in prompts]
    active = list(range(len(prompts)))
    while active:
        for i in active:
            out[i].append(nxt[i])
        active = [i for i in active if nxt[i] != EOU and len(out[i]) < max_new]
        if not active:
            break
        lg = model.forward([seqs[i] for i in active], [np.array([nxt[i]]) for i in active], phase="decode")
        for i, x in zip(active, lg):
            nxt[i] = int(np.argmax(x[-1]))
    return out


def teacher_forced(model, ids, feat):
    model.img = feat
    try:
        return ls.teacher_forced(model, ids)
    finally:
        model.img = None


# ------------------------------------------------------------------ policies
_F = dict(residual="float")
VPOLICIES = {
    "float":           None,
    "bf16":            dict(bf16=True),
    "q88":             dict(residual="q88"),                       # the BERT partition: all Q8.8
    "res_float":       dict(_F),                                   # Q8.8 tensors, float residual
    "pow2":            dict(_F, fmt="pow2"),
    "pow2+p12":        dict(_F, fmt="pow2", p_bits=12),
    "pow2+p12+vgelu":  dict(_F, fmt="pow2", p_bits=12, vop_gelu=True),   # OFFLOAD_PLAN §2.1
    "pow2+p13":        dict(_F, fmt="pow2", p_bits=13),
    "pow2+p14":        dict(_F, fmt="pow2", p_bits=14),
    "pow2+p15":        dict(_F, fmt="pow2", p_bits=15),
    "pow2+p12+in6":    dict(_F, fmt="pow2", p_bits=12, in_cap=6),
    "pow2+p14+in7":    dict(_F, fmt="pow2", p_bits=14, in_cap=7),
    "pow2+p14+in6":    dict(_F, fmt="pow2", p_bits=14, in_cap=6),
    "pow2+p14+in5":    dict(_F, fmt="pow2", p_bits=14, in_cap=5),
    "pow2_tensor+p12": dict(_F, fmt="pow2", p_bits=12, per_tensor=True),
    "pow2+hattn":      dict(_F, fmt="pow2", hattn=True),
}
TEXT_SHIPPED = "pow2+sink+p12+mix"
TEXT_FORMATS = "pow2+sink+p12"        # its exponents (llm_study formats)
VISION_SHIPPED = "pow2+p12"           # the vision formats' policy (vision_formats_<it>.json)
VISION_VOP = "pow2+p12+vgelu"         # its formats, GELU on VectorOPKernel's activation unit
                                      # (doc/plans/OFFLOAD_PLAN.md §2.1): what the scheduler
                                      # generates where the platform has the unit


def vision_policy(activations: bool) -> str:
    """The vision policy the scheduler's `vision` entry computes: VISION_VOP on a
    VectorOPKernel with the activation unit, else VISION_SHIPPED (host GELU)."""
    return VISION_VOP if activations else VISION_SHIPPED
CAL_MAX_NEW = 96                      # the calibration answers' length (independent of --quick)
CONN_K = 4096                         # the connector's K chunk (MatmulKernel max_k)
# (vision policy, text policy); the first is the reference
COMBOS = [("float", "float"), ("bf16", "bf16"), ("q88", "float"), ("pow2+p12", "float"),
          ("pow2+p14+in7", "float"), ("pow2+hattn", "float"), ("float", TEXT_SHIPPED),
          ("pow2+p12", TEXT_SHIPPED), ("pow2+p14+in7", TEXT_SHIPPED)]


def combo_name(c):
    return f"V:{c[0]} T:{c[1]}"


# ------------------------------------------------------------------ study
def build_data(assets, tok, quick):
    ev = EVAL_IDS[:6] if quick else EVAL_IDS
    evals = [(i, EVAL_PROMPTS[k % len(EVAL_PROMPTS)]) for k, i in enumerate(ev)]
    calib = [(i, CALIB_PROMPTS[k % len(CALIB_PROMPTS)]) for k, i in enumerate(CALIB_IDS)]
    pix = {i: load_pixels(image_path(i)) for i, _ in evals + calib}
    ctext = open(os.path.join(assets, "calib_wikitext2_valid.txt"), encoding="utf-8").read()
    ctext_ids = np.concatenate([[BOS], tok.encode(ctext)[:ls.CTX - 1]])
    text = open(os.path.join(assets, "heldout_wikitext2_test.txt"), encoding="utf-8").read()
    win = 256 if quick else ls.CTX
    held = np.concatenate([[BOS], tok.encode(text)[:win - 1]])
    return evals, calib, pix, ctext_ids, held


def feat_metrics(f, ref):
    rel = [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(f, ref)]
    cos = [float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(f, ref)]
    return dict(rel_mean=float(np.mean(rel)), rel_max=float(np.max(rel)), cos_min=float(np.min(cos)))


def run_combo(name, vm, tm, evals, prompts, pix, ref, max_new, log, held=None):
    """ref None -> the float reference (returned with its logits / answers)."""
    t0 = time.time()
    res = dict(combo=name)
    feats = [vm.forward(pix[i]) for i, _ in evals]
    log(f"  [{name}] vision {len(feats)} images  {time.time() - t0:.0f}s")
    if ref is not None:
        res["feat"] = feat_metrics(feats, ref["feats"])
    gens = greedy(tm, prompts, feats, max_new)
    res["gens"] = gens
    log(f"  [{name}] greedy {sum(len(g) for g in gens)} tokens  {time.time() - t0:.0f}s")
    agree = collections.Counter()
    tf_top1, tf_lp = [], []
    for k, (p, f) in enumerate(zip(prompts, feats)):
        sq = np.concatenate([p, np.array((ref or res)["gens"][k], np.int64)])
        lg = teacher_forced(tm, sq, f)
        resp = np.arange(len(sq)) >= len(p) - 1          # the positions that generate the answer
        am, lp = lg.argmax(-1)[resp], ls.log_softmax(lg[resp])
        if ref is None:
            tf_top1.append(am); tf_lp.append(lp.astype(np.float32))
            continue
        fam, flp = ref["tf_top1"][k], ref["tf_lp"][k].astype(np.float64)
        agree["resp"] += int((am == fam).sum()); agree["n"] += len(am)
        top5 = np.argpartition(-lg[resp], 5, axis=-1)[:, :5]
        agree["top5"] += int((top5 == fam[:, None]).any(-1).sum())
        agree["kl"] += float((np.exp(flp) * (flp - lp)).sum())
    log(f"  [{name}] teacher-forced  {time.time() - t0:.0f}s")
    if ref is None:
        res.update(feats=feats, tf_top1=tf_top1, tf_lp=tf_lp)
    else:
        res["agree"] = dict(agree)
        match = []
        for g, fg in zip(gens, ref["gens"]):
            n = 0
            while n < min(len(g), len(fg)) and g[n] == fg[n]:
                n += 1
            match.append(n)
        res["match"] = match
        res["identical"] = sum(int(g == fg) for g, fg in zip(gens, ref["gens"]))
    if held is not None:
        lg = ls.teacher_forced(tm, held)
        lp = ls.log_softmax(lg)
        res["ppl"] = math.exp(-float(lp[np.arange(1, len(held) - 1), held[2:]].sum()) / (len(held) - 2))
    res["vstats"] = vm.stats.summary()
    res["tstats"] = tm.stats.summary()
    res["vwsat"], res["twsat"] = vm.wsat, tm.wsat
    res["seconds"] = time.time() - t0
    return res


def calibrate(tc, vc, scale, Wt, Wv, KW, tok, calib, pix, ctext_ids, cs, sinks=(False, True)):
    """Float runs over the calibration images (and the WikiText text for the
    text model): (float text model, its position-0 sink model, vision channel
    maxima, {sink: text channel maxima}).  The calibration answers are the
    float model's greedy answers (CAL_MAX_NEW tokens), teacher-forced."""
    vf = VisionModel(Wv, KW, vc, scale, record_ch=True)
    tf = TextModel(Wt, tc, None, cos_sin=cs)
    tf0 = TextModel(Wt, tc, None, cos_sin=cs, base=tf)          # the position-0 sink
    cal_feats = [vf.forward(pix[i]) for i, _ in calib]
    cal_prompts = [prompt_ids(tok, t) for _, t in calib]
    cal_answers = greedy(tf, cal_prompts, cal_feats, CAL_MAX_NEW)
    tcal = {}
    for sink in sinks:
        cm = TextModel(Wt, tc, None, cos_sin=cs, base=tf, sink=sink, sink_model=tf0, record_ch=True)
        for p, a, f in zip(cal_prompts, cal_answers, cal_feats):
            teacher_forced(cm, np.concatenate([p, np.array(a, np.int64)]), f)
        ls.teacher_forced(cm, ctext_ids)
        tcal[sink] = cm.stats.ch
        del cm
    return tf, tf0, vf.stats.ch, tcal


def formats_json(fmt, pol, name):
    return {"policy": name, "spec": pol, "margin": ls.MARGIN,
            "exponents": {f"{k[0]}@{k[1]}": np.asarray(v).tolist() for k, v in fmt.items()}}


def cmd_formats(args):
    """The calibrated exponents of the shipped policies: the text model's
    (llm_study formats JSON: exponents + the position-0 sink K / V at the cache
    exponents; src/llama.py Formats) and the vision encoder's."""
    assets = args.assets
    out_dir = args.out or ls.study_dir(assets)
    os.makedirs(out_dir, exist_ok=True)
    tc, vc, scale, Wt, Wv = load_all(assets)
    KW = kernel_weights(Wv, vc)
    tok = ls.Tok(assets)
    _evals, calib, pix, ctext_ids, _held = build_data(assets, tok, True)
    cs = ls.rope_tables(tc, 2 * ls.CTX)
    tpol, vpol = ls.POLICIES[TEXT_FORMATS], VPOLICIES[VISION_SHIPPED]
    tf, tf0, vch, tcal = calibrate(tc, vc, scale, Wt, Wv, KW, tok, calib, pix, ctext_ids, cs,
                                   sinks=(bool(tpol.get("sink")),))
    tfmt = ls.make_formats(tpol, tcal[bool(tpol.get("sink"))], Wt, tc)
    out = formats_json(tfmt, tpol, TEXT_FORMATS)
    m = TextModel(Wt, tc, tpol, tfmt, cos_sin=cs, sink_model=tf0)
    sq = ls.Seq(tc, 1)
    m._write_sink(sq, BOS)
    out["sink_token"] = BOS
    out["sink_k_raw"] = [np.rint(sq.k[l][:, 0] * p2v(m.Eh("k", l, tc.KV))[:, None]).astype(int).tolist()
                         for l in range(tc.L)]
    out["sink_v_raw"] = [np.rint(sq.v[l][:, 0] * p2v(m.Ev("vc", l, tc.KV * tc.HD).reshape(tc.KV, tc.HD)))
                         .astype(int).tolist() for l in range(tc.L)]
    tpath = os.path.join(out_dir, f"formats_{TEXT_FORMATS}.json")
    json.dump(out, open(tpath, "w"))
    vfmt = make_vformats(vpol, vch, KW, vc, scale)
    vout = formats_json(vfmt, vpol, VISION_SHIPPED)
    vout.update(pixel_exp=0, gelu="table", scale_factor=scale)
    vpath = os.path.join(out_dir, f"vision_formats_{VISION_SHIPPED}.json")
    json.dump(vout, open(vpath, "w"))
    print(f"{tpath}: {len(tfmt)} exponent entries\n{vpath}: {len(vfmt)} exponent entries")


def cmd_study(args):
    assets = args.assets
    out_dir = args.out or ls.study_dir(assets)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "study.log"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    tc, vc, scale, Wt, Wv = load_all(assets)
    KW = kernel_weights(Wv, vc)
    tok = ls.Tok(assets)
    evals, calib, pix, ctext_ids, held = build_data(assets, tok, args.quick)
    max_new = 48 if args.quick else args.max_new
    prompts = [prompt_ids(tok, t) for _, t in evals]
    log(f"study {time.strftime('%Y-%m-%d %H:%M:%S')}  eval {len(evals)} images, calibration {len(calib)} images "
        f"+ {len(ctext_ids)} WikiText tokens, max_new {max_new}, prompt tokens {len(prompts[0])}")
    cs = ls.rope_tables(tc, 2 * ls.CTX)
    tf, tf0, vch, tcal = calibrate(tc, vc, scale, Wt, Wv, KW, tok, calib, pix, ctext_ids, cs)
    log(f"  calibration {time.time() - t0:.0f}s")
    vfmts, tfmts = {}, {}
    vfloat = VisionModel(Wv, KW, vc, scale)
    ref = run_combo("float", vfloat, tf, evals, prompts, pix, None, max_new, log, held)
    results = {"float": {k: v for k, v in ref.items() if k not in ("feats", "tf_top1", "tf_lp")}}
    combos = [tuple(c.split("/")) for c in args.combos.split(",")] if args.combos else COMBOS[1:]
    for vp, tp in combos:
        name = combo_name((vp, tp))
        vpol, tpol = VPOLICIES[vp], ls.POLICIES[tp]
        if vp not in vfmts:
            vfmts[vp] = make_vformats(vpol, vch, KW, vc, scale)
        if tp not in tfmts:
            tfmts[tp] = ls.make_formats(tpol, tcal[bool((tpol or {}).get("sink"))], Wt, tc)
        vm = vfloat if vpol is None else VisionModel(Wv, KW, vc, scale, vpol, vfmts[vp])
        tm = tf if tpol is None else TextModel(Wt, tc, tpol, tfmts[tp], cos_sin=cs, sink_model=tf0)
        r = run_combo(name, vm, tm, evals, prompts, pix, ref, max_new, log, held if tpol else None)
        results[name] = r
        a, fm = r["agree"], r["feat"]
        log(f"  [{name}] features rel {fm['rel_mean']:.4f} (max {fm['rel_max']:.4f}) cos_min {fm['cos_min']:.5f}  "
            f"answer top1 {a['resp'] / a['n']:.4f} top5 {a['top5'] / a['n']:.4f} KL {a['kl'] / a['n']:.4f}  "
            f"identical {r['identical']}/{len(prompts)}  match {r['match']}"
            + (f"  ppl {r['ppl']:.3f}" if "ppl" in r else ""))
        if vm is not vfloat:
            del vm
    json.dump(dict(results=results, evals=[[int(i), t] for i, t in evals], prompt_tokens=len(prompts[0]),
                   max_new=max_new,
                   vformats={p: {"@".join(str(x) for x in k): v for k, v in f.items()} for p, f in vfmts.items()},
                   texts={n: [tok.decode(g) for g in r["gens"]] for n, r in results.items()}),
              open(os.path.join(out_dir, "results.json"), "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else int(o))
    report(results, evals, tok, out_dir, log)
    log(f"total {time.time() - t0:.0f} s")


VCLASS_DOC = [("h", "residual (after each add)"), ("pe", "patch embedding out"), ("x", "LN1 out"),
              ("q0", "q = x.Wq"), ("k0", "k = x.Wk"), ("v", "v = x.Wv"), ("q", "q + bq"), ("k", "k + bk"),
              ("vc", "V (v + bv)"), ("s", "scores q.k"), ("pv", "P.V"), ("o", "out_proj out"),
              ("x2", "LN2 out"), ("f", "fc1 out"), ("a", "GELU out"), ("d", "fc2 out"), ("xf", "post-LN out"),
              ("img", "connector out (image features)")]


def report(results, evals, tok, out_dir, log):
    names = [n for n in results if n != "float"]
    log("\n== metrics (vs the numpy float64 reference; answer positions only) ==")
    log(f"{'combo':34s} {'feat rel':>8s} {'max':>7s} {'cos min':>8s} {'top1':>7s} {'top5':>7s} {'KL':>7s} "
        f"{'ident':>5s} {'ppl':>7s}")
    ref = results["float"]
    log(f"{'float':34s} {'':8s} {'':7s} {'':8s} {'':7s} {'':7s} {'':7s} {'':5s} {ref.get('ppl', 0):7.3f}")
    for n in names:
        r = results[n]; a = r["agree"]; f = r["feat"]
        log(f"{n:34s} {f['rel_mean']:8.4f} {f['rel_max']:7.4f} {f['cos_min']:8.5f} {a['resp'] / a['n']:7.4f} "
            f"{a['top5'] / a['n']:7.4f} {a['kl'] / a['n']:7.4f} {r['identical']:5d} "
            + (f"{r['ppl']:7.3f}" if "ppl" in r else f"{'':7s}"))
    log("\n== vision value ranges (float): max |x| per class, argmax layer ==")
    st = ref["vstats"]
    for cls, doc in VCLASS_DOC:
        if cls in st:
            log(f"  {cls:5s} {st[cls]['max']:12.2f} (L{st[cls]['argmax_layer']:2d})  {doc}")
    log("  residual max |h| per layer: " + " ".join(f"{x:.0f}" for x in st["h"]["per_layer_max"]))
    th = ref["tstats"].get("h")
    if th:
        log("  text residual max |h| per layer (image prompts): " + " ".join(f"{x:.0f}" for x in th["per_layer_max"]))
    log("\n== saturated elements (emulated) ==")
    for n in names:
        r = results[n]
        vs = {k: v["sat"] for k, v in r["vstats"].items() if v["sat"] and not k.startswith("acc:")}
        acc = {k: round(v["max"], 1) for k, v in r["vstats"].items() if k.startswith("acc:")}
        wrapped = sum(v["sat"] for k, v in r["vstats"].items() if k.startswith("acc:"))
        log(f"  [{n}] vision weights saturated {r['vwsat']}; saturated {vs or 0}; wrapped acc {wrapped}"
            + (f"; max |acc| {acc}" if acc else ""))
    with open(os.path.join(out_dir, "generations.txt"), "w") as f:
        for k, (i, t) in enumerate(evals):
            f.write(f"\n==== [{k}] COCO {i}: {t}\n")
            f.write(f"-- float ({len(ref['gens'][k])} tokens):\n{tok.decode(ref['gens'][k])}\n")
            for n in names:
                r = results[n]
                f.write(f"-- {n} (match {r['match'][k]} / {len(r['gens'][k])}):\n{tok.decode(r['gens'][k])}\n")
    log(f"\nside-by-side answers: {os.path.join(out_dir, 'generations.txt')}")


# ------------------------------------------------------------------ ablation (vision)
VABLATE = [("pe",), ("x",), ("q0", "k0", "v"), ("q", "k"), ("s",), ("p",), ("vc",), ("pv",), ("o",),
           ("x2",), ("f",), ("a",), ("d",), ("xf",), ("img",)]


def cmd_ablate(args):
    """Feature error of the vision emulation with one tensor-class group quantised
    at a time (weights always encoded), weights alone, and everything but the
    weights (float_weights) — where the base policy's error comes from."""
    assets = args.assets
    tc, vc, scale, Wt, Wv = load_all(assets)
    KW = kernel_weights(Wv, vc)
    ids = CALIB_IDS
    vf = VisionModel(Wv, KW, vc, scale, record_ch=True)
    for i in ids:
        vf.forward(load_pixels(image_path(i)))
    ev = EVAL_IDS[:args.images]
    pix = [load_pixels(image_path(i)) for i in ev]
    vref = VisionModel(Wv, KW, vc, scale)
    ref = [vref.forward(p) for p in pix]
    bases = args.base.split(",")
    if len(bases) > 1:                                  # compare whole policies
        print(f"vision policies, {len(ev)} images: feature error vs float")
        for b in bases:
            vm = VisionModel(Wv, KW, vc, scale, VPOLICIES[b], make_vformats(VPOLICIES[b], vf.stats.ch, KW, vc, scale))
            f = feat_metrics([vm.forward(p) for p in pix], ref)
            print(f"  {b:24s} rel {f['rel_mean']:.4f} (max {f['rel_max']:.4f})  cos_min {f['cos_min']:.5f}"
                  f"  weights saturated {vm.wsat}", flush=True)
        return
    base = VPOLICIES[args.base]
    fmt = make_vformats(base, vf.stats.ch, KW, vc, scale)
    runs = [("all (" + args.base + ")", dict(base)), ("weights only", dict(base, qclasses=()))]
    runs += [("+".join(g), dict(base, qclasses=g)) for g in VABLATE]
    runs += [("all, float weights", dict(base, float_weights=True))]
    runs += [(f"weights {k} only", dict(base, qclasses=(), qweights=(k,))) for k in ("pe", "q", "k", "v", "o", "f1", "f2", "c")]
    if args.only:
        runs = [r for r in runs if any(o in r[0] for o in args.only.split(","))]
    print(f"vision ablation, base {args.base}, {len(ev)} images: feature error vs float")
    for name, pol in runs:
        vm = VisionModel(Wv, KW, vc, scale, pol, fmt)
        f = feat_metrics([vm.forward(p) for p in pix], ref)
        print(f"  {name:24s} rel {f['rel_mean']:.4f} (max {f['rel_max']:.4f})  cos_min {f['cos_min']:.5f}", flush=True)


# ------------------------------------------------------------------ validation vs transformers
def hf_processor(assets):
    from transformers import AutoTokenizer, Idefics3Processor
    from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
    pc = {k: v for k, v in json.load(open(os.path.join(assets, "preprocessor_config.json"))).items()
          if k not in ("image_processor_type", "processor_class")}
    return Idefics3Processor(image_processor=Idefics3ImageProcessorPil(**pc),
                             tokenizer=AutoTokenizer.from_pretrained(assets), image_seq_len=IMG_SEQ,
                             chat_template=json.load(open(os.path.join(assets, "chat_template.json")))["chat_template"])


def cmd_validate(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText
    torch.set_num_threads(os.cpu_count())
    assets = args.assets
    tc, vc, scale, Wt, Wv = load_all(assets)
    tok = ls.Tok(assets)
    proc = hf_processor(assets)
    ids_ok = pix_ok = 0
    items = [(i, EVAL_PROMPTS[k % len(EVAL_PROMPTS)]) for k, i in enumerate(EVAL_IDS)]
    for i, t in items:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
        inp = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True),
                   images=[Image.open(image_path(i))], return_tensors="np", do_image_splitting=False)
        ids_ok += int(inp["input_ids"][0].tolist() == prompt_ids(tok, t).tolist())
        pix_ok += int(np.array_equal(inp["pixel_values"][0, 0], hf_pixel_values(load_pixels(image_path(i)))))
    print(f"prompt ids identical to the processor's {ids_ok}/{len(items)}; pixel values identical {pix_ok}/{len(items)}")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
    r = proc.apply_chat_template(msgs, add_generation_prompt=True)
    print(f"text-only template: {r!r} == ours {tok.t.decode(text_prompt_ids(tok, 'What is the capital of France?').tolist(), skip_special_tokens=False) == r}")
    hf = AutoModelForImageTextToText.from_pretrained(assets, dtype=torch.float32, attn_implementation="eager").eval()
    KW = kernel_weights(Wv, vc)
    vm = VisionModel(Wv, KW, vc, scale)
    tm = TextModel(Wt, tc, None)
    same, worst_f, worst_l = 0, 0.0, 0.0
    n_img = 4
    for i, t in items[:n_img]:
        pix = load_pixels(image_path(i))
        pv = torch.tensor(hf_pixel_values(pix))[None, None]
        ids = prompt_ids(tok, t)
        with torch.no_grad():
            hfeat = hf.model.get_image_features(pixel_values=pv).pooler_output[0].double().numpy()
            o = hf.generate(input_ids=torch.tensor(ids[None]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                            pixel_values=pv, max_new_tokens=args.max_new, do_sample=False)
        hg = o[0, len(ids):].tolist()
        feat = vm.forward(pix)
        df = float(np.abs(feat - hfeat).max())
        sq = np.concatenate([ids, np.array(hg, np.int64)])
        with torch.no_grad():
            hl = hf(input_ids=torch.tensor(sq[None]), pixel_values=pv).logits[0].double().numpy()
        nl = teacher_forced(tm, sq, feat)
        dl = float(np.abs(hl - nl).max())
        ng = greedy(tm, [ids], [feat], args.max_new)[0]
        hg_t = hg[:hg.index(EOU) + 1] if EOU in hg else hg
        same += int(ng == hg_t)
        worst_f, worst_l = max(worst_f, df), max(worst_l, dl)
        print(f"COCO {i}: features max|torch f32 - numpy f64| {df:.2e} (max |f| {np.abs(hfeat).max():.1f});  "
              f"logits {dl:.2e}, argmax agree {(hl.argmax(-1) == nl.argmax(-1)).mean():.4f};  greedy "
              f"{'identical' if ng == hg_t else 'differs'} ({len(ng)} tokens)")
        print(f"   {tok.decode(ng)!r}")
    print(f"greedy {args.max_new} tokens: torch == numpy float64 on {same}/{n_img} images; "
          f"max |feature diff| {worst_f:.2e}, max |logit diff| {worst_l:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "validate", "study", "ablate", "formats"])
    ap.add_argument("--base", default="pow2+p12", help="ablate: the vision policy")
    ap.add_argument("--images", type=int, default=3, help="ablate: evaluation images")
    ap.add_argument("--only", default=None, help="ablate: run names containing one of these")
    ap.add_argument("--assets", default=default_assets())
    ap.add_argument("--combos", default=None, help="vision/text,... (default: all COMBOS)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--quick", action="store_true", help="6 images, 48 new tokens, a 256-token text window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    {"fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study, "ablate": cmd_ablate,
     "formats": cmd_formats}[args.cmd](args)


if __name__ == "__main__":
    main()
