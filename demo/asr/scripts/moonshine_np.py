"""Moonshine in numpy float64: the study's float reference, validated against
transformers in asr_study.py validate.

  encode(cfg, W, pcm)  raw 16 kHz samples -> conv1 k127 s64 (no bias) -> tanh
                       -> GroupNorm(1 group) -> conv2 k7 s3 + GELU -> conv3 k3 s2
                       + GELU (one frame per 384 samples, 41.7 Hz) -> pre-LN
                       layers (LayerNorm without bias, RoPE on the first 32 of 36
                       head dims, interleaved pairs; GELU MLP) -> final LN
  Decoder(cfg, W, enc) RoPE self-attention with a KV cache, cross-attention over
                       the encoder states (no positions), SwiGLU MLP (fc1 ->
                       [x, gate] -> silu(gate) * x -> fc2), final LN, tied head
  greedy(...)          <s> until </s>, at most ~6.5 tokens per second of audio

The compute scales with the audio: no 30 s window.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from whisper_np import _erf, conv1d, softmax

SR = 16000
TOKENS_PER_SECOND = 6.5


def read_safetensors(path):
    """{name: float32 array} of an F32 / F16 / BF16 safetensors file, without the
    safetensors package (the scheduler's test venv has none)."""
    import struct
    out = {}
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), np.uint8)
            dt = info["dtype"]
            if dt == "F32":
                v = raw.view(np.float32)
            elif dt == "F16":
                v = raw.view(np.float16).astype(np.float32)
            elif dt == "BF16":
                v = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                raise ValueError(f"{k}: dtype {dt}")
            out[k] = v.reshape(info["shape"]).copy()
    return out


def load(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    W = {k.removeprefix("model."): v.astype(np.float64)
         for k, v in read_safetensors(os.path.join(model_dir, "model.safetensors")).items()}
    return cfg, W


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def layer_norm(x, g, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g


def head_dim(cfg, heads):
    return cfg["hidden_size"] // heads


def rope_tables(cfg, heads, positions):
    """cos / sin [T, rot / 2] for the interleaved pairs (2j, 2j + 1), frequency j."""
    rot = int(head_dim(cfg, heads) * cfg["partial_rotary_factor"])
    inv = 1.0 / (cfg["rope_theta"] ** (np.arange(0, rot, 2) / rot))
    a = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    return np.cos(a), np.sin(a)


def rope(x, heads, cs):
    """x [T, heads * hd]: rotate the pairs (2j, 2j + 1) of the first rot dims of each head."""
    cos, sin = cs
    T = x.shape[0]
    xh = x.reshape(T, heads, -1).copy()
    r = 2 * cos.shape[1]
    a, b = xh[:, :, 0:r:2].copy(), xh[:, :, 1:r:2].copy()
    xh[:, :, 0:r:2] = a * cos[:, None, :] - b * sin[:, None, :]
    xh[:, :, 1:r:2] = b * cos[:, None, :] + a * sin[:, None, :]
    return xh.reshape(T, -1)


def attention(q, k, v, heads, mask=None):
    Tq, d = q.shape
    hd = d // heads
    qh = q.reshape(Tq, heads, hd).transpose(1, 0, 2)
    kh = k.reshape(-1, heads, hd).transpose(1, 0, 2)
    vh = v.reshape(-1, heads, hd).transpose(1, 0, 2)
    s = (qh @ kh.transpose(0, 2, 1)) * hd ** -0.5
    if mask is not None:
        s = s + mask
    return (softmax(s) @ vh).transpose(1, 0, 2).reshape(Tq, d)


def frames(n_samples):
    t1 = (n_samples - 127) // 64 + 1
    t2 = (t1 - 7) // 3 + 1
    return (t2 - 3) // 2 + 1


# ------------------------------------------------------------------ encoder
def stem(cfg, W, pcm):
    """The conv stem: int16 (or float) samples -> [T, d]."""
    x = pcm.astype(np.float64) / (32768.0 if pcm.dtype == np.int16 else 1.0)
    h = np.tanh(conv1d(x[None, :], W["encoder.conv1.weight"], np.zeros(cfg["hidden_size"]), 64, 0))
    mu, var = h.mean(), h.var()
    h = (h - mu) / np.sqrt(var + 1e-5) * W["encoder.groupnorm.weight"][:, None] + W["encoder.groupnorm.bias"][:, None]
    h = gelu(conv1d(h, W["encoder.conv2.weight"], W["encoder.conv2.bias"], 3, 0))
    return gelu(conv1d(h, W["encoder.conv3.weight"], W["encoder.conv3.bias"], 2, 0)).T


def encode(cfg, W, pcm):
    h = stem(cfg, W, pcm)
    H = cfg["encoder_num_attention_heads"]
    cs = rope_tables(cfg, H, np.arange(h.shape[0]))
    for i in range(cfg["encoder_num_hidden_layers"]):
        p = f"encoder.layers.{i}."
        x = layer_norm(h, W[p + "input_layernorm.weight"])
        q = rope(x @ W[p + "self_attn.q_proj.weight"].T, H, cs)
        k = rope(x @ W[p + "self_attn.k_proj.weight"].T, H, cs)
        v = x @ W[p + "self_attn.v_proj.weight"].T
        h = h + attention(q, k, v, H) @ W[p + "self_attn.o_proj.weight"].T
        x = layer_norm(h, W[p + "post_attention_layernorm.weight"])
        f = gelu(x @ W[p + "mlp.fc1.weight"].T + W[p + "mlp.fc1.bias"])
        h = h + f @ W[p + "mlp.fc2.weight"].T + W[p + "mlp.fc2.bias"]
    return layer_norm(h, W["encoder.layer_norm.weight"])


# ------------------------------------------------------------------ decoder
class Decoder:
    def __init__(self, cfg, W, enc):
        self.cfg, self.W = cfg, W
        self.H = cfg["decoder_num_attention_heads"]
        self.L = cfg["decoder_num_hidden_layers"]
        self.xk = [enc @ W[f"decoder.layers.{i}.encoder_attn.k_proj.weight"].T for i in range(self.L)]
        self.xv = [enc @ W[f"decoder.layers.{i}.encoder_attn.v_proj.weight"].T for i in range(self.L)]
        d = cfg["hidden_size"]
        self.k = [np.zeros((0, d)) for _ in range(self.L)]
        self.v = [np.zeros((0, d)) for _ in range(self.L)]
        self.pos = 0
        self.ff = cfg["intermediate_size"]

    def step(self, ids):
        W = self.W
        n = len(ids)
        h = W["decoder.embed_tokens.weight"][ids]
        cs = rope_tables(self.cfg, self.H, np.arange(self.pos, self.pos + n))
        mask = np.triu(np.full((n, self.pos + n), -np.inf), self.pos + 1)
        for i in range(self.L):
            p = f"decoder.layers.{i}."
            x = layer_norm(h, W[p + "input_layernorm.weight"])
            q = rope(x @ W[p + "self_attn.q_proj.weight"].T, self.H, cs)
            self.k[i] = np.concatenate([self.k[i], rope(x @ W[p + "self_attn.k_proj.weight"].T, self.H, cs)])
            self.v[i] = np.concatenate([self.v[i], x @ W[p + "self_attn.v_proj.weight"].T])
            h = h + attention(q, self.k[i], self.v[i], self.H, mask) @ W[p + "self_attn.o_proj.weight"].T
            x = layer_norm(h, W[p + "post_attention_layernorm.weight"])
            q = x @ W[p + "encoder_attn.q_proj.weight"].T
            h = h + attention(q, self.xk[i], self.xv[i], self.H) @ W[p + "encoder_attn.o_proj.weight"].T
            x = layer_norm(h, W[p + "final_layernorm.weight"])
            u = x @ W[p + "mlp.fc1.weight"].T + W[p + "mlp.fc1.bias"]
            h = h + (silu(u[:, self.ff:]) * u[:, : self.ff]) @ W[p + "mlp.fc2.weight"].T + W[p + "mlp.fc2.bias"]
        self.pos += n
        x = layer_norm(h[-1], W["decoder.norm.weight"])
        return x @ W["decoder.embed_tokens.weight"].T


def max_tokens(n_samples):
    """The model card's generation limit (the prompt token included), capped by the positions."""
    return min(int(n_samples / SR * TOKENS_PER_SECOND) + 1, 194)


def greedy(cfg, dec, n_samples, first_logits=None):
    eos = cfg["eos_token_id"]
    logits = dec.step([cfg["decoder_start_token_id"]]) if first_logits is None else first_logits
    out = []
    while len(out) + 1 < max_tokens(n_samples):
        t = int(np.argmax(logits))
        if t == eos:
            break
        out.append(t)
        logits = dec.step([t])
    return out
