# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Whisper (English-only checkpoints) in numpy float64: the study's float
reference, validated against transformers in asr_study.py validate.

  log_mel(pcm)         WhisperFeatureExtractor: the 30 s window, periodic Hann
                       400, hop 160, reflect padding, |STFT|^2, the checkpoint's
                       80 mel filters, log10, max - 8 clamp, (x + 4) / 4
  encode(W, mel)       conv1 k3 + GELU, conv2 k3 s2 + GELU, + sinusoidal
                       positions, pre-LN layers, final LN -> [1500, d]
  Decoder(W, enc)      learned positions, causal self-attention with a KV
                       cache, cross-attention over the encoder states (K / V
                       once per utterance), GELU FFN, final LN, tied head
  greedy(...)          <|startoftranscript|><|notimestamps|>, the suppress
                       lists of generation_config.json, until <|endoftext|>

GELU is the exact (erf) form; attention scales q by head_dim^-0.5.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

SR, N_FFT, HOP, N_SAMPLES = 16000, 400, 160, 480000


def load(model_dir):
    """(config, generation config, weights as float64 arrays, mel filters [201, 80])."""
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    gen = json.load(open(os.path.join(model_dir, "generation_config.json")))
    W = {k.removeprefix("model."): v.astype(np.float64)
         for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
    mel = np.array(json.load(open(os.path.join(model_dir, "preprocessor_config.json")))["mel_filters"],
                   np.float64).T                     # stored [80, 201]
    return cfg, gen, W, mel


# ------------------------------------------------------------------ features
def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)          # periodic


def power_spectrum(x):
    """|STFT|^2 of the 30 s window: [3000 frames, 201 bins] (the last frame dropped)."""
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx] * hann(N_FFT)[None, :]
    return (np.abs(np.fft.rfft(fr, axis=1)) ** 2)[:-1]


def log_mel(pcm, mel_filters):
    """[80, 3000] log-mel of int16 (or float in [-1, 1)) samples, padded / cut to 30 s."""
    x = pcm.astype(np.float64) / (32768.0 if pcm.dtype == np.int16 else 1.0)
    x = np.pad(x[:N_SAMPLES], (0, max(0, N_SAMPLES - len(x))))
    m = power_spectrum(x) @ mel_filters                                # [3000, 80]
    lm = np.log10(np.maximum(m, 1e-10)).T
    lm = np.maximum(lm, lm.max() - 8.0)
    return (lm + 4.0) / 4.0


# ------------------------------------------------------------------ blocks
try:
    from scipy.special import erf as _erf
except ImportError:                                  # .venv-export has torch, not scipy
    try:
        import torch

        def _erf(x):
            return torch.special.erf(torch.from_numpy(np.asarray(x, np.float64))).numpy()
    except ImportError:
        _erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x):
    """Exact (erf) GELU."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def conv1d(x, w, b, stride=1, pad=1):
    """x [C, T], w [O, C, K] -> [O, T']."""
    C, T = x.shape
    O, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    To = (T + 2 * pad - K) // stride + 1
    cols = np.stack([xp[:, k: k + stride * (To - 1) + 1: stride] for k in range(K)], 1)   # [C, K, To]
    return np.einsum("ock,ckt->ot", w, cols) + b[:, None]


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def attention(q, k, v, heads, mask=None):
    """q [Tq, d], k / v [Tk, d] -> [Tq, d] (q already scaled)."""
    Tq, d = q.shape
    hd = d // heads
    qh = q.reshape(Tq, heads, hd).transpose(1, 0, 2)
    kh = k.reshape(-1, heads, hd).transpose(1, 0, 2)
    vh = v.reshape(-1, heads, hd).transpose(1, 0, 2)
    s = qh @ kh.transpose(0, 2, 1)
    if mask is not None:
        s = s + mask
    return (softmax(s) @ vh).transpose(1, 0, 2).reshape(Tq, d)


def sinusoids(length, channels, max_timescale=10000):
    inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-inc * np.arange(channels // 2))
    t = np.arange(length)[:, None] * inv[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], 1)


def _lin(W, p, x, bias=True):
    y = x @ W[p + ".weight"].T
    return y + W[p + ".bias"] if bias and (p + ".bias") in W else y


# ------------------------------------------------------------------ encoder
def encode(cfg, W, mel, frames=None):
    """[80, 3000] log-mel -> [1500, d] encoder states; frames: only the first
    frames encoder positions (the first 2 * frames mel columns: a shortened
    window, whisper.cpp's audio_ctx — the model was trained on 30 s)."""
    if frames is not None:
        mel = mel[:, : 2 * frames]
    h = gelu(conv1d(mel, W["encoder.conv1.weight"], W["encoder.conv1.bias"], 1, 1))
    h = gelu(conv1d(h, W["encoder.conv2.weight"], W["encoder.conv2.bias"], 2, 1)).T
    h = h + W["encoder.embed_positions.weight"][: h.shape[0]]
    H = cfg["encoder_attention_heads"]
    scale = (cfg["d_model"] // H) ** -0.5
    for i in range(cfg["encoder_layers"]):
        p = f"encoder.layers.{i}."
        x = layer_norm(h, W[p + "self_attn_layer_norm.weight"], W[p + "self_attn_layer_norm.bias"])
        q = _lin(W, p + "self_attn.q_proj", x) * scale
        k = _lin(W, p + "self_attn.k_proj", x, bias=False)
        v = _lin(W, p + "self_attn.v_proj", x)
        h = h + _lin(W, p + "self_attn.out_proj", attention(q, k, v, H))
        x = layer_norm(h, W[p + "final_layer_norm.weight"], W[p + "final_layer_norm.bias"])
        h = h + _lin(W, p + "fc2", gelu(_lin(W, p + "fc1", x)))
    return layer_norm(h, W["encoder.layer_norm.weight"], W["encoder.layer_norm.bias"])


# ------------------------------------------------------------------ decoder
class Decoder:
    """One utterance: the cross K / V computed once, the self K / V cached per layer."""

    def __init__(self, cfg, W, enc):
        self.cfg, self.W = cfg, W
        self.H = cfg["decoder_attention_heads"]
        self.scale = (cfg["d_model"] // self.H) ** -0.5
        self.L = cfg["decoder_layers"]
        self.xk = [_lin(W, f"decoder.layers.{i}.encoder_attn.k_proj", enc, bias=False) for i in range(self.L)]
        self.xv = [_lin(W, f"decoder.layers.{i}.encoder_attn.v_proj", enc) for i in range(self.L)]
        self.k = [np.zeros((0, cfg["d_model"])) for _ in range(self.L)]
        self.v = [np.zeros((0, cfg["d_model"])) for _ in range(self.L)]
        self.pos = 0

    def step(self, ids):
        """Feed tokens (one or more), return the logits of the last one."""
        W = self.W
        n = len(ids)
        h = W["decoder.embed_tokens.weight"][ids] + W["decoder.embed_positions.weight"][self.pos: self.pos + n]
        mask = np.triu(np.full((n, self.pos + n), -np.inf), self.pos + 1)
        for i in range(self.L):
            p = f"decoder.layers.{i}."
            x = layer_norm(h, W[p + "self_attn_layer_norm.weight"], W[p + "self_attn_layer_norm.bias"])
            q = _lin(W, p + "self_attn.q_proj", x) * self.scale
            self.k[i] = np.concatenate([self.k[i], _lin(W, p + "self_attn.k_proj", x, bias=False)])
            self.v[i] = np.concatenate([self.v[i], _lin(W, p + "self_attn.v_proj", x)])
            h = h + _lin(W, p + "self_attn.out_proj", attention(q, self.k[i], self.v[i], self.H, mask))
            x = layer_norm(h, W[p + "encoder_attn_layer_norm.weight"], W[p + "encoder_attn_layer_norm.bias"])
            q = _lin(W, p + "encoder_attn.q_proj", x) * self.scale
            h = h + _lin(W, p + "encoder_attn.out_proj", attention(q, self.xk[i], self.xv[i], self.H))
            x = layer_norm(h, W[p + "final_layer_norm.weight"], W[p + "final_layer_norm.bias"])
            h = h + _lin(W, p + "fc2", gelu(_lin(W, p + "fc1", x)))
        self.pos += n
        x = layer_norm(h[-1], W["decoder.layer_norm.weight"], W["decoder.layer_norm.bias"])
        return x @ W["decoder.embed_tokens.weight"].T


def prompt(gen):
    """<|startoftranscript|> then the forced ids (English-only: <|notimestamps|>)."""
    ids = [gen["decoder_start_token_id"]]
    for pos, tok in sorted(gen.get("forced_decoder_ids") or []):
        assert pos == len(ids)
        ids.append(tok)
    return ids


def window_frames(n_samples, bucket):
    """Encoder frames of the shortened window: the audio rounded up to a bucket of
    `bucket` frames (50 frames = 1 s), at most 1500."""
    n = -(-n_samples // (2 * HOP))
    return min(1500, -(-n // bucket) * bucket)


def greedy(cfg, gen, dec, max_new=440, first_logits=None):
    """Greedy decoding with the checkpoint's suppress lists; returns the generated ids (EOS excluded)."""
    sup = np.array(gen["suppress_tokens"])
    begin = np.array(gen["begin_suppress_tokens"])
    eos = gen["eos_token_id"]
    logits = dec.step(prompt(gen)) if first_logits is None else first_logits
    out = []
    while len(out) < max_new:
        lg = logits.copy()
        lg[sup] = -np.inf
        if not out:
            lg[begin] = -np.inf
        t = int(np.argmax(lg))
        if t == eos:
            break
        out.append(t)
        logits = dec.step([t])
    return out
