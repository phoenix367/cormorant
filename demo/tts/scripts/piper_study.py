#!/usr/bin/env python3
"""piper_study.py — model study of Piper en_US-lessac-medium (VITS) for the
KV260 (model-study skill, route C; doc/plans/TTS_PLAN.md §3).  Host only.

  fetch      the voice (ONNX + JSON) at the pinned revision, SHA-256 checked
  phonemize  the study's sentences -> espeak-ng IPA -> Piper phoneme ids
             (texts.json); run it with a Python that has phonemizer-fork +
             espeakng-loader (requirements-phonemize.txt)
  validate   the numpy float64 reference (piper_vits.py) against the ONNX in
             onnxruntime, noise off: text encoder stats, durations, waveform
  calibrate  power-of-two exponents per tensor from the float reference on
             the calibration sentences (Harvard list 2), one bit of headroom
             -> exponents.json
  study      the evaluation sentences (Harvard list 1 + a chat reply) under
             each policy against float with the same noise: waveform SNR,
             log-mel distance, saturation, accumulator peak; WAVs to listen
  encoder    the int16 text encoder of the library (TTS_PLAN §6): calibrate
             its exponents (-> exponents.json "encoder"), durations and
             log-mel distance against float, per softmax exponent (--p-exp)
  encoder-vsmx  the encoder's softmax on VectorOPKernel's unit (SOFTMAX_PLAN
             §5) against the host softmax, on the stored exponents: durations
             and log-mel distance against float (as encoder), the
             durations against the current library, the logits' saturation
             (-> study/encoder_vsmx.json; exponents.json untouched)
  costs      the flow and the decoder per second of audio priced by the
             bitstream's performance model (proxy graphs, in memory) + the
             host-side counts
  all        fetch validate calibrate study costs (phonemize first; encoder apart)

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/piper_study.py
           [--assets DIR] [--perf-model FILE] CMD
       /mnt/data/tts_venv/bin/python demo/tts/scripts/piper_study.py phonemize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCHED = os.path.join(REPO, "inference-scheduler")
sys.path.insert(0, HERE)

REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/en/en_US/lessac/medium"
FILES = {
    "en_US-lessac-medium.onnx": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    "en_US-lessac-medium.onnx.json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
}

# Harvard sentences (IEEE 1969, public domain): list 1 evaluates, list 2 calibrates.
EVAL = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "Hello! I am a small assistant running on an FPGA board. The answer is forty two, "
    "and the capital of France is Paris.",
]
CALIB = [
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
]
HEADROOM_BITS = 1
F_RANGE = (-10, 24)


def default_assets() -> str:
    return os.path.join(REPO, "demo", "tts", "assets", "piper-lessac-medium")


def _onnx(a):
    return os.path.join(a.assets, "en_US-lessac-medium.onnx")


# ---- fetch / phonemize --------------------------------------------------------- #

def cmd_fetch(a) -> int:
    os.makedirs(a.assets, exist_ok=True)
    bad = 0
    for name, sha in FILES.items():
        p = os.path.join(a.assets, name)
        if not os.path.exists(p):
            print(f"  downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", p)
        ok = hashlib.sha256(open(p, "rb").read()).hexdigest() == sha
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name}")
    print(f"rhasspy/piper-voices @ {REVISION} en_US/lessac/medium (MIT; dataset: Blizzard 2013 Lessac)")
    return 1 if bad else 0


def cmd_phonemize(a) -> int:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    from phonemizer.backend import EspeakBackend
    cfg = json.load(open(_onnx(a) + ".json"))
    pm = cfg["phoneme_id_map"]
    be = EspeakBackend(cfg["espeak"]["voice"], preserve_punctuation=True, with_stress=True,
                       language_switch="remove-flags")
    out = {}
    for group, texts in (("eval", EVAL), ("calib", CALIB)):
        for i, t in enumerate(texts):
            ph = be.phonemize([t])[0].strip()
            ids = [pm["^"][0]]
            for ch in ph:                                   # Piper: id, pad, id, pad ...
                if ch in pm:
                    ids += [pm[ch][0], pm["_"][0]]
            ids += [pm["$"][0]]
            out[f"{group}{i:02d}"] = {"text": t, "phonemes": ph, "ids": ids}
    json.dump(out, open(os.path.join(a.assets, "texts.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(out)} sentences -> {os.path.join(a.assets, 'texts.json')}")
    return 0


def _texts(a, group):
    d = json.load(open(os.path.join(a.assets, "texts.json")))
    return {k: v for k, v in d.items() if k.startswith(group)}


# ---- validate -------------------------------------------------------------------- #

def cmd_validate(a) -> int:
    import onnx
    import onnxruntime as ort
    import piper_vits as pv
    from onnx import helper as oh
    W = pv.load_weights(_onnx(a))
    m = onnx.load(_onnx(a))
    for n in ("/enc_p/proj/Conv_output_0", "/Ceil_output_0", "/Mul_7_output_0"):
        m.graph.output.append(oh.make_empty_tensor_value_info(n))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    worst = 0.0
    for name, t in _texts(a, "eval").items():
        ids = t["ids"]
        feeds = {"input": np.array([ids], np.int64), "input_lengths": np.array([len(ids)], np.int64),
                 "scales": np.array([0.0, 1.0, 0.0], np.float32)}
        wav_o, stats_o, ceil_o, z_o = sess.run(None, feeds)
        wav, parts = pv.synthesize(W, ids, noise_scale=0.0, noise_w=0.0, return_parts=True)
        stats = np.concatenate([parts["m_p"], parts["logs_p"]])
        d_stats = float(np.abs(stats - stats_o[0]).max())
        same_dur = np.array_equal(parts["w_ceil"], ceil_o.reshape(-1).astype(int))
        wo = wav_o.reshape(-1)
        n = min(len(wo), len(wav))
        d_z = float(np.abs(parts["z"] - z_o[0][:, :parts["z"].shape[1]]).max()) if same_dur else float("nan")
        d_w = float(np.abs(wav[:n] - wo[:n]).max())
        snr = 10 * math.log10(float(np.sum(wo[:n] ** 2)) / max(float(np.sum((wav[:n] - wo[:n]) ** 2)), 1e-30))
        worst = max(worst, d_w)
        print(f"  {name}: {len(ids):3d} ids, {n / pv.SR:5.2f} s  stats {d_stats:.1e}  durations "
              f"{'equal' if same_dur else 'DIFFER'}  z {d_z:.1e}  waveform max {d_w:.1e} (SNR {snr:.0f} dB)")
    print(f"float64 reference vs onnxruntime float32: waveform max |diff| {worst:.1e}")
    return 0


# ---- calibrate --------------------------------------------------------------------- #

def _best_input_exponent(pv, fy, items):
    """The input exponent f_x of one conv that minimises its output error on
    the calibration slices: f_w = f_y + 8 - f_x, so a coarser input buys a
    finer weight grid (ConvKernel's output shift is fixed at 8)."""
    w = items[0][2]
    fmax = max(float(np.abs(x).max()) for _, x, *_ in items) or 1.0
    f_nat = int(math.floor(math.log2(2 ** (15 - HEADROOM_BITS) / fmax)))
    best, best_err = f_nat, None
    for fx in range(f_nat - 12, f_nat + 1):
        fw = fy + 8 - fx
        if float(np.abs(w).max()) * 2.0 ** fw >= 32767.5:
            continue
        Q = pv.Quant({"x": fx, "y": fy})
        err = 0.0
        for _, x, w_, b, dil, pad, tr in items:
            ref = pv.kconv(None, "y", x, "x", w_, b, dil=dil, pad=pad, transpose=tr)
            got = pv.kconv(Q, "y", Q.q("x", x), "x", w_, b, dil=dil, pad=pad, transpose=tr)
            err += float(np.mean((got - ref) ** 2))
        if best_err is None or err < best_err:
            best, best_err = fx, err
    return best


def cmd_calibrate(a) -> int:
    import piper_vits as pv
    W = pv.load_weights(_onnx(a))
    Q = pv.Quant({})
    Q.record, Q.convs = {}, []
    for name, t in _texts(a, "calib").items():
        pv.synthesize(W, t["ids"], seed=zlib.crc32(name.encode()), Q_flow=Q, Q_dec=Q)
    exp = {}
    for k, v in Q.record.items():
        f = F_RANGE[1] if v == 0 else int(math.floor(math.log2(2 ** (15 - HEADROOM_BITS) / v)))
        exp[k] = int(min(max(f, F_RANGE[0]), F_RANGE[1]))
    by = {}
    for item in Q.convs:
        by.setdefault(item[0], []).append(item)
    moved = []
    for name, items in by.items():
        fx = _best_input_exponent(pv, exp[name], items)
        exp[name + "#in"] = fx
        moved.append(fx)
    path = os.path.join(a.assets, "exponents.json")
    keep = {k: v for k, v in (json.load(open(path)) if os.path.exists(path) else {}).items()
            if k.startswith("encoder")}                       # cmd_encoder's part
    json.dump({"exponents": exp, "max_abs": Q.record, "headroom_bits": HEADROOM_BITS, **keep},
              open(path, "w"), indent=1)
    fs = sorted(v for k, v in exp.items() if not k.endswith("#in"))
    print(f"{len(fs)} tensors, exponents {fs[0]}..{fs[-1]} (median {fs[len(fs) // 2]}); "
          f"largest |x| {max(Q.record.values()):.1f}; {len(moved)} conv input exponents searched "
          f"-> {os.path.join(a.assets, 'exponents.json')}")
    return 0


# ---- study -------------------------------------------------------------------------- #

def _mel_db(wav, sr=22050, n_fft=1024, hop=256, n_mels=80):
    win = np.hanning(n_fft + 1)[:-1]
    x = np.pad(wav, n_fft // 2, mode="reflect")
    n = 1 + (len(x) - n_fft) // hop
    fr = np.stack([x[i * hop: i * hop + n_fft] * win for i in range(n)])
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)                     # noqa: E731
    mel = lambda h: 2595 * np.log10(1 + h / 700)                     # noqa: E731
    pts = hz(np.linspace(mel(0), mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return 10 * np.log10(spec @ fb.T + 1e-10)


def _lsd(m, mref, floor_db=60.0):
    """Mean |log-mel difference| (dB) with both floored at the reference's
    peak - floor_db: pauses below that are inaudible next to the speech and
    would otherwise dominate the mean."""
    fl = mref.max() - floor_db
    return float(np.mean(np.abs(np.maximum(m, fl) - np.maximum(mref, fl))))


def _pause_floor(ref, err, frame=1024):
    """The error's level (dBFS) in the reference's pauses (frames below -50 dBFS)."""
    n = len(ref) // frame
    r = ref[: n * frame].reshape(n, frame)
    e = err[: n * frame].reshape(n, frame)
    quiet = np.sqrt((r ** 2).mean(1)) < 10 ** (-50 / 20)
    if not quiet.any():
        return float("nan")
    return 20 * math.log10(max(float(np.sqrt((e[quiet] ** 2).mean())), 1e-12))


def _write_wav(path, wav, sr=22050):
    pcm = np.clip(np.round(wav * 32767), -32768, 32767).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" +
                struct.pack("<I", len(pcm)) + pcm)


POLICIES = {
    # name: (flow policy, decoder policy); a policy is None (float), "bf16",
    # "q88" (every tensor at 2^-8, the scheduler's default), "pow2"
    # (calibrated per-tensor exponents), "opt" (pow2 + each conv's input
    # written at its searched exponent, the flow's residual chain in float on
    # the host); "+fw": float weights (what the weight grid costs)
    "bf16": ("bf16", "bf16"),
    "q88": ("q88", "q88"),
    "pow2": ("pow2", "pow2"),
    "opt": ("opt", "opt"),
    "opt+fw": ("opt+fw", "opt+fw"),
    "opt flow only": ("opt", None),
    "opt decoder only": (None, "opt"),
    "opt+hc": ("opt", "opt+hc"),
    "opt+hc+fw": ("opt+fw", "opt+hc+fw"),
    "opt, int16 chains": ("opt-i16", "opt-i16"),
    "chunk spec (the library)": ("chunk", "chunk"),
}


def _quant(pv, kind, exp):
    if kind is None:
        return None
    if kind == "bf16":
        return pv.Quant({}, bf16=True)
    if kind == "q88":
        return pv.Quant({}, default=8)
    w = "float" if kind.endswith("+fw") else "int16"
    if kind.startswith("pow2"):
        return pv.Quant({k: v for k, v in exp.items() if not k.endswith("#in")}, weights=w)
    return pv.Quant(exp, weights=w, host_chain="i16" not in kind, dec_chain="+hc" in kind)


def cmd_study(a) -> int:
    import piper_vits as pv
    W = pv.load_weights(_onnx(a))
    exp = json.load(open(os.path.join(a.assets, "exponents.json")))["exponents"]
    out_dir = os.path.join(a.assets, "study")
    os.makedirs(out_dir, exist_ok=True)
    texts = _texts(a, "eval")
    policies = {p: v for p, v in POLICIES.items() if not a.policies or p in a.policies.split(";")}
    rows, results = {p: [] for p in ["float, other flow noise", *policies]}, {}
    t0 = time.time()
    for name, t in texts.items():
        seed = zlib.crc32(name.encode())
        ref = pv.synthesize(W, t["ids"], seed=seed)
        mref = _mel_db(ref)
        other = pv.synthesize(W, t["ids"], seed=seed, z_seed=seed + 1)     # the model's own variability
        err = other - ref
        rows["float, other flow noise"].append((
            10 * math.log10(float(np.sum(ref ** 2)) / float(np.sum(err ** 2))),
            _lsd(_mel_db(other), mref), 0, 0.0, float("nan")))
        if name in ("eval00", "eval10"):
            _write_wav(os.path.join(out_dir, f"{name}_float.wav"), ref)
        for pol, (fp, dp) in policies.items():
            if fp == "chunk":
                Qf = Qd = None
                wav = pv.synthesize_chunked(W, exp, pv.front_end(W, t["ids"], seed=seed)) / 32767.0
            else:
                Qf, Qd = _quant(pv, fp, exp), _quant(pv, dp, exp)
                wav = pv.synthesize(W, t["ids"], seed=seed, Q_flow=Qf, Q_dec=Qd)
            err = wav - ref
            snr = 10 * math.log10(float(np.sum(ref ** 2)) / max(float(np.sum(err ** 2)), 1e-30))
            lsd = _lsd(_mel_db(wav), mref)
            sat = sum(sum(q.sat.values()) for q in (Qf, Qd) if q is not None)
            peak = max([q.acc_peak for q in (Qf, Qd) if q is not None] + [0.0])
            rows[pol].append((snr, lsd, sat, peak, _pause_floor(ref, err)))
            if name in ("eval00", "eval10") and pol in ("bf16", "q88", "pow2", "opt", "opt+hc", "opt, int16 chains",
                                                         "chunk spec (the library)"):
                tag = pol.split(" (")[0].replace("+", "_").replace(", ", "_").replace(" ", "_")
                _write_wav(os.path.join(out_dir, f"{name}_{tag}.wav"), wav)
        print(f"  {name}: {len(ref) / pv.SR:4.2f} s  " + "  ".join(
            f"{p} {rows[p][-1][0]:5.1f} dB" for p in policies), flush=True)
    print(f"policy vs float (same noise, {len(texts)} sentences, {time.time() - t0:.0f} s):")
    print(f"  {'policy':24s} {'SNR dB mean / min':>18s} {'log-mel dist dB':>16s} {'pause floor dBFS':>17s} "
          f"{'saturated':>10s} {'acc peak / 2^31':>16s}")
    for pol, r in rows.items():
        s = np.array(r, dtype=float)
        pf = float(np.nanmax(s[:, 4])) if not np.all(np.isnan(s[:, 4])) else float("nan")
        results[pol] = {"snr_mean": float(s[:, 0].mean()), "snr_min": float(s[:, 0].min()),
                        "lsd_mean": float(s[:, 1].mean()), "pause_floor_dbfs_max": pf,
                        "saturated": int(s[:, 2].sum()), "acc_peak_frac": float(s[:, 3].max() / 2 ** 31)}
        print(f"  {pol:24s} {s[:, 0].mean():8.1f} / {s[:, 0].min():5.1f}   {s[:, 1].mean():12.2f}   "
              f"{pf:14.1f}   {int(s[:, 2].sum()):9d}   {s[:, 3].max() / 2 ** 31:13.4f}")
    path = os.path.join(out_dir, "results.json")
    old = json.load(open(path)) if a.policies and os.path.exists(path) else {}
    json.dump({**old, **results}, open(path, "w"), indent=1)
    print(f"WAVs (eval00, eval10: float / bf16 / q88 / pow2 / opt / opt_hc / opt_int16_chains / chunk_spec) "
          f"in {out_dir}")
    return 0


# ---- encoder (TTS_PLAN §6) ----------------------------------------------------------------- #

def cmd_encoder(a) -> int:
    """Calibrate the int16 text encoder (the library's encode entries) on the
    calibration sentences, store its exponents in exponents.json
    ("encoder"), and measure it on the evaluation sentences: durations
    against the float encoder, and the audio with the float durations kept
    (so the waveforms align) against the front end on the float encoder and
    against float."""
    import piper_vits as pv
    W = pv.load_weights(_onnx(a))
    EW = pv.encoder_weights(W)
    path = os.path.join(a.assets, "exponents.json")
    doc = json.load(open(path))
    exp = doc["exponents"]
    results = {}
    for p_exp in [int(v) for v in a.p_exp.split(",")]:
        rec = {}
        for t in _texts(a, "calib").values():
            pv.encoder_float(EW, t["ids"], rec)
        E = pv.encoder_exponents(EW, rec, p_exp)
        enc_q = pv.library_encoder(EW, E)
        n_ids = n_diff = worst = 0
        lsd_f, lsd_cur, lsd_q = [], [], []
        out_dir = os.path.join(a.assets, "study")
        for name, t in _texts(a, "eval").items():
            seed = zlib.crc32(name.encode())
            ids = t["ids"]
            xf, mf, lf = pv.text_encoder(W, np.asarray(ids))
            xq, mq, lq = enc_q(ids)
            wf = np.ceil(np.exp(pv.duration_predictor(W, xf, 0.8, np.random.default_rng(seed), True)))
            wq = np.ceil(np.exp(pv.duration_predictor(W, xq, 0.8, np.random.default_rng(seed), True)))
            n_ids += len(ids)
            n_diff += int((wf != wq).sum())
            worst = max(worst, int(np.abs(wf - wq).max()))
            ref = pv.synthesize(W, ids, seed=seed)
            mref = _mel_db(ref)
            cur = pv.synthesize_chunked(W, exp, pv.front_end(W, ids, seed=seed, fast_erf=True)) / 32767.0
            # the int16 encoder's statistics with the float durations (aligned audio)
            aligned = pv.synthesize_chunked(W, exp, pv.front_end(
                W, ids, seed=seed, fast_erf=True, encoder=lambda i, xf=xf, mq=mq, lq=lq: (xf, mq, lq))) / 32767.0
            own = pv.synthesize_chunked(W, exp, pv.front_end(W, ids, seed=seed, fast_erf=True,
                                                             encoder=enc_q)) / 32767.0
            lsd_cur.append(_lsd(_mel_db(cur), mref))
            lsd_q.append(_lsd(_mel_db(aligned), mref))
            lsd_f.append(_lsd(_mel_db(aligned), _mel_db(cur)))
            if name in ("eval00", "eval10"):
                _write_wav(os.path.join(out_dir, f"{name}_enc_int16_p{p_exp}.wav"), own)
        results[p_exp] = {"exponents": E, "durations_changed": n_diff / n_ids, "max_frames": worst,
                          "lsd_current_vs_float": float(np.mean(lsd_cur)),
                          "lsd_int16_encoder_vs_float": float(np.mean(lsd_q)),
                          "lsd_int16_encoder_vs_current": float(np.mean(lsd_f))}
        r = results[p_exp]
        print(f"P at 2^-{p_exp}: durations changed for {n_diff}/{n_ids} ids ({100 * n_diff / n_ids:.1f} %, "
              f"max {worst} frame); log-mel distance vs float {r['lsd_int16_encoder_vs_float']:.3f} dB "
              f"(the float encoder: {r['lsd_current_vs_float']:.3f} dB), vs the float encoder "
              f"{r['lsd_int16_encoder_vs_current']:.3f} dB", flush=True)
    best = int(a.p_exp.split(",")[0])
    doc["encoder"] = results[best]["exponents"]
    doc["encoder_study"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "exponents"}
                            for k, v in results.items()}
    json.dump(doc, open(path, "w"), indent=1)
    print(f"encoder exponents (P at 2^-{best}) -> {path} \"encoder\"; WAVs eval00 / eval10 in "
          f"{os.path.join(a.assets, 'study')}")
    return 0


def cmd_encoder_vsmx(a) -> int:
    """The text encoder with its softmax on VectorOPKernel's unit against the
    library's host softmax, both on the stored exponents (exponents.json
    "encoder"): durations against the float encoder and against each other,
    the audio with the float durations kept against float (encoder's
    yardstick), the encoder outputs' difference and the logits' saturation."""
    import piper_vits as pv
    W = pv.load_weights(_onnx(a))
    EW = pv.encoder_weights(W)
    doc = json.load(open(os.path.join(a.assets, "exponents.json")))
    exp, E = doc["exponents"], {k: int(v) for k, v in doc["encoder"].items()}
    out_dir = os.path.join(a.assets, "study")
    pols = {"host softmax": False, "VectorOP softmax": True}
    acc = {k: {"n_diff": 0, "worst": 0, "lsd": [], "dx": [], "dst": []} for k in pols}
    n_ids = n_cross = sat = n_logits = 0
    for name, t in _texts(a, "eval").items():
        seed = zlib.crc32(name.encode())
        ids = t["ids"]
        n_ids += len(ids)
        xf, mf, lf = pv.text_encoder(W, np.asarray(ids))
        wf = np.ceil(np.exp(pv.duration_predictor(W, xf, 0.8, np.random.default_rng(seed), True)))
        mref = _mel_db(pv.synthesize(W, ids, seed=seed))
        w_of, x_of = {}, {}
        for pol, unit in pols.items():
            tr = {}
            x, st = pv.encoder_forward(EW, E, ids, trace=tr, vsmx_unit=unit)
            xq = x.astype(np.float64).T
            mq, lq = st.astype(np.float64).T[:pv.ENC_D], st.astype(np.float64).T[pv.ENC_D:]
            wq = np.ceil(np.exp(pv.duration_predictor(W, xq, 0.8, np.random.default_rng(seed), True)))
            r = acc[pol]
            r["n_diff"] += int((wf != wq).sum())
            r["worst"] = max(r["worst"], int(np.abs(wf - wq).max()))
            aligned = pv.synthesize_chunked(W, exp, pv.front_end(
                W, ids, seed=seed, fast_erf=True, encoder=lambda i, xf=xf, mq=mq, lq=lq: (xf, mq, lq))) / 32767.0
            r["lsd"].append(_lsd(_mel_db(aligned), mref))
            r["dx"].append(float(np.abs(xq - xf).max()))
            r["dst"].append(float(max(np.abs(mq - mf).max(), np.abs(lq - lf).max())))
            w_of[pol], x_of[pol] = wq, xq
            if unit:
                for k, v in tr.items():
                    if ".l" in k and k.rsplit(".", 1)[-1].startswith("l"):
                        sat += int((np.abs(v) >= 32767).sum())
                        n_logits += v.size
                if name in ("eval00", "eval10"):
                    own = pv.synthesize_chunked(W, exp, pv.front_end(
                        W, ids, seed=seed, fast_erf=True, encoder=pv.library_encoder(EW, E, True))) / 32767.0
                    _write_wav(os.path.join(out_dir, f"{name}_enc_int16_vsmx.wav"), own)
        n_cross += int((w_of["host softmax"] != w_of["VectorOP softmax"]).sum())
    res = {}
    for pol, r in acc.items():
        res[pol] = {"durations_changed": r["n_diff"] / n_ids, "durations_changed_n": r["n_diff"],
                    "max_frames": r["worst"], "lsd_vs_float_db": float(np.mean(r["lsd"])),
                    "max_abs_x_vs_float": max(r["dx"]), "max_abs_stats_vs_float": max(r["dst"])}
        print(f"  {pol:17s} durations changed against float {r['n_diff']}/{n_ids} "
              f"({100 * r['n_diff'] / n_ids:.2f} %, max {r['worst']} frame); log-mel distance vs float "
              f"{res[pol]['lsd_vs_float_db']:.3f} dB; max |x - float| {max(r['dx']):.4f}", flush=True)
    res["durations_differ_between_policies"] = n_cross
    res["logits_saturated"] = sat
    res["logits"] = n_logits
    print(f"  durations differing between the two: {n_cross}/{n_ids}; saturated logits {sat} of {n_logits}")
    path = os.path.join(out_dir, "encoder_vsmx.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"-> {path}; WAVs eval00 / eval10 _enc_int16_vsmx.wav in {out_dir}")
    return 0


# ---- costs ---------------------------------------------------------------------------- #

K_GMACS = (13.4, 30.0)       # 1 x k (k > 2) convs are not in the performance model's families


def _conv_proxy(convs):
    """ONNX proxy of 1-D convs [(tag, cin, cout, k, dil, W)] as H = 1 convs:
    time folded into rows of w0 outputs (out_w * out_ch <= 65 536 per row,
    the row's left context duplicated), channels split at in <= 1024 /
    out <= 1280, taps split so one call spans <= 64 columns."""
    from onnx import TensorProto as TP, helper as oh, numpy_helper as nph
    rng = np.random.default_rng(0)
    nodes, inits, ins, outs, info = [], [], [], [], []
    for tag, cin, cout, k, dil, W in convs:
        taps = [(t0, min(k, t0 + (64 - 1) // dil + 1)) for t0 in range(0, k, (64 - 1) // dil + 1)]
        for t0, t1 in taps:
            kk = t1 - t0
            for i0 in range(0, cin, 1024):
                ci = min(1024, cin - i0)
                for o0 in range(0, cout, 1280):
                    co = min(1280, cout - o0)
                    co16 = -(-co // 16) * 16
                    w0 = 1
                    while w0 * 2 * co16 <= 65536 and w0 * 2 <= W:
                        w0 *= 2
                    H = -(-W // w0)
                    n = len(nodes)
                    ins.append(oh.make_tensor_value_info(f"x{n}", TP.FLOAT, [1, ci, H, w0 + (kk - 1) * dil]))
                    inits.append(nph.from_array(
                        (rng.standard_normal((co, ci, 1, kk)) * 0.02).astype(np.float32), f"w{n}"))
                    nodes.append(oh.make_node("Conv", [f"x{n}", f"w{n}"], [f"y{n}"], name=f"c{n}_{tag}",
                                              kernel_shape=[1, kk], dilations=[1, dil]))
                    outs.append(oh.make_tensor_value_info(f"y{n}", TP.FLOAT, [1, co, H, w0]))
                    info.append((f"c{n}_{tag}", kk, ci * co * kk * H * w0))
    m = oh.make_model(oh.make_graph(nodes, "proxy", ins, outs, initializer=inits),
                      opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    return m, info


def piper_convs():
    """Flow + decoder convs for one second of audio (frame rate 86.13 Hz);
    transposed convs as polyphase kernel-ceil(k/s) convs with s * C_out outputs."""
    fr = round(22050 / 256)
    flow = []
    for _ in range(4):
        flow += [("flow_pre", 96, 192, 1, 1, fr)]
        for i in range(4):
            flow += [("flow_in", 192, 384, 5, 1, fr), ("flow_rs", 192, 384 if i < 3 else 192, 1, 1, fr)]
        flow += [("flow_post", 192, 96, 1, 1, fr)]
    dec = [("dec_pre", 192, 256, 7, 1, fr)]
    W, c = fr, 256
    for s, k in ((8, 16), (8, 16), (4, 8)):
        o = c // 2
        dec.append((f"dec_up{s}", c, s * o, -(-k // s), 1, W))
        W *= s
        for kk, (d1, d2) in zip((3, 5, 7), ((1, 2), (2, 6), (3, 12)), strict=True):
            dec += [(f"dec_rb{o}", o, o, kk, d1, W), (f"dec_rb{o}", o, o, kk, d2, W)]
        c = o
    dec.append(("dec_post", c, 1, 7, 1, W))
    return flow, dec


def cmd_costs(a) -> int:
    sys.path.insert(0, SCHED)
    from src.codegen import CodeGenerator
    from src.graph import OnnxGraph
    from src.perf_model import PerfModel
    pm = PerfModel.load(a.perf_model)
    print(f"performance model {os.path.relpath(a.perf_model, REPO)}")
    flow, dec = piper_convs()
    tot = [0.0, 0.0]
    for part, convs in (("flow", flow), ("decoder", dec)):
        model, info = _conv_proxy(convs)
        g = OnnxGraph(model, fuse_act=True, s2d_stem=True)
        cg = CodeGenerator(g, model_path="proxy.onnx")
        priced_s = priced_mac = unpriced_mac = 0.0
        by = {}
        for sn, (name, _k, macs) in zip(g.nodes, info, strict=True):
            b = pm.calls_band(sn.kernel_calls(cg._layouts))
            if b is None:
                unpriced_mac += macs
            else:
                priced_s += b[0] / 1e6
                priced_mac += macs
            key = name.split("_", 1)[1]
            by[key] = by.get(key, 0.0) + macs
        lo = priced_s + unpriced_mac / (K_GMACS[1] * 1e9)
        hi = priced_s + unpriced_mac / (K_GMACS[0] * 1e9)
        tot[0] += lo
        tot[1] += hi
        print(f"{part:8s} {(priced_mac + unpriced_mac) / 1e9:5.2f} GMAC per second of audio in {len(info)} calls: "
              f"{priced_mac / 1e9:.2f} GMAC priced ({priced_s:.3f} s), {unpriced_mac / 1e9:.2f} GMAC 1xk convs at "
              f"{K_GMACS[0]}-{K_GMACS[1]} GMAC/s -> {lo:.3f}-{hi:.3f} s")
    # host work per second of audio (the decoder's LeakyReLU / add / average
    # on int16 tensors, the flow's gates) and per sentence (text encoder, dp)
    fr = 22050 / 256
    host_el = 4 * 4 * 192 * fr * 3
    W, c = fr, 256
    host_el += c * W
    for s in (8, 8, 4):
        c //= 2
        W *= s
        host_el += c * W * (1 + 3 * 2 * 2 + 3 + 1)
    host_el += c * W
    print(f"host     ~{host_el / 1e6:.0f} M element ops per second of audio (LeakyReLU, adds, "
          f"averages, gates): ~{host_el * 2e-9:.2f}-{host_el * 5e-9:.2f} s at 2-5 ns each (A53, 4 threads)")
    print(f"total    kernels {tot[0]:.2f}-{tot[1]:.2f} s per second of audio; not priced: the text encoder "
          f"(on the FPGA, TTS_PLAN §6) and the duration predictor (C on the host, §7), ~0.2 GMAC per "
          f"second of audio")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--assets", default=default_assets())
    ap.add_argument("--perf-model", default=None)
    ap.add_argument("--policies", default=None,
                    help="study: only these policies (';'-separated names from POLICIES)")
    ap.add_argument("--p-exp", default="15,12",
                    help="encoder: softmax output exponents to compare (the first is stored)")
    ap.add_argument("cmd", choices=("fetch", "phonemize", "validate", "calibrate", "study", "encoder",
                                    "encoder-vsmx", "costs", "all"))
    a = ap.parse_args(argv)
    if a.perf_model is None and a.cmd in ("costs", "all"):
        sys.path.insert(0, SCHED)
        from src.perf_calls import local_bitstream_id
        a.perf_model = os.path.join(SCHED, "perf_models", "kv260", f"{local_bitstream_id()}.json")
    cmds = ("fetch", "validate", "calibrate", "study", "costs") if a.cmd == "all" else (a.cmd,)
    fn = {"fetch": cmd_fetch, "phonemize": cmd_phonemize, "validate": cmd_validate,
          "calibrate": cmd_calibrate, "study": cmd_study, "encoder": cmd_encoder,
          "encoder-vsmx": cmd_encoder_vsmx, "costs": cmd_costs}
    rc = 0
    for c in cmds:
        print(f"== {c}", flush=True)
        rc |= fn[c](a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
