#!/usr/bin/env python3
"""asr_study.py — model study of speech-to-text models for the KV260
(model-study skill, route C; doc/plans/ASR_PLAN.md).  Host only.

  fetch      the candidates' checkpoints at their pinned revisions and the
             LibriSpeech test-clean / dev-clean archives, every file checked
             against asr_study_inputs.json; the archives extracted
  select     (once) the evaluation utterances (test-clean, evenly spaced per
             speaker, <= 29 s) and the calibration ones (dev-clean) ->
             asr_study_inputs.json
  validate   the numpy float64 references (whisper_np.py, moonshine_np.py)
             against transformers float32: log-mel, encoder states,
             first logits, greedy transcripts (3 utterances per model)
  screen     every candidate in float32 through transformers on the
             evaluation set: WER (Whisper's English normaliser on both
             sides), tokens, compute per utterance -> study/screen.json
  study      (Moonshine) the int16 datapath of the plan, bit-level
             (moonshine_q.py): calibration on dev-clean (float maxima ->
             exponents), then every policy on the evaluation set: WER,
             transcripts equal to float's, teacher-forced top-1 on float's
             tokens, the encoder output's error, saturation
             -> study/<model>/results.json, transcripts.txt
  ablate     study with one group of tensor classes left float at a time
  window     Whisper with its encoder window cut to the audio, rounded up
             to buckets of N frames (whisper.cpp's audio_ctx; numpy
             float64): WER, transcripts equal to the 30 s window's
             -> study/window.json

usage: .venv-export/bin/python demo/asr/scripts/asr_study.py [--assets DIR] CMD

Assets (not in git, demo/asr/assets -> /mnt/data): one directory per
checkpoint, librispeech/ (the archives and the extracted LibriSpeech/),
study/ (results).  Deterministic: greedy decoding only, fixed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

if any(c in sys.argv for c in ("study", "ablate")):     # one BLAS thread per worker process
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_PATH = os.path.join(HERE, "asr_study_inputs.json")
INPUTS = json.load(open(INPUTS_PATH))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))
SR = 16000
MAX_SECONDS = 29.0          # one Whisper window (30 s) with room to spare


# ------------------------------------------------------------------ assets
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_dir(assets, name):
    return os.path.join(assets, name)


def libri_root(assets):
    return os.path.join(assets, "librispeech", "LibriSpeech")


def cmd_fetch(args):
    """Download what is missing or different; extract the archives."""
    bad = 0
    for name, m in INPUTS["models"].items():
        d = model_dir(args.assets, name)
        os.makedirs(d, exist_ok=True)
        for f, h in m["files"].items():
            p = os.path.join(d, f)
            if not os.path.exists(p) or sha256(p) != h:
                urllib.request.urlretrieve(
                    f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f}", p)
            ok = sha256(p) == h
            bad += not ok
            print(f"{name:16s} {f:26s} {'ok' if ok else 'SHA-256 MISMATCH'}")
    ld = os.path.join(args.assets, "librispeech")
    os.makedirs(ld, exist_ok=True)
    for part, a in INPUTS["librispeech"]["archives"].items():
        p = os.path.join(ld, f"{part}.tar.gz")
        if not os.path.exists(p) or sha256(p) != a["sha256"]:
            urllib.request.urlretrieve(a["url"], p)
        ok = sha256(p) == a["sha256"]
        bad += not ok
        print(f"{'librispeech':16s} {part + '.tar.gz':26s} {'ok' if ok else 'SHA-256 MISMATCH'}")
        if ok and not os.path.isdir(os.path.join(libri_root(args.assets), part)):
            with tarfile.open(p) as t:
                t.extractall(ld, filter="data")
    if bad:
        sys.exit(f"{bad} file(s) differ from asr_study_inputs.json")


# ------------------------------------------------------------------ data
def transcripts(assets, part):
    """{utterance id: reference text} of one LibriSpeech part."""
    out = {}
    root = os.path.join(libri_root(assets), part)
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".trans.txt"):
                for line in open(os.path.join(dp, f)):
                    uid, text = line.rstrip("\n").split(" ", 1)
                    out[uid] = text
    return out


def flac_path(assets, part, uid):
    spk, ch, _ = uid.split("-")
    return os.path.join(libri_root(assets), part, spk, ch, uid + ".flac")


def load_pcm(path):
    """16 kHz mono int16 samples (ffmpeg; LibriSpeech is 16 kHz mono already)."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                          "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, "<i2").copy()


def utterances(assets, which):
    """[(uid, reference, int16 samples)] of the evaluation or calibration set."""
    sel = INPUTS["librispeech"][which]
    refs = transcripts(assets, sel["part"])
    return [(u, refs[u], load_pcm(flac_path(assets, sel["part"], u))) for u in sel["ids"]]


def cmd_select(args):
    """Evenly spaced utterances per speaker, <= MAX_SECONDS; written into the inputs file."""
    def pick(part, per_spk):
        refs = transcripts(args.assets, part)
        by_spk = {}
        for u in sorted(refs):
            by_spk.setdefault(u.split("-")[0], []).append(u)
        ids = []
        for spk in sorted(by_spk, key=int):
            us = by_spk[spk]
            want = [round(i * (len(us) - 1) / max(per_spk - 1, 1)) for i in range(per_spk)]
            taken = set()
            for w in want:
                for j in [w] + [w + d for k in range(1, len(us)) for d in (k, -k)]:
                    if 0 <= j < len(us) and j not in taken:
                        n = len(load_pcm(flac_path(args.assets, part, us[j])))
                        if n <= MAX_SECONDS * SR:
                            taken.add(j)
                            break
            ids += [us[j] for j in sorted(taken)]
        return ids
    lib = INPUTS["librispeech"]
    lib["evaluation"]["ids"] = pick(lib["evaluation"]["part"], lib["evaluation"]["per_speaker"])
    lib["calibration"]["ids"] = pick(lib["calibration"]["part"], lib["calibration"]["per_speaker"])
    with open(INPUTS_PATH, "w") as f:
        json.dump(INPUTS, f, indent=1)
        f.write("\n")
    print(f"evaluation {len(lib['evaluation']['ids'])}, calibration {len(lib['calibration']['ids'])}"
          f" -> {INPUTS_PATH}")


# ------------------------------------------------------------------ metrics
_NORM = None


def normalise(text):
    """Whisper's English normaliser (the Open ASR leaderboard's), on hypotheses and references."""
    global _NORM
    if _NORM is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        spell = json.load(open(os.path.join(model_dir(ASSETS, "whisper-tiny.en"), "normalizer.json")))
        _NORM = EnglishTextNormalizer(spell)
    return _NORM(text)


def edit_distance(a, b):
    """Levenshtein distance between two word lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(hyps, refs):
    """Corpus WER (errors / reference words) after normalisation; and the per-utterance errors."""
    errs, words = [], 0
    for h, r in zip(hyps, refs):
        hw, rw = normalise(h).split(), normalise(r).split()
        errs.append(edit_distance(hw, rw))
        words += len(rw)
    return sum(errs) / max(words, 1), errs


# ------------------------------------------------------------------ compute
def macs(name, seconds, tokens):
    """Multiply-accumulates of one utterance: encoder (the audio), cross K / V, per decoder token."""
    cfg = json.load(open(os.path.join(model_dir(ASSETS, name), "config.json")))
    if cfg["model_type"] == "whisper":
        d, f, L, Ld = cfg["d_model"], cfg["encoder_ffn_dim"], cfg["encoder_layers"], cfg["decoder_layers"]
        T = 1500                                     # the 30 s window, whatever the audio
        enc = 3000 * d * 80 * 3 + T * d * d * 3
        enc += L * (4 * T * d * d + 2 * T * T * d + 2 * T * d * f)
        cross = Ld * 2 * T * d * d
        tok = Ld * (4 * d * d + 2 * d * d + 2 * d * f + 2 * T * d) + cfg["vocab_size"] * d
    else:
        d, f = cfg["hidden_size"], cfg["intermediate_size"]
        L, Ld = cfg["encoder_num_hidden_layers"], cfg["decoder_num_hidden_layers"]
        n = int(seconds * SR)
        t1 = (n - 127) // 64 + 1
        t2 = (t1 - 7) // 3 + 1
        T = (t2 - 3) // 2 + 1
        enc = t1 * d * 127 + t2 * 2 * d * d * 7 + T * d * 2 * d * 3
        enc += L * (4 * T * d * d + 2 * T * T * d + 2 * T * d * f)
        cross = Ld * 2 * T * d * d
        tok = Ld * (4 * d * d + 2 * d * d + 3 * d * f + 2 * T * d) + cfg["vocab_size"] * d
    return dict(enc=enc, cross=cross, per_token=tok, frames=T, total=enc + cross + tokens * tok)


def params(name):
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir(ASSETS, name), "model.safetensors"), "np") as f:
        return sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())


# ------------------------------------------------------------------ screen (transformers, float32)
def hf_transcribe(name, utts, log):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    d = model_dir(ASSETS, name)
    out = []
    if name.startswith("whisper"):
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
        fe, tok = WhisperFeatureExtractor.from_pretrained(d), WhisperTokenizer.from_pretrained(d)
        model = WhisperForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
        for i, (uid, _, pcm) in enumerate(utts):
            feats = fe(pcm.astype(np.float32) / 32768.0, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                ids = model.generate(feats, num_beams=1, do_sample=False, max_new_tokens=440)[0].tolist()
            out.append((tok.decode(ids, skip_special_tokens=True).strip(), len(ids)))
            if i % 50 == 0:
                log(f"  {name} {i}/{len(utts)}: {out[-1][0][:70]}")
    else:
        from tokenizers import Tokenizer
        from transformers import MoonshineForConditionalGeneration
        tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        model = MoonshineForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
        for i, (uid, _, pcm) in enumerate(utts):
            x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None]
            # the model card's limit: about 6.5 tokens per second of audio
            limit = min(int(len(pcm) / SR * 6.5) + 1, 194)
            with torch.no_grad():
                ids = model.generate(x, num_beams=1, do_sample=False, max_length=limit)[0].tolist()
            out.append((tok.decode(ids, skip_special_tokens=True).strip(), len(ids)))
            if i % 50 == 0:
                log(f"  {name} {i}/{len(utts)}: {out[-1][0][:70]}")
    return out


def cmd_screen(args):
    out_dir = os.path.join(args.assets, "study")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "screen.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    utts = utterances(args.assets, "evaluation")
    secs = [len(p) / SR for _, _, p in utts]
    log(f"screen {time.strftime('%Y-%m-%d %H:%M')}: {len(utts)} utterances, "
        f"{sum(secs) / 60:.1f} min of audio (mean {np.mean(secs):.1f} s, max {max(secs):.1f} s)")
    path = os.path.join(out_dir, "screen.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in args.models or list(INPUTS["models"]):
        t0 = time.time()
        hyp = hf_transcribe(name, utts, log)
        w, errs = wer([h for h, _ in hyp], [r for _, r, _ in utts])
        toks = [n for _, n in hyp]
        mc = [macs(name, s, n) for s, n in zip(secs, toks)]
        res[name] = dict(
            wer=w, params=params(name), tokens_mean=float(np.mean(toks)),
            gmac_mean=float(np.mean([m["total"] for m in mc]) / 1e9),
            enc_gmac_mean=float(np.mean([m["enc"] + m["cross"] for m in mc]) / 1e9),
            per_token_mmac=mc[0]["per_token"] / 1e6, seconds=time.time() - t0,
            hyps={u: h for (u, _, _), (h, _) in zip(utts, hyp)}, errors=dict(zip([u for u, _, _ in utts], errs)))
        log(f"{name:16s} WER {100 * w:5.2f} %  params {res[name]['params'] / 1e6:5.1f} M  "
            f"tokens {res[name]['tokens_mean']:5.1f}  {res[name]['gmac_mean']:6.2f} GMAC/utt "
            f"(encoder {res[name]['enc_gmac_mean']:6.2f}, {res[name]['per_token_mmac']:5.1f} MMAC/token)  "
            f"[{res[name]['seconds']:.0f} s]")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


# ------------------------------------------------------------------ validate (numpy float64 vs torch float32)
def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-30))


def cmd_validate(args):
    """The numpy references against transformers on the shortest, a median and the longest utterance."""
    import torch
    utts = utterances(args.assets, "evaluation")
    order = sorted(range(len(utts)), key=lambda i: len(utts[i][2]))
    pick = [utts[i] for i in (order[0], order[len(order) // 2], order[-1])]
    bad = 0
    for name in args.models or list(INPUTS["models"]):
        d = model_dir(args.assets, name)
        t0 = time.time()
        if name.startswith("whisper"):
            import whisper_np as wn
            from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
            cfg, gen, W, melf = wn.load(d)
            fe = WhisperFeatureExtractor.from_pretrained(d)
            hf = WhisperForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
            for uid, _, pcm in pick:
                mel = wn.log_mel(pcm, melf)
                hmel = fe(pcm.astype(np.float32) / 32768.0, sampling_rate=SR, return_tensors="np").input_features[0]
                enc = wn.encode(cfg, W, mel)
                with torch.no_grad():
                    henc = hf.model.encoder(torch.from_numpy(hmel[None]).float()).last_hidden_state[0].numpy()
                    hids = hf.generate(torch.from_numpy(hmel[None]).float(), num_beams=1, do_sample=False,
                                       max_new_tokens=440)[0].tolist()
                    hlog = hf(input_features=torch.from_numpy(hmel[None]).float(),
                              decoder_input_ids=torch.tensor([wn.prompt(gen)])).logits[0, -1].numpy()
                dec = wn.Decoder(cfg, W, enc)
                lg = dec.step(wn.prompt(gen))
                ids = wn.greedy(cfg, gen, dec, first_logits=lg)
                if hids[: len(wn.prompt(gen))] == wn.prompt(gen):   # transformers 5 returns only the new ids
                    hids = hids[len(wn.prompt(gen)):]
                hgen = [t for t in hids if t != gen["eos_token_id"]]
                ok = ids == hgen
                bad += not ok
                print(f"{name} {uid} ({len(pcm) / SR:4.1f} s): log-mel max |diff| {np.abs(mel - hmel).max():.2e}, "
                      f"encoder rel {rel(henc, enc):.2e}, logits rel {rel(hlog, lg):.2e}, "
                      f"tokens {'equal' if ok else 'DIFFER'} ({len(ids)})")
        else:
            import moonshine_np as mn
            from transformers import MoonshineForConditionalGeneration
            cfg, W = mn.load(d)
            hf = MoonshineForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32).eval()
            for uid, _, pcm in pick:
                x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None]
                enc = mn.encode(cfg, W, pcm)
                with torch.no_grad():
                    henc = hf.model.encoder(x).last_hidden_state[0].numpy()
                    hids = hf.generate(x, num_beams=1, do_sample=False, max_length=mn.max_tokens(len(pcm)))[0].tolist()
                    hlog = hf(input_values=x, decoder_input_ids=torch.tensor([[cfg["decoder_start_token_id"]]])
                              ).logits[0, -1].numpy()
                dec = mn.Decoder(cfg, W, enc)
                lg = dec.step([cfg["decoder_start_token_id"]])
                ids = mn.greedy(cfg, dec, len(pcm), first_logits=lg)
                hgen = [t for t in hids[1:] if t != cfg["eos_token_id"]]
                ok = ids == hgen
                bad += not ok
                print(f"{name} {uid} ({len(pcm) / SR:4.1f} s): encoder rel {rel(henc, enc):.2e} "
                      f"({enc.shape[0]} frames), logits rel {rel(hlog, lg):.2e}, "
                      f"tokens {'equal' if ok else 'DIFFER'} ({len(ids)})")
        print(f"  [{time.time() - t0:.0f} s]")
    if bad:
        sys.exit(f"{bad} greedy transcript(s) differ from transformers")


# ------------------------------------------------------------------ window (Whisper's encoder window cut to the audio)
def cmd_window(args):
    """Whisper with the encoder window rounded up to the audio in buckets (numpy
    float64): WER and transcripts equal to the 30 s window's."""
    import whisper_np as wn
    out_dir = os.path.join(args.assets, "study")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "window.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    utts = utterances(args.assets, "evaluation")
    refs = [r for _, r, _ in utts]
    for name in args.models or [m for m in INPUTS["models"] if m.startswith("whisper")]:
        cfg, gen, W, melf = wn.load(model_dir(args.assets, name))
        mels = [wn.log_mel(p, melf) for _, _, p in utts]
        full = None
        for bucket in args.buckets:
            t0 = time.time()
            hyps, frames = [], []
            for (uid, _, pcm), mel in zip(utts, mels):
                n = wn.window_frames(len(pcm), bucket)
                frames.append(n)
                dec = wn.Decoder(cfg, W, wn.encode(cfg, W, mel, n))
                hyps.append(wn.greedy(cfg, gen, dec))
            texts = [_decode_whisper(name, h) for h in hyps]
            w, errs = wer(texts, refs)
            if bucket == 1500:
                full = hyps
            same = sum(a == b for a, b in zip(hyps, full)) if full else None
            res.setdefault(name, {})[str(bucket)] = dict(
                wer=w, frames_mean=float(np.mean(frames)), same_as_30s=same,
                texts=dict(zip([u for u, _, _ in utts], texts)), errors=dict(zip([u for u, _, _ in utts], errs)))
            print(f"{name:16s} bucket {bucket:5d} frames (mean {np.mean(frames):6.1f}): WER {100 * w:5.2f} %"
                  + (f", {same}/{len(utts)} transcripts = 30 s window" if same is not None else "")
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1)


# ------------------------------------------------------------------ study (the int16 datapath, moonshine_q.py)
# policy name -> moonshine_q policy (None: float64 reference); "exact" lists classes left unquantised
POLICIES = {
    "float": None,
    "bf16": {"bf16": True},
    "unit": {"p_bits": 12},                       # every op on the kernels / units (the plan)
    "unit+p15": {"p_bits": 15},
    "unit+in7": {"p_bits": 12, "in_cap": 7},
    "unit+in9": {"p_bits": 12, "in_cap": 9},
    "unit+tensor": {"p_bits": 12, "per_tensor": True},
    "unit+hostgelu": {"p_bits": 12, "vop_gelu": False},
    "unit+fw": {"p_bits": 12, "float_weights": True},
    "unit+in6": {"p_bits": 12, "in_cap": 6},
    # inputs at 2^-7 except the encoder's GELU outputs (VectorOPKernel writes Q8.8: the GELUs
    # stay on the FPGA)
    "unit+in7v": {"p_bits": 12, "in_cap": 8, "caps": {c: 7 for c in ("x", "x2", "xf", "gn", "y", "y2", "y3", "yf")}
                  | {("a", "d"): 7}},
    # ... and the stem's host GELU output (conv3's input) at 2^-6
    "opt": {"p_bits": 12, "in_cap": 8, "caps": {c: 7 for c in ("x", "x2", "xf", "gn", "y", "y2", "y3", "yf")}
            | {("a", "d"): 7, "g2": 6}},
    # every kernel input's exponent per channel from the error balance (moonshine_q.balanced)
    "unit+bal": {"p_bits": 12, "bal": True},
    # ... except VectorOP's GELU outputs (Q8.8, so the GELUs stay on the FPGA)
    "unit+bal+vop": {"p_bits": 12, "bal": "novop"},
}
ABLATE = [("c1", "t", "gn"), ("c2", "g2", "c3", "g3"), ("x", "x2", "xf"), ("q0", "k0", "q", "k"),
          ("s", "p"), ("v", "vc", "pv", "o"), ("f", "a", "d"), ("y", "y2", "y3", "yf"),
          ("cq", "ck", "cs", "cp", "cv", "cpv", "co"), ("u", "ub", "gb"), ("logits",)]

_W = {}


def _worker_init(name, pol, fmt_path):
    import pickle
    import torch
    torch.set_num_threads(1)
    import moonshine_np as mn
    import moonshine_q as mq
    cfg, W = mn.load(model_dir(ASSETS, name))
    fmt = pickle.load(open(fmt_path, "rb"))["fmt"] if fmt_path else None
    _W.update(cfg=cfg, W=W, m=mq.QM(cfg, W, pol=pol, fmt=fmt, record=pol is None and not fmt_path))


def _calib_one(pcm):
    import moonshine_q as mq
    m = _W["m"]
    m.ch, m.ms = {}, {}
    mq.transcribe(m, _W["cfg"], pcm)
    return m.ch, m.ms


def _eval_one(args):
    """One utterance under the worker's policy: greedy ids, teacher-forced agreement with
    float's ids, the encoder output's relative error, saturation counts."""
    import moonshine_np as mn
    import moonshine_q as mq
    pcm, fids = args
    m, cfg = _W["m"], _W["cfg"]
    m.sat.clear()
    m.cnt.clear()
    xf = m.encode(pcm)
    xr = mn.encode(cfg, _W["W"], pcm)
    out = {"xf_rel": float(np.linalg.norm(xf - xr) / np.linalg.norm(xr))}
    ids, _ = mq.transcribe(m, cfg, pcm, xf=xf)
    out["ids"] = ids
    if fids is not None:
        _, am = mq.transcribe(m, cfg, pcm, forced=fids, xf=xf)
        want = list(fids) + [cfg["eos_token_id"]]
        out["tf_agree"] = int(sum(a == b for a, b in zip(am, want)))
        out["tf_n"] = len(want)
    out["sat"], out["cnt"] = dict(m.sat), dict(m.cnt)
    return out


def _pool(name, pol, fmt_path, jobs):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(jobs, initializer=_worker_init, initargs=(name, pol, fmt_path))


def cmd_study(args):
    """Calibrate (dev-clean, float), then every policy on the evaluation set against float."""
    import pickle
    from tokenizers import Tokenizer
    import moonshine_q as mq
    name = args.model
    out_dir = os.path.join(args.assets, "study", name)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "study.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    tok = Tokenizer.from_file(os.path.join(model_dir(args.assets, name), "tokenizer.json"))
    utts = utterances(args.assets, "evaluation")[: args.limit or None]
    refs = [r for _, r, _ in utts]
    log(f"study {name} {time.strftime('%Y-%m-%d %H:%M')}: {len(utts)} utterances, policies {args.policies}")
    # calibration maxima (float, dev-clean)
    cal_path = os.path.join(out_dir, "calib.pkl")
    if not os.path.exists(cal_path) or "ms" not in pickle.load(open(cal_path, "rb")):
        t0 = time.time()
        cal = utterances(args.assets, "calibration")
        with _pool(name, None, None, args.jobs) as p:
            parts = p.map(_calib_one, [pcm for _, _, pcm in cal])
        ch, ms = {}, {}
        for c, q in parts:
            for k, v in c.items():
                ch[k] = v if k not in ch else np.maximum(ch[k], v)
            for k, (sq, n) in q.items():
                ms[k] = [sq, n] if k not in ms else [ms[k][0] + sq, ms[k][1] + n]
        pickle.dump({"max": ch, "ms": {k: sq / n for k, (sq, n) in ms.items()}}, open(cal_path, "wb"))
        log(f"calibration: {len(cal)} dev-clean utterances, {len(ch)} tensor classes [{time.time() - t0:.0f} s]")
    cal = pickle.load(open(cal_path, "rb"))
    ch = cal["max"]
    res_path = os.path.join(out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) and not args.limit else {}
    import moonshine_np as mn
    cfg, W = mn.load(model_dir(args.assets, name))
    mf = mq.QM(cfg, W)
    fids = None
    pols = ["float"] + [p for p in args.policies.split(",") if p != "float"]
    if "float" in res and "ids" in res["float"] and not args.limit:      # float's run is the reference
        fids = [res["float"]["ids"][u] for u, _, _ in utts]
        ftexts = [res["float"]["texts"][u] for u, _, _ in utts]
        pols = pols[1:]
    for pname in pols:
        t0 = time.time()
        if pname.startswith("ablate:"):
            cls = pname.split(":", 1)[1].split("+")
            pol = dict(POLICIES[args.ablate_base], exact=cls)
        else:
            pol = POLICIES[pname]
        fmt_path = None
        if pol is not None and not pol.get("bf16"):
            fmt = mq.make_formats(mf, ch, pol, ms=cal["ms"])
            fmt_path = os.path.join(out_dir, f"formats_{pname.replace(':', '_').replace('+', '_')}.pkl")
            pickle.dump({"fmt": fmt, "policy": pol}, open(fmt_path, "wb"))
        with _pool(name, pol, fmt_path, args.jobs) as p:
            outs = p.map(_eval_one, [(pcm, None if fids is None else fids[i]) for i, (_, _, pcm) in enumerate(utts)])
        texts = [tok.decode(o["ids"], skip_special_tokens=True).strip() for o in outs]
        w, errs = wer(texts, refs)
        if pname == "float":
            fids = [o["ids"] for o in outs]
            ftexts = texts
        same = sum(o["ids"] == f for o, f in zip(outs, fids))
        sat, cnt = defaultdict(int), defaultdict(int)
        for o in outs:
            for k, v in o["sat"].items():
                sat[k] += v
            for k, v in o["cnt"].items():
                cnt[k] += v
        r = dict(wer=w, same=same, n=len(utts), xf_rel=float(np.mean([o["xf_rel"] for o in outs])),
                 sat={k: [sat[k], cnt[k]] for k in sorted(sat) if sat[k]}, seconds=time.time() - t0,
                 texts={u: t for (u, _, _), t in zip(utts, texts)})
        if "tf_agree" in outs[0]:
            r["tf_top1"] = sum(o["tf_agree"] for o in outs) / sum(o["tf_n"] for o in outs)
        if pname == "float":
            r["ids"] = {u: o["ids"] for (u, _, _), o in zip(utts, outs)}
        res[pname] = r
        log(f"{pname:24s} WER {100 * w:5.2f} %  same {same:3d}/{len(utts)}  "
            + (f"tf top-1 {r['tf_top1']:.4f}  " if "tf_top1" in r else "")
            + f"xf rel {r['xf_rel']:.4f}  sat {sum(sat.values())}  [{r['seconds']:.0f} s]")
        if not args.limit:
            with open(res_path, "w") as f:
                json.dump(res, f, indent=1)
    if not args.limit:
        with open(os.path.join(out_dir, "transcripts.txt"), "w") as f:
            for i, (u, ref, _) in enumerate(utts):
                f.write(f"{u}\n  ref   {ref}\n  float {ftexts[i]}\n")
                for pname in [p for p in res if p != "float"]:
                    t = res[pname]["texts"][u]
                    if t != ftexts[i]:
                        f.write(f"  {pname:5s} {t}\n")


_TOK = {}


def _decode_whisper(name, ids):
    if name not in _TOK:
        from tokenizers import Tokenizer
        _TOK[name] = Tokenizer.from_file(os.path.join(model_dir(ASSETS, name), "tokenizer.json"))
    return _TOK[name].decode(ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=ASSETS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("select")
    for c in ("screen", "validate", "window"):
        p = sub.add_parser(c)
        p.add_argument("--models", nargs="*")
        if c == "window":
            p.add_argument("--buckets", type=int, nargs="*", default=[1500, 250],
                           help="encoder frames per bucket (50 = 1 s); 1500 = the 30 s window")
    p = sub.add_parser("study")
    p.add_argument("--model", default="moonshine-tiny")
    p.add_argument("--policies", default="bf16,unit,unit+p15,unit+in7,unit+in9,unit+tensor,unit+hostgelu,unit+fw",
                   help="comma-separated names of POLICIES, or ablate:<cls>+<cls> (classes left float)")
    p.add_argument("--ablate-base", default="unit")
    p.add_argument("--limit", type=int, default=0, help="only the first N utterances (nothing saved)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p = sub.add_parser("ablate")
    p.add_argument("--model", default="moonshine-tiny")
    p.add_argument("--ablate-base", default="unit")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    if args.cmd == "ablate":
        args.policies = ",".join("ablate:" + "+".join(g) for g in ABLATE)
        args.cmd = "study"
    {"fetch": cmd_fetch, "select": cmd_select, "screen": cmd_screen, "validate": cmd_validate,
     "window": cmd_window, "study": cmd_study}[args.cmd](args)


if __name__ == "__main__":
    main()
