#!/usr/bin/env python3
"""tts_screen.py — screen TTS models for the KV260 by GMAC per second of
generated audio (doc/plans/TTS_PLAN.md §2).  Host only.

  fetch    the candidates' ONNX exports at pinned revisions (~1.2 GB)
  measure  each graph runs once in onnxruntime with profiling on; the
           profile records every node's input / output shapes, from which
           Conv / ConvTranspose / MatMul / Gemm / LSTM MACs are counted
           (grouped by the node name's first path component) and divided by
           the seconds of audio the run produced

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/tts_screen.py
           [--dir DIR] {fetch,measure,all}
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

HF = "https://huggingface.co"
PIPER = f"{HF}/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_US/lessac"
KOKORO = f"{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231"
KITTEN = {"nano": f"{HF}/onnx-community/KittenTTS-Nano-v0.8-ONNX/resolve/6078af30aab083436023d04ba845aedd9d1a9206",
          "mini": f"{HF}/onnx-community/KittenTTS-Mini-v0.8-ONNX/resolve/c473d28fdbd9eff9c71f9fe56a409f87cc4f1c9f"}
SUPERTONIC = f"{HF}/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323"
TINYTTS = "https://github.com/tronghieuit/tiny-tts"
FILES = {
    "piper_lessac_medium.onnx": f"{PIPER}/medium/en_US-lessac-medium.onnx",
    "piper_lessac_medium.onnx.json": f"{PIPER}/medium/en_US-lessac-medium.onnx.json",
    "piper_lessac_high.onnx": f"{PIPER}/high/en_US-lessac-high.onnx",
    "piper_lessac_high.onnx.json": f"{PIPER}/high/en_US-lessac-high.onnx.json",
    "kokoro.onnx": f"{KOKORO}/onnx/model.onnx",
    "kokoro_af_heart.bin": f"{KOKORO}/voices/af_heart.bin",
    "kitten_nano.onnx": f"{KITTEN['nano']}/onnx/model.onnx",
    "kitten_mini.onnx": f"{KITTEN['mini']}/onnx/model.onnx",
    **{f"supertonic_{os.path.basename(f)}": f"{SUPERTONIC}/{f}" for f in (
        "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx", "onnx/tts.json", "onnx/unicode_indexer.json", "voice_styles/F1.json")},
}


def fetch(D):
    os.makedirs(D, exist_ok=True)
    for name, url in FILES.items():
        p = os.path.join(D, name)
        if not os.path.exists(p) or not os.path.getsize(p):
            print(f"  {name}", flush=True)
            urllib.request.urlretrieve(url, p)
    if not os.path.isdir(os.path.join(D, "tinytts_repo")):
        subprocess.run(["git", "clone", "-q", "--depth", "1", TINYTTS, os.path.join(D, "tinytts_repo")],
                       check=True)


def measure(D):
    import onnxruntime as ort
    rng = np.random.default_rng(0)


    def prod(s):
        n = 1
        for d in s:
            n *= int(d)
        return n


    def shapes(lst):
        return [list(next(iter(x.values()))) if x else None for x in lst]


    def run(path, feeds, tag):
        so = ort.SessionOptions()
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(D, "prof_" + tag)
        so.log_severity_level = 3
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = s.run(None, feeds)
        prof = s.end_profiling()
        ev = json.load(open(prof))
        os.remove(prof)
        macs, ops, unk = collections.Counter(), collections.Counter(), collections.Counter()
        for e in ev:
            if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
                continue
            a = e["args"]
            op = a.get("op_name")
            ins, outs_ = shapes(a.get("input_type_shape", [])), shapes(a.get("output_type_shape", []))
            name = e["name"][: -len("_kernel_time")]
            grp = name.strip("/").split("/")[0] if "/" in name else "(root)"
            m = 0
            try:
                if op in ("Conv", "ConvInteger"):
                    m = prod(outs_[0]) * prod(ins[1][1:])
                elif op == "ConvTranspose":
                    m = prod(ins[0]) * prod(ins[1][1:])
                elif op in ("MatMul", "MatMulInteger", "FusedMatMul"):
                    m = prod(outs_[0]) * ins[0][-1]
                elif op == "Gemm":
                    m = prod(outs_[0]) * (ins[1][0] if len(ins[1]) == 2 else ins[0][-1])
                elif op in ("LSTM", "DynamicQuantizeLSTM"):
                    X, W, R = ins[0], ins[1], ins[2]
                    m = X[0] * X[1] * W[0] * W[1] * (W[2] + R[2])
                    ops["LSTM steps"] += X[0]
            except Exception:
                unk[op] += 1
            if m:
                macs[grp] += m
            ops[op] += 1
        return outs, macs, ops


    def report(model, macs, seconds, extra=""):
        tot = sum(macs.values())
        print(f"== {model}: {seconds:.2f} s of audio, {tot / 1e9:.2f} GMAC -> {tot / 1e9 / seconds:.2f} GMAC per second of audio {extra}")
        for g, v in macs.most_common(6):
            print(f"     {v / 1e9 / seconds:7.3f} GMAC/s  {g}")
        return tot / seconds


    results = {}

    # ---- Piper (VITS): phoneme ids from the voice's map, pads between ------- #
    for q in ("medium", "high"):
        cfg = json.load(open(f"{D}/piper_lessac_{q}.onnx.json"))
        pm = cfg["phoneme_id_map"]
        sr = cfg["audio"]["sample_rate"]
        text = "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst ʌv ðə tˈiː tˈiː ˈɛs sˈɪstəm ɔn ðə bˈoːɹd."
        ids = [pm["^"][0]]
        for ch in text:
            if ch in pm:
                ids += [pm[ch][0], pm["_"][0]]
        ids += [pm["$"][0]]
        feeds = {"input": np.array([ids], np.int64), "input_lengths": np.array([len(ids)], np.int64),
                 "scales": np.array([0.667, 1.0, 0.8], np.float32)}
        outs, macs, ops = run(f"{D}/piper_lessac_{q}.onnx", feeds, f"piper_{q}")
        secs = outs[0].size / sr
        results[f"piper_{q}"] = report(f"Piper lessac-{q} ({sr} Hz, {len(ids)} ids)", macs, secs)

    # ---- Kokoro / Kitten (StyleTTS2): ids 1..150 with 0 pads ----------------- #
    for name, path, sr in (("kokoro", "kokoro.onnx", 24000), ("kitten_nano", "kitten_nano.onnx", 24000),
                           ("kitten_mini", "kitten_mini.onnx", 24000)):
        T = 80
        ids = [0] + list(rng.integers(16, 150, T)) + [0]
        if name == "kokoro":
            voices = np.fromfile(f"{D}/kokoro_af_heart.bin", np.float32).reshape(-1, 1, 256)
            style = voices[len(ids) - 2]
        else:
            style = (rng.standard_normal((1, 256)) * 0.1).astype(np.float32)
        feeds = {"input_ids": np.array([ids], np.int64), "style": style.astype(np.float32),
                 "speed": np.array([1.0], np.float32)}
        try:
            outs, macs, ops = run(f"{D}/{path}", feeds, name)
        except Exception as e:
            print(f"== {name}: run failed: {str(e)[:200]}")
            continue
        secs = outs[0].size / sr
        results[name] = report(f"{name} ({sr} Hz, {T} ids)", macs, secs)

    # ---- Supertonic-3: text encoder, duration, N flow steps, vocoder --------- #
    tts = json.load(open(f"{D}/supertonic_tts.json"))
    sr = tts["ae"]["sample_rate"]
    hop = tts["ae"]["base_chunk_size"] * tts["ttl"]["chunk_compress_factor"]
    idx = json.load(open(f"{D}/supertonic_unicode_indexer.json"))
    text = "Hello world, this is a test of the text to speech system on the board."
    tid = [idx[ord(c)] if ord(c) < len(idx) and idx[ord(c)] >= 0 else 0 for c in text]
    T = len(tid)
    st = json.load(open(f"{D}/supertonic_F1.json"))
    sttl = np.array(st["style_ttl"]["data"], np.float32).reshape(st["style_ttl"]["dims"])
    sdp = np.array(st["style_dp"]["data"], np.float32).reshape(st["style_dp"]["dims"])
    tmask = np.ones((1, 1, T), np.float32)
    ti = np.array([tid], np.int64)
    o, m_te, _ = run(f"{D}/supertonic_text_encoder.onnx", {"text_ids": ti, "style_ttl": sttl, "text_mask": tmask}, "st_te")
    temb = o[0]
    o, m_dp, _ = run(f"{D}/supertonic_duration_predictor.onnx", {"text_ids": ti, "style_dp": sdp, "text_mask": tmask}, "st_dp")
    dur = float(np.asarray(o[0]).reshape(-1)[0])
    L = int(np.ceil(dur * sr / hop))
    steps = 5
    lat = rng.standard_normal((1, 144, L)).astype(np.float32)
    m_ve = collections.Counter()
    for k in range(steps):
        o, mm, _ = run(f"{D}/supertonic_vector_estimator.onnx",
                       {"noisy_latent": lat, "text_emb": temb, "style_ttl": sttl,
                        "latent_mask": np.ones((1, 1, L), np.float32), "text_mask": tmask,
                        "current_step": np.array([k], np.float32), "total_step": np.array([steps], np.float32)}, "st_ve")
        lat = o[0]
        m_ve.update(mm)
    o, m_voc, _ = run(f"{D}/supertonic_vocoder.onnx", {"latent": lat}, "st_voc")
    secs = np.asarray(o[0]).size / sr
    tot = collections.Counter({"text_encoder": sum(m_te.values()), "duration": sum(m_dp.values()),
                               f"vector_estimator x{steps}": sum(m_ve.values()), "vocoder": sum(m_voc.values())})
    results["supertonic3"] = report(f"Supertonic-3 ({sr} Hz, {T} chars, {L} latent frames, {steps} steps)", tot, secs)

    # ---- TinyTTS: decoder + flow per frame, text encoder per phoneme --------- #
    TT = f"{D}/tinytts_repo/onnx"
    Tf = 400
    g = rng.standard_normal((1, 80, 1)).astype(np.float32)
    o, m_dec, _ = run(f"{TT}/decoder.onnx", {"z": rng.standard_normal((1, 80, Tf)).astype(np.float32), "g": g}, "tt_dec")
    samples = np.asarray(o[0]).size
    o, m_flow, _ = run(f"{TT}/flow.onnx", {"z_p": rng.standard_normal((1, 80, Tf)).astype(np.float32),
                                            "y_mask": np.ones((1, 1, Tf), np.float32), "g": g}, "tt_flow")
    P = 60
    o, m_te, _ = run(f"{TT}/text_encoder.onnx", {
        "phone_ids": rng.integers(1, 50, (1, P)).astype(np.int64), "phone_lengths": np.array([P], np.int64),
        "tone_ids": np.zeros((1, P), np.int64), "language_ids": np.zeros((1, P), np.int64),
        "bert": np.zeros((1, 1024, P), np.float32), "ja_bert": np.zeros((1, 768, P), np.float32),
        "speaker_id": np.zeros((1,), np.int64)}, "tt_te")
    secs = samples / 44100
    tot = collections.Counter({"decoder": sum(m_dec.values()), "flow": sum(m_flow.values()),
                               f"text_encoder ({P} phonemes, per {secs:.1f} s)": sum(m_te.values())})
    results["tinytts"] = report(f"TinyTTS (44100 Hz, {Tf} frames, hop {samples // Tf})", tot, secs)

    json.dump(results, open(f"{D}/gmac_per_s.json", "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "assets", "screen"),
                    help="download / work directory (~1.2 GB)")
    ap.add_argument("cmd", choices=("fetch", "measure", "all"))
    a = ap.parse_args(argv)
    if a.cmd in ("fetch", "all"):
        fetch(a.dir)
    if a.cmd in ("measure", "all"):
        measure(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
