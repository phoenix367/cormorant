#!/usr/bin/env python3
"""
tts_host_emu.py — run the generated Piper project on the HOST: the generated
inference.c, tts_api.c and tts_bench.c compiled unchanged against the
software ConvKernel and the malloc-backed buffers of
inference-scheduler/test/host_emu.py, then every sample of tts_bench (the
study's sentences, chunk by chunk, stitched by the library) compared bit for
bit with the specification (piper_vits.synthesize_chunked) — the board gate
without the board.

With --lib-check, libpiper_tts.so is built the same way and driven by the chat
server's backend (demo/chat/piper_backend.py: LibTtsEngine through ctypes —
tts_encode, tts_duration and the chunks —, the espeak-ng phonemizer of the
host, the length regulator and the noise of piper_vits.front_end) for a few
sentences: its z_p must equal the front end on the spec encoder
(encoder_forward) and the spec duration predictor (duration_predictor_seq),
its samples the specification on that z_p.

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/tts_host_emu.py
           [--project demo/tts/build/piper_project] [--utts eval00,eval10] [--incoherent]
           [--lib-check] [--texts "Hello there. How are you?"]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tts_board as tb                                              # noqa: E402

sys.path.insert(0, os.path.join(tb.REPO, "inference-scheduler", "test"))
import host_emu                                                     # noqa: E402


def _emu_sources(work: str) -> str:
    emu = os.path.join(work, "emu")
    os.makedirs(emu, exist_ok=True)
    for name, text in (("inference_buf_emu.c", host_emu.buf_emu_source()), ("emu_common.h", host_emu._COMMON),
                       ("xvectoropkernel.h", host_emu._VOP), ("xmatmulkernel.h", host_emu._MM),
                       ("xconvkernel.h", host_emu._CONV)):
        with open(os.path.join(emu, name), "w") as f:
            f.write(text)
    return emu


def build(project: str, work: str, incoherent: bool = False, shared: bool = False) -> str:
    from src._conv_hw_config import CONV_TILE_IC
    from src._matmul_hw_config import MATMUL_TILE_M
    emu = _emu_sources(work)
    exe = os.path.join(work, "libpiper_tts_host.so" if shared else "tts_bench_host")
    cmd = [host_emu.which_cc(), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
           "-Wno-unused-function", "-pthread", f"-DEMU_TILE_M={MATMUL_TILE_M}",
           *(["-DEMU_INCOHERENT"] if incoherent else []),
           f"-DEMU_CONV_TILE_IC={CONV_TILE_IC}", f'-DINFERENCE_WEIGHTS_DIR="{project}"',
           f'-DTTS_API_WEIGHTS_DIR="{project}"', '-DTTS_MODEL_NAME="piper-lessac-medium"',
           "-I", os.path.join(project, "include"), "-I", os.path.join(project, "test"), "-I", emu,
           os.path.join(project, "src", "inference.c"), os.path.join(emu, "inference_buf_emu.c"),
           os.path.join(project, "test", "tts_api.c"), os.path.join(project, "test", "tts_dp.c"),
           *(["-shared", "-fPIC"] if shared else [os.path.join(project, "test", "tts_bench.c")]),
           "-lm", "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("compile failed:\n" + r.stdout + r.stderr[-5000:])
    print(f"compiled with -Werror ({time.time() - t0:.0f} s)", flush=True)
    return exe


LIB_TEXTS = ("Hello! I am a small assistant running on an FPGA board.",
             "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background.")


def lib_check(project: str, work: str, texts) -> bool:
    """The chat server's PiperBackend over the host-built libpiper_tts.so."""
    import numpy as np
    sys.path.insert(0, os.path.join(tb.REPO, "demo", "chat"))
    from chat_backend import CancelToken, SpeechRequest
    from piper_backend import LibTtsEngine, PiperBackend
    lib = build(project, work, shared=True)
    b = PiperBackend(LibTtsEngine(lib, project), os.path.join(project, "weights"))
    b.load_host()
    b.load()
    summary, W, E = tb.load(project)
    enc = tb.pv.library_encoder(tb.pv.encoder_weights(W),
                                json.load(open(os.path.join(summary["assets"], "exponents.json")))["encoder"])
    ok = b.engine.has_encode and b.engine.has_duration
    print(f"  the library's text encoder (tts_encode): {'yes' if b.engine.has_encode else 'MISSING'}, "
          f"duration predictor (tts_duration): {'yes' if b.engine.has_duration else 'MISSING'}")
    try:
        for i, text in enumerate(texts):
            job = b.prepare_speech(SpeechRequest(model=b.model_id, input=text, seed=i))
            t0 = time.time()
            out = list(b.synthesize(job, CancelToken()))
            # the backend's z_p (tts_encode through ctypes) == the front end on the spec encoder
            zref = [tb.pv.front_end(None, ids, seed=job.seed + j, fast_erf=True, encoder=enc,
                                    duration=lambda x, z: tb.pv.duration_predictor_seq(W, x, z))
                    for j, ids in enumerate(job.groups)]
            zsame = all(a.shape == r.shape and np.array_equal(a, r) for a, r in zip(job.utterances, zref, strict=True))
            ok &= zsame
            got = np.frombuffer(b"".join(out[:-1]), "<i2")
            want = np.concatenate([tb.pv.synthesize_chunked(W, E, zp) for zp in job.utterances])
            same = got.shape == want.shape and np.array_equal(got, want)
            ok &= same
            print(f"  [{text[:40]}...] {len(job.utterances)} utterance(s), {job.samples} samples, "
                  f"front end {job.frontend_ms:.0f} ms, library {time.time() - t0:.1f} s: z_p "
                  + ("bit-exact" if zsame else "MISMATCH") + ", samples " + ("bit-exact" if same else "MISMATCH"),
                  flush=True)
            tb.write_wav(os.path.join(work, f"lib_check_{i}.wav"), got)
    finally:
        b.close()
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--project", default=tb.DEFAULT_PROJECT)
    ap.add_argument("--utts", default="eval00,eval10")
    ap.add_argument("--work", default=None)
    ap.add_argument("--incoherent", action="store_true",
                    help="separate CPU / DDR buffer copies: every cache sync must be right")
    ap.add_argument("--lib-check", action="store_true",
                    help="the chat server's backend over a host-built libpiper_tts.so instead")
    ap.add_argument("--texts", action="append", default=None, help="--lib-check sentences")
    args = ap.parse_args(argv)
    project = os.path.abspath(args.project)
    work = args.work or os.path.join(project, "host_emu")
    os.makedirs(work, exist_ok=True)
    if args.lib_check:
        ok = lib_check(project, work, args.texts or LIB_TEXTS)
        print("LIB CHECK: " + ("PiperBackend samples bit-exact with the specification" if ok else "MISMATCH"))
        return 0 if ok else 1
    exe = build(project, work, args.incoherent)
    summary, W, E = tb.load(project)
    zps = tb.utterances(W, summary["assets"], [n for n in args.utts.split(",") if n])
    utts = os.path.join(work, "utts.bin")
    tb.write_utts(utts, zps)
    refs = tb.reference(project, W, E, zps)
    pcm = os.path.join(work, "pcm_host.bin")
    t0 = time.time()
    r = subprocess.run([exe, "-i", utts, "-o", pcm, "-R", "1", "-r"], capture_output=True, text=True,
                       cwd=work, env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
    print(r.stderr[-3000:])
    if r.returncode:
        print(r.stdout[-3000:])
        return 1
    res = tb.parse(r.stdout)
    print(f"tts_bench on the host emulation: {time.time() - t0:.0f} s, re-open {res.get('tts_reopen')}",
          flush=True)
    rep = tb.check_pcm(pcm, zps, refs)
    ok = all(v.get("mismatches", 1) == 0 for v in rep.values()) and \
        res.get("tts_reopen", {}).get("identical", False)
    # the text encoder: every bucket
    seqs = tb.encoder_cases(summary["assets"])
    ids_bin, enc_bin = os.path.join(work, "ids.bin"), os.path.join(work, "enc_host.bin")
    z_bin, dur_bin = os.path.join(work, "dpz.bin"), os.path.join(work, "dur_host.bin")
    tb.write_ids(ids_bin, seqs)
    zs = tb.dp_noise(seqs)
    tb.write_noise(z_bin, zs)
    r = subprocess.run([exe, "-i", utts, "-o", pcm, "-R", "2", "-e", ids_bin, "-E", enc_bin, "-d", z_bin,
                        "-D", dur_bin], capture_output=True,
                       text=True, cwd=work, env=dict(os.environ, INFERENCE_HOST_THREADS="4"))
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    E_enc = json.load(open(os.path.join(summary["assets"], "exponents.json")))["encoder"]
    erep = tb.check_enc(enc_bin, seqs, W, E_enc)
    drep = tb.check_dur(dur_bin, seqs, zs, W, E_enc)
    pr = tb.parse(r.stdout)
    ok = ok and all(v["mismatches"] == 0 for v in list(erep.values()) + list(drep.values())) and \
        all(e["reps_identical"] for e in pr["encs"] + pr.get("durs", []))
    rep["encoder"], rep["duration"] = erep, drep
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump({"check": rep, "bench": res}, f, indent=1)
    print("HOST EMULATION: " + ("PCM bit-exact with the specification" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
