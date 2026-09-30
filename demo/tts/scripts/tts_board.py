#!/usr/bin/env python3
"""
tts_board.py — build libpiper_tts.so + tts_bench on the KV260 and run the
TTS board gate (doc/plans/TTS_PLAN.md §4): the PCM of real utterances bit-exact
with the specification (piper_vits.synthesize_chunked), ms per chunk, the
real-time factor, tts_open time, CMA and the close / re-open cycle.

  upload  project sources (demo/tts/build/piper_project, generate_tts_project.py)
          -> <remote dir>/piper_project (weights/ and build/ excluded)
  weights weights/*.dat (dp.dat included) -> <weights dir>/weights (only
          changed files); weights/voice.json -> <weights dir>/ (the chat
          server's front end, demo/chat/piper_backend.py)
  build   cmake -DINFERENCE_WEIGHTS_DIR=<weights dir> + make tts_bench piper_tts
          (and build_prof/ with -DINFERENCE_PROFILING=ON when --profile)
  install <remote dir>/lib/libpiper_tts.so (remote dir: the chat install
          directory, demo/chat/chat_config.json remote.dir)
  run     tts_bench on utts.bin (z_p of the study's sentences from
          piper_vits.front_end, seeded by name) and ids.bin (the encoder
          cases: every encode bucket); pcm.bin and enc.bin downloaded
  check   every sample against the specification (chunk_forward), every
          encoder output against encoder_forward; WAVs next to the project

The board lock is held for the whole session.  Stop the chat server first
(demo/chat/deploy.py --stop) — it owns the FPGA.

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/tts_board.py
           [--project demo/tts/build/piper_project] [--utts eval00,eval10] [--reps 3]
           [--profile] [--reopen] [--skip-build] [--install-only]
           [--remote-dir /root/kv260_chat] [--weights-dir /root/piper_weights]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import wave
import zlib
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))
sys.path.insert(0, os.path.join(REPO, "demo", "chat", "scripts"))
sys.path.insert(0, os.path.join(REPO, "demo", "bert_squad", "scripts"))

import piper_vits as pv                                             # noqa: E402

DEFAULT_PROJECT = os.path.join(os.path.dirname(HERE), "build", "piper_project")
WEIGHTS_DIR = "/root/piper_weights"
RUN_DIR = "/tmp/tts_bench"
SR = 22050


def load(project: str):
    """(project.json, weights, exponents) of a generated project."""
    from src.piper import load_weights
    summary = json.load(open(os.path.join(project, "project.json")))
    assets = summary["assets"]
    W = load_weights(os.path.join(assets, "en_US-lessac-medium.onnx"))
    E = json.load(open(os.path.join(assets, "exponents.json")))["exponents"]
    return summary, W, E


def utterances(W, assets: str, names) -> dict:
    """{name: z_p [192][frames] float32} of texts.json's sentences, noise seeded
    by the name (as piper_study.py study)."""
    texts = json.load(open(os.path.join(assets, "texts.json")))
    return {n: pv.front_end(W, texts[n]["ids"], seed=zlib.crc32(n.encode())) for n in names}


def write_utts(path: str, zps: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(zps)))
        for zp in zps.values():
            f.write(struct.pack("<i", zp.shape[1]))
            f.write(np.ascontiguousarray(zp, "<f4").tobytes())


def reference(project: str, W, E, zps: dict) -> dict:
    """The specification's samples per utterance, cached in the project by
    name and z_p checksum (a regenerated project — new exponents — starts
    without the cache)."""
    out = {}
    for n, zp in zps.items():
        key = zlib.crc32(np.ascontiguousarray(zp, "<f4").tobytes())
        path = os.path.join(project, "reference", f"{n}_{key:08x}.npy")
        if os.path.exists(path):
            ref = np.load(path)
            if ref.shape == (zp.shape[1] * pv.HOP,):
                out[n] = ref
                continue
        t0 = time.time()
        out[n] = pv.synthesize_chunked(W, E, zp).astype(np.int16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, out[n])
        print(f"  spec {n}: {zp.shape[1]} frames ({time.time() - t0:.0f} s)", flush=True)
    return out


def encoder_cases(assets: str) -> dict:
    """{name: phoneme ids} over every encode bucket (32 .. 400): prefixes and
    concatenations of the study's sentences."""
    t = {k: v["ids"] for k, v in json.load(open(os.path.join(assets, "texts.json"))).items()}
    both = t["eval00"] + t["eval02"]
    long = (t["eval10"] + t["eval00"] + t["eval01"])[:400]
    return {"eval10": t["eval10"], "ids20": t["eval00"][:20], "ids60": t["eval00"][:60],
            "eval00": t["eval00"], "ids162": both, "ids400": long}


def write_ids(path: str, seqs: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(seqs)))
        for ids in seqs.values():
            f.write(struct.pack("<i", len(ids)))
            f.write(np.asarray(ids, "<i4").tobytes())


def dp_noise(seqs: dict) -> dict:
    """{name: z [2][n]}: the duration predictor's noise per encoder case
    (standard normal * 0.8, seeded by the name)."""
    return {k: np.random.default_rng(zlib.crc32(k.encode())).standard_normal((2, len(ids))) * 0.8
            for k, ids in seqs.items()}


def write_noise(path: str, zs: dict) -> None:
    with open(path, "wb") as f:
        for z in zs.values():
            f.write(np.ascontiguousarray(z, "<f8").tobytes())


def check_dur(dur_path: str, seqs: dict, zs: dict, W, E_enc) -> dict:
    """tts_duration's logw (dur.bin) against piper_vits.duration_predictor_seq on
    the spec encoder's x."""
    EW = pv.encoder_weights(W)
    raw = np.fromfile(dur_path, "<f8")
    rep, k = {}, 0
    for name, ids in seqs.items():
        n = len(ids)
        got = raw[k:k + n]
        k += n
        x, _ = pv.encoder_forward(EW, E_enc, ids)
        want = pv.duration_predictor_seq(W, x.T, zs[name])
        bad = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        rep[name] = {"ids": n, "mismatches": bad}
        print(f"  [duration {name}] {n} ids: " + ("bit-exact" if not bad else f"{bad} values differ"), flush=True)
    return rep


def check_enc(enc_path: str, seqs: dict, W, E_enc) -> dict:
    """tts_encode's x, m_p, logs_p (enc.bin) against piper_vits.encoder_forward."""
    EW = pv.encoder_weights(W)
    raw = np.fromfile(enc_path, "<f4")
    rep, k = {}, 0
    for name, ids in seqs.items():
        n = len(ids)
        got = raw[k:k + 3 * 192 * n].reshape(3, 192, n)
        k += 3 * 192 * n
        x, st = pv.encoder_forward(EW, E_enc, ids)
        want = np.stack([x.T, st[:, :192].T, st[:, 192:].T])
        bad = int((got.view(np.uint32) != want.astype(np.float32).view(np.uint32)).sum())
        rep[name] = {"ids": n, "mismatches": bad}
        print(f"  [encode {name}] {n} ids: " + ("bit-exact" if not bad else f"{bad} values differ"), flush=True)
    return rep


def write_wav(path: str, pcm: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, "<i2").tobytes())


def check_pcm(pcm_path: str, zps: dict, refs: dict, wav_dir: str = None, tag: str = "") -> dict:
    """Every utterance's samples of pcm.bin against the specification."""
    raw = np.fromfile(pcm_path, "<i2")
    rep, k = {}, 0
    for n, zp in zps.items():
        ns = zp.shape[1] * pv.HOP
        got, want = raw[k:k + ns], refs[n]
        k += ns
        bad = np.flatnonzero(got != want) if got.shape == want.shape else np.arange(ns)
        rep[n] = {"samples": int(ns), "mismatches": int(bad.size),
                  "first_mismatch": int(bad[0]) if bad.size else None,
                  "max_abs_diff": int(np.abs(got.astype(np.int32) - want).max()) if got.shape == want.shape
                  else None}
        if wav_dir:
            os.makedirs(wav_dir, exist_ok=True)
            write_wav(os.path.join(wav_dir, f"{n}{tag}.wav"), got)
        print(f"  [{n}] {ns} samples: " + ("bit-exact" if not bad.size else
              f"{bad.size} mismatches (first at {bad[0]}, max diff {rep[n]['max_abs_diff']})"), flush=True)
    if k != raw.size:
        rep["_size"] = {"expected": k, "got": int(raw.size)}
        print(f"  pcm.bin holds {raw.size} samples, expected {k}", flush=True)
    return rep


def parse(out: str) -> dict:
    res = {"utts": [], "encs": [], "profiles": {}}
    phase = None
    for line in out.splitlines():
        for key in ("TTS_OPEN", "TTS_UTT", "TTS_ENC", "TTS_DUR", "TTS_REOPEN", "TTS_SUMMARY"):
            if line.startswith(key + ":"):
                d = json.loads(line[len(key) + 1:])
                if key == "TTS_UTT":
                    res["utts"].append(d)
                elif key == "TTS_ENC":
                    res["encs"].append(d)
                elif key == "TTS_DUR":
                    res.setdefault("durs", []).append(d)
                else:
                    res[key.lower()] = d
        if line.startswith("PROFILE_PHASE:"):
            phase = line.split(":", 1)[1].strip()
        elif line.startswith("LAYERS_JSON:") and phase:
            res["profiles"][phase] = json.loads(line[len("LAYERS_JSON:"):])
            phase = None
    return res


def breakdown(profile: dict, layers: list, per: int = 1) -> dict:
    """{kind: ms per chunk} from the profiler's per-layer totals."""
    kind = {L["i"]: L["kind"] for L in layers}
    out = {}
    for L in profile["layers"]:
        if L["calls"]:
            k = kind.get(L["i"], "other")
            out[k] = out.get(k, 0.0) + L["total_us"] / 1000.0 / per
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def sync_voice(session, project: str, weights_dir: str) -> str:
    """weights/voice.json -> <weights_dir>/ when its SHA-1 differs from the
    board's copy; removes the frontend.npz that older installs left there
    (the server has not read it since the library runs the encoder and the
    duration predictor)."""
    import hashlib
    sent = []
    out, _, _ = session.exec(f"test -f {weights_dir}/frontend.npz && rm -f {weights_dir}/frontend.npz "
                             f"&& echo removed", timeout=30)
    if "removed" in out:
        sent.append("obsolete frontend.npz removed")
    sftp = session._client.open_sftp()                           # noqa: SLF001
    try:
        for name in ("voice.json",):
            local = os.path.join(project, "weights", name)
            sha = hashlib.sha1(open(local, "rb").read()).hexdigest()
            out, _, _ = session.exec(f"sha1sum {weights_dir}/{name} 2>/dev/null", timeout=60)
            if out.split()[:1] != [sha]:
                sftp.put(local, f"{weights_dir}/{name}")
                sent.append(f"{name} uploaded")
    finally:
        sftp.close()
    return ", ".join(sent) if sent else "unchanged"


def main(argv=None) -> int:
    import llm_board
    from deploy_and_run import _stream_exec, board_lock, lock_path, sync_weights
    from src.remote import RemoteSession, _green, _red

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--bert-config", default=None)
    ap.add_argument("--utts", default="eval00,eval10")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="also build and run with the profiler")
    ap.add_argument("--reopen", action="store_true", help="tts_close + tts_open cycle")
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--install-only", action="store_true",
                    help="upload, build and install libpiper_tts.so; no bench or checks")
    ap.add_argument("--jobs", type=int, default=1, help="make -j on the board (default 1)")
    ap.add_argument("--remote-dir", default=None,
                    help="install directory on the board: <dir>/lib/libpiper_tts.so and "
                         "<dir>/piper_project (default: remote.dir of demo/chat/chat_config.json, "
                         "else /root/kv260_chat)")
    ap.add_argument("--weights-dir", default=WEIGHTS_DIR)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    args = ap.parse_args(argv)
    if args.install_only and args.skip_build:
        ap.error("--install-only builds; it cannot be combined with --skip-build")
    project = os.path.abspath(args.project)
    cfg = llm_board.bert_config(args.bert_config)
    summary = json.load(open(os.path.join(project, "project.json")))
    layers = json.load(open(os.path.join(project, "layers.json")))
    names = [n for n in args.utts.split(",") if n]
    zps = refs = None
    seqs = {}
    if not args.install_only:
        summary, W, E = load(project)
        zps = utterances(W, summary["assets"], names)
        write_utts(os.path.join(project, "utts.bin"), zps)
        seqs = encoder_cases(summary["assets"])
        write_ids(os.path.join(project, "ids.bin"), seqs)
        zs = dp_noise(seqs)
        write_noise(os.path.join(project, "dpz.bin"), zs)
        if not args.no_check:
            refs = reference(project, W, E, zps)
    remote_dir = args.remote_dir or llm_board.chat_remote_dir()
    remote_proj = f"{remote_dir}/piper_project"
    lib = f"{remote_dir}/lib/libpiper_tts.so"
    results = {"project": summary, "utts": names}
    with board_lock(lock_path(cfg, args.board_lock)):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            out, _, _ = session.exec("grep -E 'CmaFree|CmaTotal' /proc/meminfo | tr -s ' '", timeout=15)
            print(f"board: {out.strip()}", flush=True)
            if not args.skip_build:
                t0 = time.monotonic()
                n = llm_board.upload_project(session, project, remote_proj,
                                             skip=("reference", "wav", "board_gate.json"))
                print(f"  upload: {n} files ({time.monotonic() - t0:.0f} s)", flush=True)
                s = sync_weights(session, Path(project) / "weights", args.weights_dir)
                print(f"  weights: {'OK' if s.ok else 'FAIL'} {s.output} ({s.duration:.0f} s)", flush=True)
                if not s.ok:
                    return 1
                print(f"  voice: {sync_voice(session, project, args.weights_dir)}", flush=True)
                llm_board.build(session, remote_proj, cfg, summary["active_kernels"], args.profile,
                                args.jobs, args.weights_dir, targets="tts_bench piper_tts")
                session.exec_checked(f"mkdir -p {os.path.dirname(lib)} && "
                                     f"cp {remote_proj}/build/libpiper_tts.so {lib}", timeout=30)
                out, _, _ = session.exec(f"nm -D --defined-only {lib} | awk '{{print $3}}' | sort",
                                         timeout=30)
                results["exported"] = [s for s in out.split() if s]
                print(f"  libpiper_tts.so exports: {results['exported']}", flush=True)
                if args.install_only:
                    print(f"installed {lib} (weights {args.weights_dir})")
                    return 0
            sftp = session._client.open_sftp()                   # noqa: SLF001
            session.exec_checked(f"mkdir -p {RUN_DIR}", timeout=15)
            sftp.put(os.path.join(project, "utts.bin"), f"{RUN_DIR}/utts.bin")
            sftp.put(os.path.join(project, "ids.bin"), f"{RUN_DIR}/ids.bin")
            sftp.put(os.path.join(project, "dpz.bin"), f"{RUN_DIR}/dpz.bin")
            runs = [("build", "main", args.reps, args.reopen)] + (
                [("build_prof", "prof", 1, False)] if args.profile else [])
            for bdir, tag, reps, reopen in runs:
                cmd = (f"cd {RUN_DIR} && {remote_proj}/{bdir}/tts_bench -i utts.bin -o pcm_{tag}.bin "
                       f"-R {reps} {'-r' if reopen else ''} -e ids.bin -E enc_{tag}.bin "
                       f"-d dpz.bin -D dur_{tag}.bin")
                t0 = time.monotonic()
                out, err, rc = _stream_exec(session, cmd, timeout=3600,
                                            on_stderr_line=lambda ln: print("    " + ln, flush=True))
                print(f"  tts_bench [{tag}] rc={rc} ({time.monotonic() - t0:.0f} s)", flush=True)
                if rc != 0:
                    print(out[-2000:], err[-2000:])
                    return 1
                res = parse(out)
                sftp.get(f"{RUN_DIR}/pcm_{tag}.bin", os.path.join(project, f"pcm_board_{tag}.bin"))
                sftp.get(f"{RUN_DIR}/enc_{tag}.bin", os.path.join(project, f"enc_board_{tag}.bin"))
                sftp.get(f"{RUN_DIR}/dur_{tag}.bin", os.path.join(project, f"dur_board_{tag}.bin"))
                if tag == "main":
                    results["bench"] = res
                else:
                    chunks = res["utts"][0]["chunks"]
                    results["profile"] = {ph: breakdown(p, layers, per=chunks if ph == "chunk" else 1)
                                          for ph, p in res["profiles"].items()}
                    results["profile_layers"] = {
                        ph: [{k: ly[k] for k in ("i", "name", "calls", "mean_us", "min_us", "total_us")
                              if k in ly} for ly in p.get("layers", []) if ly.get("calls")]
                        for ph, p in res["profiles"].items()}
            sftp.close()
        finally:
            session.close()
    bench = results["bench"]
    o = bench.get("tts_open", {})
    print(f"tts_open {o.get('open_ms')} ms, CmaFree {o.get('cma_free_kb_before')} -> "
          f"{o.get('cma_free_kb_after')} kB "
          f"({(o.get('cma_free_kb_before', 0) - o.get('cma_free_kb_after', 0)) / 1024:.1f} MB)")
    for u in bench["utts"]:
        print(f"utterance {u['i']}: {u['frames']} frames, {u['audio_s']:.2f} s audio in "
              f"{u['best_ms']:.0f} ms (RTF {u['rtf']:.3f}); chunk ms {u['chunk_ms']}; "
              f"repetitions identical {u['reps_identical']}")
    if "tts_reopen" in bench:
        print(f"re-open: {bench['tts_reopen']}")
    for e in bench.get("encs", []):
        print(f"encode {e['i']}: {e['n']} ids (bucket {e['bucket']}) in {e['best_ms']:.1f} ms; "
              f"repetitions identical {e['reps_identical']}")
    for e in bench.get("durs", []):
        print(f"duration {e['i']}: {e['n']} ids in {e['best_ms']:.1f} ms; repetitions identical {e['reps_identical']}")
    for ph, bd in results.get("profile", {}).items():
        print(f"profile ms per {'chunk' if ph == 'chunk' else ph + ' call'}: "
              + ", ".join(f"{k} {v:.2f}" for k, v in bd.items()))
    ok = all(u["reps_identical"] for u in bench["utts"]) and \
        all(e["reps_identical"] for e in bench.get("encs", []) + bench.get("durs", [])) and \
        bench.get("tts_reopen", {}).get("identical", True)
    if not args.no_check:
        results["check"] = check_pcm(os.path.join(project, "pcm_board_main.bin"), zps, refs,
                                     os.path.join(project, "wav"), "_board")
        E_enc = json.load(open(os.path.join(summary["assets"], "exponents.json")))["encoder"]
        results["check_encoder"] = check_enc(os.path.join(project, "enc_board_main.bin"), seqs, W, E_enc)
        results["check_duration"] = check_dur(os.path.join(project, "dur_board_main.bin"), seqs, zs, W, E_enc)
        ok = ok and all(r.get("mismatches", 1) == 0 for r in results["check"].values()) and \
            all(r["mismatches"] == 0 for r in results["check_encoder"].values()) and \
            all(r["mismatches"] == 0 for r in results["check_duration"].values())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print(_green("BOARD GATE: PCM bit-exact") if ok else _red("BOARD GATE: MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
