#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
cpu_baseline.py — the CPU baseline of the FPGA results: BERT-SQuAD,
SmolLM2, SmolVLM, Piper and Moonshine with onnxruntime on the KV260's four
Cortex-A53 cores (doc/plans/BERT_PLAN.md §4, doc/plans/CHAT_PLAN.md §26,
doc/plans/TTS_PLAN.md §8, doc/plans/ASR_PLAN.md §9).

  models  bert: bertsquad-12.onnx of the ONNX model zoo (the demo's assets,
          else downloaded); smollm2: the export onnx/model.onnx at the revision
          demo/chat/scripts/llm_models.json pins (downloaded), with its head on
          the last position only — a Slice before the head MatMul, as the FPGA
          library computes a prefill's logits; prepare.py checks that the
          logits equal the export's last row; smolvlm: its three exports
          (vision encoder + connector, token embedding, decoder; the revision
          demo/chat/scripts/vlm_study_inputs.json pins), the decoder edited
          likewise; piper: the voice's own ONNX (demo/tts/assets, the
          revision piper_study.py pins), fp32 only; moonshine: the
          onnx-community export (encoder, decoder, decoder with past) whose
          weights equal the checkpoint asr_study_inputs.json pins.  Every
          source is SHA-256-checked
          (PINS); the int8 copies are onnxruntime's dynamic quantisation.  On
          the host in build/venv (onnxruntime 1.23.2 = the board's, onnx,
          sympy), the models in build/ (--work: ~7 GB with every model).
  board   onnxruntime 1.23.2 in a venv in --remote-dir (beside the system
          Python the chat server uses; pip needs PyPI); the models, the
          inputs and board_bench.py uploaded where their SHA-256 differs; each
          run in a systemd scope capped below MemAvailable, after the page
          cache is dropped; --remote-dir removed afterwards unless --keep.
  report  bert: latency, EM / F1 against the gold answers (the demo's
          scoring), the spans against fp32's (= the float model's);
          smollm2 / smolvlm: decode ms / token, prefill 16 / 64 / 256 tokens
          (smolvlm: llm_image, the image through the vision encoder), the
          greedy tokens against fp32's.  With the FPGA's results — bert:
          demo/bert_squad/build/results.json of run_demo.py (the default);
          smollm2 / smolvlm: an llm_board.py --out JSON (--fpga) — the
          speed-ups.
          Written to build/<model>_results.json (--out).

The chat server owns the FPGA and most of the board's memory: stop it
(demo/chat/deploy.py --stop) or pass --stop-server (it is restarted after).
The board lock is held for the board part (src/remote/lock.py).

usage:  inference-scheduler/.venv/bin/python tools/cpu_baseline/cpu_baseline.py bert [--fpga RESULTS_JSON]
        ... cpu_baseline.py smollm2 --model 135m|360m [--fpga LLM_BOARD_JSON]
        ... cpu_baseline.py smolvlm [--fpga LLM_BOARD_JSON] [--images 39769,1268]
        ... cpu_baseline.py piper [--fpga TTS_BOARD_JSON] [--utts eval00,eval10]
        ... cpu_baseline.py moonshine --model base [--fpga ASR_BOARD_JSON] [--utts all]
        options: [--config demo/bert_squad/bert_squad_config.json] [--engines fp32,int8]
                 [--threads 4,1] [--stop-server] [--keep] [--remote-dir DIR] [--prepare-only]
                 [--out JSON] [--work DIR] [--board-lock FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parent
REPO = TOOL.parents[1]
BUILD = TOOL / "build"
ORT = "1.23.2"                  # the newest onnxruntime with an aarch64 wheel for the board's Python 3.10
CHAT_DEPLOY = REPO / "demo" / "chat" / "deploy.py"
BERT_CONFIG = REPO / "demo" / "bert_squad" / "bert_squad_config.json"
BERT_FPGA = REPO / "demo" / "bert_squad" / "build" / "results.json"
for _p in (REPO / "demo" / "chat" / "scripts", REPO / "demo" / "bert_squad" / "scripts",
           REPO / "inference-scheduler"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.remote import RemoteSession, hold_board_lock                       # noqa: E402

BERT_ZOO = ("https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/"
            "bert-squad/model/bertsquad-12.onnx")
PINS = {   # the measured files (CHAT_PLAN §26: the revisions of llm_models.json)
    "bert": {"sha256": "5f0d96a9e6b8a4a2e59f9636f8bbad09ee8a0c58c8212027cc17c5fcc9659e55",
             "size": 435852736},
    "smollm2-135m-instruct": {"revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "size": 540345794,
                              "sha256": "1e790f1d6fac94097a6516ed1f07c29049097271d536d538562cd31ffc011a71"},
    "smollm2-360m-instruct": {"revision": "a10cc1512eabd3dde888204e902eca88bddb4951", "size": 1449582810,
                              "sha256": "cf419c37a4f6073a325028245d4cae2e6135765f0f2841dbe6f48c8cd261bbcd"},
    "smolvlm-256m-instruct": {"revision": "7e3e67edbbed1bf9888184d9df282b700a323964", "parts": {
        "vision_encoder": {"size": 374316497,
                           "sha256": "e27ad810eabfa7c881b409e1ca357e7a951541019add228ff73c72718fcf6218"},
        "embed_tokens": {"size": 113541419,
                         "sha256": "a68ea4058524059ed0e2873fdbc4f33b93e13ae3d3204938d32d0566d390ae00"},
        "decoder_model_merged": {"size": 540640302,
                                 "sha256": "5ae6a66fdf47680433aa6bcc512ae8f0b98c2db01bea055fc76ee37bb0389b78"}}},
}
VLM = "smolvlm-256m-instruct"
PINS["piper"] = {"revision": "c10ece1aade47bb51c153c893d14e5bf8e5b7117", "size": 63201294,
                 "sha256": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f"}
PINS["moonshine-base"] = {   # its weights equal UsefulSensors/moonshine-base @ "checkpoint", tensor for tensor
    "repo": "onnx-community/moonshine-base-ONNX", "revision": "b1e9b6aae3c3c7298f10c3798393fdf38e8fbbad",
    "checkpoint": "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf", "parts": {
        "encoder_model": {"size": 80818781,
                          "sha256": "153e128e7abd64a74ee47f2c3f585c3171c4d46cbb368b032827934c4e01e779"},
        "decoder_model": {"size": 165786949,
                          "sha256": "d957c6418d65baee2842373c6842846c3ed6c9d9fc0ef48d790b110665e18063"},
        "decoder_with_past_model": {"size": 154634914,
                                    "sha256": "2d8740fb19673870125ed6f15370dd316c8c43d266e0846a6f456b354cebf169"}}}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log(f"  downloading {url}\n    -> {dest} ({size / 1e6:.0f} MB)")
    t0, last, done = time.monotonic(), 0.0, 0
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.monotonic() - last > 10:
                last = time.monotonic()
                log(f"    {done / 1e6:.0f} / {size / 1e6:.0f} MB")
    os.replace(tmp, dest)
    log(f"    done ({time.monotonic() - t0:.0f} s)")


def pinned(pin: dict, candidates: list[Path], url: str, dest: Path) -> Path:
    """The first candidate with the pinned SHA-256 (pin: size, sha256), else url downloaded to dest."""
    for c in candidates + [dest]:
        if c.exists() and c.stat().st_size == pin["size"] and sha256(c) == pin["sha256"]:
            return c
    download(url, dest, pin["size"])
    if sha256(dest) != pin["sha256"]:
        sys.exit(f"{dest}: SHA-256 {sha256(dest)} is not the pinned {pin['sha256']}")
    return dest


# ── host: the models ─────────────────────────────────────────────────────────

def host_venv() -> Path:
    """build/venv with onnxruntime ORT (the quantizer needs onnx and sympy)."""
    py = BUILD / "venv" / "bin" / "python"
    probe = f"import onnx, sympy, onnxruntime, sys; sys.exit(onnxruntime.__version__ != {ORT!r})"
    if py.exists() and subprocess.run([str(py), "-c", probe], capture_output=True).returncode == 0:
        return py
    log(f"host venv: {py.parent.parent} (onnxruntime {ORT}, onnx, sympy)")
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(py.parent.parent)], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", f"onnxruntime=={ORT}", "onnx", "sympy", "numpy"],
                   check=True)
    return py


def derived(src: Path, dst: Path, what: str) -> Path:
    """dst = prepare.py what src, unless dst is newer than src."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    log(f"  {what}: {src.name} -> {dst.name}")
    r = subprocess.run([str(host_venv()), str(TOOL / "prepare.py"), what, str(src), str(dst)],
                       capture_output=True, text=True)
    if r.returncode:                         # the quantizer's shape-inference chatter only on failure
        sys.exit(f"prepare.py {what} {src.name} failed:\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    for line in r.stdout.splitlines():
        if line.startswith("  "):            # prepare.py's own lines (the last-row check)
            log(line)
    return dst


def bert_models() -> dict:
    assets = REPO / "demo" / "bert_squad" / "assets" / "models"
    src = pinned(PINS["bert"], [assets / "bertsquad-12.onnx", REPO / "inference-scheduler" / "bertsquad-12.onnx"],
                 BERT_ZOO, BUILD / "bertsquad-12.onnx")
    models = {"fp32": src, "int8": derived(src, BUILD / "bertsquad-12-int8.onnx", "int8")}
    log("".join(f"  {e}: {m}\n" for e, m in models.items()).rstrip())
    return models


def llm_name(model: str) -> str:
    return {"135m": "smollm2-135m-instruct", "360m": "smollm2-360m-instruct"}[model]


def llm_models(model: str) -> dict:
    name = llm_name(model)
    pin = PINS[name]
    spec = json.loads((REPO / "demo" / "chat" / "scripts" / "llm_models.json").read_text())["models"][name]
    if spec["revision"] != pin["revision"]:
        sys.exit(f"llm_models.json pins {name} at {spec['revision']}, this tool's PINS at {pin['revision']}: "
                 "update PINS (the export's SHA-256 at the new revision)")
    url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/onnx/model.onnx"
    src = pinned(pin, [], url, BUILD / f"{name}.onnx")
    last = derived(src, BUILD / f"{name}-last-row.onnx", "last-row")
    models = {"fp32": last, "int8": derived(last, BUILD / f"{name}-last-row-int8.onnx", "int8")}
    log("".join(f"  {e}: {m}\n" for e, m in models.items()).rstrip())
    return models


def llm_prompts(a) -> Path:
    """The prompts' token ids as llm_board.py makes them."""
    import llm_project as lp
    ids = lp.with_second_turns(lp.tokenize_prompts(a.prompts.split(",")),
                               [n for n in a.second_turn.split(",") if n])
    p = BUILD / "prompts.json"
    p.write_text(json.dumps({k: [int(t) for t in v] for k, v in ids.items()}))
    return p


def vlm_models() -> dict:
    """{engine: {vision, embed, decoder}}: the pinned exports, the decoder's head
    on the last position, int8 copies of all three."""
    pin = PINS[VLM]
    spec = json.loads((REPO / "demo" / "chat" / "scripts" / "vlm_study_inputs.json").read_text())["model"]
    if spec["revision"] != pin["revision"]:
        sys.exit(f"vlm_study_inputs.json pins {VLM} at {spec['revision']}, this tool's PINS at "
                 f"{pin['revision']}: update PINS (the exports' SHA-256 at the new revision)")
    src = {part: pinned(p, [], f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/onnx/{part}.onnx",
                        BUILD / f"{VLM}-{part}.onnx") for part, p in pin["parts"].items()}
    fp32 = {"vision": src["vision_encoder"], "embed": src["embed_tokens"],
            "decoder": derived(src["decoder_model_merged"], BUILD / f"{VLM}-decoder-last-row.onnx", "last-row")}
    models = {"fp32": fp32,
              "int8": {k: derived(v, v.with_name(v.stem + "-int8.onnx"), "int8") for k, v in fp32.items()}}
    for e, parts in models.items():
        log(f"  {e}: " + ", ".join(f"{k} {v.name}" for k, v in parts.items()))
    return models


def vlm_inputs(a) -> tuple:
    """(prompts JSON, images.bin) of the image prompts as llm_board.py makes
    them (its vlm_inputs: vlm_study's evaluation prompts, the gate's cached
    pixels and ids), with the full ids and the image token."""
    import llm_project as lp
    import vlm_project as vp
    import vlm_sched_check as vc
    import vlm_study as vs
    assets = vp.default_assets()
    ids = [int(x) for x in a.images.split(",")]
    items = [(i, vs.EVAL_PROMPTS[vs.EVAL_IDS.index(i) % len(vs.EVAL_PROMPTS)]
              if i in vs.EVAL_IDS else vs.EVAL_PROMPTS[0]) for i in ids]
    sd = lp.study.study_dir(assets)
    toks = vc.prompt_tokens(items, assets, os.path.join(sd, "sched_check", "prompt_ids.json"))
    pix = vc.pixels(ids, os.path.join(sd, "sched_check"))
    image_token = json.loads((Path(assets) / "config.json").read_text())["image_token_id"]
    p = BUILD / "vlm_prompts.json"
    p.write_text(json.dumps({"image_token": image_token,
                             "prompts": [{"name": f"COCO {i}", "ids": [int(t) for t in toks[i]]} for i in ids]}))
    im = BUILD / "vlm_images.bin"
    side = pix[ids[0]].shape[0]
    im.write_bytes(np.array([len(ids), side], "<i4").tobytes()
                   + b"".join(np.ascontiguousarray(pix[i], np.uint8).tobytes() for i in ids))
    return p, im


def piper_models(a) -> dict:
    """The voice's own ONNX — how Piper itself synthesizes.  fp32 only: the graph is
    convolutions, and onnxruntime's CPU ConvInteger takes no int8 weights."""
    if a.engines != "fp32":
        sys.exit("piper: fp32 only (its convolutions have no int8 kernel in onnxruntime's CPU provider; "
                 "Piper ships and runs fp32)")
    pin = PINS["piper"]
    voice = REPO / "demo" / "tts" / "assets" / "piper-lessac-medium" / "en_US-lessac-medium.onnx"
    url = (f"https://huggingface.co/rhasspy/piper-voices/resolve/{pin['revision']}/en/en_US/lessac/medium/"
           "en_US-lessac-medium.onnx")
    src = pinned(pin, [voice], url, BUILD / "en_US-lessac-medium.onnx")
    log(f"  fp32: {src}")
    return {"fp32": src}


def piper_inputs(a) -> Path:
    """The sentences' phoneme ids (texts.json, as tts_board.py's) and the voice's scales."""
    d = REPO / "demo" / "tts" / "assets" / "piper-lessac-medium"
    texts = json.loads((d / "texts.json").read_text())
    voice = json.loads((d / "en_US-lessac-medium.onnx.json").read_text())
    inf = voice["inference"]
    p = BUILD / "piper_utts.json"
    p.write_text(json.dumps({"scales": [inf["noise_scale"], inf["length_scale"], inf["noise_w"]],
                             "sample_rate": voice["audio"]["sample_rate"],
                             "utts": [{"name": n, "ids": texts[n]["ids"]} for n in a.utts.split(",")]}))
    return p


def moonshine_models(a) -> dict:
    """{engine: {encoder, decoder, decoder_with_past}}: the export's plain graphs (the merged
    decoder hides its MatMuls in If branches, which the quantizer does not enter)."""
    if a.model != "base":
        sys.exit("moonshine: base only — the public tiny export (onnx-community/moonshine-tiny-ONNX) holds "
                 "another checkpoint than the one asr_study_inputs.json pins (its weights differ by up to 0.05)")
    pin = PINS["moonshine-base"]
    want = json.loads((REPO / "demo" / "asr" / "scripts" / "asr_study_inputs.json").read_text())
    want = want["models"]["moonshine-base"]["revision"]
    if want != pin["checkpoint"]:
        sys.exit(f"asr_study_inputs.json pins moonshine-base at {want}, the export's weights are {pin['checkpoint']}'s")
    src = {k: pinned(p, [], f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/onnx/{k}.onnx",
                     BUILD / f"moonshine-base-{k}.onnx") for k, p in pin["parts"].items()}
    fp32 = {"encoder": src["encoder_model"], "decoder": src["decoder_model"],
            "decoder_with_past": src["decoder_with_past_model"]}
    models = {"fp32": fp32,
              "int8": {k: derived(v, v.with_name(v.stem + "-int8.onnx"), "int8") for k, v in fp32.items()}}
    for e, parts in models.items():
        log(f"  {e}: " + ", ".join(f"{k} {v.name}" for k, v in parts.items()))
    return models


def moonshine_inputs(a) -> tuple:
    """(utts.bin, parts.json): asr_board.py's utterances, each split as the FPGA
    library splits it (moonshine_q.chunks), and the greedy limits."""
    sys.path.insert(0, str(REPO / "demo" / "asr" / "scripts"))
    import asr_board
    import moonshine_np as mn
    import moonshine_q as mq
    utts = asr_board.select(a.utts)
    cfg = json.loads((REPO / "demo" / "asr" / "assets" / "moonshine-base" / "config.json").read_text())
    u = BUILD / "moonshine_utts.bin"
    with open(u, "wb") as f:
        f.write(np.array([len(utts)], "<i4").tobytes())
        for _, _, pcm in utts:
            f.write(np.array([len(pcm)], "<i4").tobytes() + np.asarray(pcm, "<i2").tobytes())
    p = BUILD / "moonshine_parts.json"
    p.write_text(json.dumps({"start": cfg["decoder_start_token_id"], "eos": cfg["eos_token_id"],
                             "max_positions": cfg["max_position_embeddings"], "max_tokens": mn.max_tokens(10 ** 12),
                             "tokens_per_second": mn.TOKENS_PER_SECOND, "sample_rate": mn.SR,
                             "utts": [{"uid": uid, "parts": [[int(x), int(n)] for x, n in mq.chunks(pcm)]}
                                      for uid, _, pcm in utts]}))
    log(f"  {len(utts)} utterances, {sum(len(x) for _, _, x in utts) / 16000:.0f} s of audio")
    return u, p


# ── board ────────────────────────────────────────────────────────────────────

def progress(line: str) -> None:
    if "device_discovery" not in line:      # onnxruntime: "GPU device discovery failed" (the board has none)
        log("    " + line)


def chat_server_running() -> bool:
    return subprocess.run([sys.executable, str(CHAT_DEPLOY), "--status"], capture_output=True).returncode == 0


def on_board(cfg: dict, a, uploads: dict, runs: list) -> list:
    """uploads {remote name: local file}; runs [{"tag", "args"}] -> each run's
    "RESULT:" summary, its output downloaded to build/results/<tag>.<ext>."""
    from deploy_and_run import _stream_exec
    stopped = False
    if chat_server_running():
        if not a.stop_server:
            sys.exit("the chat server is running (it owns the FPGA and most of the memory): stop it "
                     "(demo/chat/deploy.py --stop) or pass --stop-server")
        subprocess.run([sys.executable, str(CHAT_DEPLOY), "--stop"], check=True)
        stopped = True
    user = cfg["ssh"].get("user", "root")
    sudo = "" if user == "root" else "sudo -n "
    rd = a.remote_dir or ("/root/ort_bench" if user == "root" else f"/home/{user}/ort_bench")
    out_dir = BUILD / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        hold_board_lock(cfg, a.board_lock)
        s = RemoteSession(cfg["ssh"])
        s.connect()
        try:
            s.exec_checked(f"mkdir -p {shlex.quote(rd)}")
            probe = f"import onnxruntime, sys; sys.exit(onnxruntime.__version__ != {ORT!r})"
            if s.exec(f"{rd}/venv/bin/python -c {shlex.quote(probe)}")[2]:
                log(f"board venv: {rd}/venv (onnxruntime {ORT}, numpy)")
                _, err, rc = s.exec(f"rm -rf {rd}/venv && python3 -m venv {rd}/venv && "
                                    f"{rd}/venv/bin/pip install -q onnxruntime=={ORT} numpy", timeout=1800)
                if rc:
                    sys.exit(f"pip on the board failed (it needs PyPI):\n{err[-2000:]}")
            sftp = s._client.open_sftp()                                    # noqa: SLF001
            for name, local in uploads.items():
                remote = f"{rd}/{name}"
                have = s.exec(f"sha256sum {shlex.quote(remote)} 2>/dev/null")[0].split(" ")[0]
                if have != sha256(local):
                    t0 = time.monotonic()
                    sftp.put(str(local), remote)
                    log(f"  upload {name}: {local.stat().st_size / 1e6:.0f} MB ({time.monotonic() - t0:.0f} s)")
            scope = s.exec("command -v systemd-run")[2] == 0
            for r in runs:
                s.exec(f"{sudo}sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", timeout=120)
                mem = int(s.exec("awk '/MemAvailable/ {print $2}' /proc/meminfo")[0] or 0) // 1024
                cap = max(256, mem - 300)
                pre = f"{sudo}systemd-run --scope -q -p MemoryMax={cap}M " if scope else sudo
                log(f"\n{r['tag']} (MemAvailable {mem} MB{f', capped at {cap} MB' if scope else ''})")
                out, err, rc = _stream_exec(s, f"cd {shlex.quote(rd)} && {pre}venv/bin/python board_bench.py "
                                               + " ".join(shlex.quote(x) for x in r["args"]),
                                            timeout=4 * 3600, on_stderr_line=progress)
                if rc:
                    sys.exit(f"{r['tag']}: rc {rc}" + (f" — out of memory (capped at {cap} MB): stop the "
                                                       "chat server and other services" if rc in (137, -9) else "")
                             + f"\n{err[-2000:]}")
                line = next((ln for ln in out.splitlines() if ln.startswith("RESULT: ")), None)
                if line is None:
                    sys.exit(f"{r['tag']}: no RESULT line\n{out[-2000:]}{err[-2000:]}")
                res = json.loads(line[len("RESULT: "):])
                local = out_dir / r["out"]
                sftp.get(f"{rd}/{r['out']}", str(local))
                results.append(dict(res, tag=r["tag"], engine=r["engine"], output=str(local)))
                log(f"  {json.dumps(res)}")
            sftp.close()
        finally:
            if not a.keep:
                s.exec(f"rm -rf {shlex.quote(rd)}", timeout=120)
            s.close()
    finally:
        if stopped:
            subprocess.run([sys.executable, str(CHAT_DEPLOY)], check=False)
    return results


def runs_for(a, kind: str, inputs: list) -> list:
    """(engine, threads) runs: board_bench.py's arguments; bert: the first
    thread count with --n examples, the others with --n-rest."""
    runs = []
    for t_i, t in enumerate(int(x) for x in a.threads.split(",")):
        for e in a.engines.split(","):
            tag = f"{a.model_name}-{e}-t{t}"
            out = f"{tag}.{'npz' if kind == 'bert' else 'json'}"
            parts = {"vlm": ("vision", "embed", "decoder"), "asr": ("encoder", "decoder", "decoder_with_past")}
            models = [f"{e}-{p}.onnx" for p in parts[kind]] if kind in parts else [f"{e}.onnx"]
            args = [kind, *models, *inputs, out, "--threads", str(t)]
            if kind == "bert":
                args += ["--n", str(a.n if t_i == 0 else a.n_rest)]
            elif kind in ("tts", "asr"):
                args += ["-R", str(a.reps)]
            else:
                args += ["-k", str(a.k), "-P", a.prefill_lens, "-R", str(a.reps)]
            runs.append({"tag": tag, "engine": e, "threads": t, "out": out, "args": args})
    return runs


# ── reports ──────────────────────────────────────────────────────────────────

def bert_report(a, results: list) -> dict:
    from _common import import_study, load_config, load_features
    from deploy_and_run import decode_all, score
    bcfg = load_config(a.config)
    feats, bs = load_features(bcfg), import_study(bcfg)
    ref = None
    rows = []
    for r in results:
        lg = np.load(r["output"])["logits"].astype(np.float64)
        n = len(lg)
        preds = decode_all(bs, dict(feats, examples=feats["examples"][:n]), lg)
        if ref is None and r["engine"] == "fp32":
            ref = preds
        sc = score(preds, ref[:n] if ref else None, bs)
        rows.append(dict(r, em=sc["em"], f1=sc["f1"], same_span=sc.get("same_span")))
    fpga = None
    if a.fpga and Path(a.fpga).exists():
        f = json.loads(Path(a.fpga).read_text())
        m, acc = f["metrics"], f.get("accuracy", {}).get("board", {})
        fpga = {"source": a.fpga, "n": m["examples"], "mean_ms": m["mean_ms"], "min_ms": m["min_ms"],
                "max_ms": m["max_ms"], "em": acc.get("em"), "f1": acc.get("f1"), "same_span_float": acc.get("same_span")}
    log("\nBERT-SQuAD (bertsquad-12, sequence 256)")
    log(f"  {'engine':24s} {'threads':>7s} {'n':>4s}  {'mean ms (min-max)':>26s} {'FPGA speed-up':>13s}"
        f"  {'EM / F1':>11s}  same span as fp32")
    if fpga:
        acc = f"{fpga['em']:5.1f} / {fpga['f1']:4.1f}" if fpga["em"] is not None else f"{'-':>11s}"
        same = f"{fpga['same_span_float']} / {fpga['n']} (vs float)" if fpga["same_span_float"] is not None else "-"
        log(f"  {'FPGA (' + Path(a.fpga).name + ')':24s} {'-':>7s} {fpga['n']:4d}  "
            f"{fpga['mean_ms']:9.1f} ({fpga['min_ms']:.1f}-{fpga['max_ms']:.1f}) {'-':>13s}  {acc}  {same}")
    for r in rows:
        sp = f"{r['mean_ms'] / fpga['mean_ms']:.1f}x" if fpga else "-"
        same = f"{r['same_span']} / {r['n']}" if r["same_span"] is not None else "-"
        log(f"  {'onnxruntime ' + r['engine']:24s} {r['threads']:7d} {r['n']:4d}  "
            f"{r['mean_ms']:9.1f} ({r['min_ms']:.1f}-{r['max_ms']:.1f}) {sp:>13s}  "
            f"{r['em']:5.1f} / {r['f1']:4.1f}  {same}")
    return {"model": "bertsquad-12", "fpga": fpga, "onnxruntime": rows}


def first_diff(a: list, b: list):
    """the index of the first differing token of the common length, None when all are equal."""
    n = min(len(a), len(b))
    return next((i for i, (x, y) in enumerate(zip(a[:n], b[:n], strict=True)) if x != y), None)


def llm_report(a, results: list) -> dict:
    fpga = None
    if a.fpga:
        b = json.loads(Path(a.fpga).read_text())["bench"]
        fpga = {"source": a.fpga, "decode_ms_mean": b["llm_summary"]["decode_ms_mean"],
                "prefill": [{"n": p["n"], "best_ms": p["best_ms"]} for p in b["prefill"]],
                "toks": [[st["tok"] for st in p["steps"]] for p in b["prompts"]]}
        if b.get("images"):
            fpga["image_ms"] = float(np.mean([im["ms"] for im in b["images"]]))
    rows = []
    for r in results:
        d = json.loads(Path(r["output"]).read_text())
        rows.append(dict(r, toks=[p["toks"] for p in d["prompts"]], prompt_names=[p["name"] for p in d["prompts"]]))
    ref = next((r for r in rows if r["engine"] == "fp32"), None)
    vision = bool(rows[0].get("images"))
    for r in rows:
        if vision:
            r["image_ms"] = float(np.mean([im["ms"] for im in r["images"]]))
    log(f"\n{a.model_name}: decode ms / token (mean of {rows[0]['decode_steps']} steps), "
        f"prefill best of {a.reps}" + (", llm_image the mean of the images" if vision else ""))
    lens = [p["n"] for p in rows[0]["prefill"]]
    log(f"  {'engine':24s} {'threads':>7s}" + (f" {'image ms':>9s}" if vision else "")
        + f" {'decode ms (tok/s)':>18s}  " + "  ".join(f"{'prefill ' + str(n):>12s}" for n in lens)
        + "  FPGA speed-up: " + ("image, " if vision else "") + "decode, prefill")
    if fpga:
        log(f"  {'FPGA (' + Path(a.fpga).name + ')':24s} {'-':>7s}"
            + (f" {fpga.get('image_ms', float('nan')):9.0f}" if vision else "")
            + f" {fpga['decode_ms_mean']:9.2f} ({1e3 / fpga['decode_ms_mean']:5.2f})  "
            + "  ".join(f"{p['best_ms']:9.0f} ms" for p in fpga["prefill"]))
    for r in rows:
        sp = "-"
        if fpga:
            fp = {q["n"]: q["best_ms"] for q in fpga["prefill"]}
            sp = ((f"{r['image_ms'] / fpga['image_ms']:.1f}x, " if vision and fpga.get("image_ms") else "")
                  + f"{r['decode_ms_mean'] / fpga['decode_ms_mean']:.1f}x, "
                  + " / ".join(f"{p['best_ms'] / fp[p['n']]:.1f}x" for p in r["prefill"] if p["n"] in fp))
        log(f"  {'onnxruntime ' + r['engine']:24s} {r['threads']:7d}"
            + (f" {r['image_ms']:9.0f}" if vision else "")
            + f" {r['decode_ms_mean']:9.2f} ({1e3 / r['decode_ms_mean']:5.2f})  "
            + "  ".join(f"{p['best_ms']:9.0f} ms" for p in r["prefill"]) + f"  {sp}")
    agree = {}
    if ref:
        log(f"  greedy tokens equal to onnxruntime fp32's before the first difference (of {len(ref['toks'][0])}):")
        others = ([("FPGA", fpga["toks"])] if fpga else []) + [
            (f"onnxruntime {r['engine']} t{r['threads']}", r["toks"]) for r in rows if r is not ref]
        for label, toks in others:
            if len(toks) != len(ref["toks"]):
                sys.exit(f"{label}: {len(toks)} prompts, onnxruntime's {len(ref['toks'])}: run both with "
                         "the same --prompts / --second-turn / --images")
            counts = [first_diff(t, f) for t, f in zip(toks, ref["toks"], strict=True)]
            counts = [min(len(t), len(f)) if c is None else c
                      for c, t, f in zip(counts, toks, ref["toks"], strict=True)]
            agree[label] = dict(zip(ref["prompt_names"], counts, strict=True))
            log(f"    {label:24s} " + ", ".join(f"{n} {c}" for n, c in agree[label].items()))
    return {"model": a.model_name, "fpga": {k: v for k, v in (fpga or {}).items() if k != "toks"} or None,
            "onnxruntime": [{k: v for k, v in r.items() if k != "toks"} for r in rows], "greedy_agreement": agree}


def piper_report(a, results: list) -> dict:
    rows = [dict(r, utts=json.loads(Path(r["output"]).read_text())["utts"]) for r in results]
    fpga = {}
    if a.fpga:
        sys.path.insert(0, str(REPO / "demo" / "tts" / "scripts"))
        import tts_board
        f = json.loads(Path(a.fpga).read_text())
        b = f["bench"]
        order = list(tts_board.encoder_cases(str(REPO / "demo" / "tts" / "assets" / "piper-lessac-medium")))
        for name, u in zip(f["utts"], b["utts"], strict=True):
            i = order.index(name) if name in order else None
            enc = b["encs"][i]["best_ms"] if i is not None else 0.0
            dur = b.get("durs", [{}] * len(order))[i].get("best_ms", 0.0) if i is not None else 0.0
            fpga[name] = {"encode_ms": enc, "duration_ms": dur, "chunks_ms": u["best_ms"], "audio_s": u["audio_s"],
                          "total_ms": enc + dur + u["best_ms"], "rtf": (enc + dur + u["best_ms"]) / 1e3 / u["audio_s"]}
    log("\npiper-lessac-medium: per sentence (best of the runs), the real-time factor = compute s / audio s")
    log(f"  {'sentence':10s} {'engine':22s} {'threads':>7s} {'ms':>8s} {'audio s':>8s} {'RTF':>7s}  FPGA speed-up")
    names = [u["name"] for u in rows[0]["utts"]]
    for n in names:
        if n in fpga:
            g = fpga[n]
            log(f"  {n:10s} {'FPGA':22s} {'-':>7s} {g['total_ms']:8.0f} {g['audio_s']:8.2f} {g['rtf']:7.3f}  "
                f"(encoder {g['encode_ms']:.0f} + duration {g['duration_ms']:.1f} + chunks {g['chunks_ms']:.0f} ms)")
        for r in rows:
            u = next(x for x in r["utts"] if x["name"] == n)
            sp = f"{u['rtf'] / fpga[n]['rtf']:.1f}x" if n in fpga else "-"
            log(f"  {n:10s} {'onnxruntime ' + r['engine']:22s} {r['threads']:7d} {u['best_ms']:8.0f} {u['audio_s']:8.2f} "
                f"{u['rtf']:7.3f}  {sp}")
    return {"model": "piper-lessac-medium", "fpga": fpga or None, "onnxruntime": rows}


def wer(ids_json: Path) -> dict:
    """asr_study.py score on {"ids": {uid: ids}}: the WER and the transcripts equal to the float model's."""
    import re
    out = subprocess.run([str(REPO / ".venv-export" / "bin" / "python"), str(REPO / "demo" / "asr" / "scripts" /
                          "asr_study.py"), "score", "--model", "moonshine-base", "--ids", str(ids_json)],
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) utterances, WER ([\d.]+) %", out)
    f = re.search(r"study float\s+WER [\d.]+ %, transcripts equal to the board's: (\d+)/(\d+)", out)
    return {"utterances": int(m[1]) if m else None, "wer": float(m[2]) if m else None,
            "equal_to_float": int(f[1]) if f else None}


def moonshine_report(a, results: list) -> dict:
    import statistics
    rows = []
    for r in results:
        d = json.loads(Path(r["output"]).read_text())
        ids = {u["uid"]: u["ids"] for u in d["utts"]}
        p = Path(r["output"]).with_suffix(".ids.json")
        p.write_text(json.dumps({"ids": ids}))
        steps = [u["decode_ms"] / u["steps"] for u in d["utts"] if u["steps"]]
        rows.append(dict(r, summary=d["summary"], ids=ids, score=wer(p),
                         step_ms=statistics.median(steps) if steps else None))
    fpga = None
    if a.fpga:
        f = json.loads(Path(a.fpga).read_text())
        sm = f["bench"]["summary"]
        steps = [u["decode_ms"] / u["steps"] for u in f["bench"]["utts"] if u.get("steps")]
        fpga = {"rtf": sm["rtf"], "total_ms": sm["total_ms"], "audio_s": sm.get("audio_s"),
                "step_ms": statistics.median(steps) if steps else None, "ids": f["ids"], "score": wer(Path(a.fpga))}
    ref = next((r for r in rows if r["engine"] == "fp32"), None)
    n = len(rows[0]["ids"])
    log(f"\nmoonshine-base: {n} utterances (best of {a.reps} runs each); RTF = compute s / audio s")
    log(f"  {'engine':22s} {'threads':>7s} {'total s':>8s} {'RTF':>7s} {'decode step':>11s} {'WER':>7s} "
        f"{'= float':>8s} {'= fp32 ids':>10s}  FPGA speed-up")
    def line(label, thr, total_ms, rtf, step, score, ids, sp):
        same = f"{sum(ids[u] == ref['ids'][u] for u in ref['ids'])}/{n}" if ref else "-"
        log(f"  {label:22s} {thr:>7s} {total_ms / 1e3:8.1f} {rtf:7.3f} {step:9.1f}ms {score['wer']:6.2f}% "
            f"{score['equal_to_float']:>4}/{n:<3} {same:>10s}  {sp}")
    if fpga:
        line("FPGA", "-", fpga["total_ms"], fpga["rtf"], fpga["step_ms"], fpga["score"], fpga["ids"], "-")
    for r in rows:
        line(f"onnxruntime {r['engine']}", str(r["threads"]), r["summary"]["total_ms"], r["summary"]["rtf"], r["step_ms"],
             r["score"], r["ids"], f"{r['summary']['rtf'] / fpga['rtf']:.1f}x" if fpga else "-")
    log("  (the FPGA's encoder entry also writes the cross-attention keys / values; here the decoder's first step does)")
    return {"model": "moonshine-base", "fpga": {k: v for k, v in (fpga or {}).items() if k != "ids"} or None,
            "onnxruntime": [{k: v for k, v in r.items() if k != "ids"} for r in rows]}


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("bert", help="BERT-SQuAD (bertsquad-12) on the BERT demo's inputs")
    pb.add_argument("--fpga", default=str(BERT_FPGA),
                    help="the FPGA's results.json (run_demo.py; default demo/bert_squad/build/results.json)")
    pb.add_argument("--threads", default="4,1", help="intra-op thread counts (default 4,1)")
    pb.add_argument("--n", type=int, default=50, help="examples at the first thread count (default 50)")
    pb.add_argument("--n-rest", type=int, default=5, help="examples at the other thread counts (default 5)")
    pl = sub.add_parser("smollm2", help="SmolLM2-135M / 360M with llm_board.py's protocol")
    pl.add_argument("--model", choices=("135m", "360m"), required=True)
    pl.add_argument("--fpga", default=None, help="the FPGA's results: an llm_board.py --out JSON")
    pl.add_argument("--threads", default="4", help="intra-op thread counts (default 4)")
    pl.add_argument("--prompts", default="factual,summarise,multi-turn")
    pl.add_argument("--second-turn", default="factual")
    pv = sub.add_parser("smolvlm", help="SmolVLM-256M with llm_board.py's protocol for images")
    pv.add_argument("--fpga", default=None, help="the FPGA's results: an llm_board.py --out JSON")
    pv.add_argument("--threads", default="4", help="intra-op thread counts (default 4)")
    pv.add_argument("--images", default="39769,1268", help="COCO ids, one image prompt each (llm_board.py's default)")
    pp = sub.add_parser("piper", help="the Piper voice (TTS) on tts_board.py's sentences")
    pp.add_argument("--fpga", default=None, help="the FPGA's results: a tts_board.py --out JSON")
    pp.add_argument("--threads", default="4", help="intra-op thread counts (default 4)")
    pp.add_argument("--utts", default="eval00,eval10", help="texts.json sentences (tts_board.py's default)")
    pp.add_argument("--reps", type=int, default=3, help="runs per sentence, best taken (default 3)")
    pp.add_argument("--engines", default="fp32", help="fp32 (the only one: see piper_models)")
    pm = sub.add_parser("moonshine", help="Moonshine (ASR) on asr_board.py's LibriSpeech utterances")
    pm.add_argument("--model", choices=("base", "tiny"), default="base")
    pm.add_argument("--fpga", default=None, help="the FPGA's results: an asr_board.py --out JSON")
    pm.add_argument("--threads", default="4", help="intra-op thread counts (default 4)")
    pm.add_argument("--utts", default="all", help="all (200), auto, N or utterance ids (asr_board.py's --utts)")
    pm.add_argument("--reps", type=int, default=2, help="runs per utterance, best taken (default 2, as asr_board.py)")
    for p in (pl, pv):
        p.add_argument("-k", type=int, default=32, help="greedy decode steps per prompt (default 32)")
        p.add_argument("--prefill-lens", default="16,64,256")
        p.add_argument("--reps", type=int, default=3, help="prefill repetitions, best taken (default 3)")
    for p in (pb, pl, pv, pm):
        p.add_argument("--engines", default="fp32,int8")
    for p in (pb, pl, pv, pp, pm):
        p.add_argument("--config", default=str(BERT_CONFIG),
                       help="the board (ssh, board_lock): the BERT demo's config, as llm_board.py")
        p.add_argument("--stop-server", action="store_true", help="stop the chat server for the run, restart it after")
        p.add_argument("--keep", action="store_true", help="keep --remote-dir (the venv, the models) on the board")
        p.add_argument("--remote-dir", default=None, help="default /root/ort_bench (~/ort_bench for another user)")
        p.add_argument("--prepare-only", action="store_true", help="prepare the models on the host, no board")
        p.add_argument("--out", default=None, help="the report JSON (default build/<model>_results.json)")
        p.add_argument("--work", default=None,
                       help="the host's work directory: venv, models, results (default tools/cpu_baseline/build)")
        p.add_argument("--board-lock", default=None)
    a = ap.parse_args(argv)
    global BUILD
    BUILD = Path(a.work).resolve() if a.work else BUILD
    if not Path(a.config).exists():
        sys.exit(f"{a.config} not found: copy {BERT_CONFIG.name}.example next to it and set ssh")
    cfg = json.loads(Path(a.config).read_text())
    BUILD.mkdir(parents=True, exist_ok=True)
    log("models:")
    if a.cmd == "bert":
        a.model_name = "bert"
        models = bert_models()
        from _common import load_config, preprocessed_dir
        inputs = [preprocessed_dir(load_config(a.config)) / "inputs.bin"]
    elif a.cmd == "smollm2":
        a.model_name = llm_name(a.model)
        models = llm_models(a.model)
        inputs = [llm_prompts(a)]
    elif a.cmd == "smolvlm":
        a.model_name = VLM
        models = vlm_models()
        inputs = list(vlm_inputs(a))
    elif a.cmd == "piper":
        a.model_name = "piper-lessac-medium"
        models = piper_models(a)
        inputs = [piper_inputs(a)]
    else:
        a.model_name = f"moonshine-{a.model}"
        models = moonshine_models(a)
        inputs = list(moonshine_inputs(a))
    if a.prepare_only:
        return 0
    kind = {"bert": "bert", "smollm2": "llm", "smolvlm": "vlm", "piper": "tts", "moonshine": "asr"}[a.cmd]
    runs = runs_for(a, kind, [p.name for p in inputs])
    uploads = {"board_bench.py": TOOL / "board_bench.py", **{p.name: p for p in inputs}}
    for e in a.engines.split(","):
        if kind in ("vlm", "asr"):
            uploads.update({f"{e}-{part}.onnx": m for part, m in models[e].items()})
        else:
            uploads[f"{e}.onnx"] = models[e]
    results = on_board(cfg, a, uploads, runs)
    rep = {"bert": bert_report, "piper": piper_report, "moonshine": moonshine_report}.get(a.cmd, llm_report)(a, results)
    rep.update(onnxruntime_version=ORT, date=time.strftime("%Y-%m-%d"))
    out = Path(a.out) if a.out else BUILD / f"{a.model_name}_results.json"
    out.write_text(json.dumps(rep, indent=1))
    log(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
