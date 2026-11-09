#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vjepa2_study.py — model study of V-JEPA 2 ViT-L with its Something-Something v2
classifier (facebook/vjepa2-vitl-fpc16-256-ssv2) for the KV260 (model-study skill,
route C; doc/plans/VJEPA2_PLAN.md).  Host only.

  fetch     the checkpoint and MTEB's Something-Something v2 subset (5444 labelled
            validation videos in one parquet file) at their pinned revisions, every
            file checked against vjepa2_study_inputs.json; the selected videos (the
            calibration and evaluation samples, chosen once with --record) decoded
            (ffmpeg, rgb24) and made into clips (vjepa2_q.make_clip: 16 frames, Meta's
            one-segment sampling, resize + center crop) in samples/<index>.npz, the
            clips' SHA-256 checked; --record writes the hashes and the selection
  validate  vjepa2_q's float64 reference against transformers' VJEPA2ForVideoClassification
            (float32) on the same pixels: the encoder output, the pooled row, the logits;
            then vjepa2_q.make_clip's pixels against the checkpoint's video processor
  study     every policy on the evaluation clips against the float64 reference: top-1
            agreement, top-5 overlap, KL of the class distributions, the logits' /
            encoder output's / pooled row's relative error, accuracy against the labels;
            value ranges, saturation, accumulators -> study/results.json
  ablate    the planned policy with one group of tensor classes quantised (the others exact)
  costs     MACs and int16 weight bytes per part, the projected board time
  formats   the calibrated exponents of a policy (the 16 calibration clips; default the planned
            one) -> demo/video_classification/vjepa2_vitl_ssv2_<policy>_formats.json, the
            frontend's (inference-scheduler/src/vjepa2.py) — byte-identical on a rerun

usage (.venv-export with pyarrow: demo/video_classification/scripts/requirements-study.txt):
  PY=.venv-export/bin/python
  $PY demo/video_classification/scripts/vjepa2_study.py fetch [--record]
  $PY demo/video_classification/scripts/vjepa2_study.py validate
  $PY demo/video_classification/scripts/vjepa2_study.py study [--policies a,b] [--quick | -n N]
  $PY demo/video_classification/scripts/vjepa2_study.py formats [--policy P]

Assets (not in git, demo/video_classification/assets -> /mnt/data/vjepa2_assets):
vjepa2-vitl-fpc16-256-ssv2/, datasets/ssv2_mteb/, videos/, samples/, study/.
Deterministic: fixed data, no sampling.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

if any(c in sys.argv for c in ("study", "ablate", "formats")):     # one BLAS thread per worker process
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vjepa2_q as vq   # noqa: E402

ls = vq.ls
INPUTS_PATH = os.path.join(HERE, "vjepa2_study_inputs.json")
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))
PLANNED = "pow2+p12+vgelu+vsmx+rowops"
N_CAL, N_EVAL = 16, 200              # the selection; the study evaluates the first -n of them


def inputs():
    return json.load(open(INPUTS_PATH))


def ckpt_dir(assets):
    return os.path.join(assets, "vjepa2-vitl-fpc16-256-ssv2")


def parquet_path(assets):
    return os.path.join(assets, "datasets", "ssv2_mteb", "data", "test-00000-of-00001.parquet")


def study_dir(assets):
    return os.path.join(assets, "study")


def sample_path(assets, idx):
    return os.path.join(assets, "samples", f"{idx:04d}.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(path, url, want):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or (want and sha256(path) != want):
        urllib.request.urlretrieve(url, path)
    got = sha256(path)
    return got == want if want else got


# ------------------------------------------------------------------ fetch
def decode_video(path):
    """Every frame of a video as uint8 [n, h, w, 3] (ffmpeg, rgb24, no frame-rate conversion)."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "json", path], capture_output=True, check=True)
    st = json.loads(pr.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-fps_mode", "passthrough", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "pipe:"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8)
    assert a.size % (h * w * 3) == 0, (path, a.size, h, w)
    return a.reshape(-1, h, w, 3)


def select(n_rows):
    """Evaluation: N_EVAL rows evenly over the subset; calibration: N_CAL other rows."""
    ev = [int(i) for i in np.linspace(0, n_rows - 1, N_EVAL).round()]
    cal = [i for i in (int(j) for j in np.linspace(13, n_rows - 14, N_CAL + 4).round()) if i not in set(ev)][:N_CAL]
    return {"calibration": cal, "evaluation": ev}


def cmd_fetch(args):
    import pyarrow.parquet as pq
    inp = inputs()
    bad = 0
    m = inp["model"]
    for f, h in m["files"].items():
        want = None if args.record else h
        r = _download(os.path.join(ckpt_dir(args.assets), f),
                      f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f}", want)
        if args.record:
            m["files"][f], r = r, True
        bad += not r
        print(f"{m['repo']}/{f:32s} {'ok' if r else 'SHA-256 MISMATCH'}")
    d = inp["dataset"]
    for f, h in d["files"].items():
        r = _download(os.path.join(args.assets, "datasets", "ssv2_mteb", f),
                      f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['revision']}/{f}", h)
        bad += not r
        print(f"{d['repo']}/{f:32s} {'ok' if r else 'SHA-256 MISMATCH'}")
    if bad:
        sys.exit(f"{bad} file(s) differ from {INPUTS_PATH}")
    cfg = vq.VCfg(ckpt_dir(args.assets))
    pf = pq.ParquetFile(parquet_path(args.assets))
    n_rows = pf.metadata.num_rows
    # the dataset's class ids are the checkpoint's (the names without its "[...]" brackets)
    names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    if [n.lower() for n in names] != [n.replace("[", "").replace("]", "").lower() for n in cfg.labels]:
        sys.exit("the dataset's classes differ from the checkpoint's id2label")
    if args.record or "samples" not in inp:
        inp["samples"] = select(n_rows)
    want = sorted(set(inp["samples"]["calibration"]) | set(inp["samples"]["evaluation"]))
    clips = inp.setdefault("clips", {})
    tab = pf.read()
    vids, labels = tab.column("video"), tab.column("label")
    vdir = os.path.join(args.assets, "videos")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(os.path.join(args.assets, "samples"), exist_ok=True)
    t0 = time.time()
    for k, i in enumerate(want):
        v = vids[i].as_py()
        data = v["bytes"] if isinstance(v, dict) else v
        vp = os.path.join(vdir, f"{i:04d}.webm")
        with open(vp, "wb") as fo:
            fo.write(data)
        frames = decode_video(vp)
        clip, idx = vq.make_clip(frames, cfg)
        hsh = hashlib.sha256(clip.tobytes()).hexdigest()
        rec = dict(video=hashlib.sha256(data).hexdigest(), clip=hsh, frames=int(len(frames)),
                   size=[int(frames.shape[1]), int(frames.shape[2])])
        if args.record:
            clips[str(i)] = rec
        elif clips.get(str(i), {}).get("clip") != hsh:
            print(f"  row {i}: the clip differs from the pinned SHA-256 (another ffmpeg?)")
            bad += 1
        np.savez(sample_path(args.assets, i), clip=clip, label=int(labels[i].as_py()), frame_idx=idx,
                 frames=np.stack([frames[j] for j in idx]))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(want)} clips ({time.time() - t0:.0f} s)", flush=True)
    print(f"{len(want)} clips -> {os.path.join(args.assets, 'samples')}")
    if args.record:
        json.dump(inp, open(INPUTS_PATH, "w"), indent=1)
        print(f"recorded -> {INPUTS_PATH}")
    if bad:
        sys.exit(f"{bad} clip(s) differ from {INPUTS_PATH}")


# ------------------------------------------------------------------ samples
class Sample:
    def __init__(self, assets, idx, with_frames=False):
        d = np.load(sample_path(assets, idx))
        self.idx = idx
        self.name = f"ssv2_{idx:04d}"
        self.clip = d["clip"]
        self.label = int(d["label"])
        self.frames = d["frames"] if with_frames else None


def load_samples(assets, part, n=None, with_frames=False):
    ids = inputs()["samples"][part]
    return [Sample(assets, i, with_frames) for i in (ids[:n] if n else ids)]


def load_model_inputs(assets):
    cfg = vq.VCfg(ckpt_dir(assets))
    W = ls.load_safetensors(os.path.join(ckpt_dir(assets), "model.safetensors"))
    return cfg, W, vq.kernel_weights(W, cfg)


# ------------------------------------------------------------------ the torch reference
def torch_model(assets):
    import torch
    from transformers import VJEPA2ForVideoClassification
    torch.set_num_threads(os.cpu_count())
    return VJEPA2ForVideoClassification.from_pretrained(ckpt_dir(assets), dtype=torch.float32,
                                                        attn_implementation="eager").eval()


def torch_pixels(clip, cfg):
    """The clip as the model's pixel_values_videos [1, 16, 3, S, S]: x / 255, normalised."""
    import torch
    mean, std = cfg.mean.astype(np.float32), cfg.std.astype(np.float32)
    x = (clip.astype(np.float32) * np.float32(1.0 / 255.0) - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))[None]


def torch_run(model, clip, cfg, px=None):
    import torch
    with torch.no_grad():
        px = torch_pixels(clip, cfg) if px is None else px
        enc = model.vjepa2(pixel_values_videos=px, skip_predictor=True).last_hidden_state
        pooled = model.pooler(enc)
        logits = model.classifier(pooled)
    return dict(xf=enc[0].numpy(), pooled=pooled[0].numpy(), logits=logits[0].numpy())


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cmd_validate(args):
    cfg, W, KW = load_model_inputs(args.assets)
    model = torch_model(args.assets)
    m = vq.VJEPA2(cfg, W, KW=KW)
    samples = load_samples(args.assets, "evaluation", args.n, with_frames=True)
    for s in samples:
        t0 = time.time()
        ref = torch_run(model, s.clip, cfg)
        tr = {}
        lg = m.forward(s.clip, trace=tr)
        print(f"{s.name}: encoder rel {rel(tr['xf'], ref['xf']):.1e}, pooled rel {rel(tr['pooled'], ref['pooled']):.1e}, "
              f"logits rel {rel(lg, ref['logits']):.1e} max|d| {np.abs(lg - ref['logits']).max():.1e}, top-1 "
              f"{'equal' if np.argmax(lg) == np.argmax(ref['logits']) else 'DIFFERS'} "
              f"({cfg.labels[int(np.argmax(lg))]!r}; label {cfg.labels[s.label]!r})  "
              f"numpy {time.time() - t0:.0f} s", flush=True)
    # the clip's pixels against the checkpoint's video processor (the same 16 frames)
    import torch
    from transformers import AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(ckpt_dir(args.assets))
    for s in samples:
        v = torch.from_numpy(np.ascontiguousarray(s.frames.transpose(0, 3, 1, 2)))
        pxt = proc(v, return_tensors="pt")["pixel_values_videos"]                 # [1, 16, 3, S, S]
        back = (pxt[0].numpy().transpose(0, 2, 3, 1) * cfg.std + cfg.mean) * 255.0
        d = back - s.clip.astype(np.float64)
        hf = torch_run(model, None, cfg, px=pxt)
        ours = torch_run(model, s.clip, cfg)
        print(f"{s.name}: pixels vs the video processor max|d| {np.abs(d).max():.2f}, "
              f"{(np.abs(d) > 0.5).mean() * 100:.2f} % differ; logits rel {rel(ours['logits'], hf['logits']):.1e}, "
              f"top-1 {'equal' if np.argmax(ours['logits']) == np.argmax(hf['logits']) else 'DIFFERS'}")


# ------------------------------------------------------------------ study
_G = {}


def _worker(k):
    m, samples, ref = _G["m"], _G["samples"], _G.get("ref")
    s = samples[k]
    m.stats = ls.Stats(m.cfg.NL)
    t0 = time.time()
    tr = {}
    logits = m.forward(s.clip, trace=tr)
    out = dict(logits=np.asarray(logits, np.float64), pooled=np.asarray(tr["pooled"], np.float64),
               stats=m.stats.summary(), wsat=m.wsat, sec=time.time() - t0)
    if ref is None:
        out["xf"] = tr["xf"]
    else:
        out["xf_rel"] = rel(tr["xf"], ref[k]["xf"])
    if m.record_ch:
        out["ch"] = m.stats.ch
    return out


def run_model(m, samples, jobs, ref=None, log=print, tag=""):
    import multiprocessing as mp
    _G.update(m=m, samples=samples, ref=ref)
    ctx = mp.get_context("fork")
    outs, t0 = [], time.time()
    with ctx.Pool(min(jobs, len(samples))) as pool:
        for k, o in enumerate(pool.imap(_worker, range(len(samples)))):
            outs.append(o)
            if (k + 1) % max(1, len(samples) // 8) == 0 or k + 1 == len(samples):
                log(f"    {tag} {k + 1}/{len(samples)} clips, {time.time() - t0:.0f} s "
                    f"({o['sec']:.0f} s a clip in a worker)")
    return outs


def merge_ch(outs):
    ch = {}
    for o in outs:
        for key, v in o["ch"].items():
            ch[key] = v.copy() if key not in ch else np.maximum(ch[key], v)
    return ch


def merge_stats(outs):
    tot = {}
    for o in outs:
        for cls, v in o["stats"].items():
            e = tot.setdefault(cls, dict(max=0.0, sat=0, n=0))
            e["max"] = max(e["max"], v["max"])
            e["sat"] += v["sat"]
            e["n"] += v["n"]
    return tot


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def metrics(outs, ref, samples):
    r = {k: [] for k in ("top1", "top5", "kl", "logits", "xf", "pooled", "acc", "acc5")}
    for o, rf, s in zip(outs, ref, samples):
        a, b = o["logits"], rf["logits"]
        ta, tb = np.argsort(-a), np.argsort(-b)
        r["top1"].append(float(ta[0] == tb[0]))
        r["top5"].append(len(set(ta[:5]) & set(tb[:5])) / 5.0)
        pa, pb = softmax(a), softmax(b)
        r["kl"].append(float(np.sum(pb * (np.log(pb) - np.log(np.maximum(pa, 1e-300))))))
        r["logits"].append(rel(a, b))
        r["xf"].append(o["xf_rel"] if "xf_rel" in o else rel(o["xf"], rf["xf"]))
        r["pooled"].append(rel(o["pooled"], rf["pooled"]))
        r["acc"].append(float(ta[0] == s.label))
        r["acc5"].append(float(s.label in ta[:5]))
    return {k: dict(mean=float(np.mean(v)), max=float(np.max(v))) for k, v in r.items()}


def calibrate(cfg, W, KW, assets, jobs, log):
    t0 = time.time()
    cal = load_samples(assets, "calibration")
    m = vq.VJEPA2(cfg, W, record_ch=True, KW=KW)
    ch = merge_ch(run_model(m, cal, jobs, log=log, tag="calibration"))
    log(f"  calibration: {len(cal)} clips, {time.time() - t0:.0f} s")
    return ch


def cmd_study(args):
    out_dir = args.out or study_dir(args.assets)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "study.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    cfg, W, KW = load_model_inputs(args.assets)
    n = 16 if args.quick else args.n
    samples = load_samples(args.assets, "evaluation", n)
    names = args.policies.split(",") if args.policies else [p for p in vq.POLICIES if p != "float"]
    log(f"study {time.strftime('%Y-%m-%d %H:%M:%S')}  {len(samples)} evaluation clips, {args.jobs} workers, "
        f"policies {', '.join(names)}")
    ch = calibrate(cfg, W, KW, args.assets, args.jobs, log)
    fl = vq.VJEPA2(cfg, W, KW=KW)
    ref = run_model(fl, samples, args.jobs, log=log, tag="float")
    del fl
    log(f"  float64 reference {time.time() - t0:.0f} s")
    results = {"float": dict(metrics=metrics(ref, ref, samples), stats=merge_stats(ref))}
    logits = {"float": [o["logits"].tolist() for o in ref]}
    fmts = {}
    for name in names:
        pol = vq.POLICIES[name]
        t1 = time.time()
        fmt = vq.make_formats(pol, ch, KW, cfg)
        fmts[name] = fmt
        m = vq.VJEPA2(cfg, W, pol, fmt, KW=KW)
        outs = run_model(m, samples, args.jobs, ref=ref, log=log, tag=name)
        results[name] = dict(metrics=metrics(outs, ref, samples), stats=merge_stats(outs), wsat=m.wsat)
        del m
        logits[name] = [o["logits"].tolist() for o in outs]
        mt = results[name]["metrics"]
        log(f"  [{name}] top-1 agree {mt['top1']['mean']:.3f}  top-5 overlap {mt['top5']['mean']:.3f}  "
            f"KL {mt['kl']['mean']:.4f} (max {mt['kl']['max']:.3f})  logits rel {mt['logits']['mean']:.4f}  "
            f"encoder rel {mt['xf']['mean']:.4f}  pooled rel {mt['pooled']['mean']:.4f}  acc {mt['acc']['mean']:.3f}  "
            f"{time.time() - t1:.0f} s")
    json.dump(dict(results=results, samples=[s.idx for s in samples], labels=[s.label for s in samples],
                   logits=logits, formats={k: vq.formats_json(f, k, vq.POLICIES[k]) for k, f in fmts.items()}),
              open(os.path.join(out_dir, "results.json"), "w"))
    report(results, log)
    log(f"total {time.time() - t0:.0f} s")


CLASS_DOC = (("pe", "patch embedding"), ("h", "residual"), ("x", "LN1 out"), ("q0", "q (kernel)"),
             ("qb", "q + bias"), ("q", "q after RoPE"), ("s", "scores"), ("pv", "P.V"), ("o", "attention out"),
             ("x2", "LN2 out"), ("f", "fc1 out"), ("a", "GELU out"), ("d", "fc2 out"), ("xf", "encoder out"),
             ("xc", "pooled row"), ("logits", "logits"))


def report(results, log):
    log("\n== policies vs the float64 reference ==")
    log(f"{'policy':30s} {'top-1':>6s} {'top-5':>6s} {'KL':>8s} {'KL max':>7s} {'logits':>7s} {'encoder':>8s} "
        f"{'pooled':>7s} {'acc':>6s} {'acc5':>6s}")
    for n, r in results.items():
        m = r["metrics"]
        log(f"{n:30s} {m['top1']['mean']:6.3f} {m['top5']['mean']:6.3f} {m['kl']['mean']:8.4f} {m['kl']['max']:7.3f} "
            f"{m['logits']['mean']:7.4f} {m['xf']['mean']:8.4f} {m['pooled']['mean']:7.4f} {m['acc']['mean']:6.3f} "
            f"{m['acc5']['mean']:6.3f}")
    st = results["float"]["stats"]
    log("\n== value ranges (float): max |x| per class ==")
    log("  " + "  ".join(f"{c} {st[c]['max']:.1f}" for c, _ in CLASS_DOC if c in st))
    log("\n== saturation (emulated) ==")
    for n, r in results.items():
        if "wsat" not in r:
            continue
        sat = {c: v["sat"] for c, v in r["stats"].items() if v["sat"] and not c.startswith("acc:")
               and (c != "h" or n == "q88")}
        acc = max((v["max"] for c, v in r["stats"].items() if c.startswith("acc:")), default=0.0)
        wrapped = sum(v["sat"] for c, v in r["stats"].items() if c.startswith("acc:"))
        log(f"  [{n}] weights saturated {r['wsat']}; saturated {sat or 0}; wrapped acc {wrapped}; max |acc| {acc:.1f}")


# ------------------------------------------------------------------ ablation
ABLATE = [("pe",), ("x", "x2"), ("q0", "k0", "v"), ("qb", "kb"), ("q", "k"), ("s",), ("p",), ("vc",), ("pv",),
          ("o",), ("f",), ("a",), ("d",), ("h",), ("xf", "xc", "logits")]


def cmd_ablate(args):
    cfg, W, KW = load_model_inputs(args.assets)
    samples = load_samples(args.assets, "evaluation", args.n)
    ch = calibrate(cfg, W, KW, args.assets, args.jobs, print)
    ref = run_model(vq.VJEPA2(cfg, W, KW=KW), samples, args.jobs, tag="float")
    base = vq.POLICIES[args.base]
    fmt = vq.make_formats(base, ch, KW, cfg)
    print(f"ablation of {args.base} on {len(samples)} clips: only these classes quantised (the weights always)")
    for grp in ABLATE:
        outs = run_model(vq.VJEPA2(cfg, W, dict(base, qclasses=set(grp)), fmt, KW=KW), samples, args.jobs,
                         ref=ref, tag="+".join(grp))
        mt = metrics(outs, ref, samples)
        print(f"  {'+'.join(grp):16s} top-1 {mt['top1']['mean']:.3f}  KL {mt['kl']['mean']:.4f}  "
              f"logits rel {mt['logits']['mean']:.4f}  encoder rel {mt['xf']['mean']:.4f}", flush=True)


# ------------------------------------------------------------------ formats
def formats_path(policy):
    return os.path.normpath(os.path.join(HERE, "..", f"vjepa2_vitl_ssv2_{policy}_formats.json"))


def cmd_formats(args):
    cfg, W, KW = load_model_inputs(args.assets)
    ch = calibrate(cfg, W, KW, args.assets, args.jobs, print)
    pol = vq.POLICIES[args.policy]
    fmt = vq.make_formats(pol, ch, KW, cfg)
    d = vq.formats_json(fmt, args.policy, pol)
    d["checkpoint"] = inputs()["model"]
    path = formats_path(args.policy)
    with open(path, "w") as f:
        json.dump(d, f)
    print(f"{path}: {len(fmt)} entries")


# ------------------------------------------------------------------ costs
# board references (bitstream b4e63457057a's design): SmolVLA's vision entry (SigLIP, 12 layers 768
# wide, 1024 tokens, the same partition: query-major scores, the row-mode softmax, two row pieces)
# 1313 ms an image (doc/plans/VLA_PLAN.md §4); the softmax unit's row mode 1.81 ms for 1024 x 1024
REF_MS, REF_SMX_MS = 1313.0, 1.81
REF = dict(L=12, D=768, F=3072, N=1024, P=16, H=12)


def vit_macs(L, D, F, N, patch_k):
    return N * patch_k * D + L * (4 * N * D * D + 2 * N * N * D + 2 * N * D * F)


def cmd_costs(args):
    cfg = vq.VCfg(ckpt_dir(args.assets))
    D, F, N, L = cfg.D, cfg.F, cfg.N, cfg.L
    pk = 3 * cfg.tub * cfg.P * cfg.P
    enc = vit_macs(L, D, F, N, pk)
    pool_self = vit_macs(cfg.n_pool, D, F, N, 0)
    cross = 2 * N * D * D + 2 * N * D + 2 * D * F + D * cfg.C
    tot = enc + pool_self + cross
    w = 2 * (pk * D + (L + cfg.n_pool) * (4 * D * D + 2 * D * F) + 2 * D * D + 2 * D * F + D * cfg.C)
    ref = vit_macs(REF["L"], REF["D"], REF["F"], REF["N"], 3 * REF["P"] ** 2)
    ref_smx = REF["L"] * REF["H"] * REF_SMX_MS
    rate = ref / ((REF_MS - ref_smx) / 1e3)
    smx = (L + cfg.n_pool) * cfg.H * REF_SMX_MS * (N / REF["N"]) ** 2
    lin_s = tot / rate
    print(f"MACs per clip: encoder {enc / 1e9:.0f} GMAC (attention {2 * L * N * N * D / 1e9:.0f}), pooler self layers "
          f"{pool_self / 1e9:.0f}, cross layer + classifier {cross / 1e9:.1f} = {tot / 1e9:.0f} GMAC")
    print(f"int16 weights {w / 2 ** 20:.0f} MiB; softmax rows: {(L + cfg.n_pool) * cfg.H} heads of {N} x {N}")
    print(f"projection from SmolVLA's vision entry ({ref / 1e9:.0f} GMAC, {REF_MS:.0f} ms, of it the softmax "
          f"{ref_smx:.0f} ms): {rate / 1e9:.0f} GMAC/s for the kernels' MACs -> {lin_s:.1f} s, the softmax unit "
          f"{smx / 1e3:.1f} s ({REF_SMX_MS} ms per 1024^2) -> {lin_s:.1f} s with the softmax beside ConvKernel, "
          f"{lin_s + smx / 1e3:.1f} s after it")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "validate", "study", "ablate", "costs", "formats"])
    ap.add_argument("--assets", default=ASSETS)
    ap.add_argument("--record", action="store_true", help="fetch: record the hashes and the sample selection")
    ap.add_argument("--policies", default=None, help="study: comma-separated (default: every vjepa2_q.POLICIES)")
    ap.add_argument("--base", default=PLANNED, help="ablate: the policy")
    ap.add_argument("--policy", default=PLANNED, help="formats: the policy")
    ap.add_argument("--quick", action="store_true", help="study: 16 evaluation clips")
    ap.add_argument("-n", type=int, default=100, help="validate / study / ablate: evaluation clips")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "validate" and "-n" not in sys.argv:
        args.n = 3
    {"fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study, "ablate": cmd_ablate,
     "costs": cmd_costs, "formats": cmd_formats}[args.cmd](args)


if __name__ == "__main__":
    main()
