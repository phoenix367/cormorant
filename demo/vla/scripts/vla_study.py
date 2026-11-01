#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vla_study.py — model study of SmolVLA (lerobot/smolvla_base) for the KV260
(model-study skill, route C; doc/plans/VLA_PLAN.md).  Host only.

  select    (once) the calibration and evaluation samples of the two SO100
            datasets (episodes, frames) -> vla_study_inputs.json
  fetch     the checkpoint, SmolVLM2-500M's config and tokenizer and the
            datasets' metadata, tables and videos at their pinned revisions,
            every file checked against vla_study_inputs.json; the samples'
            frames decoded (ffmpeg, rgb24; their SHA-256 checked) into
            samples/<dataset>_ep<e>_f<f>.npz with the state, the next 50
            actions and the task; --drop-videos deletes the videos after
  validate  smolvla_q's float64 reference against lerobot's SmolVLAPolicy
            (torch float32): token ids, pixels, the prefix caches, the first
            velocity and the action chunk with the same noise; then lerobot's
            default precision (from_pretrained: the VLM and the expert in bf16)
  study     every policy on the evaluation samples against float with the same
            noise: the action chunk's error (normalised units: mean |d|, its
            first 10 steps, relative L2), the yardsticks (float with other
            noise, bf16, lerobot's bf16 VLM), each one's error against the
            recorded actions; the image features', caches' and first
            velocity's errors; saturation -> study/results.json
  ablate    one part (vision, VLM, expert) quantised at a time
  formats   a policy's calibrated exponents -> demo/vla/smolvla_base_<policy>_formats.json
  costs     MACs and weight bytes per part, the projected board time

usage (the lerobot venv, demo/vla/scripts/requirements-study.txt):
  PY=demo/vla/assets/.venv-lerobot/bin/python
  $PY demo/vla/scripts/vla_study.py fetch
  $PY demo/vla/scripts/vla_study.py validate
  $PY demo/vla/scripts/vla_study.py study [--policies a,b] [--quick]

Assets (not in git, demo/vla/assets -> /mnt/data): smolvla_base/,
smolvlm2-500m-video-instruct/, datasets/<name>/, samples/, study/.
Deterministic: the noise is seeded per sample, fixed data.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

if any(c in sys.argv for c in ("study", "ablate", "formats")):   # one BLAS thread per worker process
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smolvla_q as sq   # noqa: E402

ls, vs = sq.ls, sq.vs
INPUTS_PATH = os.path.join(HERE, "vla_study_inputs.json")
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))
NOISE_SEED = 2026
YARD_SEED = 7                    # the noise yardstick: float with another seed


def inputs():
    return json.load(open(INPUTS_PATH))


def ckpt_dir(assets):
    return os.path.join(assets, "smolvla_base")


def vlm_dir(assets):
    return os.path.join(assets, "smolvlm2-500m-video-instruct")


def study_dir(assets):
    return os.path.join(assets, "study", "smolvla_base")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sample_name(s):
    return f"{s['dataset']}_ep{s['episode']}_f{s['frame']}"


def sample_path(assets, s):
    return os.path.join(assets, "samples", sample_name(s) + ".npz")


# ------------------------------------------------------------------ select / fetch
def episodes(assets, name):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(assets, "datasets", name, "meta", "episodes", "chunk-000",
                                   "file-000.parquet")).to_pydict()
    return t


def cmd_select(args):
    """Evenly spread samples: per dataset, the episodes long enough for a 50-step
    chunk (>= 200 frames); calibration from 4 of them, evaluation from 12 others,
    frames at a cycling fraction of the episode."""
    inp = inputs()
    out = {"calibration": [], "evaluation": []}
    for name in inp["datasets"]:
        ep = episodes(args.assets, name)
        ok = [i for i, n in zip(ep["episode_index"], ep["length"]) if n >= 200]
        length = dict(zip(ep["episode_index"], ep["length"]))
        cal = ok[1::len(ok) // 4][:4]
        rest = [i for i in ok if i not in cal]
        ev = rest[::max(1, len(rest) // 12)][:12]
        for part, eps, fracs in (("calibration", cal, (0.25, 0.6)), ("evaluation", ev, (0.1, 0.35, 0.6))):
            for k, e in enumerate(eps):
                f = int(fracs[k % len(fracs)] * (length[e] - 50))
                out[part].append({"dataset": name, "episode": int(e), "frame": f})
    inp["samples"] = out
    json.dump(inp, open(INPUTS_PATH, "w"), indent=1)
    print(f"{len(out['calibration'])} calibration, {len(out['evaluation'])} evaluation samples -> {INPUTS_PATH}")


def _download(path, url, want):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or (want and sha256(path) != want):
        urllib.request.urlretrieve(url, path)
    got = sha256(path)
    return got == want if want else got


def decode_frames(video, numbers, h, w):
    """The frames numbered `numbers` (ascending) of a video as uint8 [h, w, 3] (ffmpeg, rgb24)."""
    sel = "+".join(f"eq(n\\,{n})" for n in numbers)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select='{sel}'", "-fps_mode", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8)
    assert a.size == len(numbers) * h * w * 3, (video, a.size, len(numbers))
    return a.reshape(len(numbers), h, w, 3)


def cmd_fetch(args):
    import pyarrow.parquet as pq
    inp = inputs()
    bad = 0
    for key, d in (("model", ckpt_dir(args.assets)), ("vlm", vlm_dir(args.assets))):
        m = inp[key]
        for f, h in m["files"].items():
            ok = _download(os.path.join(d, f), f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f}", h)
            bad += not ok
            print(f"{m['repo']}/{f:40s} {'ok' if ok else 'SHA-256 MISMATCH'}")
    record = args.record
    samples = inp.get("samples", {})
    for name, ds in inp["datasets"].items():
        root = os.path.join(args.assets, "datasets", name)
        for f, h in ds["files"].items():
            want = None if record else h
            if f.startswith("videos/") and not any(
                    not os.path.exists(sample_path(args.assets, s)) for p in samples.values() for s in p
                    if s["dataset"] == name):
                continue                                     # every sample of this dataset extracted
            r = _download(os.path.join(root, f),
                          f"https://huggingface.co/datasets/{ds['repo']}/resolve/{ds['revision']}/{f}", want)
            if record:
                ds["files"][f] = r
                r = True
            bad += not r
            print(f"{ds['repo']}/{f:52s} {'ok' if r else 'SHA-256 MISMATCH'}")
        mine = [s for p in samples.values() for s in p if s["dataset"] == name]
        if not mine:
            continue
        info = json.load(open(os.path.join(root, "meta", "info.json")))
        fps = info["fps"]
        ep = episodes(args.assets, name)
        epi = {e: i for i, e in enumerate(ep["episode_index"])}
        data = pq.read_table(os.path.join(root, "data", "chunk-000", "file-000.parquet")).to_pydict()
        stats = json.load(open(os.path.join(root, "meta", "stats.json")))
        need = [s for s in mine if not os.path.exists(sample_path(args.assets, s)) or record]
        frames = {}
        for cam in ds["cameras"]:
            vkey = f"observation.images.{cam}"
            nums = []
            for s in need:
                i = epi[s["episode"]]
                n = round((ep[f"videos/{vkey}/from_timestamp"][i] + s["frame"] / fps) * fps)
                assert n == ep["dataset_from_index"][i] + s["frame"], (name, s, n)
                nums.append(n)
            if not nums:
                continue
            order = sorted(set(nums))
            hh, ww = info["features"][vkey]["shape"][:2]
            dec = decode_frames(os.path.join(root, "videos", vkey, "chunk-000", "file-000.mp4"), order, hh, ww)
            frames[cam] = {n: dec[order.index(n)] for n in nums}
        for s in need:
            i = epi[s["episode"]]
            g = ep["dataset_from_index"][i] + s["frame"]
            assert data["episode_index"][g] == s["episode"] and data["frame_index"][g] == s["frame"]
            acts = np.array(data["action"][g:g + 50], np.float32)
            assert len(acts) == 50 and all(e == s["episode"] for e in data["episode_index"][g:g + 50])
            imgs = {cam: frames[cam][g] for cam in ds["cameras"]}
            hs = {cam: hashlib.sha256(imgs[cam].tobytes()).hexdigest() for cam in imgs}
            if record:
                s["sha256"] = hs
            elif hs != s.get("sha256"):
                print(f"  {sample_name(s)}: decoded frames differ from the pinned SHA-256 (another ffmpeg?)")
                bad += 1
            os.makedirs(os.path.dirname(sample_path(args.assets, s)), exist_ok=True)
            np.savez(sample_path(args.assets, s), state=np.array(data["observation.state"][g], np.float32),
                     actions=acts, task=np.array(ep["tasks"][i][0]),
                     state_mean=np.array(stats["observation.state"]["mean"]), state_std=np.array(stats["observation.state"]["std"]),
                     action_mean=np.array(stats["action"]["mean"]), action_std=np.array(stats["action"]["std"]),
                     **{f"img_{cam}": imgs[cam] for cam in imgs})
        print(f"{name}: {len(need)} samples extracted")
        if args.drop_videos:
            for cam in ds["cameras"]:
                p = os.path.join(root, "videos", f"observation.images.{cam}", "chunk-000", "file-000.mp4")
                if os.path.exists(p):
                    os.remove(p)
    if record:
        json.dump(inp, open(INPUTS_PATH, "w"), indent=1)
        print(f"recorded -> {INPUTS_PATH}")
    if bad:
        sys.exit(f"{bad} file(s) differ from vla_study_inputs.json")


# ------------------------------------------------------------------ samples
class Sample:
    def __init__(self, assets, s, inp):
        d = np.load(sample_path(assets, s))
        cams = inp["datasets"][s["dataset"]]["cameras"]          # dataset camera -> model camera
        self.name = sample_name(s)
        self.frames = [d[f"img_{c}"] for c in sorted(cams, key=lambda c: cams[c])]
        self.task = str(d["task"])
        eps = 1e-8                                                # lerobot's normalizer eps
        self.state = (d["state"] - d["state_mean"]) / (d["state_std"] + eps)
        self.gt = (d["actions"] - d["action_mean"]) / (d["action_std"] + eps)
        self.raw = d


def load_samples(assets, part, n=None):
    inp = inputs()
    ss = inp["samples"][part]
    return [Sample(assets, s, inp) for s in (ss[:n] if n else ss)]


def task_ids(tok, task, cfg):
    """lerobot's tokenizer step: the task + "\\n" (NewLineTaskProcessorStep), truncated
    at tokenizer_max_length; the padding is masked, so only the valid ids."""
    if not task.endswith("\n"):
        task = task + "\n"
    return tok.encode(task)[:cfg.tok_max]


def noise(cfg, idx, seed=NOISE_SEED):
    return np.random.default_rng([seed, idx]).standard_normal((cfg.chunk, cfg.max_action))


# ------------------------------------------------------------------ the torch reference (lerobot)
def torch_policy(assets, default_dtypes=False):
    """lerobot's SmolVLAPolicy with the checkpoint, every parameter float32; with
    default_dtypes the precision of SmolVLAPolicy.from_pretrained: the VLM loaded in
    bf16 (load_vlm_weights), the expert built from the bf16 text config except the
    cross layers' new k / v linears, the projections float32."""
    import torch
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    torch.set_num_threads(os.cpu_count())
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir(assets))
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = vlm_dir(assets)
    cfg.device = "cpu"
    pol = SmolVLAPolicy(cfg).float()
    sd = load_file(os.path.join(ckpt_dir(assets), "model.safetensors"))
    missing, unexpected = pol.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and not missing, (missing[:5], unexpected[:5])
    if default_dtypes:
        ve = pol.model.vlm_with_expert
        ve.vlm.to(torch.bfloat16)
        ve.lm_expert.to(torch.bfloat16)
        for l, layer in enumerate(ve.lm_expert.layers):
            if not (ve.self_attn_every_n_layers > 0 and l % ve.self_attn_every_n_layers == 0):
                layer.self_attn.k_proj.float()
                layer.self_attn.v_proj.float()
    return pol.eval()


def torch_batch(pol, sample, cfg):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pol.config.vlm_model_name)
    t = sample.task if sample.task.endswith("\n") else sample.task + "\n"
    enc = tok(t, max_length=cfg.tok_max, truncation=True, padding="max_length", padding_side="right",
              return_tensors="pt")
    batch = {f"observation.images.{c.split('.')[-1]}": torch.tensor(f).permute(2, 0, 1)[None].float() / 255.0
             for c, f in zip(cfg.cameras, sample.frames)}
    batch = {k.replace("observation.images.observation.images.", "observation.images."): v for k, v in batch.items()}
    batch["observation.state"] = torch.tensor(sample.state, dtype=torch.float32)[None]
    return batch, enc["input_ids"], enc["attention_mask"].bool()


def torch_sample(pol, sample, cfg, nz, trace=False):
    import torch
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    batch, ids, mask = torch_batch(pol, sample, cfg)
    with torch.no_grad():
        images, img_masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        noise_t = torch.tensor(nz, dtype=torch.float32)[None]
        out = {"ids": ids[0][mask[0]].numpy(), "images": [i[0].permute(1, 2, 0).double().numpy() for i in images]}
        if trace:
            m = pol.model
            pe, pm, pa = m.embed_prefix(images, img_masks, ids, mask, state=state)
            a2 = make_att_2d_masks(pm, pa)
            _, kv = m.vlm_with_expert.forward(attention_mask=a2, position_ids=torch.cumsum(pm, 1) - 1,
                                              past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
            valid = pm[0].numpy()
            out["K"] = [kv.layers[l].keys[0].float().numpy()[:, valid] for l in range(len(kv.layers))]
            out["V"] = [kv.layers[l].values[0].float().numpy()[:, valid] for l in range(len(kv.layers))]
            out["v0"] = m.denoise_step(pm, kv, noise_t, torch.tensor([1.0]))[0].float().numpy()
        out["actions"] = pol.model.sample_actions(images, img_masks, ids, mask, state, noise=noise_t)[0].float().numpy()
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cmd_validate(args):
    import torch
    cfg = sq.VLACfg(ckpt_dir(args.assets), vlm_dir(args.assets))
    W = ls.load_safetensors(os.path.join(ckpt_dir(args.assets), "model.safetensors"))
    tok = ls.Tok(vlm_dir(args.assets))
    m = sq.SmolVLA(cfg, W)
    pol = torch_policy(args.assets)
    samples = load_samples(args.assets, "evaluation")[:args.n]
    for k, s in enumerate(samples):
        nz = noise(cfg, k)
        t0 = time.time()
        ref = torch_sample(pol, s, cfg, nz, trace=True)
        ids = task_ids(tok, s.task, cfg)
        tr = {}
        a = m.sample(s.frames, ids, s.state, nz, trace=tr)
        px = [sq.pixels(f, cfg) for f in s.frames]
        dpx = max(float(np.abs((p / 255.0) * 2 - 1 - r).max()) for p, r in zip(px, ref["images"]))
        dk = max(rel(tr["K"][l], ref["K"][l]) for l in range(cfg.text.L))
        dv = max(rel(tr["V"][l], ref["V"][l]) for l in range(cfg.text.L))
        print(f"{s.name}: ids {'equal' if list(ids) == list(ref['ids']) else f'DIFFER {list(ids)} vs {list(ref[chr(105)+chr(100)+chr(115)])}'} "
              f"({len(ids)}), pixels max|d| {dpx:.1e}, caches rel K {dk:.1e} V {dv:.1e}, "
              f"v0 rel {rel(tr['v0'], ref['v0']):.1e}, actions rel {rel(a, ref['actions']):.1e} "
              f"max|d| {np.abs(a - ref['actions']).max():.1e}  ({time.time() - t0:.0f} s)", flush=True)
    pol = torch_policy(args.assets, default_dtypes=True)      # lerobot's default precision
    for k, s in enumerate(samples):
        nz = noise(cfg, k)
        a = m.sample(s.frames, task_ids(tok, s.task, cfg), s.state, nz)
        b = torch_sample(pol, s, cfg, nz)["actions"]
        print(f"{s.name}: lerobot's default precision vs float64: actions rel {rel(b, a):.4f}, "
              f"mean|d| {np.abs(b - a)[:, :cfg.action_dim].mean():.4f}")


# ------------------------------------------------------------------ study
_G = {}


def _worker_run(job):
    """One sample through the global model (forked workers): actions + trace + stats."""
    k, seed = job
    m, cfg, samples, tok, feats = _G["m"], _G["cfg"], _G["samples"], _G["tok"], _G.get("feats")
    s = samples[k]
    for part in (m.vision, m.text, m.expert):
        part.stats = ls.Stats(part.cfg.L)
    tr = {}
    a = m.sample(s.frames, task_ids(tok, s.task, cfg), s.state, noise(cfg, k, seed),
                 feats=None if feats is None else feats[k], trace=tr)
    out = dict(actions=a, feats=tr["feats"], K=[tr["K"][0], tr["K"][-1]], V=[tr["V"][0], tr["V"][-1]], v0=tr["v0"],
               stats={p: getattr(m, p).stats.summary() for p in ("vision", "text", "expert")},
               wsat={p: getattr(m, p).wsat for p in ("vision", "text", "expert")})
    if m.vision.record_ch:
        out["ch"] = {p: getattr(m, p).stats.ch for p in ("vision", "text", "expert")}
    return out


def run_model(m, cfg, samples, tok, jobs, seed=NOISE_SEED, feats=None):
    import multiprocessing as mp
    _G.update(m=m, cfg=cfg, samples=samples, tok=tok, feats=feats)
    ctx = mp.get_context("fork")
    with ctx.Pool(min(jobs, len(samples))) as pool:
        return pool.map(_worker_run, [(k, seed) for k in range(len(samples))])


def merge_ch(outs):
    ch = {}
    for o in outs:
        for p, d in o["ch"].items():
            t = ch.setdefault(p, {})
            for key, v in d.items():
                t[key] = v.copy() if key not in t else np.maximum(t[key], v)
    return ch


def calibrate(cfg, W, KW, tok, assets, jobs, cs, log):
    t0 = time.time()
    cal = load_samples(assets, "calibration")
    m = sq.SmolVLA(cfg, W, record_ch=True, KW=KW, cos_sin=cs)
    ch = merge_ch(run_model(m, cfg, cal, tok, jobs))
    log(f"  calibration: {len(cal)} samples, {time.time() - t0:.0f} s")
    return ch


def metrics(outs, ref, samples, cfg):
    A = cfg.action_dim
    r = dict(mae=[], mae10=[], rel=[], maxd=[], gt_mae=[], feat=[], k_first=[], k_last=[], v0=[])
    for o, rf, s in zip(outs, ref, samples):
        a, b = o["actions"][:, :A], rf["actions"][:, :A]
        r["mae"].append(float(np.abs(a - b).mean()))
        r["mae10"].append(float(np.abs(a[:10] - b[:10]).mean()))
        r["rel"].append(rel(a, b))
        r["maxd"].append(float(np.abs(a - b).max()))
        r["gt_mae"].append(float(np.abs(a - s.gt).mean()))
        r["feat"].append(max(rel(x, y) for x, y in zip(o["feats"], rf["feats"])))
        r["k_first"].append(rel(o["K"][0], rf["K"][0]))
        r["k_last"].append(rel(o["K"][1], rf["K"][1]))
        r["v0"].append(rel(o["v0"][:, :A], rf["v0"][:, :A]))
    return {k: dict(mean=float(np.mean(v)), max=float(np.max(v))) for k, v in r.items()}


def merge_stats(outs):
    tot = {}
    for o in outs:
        for p, st in o["stats"].items():
            t = tot.setdefault(p, {})
            for cls, v in st.items():
                e = t.setdefault(cls, dict(max=0.0, sat=0, n=0))
                e["max"] = max(e["max"], v["max"])
                e["sat"] += v["sat"]
                e["n"] += v["n"]
    return tot


def cmd_study(args):
    out_dir = args.out or study_dir(args.assets)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "study.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    cfg = sq.VLACfg(ckpt_dir(args.assets), vlm_dir(args.assets))
    W = ls.load_safetensors(os.path.join(ckpt_dir(args.assets), "model.safetensors"))
    KW = sq.vision_kernel_weights(sq.vision_weights(W), cfg)
    tok = ls.Tok(vlm_dir(args.assets))
    cs = ls.rope_tables(cfg.text, 1024)
    samples = load_samples(args.assets, "evaluation", 6 if args.quick else None)
    log(f"study {time.strftime('%Y-%m-%d %H:%M:%S')}  {len(samples)} evaluation samples, "
        f"{len(cfg.cameras)} cameras in the config, chunk {cfg.chunk}, {cfg.steps} steps")
    ch = calibrate(cfg, W, KW, tok, args.assets, args.jobs, cs, log)
    fl = sq.SmolVLA(cfg, W, KW=KW, cos_sin=cs)
    ref = run_model(fl, cfg, samples, tok, args.jobs)
    log(f"  float reference {time.time() - t0:.0f} s")
    results = {"float": dict(metrics=metrics(ref, ref, samples, cfg), stats=merge_stats(ref))}
    alt = run_model(fl, cfg, samples, tok, args.jobs, seed=YARD_SEED,
                    feats=[o["feats"] for o in ref])
    results["float@other-noise"] = dict(metrics=metrics(alt, ref, samples, cfg))
    del fl
    acts = {"float": [o["actions"] for o in ref], "float@other-noise": [o["actions"] for o in alt]}
    if args.torch:
        pol = torch_policy(args.assets, default_dtypes=True)
        tb = []
        for k, s in enumerate(samples):
            o = torch_sample(pol, s, cfg, noise(cfg, k))
            tb.append(dict(actions=o["actions"], feats=ref[k]["feats"], K=ref[k]["K"], V=ref[k]["V"], v0=ref[k]["v0"]))
        del pol
        results["lerobot (bf16)"] = dict(metrics=metrics(tb, ref, samples, cfg))
        acts["lerobot (bf16)"] = [o["actions"] for o in tb]
        log(f"  lerobot bf16 {time.time() - t0:.0f} s")
    names = args.policies.split(",") if args.policies else [p for p in sq.POLICIES if p != "float"]
    fmts = {}
    for name in names:
        pol = sq.POLICIES[name]
        t1 = time.time()
        fmt = sq.make_formats(pol, ch["vision"], ch["text"], ch["expert"], cfg, KW, W)
        fmts[name] = fmt
        m = sq.SmolVLA(cfg, W, pol, fmt, KW=KW, cos_sin=cs)
        outs = run_model(m, cfg, samples, tok, args.jobs)
        del m
        results[name] = dict(metrics=metrics(outs, ref, samples, cfg), stats=merge_stats(outs),
                             wsat=outs[0]["wsat"])
        acts[name] = [o["actions"] for o in outs]
        mt = results[name]["metrics"]
        log(f"  [{name}] actions mean|d| {mt['mae']['mean']:.4f} (first 10 {mt['mae10']['mean']:.4f}) "
            f"rel {mt['rel']['mean']:.4f} (max {mt['rel']['max']:.4f})  feat {mt['feat']['mean']:.4f}  "
            f"K0 {mt['k_first']['mean']:.4f} K15 {mt['k_last']['mean']:.4f}  v0 {mt['v0']['mean']:.4f}  "
            f"{time.time() - t1:.0f} s")
    json.dump(dict(results=results, samples=[s.name for s in samples],
                   actions={n: [a.tolist() for a in v] for n, v in acts.items()},
                   gt=[s.gt.tolist() for s in samples]),
              open(os.path.join(out_dir, "results.json"), "w"), indent=0)
    report(results, cfg, log)
    log(f"total {time.time() - t0:.0f} s")


def report(results, cfg, log):
    log("\n== action chunk vs float (same noise; normalised units, the first "
        f"{cfg.action_dim} dims; gt = the recorded actions) ==")
    log(f"{'policy':30s} {'mean|d|':>8s} {'first10':>8s} {'rel':>7s} {'rel max':>7s} {'max|d|':>7s} "
        f"{'gt mae':>7s} {'feat':>7s} {'K0':>7s} {'K15':>7s} {'v0':>7s}")
    for n, r in results.items():
        m = r["metrics"]
        log(f"{n:30s} {m['mae']['mean']:8.4f} {m['mae10']['mean']:8.4f} {m['rel']['mean']:7.4f} "
            f"{m['rel']['max']:7.4f} {m['maxd']['max']:7.3f} {m['gt_mae']['mean']:7.4f} {m['feat']['mean']:7.4f} "
            f"{m['k_first']['mean']:7.4f} {m['k_last']['mean']:7.4f} {m['v0']['mean']:7.4f}")
    st = results["float"].get("stats", {})
    log("\n== value ranges (float): max |x| per class ==")
    for p in ("vision", "text", "expert"):
        cl = st.get(p, {})
        log(f"  {p:7s} " + "  ".join(f"{c} {v['max']:.1f}" for c, v in sorted(cl.items()) if not c.startswith("acc:")))
    log("\n== saturated elements (emulated) ==")
    for n, r in results.items():
        if "wsat" not in r:
            continue
        # "h" counts |h| > 128 (Q8.8), a saturation only for the q88 residual
        sat = {f"{p}.{c}": v["sat"] for p, cl in r["stats"].items() for c, v in cl.items() if v["sat"]
               and not c.startswith("acc:") and (c != "h" or n == "q88")}
        acc = {f"{p}.{c}": round(v["max"], 1) for p, cl in r["stats"].items() for c, v in cl.items()
               if c.startswith("acc:")}
        wrapped = sum(v["sat"] for p, cl in r["stats"].items() for c, v in cl.items() if c.startswith("acc:"))
        log(f"  [{n}] weights saturated {r['wsat']}; saturated {sat or 0}; wrapped acc {wrapped}; "
            f"max |acc| {max(acc.values()) if acc else 0}")


# ------------------------------------------------------------------ ablation
def cmd_ablate(args):
    cfg = sq.VLACfg(ckpt_dir(args.assets), vlm_dir(args.assets))
    W = ls.load_safetensors(os.path.join(ckpt_dir(args.assets), "model.safetensors"))
    KW = sq.vision_kernel_weights(sq.vision_weights(W), cfg)
    tok = ls.Tok(vlm_dir(args.assets))
    cs = ls.rope_tables(cfg.text, 1024)
    samples = load_samples(args.assets, "evaluation", args.n)
    ch = calibrate(cfg, W, KW, tok, args.assets, args.jobs, cs, print)
    ref = run_model(sq.SmolVLA(cfg, W, KW=KW, cos_sin=cs), cfg, samples, tok, args.jobs)
    base = args.base
    full = sq.make_formats(sq.POLICIES[base], ch["vision"], ch["text"], ch["expert"], cfg, KW, W)
    print(f"ablation of {base}, {len(samples)} samples: action mean|d| vs float (rel)")
    runs = [("all", ("vision", "text", "expert")), ("vision only", ("vision",)), ("VLM only", ("text",)),
            ("expert only", ("expert",)), ("VLM + expert", ("text", "expert"))]
    for name, parts in runs:
        pol = sq.part_policy(base, parts)
        fmt = {p: (full[p] if p in parts else {}) for p in full}
        outs = run_model(sq.SmolVLA(cfg, W, pol, fmt, KW=KW, cos_sin=cs), cfg, samples, tok, args.jobs)
        mt = metrics(outs, ref, samples, cfg)
        print(f"  {name:16s} mean|d| {mt['mae']['mean']:.4f} (first 10 {mt['mae10']['mean']:.4f})  "
              f"rel {mt['rel']['mean']:.4f}  v0 {mt['v0']['mean']:.4f}", flush=True)


# ------------------------------------------------------------------ formats
def formats_path(policy):
    return os.path.normpath(os.path.join(HERE, "..", f"smolvla_base_{policy}_formats.json"))


def cmd_formats(args):
    cfg = sq.VLACfg(ckpt_dir(args.assets), vlm_dir(args.assets))
    W = ls.load_safetensors(os.path.join(ckpt_dir(args.assets), "model.safetensors"))
    KW = sq.vision_kernel_weights(sq.vision_weights(W), cfg)
    tok = ls.Tok(vlm_dir(args.assets))
    ch = calibrate(cfg, W, KW, tok, args.assets, args.jobs, ls.rope_tables(cfg.text, 1024), print)
    pol = sq.POLICIES[args.policy]
    fmt = sq.make_formats(pol, ch["vision"], ch["text"], ch["expert"], cfg, KW, W)
    path = formats_path(args.policy)
    d = sq.formats_json(fmt, args.policy, pol)
    d["checkpoint"] = inputs()["model"]
    json.dump(d, open(path, "w"))
    print(f"{path}: " + ", ".join(f"{p} {len(f)} entries" for p, f in fmt.items()))


# ------------------------------------------------------------------ costs
def part_costs(cfg, n_cam, n_lang):
    """MACs and int16 weight bytes per part of one action chunk."""
    v, t, e = cfg.vision, cfg.text, cfg.expert
    N, D, F = v.N, v.D, v.F
    vis = N * 3 * v.P * v.P * D + v.L * (4 * N * D * D + 2 * N * N * D + 2 * N * D * F) \
        + (N // cfg.scale ** 2) * D * cfg.scale ** 2 * t.D
    vis_w = 3 * v.P * v.P * D + v.L * (4 * D * D + 2 * D * F) + D * cfg.scale ** 2 * t.D
    T = n_cam * cfg.img_tokens + n_lang + 1
    kv = t.KV * t.HD
    per = t.D * (t.H * t.HD + 2 * kv) + t.H * t.HD * t.D + 3 * t.D * t.F
    vlm = (t.L - 1) * (T * per + 2 * t.H * T * T * t.HD) + T * t.D * 2 * kv
    vlm_w = (t.L - 1) * per + t.D * 2 * kv
    n = cfg.chunk
    xl = [l for l in range(e.L) if not cfg.self_layer(l)]
    cross = len(xl) * 2 * T * kv * kv
    per_e = e.D * e.H * e.HD + e.H * e.HD * e.D + 3 * e.D * e.F
    step = e.L * n * per_e + (e.L - len(xl)) * n * e.D * 2 * kv \
        + sum(2 * e.H * n * (T + (n if cfg.self_layer(l) else 0)) * e.HD for l in range(e.L)) \
        + n * cfg.max_action * e.D + n * e.D * e.D + n * e.D * cfg.max_action
    exp_w = e.L * per_e + (e.L - len(xl)) * e.D * 2 * kv + len(xl) * 2 * kv * kv \
        + cfg.max_action * e.D + e.D * e.D + e.D * cfg.max_action
    return dict(T=T, vision=vis * n_cam, vision_w=2 * vis_w, vlm=vlm, vlm_w=2 * vlm_w, cross=cross,
                step=step, expert=cfg.steps * step, expert_w=2 * exp_w)


# board references (doc/plans/CHAT_PLAN.md §26, bitstream 04259a7af917): SmolVLM-256M's vision
# encoder (the same SigLIP encoder) 1614 ms per image; SmolLM2-360M (32 layers 960 wide)
# prefill 16 / 64 / 256 tokens 191 / 233 / 1024 ms
VISION_MS = 1614.0
PREFILL_360M = ((16, 191.0), (64, 233.0), (256, 1024.0))


def prefill_ms(T):
    xs, ys = zip(*PREFILL_360M)
    return float(np.interp(T, xs, ys)) if T <= xs[-1] else ys[-1] + (T - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])


def cmd_costs(args):
    cfg = sq.VLACfg(ckpt_dir(args.assets), vlm_dir(args.assets))
    t = cfg.text
    per360 = t.D * (t.H * t.HD + 2 * t.KV * t.HD) + t.H * t.HD * t.D + 3 * t.D * t.F
    e = cfg.expert
    per_e = e.D * e.H * e.HD + e.H * e.HD * e.D + 3 * e.D * e.F + e.D * t.KV * t.HD   # the self layers' k / v
    ratio = per_e / per360
    print(f"MACs and int16 weights per action chunk ({cfg.chunk} actions, {cfg.steps} steps), "
          f"projected from the board's SmolVLM vision and SmolLM2-360M prefill times:")
    for n_cam in (1, 2, 3):
        c = part_costs(cfg, n_cam, args.lang)
        T = c["T"]
        vis_ms = VISION_MS * n_cam
        vlm_ms = prefill_ms(T) * (t.L - 1 + 0.25) / 32           # 15 full layers + the last one's k / v
        step_ms = prefill_ms(cfg.chunk) / 32 * e.L * ratio        # a 50-token prefill of 16 expert layers
        tot = vis_ms + vlm_ms + cfg.steps * step_ms
        print(f"  {n_cam} camera(s), T = {T:3d}: vision {c['vision'] / 1e9:6.1f} GMAC ({vis_ms:5.0f} ms), "
              f"VLM {c['vlm'] / 1e9:5.1f} GMAC ({vlm_ms:4.0f} ms), cross K'V' {c['cross'] / 1e9:4.2f} GMAC, "
              f"expert {c['expert'] / 1e9:5.1f} GMAC ({cfg.steps} x {step_ms:3.0f} ms)  ->  ~{tot / 1000:.1f} s")
    c = part_costs(cfg, 1, args.lang)
    tot_w = c["vision_w"] + c["vlm_w"] + c["expert_w"]
    print(f"weights (int16): vision + connector {c['vision_w'] / 2 ** 20:.0f} MiB, VLM {c['vlm_w'] / 2 ** 20:.0f} MiB, "
          f"expert {c['expert_w'] / 2 ** 20:.0f} MiB = {tot_w / 2 ** 20:.0f} MiB; the embedding table "
          f"{49280 * t.D * 2 / 2 ** 20:.0f} MiB (host, bf16)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["select", "fetch", "validate", "study", "ablate", "formats", "costs"])
    ap.add_argument("--assets", default=ASSETS)
    ap.add_argument("--record", action="store_true", help="fetch: record the SHA-256 of the datasets' files and frames")
    ap.add_argument("--drop-videos", action="store_true", help="fetch: delete the videos once the frames are out")
    ap.add_argument("--policies", default=None, help="study: comma-separated (default: all of smolvla_q.POLICIES)")
    ap.add_argument("--policy", default="pow2+p12+vgelu+vsmx+rowops", help="formats: the policy")
    ap.add_argument("--base", default="pow2+p12", help="ablate: the policy")
    ap.add_argument("--quick", action="store_true", help="study: 6 evaluation samples")
    ap.add_argument("--torch", action="store_true", help="study: lerobot's default-precision yardstick (torch, bf16 VLM and expert)")
    ap.add_argument("-n", type=int, default=3, help="validate / ablate: samples")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--lang", type=int, default=12, help="costs: language tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    {"select": cmd_select, "fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study, "ablate": cmd_ablate,
     "formats": cmd_formats, "costs": cmd_costs}[args.cmd](args)


if __name__ == "__main__":
    main()
