#!/usr/bin/env python3
"""postprocess.py — the stereo_depth demo's results (doc/plans/STEREO_PLAN.md).

Reads the board's build/results/disparity.bin (float32 [H][W] per pair, at
the network's resolution) and build/data/meta.json:

  * per pair the disparity at the pair's own resolution: the padding cropped
    (the image sits at the bottom left), scaled back up (bilinear) and
    divided by the scale -> build/results/disparity/<name>.png (the left
    image beside the disparity, turbo colours) and <name>.pfm;
  * EPE and bad-1 / bad-2 against the ground truth (non-occluded pixels) of
    every Middlebury / ETH3D pair;
  * for a RealSense pair (scripts/capture.py: meta.json with the camera's
    calibration) the depth, fx * baseline / disparity, beside the camera's own
    depth -> <name>_depth.png, and their agreement;
  * the first ``check`` maps bit for bit against the scheduler's simulation
    of the generated project (generate_project.graph_and_generator) on the
    same raw inputs.

usage: inference-scheduler/.venv/bin/python demo/stereo_depth/scripts/postprocess.py [--check N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(HERE))
DATA = DEMO / "build" / "data"
RESULTS = DEMO / "build" / "results"
MAX_DISP = 192


def turbo(x: np.ndarray) -> np.ndarray:
    """Google's turbo colour map (polynomial fit), x in [0, 1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def read_pfm(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        sc = float(f.readline())
        d = np.fromfile(f, "<f4" if sc < 0 else ">f4").reshape(h, w)
    return np.flipud(d).astype(np.float64)


def write_pfm(p: Path, d: np.ndarray) -> None:
    with open(p, "wb") as f:
        f.write(b"Pf\n" + f"{d.shape[1]} {d.shape[0]}\n-1\n".encode())
        np.flipud(d).astype("<f4").tofile(f)


def to_pair(disp: np.ndarray, m: dict) -> np.ndarray:
    """The network-resolution map -> the pair's own resolution, in its pixels."""
    sw, sh = m["scaled"]
    d = disp[m["height"] - sh:, :sw]
    w0, h0 = m["size"]
    if (sw, sh) != (w0, h0):
        d = np.asarray(Image.fromarray(d.astype(np.float32), mode="F").resize((w0, h0), Image.BILINEAR),
                       dtype=np.float64) / m["scale"]
    return d.astype(np.float64)


def metrics(d: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    e = np.abs(d - gt)[valid]
    return {"epe": float(e.mean()), "bad1": float((e > 1).mean() * 100), "bad2": float((e > 2).mean() * 100)}


def realsense_depth(d: np.ndarray, src: Path, left: Image.Image, png: Path) -> dict:
    """A RealSense pair (scripts/capture.py): depth = fx * baseline / disparity
    against the camera's own depth (depth_rs.png).  Writes the left image, our
    depth and the camera's side by side on one colour scale (near red, far
    blue, the camera's 2nd-98th percentile range) and returns the agreement
    where both have a depth."""
    meta = json.loads((src / "meta.json").read_text())
    with np.errstate(divide="ignore"):
        ours = np.where(d > 0.5, meta["fx"] * meta["baseline_m"] / np.maximum(d, 1e-6), 0.0)
    rs_d = np.asarray(Image.open(src / "depth_rs.png"), dtype=np.float64) * meta["depth_scale"]
    both = (ours > 0) & (rs_d > 0)
    out = {"realsense": {"fx": meta["fx"], "baseline_mm": meta["baseline_m"] * 1000, "emitter": meta["emitter"]}}
    if both.any():
        rel = np.abs(ours[both] - rs_d[both]) / rs_d[both]
        out["realsense"].update({"compared_px": int(both.sum()), "median_rel_diff": float(np.median(rel)),
                                 "within_5pct": float((rel < 0.05).mean() * 100),
                                 "within_10pct": float((rel < 0.10).mean() * 100),
                                 "median_depth_m": float(np.median(ours[both])),
                                 "rs_coverage": float((rs_d > 0).mean() * 100)})
    lo, hi = (np.percentile(rs_d[rs_d > 0], [2, 98]) if (rs_d > 0).any() else (0.3, 5.0))

    def col(z):
        rgb = turbo(1.0 - (np.clip(z, lo, hi) - lo) / max(hi - lo, 1e-6))
        rgb[z <= 0] = 0                                       # no depth: black
        return Image.fromarray(rgb)
    w, h = left.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(left, (0, 0))
    canvas.paste(col(ours), (w, 0))
    canvas.paste(col(rs_d), (2 * w, 0))
    canvas.save(png)
    return out


def bit_exact(maps: np.ndarray, meta: list, n: int) -> dict:
    """The first n board maps against the scheduler's simulation (float32 bits)."""
    if n <= 0:
        return {}
    from generate_project import graph_and_generator
    _g, cg = graph_and_generator()
    H, W, f = meta[0]["height"], meta[0]["width"], meta[0]["input_exp"]
    raw = np.fromfile(DATA / "pairs.bin", "<i2")
    per = 3 * H * W
    differ = []
    for i in range(min(n, len(meta))):
        left = raw[2 * i * per:(2 * i + 1) * per].astype(np.float64).reshape(1, 3, H, W) * 2.0 ** -f
        right = raw[(2 * i + 1) * per:(2 * i + 2) * per].astype(np.float64).reshape(1, 3, H, W) * 2.0 ** -f
        sim = cg._forward_pass({"left": left, "right": right})["disparity"].astype(np.float32)
        if not np.array_equal(sim.view(np.uint32), maps[i].view(np.uint32)):
            differ.append(meta[i]["name"])
            print(f"  {meta[i]['name']}: DIFFERS from the simulation "
                  f"(max |diff| {np.abs(sim - maps[i]).max():.4g} px)")
    return {"checked": min(n, len(meta)), "differ": differ}


def postprocess(check: int = 2) -> dict:
    meta = json.loads((DATA / "meta.json").read_text())
    H, W = meta[0]["height"], meta[0]["width"]
    maps = np.fromfile(RESULTS / "disparity.bin", "<f4").reshape(-1, H, W)
    if len(maps) != len(meta):
        raise SystemExit(f"{len(maps)} maps for {len(meta)} pairs")
    out_dir = RESULTS / "disparity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, disp in zip(meta, maps, strict=True):
        d = to_pair(disp, m)
        src = Path(m["dir"])
        left = Image.open(src / ("left.png" if m["set"] == "user" else "im0.png")).convert("RGB")
        row = {"name": m["name"], "set": m["set"], "size": m["size"], "scale": m["scale"],
               "range": [float(d.min()), float(d.max())]}
        top = float(np.percentile(d, 99)) or 1.0
        if m["set"] != "user":
            gt = read_pfm(src / "disp0GT.pfm")
            mask = np.asarray(Image.open(src / "mask0nocc.png")) == 255
            valid = np.isfinite(gt) & (gt > 0) & (gt < MAX_DISP) & mask
            row.update(metrics(d, np.where(np.isfinite(gt), gt, 0.0), valid))
            top = float(np.max(gt[valid])) if valid.any() else top
        if (src / "meta.json").exists():
            row.update(realsense_depth(d, src, left, out_dir / f"{m['name']}_depth.png"))
        rows.append(row)
        pic = Image.fromarray(turbo(d / top))
        canvas = Image.new("RGB", (left.width * 2, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(pic, (left.width, 0))
        canvas.save(out_dir / f"{m['name']}.png")
        write_pfm(out_dir / f"{m['name']}.pfm", d)
    r = {"pairs": len(rows), "labelled": sum(1 for x in rows if "epe" in x), "annotated": str(out_dir),
         "per_pair": rows}
    lab = [x for x in rows if "epe" in x]
    if lab:
        for k in ("epe", "bad1", "bad2"):
            r[k] = float(np.mean([x[k] for x in lab]))
    r["bit_exact"] = bit_exact(maps, meta, check)
    (RESULTS / "quality.json").write_text(json.dumps(r, indent=1))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--check", type=int, default=2, help="pairs checked bit for bit against the simulation")
    a = ap.parse_args(argv)
    r = postprocess(a.check)
    if "epe" in r:
        print(f"{r['labelled']} pairs with ground truth: EPE {r['epe']:.3f} px, bad-1 {r['bad1']:.2f} %, "
              f"bad-2 {r['bad2']:.2f} %")
    be = r["bit_exact"]
    if be:
        print("bit-exact against the simulation: " + ("yes" if not be["differ"] else f"NO {be['differ']}")
              + f" ({be['checked']} pairs)")
    print(f"-> {r['annotated']}")
    return 0 if not be or not be["differ"] else 1


if __name__ == "__main__":
    sys.exit(main())
