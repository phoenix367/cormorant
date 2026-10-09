#!/usr/bin/env python3
"""stereo_study.py — can the KV260 run a stereo depth network?  (doc/plans/STEREO_PLAN.md)

LightStereo-S (OpenStereo, Guo et al. 2024, arXiv 2406.19833): a MobileNetV2
backbone, a correlation cost volume over 48 disparities at 1/4 resolution,
2D cost aggregation (MobileNetV2 blocks, stripe attention from the left
image), soft-argmax regression and a learned 4x context upsampling.  Every
layer is 2D — no 3D convolution, no grid_sample, no recurrence.

  fetch     the checkpoints (Hugging Face XiandaGuo/OpenStereo at a pinned
            revision), OpenStereo at a pinned commit (the reference code), the
            Middlebury MiddEval3 quarter-resolution and ETH3D low-res two-view
            training sets (SHA-256 checked)
  validate  this script's functional model (float) against OpenStereo's module
  study     numeric policies on the 15 + 27 pairs with ground truth: EPE and
            bad-1 / bad-2 against the ground truth and against float
  costs     MACs per stage at a resolution, split by where each would run

The functional model is the emulation of the planned partition (the
specification a frontend would have to match bit for bit):
  ConvKernel  every conv / deconv: int16 operands, an exact sum, the output
              floor(acc / 2^8) saturated (BatchNorm folded into the weights)
  VectorOP    residual adds (exact, saturating), the attention MUL (floor),
              LeakyReLU (rounded to nearest, ties to even), ReLU / ReLU6
              (exact), both softmaxes (the unit's column mode, P at 2^-15)
  host        InstanceNorm, the replicate pad, the correlation volume (exact
              mean), the context upsampling (double, float32 output)

usage: .venv-export/bin/python demo/stereo_depth/scripts/stereo_study.py fetch|validate|study|costs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_assets import (ASSETS, CKPT_DIR, CKPTS, DATA_DIR, DEMO, FORMATS, OPENSTEREO,  # noqa: E402,F401
                           STUDY_DIR, fetch, sha256, write_formats)

MAX_DISP = 192
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def cmd_fetch(_a) -> None:
    fetch(study=True)


# ── data ──────────────────────────────────────────────────────────────────────

def read_pfm(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        d = np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(h, w)
    return np.flipud(d).astype(np.float64)


def read_rgb(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def pairs():
    """[(set, name, dir)] — Middlebury MiddEval3 trainingQ, then ETH3D two-view training."""
    out = []
    for s, root in (("middlebury", DATA_DIR / "MiddEval3" / "trainingQ"), ("eth3d", DATA_DIR / "eth3d")):
        if root.exists():
            out += [(s, d.name, d) for d in sorted(root.iterdir()) if (d / "disp0GT.pfm").exists()]
    return out


def load_pair(d: Path):
    from PIL import Image
    im_l, im_r = read_rgb(d / "im0.png"), read_rgb(d / "im1.png")
    gt = read_pfm(d / "disp0GT.pfm")
    mask = np.asarray(Image.open(d / "mask0nocc.png")) == 255
    valid = np.isfinite(gt) & (gt > 0) & (gt < MAX_DISP) & mask
    return im_l, im_r, np.where(np.isfinite(gt), gt, 0.0), valid


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    from PIL import Image
    h, w = img.shape[:2]
    im = Image.fromarray(img.astype(np.uint8)).resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def upscale(disp: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(disp))[None, None]
    return F.interpolate(t, (h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


def prep(img: np.ndarray, th: int, tw: int) -> torch.Tensor:
    """OpenStereo's evaluation transform: RightTopPad (edge) to th x tw, /255, ImageNet normalisation."""
    h, w = img.shape[:2]
    img = np.pad(img, ((th - h, 0), (0, tw - w), (0, 0)), "edge")
    x = (img / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


# ── numerics ──────────────────────────────────────────────────────────────────

def tag(t: torch.Tensor, f: int) -> torch.Tensor:
    """Record a tensor's exponent (its raw int16 value is t * 2^f)."""
    t.f = f
    return t


class Num:
    """Where each tensor is rounded, and at which power-of-two exponent.

    Policies: 'float' rounds nothing; 'calib' rounds nothing and records the
    largest |value| of every site; 'q88' is the planned partition with every
    int16 tensor at 2^-8 (the softmax P at 2^-15, as the VectorOP unit writes
    it); 'pow2' gives every site the exponent of its calibrated range with one
    bit of headroom (`table`), within what the kernels allow: ReLU6 outputs at
    2^-8 (VectorOP's RELU6 clamps at raw 6.0 in Q8.8), a conv's weight at
    f_y + 8 - f_x and its bias at f_y (numeric.encode_conv_weights), adds and
    concats at the coarser operand exponent (the finer operand rescaled with a
    floor, a VectorOP MUL by 2^-k), a MUL at f_a + f_b - 8.

    Ablation options, '+'-separated after the base: 'fw' keeps the weights
    float; 'x<stage>' keeps a stage exact; 'o<stage>' rounds only that stage
    (stages: input, backbone, fpn, correlation, aggregation, attention,
    regression, refine, upsample)."""

    def __init__(self, policy: str, table: dict | None = None):
        self.policy = policy
        base, *opts = policy.split("+")
        self.q = base in ("q88", "pow2", "bf16")
        self.bf16 = base == "bf16"     # the yardstick: every tensor and weight rounded to bfloat16
        self.r6 = "r6" in opts         # ReLU6 outputs at their calibrated exponent too (not at 2^-8)
        self.pc = "pc" in opts or "pc5" in opts   # per-output-channel weight exponents (+ a rescale)
        self.pc_min = 0.05 if "pc5" in opts else 0.0  # pc5: only where per-tensor rounding loses > 5 %
        self.hmul = "hmul" in opts     # the attention MUL on the host, at its calibrated exponent
        self.calib = base == "calib"
        self.pow2 = base == "pow2"
        self.table = table or {}
        self.maxes: dict[str, float] = {}
        self.fw = "fw" in opts
        self.exempt = {o[1:] for o in opts if o.startswith("x")}
        self.only = {o[1:] for o in opts if o.startswith("o")}
        self.stage = "input"
        self.stats = {"sat": 0, "n": 0, "wsat": 0, "requant": 0, "sat_by_stage": {}}

    def on(self) -> bool:
        return self.q and self.stage not in self.exempt and (not self.only or self.stage in self.only)

    def exp(self, site: str, default: int = 8) -> int:
        return self.table.get(site, default) if self.pow2 else default

    def record(self, site, t):
        if self.calib and site:
            self.maxes[site] = max(self.maxes.get(site, 0.0), float(t.abs().max()))
            if t.dim() == 4:
                c = t.abs().amax(dim=(0, 2, 3)).tolist()
                prev = self.maxes.get(site + ":c")
                self.maxes[site + ":c"] = c if prev is None else [max(a, b) for a, b in zip(prev, c, strict=True)]

    def _sat(self, t: torch.Tensor, f: int) -> torch.Tensor:
        if self.bf16:
            return t
        lo, hi = -32768 / 2 ** f, 32767 / 2 ** f
        n = int(((t < lo) | (t > hi)).sum())
        st = self.stats
        st["sat"] += n
        st["n"] += t.numel()
        st["sat_by_stage"][self.stage] = st["sat_by_stage"].get(self.stage, 0) + n
        return t.clamp(lo, hi)

    def w(self, t, f=8):                    # a weight or bias, rounded once, half to even
        if not self.on() or self.fw:
            return t
        if self.bf16:
            return t.to(torch.bfloat16).double()
        r = torch.round(t * 2 ** f)
        self.stats["wsat"] += int(((r < -32768) | (r > 32767)).sum())
        return r.clamp(-32768, 32767) / 2 ** f

    def w_pc(self, t, fx, fy, site):
        """Per output channel c: the finest f_w,c whose weights fit int16 and
        whose output f_w,c + f_x - 8 holds the channel's calibrated range
        (one bit of headroom); never coarser than the per-tensor f_y + 8 - f_x.
        The kernel writes channel c at f_w,c + f_x - 8; a VectorOP MUL by
        2^-(f_w,c - f_w) floors it to f_y: floor of a floor, the same as one
        floor at f_y."""
        if not self.on() or self.fw:
            return t
        oc = t.shape[0]
        cmax = self.table.get(site + ":c")
        out = torch.empty_like(t)
        for c in range(oc):
            wmax = float(t[c].abs().max())
            fw_w = 15 if wmax == 0 else math.floor(math.log2(32767 / wmax))
            fy_c = 15 if not cmax or cmax[c] == 0 else math.floor(math.log2(16383 / cmax[c]))
            fwc = max(fy + 8 - fx, min(fw_w, fy_c + 8 - fx, 24))
            out[c] = torch.round(t[c] * 2 ** fwc) / 2 ** fwc
        return out

    def kernel(self, t, f=8, site=None):    # a kernel output: floor(acc / 2^8)
        self.record(site, t)
        if self.on() and self.bf16:
            return tag(t.to(torch.bfloat16).double(), f)
        return tag(self._sat(torch.floor(t * 2 ** f) / 2 ** f, f) if self.on() else t, f)

    def host(self, t, f=8, site=None):      # a host op / VectorOP rounded output: half to even
        self.record(site, t)
        if self.on() and self.bf16:
            return tag(t.to(torch.bfloat16).double(), f)
        return tag(self._sat(torch.round(t * 2 ** f) / 2 ** f, f) if self.on() else t, f)

    def requant(self, t, f):                # to a coarser exponent: floor (VectorOP MUL by 2^-k)
        if not self.on() or self.bf16 or getattr(t, "f", f) <= f:
            return t
        self.stats["requant"] += 1
        return tag(torch.floor(t * 2 ** f) / 2 ** f, f)

    def add(self, a, b, site=None):         # VectorOP ADD: exact, saturating
        fa, fb = getattr(a, "f", 8), getattr(b, "f", 8)
        self.record(site, a + b)
        f = min(fa, fb, self.exp(site, min(fa, fb))) if site else min(fa, fb)
        if not self.on():
            return tag(a + b, f)
        return tag(self._sat(self.requant(a, f) + self.requant(b, f), f), f)

    def concat(self, ts, site=None):        # a host copy: one exponent
        f = min(getattr(t, "f", 8) for t in ts)
        return tag(torch.cat([self.requant(t, f) for t in ts], 1), f)

    def mul(self, a, b, site=None):         # VectorOP MUL: raw product >> 8, truncated, saturating
        f = getattr(a, "f", 8) + getattr(b, "f", 8) - 8
        y = self.kernel(a * b, f, site)
        return self.requant(y, min(f, self.exp(site, f))) if site else y


# ── the model, functional ─────────────────────────────────────────────────────

def load_state(name: str) -> dict:
    sd = torch.load(CKPT_DIR / Path(CKPTS[name][0]).name, map_location="cpu", weights_only=False)["model_state"]
    return {k: v.double() for k, v in sd.items() if "num_batches" not in k}


def fold(sd: dict, conv: str, bn: str | None, transposed: bool = False):
    """Conv (or ConvTranspose) weight + bias with the BatchNorm folded in."""
    w = sd[conv + ".weight"]
    b = sd.get(conv + ".bias", torch.zeros(w.shape[1] if transposed else w.shape[0], dtype=w.dtype))
    if bn is not None:
        s = sd[bn + ".weight"] / torch.sqrt(sd[bn + ".running_var"] + 1e-5)
        w = w * (s.view(1, -1, 1, 1) if transposed else s.view(-1, 1, 1, 1))
        b = (b - sd[bn + ".running_mean"]) * s + sd[bn + ".bias"]
    return w, b


class LightStereoF:
    """LightStereo-S as explicit tensor operations under a Num policy.
    Every rounding of the planned partition is a Num call; under 'float' it is
    exactly OpenStereo's module (`validate`)."""

    MOBILENETV2 = [  # timm mobilenetv2_100: (stage, blocks, expansion, out, stride)
        (0, 1, 1, 16, 1), (1, 2, 6, 24, 2), (2, 3, 6, 32, 2), (3, 4, 6, 64, 2), (4, 3, 6, 96, 1),
        (5, 3, 6, 160, 2), (6, 1, 6, 320, 1)]

    def __init__(self, sd: dict, num: Num, stripe_max: int = 7):
        self.sd, self.n, self.stripe_max = sd, num, stripe_max
        self.macs: dict[str, float] = {}
        self.stage = "input"

    @property
    def stage(self) -> str:
        return self.n.stage

    @stage.setter
    def stage(self, v: str) -> None:
        self.n.stage = v

    # ConvKernel calls
    def conv(self, x, conv, bn=None, stride=1, pad=0, groups=1, act=None, transposed=False, out_pad=0,
             cap=None):
        w, b = fold(self.sd, conv, bn, transposed)
        n = self.n
        fx = getattr(x, "f", 8)
        fy = 8 if act == "relu6" and not n.r6 else n.exp(conv)
        if cap is not None:
            fy = min(fy, cap)
        wq = n.w(w, fy + 8 - fx)
        if n.pc and n.on() and not n.fw and groups == 1 and \
                float((wq - w).norm() / max(float(w.norm()), 1e-12)) > n.pc_min:
            wq = n.w_pc(w if not transposed else w.transpose(0, 1), fx, fy, conv)
            if transposed:
                wq = wq.transpose(0, 1)
            n.stats["pc_convs"] = n.stats.get("pc_convs", 0) + 1
        bq = n.w(b, fy)
        if n.on():
            n.stats.setdefault("wq", {})[conv] = (fx, fy, fy + 8 - fx,
                                                  float((wq - w).norm() / max(float(w.norm()), 1e-12)))
        if transposed:
            y = F.conv_transpose2d(x, wq, bq, stride=stride, padding=pad, output_padding=out_pad)
            macs = x.numel() * w.shape[1] * w.shape[2] * w.shape[3]
        else:
            y = F.conv2d(x, wq, bq, stride=stride, padding=pad, groups=groups)
            macs = y.numel() * (w.shape[1] * w.shape[2] * w.shape[3])
        self.macs[self.stage] = self.macs.get(self.stage, 0) + macs
        y = n.kernel(y, fy, conv)
        return self.act(y, act)

    def act(self, y, act):
        f = getattr(y, "f", 8)
        if act == "relu":
            return tag(F.relu(y), f)
        if act == "relu6":
            return tag(y.clamp(0, 6), f)
        if act is not None and act.startswith("leaky"):
            return self.n.host(F.leaky_relu(y, float(act[5:]) if len(act) > 5 else 0.01), f)
        return y

    def instnorm(self, x, site):  # host, double, rounded half to even
        m = x.mean(dim=(2, 3), keepdim=True)
        v = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        return self.n.host((x - m) / torch.sqrt(v + 1e-5), self.n.exp(site), site)

    def stripe(self, x, conv, horizontal: bool):
        """A depthwise 1 x k (or k x 1) conv with bias; k > stripe_max runs as
        pieces of <= stripe_max taps (each a ConvKernel call, floored), summed
        by VectorOP adds."""
        w = self.sd[conv + ".weight"]
        b = self.sd[conv + ".bias"]
        k = w.shape[3] if horizontal else w.shape[2]
        fx, fy = getattr(x, "f", 8), self.n.exp(conv)
        if k <= self.stripe_max:
            y = F.conv2d(x, self.n.w(w, fy + 8 - fx), self.n.w(b, fy),
                         padding=(0, k // 2) if horizontal else (k // 2, 0), groups=x.shape[1])
            self.macs[self.stage] = self.macs.get(self.stage, 0) + y.numel() * k
            return self.n.kernel(y, fy, conv)
        out = None
        for s in range(0, k, self.stripe_max):
            e = min(k, s + self.stripe_max)
            wp = w[..., s:e] if horizontal else w[:, :, s:e, :]
            # taps s..e-1 of a centred k-tap conv: zero-pad so the piece reads x[i + t - k//2]
            lo, hi = k // 2 - s, (e - 1) - k // 2
            xp = F.pad(x, (lo, max(0, hi), 0, 0) if horizontal else (0, 0, lo, max(0, hi)))
            if hi < 0:   # the piece ends before the centre: drop the surplus columns / rows
                xp = xp[..., :hi] if horizontal else xp[:, :, :hi, :]
            y = F.conv2d(xp, self.n.w(wp, fy + 8 - fx), self.n.w(b, fy) if s == 0 else None, groups=x.shape[1])
            self.macs[self.stage] = self.macs.get(self.stage, 0) + y.numel() * (e - s)
            y = self.n.kernel(y, fy, conv)
            out = y if out is None else self.n.add(out, y, conv + ".sum")
        return out

    # blocks
    def mbv2_block(self, x, p, expand, stride, skip):
        if expand == 1:   # timm DepthwiseSeparableConv
            h = self.conv(x, p + ".conv_dw", p + ".bn1", stride, 1, groups=x.shape[1], act="relu6")
            h = self.conv(h, p + ".conv_pw", p + ".bn2")
        else:             # timm InvertedResidual
            h = self.conv(x, p + ".conv_pw", p + ".bn1", act="relu6")
            h = self.conv(h, p + ".conv_dw", p + ".bn2", stride, 1, groups=h.shape[1], act="relu6")
            h = self.conv(h, p + ".conv_pwl", p + ".bn3")
        return self.n.add(x, h, p + ".add") if skip else h

    def mbv2_stage(self, x, stage):
        _, nb, e, out, s = self.MOBILENETV2[stage]
        for j in range(nb):
            st = s if j == 0 else 1
            x = self.mbv2_block(x, self._bp(stage, j), e, st, st == 1 and x.shape[1] == out)
        return x

    def _bp(self, stage, j):
        # LightStereo's Backbone keeps timm's blocks as block0..block4 (block3 = blocks[3:5])
        if stage <= 2:
            return f"backbone.block{stage}.{j}"
        if stage in (3, 4):     # nn.Sequential(blocks[3:5]) keeps timm's indices 3 and 4
            return f"backbone.block3.{stage}.{j}"
        return f"backbone.block4.{j}"

    def fpn(self, low, high, p):
        x = self.conv(low, p + ".deconv.block.0", p + ".deconv.block.1", 2, 1, act="leaky0.2", transposed=True)
        x = self.n.concat([high, x])
        return self.conv(x, p + ".conv.block.0", p + ".conv.block.1", 1, 1, act="leaky0.2")

    def backbone(self, img):
        self.stage = "backbone"
        c1 = self.conv(img, "backbone.conv_stem", "backbone.bn1", 2, 1, act="relu6")
        c1 = self.mbv2_stage(c1, 0)
        c2 = self.mbv2_stage(c1, 1)
        c3 = self.mbv2_stage(c2, 2)
        c4 = self.mbv2_stage(self.mbv2_stage(c3, 3), 4)
        c5 = self.mbv2_stage(c4, 5)
        self.stage = "fpn"
        p4 = self.fpn(c5, c4, "backbone.fpn_layer4")
        p3 = self.fpn(p4, c3, "backbone.fpn_layer3")
        p2 = self.fpn(p3, c2, "backbone.fpn_layer2")
        p2 = tag(F.pad(p2, (1, 1, 1, 1), mode="replicate"), p2.f)   # host copy (exact)
        p2 = self.conv(p2, "backbone.out_conv.block.0")
        p2 = self.instnorm(p2, "backbone.out_conv.in")
        return [p2, p3, p4, c5]

    def correlation(self, fl, fr, d):
        """The cost volume: host op, the exact mean over channels, rounded."""
        b, c, h, w = fl.shape
        vol = torch.zeros(b, d, h, w, dtype=fl.dtype)
        for i in range(d):
            vol[:, i, :, i:] = (fl[:, :, :, i:] * fr[:, :, :, :w - i]).mean(1)
        self.macs["correlation"] = self.macs.get("correlation", 0) + c * h * sum(w - i for i in range(d))
        return self.n.host(vol, self.n.exp("correlation"), "correlation")

    def mv2res(self, x, p, stride, skip):
        h = self.conv(x, p + ".pwconv.0", p + ".pwconv.1", act="relu6")
        h = self.conv(h, p + ".dwconv.0", p + ".dwconv.1", stride, 1, groups=h.shape[1], act="relu6")
        h = self.conv(h, p + ".pwliner.0", p + ".pwliner.1")
        return self.n.add(x, h, p + ".add") if skip else h

    def attention(self, cost, feat, p):
        outer, self.stage = self.stage, "attention"
        try:
            return self._attention(cost, feat, p)
        finally:
            self.stage = outer

    def _attention(self, cost, feat, p):
        a = self.conv(feat, p + ".conv0")
        a0 = self.stripe(self.stripe(a, p + ".conv0_1", True), p + ".conv0_2", False)
        a1 = self.stripe(self.stripe(a, p + ".conv1_1", True), p + ".conv1_2", False)
        a2 = self.stripe(self.stripe(a, p + ".conv2_1", True), p + ".conv2_2", False)
        a = self.n.add(self.n.add(self.n.add(a, a0, p + ".sum0"), a1, p + ".sum1"), a2, p + ".sum2")
        # conv3's exponent leaves the MUL's (f_a + f_cost - 8) within its range
        if self.n.hmul:     # a host product, rounded at the calibrated exponent
            a = self.conv(a, p + ".conv3")
            return self.n.host(a * cost, self.n.exp(p + ".mul"), p + ".mul")
        cap = self.n.exp(p + ".mul", 99) + 8 - getattr(cost, "f", 8) if self.n.pow2 else None
        a = self.conv(a, p + ".conv3", cap=cap)
        return self.n.mul(a, cost, p + ".mul")

    def aggregation(self, x, fl):
        p = "cost_agg"
        x = self.mv2res(x, p + ".conv0.0", 1, True)
        x = self.attention(x, fl[0], p + ".att0")
        c1 = self.mv2res(x, p + ".conv1", 2, False)
        c2 = self.mv2res(c1, p + ".conv2.0", 1, True)
        c2 = self.attention(c2, fl[1], p + ".att2")
        c3 = self.mv2res(c2, p + ".conv3", 2, False)
        c4 = c3
        for j in range(3):
            c4 = self.mv2res(c4, f"{p}.conv4.{j}", 1, True)
        c4 = self.attention(c4, fl[2], p + ".att4")
        c5 = self.conv(c4, p + ".conv5.0", p + ".conv5.1", 2, 1, transposed=True, out_pad=1)
        c5 = self.act(self.n.add(c5, self.mv2res(c2, p + ".redir2", 1, True), p + ".c5add"), "relu")
        c6 = self.conv(c5, p + ".conv6.0", p + ".conv6.1", 2, 1, transposed=True, out_pad=1)
        return self.act(self.n.add(c6, self.mv2res(x, p + ".redir1", 1, True), p + ".c6add"), "relu")

    def softmax_unit(self, logits):
        """VectorOP's softmax unit, column mode: P at 2^-15 (the unit is within
        1 LSB of the exact softmax; emulated exactly rounded)."""
        p = torch.softmax(logits, dim=1)
        return self.n.host(p, 15)            # P at 2^-15 whatever the logits' exponent

    def run(self, left, right):
        """The whole network: disparity [H, W] at full resolution (float)."""
        n = self.n
        x = n.host(torch.cat([left, right], 0), n.exp("input"), "input")
        f = self.backbone(x)
        fl, fr = [tag(t[:1], t.f) for t in f], tag(f[0][1:], f[0].f)
        self.stage = "correlation"
        vol = self.correlation(fl[0], fr, MAX_DISP // 4)
        self.stage = "aggregation"
        agg = self.aggregation(vol, fl)
        self.stage = "regression"
        prob = self.softmax_unit(agg)
        # Σ_d P_d · d: a 1 x 1 ConvKernel call, P at 2^-15, d exact at f_w = f_out + 8 - 15
        dvals = torch.arange(MAX_DISP // 4, dtype=prob.dtype).view(1, -1, 1, 1)
        init_disp = n.kernel((prob * dvals).sum(1, keepdim=True), n.exp("regression"), "regression")
        self.macs[self.stage] = self.macs.get(self.stage, 0) + prob.numel()
        self.stage = "refine"
        img1 = tag(x[:1], x.f)
        xs = self.conv(fl[0], "refine_1.0.block.0", pad=1)
        xs = self.act(self.instnorm(xs, "refine_1.0.in"), "leaky")
        xs = self.conv(xs, "refine_1.1.block.0", pad=1)
        xs = self.act(self.instnorm(xs, "refine_1.1.in"), "relu")
        st = self.conv(img1, "stem_2.0.block.0", "stem_2.0.block.1", 2, 1, act="leaky")
        st = self.conv(st, "stem_2.1.block.0", "stem_2.1.block.1", 1, 1, act="relu")
        xs = self.fpn(xs, st, "refine_2")
        xs = self.conv(xs, "refine_3.block.0", None, 2, 1, transposed=True)
        spx = self.softmax_unit(xs)                     # [1, 9, H, W]
        self.stage = "upsample"
        disp = context_upsample(init_disp * 4.0, spx)   # host, double -> float32
        self.macs[self.stage] = self.macs.get(self.stage, 0) + spx.numel()
        return disp[0], init_disp


def context_upsample(disp_low, w):
    b, _, h, ww = disp_low.shape
    u = F.unfold(disp_low, kernel_size=3, padding=1).reshape(b, 9, h, ww)
    u = F.interpolate(u, (h * 4, ww * 4), mode="nearest")
    return (u * w).sum(1)


# ── validate ──────────────────────────────────────────────────────────────────

def openstereo_model(name: str):
    """OpenStereo's LightStereo module, imported file by file (its package
    __init__ pulls the dataset code and OpenCV)."""
    sys.path.insert(0, str(OPENSTEREO))
    for pkg in ("stereo", "stereo.modeling", "stereo.modeling.common", "stereo.modeling.cost_volume",
                "stereo.modeling.disp_pred", "stereo.modeling.disp_refinement", "stereo.modeling.models",
                "stereo.modeling.models.lightstereo"):
        m = types.ModuleType(pkg)
        m.__path__ = [str(OPENSTEREO / pkg.replace(".", "/"))]
        sys.modules[pkg] = m
    def load(mod):
        spec = importlib.util.spec_from_file_location(mod, OPENSTEREO / (mod.replace(".", "/") + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod] = m
        spec.loader.exec_module(m)
        return m
    for mod in ("stereo.modeling.common.basic_block_2d", "stereo.modeling.cost_volume.cost_volume",
                "stereo.modeling.disp_pred.disp_regression", "stereo.modeling.disp_refinement.disp_refinement",
                "stereo.modeling.models.lightstereo.backbone", "stereo.modeling.models.lightstereo.aggregation"):
        load(mod)
    ls = load("stereo.modeling.models.lightstereo.lightstereo")
    import timm
    orig = timm.create_model
    def create(*a, **k):   # no ImageNet download; timm >= 1.0 folds act1 into bn1 (BatchNormAct2d)
        m = orig(*a, **{**k, "pretrained": False})
        if not hasattr(m, "act1"):
            m.act1 = torch.nn.Identity()
        return m
    timm.create_model = create
    cfg = types.SimpleNamespace(MAX_DISP=MAX_DISP, LEFT_ATT=True, AGGREGATION_BLOCKS=[1, 2, 4], EXPANSE_RATIO=4)
    cfg.get = lambda k, d=None: getattr(cfg, k, d)
    model = ls.LightStereo(cfg)
    timm.create_model = orig
    sd = torch.load(CKPT_DIR / Path(CKPTS[name][0]).name, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(sd, strict=True)
    return model.eval()


def cmd_validate(a) -> None:
    torch.set_grad_enabled(False)
    s, name, d = pairs()[0]
    im_l, im_r, gt, valid = load_pair(d)
    th, tw = math.ceil(im_l.shape[0] / 32) * 32, math.ceil(im_l.shape[1] / 32) * 32
    L, R = prep(im_l, th, tw), prep(im_r, th, tw)
    ref = openstereo_model(a.ckpt).double()
    t0 = time.time()
    want = ref({"left": L, "right": R})["disp_pred"][0, 0]
    got, _ = LightStereoF(load_state(a.ckpt), Num("float")).run(L, R)
    err = (got - want).abs().max().item()
    print(f"{s}/{name} {tw}x{th}: OpenStereo vs functional (float64): max |diff| {err:.3e} px "
          f"({time.time() - t0:.1f} s); disparity range {want.min():.2f} … {want.max():.2f}")
    ref32 = openstereo_model(a.ckpt)
    want32 = ref32({"left": L.float(), "right": R.float()})["disp_pred"][0, 0].double()
    print(f"  OpenStereo float32 vs float64: max |diff| {(want32 - want).abs().max():.3e} px, "
          f"mean {(want32 - want).abs().mean():.3e}")
    if err > 1e-4:   # float32 itself differs by ~2e-4 px
        raise SystemExit("validate: FAIL")
    print("validate: ok")


# ── study ─────────────────────────────────────────────────────────────────────

def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    e = np.abs(pred - gt)[valid]
    return {"epe": float(e.mean()), "bad1": float((e > 1).mean() * 100), "bad2": float((e > 2).mean() * 100)}


def run_policy(sd, policy, L, R, h, w, stripe_max=7, table=None):
    n = Num(policy, table)
    m = LightStereoF(sd, n, stripe_max)
    disp, init = m.run(L, R)
    th = L.shape[2]
    return disp[th - h:, :w].numpy(), n.stats, m.macs


def cmd_study(a) -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    sd = load_state(a.ckpt)
    policies = a.policies.split(",")
    table = None
    if any(p.startswith("pow2") for p in policies):
        table = json.loads(formats_path(a.ckpt).read_text())["exponents"]
    rows = []
    sel = pairs()
    sel = sel[:: a.stride]
    if a.limit:
        sel = sel[: a.limit]
    t_start = time.time()
    for s, name, d in sel:
        im_l, im_r, gt, valid = load_pair(d)
        H0, W0 = im_l.shape[:2]
        scale = a.scale
        if a.fit:              # the demo's input: scaled down to fit H x W, padded (prepare.py)
            scale = min(1.0, a.fit[1] / W0, a.fit[0] / H0)
        if scale != 1.0:       # a lower input resolution: the pair resized, the disparity scaled back
            im_l, im_r = resize(im_l, scale), resize(im_r, scale)
        h, w = im_l.shape[:2]
        th, tw = (a.fit[0], a.fit[1]) if a.fit else (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
        L, R = prep(im_l, th, tw), prep(im_r, th, tw)
        row = {"set": s, "name": name, "size": [w, h], "valid": int(valid.sum()), "scale": scale}
        ref = None
        for p in policies:
            t0 = time.time()
            disp, st, _ = run_policy(sd, p, L, R, h, w, table=table)
            if scale != 1.0 and a.fit:   # as postprocess.py: PIL bilinear, / scale
                from PIL import Image
                disp = np.asarray(Image.fromarray(disp.astype(np.float32), mode="F").resize((W0, H0), Image.BILINEAR),
                                  dtype=np.float64) / scale
            elif scale != 1.0:
                disp = upscale(disp, H0, W0) / scale
            row[p] = metrics(disp, gt, valid)
            row[p]["sat_frac"] = st["sat"] / max(1, st["n"])
            row[p]["wsat"] = st["wsat"]
            row[p]["requant"] = st["requant"]
            row[p]["pc_convs"] = st.get("pc_convs", 0)
            row[p]["seconds"] = round(time.time() - t0, 1)
            if ref is None:
                ref = disp
            else:
                diff = np.abs(disp - ref)
                row[p]["vs_float_mean"] = float(diff.mean())
                row[p]["vs_float_gt1"] = float((diff > 1).mean() * 100)
        rows.append(row)
        print(f"{s:10s} {name:18s} {w}x{h} " + "  ".join(
            f"{p}: EPE {row[p]['epe']:.3f} bad2 {row[p]['bad2']:5.2f}" +
            (f" Δfloat {row[p]['vs_float_mean']:.3f}" if "vs_float_mean" in row[p] else "") for p in policies),
            flush=True)
    summary = {}
    for s in sorted({r["set"] for r in rows}) + ["all"]:
        rs = [r for r in rows if s in ("all", r["set"])]
        summary[s] = {p: {k: float(np.mean([r[p][k] for r in rs])) for k in rs[0][p] if k != "seconds"}
                      for p in policies}
    out = {"ckpt": a.ckpt, "policies": policies, "pairs": rows, "summary": summary,
           "seconds": round(time.time() - t_start, 1)}
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    path = STUDY_DIR / f"study_{a.ckpt}{'_' + a.tag if a.tag else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print()
    for s, v in summary.items():
        print(f"{s:10s} " + "  ".join(
            f"{p}: EPE {v[p]['epe']:.3f} bad1 {v[p]['bad1']:.2f} bad2 {v[p]['bad2']:.2f}"
            + (f" Δfloat {v[p]['vs_float_mean']:.3f} (>1px {v[p]['vs_float_gt1']:.2f} %)" if "vs_float_mean" in v[p] else "")
            + f" sat {v[p]['sat_frac']:.2e}" for p in policies))
    print(f"-> {path} ({out['seconds']} s)")


# ── calibrate ─────────────────────────────────────────────────────────────────

def formats_path(ckpt: str) -> Path:
    return STUDY_DIR / f"formats_{ckpt}.json"


def cmd_calibrate(a) -> None:
    """Every site's largest |value| over the MiddEval3 test pairs (no ground
    truth; disjoint from the study's pairs) -> its exponent with one bit of
    headroom: f = floor(log2(16383 / max)), within [0, 15]."""
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    sd = load_state(a.ckpt)
    root = DATA_DIR / "MiddEval3" / "testQ"
    maxes: dict[str, float] = {}
    t0 = time.time()
    for d in sorted(root.iterdir()):
        im_l, im_r = read_rgb(d / "im0.png"), read_rgb(d / "im1.png")
        th, tw = math.ceil(im_l.shape[0] / 32) * 32, math.ceil(im_l.shape[1] / 32) * 32
        n = Num("calib")
        LightStereoF(sd, n).run(prep(im_l, th, tw), prep(im_r, th, tw))
        for k, v in n.maxes.items():
            if k.endswith(":c"):   # per-channel maxima
                maxes[k] = v if k not in maxes else [max(a, b) for a, b in zip(maxes[k], v, strict=True)]
            else:
                maxes[k] = max(maxes.get(k, 0.0), v)
    table = {k: int(max(0, min(15, math.floor(math.log2(16383 / v))))) if v > 0 else 15
             for k, v in maxes.items() if not k.endswith(":c")}
    table.update({k: v for k, v in maxes.items() if k.endswith(":c")})
    out = {"ckpt": a.ckpt, "calibration": "MiddEval3 testQ (15 pairs)", "headroom_bits": 1,
           "exponents": table, "max_abs": maxes}
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    formats_path(a.ckpt).write_text(json.dumps(out, indent=1))
    if a.ckpt == "anything-s":
        write_formats(maxes, a.ckpt, out["calibration"])
        print(f"-> {FORMATS} (the demo's formats)")
    hist = {}
    for f in (v for k, v in table.items() if not k.endswith(":c")):
        hist[f] = hist.get(f, 0) + 1
    print(f"{sum(not k.endswith(':c') for k in table)} sites ({time.time() - t0:.0f} s); exponents: "
          + ", ".join(f"2^-{f}: {n}" for f, n in sorted(hist.items())))
    for k in ("input", "correlation", "regression", "backbone.out_conv.in"):
        print(f"  {k}: max {maxes.get(k, 0):.3f} -> 2^-{table.get(k)}")
    print(f"-> {formats_path(a.ckpt)}")


# ── costs ─────────────────────────────────────────────────────────────────────

def cmd_costs(a) -> None:
    torch.set_grad_enabled(False)
    h, w = a.height, a.width
    m = LightStereoF(load_state(a.ckpt), Num("float"))
    L = torch.zeros(1, 3, h, w, dtype=torch.float64)
    m.run(L, L)
    tot = sum(m.macs.values())
    print(f"LightStereo-S at {w}x{h}: {tot / 1e9:.2f} GMAC")
    for k, v in m.macs.items():
        print(f"  {k:12s} {v / 1e6:9.1f} MMAC  {100 * v / tot:5.1f} %")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    v = sub.add_parser("validate")
    v.add_argument("--ckpt", default="anything-s", choices=sorted(CKPTS))
    s = sub.add_parser("study")
    s.add_argument("--ckpt", default="anything-s", choices=sorted(CKPTS))
    s.add_argument("--policies", default="float,q88")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--stride", type=int, default=1, help="every Nth pair (a quick ablation)")
    s.add_argument("--tag", default="", help="suffix of the results file")
    s.add_argument("--scale", type=float, default=1.0, help="run at this fraction of the pairs' resolution")
    s.add_argument("--fit", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="the demo's input: each pair scaled down to fit H x W and padded (prepare.py)")
    s.add_argument("--threads", type=int, default=8)
    k = sub.add_parser("calibrate")
    k.add_argument("--ckpt", default="anything-s", choices=sorted(CKPTS))
    k.add_argument("--threads", type=int, default=8)
    c = sub.add_parser("costs")
    c.add_argument("--ckpt", default="anything-s", choices=sorted(CKPTS))
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--width", type=int, default=640)
    a = ap.parse_args(argv)
    {"fetch": cmd_fetch, "validate": cmd_validate, "study": cmd_study, "calibrate": cmd_calibrate,
     "costs": cmd_costs}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
