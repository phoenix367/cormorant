# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""onnx_qat.py — fine-tuning an ONNX CNN for the 8-bit configuration (doc/plans/INT8_PLAN.md
§9.8): ``Net``, onnx_i8.Graph's nodes in torch with the spec of an onnx_i8 policy straight
through, and ``distill``, distillation from the float model; the fine-tuned weights written
back into the ONNX graph (``export``) for onnx_i8 to judge bit for bit.

  weights   every Conv's / Gemm's int8 per output channel (int8_spec's least-error exponent of
            three, chosen at every step), straight through; the biases train in float (the
            kernels add them on the accumulator's fine grid)
  tensors   each at its format from onnx_i8.calibrate (static per tensor or per channel, blocks
            of 32 channels, int16 where the policy keeps 16 bits), straight-through rounding;
            the exponents fixed while training

Used by demo/object_detection/scripts/yolo_qat.py (loss "sigmoid": the head maps) and
cnn_i8_study.py qat (loss "softmax": the classifiers' logits)."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "chat", "scripts")))
import int8_fq as fq  # noqa: E402
import onnx_i8 as oi  # noqa: E402


class Net(torch.nn.Module):
    """onnx_i8.Graph's nodes in torch; quant: the formats' rounding and the int8 weights."""

    def __init__(self, g, fm, dev):
        super().__init__()
        self.g, self.fm = g, fm
        self.w = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        self.key = {}
        for k, n in enumerate(g.nodes):
            if n.op == "Conv":
                W = g.init[n.inputs[1]]
            elif n.op == "Gemm":
                W = g.init[n.inputs[1]] * n.attrs.get("alpha", 1.0)
                W = W if n.attrs.get("transB") else W.T            # [N, K]: output channels first
            else:
                continue
            self.key[id(n)] = f"c{k}"
            self.w[f"c{k}"] = torch.nn.Parameter(torch.tensor(W, dtype=torch.float32))
            b = (g.init[n.inputs[2]] * (n.attrs.get("beta", 1.0) if n.op == "Gemm" else 1.0)
                 if len(n.inputs) > 2 and n.inputs[2] else np.zeros(W.shape[0]))
            self.b[f"c{k}"] = torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.dev = dev
        self.to(dev)

    def fq(self, y, f):
        if f is None or f.kind == "o16":
            return y
        if f.kind == "b":                              # blocks of 32 channels at every pixel
            if y.dim() == 2:
                return fq.fq_blocks(y, f.f)
            return fq.fq_blocks(y.permute(0, 2, 3, 1), f.f).permute(0, 3, 1, 2)
        lo, hi = f.range()
        if f.kind == "c":                              # an exponent per channel
            e = torch.tensor(f.f, dtype=y.dtype, device=y.device).reshape((1, -1) + (1,) * (y.dim() - 2))
            return fq.fq_static(y, e, lo, hi)
        return fq.fq_static(y, f.f, lo, hi)

    @staticmethod
    def _act(y, act):
        if act is None:
            return y
        return F.relu(y) if act[0] == "relu" else torch.clamp(y, act[1], act[2])

    def forward(self, x, quant=True):
        g = self.g
        env = {g.inputs[0]: self.fq(x, self.fm[g.inputs[0]]) if quant else x}
        for n in g.nodes:
            a = n.attrs
            xs = [env.get(i) for i in n.inputs]
            if n.op in ("Conv", "Gemm"):
                k = self.key[id(n)]
                W = fq.wq_spec(self.w[k]) if quant else self.w[k]
                if n.op == "Gemm":
                    y = F.linear(xs[0], W, self.b[k])
                else:
                    pt, pl, pb, pr = oi.auto_pads(a, xs[0].shape[2], xs[0].shape[3], W.shape[2], W.shape[3],
                                                  *a.get("strides", [1, 1]), *a.get("dilations", [1, 1]))
                    sym = (pt, pl) == (pb, pr)
                    xi = xs[0] if sym else F.pad(xs[0], (pl, pr, pt, pb))
                    y = F.conv2d(xi, W, self.b[k], stride=tuple(a.get("strides", [1, 1])),
                                 padding=(pt, pl) if sym else 0, dilation=tuple(a.get("dilations", [1, 1])),
                                 groups=a.get("group", 1))
                y = self._act(y, n.act)
            elif n.op == "SiLU":
                y = F.silu(xs[0])
            elif n.op == "Add":
                y = self._act(xs[0] + xs[1], n.act)
            elif n.op == "Relu":
                y = F.relu(xs[0])
            elif n.op == "Clip":
                _, lo, hi = g._act_of(n)
                y = torch.clamp(xs[0], lo, hi)
            elif n.op == "Concat":
                y = torch.cat(xs, dim=a["axis"])
            elif n.op in ("MaxPool", "AveragePool"):
                kh, kw = a["kernel_shape"]
                pt, pl, pb, pr = oi.auto_pads(a, xs[0].shape[2], xs[0].shape[3], kh, kw, *a.get("strides", [1, 1]))
                assert (pt, pl) == (pb, pr)
                pool = F.max_pool2d if n.op == "MaxPool" else F.avg_pool2d
                kwargs = {} if n.op == "MaxPool" else dict(count_include_pad=bool(a.get("count_include_pad", 0)))
                y = pool(xs[0], (kh, kw), tuple(a.get("strides", [1, 1])), (pt, pl), **kwargs)
            elif n.op == "GlobalAveragePool":
                y = xs[0].mean(dim=(2, 3), keepdim=True)
            elif n.op == "Resize":
                y = F.interpolate(xs[0], scale_factor=2.0, mode="nearest")
            elif n.op in oi.VIEW:
                shape = oi.view([np.zeros(xs[0].shape)] + [g.init.get(i) for i in n.inputs[1:]], n, a).shape
                y = xs[0].reshape(shape)
            else:
                raise NotImplementedError(n.op)
            env[n.outputs[0]] = self.fq(y, self.fm.get(n.outputs[0])) if quant else y
        return [env[o] for o in g.outputs]


def distill(g, fm, batches, steps, lr, loss="softmax", warmup=50, log_every=50):
    """Distillation from the float model: ``batches()`` yields float32 [B, C, H, W] inputs; loss
    "softmax" (KL of the logits) or "sigmoid" (the mean squared difference of the sigmoids of
    every output).  Returns (the student, the loss per step)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, teacher = Net(g, fm, dev), Net(g, fm, dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t0, log = time.time(), []
    for step in range(steps):
        x = torch.tensor(next(batches), device=dev)
        with torch.no_grad():
            ref = teacher(x, quant=False)
        out = student(x, quant=True)
        if loss == "softmax":
            lv = sum(F.kl_div(F.log_softmax(o.flatten(1), -1), F.log_softmax(r.flatten(1), -1),
                              log_target=True, reduction="batchmean") for o, r in zip(out, ref, strict=True))
        else:
            lv = sum(F.mse_loss(torch.sigmoid(o), torch.sigmoid(r)) for o, r in zip(out, ref, strict=True))
        opt.zero_grad(set_to_none=True)
        lv.backward()
        opt.step()
        sched.step()
        log.append(lv.item())
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {np.mean(log[-log_every:]):.3e}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    return student, log


def export(student, src, out):
    """The student's weights (Conv / Gemm, float32) into a copy of the ONNX graph ``src``."""
    import onnx
    from onnx import numpy_helper
    g = student.g
    m = onnx.load(src)
    trained = {}
    for n in g.nodes:
        if id(n) not in student.key:
            continue
        k = student.key[id(n)]
        W = student.w[k].detach().cpu().numpy()
        if n.op == "Gemm":
            W = (W if n.attrs.get("transB") else W.T) / n.attrs.get("alpha", 1.0)
        trained[n.inputs[1]] = W
        if len(n.inputs) > 2 and n.inputs[2]:
            b = student.b[k].detach().cpu().numpy()
            trained[n.inputs[2]] = b / (n.attrs.get("beta", 1.0) if n.op == "Gemm" else 1.0)
    for t in m.graph.initializer:
        if t.name in trained:
            t.CopyFrom(numpy_helper.from_array(trained[t.name].astype(np.float32).reshape(tuple(t.dims)), t.name))
    onnx.save(m, out)
