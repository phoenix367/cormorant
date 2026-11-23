# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The 8-bit configuration's rounding in torch, straight through (doc/plans/INT8_PLAN.md §6,
inference-scheduler/src/int8_spec.py): the fine-tuning tools' fake quantization
(demo/chat/scripts/llm_qat.py, demo/object_detection/scripts/yolo_qat.py)."""

import torch
import torch.nn.functional as F


def rste(x):
    """round half to even, gradient 1."""
    return x + (torch.round(x) - x).detach()


def fq_static(x, f, lo, hi):
    """int8 static at exponent f (value = q * 2^-f), per-head f broadcast by the caller."""
    s = torch.exp2(-f.to(x.dtype)) if torch.is_tensor(f) else 2.0 ** -f
    return torch.clamp(rste(x / s), lo, hi) * s


def fq_blocks(x, f, dmax=7, block=32):
    """the producers' block rule (int8_spec.block_round_float) along the last axis: per block
    the largest shift d in [0, dmax] whose rounded values fit [-128, 127]; straight through."""
    sh = x.shape
    n = sh[-1]
    pad = (-n) % block
    xp = F.pad(x, (0, pad)) if pad else x
    B = xp.reshape(*xp.shape[:-1], -1, block)
    with torch.no_grad():
        Bd = B.detach()
        d = torch.zeros(B.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
        for dd in range(1, dmax + 1):
            r = torch.round(Bd * 2.0 ** (f + dd))
            fits = ((r.amin(-1, keepdim=True) >= -128) & (r.amax(-1, keepdim=True) <= 127))
            d = torch.where(fits, torch.full_like(d, dd), d)
        s = torch.exp2(-(f + d))
    q = torch.clamp(rste(B / s), -128, 127) * s
    q = q.reshape(xp.shape)
    return q[..., :n] if pad else q


def w_scale(Wd):
    """the spec's per-output-channel step 2^-f of a weight [O, ...] (output channels first):
    candidates f0 (the finest exponent at which max |w| rounds into 127), f0 + 1, f0 + 2; the
    least squared error (int8_spec.encode_weights).  Returns [O, 1, ...]."""
    W2 = Wd.reshape(Wd.shape[0], -1)
    amax = W2.abs().amax(1, keepdim=True).clamp_min(1e-30)
    f0 = torch.floor(torch.log2(127.0 / amax))
    f0 = torch.where(torch.round(amax * torch.exp2(f0)) > 127, f0 - 1, f0)
    f0 = torch.where(torch.round(amax * torch.exp2(f0 + 1)) <= 127, f0 + 1, f0)
    best_f, best_e = f0, None
    for k in range(3):
        f = f0 + k
        e = ((torch.clamp(torch.round(W2 * torch.exp2(f)), -127, 127) * torch.exp2(-f) - W2) ** 2).sum(1, keepdim=True)
        if best_e is None:
            best_e = e
        else:
            take = e < best_e
            best_f, best_e = torch.where(take, f, best_f), torch.where(take, e, best_e)
    return torch.exp2(-best_f).reshape((Wd.shape[0],) + (1,) * (Wd.dim() - 1))


def wq_spec(W):
    """W's int8 values per output channel (first axis), straight through inside [-127, 127]."""
    with torch.no_grad():
        s = w_scale(W.detach())
    return torch.clamp(rste(W / s), -127, 127) * s
