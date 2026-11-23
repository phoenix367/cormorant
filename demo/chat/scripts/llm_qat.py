#!/usr/bin/env python3
# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune a Llama checkpoint for the 8-bit configuration: distillation against the integer
specification (doc/plans/INT8_PLAN.md §9.5).

The student's forward follows inference-scheduler/src/int8_spec.py with the policy's own 8-bit
formats (the JSON `llm_study.py formats --base <policy>+i8ob` writes; its exponents table):
  linears   int8 weights per output channel (the spec's least-error exponent of three, chosen
            at every optimizer step), inputs in blocks of 32 within 8 shifts of the calibrated
            base (x / x2 / a: the int16 exponent - 8; the o input: pvb)
  attention q / K / V static per head after RoPE (q8 / k8 / vc8), P in blocks of 32 keys at base
            2^-6
  LM head   int8 weights (frozen: tied to the embedding); input blocked at the xf base or, with
            --head-a16, int16 at that base + 8 (the W8A16 head, modifier i8hw)
with straight-through rounding; the norms, the residual, RoPE and the softmax in float.  Only
the linears' weights train.  Loss: KL(float teacher || student) over random windows of a plain
training text (WikiText-2 train in INT8_PLAN), position 0 the float sink (<|im_start|>), as the
deployed stack.  Writes OUT/ as a checkpoint directory (the source's files, the trained weights
as float32 model.safetensors, qat_log.json) that llm_study.py --assets reads.

The 4 GB GTX 1050 of the dev PC holds SmolLM2-135M (student, teacher, AdamW): ~4 s a step of
4 x 256 tokens; 1000 steps ~70 min.  For SmolLM2-360M (doc/plans/INT8_PLAN.md §9.9):
--topk K keeps the teacher off the GPU while the student trains — it first computes every
training window's K most likely tokens (log-probabilities, on the CPU side; --topk-cache
keeps them in a file) and the evaluation windows' argmax, then leaves; the loss is the
exact KL over those K and one bucket for the rest of the vocabulary (renormalising over the
K instead sharpened the student and raised its perplexity, INT8_PLAN §9.9) —
--optim adafactor (factored second moments, no first) replaces AdamW's two full-size
moments (it lacks AdamW's momentum: on 360M the evaluation got worse), --optim adamw-cpu
keeps AdamW's state and master weights in host RAM (the step's gradients copied down, the
updated weights back), and --groups 2 trains every other layer in a step (half the
gradients held).  The repository's .venv-export has a CPU torch (~17 s a step); a CUDA torch for
the Pascal GPU is torch 2.14.0+cu126 (CUDA 13 dropped sm_61).

usage: llm_qat.py ASSETS FORMATS.json OUT_DIR --train-text FILE [--steps 1000] [--head-a16]
           [--topk 64 --optim adamw-cpu --groups 2 --seq 128 --accum 8]
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

from int8_fq import fq_blocks, fq_static, rste, w_scale, wq_spec

torch.set_num_threads(8)
ap = argparse.ArgumentParser()
ap.add_argument("assets")
ap.add_argument("formats")
ap.add_argument("out")
ap.add_argument("--train-text", required=True, help="plain training text (e.g. WikiText-2 train)")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--eval-windows", type=int, default=4)
ap.add_argument("--head-a16", action="store_true")
ap.add_argument("--topk", type=int, default=0, help="the teacher's top-K precomputed (0: the teacher live)")
ap.add_argument("--topk-cache", default=None, help="--topk: a file to keep the teacher's top-K in (reused)")
ap.add_argument("--optim", choices=["adamw", "adamw-cpu", "adafactor"], default="adamw")
ap.add_argument("--groups", type=int, default=1,
                help="train the layers in this many interleaved groups, one a step (gradient memory / groups)")
a = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1023
tok = AutoTokenizer.from_pretrained(a.assets)
SINK = tok.convert_tokens_to_ids("<|im_start|>")
fmt = json.load(open(a.formats))["exponents"]    # "cls@l" -> int or [ints] (llm_study.py formats)


def F8(cls, layer):
    return fmt[f"{cls}@{layer}"]


def windows(path, n):
    ids = tok(open(path).read())["input_ids"]
    return [torch.tensor([[SINK] + ids[i * T:(i + 1) * T]]).to(DEV) for i in range(min(n, len(ids) // T))]


test = windows(os.path.join(a.assets, "heldout_wikitext2_test.txt"), a.eval_windows)
train_ids = torch.tensor(tok(open(a.train_text).read())["input_ids"])


class QLinFn(torch.autograd.Function):
    """y = x · wq(W)^T with the per-channel step s given; the int8 weight recomputed in the
    backward pass (no stored copy); straight through inside [-127, 127]."""

    @staticmethod
    def forward(ctx, x, W, s):
        ctx.save_for_backward(x, W, s)
        return x @ (torch.clamp(torch.round(W / s), -127, 127) * s).t()

    @staticmethod
    def backward(ctx, gy):
        x, W, s = ctx.saved_tensors
        r = torch.round(W / s)
        gx = gy @ (torch.clamp(r, -127, 127) * s)
        gW = (gy.reshape(-1, gy.shape[-1]).t() @ x.reshape(-1, x.shape[-1])) * ((r >= -127) & (r <= 127))
        return gx, gW, None


STEP = [0]                # the optimizer step: the weights (and so their steps) change only there


class QLin(nn.Module):
    def __init__(self, lin, f_in):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.f_in = f_in
        self._s, self._at = None, -1

    def forward(self, x):
        if self._at != STEP[0] or not self.training:
            with torch.no_grad():
                self._s = w_scale(self.weight.detach())
            self._at = STEP[0]
        return QLinFn.apply(fq_blocks(x, self.f_in), self.weight, self._s)


class QHead(nn.Module):
    def __init__(self, emb, f_in, a16):
        super().__init__()
        self.emb, self.f_in, self.a16 = emb, f_in, a16

    def forward(self, x):
        if self.a16:
            s = 2.0 ** -(self.f_in + 8)
            x = torch.clamp(rste(x / s), -32768, 32767) * s
        else:
            x = fq_blocks(x, self.f_in)
        return F.linear(x, wq_spec(self.emb.weight))


LAYER = {}


def spec_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    li = module.layer_idx
    fq = torch.tensor(F8("q8", li), device=query.device, dtype=query.dtype).view(1, -1, 1, 1)
    fk = torch.tensor(F8("k8", li), device=query.device, dtype=query.dtype).view(1, -1, 1, 1)
    fv = torch.tensor(F8("vc8", li), device=query.device, dtype=query.dtype).view(1, -1, 1, 1)
    q = fq_static(query, fq, -128, 127)
    k = fq_static(key, fk, -127, 127)
    v = fq_static(value, fv, -127, 127)
    k = repeat_kv(k, module.num_key_value_groups)
    v = repeat_kv(v, module.num_key_value_groups)
    w = (q @ k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w.masked_fill(~attention_mask, float("-inf")) if attention_mask.dtype == torch.bool else w + attention_mask
    p = fq_blocks(torch.softmax(w.float(), -1), 6).to(query.dtype)
    return (p @ v).transpose(1, 2).contiguous(), p


AttentionInterface.register("spec8", spec_attention)
ALL_MASK_ATTENTION_FUNCTIONS.register("spec8", eager_mask)


def load(student):
    m = AutoModelForCausalLM.from_pretrained(a.assets, dtype=torch.float32)
    if student:
        for li, L in enumerate(m.model.layers):
            A, M = L.self_attn, L.mlp
            A.layer_idx = li
            for nm, ci in (("q_proj", "x"), ("k_proj", "x"), ("v_proj", "x"), ("o_proj", "pvb")):
                f = F8(ci, li) - 8 if ci == "x" else F8(ci, li)
                setattr(A, nm, QLin(getattr(A, nm), f))
            for nm, ci in (("gate_proj", "x2"), ("up_proj", "x2"), ("down_proj", "a")):
                setattr(M, nm, QLin(getattr(M, nm), F8(ci, li) - 8))
        m.lm_head = QHead(m.model.embed_tokens, F8("xf", m.config.num_hidden_layers) - 8, a.head_a16)
        m.config._attn_implementation = "spec8"
    return m.to(DEV)


teacher = load(False).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    sink = teacher(torch.tensor([[SINK]], device=DEV), use_cache=True).past_key_values
rng = random.Random(1)
starts = [[rng.randrange(len(train_ids) - a.seq) for _ in range(a.accum)] for _ in range(a.steps)]


def teach(x):
    return teacher(x, past_key_values=copy.deepcopy(sink), use_cache=True).logits[0]


ref_argmax = None
topk = None
if a.topk:                                          # the teacher's answers, then the GPU is the student's
    t1 = time.time()
    key = dict(assets=os.path.abspath(a.assets), k=a.topk, seq=a.seq, accum=a.accum, steps=a.steps,
               text=os.path.abspath(a.train_text), windows=a.eval_windows)
    cached = torch.load(a.topk_cache) if a.topk_cache and os.path.exists(a.topk_cache) else None
    if cached is not None and cached["key"] == key:
        ref_argmax, topk = [r.to(DEV) for r in cached["ref_argmax"]], cached["topk"]
    else:
        with torch.no_grad():
            ref_argmax = [teach(x[:, 1:])[:-1].argmax(-1) for x in test]
            topk = []
            for row in starts:
                r = []
                for i in row:      # the true log-probabilities of the K most likely (not renormalised)
                    v, ix = torch.topk(F.log_softmax(teach(train_ids[i:i + a.seq][None].to(DEV)).float(), -1),
                                       a.topk, -1)
                    r.append((v.cpu(), ix.int().cpu()))
                topk.append(r)
        if a.topk_cache:
            torch.save(dict(key=key, ref_argmax=[r.cpu() for r in ref_argmax], topk=topk), a.topk_cache)
    del teacher
    torch.cuda.empty_cache()
    print(f"teacher top-{a.topk}: {sum(len(r) for r in topk)} windows [{time.time() - t1:.0f} s]", flush=True)
student = load(True)


def evaluate():
    student.eval()
    nll, cnt, agree = 0.0, 0, 0
    with torch.no_grad():
        for k, x in enumerate(test):
            lg = student(x[:, 1:], past_key_values=copy.deepcopy(sink), use_cache=True).logits[0][:-1]
            ref = ref_argmax[k] if ref_argmax is not None else teach(x[:, 1:])[:-1].argmax(-1)
            nll += F.cross_entropy(lg, x[0, 2:], reduction="sum").item()
            cnt += lg.shape[0]
            agree += (lg.argmax(-1) == ref).sum().item()
    student.train()
    return math.exp(nll / cnt), agree / cnt


wparams = [m.weight for m in student.modules() if isinstance(m, QLin)]
wgroups = [[m.weight for m in layer.modules() if isinstance(m, QLin)]
           for layer in student.model.layers]                 # per layer, for --groups
for p in student.parameters():
    p.requires_grad_(False)
for p in wparams:
    p.requires_grad_(True)
cpu_of = None
if a.optim == "adafactor":
    from transformers.optimization import Adafactor
    opt = Adafactor(wparams, lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False,
                    beta1=None, weight_decay=0.0)
elif a.optim == "adamw-cpu":                        # AdamW's state and master weights in host RAM
    cpu_of = {id(p): torch.nn.Parameter(p.detach().cpu().clone()) for p in wparams}
    opt = torch.optim.AdamW(list(cpu_of.values()), lr=a.lr, weight_decay=0.0)
else:
    opt = torch.optim.AdamW(wparams, lr=a.lr, weight_decay=0.0)


def opt_step():
    if cpu_of is None:
        opt.step()
        return
    for p in wparams:                               # the step's gradients down, the updated weights back
        c = cpu_of[id(p)]
        c.grad = None if p.grad is None else p.grad.detach().cpu()
        p.grad = None
    opt.step()
    with torch.no_grad():
        for p in wparams:
            c = cpu_of[id(p)]
            if c.grad is not None:
                p.copy_(c.to(p.device, non_blocking=True))
                c.grad = None
log = dict(args=vars(a), evals=[])
t0 = time.time()
p0, g0 = evaluate()
log["evals"].append(dict(step=0, ppl=p0, agree=g0))
print(f"step 0 ppl {p0:.3f} agree {g0:.4f}", flush=True)
WARM = 50
for step in range(1, a.steps + 1):
    STEP[0] = step
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1.0, step / WARM) * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps))
    opt.zero_grad(set_to_none=True)
    for p in wparams:
        p.grad = None
    if a.groups > 1:                                # this step's layers: every groups-th from step % groups
        for li, ps in enumerate(wgroups):
            for p in ps:
                p.requires_grad_(li % a.groups == step % a.groups)
    tot = 0.0
    for j, i in enumerate(starts[step - 1]):
        x = train_ids[i:i + a.seq][None].to(DEV)
        sl = student(x, past_key_values=copy.deepcopy(sink), use_cache=True).logits[0]
        if topk is not None:
            # the exact KL of the coarse distribution: the teacher's top K and one bucket for the
            # rest (renormalising over the K would push the student's tail mass into them)
            lt, ix = (t.to(DEV) for t in topk[step - 1][j])
            ls = torch.gather(F.log_softmax(sl.float(), -1), 1, ix.long())
            pt = lt.exp()
            tail_t = (1.0 - pt.sum(-1)).clamp_min(1e-8)
            tail_s = (1.0 - ls.exp().sum(-1)).clamp_min(1e-8)
            loss = ((pt * (lt - ls)).sum(-1) + tail_t * (tail_t.log() - tail_s.log())).mean()
        else:
            with torch.no_grad():
                tl = teach(x)
            loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
        (loss / a.accum).backward()
        tot += loss.item() / a.accum
    opt_step()
    if step % 25 == 0:
        mem = f", GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB" if DEV.type == "cuda" else ""
        print(f"step {step} kl {tot:.4f} [{time.time() - t0:.0f} s{mem}]", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        p_, g_ = evaluate()
        log["evals"].append(dict(step=step, ppl=p_, agree=g_, kl=tot))
        print(f"step {step} ppl {p_:.3f} agree {g_:.4f} [{time.time() - t0:.0f} s]", flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump(log, open(os.path.join(a.out, "qat_log.json"), "w"), indent=1)

# the trained weights as a checkpoint directory (the linears' float weights; tied head unchanged)
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.assets):
    if f != "model.safetensors" and os.path.isfile(os.path.join(a.assets, f)):
        shutil.copy(os.path.join(a.assets, f), os.path.join(a.out, f))
out = AutoModelForCausalLM.from_pretrained(a.assets, dtype=torch.float32)
with torch.no_grad():
    for L, S in zip(out.model.layers, student.model.layers, strict=True):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(L.self_attn, nm).weight.copy_(getattr(S.self_attn, nm).weight.cpu())
        for nm in ("gate_proj", "up_proj", "down_proj"):
            getattr(L.mlp, nm).weight.copy_(getattr(S.mlp, nm).weight.cpu())
out.config.torch_dtype = "float32"                  # no bf16 re-rounding of the trained weights
out.save_pretrained(a.out, safe_serialization=True)
json.dump(log, open(os.path.join(a.out, "qat_log.json"), "w"), indent=1)
print("saved", a.out, flush=True)
