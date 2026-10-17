"""
Speech-to-text ops (Moonshine, doc/plans/ASR_PLAN.md §3; the frontend
src/moonshine.py).  The specification is demo/asr/scripts/moonshine_q.py: the
generated project computes what it computes, bit for bit.

Host ops (numpy reference + the C helpers of ASR_C):

  AsrFrames     pcm [N] (int16 at 2^-15, a host input), n (valid samples, i32)
                -> conv1's polyphase image [1][64][rows][w + 1]:
                img[c][r][j] = pcm[64 (r w + j) + c] (0 past n) — conv1 (127
                taps, stride 64; a zero 128th tap) is a 1 x 2 conv over it
  AsrStemNorm   c1 [1][d][rows][w] (conv1's output, per-channel exponents), n
                -> tanh (an int16 table at 2^-15 per input exponent), then the
                utterance's GroupNorm(1 group) over the valid frames — S1 / S2
                exact integer sums, mean = S1 / N, var = S2 / N - mean^2,
                y = ((raw - mean) / sqrt(var + eps 2^30)) gamma[c] + beta[c] —
                written (round half even, saturate) into conv2's folded input
  AsrStemGelu   c [1][C][rows][w] (a stem conv's output, bias included), n
                -> GELU (erf) from an int32 table at 2^-16 per input exponent,
                rounded half even to the output's exponents: conv3's folded
                input (int16), or (``residual``) the int32 residual [T][C] at
                2^-15 (transposed)
  AsrArgmax     logits [1][V] (int16 at one exponent) -> id [1] (i32 host):
                the first maximum

Row-ops nodes (the unit's jobs; src/rowops_nodes.py's conventions):

  AsrCrossPrep  k0 / v0 [T][D] (the cross K / V MatMuls of the encoder output)
                -> the decoder layer's cross caches (DMA states [C][D],
                group-major, the V cache interleaved by its group_kw): raw
                copies (the caches carry the projections' exponents)
  AsrCrossQ     q0 [1][D] (a decode step's cross q MatMul) -> the q.K^T x
                image [H][HD][1] (raw, one copy)
  AsrSwiGLU     u [R][2F] (fc1 without its bias) -> a [R][F]: the halves plus
                the bias rounded at the staging states' exponents (two COPYs
                with a bias, rowops_spec.bias_round: x -> ub, gate -> gb), then
                SiLU(gb) * ub (MULACT, rowops_spec.silu_mul)

A conv tensor's per-channel exponents are its ``chexp`` (numeric.py), else
its scalar ``exp``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np

from . import rowops_spec as rs
from .host_nodes import HostContext, _attrs, _c_double, _label, _resolve
from .llm_nodes import HostTable, LlmNode, _const_array, _i32, _llm_inputs
from .nodes import SchedulerError
from .rowops_nodes import (NO_WINDOW, RowOpsNode, _exps, _need_unit, _pblock, row_stride_bytes,
                           tables_tensor, vimage_jobs, zero_param_tensor)

GELU_BITS = 16                       # the stem GELU table's fixed point (int32 at 2^-16)
RES_BITS = 15                        # the int32 residual's exponent
PHASES = 64                          # conv1's stride: one polyphase image row per 64 samples


def _require(cond, node, msg):
    if not cond:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {msg}")


def chexp(t, F: int) -> np.ndarray:
    """Per-channel (axis 1) exponents of a 4-D conv tensor."""
    if t.chexp is not None:
        return np.asarray(t.chexp, np.int64)
    return np.full(int(t.shape[1]), F if t.exp is None else int(np.asarray(t.exp)), np.int64)


def _scalar_exp(t, F: int, node, what: str) -> int:
    e = t.exp_channels(F) if t.chexp is None else t.chexp
    _require(t.chexp is None and (np.asarray(e) == np.asarray(e).reshape(-1)[0]).all(), node,
             f"{what} '{t.onnx_name}': one exponent")
    return int(np.asarray(e).reshape(-1)[0])


# ------------------------------------------------------------------ #
# The tables (computed here, shipped as data: no libm on the board)     #
# ------------------------------------------------------------------ #

_ERF = np.vectorize(math.erf, otypes=[np.float64])
_TANH = np.vectorize(math.tanh, otypes=[np.float64])


def tanh_table(f: int) -> np.ndarray:
    """round_half_even(tanh(r 2^-f) 2^15) for every raw int16 r (index r + 32768),
    saturated — the tanh of conv1's output at 2^-15."""
    x = (np.arange(65536) - 32768) / 2.0 ** int(f)
    return np.clip(np.rint(_TANH(x) * 32768.0), -32768, 32767).astype(np.int64)


def gelu16_table(f: int) -> np.ndarray:
    """round_half_even(GELU(r 2^-f) 2^16) for every raw int16 r, saturated to int32
    (GELU erf in vectorop_act.act_values' operation order)."""
    x = (np.arange(65536) - 32768) / 2.0 ** int(f)
    v = np.rint(x * (0.5 * (1.0 + _ERF(x / math.sqrt(2.0)))) * 2.0 ** GELU_BITS)
    return np.clip(v, -2.0 ** 31, 2.0 ** 31 - 1).astype(np.int64)


def rne_shr(v, k) -> np.ndarray:
    """round half to even of v / 2^k per element (k >= 0; k < 0 shifts left)."""
    v = np.asarray(v, np.int64)
    k = np.broadcast_to(np.asarray(k, np.int64), v.shape)
    kk = np.maximum(k, 1)
    q = v >> kk
    r = v - (q << kk)
    half = np.int64(1) << (kk - 1)
    rd = q + ((r > half) | ((r == half) & ((q & 1) == 1)))
    return np.where(k <= 0, v << np.maximum(-k, 0), rd)


def stem_frames(n: int) -> Tuple[int, int, int]:
    """(t1, t2, T): the valid frames after conv1 (k 127, s 64), conv2 (k 7, s 3) and
    conv3 (k 3, s 2) of n samples (0 where the audio is too short)."""
    t1 = (n - 127) // 64 + 1 if n >= 127 else 0
    t2 = (t1 - 7) // 3 + 1 if t1 >= 7 else 0
    T = (t2 - 3) // 2 + 1 if t2 >= 3 else 0
    return max(t1, 0), max(t2, 0), max(T, 0)


def _fold_src(rows: int, width: int, step: int) -> np.ndarray:
    """[rows][width] source frame of each folded position: r * step + j."""
    return np.arange(rows)[:, None] * step + np.arange(width)[None, :]


# ------------------------------------------------------------------ #
# Host ops                                                             #
# ------------------------------------------------------------------ #

class AsrNode(LlmNode):
    helpers: ClassVar[Tuple[str, ...]] = ("llm", "asr")


@dataclass
class AsrFramesNode(AsrNode):
    """pcm [N], n -> conv1's polyphase image [1][64][rows][w + 1] (exponent 15)."""
    N:    int = 1
    rows: int = 1
    w:    int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        pcm, n = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        _require(len(y.shape) == 4 and int(y.shape[0]) == 1 and int(y.shape[1]) == PHASES, node,
                 f"output [1][{PHASES}][rows][w + 1]")
        rows, w = int(y.shape[2]), int(y.shape[3]) - 1
        sn = cls(onnx_node=node, inputs=[pcm, n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, N=pcm.numel, rows=rows, w=w)
        sn._want(pcm, "i16", "pcm")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        _require(_scalar_exp(pcm, ctx.frac_bits, node, "pcm") == 15
                 and _scalar_exp(y, ctx.frac_bits, node, "output") == 15, node, "pcm / image at 2^-15")
        return sn

    def describe(self):
        return f"{self.N} samples -> [{PHASES}][{self.rows}][{self.w + 1}] (conv1's polyphase image)"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"asr_frames({ins[0]}, {self.N}u, (unsigned){ins[1]}[0], {self.rows}u, {self.w}u, {out});"]

    def reference(self, ins, dtype):
        pcm = np.rint(np.asarray(ins[0], np.float64).reshape(-1) * 32768.0)
        n = min(max(_i32(ins[1]), 0), self.N)
        blk = _fold_src(self.rows, self.w + 1, self.w)            # [rows][w + 1] phase blocks
        idx = blk[None, :, :] * PHASES + np.arange(PHASES)[:, None, None]
        y = np.where(idx < n, pcm[np.minimum(idx, self.N - 1)], 0.0)
        return (y / 32768.0).reshape(self.output.shape)


@dataclass
class AsrStemNormNode(AsrNode):
    """c1 [1][d][rows][w], n -> tanh -> GroupNorm(1 group) -> conv2's folded input
    [1][d][rows2][width2] (frame of (r, j) = r * step2 + j; 0 past the valid frames)."""
    d:      int = 1
    rows:   int = 1
    w:      int = 1
    rows2:  int = 1
    width2: int = 1
    step2:  int = 1
    eps:    float = 1e-5
    fin:    Optional[np.ndarray] = field(default=None, repr=False)     # [d]
    fout:   Optional[np.ndarray] = field(default=None, repr=False)     # [d]
    gamma:  Optional[np.ndarray] = field(default=None, repr=False)
    beta:   Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        ins = _llm_inputs(node, tensors)
        _require(len(ins) == 4, node, "inputs: c1, n, gamma, beta")
        c1, n = ins[:2]
        y = _resolve(tensors, node.output[0], node)
        F = ctx.frac_bits
        a = _attrs(node)
        _require(len(c1.shape) == 4 and len(y.shape) == 4 and int(c1.shape[1]) == int(y.shape[1]), node,
                 "c1 [1][d][rows][w] -> [1][d][rows2][width2]")
        d = int(c1.shape[1])
        gamma = _const_array(ctx, tensors, node.input[2], node, "gamma").astype(np.float64).reshape(-1)
        beta = _const_array(ctx, tensors, node.input[3], node, "beta").astype(np.float64).reshape(-1)
        _require(gamma.size == d and beta.size == d, node, f"gamma / beta [{d}]")
        eps = a.get("axi_eps", b"1e-05")
        sn = cls(onnx_node=node, inputs=[c1, n], output=y, index=index, align_elems=align_elems, F=F,
                 d=d, rows=int(c1.shape[2]), w=int(c1.shape[3]), rows2=int(y.shape[2]),
                 width2=int(y.shape[3]), step2=int(a["step"]),
                 eps=float(eps.decode() if isinstance(eps, bytes) else eps),
                 fin=chexp(c1, F), fout=chexp(y, F), gamma=gamma, beta=beta)
        sn._want(c1, None, "c1")
        sn._want(n, "i32", "n")
        sn._want(y, None, "output")
        return sn

    def tables(self):
        return [HostTable(f"_asr_tanh_f{f}", "i16", tanh_table(f)) for f in sorted(set(int(v) for v in self.fin))]

    def c_file_consts(self, dtype):  # noqa: ARG002
        fs = sorted(set(int(v) for v in self.fin))
        tabs = ", ".join(f"_asr_tanh_f{f}" for f in fs)
        sel = ", ".join(str(fs.index(int(v))) for v in self.fin)
        gl = ", ".join(_c_double(v) for v in self.gamma)
        bl = ", ".join(_c_double(v) for v in self.beta)
        fo = ", ".join(str(int(v)) for v in self.fout)
        o = self.output.c_name
        del tabs
        return [f"static const unsigned char _asr_ts_{o}[{self.d}] = {{{sel}}};",
                f"static const double _asr_g_{o}[{self.d}] = {{{gl}}};",
                f"static const double _asr_b_{o}[{self.d}] = {{{bl}}};",
                f"static const signed char _asr_no_{o}[{self.d}] = {{{fo}}};"]

    def describe(self):
        return (f"tanh + GroupNorm of [{self.d}][{self.rows}x{self.w}] -> conv2's folded input "
                f"[{self.rows2}][{self.width2}] (step {self.step2}), tanh tables for exponents "
                f"{sorted(set(int(v) for v in self.fin))}")

    def c_call(self, ins, out, scratch, direct, dtype):
        o = self.output.c_name
        fs = sorted(set(int(v) for v in self.fin))
        return ["{",
                f"    const int16_t *const _tt[{len(fs)}] = {{{', '.join(f'_asr_tanh_f{f}' for f in fs)}}};",
                "    asr_norm_t _a;",
                f"    _a.c1 = {ins[0]}; _a.y = {out}; _a.tab = _tt; _a.tsel = _asr_ts_{o};",
                f"    _a.g = _asr_g_{o}; _a.b = _asr_b_{o};",
                f"    _a.fo = _asr_no_{o}; _a.d = {self.d}u; _a.rows = {self.rows}u; _a.w = {self.w}u;",
                f"    _a.rows2 = {self.rows2}u; _a.width2 = {self.width2}u; _a.step2 = {self.step2}u;",
                f"    _a.n = (unsigned){ins[1]}[0]; _a.eps = {_c_double(self.eps)};",
                "    asr_stem_norm(&_a);",
                "}"]

    def reference(self, ins, dtype):
        d = self.d
        c1 = np.asarray(ins[0], np.float64).reshape(d, self.rows * self.w)
        t1 = min(stem_frames(max(_i32(ins[1]), 0))[0], self.rows * self.w)
        raw = np.rint(c1[:, :t1] * np.power(2.0, self.fin.astype(np.float64))[:, None]).astype(np.int64)
        t = np.empty_like(raw)
        for f in np.unique(self.fin):
            cc = self.fin == f
            t[cc] = tanh_table(int(f))[raw[cc] + 32768]
        y = np.zeros((d, self.rows2, self.width2))
        if t1:
            N = t.size
            S1, S2 = int(t.sum()), int((t * t).sum())
            mean = float(S1) / N
            var = float(S2) / N - mean * mean
            den = math.sqrt(var + self.eps * 1073741824.0)
            v = ((t.astype(np.float64) - mean) / den) * self.gamma[:, None] + self.beta[:, None]
            q = np.clip(np.rint(v * np.power(2.0, self.fout.astype(np.float64))[:, None]), -32768, 32767)
            src = _fold_src(self.rows2, self.width2, self.step2)
            ok = src < t1
            y = np.where(ok[None], q[:, np.minimum(src, t1 - 1)], 0.0)
        return (y / np.power(2.0, self.fout.astype(np.float64))[:, None, None]).reshape(self.output.shape)


@dataclass
class AsrStemGeluNode(AsrNode):
    """c [1][C][rows][w] (a stem conv's output), n -> GELU: conv3's folded input
    [1][C][rows2][width2] (int16, per-channel exponents) or ``residual`` the int32
    residual [T][C] at 2^-15 (frames past the valid ones 0)."""
    C:        int = 1
    rows:     int = 1
    w:        int = 1
    stage:    int = 2                     # 2: conv2's output (t2 frames), 3: conv3's (T)
    residual: bool = False
    rows2:    int = 1
    width2:   int = 1
    step2:    int = 1
    T:        int = 1
    fin:      Optional[np.ndarray] = field(default=None, repr=False)
    fout:     Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        c, n = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        F = ctx.frac_bits
        a = _attrs(node)
        _require(len(c.shape) == 4, node, "c [1][C][rows][w]")
        C = int(c.shape[1])
        residual = int(a.get("residual", 0)) != 0
        sn = cls(onnx_node=node, inputs=[c, n], output=y, index=index, align_elems=align_elems, F=F, C=C,
                 rows=int(c.shape[2]), w=int(c.shape[3]), stage=int(a["stage"]), residual=residual,
                 fin=chexp(c, F))
        sn._want(c, None, "c")
        sn._want(n, "i32", "n")
        if residual:
            sn._want(y, "dma:i32", "output")
            _require(len(y.shape) == 2 and int(y.shape[1]) == C, node, f"the residual [T][{C}]")
            _require((y.exp_channels(F) == RES_BITS).all(), node, f"the residual at 2^-{RES_BITS}")
            sn.T = int(y.shape[0])
        else:
            sn._want(y, None, "output")
            _require(len(y.shape) == 4 and int(y.shape[1]) == C, node, "[1][C][rows2][width2]")
            sn.rows2, sn.width2, sn.step2 = int(y.shape[2]), int(y.shape[3]), int(a["step"])
            sn.fout = chexp(y, F)
            _require((sn.fout <= GELU_BITS + 15).all(), node, "output exponents")
        _require(sn.stage in (2, 3), node, "stage 2 or 3")
        return sn

    def tables(self):
        return [HostTable(f"_asr_gelu16_f{f}", "i32", gelu16_table(f))
                for f in sorted(set(int(v) for v in self.fin))]

    def c_file_consts(self, dtype):  # noqa: ARG002
        fs = sorted(set(int(v) for v in self.fin))
        o = self.output.c_name
        out = [f"static const unsigned char _asr_gs_{o}[{self.C}] = "
               f"{{{', '.join(str(fs.index(int(v))) for v in self.fin)}}};"]
        if not self.residual:
            out.append(f"static const signed char _asr_gk_{o}[{self.C}] = "
                       f"{{{', '.join(str(GELU_BITS - int(v)) for v in self.fout)}}};")
        return out

    def _frames(self, n: int) -> int:
        t = stem_frames(max(n, 0))[1 if self.stage == 2 else 2]
        return min(t, self.rows * self.w)

    def describe(self):
        to = (f"the int32 residual [{self.T}][{self.C}]" if self.residual else
              f"conv3's folded input [{self.rows2}][{self.width2}] (step {self.step2})")
        return (f"GELU of [{self.C}][{self.rows}x{self.w}] -> {to}, tables for exponents "
                f"{sorted(set(int(v) for v in self.fin))}")

    def c_call(self, ins, out, scratch, direct, dtype):
        o = self.output.c_name
        fs = sorted(set(int(v) for v in self.fin))
        tabs = f"    const int32_t *const _gt[{len(fs)}] = {{{', '.join(f'_asr_gelu16_f{f}' for f in fs)}}};"
        common = (f"{ins[0]}, _gt, _asr_gs_{o}, {self.C}u, {self.rows}u, {self.w}u, "
                  f"asr_frames_at({self.stage}u, (unsigned){ins[1]}[0], {self.rows * self.w}u)")
        if self.residual:
            return ["{", tabs, f"    asr_gelu_res({common}, {self.T}u, (int32_t *){out});", "}"]
        return ["{", tabs,
                f"    asr_gelu_fold({common}, _asr_gk_{o}, {self.rows2}u, {self.width2}u, {self.step2}u, {out});",
                "}"]

    def reference(self, ins, dtype):
        c = np.asarray(ins[0], np.float64).reshape(self.C, self.rows * self.w)
        tv = self._frames(_i32(ins[1]))
        raw = np.rint(c[:, :tv] * np.power(2.0, self.fin.astype(np.float64))[:, None]).astype(np.int64)
        t16 = np.empty_like(raw)
        for f in np.unique(self.fin):
            cc = self.fin == f
            t16[cc] = gelu16_table(int(f))[raw[cc] + 32768]
        if self.residual:
            y = np.zeros((self.T, self.C))
            y[:tv] = (rne_shr(t16, GELU_BITS - RES_BITS) / 2.0 ** RES_BITS).T
            return y.reshape(self.output.shape)
        q = np.clip(rne_shr(t16, (GELU_BITS - self.fout)[:, None]), -32768, 32767)
        out = np.zeros((self.C, self.rows2, self.width2))
        if tv:
            src = _fold_src(self.rows2, self.width2, self.step2)
            out = np.where((src < tv)[None], q[:, np.minimum(src, tv - 1)], 0.0)
        return (out / np.power(2.0, self.fout.astype(np.float64))[:, None, None]).reshape(self.output.shape)


@dataclass
class AsrFrameCountNode(AsrNode):
    """n (valid samples, i32) -> the valid encoder frames (i32), at most ``cap``."""
    cap: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        (n,) = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        sn = cls(onnx_node=node, inputs=[n], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, cap=int(_attrs(node)["cap"]))
        sn._want(n, "i32", "n")
        sn._want(y, "i32", "output")
        return sn

    def describe(self):
        return f"the encoder frames of n samples (at most {self.cap})"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"{out}[0] = (int32_t)asr_frames_at(3u, (unsigned){ins[0]}[0], {self.cap}u);"]

    def reference(self, ins, dtype):
        return np.array([float(min(stem_frames(max(_i32(ins[0]), 0))[2], self.cap))])


@dataclass
class AsrArgmaxNode(AsrNode):
    """logits [1][V] (int16, one exponent) -> id [1] (i32 host): the first maximum."""
    V: int = 1

    @classmethod
    def from_onnx_node(cls, node, tensors, index, align_elems, ctx: HostContext):
        (lg,) = _llm_inputs(node, tensors)
        y = _resolve(tensors, node.output[0], node)
        sn = cls(onnx_node=node, inputs=[lg], output=y, index=index, align_elems=align_elems,
                 F=ctx.frac_bits, V=lg.numel)
        sn._want(lg, None, "logits")
        sn._want(y, "i32", "output")
        _scalar_exp(lg, ctx.frac_bits, node, "logits")
        _require(y.numel == 1, node, "output [1]")
        return sn

    def describe(self):
        return f"argmax over {self.V} logits"

    def c_call(self, ins, out, scratch, direct, dtype):
        return [f"{out}[0] = (int32_t)asr_argmax({ins[0]}, {self.V}u);"]

    def reference(self, ins, dtype):
        return np.array([float(np.argmax(np.asarray(ins[0], np.float64).reshape(-1)))])


# ------------------------------------------------------------------ #
# Row-ops nodes                                                        #
# ------------------------------------------------------------------ #

def _jobs_call(node, jobs, layouts) -> str:      # noqa: ARG001
    cn = lambda t: "NULL" if t is None else t.c_name   # noqa: E731
    lines = ["    {"]
    for i, j in enumerate(jobs):
        if i:
            lines.append("        kernel_wait(KERNEL_ROWOPS);")
        lines.append(f"        /* {j[0]} */")
        lines.append(node.call(cn(j[1]), j[2], cn(j[3]), j[4], cn(j[5]), j[6], cn(j[7]), j[8], j[9], j[10],
                               j[11], j[12], j[13], j[14], j[15], j[16], j[17], 0, j[18], 0, j[19], j[20],
                               indent="        "))
    lines.append("    }")
    return "\n".join(lines)


def _jobs_calls(jobs) -> list:
    from .perf_calls import KernelCall, merge
    return merge([KernelCall.of("RowOpsKernel", op=j[8], n=j[9], rows=j[10], blocks=j[11], a_s1=j[12],
                                a_s2=j[13], b_s1=j[14], b_s2=j[15], c_s1=j[16], c_s2=j[17], aux=j[18],
                                win_lo=j[19], win_hi=j[20]) for j in jobs])


@dataclass
class AsrCrossPrepRoNode(RowOpsNode):
    """k0 / v0 [T][D] -> the cross caches (the output is the K cache, a state
    written in place; the V cache too): one COPY with the heads as blocks into
    the K cache, the V cache likewise (group_kw 1) or per head the TRANSPOSEs of
    vimage_jobs (group_kw K, T % 16K == 0)."""
    T:  int = 1
    H:  int = 1
    HD: int = 16
    C:  int = 16
    K:  int = 1

    @property
    def D(self) -> int:
        return self.H * self.HD

    @property
    def k0(self):
        return self.inputs[0]

    @property
    def v0(self):
        return self.inputs[1]

    @property
    def cv(self):
        return self.inputs[2]

    @property
    def pz(self):
        return self.inputs[3]

    def state_writes(self):
        return [self.output, self.cv]

    def state_updates(self):
        return self.state_writes()

    def _jobs(self, layouts):
        nd, al = self.onnx_node, self.align_elems
        sk = row_stride_bytes(self.k0, layouts, self.D, 2, nd, al)
        sv = row_stride_bytes(self.v0, layouts, self.D, 2, nd, al)
        T, H, HD, C, K = self.T, self.H, self.HD, self.C, self.K
        cp = rs.opword(rs.OP_COPY, p_bcast=True)
        jobs = [("k0 -> the K cache rows (the heads as blocks)", self.k0, 0, None, 0, self.output, 0, self.pz, cp,
                 HD, T, H, sk, HD * 2, 0, 0, HD * 2, C * HD * 2, 0, 0, NO_WINDOW)]
        if K == 1:
            jobs.append(("v0 -> the V cache rows (the heads as blocks)", self.v0, 0, None, 0, self.cv, 0, self.pz,
                         cp, HD, T, H, sv, HD * 2, 0, 0, HD * 2, C * HD * 2, 0, 0, NO_WINDOW))
        else:
            xp = rs.opword(rs.OP_XPOSE)
            for h in range(H):
                for a_off, c_off, rows, blocks, a_s1, a_s2, b_s1, b_s2, c_s1, c_s2, R in \
                        vimage_jobs(0, T, HD, K, sv):
                    jobs.append((f"the V image of head {h}", self.v0, h * HD * 2 + a_off, None, 0, self.cv,
                                 h * C * HD * 2 + c_off, None, xp, HD, rows, blocks, a_s1, a_s2, b_s1, b_s2,
                                 c_s1, c_s2, R, 0, NO_WINDOW))
        return jobs

    def kernel_calls(self, layouts: dict) -> list:
        return _jobs_calls(self._jobs(layouts))

    def chain_jobs(self, layouts: dict) -> int:
        return len(self._jobs(layouts))

    def describe(self) -> str:
        return (f"T={self.T} heads {self.H} head_dim {self.HD} -> the cross caches ({self.C} rows, V "
                f"interleave {self.K}) on RowOpsKernel")

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] AsrCrossPrep({self.k0.onnx_name}, {self.v0.onnx_name}) -> "
                f"{self.output.onnx_name} + {self.cv.onnx_name}  {self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        return _jobs_call(self, self._jobs(layouts), layouts)

    def reference(self, ins, dtype):  # noqa: ARG002
        """The V cache's rows [0, T) in place; the K cache returned whole — its rows
        past T zero (the decoder never reads them: keys = roundup(frames, Q) <= T)."""
        T, D = self.T, self.D
        a = {t.onnx_name: x for t, x in zip(self.inputs, ins, strict=True)}
        cv = a[self.cv.onnx_name]
        cv[:T] = np.asarray(a[self.v0.onnx_name], np.float64).reshape(T, D)
        ck = np.zeros((self.C, D))
        ck[:T] = np.asarray(a[self.k0.onnx_name], np.float64).reshape(T, D)
        return ck


def from_asr_cross_prep(node, tensors, index, align_elems, ctx: HostContext):
    """AsrCrossPrep(k0, v0, cache_v) -> cache_k (both caches DMA states written in place)."""
    _need_unit(node, "AsrCrossPrep")
    F = ctx.frac_bits
    k0, v0, cv = _llm_inputs(node, tensors)
    ck = _resolve(tensors, node.output[0], node)
    a = _attrs(node)
    H, HD = int(a["num_heads"]), int(a["head_dim"])
    D = H * HD
    T = k0.numel // D
    C, K = int(ck.shape[0]), int(cv.group_kw)
    _require(k0.numel == v0.numel == T * D and k0.shape[-1] == D, node, f"k0 / v0 [T][{D}]")
    for t, w in ((k0, "k0"), (v0, "v0")):
        _require(t.dma is None and t.host is None and not t.is_state, node, f"{w}: an int16 DMA tensor")
    for t, w in ((ck, "cache_k"), (cv, "cache_v")):
        _require(t.is_state and not t.is_host and t.group_layout == (H, HD) and int(t.shape[0]) == C, node,
                 f"{w} must be a DMA state stored group-major [{H}][{C}][{HD}]")
    _require(ck.group_kw == 1, node, "the K cache rows must not be interleaved")
    _require(T <= C and T % 8 == 0 and (K == 1 or T % (16 * K) == 0), node,
             f"rows {T}: <= {C}, % 8 and % the V image's {16 * K}-row blocks")
    _require(np.array_equal(_exps(k0, F), _exps(ck, F)) and np.array_equal(_exps(v0, F), _exps(cv, F)), node,
             "the caches must carry the projections' exponents (raw copies)")
    return AsrCrossPrepRoNode(onnx_node=node, inputs=[k0, v0, cv, zero_param_tensor(tensors),
                                                      tables_tensor(tensors)],
                              output=ck, index=index, align_elems=align_elems, F=F, T=T, H=H, HD=HD, C=C, K=K)


@dataclass
class AsrCrossQRoNode(RowOpsNode):
    """q0 [1][D] -> the q.K^T x image [H][HD][1] (raw): one COPY."""
    D: int = 16

    def kernel_calls(self, layouts: dict) -> list:  # noqa: ARG002
        from .perf_calls import KernelCall
        return [KernelCall.of("RowOpsKernel", op=rs.opword(rs.OP_COPY, p_bcast=True), n=self.D, rows=1,
                              blocks=1, a_s1=self.D * 2, c_s1=self.D * 2, win_hi=NO_WINDOW)]

    def describe(self) -> str:
        return f"{self.D} values -> the q image on RowOpsKernel (one COPY)"

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] AsrCrossQ({self.inputs[0].onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:  # noqa: ARG002
        return self.call(self.inputs[0].c_name, 0, "NULL", 0, self.output.c_name, 0, self.inputs[1].c_name,
                         rs.opword(rs.OP_COPY, p_bcast=True), self.D, 1, 1, self.D * 2, 0, 0, 0,
                         self.D * 2, 0, 0, 0, 0, 0, NO_WINDOW)

    def reference(self, ins, dtype):  # noqa: ARG002
        q0 = self.inputs[0]
        raw = np.rint(np.asarray(ins[0], np.float64).reshape(-1) * np.power(2.0, _exps(q0, self.F)))
        return raw.reshape(self.output.shape)


def from_asr_cross_q(node, tensors, index, align_elems, ctx: HostContext):
    _need_unit(node, "AsrCrossQ")
    (q0,) = _llm_inputs(node, tensors)
    y = _resolve(tensors, node.output[0], node)
    D = q0.numel
    _require(q0.dma is None and q0.host is None and y.numel == D and len(y.shape) == 3 and int(y.shape[2]) == 1,
             node, f"q0 [1][{D}] -> [H][HD][1]")
    _require(not y.exp_channels(ctx.frac_bits).any(), node, "the q image holds raw integers (exponent 0)")
    return AsrCrossQRoNode(onnx_node=node, inputs=[q0, zero_param_tensor(tensors), tables_tensor(tensors)],
                           output=y, index=index, align_elems=align_elems, F=ctx.frac_bits, D=D)


@dataclass
class AsrSwiGLURoNode(RowOpsNode):
    """u [R][2F] -> a [R][F]: COPY with the bias of the x half -> ub, of the gate half
    -> gb (the staging states), MULACT SiLU(gb) * ub -> a."""
    R:  int = 1
    Fh: int = 1
    bias: Optional[np.ndarray] = field(default=None, repr=False)
    bc: Tuple[bool, bool, bool] = (False, False, False)

    @property
    def u(self):
        return self.inputs[0]

    @property
    def ub(self):
        return self.inputs[1]

    @property
    def gb(self):
        return self.inputs[2]

    def state_writes(self):
        return [self.ub, self.gb]

    def state_updates(self):
        return self.state_writes()

    def _jobs(self, layouts):
        nd, al = self.onnx_node, self.align_elems
        Fh = self.Fh
        su = row_stride_bytes(self.u, layouts, 2 * Fh, 2, nd, al)
        pu, pg, pm = self.inputs[3], self.inputs[4], self.inputs[5]
        mul = rs.opword(rs.OP_MULACT, rs.K_I16, rs.K_I16, rs.K_I16, b_en=True, p_bcast=self.bc[2])
        return [("x half + bias -> ub", self.u, 0, None, 0, self.ub, 0, pu,
                 rs.opword(rs.OP_COPY, p_bcast=self.bc[0]), Fh, self.R, 1, su, 0, 0, 0, Fh * 2, 0, 0, 0, NO_WINDOW),
                ("gate half + bias -> gb", self.u, Fh * 2, None, 0, self.gb, 0, pg,
                 rs.opword(rs.OP_COPY, p_bcast=self.bc[1]), Fh, self.R, 1, su, 0, 0, 0, Fh * 2, 0, 0, 0, NO_WINDOW),
                ("SiLU(gb) * ub -> a", self.gb, 0, self.ub, 0, self.output, 0, pm, mul, Fh, self.R, 1,
                 Fh * 2, 0, Fh * 2, 0, row_stride_bytes(self.output, layouts, Fh, 2, nd, al), 0, 0, 0, NO_WINDOW)]

    def kernel_calls(self, layouts: dict) -> list:
        return _jobs_calls(self._jobs(layouts))

    def chain_jobs(self, layouts: dict) -> int:
        return 3

    def describe(self) -> str:
        return f"rows={self.R} F={self.Fh} on RowOpsKernel (two COPYs with the fc1 bias, MULACT)"

    def emit_comment(self) -> str:
        return (f"    /* [{self.index}] AsrSwiGLU({self.u.onnx_name}) -> {self.output.onnx_name}  "
                f"{self.describe()} */")

    def emit_call(self, layouts: dict) -> str:
        return _jobs_call(self, self._jobs(layouts), layouts)

    def reference(self, ins, dtype):  # noqa: ARG002
        F, Fh, R = self.F, self.Fh, self.R
        fu = _exps(self.u, F)
        raw = np.rint(np.asarray(ins[0], np.float64).reshape(R, 2 * Fh) * np.power(2.0, fu)[None, :]).astype(np.int64)
        fub, fgb, fa = _exps(self.ub, F), _exps(self.gb, F), _exps(self.output, F)
        ub = rs.bias_round(raw[:, :Fh], fu[:Fh], self.bias[:Fh], fub)
        gb = rs.bias_round(raw[:, Fh:], fu[Fh:], self.bias[Fh:], fgb)
        a = {t.onnx_name: x for t, x in zip(self.inputs, ins, strict=False)}
        a[self.ub.onnx_name][...] = (ub / np.power(2.0, fub)[None, :]).reshape(a[self.ub.onnx_name].shape)
        a[self.gb.onnx_name][...] = (gb / np.power(2.0, fgb)[None, :]).reshape(a[self.gb.onnx_name].shape)
        y = rs.silu_mul(gb, fgb, ub, fub, fa)
        return (y / np.power(2.0, fa)[None, :]).reshape(self.output.shape)


def from_asr_swiglu(node, tensors, index, align_elems, ctx: HostContext):
    """AsrSwiGLU(u, bias, ub, gb) -> a (ub / gb: int16 DMA staging states)."""
    _need_unit(node, "AsrSwiGLU")
    F = ctx.frac_bits
    ins = _llm_inputs(node, tensors)
    _require(len(ins) == 4, node, "inputs: u, bias, ub, gb")
    u, _b, ub, gb = ins
    y = _resolve(tensors, node.output[0], node)
    Fh = int(y.shape[-1])
    R = y.numel // Fh
    bias = _const_array(ctx, tensors, node.input[1], node, "bias").astype(np.float64).reshape(-1)
    _require(u.numel == 2 * R * Fh and int(u.shape[-1]) == 2 * Fh and bias.size == 2 * Fh, node,
             f"u [R][{2 * Fh}], bias [{2 * Fh}]")
    _require(u.dma is None and u.host is None and y.dma is None and y.host is None, node, "int16 DMA tensors")
    for t, w in ((ub, "ub"), (gb, "gb")):
        _require(t.is_state and not t.is_host and t.dma is None and list(t.shape) == [R, Fh], node,
                 f"{w}: an int16 DMA state [{R}][{Fh}]")
    _require(Fh % align_elems == 0, node, f"F {Fh} % {align_elems}")
    fu, fub, fgb, fa = _exps(u, F), _exps(ub, F), _exps(gb, F), _exps(y, F)
    _require((fgb <= rs.G_BITS).all(), node, f"gate exponents above {rs.G_BITS}")
    sh_c = rs.SILU_BITS + fub - fa
    _require((sh_c >= 0).all() and (sh_c <= 63).all(), node, "x / output exponents beyond the shifts")
    try:
        bq_u, sa_u, sc_u = rs.bias_params(bias[:Fh], fu[:Fh], fub)
        bq_g, sa_g, sc_g = rs.bias_params(bias[Fh:], fu[Fh:], fgb)
    except ValueError as e:
        raise SchedulerError(f"{node.op_type} node '{_label(node)}': {e}") from None
    pu, bu = _pblock(tensors, f"{y.onnx_name}__rowops_pu", 0, bq_u, sa_u, 0, sc_u, Fh)
    pg, bg = _pblock(tensors, f"{y.onnx_name}__rowops_pg", 0, bq_g, sa_g, 0, sc_g, Fh)
    pm, bm = _pblock(tensors, f"{y.onnx_name}__rowops_pm", 0, 0, rs.G_BITS - fgb, 0, sh_c, Fh)
    return AsrSwiGLURoNode(onnx_node=node, inputs=[u, ub, gb, pu, pg, pm, tables_tensor(tensors)], output=y,
                           index=index, align_elems=align_elems, F=F, R=R, Fh=Fh, bias=bias, bc=(bu, bg, bm))


ASR_OP_FACTORIES = {
    "AsrFrames":    AsrFramesNode.from_onnx_node,
    "AsrStemNorm":  AsrStemNormNode.from_onnx_node,
    "AsrStemGelu":  AsrStemGeluNode.from_onnx_node,
    "AsrArgmax":    AsrArgmaxNode.from_onnx_node,
    "AsrFrameCount": AsrFrameCountNode.from_onnx_node,
    "AsrCrossPrep": from_asr_cross_prep,
    "AsrCrossQ":    from_asr_cross_q,
    "AsrSwiGLU":    from_asr_swiglu,
}


# ------------------------------------------------------------------ #
# C helpers                                                            #
# ------------------------------------------------------------------ #

ASR_C = r"""
/* ---- speech-to-text host ops (src/asr_nodes.py; doc/plans/ASR_PLAN.md) ---- */

/* conv1's polyphase image: img[c][r][j] = pcm[64 (r w + j) + c], 0 past n. */
static void asr_frames(const int16_t *pcm, unsigned N, unsigned n, unsigned rows, unsigned w, Data_t *img)
{
    const unsigned W = w + 1u;
    unsigned       c, r, j;
    if (n > N) n = N;
    for (c = 0u; c < 64u; c++)
        for (r = 0u; r < rows; r++) {
            Data_t *o = img + ((size_t)c * rows + r) * W;
            for (j = 0u; j < W; j++) {
                const size_t i = ((size_t)r * w + j) * 64u + c;
                o[j] = i < n ? (Data_t)pcm[i] : (Data_t)0;
            }
        }
}

/* The valid frames of a stem stage (1: conv1, 2: conv2, 3: conv3) of n samples,
 * at most cap. */
static unsigned asr_frames_at(unsigned stage, unsigned n, unsigned cap)
{
    unsigned t = n >= 127u ? (n - 127u) / 64u + 1u : 0u;
    if (stage >= 2u) t = t >= 7u ? (t - 7u) / 3u + 1u : 0u;
    if (stage >= 3u) t = t >= 3u ? (t - 3u) / 2u + 1u : 0u;
    return t < cap ? t : cap;
}

typedef struct {
    const Data_t               *c1;
    Data_t                     *y;
    const int16_t *const       *tab;    /* tanh table per distinct input exponent */
    const unsigned char        *tsel;   /* the table of each channel */
    const double               *g, *b;  /* GroupNorm gamma / beta */
    const signed char          *fo;     /* the output's exponent per channel */
    unsigned                    d, rows, w, rows2, width2, step2, n;
    double                      eps;
} asr_norm_t;

/* tanh -> GroupNorm(1 group) over the valid frames -> conv2's folded input. */
static void asr_stem_norm(asr_norm_t *a)
{
    const unsigned L = a->rows * a->w, t1 = asr_frames_at(1u, a->n, L);
    const size_t   no = (size_t)a->rows2 * a->width2;
    int16_t       *t = NULL;
    long long      s1 = 0, s2 = 0;
    double         mean, var, den;
    unsigned       c, f, r, j;
    memset(a->y, 0, (size_t)a->d * no * sizeof(Data_t));
    if (!t1) return;
    t = (int16_t *)malloc((size_t)a->d * t1 * sizeof(int16_t));
    if (!t) return;
    for (c = 0u; c < a->d; c++) {
        const int16_t *tb = a->tab[a->tsel[c]];
        const int16_t *x = (const int16_t *)a->c1 + (size_t)c * L;
        int16_t       *tr = t + (size_t)c * t1;
        for (f = 0u; f < t1; f++) {
            const int v = tb[(int)x[f] + 32768];
            tr[f] = (int16_t)v;
            s1 += v;
            s2 += (long long)v * v;
        }
    }
    {
        const double N = (double)a->d * (double)t1;
        mean = (double)s1 / N;
        var = (double)s2 / N - mean * mean;
        den = sqrt(var + a->eps * 1073741824.0);
    }
    for (c = 0u; c < a->d; c++) {
        const int16_t *tr = t + (size_t)c * t1;
        const double   g = a->g[c], b = a->b[c], sc = ldexp(1.0, a->fo[c]);
        Data_t        *o = a->y + (size_t)c * no;
        for (r = 0u; r < a->rows2; r++)
            for (j = 0u; j < a->width2; j++) {
                const unsigned s = r * a->step2 + j;
                double         q;
                if (s >= t1) continue;
                q = nearbyint((((double)tr[s] - mean) / den * g + b) * sc);
                o[(size_t)r * a->width2 + j] = (Data_t)(int16_t)(q > 32767.0 ? 32767.0 : q < -32768.0 ? -32768.0 : q);
            }
    }
    free(t);
}

/* round half to even of v / 2^k (k >= 1), int64 */
static long long asr_rne_shr(long long v, int k)
{
    long long q, r, h;
    if (k <= 0) return v * (1ll << -k);
    q = v >> k;
    r = v - (q << k);
    h = 1ll << (k - 1);
    return q + ((r > h) || (r == h && (q & 1)));
}

/* GELU of a stem conv's output -> the next conv's folded input (int16). */
static void asr_gelu_fold(const Data_t *x, const int32_t *const *tab, const unsigned char *tsel, unsigned C,
                          unsigned rows, unsigned w, unsigned tv, const signed char *k, unsigned rows2,
                          unsigned width2, unsigned step2, Data_t *y)
{
    const size_t L = (size_t)rows * w, no = (size_t)rows2 * width2;
    unsigned     c, r, j;
    for (c = 0u; c < C; c++) {
        const int32_t *tb = tab[tsel[c]];
        const int16_t *xr = (const int16_t *)x + (size_t)c * L;
        Data_t        *o = y + (size_t)c * no;
        for (r = 0u; r < rows2; r++)
            for (j = 0u; j < width2; j++) {
                const unsigned s = r * step2 + j;
                long long      q;
                if (s >= tv) {
                    o[(size_t)r * width2 + j] = 0;
                    continue;
                }
                q = asr_rne_shr((long long)tb[(int)xr[s] + 32768], k[c]);
                o[(size_t)r * width2 + j] = (Data_t)(int16_t)(q > 32767 ? 32767 : q < -32768 ? -32768 : q);
            }
    }
}

/* GELU of conv3's output -> the int32 residual [T][C] at 2^-15 (transposed). */
static void asr_gelu_res(const Data_t *x, const int32_t *const *tab, const unsigned char *tsel, unsigned C,
                         unsigned rows, unsigned w, unsigned tv, unsigned T, int32_t *y)
{
    const size_t L = (size_t)rows * w;
    unsigned     c, f;
    memset(y, 0, (size_t)T * C * sizeof(int32_t));
    if (tv > T) tv = T;
    for (c = 0u; c < C; c++) {
        const int32_t *tb = tab[tsel[c]];
        const int16_t *xr = (const int16_t *)x + (size_t)c * L;
        for (f = 0u; f < tv; f++)
            y[(size_t)f * C + c] = (int32_t)asr_rne_shr((long long)tb[(int)xr[f] + 32768], 1);
    }
}

/* The first maximum of V int16 logits. */
static unsigned asr_argmax(const Data_t *lg, unsigned V)
{
    const int16_t *l = (const int16_t *)lg;
    unsigned       i, best = 0u;
    for (i = 1u; i < V; i++)
        if (l[i] > l[best]) best = i;
    return best;
}
"""


def asr_c_helpers() -> str:
    return ASR_C


__all__ = ("ASR_OP_FACTORIES", "ASR_C", "asr_c_helpers", "tanh_table", "gelu16_table", "stem_frames",
           "AsrFramesNode", "AsrStemNormNode", "AsrStemGeluNode", "AsrArgmaxNode", "AsrFrameCountNode", "AsrCrossPrepRoNode",
           "AsrCrossQRoNode", "AsrSwiGLURoNode")
