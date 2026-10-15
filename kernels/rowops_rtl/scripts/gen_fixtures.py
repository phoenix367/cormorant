#!/usr/bin/env python3
"""Writes the RowOpsKernel's RTL fixtures from the Python specification
(inference-scheduler/src/rowops_spec.py): hw/test_data/rowops_test_data/
manifest.txt + test_NN_{a,b,p,c}.hex — the DDR images of a job's regions and
the whole C region after the job (untouched bytes keep the 0xDE poison).
The testbench runs them against the RTL and against its own C++ oracle, so
the oracle is checked against the Python specification by the same files.

Descriptor chains (doc/kernels/ROWOPS_RTL_KERNEL.md §1.6): chains/manifest.txt
+ chains/chain_NN_{mem,out}.hex — one segment per chain holding every job's
regions (inputs, parameter blocks, the outputs poisoned 0xDE, the gaps 0x5A)
before and after the chain; later jobs read what earlier ones wrote.

    python3 scripts/gen_fixtures.py [--out DIR] [--check]

Formats: 16-byte words as 32 hex digits per line, byte 0 first.  Manifest:
  idx op n c1 c2 a_s1 a_s2 b_s1 b_s2 c_s1 c_s2 eps aux bq_sh win_lo win_hi label
chains/manifest.txt, one line per job (a, b, c, p: byte offsets into the segment):
  chain job op n c1 c2 a a_s1 a_s2 b b_s1 b_s2 c c_s1 c_s2 p eps aux bq_sh win_lo win_hi label
"""
import argparse
import importlib.util
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "hw", "test_data", "rowops_test_data")
SPEC = os.path.join(REPO, "inference-scheduler", "src", "rowops_spec.py")
_spec = importlib.util.spec_from_file_location("rowops_spec", SPEC)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

OP = dict(COPY=1, ADDX=2, DEQUANT=3, MULACT=4, NORM=5, ROPE=6, XPOSE=7, FOLD=9)
I16, I32, F32 = 0, 1, 2
F_B_EN, F_P_BCAST, F_P_KEEP, F_RELU, F_NORM_LN = 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16
POISON = 0xDE


def opword(code, ak=I16, bk=I16, ck=I16, b_en=False, bcast=False, relu=False, ln=False, p_keep=False):
    return (code | (ak << 4) | (bk << 6) | (ck << 8) | (F_B_EN if b_en else 0) | (F_P_BCAST if bcast else 0)
            | (F_RELU if relu else 0) | (F_NORM_LN if ln else 0) | (F_P_KEEP if p_keep else 0))


def kbytes(k):
    return 2 if k == I16 else 4


def round16(b):
    return (b + 15) & ~15


class Region:
    """A byte image written at computed offsets."""

    def __init__(self, size):
        self.b = bytearray([POISON]) * 0 + bytearray(size)

    def put(self, off, data: bytes):
        self.b[off:off + len(data)] = data


def pack(vals, kind):
    fmt = "<h" if kind == I16 else ("<i" if kind == I32 else "<I")
    return b"".join(struct.pack(fmt, int(v)) for v in vals)


def param_bytes(params):
    """params: list of (g, bq, sh_a, sh_b, sh_c) -> 16-byte entries."""
    out = bytearray()
    for g, bq, sa, sb, sc in params:
        out += struct.pack("<ii", int(g), int(bq)) + bytes([sa, sb, sc]) + bytes(5)
    return bytes(out)


class Case:
    def __init__(self, label, op, n, c1, c2, strides, a_rows, b_rows, params, expect_rows,
                 kinds, eps=0, aux=0, bq_sh=0, win=(0, 0xFFFFFFFF), xpose=None, raw=None):
        """a_rows / b_rows / expect_rows: {(i2, i1): list of ints}; kinds (ak, bk, ck);
        strides (a_s1, a_s2, b_s1, b_s2, c_s1, c_s2).  xpose: (R, tiles) for TRANSPOSE
        with tiles {(i2, i1): [row lists]} and the expected granules computed here.
        raw: the (a, b, p, c) images as they are (FOLD)."""
        self.label, self.op, self.n, self.c1, self.c2 = label, op, n, c1, c2
        self.strides, self.a_rows, self.b_rows, self.params = strides, a_rows, b_rows, params
        self.expect_rows, self.kinds, self.eps, self.aux, self.bq_sh, self.win = expect_rows, kinds, eps, aux, bq_sh, win
        self.xpose = xpose
        self.raw = raw

    def images(self):
        if self.raw is not None:
            return self.raw
        a_s1, a_s2, b_s1, b_s2, c_s1, c_s2 = self.strides
        ak, bk, ck = self.kinds
        n, c1, c2 = self.n, self.c1, self.c2
        if self.xpose is None:
            a_row = round16(kbytes(ak) * n)
            b_row = round16(8 * (self.aux & 0xFFF)) if self.op & 15 == OP["ROPE"] else round16(kbytes(bk) * n)
            c_row = round16(kbytes(ck) * n)
            a_img = bytearray((c2 - 1) * a_s2 + (c1 - 1) * a_s1 + a_row)
            b_img = bytearray((c2 - 1) * b_s2 + (c1 - 1) * b_s1 + b_row) if self.b_rows else bytearray()
            c_img = bytearray([POISON] * ((c2 - 1) * c_s2 + (c1 - 1) * c_s1 + c_row))
            for (i2, i1), vals in self.a_rows.items():
                a_img[i2 * a_s2 + i1 * a_s1: i2 * a_s2 + i1 * a_s1 + len(vals) * kbytes(ak)] = pack(vals, ak)
            for (i2, i1), vals in self.b_rows.items():
                kb = I32 if self.op & 15 == OP["ROPE"] else bk
                b_img[i2 * b_s2 + i1 * b_s1: i2 * b_s2 + i1 * b_s1 + len(vals) * kbytes(kb)] = pack(vals, kb)
            for (i2, i1), vals in self.expect_rows.items():
                off = i2 * c_s2 + i1 * c_s1
                data = pack(vals, ck)
                data += bytes(round16(len(data)) - len(data))             # the zero tail lanes
                c_img[off: off + len(data)] = data
        else:
            R, tiles = self.xpose
            L = n
            a_img = bytearray((c2 - 1) * a_s2 + (c1 - 1) * a_s1 + (R - 1) * b_s1 + round16(2 * L))
            b_img = bytearray()
            c_img = bytearray([POISON] * round16((c2 - 1) * c_s2 + (c1 - 1) * c_s1 + (L - 1) * b_s2 + 2 * R))
            for (i2, i1), rows in tiles.items():
                for r, vals in enumerate(rows):
                    off = i2 * a_s2 + i1 * a_s1 + r * b_s1
                    a_img[off: off + 2 * L] = pack(vals, I16)
                for d in range(L):
                    off = i2 * c_s2 + i1 * c_s1 + d * b_s2
                    c_img[off: off + 2 * R] = pack([rows[r][d] for r in range(R)], I16)
                if b_s2 == 2 * R:                                         # one run: zero tail lanes
                    off = i2 * c_s2 + i1 * c_s1 + 2 * R * L
                    c_img[off: off + round16(2 * R * L) - 2 * R * L] = bytes(round16(2 * R * L) - 2 * R * L)
        p_img = param_bytes(self.params) if self.params else b""
        return bytes(a_img), bytes(b_img), bytes(p_img), bytes(c_img)


def table_image():
    """The image OP_TLOAD reads: rowops_spec.table_image (one definition)."""
    return rs.table_image()


def hexlines(img):
    img = bytes(img) + bytes((-len(img)) % 16)
    return "\n".join(img[i:i + 16].hex() for i in range(0, len(img), 16)) + ("\n" if img else "")


# ---- case builders -------------------------------------------------------------------
def rne_shr(v, k):
    return int(rs.rne_shr(np.int64(v), k)) if k > 0 else int(v) << (-k)


def stream_rows(op, n, c1, c2, ak, bk, ck, params, bq_sh, rng, b_en, relu=False, win=(0, 0xFFFFFFFF),
                vals_a=None):
    """COPY / ADDX rows and their expected outputs per the spec's formula."""
    lo_a, hi_a = (-(1 << 30), 1 << 30) if ak == I32 else (-32768, 32767)
    lo_b, hi_b = (-(1 << 30), 1 << 30) if bk == I32 else (-32768, 32767)
    a_rows, b_rows, out = {}, {}, {}
    for i2 in range(c2):
        for i1 in range(c1):
            a = rng.integers(lo_a, hi_a + 1, n) if vals_a is None else vals_a[(i2, i1)]
            b = rng.integers(lo_b, hi_b + 1, n) if b_en else np.zeros(n, np.int64)
            a_rows[(i2, i1)] = [int(v) for v in a]
            if b_en:
                b_rows[(i2, i1)] = [int(v) for v in b]
            zero = i1 < win[0] or i1 >= win[1]
            res = []
            for c in range(n):
                g, bq, sa, sb, sc = params[0] if len(params) == 1 else params[c]
                x = 0 if zero else int(a[c])
                s = (x << sa) + ((int(b[c]) << sb) if b_en else 0) + (bq << bq_sh)
                v = rne_shr(s, sc)
                v = max(-32768, min(32767, v)) if ck == I16 else max(-(1 << 31), min((1 << 31) - 1, v))
                if relu and v < 0:
                    v = 0
                res.append(v)
            out[(i2, i1)] = res
    return a_rows, b_rows, out


def strides_for(n, c1, ak, bk, ck, b_row=None, pad=(0, 16, 32)):
    a_s1 = round16(kbytes(ak) * n) + pad[0]
    b_s1 = (round16(kbytes(bk) * n) if b_row is None else round16(b_row)) + pad[1]
    c_s1 = round16(kbytes(ck) * n) + pad[2]
    return (a_s1, a_s1 * c1 + 16, b_s1, b_s1 * c1, c_s1, c_s1 * c1 + 32)


def build_cases():
    rng = np.random.default_rng(2026)
    cases = []

    def add(c):
        cases.append(c)

    # 1. COPY identity, one word
    n, c1 = 8, 1
    prm = [(0, 0, 0, 0, 0)]
    a, b, o = stream_rows(OP["COPY"], n, c1, 1, I16, I16, I16, prm, 0, rng, False)
    add(Case("copy_identity_8", opword(OP["COPY"], bcast=True), n, c1, 1, strides_for(n, c1, I16, I16, I16),
             a, b, prm, o, (I16, I16, I16)))
    # 2. COPY with a tail, three rows, shift + bias, relu
    n, c1 = 13, 3
    prm = [(0, -300, 3, 0, 5)]
    a, b, o = stream_rows(OP["COPY"], n, c1, 1, I16, I16, I16, prm, 4, rng, False, relu=True)
    add(Case("copy_tail13_relu", opword(OP["COPY"], bcast=True, relu=True), n, c1, 1,
             strides_for(n, c1, I16, I16, I16), a, b, prm, o, (I16, I16, I16), bq_sh=4))
    # 3. COPY i32 -> i16 per-column shifts, two blocks
    n, c1, c2 = 21, 2, 2
    prm = [(0, int(rng.integers(-1000, 1000)), int(rng.integers(0, 4)), 0, int(rng.integers(10, 20))) for _ in range(n)]
    a, b, o = stream_rows(OP["COPY"], n, c1, c2, I32, I16, I16, prm, 2, rng, False)
    add(Case("copy_i32_to_i16_blocks", opword(OP["COPY"], ak=I32), n, c1, c2, strides_for(n, c1, I32, I16, I16),
             a, b, prm, o, (I32, I16, I16), bq_sh=2))
    # 4. ADDX residual add: h i32 += d i16 << 7
    n, c1 = 576, 4
    prm = [(0, 0, 0, 7, 0)]
    a, b, o = stream_rows(OP["ADDX"], n, c1, 1, I32, I16, I32, prm, 0, rng, True)
    add(Case("resadd_576x4", opword(OP["ADDX"], ak=I32, bk=I16, ck=I32, b_en=True, bcast=True), n, c1, 1,
             strides_for(n, c1, I32, I16, I32), a, b, prm, o, (I32, I16, I32)))
    # 5. ADDX i16 + i16 -> i16 with per-column shifts, bias, window
    n, c1 = 40, 6
    prm = [(0, int(rng.integers(-2000, 2000)), int(rng.integers(0, 3)), int(rng.integers(0, 3)),
            int(rng.integers(2, 6))) for _ in range(n)]
    a, b, o = stream_rows(OP["ADDX"], n, c1, 1, I16, I16, I16, prm, 3, rng, True, win=(1, 5))
    add(Case("addx_i16_window", opword(OP["ADDX"], b_en=True), n, c1, 1, strides_for(n, c1, I16, I16, I16),
             a, b, prm, o, (I16, I16, I16), bq_sh=3, win=(1, 5)))
    # 6. ADDX i16 + i32 -> i32
    n, c1 = 9, 3
    prm = [(0, 5, 10, 0, 1)]
    a, b, o = stream_rows(OP["ADDX"], n, c1, 1, I16, I32, I32, prm, 20, rng, True)
    add(Case("addx_i16_i32_to_i32", opword(OP["ADDX"], ak=I16, bk=I32, ck=I32, b_en=True, bcast=True), n, c1, 1,
             strides_for(n, c1, I16, I32, I32), a, b, prm, o, (I16, I32, I32), bq_sh=20))
    # 7. DEQUANT i16 -> f32 per column
    n, c1 = 30, 2
    prm = [(0, 0, 0, 0, int(rng.integers(0, 16))) for _ in range(n)]
    a_rows, out = {}, {}
    for i1 in range(c1):
        a = rng.integers(-32768, 32768, n)
        a_rows[(0, i1)] = [int(v) for v in a]
        out[(0, i1)] = [struct.unpack("<I", struct.pack("<f", np.float32(int(a[c]) / 2.0 ** prm[c][4])))[0]
                        for c in range(n)]
    add(Case("dequant_i16", opword(OP["DEQUANT"], ck=F32), n, c1, 1, strides_for(n, c1, I16, I16, F32),
             a_rows, {}, prm, out, (I16, I16, F32)))
    # 8. DEQUANT i32 -> f32, broadcast
    n, c1 = 7, 2
    prm = [(0, 0, 0, 0, 15)]
    a_rows, out = {}, {}
    for i1 in range(c1):
        a = rng.integers(-(1 << 31), 1 << 31, n)
        a_rows[(0, i1)] = [int(v) for v in a]
        out[(0, i1)] = [struct.unpack("<I", struct.pack("<f", np.float32(int(v) / 2.0 ** 15)))[0] for v in a]
    add(Case("dequant_i32_bcast", opword(OP["DEQUANT"], ak=I32, ck=F32, bcast=True), n, c1, 1,
             strides_for(n, c1, I32, I16, F32), a_rows, {}, prm, out, (I32, I16, F32)))
    # 9. MULACT per-column gate exponents
    n, c1 = 48, 3
    f_g = rng.integers(9, 15, n)
    f_u = rng.integers(8, 15, n)
    f_o = rng.integers(6, 10, n)
    prm = [(0, 0, int(14 - f_g[c]), 0, int(20 + f_u[c] - f_o[c])) for c in range(n)]
    a_rows, b_rows, out = {}, {}, {}
    for i1 in range(c1):
        g = rng.integers(-32768, 32768, n)
        u = rng.integers(-32768, 32768, n)
        a_rows[(0, i1)] = [int(v) for v in g]
        b_rows[(0, i1)] = [int(v) for v in u]
        out[(0, i1)] = [int(v) for v in rs.silu_mul(g[None, :], f_g, u[None, :], f_u, f_o)[0]]
    add(Case("mulact_48x3", opword(OP["MULACT"], b_en=True), n, c1, 1, strides_for(n, c1, I16, I16, I16),
             a_rows, b_rows, prm, out, (I16, I16, I16)))
    # 10. MULACT one row of 1536, broadcast exponents
    n, c1 = 1536, 1
    prm = [(0, 0, 2, 0, 24)]
    g = rng.integers(-32768, 32768, n)
    u = rng.integers(-32768, 32768, n)
    out = {(0, 0): [int(v) for v in rs.silu_mul(g[None, :], 12, u[None, :], 12, 8)[0]]}
    add(Case("mulact_1536_bcast", opword(OP["MULACT"], b_en=True, bcast=True), n, c1, 1,
             strides_for(n, c1, I16, I16, I16), {(0, 0): [int(v) for v in g]}, {(0, 0): [int(v) for v in u]},
             prm, out, (I16, I16, I16)))

    # 11. NORM rms on the int32 residual, 576 x 3
    def norm_case(label, ln, ak, n, c1, eps, f_out_lo=7, f_out_hi=10, rows_scale=None, beta_max=0.9):
        f_in = 15 if ak == I32 else 8
        gamma = rng.uniform(0.05, 3.0, n) * rng.choice([-1, 1], n)
        beta = rng.uniform(-beta_max, beta_max, n) if ln else None
        G, g, Bq, E = rs.norm_params(gamma, n, eps, f_in, beta=beta, rms=not ln)
        bb = rs.beta_bits(beta)                                     # 20, coarser for |beta| >= 2
        bq = np.rint((beta if ln else np.zeros(n)) * 2.0 ** bb).astype(np.int64)
        f_out = rng.integers(f_out_lo, f_out_hi + 1, n)
        prm = [(int(G[c]), int(bq[c]), 0, 0, int(24 + g - f_out[c])) for c in range(n)]
        a_rows, out = {}, {}
        scales = rows_scale or [1.0, 30.0, 0.02, 300.0]
        for i1 in range(c1):
            x = rng.standard_normal(n) * scales[i1 % len(scales)]
            if ak == I32:
                x_raw = np.clip(np.rint(x * 2.0 ** 15), -(1 << 31), (1 << 31) - 1).astype(np.int64)
                if i1 == 1:
                    x_raw[3] = 25982 << 15                    # the sink's magnitude
            else:
                x_raw = np.clip(np.rint(x * 256.0), -32768, 32767).astype(np.int64)
            a_rows[(0, i1)] = [int(v) for v in x_raw]
            if ln:
                y = rs.layer_norm(x_raw[None, :], f_in, gamma, beta, eps, f_out)[0]
            else:
                y = rs.rms_norm(x_raw[None, :], f_in, gamma, eps, f_out)[0]
            out[(0, i1)] = [int(v) for v in y]
        return Case(label, opword(OP["NORM"], ak=ak, ln=ln), n, c1, 1, strides_for(n, c1, ak, I16, I16),
                    a_rows, {}, prm, out, (ak, I16, I16), eps=E, bq_sh=rs.P1_BITS + g - bb)
    add(norm_case("rms_i32_576x3", False, I32, 576, 3, 1e-5))
    add(norm_case("ln_i16_768x3", True, I16, 768, 3, 1e-12, rows_scale=[0.5, 4.0, 30.0]))
    add(norm_case("ln_i32_768x4", True, I32, 768, 4, 1e-6))
    add(norm_case("rms_i16_40x5", False, I16, 40, 5, 1e-5, rows_scale=[1.0, 20.0, 60.0]))
    add(norm_case("ln_i16_beta4_64x2", True, I16, 64, 2, 1e-5, beta_max=4.5))   # beta beyond 2: Bq' at 2^-18

    # 15. ROPE: HD 64, 3 heads, 5 rows
    HD, H, T = 64, 3, 5
    n = HD * H
    inv = (1.0 / (10000.0 ** (np.arange(HD // 2) * 2 / HD))).astype(np.float32)
    ang = (np.arange(T, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32)
    cos = np.concatenate([np.cos(ang.astype(np.float64)).astype(np.float32)] * 2, 1).astype(np.float64)
    sin = np.concatenate([np.sin(ang.astype(np.float64)).astype(np.float32)] * 2, 1).astype(np.float64)
    c_raw, s_raw = rs.rope_tables_raw(cos, sin)
    f_x = rng.integers(10, 15, (H, HD))
    f_out = np.array([7, 6, 7])
    half = HD // 2
    f_rot = np.concatenate([f_x[:, half:], f_x[:, :half]], 1)
    fp = np.maximum(f_x, f_rot)
    prm = []
    for h in range(H):
        for d in range(HD):
            prm.append((0, 0, int(fp[h, d] - f_x[h, d]), int(fp[h, d] - f_rot[h, d]), int(fp[h, d] + 23 - f_out[h])))
    x_raw = rng.integers(-32768, 32768, (T, H, HD))
    y = rs.rope(x_raw, f_x, c_raw, s_raw, f_out)
    a_rows = {(0, t): [int(v) for v in x_raw[t].reshape(-1)] for t in range(T)}
    b_rows = {(0, t): [int(v) for v in c_raw[t]] + [int(v) for v in s_raw[t]] for t in range(T)}
    out = {(0, t): [int(v) for v in y[t].reshape(-1)] for t in range(T)}
    add(Case("rope_hd64_h3_t5", opword(OP["ROPE"], bk=I32, b_en=True), n, T, 1,
             strides_for(n, T, I16, I32, I16, b_row=8 * HD), a_rows, b_rows, prm, out, (I16, I32, I16), aux=HD))

    # 16. TRANSPOSE: the V-cache interleave (R = 4, contiguous granules) and the
    #     q image (R = 8, granules at stride 2 * G * T), and R = 1 straddling words
    def xpose_case(label, R, L, c1, c2, gran_stride, tile_stride=None):
        tiles = {(i2, i1): [[int(v) for v in rng.integers(-32768, 32768, L)] for _ in range(R)]
                 for i2 in range(c2) for i1 in range(c1)}
        b_s1 = round16(2 * L) + 16
        a_s1 = b_s1 * R + 16
        a_s2 = a_s1 * c1 + 32
        c_s1 = tile_stride if tile_stride is not None else round16(gran_stride * (L - 1) + 2 * R) + 16
        c_s2 = c_s1 * c1 + 16
        return Case(label, opword(OP["XPOSE"]), L, c1, c2, (a_s1, a_s2, b_s1, gran_stride, c_s1, c_s2),
                    {}, {}, [], {}, (I16, I16, I16), aux=R, xpose=(R, tiles))
    add(xpose_case("xpose_r4_contig_L64", 4, 64, 3, 1, 8))
    add(xpose_case("xpose_r8_qimage_L64", 8, 64, 2, 2, 2 * 3 * 16, tile_stride=16))
    add(xpose_case("xpose_r1_straddle_L37", 1, 37, 2, 1, 6))
    add(xpose_case("xpose_r3_L100", 3, 100, 1, 1, 14))
    add(xpose_case("xpose_r3_contig_L37", 3, 37, 2, 1, 6))      # a contiguous run with a tail word

    # 17. The attention prep's jobs (HOSTOPS_PLAN §10, rowops_nodes.AttnPrepRoNode) at
    #     a small geometry — HD 16, KV 2, 16K-row V blocks with K = 4, staging rows of
    #     KV * HD int16 (64 bytes), cache rows C = 128:
    HD, KV, K, C = 16, 2, 4, 128
    RS = KV * HD * 2
    #     (a) the q staging zeroed: COPY with the window [0, 0) — every row reads as zero
    n, c1 = 3 * HD, 4
    prm = [(0, 0, 0, 0, 0)]
    a, b, o = stream_rows(OP["COPY"], n, c1, 1, I16, I16, I16, prm, 0, rng, False, win=(0, 0))
    add(Case("copy_window_zero_48x4", opword(OP["COPY"], bcast=True), n, c1, 1,
             strides_for(n, c1, I16, I16, I16), a, b, prm, o, (I16, I16, I16), win=(0, 0)))
    #     (b) ks -> the K cache rows [pos, pos + n), group-major: COPY of n = HD with blocks = KV
    #         (a_s2 = HD * 2 inside the staging row, c_s2 = C * HD * 2 between the groups), 5 rows
    n, c1, c2 = HD, 5, KV
    a, b, o = stream_rows(OP["COPY"], n, c1, c2, I16, I16, I16, prm, 0, rng, False)
    add(Case("copy_kcache_blocks_hd16", opword(OP["COPY"], bcast=True), n, c1, c2,
             (RS, HD * 2, 0, 0, HD * 2, C * HD * 2), a, b, prm, o, (I16, I16, I16)))
    #     (c) the whole 16K-row blocks of the V image: R = K tiles, the tile's rows 16 staging
    #         rows apart (b_s1 = 16 RS > a_s1 = RS), 16 tiles per block, 2 blocks, contiguous granules
    def xpose_geom(label, R, L, c1, c2, strides):
        tiles = {(i2, i1): [[int(v) for v in rng.integers(-32768, 32768, L)] for _ in range(R)]
                 for i2 in range(c2) for i1 in range(c1)}
        return Case(label, opword(OP["XPOSE"]), L, c1, c2, strides, {}, {}, [], {}, (I16, I16, I16),
                    aux=R, xpose=(R, tiles))
    add(xpose_geom("xpose_vimage_whole_k4", K, HD, 16, 2,
                   (RS, 16 * K * RS, 16 * RS, 2 * K, HD * K * 2, 16 * K * HD * 2)))
    #     (d) the boundary rows of the V image: R = 1 granules 2K bytes apart, the rows HD*K*2
    #         apart, the complete 16-row groups as blocks 2 bytes apart (c_s2 = 2) — the other
    #         keys' bytes between the granules must keep their values (byte strobes)
    add(xpose_geom("xpose_vimage_edge_r1_groups", 1, HD, 16, 2,
                   (RS, 16 * RS, 0, 2 * K, HD * K * 2, 2)))
    add(xpose_geom("xpose_vimage_edge_r1_partial", 1, HD, 5, 1,
                   (RS, 16 * RS, 0, 2 * K, HD * K * 2, 2)))
    #     (e) the q image: R = 8 tiles of the q staging (H = 3 heads of HD as blocks: a_s2 = HD * 2),
    #         granules 2 * G * T apart (G = 3, T = 16), the tiles 16 bytes apart, the heads 2T apart
    G, T = 3, 16
    SQ = G * HD * 2
    add(xpose_geom("xpose_qimage_r8_g3_t16", 8, HD, T // 8, G,
                   (8 * SQ, HD * 2, SQ, 2 * G * T, 16, 2 * T)))
    #     (f) the vision / TTS prep's COPY with a bias (rowops_spec.bias_round, phase 4b): per
    #         column Bq' = round(b 2^fp), sh_a = fp - f_x, sh_c = fp - f_out at fp = bias_exp, RELU
    n, c1 = 40, 3
    bias = rng.uniform(-3.0, 3.0, n)
    f_x = rng.integers(6, 13, n)
    f_out = rng.integers(7, 11, n)
    bq, sa, sc = rs.bias_params(bias, f_x, f_out)
    prm = [(0, int(bq[c]), int(sa[c]), 0, int(sc[c])) for c in range(n)]
    a, b, o = stream_rows(OP["COPY"], n, c1, 1, I16, I16, I16, prm, 0, rng, False, relu=True)
    add(Case("copy_bias_relu_percol_40x3", opword(OP["COPY"], relu=True), n, c1, 1,
             strides_for(n, c1, I16, I16, I16), a, b, prm, o, (I16, I16, I16)))
    # the spec agrees with the fixture's formula
    x_raw = np.array([a[(0, i)] for i in range(c1)], np.int64)
    assert (rs.bias_round(x_raw, f_x, bias, f_out, relu=True) == np.array([o[(0, i)] for i in range(c1)])).all()

    # 18. FOLD (op 9, HOSTOPS_PLAN §11: Piper's TtsPrep) — rowops_spec.fold on C source rows
    #     of L int16 a_s2 bytes apart: W = w0 + hl + hr columns per row from s0 = t0 - hl,
    #     the window [lo, hi), LeakyReLU 1/m, s bits coarser; the output one contiguous run
    def fold_case(label, C, L, rows, w0, hl, hr, t0, m, s, win, pad=0):
        W, s0 = w0 + hl + hr, t0 - hl
        stride = L + pad                                           # elements between the source rows
        src = rng.integers(-32768, 32768, (C - 1) * stride + L)
        src[::5] = rng.integers(-300, 300, src[::5].size)          # small values: ties
        src[:8] = [-32768, 32767, -1, 1, 0, -5 * m, 5 * m, -m]
        x = np.stack([src[k * stride: k * stride + L] for k in range(C)])
        y = rs.fold(x, rows, w0, s0, W, win[0], win[1], m, s)
        a_img = pack(src, I16)
        a_img += bytes(round16(len(a_img)) - len(a_img))
        c_img = pack(y.reshape(-1), I16)
        c_img += bytes(round16(len(c_img)) - len(c_img))
        return Case(label, OP["FOLD"], W, rows, C, (0, 2 * stride, w0, s0 & 0xFFFFFFFF, 0, 0), {}, {}, [], {},
                    (I16, I16, I16), aux=rs.fold_aux(m, s), win=win, raw=(a_img, b"", b"", c_img))
    add(fold_case("fold_flow_w36_m1_s4", 6, 64, 2, 32, 2, 2, 0, 1, 4, (0, 64)))
    add(fold_case("fold_leaky_m10_window", 4, 96, 3, 32, 1, 1, 0, 10, 1, (5, 90)))
    add(fold_case("fold_m100_halo36", 2, 128, 4, 32, 36, 0, 0, 100, 2, (0, 128)))
    add(fold_case("fold_t0_12_hr24", 3, 128, 4, 32, 0, 24, 12, 10, 0, (0, 128)))
    add(fold_case("fold_window_empty", 2, 64, 3, 24, 1, 1, 0, 10, 1, (40, 40)))
    add(fold_case("fold_m7_s9_w8", 3, 40, 5, 8, 0, 0, 0, 7, 9, (0, 40)))
    add(fold_case("fold_w1_rows9", 2, 30, 9, 3, 0, 0, 1, 10, 2, (0, 30)))
    add(fold_case("fold_unaligned_rows", 3, 50, 2, 24, 3, 2, 0, 10, 3, (2, 47), pad=3))
    return cases


# ---- descriptor chains ------------------------------------------------------------------
GAP = 0x5A


class Seg:
    """A chain's DDR segment: regions allocated 16-byte aligned with 64-byte gaps (0x5A).
    `b0` is the image before the chain; the jobs' outputs are computed in order from `b`
    (the image as the jobs before them left it) and written into it."""

    def __init__(self):
        self.b0 = bytearray()
        self.b = bytearray()

    def alloc(self, nbytes, fill=None):
        assert len(self.b) == len(self.b0)
        off = round16(len(self.b0) + 64)
        new = bytes([GAP]) * (off - len(self.b0))
        new += bytes(fill) if fill is not None else bytes([POISON]) * nbytes
        new += bytes([POISON]) * max(0, off + nbytes - len(self.b0) - len(new))
        self.b0 += new
        self.b += new
        return off

    def rows(self, off, s1, s2, c1, c2, n, kind):
        """[c2 * c1][n] int64 of the rows (i2, i1) at off + i2 s2 + i1 s1."""
        fmt = {I16: "<h", I32: "<i"}[kind]
        nb = kbytes(kind)
        out = np.zeros((c2 * c1, n), np.int64)
        for i2 in range(c2):
            for i1 in range(c1):
                o = off + i2 * s2 + i1 * s1
                out[i2 * c1 + i1] = [struct.unpack_from(fmt, self.b, o + nb * c)[0] for c in range(n)]
        return out

    def put_rows(self, off, s1, s2, c1, c2, vals, kind):
        """Whole-word rows: the lanes past n of the last word written 0."""
        for i2 in range(c2):
            for i1 in range(c1):
                data = pack(vals[i2 * c1 + i1], kind)
                data += bytes(round16(len(data)) - len(data))
                o = off + i2 * s2 + i1 * s1
                self.b[o: o + len(data)] = data


def stream_formula(x, b, prm, bq_sh, ck, relu=False):
    """COPY / ADDX per column (rowops_spec's formula: one rounding, saturation, RELU)."""
    rows, n = x.shape
    y = np.zeros((rows, n), np.int64)
    for r in range(rows):
        for c in range(n):
            g, bq, sa, sb, sc = prm[0] if len(prm) == 1 else prm[c]
            v = rne_shr((int(x[r, c]) << sa) + ((int(b[r, c]) << sb) if b is not None else 0) + (bq << bq_sh), sc)
            v = max(-32768, min(32767, v)) if ck == I16 else max(-(1 << 31), min((1 << 31) - 1, v))
            y[r, c] = 0 if (relu and v < 0) else v
    return y


def chain_job(label, op, n, c1, c2, a=0, a_s=(0, 0), b=0, b_s=(0, 0), c=0, c_s=(0, 0), p=0, eps=0, aux=0,
              bq_sh=0, win=(0, 0xFFFFFFFF)):
    return dict(label=label, op=op, n=n, c1=c1, c2=c2, a=a, a_s1=a_s[0], a_s2=a_s[1], b=b, b_s1=b_s[0],
                b_s2=b_s[1], c=c, c_s1=c_s[0], c_s2=c_s[1], p=p, eps=eps, aux=aux, bq_sh=bq_sh, win=win)


def chain_decode_prep(rng):
    """A decode step's attention prep (rowops_nodes.AttnPrepRoNode at T = 1, HOSTOPS_PLAN §10,
    RPU_PLAN §13.4) at a small geometry — HD 16, KV 2, G 2 (H 4), position 5 of a K cache of
    32 rows, the V image's keys interleaved by 4: RoPE of q and k, the rotated k into the K
    cache (the groups as blocks), v re-rounded, the q image per group from the rotated q
    (TRANSPOSE R = G, contiguous), v's granules into the V image (R = 1, byte strobes: the
    other keys' bytes keep their values), and the q staging zeroed after its transpose."""
    HD, KV, G, CR, K, pos = 16, 2, 2, 32, 4, 5
    H = KV * G
    seg, jobs = Seg(), []
    # inputs: q, k, v rows (int16), one RoPE cos / sin row
    q = rng.integers(-32768, 32768, H * HD)
    k = rng.integers(-32768, 32768, KV * HD)
    v = rng.integers(-32768, 32768, KV * HD)
    inv = (1.0 / (10000.0 ** (np.arange(HD // 2) * 2 / HD))).astype(np.float32)
    ang = (np.float32(pos) * inv).astype(np.float32)
    cos = np.concatenate([np.cos(ang.astype(np.float64)).astype(np.float32)] * 2).astype(np.float64)
    sin = np.concatenate([np.sin(ang.astype(np.float64)).astype(np.float32)] * 2).astype(np.float64)
    c_raw, s_raw = rs.rope_tables_raw(cos[None, :], sin[None, :])
    o_q = seg.alloc(2 * H * HD, pack(q, I16))
    o_k = seg.alloc(2 * KV * HD, pack(k, I16))
    o_v = seg.alloc(2 * KV * HD, pack(v, I16))
    o_cs = seg.alloc(8 * HD, pack(list(c_raw[0]) + list(s_raw[0]), I32))
    # the caches hold older positions: their bytes must survive
    o_kc = seg.alloc(2 * KV * CR * HD, pack(rng.integers(-32768, 32768, KV * CR * HD), I16))
    VI = 2 * HD * K                                             # bytes of a V image row of K keys
    o_vi = seg.alloc(KV * (CR // K) * VI, pack(rng.integers(-32768, 32768, KV * (CR // K) * VI // 2), I16))
    o_qi = seg.alloc(KV * 128)                                  # the q image: 64 bytes per group, 128 apart
    o_qs = seg.alloc(2 * H * HD)                                # staging rows
    o_ks = seg.alloc(2 * KV * HD)
    o_vs = seg.alloc(2 * KV * HD)

    def rope_job(label, src, dst, heads):
        n = heads * HD
        f_x = rng.integers(10, 15, (heads, HD))
        f_out = rng.integers(6, 9, heads)
        half = HD // 2
        f_rot = np.concatenate([f_x[:, half:], f_x[:, :half]], 1)
        fp = np.maximum(f_x, f_rot)
        prm = [(0, 0, int(fp[h, d] - f_x[h, d]), int(fp[h, d] - f_rot[h, d]), int(fp[h, d] + 23 - f_out[h]))
               for h in range(heads) for d in range(HD)]
        o_p = seg.alloc(16 * n, param_bytes(prm))
        x = seg.rows(src, 0, 0, 1, 1, n, I16).reshape(1, heads, HD)
        cs = seg.rows(o_cs, 0, 0, 1, 1, 2 * HD, I32)
        y = rs.rope(x, f_x, cs[:, :HD], cs[:, HD:], f_out[:, None] * np.ones((1, HD), np.int64))
        seg.put_rows(dst, 0, 0, 1, 1, y.reshape(1, n), I16)
        jobs.append(chain_job(label, opword(OP["ROPE"], bk=I32, b_en=True), n, 1, 1, a=src, a_s=(round16(2 * n), 0),
                              b=o_cs, b_s=(8 * HD, 0), c=dst, c_s=(round16(2 * n), 0), p=o_p, aux=HD))
    rope_job("rope_q", o_q, o_qs, H)
    rope_job("rope_k", o_k, o_ks, KV)
    # the rotated k (rope_k's output) into the K cache row pos of every group
    o_p1 = seg.alloc(16, param_bytes([(0, 0, 0, 0, 0)]))
    x = seg.rows(o_ks, 0, 2 * HD, 1, KV, HD, I16)
    seg.put_rows(o_kc + 2 * pos * HD, 2 * HD, 2 * CR * HD, 1, KV, x, I16)
    jobs.append(chain_job("copy_kcache", opword(OP["COPY"], bcast=True), HD, 1, KV, a=o_ks, a_s=(2 * HD, 2 * HD),
                          c=o_kc + 2 * pos * HD, c_s=(2 * HD, 2 * CR * HD), p=o_p1))
    # v re-rounded per column into its staging row
    prm = [(0, 0, 0, 0, int(rng.integers(0, 4))) for _ in range(KV * HD)]
    o_p2 = seg.alloc(16 * KV * HD, param_bytes(prm))
    x = seg.rows(o_v, 0, 0, 1, 1, KV * HD, I16)
    seg.put_rows(o_vs, 0, 0, 1, 1, stream_formula(x, None, prm, 0, I16), I16)
    jobs.append(chain_job("copy_v_reround", opword(OP["COPY"]), KV * HD, 1, 1, a=o_v, a_s=(2 * KV * HD, 0),
                          c=o_vs, c_s=(2 * KV * HD, 0), p=o_p2))
    # the q image of every group from the rotated q: tile = the group's G heads (rows HD apart)
    for g in range(KV):
        rows_ = seg.rows(o_qs + 2 * g * G * HD, 2 * HD, 0, G, 1, HD, I16)
        img = [int(rows_[r, d]) for d in range(HD) for r in range(G)]
        seg.put_rows(o_qi + 128 * g, 0, 0, 1, 1, [img], I16)
    jobs.append(chain_job("xpose_qimage", opword(OP["XPOSE"]), HD, 1, KV, a=o_qs, a_s=(0, 2 * G * HD),
                          b_s=(2 * HD, 2 * G), c=o_qi, c_s=(0, 128), aux=G))
    # v's granules into the V image: key pos is lane pos % K of row pos // K, groups VI * CR / K apart
    c0 = o_vi + (pos // K) * VI + 2 * (pos % K)
    for g in range(KV):
        vals = seg.rows(o_vs + 2 * g * HD, 0, 0, 1, 1, HD, I16)[0]
        for d in range(HD):
            o = c0 + g * (CR // K) * VI + d * 2 * K
            seg.b[o: o + 2] = pack([vals[d]], I16)
    jobs.append(chain_job("xpose_vgranules_r1", opword(OP["XPOSE"]), HD, 1, KV, a=o_vs, a_s=(0, 2 * HD),
                          b_s=(0, 2 * K), c=c0, c_s=(0, (CR // K) * VI), aux=1))
    # the q staging zeroed (COPY, window [0, 0)) — after the q image read it
    seg.put_rows(o_qs, 0, 0, 1, 1, np.zeros((1, H * HD), np.int64), I16)
    jobs.append(chain_job("copy_zero_qstaging", opword(OP["COPY"], bcast=True), H * HD, 1, 1, a=o_q,
                          a_s=(2 * H * HD, 0), c=o_qs, c_s=(2 * H * HD, 0), p=o_p1, win=(0, 0)))
    return jobs, seg


def chain_residual_norm(rng):
    """The residual stream of a decoder layer and Piper's fold at n = 96: the residual add
    (ADDX, int32), RMSNorm of its output (P loaded), RMSNorm of other rows with P_KEEP (its
    own p a decoy block), SiLU * up of the normalised rows (the gate at the norm's output
    exponents), DEQUANT of the residual, FOLD of the SiLU output rows (a halo, the window,
    m 10, s 1), and a windowed COPY of the residual to int16 with a bias and RELU."""
    n, c1 = 96, 3
    seg, jobs = Seg(), []
    h = np.clip(np.rint(rng.standard_normal((c1, n)) * 2.0 ** 15 * np.array([[1.0], [20.0], [0.05]])),
                -(1 << 31), (1 << 31) - 1).astype(np.int64)
    d = rng.integers(-32768, 32768, (c1, n))
    o_h = seg.alloc(4 * n * c1, pack(h.reshape(-1), I32))
    o_d = seg.alloc(2 * n * c1, pack(d.reshape(-1), I16))
    h2 = np.clip(np.rint(rng.standard_normal((2, n)) * 2.0 ** 15 * 3.0), -(1 << 31), (1 << 31) - 1).astype(np.int64)
    o_h2 = seg.alloc(4 * n * 2, pack(h2.reshape(-1), I32))
    u = rng.integers(-32768, 32768, (c1, n))
    o_u = seg.alloc(2 * n * c1, pack(u.reshape(-1), I16))
    s32, s16 = 4 * n, round16(2 * n)
    o_r = seg.alloc(s32 * c1)                                   # outputs
    o_x = seg.alloc(s16 * c1)
    o_x2 = seg.alloc(s16 * 2)
    o_m = seg.alloc(s16 * c1)
    o_f = seg.alloc(s32 * c1)
    o_c = seg.alloc(s16 * c1)
    # 0. the residual add: r = h + (d << 7)
    prm = [(0, 0, 0, 7, 0)]
    o_p = seg.alloc(16, param_bytes(prm))
    r = stream_formula(seg.rows(o_h, s32, 0, c1, 1, n, I32), seg.rows(o_d, 2 * n, 0, c1, 1, n, I16), prm, 0, I32)
    seg.put_rows(o_r, s32, 0, c1, 1, r, I32)
    jobs.append(chain_job("resadd", opword(OP["ADDX"], ak=I32, bk=I16, ck=I32, b_en=True, bcast=True), n, c1, 1,
                          a=o_h, a_s=(s32, s32 * c1), b=o_d, b_s=(2 * n, 2 * n * c1), c=o_r, c_s=(s32, s32 * c1),
                          p=o_p))
    # 1. RMSNorm of the residual (rowops_spec.rms_norm), per-column output exponents
    gamma = rng.uniform(0.05, 3.0, n) * rng.choice([-1, 1], n)
    eps = 1e-5
    G, g, _b, E = rs.norm_params(gamma, n, eps, 15, rms=True)
    f_x = rng.integers(7, 11, n)
    prm = [(int(G[c]), 0, 0, 0, int(24 + g - f_x[c])) for c in range(n)]
    o_pn = seg.alloc(16 * n, param_bytes(prm))
    bq_sh = rs.P1_BITS + g - rs.beta_bits(None)
    x = rs.rms_norm(seg.rows(o_r, s32, 0, c1, 1, n, I32), 15, gamma, eps, f_x)
    seg.put_rows(o_x, s16, 0, c1, 1, x, I16)
    jobs.append(chain_job("rms_norm", opword(OP["NORM"], ak=I32), n, c1, 1, a=o_r, a_s=(s32, s32 * c1), c=o_x,
                          c_s=(s16, s16 * c1), p=o_pn, eps=E, bq_sh=bq_sh))
    # 2. the same norm of other rows with P_KEEP: p points at a decoy (other gammas)
    decoy = [(int(-G[c]), 0, 0, 0, int(24 + g - f_x[c]) ^ 1) for c in range(n)]
    o_pd = seg.alloc(16 * n, param_bytes(decoy))
    x2 = rs.rms_norm(seg.rows(o_h2, s32, 0, 2, 1, n, I32), 15, gamma, eps, f_x)
    seg.put_rows(o_x2, s16, 0, 2, 1, x2, I16)
    jobs.append(chain_job("rms_norm_keep", opword(OP["NORM"], ak=I32, p_keep=True), n, 2, 1, a=o_h2,
                          a_s=(s32, s32 * 2), c=o_x2, c_s=(s16, s16 * 2), p=o_pd, eps=E, bq_sh=bq_sh))
    # 3. SiLU * up: the gate is the norm's output at its exponents f_x (rowops_spec.silu_mul)
    f_u = rng.integers(8, 13, n)
    f_o = rng.integers(6, 10, n)
    prm = [(0, 0, int(14 - f_x[c]), 0, int(20 + f_u[c] - f_o[c])) for c in range(n)]
    o_pm = seg.alloc(16 * n, param_bytes(prm))
    mres = rs.silu_mul(seg.rows(o_x, s16, 0, c1, 1, n, I16), f_x, seg.rows(o_u, 2 * n, 0, c1, 1, n, I16), f_u, f_o)
    seg.put_rows(o_m, s16, 0, c1, 1, mres, I16)
    jobs.append(chain_job("silu_mul", opword(OP["MULACT"], b_en=True), n, c1, 1, a=o_x, a_s=(s16, s16 * c1), b=o_u,
                          b_s=(2 * n, 2 * n * c1), c=o_m, c_s=(s16, s16 * c1), p=o_pm))
    # 4. DEQUANT of the residual (int32 at 2^-15 -> float32)
    o_pq = seg.alloc(16, param_bytes([(0, 0, 0, 0, 15)]))
    rr = seg.rows(o_r, s32, 0, c1, 1, n, I32)
    fl = [[struct.unpack("<I", struct.pack("<f", np.float32(int(vv) / 2.0 ** 15)))[0] for vv in row] for row in rr]
    for i1 in range(c1):
        seg.b[o_f + i1 * s32: o_f + i1 * s32 + 4 * n] = pack(fl[i1], F32)
    jobs.append(chain_job("dequant_residual", opword(OP["DEQUANT"], ak=I32, ck=F32, bcast=True), n, c1, 1, a=o_r,
                          a_s=(s32, s32 * c1), c=o_f, c_s=(s32, s32 * c1), p=o_pq))
    # 5. FOLD of the SiLU output rows (the channels s16 apart): 2 rows of 44 from -2, 40 apart
    rows_, w0, hl, W, m, sh = 2, 40, 2, 44, 10, 1
    src = seg.rows(o_m, s16, 0, c1, 1, n, I16)
    y = rs.fold(src, rows_, w0, -hl, W, 0, n, m, sh)
    o_fo = seg.alloc(round16(2 * c1 * rows_ * W))
    data = pack(y.reshape(-1), I16)
    seg.b[o_fo: o_fo + round16(len(data))] = data + bytes(round16(len(data)) - len(data))
    jobs.append(chain_job("fold_silu_rows", OP["FOLD"], W, rows_, c1, a=o_m, a_s=(0, s16), b_s=(w0, (-hl) & 0xFFFFFFFF),
                          c=o_fo, aux=rs.fold_aux(m, sh), win=(0, n)))
    # 6. the residual to int16 with a bias and RELU, rows outside [1, 2) read as zeros
    prm = [(0, int(rng.integers(-3000, 3000)), 0, 0, int(rng.integers(5, 9))) for _ in range(n)]
    o_pc = seg.alloc(16 * n, param_bytes(prm))
    xr = seg.rows(o_r, s32, 0, c1, 1, n, I32)
    xr[[0, 2]] = 0
    seg.put_rows(o_c, s16, 0, c1, 1, stream_formula(xr, None, prm, 4, I16, relu=True), I16)
    jobs.append(chain_job("copy_window_relu", opword(OP["COPY"], ak=I32, relu=True), n, c1, 1, a=o_r,
                          a_s=(s32, s32 * c1), c=o_c, c_s=(s16, s16 * c1), p=o_pc, bq_sh=4, win=(1, 2)))
    return jobs, seg


def build_chains():
    """The chain fixtures: (jobs, segment before, segment after) per chain."""
    rng = np.random.default_rng(2028)
    out = []
    for build in (chain_decode_prep, chain_residual_norm):
        jobs, seg = build(rng)
        assert len(seg.b) == len(seg.b0)
        out.append((jobs, bytes(seg.b0), bytes(seg.b)))
    return out


def chain_files():
    """{relative path: text} of the chain fixtures (chains/)."""
    files = {}
    lines = ["# RowOpsKernel RTL chain fixtures (scripts/gen_fixtures.py): one line per job, the jobs of a chain "
             "in order; a b c p are byte offsets into the chain's segment (chain_NN_mem.hex before the chain, "
             "chain_NN_out.hex after it)",
             "# chain job op n c1 c2 a a_s1 a_s2 b b_s1 b_s2 c c_s1 c_s2 p eps aux bq_sh win_lo win_hi label"]
    for ci, (jobs, before, after) in enumerate(build_chains()):
        files[f"chains/chain_{ci:02d}_mem.hex"] = hexlines(before)
        files[f"chains/chain_{ci:02d}_out.hex"] = hexlines(after)
        for ji, j in enumerate(jobs):
            lines.append(f"{ci} {ji} {j['op']} {j['n']} {j['c1']} {j['c2']} {j['a']} {j['a_s1']} {j['a_s2']} "
                         f"{j['b']} {j['b_s1']} {j['b_s2']} {j['c']} {j['c_s1']} {j['c_s2']} {j['p']} {j['eps']} "
                         f"{j['aux']} {j['bq_sh']} {j['win'][0]} {j['win'][1]} {j['label']}")
    files["chains/manifest.txt"] = "\n".join(lines) + "\n"
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 if the files on disk differ")
    args = ap.parse_args()
    cases = build_cases()
    files = {}
    lines = ["# RowOpsKernel RTL fixtures (scripts/gen_fixtures.py): idx op n c1 c2 a_s1 a_s2 b_s1 b_s2 "
             "c_s1 c_s2 eps aux bq_sh win_lo win_hi label"]
    for i, c in enumerate(cases):
        a, b, p, cimg = c.images()
        files[f"test_{i:02d}_a.hex"] = hexlines(a)
        files[f"test_{i:02d}_b.hex"] = hexlines(b)
        files[f"test_{i:02d}_p.hex"] = hexlines(p)
        files[f"test_{i:02d}_c.hex"] = hexlines(cimg)
        s = c.strides
        lines.append(f"{i} {c.op} {c.n} {c.c1} {c.c2} {s[0]} {s[1]} {s[2]} {s[3]} {s[4]} {s[5]} {c.eps} {c.aux} "
                     f"{c.bq_sh} {c.win[0]} {c.win[1]} {c.label}")
    files["tables.hex"] = hexlines(table_image())
    files["manifest.txt"] = "\n".join(lines) + "\n"
    files.update(chain_files())
    if args.check:
        bad = [f for f, t in files.items()
               if not os.path.exists(os.path.join(args.out, f)) or open(os.path.join(args.out, f)).read() != t]
        if bad:
            print(f"{args.out}: {len(bad)} files differ: {bad[:4]}", file=sys.stderr)
            return 1
        print(f"{args.out}: {len(cases)} cases and the chains up to date")
        return 0
    os.makedirs(os.path.join(args.out, "chains"), exist_ok=True)
    for d, pre in ((args.out, "test_"), (os.path.join(args.out, "chains"), "chain_")):
        for f in os.listdir(d):
            if f.startswith(pre) and f.endswith(".hex"):
                os.remove(os.path.join(d, f))
    for f, t in files.items():
        open(os.path.join(args.out, f), "w").write(t)
    print(f"{args.out}: {len(cases)} cases + the chains, {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
