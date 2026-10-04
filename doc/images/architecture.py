#!/usr/bin/env python3
"""
architecture.py — draws doc/images/architecture.svg, the layered system
diagram of the main README: applications, the software on the board and on
the host, the FPGA kernels and the Kria KV260, with the cross-cutting
verification and performance tooling as pillars.

usage: python3 doc/images/architecture.py   (needs Pillow, for text widths)

Every label must fit its box: the script measures each line with a
sans-serif font a little wider than the browser's and stops when one does
not fit.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from PIL import ImageFont

OUT = Path(__file__).resolve().parent / "architecture.svg"
W = 1600
FONT = "Segoe UI, Helvetica Neue, Helvetica, Arial, sans-serif"
MEASURE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 100)

# palette (after the boardmix example)
HEADER_BG, TITLE = "#e9eaee", "#1f2328"
APP_FILL, APP_STROKE, APP_CHIP = "#c8f7d8", "#1fbf8f", "#2fd8a8"
SW_FILL, SW_STROKE, SW_CHIP, SW_OLIVE = "#d7eefb", "#e8b6f0", "#9b6bff", "#6f9a3c"
HW_FILL, HW_STROKE, HW_CHIP = "#aedcf6", "#3fd09a", "#bdf3d9"
BOARD_CHIP, BRACE, ARROW = "#e1e1e3", "#9b6bff", "#6b6b70"
PILLAR_V, PILLAR_P, GEN = "#9d174d", "#86198f", "#7c3aed"   # pillars: white text at > 7:1

out: list[str] = []
overflow: list[str] = []


def width_of(text: str, size: float) -> float:
    return MEASURE.getlength(text) * size / 100.0


def text(x, y, s, size=15, fill=TITLE, weight="normal", anchor="middle", rotate=None):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}" dominant-baseline="central"{tr}>{html.escape(s)}</text>')


def rect(x, y, w, h, fill, stroke=None, sw=2, r=6):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{st}/>')


def chip(x, y, w, h, lines, fill, color="#ffffff", size=14.5, bold_first=False):
    rect(x, y, w, h, fill, r=4)
    lh = size * 1.25
    y0 = y + h / 2 - lh * (len(lines) - 1) / 2
    for i, s in enumerate(lines):
        if width_of(s, size) > w - 14:
            overflow.append(f"{width_of(s, size):.0f} > {w - 14:.0f} px: {s!r}")
        text(x + w / 2, y0 + i * lh, s, size, color, "bold" if bold_first and i == 0 else "normal")


def chips(x, y, w, h, items, fill, color="#ffffff", gap=18, size=14.5, bold_first=False):
    n = len(items)
    cw = (w - gap * (n - 1)) / n
    for i, lines in enumerate(items):
        chip(x + i * (cw + gap), y, cw, h, lines, fill, color, size, bold_first)


def arrow_up(cx, y_top, y_bot, w=34, fill=ARROW):
    head = 22
    out.append(f'<polygon fill="{fill}" points="{cx - w / 2},{y_top + head} {cx},{y_top} {cx + w / 2},{y_top + head} '
               f'{cx + w / 4},{y_top + head} {cx + w / 4},{y_bot} {cx - w / 4},{y_bot} {cx - w / 4},{y_top + head}"/>')


def arrow_down(cx, y_top, y_bot, w=34, fill=ARROW):
    head = 22
    out.append(f'<polygon fill="{fill}" points="{cx - w / 4},{y_top} {cx + w / 4},{y_top} {cx + w / 4},{y_bot - head} '
               f'{cx + w / 2},{y_bot - head} {cx},{y_bot} {cx - w / 2},{y_bot - head} {cx - w / 4},{y_bot - head}"/>')


def note(x, y, s, fill="#4b5563", size=13.5):
    """An arrow's label, left-aligned beside it."""
    text(x, y, s, size, fill, "500", "start")


def arrow_both(cx, y_top, y_bot, w=34):
    head = 20
    out.append(f'<polygon fill="{ARROW}" points="{cx},{y_top} {cx + w / 2},{y_top + head} {cx + w / 4},{y_top + head} '
               f'{cx + w / 4},{y_bot - head} {cx + w / 2},{y_bot - head} {cx},{y_bot} {cx - w / 2},{y_bot - head} '
               f'{cx - w / 4},{y_bot - head} {cx - w / 4},{y_top + head} {cx - w / 2},{y_top + head}"/>')


def brace(x, y1, y2, label):
    m = (y1 + y2) / 2
    out.append(f'<path d="M{x},{y1} q12,0 12,12 L{x + 12},{m - 12} q0,12 12,12 q-12,0 -12,12 '
               f'L{x + 12},{y2 - 12} q0,12 -12,12" fill="none" stroke="{BRACE}" stroke-width="2.5"/>')
    for i, s in enumerate(label):
        text(x + 32, m + (i - (len(label) - 1) / 2) * 24, s, 21, TITLE, "500", "start")


# ── layout ──────────────────────────────────────────────────────────────────
L, R = 210, 1390                         # the main column
y = 0
rect(0, 0, W, 96, HEADER_BG, r=0)
text(W / 2, 48, "Cormorant — system architecture", 38, TITLE, "600")
y = 130

# applications
app_top = y
rect(L, y, R - L, 96, APP_FILL, APP_STROKE, 2.5, 4)
chips(L + 22, y + 16, R - L - 44, 64, [
    ["Image classification", "ResNet-18 · MobileNet"], ["Camera", "MobileNet V1, live"],
    ["MNIST", "ConvNet · LeNet"], ["Question answering", "BERT-base SQuAD"],
    ["Chat server", "OpenAI API · SmolLM2"], ["Text to speech", "Piper (VITS)"]],
    APP_CHIP, TITLE, gap=16, size=14, bold_first=True)
app_bot = y + 96
y = app_bot
for cx in (L + 230, (L + R) / 2, R - 230):
    arrow_down(cx, y + 10, y + 52)
note((L + R) / 2 + 30, y + 31, "call")
y += 62

# software
sw_top = y
rect(L, y, R - L, 92, SW_FILL, SW_STROKE, 2, 4)
text(L + 20, y + 30, "Model libraries", 19, TITLE, "600", "start")
text(L + 20, y + 58, "generated C, one per model", 13.5, "#4b5563", anchor="start")
chips(L + 270, y + 16, R - L - 290, 60, [
    ["CNN projects", "ResNet · MobileNet"], ["libbert_squad.so", "BERT-base"],
    ["libsmollm2.so", "SmolLM2 135M, 360M"], ["libsmolvlm_256m.so", "SmolVLM image chat"],
    ["libpiper_tts.so", "Piper TTS"]], SW_OLIVE, gap=14, size=13.5, bold_first=True)
lib_bot = y + 92
y += 152

half = (R - L - 20) / 2
rect(L, y, half, 250, SW_FILL, SW_STROKE, 2, 4)
text(L + half / 2, y + 28, "Runtime — on the board", 19, TITLE, "600")
cw = (half - 60) / 2
rows = [[["XRT buffers in CMA", "cache clean / invalidate"], ["Kernel drivers", "Xilinx driver API, over UIO"]],
        [["kernel_wait", "kernels on different lanes overlap"], ["Host ops in C", "Softmax · LayerNorm · attention"]],
        [["Per-layer profiler", "INFERENCE_PROFILING"], ["Linux on the A53", "Ubuntu 22.04 · XRT 2.13"]]]
for r_i, row in enumerate(rows):
    for c_i, lines in enumerate(row):
        chip(L + 20 + c_i * (cw + 20), y + 52 + r_i * 66, cw, 56, lines, SW_CHIP, size=13.5, bold_first=True)
x2 = L + half + 20
rect(x2, y, half, 250, SW_FILL, SW_STROKE, 2, 4)
text(x2 + half / 2, y + 28, "Inference scheduler — design time, on the host", 19, TITLE, "600")
rows = [[["Frontends", "ONNX · Llama · ViT · Piper"], ["Graph + fusion", "LayerNorm · GELU"]],
        [["Kernel mapping", "MatMul → ConvKernel · image"], ["DAG + event stream", "buffer pool coloring"]],
        [["Fixed-point simulator", "the bit-exact expectations"], ["C code generator", "planning with --plan"]]]
for r_i, row in enumerate(rows):
    for c_i, lines in enumerate(row):
        chip(x2 + 20 + c_i * (cw + 20), y + 52 + r_i * 66, cw, 56, lines, SW_CHIP, size=13.5, bold_first=True)
gx = x2 + half / 2                       # the scheduler writes the libraries' source
arrow_up(gx, lib_bot + 8, y - 8, w=38, fill=GEN)
note(gx + 32, (lib_bot + y) / 2, "generates the C source of every library", GEN, 14.5)
note(L + half / 2 + 32, (lib_bot + y) / 2, "built on the runtime", "#4b5563", 14)
arrow_down(L + half / 2, lib_bot + 8, y - 8)
sw_bot = y + 250
y = sw_bot
for cx in (L + 230, (L + R) / 2, R - 230):
    arrow_down(cx, y + 10, y + 52)
note((L + R) / 2 + 30, y + 31, "drive the kernels: AXI-Lite registers · start · wait")
y += 62

# hardware
hw_top = y
rect(L, y, R - L, 110, HW_FILL, HW_STROKE, 2.5, 4)
text(L + 20, y + 40, "FPGA kernels", 19, TITLE, "600", "start")
text(L + 20, y + 68, "HLS + SystemVerilog", 13.5, "#334155", anchor="start")
chips(L + 230, y + 18, R - L - 250, 74, [
    ["VectorOPKernel", "element-wise, 8 lanes"], ["MatmulKernel", "SystemVerilog, 128 MACs"],
    ["ConvKernel", "16 × 16 MAC grid, 512 MACs"], ["PoolingKernel", "8 channel lanes"]],
    HW_CHIP, TITLE, gap=18, size=14, bold_first=True)
y += 110
for cx in (L + 330, (L + R) / 2, R - 330):
    arrow_both(cx, y + 8, y + 58)
note((L + R) / 2 + 30, y + 33, "data: 128-bit AXI masters ⇄ DDR")
y += 66
rect(L, y, R - L, 96, "#ffffff", "#c9cbd1", 2, 4)
text(L + 20, y + 36, "Kria KV260", 19, TITLE, "600", "start")
text(L + 20, y + 62, "Zynq UltraScale+ K26", 13.5, "#4b5563", anchor="start")
chips(L + 230, y + 18, R - L - 250, 60, [
    ["Programmable logic", "kernels at 100 MHz"], ["4 × Arm Cortex-A53", "Linux, host ops"],
    ["4 GB DDR4", "1 GB CMA for the kernels"], ["AXI HPC0 / HPC1", "weights on their own port"]],
    BOARD_CHIP, TITLE, gap=18, size=14, bold_first=True)
hw_bot = y + 96
y = hw_bot + 18
rect(L, y, R - L, 44, "#e6dcfb", r=4)
text((L + R) / 2, y + 22, "Bitstream and device-tree overlay — Vivado block design (hw/cormorant_hw_128)", 16, "#5b3fc4", "500")
bot = y + 44

# pillars: cross-cutting
for i, (title, sub) in enumerate((("Verification", "bit-exact: simulator · C-sim · RTL · board · fact registry"),
                                  ("Performance", "measured models · per-layer profiler · timeline"))):
    px = 40 + i * 84
    rect(px, sw_top, 70, bot - sw_top, (PILLAR_V, PILLAR_P)[i], r=4)
    cy = (sw_top + bot) / 2
    text(px + 25, cy, title, 21, "#ffffff", "bold", rotate=-90)
    text(px + 49, cy, sub, 14.5, "#ffffff", "500", rotate=-90)

# braces
brace(R + 18, app_top, app_bot, ["Applications"])
brace(R + 18, sw_top, sw_bot, ["Software"])
brace(R + 18, hw_top, bot, ["Hardware"])

if overflow:
    sys.exit("labels that do not fit:\n  " + "\n  ".join(overflow))
H = bot + 36
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'font-family="{FONT}">\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(out) + "\n</svg>\n")
OUT.write_text(svg)
print(f"{OUT} ({W} x {H:.0f})")
