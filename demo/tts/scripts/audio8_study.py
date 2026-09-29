#!/usr/bin/env python3
"""audio8_study.py — model study of Audio8 TTS Preview 0.1B (model-study
skill, route C; doc/plans/TTS_PLAN.md §1).  Host only.  The study stops at
the speed gate, so no weights are downloaded:

  fetch    config, modeling code, license and README at the pinned revision
           (SHA-256 checked) and the model.safetensors header (an HTTP range
           request)
  params   parameters per component from the header; the codec decoder's
           from its architecture (modeling_arktts_codec.py)
  costs    one audio frame (1 slow step + 10 fast passes, 9 fast heads) and
           one second of codec decoding priced by the bitstream's
           performance model (proxy graphs with the real shapes, split at the
           kernel bounds, built in memory); the host-side estimates; the
           weight bandwidth real time would need
  all      fetch, params, costs

usage: inference-scheduler/.venv/bin/python demo/tts/scripts/audio8_study.py
           [--assets DIR] [--perf-model FILE] {fetch,params,costs,all}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCHED = os.path.join(REPO, "inference-scheduler")
sys.path.insert(0, SCHED)

HF_REPO = "Edge0/Audio8-TTS-Preview-0.1b"
REVISION = "b476f0208438dfa791abee44d11029f055aeae04"      # 2026-08-25
LICENSE = "Audio8 Community License v1.0 (non-commercial free; commercial < US$2M revenue)"
FILES = {
    "config.json": "65c299d24934dcb47c34cea10c39ad401a817fb785f041dde37b8dde65b77e4e",
    "configuration_arktts.py": "7936ca9839bdf4f0a0cfa890b62f646ed94b162b7e351d18b89d1bb1f4542c23",
    "modeling_arktts.py": "50bc8f9adffa79b231c8b9e8fe7a56cc9b304bd46d1c890fbd24b0d1fd954b12",
    "modeling_arktts_codec.py": "c9d579f1a876cbceb3f7ef501dfccc17a4986933ba758de36c84450ba4905925",
    "generation_config.json": "f7ae691ad4bfd17aa206118c0c81ce8960da1562b20429a36e8e028820839412",
    "LICENSE": "d6386c15f43c086f5eec426bd0037cd9628490049cca023e4c0ac9e2a4559f1a",
    "README.md": "5c3e244037d6850c517b4bdb8b86ff7d17acca2abc8518c4c8d408f89aebce37",
}
LFS = {   # not downloaded; the Hub's LFS SHA-256 for the record
    "model.safetensors": (339605208, "69b162eb71da66b3a3f6ed7f9aa5e623853526717fd798d347b34919d58a4734"),
    "codec.pth": (1349857559, "c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048"),
}
HEADER_SHA256 = "ae8da6eeb8056544518e2ab285c8374a1084c9172953420e7544ca057433b6ed"

SR, FRAME = 44100, 2048                    # codec: 44.1 kHz, 2048 samples per model frame
FPS = SR / FRAME                           # 21.53 frames per second of audio
LATENT = SR / 512                          # codec latent rate (hop 512), 86.13 Hz
# Host-side estimates (not priced by the performance model):
#  - AR host ops per frame: 24 slow + 40 fast layer-steps at the SmolLM2 decode's
#    0.2-0.45 ms per layer-step (norms, attention, SiLU, residual; the Mamba-2
#    step adds ~50 k MACs per layer) plus ~10 top-p samplings over 4096 logits
AR_HOST_MS = (18.0, 30.0)
#  - Snake (x + sin^2(a x) / a) on the A53, 4 threads, double sin
SNAKE_NS = (5.0, 13.0)
#  - the 1x7 dilated convs are not in the performance model's families: bounded
#    by the priced 1x1 convs' rate (low) and ResNet-18's 3x3 rate (high)
K7_GMACS = (13.4, 30.0)
PORT_GBS = 2.94                            # GEMV weight streaming, both ports (priced)
DDR_PEAK_GBS = 19.2                        # K26 SOM: 64-bit DDR4-2400


def default_assets() -> str:
    return os.path.join(REPO, "demo", "tts", "assets", "audio8-tts-preview-0.1b")


def _url(name: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/{REVISION}/{name}"


def _get(url: str, rng=None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def cmd_fetch(a) -> int:
    os.makedirs(a.assets, exist_ok=True)
    bad = 0
    for name, sha in FILES.items():
        p = os.path.join(a.assets, name)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(_get(_url(name)))
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        ok = got == sha
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name}")
    hp = os.path.join(a.assets, "model.safetensors.header.json")
    if not os.path.exists(hp):
        n = struct.unpack("<Q", _get(_url("model.safetensors"), (0, 7)))[0]
        with open(hp, "wb") as f:
            f.write(_get(_url("model.safetensors"), (8, 8 + n - 1)))
    ok = hashlib.sha256(open(hp, "rb").read()).hexdigest() == HEADER_SHA256
    bad += not ok
    print(f"  {'OK ' if ok else 'BAD'} model.safetensors header")
    print(f"{HF_REPO} @ {REVISION}; {LICENSE}")
    return 1 if bad else 0


def _header(a) -> dict:
    h = json.load(open(os.path.join(a.assets, "model.safetensors.header.json")))
    h.pop("__metadata__", None)
    return h


def component(name: str) -> str:
    if "embed" in name or name.startswith("codebook_embeddings"):
        return "embedding tables (host, one row per token)"
    if name.startswith("slow.") or name.startswith("semantic_output"):
        return "slow step (once per frame)"
    return "fast step (10 passes per frame)"


def codec_decoder_params() -> dict:
    """Parameters of the decode path (modeling_arktts_codec.py)."""
    post = 8 * (1024 * (16 * 64 + 2 * 8 * 64) + 1024 * 1024 + 3 * 1024 * 1216)
    up = 2 * (1024 * 1024 * 2 + 2 * 1024 * 4096 + 1024 * 7)
    dac = 1024 * 1536 * 7
    c = 1536
    for s in (8, 8, 4, 2):
        o = c // 2
        dac += c * o * 2 * s + 3 * (o * o * 7 + o * o)
        c = o
    dac += c * 7
    return {"post transformer (8 layers, dim 1024)": post, "upsampler (2 x ConvNeXt)": up,
            "DAC decoder (convs)": dac}


def cmd_params(a) -> int:
    h = _header(a)
    per = Counter()
    for k, m in h.items():
        per[component(k)] += int(np.prod(m["shape"]))
    tot = sum(per.values())
    print(f"main model {tot / 1e6:.1f} M parameters (bf16):")
    for k, v in per.most_common():
        print(f"  {v / 1e6:7.1f} M  {k}")
    cd = codec_decoder_params()
    print(f"codec decoder {sum(cd.values()) / 1e6:.1f} M parameters:")
    for k, v in cd.items():
        print(f"  {v / 1e6:7.1f} M  {k}")
    return 0


# ---- proxy graphs ------------------------------------------------------- #

def _matmuls(mats, seed=0):
    from onnx import TensorProto as TP, helper as oh, numpy_helper as nph
    rng = np.random.default_rng(seed)
    nodes, inits, outs, ins = [], [], [], {}
    for i, (tag, k, m) in enumerate(mats):
        x = f"x{k}"
        ins.setdefault(x, oh.make_tensor_value_info(x, TP.FLOAT, [1, k]))
        w, y = f"w{i}_{tag}", f"y{i}_{tag}"
        inits.append(nph.from_array((rng.standard_normal((k, m)) * 0.02).astype(np.float32), w))
        nodes.append(oh.make_node("MatMul", [x, w], [y], name=f"mm{i}_{tag}"))
        outs.append(oh.make_tensor_value_info(y, TP.FLOAT, [1, m]))
    m = oh.make_model(oh.make_graph(nodes, "proxy", list(ins.values()), outs, initializer=inits),
                      opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def slow_step():
    mats = []
    for _ in range(24):
        mats += [("q", 512, 512), ("k", 512, 128), ("v", 512, 128), ("o", 512, 512),
                 ("mamba_in", 512, 1688), ("mamba_out", 768, 512),
                 ("gate", 512, 768), ("up", 512, 768), ("down", 768, 512)]
    return mats + [("semantic_head", 512, 4097)]


def fast_layers():
    mats = []
    for _ in range(4):        # FFN 4864 split at gemv_max_m / max_k 4096
        mats += [("wqkv", 512, 768), ("wo", 512, 512),
                 ("w1a", 512, 2432), ("w1b", 512, 2432), ("w3a", 512, 2432), ("w3b", 512, 2432),
                 ("w2a", 2432, 512), ("w2b", 2432, 512)]
    return mats


def codec_dac_1s():
    """The DAC decoder for one second of audio: 1-D convs as H = 1 convs with
    time folded into rows (w0 outputs per row, the row's left context
    duplicated — ConvKernel keeps one padded output row of out_w * out_ch <=
    65 536 accumulators); transposed convs as their polyphase kernel-2 convs
    (s * C_out channels, the same MACs); channels split at in <= 1024 /
    out <= 1280."""
    from onnx import TensorProto as TP, helper as oh, numpy_helper as nph
    rng = np.random.default_rng(0)
    nodes, inits, ins, outs, info = [], [], [], [], []

    def conv(tag, cin, cout, k, dil, W):
        for i0 in range(0, cin, 1024):
            ci = min(1024, cin - i0)
            for o0 in range(0, cout, 1280):
                co = min(1280, cout - o0)
                co16 = -(-co // 16) * 16
                w0 = 1
                while w0 * 2 * co16 <= 65536 and w0 * 2 <= W:
                    w0 *= 2
                H = -(-W // w0)
                n = len(nodes)
                x, w, y = f"x{n}", f"w{n}", f"y{n}"
                ins.append(oh.make_tensor_value_info(x, TP.FLOAT, [1, ci, H, w0 + (k - 1) * dil]))
                inits.append(nph.from_array(
                    (rng.standard_normal((co, ci, 1, k)) * 0.02).astype(np.float32), w))
                nodes.append(oh.make_node("Conv", [x, w], [y], name=f"c{n}_{tag}",
                                          kernel_shape=[1, k], dilations=[1, dil]))
                outs.append(oh.make_tensor_value_info(y, TP.FLOAT, [1, co, H, w0]))
                info.append((f"c{n}_{tag}", k, ci * co * k * H * w0))

    lat = round(LATENT)
    conv("conv0", 1024, 1536, 7, 1, lat)
    W, cin = lat, 1536
    for s in (8, 8, 4, 2):
        cout = cin // 2
        conv(f"up{s}", cin, s * cout, 2, 1, W)
        W *= s
        for d in (1, 3, 9):
            conv(f"ru{cout}d{d}", cout, cout, 7, d, W)
            conv(f"ru{cout}p", cout, cout, 1, 1, W)
        cin = cout
    conv("out", cin, 1, 7, 1, W)
    m = oh.make_model(oh.make_graph(nodes, "codec", ins, outs, initializer=inits),
                      opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    return m, info


def snake_per_second() -> float:
    """Snake evaluations per second of audio: before each transposed conv and
    both convs of every residual unit, plus the output."""
    n, W, c = 0.0, LATENT, 1536
    for s in (8, 8, 4, 2):
        n += c * W
        c, W = c // 2, W * s
        n += 6 * c * W
    return n + c * W


def _priced(model, pm):
    from src.codegen import CodeGenerator
    from src.graph import OnnxGraph
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True)
    cg = CodeGenerator(g, model_path="proxy.onnx")
    out = {}
    for sn in g.nodes:
        b = pm.calls_band(sn.kernel_calls(cg._layouts))
        out[sn.onnx_node.name] = None if b is None else (b[0] / 1e3, b[1])
    return out


def _sum(prices):
    tot = sum(v[0] for v in prices.values() if v)
    band = sum(v[0] * v[1] for v in prices.values() if v) / tot if tot else 0.0
    return tot, band


def cmd_costs(a) -> int:
    from src.perf_model import PerfModel
    pm = PerfModel.load(a.perf_model)
    print(f"performance model {os.path.relpath(a.perf_model, REPO)}")
    slow = _priced(_matmuls(slow_step()), pm)
    fast = _priced(_matmuls(fast_layers()), pm)
    head = _priced(_matmuls([("fast_head", 512, 4096)]), pm)
    if any(v is None for d in (slow, fast, head) for v in d.values()):
        print("  unpriced MatMuls — the numbers below are incomplete")
    s_ms, s_b = _sum(slow)
    f_ms, f_b = _sum(fast)
    h_ms, _ = _sum(head)
    frame_k = s_ms + 10 * f_ms + 9 * h_ms
    w_slow = sum(k * m for _, k, m in slow_step())
    w_fast = 10 * sum(k * m for _, k, m in fast_layers()) + 9 * 512 * 4096
    print(f"slow step   {s_ms:7.2f} ms (±{s_b * 100:.0f} %), {w_slow / 1e6:.1f} M weights")
    print(f"fast pass   {f_ms:7.2f} ms + head {h_ms:.2f} ms; per frame 10 passes + 9 heads = "
          f"{10 * f_ms + 9 * h_ms:.1f} ms, {w_fast / 1e6:.1f} M weights")
    print(f"frame       {frame_k:7.1f} ms kernels + {AR_HOST_MS[0]:.0f}-{AR_HOST_MS[1]:.0f} ms host "
          f"for {1e3 / FPS:.1f} ms of audio")
    ar = [(frame_k + h) * FPS / 1e3 for h in AR_HOST_MS]
    print(f"  -> AR {ar[0]:.2f}-{ar[1]:.2f} s per second of audio")

    model, info = codec_dac_1s()
    prices = _priced(model, pm)
    priced_ms = sum(p[0] for p in prices.values() if p)
    priced_mac = sum(m for n, _, m in info if prices[n])
    rate_k2 = [m / (prices[n][0] / 1e3) for n, k, m in info if k == 2 and prices[n]]
    k2 = sum(m for n, k, m in info if k == 2 and not prices[n])
    k7 = sum(m for n, k, m in info if k == 7 and not prices[n])
    other = sum(m for n, k, m in info if k not in (2, 7) and not prices[n])
    r2 = rate_k2[0] if rate_k2 else priced_mac / (priced_ms / 1e3)
    k2_s = k2 / r2
    k7_s = [k7 / (r * 1e9) for r in reversed(K7_GMACS)]
    other_s = other / r2
    tot_mac = sum(m for _, _, m in info)
    print(f"codec DAC   {tot_mac / 1e9:.1f} GMAC per second of audio in {len(info)} convs: "
          f"{priced_mac / 1e9:.1f} GMAC priced ({priced_ms / 1e3:.2f} s), "
          f"{k2 / 1e9:.1f} GMAC polyphase convs at {r2 / 1e9:.1f} GMAC/s ({k2_s:.2f} s), "
          f"{k7 / 1e9:.1f} GMAC 1x7 convs at {K7_GMACS[0]}-{K7_GMACS[1]} GMAC/s "
          f"({k7_s[0]:.2f}-{k7_s[1]:.2f} s)" + (f", {other / 1e9:.1f} GMAC other" if other else ""))
    sn = snake_per_second()
    snake = [sn * ns * 1e-9 for ns in SNAKE_NS]
    codec = [priced_ms / 1e3 + k2_s + other_s + k7_s[i] + snake[i] for i in (0, 1)]
    print(f"  Snake     {sn / 1e6:.0f} M evaluations per second of audio: {snake[0]:.2f}-{snake[1]:.2f} s (host)")
    print(f"  -> codec {codec[0]:.2f}-{codec[1]:.2f} s per second of audio "
          f"(+ the post transformer / upsampler, ~0.05 s)")
    print(f"total       {ar[0] + codec[0]:.1f}-{ar[1] + codec[1]:.1f} s per second of audio in sequence; "
          f">= {max(ar[0], codec[0]):.1f} s with AR and codec on their own lanes")
    need = (w_slow + w_fast) * 2 * FPS / 1e9
    print(f"real time   the AR streams {(w_slow + w_fast) * 2 / 1e6:.0f} MB of int16 weights per frame: "
          f"{need:.1f} GB/s ({need / 2:.1f} GB/s at int8) against {PORT_GBS} GB/s through the kernels' "
          f"ports and {DDR_PEAK_GBS} GB/s DDR peak")
    return 0


def main(argv=None) -> int:
    from src.perf_calls import local_bitstream_id
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--assets", default=default_assets())
    ap.add_argument("--perf-model", default=None,
                    help="default: perf_models/kv260/<the local bitstream id>.json")
    ap.add_argument("cmd", choices=("fetch", "params", "costs", "all"))
    a = ap.parse_args(argv)
    if a.perf_model is None:
        a.perf_model = os.path.join(SCHED, "perf_models", "kv260", f"{local_bitstream_id()}.json")
    rc = 0
    for c in (("fetch", "params", "costs") if a.cmd == "all" else (a.cmd,)):
        print(f"== {c}")
        rc |= {"fetch": cmd_fetch, "params": cmd_params, "costs": cmd_costs}[c](a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
