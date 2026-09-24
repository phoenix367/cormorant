#!/usr/bin/env python3
"""ConvKernel cycle model (architecture as of CONV_OPTIMISATION.md §2.34).
See SKILL.md for usage.  Constants come from platforms/<AXI_PLATFORM>.json."""
import argparse, json, math, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

def load_platform(name):
    cfg = json.load(open(os.path.join(ROOT, "platforms", f"{name}.json")))
    c = cfg["kernels"]["conv"]
    return dict(TILE_M=c["tile_m"], TILE_IC=c["tile_ic"], ROWS=c["max_line_buf_rows"],
                COLS=c["max_line_buf_cols"], ACC=c["max_acc_persist_entries"],
                MPG=c["max_m_per_group"], CLOCK=cfg.get("clock", 150), PORT_ELEMS=8)

def geom(P, in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw, dwise):
    m_tiles = -(-out_ch // P["TILE_M"]); ic_tiles = -(-in_ch // P["TILE_IC"])
    mtg = min(P["MPG"], m_tiles); groups = -(-m_tiles // mtg)
    per = max(1, min(oh, P["ACC"] // (ow * m_tiles * P["TILE_M"])))
    if not dwise and groups > 1:
        win = (kh - 1) * dh + 1
        cap = (P["ROWS"] - win) // sh + 1 if win < P["ROWS"] else 1
        per = min(per, cap)
    chunks = -(-oh // per)
    winw = (kw - 1) * dw + 1
    owpt = min(ow, (P["COLS"] - winw) // sw + 1 if winw < P["COLS"] else 1)
    owt = -(-ow // owpt)
    return m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt

INVOKE_OVERHEAD = 1000   # geometry dividers, bias load, DATAFLOW start-up, first-burst latencies
ROW_LOAD_LATENCY = 40    # first read data after a row's requests are issued

def model_layer(P, in_ch, out_ch, in_h, in_w, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl, dwise):
    """Cycle model.  The patch producer is SEQUENTIAL per row: it loads a row's
    channel runs from DDR (one 16-bit element per cycle, requests pipelined)
    and only then streams that row's patches, and the patch FIFO holds only
    ~5 pixels, so input loads are serial with compute, not overlapped."""
    m_tiles, ic_tiles, mtg, groups, per, chunks, owpt, owt = geom(P, in_ch, out_ch, oh, ow, kh, kw, sh, sw, dh, dw, dwise)
    E, T = P["PORT_ELEMS"], P["TILE_IC"]
    sweep = fill = ph1 = ph3 = loads = 0
    for c in range(chunks):
        rows = min(per, oh - c * per)
        ph1 += rows * ow * m_tiles
        ph3 += rows * ow * out_ch                       # 1 element/cycle drain (+ writer at same rate)
        r0 = c * per * sh - pt
        r1 = (c * per + rows - 1) * sh + (kh - 1) * dh - pt
        in_rows = max(0, min(r1, in_h - 1) - max(r0, 0) + 1)   # rows actually fetched for this chunk
        if dwise:
            for mt in range(m_tiles):
                mv = min(P["TILE_M"], out_ch - mt * P["TILE_M"])
                fill += mv * math.ceil(kh * kw / E)             # one beat per 8 positions
                for t in range(owt):
                    tw = min(owpt, ow - t * owpt)
                    cols = min(in_w, (tw - 1) * sw + (kw - 1) * dw + 1)
                    sweep += rows * tw * (kh * kw + 6)
                    loads += in_rows * (mv * cols + ROW_LOAD_LATENCY)
        else:
            for ict in range(ic_tiles):
                icv = min(T, in_ch - ict * T)
                lanes = E if (ict == ic_tiles - 1 and icv <= E) else T
                for t in range(owt):
                    tw = min(owpt, ow - t * owpt)
                    cols = min(in_w, (tw - 1) * sw + (kw - 1) * dw + 1)
                    loads += in_rows * (icv * cols + ROW_LOAD_LATENCY)   # grp 0 only
                    for g in range(groups):
                        mt0 = g * mtg; G = min(mtg, m_tiles - mt0)
                        mv_sum = sum(min(P["TILE_M"], out_ch - (mt0 + i) * P["TILE_M"]) for i in range(G))
                        fill += mv_sum * kh * kw * lanes / E        # beats at 1/cycle
                        sweep += rows * tw * (G * kh * kw + 2 * G + 6)
    total = sweep + fill + ph1 + ph3 + loads + INVOKE_OVERHEAD
    return dict(total=total, sweep=sweep, fill=fill, ph1=ph1, ph3=ph3, loads=loads,
                chunks=chunks, rows=per, groups=groups, ic_tiles=ic_tiles, owt=owt)

def fmt_layer(name, r, P):
    t = r["total"]
    return (f"{name[:40]:40s} {t/1e3:9.1f} k cyc {t/P['CLOCK']/1e3:7.2f} ms | "
            f"MAC {100*r['sweep']/t:4.0f}% fill {100*r['fill']/t:4.0f}% bias {100*r['ph1']/t:3.0f}% "
            f"drain/write {100*r['ph3']/t:4.0f}% loads {100*r['loads']/t:4.0f}% | "
            f"chunks {r['chunks']}x{r['rows']} groups {r['groups']} ic_tiles {r['ic_tiles']} owt {r['owt']}")

def run_models(P, paths):
    import onnx
    from onnx import shape_inference
    for path in paths:
        m = shape_inference.infer_shapes(onnx.load(path)); g = m.graph
        sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(g.value_info) + list(g.input) + list(g.output)}
        init = {i.name: i for i in g.initializer}
        tot = dict(total=0, sweep=0, fill=0, ph1=0, ph3=0, loads=0); rows = []
        for n in g.node:
            if n.op_type != "Conv": continue
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            M, _, kh, kw = init[n.input[1]].dims; dwise = a.get("group", 1) > 1
            C = sh[n.input[0]][1]; ih, iw = sh[n.input[0]][2:4]; oh, ow = sh[n.output[0]][2:4]
            s_ = a.get("strides", [1, 1]); d_ = a.get("dilations", [1, 1]); p_ = a.get("pads", [0, 0, 0, 0])
            r = model_layer(P, C, M, ih, iw, oh, ow, kh, kw, s_[0], s_[1], d_[0], d_[1], p_[0], p_[1], dwise)
            for k in tot: tot[k] += r[k]
            rows.append((r["total"], fmt_layer(n.name or n.output[0], r, P)))
        print(f"\n{os.path.basename(path)}: {len(rows)} Conv, total {tot['total']/1e6:.2f} M cycles = {tot['total']/P['CLOCK']/1e3:.0f} ms @{P['CLOCK']} MHz | "
              f"MAC {100*tot['sweep']/tot['total']:.0f}% fill {100*tot['fill']/tot['total']:.0f}% bias {100*tot['ph1']/tot['total']:.0f}% drain/write {100*tot['ph3']/tot['total']:.0f}% loads {100*tot['loads']/tot['total']:.0f}%")
        for _, line in sorted(rows, reverse=True)[:8]: print("  " + line)

def run_validate(P, report):
    d = json.load(open(report))
    print(f"{'case':58s} {'model':>9s} {'measured':>9s}  delta")
    errs = []
    for t in d["tests"]:
        g = t.get("geometry") or {}
        try:
            r = model_layer(P, g["in_ch"], g["out_ch"], g["in_h"], g["in_w"], g["out_h"], g["out_w"], g["kh"], g["kw"],
                            g["stride_h"], g["stride_w"], g["dilation_h"], g["dilation_w"], g["pad_top"], g["pad_left"], bool(g["is_depthwise"]))
        except KeyError:
            print(f"{t['label'][:58]:58s}  (geometry fields missing in report)"); continue
        meas = t["duration_ns"] / 10.0
        model = (r["total"] - INVOKE_OVERHEAD) * g.get("batch", 1) + INVOKE_OVERHEAD
        errs.append(abs(model - meas) / meas)
        print(f"{t['label'][:58]:58s} {model/1e3:8.1f}k {meas/1e3:8.1f}k {100*(model-meas)/meas:+6.1f}%")
    big = [e for e, t in zip(errs, d["tests"]) if t["duration_ns"] > 200000]
    print(f"mean |error| all: {100*sum(errs)/len(errs):.1f} %   cases > 20 k cycles: {100*sum(big)/max(1,len(big)):.1f} %")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--platform", default=os.environ.get("AXI_PLATFORM", "kv260"))
    ap.add_argument("--case", nargs="+", type=int, metavar="N", help="C M H W kh kw [sh sw dh dw pt pl pb pr dw]")
    ap.add_argument("--validate", metavar="conv_test_report.json")
    a = ap.parse_args(); P = load_platform(a.platform)
    if a.case:
        v = a.case + [1, 1, 1, 1, 0, 0, 0, 0, 0][len(a.case) - 6:]
        C, M, H, W, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, dwise = v[:15]
        oh = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1; ow = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
        r = model_layer(P, C, M, H, W, oh, ow, kh, kw, sh, sw, dh, dw, pt, pl, bool(dwise))
        print(fmt_layer(f"C={C} M={M} {H}x{W}->{oh}x{ow} {kh}x{kw} s{sh}", r, P))
    if a.validate: run_validate(P, a.validate)
    if a.models: run_models(P, a.models)

if __name__ == "__main__":
    main()
