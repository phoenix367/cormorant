#!/usr/bin/env python3
"""Work per engine in the schedule the inference scheduler generates for a
model (BERT_PLAN phase 1 gate (c) / phase-2 sizing): kernel calls and MACs,
VectorOP element ops, host-op element counts and DMA-buffer traffic, and a
first-order latency estimate.

usage:  inference-scheduler/.venv/bin/python demo/bert_squad/scripts/bert_schedule_stats.py [model.onnx]
        (default: $BERT_SQUAD_MODEL or inference-scheduler/bertsquad-12-simplified.onnx)

MatMuls are listed per engine (BERT_PLAN §2 2A: MatmulKernel, or ConvKernel
with swapped operand roles) with the scheduler's engine cost model
(inference-scheduler/src/cost_model.py: the conv cycle model and the
board-calibrated MatmulKernel block model, cycles at 100 MHz) — the same
numbers the engine choice used.

Rates used for the rest of the estimate (edit below): VectorOP 8 elem/cycle at 100 MHz, host memcpy to /
from the non-cacheable BO 1 GB/s, host double math per element (A53 @ 1.3
GHz, one core): exp 25 ns, tanh / erf 30 ns, LayerNorm 12 ns, cached copies
2 ns.  They are placeholders for the phase-1 board profile.
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "inference-scheduler"))

from src.codegen import CodeGenerator                     # noqa: E402
from src.graph import OnnxGraph                            # noqa: E402
from src.host_nodes import GatherNode, HostNode, SliceNode  # noqa: E402
from src.cost_model import CALL_OVERHEAD, matmul_cycles    # noqa: E402
from src.nodes import MatmulConvNode, MatmulNode, ScheduledNode  # noqa: E402

CLOCK_HZ = 100e6
VOP_ELEMS_PER_S = 8 * 100e6
BO_BYTES_PER_S = 1.0e9
NS_PER_ELEM = {"Softmax": 25e-9, "Gelu": 30e-9, "LayerNormalization": 12e-9}
COPY_NS = 2e-9


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "BERT_SQUAD_MODEL", os.path.join(REPO, "inference-scheduler", "bertsquad-12-simplified.onnx"))
    g = OnnxGraph(model, fuse_act=True, s2d_stem=True)
    cg = CodeGenerator(g, model_path=model)
    bpe = cg._dtype.bytes_per_elem

    vop_calls = 0
    # (engine, kind) -> nodes / calls / MACs / model cycles on the chosen engine /
    # model cycles MatmulKernel would take
    mm = collections.defaultdict(collections.Counter)
    geo = collections.defaultdict(collections.Counter)
    vop = collections.Counter()
    host = collections.defaultdict(lambda: collections.Counter())
    for sn in g.nodes:
        if isinstance(sn, (MatmulNode, MatmulConvNode)):
            macs = sn.n * sn.k * sn.m * sn.batch * sn.outer_count
            kind = "linear (constant B)" if sn.inputs[1].is_weight else "attention (runtime B)"
            if isinstance(sn, MatmulConvNode):
                eng, calls, cyc, mm_cyc = "ConvKernel", sn.calls, sn.est_conv_cycles, sn.est_matmul_cycles
                geo[(kind, f"{sn.n}x{sn.k}.{sn.k}x{sn.m}" + (f" b{sn.batch}" if sn.batch > 1 else ""),
                     f"1x{sn.kw} s(1,{sn.kw}) in_ch {sn.k // sn.kw} out {sn.out_h}x{sn.out_w}"
                     + (f" x{sn.calls} calls" if sn.calls > 1 else ""))].update(
                    nodes=1, cyc=sn.est_conv_cycles, mm_cyc=sn.est_matmul_cycles)
            else:
                eng, calls = "MatmulKernel", sn.outer_count
                cyc = mm_cyc = (matmul_cycles(sn.n, sn.k, sn.m, sn.batch * sn.outer_count)
                                + CALL_OVERHEAD)
            c = mm[(eng, kind)]
            c.update(nodes=1, calls=calls, macs=macs, cyc=cyc, mm_cyc=mm_cyc)
        elif isinstance(sn, ScheduledNode):
            vop[sn.onnx_node.op_type] += sn.output.numel
            vop_calls += 1
        elif isinstance(sn, HostNode) and not (isinstance(sn, SliceNode) and sn.is_view):
            op = sn.onnx_node.op_type
            ins, (_o, o_cnt, _io), _s, _t = cg._host_stage_plan(sn)
            in_bytes = sum(cnt for _t_, _off, cnt, _io_ in ins) * bpe
            if isinstance(sn, GatherNode):
                in_bytes += sn.n_idx * sn.row_len * bpe          # rows straight from the table BO
            h = host[op]
            h["nodes"] += 1
            h["elems"] += sn.output.numel
            h["bo_bytes"] += in_bytes + o_cnt * bpe

    print(f"model: {os.path.basename(model)}   nodes: {len(g.nodes)}   "
          f"pool: {cg._compute_pool_layout()[1] * bpe / 2**20:.1f} MiB   "
          f"weights: {sum(t.numel for t in g.weight_tensors) * bpe / 2**20:.1f} MiB")
    t_mm = sum(c["cyc"] for c in mm.values()) / CLOCK_HZ
    t_mm_only = sum(c["mm_cyc"] for c in mm.values()) / CLOCK_HZ
    print(f"\nMatMuls ({sum(c['nodes'] for c in mm.values())}, "
          f"{sum(c['macs'] for c in mm.values()) / 1e9:.2f} GMAC) — engine cost model at 100 MHz: "
          f"{t_mm:.3f} s  (all on MatmulKernel: {t_mm_only:.2f} s)")
    print(f"  {'engine':13s} {'kind':22s} {'nodes':>5s} {'calls':>5s} {'GMAC':>7s} {'model':>9s}"
          f" {'MatmulKernel':>13s}")
    for (eng, kind), c in sorted(mm.items()):
        print(f"  {eng:13s} {kind:22s} {c['nodes']:5d} {c['calls']:5d} {c['macs'] / 1e9:7.2f}"
              f" {c['cyc'] / CLOCK_HZ * 1e3:7.1f} ms {c['mm_cyc'] / CLOCK_HZ * 1e3:10.1f} ms")
    if geo:
        print("  ConvKernel geometries (per node: model ms on ConvKernel / on MatmulKernel):")
        for (_kind, shape, conv), c in sorted(geo.items()):
            print(f"    {c['nodes']:3d} x {shape:24s} {conv:42s}"
                  f" {c['cyc'] / c['nodes'] / CLOCK_HZ * 1e3:7.2f} / "
                  f"{c['mm_cyc'] / c['nodes'] / CLOCK_HZ * 1e3:7.2f} ms")
    vop_elems = sum(vop.values())
    print(f"\nVectorOPKernel: {vop_calls} calls, {vop_elems / 1e6:.1f} M element ops "
          f"(~{vop_elems / VOP_ELEMS_PER_S * 1e3:.0f} ms)")
    for k, v in sorted(vop.items()):
        print(f"  {k:6s} {v / 1e6:8.2f} M")
    t_host = 0.0
    print("\nHost CPU ops:")
    print(f"  {'op':20s} {'nodes':>5s} {'elements':>12s} {'BO traffic':>11s} {'est.':>8s}")
    for op, h in sorted(host.items()):
        t = h["bo_bytes"] / BO_BYTES_PER_S + h["elems"] * NS_PER_ELEM.get(op, COPY_NS)
        t_host += t
        print(f"  {op:20s} {h['nodes']:5d} {h['elems'] / 1e6:10.2f} M {h['bo_bytes'] / 2**20:8.1f} MiB"
              f" {t * 1e3:6.0f} ms")
    tot_bo = sum(h["bo_bytes"] for h in host.values())
    print(f"  {'total':20s} {sum(h['nodes'] for h in host.values()):5d} "
          f"{sum(h['elems'] for h in host.values()) / 1e6:10.2f} M {tot_bo / 2**20:8.1f} MiB"
          f" {t_host * 1e3:6.0f} ms")
    t_vop = vop_elems / VOP_ELEMS_PER_S
    print(f"\nfirst-order latency (serial): MatMuls {t_mm:.2f} s + VectorOP {t_vop:.2f} s"
          f" + host {t_host:.2f} s = {t_mm + t_vop + t_host:.1f} s")


if __name__ == "__main__":
    main()
