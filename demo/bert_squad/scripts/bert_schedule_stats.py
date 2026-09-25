#!/usr/bin/env python3
"""Work per engine in the schedule the inference scheduler generates for a
model (BERT_PLAN phase 1 gate (c) / phase-2 sizing): kernel calls and MACs,
VectorOP element ops, host-op element counts and DMA-buffer traffic, and a
first-order latency estimate.

usage:  inference-scheduler/.venv/bin/python demo/bert_squad/scripts/bert_schedule_stats.py [model.onnx]
        (default: $BERT_SQUAD_MODEL or inference-scheduler/bertsquad-12-simplified.onnx)

Rates used for the estimate (edit below): MatmulKernel 2.3 GMAC/s (measured
on 256^3, BERT_PLAN §0), VectorOP 8 elem/cycle at 100 MHz, host memcpy to /
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
from src.nodes import MatmulNode, ScheduledNode            # noqa: E402

MM_MACS_PER_S = 2.3e9
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

    mm_calls = vop_calls = 0
    mm = collections.Counter()
    vop = collections.Counter()
    host = collections.defaultdict(lambda: collections.Counter())
    for sn in g.nodes:
        if isinstance(sn, MatmulNode):
            macs = sn.n * sn.k * sn.m * sn.batch * sn.outer_count
            kind = "linear (constant B)" if sn.inputs[1].is_weight else "attention (runtime B)"
            mm[kind] += macs
            mm[kind + " calls"] += sn.outer_count
            mm_calls += sn.outer_count
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

    total_macs = sum(v for k, v in mm.items() if not k.endswith("calls"))
    print(f"model: {os.path.basename(model)}   nodes: {len(g.nodes)}   "
          f"pool: {cg._compute_pool_layout()[1] * bpe / 2**20:.1f} MiB   "
          f"weights: {sum(t.numel for t in g.weight_tensors) * bpe / 2**20:.1f} MiB")
    print(f"\nMatmulKernel: {mm_calls} calls, {total_macs / 1e9:.2f} GMAC")
    for k in sorted(k for k in mm if not k.endswith("calls")):
        print(f"  {k:24s} {mm[k + ' calls']:4d} calls  {mm[k] / 1e9:7.2f} GMAC"
              f"  ~{mm[k] / MM_MACS_PER_S:6.2f} s")
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
    t_mm = total_macs / MM_MACS_PER_S
    t_vop = vop_elems / VOP_ELEMS_PER_S
    print(f"\nfirst-order latency (serial): MatmulKernel {t_mm:.1f} s + VectorOP {t_vop:.2f} s"
          f" + host {t_host:.2f} s = {t_mm + t_vop + t_host:.1f} s")


if __name__ == "__main__":
    main()
