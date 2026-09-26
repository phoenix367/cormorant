#!/usr/bin/env python3
"""
llm_attn_kernel_bench.py — what decode attention would cost on the FPGA
(policy p12, doc/CHAT_PLAN.md §10.5 item 5) next to the shipped host region
(xattn): the kernel calls the p12 decode path issues per layer and KV group,
timed on the board with the per-layer profiler.

Per KV group (3 q heads of 64 channels) and S cached keys, p12 decode runs
  scores  MatmulKernel [3][64] x [64][S]   B = the K cache kept transposed and
                                            tile-packed ([S/32][64][32]: the
                                            packed-B layout, b_packed = 1)
  softmax on the host (P at 2^-12, exp table)
  P.V     MatmulKernel [3][S] x [S][64]    B = the V cache, row-major
                                            (activation path, b_packed = 0)
i.e. 2 x 90 = 180 kernel calls per token for SmolLM2-135M, plus host
RoPE / softmax work and a cache flush / invalidate at every hand-off.  The
benchmark builds one graph per S with 3 independent copies of each MatMul
(the K / V caches as constants of the same byte size and layout), runs it
N times under the profiler and reports the mean time per call; the
per-token estimate is 90 x (t_qk + t_pv) + the host softmax.

usage: inference-scheduler/.venv/bin/python demo/chat/scripts/llm_attn_kernel_bench.py
           [--lens 64,256,512,1024] [--iters 200] [--board-lock FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_llm_project as glp                                  # noqa: E402
import llm_board                                                    # noqa: E402
import llm_project as lp                                            # noqa: E402
from inference_scheduler import main as sched_main                  # noqa: E402

from deploy_and_run import _stream_exec, board_lock                 # noqa: E402
from src.codegen import CodeGenerator                               # noqa: E402
from src.graph import OnnxGraph                                     # noqa: E402
from src.remote import RemoteSession                                # noqa: E402

REMOTE = "/tmp/llm_attn_bench"
BENCH_C = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "inference.h"
#include "inference_prof.h"
int main(int argc, char **argv)
{
    int it, n = argc > 1 ? atoi(argv[1]) : 200;
    inference_buf_t *q = NULL, *p = NULL, *v = NULL, *outs[6];
    struct timespec a, b;
    if (inference_init(INFERENCE_MATMULKERNEL_INSTANCE) != 0) return 1;
    inference_prof_init(inference_num_layers(), inference_layer_names_ptr());
    q = inference_buf_alloc(INFERENCE_Q_SIZE);
    p = inference_buf_alloc(INFERENCE_P_SIZE);
    v = inference_buf_alloc(INFERENCE_V_SIZE);
    for (it = 0; it < 6; it++) outs[it] = inference_buf_alloc(it < 3 ? INFERENCE_S0_SIZE : INFERENCE_O0_SIZE);
    inference_run(p, q, v, outs[3], outs[4], outs[5], outs[0], outs[1], outs[2]);
    inference_prof_reset();
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (it = 0; it < n; it++)
        inference_run(p, q, v, outs[3], outs[4], outs[5], outs[0], outs[1], outs[2]);
    clock_gettime(CLOCK_MONOTONIC, &b);
    printf("WALL_MS %.4f\n", ((b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6) / n);
    inference_prof_dump_json(stdout);
    inference_deinit();
    return 0;
}
"""


def model(S: int, seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    vi = oh.make_tensor_value_info
    nodes, inits = [], []
    for g in range(3):
        kt = rng.normal(0, 0.3, (64, S)).astype(np.float32)
        inits.append(nph.from_array(kt, f"kt{g}"))
        nodes.append(oh.make_node("MatMul", ["q", f"kt{g}"], [f"s{g}"], name=f"qk{g}"))
        nodes.append(oh.make_node("MatMul", ["p", "v"], [f"o{g}"], name=f"pv{g}"))
    g = oh.make_graph(nodes, f"attn_{S}", [vi("p", TensorProto.FLOAT, [3, S]),
                                            vi("q", TensorProto.FLOAT, [3, 64]),
                                            vi("v", TensorProto.FLOAT, [S, 64])],
                      [vi(f"o{g}", TensorProto.FLOAT, [3, 64]) for g in range(3)]
                      + [vi(f"s{g}", TensorProto.FLOAT, [3, S]) for g in range(3)],
                      initializer=inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--lens", default="64,256,512,1024")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--board-lock", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = llm_board.bert_config()
    lens = [int(x) for x in args.lens.split(",")]
    work = tempfile.mkdtemp(prefix="llm_attn_bench_")
    dd = glp.driver_dirs_from_config(
        os.path.join(lp._main_checkout(), "demo", "bert_squad", "bert_squad_config.json"))
    projects = {}
    for S in lens:
        g = OnnxGraph(model(S), fuse_act=True, s2d_stem=True)
        names = [(sn.onnx_node.name, type(sn).__name__, getattr(sn, "b_packed", None))
                 for sn in g.nodes]
        cg = CodeGenerator(g, model_path=f"attn_{S}.onnx")
        d = os.path.join(work, f"attn_{S}")
        os.makedirs(os.path.join(d, "test"), exist_ok=True)
        tmp_onnx = os.path.join(work, f"attn_{S}.onnx")
        onnx.save(model(S), tmp_onnx)
        if sched_main(["--out-dir", d, tmp_onnx, "--no-report", "--embed-large-weights"]) != 0:
            return 1
        with open(os.path.join(d, "test", "bench.c"), "w") as f:
            f.write(BENCH_C)
        with open(os.path.join(d, "CMakeLists.txt"), "a") as f:
            f.write("\nadd_executable(attn_bench test/bench.c)\n"
                    "target_link_libraries(attn_bench PRIVATE inference m)\n"
                    "if(DEFINED INFERENCE_MATMULKERNEL_INSTANCE)\n"
                    "  target_compile_definitions(attn_bench PRIVATE "
                    "INFERENCE_MATMULKERNEL_INSTANCE=${INFERENCE_MATMULKERNEL_INSTANCE})\n"
                    "endif()\n")
        glp.populate_drivers(d, dd, cg._active_kernels)
        projects[S] = (d, names)
        del cg
    results = {}
    uio = cfg["remote"]["uio_devices"]["MatmulKernel"]
    with board_lock(args.board_lock or cfg.get("board_lock")):
        session = RemoteSession(cfg["ssh"])
        session.connect()
        try:
            for S, (d, names) in projects.items():
                rp = f"{REMOTE}/attn_{S}"
                llm_board.upload_project(session, d, rp)
                cmd = (f"cmake -S {rp} -B {rp}/build -DINFERENCE_TARGET=LINUX -DINFERENCE_PROFILING=ON "
                       f"-DINFERENCE_BUILD_TEST=OFF -DINFERENCE_MATMULKERNEL_INSTANCE=\\\"{uio}\\\" "
                       f"> {rp}/cmake.log 2>&1 && make -C {rp}/build -j4 attn_bench > {rp}/make.log 2>&1"
                       f" && {rp}/build/attn_bench {args.iters}")
                out, err, rc = session.exec(cmd, timeout=1800)
                if rc != 0:
                    print(out[-2000:], err[-2000:])
                    return 1
                wall = float(next(ln.split()[1] for ln in out.splitlines() if ln.startswith("WALL_MS")))
                lj = json.loads(next(ln for ln in out.splitlines()
                                     if ln.startswith("LAYERS_JSON:"))[len("LAYERS_JSON:"):])
                per = {}
                for L in lj["layers"]:
                    kind = "qk" if L["name"].startswith("qk") else "pv"
                    per.setdefault(kind, []).append(L["mean_us"])
                t_qk, t_pv = float(np.mean(per["qk"])), float(np.mean(per["pv"]))
                results[S] = {"qk_us": t_qk, "pv_us": t_pv, "graph_wall_ms": wall,
                              "per_token_kernel_ms": 90 * (t_qk + t_pv) / 1000.0,
                              "nodes": names}
                print(f"S={S}: q.K^T {t_qk:.1f} us, P.V {t_pv:.1f} us per call -> "
                      f"{90 * (t_qk + t_pv) / 1000.0:.2f} ms of kernel calls per token (90 groups)",
                      flush=True)
            session.exec(f"rm -rf {REMOTE}", timeout=60)
        finally:
            session.close()
    shutil.rmtree(work, ignore_errors=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
