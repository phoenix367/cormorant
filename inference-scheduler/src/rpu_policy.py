"""Who drives the kernels: the A53 itself or the Cortex-R5 firmware
(doc/plans/RPU_PLAN.md §9.3).  Small models stay on the A53, large ones go to
the R5.

Driving the kernels from the R5 costs about CALL_US per kernel call more
than the A53 (the R5 writes the call's changed registers across the LPD–FPD
bridge — posted, the window is device memory — and takes the interrupt) and
BATCH_US per batch (the A53 hands the batch over and waits for its end; a
batch ends before every host op and at the output sync).  Measured on the
KV260 on 2026-10-13 (bitstream 1cedaf2ed816, protocol v4, R5 minus A53
latency, RPU_PLAN §9.4): chains of 1 / 16 / 128 tiny VectorOP calls +3 /
+21 / +146 µs, MNIST convnet +17 µs for 8 calls and 1 batch, LeNet +11 µs
for 10 + 1, ResNet-18 +36 µs for 41 + 1, MobileNet v1 +47 µs for 57 + 1,
MobileNet v2 +111 µs for 100 + 1, YOLOv5n +276 µs for 127 + 14.  The
least-squares fit, 1.0 µs a call and 10.6 µs a batch, is within 21 µs of
every one.  (Protocol v3 — every register strongly ordered, the whole ring
cleaned per batch — cost 8.4 µs and 61 µs.)

Since protocol v5 (RPU_PLAN §11.2) the R5 sleeps after R5_IDLE_US without a
command and each inference pays RUN_US once — the R5 woken for its first batch
and the A53 asleep on the doorbell through its last, which waits for the
outputs — and WAKE_US for every further wake, after a host interval longer
than R5_IDLE_US.  Measured on 2026-10-13 evening (v5, the A53 cores' deep
idle on, rpu_board.py --mode r5, R5 minus A53 per run): MNIST convnet +42 µs
(its runs back to back: no R5 wake), LeNet +82 µs, ResNet-18 +112, MobileNet
v1 +151 / +140, MobileNet v2 +245, YOLOv5n +825 (14 batches, 3.4 wakes): the
convnet against LeNet gives the A53's share (~23 µs over the calls and the
batch) and the R5's wake (~38 µs).  LeNet: 10 + 10.6 + 61 = 82 µs, 6.6 % of
its 1.22 ms -> the A53.  The larger models' calls cost more than 1.0 µs
(MobileNet v2 ~1.8: more registers change per call) and YOLO's long inner
batches each ~30 µs more (the A53 asleep on the doorbell); neither is
modelled — a per-long-batch term from the replay's batch lengths counted all
151 decode batches of SmolLM2-135M as long where the board's doorbell count
says about a fifth, and would have put decode on the A53 against the
measurement (+1.2 ms, 2.4 %, on the R5) — and both only make large models'
small shares larger.

A model is "large" — the R5 drives it — when that overhead is at most
MAX_OVERHEAD of its predicted inference time.  The prediction is the
bitstream's performance model where one exists (perf_models/kv260/, as
--plan uses it), with the analytic cost model for any call it cannot price;
with neither (no kernel call priced), a model with at least
MIN_CALLS_WITHOUT_MODEL calls per batch counts as large.  The generated
header carries the numbers (INFERENCE_RPU_CALLS / _BATCHES / _PREDICTED_US /
_OVERHEAD_US / _RECOMMENDED, per entry in a multi-entry project); the
library follows INFERENCE_RPU_RECOMMENDED per run when the firmware runs,
unless the environment says INFERENCE_RPU=0 / 1 (runtime/inference_rpu.c).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

CALL_US = 1.0
BATCH_US = 10.6
RUN_US = 61.0                   # once per inference: the R5 woken for its first batch + the A53 asleep on
                                # the doorbell through its last (the outputs)
WAKE_US = 38.0                  # every further R5 wake: after a host interval longer than R5_IDLE_US
R5_IDLE_US = 1000.0             # runtime/rpu/rpu_ctrl.h RPU_IDLE_SPIN_US
MAX_OVERHEAD = 0.05
MIN_CALLS_WITHOUT_MODEL = 32

_PERF_MODEL: Dict[str, object] = {}


def estimate_us(c, mhz: Optional[float] = None) -> float:
    """The analytic cost model's time of one kernel call (µs, at the kernel clock
    ``mhz``, default KERNEL_MHZ): the calibration's sizing estimate
    (perf_calibrate.py) and the policy's fallback prediction."""
    from .cost_model import CALL_OVERHEAD, KERNEL_MHZ, conv_batch_cycles, gemv_cycles, matmul_cycles
    f = c.fields
    if c.kernel == "ConvKernel":
        try:
            cyc = conv_batch_cycles(f["batch"], in_ch=f["in_ch"], out_ch=f["out_ch"],
                                    in_h=f["in_h"], in_w=f["in_w"], oh=f["out_h"], ow=f["out_w"],
                                    kh=f["kh"], kw=f["kw"], sh=f["stride_h"], sw=f["stride_w"],
                                    dh=f["dilation_h"], dw=f["dilation_w"],
                                    pt=f["pad_top"], pl=f["pad_left"])
            if f["is_dw"]:
                cyc /= max(1, f["in_ch"] // 16)
        except Exception:                                    # noqa: BLE001 — sizing only
            cyc = 1e6
    elif c.kernel == "MatmulKernel":
        cyc = (gemv_cycles(f["n"], f["k"], f["m"], f["batch"], f["gemv_kw"]) if f["gemv_kw"]
               else matmul_cycles(f["n"], f["k"], f["m"], f["batch"]))
    elif c.kernel == "VectorOPKernel" and f["op"] in (10, 11):
        # vo_smx with ideal memory (SOFTMAX_PLAN §4.2)
        cyc = (60 + f["outer"] // 16 * (455 + 6 * f["size"]) if f["op"] == 11
               else 75 + f["outer"] * (58 + 3 * math.ceil(f["size"] / 8)))
    elif c.kernel == "VectorOPKernel":
        cyc = f["outer"] * math.ceil(f["size"] / 8) * (8 if f["op"] == 3 else 1) + 500
    elif c.kernel == "RowOpsKernel":
        # four lanes per cycle; NORM reads its row twice; ROPE loads its B row (HD / 2
        # words) per row; TRANSPOSE loads a tile and emits one granule per cycle
        # (contiguous) or per two (scattered); FOLD eight lanes (ROWOPS_RTL_KERNEL.md
        # §Performance)
        rows, n, op = f["rows"] * f["blocks"], f["n"], f["op"] & 15
        if op == 9:                                   # FOLD: eight lanes, a row's words + one
            cyc = rows * (math.ceil(n / 8) + 1) + 500
        elif op == 7:
            R = f["aux"] & 15
            cyc = rows * (R * math.ceil(n / 8) + (n if f["b_s2"] == 2 * R else 2 * n)) + 500
        elif op == 6:
            cyc = rows * (math.ceil(n / 4) + (f["aux"] & 0xFFF) // 2 + 11) + 540
        else:
            passes = 2 if op == 5 else 1
            cyc = rows * (passes * math.ceil(n / 4) + 40) + 500
    else:
        cyc = f["batch"] * f["channels"] * (f["out_h"] * f["out_w"] * f["pool_h"] * f["pool_w"]
                                            + f["in_h"] * f["in_w"]) / 8 + 500
    return (cyc + CALL_OVERHEAD) / (mhz or KERNEL_MHZ)


def _default_perf_model():
    """The performance model of the bitstream bitstream_config_kv260.json names
    (as --plan resolves it), or None."""
    from .perf_calls import local_bitstream_id
    from .perf_model import PerfModel, default_model_path
    try:
        p = default_model_path(local_bitstream_id())
    except Exception:                                        # noqa: BLE001 — no config, no model
        return None
    if p is None:
        return None
    key = str(p)
    if key not in _PERF_MODEL:
        _PERF_MODEL[key] = PerfModel.load(key)
    return _PERF_MODEL[key]


def _calls_of(sn, layouts):
    if not hasattr(sn, "kernel_calls"):
        return []
    try:
        return list(sn.kernel_calls(layouts))
    except Exception:                                        # noqa: BLE001 — a runtime-keyed node
        return [None]


def rpu_estimate(cg) -> dict:
    """The policy's numbers for one generated graph: ``calls`` (kernel calls per
    inference), ``batches`` (R5 batches: one per host op that follows kernel
    work, plus the final one), ``predicted_us`` (the inference as predicted;
    None without any model), ``overhead_us``, ``recommended`` and ``basis``."""
    from .codegen.timing import simulate
    from .host_model import HostModel, default_host_model_path

    layouts = cg._layouts
    events = cg._compute_event_stream()
    by_index = {sn.index: sn for sn in cg._graph.nodes}
    calls = 0
    batches = 0
    pending = False
    for ev in events:
        if ev[0] in ("start", "start_sync"):
            calls += len(_calls_of(by_index[ev[1]], layouts))
            pending = True
        elif ev[0] == "cpu" and pending:
            batches += 1
            pending = False
    if pending:
        batches += 1
    pm = _default_perf_model()
    hp = default_host_model_path()
    hm = HostModel.load(hp) if hp.exists() else HostModel()

    def kernel_us(sn):
        """The node's kernel time: the performance model, else the analytic model;
        None when a call cannot be listed (a runtime-keyed attention node)."""
        cs = _calls_of(sn, layouts)
        if not cs:
            return 0.0
        if any(c is None for c in cs):
            return None
        if pm is not None:
            t = pm.calls_us(cs)
            if t is not None:
                return t
        return sum(estimate_us(c) for c in cs)

    predicted_us: Optional[float] = None
    basis = "call count"
    r5_wakes = 1 if calls else 0
    if calls:
        tl = simulate(cg, kernel_us, hm.us, events=events)
        basis = (f"performance model {pm.platform}/{pm.bitstream}" if pm is not None
                 else "analytic cost model")
        predicted_us = tl.total_us if tl.total_us > 0 else None
        if predicted_us is None:
            basis = "call count"
        else:
            r5_wakes = _r5_wakes(tl)
    overhead_us = (calls * CALL_US + batches * BATCH_US
                   + (RUN_US + (r5_wakes - 1) * WAKE_US if calls else 0.0))
    if predicted_us is not None:
        recommended = overhead_us <= MAX_OVERHEAD * predicted_us
    else:
        recommended = calls >= MIN_CALLS_WITHOUT_MODEL * max(1, batches)
    return {"calls": calls, "batches": batches, "r5_wakes": r5_wakes,
            "predicted_us": predicted_us, "overhead_us": overhead_us,
            "recommended": bool(recommended and calls > 0), "basis": basis}


def _r5_wakes(tl) -> int:
    """The R5's wake-ups per inference, from the timed replay: one at the start,
    one after every host interval (back-to-back host ops merged) longer than
    R5_IDLE_US that kernel work follows."""
    host = sorted((t0, t1) for lane, _n, t0, t1, kind in tl.spans if lane == "CPU" and kind == "host")
    merged = []
    for t0, t1 in host:
        if merged and t0 <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    kernels = [t0 for lane, _n, t0, _t1, kind in tl.spans if kind == "kernel"]
    return 1 + sum(1 for t0, t1 in merged if t1 - t0 > R5_IDLE_US and any(k >= t1 - 1e-6 for k in kernels))


def header_lines(est: dict, suffix: str = "") -> list:
    """The policy as C macros (``suffix`` names the entry of a multi-entry project)."""
    pred = est["predicted_us"]
    why = (f"{est['overhead_us'] / pred * 100:.1f} % of {pred / 1e3:.2f} ms predicted ({est['basis']}; "
           f"{est['r5_wakes']} R5 wake(s))"
           if pred else f"no prediction: {est['calls']} calls, {est['batches']} batch(es)")
    return [
        f"#define INFERENCE_RPU_CALLS{suffix}         {est['calls']}u",
        f"#define INFERENCE_RPU_BATCHES{suffix}       {est['batches']}u",
        f"#define INFERENCE_RPU_PREDICTED_US{suffix}  {int(round(pred)) if pred else 0}u",
        f"#define INFERENCE_RPU_OVERHEAD_US{suffix}   {int(round(est['overhead_us']))}u",
        f"#define INFERENCE_RPU_RECOMMENDED{suffix}   {1 if est['recommended'] else 0}"
        f"   /* {'the R5 drives the kernels' if est['recommended'] else 'the A53 drives the kernels'}: {why} */",
    ]


__all__ = ("CALL_US", "BATCH_US", "RUN_US", "WAKE_US", "R5_IDLE_US",
           "MAX_OVERHEAD", "MIN_CALLS_WITHOUT_MODEL",
           "estimate_us", "rpu_estimate", "header_lines")
