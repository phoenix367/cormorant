"""Generate ONNX test models that use PoolingKernel nodes."""

import os
import sys
import onnx
from onnx import helper, TensorProto

# Pull bounds from the platform JSON so violator/at-limit models always
# size against the active PoolingKernel configuration.  Avoids the trap of
# a hard-coded pool_h=8 silently becoming a legal value the moment the
# JSON max_kh is bumped (which would make the "must raise" test pass
# without the validator ever firing).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src._pool_hw_config import (   # noqa: E402
    POOL_MAX_KH,
    POOL_MAX_KW,
    POOL_MAX_LINE_BUF_ROWS,
    POOL_MAX_LINE_BUF_COLS,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def _save(model, name: str) -> None:
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


def _vi(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _opset(v: int = 13):
    return [helper.make_opsetid("", v)]


# ---------------------------------------------------------------------------
# pool_maxpool_simple: 2x2 MaxPool, stride=2, no padding
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_maxpool_simple() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_maxpool_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_maxpool_simple.onnx")


# ---------------------------------------------------------------------------
# pool_avgpool_simple: 2x2 AveragePool, stride=2, no padding
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_avgpool_simple() -> None:
    node = helper.make_node(
        "AveragePool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_avgpool_simple",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_avgpool_simple.onnx")


# ---------------------------------------------------------------------------
# pool_maxpool_padded: 3x3 MaxPool, stride=1, pad=1 (same-size output)
# X[1,4,8,8] → Y[1,4,8,8]
# ---------------------------------------------------------------------------
def gen_maxpool_padded() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_maxpool_padded",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 8, 8])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_maxpool_padded.onnx")


# ---------------------------------------------------------------------------
# pool_avgpool_count_pad: 2x2 AveragePool with count_include_pad=1
# X[1,2,6,6] → Y[1,2,3,3]
# ---------------------------------------------------------------------------
def gen_avgpool_count_pad() -> None:
    node = helper.make_node(
        "AveragePool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], count_include_pad=1,
    )
    graph = helper.make_graph(
        [node], "pool_avgpool_count_pad",
        inputs=[_vi("X", [1, 2, 6, 6])],
        outputs=[_vi("Y", [1, 2, 3, 3])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_avgpool_count_pad.onnx")


# ---------------------------------------------------------------------------
# pool_global_max: GlobalMaxPool
# X[1,8,4,4] → Y[1,8,1,1]
# ---------------------------------------------------------------------------
def gen_global_max() -> None:
    node = helper.make_node(
        "GlobalMaxPool", inputs=["X"], outputs=["Y"],
    )
    graph = helper.make_graph(
        [node], "pool_global_max",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 8, 1, 1])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_global_max.onnx")


# ---------------------------------------------------------------------------
# pool_global_avg: GlobalAveragePool
# X[1,8,4,4] → Y[1,8,1,1]
# ---------------------------------------------------------------------------
def gen_global_avg() -> None:
    node = helper.make_node(
        "GlobalAveragePool", inputs=["X"], outputs=["Y"],
    )
    graph = helper.make_graph(
        [node], "pool_global_avg",
        inputs=[_vi("X", [1, 8, 4, 4])],
        outputs=[_vi("Y", [1, 8, 1, 1])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_global_avg.onnx")


# ---------------------------------------------------------------------------
# pool_lp_p2: LpPool p=2 (default), 2x2, stride=2
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_lp_p2() -> None:
    node = helper.make_node(
        "LpPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], p=2,
    )
    graph = helper.make_graph(
        [node], "pool_lp_p2",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset(18)), "pool_lp_p2.onnx")


# ---------------------------------------------------------------------------
# pool_lp_p1: LpPool p=1, 2x2, stride=2
# X[1,4,8,8] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_lp_p1() -> None:
    node = helper.make_node(
        "LpPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2], p=1,
    )
    graph = helper.make_graph(
        [node], "pool_lp_p1",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset(18)), "pool_lp_p1.onnx")


# ---------------------------------------------------------------------------
# pool_then_relu: MaxPool followed by Relu (PoolingKernel + VectorOPKernel)
# X[1,4,8,8] → Z[1,4,4,4] → Y[1,4,4,4]
# ---------------------------------------------------------------------------
def gen_pool_then_relu() -> None:
    pool = helper.make_node("MaxPool",  inputs=["X"], outputs=["Z"], kernel_shape=[2, 2], strides=[2, 2])
    relu = helper.make_node("Relu",     inputs=["Z"], outputs=["Y"])
    graph = helper.make_graph(
        [pool, relu], "pool_then_relu",
        inputs=[_vi("X", [1, 4, 8, 8])],
        outputs=[_vi("Y", [1, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_then_relu.onnx")


# ---------------------------------------------------------------------------
# pool_batch2: MaxPool with batch=2
# X[2,4,8,8] → Y[2,4,4,4]
# ---------------------------------------------------------------------------
def gen_pool_batch2() -> None:
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "pool_batch2",
        inputs=[_vi("X", [2, 4, 8, 8])],
        outputs=[_vi("Y", [2, 4, 4, 4])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()), "pool_batch2.onnx")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 128-bit x port coverage (POOL_OPTIMIZATION §2.13): row segments that start
# at every 16-byte lane offset, rows wider than the 64-column line buffer
# (ow-tiling), several channel tiles, and batch slices at odd offsets.
# ---------------------------------------------------------------------------
def _pool(name, op, x_shape, y_shape, **attrs):
    node = helper.make_node(op, inputs=["X"], outputs=["Y"], **attrs)
    graph = helper.make_graph([node], name, inputs=[_vi("X", x_shape)], outputs=[_vi("Y", y_shape)])
    _save(helper.make_model(graph, opset_imports=_opset()), name + ".onnx")


def gen_pool_w13_maxpool() -> None:
    """X[1,4,9,13] 2x2 s2 -> Y[1,4,4,6]: 13-element rows, every run starts
    at a different lane."""
    _pool("pool_w13_maxpool", "MaxPool", [1, 4, 9, 13], [1, 4, 4, 6],
          kernel_shape=[2, 2], strides=[2, 2])


def gen_pool_w77_avgpool_tiled() -> None:
    """X[1,8,20,77] 3x3 s2 pad1 -> Y[1,8,10,39]: in_w > 64 columns forces
    ow-tiling; one 8-channel tile."""
    _pool("pool_w77_avgpool_tiled", "AveragePool", [1, 8, 20, 77], [1, 8, 10, 39],
          kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])


def gen_pool_c24_w17_maxpool() -> None:
    """X[1,24,17,17] 3x3 s1 pad1 -> Y[1,24,17,17]: three channel tiles,
    17-element rows."""
    _pool("pool_c24_w17_maxpool", "MaxPool", [1, 24, 17, 17], [1, 24, 17, 17],
          kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])


def gen_pool_batch3_odd_offsets() -> None:
    """X[3,5,7,30] 2x2 s2 -> Y[3,5,3,15]: 1050-element batch slices, so
    every batch starts at a non-16-byte offset."""
    _pool("pool_batch3_odd_offsets", "MaxPool", [3, 5, 7, 30], [3, 5, 3, 15],
          kernel_shape=[2, 2], strides=[2, 2])


def gen_pool_global_avg_w7() -> None:
    """GlobalAveragePool X[1,12,7,7] -> Y[1,12,1,1]: 49-element channel
    planes (every plane starts at a different lane), two channel tiles;
    7 is the kernel's kMaxPoolH/W."""
    _pool("pool_global_avg_w7", "GlobalAveragePool", [1, 12, 7, 7], [1, 12, 1, 1])


ALL_GENERATORS = [
    gen_maxpool_simple,
    gen_avgpool_simple,
    gen_maxpool_padded,
    gen_avgpool_count_pad,
    gen_global_max,
    gen_global_avg,
    gen_lp_p2,
    gen_lp_p1,
    gen_pool_then_relu,
    gen_pool_batch2,
    gen_pool_w13_maxpool,
    gen_pool_w77_avgpool_tiled,
    gen_pool_c24_w17_maxpool,
    gen_pool_batch3_odd_offsets,
    gen_pool_global_avg_w7,
]


# ---------------------------------------------------------------------------
# Hardware-bound violation models — must each raise SchedulerError when
# loaded by OnnxGraph.  The bounds come from platforms/<AXI_PLATFORM>.json
# (kernels.pool) via ``_pool_hw_config`` — the same JSON the C++ CMake
# build reads, so the violator/at-limit fixtures track whatever the
# active platform configures the kernel for.
#
# Each model violates exactly one bound by exactly one unit so the test
# can identify which constraint fired.  Geometries are otherwise minimal
# to keep ONNX shape-inference fast and the model files tiny.
# ---------------------------------------------------------------------------

def gen_unsupported_pool_h_too_large() -> None:
    """pool_h = kMaxPoolH + 1 (window taller than the adder tree)."""
    pool_h = POOL_MAX_KH + 1
    in_h   = pool_h + 4
    out_h  = in_h - pool_h + 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[pool_h, 3], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_pool_h",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, out_h, 6])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_pool_h.onnx")


def gen_unsupported_pool_w_too_large() -> None:
    """pool_w = kMaxPoolW + 1."""
    pool_w = POOL_MAX_KW + 1
    in_w   = pool_w + 4
    out_w  = in_w - pool_w + 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, pool_w], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_pool_w",
        inputs=[_vi("X", [1, 4, 8, in_w])],
        outputs=[_vi("Y", [1, 4, 6, out_w])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_pool_w.onnx")


def gen_unsupported_dil_h_overflows_line_buf() -> None:
    """Vertical pool span one above kMaxLineBufRows — must raise.

    Uses pool_h=2 with dil_h = MAX_ROWS so span = (pool_h-1)*dil_h + 1
    = MAX_ROWS + 1, exactly one element over the line-buffer-row capacity.
    pool_h stays within kMaxPoolH so the violation isolates the
    line-buffer-row constraint, not the pool-height constraint.
    """
    dh   = POOL_MAX_LINE_BUF_ROWS
    span = dh + 1                       # (2-1)*dh + 1
    in_h = span                         # out_h = in_h - span + 1 = 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 3], strides=[1, 1], dilations=[dh, 1],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_dil_h",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, 1, 6])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_dil_h.onnx")


def gen_unsupported_dil_w_overflows_line_buf() -> None:
    """Horizontal pool span one above kMaxLineBufCols — must raise.

    pool_w=2 with dil_w = MAX_COLS so span = MAX_COLS + 1.
    """
    dw   = POOL_MAX_LINE_BUF_COLS
    span = dw + 1
    in_w = span                         # out_w = 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[3, 2], strides=[1, 1], dilations=[1, dw],
    )
    graph = helper.make_graph(
        [node], "pool_unsupported_dil_w",
        inputs=[_vi("X", [1, 4, 8, in_w])],
        outputs=[_vi("Y", [1, 4, 6, 1])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_unsupported_dil_w.onnx")


def gen_pool_h_at_limit() -> None:
    """Boundary-case: pool_h == kMaxPoolH; must parse OK."""
    pool_h = POOL_MAX_KH
    in_h   = pool_h + 5
    out_h  = in_h - pool_h + 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[pool_h, 3], strides=[1, 1],
    )
    graph = helper.make_graph(
        [node], "pool_pool_h_at_limit",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, out_h, 6])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_pool_h_at_limit.onnx")


def gen_dil_h_at_line_buf_limit() -> None:
    """Boundary-case: vertical span == kMaxLineBufRows; must parse OK.

    pool_h=2 with dil_h = MAX_ROWS - 1 gives span = MAX_ROWS exactly.
    """
    dh    = POOL_MAX_LINE_BUF_ROWS - 1
    span  = dh + 1                      # exactly MAX_ROWS
    in_h  = span + 4                    # gives out_h = 5, mild non-degenerate
    out_h = in_h - span + 1
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=[2, 3], strides=[1, 1], dilations=[dh, 1],
    )
    graph = helper.make_graph(
        [node], "pool_dil_h_at_limit",
        inputs=[_vi("X", [1, 4, in_h, 8])],
        outputs=[_vi("Y", [1, 4, out_h, 6])],
    )
    _save(helper.make_model(graph, opset_imports=_opset()),
          "pool_dil_h_at_limit.onnx")


_UNSUPPORTED_POOL_GENERATORS = [
    gen_unsupported_pool_h_too_large,
    gen_unsupported_pool_w_too_large,
    gen_unsupported_dil_h_overflows_line_buf,
    gen_unsupported_dil_w_overflows_line_buf,
    gen_pool_h_at_limit,
    gen_dil_h_at_line_buf_limit,
]
ALL_GENERATORS += _UNSUPPORTED_POOL_GENERATORS


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating pool test models in {OUT_DIR}/")
    for gen in ALL_GENERATORS:
        gen()
    print("Done.")


if __name__ == "__main__":
    main()
