"""
VectorOPKernel's activation unit in the scheduler (doc/plans/ACTIVATIONS_PLAN.md).

The kernel (``kernels/vectorop_rtl/rtl/vo_act.sv``) computes LeakyReLU, SiLU,
GELU (erf) and GELU (tanh) as ops 6-9 and as acts 3-6 after any op: the exact
function of the Q8.8 value rounded to the nearest Q8.8 value, ties to even —
the host ops' write-back (``DataType.host_quantize``).  IEEE double with libm
gives exactly that for every Q8.8 input (the nearest rounding tie is
1.6e-5 LSB away), so ``act_values`` here, the C++ model (``VectorOP.cpp``
``act_fn``) and the RTL's table agree bit for bit.

Mapping (``from_onnx``; only when the platform's VectorOPKernel has the unit,
``_vectorop_hw_config.VECTOROP_ACTIVATIONS``, and only on ``ap_fixed<16,8>``
DMA tensors without a power-of-two exponent — the table is Q8.8's):

  * ``Gelu`` (native, ``approximate`` none / tanh, or a fused pattern with the
    graph's own constants): OP_GELU / OP_GELU_TANH when the node's Q8.8
    function equals the kernel's on all 65 536 inputs (BERT's float32
    constants do), else the host op ``GeluNode`` as before;
  * ``LeakyRelu``: OP_LEAKY_RELU with the slope quantised to 2^-16
    (``alpha`` register: round(alpha * 65536), 0 <= alpha < 1);
  * ``Silu`` (``fusion.fuse_patterns``' node for ``Mul(x, Sigmoid(x))``):
    OP_SILU.

``OnnxGraph._fuse_activations`` then folds them into a producing
Add / Sub / Mul / Div (the act register), as Relu / Clip(0,6).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from . import _vectorop_hw_config
from .dtype import AP_FIXED_16_8, ApFixed, DataType
from .host_nodes import GeluNode, HostContext, libm
from .nodes import (ACT_GELU, ACT_GELU_TANH, ACT_LEAKY_RELU, ACT_SILU, OP_GELU,
                    OP_GELU_TANH, OP_LEAKY_RELU, OP_SILU, ScheduledNode, SchedulerError)

# ONNX op types this module maps ("Silu" is fusion.py's node for x * Sigmoid(x)).
ONNX_OP_TYPES = frozenset({"Gelu", "LeakyRelu", "Silu"})

# The activation codes vo_act computes (ReLU / ReLU6 are the ALU's).
UNIT_ACTS = frozenset({ACT_LEAKY_RELU, ACT_SILU, ACT_GELU, ACT_GELU_TANH})

ALPHA_ONE = 1 << 16          # the alpha register's scale: slope = alpha / 2^16


def enabled() -> bool:
    """The platform's VectorOPKernel has the activation unit (read at call time,
    so a test can patch ``_vectorop_hw_config.VECTOROP_ACTIVATIONS``)."""
    return _vectorop_hw_config.VECTOROP_ACTIVATIONS


def act_values(act: int, x, alpha: int = 0) -> np.ndarray:
    """The unrounded activation ``act`` (ACT_LEAKY_RELU .. ACT_GELU_TANH) of the
    float64 values ``x``, in double — the formulas of VectorOP.cpp ``act_fn``."""
    x = np.asarray(x, np.float64)
    if act == ACT_LEAKY_RELU:
        return np.where(x >= 0.0, x, x * ((alpha & 0xFFFF) / ALPHA_ONE))
    if act == ACT_SILU:
        return x / (1.0 + libm("exp", -x))
    if act == ACT_GELU:
        return x * (0.5 * (1.0 + libm("erf", x / math.sqrt(2.0))))
    if act == ACT_GELU_TANH:
        return x * (0.5 * (1.0 + libm("tanh", math.sqrt(2.0 / math.pi)
                                       * (x + 0.044715 * (x * x * x)))))
    raise ValueError(f"act {act} is not an activation of the unit")


def apply(act: int, x, dtype: DataType, alpha: int = 0) -> np.ndarray:
    """What the kernel writes for the Q8.8 op results ``x``: the activation
    rounded to nearest, ties to even, saturated."""
    return dtype.host_quantize(act_values(act, x, alpha))


def is_q88(dtype: DataType) -> bool:
    """The element type the kernel's activation unit computes in."""
    return isinstance(dtype, ApFixed) and dtype.name == "ap_fixed<16,8>"


@lru_cache(maxsize=None)
def _all_q88() -> np.ndarray:
    return np.arange(-32768, 32768, dtype=np.float64) / 256.0


@lru_cache(maxsize=None)
def kernel_table(act: int) -> np.ndarray:
    """The kernel's output for every Q8.8 input (ascending raw -32768 .. 32767)."""
    return apply(act, _all_q88(), AP_FIXED_16_8)


def gelu_act(node: GeluNode, dtype: DataType) -> Optional[int]:
    """ACT_GELU / ACT_GELU_TANH when ``node`` (with the graph's constants)
    computes the kernel's function on every Q8.8 input, else None."""
    if not is_q88(dtype):
        return None
    act = ACT_GELU_TANH if node.approximate == "tanh" else ACT_GELU
    return act if np.array_equal(node.values(_all_q88(), dtype), kernel_table(act)) else None


def leaky_alpha(alpha: float, name: str) -> int:
    """The alpha register for an ONNX LeakyRelu slope: round(alpha * 2^16)."""
    q = int(round(float(alpha) * ALPHA_ONE))
    if not (0.0 <= float(alpha) and q < ALPHA_ONE):
        raise SchedulerError(
            f"LeakyRelu node '{name}': alpha={alpha} — VectorOPKernel takes a slope "
            f"0 <= alpha < 1 (quantised to 2^-16).")
    return q


def _eligible(node, tensors, dtype: DataType) -> Optional[str]:
    """Why the node cannot run on the activation unit, or None."""
    if not enabled():
        return ("the platform's VectorOPKernel has no activation unit "
                "(platforms/<platform>.json kernels.vectorop.activations, "
                "AXI_VECTOROP_ACTIVATIONS)")
    if not is_q88(dtype):
        return f"the activation unit computes in ap_fixed<16,8>, not {dtype.name}"
    for name in (node.input[0], node.output[0]):
        t = tensors.get(name)
        if t is None:
            raise SchedulerError(
                f"Tensor '{name}' of node '{node.name or node.op_type}' was not "
                f"found in the graph.")
        if t.exp is not None or t.host is not None:
            return f"tensor '{name}' is not a plain Q8.8 DMA tensor (exponent / host memory)"
    return None


def _vectorop_node(node, tensors, index: int, align_elems: int, op_code: int,
                   alpha: int = 0) -> ScheduledNode:
    sn = ScheduledNode(onnx_node=node, op_code=op_code, arity=1,
                       inputs=[tensors[node.input[0]]], output=tensors[node.output[0]],
                       index=index, align_elems=align_elems)
    sn.alpha = alpha
    sn.validate()
    return sn


def from_onnx(node, tensors, index: int, align_elems: int, ctx: HostContext,
              dtype: DataType):
    """The scheduled node of an ONNX Gelu / LeakyRelu / Silu (see the module doc)."""
    label = node.name or node.op_type
    why = _eligible(node, tensors, dtype)
    if node.op_type == "Gelu":
        host = GeluNode.from_onnx_node(node, tensors, index, align_elems, ctx)
        act = None if why else gelu_act(host, dtype)
        if act is None:
            return host
        return _vectorop_node(node, tensors, index, align_elems,
                              OP_GELU_TANH if act == ACT_GELU_TANH else OP_GELU)
    if why:
        raise SchedulerError(f"{node.op_type} node '{label}' needs VectorOPKernel's "
                             f"activation unit: {why}.")
    if node.op_type == "LeakyRelu":
        attrs = {a.name: a for a in node.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 0.01
        return _vectorop_node(node, tensors, index, align_elems, OP_LEAKY_RELU,
                              leaky_alpha(alpha, label))
    if node.op_type == "Silu":
        return _vectorop_node(node, tensors, index, align_elems, OP_SILU)
    raise SchedulerError(f"Node '{label}': {node.op_type} is not an activation of "
                         f"VectorOPKernel's unit.")


__all__ = ("ONNX_OP_TYPES", "UNIT_ACTS", "ALPHA_ONE", "enabled", "act_values", "apply",
           "is_q88", "kernel_table", "gelu_act", "leaky_alpha", "from_onnx")
