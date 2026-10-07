"""Generate ONNX test models for VectorOPKernel's activation unit
(doc/plans/ACTIVATIONS_PLAN.md): LeakyReLU, SiLU, GELU and GELU tanh, as
ops and fused after an Add / Mul / Div.

The single-activation models take a [1, 65536] input: the test harness fills
inputs with the ramp ``p[pos] = (Data_t)pos``, so every Q8.8 value appears
once and the board run checks the activation on every input against the
scheduler's simulation.  Generated with the platform's activation unit on
(``kernels.vectorop.activations``); run with ``AXI_VECTOROP_ACTIVATIONS=0``
the scheduler keeps the GELUs on the host and rejects the others.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
OPSET = 20            # native Gelu


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name=name)


def _save(nodes, inputs, outputs, inits, name):
    graph = helper.make_graph(nodes, name.replace(".onnx", ""), inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    out = os.path.join(OUT_DIR, name)
    onnx.save(model, out)
    print(f"  {out}")


ALL = [1, 65536]


def gen_gelu():
    _save([helper.make_node("Gelu", ["X"], ["Y"], approximate="none")],
          [_vi("X", ALL)], [_vi("Y", ALL)], [], "act_gelu.onnx")


def gen_gelu_tanh():
    _save([helper.make_node("Gelu", ["X"], ["Y"], approximate="tanh")],
          [_vi("X", ALL)], [_vi("Y", ALL)], [], "act_gelu_tanh.onnx")


def gen_leaky_relu():
    _save([helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.1)],
          [_vi("X", ALL)], [_vi("Y", ALL)], [], "act_leaky_relu.onnx")


def gen_silu():
    _save([helper.make_node("Sigmoid", ["X"], ["S"]),
           helper.make_node("Mul", ["X", "S"], ["Y"])],
          [_vi("X", ALL)], [_vi("Y", ALL)], [], "act_silu.onnx")


def gen_bias_gelu():
    """Broadcast bias Add -> Gelu (one call: act GELU), the BERT FFN shape."""
    rng = np.random.default_rng(20)
    _save([helper.make_node("Add", ["X", "bias"], ["H"]),
           helper.make_node("Gelu", ["H"], ["Y"])],
          [_vi("X", [1, 64, 256])], [_vi("Y", [1, 64, 256])],
          [_init("bias", rng.uniform(-2.0, 2.0, (256,)))], "act_bias_gelu.onnx")


def gen_mul_leaky_relu():
    """Mul -> LeakyRelu(0.01) (one call: act LEAKY_RELU, alpha 655)."""
    rng = np.random.default_rng(21)
    _save([helper.make_node("Mul", ["X", "scale"], ["H"]),
           helper.make_node("LeakyRelu", ["H"], ["Y"], alpha=0.01)],
          [_vi("X", [1, 4096])], [_vi("Y", [1, 4096])],
          [_init("scale", rng.uniform(-1.5, 1.5, (1, 4096)))], "act_mul_leaky_relu.onnx")


def gen_div_silu():
    """Div -> x * Sigmoid(x) (one call: DIV, one lane per cycle, act SILU)."""
    rng = np.random.default_rng(22)
    _save([helper.make_node("Div", ["X", "d"], ["H"]),
           helper.make_node("Sigmoid", ["H"], ["S"]),
           helper.make_node("Mul", ["S", "H"], ["Y"])],
          [_vi("X", [1, 1024])], [_vi("Y", [1, 1024])],
          [_init("d", rng.uniform(0.5, 4.0, (1, 1024)))], "act_div_silu.onnx")


def gen_gelu_leaky_chain():
    """Gelu -> LeakyRelu: two calls (nothing is fused into an activation op)."""
    _save([helper.make_node("Gelu", ["X"], ["H"], approximate="tanh"),
           helper.make_node("LeakyRelu", ["H"], ["Y"], alpha=0.2)],
          [_vi("X", [1, 2048])], [_vi("Y", [1, 2048])], [], "act_gelu_leaky_chain.onnx")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn in (gen_gelu, gen_gelu_tanh, gen_leaky_relu, gen_silu, gen_bias_gelu,
               gen_mul_leaky_relu, gen_div_silu, gen_gelu_leaky_chain):
        fn()


if __name__ == "__main__":
    main()
