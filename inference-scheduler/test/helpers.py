import os
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
def _model(name: str) -> str:
    return os.path.join(MODELS_DIR, name)
def _models_exist() -> bool:
    required = ["single_add.onnx", "relu_chain.onnx", "mixed_ops.onnx", "unsupported.onnx"]
    return all(os.path.isfile(_model(m)) for m in required)


def matmul_impl(impl: str):
    """Patch the MatmulKernel the scheduler's cost model assumes
    (``kernels.matmul.impl``, "rtl" on the KV260): "hls" models the
    bitstreams built with the Vitis HLS kernel (caa67f49a5a3 and older),
    whose engine choices many tests pin.  A context manager or decorator; it
    does not cover setUpClass."""
    from unittest import mock

    from src import cost_model
    return mock.patch.object(cost_model, "MATMUL_IMPL", impl)
