#!/usr/bin/env python3
"""
gen_matmul_models.py — create ONNX models consisting exclusively of MatMul nodes.

These models target the MatmulKernel IP core (ap_fixed<16,8>, tiled N×K×M
matrix multiply with kTileN=4, kTileM=16, kTileK=256; kMaxK is read from
platforms/<AXI_PLATFORM>.json (kernels.matmul.max_k)).

Models produced
---------------
  mm_1x1.onnx              degenerate 1×1×1: X[1,1] @ W[1,1] -> Y[1,1]
  mm_exact_tile.onnx       N=4 K=4 M=16 — exact kTileN/kTileM boundaries
  mm_partial_n.onnx        N=5 K=4 M=16 — partial last N-tile (5 = kTileN+1)
  mm_partial_m.onnx        N=4 K=4 M=17 — partial last M-tile (17 = kTileM+1)
  mm_partial_k.onnx        N=4 K=3 M=16 — K not divisible by kTileN
  mm_large_k.onnx          N=4 K=512 M=16 — two full kTileK iterations (K=2×256)
  mm_nlp_proj.onnx         N=10 K=768 M=768 — BERT hidden-dim projection
  mm_two_layer.onnx        two sequential MatMuls: X@W1->H@W2->Y
  mm_three_layer.onnx      three sequential MatMuls: X@W1->H1@W2->H2@W3->Y
  mm_batch.onnx            batch=2, both inputs runtime: A[2,4,4]@B[2,4,16]->Y
  mm_batch_broadcast.onnx  batch=2, weight broadcasts: A[2,4,4]@W[4,16]->Y[2,4,16]
  mm_4d_3d.onnx            4-D×3-D broadcast: X[2,3,4,16]@W[3,16,8]->Y[2,3,4,8]
  mm_3d_4d.onnx            3-D×4-D broadcast: X[3,4,16]@W[2,3,16,8]->Y[2,3,4,8]
  mm_3d_3d.onnx            3-D×3-D batched weight: X[3,4,16]@W[3,16,8]->Y[3,4,8]
  mm_4d_4d.onnx            4-D×4-D batched weight: X[2,3,4,16]@W[2,3,16,8]->Y[2,3,4,8]
  mm_5d_3d.onnx            5-D×3-D broadcast: X[2,2,2,4,16]@W[2,16,8]->Y[2,2,2,4,8]
  mm_3d_5d.onnx            3-D×5-D broadcast: X[2,4,16]@W[2,2,2,16,8]->Y[2,2,2,4,8]
  mm_5d_4d.onnx            5-D×4-D broadcast: X[2,2,2,4,16]@W[2,2,16,8]->Y[2,2,2,4,8]
  mm_4d_5d.onnx            4-D×5-D broadcast: X[2,2,4,16]@W[2,2,2,16,8]->Y[2,2,2,4,8]
  mm_5d_2d.onnx            5-D×2-D broadcast: X[2,2,2,4,16]@W[16,8]->Y[2,2,2,4,8]
  mm_2d_5d.onnx            2-D×5-D broadcast: X[4,16]@W[2,2,2,16,8]->Y[2,2,2,4,8]
  mm_sat_pos.onnx          positive saturation: K=2 x=1.0 w=100.0 -> 200 -> max
  mm_sat_neg.onnx          negative saturation: K=2 x=1.0 w=-100.0 -> -200 -> min

Expected output values (for uniform input X=1.0 unless noted)
--------------------------------------------------------------
  mm_1x1              Y = 1.0
  mm_exact_tile       Y = 4 * 1.0 * 0.25 = 1.0
  mm_partial_n        Y = 1.0  (same W)
  mm_partial_m        Y = 1.0  (same W)
  mm_partial_k        Y = 3 * 1.0 * (1/3) = 1.0
  mm_large_k          Y = 512 * 1.0 * (1/512) = 1.0
  mm_nlp_proj         Y = 768 * 1.0 * (3/256) = 9.0  (matches Vivado tb)
  mm_two_layer        Y = 1.0
  mm_three_layer      Y = 1.0
  mm_batch            Y = 1.0  (for A=1.0 B=0.25)
  mm_batch_broadcast  Y = 1.0  (for A=1.0)
  mm_4d_3d            Y = 1.0  (for X=1.0; W=(1/16)*ones broadcasts outer batch)
  mm_3d_4d            Y = 1.0  (for X=1.0; W=(1/16)*ones; A broadcasts outer batch)
  mm_3d_3d            Y = 1.0  (for X=1.0; W=(1/16)*ones per batch)
  mm_4d_4d            Y = 1.0  (for X=1.0; W=(1/16)*ones per batch pair)
  mm_5d_3d            Y = 1.0  (for X=1.0; W=(1/16)*ones; outer_count=4)
  mm_3d_5d            Y = 1.0  (for X=1.0; W=(1/16)*ones; outer_count=4)
  mm_5d_4d            Y = 1.0  (for X=1.0; W=(1/16)*ones; outer_count=2)
  mm_4d_5d            Y = 1.0  (for X=1.0; W=(1/16)*ones; outer_count=2)
  mm_5d_2d            Y = 1.0  (for X=1.0; W=(1/16)*ones; batch=8, b_stride=0)
  mm_2d_5d            Y = 1.0  (for X=1.0; W=(1/16)*ones; batch=8, a_stride=0)
  mm_sat_pos          Y = saturate(200.0)  = ap_fixed max ≈ +127.996
  mm_sat_neg          Y = saturate(-200.0) = ap_fixed min = -128.0

Run from the inference-scheduler directory:
  python test/gen_matmul_models.py [--out-dir /path/to/models]
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
from onnx import TensorProto

# Pull kMaxK from the platform JSON so the violator and at-limit models
# always size against the active MatmulKernel configuration.  Avoids the
# trap of a hard-coded K=2049 silently becoming a legal value the moment
# the JSON max_k is bumped (which would make the "must raise" test pass
# without the validator ever firing).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src._matmul_hw_config import MATMUL_MAX_K  # noqa: E402


DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "models")


# ------------------------------------------------------------------ #
# Helpers (same style as gen_test_models.py)                          #
# ------------------------------------------------------------------ #

def _save(model: onnx.ModelProto, path: str) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  saved: {path}")


def _float32(name: str, shape: list) -> onnx.ValueInfoProto:
    return oh.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _initializer(name: str, data: np.ndarray) -> onnx.TensorProto:
    return nph.from_array(data.astype(np.float32), name=name)


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ------------------------------------------------------------------ #
# Model 1: 1×1×1 degenerate                                           #
# ------------------------------------------------------------------ #

def make_mm_1x1(out_dir: str) -> None:
    """X[1,1] @ W[1,1] -> Y[1,1].  W=[[1.0]] so Y=X."""
    W = np.array([[1.0]], dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_1x1",
        [_float32("X", [1, 1])],
        [_float32("Y", [1, 1])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_1x1.onnx"))


# ------------------------------------------------------------------ #
# Model 2: exact tile — N=4, K=4, M=16                                #
# ------------------------------------------------------------------ #

def make_mm_exact_tile(out_dir: str) -> None:
    """X[4,4] @ W[4,16] -> Y[4,16].

    N=kTileN=4, M=kTileM=16, K=4 < kTileK — both tile boundaries exact.
    W = 0.25 * ones(4,16):  Y[n,m] = 4 * X_row_mean * 0.25.
    For uniform X=1.0: Y = 1.0.
    """
    W = np.full((4, 16), 0.25, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_exact_tile",
        [_float32("X", [4, 4])],
        [_float32("Y", [4, 16])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_exact_tile.onnx"))


# ------------------------------------------------------------------ #
# Model 3: partial N-tile — N=5, K=4, M=16                            #
# ------------------------------------------------------------------ #

def make_mm_partial_n(out_dir: str) -> None:
    """X[5,4] @ W[4,16] -> Y[5,16].

    N=5 = kTileN+1: last N-tile has 1 valid row.
    For uniform X=1.0: Y = 1.0.
    """
    W = np.full((4, 16), 0.25, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_partial_n",
        [_float32("X", [5, 4])],
        [_float32("Y", [5, 16])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_partial_n.onnx"))


# ------------------------------------------------------------------ #
# Model 4: partial M-tile — N=4, K=4, M=17                            #
# ------------------------------------------------------------------ #

def make_mm_partial_m(out_dir: str) -> None:
    """X[4,4] @ W[4,17] -> Y[4,17].

    M=17 = kTileM+1: last M-tile has 1 valid column.
    For uniform X=1.0: Y = 1.0.
    """
    W = np.full((4, 17), 0.25, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_partial_m",
        [_float32("X", [4, 4])],
        [_float32("Y", [4, 17])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_partial_m.onnx"))


# ------------------------------------------------------------------ #
# Model 5: partial K — N=4, K=3, M=16                                 #
# ------------------------------------------------------------------ #

def make_mm_partial_k(out_dir: str) -> None:
    """X[4,3] @ W[3,16] -> Y[4,16].

    K=3 < kTileK=256, not divisible by kTileN=4.
    W = (1/3)*ones(3,16): Y = 3 * 1.0 * (1/3) = 1.0 for uniform X=1.0.
    """
    W = np.full((3, 16), 1.0 / 3.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_partial_k",
        [_float32("X", [4, 3])],
        [_float32("Y", [4, 16])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_partial_k.onnx"))


# ------------------------------------------------------------------ #
# Model 6: large K — N=4, K=512, M=16  (two full kTileK iterations)   #
# ------------------------------------------------------------------ #

def make_mm_large_k(out_dir: str) -> None:
    """X[4,512] @ W[512,16] -> Y[4,16].

    K=512 = 2*kTileK: exercises the k-tile outer loop with exactly 2 passes.
    W = (1/512)*ones(512,16): Y = 512 * 1.0 * (1/512) = 1.0 for uniform X=1.0.
    """
    W = np.full((512, 16), 1.0 / 512.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_large_k",
        [_float32("X", [4, 512])],
        [_float32("Y", [4, 16])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_large_k.onnx"))


# ------------------------------------------------------------------ #
# Model 7: NLP projection — N=10, K=768, M=768                        #
# ------------------------------------------------------------------ #

def make_mm_nlp_proj(out_dir: str) -> None:
    """X[10,768] @ W[768,768] -> Y[10,768].

    Representative BERT hidden-dim projection.
      N=10:  partial last N-tile (10 = 2*kTileN + 2 remainder)
      K=768: exactly 3 kTileK iterations (768 = 3*256)
      M=768: exactly 48 kTileM iterations (768 = 48*16)

    W = (3/256) per element — mirrors the Vivado testbench mm_10x768x768 case:
      ap_fixed<16,8> encoding: A_raw=0x0100 (1.0), W_raw=0x0003 (3/256)
      Y_raw = 768 * 256 * 3 / 256 = 2304 = 0x0900  (9.0)
    For uniform X=1.0: Y = 768 * 1.0 * (3/256) = 9.0.
    """
    W = np.full((768, 768), 3.0 / 256.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_nlp_proj",
        [_float32("X", [10, 768])],
        [_float32("Y", [10, 768])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_nlp_proj.onnx"))


# ------------------------------------------------------------------ #
# Model 8: two-layer MLP backbone (no activations)                     #
# ------------------------------------------------------------------ #

def make_mm_two_layer(out_dir: str) -> None:
    """X[4,16] @ W1[16,32] -> H[4,32];  H @ W2[32,16] -> Y[4,16].

    W1 = (1/16)*ones(16,32), W2 = (1/32)*ones(32,16).
    For uniform X=1.0: H = 1.0, Y = 1.0.
    """
    W1 = np.full((16, 32), 1.0 / 16.0, dtype=np.float32)
    W2 = np.full((32, 16), 1.0 / 32.0, dtype=np.float32)
    graph = oh.make_graph(
        [
            oh.make_node("MatMul", ["X",  "W1"], ["H"]),
            oh.make_node("MatMul", ["H",  "W2"], ["Y"]),
        ],
        "mm_two_layer",
        [_float32("X", [4, 16])],
        [_float32("Y", [4, 16])],
        initializer=[_initializer("W1", W1), _initializer("W2", W2)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_two_layer.onnx"))


# ------------------------------------------------------------------ #
# Model 9: three-layer MLP backbone (no activations)                   #
# ------------------------------------------------------------------ #

def make_mm_three_layer(out_dir: str) -> None:
    """X[4,64]@W1->H1[4,32]; H1@W2->H2[4,16]; H2@W3->Y[4,8].

    W1=(1/64)*ones(64,32), W2=(1/32)*ones(32,16), W3=(1/16)*ones(16,8).
    For uniform X=1.0: H1 = H2 = Y = 1.0.
    """
    W1 = np.full((64, 32), 1.0 / 64.0, dtype=np.float32)
    W2 = np.full((32, 16), 1.0 / 32.0, dtype=np.float32)
    W3 = np.full((16,  8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [
            oh.make_node("MatMul", ["X",  "W1"], ["H1"]),
            oh.make_node("MatMul", ["H1", "W2"], ["H2"]),
            oh.make_node("MatMul", ["H2", "W3"], ["Y"]),
        ],
        "mm_three_layer",
        [_float32("X", [4, 64])],
        [_float32("Y", [4,  8])],
        initializer=[
            _initializer("W1", W1),
            _initializer("W2", W2),
            _initializer("W3", W3),
        ],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_three_layer.onnx"))


# ------------------------------------------------------------------ #
# Model 10: batched MatMul — both inputs runtime                       #
# ------------------------------------------------------------------ #

def make_mm_batch(out_dir: str) -> None:
    """A[2,4,4] @ B[2,4,16] -> Y[2,4,16].

    batch=2, N=4, K=4, M=16.  Both A and B are runtime inputs.
    MatmulKernel: a_batch_stride=N*K=16, b_batch_stride=K*M=64.
    For A=1.0, B=0.25: Y = 4 * 1.0 * 0.25 = 1.0 per batch element.
    """
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["A", "B"], ["Y"])],
        "mm_batch",
        [_float32("A", [2, 4,  4]),
         _float32("B", [2, 4, 16])],
        [_float32("Y", [2, 4, 16])],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_batch.onnx"))


# ------------------------------------------------------------------ #
# Model 11: batched MatMul — weight broadcasts (b_batch_stride=0)      #
# ------------------------------------------------------------------ #

def make_mm_batch_broadcast(out_dir: str) -> None:
    """A[2,4,4] @ W[4,16] -> Y[2,4,16].

    batch=2, N=4, K=4, M=16.  A is a runtime input; W is a constant weight
    shared across both batch elements (MatmulKernel b_batch_stride=0).
    W = 0.25*ones(4,16): Y = 1.0 for uniform A=1.0.
    """
    W = np.full((4, 16), 0.25, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["A", "W"], ["Y"])],
        "mm_batch_broadcast",
        [_float32("A", [2, 4, 4])],
        [_float32("Y", [2, 4, 16])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_batch_broadcast.onnx"))


# ------------------------------------------------------------------ #
# Model 12: 4-D × 3-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_4d_3d(out_dir: str) -> None:
    """X[2,3,4,16] @ W[3,16,8] -> Y[2,3,4,8].

    A is 4-D [b1=2, b2=3, N=4, K=16]; W is 3-D [b2=3, K=16, M=8].
    W has the inner batch dimension (b2=3) but NOT the outer one (b1=2).
    The inference-scheduler emits an outer loop of b1=2 calls to
    XMatmulkernel, each handling batch=b2=3 inner matrix multiplications:

        for _i in range(b1=2):
            run_matmul_at(X, _i*192,  W, 0,  Y, _i*96,
                          n=4, k=16, m=8, batch=3,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = b2*N*K = 3*4*16  = 192
      b_outer_stride = 0                (W has no b1 dimension)
      c_outer_stride = b2*N*M = 3*4*8   =  96

    W = (1/K) = (1/16)*ones(3,16,8):  Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    In ap_fixed<16,8>: W_raw=0x0010 (1/16), X_raw=0x0100 (1.0).
    """
    W = np.full((3, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_4d_3d",
        [_float32("X", [2, 3, 4, 16])],
        [_float32("Y", [2, 3, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_4d_3d.onnx"))


# ------------------------------------------------------------------ #
# Model 13: 3-D × 4-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_3d_4d(out_dir: str) -> None:
    """X[3,4,16] @ W[2,3,16,8] -> Y[2,3,4,8].

    A is 3-D [b2=3, N=4, K=16]; W is 4-D [b1=2, b2=3, K=16, M=8].
    W has the outer batch dimension (b1=2) that is absent from A.
    The inference-scheduler emits an outer loop of b1=2 calls to
    XMatmulkernel, each handling batch=b2=3 inner matrix multiplications:

        for _i in range(b1=2):
            run_matmul_at(X, 0,        W, _i*384,  Y, _i*96,
                          n=4, k=16, m=8, batch=3,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = 0                (A has no b1 dimension)
      b_outer_stride = b2*K*M = 3*16*8 = 384
      c_outer_stride = b2*N*M = 3*4*8  =  96

    W = (1/K) = (1/16)*ones(2,3,16,8):  Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    In ap_fixed<16,8>: W_raw=0x0010 (1/16), X_raw=0x0100 (1.0).
    """
    W = np.full((2, 3, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_3d_4d",
        [_float32("X", [3, 4, 16])],
        [_float32("Y", [2, 3, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_3d_4d.onnx"))


# ------------------------------------------------------------------ #
# Model 14: 3-D × 3-D batched weight                                  #
# ------------------------------------------------------------------ #

def make_mm_3d_3d(out_dir: str) -> None:
    """X[3,4,16] @ W[3,16,8] -> Y[3,4,8].

    A is 3-D [batch=3, N=4, K=16]; W is 3-D [batch=3, K=16, M=8].
    Both have identical batch dimensions — no broadcasting.
    MatmulKernel: batch=3, n=4, k=16, m=8,
                  a_batch_stride=N*K=64, b_batch_stride=K*M=128,
                  c_batch_stride=N*M=32.

    W = (1/K) = (1/16)*ones(3,16,8):  Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    In ap_fixed<16,8>: W_raw=0x0010 (1/16), X_raw=0x0100 (1.0).
    """
    W = np.full((3, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_3d_3d",
        [_float32("X", [3, 4, 16])],
        [_float32("Y", [3, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_3d_3d.onnx"))


# ------------------------------------------------------------------ #
# Model 15: 4-D × 4-D batched weight                                  #
# ------------------------------------------------------------------ #

def make_mm_4d_4d(out_dir: str) -> None:
    """X[2,3,4,16] @ W[2,3,16,8] -> Y[2,3,4,8].

    A is 4-D [b1=2, b2=3, N=4, K=16]; W is 4-D [b1=2, b2=3, K=16, M=8].
    Both batch dimensions match — batch dims are flattened to batch=b1*b2=6.
    MatmulKernel: batch=6, n=4, k=16, m=8,
                  a_batch_stride=N*K=64, b_batch_stride=K*M=128,
                  c_batch_stride=N*M=32.

    W = (1/K) = (1/16)*ones(2,3,16,8):  Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    In ap_fixed<16,8>: W_raw=0x0010 (1/16), X_raw=0x0100 (1.0).
    """
    W = np.full((2, 3, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_4d_4d",
        [_float32("X", [2, 3, 4, 16])],
        [_float32("Y", [2, 3, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_4d_4d.onnx"))


# ------------------------------------------------------------------ #
# Model 16: 5-D × 3-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_5d_3d(out_dir: str) -> None:
    """X[2,2,2,4,16] @ W[2,16,8] -> Y[2,2,2,4,8].

    A is 5-D [b0=2, b1=2, b2=2, N=4, K=16]; W is 3-D [b2=2, K=16, M=8].
    W lacks the two outer batch dims (b0,b1); the scheduler emits an outer
    loop of b0*b1=4 calls advancing A and Y while W repeats at offset 0.

        for _i in range(b0*b1=4):
            run_matmul_at(X, _i*128,  W, 0,  Y, _i*64,
                          n=4, k=16, m=8, batch=2,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = b2*N*K = 2*4*16 = 128
      b_outer_stride = 0
      c_outer_stride = b2*N*M = 2*4*8  =  64

    W = (1/16)*ones(2,16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((2, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_5d_3d",
        [_float32("X", [2, 2, 2, 4, 16])],
        [_float32("Y", [2, 2, 2, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_5d_3d.onnx"))


# ------------------------------------------------------------------ #
# Model 17: 3-D × 5-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_3d_5d(out_dir: str) -> None:
    """X[2,4,16] @ W[2,2,2,16,8] -> Y[2,2,2,4,8].

    A is 3-D [b2=2, N=4, K=16]; W is 5-D [b0=2, b1=2, b2=2, K=16, M=8].
    A lacks the two outer batch dims (b0,b1); the scheduler emits an outer
    loop of b0*b1=4 calls advancing W and Y while A repeats at offset 0.

        for _i in range(b0*b1=4):
            run_matmul_at(X, 0,  W, _i*256,  Y, _i*64,
                          n=4, k=16, m=8, batch=2,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = 0
      b_outer_stride = b2*K*M = 2*16*8 = 256
      c_outer_stride = b2*N*M = 2*4*8  =  64

    W = (1/16)*ones(2,2,2,16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((2, 2, 2, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_3d_5d",
        [_float32("X", [2, 4, 16])],
        [_float32("Y", [2, 2, 2, 4, 8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_3d_5d.onnx"))


# ------------------------------------------------------------------ #
# Model 18: 5-D × 4-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_5d_4d(out_dir: str) -> None:
    """X[2,2,2,4,16] @ W[2,2,16,8] -> Y[2,2,2,4,8].

    A is 5-D [b0=2, b1=2, b2=2, N=4, K=16]; W is 4-D [b1=2, b2=2, K=16, M=8].
    W lacks only the outermost dim (b0=2); the scheduler emits an outer
    loop of b0=2 calls advancing A and Y while W repeats at offset 0.

        for _i in range(b0=2):
            run_matmul_at(X, _i*256,  W, 0,  Y, _i*128,
                          n=4, k=16, m=8, batch=4,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = b1*b2*N*K = 2*2*4*16 = 256
      b_outer_stride = 0
      c_outer_stride = b1*b2*N*M = 2*2*4*8  = 128

    W = (1/16)*ones(2,2,16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((2, 2, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_5d_4d",
        [_float32("X", [2, 2, 2, 4, 16])],
        [_float32("Y", [2, 2, 2, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_5d_4d.onnx"))


# ------------------------------------------------------------------ #
# Model 19: 4-D × 5-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_4d_5d(out_dir: str) -> None:
    """X[2,2,4,16] @ W[2,2,2,16,8] -> Y[2,2,2,4,8].

    A is 4-D [b1=2, b2=2, N=4, K=16]; W is 5-D [b0=2, b1=2, b2=2, K=16, M=8].
    A lacks only the outermost dim (b0=2); the scheduler emits an outer
    loop of b0=2 calls advancing W and Y while A repeats at offset 0.

        for _i in range(b0=2):
            run_matmul_at(X, 0,  W, _i*512,  Y, _i*128,
                          n=4, k=16, m=8, batch=4,
                          a_stride=64, b_stride=128, c_stride=32)

    Outer strides:
      a_outer_stride = 0
      b_outer_stride = b1*b2*K*M = 2*2*16*8 = 512
      c_outer_stride = b1*b2*N*M = 2*2*4*8  = 128

    W = (1/16)*ones(2,2,2,16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((2, 2, 2, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_4d_5d",
        [_float32("X", [2, 2, 4, 16])],
        [_float32("Y", [2, 2, 2, 4, 8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_4d_5d.onnx"))


# ------------------------------------------------------------------ #
# Model 20: 5-D × 2-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_5d_2d(out_dir: str) -> None:
    """X[2,2,2,4,16] @ W[16,8] -> Y[2,2,2,4,8].

    A is 5-D [b0=2, b1=2, b2=2, N=4, K=16]; W is 2-D [K=16, M=8].
    W broadcasts across all three batch dims via a single kernel call:
    batch=b0*b1*b2=8, b_batch_stride=0.

        run_matmul(X, W, Y,
                   n=4, k=16, m=8, batch=8,
                   a_stride=64, b_stride=0, c_stride=32)

    W = (1/16)*ones(16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_5d_2d",
        [_float32("X", [2, 2, 2, 4, 16])],
        [_float32("Y", [2, 2, 2, 4,  8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_5d_2d.onnx"))


# ------------------------------------------------------------------ #
# Model 21: 2-D × 5-D broadcasting                                    #
# ------------------------------------------------------------------ #

def make_mm_2d_5d(out_dir: str) -> None:
    """X[4,16] @ W[2,2,2,16,8] -> Y[2,2,2,4,8].

    A is 2-D [N=4, K=16]; W is 5-D [b0=2, b1=2, b2=2, K=16, M=8].
    A broadcasts across all three batch dims via a single kernel call:
    batch=b0*b1*b2=8, a_batch_stride=0.

        run_matmul(X, W, Y,
                   n=4, k=16, m=8, batch=8,
                   a_stride=0, b_stride=128, c_stride=32)

    W = (1/16)*ones(2,2,2,16,8): Y = K*(1/K)*1.0 = 1.0 for uniform X=1.0.
    """
    W = np.full((2, 2, 2, 16, 8), 1.0 / 16.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_2d_5d",
        [_float32("X", [4, 16])],
        [_float32("Y", [2, 2, 2, 4, 8])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_2d_5d.onnx"))


# ------------------------------------------------------------------ #
# Model 22: positive saturation                                        #
# ------------------------------------------------------------------ #

def make_mm_sat_pos(out_dir: str) -> None:
    """X[4,2] @ W[2,1] -> Y[4,1].  Saturates to ap_fixed<16,8> max.

    W = [[100.0],[100.0]]: Y = X[:,0]*100 + X[:,1]*100 = 200*X_row_mean.
    For X=1.0: Y_float=200.0 > 127.996 → saturates to 0x7FFF ≈ 127.996.
    """
    W = np.full((2, 1), 100.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_sat_pos",
        [_float32("X", [4, 2])],
        [_float32("Y", [4, 1])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_sat_pos.onnx"))


# ------------------------------------------------------------------ #
# Model 23: negative saturation                                        #
# ------------------------------------------------------------------ #

def make_mm_sat_neg(out_dir: str) -> None:
    """X[4,2] @ W[2,1] -> Y[4,1].  Saturates to ap_fixed<16,8> min.

    W = [[-100.0],[-100.0]]: Y = -(X[:,0]+X[:,1])*100 = -200*X_row_mean.
    For X=1.0: Y_float=-200.0 < -128.0 → saturates to 0x8000 = -128.0.
    """
    W = np.full((2, 1), -100.0, dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_sat_neg",
        [_float32("X", [4, 2])],
        [_float32("Y", [4, 1])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph), os.path.join(out_dir, "mm_sat_neg.onnx"))


# ------------------------------------------------------------------ #
# Hardware-bound violation / boundary models                           #
#                                                                      #
# kMaxK comes from platforms/<AXI_PLATFORM>.json (kernels.matmul.max_k) #
# via ``_matmul_hw_config``.  The over-limit model violates kMaxK by   #
# exactly one unit; the boundary model exercises the inclusive `≤`     #
# boundary.  Geometries are otherwise minimal (N=1, M=1) to keep       #
# weight tensors tiny.                                                 #
# ------------------------------------------------------------------ #

def make_mm_unsupported_k_too_large(out_dir: str) -> None:
    """X[1,kMaxK+1] @ W[kMaxK+1,1] -> Y[1,1]: K violates kMaxK by one."""
    K = MATMUL_MAX_K + 1
    W = np.zeros((K, 1), dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_unsupported_k",
        [_float32("X", [1, K])],
        [_float32("Y", [1, 1])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph),
          os.path.join(out_dir, "mm_unsupported_k.onnx"))


def make_mm_k_at_limit(out_dir: str) -> None:
    """Boundary: K == kMaxK; must parse OK (bound is `≤`, not `<`)."""
    K = MATMUL_MAX_K
    W = np.zeros((K, 1), dtype=np.float32)
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["X", "W"], ["Y"])],
        "mm_k_at_limit",
        [_float32("X", [1, K])],
        [_float32("Y", [1, 1])],
        initializer=[_initializer("W", W)],
    )
    _save(_make_model(graph),
          os.path.join(out_dir, "mm_k_at_limit.onnx"))


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #



# ------------------------------------------------------------------ #
# Packed tile-major B and 128-bit A/B port coverage (MATMUL_OPTIMISATION #
# §3 / §3b).  Constant Bs below are emitted packed (b_packed = 1);       #
# activation Bs and shared constants stay row-major.  Weights are small   #
# so ramp inputs never saturate the ap_fixed<16,8> output.                #
# ------------------------------------------------------------------ #

def _w(shape, scale, seed):
    return (np.random.RandomState(seed).uniform(-1.0, 1.0, size=shape) * scale).astype(np.float32)


def make_mm_packed_fc_256x10(out_dir: str) -> None:
    """A[1,256] @ W[256,10] -> Y[1,10]: the MNIST convnet's Gemm geometry.
    One m-tile (m padded 10 -> 16), one k-tile, 8 back-to-back 64-word
    requests per block."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "W"], ["Y"])], "mm_packed_fc_256x10",
                          [_float32("A", [1, 256])], [_float32("Y", [1, 10])],
                          initializer=[_initializer("W", _w((256, 10), 0.02, 11))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_fc_256x10.onnx"))


def make_mm_packed_fc_512x1000(out_dir: str) -> None:
    """A[1,512] @ W[512,1000] -> Y[1,1000]: ResNet-18 classifier geometry
    (2 k-tiles x 63 m-tiles, m padded to 1008; 1 MB packed weight .dat)."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "W"], ["Y"])], "mm_packed_fc_512x1000",
                          [_float32("A", [1, 512])], [_float32("Y", [1, 1000])],
                          initializer=[_initializer("W", _w((512, 1000), 0.01, 12))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_fc_512x1000.onnx"))


def make_mm_packed_fc_1280x1001(out_dir: str) -> None:
    """A[1,1280] @ W[1280,1001] -> Y[1,1001]: MobileNet v2 classifier
    geometry (5 k-tiles x 63 m-tiles)."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "W"], ["Y"])], "mm_packed_fc_1280x1001",
                          [_float32("A", [1, 1280])], [_float32("Y", [1, 1001])],
                          initializer=[_initializer("W", _w((1280, 1001), 0.005, 13))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_fc_1280x1001.onnx"))


def make_mm_packed_batch_const(out_dir: str) -> None:
    """A[3,2,8] @ W[3,8,20] -> Y[3,2,20]: batched CONSTANT B.  Every slice
    is packed; b_batch_stride is rescaled from 8*20 to 8*32 packed elements."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "W"], ["Y"])], "mm_packed_batch_const",
                          [_float32("A", [3, 2, 8])], [_float32("Y", [3, 2, 20])],
                          initializer=[_initializer("W", _w((3, 8, 20), 0.1, 14))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_batch_const.onnx"))


def make_mm_packed_4d_3d_const(out_dir: str) -> None:
    """A[2,3,4,8] @ W[3,8,20] -> Y[2,3,4,20]: outer loop (run_matmul_at)
    over a CONSTANT packed B; b_outer_stride stays 0, b_batch_stride is
    rescaled to packed slices."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "W"], ["Y"])], "mm_packed_4d_3d_const",
                          [_float32("A", [2, 3, 4, 8])], [_float32("Y", [2, 3, 4, 20])],
                          initializer=[_initializer("W", _w((3, 8, 20), 0.1, 15))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_4d_3d_const.onnx"))


def make_mm_shared_const_row_major(out_dir: str) -> None:
    """W[8,8] feeds BOTH a MatMul (as B) and an Add: the scheduler must keep
    it row-major (b_packed = 0) or the Add would read a packed image."""
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["A", "W"], ["Y"]),
         oh.make_node("Add", ["X2", "W"], ["Z"])],
        "mm_shared_const_row_major",
        [_float32("A", [4, 8]), _float32("X2", [8, 8])],
        [_float32("Y", [4, 8]), _float32("Z", [8, 8])],
        initializer=[_initializer("W", _w((8, 8), 0.2, 16))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_shared_const_row_major.onnx"))


def make_mm_unaligned_rows(out_dir: str) -> None:
    """A[3,5,13] @ B[3,13,5] -> Y[3,5,5], both runtime inputs: k=13 and m=5
    put every row segment and batch slice (65 elements) at a non-16-byte
    offset, so the 128-bit A/B ports must extract lanes at all 8 shifts."""
    graph = oh.make_graph([oh.make_node("MatMul", ["A", "B"], ["Y"])], "mm_unaligned_rows",
                          [_float32("A", [3, 5, 13]), _float32("B", [3, 13, 5])],
                          [_float32("Y", [3, 5, 5])])
    _save(_make_model(graph), os.path.join(out_dir, "mm_unaligned_rows.onnx"))


def make_mm_relu_then_packed_odd(out_dir: str) -> None:
    """Relu(A[1,7,13]) @ W[13,9] -> Y[1,7,9]: A is an intermediate tensor
    (VectorOP output) with 13-element rows, B is a packed constant with
    k=13, m=9 (m padded to 16)."""
    graph = oh.make_graph(
        [oh.make_node("Relu", ["A"], ["R"]),
         oh.make_node("MatMul", ["R", "W"], ["Y"])],
        "mm_relu_then_packed_odd",
        [_float32("A", [1, 7, 13])], [_float32("Y", [1, 7, 9])],
        initializer=[_initializer("W", _w((13, 9), 0.1, 17))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_relu_then_packed_odd.onnx"))


def make_mm_packed_then_activation(out_dir: str) -> None:
    """MatMul(A, W const) -> MatMul(., B input): two MatmulKernel calls in
    one inference with b_packed = 1 then 0.  Regression for the register
    being rewritten on every call — a stale b_packed = 1 would make the
    second (row-major) product wrong."""
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["A", "W"], ["T"]),
         oh.make_node("MatMul", ["T", "B"], ["Y"])],
        "mm_packed_then_activation",
        [_float32("A", [2, 24]), _float32("B", [20, 6])],
        [_float32("Y", [2, 6])],
        initializer=[_initializer("W", _w((24, 20), 0.1, 18))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_then_activation.onnx"))


def make_mm_packed_two_layer_odd(out_dir: str) -> None:
    """[1,40] @ W1[40,24] -> Relu -> @ W2[24,10]: two packed constants whose
    m (24, 10) are not multiples of 16 and whose k (40, 24) are partial
    k-tiles."""
    graph = oh.make_graph(
        [oh.make_node("MatMul", ["A", "W1"], ["T"]),
         oh.make_node("Relu", ["T"], ["R"]),
         oh.make_node("MatMul", ["R", "W2"], ["Y"])],
        "mm_packed_two_layer_odd",
        [_float32("A", [1, 40])], [_float32("Y", [1, 10])],
        initializer=[_initializer("W1", _w((40, 24), 0.1, 19)),
                     _initializer("W2", _w((24, 10), 0.1, 20))])
    _save(_make_model(graph), os.path.join(out_dir, "mm_packed_two_layer_odd.onnx"))


_ALL_MAKERS = [
    make_mm_1x1,
    make_mm_exact_tile,
    make_mm_partial_n,
    make_mm_partial_m,
    make_mm_partial_k,
    make_mm_large_k,
    make_mm_nlp_proj,
    make_mm_two_layer,
    make_mm_three_layer,
    make_mm_batch,
    make_mm_batch_broadcast,
    make_mm_4d_3d,
    make_mm_3d_4d,
    make_mm_3d_3d,
    make_mm_4d_4d,
    make_mm_5d_3d,
    make_mm_3d_5d,
    make_mm_5d_4d,
    make_mm_4d_5d,
    make_mm_5d_2d,
    make_mm_2d_5d,
    make_mm_sat_pos,
    make_mm_sat_neg,
    make_mm_unsupported_k_too_large,
    make_mm_k_at_limit,
    make_mm_packed_fc_256x10,
    make_mm_packed_fc_512x1000,
    make_mm_packed_fc_1280x1001,
    make_mm_packed_batch_const,
    make_mm_packed_4d_3d_const,
    make_mm_shared_const_row_major,
    make_mm_unaligned_rows,
    make_mm_relu_then_packed_odd,
    make_mm_packed_then_activation,
    make_mm_packed_two_layer_odd,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MatMul ONNX test models.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help=f"Directory to write .onnx files (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Writing {len(_ALL_MAKERS)} MatMul test models to: {args.out_dir}\n")

    for maker in _ALL_MAKERS:
        maker(args.out_dir)

    print(f"\nDone — {len(_ALL_MAKERS)} models generated.")


if __name__ == "__main__":
    main()
