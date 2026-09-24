"""
ONNX model loading and graph resolution.

OnnxGraph
---------
  1. Loads the model and runs shape inference so every intermediate tensor
     has a known shape.
  2. Builds a flat dict of TensorInfo objects covering:
       - constant weights  (model.graph.initializer)
       - graph inputs      (model.graph.input)
       - graph outputs     (model.graph.output)
       - intermediate      (model.graph.value_info, produced by shape inference)
  3. Wraps each NodeProto in a ScheduledNode (validates op support and shapes).
  4. Exposes the ordered node list ready for code generation.
"""

from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
import onnx
import onnx.helper as onnx_helper
import onnx.numpy_helper as nph
from onnx import shape_inference, TensorProto

from typing import Union
from .tensor import TensorInfo
from .nodes  import (
    ACT_NONE, _pack_matmul_b, ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode,
                     POOL_OP_TYPES, VECTOROP_OP_TYPES, RESHAPE_OP_TYPES, SchedulerError)
from .dtype  import DataType, AP_FIXED_16_8

_ALL_SUPPORTED_OP_TYPES: frozenset = (
    {"MatMul", "Conv", "Gemm"} | POOL_OP_TYPES | VECTOROP_OP_TYPES | RESHAPE_OP_TYPES
)


# ------------------------------------------------------------------ #
# ONNX dtype → numpy dtype string                                     #
# ------------------------------------------------------------------ #

_ONNX_DTYPE_MAP = {
    TensorProto.FLOAT:   "float32",
    TensorProto.DOUBLE:  "float64",
    TensorProto.INT8:    "int8",
    TensorProto.INT16:   "int16",
    TensorProto.INT32:   "int32",
    TensorProto.INT64:   "int64",
    TensorProto.UINT8:   "uint8",
    TensorProto.UINT16:  "uint16",
    TensorProto.UINT32:  "uint32",
    TensorProto.UINT64:  "uint64",
    TensorProto.FLOAT16: "float16",
    TensorProto.BOOL:    "bool",
}


def _onnx_dtype_name(onnx_dtype: int) -> str:
    return _ONNX_DTYPE_MAP.get(onnx_dtype, f"onnx_dtype_{onnx_dtype}")


def _shape_from_type_proto(tp: onnx.TypeProto) -> List[int]:
    """Extract a concrete integer shape list from a TypeProto."""
    if not tp.HasField("tensor_type"):
        return []
    shape = tp.tensor_type.shape
    if shape is None:
        return []
    dims = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            # Symbolic / dynamic dimension — use 0 as placeholder
            dims.append(0)
    return dims


class OnnxGraph:
    """Parsed, validated, and resolved ONNX computation graph."""

    @staticmethod
    def _preprocess_model(model: onnx.ModelProto):
        """Rewrite Gemm → MatMul + Add (when applicable) so downstream node
        classes only see the supported op set.

        Returns ``(rewritten_model, gemm_decomposed_count)``.  The count is
        zero for graphs that contained no Gemm nodes (in which case the
        original model is returned unmodified).
        """
        """
        Simplify the ONNX graph before scheduling:

          1. Decompose Gemm (alpha=1, beta=1, transA=0) into MatMul + Add
             so existing MatmulNode / ScheduledNode handle it.  When
             transB=1, the constant B initializer is transposed offline
             and a new "<B>_T" initializer is appended so the rewritten
             MatMul can read it as a row-major tensor (transB=0
             semantics).  transB=1 with a non-constant B is rejected
             — runtime transpose is not supported.

        Requires that shape inference has already been run on the model
        so that intermediate shapes are available for the new MatMul output.
        """
        graph = model.graph

        # Build a shape map from all known tensors (inputs, outputs, value_info,
        # and initializers — initializers don't appear in value_info).
        shape_map: Dict[str, List[int]] = {}
        for init in graph.initializer:
            arr = nph.to_array(init)
            shape_map[init.name] = list(arr.shape)
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            dims = [
                d.dim_value if d.HasField("dim_value") else 0
                for d in vi.type.tensor_type.shape.dim
            ]
            shape_map[vi.name] = dims

        gemm_counter = [0]
        new_nodes: List[onnx.NodeProto] = []
        new_value_info: List[onnx.ValueInfoProto] = []

        for node in graph.node:
            if node.op_type != "Gemm":
                new_nodes.append(node)
                continue

            attrs = {a.name: a for a in node.attribute}
            alpha  = attrs["alpha"].f  if "alpha"  in attrs else 1.0
            beta   = attrs["beta"].f   if "beta"   in attrs else 1.0
            transA = attrs["transA"].i if "transA" in attrs else 0
            transB = attrs["transB"].i if "transB" in attrs else 0

            if abs(alpha - 1.0) > 1e-6 or abs(beta - 1.0) > 1e-6:
                raise SchedulerError(
                    f"Gemm node '{node.name}': alpha={alpha}, beta={beta}. "
                    f"Only alpha=1, beta=1 is supported."
                )
            if transA != 0:
                raise SchedulerError(
                    f"Gemm node '{node.name}': transA={transA}. "
                    f"Only transA=0 is supported (A is a runtime tensor; "
                    f"offline transpose is not feasible)."
                )
            if transB not in (0, 1):
                raise SchedulerError(
                    f"Gemm node '{node.name}': transB={transB}. "
                    f"Must be 0 or 1."
                )

            A = node.input[0]
            B = node.input[1]
            C = node.input[2] if len(node.input) >= 3 and node.input[2] else None
            Y = node.output[0]

            # transB=1: transpose the constant B initializer offline so the
            # rewritten MatMul reads it row-major.  We append a new
            # "<B>_T" initializer rather than mutating B in place so any
            # other consumer of the original tensor is left untouched.
            if transB == 1:
                b_init = next(
                    (init for init in graph.initializer if init.name == B),
                    None,
                )
                if b_init is None:
                    raise SchedulerError(
                        f"Gemm node '{node.name}': transB=1 requires B '{B}' "
                        f"to be a constant initializer; runtime transpose is "
                        f"not supported."
                    )
                arr = nph.to_array(b_init)
                if arr.ndim != 2:
                    raise SchedulerError(
                        f"Gemm node '{node.name}': transB=1 with non-2D B "
                        f"(shape={list(arr.shape)}) is not supported."
                    )
                new_B = f"{B}_T"
                if not any(init.name == new_B for init in graph.initializer):
                    graph.initializer.append(
                        nph.from_array(arr.T.copy(), name=new_B)
                    )
                    shape_map[new_B] = list(arr.T.shape)
                B = new_B

            gemm_counter[0] += 1
            n = gemm_counter[0]

            if C:
                # Gemm → MatMul(A,B)→tmp  +  Add(tmp,C)→Y
                tmp = f"_gemm_mm_out_{n}"
                # Infer tmp shape: A[-2] × B[-1]
                a_shape = shape_map.get(A, [])
                b_shape = shape_map.get(B, [])
                if len(a_shape) >= 2 and len(b_shape) >= 2:
                    tmp_shape = a_shape[:-1] + [b_shape[-1]]
                else:
                    tmp_shape = []
                if tmp_shape:
                    new_value_info.append(
                        onnx_helper.make_tensor_value_info(
                            tmp, TensorProto.FLOAT, tmp_shape
                        )
                    )
                new_nodes.append(
                    onnx_helper.make_node("MatMul", inputs=[A, B], outputs=[tmp],
                                          name=f"_gemm_matmul_{n}")
                )
                new_nodes.append(
                    onnx_helper.make_node("Add", inputs=[tmp, C], outputs=[Y],
                                          name=f"_gemm_add_{n}")
                )
            else:
                # No bias: Gemm → MatMul(A,B)→Y
                new_nodes.append(
                    onnx_helper.make_node("MatMul", inputs=[A, B], outputs=[Y],
                                          name=f"_gemm_matmul_{n}")
                )

        if gemm_counter[0] == 0:
            return model, 0  # nothing changed

        new_graph = onnx_helper.make_graph(
            new_nodes,
            graph.name,
            list(graph.input),
            list(graph.output),
            initializer=list(graph.initializer),
            value_info=list(graph.value_info) + new_value_info,
        )
        new_model = onnx_helper.make_model(
            new_graph, opset_imports=list(model.opset_import)
        )
        new_model.ir_version = model.ir_version
        return new_model, gemm_counter[0]

    def __init__(self, model_path: str,
                 dtype: DataType = None,
                 fuse_act: bool = False) -> None:
        """
        fuse_act: fold a Relu / Clip(0,6) node into the VectorOP node that
        produces its input (the kernel's `act` register) when the producer's
        output has no other consumer and is not a graph output.  Off by
        default so generated code is unchanged unless asked for; the CLI
        enables it.  ``self.act_fused_count`` reports how many were folded.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        _dtype      = dtype if dtype is not None else AP_FIXED_16_8
        align_elems = _dtype.align_elems

        # Load and validate
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Run shape inference so every intermediate tensor gets a shape
        model = shape_inference.infer_shapes(model)

        # Simplify: decompose Gemm → MatMul + Add.  The count is exposed via
        # ``self.gemm_decomposed_count`` so the report generator can list it
        # as an applied transformation.
        model, self.gemm_decomposed_count = OnnxGraph._preprocess_model(model)

        graph = model.graph

        # ---------------------------------------------------------- #
        # Build the tensor registry                                    #
        # ---------------------------------------------------------- #
        self._tensors: Dict[str, TensorInfo] = {}

        # 1. Constant weights / initializers
        for init in graph.initializer:
            arr = nph.to_array(init).copy()
            ti  = TensorInfo(
                onnx_name=init.name,
                shape=list(arr.shape),
                dtype=_onnx_dtype_name(init.data_type),
                data=arr.astype(np.float32),   # always store as float32
            )
            self._tensors[init.name] = ti

        # 2. Graph inputs (may overlap with initializers for older opsets)
        for vi in graph.input:
            if vi.name in self._tensors:
                continue  # already registered as initializer
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # 3. Intermediate tensors (shape-inferred by onnx.shape_inference)
        for vi in graph.value_info:
            if vi.name in self._tensors:
                continue
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # 4. Graph outputs
        for vi in graph.output:
            if vi.name in self._tensors:
                continue
            shape = _shape_from_type_proto(vi.type)
            dtype = _onnx_dtype_name(vi.type.tensor_type.elem_type)
            self._tensors[vi.name] = TensorInfo(
                onnx_name=vi.name,
                shape=shape,
                dtype=dtype,
                data=None,
            )

        # ---------------------------------------------------------- #
        # Identify model boundaries                                    #
        # ---------------------------------------------------------- #
        # Graph inputs that are NOT in the initializer set are true
        # model inputs (data that the caller supplies at run time).
        init_names = {init.name for init in graph.initializer}
        self._input_names: List[str] = [
            vi.name for vi in graph.input if vi.name not in init_names
        ]
        self._output_names: List[str] = [
            vi.name for vi in graph.output
        ]

        # ---------------------------------------------------------- #
        # Resolve nodes                                               #
        # ---------------------------------------------------------- #
        self._nodes: List[Union[ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode]] = []
        for idx, node in enumerate(graph.node):
            if node.op_type == "MatMul":
                sn = MatmulNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type == "Conv":
                sn = ConvNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type in POOL_OP_TYPES:
                sn = PoolNode.from_onnx_node(node, self._tensors, idx, align_elems)
            elif node.op_type in RESHAPE_OP_TYPES:
                sn = ReshapeNode.from_onnx_node(node, self._tensors, idx, align_elems)
            else:
                if node.op_type not in VECTOROP_OP_TYPES:
                    raise SchedulerError(
                        f"Node '{node.name or node.op_type}' "
                        f"(op_type='{node.op_type}') is not supported.\n"
                        f"Supported ops: {sorted(_ALL_SUPPORTED_OP_TYPES)}"
                    )
                sn = ScheduledNode.from_onnx_node(node, self._tensors, idx, align_elems)
            self._nodes.append(sn)

        self.act_fused_count = self._fuse_activations() if fuse_act else 0
        self._pack_matmul_weights()

    # ------------------------------------------------------------------ #
    # Activation fusion (VectorOPKernel `act` register)                    #
    # ------------------------------------------------------------------ #

    def _fuse_activations(self) -> int:
        """Fold Relu / Clip(0,6) nodes into their producing VectorOP node.

        A unary activation node R with input T is folded into the
        ScheduledNode P that produces T when
          * P is a VectorOP ScheduledNode with no activation fused yet,
          * T is not a graph output (it must not be materialised),
          * R is T's only consumer, and
          * R's output has the same element count as T (unary nodes are
            validated that way already).
        P then writes R's output directly with ``act`` set, R disappears
        from the node list and node indices are renumbered.  Returns the
        number of folded nodes.
        """
        consumers: Dict[str, List[int]] = {}
        for pos, sn in enumerate(self._nodes):
            for t in sn.inputs:
                consumers.setdefault(t.onnx_name, []).append(pos)
        producer_pos: Dict[str, int] = {
            sn.output.onnx_name: pos for pos, sn in enumerate(self._nodes)
        }
        graph_outputs = set(self._output_names)

        removed: set = set()
        fused = 0
        for pos, sn in enumerate(self._nodes):
            if not isinstance(sn, ScheduledNode):
                continue
            act = sn.fusable_act
            if act is None:
                continue
            src = sn.inputs[0]
            ppos = producer_pos.get(src.onnx_name)
            if ppos is None:
                continue                                 # graph input / weight
            prod = self._nodes[ppos]
            if not isinstance(prod, ScheduledNode) or prod.act != ACT_NONE:
                continue
            if src.onnx_name in graph_outputs:
                continue
            if consumers.get(src.onnx_name, []) != [pos]:
                continue
            if src.numel != sn.output.numel:
                continue
            prod.act    = act
            prod.output = sn.output
            prod.fused_nodes.append(sn.onnx_node)
            producer_pos[sn.output.onnx_name] = ppos
            removed.add(pos)
            fused += 1

        if fused:
            self._nodes = [sn for pos, sn in enumerate(self._nodes)
                           if pos not in removed]
            for idx, sn in enumerate(self._nodes):
                sn.index = idx
        return fused

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def _pack_matmul_weights(self) -> None:
        """Emit constant MatMul B operands in MatmulKernel's tile-major packed
        layout (MATMUL_OPTIMISATION §3b, MatmulNode.b_packed).

        A constant is packed only when EVERY node that reads it is a MatMul
        using it as B with the same (k, m): the packed image is a different
        byte layout, so a tensor also consumed elsewhere (a VectorOP, a
        Reshape alias, the A side of another MatMul) must keep its row-major
        form.  Such tensors, and non-constant Bs (activations), stay
        row-major and the kernel reads them through its per-row path.
        """
        readers: dict = {}
        for sn in self._nodes:
            for t in getattr(sn, "inputs", []):
                readers.setdefault(t.onnx_name, []).append(sn)
        done: set = set()
        for sn in self._nodes:
            if not isinstance(sn, MatmulNode):
                continue
            b = sn.inputs[1]
            if b.data is None or b.onnx_name in done:
                continue
            users = readers.get(b.onnx_name, [])
            if not all(isinstance(u, MatmulNode) and u.inputs[1] is b
                       and u.k == sn.k and u.m == sn.m for u in users):
                continue
            _pack_matmul_b(b, sn.k, sn.m)
            for u in users:
                u.pack_b()
            done.add(b.onnx_name)

    @property
    def nodes(self) -> List[Union[ScheduledNode, MatmulNode, ConvNode, PoolNode, ReshapeNode]]:
        return self._nodes

    @property
    def input_tensors(self) -> List[TensorInfo]:
        return [self._tensors[n] for n in self._input_names]

    @property
    def output_tensors(self) -> List[TensorInfo]:
        return [self._tensors[n] for n in self._output_names]

    @property
    def weight_tensors(self) -> List[TensorInfo]:
        """All constant initializer tensors, in declaration order."""
        seen = set()
        weights = []
        for sn in self._nodes:
            for t in sn.inputs:
                if t.is_weight and t.onnx_name not in seen:
                    seen.add(t.onnx_name)
                    weights.append(t)
        return weights

    @property
    def intermediate_tensors(self) -> List[TensorInfo]:
        """Non-constant, non-input, non-output tensors (writable buffers)."""
        boundary = (
            {t.onnx_name for t in self.input_tensors}
            | {t.onnx_name for t in self.output_tensors}
            | {t.onnx_name for t in self.weight_tensors}
        )
        seen = set()
        result = []
        for sn in self._nodes:
            for t in [sn.output] + sn.inputs:
                if t.onnx_name not in boundary and t.onnx_name not in seen:
                    seen.add(t.onnx_name)
                    result.append(t)
        return result

    def get_tensor(self, name: str) -> TensorInfo:
        return self._tensors[name]
