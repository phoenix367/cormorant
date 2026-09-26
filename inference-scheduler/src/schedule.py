"""
Data-flow DAG over the scheduled node list.

Builds explicit producer / consumer relationships from each node's
input and output tensors so the parallel scheduler (added on top of this
module) can answer:

  * which nodes are *ready* (all dependencies produced)?
  * which nodes are *concurrency-independent* and may therefore run on
    different hardware kernel lanes simultaneously?

Edges
-----
There is an edge ``u -> v`` iff some intermediate tensor produced by
``u`` is consumed by ``v``. Constant weights / initializers and graph
inputs are treated as *external* tensors — they impose no edge because
they are always available before ``inference_run()`` enters its body.

Reshape nodes
-------------
``ReshapeNode`` is a buffer alias and emits no hardware call, but it
still appears in the DAG as a normal node so any downstream consumer of
the reshaped tensor is correctly ordered after the producer of the
underlying source. The event emitter ignores nodes whose ``kernel_name``
is the empty string.

Host ops
--------
``SpaceToDepthNode`` also has an empty ``kernel_name`` (it runs on the
CPU), but unlike a Reshape it produces a new buffer: it is an ordinary
DAG node with a real producer edge to its consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple


@dataclass
class DagNode:
    """One DAG node — wraps a scheduler node with its in/out edges."""

    sched: object               # ScheduledNode | MatmulNode | ConvNode | PoolNode | ReshapeNode
    index: int                  # original graph order (sched.index)
    kernel_name: str            # "" for ReshapeNode (no hardware call)
    preds: Set[int] = field(default_factory=set)
    succs: Set[int] = field(default_factory=set)


@dataclass
class Dag:
    """Data-flow DAG over an :class:`OnnxGraph`'s scheduled node list."""

    nodes: List[DagNode]                   # in original graph order
    by_index: Dict[int, DagNode]
    producer_of: Dict[str, int]            # tensor onnx_name -> producing node index
    external_tensors: FrozenSet[str]       # graph inputs + constant weights

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_graph(cls, graph) -> "Dag":
        """Build a DAG from an :class:`OnnxGraph`.

        The graph's ``nodes`` list is the source of truth for ordering and
        for ``sched.index`` values.
        """
        externals: Set[str] = {t.onnx_name for t in graph.input_tensors}
        externals.update(t.onnx_name for t in graph.weight_tensors)
        # persistent states (src/numeric.py) that no node of this graph
        # produces are always available, like weights
        produced = {sn.output.onnx_name for sn in graph.nodes}
        externals.update(t.onnx_name for t in getattr(graph, "state_tensors", [])
                         if t.onnx_name not in produced)

        producer: Dict[str, int] = {}
        for sn in graph.nodes:
            out_name = sn.output.onnx_name
            if out_name in producer:
                raise ValueError(
                    f"Tensor '{out_name}' is produced by both node "
                    f"{producer[out_name]} and {sn.index}; ONNX graph "
                    f"is malformed."
                )
            producer[out_name] = sn.index

        dag_nodes: List[DagNode] = []
        by_index: Dict[int, DagNode] = {}
        for sn in graph.nodes:
            dn = DagNode(
                sched=sn,
                index=sn.index,
                kernel_name=getattr(type(sn), "kernel_name", ""),
            )
            dag_nodes.append(dn)
            by_index[sn.index] = dn

        for sn in graph.nodes:
            consumer = by_index[sn.index]
            for t in sn.inputs:
                name = t.onnx_name
                if name in externals or t.is_weight:
                    continue
                prod_idx = producer.get(name)
                if prod_idx is None:
                    raise ValueError(
                        f"Node {sn.index} ('{sn.onnx_node.op_type}') consumes "
                        f"tensor '{name}', but no producing node was found and "
                        f"it is not a graph input or constant weight."
                    )
                if prod_idx == sn.index:
                    continue
                consumer.preds.add(prod_idx)
                by_index[prod_idx].succs.add(sn.index)

        return cls(
            nodes=dag_nodes,
            by_index=by_index,
            producer_of=producer,
            external_tensors=frozenset(externals),
        )

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def predecessors(self, idx: int) -> Set[int]:
        return self.by_index[idx].preds

    def successors(self, idx: int) -> Set[int]:
        return self.by_index[idx].succs

    def roots(self) -> List[int]:
        """Indices of nodes with no DAG predecessors (ready immediately)."""
        return sorted(n.index for n in self.nodes if not n.preds)

    def leaves(self) -> List[int]:
        """Indices of nodes with no DAG successors (graph outputs)."""
        return sorted(n.index for n in self.nodes if not n.succs)

    def topological_order(self) -> List[int]:
        """Kahn's algorithm; returns node indices in a valid topological order.

        Ties are broken by the node's original graph index, so the result is
        deterministic and matches ``graph.nodes`` order whenever the graph is
        already a chain.
        """
        in_deg: Dict[int, int] = {n.index: len(n.preds) for n in self.nodes}
        ready = sorted(idx for idx, d in in_deg.items() if d == 0)
        order: List[int] = []
        while ready:
            idx = ready.pop(0)
            order.append(idx)
            for s in sorted(self.by_index[idx].succs):
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    lo, hi = 0, len(ready)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if ready[mid] < s:
                            lo = mid + 1
                        else:
                            hi = mid
                    ready.insert(lo, s)
        if len(order) != len(self.nodes):
            raise ValueError(
                f"DAG contains a cycle: only {len(order)} of "
                f"{len(self.nodes)} nodes scheduled."
            )
        return order

    def ancestors(self) -> Dict[int, FrozenSet[int]]:
        """For every node index, the frozenset of all transitive ancestors."""
        out: Dict[int, Set[int]] = {n.index: set() for n in self.nodes}
        for idx in self.topological_order():
            anc = out[idx]
            for p in self.by_index[idx].preds:
                anc.add(p)
                anc.update(out[p])
        return {k: frozenset(v) for k, v in out.items()}

    def independent_pairs(self) -> List[Tuple[int, int]]:
        """All ``(u, v)`` with ``u < v`` whose execution order is unconstrained.

        Two nodes are concurrency-independent iff neither is a transitive
        ancestor of the other.  Quadratic; intended for tests, not for the
        scheduler's hot path.
        """
        anc = self.ancestors()
        ids = sorted(self.by_index.keys())
        out: List[Tuple[int, int]] = []
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                if u in anc[v] or v in anc[u]:
                    continue
                out.append((u, v))
        return out
