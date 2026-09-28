"""
The timed event-stream replay (src/codegen/timing.py, TACTICS_PLAN §4.4):
the CPU issues kernel starts without blocking, waits where the event stream
waits, runs host ops inline; a multi-call MatmulConvNode blocks until its
last call is issued.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.codegen.timing import simulate  # noqa: E402


class _SN:
    def __init__(self, index, lane, calls=1):
        self.index, self.lane, self.calls = index, lane, calls
        if calls > 1:
            self.conv_n = 1


class _Graph:
    def __init__(self, nodes):
        self.nodes = nodes


class _CG:
    def __init__(self, nodes, events):
        self._graph = _Graph(nodes)
        self._events = events

    def _kernel_id_of(self, sn):
        return sn.lane

    def _compute_event_stream(self):
        return self._events


class TestSimulate(unittest.TestCase):
    def test_overlap_and_waits(self):
        # conv(0) 100 us || host(1) 30 us; wait conv; host(2) 10 us; matmul(3) 50; drain
        nodes = [_SN(0, "KERNEL_CONV"), _SN(1, None), _SN(2, None), _SN(3, "KERNEL_MATMUL")]
        ev = [("start", 0), ("cpu", 1), ("wait", "KERNEL_CONV", 0), ("cpu", 2),
              ("start", 3), ("drain", "KERNEL_MATMUL", 3)]
        dur = {0: 100.0, 1: 30.0, 2: 10.0, 3: 50.0}
        tl = simulate(_CG(nodes, ev), lambda sn: dur[sn.index], lambda sn: dur[sn.index],
                      issue_us=1.0)
        # 0: start (t=1), host 30 (t=31), wait until 100, host 10 (110), start (111), drain 160
        self.assertAlmostEqual(tl.total_us, 160.0)
        self.assertAlmostEqual(tl.cpu_us, 1 + 30 + 10 + 1)
        self.assertEqual(tl.lane_us, {"KERNEL_CONV": 100.0, "KERNEL_MATMUL": 50.0})
        self.assertAlmostEqual(tl.wait_us, 69.0 + 49.0)
        self.assertEqual([w[0] for w in tl.top_waits()], [0, 3])

    def test_same_lane_serialises(self):
        nodes = [_SN(0, "KERNEL_CONV"), _SN(1, "KERNEL_CONV")]
        ev = [("start", 0), ("wait", "KERNEL_CONV", 0), ("start", 1), ("drain", "KERNEL_CONV", 1)]
        tl = simulate(_CG(nodes, ev), lambda sn: 40.0, lambda sn: 0.0, issue_us=0.0)
        self.assertAlmostEqual(tl.total_us, 80.0)

    def test_multi_call_node(self):
        # 4 calls of 25 us: the CPU is busy until the last call is issued
        nodes = [_SN(0, "KERNEL_CONV", calls=4), _SN(1, None)]
        ev = [("start", 0), ("cpu", 1), ("drain", "KERNEL_CONV", 0)]
        tl = simulate(_CG(nodes, ev), lambda sn: 100.0, lambda sn: 10.0, issue_us=0.0)
        self.assertAlmostEqual(tl.node_end[0], 100.0)
        self.assertAlmostEqual(tl.node_end[1], 85.0)          # 75 us of loop, then the host op
        self.assertAlmostEqual(tl.total_us, 100.0)

    def test_unpriced(self):
        nodes = [_SN(0, "KERNEL_CONV")]
        tl = simulate(_CG(nodes, [("start", 0), ("drain", "KERNEL_CONV", 0)]),
                      lambda sn: None, lambda sn: None, issue_us=0.0)
        self.assertEqual(tl.unpriced, [0])
        self.assertEqual(tl.total_us, 0.0)


if __name__ == "__main__":
    unittest.main()
