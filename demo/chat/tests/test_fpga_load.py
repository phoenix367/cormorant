# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""fpga_load.py (the kv260-chat unit's first ExecStartPre) on a fake sysfs:
when the bundle's bitstream counts as loaded, why not, --check's exit
status, the refusal while another process holds the FPGA, the stale
interrupt check, and the Cortex-R5's overlays above ours (stdlib only)."""

import contextlib
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from unittest import mock

import _util  # noqa: F401  (demo/chat on sys.path)
import fpga_load

BIN = b"\x00\x09\x0f\xf0" * 64
ID = hashlib.sha256(BIN).hexdigest()[:12]
NODES = ["fabric_vecop@a0000000", "fabric_matmul@a0010000"]
INTERRUPTS = (" 66:   0   0   0   0     GICv2 121 Level     fabric_vecop\n"
              " 67:   5   0   0   0     GICv2 122 Level     fabric_matmul\n")


class FakeBoard(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = self.td.name
        self.paths = {k: os.path.join(root, k) for k in ("firmware", "overlays", "fpga", "uio", "bundle",
                                                         "remoteproc", "dt")}
        for p in self.paths.values():
            os.makedirs(p)
        self.irq_file = os.path.join(root, "interrupts")
        self.patches = [mock.patch.object(fpga_load, "FIRMWARE", self.paths["firmware"]),
                        mock.patch.object(fpga_load, "OVERLAYS", self.paths["overlays"]),
                        mock.patch.object(fpga_load, "FPGA", self.paths["fpga"]),
                        mock.patch.object(fpga_load, "UIO_CLASS", self.paths["uio"]),
                        mock.patch.object(fpga_load, "INTERRUPTS", self.irq_file),
                        mock.patch.object(fpga_load, "REMOTEPROC", self.paths["remoteproc"]),
                        mock.patch.object(fpga_load, "LIVE_DT", self.paths["dt"])]
        for p in self.patches:
            p.start()
        self.load = {"id": ID, "overlay": "pl", "pl0_hz": 100000000, "axi_writes": [],
                     "uio_nodes": NODES, "uio_irqs": {NODES[0]: 121, NODES[1]: 122}}
        with open(os.path.join(self.paths["bundle"], "load.json"), "w") as f:
            json.dump(self.load, f)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def put(self, rel, text):
        path = os.path.join(self.td.name, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb" if isinstance(text, bytes) else "w") as f:
            f.write(text)

    def board_runs_it(self):
        self.put("overlays/pl/status", "applied\n")
        self.put("firmware/pl.bin", BIN)
        self.put("fpga/state", "operating\n")
        self.put("uio/uio4/name", "fabric_vecop\n")
        self.put("uio/uio5/name", "fabric_matmul\n")
        self.put("interrupts", INTERRUPTS)

    def check(self, held=()):
        with mock.patch.object(fpga_load, "holders", return_value=list(held)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = fpga_load.main(["--check", self.paths["bundle"]])
        return rc, out.getvalue()


class TestLoaded(FakeBoard):

    def test_loaded(self):
        self.board_runs_it()
        self.assertEqual(fpga_load.loaded(self.load), [])
        self.assertEqual(self.check()[0], 0)

    def test_after_a_boot(self):
        self.put("fpga/state", "operating\n")                  # the starter-kit design
        why = fpga_load.loaded(self.load)
        self.assertEqual(len(why), 3, why)                     # no overlay, no pl.bin, no UIO devices
        rc, out = self.check()
        self.assertEqual(rc, 3)
        self.assertIn("not loaded", out)

    def test_another_bitstream(self):
        self.board_runs_it()
        self.put("firmware/pl.bin", b"other")
        (why,) = fpga_load.loaded(self.load)
        self.assertIn(f"not {ID}", why)

    def test_missing_uio_device(self):
        self.board_runs_it()
        os.remove(os.path.join(self.paths["uio"], "uio5", "name"))
        self.assertEqual(fpga_load.loaded(self.load), ["no UIO device fabric_matmul"])

    def test_in_use(self):
        self.board_runs_it()
        rc, out = self.check(held=["123 ./bench"])
        self.assertEqual(rc, 3)
        self.assertIn("in use by pid 123", out)
        with mock.patch.object(fpga_load, "holders", return_value=["123 ./bench"]), \
                mock.patch.object(fpga_load, "load_bundle") as load, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(fpga_load.main([self.paths["bundle"]]), 1)
        load.assert_not_called()
        self.assertIn("in use", out.getvalue())

    def test_stale_interrupt(self):
        self.board_runs_it()
        self.assertEqual(fpga_load.stale_irqs(self.load["uio_irqs"]), {})
        self.put("interrupts", INTERRUPTS.replace("122", "61"))
        self.assertEqual(fpga_load.stale_irqs(self.load["uio_irqs"]), {NODES[1]: (61, 122)})

    def test_no_bundle(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(fpga_load.main([os.path.join(self.td.name, "nowhere")]), 1)
        self.assertIn("no bundle", out.getvalue())


class TestR5Overlays(FakeBoard):
    """The Cortex-R5's overlays stay after its firmware stops, and the kernel refuses
    to remove ours from under them (2026-10-28: "#2 overlaps with #4 @/__symbols__").
    Their order comes from the live phandles (on the board: 155 / 169 / 172)."""

    def overlays(self, *names):
        for n in names:
            os.makedirs(os.path.join(self.paths["overlays"], n), exist_ok=True)

    def phandles(self, **ph):
        paths = {"fabric_vecop": "/axi/fabric_vecop@a0000000", "tcm_0a": "/tcm@ffe00000",
                 "rpu_ipi": "/axi/rpu_ipi@ff340000"}
        for label, p in ph.items():
            self.put(f"dt/__symbols__/{label}", paths[label].encode() + b"\0")
            self.put(f"dt{paths[label]}/phandle", struct.pack(">I", p))

    def r5(self, state):
        self.put("remoteproc/remoteproc0/state", state + "\n")

    def test_above_ours_newest_first(self):
        self.overlays("pl", "rpu", "rpu-ipi")
        self.phandles(fabric_vecop=155, tcm_0a=169, rpu_ipi=172)
        self.r5("offline")
        self.assertEqual(fpga_load.r5_overlays_above("pl", "fabric_vecop"), ["rpu-ipi", "rpu"])

    def test_below_ours_left(self):
        self.overlays("pl", "rpu", "rpu-ipi")
        self.phandles(fabric_vecop=200, tcm_0a=169, rpu_ipi=172)
        self.assertEqual(fpga_load.r5_overlays_above("pl", "fabric_vecop"), [])

    def test_unknown_order_refused(self):
        self.overlays("pl", "rpu")
        self.phandles(fabric_vecop=155)
        with self.assertRaisesRegex(fpga_load.LoadError, "cannot tell.*tcm_0a.*reboot"):
            fpga_load.r5_overlays_above("pl", "fabric_vecop")

    def test_running_firmware_refused(self):
        self.overlays("pl", "rpu")
        self.phandles(fabric_vecop=155, tcm_0a=169)
        self.r5("running")
        with self.assertRaisesRegex(fpga_load.LoadError, "Cortex-R5 firmware is running"):
            fpga_load.r5_overlays_above("pl", "fabric_vecop")

    def test_load_removes_them_before_ours(self):
        self.overlays("pl", "rpu", "rpu-ipi")
        self.phandles(fabric_vecop=155, tcm_0a=169, rpu_ipi=172)
        removed = []

        class Stop(Exception):
            pass
        with mock.patch.object(fpga_load, "remove_overlay", side_effect=removed.append), \
                mock.patch.object(fpga_load, "check_kernels_free", side_effect=Stop):
            with self.assertRaises(Stop):
                fpga_load.load_bundle(self.load, self.paths["bundle"])
        self.assertEqual(removed, ["pynq", "rpu-ipi", "rpu", "pl"])

if __name__ == "__main__":
    unittest.main()
