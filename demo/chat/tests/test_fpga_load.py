# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""fpga_load.py (the kv260-chat unit's first ExecStartPre) on a fake sysfs:
when the bundle's bitstream counts as loaded, why not, --check's exit
status, the refusal while another process holds the FPGA, and the stale
interrupt check (stdlib only)."""

import contextlib
import hashlib
import io
import json
import os
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
        self.paths = {k: os.path.join(root, k) for k in ("firmware", "overlays", "fpga", "uio", "bundle")}
        for p in self.paths.values():
            os.makedirs(p)
        self.irq_file = os.path.join(root, "interrupts")
        self.patches = [mock.patch.object(fpga_load, "FIRMWARE", self.paths["firmware"]),
                        mock.patch.object(fpga_load, "OVERLAYS", self.paths["overlays"]),
                        mock.patch.object(fpga_load, "FPGA", self.paths["fpga"]),
                        mock.patch.object(fpga_load, "UIO_CLASS", self.paths["uio"]),
                        mock.patch.object(fpga_load, "INTERRUPTS", self.irq_file)]
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


if __name__ == "__main__":
    unittest.main()
