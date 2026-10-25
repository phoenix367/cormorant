# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The systemd units deploy.py and demo/miranda/deploy.py install: unit-file
quoting (each ExecStart word reads back as the argument), the chat unit's
start sequence (fpga_load.py, the R5 firmware, the CMA clean-up, Type=notify),
Miranda's ordering after the chat server without pulling it in, the server's
READY=1, and the FPGA bundle step refusing a config that names another
bitstream than the board runs.  Needs the scheduler venv (deploy.py imports
numpy and paramiko); skipped in a stdlib-only run."""

import contextlib
import importlib.util
import io
import os
import re
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _util

try:                                  # src.remote exits (SystemExit) when paramiko is missing
    with contextlib.redirect_stderr(io.StringIO()):
        import deploy
except (ImportError, SystemExit):                              # pragma: no cover
    deploy = None


def unquote(line: str) -> list:
    """The words of a unit-file command line as systemd reads the ones
    deploy.systemd_word writes: bare words, or double quotes with \\\\ \\" \\n;
    %% and $$ stand for % and $."""
    words = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"|(\S+)', line):
        w = m.group(2) if m.group(1) is None else re.sub(
            r"\\(.)", lambda e: "\n" if e.group(1) == "n" else e.group(1), m.group(1))
        words.append(w.replace("%%", "%").replace("$$", "$"))
    return words


def cfg(rpu=False, fpga=True, env=None):
    return {"remote": {"dir": "/root/kv260_chat", "weights_dir": "/root/bert_squad_weights"},
            "server": {"host": "0.0.0.0", "port": 8000, "api_key": None, "backends": ["bert-squad"],
                       "max_windows": 8, "doc_stride": 128, "queue_timeout": 120, "max_queue": 16,
                       "resident": "auto", "env": env or {}, "launcher": "systemd"},
            "fpga": {"bitstream_config": "/x/bitstream_config_kv260.json" if fpga else None},
            "rpu": {"enabled": rpu, "bsp_dir": None, "firmware": "rpu_firmware.elf"},
            "health_timeout": 180, "_bert": {}}


def directives(text: str, key: str) -> list:
    return [ln.split("=", 1)[1] for ln in text.splitlines() if ln.startswith(key + "=")]


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestQuoting(unittest.TestCase):

    def test_plain_words_stay_bare(self):
        for w in ("/usr/bin/python3", "--port", "8000", "-1", "a=b", "x@y:z,w.v"):
            self.assertEqual(deploy.systemd_word(w), w)

    def test_round_trip(self):
        words = ["a b", '[":", "\\"", "*"]', "50%", "$HOME", "tab\tx", "line\nbreak", "back\\slash", ""]
        line = deploy.exec_line(words)
        self.assertEqual(unquote(line), words)
        self.assertIn("%%", line)
        self.assertIn("$$HOME", line)

    def test_environment_keeps_dollar(self):
        self.assertEqual(deploy.systemd_word("A=x $y%", env=True), '"A=x $y%%"')


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestChatUnit(unittest.TestCase):

    def test_start_sequence(self):
        text = deploy.chat_unit(cfg(rpu=True))
        self.assertEqual(directives(text, "Type"), ["notify"])
        pre = directives(text, "ExecStartPre")
        self.assertEqual(unquote(pre[0]), ["/usr/bin/python3", "/root/kv260_chat/fpga_load.py",
                                           "/root/kv260_chat/fpga"])
        self.assertEqual(unquote(pre[1]), ["/bin/bash", "/root/kv260_chat/rpu/rpu_up.sh"])
        self.assertTrue(pre[2].startswith("-/bin/sh -c "))     # the CMA clean-up may fail
        self.assertEqual(directives(text, "ExecStopPost"), ["-/bin/sh /root/kv260_chat/rpu/rpu_down.sh"])
        self.assertEqual(directives(text, "TimeoutStartSec"), ["300"])
        self.assertEqual(directives(text, "WantedBy"), ["multi-user.target"])

    def test_exec_start_is_the_server_argv(self):
        c = cfg()
        c["smollm2"] = {"lib": "/l.so", "context": 1024, "reserve": 256, "temperature": 0.2, "top_p": 0.9,
                        "top_k": 50, "repetition_penalty": 1.1, "dry_multiplier": 0.8, "dry_base": 1.75,
                        "dry_allowed_length": 2, "dry_penalty_last_n": -1,
                        "dry_sequence_breakers": [":", "\"", "*"], "loop_guard": True, "cma_mb": 330}
        c["server"]["backends"].append("smollm2")
        (start,) = directives(deploy.chat_unit(c), "ExecStart")
        self.assertEqual(unquote(start), deploy.server_argv(c))

    def test_without_rpu_or_bundle(self):
        text = deploy.chat_unit(cfg(fpga=False, env={"INFERENCE_RPU": 0, "_note": "x"}))
        self.assertEqual(len(directives(text, "ExecStartPre")), 1)  # only the CMA clean-up
        self.assertEqual(directives(text, "ExecStopPost"), [])
        self.assertEqual(directives(text, "Environment"), ["INFERENCE_RPU=0"])


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestMirandaUnit(unittest.TestCase):

    def test_after_the_chat_server_without_starting_it(self):
        spec = importlib.util.spec_from_file_location(
            "miranda_deploy", os.path.join(_util.DEMO, "miranda", "deploy.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        argv = ["/usr/bin/python3", "-u", "/root/kv260_chat/miranda/miranda_server.py", "--wake", "hi there"]
        text = m.unit_text(cfg(), argv)
        self.assertIn("kv260-chat.service", directives(text, "After")[0].split())
        for key in ("Wants", "Requires", "BindsTo", "Requisite"):
            self.assertEqual(directives(text, key), [], key)
        self.assertEqual(unquote(directives(text, "ExecStart")[0]), argv)
        self.assertEqual(directives(text, "Restart"), ["on-failure"])


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestReady(unittest.TestCase):

    def test_sd_notify_sends_ready(self):
        with tempfile.TemporaryDirectory() as td, \
                socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            path = os.path.join(td, "notify")
            s.bind(path)
            with mock.patch.dict(os.environ, {"NOTIFY_SOCKET": path}):
                _util.server_mod.sd_notify("READY=1")
            self.assertEqual(s.recv(64), b"READY=1")
        with mock.patch.dict(os.environ, {}, clear=True):
            _util.server_mod.sd_notify("READY=1")              # outside systemd: nothing


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestStageFpga(unittest.TestCase):

    def test_refuses_another_bitstream_than_the_board_runs(self):
        session = mock.MagicMock()
        session.exec.return_value = ("applied\n" + "a" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
        with mock.patch("src.bitstream.bundle.resolve_bitstream_config",
                        return_value={"bit": Path("/x.bit"), "hwh": Path("/x.hwh"), "dtbo": Path("/pl.dtbo"),
                                      "overlay_name": "pl", "xclbinutil": "xclbinutil"}), \
                mock.patch("src.perf_calls.bitstream_id", return_value="bbbbbbbbbbbb"), \
                mock.patch("src.bitstream.bundle.make_bundle") as make, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(deploy.stage_fpga(session, cfg()))
        make.assert_not_called()
        self.assertIn("the board runs aaaaaaaaaaaa", out.getvalue())

    def test_board_bitstream_needs_the_overlay(self):
        session = mock.MagicMock()
        session.exec.return_value = ("c" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
        self.assertIsNone(deploy.board_bitstream(session, "pl"))
        session.exec.return_value = ("applied\n" + "c" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
        self.assertEqual(deploy.board_bitstream(session, "pl"), "c" * 12)


if __name__ == "__main__":
    unittest.main()
