# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""The systemd units deploy.py and demo/miranda/deploy.py install: unit-file
quoting (each ExecStart word reads back as the argument), the chat unit's
start sequence (fpga_load.py, the R5 firmware, the CMA clean-up, Type=notify),
Miranda's ordering after the chat server without pulling it in, the server's
READY=1, its stop on a second SIGTERM, the nohup launcher's start and stop, and
the FPGA bundle step refusing a config that names another bitstream than the
board runs.  Needs the scheduler venv (deploy.py imports
numpy and paramiko); skipped in a stdlib-only run."""

import contextlib
import importlib.util
import io
import os
import re
import signal
import socket
import subprocess
import sys
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


class TestStop(unittest.TestCase):

    # The server with a stderr that sends it a second SIGTERM from inside the write of
    # its "shutting down" line, while log() holds its lock: the stop that hung (2026-10-27).
    WRAP = ("import os, runpy, signal, sys\n"
            "class Err:\n"
            "    def __init__(self, f): self.f, self.sent = f, False\n"
            "    def write(self, s):\n"
            "        if 'shutting down' in s and not self.sent:\n"
            "            self.sent = True\n"
            "            os.kill(os.getpid(), signal.SIGTERM)\n"
            "        return self.f.write(s)\n"
            "    def __getattr__(self, a): return getattr(self.f, a)\n"
            "sys.stderr = Err(sys.stderr)\n"
            "sys.argv = sys.argv[1:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n")

    def test_second_sigterm_inside_the_first(self):
        p = subprocess.Popen([sys.executable, "-c", self.WRAP, os.path.join(_util.CHAT, "kv260_chat_server.py"),
                              "--backend", "echo", "--host", "127.0.0.1", "--port", "0"],
                             cwd=_util.CHAT, stderr=subprocess.PIPE, text=True)
        try:
            for ln in p.stderr:
                if "listening" in ln:
                    break
            p.send_signal(signal.SIGTERM)
            _, err = p.communicate(timeout=20)
        finally:
            p.kill()
            p.wait()
        self.assertEqual(p.returncode, 0, err)
        self.assertEqual(err.count("shutting down"), 1, err)
        self.assertIn("stopped", err)


@unittest.skipIf(deploy is None, "deploy.py needs the scheduler venv (numpy, paramiko)")
class TestNohupLauncher(unittest.TestCase):
    """deploy.py's nohup start and stop in a local shell, for a stand-in server
    that counts its SIGTERMs and takes 0.5 s to stop."""

    FAKE = ("import signal, sys, time\n"
            "n = []\n"
            "signal.signal(signal.SIGTERM, lambda *a: n.append(1))\n"
            "print('ready', flush=True)\n"
            "while not n: time.sleep(0.05)\n"
            "time.sleep(0.5)\n"
            "open(sys.argv[1], 'w').write(str(len(n)))\n")

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = self.td.name
        self.server = os.path.join(self.dir, "kv260_chat_server.py")
        Path(self.server).write_text(self.FAKE)
        self.cfg = cfg(env={"INFERENCE_RPU": 0})
        self.cfg["remote"]["dir"] = self.dir
        self.cfg["_bert"] = {"run": {"use_sudo": False}}
        self.pid = None

    def tearDown(self):
        subprocess.run(["pkill", "-KILL", "-f", self.server])  # a start that failed leaves it running
        self.td.cleanup()

    def server_alive(self):
        with contextlib.suppress(OSError):
            return self.server in Path(f"/proc/{self.pid}/cmdline").read_text()
        return False

    def sh(self, cmd, timeout):
        """Like session.exec: returns once every holder of the output pipes closed them."""
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)

    def test_start_returns_with_the_server_pid_and_stop_signals_it_once(self):
        with mock.patch.object(deploy, "server_argv",
                               return_value=[sys.executable, self.server, os.path.join(self.dir, "signals")]):
            r = self.sh(deploy.nohup_start(self.cfg), timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.pid = int(Path(self.dir, "server.pid").read_text())
        self.assertTrue(self.server_alive())                  # its own pid, not a shell's or sudo's
        self.assertIn("INFERENCE_RPU=0", Path(f"/proc/{self.pid}/environ").read_text())
        self.assertTrue(_util.wait_until(lambda: "ready" in Path(self.dir, "server.log").read_text()))

        r = self.sh(deploy.nohup_stop(self.cfg), timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.server_alive())                 # the stop waited for it
        self.assertEqual(Path(self.dir, "signals").read_text(), "1")
        self.assertFalse(os.path.exists(os.path.join(self.dir, "server.pid")))

    def test_stop_leaves_another_process_alone(self):
        other = subprocess.Popen(["sleep", "30"])
        try:
            Path(self.dir, "server.pid").write_text(f"{other.pid}\n")
            r = self.sh(deploy.nohup_stop(self.cfg), timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIsNone(other.poll())
            self.assertFalse(os.path.exists(os.path.join(self.dir, "server.pid")))
        finally:
            other.kill()
            other.wait()


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

    def staged_load(self, status):
        """The commands load_staged_bitstream runs when the board's overlay
        'pl' reads `status` and the staged bundle is another bitstream's."""
        session = mock.MagicMock()

        def run(cmd, timeout=None):
            if "load.json" in cmd:
                return ('{"id": "aaaaaaaaaaaa", "overlay": "pl"}\n', "", 0)
            if "overlays/pl/status" in cmd:
                return (status + "c" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
            if "--check" in cmd:                               # the bundle's bitstream is not the one loaded
                return ("", "", 3)
            return ("", "", 0)
        session.exec.side_effect = run
        with contextlib.redirect_stdout(io.StringIO()):
            deploy.load_staged_bitstream(session, cfg())
        return [c.args[0] for c in session.exec.call_args_list]

    def test_staged_bundle_loads_after_a_boot(self):
        self.assertTrue(any("fpga_load.py" in c for c in self.staged_load("")))

    def test_staged_bundle_leaves_a_newer_bitstream_alone(self):
        # the board runs the bitstream the deploy is about to stage: swapping it for the
        # older bundle removed its overlay (refused under the R5's, 2026-10-28)
        self.assertFalse(any("fpga_load.py" in c for c in self.staged_load("applied\n")))

    def test_board_bitstream_needs_the_overlay(self):
        session = mock.MagicMock()
        session.exec.return_value = ("c" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
        self.assertIsNone(deploy.board_bitstream(session, "pl"))
        session.exec.return_value = ("applied\n" + "c" * 64 + "  /lib/firmware/pl.bin\n", "", 0)
        self.assertEqual(deploy.board_bitstream(session, "pl"), "c" * 12)


if __name__ == "__main__":
    unittest.main()
