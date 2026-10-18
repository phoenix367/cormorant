# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Persistent SSH + SFTP session for remote board interactions."""

import os
import socket
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

try:
    import paramiko
    import paramiko.sftp_client
except ImportError:
    print("error: paramiko is required.  pip install paramiko", file=sys.stderr)
    sys.exit(1)


class RemoteSession:
    """
    Persistent SSH+SFTP connection to a remote board.

    All remote commands run via exec(); directory uploads via upload_dir().
    A single transport is reused for the entire session.
    """

    def __init__(self, ssh_cfg: dict) -> None:
        self._cfg    = ssh_cfg
        self._client: Optional[paramiko.SSHClient] = None

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        cfg    = self._cfg
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict = {
            "hostname":       cfg["host"],
            "username":       cfg["user"],
            "port":           int(cfg["port"]),
            "timeout":        float(cfg["connect_timeout"]),
            "allow_agent":    True,
            "look_for_keys":  True,
            "banner_timeout": 30,
        }
        key_file = cfg.get("key_file")
        if key_file:
            kwargs["key_filename"]  = os.path.expanduser(key_file)
            kwargs["look_for_keys"] = False
        password = cfg.get("password")
        if password:
            kwargs["password"] = password
        client.connect(**kwargs)
        self._client = client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ── remote command execution ─────────────────────────────────────────────

    def exec(self, command: str, timeout: int = 120) -> Tuple[str, str, int]:
        """Run *command* on the remote host.  Returns (stdout, stderr, exit_code)."""
        if self._client is None:
            raise RuntimeError("RemoteSession: not connected")
        try:
            _, stdout_ch, stderr_ch = self._client.exec_command(
                command, timeout=float(timeout), get_pty=False)
            stdout = stdout_ch.read().decode("utf-8", errors="replace")
            stderr = stderr_ch.read().decode("utf-8", errors="replace")
            rc     = stdout_ch.channel.recv_exit_status()
        except socket.timeout as exc:
            raise TimeoutError(
                f"Remote command timed out after {timeout}s:\n  {command}") from exc
        return stdout, stderr, rc

    def exec_checked(self, command: str, timeout: int = 120) -> Tuple[str, str]:
        """Like exec() but raises RuntimeError on non-zero exit."""
        out, err, rc = self.exec(command, timeout=timeout)
        if rc != 0:
            raise RuntimeError(
                f"Remote command failed (rc={rc}):\n  {command}\n"
                f"stdout:\n{out}\nstderr:\n{err}"
            )
        return out, err

    # ── file transfer ────────────────────────────────────────────────────────

    def upload_dir(self, local_dir: Path, remote_dir: str,
                   on_file: Optional[Callable[[str], None]] = None) -> int:
        """Recursively upload *local_dir* to *remote_dir* via SFTP.

        *on_file* is called with the relative file path before each upload.
        Returns the number of files uploaded.
        """
        sftp = self._client.open_sftp()
        n    = 0
        try:
            self._mkdir_p(sftp, remote_dir)
            for local_path in sorted(local_dir.rglob("*")):
                rel    = local_path.relative_to(local_dir)
                remote = f"{remote_dir}/{rel.as_posix()}"
                if local_path.is_dir():
                    try:
                        sftp.mkdir(remote)
                    except OSError:
                        pass
                else:
                    if on_file:
                        on_file(str(rel))
                    sftp.put(str(local_path), remote)
                    n += 1
        finally:
            sftp.close()
        return n

    @staticmethod
    def _mkdir_p(sftp: "paramiko.SFTPClient", remote_path: str) -> None:
        """Create *remote_path* and all missing parent directories."""
        parts   = Path(remote_path).parts
        current = ""
        for part in parts:
            if part == "/":
                current = "/"
                continue
            current = current.rstrip("/") + "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                try:
                    sftp.mkdir(current)
                except OSError:
                    pass
