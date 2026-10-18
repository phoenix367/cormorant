# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Remote prerequisite checks (cmake, make, gcc, XRT, sudo, UIO devices)."""

from typing import TYPE_CHECKING

from .colors import _green, _red, _dim
from .config import uio_devices_from_cfg

if TYPE_CHECKING:
    from .session import RemoteSession


def check_prerequisites(session: "RemoteSession", cfg: dict,
                         label_width: int = 32) -> bool:
    """Verify that the remote machine has tools and devices needed to build
    and run the inference project.  Returns True when all checks pass."""
    uio_devices = uio_devices_from_cfg(cfg)

    checks = [
        ("cmake --version 2>&1 | head -1", "cmake"),
        ("make --version  2>&1 | head -1", "make"),
        ("gcc  --version  2>&1 | head -1", "gcc"),
        (
            "pkg-config --exists xrt 2>/dev/null && echo 'xrt via pkg-config' || "
            "{ [ -f /opt/xilinx/xrt/include/xrt.h ] && echo 'xrt at /opt/xilinx/xrt'; } || "
            "{ [ -f /usr/include/xrt/xrt.h ]        && echo 'xrt at /usr/include/xrt'; } || "
            "echo '__MISSING__'",
            "xrt headers",
        ),
        (
            "sudo -n true 2>/dev/null && echo 'passwordless sudo OK' || "
            "[ \"$(id -u)\" = '0' ] && echo 'running as root' || "
            "echo '__MISSING__'",
            "sudo / root",
        ),
    ]

    for kernel_name, uio_name in uio_devices.items():
        checks.append((
            f"match=$(grep -rl '^{uio_name}$' /sys/class/uio/*/name 2>/dev/null | head -1); "
            f"[ -n \"$match\" ] && "
            f"echo \"/dev/uio$(echo $match | grep -o 'uio[0-9]*' | tail -1 | sed 's/uio//')\" "
            f"|| echo '__MISSING__'",
            f"uio ({kernel_name}: {uio_name})",
        ))

    if not uio_devices:
        checks.append((
            "grep -rl '' /sys/class/uio/*/name 2>/dev/null | head -1 | "
            "xargs -r cat 2>/dev/null || echo '__MISSING__'",
            "uio devices (using header defaults)",
        ))

    all_ok = True
    for cmd, label in checks:
        out, _, rc = session.exec(cmd, timeout=15)
        missing = "__MISSING__" in out or (rc != 0 and not out.strip())
        if missing:
            print(f"    {_red('MISSING')} {label}")
            all_ok = False
        else:
            desc = out.strip().splitlines()[0][:70]
            print(f"    {_green('OK')}      {label:<{label_width}} {_dim(desc)}")
    return all_ok
