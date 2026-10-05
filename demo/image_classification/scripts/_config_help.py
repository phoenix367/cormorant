"""Shared error formatting for the image-classification demo scripts.

The three demo subscripts (download_assets.py, generate_project.py,
deploy_and_run.py) and the top-level run_demo.py all need the same
"missing config" message.  Keeping it here avoids three drifting copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
EXAMPLE  = DEMO_DIR / "image_classification_config.json.example"


def format_missing_config(path: Path) -> str:
    """Return a multi-line, copy-pasteable bootstrap message."""
    lines = [
        "",
        "ERROR: image-classification demo config not found.",
        "",
        f"  Expected at: {path}",
    ]
    if EXAMPLE.exists():
        lines += [
            f"  Example   : {EXAMPLE}",
            "",
            "To bootstrap the demo:",
            "",
            f"  cp '{EXAMPLE}' '{path}'",
            f"  $EDITOR '{path}'",
            "",
            "Required edits before the first run:",
            "  • ssh.host         — KV260 hostname or IP (e.g. kv260.local)",
            "  • ssh.user         — SSH user on the board (typically 'root')",
            "  • ssh.key_file     — path to your SSH private key, or null + ssh.password",
            "  • local.driver_dirs.* — driver paths for each kernel.",
            "                          Build them first in build/ with:",
            "                            make driver_vectorop_rtl",
            "                            make driver_matmul_rtl",
            "                            make synthesize_conv_kv260",
            "                            make driver_pool_rtl",
            "",
            "Optional but commonly tuned:",
            "  • remote.uio_devices  — DT node labels of your loaded overlay",
            "                          (verify on the board: cat /sys/class/uio/uio*/name)",
            "  • run.top_k           — number of predictions printed per image",
            "  • preprocess.*        — input size / normalisation / resize mode",
            "",
            f"See {DEMO_DIR / 'README.md'} for the full description of each field.",
            "",
        ]
    else:
        lines += [
            "",
            "  The example config (image_classification_config.json.example) is",
            "  also missing — your checkout looks incomplete.  Restore it from",
            "  git or re-clone the repository.",
            "",
        ]
    return "\n".join(lines)


def missing_config_die(path: Path, *, exit_code: int = 2) -> None:
    """Print the bootstrap message to stderr and exit ``exit_code``."""
    print(format_missing_config(path), file=sys.stderr)
    sys.exit(exit_code)
