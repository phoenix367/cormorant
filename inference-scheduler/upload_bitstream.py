#!/usr/bin/env python3
"""
upload_bitstream.py — Upload and load a Cormorant bitstream on KV260.

Converts the Vivado .bit file to a flat .bin, uploads it to /lib/firmware/
on the board, loads it via the FPGA manager sysfs interface, configures the
Zynq MPSoC PS AXI fabric registers from the .hwh metadata, then applies
the device tree overlay (.dtbo) so that UIO devices appear in /dev/.

See src/bitstream/ for the implementation and src/bitstream/platforms/kv260.py
for KV260-specific register tables and xclbin metadata.

Configuration
─────────────
File paths (bit / hwh / dtbo) can be supplied via a "bitstream" section in
the JSON config file.  Paths are resolved relative to the config file's
directory, so configs are portable across checkouts.

  {
    "ssh": { "host": "...", ... },
    "bitstream": {
      "bit":          "path/to/design.bit",
      "hwh":          "path/to/design.hwh",
      "dtbo":         "path/to/pl.dtbo",
      "overlay_name": null,
      "xclbinutil":   "xclbinutil"
    }
  }

CLI flags always override config values.  The hwh path is also auto-detected
as <bit_stem>.hwh when it sits alongside the .bit file.

Usage:
  # All paths from config
  python upload_bitstream.py --config bitstream_config_kv260.json

  # Override individual paths on the CLI
  python upload_bitstream.py --config bitstream_config_kv260.json \\
      --bit   ../hw/.../design_cormorant_wrapper.bit \\
      --dtbo  ../dts/kv260/pl.dtbo

  # Check board readiness only
  python upload_bitstream.py --config bitstream_config_kv260.json --check-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bitstream import check_board, upload_bitstream  # noqa: E402
from src.bitstream.board import fpga_state, list_uio_devices  # noqa: E402
from src.remote import (  # noqa: E402
    _green, _red, _dim,
    load_config,
    RemoteSession, hold_board_lock,
)

_BITSTREAM_DEFAULTS = {
    "bitstream": {
        "bit":          None,
        "hwh":          None,
        "dtbo":         None,
        "overlay_name": None,
        "xclbinutil":   "xclbinutil",
    }
}


def _cfg_path(value: str | None, config_dir: Path) -> Path | None:
    """Resolve a config-file path relative to the config file's directory."""
    if value is None:
        return None
    p = Path(value).expanduser()
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()


def _resolve_hwh(explicit: Path | None, bit_path: Path) -> Path:
    """
    Resolve the HWH path.
    1. Explicit value (from CLI or config)
    2. <bit_stem>.hwh alongside the .bit file
    """
    if explicit is not None:
        return explicit
    candidate = bit_path.with_suffix(".hwh")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"HWH file not found alongside {bit_path.name} "
        f"(looked for {candidate}).  Set bitstream.hwh in config or pass --hwh."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload a Cormorant bitstream to a KV260 board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", required=True, metavar="FILE",
        help="JSON config file.  May contain a 'bitstream' section with file paths.",
    )
    p.add_argument(
        "--bit", metavar="FILE", default=None,
        help="Vivado .bit file (overrides config).",
    )
    p.add_argument(
        "--hwh", metavar="FILE", default=None,
        help="Hardware-handoff .hwh (overrides config).  Default: auto-detect alongside .bit.",
    )
    p.add_argument(
        "--dtbo", metavar="FILE", default=None,
        help="Device tree overlay .dtbo (overrides config).",
    )
    p.add_argument(
        "--overlay-name", metavar="NAME", default=None,
        help=(
            "Configfs overlay dir name and /lib/firmware/<name>.bin "
            "(overrides config).  Default: .dtbo stem."
        ),
    )
    p.add_argument(
        "--xclbinutil", metavar="CMD", default=None,
        help="xclbinutil command or path (overrides config).  Default: xclbinutil from PATH.",
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="Verify board readiness only; do not upload or load anything.",
    )
    return p


def main() -> int:
    args   = _build_parser().parse_args()
    cfg    = load_config(args.config, _BITSTREAM_DEFAULTS)
    bs_cfg = cfg["bitstream"]

    config_dir = Path(args.config).expanduser().resolve().parent

    # ── resolve paths: CLI > config > built-in default ──────────────────────
    bit_path = (
        Path(args.bit).expanduser().resolve() if args.bit
        else _cfg_path(bs_cfg.get("bit"), config_dir)
    )
    dtbo_path = (
        Path(args.dtbo).expanduser().resolve() if args.dtbo
        else _cfg_path(bs_cfg.get("dtbo"), config_dir)
    )
    hwh_explicit = (
        Path(args.hwh).expanduser().resolve()  if args.hwh
        else _cfg_path(bs_cfg.get("hwh"), config_dir)
    )
    overlay_name = (
        args.overlay_name
        or bs_cfg.get("overlay_name")
        or (dtbo_path.stem if dtbo_path else None)
    )
    xclbinutil = args.xclbinutil or bs_cfg.get("xclbinutil") or "xclbinutil"

    # ── validate required inputs ─────────────────────────────────────────────
    if not args.check_only:
        if bit_path is None:
            print(
                _red("error: bit path is required — "
                     "set bitstream.bit in config or pass --bit"),
                file=sys.stderr,
            )
            return 1
        if dtbo_path is None:
            print(
                _red("error: dtbo path is required — "
                     "set bitstream.dtbo in config or pass --dtbo"),
                file=sys.stderr,
            )
            return 1

        try:
            hwh_path = _resolve_hwh(hwh_explicit, bit_path)
        except FileNotFoundError as exc:
            print(_red(f"error: {exc}"), file=sys.stderr)
            return 1

        for label, path in [(".bit", bit_path), (".dtbo", dtbo_path), (".hwh", hwh_path)]:
            if not path.exists():
                print(_red(f"error: {label} file not found: {path}"), file=sys.stderr)
                if label == ".bit":
                    print(
                        "       Build with: cd <repo>/build && make build_hw_kv260",
                        file=sys.stderr,
                    )
                return 1
    else:
        hwh_path = hwh_explicit  # not needed for --check-only

    # ── connect ──────────────────────────────────────────────────────────────
    hold_board_lock(cfg)                  # one job per board (src/remote/lock.py)
    ssh_cfg = cfg["ssh"]
    print(f"Connecting to {ssh_cfg['user']}@{ssh_cfg['host']}:{ssh_cfg['port']} …")
    session = RemoteSession(ssh_cfg)
    session.connect()
    print(f"  {_green('Connected')}")

    try:
        if args.check_only:
            print("\nBoard readiness check")
            ok    = check_board(session)
            state = fpga_state(session)
            print(f"\n  FPGA state:  {_dim(state)}")
            uios = list_uio_devices(session)
            if uios:
                print("  UIO devices:")
                for name, dev in uios:
                    print(f"    {dev}  {name}")
            else:
                print("  UIO devices: none")
            return 0 if ok else 1

        print(f"\n  .bit         {bit_path}")
        print(f"  .hwh         {hwh_path}")
        print(f"  .dtbo        {dtbo_path}")
        print(f"  name         {overlay_name}")
        print(f"  xclbinutil   {xclbinutil}")

        upload_bitstream(
            session, bit_path, hwh_path, dtbo_path, overlay_name, xclbinutil)

    except (RuntimeError, TimeoutError) as exc:
        print(f"\n{_red('Error:')} {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
