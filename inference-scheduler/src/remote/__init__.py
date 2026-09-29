"""
src.remote — shared SSH/config utilities for run_remote_*.py scripts.
"""

from .colors  import _green, _red, _yellow, _bold, _dim, _cyan
from .config  import SHARED_DEFAULTS, deep_merge, load_config, uio_devices_from_cfg
from .session import RemoteSession
from .checks  import check_prerequisites
from .lock    import LOCK_ENV, board_lock, default_lock_path, hold_board_lock, lock_path

__all__ = [
    "_green", "_red", "_yellow", "_bold", "_dim", "_cyan",
    "SHARED_DEFAULTS", "deep_merge", "load_config", "uio_devices_from_cfg",
    "RemoteSession",
    "check_prerequisites",
    "LOCK_ENV", "board_lock", "default_lock_path", "hold_board_lock", "lock_path",
]
