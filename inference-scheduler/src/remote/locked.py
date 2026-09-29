"""Run a command under the KV260 board lock, or print the lock path
(src/remote/lock.py):

    .venv/bin/python -m src.remote.locked --config CFG.json -- CMD ARG ...
    .venv/bin/python -m src.remote.locked --host 192.168.100.8 --print
"""

from .lock import main

if __name__ == "__main__":
    raise SystemExit(main())
