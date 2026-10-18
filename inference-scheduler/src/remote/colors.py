# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Terminal colour helpers. Colours are suppressed when stdout is not a TTY."""

import sys

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def _green(s: str)  -> str: return _c("32", s)
def _red(s: str)    -> str: return _c("31", s)
def _yellow(s: str) -> str: return _c("33", s)
def _bold(s: str)   -> str: return _c("1",  s)
def _dim(s: str)    -> str: return _c("2",  s)
def _cyan(s: str)   -> str: return _c("36", s)
