# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""
Hugging Face safetensors checkpoints as float32 arrays (numpy only: the
studies load this file by path, without the scheduler package).

A checkpoint is a ``.safetensors`` file or a directory holding
``model.safetensors`` or a sharded ``model.safetensors.index.json``.
bf16 / f16 / f32 tensors become float32 exactly.  ``LazySafetensors`` and
``Tracked`` record which tensors were read: ``check_consumed`` refuses a
checkpoint with tensors nobody read (biases, extra norms of a block the
reader does not implement — doc/plans/CHAT_PLAN.md §28).
"""

from __future__ import annotations

import fnmatch
import json
import os
import struct
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

INDEX = "model.safetensors.index.json"


def checkpoint_files(path: str) -> List[str]:
    """The safetensors files of a checkpoint: ``path`` itself (a file), or a
    directory's ``model.safetensors`` or the shards its index names."""
    if not os.path.isdir(path):
        return [path]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    index = os.path.join(path, INDEX)
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(path, s) for s in shards]
    raise FileNotFoundError(f"{path}: neither model.safetensors nor {INDEX}")


def _st_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """A safetensors tensor's bytes -> float32 (bf16 upcast exactly)."""
    if dtype == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return raw.view(np.float32).copy()
    raise ValueError(f"dtype {dtype}")


def _st_header(f) -> Tuple[int, dict]:
    n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n, json.loads(f.read(n))


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """bf16 / f16 / f32 safetensors -> float32 arrays (bf16 upcast exactly);
    ``path`` as in ``checkpoint_files``."""
    out = {}
    for fp in checkpoint_files(path):
        with open(fp, "rb") as f:
            _, hdr = _st_header(f)
            blob = np.frombuffer(f.read(), np.uint8)
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            if name in out:
                raise ValueError(f"{name}: in two shards of {path}")
            a, b = m["data_offsets"]
            try:
                arr = _st_float32(blob[a:b], m["dtype"])
            except ValueError:
                raise ValueError(f"{name}: dtype {m['dtype']}") from None
            out[name] = arr.reshape(m["shape"])
    return out


class LazySafetensors(Mapping):
    """``load_safetensors`` without the float32 copy of the checkpoint: the
    files are memory-mapped and a tensor becomes float32 when read — every read
    converts anew and nothing is kept (the same values as load_safetensors).
    For frontends that build one entry at a time (generate_llm_project.py).
    ``select({name: name in the file})`` gives a renamed subset; every view
    shares the record of the tensors read (``unread()``)."""

    def __init__(self, path: str, names: Optional[Dict[str, str]] = None):
        self.path = path
        self._blobs, self._meta = [], {}
        for fp in checkpoint_files(path):
            with open(fp, "rb") as f:
                off, hdr = _st_header(f)
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if k in self._meta:
                    raise ValueError(f"{k}: in two shards of {path}")
                self._meta[k] = (len(self._blobs), v)
            self._blobs.append(np.memmap(fp, np.uint8, mode="r", offset=off))
        self._names = names if names is not None else {k: k for k in self._meta}
        self._read = set()

    def select(self, names: Dict[str, str]) -> "LazySafetensors":
        out = object.__new__(LazySafetensors)
        out.path, out._blobs, out._meta, out._read = self.path, self._blobs, self._meta, self._read
        out._names = {n: self._names[f] for n, f in names.items()}
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        fname = self._names[name]
        i, m = self._meta[fname]
        a, b = m["data_offsets"]
        try:
            arr = _st_float32(np.asarray(self._blobs[i][a:b]), m["dtype"])
        except ValueError:
            raise ValueError(f"{name}: dtype {m['dtype']}") from None
        self._read.add(fname)
        return arr.reshape(m["shape"])

    def __contains__(self, name) -> bool:
        return name in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def unread(self) -> List[str]:
        """The checkpoint's tensors (file names) no view has read."""
        return sorted(set(self._meta) - self._read)


class Tracked(Mapping):
    """A read-only view of a {name: array} dict that records the names read
    (``Mapping.get`` / ``items`` / ``values`` read through ``__getitem__``)."""

    def __init__(self, d: Mapping):
        self._d = d
        self._read = set()

    def __getitem__(self, name: str) -> np.ndarray:
        v = self._d[name]
        self._read.add(name)
        return v

    def __contains__(self, name) -> bool:
        return name in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def unread(self) -> List[str]:
        return sorted(set(self._d) - self._read)


# Tensors Hugging Face itself never uses: RoPE buffers some exports keep.
NEVER_READ = ("*.rotary_emb.inv_freq",)


def check_consumed(W, what: str, ignore: Iterable[str] = ()) -> None:
    """Raise when ``W`` (LazySafetensors / Tracked) holds tensors nobody read
    — weights of a block feature the reader does not implement.  ``ignore``:
    fnmatch patterns of tensors that are legitimately unused (a tied
    checkpoint's stored lm_head)."""
    pats = tuple(NEVER_READ) + tuple(ignore)
    left = [n for n in W.unread() if not any(fnmatch.fnmatchcase(n, p) for p in pats)]
    if left:
        more = f" (+{len(left) - 8} more)" if len(left) > 8 else ""
        raise ValueError(f"{what}: {len(left)} checkpoint tensors are not used by this stack — "
                         f"a block feature it does not implement: {', '.join(left[:8])}{more}")


__all__ = ("INDEX", "LazySafetensors", "NEVER_READ", "Tracked", "check_consumed",
           "checkpoint_files", "load_safetensors")
