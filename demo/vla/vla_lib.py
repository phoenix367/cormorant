# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vla_lib.py — libsmolvla.so from Python (ctypes; numpy only): SmolVLA on the
KV260's kernels (demo/vla/src/vla_api.h, doc/plans/VLA_PLAN.md).

    from vla_lib import SmolVLA
    vla = SmolVLA("/root/smolvla/lib/libsmolvla.so")
    vla.set_image(0, top_rgb)                   # uint8 [h][w][3], any size
    vla.set_image(1, wrist_rgb)
    actions = vla.act(2, ids, state, seed=7)    # [chunk][action] float32, normalised

``ids``: the tokenizer's ids of the task + "\\n" (SmolVLM2's tokenizer.json, the
project's weights/tokenizer.json); ``state``: the robot's state normalised by
the dataset's mean / std; the actions are normalised the same way (the
robot's are the first ``action_dim``).
"""

from __future__ import annotations

import ctypes

import numpy as np


class VlaStats(ctypes.Structure):
    _fields_ = [("image_ms", ctypes.c_double), ("prefix_ms", ctypes.c_double), ("expert_ms", ctypes.c_double),
                ("act_ms", ctypes.c_double), ("cameras", ctypes.c_int), ("steps", ctypes.c_int)]


class SmolVLA:
    def __init__(self, lib: str, weights_dir: str = None):
        self.lib = ctypes.CDLL(lib)
        L = self.lib
        L.vla_open.argtypes = [ctypes.c_char_p]
        L.vla_last_error.restype = ctypes.c_char_p
        L.vla_set_image.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.vla_act.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p]
        L.vla_noise.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        L.vla_resize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.vla_last_stats.argtypes = [ctypes.POINTER(VlaStats)]
        L.vla_weights_dir.restype = ctypes.c_char_p
        rc = L.vla_open(weights_dir.encode() if weights_dir else None)
        if rc:
            raise RuntimeError(f"vla_open: {rc} {L.vla_last_error().decode()}")
        self.cameras = L.vla_cameras()
        self.image_size = L.vla_image_size()
        self.tok_max = L.vla_tok_max()
        self.state_size = L.vla_state_size()
        self.action_size = L.vla_action_size()
        self.state_dim = L.vla_state_dim()
        self.action_dim = L.vla_action_dim()
        self.chunk = L.vla_chunk()
        self.steps = L.vla_steps()

    def close(self):
        self.lib.vla_close()

    def _check(self, rc, what):
        if rc:
            raise RuntimeError(f"{what}: {rc} {self.lib.vla_last_error().decode()}")

    def set_image(self, cam: int, rgb: np.ndarray) -> None:
        a = np.ascontiguousarray(rgb, np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("rgb: uint8 [h][w][3]")
        self._check(self.lib.vla_set_image(cam, a.ctypes.data, a.shape[0], a.shape[1]), "vla_set_image")

    def noise(self, seed: int) -> np.ndarray:
        out = np.empty((self.chunk, self.action_size), np.float32)
        self.lib.vla_noise(ctypes.c_uint64(seed), out.ctypes.data)
        return out

    def act(self, cameras: int, ids, state, noise: np.ndarray = None, seed: int = 0) -> np.ndarray:
        ids = np.ascontiguousarray(ids, np.int32)
        st = np.ascontiguousarray(state, np.float32)
        nz = self.noise(seed) if noise is None else np.ascontiguousarray(noise, np.float32)
        if nz.shape != (self.chunk, self.action_size):
            raise ValueError(f"noise [{self.chunk}][{self.action_size}]")
        out = np.empty((self.chunk, self.action_size), np.float32)
        self._check(self.lib.vla_act(cameras, ids.ctypes.data, ids.size, st.ctypes.data, st.size, nz.ctypes.data,
                                     out.ctypes.data), "vla_act")
        return out

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(rgb, np.uint8)
        out = np.empty((self.image_size, self.image_size, 3), np.uint8)
        self._check(self.lib.vla_resize(a.ctypes.data, a.shape[0], a.shape[1], out.ctypes.data), "vla_resize")
        return out

    def stats(self) -> dict:
        s = VlaStats()
        self.lib.vla_last_stats(ctypes.byref(s))
        return {k: getattr(s, k) for k, _ in VlaStats._fields_}
