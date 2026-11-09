# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""vjepa2_lib.py — libvjepa2.so from Python (ctypes; numpy only): the V-JEPA 2
video classifier on the KV260's kernels (demo/video_classification/src/vjepa2_api.h,
doc/plans/VJEPA2_PLAN.md).

    from vjepa2_lib import VJepa2
    vj = VJepa2("/root/vjepa2/lib/libvjepa2.so")
    logits = vj.classify(frames)                 # uint8 [n][h][w][3], a decoded video
    print(vj.labels[int(logits.argmax())])

``classify`` samples the clip's 16 frames, resizes and crops them and runs the
network; ``classify_clip`` takes 16 frames already at the model's size
(``frame``).
"""

from __future__ import annotations

import ctypes

import numpy as np


class VJepa2Stats(ctypes.Structure):
    _fields_ = [("prep_ms", ctypes.c_double), ("run_ms", ctypes.c_double), ("total_ms", ctypes.c_double)]


class VJepa2:
    def __init__(self, lib: str, weights_dir: str = None):
        self.lib = ctypes.CDLL(lib)
        L = self.lib
        L.vjepa2_open.argtypes = [ctypes.c_char_p]
        L.vjepa2_last_error.restype = ctypes.c_char_p
        L.vjepa2_label.restype = ctypes.c_char_p
        L.vjepa2_sample_frames.argtypes = [ctypes.c_int, ctypes.c_void_p]
        L.vjepa2_frame.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.vjepa2_classify_clip.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.vjepa2_classify.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.vjepa2_last_stats.argtypes = [ctypes.POINTER(VJepa2Stats)]
        L.vjepa2_weights_dir.restype = ctypes.c_char_p
        rc = L.vjepa2_open(weights_dir.encode() if weights_dir else None)
        if rc:
            raise RuntimeError(f"vjepa2_open: {rc} {L.vjepa2_last_error().decode()}")
        self.frames = L.vjepa2_frames()
        self.crop = L.vjepa2_crop()
        self.short_side = L.vjepa2_short_side()
        self.classes = L.vjepa2_classes()
        self.labels = [L.vjepa2_label(c).decode() for c in range(self.classes)]

    def close(self):
        self.lib.vjepa2_close()

    def _check(self, rc, what):
        if rc:
            raise RuntimeError(f"{what}: {rc} {self.lib.vjepa2_last_error().decode()}")

    def sample_frames(self, n: int) -> np.ndarray:
        idx = np.zeros(self.frames, np.int32)
        self._check(self.lib.vjepa2_sample_frames(n, idx.ctypes.data), "vjepa2_sample_frames")
        return idx

    def frame(self, rgb: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(rgb, np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("rgb: uint8 [h][w][3]")
        out = np.empty((self.crop, self.crop, 3), np.uint8)
        self._check(self.lib.vjepa2_frame(a.ctypes.data, a.shape[0], a.shape[1], out.ctypes.data), "vjepa2_frame")
        return out

    def classify_clip(self, clip: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(clip, np.uint8)
        if a.shape != (self.frames, self.crop, self.crop, 3):
            raise ValueError(f"clip: uint8 [{self.frames}][{self.crop}][{self.crop}][3]")
        out = np.empty(self.classes, np.float32)
        self._check(self.lib.vjepa2_classify_clip(a.ctypes.data, out.ctypes.data), "vjepa2_classify_clip")
        return out

    def classify(self, frames: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(frames, np.uint8)
        if a.ndim != 4 or a.shape[3] != 3:
            raise ValueError("frames: uint8 [n][h][w][3]")
        out = np.empty(self.classes, np.float32)
        self._check(self.lib.vjepa2_classify(a.ctypes.data, a.shape[0], a.shape[1], a.shape[2], out.ctypes.data),
                    "vjepa2_classify")
        return out

    def stats(self) -> dict:
        st = VJepa2Stats()
        self.lib.vjepa2_last_stats(ctypes.byref(st))
        return {f: getattr(st, f) for f, _ in VJepa2Stats._fields_}
