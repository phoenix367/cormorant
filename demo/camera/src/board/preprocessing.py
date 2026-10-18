# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
preprocessing.py — board-side camera-frame preprocessing for the KV260
camera demo.  Runs ON the KV260 (imported by camera_loop.py).

Converts an OpenCV BGR camera frame into the exact NCHW ap_fixed<16,8>
int16 blob the generated inference project expects on its input buffer.

The encoding MUST stay identical to demo/image_classification's
download_assets.py::_encode_pixels so the camera demo's predictions match
the static-image demo bit for bit:

  1. resize to input_size x input_size,
  2. BGR -> RGB  (OpenCV frames are BGR; the model was trained on RGB —
     skipping this swap silently corrupts every prediction),
  3. normalize  (tf | unit | imagenet | none),
  4. HWC -> CHW,
  5. scale by 256 and pack as little-endian int16 (the ap_fixed<16,8>
     bit pattern; 1.0 -> 0x0100).

Pure numpy + OpenCV — deliberately no Pillow dependency, since the board
image may not ship it.
"""

import cv2
import numpy as np


# Standard torchvision ImageNet normalisation constants (RGB order) — used
# by the ONNX Model Zoo MobileNetV2 / ResNet / etc. trained in PyTorch.
# Pre-scaled by 255 so the encoder can operate directly on the input
# [0, 255] range: out = (pixel - mean) / std (matches the ResNet18 spec).
_IMAGENET_MEAN_255 = np.array((123.675, 116.28, 103.53), dtype=np.float32)
_IMAGENET_STD_255  = np.array(( 58.395,  57.12,  57.375), dtype=np.float32)


def preprocess_frame(bgr_frame: np.ndarray, input_size: int,
                     normalize: str) -> bytes:
    """Encode one BGR camera frame into the FPGA input blob.

    Returns ``3 * input_size * input_size * 2`` bytes (int16, NCHW).
    """
    resized = cv2.resize(bgr_frame, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]                       # BGR -> RGB

    a = rgb.astype(np.float32)
    if normalize == "tf":
        a = (a - 127.5) / 127.5                     # [-1, 1]  (TF MobileNet)
    elif normalize == "unit":
        a = a / 255.0                               # [0, 1]   (Keras-style)
    elif normalize == "none":
        a = a / 256.0                               # ~ raw byte / 256
    elif normalize == "imagenet":
        # ONNX Model Zoo ResNet18 spec, equivalent to the torchvision recipe:
        # (pixel - 255*mean) / (255*std).  Constants pre-scaled into the
        # 0..255 input range so we skip the intermediate /255 step.
        a = (a - _IMAGENET_MEAN_255) / _IMAGENET_STD_255
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r} "
                         f"(expected 'tf', 'unit', 'imagenet', 'none')")

    a = np.transpose(a, (2, 0, 1))                  # HWC -> CHW
    bits = np.rint(a * 256.0)                       # ap_fixed<16,8>
    bits = np.clip(bits, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(bits.reshape(-1)).tobytes()
