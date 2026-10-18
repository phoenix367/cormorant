# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
visualization.py — OpenCV overlay helpers for the KV260 camera demo.

draw_text_with_background / get_contrasting_color / calculate_luminance are
adapted from the kria-camera-demo project (Copyright 2026 GradeBuilder SL,
Apache License 2.0) — see that project's LICENSE.  annotate_frame() is new
and composes the demo's per-frame overlay.

Runs ON the KV260 (imported by camera_loop.py).
"""

import cv2


def calculate_luminance(bgr_color):
    """Relative luminance of a BGR color (ITU-R BT.709), range [0, 255]."""
    b, g, r = bgr_color[0], bgr_color[1], bgr_color[2]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def get_contrasting_color(image, x, y, width, height):
    """White or black, whichever contrasts with the mean background colour."""
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + width), min(img_h, y + height)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (255, 255, 255)
    luminance = calculate_luminance(cv2.mean(roi)[:3])
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def draw_text_with_background(image, text, position, font, font_scale,
                              text_color, thickness, bg_opacity=0.6):
    """Draw *text* with a semi-transparent contrasting background box."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale,
                                                 thickness)
    x, y = position
    pad = 5
    bg = (x - pad, y - text_h - pad, x + text_w + pad, y + baseline + pad)

    overlay = image.copy()
    bg_color = tuple(255 - c for c in text_color)
    cv2.rectangle(overlay, (bg[0], bg[1]), (bg[2], bg[3]), bg_color, -1)
    cv2.addWeighted(overlay, bg_opacity, image, 1 - bg_opacity, 0, image)
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)
    return image


def annotate_frame(bgr_frame, predictions, infer_ms, disp_fps, frame_idx,
                   power_w=None):
    """Compose the camera-demo overlay onto a copy of *bgr_frame*.

    predictions : list of (label_str, probability) sorted best-first.
    infer_ms    : FPGA inference latency for this frame.
    disp_fps    : measured host-display refresh rate.
    power_w     : whole-board (SOM) power draw in watts, or None if no
                  power sensor was available.
    Returns the annotated BGR image (the input frame is left untouched).
    """
    out = bgr_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    # Top-1 banner.
    if predictions:
        label, prob = predictions[0]
        draw_text_with_background(
            out, f"{label}  {prob * 100.0:.1f}%",
            (14, 40), font, 0.9, white, 2)

    # Remaining top-K, smaller, stacked below the banner.
    y = 74
    for rank, (label, prob) in enumerate(predictions[1:], start=2):
        draw_text_with_background(
            out, f"{rank}. {label}  {prob * 100.0:.1f}%",
            (14, y), font, 0.5, white, 1)
        y += 26

    # Performance line along the bottom edge.
    power_str = f"{power_w:.1f} W" if power_w is not None else "n/a"
    h = out.shape[0]
    draw_text_with_background(
        out,
        f"infer {infer_ms:.0f} ms   display {disp_fps:.1f} fps   "
        f"power {power_str}   frame {frame_idx}",
        (14, h - 16), font, 0.5, white, 1)
    return out
