"""vlm_image.py — images of OpenAI chat requests for SmolVLM
(doc/plans/CHAT_PLAN.md §23): an `image_url` part's URL (a base64 data URL)
-> the 512 x 512 RGB pixels the library's llm_image() takes.

Preprocessing = transformers' Idefics3 image processor with image splitting
off (vlm_study.load_pixels, the study's and the calibration's input): RGB,
a LANCZOS resize of the longest edge to 2048 (aspect kept, the short side
rounded up to even), a LANCZOS resize to 512 x 512.  Needs Pillow (the
board: python3-pil).  Decoded images are kept in a small LRU cache by the
URL's hash — chat clients resend the whole history every turn.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

MAX_URL_BYTES = 32 * 1024 * 1024        # a data URL of up to ~24 MB of image
MAX_PIXELS = 50_000_000                 # decompression-bomb guard


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class Image:
    pixels: bytes          # size x size x 3 RGB, row-major
    size: int
    sha: str               # of the pixels: the prefix cache's key
    width: int = 0         # the original's
    height: int = 0


def _max_len_size(h: int, w: int, max_len: int):
    """transformers' _resize_output_size_rescale_to_max_len (even short side)."""
    ar = w / h
    if w >= h:
        w, h = max_len, int(max_len / ar)
        h += h % 2
    else:
        h, w = max_len, int(max_len * ar)
        w += w % 2
    return max(h, 1), max(w, 1)


def url_bytes(url: str) -> bytes:
    if not isinstance(url, str) or not url:
        raise ImageError("an image_url part needs a url")
    if len(url) > MAX_URL_BYTES:
        raise ImageError(f"the image URL is longer than {MAX_URL_BYTES} bytes")
    if not url.startswith("data:"):
        raise ImageError("only data URLs (data:image/...;base64,...) are supported; "
                         "the server does not fetch remote images")
    head, sep, data = url.partition(",")
    if not sep or ";base64" not in head:
        raise ImageError("the data URL must be base64-encoded (data:image/png;base64,...)")
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ImageError(f"invalid base64 in the data URL: {e}") from None


def preprocess(data: bytes, size: int = 512, longest: int = 2048) -> Image:
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImageError("image input needs Pillow on the server (python3-pil)") from None
    PILImage.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        im = PILImage.open(io.BytesIO(data))
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise ImageError(f"the image has {w} x {h} pixels (at most {MAX_PIXELS})")
        im = im.convert("RGB")
    except ImageError:
        raise
    except Exception as e:                                   # PIL raises many kinds
        raise ImageError(f"cannot decode the image: {e}") from None
    h2, w2 = _max_len_size(h, w, longest)
    lanczos = getattr(PILImage, "Resampling", PILImage).LANCZOS
    im = im.resize((w2, h2), lanczos).resize((size, size), lanczos)
    px = im.tobytes()
    return Image(px, size, hashlib.sha256(px).hexdigest(), w, h)


class ImageCache:
    """URL -> Image, LRU over the URL's SHA-256."""

    def __init__(self, size: int = 512, entries: int = 16):
        self.size = size
        self.entries = entries
        self._d: "OrderedDict[str, Image]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, url: str) -> Image:
        key = hashlib.sha256(url.encode("utf-8", "surrogatepass")).hexdigest()
        with self._lock:
            im: Optional[Image] = self._d.get(key)
            if im is not None:
                self._d.move_to_end(key)
                self.hits += 1
                return im
        im = preprocess(url_bytes(url), self.size)
        with self._lock:
            self.misses += 1
            self._d[key] = im
            while len(self._d) > self.entries:
                self._d.popitem(last=False)
        return im


__all__ = ("Image", "ImageCache", "ImageError", "preprocess", "url_bytes")
