"""yolo_post.py — YOLOv5 pre- and post-processing in numpy (doc/plans/YOLO_PLAN.md).

The FPGA graph ends at the three Detect convs (raw head maps [255][ny][nx]
at strides 8 / 16 / 32); everything after them runs here, as in Ultralytics'
YOLOv5 v7.0:

  letterbox()   an image -> the network input [3][640][640] in [0, 1]
                (scale to fit, centred, grey 114 padding; utils/augmentations.py)
  decode()      the head maps -> [N][85] candidates: cx cy w h (input pixels),
                objectness, 80 class probabilities (models/yolo.py Detect)
  nms()         class-aware greedy NMS (utils/general.py non_max_suppression)
  to_image()    boxes back to the original image's pixels
  evaluate()    mAP@0.5 and mAP@0.5:0.95 against labels (val.py's matching and
                ap_per_class, 101-point interpolation)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

SIZE = 640
NC = 80
STRIDES = (8, 16, 32)
ANCHORS = np.array([[10, 13, 16, 30, 33, 23],                # P3/8
                    [30, 61, 62, 45, 59, 119],               # P4/16
                    [116, 90, 156, 198, 373, 326]],          # P5/32
                   np.float64).reshape(3, 3, 2)
MAX_WH = 7680                                                # class offset of the batched NMS
IOU_V = np.linspace(0.5, 0.95, 10)                           # mAP@0.5:0.95's thresholds


def letterbox(img: Image.Image, size: int = SIZE, fill: int = 114):
    """(x [3][size][size] float64 in [0, 1], ratio, (left, top)): the image
    scaled by ``ratio`` to fit (bilinear), placed at (left, top) on a grey
    canvas — YOLOv5's letterbox with auto=False."""
    im = img.convert("RGB")
    w, h = im.size
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((size - nw) / 2 - 0.1)), int(round((size - nh) / 2 - 0.1))
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(im, (left, top))
    return np.asarray(canvas, np.float64).transpose(2, 0, 1) / 255.0, r, (left, top)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode(heads: Sequence[np.ndarray]) -> np.ndarray:
    """The three raw head maps ([1][255][ny][nx] or [255][ny][nx], P3 / P4 /
    P5) -> [N][85]: cx, cy, w, h in input pixels, objectness, class scores;
    rows in the export's order (level, anchor, y, x)."""
    out = []
    for i, h in enumerate(heads):
        h = np.asarray(h, np.float64)
        ny, nx = h.shape[-2:]
        y = _sigmoid(h.reshape(3, NC + 5, ny, nx)).transpose(0, 2, 3, 1)        # [3][ny][nx][85]
        gy, gx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        grid = np.stack([gx, gy], -1)[None].astype(np.float64)
        xy = (y[..., :2] * 2.0 - 0.5 + grid) * STRIDES[i]
        wh = (y[..., 2:4] * 2.0) ** 2 * ANCHORS[i][:, None, None, :]
        out.append(np.concatenate([xy, wh, y[..., 4:]], -1).reshape(-1, NC + 5))
    return np.concatenate(out, 0)


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    o = np.empty_like(b)
    o[:, 0], o[:, 1] = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    o[:, 2], o[:, 3] = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    return o


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a [N][4] with every box in b [M][4] (xyxy) -> [N][M]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])           # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
    order, keep = np.argsort(-scores, kind="stable"), []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        o = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][o <= iou]
    return keep


def nms(pred: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 300,
        multi_label: bool = False) -> np.ndarray:
    """decode()'s rows -> detections [K][6]: x1 y1 x2 y2 (input pixels), score,
    class.  score = objectness x class probability; candidates above ``conf``
    (objectness first), the best class or every class above ``conf``
    (``multi_label``, as val.py), class-aware greedy NMS at ``iou``."""
    x = pred[pred[:, 4] > conf]
    if not len(x):
        return np.zeros((0, 6))
    sc = x[:, 5:] * x[:, 4:5]
    box = xywh2xyxy(x[:, :4])
    if multi_label:
        i, j = np.nonzero(sc > conf)
        det = np.concatenate([box[i], sc[i, j, None], j[:, None].astype(np.float64)], 1)
    else:
        j = sc.argmax(1)
        s = sc[np.arange(len(j)), j]
        det = np.concatenate([box, s[:, None], j[:, None].astype(np.float64)], 1)[s > conf]
    if not len(det):
        return np.zeros((0, 6))
    det = det[np.argsort(-det[:, 4], kind="stable")[:30000]]
    keep = _greedy_nms(det[:, :4] + det[:, 5:6] * MAX_WH, det[:, 4], iou)[:max_det]
    return det[keep]


def to_image(det: np.ndarray, ratio: float, pad: Tuple[int, int], shape: Tuple[int, int]) -> np.ndarray:
    """Detections in input pixels -> the original image's (w, h = shape), clipped."""
    d = det.copy()
    d[:, [0, 2]] = ((d[:, [0, 2]] - pad[0]) / ratio).clip(0, shape[0])
    d[:, [1, 3]] = ((d[:, [1, 3]] - pad[1]) / ratio).clip(0, shape[1])
    return d


def labels_xyxy(lines: Iterable[str], shape: Tuple[int, int]) -> np.ndarray:
    """YOLO label lines (class cx cy w h, normalised) -> [L][5]: class, x1 y1 x2 y2 in pixels."""
    w, h = shape
    rows = [[float(v) for v in ln.split()] for ln in lines if ln.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows)
    b = xywh2xyxy(a[:, 1:5] * np.array([w, h, w, h]))
    return np.concatenate([a[:, :1], b], 1)


def match(det: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """val.py process_batch: [K][10] bool, detection k a true positive at
    IoU threshold t (each label matched once, best IoU first)."""
    correct = np.zeros((len(det), len(IOU_V)), bool)
    if not len(det) or not len(lab):
        return correct
    iou = box_iou(lab[:, 1:], det[:, :4])
    same = lab[:, :1] == det[:, 5]
    for t, thr in enumerate(IOU_V):
        x = np.nonzero((iou >= thr) & same)
        if x[0].size:
            m = np.stack([x[0], x[1], iou[x[0], x[1]]], 1)
            if x[0].size > 1:
                m = m[np.argsort(-m[:, 2], kind="stable")]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), t] = True
    return correct


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate(stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """``stats``: per image (correct [K][10], scores [K], classes [K], label
    classes [L]) -> {"map50", "map", "classes", "labels", "detections"}
    (val.py's ap_per_class over the classes with labels)."""
    tp = np.concatenate([s[0] for s in stats], 0) if stats else np.zeros((0, 10), bool)
    conf = np.concatenate([s[1] for s in stats])
    pcls = np.concatenate([s[2] for s in stats])
    tcls = np.concatenate([s[3] for s in stats])
    order = np.argsort(-conf, kind="stable")
    tp, pcls = tp[order], pcls[order]
    ap = []
    for c in np.unique(tcls):
        sel = pcls == c
        n_l, n_p = int((tcls == c).sum()), int(sel.sum())
        if not n_p:
            ap.append(np.zeros(len(IOU_V)))
            continue
        fpc, tpc = (1 - tp[sel]).cumsum(0), tp[sel].cumsum(0)
        rec = tpc / (n_l + 1e-16)
        pre = tpc / (tpc + fpc)
        ap.append(np.array([_ap(rec[:, t], pre[:, t]) for t in range(len(IOU_V))]))
    ap = np.array(ap) if ap else np.zeros((0, len(IOU_V)))
    return {"map50": float(ap[:, 0].mean()) if len(ap) else 0.0,
            "map": float(ap.mean()) if len(ap) else 0.0,
            "classes": int(len(ap)), "labels": int(len(tcls)), "detections": int(len(conf))}
