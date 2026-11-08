#!/usr/bin/env python3
"""Face detection worker (SCRFD, ONNX Runtime).

Request : {"image": "/path/to/photo.jpg"}
Response: {"model": "scrfd-500m",
           "faces": [{"bbox": [x, y, w, h], "confidence": 0.98,
                      "landmarks": [[x, y], ... 5 points ...]}, ...]}

Model file: models/det_500m.onnx next to this script (or
DAHLIA_FACE_MODEL). Pure numpy decoding (anchors + NMS); image I/O via
Pillow. No OpenCV dependency.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run_loop  # noqa: E402

MODEL_NAME = "scrfd-500m"
INPUT_SIZE = 640
SCORE_THRESH = 0.5
NMS_THRESH = 0.4
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2

_session = None


def session():
    global _session
    if _session is None:
        import onnxruntime as ort

        model = os.environ.get("DAHLIA_FACE_MODEL") or str(
            Path(__file__).resolve().parent / "models" / "det_500m.onnx"
        )
        if not Path(model).exists():
            raise FileNotFoundError(f"face model not found: {model}")
        _session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return _session


def _anchor_centers(h: int, w: int, stride: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    centers = np.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], axis=-1)
    return centers.reshape(-1, 2).astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thresh]
    return keep


def detect(image_path: str) -> list[dict]:
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    r = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    resized = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    canvas.paste(resized, (0, 0))
    arr = np.asarray(canvas).astype(np.float32)
    blob = np.transpose((arr - 127.5) / 128.0, (2, 0, 1))[None, ...]

    sess = session()
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: blob})

    # SCRFD output layout: per stride -> scores, bboxes, keypoints
    # appear grouped as [s8, s16, s32, b8, b16, b32, k8, k16, k32].
    fmc = len(STRIDES)
    all_boxes, all_scores, all_kps = [], [], []
    for idx, stride in enumerate(STRIDES):
        scores = outputs[idx].reshape(-1)
        deltas = outputs[idx + fmc].reshape(-1, 4)
        kps = outputs[idx + 2 * fmc].reshape(-1, 5, 2)
        gh, gw = INPUT_SIZE // stride, INPUT_SIZE // stride
        centers = _anchor_centers(gh, gw, stride)
        centers = np.repeat(centers, NUM_ANCHORS, axis=0)
        # Positive offsets from anchor centers (distance2bbox / kps).
        boxes = np.stack(
            [
                centers[:, 0] - deltas[:, 0] * stride,
                centers[:, 1] - deltas[:, 1] * stride,
                centers[:, 0] + deltas[:, 2] * stride,
                centers[:, 1] + deltas[:, 3] * stride,
            ],
            axis=-1,
        )
        points = centers[:, None, :] + kps * stride
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_kps.append(points)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    kps = np.concatenate(all_kps)
    mask = scores > SCORE_THRESH
    boxes, scores, kps = boxes[mask], scores[mask], kps[mask]
    if len(boxes) == 0:
        return []
    keep = _nms(boxes, scores, NMS_THRESH)

    faces = []
    for i in keep:
        x1, y1, x2, y2 = (boxes[i] / r).tolist()
        lm = (kps[i] / r).tolist()
        faces.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "confidence": float(scores[i]),
                "landmarks": [[float(px), float(py)] for px, py in lm],
            }
        )
    faces.sort(key=lambda f: -f["confidence"])
    return faces


def handle(req: dict) -> dict:
    faces = detect(req["image"])
    return {"model": MODEL_NAME, "faces": faces}


if __name__ == "__main__":
    run_loop(handle)
