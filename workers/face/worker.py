#!/usr/bin/env python3
"""Face detection worker (YuNet, ONNX Runtime).

Request : {"image": "/path/to/photo.jpg"}
Response: {"model": "yunet-2023mar",
           "faces": [{"bbox": [x, y, w, h], "confidence": 0.98,
                      "landmarks": [[x, y], ... 5 points ...]}, ...]}

Model file: models/face_detection_yunet_2023mar.onnx next to this script (or
DAHLIA_FACE_MODEL). NumPy decoding and NMS; image I/O via
Pillow. No OpenCV dependency.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_image_rgb, run_loop  # noqa: E402

MODEL_NAME = "yunet-2023mar"
INPUT_SIZE = 640
SCORE_THRESH = 0.5
NMS_THRESH = 0.4
STRIDES = (8, 16, 32)
OUTPUT_NAMES = [f"{kind}_{stride}" for kind in ("cls", "obj", "bbox", "kps") for stride in STRIDES]

_session = None


def session():
    global _session
    if _session is None:
        import onnxruntime as ort

        model = os.environ.get("DAHLIA_FACE_MODEL") or str(
            Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
        )
        if not Path(model).exists():
            raise FileNotFoundError(f"face model not found: {model}")
        _session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return _session


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
    img = load_image_rgb(image_path)
    orig_w, orig_h = img.size
    sess = session()
    model_input = sess.get_inputs()[0]
    h, w = (dim if isinstance(dim, int) else INPUT_SIZE for dim in model_input.shape[2:])
    r = min(w / orig_w, h / orig_h)
    new_w, new_h = max(1, round(orig_w * r)), max(1, round(orig_h * r))
    resized = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (w, h))
    canvas.paste(resized, (0, 0))
    blob = np.asarray(canvas, dtype=np.float32)[:, :, ::-1].transpose(2, 0, 1)[None, ...]
    outputs = dict(zip(OUTPUT_NAMES, sess.run(OUTPUT_NAMES, {model_input.name: blob})))

    all_boxes, all_scores, all_kps = [], [], []
    for stride in STRIDES:
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0, 1)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0, 1)
        scores = np.sqrt(cls * obj)
        indices = np.flatnonzero(scores >= SCORE_THRESH)
        rows, cols = np.divmod(indices, w // stride)
        grid = np.stack([cols, rows], axis=1).astype(np.float32)
        deltas = outputs[f"bbox_{stride}"].reshape(-1, 4)[indices]
        centers = (grid + deltas[:, :2]) * stride
        half_sizes = np.exp(deltas[:, 2:]) * (stride / 2)
        boxes = np.concatenate([centers - half_sizes, centers + half_sizes], axis=1)
        kps = outputs[f"kps_{stride}"].reshape(-1, 5, 2)[indices]
        points = (grid[:, None, :] + kps) * stride
        all_boxes.append(boxes)
        all_scores.append(scores[indices])
        all_kps.append(points)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    kps = np.concatenate(all_kps)
    if len(boxes) == 0:
        return []
    keep = _nms(boxes, scores, NMS_THRESH)

    faces = []
    scale = np.array([orig_w / new_w, orig_h / new_h])
    for i in keep:
        x1, y1, x2, y2 = (boxes[i].reshape(2, 2) * scale).ravel().tolist()
        lm = (kps[i] * scale).tolist()
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
