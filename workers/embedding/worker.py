#!/usr/bin/env python3
"""Face embedding worker (ArcFace family, ONNX Runtime).

Request : {"image": "/path/to/photo.jpg"}
Response: {"model": "arcface-mbf",
           "faces": [{"bbox": [...], "confidence": 0.98,
                      "embedding": [512 floats]}, ...]}

The worker re-runs SCRFD detection internally to obtain the 5-point
landmarks needed for alignment (the database intentionally stores
only bbox + confidence). Faces are returned in detection order, so
callers can match them to stored faces by bbox proximity.

Model file: models/w600k_mbf.onnx next to this script (or
DAHLIA_EMBED_MODEL).
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "face"))
from common import run_loop  # noqa: E402
import worker as face_worker  # workers/face/worker.py  # noqa: E402

MODEL_NAME = "arcface-mbf"
SIZE = 112
# ArcFace reference landmarks for a 112x112 aligned crop.
TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

_session = None


def session():
    global _session
    if _session is None:
        import onnxruntime as ort

        model = os.environ.get("DAHLIA_EMBED_MODEL") or str(
            Path(__file__).resolve().parent / "models" / "w600k_mbf.onnx"
        )
        if not Path(model).exists():
            raise FileNotFoundError(f"embedding model not found: {model}")
        _session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return _session


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale+rotation+translation)
    mapping src points onto dst points. Returns a 2x3 matrix."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = (dst_c.T @ src_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    d_mat = np.diag([1.0, d])
    rot = u @ d_mat @ vt
    var = (src_c**2).sum() / len(src)
    scale = (s * [1.0, d]).sum() / var
    trans = dst_mean - scale * (rot @ src_mean)
    return np.hstack([scale * rot, trans[:, None]]).astype(np.float64)


def _warp_affine(img: Image.Image, m_src_to_dst: np.ndarray) -> Image.Image:
    """Warp with PIL using the inverse map (PIL expects dst->src)."""
    m3 = np.vstack([m_src_to_dst, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m3)[:2]
    a, b, c = inv[0]
    d, e, f = inv[1]
    return img.transform(
        (SIZE, SIZE), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR
    )


def embed(image_path: str) -> list[dict]:
    img = Image.open(image_path).convert("RGB")
    faces = face_worker.detect(image_path)
    sess = session()
    input_name = sess.get_inputs()[0].name
    out = []
    for face in faces:
        landmarks = np.array(face["landmarks"], dtype=np.float64)
        mat = _similarity_transform(landmarks, TEMPLATE)
        crop = _warp_affine(img, mat)
        arr = np.asarray(crop).astype(np.float32)
        blob = np.transpose((arr - 127.5) / 127.5, (2, 0, 1))[None, ...]
        vec = sess.run(None, {input_name: blob})[0].reshape(-1).astype(np.float64)
        vec /= np.linalg.norm(vec) + 1e-12
        out.append(
            {
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "embedding": [float(v) for v in vec],
            }
        )
    return out


def handle(req: dict) -> dict:
    return {"model": MODEL_NAME, "faces": embed(req["image"])}


if __name__ == "__main__":
    run_loop(handle)
