#!/usr/bin/env python3
"""Face mesh worker: MediaPipe FaceLandmarker (Tasks API) over a face
crop, for Dahlia's built-in 3D face preview.

Request : {"image": "<face crop jpg/png>"}
Response: {"found": true,
           "points": [[x, y, z], ...],        # 478 landmarks, centred
           "triangles": [[a, b, c], ...],     # Delaunay, long edges cut
           "blendshapes": {"jawOpen": 0.12, ...}}
          {"found": false} when no face is detected.

Model: models/face_landmarker.task next to this script
($DAHLIA_FACELANDMARKER_MODEL overrides). 3.6 MB, Apache-2.0
(Google MediaPipe). Output coordinates are recentred on the landmark
centroid and scaled so the face is roughly 2 units wide; z is flipped
so the nose points toward the viewer (verified on the samples).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run_loop  # noqa: E402

_landmarker = None


def _model_path() -> Path:
    env = os.environ.get("DAHLIA_FACELANDMARKER_MODEL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = _model_path()
        if not model.exists():
            raise RuntimeError(f"face landmarker model not found: {model}")
        base = mp_python.BaseOptions(model_asset_path=str(model))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def triangulate(pts2d: np.ndarray) -> list[list[int]]:
    """Delaunay over the 2D landmarks; drop triangles with any long
    edge (they would stretch across the silhouette/background)."""
    from scipy.spatial import Delaunay

    tris = Delaunay(pts2d).simplices
    out = []
    for a, b, c in tris:
        edges = [
            np.linalg.norm(pts2d[a] - pts2d[b]),
            np.linalg.norm(pts2d[b] - pts2d[c]),
            np.linalg.norm(pts2d[c] - pts2d[a]),
        ]
        if max(edges) < 0.16:
            out.append([int(a), int(b), int(c)])
    return out


def handle(req: dict) -> dict:
    try:
        import mediapipe as mp
        import scipy  # noqa: F401
    except ImportError:
        # The lean portable package ships without the optional
        # facemesh component; fail with an explanation, not a traceback.
        raise RuntimeError(
            "3D顔プレビュー用コンポーネント（mediapipe）が未導入のため、"
            "この機能は現在のパッケージでは使えません"
        )

    image_path = req["image"]
    landmarker = _get_landmarker()
    # Load via PIL and force RGB: grayscale JPEGs (e.g. historical
    # B/W portraits, and Dahlia's own crops of them) would otherwise
    # reach the detector as 1-channel tensors and fail inference.
    from PIL import Image as PILImage

    arr = np.ascontiguousarray(
        np.asarray(PILImage.open(image_path).convert("RGB"), dtype=np.uint8)
    )
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return {"found": False}
    lms = result.face_landmarks[0]
    raw = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)

    # Centre + scale into viewer space; flip Y (image coords are
    # top-left origin) and Z (nose toward viewer).
    centre = raw[:, :2].mean(axis=0)
    width = max(raw[:, 0].max() - raw[:, 0].min(), 1e-6)
    scale = 2.0 / width
    pts = np.empty_like(raw)
    pts[:, 0] = (raw[:, 0] - centre[0]) * scale
    pts[:, 1] = -(raw[:, 1] - centre[1]) * scale
    pts[:, 2] = -raw[:, 2] * scale

    blendshapes: dict[str, float] = {}
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            blendshapes[cat.category_name] = round(float(cat.score), 4)

    return {
        "found": True,
        "points": [[round(float(v), 4) for v in p] for p in pts],
        "uvs": [[round(float(p[0]), 4), round(1.0 - float(p[1]), 4)] for p in raw],
        "triangles": triangulate(raw[:, :2]),
        "blendshapes": blendshapes,
    }


if __name__ == "__main__":
    run_loop(handle)
