#!/usr/bin/env python3
"""Face mesh worker: MediaPipe FaceLandmarker (Tasks API) over a face
crop, for Dahlia's built-in 3D face preview.

Request : {"image": "<face crop jpg/png>"}
Response: {"found": true,
           "points": [[x, y, z], ...],        # 478 landmarks, centred
           "triangles": [[a, b, c], ...],     # canonical face tessellation
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
from common import load_image_rgb, run_loop  # noqa: E402

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


_triangles = None


def triangulate(pts: np.ndarray) -> list[list[int]]:
    """Canonical MediaPipe face-mesh triangulation (tessellation.json,
    derived from FACEMESH_TESSELATION, Apache-2.0). Designed topology
    beats a geometric Delaunay here: no facets stretch across the
    mouth/eye openings, and there is no scipy dependency. Indices
    cover the 468-point mesh; iris points (468+) stay untriangulated.
    The tessellation edges are undirected, so each triangle is wound
    to face away from the landmark centroid -- without this the mesh
    shades like crumpled foil."""
    global _triangles
    if _triangles is None:
        import json

        path = Path(__file__).resolve().parent / "tessellation.json"
        _triangles = json.loads(path.read_text(encoding="utf-8"))["triangles"]
    tris = [list(t) for t in _triangles if max(t) < len(pts)]
    _orient_consistently(tris, pts)
    return [[int(a), int(b), int(c)] for a, b, c in tris]


def _orient_consistently(tris: list, pts: np.ndarray) -> None:
    """Make all triangle windings consistent in place. Adjacent
    triangles must traverse their shared edge in opposite directions;
    propagate that constraint by BFS from a seed, then flip the whole
    mesh if the majority of normals end up pointing at the centroid.
    A per-triangle centroid test alone miswinds triangles in concave
    regions (eye sockets), which shade as dark folded bands."""
    from collections import defaultdict, deque

    edge_map = defaultdict(list)
    for i, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map[(min(u, v), max(u, v))].append(i)

    def directed(tri, key):
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            if (min(u, v), max(u, v)) == key:
                return (u, v)
        return None

    centre = pts.mean(axis=0)

    def seed_outward(i):
        a, b, c = tris[i]
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        tc = (pts[a] + pts[b] + pts[c]) / 3.0
        if float(np.dot(n, tc - centre)) < 0:
            tris[i] = [a, c, b]

    seen = [False] * len(tris)
    for start in range(len(tris)):
        if seen[start]:
            continue
        seed_outward(start)
        seen[start] = True
        component = [start]
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for key, others in edge_map.items():
                if i not in others:
                    continue
                for j in others:
                    if j == i or seen[j]:
                        continue
                    if directed(tris[i], key) == directed(tris[j], key):
                        tris[j] = [tris[j][0], tris[j][2], tris[j][1]]
                    seen[j] = True
                    component.append(j)
                    queue.append(j)
        # A component can be consistently wound yet globally flipped
        # (its seed sits in a concavity -- the eye/brow patch did
        # exactly this and shaded as a dark band). Decide per
        # component by majority vote of its own triangles.
        score = 0.0
        for i in component:
            a, b, c = tris[i]
            n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            tc = (pts[a] + pts[b] + pts[c]) / 3.0
            score += float(np.dot(n, tc - centre))
        if score < 0:
            for i in component:
                a, b, c = tris[i]
                tris[i] = [a, c, b]


def handle(req: dict) -> dict:
    try:
        import mediapipe as mp
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
    # EXIF orientation is applied so the preview matches what the
    # other workers saw.
    arr = np.ascontiguousarray(
        np.asarray(load_image_rgb(image_path), dtype=np.uint8)
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
        "triangles": triangulate(pts),
        "blendshapes": blendshapes,
    }


if __name__ == "__main__":
    run_loop(handle)
