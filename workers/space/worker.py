#!/usr/bin/env python3
"""Space worker: 3D positions for the built-in Identity Space view,
plus on-demand monocular depth for the 3D relief view.

Requests:
  {"items": [{"face_id": 1, "vector": [...]}, ...], "method": "pca"}
    -> {"points": [{"face_id": 1, "pos": [x, y, z]}, ...],
        "method": "pca" | "umap"}
    PCA by default. UMAP (cosine) when requested AND at least 20
    faces are present - below that UMAP finds no structure and the
    caller silently gets PCA (the response reports what was used).

  {"op": "add_items", "items": [...]}
      Accumulate items in the session; answered with {"total": n}.
      Lets the caller stream a large library in modest chunks
      instead of one giant JSON line (W23). Items are converted to
      a float64 array on arrival so the session does not hold the
      far heavier JSON/list representation.
  {"op": "reduce", "method": "pca"}
      Reduce everything accumulated so far (and clear it);
      answered with the same {"points": ..., "method": ...} as the
      one-shot form.
  {"op": "clear"}
      Drop accumulated items without reducing them.

  {"depth": {"image": "<face crop jpg>", "out": "<png to write>"}}
    -> {"depth_ready": true, "path": "<png>"}
    -> {"depth_ready": false, "reason": "..."} when the Depth
       Anything V2 model is not installed. Model lookup:
       $DAHLIA_DEPTH_MODEL, then <repo>/models/depth-anything-v2-small.onnx,
       then <workers>/models/depth-anything-v2-small.onnx.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_image_rgb, report_progress, run_loop  # noqa: E402

SCALE = 4.2
UMAP_MIN_POINTS = 20


def pca3(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0)
    if len(vectors) < 4:
        out = np.zeros((len(vectors), 3))
        out[:, : min(3, vectors.shape[1])] = centered[:, :3]
        return out
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:3].T
    scale = np.abs(coords).max()
    if scale > 0:
        coords = coords / scale * SCALE
    return coords


def umap3(vectors: np.ndarray):
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=max(2, min(15, len(vectors) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(normed)
    scale = np.abs(coords).max()
    if scale > 0:
        coords = coords / scale * SCALE
    return coords


def reduce3(items, method: str):
    return reduce3_vectors(
        [it["face_id"] for it in items],
        np.array([it["vector"] for it in items], dtype=np.float64),
        method,
    )


def reduce3_vectors(face_ids, vectors: np.ndarray, method: str):
    # The reduction itself is one numerical call (an SVD, or a UMAP
    # fit) with no meaningful sub-steps, so progress is reported as
    # start/end checkpoints: the start line tells the supervisor the
    # worker is alive before a long silent computation begins.
    report_progress(0, 1)
    used = "pca"
    coords = None
    if method == "umap" and len(face_ids) >= UMAP_MIN_POINTS:
        coords = umap3(vectors)
        if coords is not None:
            used = "umap"
    if coords is None:
        coords = pca3(vectors)
    points = [
        {"face_id": fid, "pos": [round(float(c), 4) for c in row]}
        for fid, row in zip(face_ids, coords)
    ]
    report_progress(1, 1)
    return points, used


_session = None


def _depth_model_path():
    env = os.environ.get("DAHLIA_DEPTH_MODEL")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    for cand in (
        here.parent.parent.parent / "models" / "depth-anything-v2-small.onnx",
        here.parent.parent / "models" / "depth-anything-v2-small.onnx",
    ):
        if cand.exists():
            return str(cand)
    return None


def make_depth(image_path: str, out_path: str) -> dict:
    global _session
    model = _depth_model_path()
    if model is None:
        return {"depth_ready": False, "reason": "depth model not installed"}
    import onnxruntime as ort
    from PIL import Image

    if _session is None:
        _session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    img = load_image_rgb(image_path).resize((518, 518))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = np.transpose((arr - mean) / std, (2, 0, 1))[None, ...]
    depth = _session.run(None, {"pixel_values": blob})[0][0].astype(np.float32)
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return {"depth_ready": False, "reason": "flat depth result"}
    near = (depth - lo) / (hi - lo)  # larger raw value = nearer (verified)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((near * 255).astype(np.uint8), mode="L").save(out_path)
    return {"depth_ready": True, "path": out_path}


# Items accumulated via add_items within this worker session, kept
# as float64 arrays (one per chunk) plus their face ids (W23).
_pending_ids: list = []
_pending_vecs: list = []


def _take_pending():
    ids, vecs = _pending_ids[:], _pending_vecs[:]
    _pending_ids.clear()
    _pending_vecs.clear()
    if not vecs:
        return [], None
    vectors = vecs[0] if len(vecs) == 1 else np.vstack(vecs)
    return ids, vectors


def handle(req: dict) -> dict:
    if "depth" in req:
        d = req["depth"]
        return make_depth(d["image"], d["out"])
    op = req.get("op")
    if op == "add_items":
        items = req.get("items", [])
        if items:
            _pending_ids.extend(it["face_id"] for it in items)
            _pending_vecs.append(
                np.array([it["vector"] for it in items], dtype=np.float64)
            )
        return {"total": len(_pending_ids)}
    if op == "clear":
        _pending_ids.clear()
        _pending_vecs.clear()
        return {"total": 0}
    if op == "reduce":
        # Take the accumulated items; the buffer is cleared either
        # way so a failed run cannot leak stale items into the next.
        ids, vectors = _take_pending()
        if vectors is None:
            return {"points": [], "method": "pca"}
        points, used = reduce3_vectors(ids, vectors, req.get("method", "pca"))
        return {"points": points, "method": used}
    items = req.get("items", [])
    if not items:
        return {"points": [], "method": "pca"}
    points, used = reduce3(items, req.get("method", "pca"))
    return {"points": points, "method": used}


if __name__ == "__main__":
    run_loop(handle)
