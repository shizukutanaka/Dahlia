#!/usr/bin/env python3
"""Clustering worker: DBSCAN over face embeddings (cosine distance).

Requests:
  {"items": [...], "eps": 0.62, "min_samples": 2}
      One-shot clustering (original form).
  {"op": "add_items", "items": [...]}
      Accumulate items in the session; answered with {"total": n}.
      Lets the caller stream a large library in modest chunks
      instead of one giant JSON line (W22). Items are converted to
      a float64 array on arrival so the session does not hold the
      far heavier JSON/list representation.
  {"op": "cluster", "eps": 0.62, "min_samples": 2}
      Cluster everything accumulated so far (and clear it).
  {"op": "clear"}
      Drop accumulated items without clustering.

Response: {"clusters": [[face_id, ...], ...], "noise": [face_id, ...]}
(add_items/clear answer with {"total": n} instead.)

Implemented directly on numpy (no scikit-learn dependency).

Scale (W21): the neighbor search never materializes the full
n x n distance matrix. Rows are processed in blocks sized to a
fixed element budget, so peak extra memory stays around
BLOCK_ELEMS float64 values no matter how large n grows (the old
dense matrix needed ~800 MB at 10k faces and ~20 GB at 50k).
The per-pair arithmetic is the same cosine distance as the dense
version, and the diagonal is pinned to exactly 0.0 just like
fill_diagonal did, so the resulting labels are unchanged; the
blocked path is tested for equality against a dense reference
implementation in tests/test_workers.py.
"""
import sys
from pathlib import Path

import numpy as np

DEFAULT_EPS = 0.62

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import report_progress, run_loop  # noqa: E402

UNVISITED, NOISE = -2, -1

# Block budget for the neighbor search: 2^22 float64 = 32 MB.
BLOCK_ELEMS = 1 << 22


def neighbor_lists(unit: np.ndarray, eps: float) -> list:
    """For each row, the indices within cosine distance eps.

    Computed block by block; only one (rows x n) block of distances
    exists at a time. Includes the point itself (distance 0), as
    the dense implementation did.
    """
    n = unit.shape[0]
    if n == 0:
        return []
    rows = max(1, min(n, BLOCK_ELEMS // n))
    out = []
    for start in range(0, n, rows):
        dist = 1.0 - unit[start:start + rows] @ unit.T
        idx = np.arange(dist.shape[0])
        dist[idx, start + idx] = 0.0  # the old fill_diagonal(0.0)
        for i in range(dist.shape[0]):
            out.append(np.flatnonzero(dist[i] <= eps))
        report_progress(min(start + rows, n), n)
    return out


def dbscan(neighbors: list, min_samples: int) -> np.ndarray:
    n = len(neighbors)
    labels = np.full(n, UNVISITED, dtype=int)
    cluster = 0
    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        if len(neighbors[i]) < min_samples:
            labels[i] = NOISE
            continue
        labels[i] = cluster
        queue = list(neighbors[i])
        head = 0
        while head < len(queue):
            j = queue[head]
            head += 1
            if labels[j] == NOISE:
                labels[j] = cluster
            if labels[j] != UNVISITED:
                continue
            labels[j] = cluster
            if len(neighbors[j]) >= min_samples:
                queue.extend(neighbors[j])
        cluster += 1
    return labels


def cluster_vectors(face_ids: list, vectors: np.ndarray, eps: float, min_samples: int) -> dict:
    if not face_ids:
        return {"clusters": [], "noise": []}
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    labels = dbscan(neighbor_lists(unit, eps), min_samples)
    clusters: dict[int, list[int]] = {}
    noise: list[int] = []
    for face_id, lab in zip(face_ids, labels):
        if lab == NOISE:
            noise.append(face_id)
        else:
            clusters.setdefault(int(lab), []).append(face_id)
    return {
        "clusters": [clusters[k] for k in sorted(clusters)],
        "noise": noise,
    }


def cluster_items(items: list, eps: float, min_samples: int) -> dict:
    if not items:
        return {"clusters": [], "noise": []}
    vectors = np.array([it["vector"] for it in items], dtype=np.float64)
    return cluster_vectors([it["face_id"] for it in items], vectors, eps, min_samples)


# Items accumulated via add_items within this worker session, kept
# as float64 arrays (one per chunk) plus their face ids.
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
    eps = float(req.get("eps", DEFAULT_EPS))
    min_samples = int(req.get("min_samples", 2))
    if op == "cluster":
        # Take the accumulated items; the buffer is cleared either
        # way so a failed run cannot leak stale items into the next.
        ids, vectors = _take_pending()
        if vectors is None:
            return {"clusters": [], "noise": []}
        return cluster_vectors(ids, vectors, eps, min_samples)
    return cluster_items(req.get("items", []), eps, min_samples)


if __name__ == "__main__":
    run_loop(handle)
