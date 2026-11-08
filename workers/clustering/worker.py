#!/usr/bin/env python3
"""Clustering worker: DBSCAN over face embeddings (cosine distance).

Request : {"items": [{"face_id": 1, "vector": [...]}, ...],
           "eps": 0.68, "min_samples": 2}
Response: {"clusters": [[face_id, ...], ...], "noise": [face_id, ...]}

Implemented directly on numpy (no scikit-learn dependency); photo
libraries are small enough for an O(n^2) distance matrix.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run_loop  # noqa: E402

UNVISITED, NOISE = -2, -1


def dbscan(dist: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = dist.shape[0]
    labels = np.full(n, UNVISITED, dtype=int)
    neighbors = [np.where(dist[i] <= eps)[0] for i in range(n)]
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


def handle(req: dict) -> dict:
    items = req.get("items", [])
    eps = float(req.get("eps", 0.68))  # tuned on tests/samples: same-person distance 0.65, different >= 0.85
    min_samples = int(req.get("min_samples", 2))
    if not items:
        return {"clusters": [], "noise": []}
    vectors = np.array([it["vector"] for it in items], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    labels = dbscan(dist, eps, min_samples)
    clusters: dict[int, list[int]] = {}
    noise: list[int] = []
    for item, lab in zip(items, labels):
        if lab == NOISE:
            noise.append(item["face_id"])
        else:
            clusters.setdefault(int(lab), []).append(item["face_id"])
    return {
        "clusters": [clusters[k] for k in sorted(clusters)],
        "noise": noise,
    }


if __name__ == "__main__":
    run_loop(handle)
