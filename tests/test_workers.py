"""Worker tests (run with the Dahlia venv python).

  .venv/bin/python -m pytest tests/test_workers.py
  # or: .venv/bin/python -m unittest tests.test_workers

Covers the clustering algorithm and the face worker over the sample
photos (tests/samples). Model files are required for the face tests;
they are skipped when absent.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT / "workers" / "clustering"))
sys.path.insert(0, str(ROOT / "workers" / "face"))

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clustering_worker = _load("clustering_worker", ROOT / "workers" / "clustering" / "worker.py")


def unit(v):
    v = np.array(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


class ClusteringTest(unittest.TestCase):
    def test_two_people_and_noise(self):
        items = [
            {"face_id": 1, "vector": unit([1.0, 0.05, 0.0])},
            {"face_id": 2, "vector": unit([0.98, 0.1, 0.0])},
            {"face_id": 3, "vector": unit([0.0, 0.05, 1.0])},
            {"face_id": 4, "vector": unit([0.02, 0.0, 0.99])},
            {"face_id": 5, "vector": unit([0.0, 1.0, 0.0])},  # outlier
        ]
        r = clustering_worker.handle({"items": items, "eps": 0.3, "min_samples": 2})
        clusters = sorted(sorted(c) for c in r["clusters"])
        self.assertEqual(clusters, [[1, 2], [3, 4]])
        self.assertEqual(r["noise"], [5])

    def test_empty(self):
        r = clustering_worker.handle({"items": []})
        self.assertEqual(r, {"clusters": [], "noise": []})


@unittest.skipUnless(
    (ROOT / "workers" / "face" / "models" / "det_500m.onnx").exists(),
    "face model not downloaded",
)
class FaceWorkerTest(unittest.TestCase):
    def test_detects_portrait_and_rejects_landscape(self):
        face_worker = _load("face_worker", ROOT / "workers" / "face" / "worker.py")

        faces = face_worker.detect(str(ROOT / "tests" / "samples" / "einstein_a.jpg"))
        self.assertEqual(len(faces), 1)
        self.assertGreater(faces[0]["confidence"], 0.5)
        x, y, w, h = faces[0]["bbox"]
        self.assertTrue(w > 50 and h > 50 and x >= 0 and y >= 0)

        faces = face_worker.detect(str(ROOT / "tests" / "samples" / "landscape.jpg"))
        self.assertEqual(faces, [])


if __name__ == "__main__":
    unittest.main()


sys.path.insert(0, str(ROOT / "workers" / "space"))
space_worker = _load("space_worker", ROOT / "workers" / "space" / "worker.py")


class SpaceWorkerTest(unittest.TestCase):
    def test_pca_keeps_clusters_separate(self):
        rng = np.random.default_rng(7)
        items = []
        base_a = np.zeros(16); base_a[0] = 1.0
        base_b = np.zeros(16); base_b[2] = 1.0
        for i in range(6):  # cluster A around e0
            v = base_a + rng.normal(0, 0.03, 16)
            items.append({"face_id": 100 + i, "vector": v.tolist()})
        for i in range(6):  # cluster B around e2
            v = base_b + rng.normal(0, 0.03, 16)
            items.append({"face_id": 200 + i, "vector": v.tolist()})
        r = space_worker.handle({"items": items, "method": "pca"})
        self.assertEqual(r["method"], "pca")
        pos = {p["face_id"]: np.array(p["pos"]) for p in r["points"]}
        self.assertEqual(len(pos), 12)
        ca = np.mean([pos[100 + i] for i in range(6)], axis=0)
        cb = np.mean([pos[200 + i] for i in range(6)], axis=0)
        within = max(np.linalg.norm(pos[100 + i] - ca) for i in range(6))
        between = np.linalg.norm(ca - cb)
        self.assertGreater(between, within * 3)

    def test_umap_falls_back_below_20_points(self):
        items = [{"face_id": i, "vector": unit([1.0, i * 0.01, 0.0])} for i in range(5)]
        r = space_worker.handle({"items": items, "method": "umap"})
        self.assertEqual(r["method"], "pca")
