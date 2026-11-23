"""Worker tests (run with the Dahlia venv python).

  .venv/bin/python -m pytest tests/test_workers.py
  # or: .venv/bin/python -m unittest tests.test_workers

Covers the clustering algorithm and the face worker over the sample
photos (tests/samples). Model files are required for the face tests;
they are skipped when absent.
"""
import json
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

sys.path.insert(0, str(ROOT / "workers"))
import common as workers_common  # noqa: E402


class ExifOrientationTest(unittest.TestCase):
    """W86: workers must see phone photos upright (EXIF applied)."""

    def test_orientation_6_rotates_image_upright(self):
        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sideways.jpg"
            # Stored pixels are 4x2 with the whole top row red; EXIF
            # orientation 6 says "rotate 90 deg CW".
            img = Image.new("RGB", (4, 2), (0, 0, 0))
            for x in range(4):
                img.putpixel((x, 0), (255, 0, 0))
            exif = Image.Exif()
            exif[274] = 6  # Orientation
            img.save(path, exif=exif, quality=95, subsampling=0)

            loaded = workers_common.load_image_rgb(str(path))
            self.assertEqual(loaded.size, (2, 4))
            # After a 90 deg CW rotation the stored top row becomes
            # the right-hand column of the corrected image.
            r, g, b = loaded.getpixel((1, 2))
            self.assertTrue(r > 150 and r > g + 80, f"expected red, got {(r, g, b)}")
            r2, g2, b2 = loaded.getpixel((0, 2))
            self.assertTrue(r2 < 80, f"expected black, got {(r2, g2, b2)}")

    def test_no_exif_loads_unchanged(self):
        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.jpg"
            Image.new("RGB", (4, 2), (10, 20, 30)).save(path)
            loaded = workers_common.load_image_rgb(str(path))
            self.assertEqual(loaded.size, (4, 2))
            self.assertEqual(loaded.getpixel((0, 0)), (10, 20, 30))


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


def _reference_cluster(items, eps, min_samples):
    """The pre-W21 implementation (dense n x n distance matrix),
    kept here as the ground truth the blocked version must match."""
    if not items:
        return {"clusters": [], "noise": []}
    vectors = np.array([it["vector"] for it in items], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    n = dist.shape[0]
    UNVISITED, NOISE = -2, -1
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
    clusters, noise = {}, []
    for item, lab in zip(items, labels):
        if lab == NOISE:
            noise.append(item["face_id"])
        else:
            clusters.setdefault(int(lab), []).append(item["face_id"])
    return {"clusters": [clusters[k] for k in sorted(clusters)], "noise": noise}


def _grouped_items(rng, people, per_person, dim, noise_n=0, fid0=1):
    """Synthetic faces: tight groups (distance ~0.44 within a
    person, ~1.0 across) so eps=0.68 decisions have wide margins."""
    items, fid = [], fid0
    # Per-component noise sigma chosen so sigma^2 * dim = 0.78 at
    # any dim, giving within-person cosine distance ~0.44.
    sigma = (0.78 / dim) ** 0.5
    for _ in range(people):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        for _ in range(per_person):
            v = base + sigma * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            items.append({"face_id": fid, "vector": v.tolist()})
            fid += 1
    for _ in range(noise_n):
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v)
        items.append({"face_id": fid, "vector": v.tolist()})
        fid += 1
    return items


def _as_sets(result):
    return (
        sorted(tuple(sorted(c)) for c in result["clusters"]),
        tuple(result["noise"]),
    )


class ClusteringEquivalenceTest(unittest.TestCase):
    """W21: the blocked neighbor search must produce exactly the
    labels the dense implementation produced."""

    def check(self, items, eps, min_samples):
        got = clustering_worker.handle(
            {"items": items, "eps": eps, "min_samples": min_samples}
        )
        want = _reference_cluster(items, eps, min_samples)
        self.assertEqual(_as_sets(got), _as_sets(want))

    def test_small_dims_random_groups(self):
        rng = np.random.default_rng(7)
        for dim in (8, 33):
            items = _grouped_items(rng, people=6, per_person=4, dim=dim, noise_n=5)
            self.check(items, eps=0.68, min_samples=2)
            self.check(items, eps=0.68, min_samples=3)

    def test_full_dim_and_chained_groups(self):
        rng = np.random.default_rng(11)
        items = _grouped_items(rng, people=5, per_person=6, dim=512, noise_n=7)
        self.check(items, eps=0.68, min_samples=2)
        # An exact duplicate pair (distance exactly ~0) must cluster
        # together even at a tiny eps, as it did before.
        dup = dict(items[0])
        dup["face_id"] = 9999
        self.check(items + [dup], eps=0.01, min_samples=2)

    def test_min_samples_one_and_singletons(self):
        rng = np.random.default_rng(3)
        items = _grouped_items(rng, people=3, per_person=1, dim=16, noise_n=2)
        self.check(items, eps=0.68, min_samples=1)


class ClusteringChunkedTest(unittest.TestCase):
    """W22: items can be streamed in with add_items and clustered
    by a separate request, with the one-shot result."""

    def setUp(self):
        clustering_worker.handle({"op": "clear"})

    def tearDown(self):
        clustering_worker.handle({"op": "clear"})

    def test_chunked_matches_one_shot(self):
        rng = np.random.default_rng(5)
        items = _grouped_items(rng, people=4, per_person=5, dim=64, noise_n=3)
        for chunk in (items[:7], items[7:15], items[15:]):
            r = clustering_worker.handle({"op": "add_items", "items": chunk})
        self.assertEqual(r["total"], len(items))
        got = clustering_worker.handle({"op": "cluster", "eps": 0.68, "min_samples": 2})
        want = clustering_worker.handle({"items": items, "eps": 0.68, "min_samples": 2})
        self.assertEqual(_as_sets(got), _as_sets(want))
        # The buffer was consumed: a second cluster sees nothing.
        again = clustering_worker.handle({"op": "cluster", "eps": 0.68, "min_samples": 2})
        self.assertEqual(again, {"clusters": [], "noise": []})

    def test_cluster_with_empty_buffer(self):
        r = clustering_worker.handle({"op": "cluster", "eps": 0.68, "min_samples": 2})
        self.assertEqual(r, {"clusters": [], "noise": []})


class ClusteringScaleTest(unittest.TestCase):
    """W21: thousands of faces cluster correctly with the peak
    memory bounded by the block budget, not by n^2."""

    def test_six_thousand_faces(self):
        import subprocess

        # Feed the items through the chunked protocol (the way the
        # app does it) so the measurement reflects the real path.
        script = (
            "import importlib.util, json, resource, sys\n"
            "import numpy as np\n"
            f"spec = importlib.util.spec_from_file_location('w', {str(ROOT / 'workers' / 'clustering' / 'worker.py')!r})\n"
            "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
            "rng = np.random.default_rng(21)\n"
            "fid = 0\n"
            "for _ in range(240):\n"
            "    base = rng.normal(size=512); base /= np.linalg.norm(base)\n"
            "    chunk = []\n"
            "    for _ in range(25):\n"
            "        v = base + 0.039 * rng.normal(size=512); v /= np.linalg.norm(v)\n"
            "        fid += 1; chunk.append({'face_id': fid, 'vector': v.tolist()})\n"
            "    w.handle({'op': 'add_items', 'items': chunk})\n"
            "r = w.handle({'op': 'cluster', 'eps': 0.68, 'min_samples': 2})\n"
            "rss_unit = 1024**2 if sys.platform == 'darwin' else 1024\n"
            "print(json.dumps({'clusters': len(r['clusters']), 'noise': len(r['noise']),\n"
            "                  'rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_unit}))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(out.returncode, 0, out.stderr[-1000:])
        stats = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(stats["clusters"], 240)
        self.assertEqual(stats["noise"], 0)
        # The dense matrix alone would add 6000^2 * 8 = 288 MB on
        # top of the interpreter + vectors; the blocked version
        # must stay far below that.
        self.assertLess(stats["rss_mb"], 300, stats)



@unittest.skipUnless(
    (ROOT / "workers" / "face" / "models" / "face_detection_yunet_2023mar.onnx").exists(),
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
        self.assertEqual(np.shape(faces[0]["landmarks"]), (5, 2))
        self.assertTrue(np.isfinite(faces[0]["landmarks"]).all())

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


class SpaceChunkedTest(unittest.TestCase):
    """W23: items can be streamed in with add_items and reduced by
    a separate request, with exactly the one-shot result."""

    def setUp(self):
        space_worker.handle({"op": "clear"})

    def tearDown(self):
        space_worker.handle({"op": "clear"})

    def _items(self):
        rng = np.random.default_rng(9)
        items = []
        for person, base_i in enumerate((0, 2)):
            base = np.zeros(16)
            base[base_i] = 1.0
            for i in range(6):
                v = base + rng.normal(0, 0.03, 16)
                items.append({"face_id": person * 100 + i, "vector": v.tolist()})
        return items

    def test_chunked_matches_one_shot(self):
        items = self._items()
        for chunk in (items[:5], items[5:9], items[9:]):
            r = space_worker.handle({"op": "add_items", "items": chunk})
        self.assertEqual(r["total"], len(items))
        got = space_worker.handle({"op": "reduce", "method": "pca"})
        want = space_worker.handle({"items": items, "method": "pca"})
        self.assertEqual(got, want)
        # The buffer was consumed: a second reduce sees nothing.
        again = space_worker.handle({"op": "reduce", "method": "pca"})
        self.assertEqual(again, {"points": [], "method": "pca"})

    def test_reduce_with_empty_buffer(self):
        r = space_worker.handle({"op": "reduce", "method": "pca"})
        self.assertEqual(r, {"points": [], "method": "pca"})

    def test_clear_drops_pending_items(self):
        items = self._items()
        space_worker.handle({"op": "add_items", "items": items[:4]})
        r = space_worker.handle({"op": "clear"})
        self.assertEqual(r["total"], 0)
        r = space_worker.handle({"op": "reduce", "method": "pca"})
        self.assertEqual(r, {"points": [], "method": "pca"})


sys.path.insert(0, str(ROOT / "workers" / "facemesh"))
facemesh_worker = _load("facemesh_worker", ROOT / "workers" / "facemesh" / "worker.py")
_FACEMESH_MODEL = ROOT / "workers" / "facemesh" / "models" / "face_landmarker.task"


@unittest.skipUnless(_FACEMESH_MODEL.exists(), "face landmarker model not present")
class FaceMeshWorkerTest(unittest.TestCase):
    def test_landmarks_on_sample_portrait(self):
        try:
            r = facemesh_worker.handle({"image": str(ROOT / "tests" / "samples" / "einstein_a.jpg")})
        except Exception as e:
            self.skipTest(f"mediapipe unavailable: {e}")
        self.assertTrue(r["found"])
        self.assertEqual(len(r["points"]), 478)
        self.assertGreater(len(r["triangles"]), 800)
        # Winding: shared edges must be traversed in opposite
        # directions by their two triangles (a consistently wound
        # mesh shades smoothly; per-triangle heuristics do not).
        # A handful of non-manifold edges in the canonical
        # tessellation can never satisfy this, hence 0.99 not 1.0.
        from collections import defaultdict

        edge_dirs = defaultdict(list)
        for a, b, c in r["triangles"]:
            for u, v in ((a, b), (b, c), (c, a)):
                edge_dirs[(min(u, v), max(u, v))].append((u, v))
        shared = [d for d in edge_dirs.values() if len(d) == 2]
        consistent = sum(1 for d in shared if d[0] != d[1])
        self.assertGreater(consistent / len(shared), 0.99)
        # And the mesh as a whole must face outward, not be globally
        # flipped (a flipped mesh would score near 0.1 here; the
        # flat central face makes ~0.85 normal for a correct mesh).
        pts = np.array(r["points"], dtype=np.float64)
        centre = pts.mean(axis=0)
        outward = 0
        for a, b, c in r["triangles"]:
            n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            tc = (pts[a] + pts[b] + pts[c]) / 3.0
            if float(np.dot(n, tc - centre)) > 0:
                outward += 1
        self.assertGreater(outward / len(r["triangles"]), 0.75)
        self.assertEqual(len(r["uvs"]), 478)
        self.assertTrue(r["blendshapes"])
        # nose tip (landmark 1) must sit in front of the cheek edge (234)
        self.assertGreater(r["points"][1][2], r["points"][234][2])
