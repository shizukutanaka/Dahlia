"""Regression tests for the accuracy evaluation harness
(scripts/evaluate_accuracy.py).

The full evaluation is a script; these tests pin its key claims so
a worker change, a threshold change or (in the model-backed cases)
a model swap that degrades measured accuracy fails loudly here:

  - the metric functions themselves (known-answer cases),
  - the production eps sits on the synthetic quality plateau,
  - detection reproduces the labeled sample counts exactly, and
    the real portraits keep their same/different distance margins
    around the production eps (model tests skip when the model
    files are absent).

Run: .venv/bin/python -m unittest tests.test_evaluation
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("evaluate_accuracy", ROOT / "scripts" / "evaluate_accuracy.py")


class MetricKnownAnswerTest(unittest.TestCase):
    def test_pairwise_perfect(self):
        truth = {1: 0, 2: 0, 3: 1, 4: 1, 5: -1}
        m = ev.pairwise_metrics(truth, [[1, 2], [3, 4]])
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0))

    def test_pairwise_merged_people_hurt_precision(self):
        truth = {1: 0, 2: 0, 3: 1, 4: 1}
        m = ev.pairwise_metrics(truth, [[1, 2, 3, 4]])
        # 6 predicted pairs, only 2 truly same.
        self.assertAlmostEqual(m["precision"], 2 / 6)
        self.assertEqual(m["recall"], 1.0)

    def test_pairwise_split_person_hurts_recall(self):
        truth = {1: 0, 2: 0, 3: 0, 4: 0}
        m = ev.pairwise_metrics(truth, [[1, 2], [3, 4]])
        self.assertEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 2 / 6)

    def test_noise_face_in_a_cluster_is_a_false_pair(self):
        truth = {1: 0, 2: 0, 3: -1}
        m = ev.pairwise_metrics(truth, [[1, 2, 3]])
        self.assertAlmostEqual(m["precision"], 1 / 3)

    def test_purity(self):
        truth = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        self.assertEqual(ev.purity(truth, [[1, 2], [3, 4, 5]]), 1.0)
        # One mixed cluster of 4 with a 3/1 split.
        self.assertAlmostEqual(ev.purity({1: 0, 2: 0, 3: 0, 4: 1}, [[1, 2, 3, 4]]), 0.75)


class SyntheticQualityTest(unittest.TestCase):
    """The production eps must deliver near-perfect clustering on
    calibrated synthetic data and sit on the sweep plateau (not at
    a cliff edge) - the regression form of the eps decision in
    docs/EVALUATION.md."""

    def check_level(self, target):
        items, truth = ev.synthetic_dataset(42, 40, 6, 24, target)
        r = ev.cluster(items, ev.PROD_EPS)
        m = ev.pairwise_metrics(truth, r["clusters"])
        self.assertGreaterEqual(m["f1"], 0.98, f"within~{target}: {m}")
        self.assertGreaterEqual(ev.purity(truth, r["clusters"]), 0.98)
        sweep = {}
        for eps in ev.SWEEP_EPS:
            rr = ev.cluster(items, eps)
            sweep[eps] = ev.pairwise_metrics(truth, rr["clusters"])["f1"]
        self.assertGreaterEqual(m["f1"], max(sweep.values()) - 0.01,
                                f"eps {ev.PROD_EPS} not on plateau: {sweep}")

    def test_typical_level(self):
        self.check_level(ev.LEVELS["typical"])

    def test_hard_level(self):
        self.check_level(ev.LEVELS["hard"])


@unittest.skipUnless(ev.models_present(), "face/embedding models not downloaded")
class RealModelEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face = _load("face_worker", ROOT / "workers" / "face" / "worker.py")
        cls.emb = _load("embedding_worker", ROOT / "workers" / "embedding" / "worker.py")

    def test_detection_counts_on_all_samples(self):
        for row in ev.evaluate_detection(self.face):
            self.assertTrue(row["ok"], row)

    def test_real_pair_margins_and_partition(self):
        real = ev.evaluate_real_embeddings(self.emb)
        self.assertNotIn("error", real)
        self.assertLess(real["same_person_distance"], ev.PROD_EPS)
        self.assertGreater(min(real["different_person_distances"]), ev.PROD_EPS)
        self.assertTrue(real["partition_at_production_eps"]["ok"])


if __name__ == "__main__":
    unittest.main()
