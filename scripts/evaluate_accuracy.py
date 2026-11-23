#!/usr/bin/env python3
"""Accuracy evaluation harness: face detection and identity
clustering, measured instead of assumed.

Three parts:

  A. Detection on the labeled sample photos (tests/samples): the
     expected face count per photo is known, so detection is
     checked exactly.
  B. The real sample portraits through the embedding worker: the
     pairwise cosine distances (same person vs different people),
     the margins they leave around the production eps, and the
     end-to-end clustering partition at production settings. This
     is the part a future model swap must re-run: a better model
     widens the same/different gap.
  C. Synthetic clustering quality: generated embeddings with known
     ground truth, calibrated so the within-person cosine distance
     matches a target difficulty (0.30 easy / 0.45 typical /
     0.60 hard).
     Reports pairwise precision/recall/F1 and purity at the
     production eps, plus an eps sweep showing where the
     production value sits relative to the quality plateau.

Only in-repo samples and synthetic data are used; no external
face dataset is downloaded (the sample photos are public-domain
fixtures, see tests/samples/SOURCES.md).

Usage:
  .venv/bin/python scripts/evaluate_accuracy.py [--json out.json]
                                                [--skip-models]

Exit code is 0 when every gate passes, 1 otherwise, so the script
can gate a model swap or a threshold change. Production settings
use the clustering worker's default eps, which must match CLUSTER_EPS
in pipeline.rs.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT / "workers" / "face"))
sys.path.insert(0, str(ROOT / "workers" / "embedding"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clustering_worker = _load("clustering_worker", ROOT / "workers" / "clustering" / "worker.py")

# Production clustering parameters (defaults of the clustering
# worker's handle(); pipeline.rs CLUSTER_EPS carries the same eps).
PROD_EPS = clustering_worker.DEFAULT_EPS
PROD_MIN_SAMPLES = 2

SAMPLES = ROOT / "tests" / "samples"
# Labeled expectations for part A/B (see tests/samples/SOURCES.md).
EXPECTED_FACES = {
    "einstein_a.jpg": 1,
    "einstein_b.jpg": 1,
    "curie_a.jpg": 1,
    "landscape.jpg": 0,
}
SAME_PERSON = ("einstein_a.jpg", "einstein_b.jpg")
OTHER_PERSON = "curie_a.jpg"

DIM = 512
LEVELS = {"easy": 0.30, "typical": 0.45, "hard": 0.60}
SWEEP_EPS = sorted({0.40, 0.50, 0.55, 0.60, 0.65, 0.68, 0.72, 0.76, 0.80, 0.85, 0.90, PROD_EPS})


# ------------------------------------------------------------------
# Metrics

def pairwise_metrics(truth: dict, clusters: list) -> dict:
    """Pairwise precision/recall/F1 of a clustering against ground
    truth labels. truth maps face_id -> person label; singleton
    (noise) faces carry unique negative labels, so a pair counts as
    truly-same only for two faces of one real person."""
    pred_same = set()
    for c in clusters:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                pred_same.add((c[i], c[j]) if c[i] < c[j] else (c[j], c[i]))
    ids = sorted(truth)
    tp = fp = fn = 0
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            truly = truth[a] == truth[b] and truth[a] >= 0
            pred = (a, b) in pred_same
            if truly and pred:
                tp += 1
            elif pred and not truly:
                fp += 1
            elif truly and not pred:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def purity(truth: dict, clusters: list) -> float:
    """Size-weighted purity of the predicted clusters: for each
    cluster, the share of its largest single true-person group."""
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 1.0
    good = 0
    for c in clusters:
        counts: dict = {}
        for fid in c:
            lab = truth[fid]
            if lab >= 0:
                counts[lab] = counts.get(lab, 0) + 1
        good += max(counts.values()) if counts else 0
    return good / total


# ------------------------------------------------------------------
# Part C data

def synthetic_dataset(seed: int, people: int, per_person: int,
                      noise_n: int, within_target: float):
    """Embeddings with known ground truth. A person's faces are a
    shared unit base vector plus Gaussian noise, scaled so the
    expected within-person cosine distance equals within_target:
    E[distance] ~= s / (1 + s) with s = sigma^2 * dim."""
    rng = np.random.default_rng(seed)
    sigma = (within_target / (1.0 - within_target) / DIM) ** 0.5
    items, truth = [], {}
    fid = 1
    for person in range(people):
        base = rng.normal(size=DIM)
        base /= np.linalg.norm(base)
        for _ in range(per_person):
            v = base + sigma * rng.normal(size=DIM)
            v /= np.linalg.norm(v)
            items.append({"face_id": fid, "vector": v.tolist()})
            truth[fid] = person
            fid += 1
    for i in range(noise_n):
        v = rng.normal(size=DIM)
        v /= np.linalg.norm(v)
        items.append({"face_id": fid, "vector": v.tolist()})
        truth[fid] = -(i + 1)  # unique label: a person of one
        fid += 1
    return items, truth


def cluster(items, eps, min_samples=PROD_MIN_SAMPLES):
    return clustering_worker.handle(
        {"items": items, "eps": eps, "min_samples": min_samples}
    )


# ------------------------------------------------------------------
# Parts A + B (need the real models)

def models_present() -> bool:
    return (
        (ROOT / "workers" / "face" / "models" / "face_detection_yunet_2023mar.onnx").exists()
        and (ROOT / "workers" / "embedding" / "models" / "face_recognition_sface_2021dec.onnx").exists()
    )


def evaluate_detection(face_worker) -> list:
    rows = []
    for name, expected in EXPECTED_FACES.items():
        faces = face_worker.detect(str(SAMPLES / name))
        rows.append({
            "file": name,
            "expected": expected,
            "detected": len(faces),
            "confidences": [round(float(f["confidence"]), 3) for f in faces],
            "ok": len(faces) == expected,
        })
    return rows


def evaluate_real_embeddings(embedding_worker) -> dict:
    vecs = {}
    for name in ("einstein_a.jpg", "einstein_b.jpg", "curie_a.jpg"):
        faces = embedding_worker.embed(str(SAMPLES / name))
        if len(faces) != 1:
            return {"ok": False, "error": f"{name}: expected 1 embedded face, got {len(faces)}"}
        vecs[name] = np.array(faces[0]["embedding"], dtype=np.float64)

    def dist(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(1.0 - a @ b)

    d_same = dist(vecs[SAME_PERSON[0]], vecs[SAME_PERSON[1]])
    d_diff = sorted(
        dist(vecs[p], vecs[OTHER_PERSON]) for p in SAME_PERSON
    )
    items = [
        {"face_id": i + 1, "vector": vecs[n].tolist()}
        for i, n in enumerate(("einstein_a.jpg", "einstein_b.jpg", "curie_a.jpg"))
    ]
    r = cluster(items, PROD_EPS)
    partition_ok = (
        sorted(sorted(c) for c in r["clusters"]) == [[1, 2]] and r["noise"] == [3]
    )
    return {
        "ok": bool(d_same < PROD_EPS < d_diff[0] and partition_ok),
        "same_person_distance": round(d_same, 4),
        "different_person_distances": [round(d, 4) for d in d_diff],
        "eps": PROD_EPS,
        "margin_above_same": round(PROD_EPS - d_same, 4),
        "margin_below_different": round(d_diff[0] - PROD_EPS, 4),
        "partition_at_production_eps": {
            "clusters": r["clusters"], "noise": r["noise"], "ok": partition_ok,
        },
    }


# ------------------------------------------------------------------
# Part C

def evaluate_synthetic(seeds=(42, 7), people=40, per_person=6, noise_n=24) -> dict:
    levels = {}
    for level, target in LEVELS.items():
        sweep = {eps: [] for eps in SWEEP_EPS}
        prod_rows = []
        for seed in seeds:
            items, truth = synthetic_dataset(seed, people, per_person, noise_n, target)
            for eps in SWEEP_EPS:
                r = cluster(items, eps)
                sweep[eps].append(pairwise_metrics(truth, r["clusters"])["f1"])
            r = cluster(items, PROD_EPS)
            m = pairwise_metrics(truth, r["clusters"])
            noise_ids = {fid for fid, lab in truth.items() if lab < 0}
            prod_rows.append({
                **m,
                "purity": purity(truth, r["clusters"]),
                "clusters": len(r["clusters"]),
                "noise_recall": (
                    len(noise_ids & set(r["noise"])) / len(noise_ids) if noise_ids else 1.0
                ),
            })
        mean = lambda rows, k: round(float(np.mean([row[k] for row in rows])), 4)
        sweep_mean = {f"{eps:.2f}": round(float(np.mean(v)), 4) for eps, v in sweep.items()}
        best = max(sweep_mean.values())
        prod_f1 = mean(prod_rows, "f1")
        levels[level] = {
            "within_person_distance_target": target,
            "at_production_eps": {
                "precision": mean(prod_rows, "precision"),
                "recall": mean(prod_rows, "recall"),
                "f1": prod_f1,
                "purity": mean(prod_rows, "purity"),
                "clusters": mean(prod_rows, "clusters"),
                "people": people,
                "noise_recall": mean(prod_rows, "noise_recall"),
            },
            "eps_sweep_f1": sweep_mean,
            "best_sweep_f1": best,
            # The production eps must sit on the quality plateau,
            # not at the edge of a cliff.
            "on_plateau": prod_f1 >= best - 0.01,
        }
    return levels


# ------------------------------------------------------------------

def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write the full results as JSON")
    ap.add_argument("--skip-models", action="store_true",
                    help="run only the synthetic part (no model files needed)")
    args = ap.parse_args()

    report = {
        "head": git_head(),
        "production": {"eps": PROD_EPS, "min_samples": PROD_MIN_SAMPLES},
        "gates": {},
    }
    gates = report["gates"]

    if not args.skip_models:
        if not models_present():
            print("models not present: run with --skip-models, or place "
                  "face_detection_yunet_2023mar.onnx / face_recognition_sface_2021dec.onnx in the workers' models/")
            return 1
        face_worker = _load("face_worker", ROOT / "workers" / "face" / "worker.py")
        embedding_worker = _load("embedding_worker", ROOT / "workers" / "embedding" / "worker.py")

        det = evaluate_detection(face_worker)
        report["detection"] = det
        gates["detection_exact"] = all(r["ok"] for r in det)
        print("== A. Detection on samples ==")
        for r in det:
            print(f"  {r['file']:<16} expected {r['expected']}  detected {r['detected']}"
                  f"  conf {r['confidences']}  {'OK' if r['ok'] else 'MISMATCH'}")

        real = evaluate_real_embeddings(embedding_worker)
        report["real_embeddings"] = real
        gates["real_pair_margins"] = bool(
            real.get("same_person_distance", 9) < PROD_EPS
            < min(real.get("different_person_distances", [0]))
        )
        gates["real_partition"] = bool(
            real.get("partition_at_production_eps", {}).get("ok")
        )
        print("== B. Real sample embeddings ==")
        if "error" in real:
            print(f"  ERROR: {real['error']}")
        else:
            print(f"  same-person distance       {real['same_person_distance']}")
            print(f"  different-person distances {real['different_person_distances']}")
            print(f"  eps {PROD_EPS}: margin above same +{real['margin_above_same']},"
                  f" below different +{real['margin_below_different']}")
            print(f"  partition at production eps: {real['partition_at_production_eps']}")

    levels = evaluate_synthetic()
    report["synthetic"] = levels
    gates["synthetic_f1"] = all(
        lv["at_production_eps"]["f1"] >= 0.98 for lv in levels.values()
    )
    gates["synthetic_on_plateau"] = all(lv["on_plateau"] for lv in levels.values())
    print("== C. Synthetic clustering quality (pairwise, at production eps) ==")
    for level, lv in levels.items():
        p = lv["at_production_eps"]
        print(f"  {level:<8} within~{lv['within_person_distance_target']:.2f}  "
              f"P {p['precision']:.3f}  R {p['recall']:.3f}  F1 {p['f1']:.3f}  "
              f"purity {p['purity']:.3f}  clusters {p['clusters']:.0f}/{p['people']}  "
              f"noise recall {p['noise_recall']:.3f}  plateau {'yes' if lv['on_plateau'] else 'NO'}")
    print("  eps sweep (mean F1):")
    for level, lv in levels.items():
        cells = "  ".join(f"{e}:{f:.2f}" for e, f in lv["eps_sweep_f1"].items())
        print(f"    {level:<8} {cells}")

    print("== Gates ==")
    for name, ok in gates.items():
        print(f"  {name:<24} {'PASS' if ok else 'FAIL'}")
    passed = all(gates.values())

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.json}")
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
