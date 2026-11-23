#!/usr/bin/env python3
"""Measure real SFace separation on a reproducible, locally cached LFW subset."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

from evaluate_accuracy import ROOT, SAMPLES, _load, cluster, pairwise_metrics

REVISION = "274dfbca8b2dc9c6d21ae80f8627c911222eb347"
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def download_subset(cache: Path) -> dict[str, list[Path]]:
    by_person = defaultdict(list)
    for offset in range(0, 500, 100):
        params = urlencode(dict(dataset="bitmind/lfw", config="default", split="train",
                               offset=offset, length=100))
        with urlopen(f"{ROWS_URL}?{params}", timeout=60) as response:
            for entry in json.load(response)["rows"]:
                row = entry["row"]
                filename = row["filename"]
                url = row["image"]["src"]
                if f"/{REVISION}/" not in url:
                    raise ValueError("LFW mirror revision changed")
                by_person[filename.rsplit("_", 1)[0]].append((filename, url))

    def fetch(item):
        name, url = item
        path = cache / name
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                data = response.read()
            with Image.open(BytesIO(data)) as img:
                img.verify()
            path.write_bytes(data)
        return path

    people = sorted(person for person, photos in by_person.items() if len(photos) >= 8)[:10]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return {
            person: list(pool.map(fetch, sorted(by_person[person])[:10]))
            for person in people
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="local directory for LFW photos (not for distribution)")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    people = download_subset(args.cache)
    embedding = _load("embedding_worker", ROOT / "workers" / "embedding" / "worker.py")
    items, truth, rejected = [], {}, []
    for label, (person, paths) in enumerate(people.items()):
        print(f"{person}: {', '.join(p.name for p in paths)}", flush=True)
        for path in paths:
            faces = embedding.embed(str(path))
            if len(faces) != 1:
                rejected.append((path.name, len(faces)))
                continue
            fid = len(items)
            items.append({"face_id": fid, "vector": faces[0]["embedding"]})
            truth[fid] = label

    print(f"LFW: {len(items)} usable images from {len(set(truth.values()))} people, "
          f"rejected (file, face count): {rejected}")
    vec = np.array([item["vector"] for item in items])
    labels = np.array(list(truth.values()))
    i, j = np.triu_indices(len(vec), k=1)
    dist = 1.0 - (vec @ vec.T)[i, j]
    same = labels[i] == labels[j]
    for name, distances in [("same", dist[same]), ("different", dist[~same])]:
        percentiles = np.percentile(distances, [0, 1, 5, 50, 95, 99, 100])
        print(f"{name:9s} n={len(distances)} min/p01/p05/p50/p95/p99/max "
              + " ".join(f"{x:.4f}" for x in percentiles))

    print("eps   same edges   cross edges   clusters   mixed   noise   precision   recall   F1")
    for eps in np.arange(45, 71) / 100:
        result = cluster(items, eps)
        groups = result["clusters"]
        mixed = sum(len({truth[fid] for fid in group}) > 1 for group in groups)
        m = pairwise_metrics(truth, groups)
        print(f"{eps:.2f}  {np.sum(same & (dist <= eps)):10d}   "
              f"{np.sum(~same & (dist <= eps)):11d}   {len(groups):8d}   "
              f"{mixed:5d}   {len(result['noise']):5d}   {m['precision']:9.4f}   "
              f"{m['recall']:.4f}   {m['f1']:.4f}")

    names = ("einstein_a.jpg", "einstein_b.jpg", "curie_a.jpg")
    fixtures = [embedding.embed(str(SAMPLES / name)) for name in names]
    if any(len(faces) != 1 for faces in fixtures):
        raise ValueError("expected one face per fixture portrait")
    a, b, c = [np.array(faces[0]["embedding"]) for faces in fixtures]
    print(f"Fixtures: same {1 - a @ b:.4f}, different {1 - a @ c:.4f} / {1 - b @ c:.4f}")


if __name__ == "__main__":
    main()
