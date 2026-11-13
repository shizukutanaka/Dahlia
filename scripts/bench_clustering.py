#!/usr/bin/env python3
"""Clustering scale benchmark (W21): peak memory + wall time of the
clustering worker's handle() over synthetic face embeddings.

Dataset: groups of 25 near-duplicate faces per synthetic person
(intra-person cosine distance ~0.44, inter-person ~1.0, eps = 0.68)
plus singleton noise faces, 512-dim unit vectors like ArcFace output.

Usage:
  .venv/bin/python scripts/bench_clustering.py [--sizes 10000,50000]
      [--worker PATH]   worker.py to measure (default: the repo's)

Each size runs in a subprocess so peak RSS is measured cleanly.
"""
import argparse
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUP = 25
DIM = 512


def make_items(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    people = max(1, (n * 4 // 5) // GROUP)
    noise_n = n - people * GROUP
    items = []
    fid = 0
    for _ in range(people):
        base = rng.normal(size=DIM)
        base /= np.linalg.norm(base)
        for _ in range(GROUP):
            v = base + 0.039 * rng.normal(size=DIM)
            v /= np.linalg.norm(v)
            fid += 1
            items.append({"face_id": fid, "vector": v.tolist()})
    for _ in range(noise_n):
        v = rng.normal(size=DIM)
        v /= np.linalg.norm(v)
        fid += 1
        items.append({"face_id": fid, "vector": v.tolist()})
    return items


def stream_items(mod, n: int, rng):
    """Generate and feed items person by person, retaining nothing
    list-shaped: exactly what the worker sees from the app (W22)."""
    people = max(1, (n * 4 // 5) // GROUP)
    fid = 0
    for _ in range(people):
        base = rng.normal(size=DIM)
        base /= np.linalg.norm(base)
        chunk = []
        for _ in range(GROUP):
            v = base + 0.039 * rng.normal(size=DIM)
            v /= np.linalg.norm(v)
            fid += 1
            chunk.append({"face_id": fid, "vector": v.tolist()})
        mod.handle({"op": "add_items", "items": chunk})
    while fid < n:
        v = rng.normal(size=DIM)
        v /= np.linalg.norm(v)
        fid += 1
        mod.handle({"op": "add_items", "items": [{"face_id": fid, "vector": v.tolist()}]})


def run_one(worker_path: str, n: int, chunked: bool, stream: bool = False) -> dict:
    spec = importlib.util.spec_from_file_location("bench_worker", worker_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t0 = time.perf_counter()
    if stream:
        stream_items(mod, n, np.random.default_rng(42))
        r = mod.handle({"op": "cluster", "eps": 0.68, "min_samples": 2})
    elif chunked:
        # Stream prebuilt items in chunks, then cluster.
        items = make_items(n)
        for s in range(0, len(items), 500):
            mod.handle({"op": "add_items", "items": items[s:s + 500]})
        r = mod.handle({"op": "cluster", "eps": 0.68, "min_samples": 2})
    else:
        items = make_items(n)
        r = mod.handle({"items": items, "eps": 0.68, "min_samples": 2})
    dt = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "n": n,
        "seconds": round(dt, 2),
        "peak_rss_mb": round(peak_mb),
        "clusters": len(r["clusters"]),
        "clustered": sum(len(c) for c in r["clusters"]),
        "noise": len(r["noise"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000")
    ap.add_argument("--worker", default=str(ROOT / "workers" / "clustering" / "worker.py"))
    ap.add_argument("--chunked", action="store_true",
                    help="feed items via add_items + cluster (the app path)")
    ap.add_argument("--stream", action="store_true",
                    help="generate items on the fly while feeding (no list retained)")
    ap.add_argument("--run-one", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_one:
        print(json.dumps(run_one(args.worker, args.run_one, args.chunked, args.stream)))
        return

    for n in [int(s) for s in args.sizes.split(",") if s]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", args.worker,
               "--run-one", str(n)]
        if args.chunked:
            cmd.append("--chunked")
        if args.stream:
            cmd.append("--stream")
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"n={n}: FAILED\n{out.stderr[-2000:]}")
            continue
        r = json.loads(out.stdout.strip().splitlines()[-1])
        print(
            f"n={r['n']:>6}: {r['seconds']:>7.2f}s  peak RSS {r['peak_rss_mb']:>6} MB  "
            f"clusters={r['clusters']} clustered={r['clustered']} noise={r['noise']}"
        )


if __name__ == "__main__":
    main()
