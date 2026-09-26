"""Training set: same Kmax blocking + prune on train S1, negatives from real blocking output.

Plan §5.5: sample 150-300K random train S1 (fixed seed), retrieve/prune against the
full train S2/S3 index, build ~39 features, label from ground truth. Negatives come
from blocking look-alikes, never random. Caches are written as float32 .npy + a
Parquet key table.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking import retrieve as rt  # noqa: E402
from src.blocking.evaluate_blocking import load_gt, read_gt_ids  # noqa: E402
from src.matching import pair_features as pf  # noqa: E402

TRAIN_SET_DIR = config.CACHE_DIR / "train_set"


def load_records(path: Path, wanted: set, limit: int = 0):
    """id -> (name, address) for rows whose id is in ``wanted``."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            eid = row[0]
            if eid in wanted:
                out[eid] = (row[1], row[2])
                if limit and len(out) >= limit:
                    break
    return out


def build(n_sample=150_000, kmax=50, seed=42, n_jobs=8, force=False):
    X_path = TRAIN_SET_DIR / "X.npy"
    if X_path.exists() and not force:
        print(f"[train_set] {X_path} exists — skipping")
        return TRAIN_SET_DIR
    TRAIN_SET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gt_ids = read_gt_ids(config.TRAIN_DIR / "train_ground_truth.tsv")
    rng = random.Random(seed)
    sample = gt_ids if n_sample >= len(gt_ids) else rng.sample(gt_ids, n_sample)
    del gt_ids
    sample_set = set(sample)
    gt = load_gt(config.TRAIN_DIR / "train_ground_truth.tsv", sample_set)
    print(f"[train_set] sampled {len(sample):,} S1 ({time.time()-t0:.0f}s)", flush=True)

    raw = rt.retrieve("train", kmax=kmax, subset=sample_set, n_jobs=n_jobs,
                      prune=True, with_records=True)
    total_pairs = sum(len(v) for v in raw.values())
    cand_ids = {c for row in raw.values() for _s, c, _n, _a in row}
    s1_recs = load_records(config.TRAIN_DIR / "train_source1.tsv", sample_set)
    print(f"[train_set] {total_pairs:,} pruned pairs, {len(cand_ids):,} distinct "
          f"candidates, {len(s1_recs):,} S1 records ({time.time()-t0:.0f}s)", flush=True)

    Xs, ys, keys_s1, keys_c = [], [], [], []
    CHUNK = 10_000
    for start in range(0, len(sample), CHUNK):
        jobs = []
        for sid in sample[start:start + CHUNK]:
            row = raw.get(sid, [])
            if not row:
                continue
            best = row[0][0]
            n1, a1 = s1_recs[sid]
            truth = set(gt[sid])
            for j, (score, cid, cname, caddr) in enumerate(row):
                ctx = (score, j + 1, (score / best if best > 0 else 0.0),
                       best - score, score - (row[j + 1][0] if j + 1 < len(row) else 0.0), cid)
                jobs.append((n1, a1, cname, caddr, ctx))
                ys.append(1 if cid in truth else 0)
                keys_s1.append(sid)
                keys_c.append(cid)
        Xs.append(pf.build_matrix(jobs, n_jobs=n_jobs))
        if (start // CHUNK) % 5 == 0:
            print(f"[train_set] {start+CHUNK:,}/{len(sample):,} S1 "
                  f"({time.time()-t0:.0f}s)", flush=True)

    X = np.vstack(Xs)
    y = np.asarray(ys, dtype=np.int8)
    np.save(X_path, X)
    np.save(TRAIN_SET_DIR / "y.npy", y)
    pq.write_table(pa.table({"source1_entity_id": keys_s1, "candidate_id": keys_c}),
                   TRAIN_SET_DIR / "keys.parquet")
    meta = {
        "n_sample": len(sample), "kmax": kmax, "seed": seed,
        "positives": int(y.sum()), "pairs": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "feature_names": pf.FEATURE_NAMES,
        "s1_with_candidates": sum(1 for s in sample if raw.get(s)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (TRAIN_SET_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (TRAIN_SET_DIR / "sample_ids.json").write_text(json.dumps(sample))
    print(f"[train_set] X={X.shape} positives={int(y.sum()):,} "
          f"({time.time()-t0:.0f}s)")
    return TRAIN_SET_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=150_000)
    ap.add_argument("--kmax", type=int, default=config.K_MAX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    build(a.sample, a.kmax, a.seed, a.n_jobs, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
