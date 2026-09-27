"""Training set: same Kmax blocking + prune on train S1, negatives from real blocking output.

Plan §5.5: sample train S1 (fixed seed), retrieve/prune against the
full train S2/S3 index, build ~39 features, label from ground truth. Negatives come
from blocking look-alikes, never random. Caches are written as float32 .npy + a
Parquet key table.

Memory design: chunks are tiny, the retrieval dict is consumed and freed per S1,
features stream into a disk memmap (never a full in-RAM matrix), and keys/labels
accumulate as compact numpy arrays. Peak RSS stays well under 6 GiB even at
500K S1 (~13M pairs).
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking import build_index as bi  # noqa: E402
from src.blocking import retrieve as rt  # noqa: E402
from src.blocking.evaluate_blocking import load_gt, read_gt_ids  # noqa: E402
from src.cleaning import text_cleaner as tc  # noqa: E402
from src.matching import pair_features as pf  # noqa: E402

TRAIN_SET_DIR = config.CACHE_DIR / "train_set"
IDF_PATH = TRAIN_SET_DIR / "name_idf.json"
N_FEATS = len(pf.FEATURE_NAMES)


def _shard_counter(args):
    """(n_docs, Counter of core-name token DF) for one index shard."""
    shard_dir, name = args
    names = np.load(shard_dir / name.replace(".npz", ".names.npy"), allow_pickle=True)
    df = Counter()
    for nm in names:
        toks = set(tc.name_views(str(nm))[1].split())
        if toks:
            df.update(toks)
    return len(names), df


def build_name_idf(n_jobs=4, force=False):
    """Token->IDF over the train S2/S3 candidate corpus (core name view).

    Saved as {"default": max_idf, "weights": {tok: idf}}, keeping only tokens
    with DF >= 2. Tokens missing from the map default to the max IDF (so hapax
    and unknown tokens stay maximally rare); a vocabulary fully unseen at
    training time (France at inference) degrades to uniform weights, i.e. plain
    Jaccard. Cached; ``--force`` rebuilds.
    """
    if IDF_PATH.exists() and not force:
        return IDF_PATH
    t0 = time.time()
    idx_dir, meta = bi.load_index("train")
    shard_dir = idx_dir / "shards"
    jobs = [(shard_dir, s) for c in meta["countries"].values() for s in c["shards"]]
    n_docs, df = 0, Counter()
    if n_jobs > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(n_jobs, len(jobs))) as pool:
            for n, c in pool.imap_unordered(_shard_counter, jobs):
                n_docs += n
                df.update(c)
    else:
        for n, c in map(_shard_counter, jobs):
            n_docs += n
            df.update(c)
    weights = {t: math.log((n_docs + 1) / (d + 1)) + 1.0 for t, d in df.items() if d >= 2}
    default = math.log((n_docs + 1) / 2.0) + 1.0
    IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDF_PATH.write_text(json.dumps({"default": default, "weights": weights}))
    print(f"[train_set] name IDF: {len(weights):,} tokens over {n_docs:,} docs "
          f"-> {IDF_PATH} ({time.time()-t0:.0f}s)", flush=True)
    return IDF_PATH


def load_name_idf(path: Path):
    d = json.loads(Path(path).read_text())
    return d["weights"], float(d["default"])


def load_pseudo(path: Path):
    """pseudo-label TSV (s1_id, candidate_id[, prob]) -> {s1: set(cands)}."""
    truth = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0]:
                truth.setdefault(row[0], set()).add(row[1])
    return truth


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


def build_jobs(sids, raw, s1_recs, gt):
    """Flatten one wave's retrieval rows into (job, label, s1, cand) tuples."""
    jobs, ys, ks1, kc = [], [], [], []
    append_j, append_y, append_1, append_c = jobs.append, ys.append, ks1.append, kc.append
    for sid in sids:
        row = raw.get(sid)
        if not row:
            continue
        best = row[0][0]
        n1, a1 = s1_recs[sid]
        truth = set(gt[sid])
        for j, (score, cid, cname, caddr) in enumerate(row):
            append_j((n1, a1, cname, caddr,
                      (score, j + 1, (score / best if best > 0 else 0.0),
                       best - score,
                       score - (row[j + 1][0] if j + 1 < len(row) else 0.0), cid)))
            append_y(1 if cid in truth else 0)
            append_1(sid)
            append_c(cid)
    return jobs, ys, ks1, kc


def _stream_records(path: Path, wanted: set):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if row[0] in wanted:
                yield row[0], (row[1], row[2])


def build(n_sample=500_000, kmax=50, seed=42, n_jobs=4, force=False, wave_s1=25_000,
          pseudo=None, pseudo_sample=50_000):
    X_path = TRAIN_SET_DIR / "X.npy"
    if X_path.exists() and not force:
        print(f"[train_set] {X_path} exists — skipping")
        return TRAIN_SET_DIR
    TRAIN_SET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    idf_used = False
    try:
        weights, default = load_name_idf(build_name_idf(n_jobs=n_jobs, force=force))
        pf.set_idf(weights, default)
        idf_used = True
    except Exception as e:
        print(f"[train_set] WARNING: name IDF unavailable ({e}); "
              f"name_idf_soft_jaccard degrades to plain Jaccard", flush=True)
        pf.set_idf({}, 1.0)

    gt_ids = read_gt_ids(config.TRAIN_DIR / "train_ground_truth.tsv")
    rng = random.Random(seed)
    sample = gt_ids if n_sample >= len(gt_ids) else rng.sample(gt_ids, n_sample)
    del gt_ids
    sample_set = set(sample)
    gt = load_gt(config.TRAIN_DIR / "train_ground_truth.tsv", sample_set)
    s1_recs = dict(_stream_records(config.TRAIN_DIR / "train_source1.tsv", sample_set))
    print(f"[train_set] sampled {len(sample):,} S1, {len(s1_recs):,} records "
          f"({time.time()-t0:.0f}s)", flush=True)

    # retrieval happens in waves; each wave's rows are freed after featurization
    waves = [sample[i:i + wave_s1] for i in range(0, len(sample), wave_s1)]
    del sample, sample_set

    n_reserve = 50 * (n_sample + (pseudo_sample if pseudo else 0))  # prune caps at 45/S1
    X_mm = np.memmap(TRAIN_SET_DIR / "X.mmap", dtype=np.float32, mode="w+",
                     shape=(n_reserve, N_FEATS))
    y_mm = np.memmap(TRAIN_SET_DIR / "y.mmap", dtype=np.int8, mode="w+",
                     shape=(n_reserve,))
    kw = pq.ParquetWriter(TRAIN_SET_DIR / "keys.tmp.parquet",
                          pa.schema([("source1_entity_id", pa.string()),
                                     ("candidate_id", pa.string())]))

    def consume(jobs, ys, ks1, kc):
        nonlocal wpos
        Xc = pf.build_matrix(jobs, n_jobs=n_jobs)
        del jobs
        n = Xc.shape[0]
        X_mm[wpos:wpos + n] = Xc
        del Xc
        y_mm[wpos:wpos + n] = np.asarray(ys, dtype=np.int8)
        npos = int(sum(ys))
        del ys
        kw.write_table(pa.table({"source1_entity_id": ks1, "candidate_id": kc}))
        del ks1, kc
        wpos += n
        X_mm.flush()
        y_mm.flush()
        return npos

    wpos = 0
    for wi, wave in enumerate(waves):
        raw = rt.retrieve("train", kmax=kmax, subset=set(wave), n_jobs=n_jobs,
                          prune=True, with_records=True)
        jobs, ys, ks1, kc = build_jobs(wave, raw, s1_recs, gt)
        del raw
        for sid in wave:  # release S1 strings already consumed
            s1_recs.pop(sid, None)
        consume(jobs, ys, ks1, kc)
        json.dump({"done_waves": wi + 1, "n_waves": len(waves), "pairs": int(wpos)},
                  open(TRAIN_SET_DIR / "partial.json", "w"))
        print(f"[train_set] wave {wi+1}/{len(waves)} ({wpos:,} pairs, "
              f"{time.time()-t0:.0f}s)", flush=True)

    # ---- pseudo-label extension: test-corpus S1s labelled by a prior inference run
    n_pseudo, pos_pseudo, pseudo_used = 0, 0, []
    if pseudo:
        pgt = load_pseudo(Path(pseudo))
        sids = sorted(pgt)
        if pseudo_sample and pseudo_sample < len(sids):
            sids = random.Random(seed + 1).sample(sids, pseudo_sample)
        s1_recs_p = dict(_stream_records(config.TEST_DIR / "test_source1.tsv", set(sids)))
        sids = [s for s in sids if s in s1_recs_p]
        pseudo_used = sids
        print(f"[train_set] pseudo: {len(sids):,} S1 from {pseudo} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for wi, wave in enumerate([sids[i:i + wave_s1]
                                   for i in range(0, len(sids), wave_s1)]):
            raw = rt.retrieve("test", kmax=kmax, subset=set(wave), n_jobs=n_jobs,
                              prune=True, with_records=True)
            jobs, ys, ks1, kc = build_jobs(wave, raw, s1_recs_p, pgt)
            del raw
            pos_pseudo += consume(jobs, ys, ks1, kc)
            n_pseudo += len(wave)
            print(f"[train_set] pseudo wave {wi+1} ({wpos:,} pairs, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        del pgt, s1_recs_p

    del gt, s1_recs
    kw.close()
    X_mm.flush()
    y_mm.flush()
    X = np.array(X_mm[:wpos])
    y = np.array(y_mm[:wpos])
    del X_mm, y_mm
    np.save(X_path, X)
    del X
    np.save(TRAIN_SET_DIR / "y.npy", y)
    (TRAIN_SET_DIR / "X.mmap").unlink(missing_ok=True)
    (TRAIN_SET_DIR / "y.mmap").unlink(missing_ok=True)
    (TRAIN_SET_DIR / "keys.tmp.parquet").replace(TRAIN_SET_DIR / "keys.parquet")
    (TRAIN_SET_DIR / "partial.json").unlink(missing_ok=True)
    # Pseudo entities have no GT in train_ground_truth.tsv — train_matcher must
    # keep them out of the val split, or they pollute the honest macro F0.5.
    (TRAIN_SET_DIR / "pseudo_s1.json").write_text(json.dumps(pseudo_used))
    meta = {
        "n_sample": n_sample, "kmax": kmax, "seed": seed,
        "positives": int(y.sum()), "pairs": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "feature_names": pf.FEATURE_NAMES, "name_idf": idf_used,
        "pseudo": {"path": str(pseudo) if pseudo else None,
                   "s1s": n_pseudo, "positives": pos_pseudo}
                   if pseudo else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (TRAIN_SET_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_set] X=({len(y):,}, {N_FEATS}) positives={int(y.sum()):,} "
          f"({time.time()-t0:.0f}s)")
    return TRAIN_SET_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--kmax", type=int, default=config.K_MAX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--wave-s1", type=int, default=25_000)
    ap.add_argument("--pseudo", type=Path, default=None,
                    help="pseudo-label TSV from make_pseudo_labels.py; adds "
                         "high-confidence test-corpus pairs as extra training data")
    ap.add_argument("--pseudo-sample", type=int, default=50_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    build(a.sample, a.kmax, a.seed, a.n_jobs, a.force, a.wave_s1,
          a.pseudo, a.pseudo_sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
