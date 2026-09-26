"""Blocking eval: recall@K, score calibration, per-country slices, missed-pair dumps.

The plan (§2, §5.3, §11 item 5) insists the recall@K curve be **measured locally**
on real train data, not inherited from the prior log. This module does exactly that:
it samples train S1 (true matches known), retrieves raw top-Kmax and reports:

  * micro recall@K for K in {1,3,5,10,20,30,50,75,100}
  * the same after §2 adaptive pruning (the actual candidate set)
  * score distributions for true matches vs singletons (to calibrate the prune
    floors / singleton gate, which are prior-log values otherwise)
  * per-country and per-source slices
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking import retrieve as rt  # noqa: E402

KS = (1, 3, 5, 10, 20, 30, 50, 75, 100)


def load_gt(path: Path, subset: set = None):
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if subset is not None and row[0] not in subset:
                continue
            gt[row[0]] = [x for x in row[1].split(",") if x] if len(row) > 1 else []
    return gt


def read_gt_ids(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        return [row[0] for row in rd]


def load_countries(corpus="train", subset: set = None):
    p = rt.S1_DIRS[corpus] / rt.S1_FILES[corpus]
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if subset is None or row[0] in subset:
                out[row[0]] = row[3]
    return out


def sample_ids(gt_ids, n, seed=42):
    rng = random.Random(seed)
    if n >= len(gt_ids):
        return list(gt_ids)
    return rng.sample(gt_ids, n)


def recall_curve(raw, gt, sample, ks=KS):
    hit = {k: 0 for k in ks}
    tot = 0
    for sid in sample:
        truth = set(gt.get(sid, []))
        tot += len(truth)
        ranked = [i for _s, i in raw.get(sid, [])]
        for k in ks:
            hit[k] += len(set(ranked[:k]) & truth)
    return {k: (hit[k] / tot if tot else 0.0) for k in ks}, tot


def diagnostic(raw, gt, sample, countries):
    """Best true-match rank/score and singleton top-1 score distributions."""
    rank_hist, true_scores, singleton_top, nonsingleton_top = [], [], [], []
    missed = []
    for sid in sample:
        truth = set(gt.get(sid, []))
        row = raw.get(sid, [])
        top = row[0][0] if row else 0.0
        ranked = [i for _s, i in row]
        if not truth:
            singleton_top.append(top)
            continue
        nonsingleton_top.append(top)
        ranks = [ranked.index(t) + 1 for t in truth if t in ranked]
        for t in truth:
            if t in ranked:
                true_scores.append(row[ranked.index(t)][0])
            else:
                missed.append((sid, t, countries.get(sid, "?")))
        rank_hist.append(max(ranks) if ranks else 0)
    return {
        "best_true_rank_hist": {str(r): rank_hist.count(r) for r in sorted(set(rank_hist))},
        "true_score_pcts": _pcts(true_scores),
        "singleton_top1_pcts": _pcts(singleton_top),
        "nonsingleton_top1_pcts": _pcts(nonsingleton_top),
        "n_missed_pairs": len(missed),
        "missed_by_country": _count(missed, 2),
        "missed_examples": missed[:20],
    }


def _pcts(x):
    if not x:
        return {}
    a = np.array(x)
    return {f"p{p}": round(float(np.percentile(a, p)), 4)
            for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}


def _count(rows, idx):
    d = {}
    for r in rows:
        d[r[idx]] = d.get(r[idx], 0) + 1
    return d


SWEEP_CONFIGS = [
    ("B_loose2", dict(floors=(0.30, 0.20), caps=(20, 40))),
    ("C_loose3", dict(floors=(0.25, 0.15), caps=(25, 50))),
    ("K_30_18", dict(floors=(0.30, 0.18), caps=(30, 45))),
    ("L_28_17", dict(floors=(0.28, 0.17), caps=(30, 45))),
    ("M_32_20", dict(floors=(0.32, 0.20), caps=(28, 40))),
    ("N_28_16", dict(floors=(0.28, 0.16), caps=(28, 45))),
    ("O_27_16", dict(floors=(0.27, 0.16), caps=(27, 45))),
    ("P_30_18_cf", dict(floors=(0.30, 0.18), caps=(50, 50))),
]


def oracle_f05_raw(raw, gt, sample, kmax):
    """Upper bound of ANY classifier restricted to this candidate set (macro F0.5)."""
    from src.matching.scorer import f05_entity
    vals = []
    for sid in sample:
        truth = set(gt.get(sid, []))
        got = {i for _s, i in raw.get(sid, [])[:kmax]} & truth
        vals.append(f05_entity(got, truth))
    return sum(vals) / max(len(vals), 1)


def prune_sweep(raw, gt, sample, kmax=50, configs=SWEEP_CONFIGS):
    """Recall + avg candidates for each adaptive-prune config, applied to top-``kmax``."""
    out = []
    for name, kw in configs:
        hit = tot = 0
        ncand = 0
        for sid in sample:
            truth = set(gt.get(sid, []))
            row = raw.get(sid, [])[:kmax]
            tot += len(truth)
            scores = np.array([s for s, _ in row], dtype=np.float32)
            ids = np.array([i for _, i in row], dtype=object)
            _s, kept = rt.adaptive_prune(scores, ids, **kw)
            ncand += len(kept)
            hit += len(set(kept) & truth)
        out.append({"config": name, **{k: str(v) for k, v in kw.items()},
                    "pruned_recall": hit / tot if tot else 0.0,
                    "avg_candidates": ncand / max(len(sample), 1)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    gt_ids = read_gt_ids(config.TRAIN_DIR / "train_ground_truth.tsv")
    sample = sample_ids(gt_ids, a.sample, a.seed)
    sample_set = set(sample)
    del gt_ids
    gt = load_gt(config.TRAIN_DIR / "train_ground_truth.tsv", sample_set)
    countries = load_countries("train", sample_set)
    print(f"[eval] sampled {len(sample):,} train S1; retrieving top-{a.kmax} "
          f"({time.time()-t0:.0f}s)", flush=True)

    raw = rt.retrieve("train", kmax=a.kmax, subset=set(sample), n_jobs=a.n_jobs)

    curve, tot = recall_curve(raw, gt, sample)
    pruned = rt.prune_results({k: v[:min(a.kmax, 50)] for k, v in raw.items()})
    pcurve, _ = recall_curve({k: [(s, i) for s, i in v] for k, v in pruned.items()},
                             gt, sample, ks=(min(a.kmax, 50),))
    diag = diagnostic(raw, gt, sample, countries)
    sweep50 = prune_sweep(raw, gt, sample, kmax=50)
    sweep100 = prune_sweep(raw, gt, sample, kmax=min(a.kmax, 100))
    oracle50 = oracle_f05_raw(raw, gt, sample, 50)
    oracle100 = oracle_f05_raw(raw, gt, sample, min(a.kmax, 100))

    report = {
        "sample": len(sample),
        "kmax": a.kmax,
        "true_pairs_in_sample": tot,
        "recall_at_k": curve,
        "oracle_macro_f05_k50": oracle50,
        "oracle_macro_f05_k100": oracle100,
        "pruned_recall_at_kmax": pcurve,
        "avg_candidates_pruned": (sum(len(v) for v in pruned.values()) / max(len(pruned), 1)),
        "prune_sweep_top50": sweep50,
        "prune_sweep_top100": sweep100,
        "diagnostics": diag,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[eval] ORACLE macro F0.5 on our candidates: K=50 {oracle50:.4f} | "
          f"K=100 {oracle100:.4f}")
    for label, sweep in (("top-50", sweep50), ("top-100", sweep100)):
        print(f"[eval] prune sweep ({label}):")
        for row in sweep:
            print(f"    {row['config']:12s} floors={row['floors']:10s} caps={row['caps']:8s} "
                  f"recall={row['pruned_recall']:.4f} avg_cand={row['avg_candidates']:.1f}")
    print(json.dumps(report, indent=2, default=str))
    if a.out:
        a.out.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
