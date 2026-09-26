"""LightGBM matcher: S1-grouped split, early stop, refit 1.1x iters on 100%.

Plan §5.5-§5.7. Thresholds are tuned on the *honest* held-out val predictions from
the early-stopped model (val never seen by that model); the final production model is
then refit on 100% of entities at 1.1x the early-stopped iteration count.

Every run prints the oracle / all-empty / top-1 baselines alongside the tuned rule.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking.evaluate_blocking import load_gt  # noqa: E402
from src.matching import pair_features as pf  # noqa: E402
from src.matching import selection as sel  # noqa: E402

TRAIN_SET_DIR = config.CACHE_DIR / "train_set"
MODEL_DIR = config.MODEL_DIR

LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1, "min_child_samples": 50,
    "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
    "n_estimators": 2500, "random_state": 42, "n_jobs": 16, "verbose": -1,
}


def load_training_set():
    X = np.load(TRAIN_SET_DIR / "X.npy")
    y = np.load(TRAIN_SET_DIR / "y.npy")
    keys = pq.read_table(TRAIN_SET_DIR / "keys.parquet").to_pandas()
    meta = json.loads((TRAIN_SET_DIR / "meta.json").read_text())
    return X, y, keys["source1_entity_id"].to_numpy(), keys["candidate_id"].to_numpy(), meta


def padded_val(probs, v_s1, v_cand, v_y, val_ids, gt, absent=sel.ABSENT):
    idx = {sid: i for i, sid in enumerate(val_ids)}
    codes = np.fromiter((idx.get(s, -1) for s in v_s1), dtype=np.int64, count=len(v_s1))
    keep = codes >= 0
    codes, cands, yy = codes[keep], v_cand[keep], v_y[keep].astype(bool)
    probs = np.asarray(probs)[keep]
    order = np.argsort(codes, kind="stable")  # group rows by entity
    codes, cands, yy, probs = codes[order], cands[order], yy[order], probs[order]
    counts = np.bincount(codes, minlength=len(val_ids))
    max_c = max(int(counts.max()), 1)
    P = np.full((len(val_ids), max_c), absent, np.float32)
    T = np.zeros((len(val_ids), max_c), bool)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(codes)) - np.repeat(offs, counts)
    P[codes, pos] = probs[keep]
    T[codes, pos] = yy
    true_total = np.array([len(gt.get(s, [])) for s in val_ids], dtype=np.int64)
    return P, T, true_total, cands


def train(n_sample=None, seed=42, val_frac=0.2, save=True, n_jobs=16, device=None):
    t0 = time.time()
    X, y, s1, cand, meta = load_training_set()
    if n_sample and n_sample < len(X):
        rs = np.random.RandomState(seed)
        idx = np.sort(rs.choice(len(X), n_sample, replace=False))
        X, y, s1, cand = X[idx], y[idx], s1[idx], cand[idx]

    uniq = np.unique(s1)
    rng = random.Random(seed)
    val_ids = set(rng.sample(list(uniq), max(1, int(len(uniq) * val_frac))))
    is_val = np.fromiter((s in val_ids for s in s1), dtype=bool, count=len(s1))
    print(f"[train] pairs={len(y):,} pos={int(y.sum()):,} entities={len(uniq):,} "
          f"val_entities={len(val_ids):,} ({time.time()-t0:.0f}s)", flush=True)

    dev = config.lgb_device_params(device)
    params = dict(LGB_PARAMS, n_jobs=n_jobs, **dev)
    print(f"[train] device={dev.get('device_type')} ({time.time()-t0:.0f}s)", flush=True)
    dtrain = lgb.Dataset(X[~is_val], label=y[~is_val], feature_name=pf.FEATURE_NAMES)
    dval = lgb.Dataset(X[is_val], label=y[is_val], reference=dtrain)
    bst = lgb.train(params, dtrain, num_boost_round=2500, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(200)])
    best_iter = bst.best_iteration or 2500
    print(f"[train] early stop at iter {best_iter} "
          f"(val logloss {bst.best_score['valid_0']['binary_logloss']:.5f}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    probs = bst.predict(X[is_val])
    val_ids_sorted = sorted(val_ids)
    gt = load_gt(config.TRAIN_DIR / "train_ground_truth.tsv", set(val_ids_sorted))
    P, T, true_total, v_cands = padded_val(probs, s1[is_val], cand[is_val], y[is_val],
                                           val_ids_sorted, gt)

    baselines = {
        "oracle_f05_on_our_candidates": sel.oracle_f05(T, true_total),
        "all_empty_f05": sel.all_empty_f05(true_total),
        "top1_f05": sel.top1_f05(P, T, true_total),
        "positive_pair_rate": float(y.mean()),
        "singleton_rate_val": float((true_total == 0).mean()),
    }
    print("[train] baselines:", json.dumps({k: round(v, 4) for k, v in baselines.items()}))

    best, best_f, results = sel.grid_search(P, T, true_total)
    print(f"[train] best rule {best} -> val macro F0.5 = {best_f:.4f}")

    out = MODEL_DIR
    if save:
        out.mkdir(parents=True, exist_ok=True)
        final = lgb.train(params, lgb.Dataset(X, label=y, feature_name=pf.FEATURE_NAMES),
                          num_boost_round=max(50, int(best_iter * 1.1)))
        final.save_model(str(out / "lgbm.txt"))
        json.dump({
            "best_params": best, "val_macro_f05": best_f, "best_iter": best_iter,
            "baselines": baselines, "feature_names": pf.FEATURE_NAMES,
            "n_estimators_final": max(50, int(best_iter * 1.1)),
            "val_entities": val_ids_sorted, "seed": seed, "val_frac": val_frac,
            "device": dev.get("device_type"),
        }, open(out / "meta.json", "w"), indent=2)
        np.save(out / "val_probs.npy", P)
        np.save(out / "val_true.npy", T)
        np.save(out / "val_true_total.npy", true_total)
        json.dump(val_ids_sorted, open(out / "val_ids.json", "w"))
        print(f"[train] saved model -> {out} ({time.time()-t0:.0f}s)")

    return {"baselines": baselines, "best": best, "best_f05": best_f, "grid": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "gpu", "cuda"])
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    train(a.sample, a.seed, a.val_frac, not a.no_save, a.n_jobs, a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
