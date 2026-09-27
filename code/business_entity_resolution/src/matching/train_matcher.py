"""LightGBM matcher: S1-grouped split, early stop, refit 1.1x iters on 100%.

Plan §5.5-§5.7. Thresholds are tuned on the *honest* held-out val predictions from
the early-stopped model (val never seen by that model); the final production model is
then refit on 100% of entities at 1.1x the early-stopped iteration count.

Every run prints the oracle / all-empty / top-1 baselines alongside the tuned rule.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
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

PARAM_KEYS = ("tau_singleton", "tau_match", "gamma_rel", "max_matches")


def load_training_set():
    X = np.load(TRAIN_SET_DIR / "X.npy")
    if X.ndim != 2 or X.shape[1] != len(pf.FEATURE_NAMES):
        raise RuntimeError(
            f"cached training set has {X.shape[1] if X.ndim == 2 else '?'} features "
            f"but FEATURE_NAMES now has {len(pf.FEATURE_NAMES)} — the feature set "
            f"changed, rebuild it: python3 -m src.matching.build_training_set --force")
    y = np.load(TRAIN_SET_DIR / "y.npy")
    keys = pq.read_table(TRAIN_SET_DIR / "keys.parquet").to_pandas()
    meta = json.loads((TRAIN_SET_DIR / "meta.json").read_text())
    p_path = TRAIN_SET_DIR / "pseudo_s1.json"
    pseudo = set(json.loads(p_path.read_text())) if p_path.exists() else set()
    return (X, y, keys["source1_entity_id"].to_numpy(), keys["candidate_id"].to_numpy(),
            meta, pseudo)


def _countries_for(ids):
    """country string per S1 id, from train_source1.tsv (col 3)."""
    want = set(ids)
    out = {}
    with open(config.TRAIN_DIR / "train_source1.tsv", newline="",
              encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if row[0] in want:
                out[row[0]] = row[3]
    return out


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
    cmat = np.full((len(val_ids), max_c), "", object)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(codes)) - np.repeat(offs, counts)
    P[codes, pos] = probs
    T[codes, pos] = yy
    cmat[codes, pos] = cands
    true_total = np.array([len(gt.get(s, [])) for s in val_ids], dtype=np.int64)
    return P, T, true_total, cmat


def _probe_cuda(X, y, n_jobs):
    """3-round CUDA train on a small slice — cheap check before committing."""
    try:
        lgb.train({"objective": "binary", "device_type": "cuda", "verbose": -1,
                   "num_leaves": 15, "n_jobs": n_jobs},
                  lgb.Dataset(X[:5000], label=y[:5000]), num_boost_round=3)
        return True
    except Exception as e:
        print(f"[train] CUDA probe failed ({str(e)[:160]}); falling back to CPU",
              flush=True)
        return False


def train(n_sample=None, seed=42, val_frac=0.2, save=True, n_jobs=16, device=None,
          seeds=1):
    t0 = time.time()
    X, y, s1, cand, meta, pseudo = load_training_set()
    if n_sample and n_sample < len(X):
        rs = np.random.RandomState(seed)
        idx = np.sort(rs.choice(len(X), n_sample, replace=False))
        X, y, s1, cand = X[idx], y[idx], s1[idx], cand[idx]

    uniq = np.unique(s1)
    # Pseudo-labelled test entities have no GT in train_ground_truth.tsv; scoring
    # them as val would fake-zero them (they would look like singletons).
    val_pool = [u for u in uniq if u not in pseudo]
    if pseudo:
        print(f"[train] excluding {len(uniq) - len(val_pool):,} pseudo entities "
              f"from the val pool", flush=True)
    rng = random.Random(seed)
    val_ids = set(rng.sample(val_pool, max(1, int(len(val_pool) * val_frac))))
    is_val = np.fromiter((s in val_ids for s in s1), dtype=bool, count=len(s1))
    print(f"[train] pairs={len(y):,} pos={int(y.sum()):,} entities={len(uniq):,} "
          f"val_entities={len(val_ids):,} name_idf={meta.get('name_idf', False)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    dev = config.lgb_device_params(device)
    if dev.get("device_type") == "cuda" and not _probe_cuda(X, y, n_jobs):
        dev = {"device_type": "cpu"}
    params = dict(LGB_PARAMS, n_jobs=n_jobs, **dev)
    print(f"[train] device={dev.get('device_type')} seeds={seeds} "
          f"({time.time()-t0:.0f}s)", flush=True)
    dtrain = lgb.Dataset(X[~is_val], label=y[~is_val], feature_name=pf.FEATURE_NAMES)
    dval = lgb.Dataset(X[is_val], label=y[is_val], reference=dtrain)
    probs_sum = np.zeros(int(is_val.sum()), np.float64)
    best_iters = []
    for si in range(seeds):
        sparams = dict(params,
                       random_state=seed + si * 1000,
                       feature_fraction_seed=seed + si * 1000,
                       bagging_seed=seed + si * 1000)
        bst = lgb.train(sparams, dtrain, num_boost_round=2500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100, verbose=False),
                                   lgb.log_evaluation(200)])
        best_iter = bst.best_iteration or 2500
        best_iters.append(best_iter)
        probs_sum += bst.predict(X[is_val])
        print(f"[train] seed {si}: early stop at iter {best_iter} "
              f"(val logloss {bst.best_score['valid_0']['binary_logloss']:.5f}) "
              f"({time.time()-t0:.0f}s)", flush=True)
    probs = probs_sum / seeds
    best_iter = best_iters[0]

    val_ids_sorted = sorted(val_ids)
    gt = load_gt(config.TRAIN_DIR / "train_ground_truth.tsv", set(val_ids_sorted))
    P, T, true_total, cmat = padded_val(probs, s1[is_val], cand[is_val], y[is_val],
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

    # ---- one-to-one resolution of the selected claims (GT is a perfect matching)
    dedup_f, n_claims, n_kept = sel.apply_one_to_one_f05(
        P, T, true_total, cmat, val_ids_sorted,
        **{k: best[k] for k in ("tau_singleton", "tau_match", "gamma_rel")})
    print(f"[train] one-to-one: {n_claims:,} claims -> {n_kept:,} kept, "
          f"val macro F0.5 {dedup_f:.4f} (delta {dedup_f - best_f:+.4f})")

    # ---- per-country thresholds (France has no val labels -> falls back to global)
    countries = _countries_for(val_ids_sorted)
    c_arr = np.array([countries.get(s, "?") for s in val_ids_sorted])
    per_country = {}
    for c in sorted(set(c_arr)):
        m = c_arr == c
        if m.sum() < 500:
            print(f"[train] {c}: {int(m.sum())} val entities — too few to tune, "
                  f"will use global rule")
            continue
        best_c, f_c, _ = sel.grid_search(P[m], T[m], true_total[m])
        per_country[c] = dict(best_c, f05=round(f_c, 4), val_entities=int(m.sum()))
        print(f"[train] {c}: {best_c} -> F0.5 {f_c:.4f} ({int(m.sum())} entities)")

    # ---- test-mix-weighted val metric (val is US-heavy; test is India-heavy)
    mix = config.TEST_COUNTRY_MIX
    cnt = Counter(c_arr)
    w = np.array([mix.get(c, min(mix.values())) / max(cnt[c], 1) for c in c_arr])
    f_e = sel.per_entity_f05(P, T, true_total,
                             **{k: best[k] for k in ("tau_singleton", "tau_match",
                                                     "gamma_rel")})[0]
    w_f = float((np.asarray(f_e) * w).sum() / w.sum())
    print(f"[train] test-mix-weighted val macro F0.5 = {w_f:.4f} "
          f"(mix {json.dumps(mix)})", flush=True)

    out = MODEL_DIR
    if save:
        out.mkdir(parents=True, exist_ok=True)
        final_dev = dict(dev)
        final_params = dict(params)
        if dev.get("device_type") == "gpu":
            # OpenCL tree learner is flaky on full refits (left_count fatal); the CPU
            # refit of the same data is numerically equivalent and always succeeds.
            # The CUDA backend refits on the GPU without issue, so it is kept there.
            final_dev = {"device_type": "cpu"}
            final_params = dict(LGB_PARAMS, n_jobs=n_jobs, **final_dev)
            print("[train] refit on CPU (OpenCL refit is unstable on this build)",
                  flush=True)
        for si, bi_iter in enumerate(best_iters):
            sparams = dict(final_params,
                           random_state=seed + si * 1000,
                           feature_fraction_seed=seed + si * 1000,
                           bagging_seed=seed + si * 1000)
            final = lgb.train(sparams, lgb.Dataset(X, label=y,
                                                   feature_name=pf.FEATURE_NAMES),
                              num_boost_round=max(50, int(bi_iter * 1.1)))
            final.save_model(str(out / ("lgbm.txt" if si == 0 else f"lgbm_s{si}.txt")))
        json.dump({
            "best_params": best, "val_macro_f05": best_f, "best_iter": best_iter,
            "baselines": baselines, "feature_names": pf.FEATURE_NAMES,
            "n_estimators_final": max(50, int(best_iter * 1.1)),
            "val_entities": val_ids_sorted, "seed": seed, "val_frac": val_frac,
            "device": dev.get("device_type"), "n_models": seeds,
            "best_iters": best_iters,
            "val_macro_f05_one_to_one": dedup_f,
            "val_macro_f05_weighted": w_f,
            "best_params_per_country": per_country,
            "name_idf": meta.get("name_idf", False),
        }, open(out / "meta.json", "w"), indent=2)
        np.save(out / "val_probs.npy", P)
        np.save(out / "val_true.npy", T)
        np.save(out / "val_true_total.npy", true_total)
        json.dump(val_ids_sorted, open(out / "val_ids.json", "w"))
        print(f"[train] saved model -> {out} ({time.time()-t0:.0f}s)")

    return {"baselines": baselines, "best": best, "best_f05": best_f,
            "dedup_f05": dedup_f, "weighted_f05": w_f, "per_country": per_country,
            "grid": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "gpu", "cuda"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="seed-bagged ensemble size (models averaged at inference)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    train(a.sample, a.seed, a.val_frac, not a.no_save, a.n_jobs, a.device, a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
