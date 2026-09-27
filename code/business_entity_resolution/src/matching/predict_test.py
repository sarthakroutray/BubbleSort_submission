"""Test inference: per-country, per-batch, checkpointed → the two output TSVs.

Plan §5.8, made memory-lean after an OOM/reboot. Design:
  * one country at a time (France 1.4M / India 4.7M / US 3.8M docs);
  * top-Kmax for the whole country once (only scores + column ids held), then
    candidate records materialised one shard at a time, per batch;
  * pairs scored in small batches, selection applied, rows appended to a
    per-country checkpoint file (disk, not RAM) WITH the match probabilities;
  * the two output TSVs are assembled from the per-country checkpoints, so a
    restart resumes finished countries instead of recomputing them.

v2 additions:
  * seed-ensemble models (``lgbm*.txt``) are averaged per pair;
  * selection params are per-country when ``meta.json`` has
    ``best_params_per_country`` (France falls back to the global rule);
  * assembly resolves contested claims one-to-one (GT is a perfect matching):
    when two S1s claim the same S2/S3, only the higher-probability claim is
    kept — ``--no-dedup`` disables this.

``matching_results ⊆ candidate_pairs`` by construction (selection only drops).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking import retrieve as rt  # noqa: E402
from src.matching import pair_features as pf  # noqa: E402
from src.matching import selection as sel  # noqa: E402

CKPT = config.BATCH_DIR
PARAM_KEYS = ("tau_singleton", "tau_match", "gamma_rel", "max_matches")


def load_model():
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    paths = sorted(config.MODEL_DIR.glob("lgbm*.txt"))
    if not paths:
        raise FileNotFoundError(f"no lgbm*.txt under {config.MODEL_DIR}")
    boosters = [lgb.Booster(model_file=str(p)) for p in paths]
    return boosters, meta


def _load_idf(meta):
    """Activate the IDF map the model was trained with (plain Jaccard if not)."""
    if not meta.get("name_idf"):
        pf.set_idf({}, 1.0)
        return
    path = config.CACHE_DIR / "train_set" / "name_idf.json"
    if not path.exists():
        raise FileNotFoundError(
            f"model trained with the IDF feature but {path} is missing — rebuild "
            f"the training set (or copy name_idf.json into the work cache)")
    d = json.loads(path.read_text())
    pf.set_idf(d["weights"], float(d["default"]))


def params_for(meta, country):
    """(selection params, tuned_for_this_country) — France falls back to global."""
    params = {k: meta["best_params"][k] for k in PARAM_KEYS if k in meta["best_params"]}
    per = (meta.get("best_params_per_country") or {}).get(str(country))
    if per:
        params.update({k: per[k] for k in ("tau_singleton", "tau_match", "gamma_rel")
                       if k in per})
        return params, True
    return params, False


def _select(cand_probs, params):
    cands = [{"entity_id": c, "prob": float(p)} for c, p in cand_probs]
    return sel.select_matches(cands, return_probs=True, **params)


def _predict(boosters, X, device):
    """LightGBM prediction, averaged over the ensemble; cuda/gpu = CUDA/OpenCL.

    ``**kwargs`` on ``Booster.predict`` become prediction parameters, so the
    device is passed as ``device_type=<backend>`` directly (the old
    ``kwargs={...}`` form was rejected and always fell back to CPU).
    """
    if not len(X):
        return np.zeros(0)
    nf = boosters[0].num_feature()
    if X.shape[1] > nf:
        # v2 features are appended AFTER the original 38, so a model trained on an
        # older feature set can slice them off and stay byte-compatible.
        X = X[:, :nf]
    elif X.shape[1] < nf:
        raise ValueError(f"model expects {nf} features, got {X.shape[1]} — "
                         f"rebuild the training set / retrain")
    if device in ("gpu", "cuda"):
        try:
            return np.mean([b.predict(X, device_type=device) for b in boosters],
                           axis=0)
        except Exception as e:  # pragma: no cover - fall back, CPU metrics identical
            print(f"[predict] GPU predict failed ({str(e)[:120]}); using CPU",
                  flush=True)
    return np.mean([b.predict(X) for b in boosters], axis=0)


def process_country(country, idx_dir, imeta, boosters, params, kmax, batch, n_jobs,
                    ckpt: Path, limit=0):
    layout = rt.shard_layout(idx_dir, imeta, country)
    ids, names, addrs = rt.load_s1_country("test", country, limit)
    nq = len(ids)
    idf = np.load(idx_dir / f"idf_{rt.slug(str(country))}.npy")
    q = rt.build_queries(names, addrs, idf)
    best_v, best_c = rt.topk_country(idx_dir, layout, q, kmax, n_jobs)
    del q
    partial = ckpt.with_suffix(".partial.tsv")
    with open(partial, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for start in range(0, nq, batch):
            end = min(start + batch, nq)
            cid, cn, ca = rt.gather_records(idx_dir, layout, best_c[start:end], True)
            jobs, rows = [], []
            for r in range(end - start):
                i = start + r
                js, jids = rt.adaptive_prune(best_v[i], cid[r])
                rank = {c: k for k, c in enumerate(cid[r])}
                n1, a1 = str(names[i]), str(addrs[i])
                best = js[0] if js else 0.0
                row_jobs = []
                for jj, (sc, c) in enumerate(zip(js, jids)):
                    k = rank[c]
                    nxt = js[jj + 1] if jj + 1 < len(js) else 0.0
                    ctx = (sc, jj + 1, (sc / best if best > 0 else 0.0),
                           best - sc, sc - nxt, c)
                    row_jobs.append((n1, a1, str(cn[r][k]), str(ca[r][k]), ctx))
                jobs.extend(row_jobs)
                rows.append((i, jids, len(row_jobs)))
            X = pf.build_matrix(jobs, n_jobs=n_jobs) if jobs else np.zeros(
                (0, len(pf.FEATURE_NAMES)), np.float32)
            probs = _predict(boosters, X, config.LGB_DEVICE) if len(X) else np.zeros(0)
            off = 0
            for i, jids, cnt in rows:
                p = probs[off:off + cnt]
                off += cnt
                sel_pairs = _select(list(zip(jids, p)), params)
                mids = [m for m, _ in sel_pairs]
                mps = [mp for _, mp in sel_pairs]
                w.writerow([str(ids[i]), ",".join(dict.fromkeys(jids)),
                            ",".join(mids), ",".join(f"{x:.6f}" for x in mps)])
            del cid, cn, ca, jobs, rows, X, probs
            print(f"[predict] {country} {end:,}/{nq:,}", flush=True)
    partial.replace(ckpt)
    del ids, names, addrs, best_v, best_c
    return nq


def predict(kmax=50, batch=20_000, n_jobs=4, force=False, countries=None, limit=0,
            dedup=True):
    CKPT.mkdir(parents=True, exist_ok=True)
    boosters, meta = load_model()
    _load_idf(meta)
    print(f"[predict] models={len(boosters)} val_f05={meta['val_macro_f05']:.4f} "
          f"(one-to-one {meta.get('val_macro_f05_one_to_one', float('nan')):.4f})",
          flush=True)
    idx_dir, imeta = rt.bi.load_index("test")
    all_countries = sorted(imeta["countries"].keys())
    todo = [c for c in all_countries if countries is None or str(c) in countries]
    t0 = time.time()
    done = []
    for country in todo:
        ckpt = CKPT / f"{rt.slug(str(country))}.rows.tsv"
        if ckpt.exists() and not force:
            print(f"[predict] {country}: checkpoint exists, skipping", flush=True)
            done.append(country)
            continue
        params, tuned = params_for(meta, country)
        print(f"[predict] {country}: params={params} "
              f"({'per-country' if tuned else 'global rule'})", flush=True)
        nq = process_country(country, idx_dir, imeta, boosters, params, kmax, batch,
                             n_jobs, ckpt, limit)
        done.append(country)
        print(f"[predict] {country}: done {nq:,} ({time.time()-t0:.0f}s)", flush=True)

    if limit and len(done) < len(all_countries):
        print(f"[predict] --limit run: assembling only {done} "
              f"(skipping absent country checkpoints)", flush=True)
    _assemble(done, t0, dedup=dedup)
    return config.MATCHING_FILE, config.CANDIDATE_FILE


def _assemble(countries, t0, dedup=True):
    """Stream checkpoints -> candidate_pairs.tsv; one-to-one resolve -> matching."""
    outs = {
        config.CANDIDATE_FILE: "candidate_entity_ids",
        config.MATCHING_FILE: "matched_entity_ids",
    }
    handles, tmps = {}, {}
    for path, col in outs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        h = open(tmp, "w", newline="", encoding="utf-8")
        csv.writer(h, delimiter="\t", lineterminator="\n").writerow(
            ["source1_entity_id", col])
        handles[path] = h
        tmps[path] = tmp

    sids, ids_all, probs_all = [], [], []
    n_rank_proxy = 0
    n = 0
    wcand = csv.writer(handles[config.CANDIDATE_FILE], delimiter="\t",
                       lineterminator="\n")
    found = 0
    for country in countries:
        ckpt = CKPT / f"{rt.slug(str(country))}.rows.tsv"
        if not ckpt.exists():
            print(f"[predict] WARNING: no checkpoint for {country} — skipped in "
                  f"assembly (partial run?)", flush=True)
            continue
        found += 1
        with open(ckpt, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if not row:
                    continue
                sid, cand = row[0], row[1]
                mids = row[2].split(",") if len(row) > 2 and row[2] else []
                ps = None
                if len(row) > 3 and row[3]:
                    ps = [float(x) for x in row[3].split(",") if x]
                    if len(ps) != len(mids):
                        ps = None
                if ps is None:
                    # Legacy checkpoint: no probs, but matches are stored in
                    # probability order — use within-row rank as the score
                    # (a contest is always inside one country, so scales never mix).
                    n_rank_proxy += 1 if mids else 0
                    ps = [-float(i) for i in range(len(mids))]
                wcand.writerow([sid, cand])
                sids.append(sid)
                ids_all.append(mids)
                probs_all.append(ps)
                n += 1

    removed = 0
    if not found:
        raise RuntimeError("no country checkpoints found — run inference first")
    if dedup:
        claims = [(sid, c, p) for sid, mids, ps in zip(sids, ids_all, probs_all)
                  for c, p in zip(mids, ps)]
        kept = sel.resolve_one_to_one(claims)
        before = sum(len(x) for x in ids_all)
        ids_all = [[c for c, p in zip(mids, ps) if (sid, c) in kept]
                   for sid, mids, ps in zip(sids, ids_all, probs_all)]
        removed = before - sum(len(x) for x in ids_all)
        proxy = f" ({n_rank_proxy:,} rows ranked by position)" if n_rank_proxy else ""
        print(f"[predict] one-to-one: {len(claims):,} claims -> {before - removed:,} "
              f"kept ({removed:,} contested claims removed){proxy}", flush=True)

    wmatch = csv.writer(handles[config.MATCHING_FILE], delimiter="\t",
                        lineterminator="\n")
    for sid, mids in zip(sids, ids_all):
        wmatch.writerow([sid, ",".join(dict.fromkeys(mids))])

    for path, h in handles.items():
        h.close()
        tmps[path].replace(path)
    print(f"[predict] assembled {n:,} rows -> {config.MATCHING_FILE} + "
          f"{config.CANDIDATE_FILE} ({time.time()-t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmax", type=int, default=config.K_MAX)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--countries", default=None,
                    help="comma list to (re)process, e.g. 'France'; default all")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N S1 rows per country (smoke tests)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable the one-to-one contested-claim pass")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    countries = set(a.countries.split(",")) if a.countries else None
    predict(a.kmax, a.batch, a.n_jobs, a.force, countries, a.limit,
            not a.no_dedup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
