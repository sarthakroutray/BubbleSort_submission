"""Test inference: per-country, per-batch, checkpointed → the two output TSVs.

Plan §5.8, made memory-lean after an OOM/reboot. Design:
  * one country at a time (France 1.4M / India 4.7M / US 3.8M docs);
  * top-Kmax for the whole country once (only scores + column ids held), then
    candidate records materialised one shard at a time, per batch;
  * pairs scored in small batches, selection applied, rows appended to a
    per-country checkpoint file (disk, not RAM);
  * the two output TSVs are assembled from the per-country checkpoints, so a
    restart resumes finished countries instead of recomputing them.

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


def load_model():
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(config.MODEL_DIR / "lgbm.txt"))
    keys = ("tau_singleton", "tau_match", "gamma_rel", "max_matches")
    params = {k: v for k, v in meta["best_params"].items() if k in keys}
    return booster, meta, params


def _select(cand_probs, params):
    cands = [{"entity_id": c, "prob": float(p)} for c, p in cand_probs]
    return sel.select_matches(cands, **params)


def _predict(booster, X, device):
    """LightGBM prediction; ``gpu`` uses the OpenCL backend (plan §7)."""
    if device in ("gpu", "cuda") and len(X):
        try:
            return booster.predict(X, kwargs={"device_type": "gpu"})
        except Exception as e:  # pragma: no cover - fall back, CPU metrics identical
            print(f"[predict] GPU predict failed ({e}); using CPU", flush=True)
    return booster.predict(X)


def process_country(country, idx_dir, imeta, booster, params, kmax, batch, n_jobs,
                    ckpt: Path):
    layout = rt.shard_layout(idx_dir, imeta, country)
    ids, names, addrs = rt.load_s1_country("test", country)
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
            probs = _predict(booster, X, config.LGB_DEVICE) if len(X) else np.zeros(0)
            off = 0
            for i, jids, cnt in rows:
                p = probs[off:off + cnt]
                off += cnt
                sel_ids = _select(list(zip(jids, p)), params)
                w.writerow([str(ids[i]), ",".join(dict.fromkeys(jids)),
                            ",".join(sel_ids)])
            del cid, cn, ca, jobs, rows, X, probs
            print(f"[predict] {country} {end:,}/{nq:,}", flush=True)
    partial.replace(ckpt)
    del ids, names, addrs, best_v, best_c
    return nq


def predict(kmax=50, batch=20_000, n_jobs=4, force=False):
    CKPT.mkdir(parents=True, exist_ok=True)
    booster, meta, params = load_model()
    print(f"[predict] params={params} val_f05={meta['val_macro_f05']:.4f}", flush=True)
    idx_dir, imeta = rt.bi.load_index("test")
    countries = sorted(imeta["countries"].keys())
    t0 = time.time()
    for country in countries:
        ckpt = CKPT / f"{rt.slug(str(country))}.rows.tsv"
        if ckpt.exists() and not force:
            print(f"[predict] {country}: checkpoint exists, skipping", flush=True)
            continue
        nq = process_country(country, idx_dir, imeta, booster, params, kmax, batch,
                             n_jobs, ckpt)
        print(f"[predict] {country}: done {nq:,} ({time.time()-t0:.0f}s)", flush=True)

    _assemble(countries, t0)
    return config.MATCHING_FILE, config.CANDIDATE_FILE


def _assemble(countries, t0):
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
    n = 0
    for country in countries:
        ckpt = CKPT / f"{rt.slug(str(country))}.rows.tsv"
        if not ckpt.exists():
            raise RuntimeError(f"missing checkpoint for {country}")
        with open(ckpt, newline="", encoding="utf-8") as f:
            for sid, cand, match in csv.reader(f, delimiter="\t"):
                csv.writer(handles[config.CANDIDATE_FILE], delimiter="\t",
                           lineterminator="\n").writerow([sid, cand])
                csv.writer(handles[config.MATCHING_FILE], delimiter="\t",
                           lineterminator="\n").writerow([sid, match])
                n += 1
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
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    predict(a.kmax, a.batch, a.n_jobs, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
