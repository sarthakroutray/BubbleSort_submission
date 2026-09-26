"""F0.5 selection: singleton gate + relative threshold, cap 12. Shared val/test.

``select_matches`` is the per-entity rule (plan §5.7). The vectorized helpers apply
the same rule over a padded (entity, candidate) matrix so the threshold grid can be
searched on macro F0.5.

``true_total`` is the number of ground-truth matches per entity (from GT, NOT just
the ones blocking found) — it is what makes recall, and therefore the oracle, honest.
"""

import itertools

import numpy as np

ABSENT = -1.0


def select_matches(candidates, tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70,
                   max_matches=12):
    if not candidates:
        return []
    best = max(c["prob"] for c in candidates)
    if best < tau_singleton:
        return []
    thr = max(tau_match, best * gamma_rel)
    kept = sorted((c for c in candidates if c["prob"] >= thr),
                  key=lambda c: c["prob"], reverse=True)
    return [c["entity_id"] for c in kept[:max_matches]]


def _f05(p, r):
    denom = 0.25 * p + r
    return np.where(denom > 0, 1.25 * p * r / np.where(denom > 0, denom, 1.0), 0.0)


def _f_from_counts(pred_count, hit, true_total):
    true_total = np.asarray(true_total, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred_count > 0, hit / np.maximum(pred_count, 1), 0.0)
        r = np.where(true_total > 0, hit / np.maximum(true_total, 1), 1.0)
    return np.where((pred_count == 0) & (true_total == 0), 1.0,
                    np.where((pred_count == 0) | (true_total == 0), 0.0, _f05(p, r)))


def evaluate_selection(P, T, true_total, tau_singleton=0.68, tau_match=0.60,
                       gamma_rel=0.70, max_matches=12):
    """P/T padded with ABSENT/False. Macro F0.5 over ALL entities."""
    P = np.asarray(P, dtype=np.float32)
    T = np.asarray(T, dtype=bool)
    best = P.max(axis=1)
    empty = best < tau_singleton
    thr = np.where(empty, np.inf, np.maximum(tau_match, best * gamma_rel))[:, None]
    keep = P >= thr
    masked = np.where(keep, P, -2.0)
    cap = min(max_matches, masked.shape[1])
    order = np.argsort(-masked, axis=1)[:, :cap]
    kept = np.take_along_axis(masked, order, axis=1) > ABSENT
    hits = (np.take_along_axis(T, order, axis=1) & kept).sum(axis=1)
    f = _f_from_counts(kept.sum(axis=1), hits, true_total)
    return float(f.mean()), kept.sum(axis=1), hits


def oracle_f05(T, true_total):
    """Perfect classifier over OUR candidates: predict exactly the true candidates."""
    T = np.asarray(T, dtype=bool)
    hits = T.sum(axis=1)
    f = _f_from_counts(hits, hits, true_total)
    return float(f.mean())


def all_empty_f05(true_total):
    return float((np.asarray(true_total) == 0).mean())


def top1_f05(P, T, true_total, thr=0.5):
    P = np.asarray(P, dtype=np.float32)
    T = np.asarray(T, dtype=bool)
    top = P.argmax(axis=1)
    p = P[np.arange(len(P)), top]
    pred = ((p >= thr) & (P.max(axis=1) >= 0)).astype(np.int64)
    hit = (T[np.arange(len(P)), top] & pred.astype(bool)).astype(np.int64)
    f = _f_from_counts(pred, hit, true_total)
    return float(f.mean())


def grid_search(P, T, true_total,
                tau_singleton=(0.55, 0.60, 0.65, 0.68, 0.70, 0.75),
                tau_match=(0.45, 0.50, 0.55, 0.60, 0.65),
                gamma_rel=(0.60, 0.70, 0.80, 0.90), max_matches=12):
    results = []
    for ts, tm, gr in itertools.product(tau_singleton, tau_match, gamma_rel):
        f, _, _ = evaluate_selection(P, T, true_total, ts, tm, gr, max_matches)
        results.append({"tau_singleton": ts, "tau_match": tm, "gamma_rel": gr, "f05": f})
    best = max(results, key=lambda d: d["f05"])
    return best, best["f05"], results
