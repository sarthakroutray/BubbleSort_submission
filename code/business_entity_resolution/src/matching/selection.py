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
                   max_matches=12, return_probs=False):
    if not candidates:
        return []
    best = max(c["prob"] for c in candidates)
    if best < tau_singleton:
        return []
    thr = max(tau_match, best * gamma_rel)
    kept = sorted((c for c in candidates if c["prob"] >= thr),
                  key=lambda c: c["prob"], reverse=True)[:max_matches]
    if return_probs:
        return [(c["entity_id"], c["prob"]) for c in kept]
    return [c["entity_id"] for c in kept]


def resolve_one_to_one(claims):
    """Greedy max-weight one-to-one resolution over contested claims.

    GT is a perfect matching (each S2/S3 belongs to at most one S1), so a
    candidate selected by two S1s is a guaranteed FP for one of them. Two-pass
    greedy (plan §8.1):
      1. every S1's single best claim competes globally, highest prob wins;
      2. all remaining claims fill in prob-desc order on unclaimed candidates.

    Pass 1 guarantees an S1 keeps at least its best claim unless that claim
    itself lost to a stronger S1 — the evidence then says the candidate
    belongs elsewhere.

    claims: iterable of (s1_id, candidate_id, prob). Returns {(s1, cand)}.
    """
    claims = list(claims)
    best = {}
    for s1, cand, p in claims:
        cur = best.get(s1)
        if cur is None or p > cur[0]:
            best[s1] = (p, cand)
    kept = set()
    used = {}
    for s1, (p, cand) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        if cand not in used:
            used[cand] = s1
            kept.add((s1, cand))
    rest = [(p, s1, cand) for s1, cand, p in claims
            if cand != best[s1][1] and (s1, cand) not in kept]
    for p, s1, cand in sorted(rest, key=lambda t: (-t[0], t[1], t[2])):
        if cand not in used:
            used[cand] = s1
            kept.add((s1, cand))
    return kept


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


def per_entity_f05(P, T, true_total, tau_singleton=0.68, tau_match=0.60,
                   gamma_rel=0.70, max_matches=12):
    """Per-entity F0.5 under the selection rule. Returns (f, kept, order, masked, hits).

    ``kept[i, j]`` / ``order[i, j]`` / ``masked[i, j]`` describe the j-th kept
    candidate of entity i (prob-desc), aligned with the padded candidate matrix
    — exactly what ``apply_one_to_one_f05`` needs to resolve claims globally.
    """
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
    return f, kept, order, masked, hits


def evaluate_selection(P, T, true_total, tau_singleton=0.68, tau_match=0.60,
                       gamma_rel=0.70, max_matches=12):
    """P/T padded with ABSENT/False. Macro F0.5 over ALL entities."""
    f, kept, _order, _masked, hits = per_entity_f05(P, T, true_total,
                                                    tau_singleton, tau_match,
                                                    gamma_rel, max_matches)
    return float(f.mean()), kept.sum(axis=1), hits


def apply_one_to_one_f05(P, T, true_total, cmat, s1_ids, tau_singleton=0.68,
                         tau_match=0.60, gamma_rel=0.70, max_matches=12):
    """Macro F0.5 after global one-to-one resolution of the selected claims.

    ``cmat`` is the padded candidate-id matrix aligned with P/T (same layout
    ``padded_val`` produces). Applies the identical selection rule, then drops
    contested claims via ``resolve_one_to_one`` before scoring.
    """
    f, kept, order, masked, _hits = per_entity_f05(P, T, true_total,
                                                   tau_singleton, tau_match,
                                                   gamma_rel, max_matches)
    n, cap = kept.shape
    claims, col_of = [], []
    for i in range(n):
        for j in range(cap):
            if kept[i, j]:
                claims.append((s1_ids[i], cmat[i, order[i, j]], float(masked[i, j])))
                col_of.append((i, order[i, j]))
    kept_set = resolve_one_to_one(claims)
    pred_counts = np.zeros(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int64)
    for idx, (i, c) in enumerate(col_of):
        if claims[idx][:2] in kept_set:
            pred_counts[i] += 1
            if T[i, c]:
                hits[i] += 1
    f2 = _f_from_counts(pred_counts, hits, true_total)
    return float(f2.mean()), int(len(claims)), int(len(kept_set))


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
