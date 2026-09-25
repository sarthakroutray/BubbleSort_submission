"""F0.5 selection: singleton gate + relative threshold, cap 10. Shared val/test."""


def select_matches(candidates, tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70,
                   max_matches=10):
    if not candidates:
        return []
    best = max(c["prob"] for c in candidates)
    if best < tau_singleton:
        return []
    thr = max(tau_match, best * gamma_rel)
    return [c["entity_id"] for c in candidates if c["prob"] >= thr][:max_matches]
