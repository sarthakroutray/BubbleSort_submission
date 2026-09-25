"""Macro F0.5 scorer. Empty/empty=1.0, empty/non-empty=0.0. Sanity: P=2/3,R=1→0.714."""


def f05_entity(pred, true):
    pred, true = set(pred), set(true)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    inter = len(pred & true)
    p = inter / len(pred)
    r = inter / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    return sum(f05_entity(pred_map.get(k, []), true_map.get(k, [])) for k in true_map) / len(true_map)
