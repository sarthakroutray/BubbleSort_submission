"""~39 pair features per (s1, s2|s3) via rapidfuzz; float32 arrays.

Spec: FINAL_RESEARCH_AND_IMPLEMENTATION_PLAN.md §5.4. ``country`` is deliberately
excluded (overfits {US, India}, zero observed gain). Similarities are computed on
both the raw and the legal-suffix-stripped ("core") name views, and on phonetic
skeletons. Numbers get their own block including ``num_conflict_flag``.

Parallelism (plan §5.4): workers receive only their own chunk of record tuples
(no shared big dicts — fork+read stays CoW-friendly) and return float32 arrays.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning import text_cleaner as tc  # noqa: E402

FEATURE_NAMES = [
    "ret_score", "ret_rank", "ret_score_over_best", "ret_gap_best", "ret_gap_next",
    "name_ratio", "name_token_sort", "name_token_set", "name_partial", "name_wratio",
    "name_jaro",
    "core_ratio", "core_token_sort", "core_token_set", "core_partial", "core_wratio",
    "core_jaro", "core_token_jaccard",
    "core_concat_ratio", "core_len_diff",
    "skel_ratio", "skel_token_sort", "skel_token_set", "skel_partial",
    "addr_ratio", "addr_token_set", "addr_partial", "addr_token_jaccard",
    "addr_word_jaccard",
    "num_jaccard", "num_shared", "num_conflict", "num_longest_shared_len",
    "postcode_delta",
    "addr_empty_s1", "addr_empty_cand", "name_nonlatin_cand", "is_s3",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _postcode_delta(a, b) -> float:
    na = [int(x) for x in a if x.isdigit() and len(x) >= 4]
    nb = [int(x) for x in b if x.isdigit() and len(x) >= 4]
    if not na or not nb:
        return 0.0
    return 1.0 if min(abs(x - y) for x in na for y in nb) <= 1 else 0.0


def pair_vec(job):
    """job = (name1, addr1, name2, addr2, ctx) where ctx = dict-like tuple."""
    name1, addr1, name2, addr2, ctx = job
    score, rank, score_over_best, gap_best, gap_next, cand_id = ctx

    raw1, core1 = tc.name_views(name1)
    raw2, core2 = tc.name_views(name2)
    sk1 = tc.skeleton_text(name1)
    sk2 = tc.skeleton_text(name2)
    aw1 = set(tc.address_words(addr1))
    aw2 = set(tc.address_words(addr2))
    num1 = set(tc.numbers(addr1))
    num2 = set(tc.numbers(addr2))

    a = np.empty(len(FEATURE_NAMES), dtype=np.float32)
    a[0] = score
    a[1] = rank
    a[2] = score_over_best
    a[3] = gap_best
    a[4] = gap_next
    a[5] = fuzz.ratio(raw1, raw2) / 100.0
    a[6] = fuzz.token_sort_ratio(raw1, raw2) / 100.0
    a[7] = fuzz.token_set_ratio(raw1, raw2) / 100.0
    a[8] = fuzz.partial_ratio(raw1, raw2) / 100.0
    a[9] = fuzz.WRatio(raw1, raw2) / 100.0
    a[10] = JaroWinkler.similarity(raw1, raw2)
    a[11] = fuzz.ratio(core1, core2) / 100.0
    a[12] = fuzz.token_sort_ratio(core1, core2) / 100.0
    a[13] = fuzz.token_set_ratio(core1, core2) / 100.0
    a[14] = fuzz.partial_ratio(core1, core2) / 100.0
    a[15] = fuzz.WRatio(core1, core2) / 100.0
    a[16] = JaroWinkler.similarity(core1, core2)
    a[17] = _jaccard(set(core1.split()), set(core2.split()))
    a[18] = fuzz.ratio(core1.replace(" ", ""), core2.replace(" ", "")) / 100.0
    a[19] = abs(len(core1) - len(core2))
    a[20] = fuzz.ratio(sk1, sk2) / 100.0
    a[21] = fuzz.token_sort_ratio(sk1, sk2) / 100.0
    a[22] = fuzz.token_set_ratio(sk1, sk2) / 100.0
    a[23] = fuzz.partial_ratio(sk1, sk2) / 100.0
    a[24] = fuzz.ratio(addr1, addr2) / 100.0
    a[25] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    a[26] = fuzz.partial_ratio(addr1, addr2) / 100.0
    a[27] = _jaccard(set(addr1.split()), set(addr2.split()))
    a[28] = _jaccard(aw1, aw2)
    a[29] = _jaccard(num1, num2)
    a[30] = float(len(num1 & num2))
    a[31] = 1.0 if (num1 and num2 and not (num1 & num2)) else 0.0
    a[32] = float(max((len(x) for x in (num1 & num2)), default=0))
    a[33] = _postcode_delta(addr1.split(), addr2.split())
    a[34] = 1.0 if not addr1.strip() else 0.0
    a[35] = 1.0 if not addr2.strip() else 0.0
    a[36] = 1.0 if any(ord(c) > 127 for c in name2) else 0.0
    a[37] = 1.0 if cand_id.startswith("S3-") else 0.0
    return a


def _worker(jobs):
    if not jobs:
        return None
    return np.vstack([pair_vec(j) for j in jobs])


def _waves(jobs, per_wave):
    for i in range(0, len(jobs), per_wave):
        yield jobs[i:i + per_wave]


def build_matrix(jobs, n_jobs=1, per_wave=50_000):
    """jobs: list of tuples. Returns float32 [n, len(FEATURE_NAMES)]."""
    if not jobs:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    if n_jobs > 1 and len(jobs) > per_wave:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            mats = [m for m in pool.imap(_worker, _waves(jobs, per_wave), chunksize=1)
                    if m is not None]
    else:
        mats = [m for m in (_worker(w) for w in _waves(jobs, per_wave)) if m is not None]
    return np.vstack(mats).astype(np.float32)
