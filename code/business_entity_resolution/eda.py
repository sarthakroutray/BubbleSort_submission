"""First-contact EDA: counts, singletons, missingness, scripts, country rule.

Stdlib-only, streaming (safe on 16 GB RAM). Verifies every data claim quoted in
docs/FINAL_RESEARCH_AND_IMPLEMENTATION_PLAN.md §3. Run:

    python3 scripts/eda.py <dataset_dir>     # dir containing train/ and test/

Prints a JSON report: row counts, ID integrity, singleton rate, matches/entity
histogram, missingness (overall + among GT matches), non-Latin/ASCII-strip trap,
country distributions, same-country pair rate, GT one-to-one check.
"""
import csv
import json
import re
import sys
from collections import Counter

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
ID_RE = re.compile(r"^S([123])-(\d+)$")


def enc_id(m):
    return int(m.group(1)) * 10**13 + int(m.group(2))


def open_tsv(path):
    return open(path, newline="", encoding="utf-8", errors="replace")


def is_non_latin(s):
    return any(ord(c) > 127 for c in s)


def ascii_stripped_empty(s):
    return s.encode("ascii", "ignore").decode().strip() == ""


def scan_source(path, matched_set=None, matched_country_sets=None):
    r = {
        "rows": 0, "bad_id": 0, "dup_id": 0, "digit_width": [10**9, 0],
        "missing_name": 0, "missing_addr": 0,
        "nonlatin_name": 0, "nonlatin_name_ascii_stripped_empty": 0,
        "nonlatin_addr": 0,
        "countries": Counter(),
    }
    if matched_set is not None:
        r["matched"] = {"rows": 0, "missing_addr": 0, "nonlatin_name": 0,
                        "missing_name": 0, "countries": Counter()}
    seen = set()
    with open_tsv(path) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            r["rows"] += 1
            m = ID_RE.match(eid)
            if not m:
                r["bad_id"] += 1
                continue
            n = enc_id(m)
            w = len(m.group(2))
            r["digit_width"][0] = min(r["digit_width"][0], w)
            r["digit_width"][1] = max(r["digit_width"][1], w)
            if n in seen:
                r["dup_id"] += 1
            seen.add(n)
            if not name.strip():
                r["missing_name"] += 1
            if not addr.strip():
                r["missing_addr"] += 1
            nl = is_non_latin(name)
            if nl:
                r["nonlatin_name"] += 1
                if ascii_stripped_empty(name):
                    r["nonlatin_name_ascii_stripped_empty"] += 1
            if is_non_latin(addr):
                r["nonlatin_addr"] += 1
            r["countries"][country] += 1
            if matched_set is not None and n in matched_set:
                mr = r["matched"]
                mr["rows"] += 1
                if not addr.strip():
                    mr["missing_addr"] += 1
                if not name.strip():
                    mr["missing_name"] += 1
                if nl:
                    mr["nonlatin_name"] += 1
                mr["countries"][country] += 1
                if matched_country_sets is not None:
                    matched_country_sets.setdefault(country, set()).add(n)
    del seen
    return r


def main():
    out = {}
    gt = f"{DATA}/train/train_ground_truth.tsv"

    # ---- Pass 1: ground truth ----
    p1 = {
        "rows": 0, "singletons": 0, "total_pairs": 0,
        "list_size_hist": Counter(), "max_list": 0,
        "s1_dup_rows": 0, "bad_s1_id": 0, "bad_match_id": 0,
        "s1_self_match": 0, "dup_within_list": 0,
        "s2_in_list": 0, "s3_in_list": 0,
        "digit_width_s1": [10**9, 0], "digit_width_m": [10**9, 0],
    }
    s1_ids = set()
    matched_set = set()
    matched_dup_across_rows = 0
    with open_tsv(gt) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) < 2:
                row = row + [""]
            sid, lst = row[0], row[1]
            p1["rows"] += 1
            m = ID_RE.match(sid)
            if not m:
                p1["bad_s1_id"] += 1
                continue
            n = enc_id(m)
            p1["digit_width_s1"][0] = min(p1["digit_width_s1"][0], len(m.group(2)))
            p1["digit_width_s1"][1] = max(p1["digit_width_s1"][1], len(m.group(2)))
            if n in s1_ids:
                p1["s1_dup_rows"] += 1
            s1_ids.add(n)
            ids = [x for x in lst.split(",") if x]
            k = len(ids)
            p1["list_size_hist"][k] += 1
            p1["max_list"] = max(p1["max_list"], k)
            p1["total_pairs"] += k
            if k == 0:
                p1["singletons"] += 1
            seen_local = set()
            for x in ids:
                mm = ID_RE.match(x)
                if not mm:
                    p1["bad_match_id"] += 1
                    continue
                if mm.group(1) == "1":
                    p1["s1_self_match"] += 1
                    continue
                if mm.group(1) == "2":
                    p1["s2_in_list"] += 1
                else:
                    p1["s3_in_list"] += 1
                p1["digit_width_m"][0] = min(p1["digit_width_m"][0], len(mm.group(2)))
                p1["digit_width_m"][1] = max(p1["digit_width_m"][1], len(mm.group(2)))
                mi = enc_id(mm)
                if mi in seen_local:
                    p1["dup_within_list"] += 1
                seen_local.add(mi)
                if mi in matched_set:
                    matched_dup_across_rows += 1
                matched_set.add(mi)
    p1["distinct_matched_ids"] = len(matched_set)
    p1["matched_id_claimed_twice"] = matched_dup_across_rows
    p1["mean_matches_per_entity"] = round(p1["total_pairs"] / max(p1["rows"], 1), 3)
    p1["singleton_rate"] = round(p1["singletons"] / max(p1["rows"], 1), 4)
    p1["list_size_hist"] = {str(k): v for k, v in sorted(p1["list_size_hist"].items())}
    out["train_ground_truth"] = p1
    out["train_s1_ids_in_gt"] = len(s1_ids)

    matched_country_sets = {"US": set(), "India": set(), "France": set()}

    # ---- Pass 2-3: train sources ----
    out["train_source1"] = scan_source(f"{DATA}/train/train_source1.tsv")
    out["train_source2"] = scan_source(f"{DATA}/train/train_source2.tsv",
                                       matched_set, matched_country_sets)
    out["train_source3"] = scan_source(f"{DATA}/train/train_source3.tsv",
                                       matched_set, matched_country_sets)

    # ---- Pass 4: same-country check on train pairs ----
    s1_country = {}
    with open_tsv(f"{DATA}/train/train_source1.tsv") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            m = ID_RE.match(row[0])
            if m:
                s1_country[enc_id(m)] = row[3] if len(row) > 3 else ""
    same = cross = unknown_match = unknown_s1 = 0
    cross_examples = []
    with open_tsv(gt) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) < 2:
                row = row + [""]
            m = ID_RE.match(row[0])
            if not m:
                continue
            sc = s1_country.get(enc_id(m))
            if sc is None:
                unknown_s1 += 1
                continue
            for x in row[1].split(","):
                if not x:
                    continue
                mm = ID_RE.match(x)
                if not mm:
                    continue
                mi = enc_id(mm)
                mc = None
                for c, s in matched_country_sets.items():
                    if mi in s:
                        mc = c
                        break
                if mc is None:
                    unknown_match += 1
                elif mc == sc:
                    same += 1
                else:
                    cross += 1
                    if len(cross_examples) < 5:
                        cross_examples.append([row[0], x, sc, mc])
    out["train_country_pairs"] = {
        "same": same, "cross": cross,
        "same_rate": round(same / max(same + cross, 1), 6),
        "unknown_match_id": unknown_match, "unknown_s1_id": unknown_s1,
        "cross_examples": cross_examples,
    }
    del s1_country, matched_set, matched_country_sets, s1_ids

    # ---- Pass 5: test sources ----
    out["test_source1"] = scan_source(f"{DATA}/test/test_source1.tsv")
    out["test_source2"] = scan_source(f"{DATA}/test/test_source2.tsv")
    out["test_source3"] = scan_source(f"{DATA}/test/test_source3.tsv")

    for k, v in out.items():
        if isinstance(v, dict) and "countries" in v:
            v["countries"] = dict(v["countries"].most_common())
            if "matched" in v:
                v["matched"]["countries"] = dict(v["matched"]["countries"].most_common())
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
