#!/usr/bin/env python3
"""Format validator for matching_results.tsv + candidate_pairs.tsv. Stdlib only.

Checks every rule in the problem statement; prints PASS (exit 0) or numbered
issues (exit 1). Scores nothing.

Usage:
    python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
"""

import argparse
import csv
import sys
from pathlib import Path


def read_ids(path, list_col):
    rows = {}
    dup_rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", list_col]:
            return None, [f"bad header {reader.fieldnames}, expected ['source1_entity_id', '{list_col}']"], []
        for i, row in enumerate(reader, start=2):
            sid = row["source1_entity_id"]
            if sid in rows:
                dup_rows.append(sid)
            raw = row[list_col].strip()
            ids = [x for x in raw.split(",") if x] if raw else []
            rows[sid] = ids
    return rows, [], dup_rows


def load_source_ids(test_dir):
    valid = set()
    s1 = set()
    for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        p = test_dir / name
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                valid.add(eid)
                if eid.startswith("S1-"):
                    s1.add(eid)
    return valid, s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    issues = []

    test_dir = Path(a.test_dir)
    try:
        valid_ids, test_s1 = load_source_ids(test_dir)
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        return 1

    match, herr, hdup = read_ids(a.matching, "matched_entity_ids")
    if match is None:
        issues.extend(herr)
        match = {}
    issues.extend(f"matching: duplicate row for {d}" for d in hdup)

    cand, cerr, cdup = read_ids(a.candidate, "candidate_entity_ids")
    if cand is None:
        issues.extend(cerr)
        cand = {}
    issues.extend(f"candidate: duplicate row for {d}" for d in cdup)

    for sid in sorted(test_s1 - set(match)):
        issues.append(f"matching: missing row for {sid}")
    for sid in sorted(set(match) - test_s1):
        issues.append(f"matching: unknown S1 row {sid}")
    for sid in sorted(test_s1 - set(cand)):
        issues.append(f"candidate: missing row for {sid}")
    for sid in sorted(set(cand) - test_s1):
        issues.append(f"candidate: unknown S1 row {sid}")

    for label, table in (("matching", match), ("candidate", cand)):
        for sid, ids in table.items():
            if len(ids) != len(set(ids)):
                issues.append(f"{label}: duplicate IDs in list for {sid}")
            for x in ids:
                if x not in valid_ids:
                    issues.append(f"{label}: {sid} references unknown ID {x}")
                elif x.startswith("S1-"):
                    issues.append(f"{label}: {sid} self-matches S1 ID {x}")

    for sid, ids in match.items():
        if sid in cand and set(ids) - set(cand[sid]):
            issues.append(f"warning: {sid} has matches not present in candidates")

    if issues:
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
