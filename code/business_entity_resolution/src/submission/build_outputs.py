"""Assemble candidate_pairs.tsv + matching_results.tsv from batch checkpoints.

Also provides the format-exact empty baseline (every test S1 -> empty list),
which pins the output contract before any modelling (plan §5.1, §11 item 3).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_s1_ids(test_source1: Path) -> list[str]:
    ids = []
    with open(test_source1, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if row:
                ids.append(row[0])
    return ids


def _join(ids) -> str:
    return ",".join(dict.fromkeys(ids))


def write_pairs(path: Path, rows, header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for sid, ids in rows:
            w.writerow([sid, _join(ids)])
    tmp.replace(path)


def write_empty_baseline(test_dir: Path = None, output_dir: Path = None) -> None:
    test_dir = Path(test_dir or config.TEST_DIR)
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    s1 = read_s1_ids(test_dir / "test_source1.tsv")
    write_pairs(output_dir / "matching_results.tsv", ((s, []) for s in s1), MATCH_HEADER)
    write_pairs(output_dir / "candidate_pairs.tsv", ((s, []) for s in s1), CAND_HEADER)
    print(f"wrote empty baseline for {len(s1):,} test S1 rows -> {output_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--empty", action="store_true",
                    help="write the format-exact all-empty baseline")
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.empty:
        write_empty_baseline(a.test_dir, a.output_dir)
        return 0
    ap.error("nothing to do; pass --empty")


if __name__ == "__main__":
    sys.exit(main())
