"""Build pseudo-labels from a previous inference run's per-country checkpoints.

France (15% of test) is unseen in training; its only labeled signal is the
model's own high-confidence test predictions. This reads the checkpoint TSVs
written by ``predict_test`` (s1_id, candidates, matches, probs) and writes a
pseudo-label TSV of (s1_id, candidate_id, prob) rows with prob >= --min-prob,
which ``build_training_set --pseudo`` consumes as extra training truth.

Only countries NOT present in the train corpus are used by default (train has
India/US, so this picks up France) — pass --countries to override.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402

TRAIN_COUNTRIES = {"india", "us"}


def _slug_country(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def build(ckpt_dir: Path, out: Path, min_prob: float, countries=None):
    ckpt_files = sorted(ckpt_dir.glob("*.rows.tsv"))
    if not ckpt_files:
        raise FileNotFoundError(f"no *.rows.tsv checkpoints under {ckpt_dir}")
    picked = []
    for f in ckpt_files:
        c = _slug_country(f.stem.replace(".rows", ""))
        if countries is not None:
            if c in countries:
                picked.append(f)
        elif c not in TRAIN_COUNTRIES:
            picked.append(f)
    if not picked:
        raise FileNotFoundError(
            f"no candidate checkpoint countries under {ckpt_dir} "
            f"(found: {[f.stem for f in ckpt_files]})")
    print(f"[pseudo] using checkpoints: {[f.name for f in picked]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    n_rows, n_pairs, seen = 0, 0, set()
    with open(tmp, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_id", "prob"])
        for f in picked:
            with open(f, newline="", encoding="utf-8") as fi:
                for row in csv.reader(fi, delimiter="\t"):
                    if len(row) < 4 or not row[2]:
                        continue
                    mids = row[2].split(",")
                    probs = [float(x) for x in row[3].split(",") if x]
                    if len(probs) != len(mids):
                        continue
                    for c, p in zip(mids, probs):
                        if p >= min_prob and (row[0], c) not in seen:
                            seen.add((row[0], c))
                            w.writerow([row[0], c, f"{p:.6f}"])
                            n_pairs += 1
            n_rows += 1
            print(f"[pseudo] {f.name}: cumulative {n_pairs:,} pairs", flush=True)
    tmp.replace(out)
    s1s = len({s for s, _ in seen})
    print(f"[pseudo] wrote {n_pairs:,} pseudo pairs over {s1s:,} S1 -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, default=config.BATCH_DIR)
    ap.add_argument("--out", type=Path, default=config.CACHE_DIR / "pseudo_labels.tsv")
    ap.add_argument("--min-prob", type=float, default=0.85,
                    help="keep only matches with prob >= this (precision first: "
                         "F0.5 punishes false pseudo-labels 4x)")
    ap.add_argument("--countries", default=None,
                    help="comma list to restrict, e.g. 'france'; default = every "
                         "checkpoint country not in the train corpus")
    a = ap.parse_args(argv)
    countries = {c.strip().lower() for c in a.countries.split(",")} \
        if a.countries else None
    build(a.ckpt_dir, a.out, a.min_prob, countries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
