"""Pass 1: stream S2/S3 → FeatureHasher(2**22) chunks + per-country DF → weighted shards.

Design (plan §5.3, §7):
  * OR-semantics feature bag, prefixed: n_ b_ k_ d_ dw_ a_ ak_  (no concatenation feats).
  * Stable hashing of each feature string to 2**22 dims (crc32 — zero vocab RAM).
  * Block within country (measured: 100% of GT pairs are same-country; countries
    are derived from the data, never hard-coded).
  * Per-country IDF, drop DF > 0.1% within country, L2-normalize, shards <=2M rows.

Two streaming passes over the raw TSVs (cheap, cacheable):
  pass 1: dedup hashed features per chunk -> on-disk npz cache + per-country DF.
  pass 2: load cache -> apply IDF + L2 -> write <=2M-row shards.

Every stage is idempotent: an existing shard/meta is reused unless --force.
"""

import argparse
import binascii
import csv
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.cleaning.text_cleaner import (  # noqa: E402
    address_words,
    name_views,
    number_word_pairs,
    skeleton,
)

H = config.HASH_FEATURES
COMMON_DF_FRAC = 0.001  # drop features in >0.1% of a country's docs (plan §5.3)

CORPUS_FILES = {
    "train": ["train_source2.tsv", "train_source3.tsv"],
    "test": ["test_source2.tsv", "test_source3.tsv"],
}
CORPUS_DIRS = {"train": config.TRAIN_DIR, "test": config.TEST_DIR}


def corpus_index_dir(corpus: str) -> Path:
    return config.INDEX_DIR / corpus


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


@lru_cache(maxsize=1_500_000)
def hid(feature: str) -> int:
    """Stable feature-hash into [0, H)."""
    return binascii.crc32(feature.encode("utf-8")) % H


@lru_cache(maxsize=500_000)
def _name_features(name: str):
    # v1 bag (validated): adjacent bigrams only. Ordered-subset variants (v2-v5:
    # concat/c_/anchor bigrams) were all measured and REJECTED — they dilute the
    # top-K ordering (v5: recall@50 0.9425 -> 0.9072) for negligible deep gains.
    raw, _core = name_views(name)
    toks = raw.split()
    feats = []
    for t in toks:
        feats.append("n_" + t)
    for i in range(len(toks) - 1):
        feats.append("b_" + toks[i] + "_" + toks[i + 1])
    for t in toks:
        s = skeleton(t)
        if s:
            feats.append("k_" + s)
    return tuple(feats)


@lru_cache(maxsize=500_000)
def _addr_features(address: str):
    if not address:
        return ()
    words = address_words(address)
    feats = []
    for t in words:
        feats.append("a_" + t)
    for t in words:
        if t.isdigit():
            feats.append("d_" + t)
        s = skeleton(t)
        if s:
            feats.append("ak_" + s)
    for p in number_word_pairs(address):
        feats.append("dw_" + p)
    return tuple(feats)


def record_bag(name: str, address: str):
    return _name_features(name or "") + _addr_features(address or "")


def dedup_features(bags):
    """Hash a list of bags -> (doc_row, feature_idx, count) deduplicated per doc."""
    n = len(bags)
    lens = np.fromiter((len(b) for b in bags), dtype=np.int64, count=n)
    total = int(lens.sum())
    if total == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32))
    flat = np.fromiter((hid(f) for b in bags for f in b), dtype=np.int64, count=total)
    rows = np.repeat(np.arange(n, dtype=np.int64), lens)
    keys = rows * H + flat
    ku, kc = np.unique(keys, return_counts=True)
    return (ku // H, (ku % H).astype(np.int32), kc.astype(np.float32))


def csr_from(doc_rows, feat_idx, counts, nrows):
    per_row = np.bincount(doc_rows, minlength=nrows)
    indptr = np.empty(nrows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(per_row, out=indptr[1:])
    return sp.csr_matrix((counts.astype(np.float32), feat_idx, indptr),
                         shape=(nrows, H))


def stream_source(path: Path, usecols=("entity_id", "business_name", "business_address", "country")):
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         chunksize=config.CHUNK_SIZE, quoting=csv.QUOTE_MINIMAL,
                         usecols=list(usecols))
    for df in reader:
        yield tuple(df[c].to_numpy() for c in usecols)


def build(corpus: str, force: bool = False) -> Path:
    out = corpus_index_dir(corpus)
    meta_path = out / "meta.json"
    if meta_path.exists() and not force:
        print(f"[build_index] {meta_path} exists — skipping (use --force to rebuild)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "chunks"
    shard_dir = out / "shards"
    chunk_dir.mkdir(exist_ok=True)
    shard_dir.mkdir(exist_ok=True)

    files = [CORPUS_DIRS[corpus] / f for f in CORPUS_FILES[corpus]]
    df_counts: dict = {}
    chunk_records = []  # (chunk_path, country, nrows)
    t0 = time.time()
    total_docs = 0

    # ---------------- Pass 1: hashed features + DF ----------------
    for src in files:
        for ci, (ids, names, addrs, countries) in enumerate(stream_source(src)):
            bags = [record_bag(n, a) for n, a in zip(names, addrs)]
            d_u, f_u, counts = dedup_features(bags)
            total_docs += len(ids)
            ca = countries[d_u] if len(d_u) else np.zeros(0, dtype="<U16")
            for country in np.unique(countries):
                crow = countries == country
                nrows = int(crow.sum())
                remap = np.full(len(countries), -1, dtype=np.int64)
                remap[np.nonzero(crow)[0]] = np.arange(nrows)
                sel = ca == country
                cpath = chunk_dir / f"{src.stem}__{ci:05d}__{slug(str(country))}.npz"
                if not cpath.exists() or force:
                    row_index = remap[d_u[sel]] if len(d_u) else np.zeros(0, np.int64)
                    np.savez(cpath, f=f_u[sel], counts=counts[sel],
                             row_index=row_index, ids=ids[crow],
                             names=names[crow], addrs=addrs[crow],
                             nrows=np.int64(nrows))
                d = df_counts.get(country)
                if d is None:
                    d = np.zeros(H, dtype=np.int64)
                    df_counts[country] = d
                d += np.bincount(f_u[sel], minlength=H)
                chunk_records.append((cpath, country, nrows))
            print(f"[build_index] {src.stem} chunk {ci}: {len(ids):,} docs "
                  f"({total_docs:,} total, {time.time()-t0:.0f}s)", flush=True)

    # ---------------- IDF ----------------
    idfs = {}
    for country, d in df_counts.items():
        n = int(sum(r[2] for r in chunk_records if r[1] == country))
        idf = (np.log((n + 1.0) / (d + 1.0)) + 1.0).astype(np.float32)
        idf[d > COMMON_DF_FRAC * n] = 0.0
        if force or not (out / f"idf_{slug(str(country))}.npy").exists():
            np.save(out / f"idf_{slug(str(country))}.npy", idf)
        idfs[country] = idf
        print(f"[build_index] country {country!r}: {n:,} docs, "
              f"{int((idf > 0).sum()):,} kept features (DF<={COMMON_DF_FRAC:.3%})", flush=True)

    # ---------------- Pass 2: weight + L2 + shards ----------------
    countries = [c for c in df_counts]
    for country in countries:
        idf = idfs[country]
        blocks, block_rows, shard_i = [], 0, 0
        cc = [r for r in chunk_records if r[1] == country]
        cc.sort(key=lambda r: str(r[0]))
        for cpath, _c, _n in cc:
            with np.load(cpath, allow_pickle=True) as z:
                f_u, counts, row_index = z["f"], z["counts"], z["row_index"]
                ids, names, addrs = z["ids"], z["names"], z["addrs"]
                nrows = int(z["nrows"])
            if nrows == 0:
                continue
            data = counts * idf[f_u]
            m = csr_from(row_index, f_u, data, nrows)
            m.eliminate_zeros()
            norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            m = sp.diags((1.0 / norms).astype(np.float32)) @ m
            m = m.tocsr().astype(np.float32)
            m.sum_duplicates()
            blocks.append((m, ids, names, addrs))
            block_rows += nrows
            if block_rows >= config.INDEX_SHARD_SIZE:
                shard_i = _flush_shard(shard_dir, country, blocks, shard_i)
                blocks, block_rows = [], 0
        if blocks:
            shard_i = _flush_shard(shard_dir, country, blocks, shard_i)
        print(f"[build_index] country {country!r}: {shard_i} shards written "
              f"({time.time()-t0:.0f}s)", flush=True)

    shards = {}
    for p in sorted(shard_dir.glob("*.npz")):
        country = p.name.split("__")[0]
        shards.setdefault(country, []).append(p.name)
    meta = {
        "corpus": corpus,
        "H": H,
        "chunk_size": config.CHUNK_SIZE,
        "shard_size": config.INDEX_SHARD_SIZE,
        "common_df_frac": COMMON_DF_FRAC,
        "files": [str(f) for f in files],
        "total_indexed_docs": total_docs,
        "countries": {
            str(c): {
                "n_docs": int(sum(r[2] for r in chunk_records if r[1] == c)),
                "n_features_kept": int((idfs[c] > 0).sum()),
                "shards": shards.get(slug(str(c)), []),
            }
            for c in countries
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[build_index] done: {total_docs:,} docs, "
          f"{sum(len(v) for v in shards.values())} shards, {time.time()-t0:.0f}s")
    return out


def _flush_shard(shard_dir: Path, country, blocks, shard_i) -> int:
    mats = [b[0] for b in blocks]
    ids = np.concatenate([np.asarray(b[1], dtype=object) for b in blocks])
    names = np.concatenate([np.asarray(b[2], dtype=object) for b in blocks])
    addrs = np.concatenate([np.asarray(b[3], dtype=object) for b in blocks])
    m = sp.vstack(mats, format="csr").astype(np.float32)
    m.sort_indices()
    base = f"{slug(str(country))}__{shard_i:04d}"
    sp.save_npz(shard_dir / f"{base}.npz", m, compressed=False)
    np.save(shard_dir / f"{base}.ids.npy", ids, allow_pickle=True)
    np.save(shard_dir / f"{base}.names.npy", names, allow_pickle=True)
    np.save(shard_dir / f"{base}.addrs.npy", addrs, allow_pickle=True)
    del mats, blocks[:]
    return shard_i + 1


def load_index(corpus: str):
    out = corpus_index_dir(corpus)
    meta = json.loads((out / "meta.json").read_text())
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["train", "test"], default="test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    build(a.corpus, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
