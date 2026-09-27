"""Pass 2: sparse dot-product top-Kmax per country shard → adaptive prune → candidates.

Block within country (countries derived from data). Queries and indexed docs are
weighted with the same per-country IDF and L2-normalized, so cosine similarity is
a plain sparse dot product. ``awesome_cossim_topn`` (sparse-dot-topn 1.2.0) does the
top-K matmul; shards are visited one at a time to stay inside the RAM budget.

Two entry points:
  * ``retrieve``          -> in-memory dict of raw top-K, for small samples/eval.
  * ``retrieve_to_tsv``   -> streaming, memory-bounded, writes candidate_pairs.tsv.
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.blocking import build_index as bi  # noqa: E402

warnings.filterwarnings("ignore", message=".*awesome_cossim_topn.*")

S1_FILES = {"train": "train_source1.tsv", "test": "test_source1.tsv"}
S1_DIRS = {"train": config.TRAIN_DIR, "test": config.TEST_DIR}


def slug(name: str) -> str:
    return bi.slug(name)


# ---------------------------------------------------------------- pruning
def adaptive_prune(scores, ids, s_hi=0.85, s_lo=0.15, floors=(0.28, 0.17),
                   caps=(30, 45), singleton_keep=3):
    """Adaptive prune (plan §2 steps 2-4), floors/caps recalibrated on measured data.

    The plan's prior-log floors (0.50/0.35, caps 10/30) were measured to lose ~7 pts
    of recall (0.8734 at 11.6 cand/S1) on this pipeline. A local sweep over 40K train
    S1 picked (0.28/0.17, caps 30/45): pruned recall 0.9355 vs the 0.9436 K=50
    ceiling, i.e. <1 pt lost, at ~30.8 candidates/S1 (§3.5).
    """
    n = len(scores)
    if n == 0 or scores[0] <= 0:
        return [], []
    if scores[0] >= s_lo:
        if scores[0] > s_hi:
            thr, cap = floors[0] * scores[0], caps[0]
        else:
            thr, cap = floors[1] * scores[0], caps[1]
        keep = np.nonzero(scores >= thr)[0][:cap]
    else:
        keep = np.arange(min(singleton_keep, n))
    return [float(scores[i]) for i in keep], [str(ids[i]) for i in keep]


# ---------------------------------------------------------------- queries
def load_s1(corpus: str, subset: set = None):
    """Return (ids, names, addrs, countries, positions) as object arrays."""
    path = S1_DIRS[corpus] / S1_FILES[corpus]
    ids, names, addrs, countries = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for i, row in enumerate(rd):
            if subset is not None and row[0] not in subset:
                continue
            ids.append(row[0])
            names.append(row[1])
            addrs.append(row[2])
            countries.append(row[3])
    pos = np.arange(len(ids), dtype=np.int64)
    return (np.array(ids, dtype=object), np.array(names, dtype=object),
            np.array(addrs, dtype=object), np.array(countries, dtype=object), pos)


def country_queries(store, mask, idf):
    names, addrs = store[1][mask], store[2][mask]
    bags = [bi.record_bag(n, a) for n, a in zip(names, addrs)]
    d_u, f_u, counts = bi.dedup_features(bags)
    m = bi.csr_from(d_u, f_u, counts * idf[f_u], len(bags))
    m.eliminate_zeros()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags((1.0 / norms).astype(np.float32)) @ m).tocsr().astype(np.float32)


def _densify_pad(res, nq, K):
    res = res.tocsr()
    vals = np.full((nq, K), -1.0, dtype=np.float32)
    cols = np.full((nq, K), -1, dtype=np.int64)
    if res.nnz == 0:
        return vals, cols
    lens = np.diff(res.indptr)
    rows = np.repeat(np.arange(nq), lens)
    pos = np.arange(res.nnz) - np.repeat(res.indptr[:-1], lens)
    vals[rows, pos] = res.data
    cols[rows, pos] = res.indices
    return vals, cols


def _merge_topk(vals, cols, best_v, best_c, K):
    av = np.concatenate([best_v, vals], axis=1)
    ac = np.concatenate([best_c, cols], axis=1)
    idx = np.argpartition(-av, K - 1, axis=1)[:, :K]
    return np.take_along_axis(av, idx, axis=1), np.take_along_axis(ac, idx, axis=1)


def _order(best_v, best_c):
    order = np.argsort(-best_v, axis=1)
    return np.take_along_axis(best_v, order, axis=1), np.take_along_axis(best_c, order, axis=1)


def bi_topn(q, B, kmax, n_jobs):
    from sparse_dot_topn import awesome_cossim_topn
    return awesome_cossim_topn(q, B, kmax, lower_bound=0.0,
                               use_threads=True, n_jobs=n_jobs)


def _shard_path(shard_dir, name, suffix):
    return shard_dir / name.replace(".npz", suffix)


def retrieve_country(idx_dir, meta, store, country, kmax, n_jobs=16, with_records=False):
    """Return (positions, scores[n,K], entity_ids[n,K], names[n,K], addrs[n,K]).

    Two passes over the country's shards keep resident memory bounded: pass A does
    the top-K matmul (sparse matrix only), pass B re-reads one shard's id/name/address
    arrays at a time to materialise the winners. The full-country record arrays are
    never held simultaneously (matters for India: 4.7M docs).
    """
    mask = np.asarray(store[3] == country)
    pos = store[4][mask]
    if len(pos) == 0:
        empty = np.zeros((0, kmax), object)
        return pos, np.zeros((0, kmax), np.float32), empty, empty, empty
    idf = np.load(idx_dir / f"idf_{slug(str(country))}.npy")
    q = country_queries(store, mask, idf)
    nq = q.shape[0]
    best_v = np.full((nq, kmax), -1.0, np.float32)
    best_c = np.full((nq, kmax), -1, np.int64)
    shard_dir = idx_dir / "shards"
    offsets = []
    offset = 0
    for name in meta["countries"][str(country)]["shards"]:
        B = sp.load_npz(shard_dir / name).tocsr()
        res = bi_topn(q, B, kmax, n_jobs)
        vals, cols = _densify_pad(res, nq, kmax)
        cols = np.where(cols >= 0, cols + offset, cols)
        best_v, best_c = _merge_topk(vals, cols, best_v, best_c, kmax)
        offsets.append((name, offset))
        offset += B.shape[0]
        del B, res
    best_v, best_c = _order(best_v, best_c)

    cid = np.full((nq, kmax), "", object)
    cn = np.full((nq, kmax), "", object) if with_records else None
    ca = np.full((nq, kmax), "", object) if with_records else None
    for name, off in offsets:
        ids = np.load(_shard_path(shard_dir, name, ".ids.npy"), allow_pickle=True)
        sel = (best_c >= off) & (best_c < off + len(ids))
        if not sel.any():
            del ids
            continue
        local = best_c[sel] - off
        cid[sel] = ids[local]
        if with_records:
            names = np.load(_shard_path(shard_dir, name, ".names.npy"), allow_pickle=True)
            addrs = np.load(_shard_path(shard_dir, name, ".addrs.npy"), allow_pickle=True)
            cn[sel] = names[local]
            ca[sel] = addrs[local]
            del names, addrs
        del ids
    del q
    cid = np.where(best_c >= 0, cid, "")
    if with_records:
        cn = np.where(best_c >= 0, cn, "")
        ca = np.where(best_c >= 0, ca, "")
    return pos, best_v, cid, cn, ca


# ---------------------------------------------------------------- public API
def retrieve(corpus: str, kmax: int = 50, subset: set = None, n_jobs: int = 16,
             verbose=True, prune: bool = False, with_records: bool = False):
    """Top-Kmax per S1. Returns dict s1_id -> [(score, cand_id[, name, addr]), ...].

    ``prune=True`` applies the §2 adaptive prune per row on the fly; ``with_records``
    attaches the candidate name/address (read from the index shards).
    """
    idx_dir, meta = bi.load_index(corpus)
    store = load_s1(corpus, subset)
    results = {}
    t0 = time.time()
    for country in sorted(meta["countries"].keys()):
        pos, scores, cand_ids, cnames, caddrs = retrieve_country(
            idx_dir, meta, store, country, kmax, n_jobs, with_records)
        for i in range(len(pos)):
            if prune:
                s, ids = adaptive_prune(scores[i], cand_ids[i])
                if with_records:
                    rank = {c: k for k, c in enumerate(cand_ids[i])}
                    results[str(store[0][pos[i]])] = [
                        (sc, c, str(cnames[i][rank[c]]), str(caddrs[i][rank[c]]))
                        for sc, c in zip(s, ids)]
                else:
                    results[str(store[0][pos[i]])] = list(zip(s, ids))
            else:
                row = []
                for j in range(scores.shape[1]):
                    if scores[i, j] > 0 and cand_ids[i, j]:
                        if with_records:
                            row.append((float(scores[i, j]), str(cand_ids[i, j]),
                                        str(cnames[i, j]), str(caddrs[i, j])))
                        else:
                            row.append((float(scores[i, j]), str(cand_ids[i, j])))
                results[str(store[0][pos[i]])] = row
        if verbose:
            print(f"[retrieve] {country}: {len(pos):,} queries, {time.time()-t0:.0f}s", flush=True)
    return results


def prune_results(raw):
    out = {}
    for sid, row in raw.items():
        scores = np.array([r[0] for r in row], dtype=np.float32)
        ids = np.array([r[1] for r in row], dtype=object)
        out[sid] = list(zip(*adaptive_prune(scores, ids)))
    return out


# --------------------------------------------- lean, shard-at-a-time inference
def shard_layout(idx_dir, meta, country):
    """[(shard_name, global_offset, n_rows)] for one country, no matrix load."""
    shard_dir = idx_dir / "shards"
    layout, offset = [], 0
    for name in meta["countries"][str(country)]["shards"]:
        with np.load(shard_dir / name) as z:
            rows = int(z["indptr"].shape[0] - 1)
        layout.append((name, offset, rows))
        offset += rows
    return layout


def load_s1_country(corpus: str, country: str, limit: int = 0):
    """Only one country's S1 (id, name, address) — keeps peak RAM low."""
    path = S1_DIRS[corpus] / S1_FILES[corpus]
    ids, names, addrs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for row in rd:
            if row[3] == country:
                ids.append(row[0])
                names.append(row[1])
                addrs.append(row[2])
                if limit and len(ids) >= limit:
                    break
    return (np.array(ids, dtype=object), np.array(names, dtype=object),
            np.array(addrs, dtype=object))


def build_queries(names, addrs, idf, chunk=100_000):
    """L2-normalised IDF-weighted query matrix, built in bounded chunks."""
    mats = []
    for s in range(0, len(names), chunk):
        nms, ads = names[s:s + chunk], addrs[s:s + chunk]
        bags = [bi.record_bag(n, a) for n, a in zip(nms, ads)]
        d_u, f_u, counts = bi.dedup_features(bags)
        mats.append(bi.csr_from(d_u, f_u, counts * idf[f_u], len(bags)))
        del bags
    m = sp.vstack(mats, format="csr").astype(np.float32) if len(mats) > 1 else mats[0]
    m.eliminate_zeros()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags((1.0 / norms).astype(np.float32)) @ m).tocsr().astype(np.float32)


def topk_country(idx_dir, layout, q, kmax, n_jobs=16):
    """Top-Kmax across a country's shards; shards visited one at a time."""
    shard_dir = idx_dir / "shards"
    nq = q.shape[0]
    best_v = np.full((nq, kmax), -1.0, np.float32)
    best_c = np.full((nq, kmax), -1, np.int64)
    for name, off, _rows in layout:
        B = sp.load_npz(shard_dir / name).tocsr()
        res = bi_topn(q, B, kmax, n_jobs)
        vals, cols = _densify_pad(res, nq, kmax)
        cols = np.where(cols >= 0, cols + off, cols)
        best_v, best_c = _merge_topk(vals, cols, best_v, best_c, kmax)
        del B, res
    return _order(best_v, best_c)


def gather_records(idx_dir, layout, best_c, with_records=True):
    """Materialise ids/names/addrs for the winning columns, one shard at a time."""
    shard_dir = idx_dir / "shards"
    nq, kmax = best_c.shape
    cid = np.full((nq, kmax), "", object)
    cn = np.full((nq, kmax), "", object) if with_records else None
    ca = np.full((nq, kmax), "", object) if with_records else None
    for name, off, rows in layout:
        sel = (best_c >= off) & (best_c < off + rows)
        if not sel.any():
            continue
        local = best_c[sel] - off
        ids = np.load(_shard_path(shard_dir, name, ".ids.npy"), allow_pickle=True)
        cid[sel] = ids[local]
        del ids
        if with_records:
            names = np.load(_shard_path(shard_dir, name, ".names.npy"), allow_pickle=True)
            addrs = np.load(_shard_path(shard_dir, name, ".addrs.npy"), allow_pickle=True)
            cn[sel] = names[local]
            ca[sel] = addrs[local]
            del names, addrs
    cid = np.where(best_c >= 0, cid, "")
    if with_records:
        cn = np.where(best_c >= 0, cn, "")
        ca = np.where(best_c >= 0, ca, "")
    return cid, cn, ca


def retrieve_to_tsv(corpus: str, path: Path, kmax: int = 50, prune: bool = True,
                    n_jobs: int = 16):
    """Streaming, memory-bounded retrieval over ALL S1 -> candidate TSV."""
    idx_dir, meta = bi.load_index(corpus)
    store = load_s1(corpus)
    n = len(store[0])
    results = [""] * n
    t0 = time.time()
    for country in sorted(meta["countries"].keys()):
        pos, scores, cand_ids, _cn, _ca = retrieve_country(
            idx_dir, meta, store, country, kmax, n_jobs)
        for i in range(len(pos)):
            row_s, row_i = scores[i], cand_ids[i]
            if prune:
                _s, kept = adaptive_prune(row_s, row_i)
            else:
                kept = [str(row_i[j]) for j in range(len(row_s)) if row_s[j] > 0 and row_i[j]]
            results[pos[i]] = ",".join(dict.fromkeys(kept))
        print(f"[retrieve] {country}: {len(pos):,} queries, {time.time()-t0:.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for sid, s in zip(store[0], results):
            w.writerow([sid, s])
    tmp.replace(path)
    avg = sum(s.count(",") + 1 for s in results if s) / max(sum(1 for s in results if s), 1)
    print(f"[retrieve] wrote {path} — {n:,} rows, avg {avg:.1f} candidates/S1")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["train", "test"], default="test")
    ap.add_argument("--kmax", type=int, default=config.K_MAX)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.corpus == "test":
        retrieve_to_tsv("test", a.out or config.CANDIDATE_FILE, a.kmax, not a.no_prune)
    else:
        raw = retrieve("train", a.kmax)
        print(f"[retrieve] train: {len(raw)} S1, avg raw {sum(len(v) for v in raw.values())/max(len(raw),1):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
