# Methodology Documentation

Team: BubbleSort · Task: Amazon ML Challenge 2026 — Business Entity Resolution
Source of truth: `docs/FINAL_RESEARCH_AND_IMPLEMENTATION_PLAN.md` (dataset-verified v1.2).
All numbers below are **measured on the shipped data**; the sections that supersede the
prior-log figures are marked in the plan (§3.5, §3.6).

---

## 1. Methodology Used

The pipeline links every test `S1` record to the `S2`/`S3` records of the same
real-world business. It is a classical, offline, three-stage design:

```
dataset/*.tsv ──▶ 1. INGEST    chunked pd.read_csv(sep="\t", keep_default_na=False)
                  2. NORMALIZE anyascii → lower/fold → number norms → skeleton + core name
                  3. BLOCK     prefixed feature bag → FeatureHasher(2²²) → per-country IDF
                               → shards ≤2M rows → sparse top-K=50 → adaptive prune
                               → candidate_pairs.tsv
candidates ──▶    4. FEATURES  38 float32 features/pair (rapidfuzz C++)
                  5. TRAIN     LightGBM, S1-grouped 80/20 split
                  6. SELECT    singleton gate + relative threshold, cap 12
                  7. INFER     per-country / per-batch, checkpointed → matching_results.tsv
```

Design choices that were driven by measured properties of the data:

- **Country-agnostic throughout.** `country` is an open string set (train = US+India,
  test adds France at 15%). Country is used *only* to bucket blocking (measured: 100% of
  7,638,365 GT pairs are same-country), never as a filter, key or model feature. No
  `{US, India}` constant appears anywhere; France works day one.
- **Offline only.** No geocoding, registries, internet augmentation or external data.
  `anyascii`/`rapidfuzz`/LightGBM are local libraries.
- **Format first.** Both output TSVs were produced as a format-exact all-empty baseline
  and validated **before** any modelling, to pin the contract.

---

## 2. Candidate Generation / Blocking Strategy

**Feature bag (OR-semantics, `n_ b_ k_ d_ dw_ a_ ak_`).** Per record, order-independent:
name unigrams (`n_`), name bigrams (`b_`), phonetic consonant skeleton per name token
(`k_`), address numbers with leading zeros stripped (`d_`), house-number + next-word
(`dw_`, e.g. `203_26th`), canonicalised address words (`a_`), address-word skeletons
(`ak_`). No concatenation features (measured to cost recall in the prior log).

**Hashing.** Each feature string is hashed to `2²² = 4,194,304` dimensions with a stable
CRC32 (zero vocabulary RAM). A per-chunk `numpy` unique over `(row, feature)` keys gives
`tf` counts directly.

**Per-country IDF.** Corpus document frequency is accumulated per country (dense int64
`bincount`); weight `idf = log((N+1)/(df+1)) + 1`; features present in **> 0.1%** of a
country's documents are dropped (auto-suppresses `pvt/ltd/llc/rue/sarl/road` per
geography without hard-coding vocabularies). Vectors are L2-normalised, so cosine
similarity is a plain sparse dot product.

**Index.** S2/S3 are streamed in 250K chunks, deduplicated per chunk and cached to disk;
the weighted matrix is rebuilt from cache and written as ≤2M-row shards per country.
Train: 10,320,219 docs → 6 shards (518 s). Test: 9,969,589 docs → 6 shards (579 s).
Retrieval uses `sparse_dot_topn`'s `awesome_cossim_topn` (K=50) one shard at a time.

**Adaptive pruning** (plan §2), recalibrated on measured data:

| config | pruned recall | avg cand/S1 |
|---|---|---|
| prior-log floors (0.50/0.35, caps 10/30) | 0.8734 | 11.6 |
| floors 0.30/0.20, caps 20/40 | 0.9291 | 23.9 |
| **adopted: floors 0.28/0.17, caps 30/45** | **0.9355** | **30.8** |

**Measured recall@K** (40,000 random train S1, 137,937 true pairs, full train index):

| K | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| recall@K | 0.257 | 0.819 | 0.901 | 0.926 | **0.9436** | 0.9527 |

The prior-log floors lost ~7 points of recall here, so they were loosened to keep
pruned recall < 1 pt below the K=50 ceiling (0.9355). Candidate sets are therefore
larger than the 18–24/S1 efficiency target — a deliberate trade under a precision-heavy
metric (the plan's own guard: "raise the floors rather than accept the loss").

**Test volume:** 1,732,544 S1 → avg **30.21** candidates/S1; the pruned set *is*
`candidate_pairs.tsv`, so `matching_results ⊆ candidate_pairs` by construction.

---

## 3. Model Architecture and Feature Engineering

**38 pair features** per `(S1, S2|S3)` pair (all `float32`, excluded `country`):

- **Retrieval context (5):** cosine score, rank, score/best, gap-to-best, gap-to-next.
- **Name (11):** ratio, token-sort, token-set, partial, WRatio, Jaro-Winkler on the raw
  view, and the same on the legal-suffix-stripped **core** view, plus core token Jaccard,
  concatenated-core ratio (run-together domains) and core length diff.
- **Phonetic (4):** the rapidfuzz ratios on the consonant-skeleton strings.
- **Address (5):** ratio, token-set, partial, raw-token Jaccard, canonical-word Jaccard.
- **Numbers (5):** set Jaccard, shared count, **num_conflict_flag** (both have numbers,
  none shared → strong negative), longest shared number length, postcode ±1 flag.
- **Flags (4):** empty address on either side, non-Latin candidate name, is_S3.

Normalisation feeding these: `anyascii` transliteration (mandatory — 6.2% of train /
7.4% of test S2/S3 names vanish under naive ASCII strip), lowercasing, digit/letter
splitting, leading-zero and ordinal stripping, `0→o` between letters, legal-form core
view, and an `lru_cache`d phonetic skeleton (`tirupti`/`tirupati` → `trpt`).

**Training set.** 200,000 random train S1 (fixed seed) → blocking + prune against the
full train index → **6,156,110 pairs / 647,240 positives** (10.5%). Negatives are real
blocking look-alikes, never random.

**Model.** LightGBM, split **by S1 entity** 80/20 (never by row, which leaks):

```python
{'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
 'learning_rate':0.05,'num_leaves':127,'max_depth':-1,'min_child_samples':50,
 'feature_fraction':0.85,'bagging_fraction':0.85,'bagging_freq':5,
 'n_estimators':2500,'random_state':42,'n_jobs':16,'verbose':-1}
```

Early stop at iteration 500; the production model is refit on 100% of entities at
1.1× that iteration count.

**Selection** (`select_matches`): singleton gate (drop all candidates if best prob <
`τ_single`), relative floor (`thr = max(τ_match, best·γ_rel)`), sorted by probability
**before** the cap of **12** (measured GT max = 11). Thresholds were grid-searched on the
held-out val entities, including zero-candidate ones.

**Validation (S1-grouped 80/20, 40,000 val entities):**

| metric | value |
|---|---|
| Tuned rule val macro F₀.₅ | **0.9412** (τ_single=0.70, τ_match=0.60, γ_rel=0.70, cap 12) |
| Oracle (perfect classifier on our candidates) | 0.9775 |
| All-empty baseline | 0.0549 |
| Top-1 (p ≥ 0.5) | 0.6783 |

---

## 4. Other Relevant Information

**Compute.** Local Ryzen 7 (16 threads) + 16 GB RAM + RTX 5060 Laptop (8 GB).
- GPU: CUDA was **not** available (no `nvcc`; the pip LightGBM wheel is `USE_GPU=ON`
  but not `USE_CUDA`), so the GPU was driven through LightGBM's **OpenCL** backend
  (`device_type='gpu'`, `ER_LGB_DEVICE=gpu`). A full 6.16M-pair training run on the GPU
  reproduced the CPU metrics (val F₀.₅ 0.9403 vs 0.9412; oracle/all-empty/top-1
  identical), confirming the CPU/GPU parity the plan expects. The **CPU-trained model is
  shipped** (marginally better). Sparse retrieval and rapidfuzz features have no GPU
  path and stay CPU-bound.
- RAM discipline: 250K-row chunks, ≤2M-row shards, float32 caches, per-country/batch
  inference with candidate records materialised one shard at a time. An earlier
  all-in-RAM inference variant OOM'd during the India slice; the shipped `predict_test.py`
  holds only one shard's records and streams results to disk checkpoints.

**Runtimes** (no sampling; whole dataset): train index 518 s · test index 579 s ·
training 631 s · test inference 2,013 s.

**Error analysis / ablations.**
- The prune-floor sweep (§2) is the key ablation; the prior-log floors were rejected on
  measurement.
- The OpenCL `--sample` flag subsamples *rows*, which fragments per-entity candidate
  sets and deflates macro F₀.₅ — it is a speed knob, not a valid evaluation; full-scale
  runs are used for every reported metric.
- Score calibration: true-match cosine p10 = 0.457 / p50 = 0.782, while singleton top-1
  scores are p50 = 0.582 / p95 = 0.843 — heavy overlap, so singleton separation is done
  by the classifier, not by blocking.

**Licence compliance.** All components are MIT/Apache-2.0: LightGBM (MIT),
scikit-learn (BSD), scipy/numpy/pandas (BSD), rapidfuzz (MIT), anyascii (MIT),
sparse-dot-topn (Apache-2.0), pyarrow (Apache-2.0). No model exceeds 8B parameters; no
external data or APIs are used.

**Reproducibility.** Every stage is idempotent (skip-if-exists, `--force` to rebuild)
and reads paths from `ER_DATA_DIR` / `ER_WORK_DIR` / `ER_OUTPUT_DIR`. `code/` is
self-contained: `eda.py` and the organizer's `utils/validate_submission.py` are bundled
inside `code/business_entity_resolution/`, and all entry points resolve from the package
root (see `README.md`).

**Validation.** `utils/validate_submission.py --test-dir <test>` → **PASS** (exit 0) on
the full nested test set: every test S1 appears exactly once in both files, no duplicate
or unknown IDs, no S1 self-matches, and no match outside its candidate set.
