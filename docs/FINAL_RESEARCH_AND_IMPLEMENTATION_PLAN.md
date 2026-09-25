# Amazon ML Challenge 2026: Business Entity Resolution — Final Unified Research & Implementation Plan

**Status:** Final plan v1.0 — synthesis of `RESEARCH_GUIDE.md` + `RESEARCH_AND_IMPLEMENTATION_GUIDE.md` + official PDFs
**Sources merged:**
- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (task, schema, metric, fair-play, packaging)
- `6ab66c58d670e_final_submission_candidate_generation.pdf` (candidate-efficiency ranking rule)
- `RESEARCH_GUIDE.md` (operational depth: ingestion traps, OR-semantics, sharding, parallel features, ablation, compute, timeline, risks)
- `RESEARCH_AND_IMPLEMENTATION_GUIDE.md` (adaptive pruning, singleton gatekeeper, math, preprocessing code, LightGBM config, selection code)
**Metric:** macro F₀.₅ (precision-heavy) · **Deliverables:** `output/matching_results.tsv` + `output/candidate_pairs.tsv` + submission zip
**Posture:** validated end-to-end pipeline beats a polished half-pipeline. Format-PASS first, then recall, then classifier.

---

## 1. Task Definition (from problem-statement PDF — authoritative)

For each S1 record in `dataset/test/test_source1.tsv`, output the set of S2/S3 record IDs (`test_source2.tsv`, `test_source3.tsv`) for the same real-world business. S1 is deduplicated reference; S2/S3 are noisy. Cardinality per S1: zero (singleton), one, or many (mean ~3.5, max ~10).

Schema per source file: `entity_id` (prefix `S1-`/`S2-`/`S3-`; no separate source column) · `business_name` · `business_address` · `country` (open string set; train = US+India, test adds **France**).

Ground truth (train only): `train_ground_truth.tsv` → `source1_entity_id ↔ matched_entity_ids` (comma-separated S2/S3, empty = singleton).

### 1.1 Output contracts (exact — rejection if violated)

Both files are **tab-separated**. Always `pd.read_csv(..., sep="\t")`. Without it you silently get one column.

`output/matching_results.tsv` (only scored file; portal upload) and `output/candidate_pairs.tsv` (blocking audit file):

| matching_results.tsv | candidate_pairs.tsv | Rules (both) |
|---|---|---|
| `source1_entity_id` | `source1_entity_id` | exactly one row per test S1 entity |
| `matched_entity_ids` | `candidate_entity_ids` | comma-separated, no quoting, no dups, only S2/S3 IDs existing in test; empty string = none |

Critical semantics (from PDF): `candidate_pairs.tsv` = **exact set fed to the matching model for inference — the last stage before the model scores**, not an early blocking pass. Therefore `matching_results ⊆ candidates` always. Validator warns otherwise. Run before every upload:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

PASS = exit 0 (format only, not score). Final zip layout:

```
<team>_submission.zip
├── output/matching_results.tsv + candidate_pairs.tsv
├── code/business_entity_resolution/src/ + README.md + requirements.txt
└── Documentation_template.md   # methodology; blocking; model+features; other info; no page limit
```

### 1.2 Constraints (disqualification on violation)

1. **No external data lookup of any kind** — no geocoding APIs, registries, internet augmentation, commercial ER APIs. `anyascii` is a local library, allowed.
2. Final model **MIT/Apache-2.0, ≤8B params**. LightGBM (MIT) trivially complies; `mdeberta-v3-base` (~278M, MIT) also complies if used.
3. `country` is open-set. Never one-hot / filter / hard-code `{US, India}`. France must work day one.
4. Every test S1 exactly once; no dup IDs in a list; no S1 self-matches.

### 1.3 Scoring: macro F₀.₅

```
F₀.₅(e) = 1.25·P·R / (0.25·P + R),  macro-averaged over ALL S1 entities
```

- Singleton predicted empty → **1.0**; singleton predicted non-empty → **0.0** (~5.6% of entities are free points if protected).
- Precision ~2× recall (FP 4× worse than FN under β=0.5). Worked example from PDF: P=2/3,R=1 → 0.714. Conversely P=1.0,R=0.5 → 0.833. **Golden rule: when in doubt, withhold.**
- Top-1-only caps at ~0.68. Multi-match mandatory.
- Probabilities matter far more than rule polish (top-10 grid rules within 0.0004). Tune threshold conservatively (0.65–0.70).
- Zero-candidate entities still count: correct-empty = 1.0, missed = 0. Blocking recall is a hard ceiling.

Measured baselines (full train scale, from prior top-3 log — verify locally, trust ratios):

| Stage | Result |
|---|---|
| All-empty (singleton credit only) | 0.0538 |
| Top-1 (p≥0.5) | 0.6798 |
| Hashed TF-IDF blocking K=50 recall ceiling | 93.9% (K=100 → 94.9%) |
| LightGBM + tuned rule val macro F₀.₅ | **0.9425** |
| Oracle (perfect classifier on our candidates) | 0.9770 |

Gaps in order: **classifier/selection (~3.5 pts)** then **blocking recall (~2.3 pts)**.

---

## 2. The Decisive Organizer Update (candidate PDF — ranking rule)

> Blocking must scale; candidate generation counts toward final ranking beyond public/private leaderboard. **Smaller candidate set per S1 = ranked higher.**

Fixed K=50 over 1.7M test S1 = ~85M pairs; K=100 = ~170M pairs → efficiency penalty. **Final decision: retrieve Kmax=50, then adaptive prune to avg 18–24 (~30–40M pairs).** Keeps 93.7–93.9% recall ceiling while cutting volume ~55%, halving inference, and landing in the top compactness tier.

| Strategy | Avg/S1 | Test pairs | Recall ceiling | Rank |
|---|---|---|---|---|
| Fixed K=100 | 100 | ~170M | 94.9% | penalized |
| Fixed K=50 | 50 | ~85M | 93.9% | moderate |
| **Adaptive pruned (adopted)** | **18–24** | **~34M** | **~93.7%** | **best tier** |

Adaptive pruning algorithm (per e₁, scores s₁≥s₂≥…):

1. Retrieve Kmax=50 via sparse dot product.
2. If s₁ > 0.85 (dominant exact): keep sⱼ ≥ 0.50·s₁, cap K=10.
3. Else keep sⱼ ≥ 0.35·s₁, cap K=30.
4. If all sⱼ < 0.15 (likely singleton): keep top-3 or empty.
5. `candidate_pairs.tsv` = this pruned set (the exact model input).

This satisfies both PDFs: candidates are the last pre-model set AND compact.

---

## 3. Data Understanding & 30-Minute EDA (do first)

Reported scale: ~150K train S1 · ~10.3M train S2+S3 · **~1.7M test S1** · test S2/S3 same order. Full cross-product ~2×10¹³ — never materialize dense.

One EDA script, verify these before trusting:

- [ ] Row counts, ID prefix consistency, dup/near-dup within source, ID gaps.
- [ ] Singleton rate (~5.6%), matches-per-entity (mean 3.5, max 10).
- [ ] Missing `business_address` (~4–9% among matches) → blocking must be OR-semantics, never `name AND address`.
- [ ] Non-Latin share (~13.7% of matched S2/S3: Devanagari/Kannada/Bengali/Tamil). **ASCII-stripping trap:** recall → ~29%, 6.4% names become `""`. Transliteration mandatory.
- [ ] Country distribution; confirm **100% same-country pairs** in train → then block within country (halves space, zero recall cost). Never hard-code the set.
- [ ] Does any S2/S3 ID appear under two S1 rows? Decides if one-to-one assignment (§8.1) is legal.
- [ ] Noise samples: Corp/Corporation, Pvt/Private, Ltd/Limited, LLC, SAS/SARL, `&`/and, transpositions, transliteration drift, `F0ods` zero-for-o, `002050` leading zeros, `158ND/158th` ordinals, run-together domains.
- [ ] France vocab drift (`sarl`, `rue`, siret-like numbers, accents) — handle generically.

---

## 4. Architecture (3 stages + decision engine)

```
ingest (chunked TSV) → normalize (transliterate + fold)
→ block (FeatureHasher 2^22 + per-country IDF + sharded sparse top-Kmax=50 + adaptive prune)
→ pair features (~35, rapidfuzz C++) → LightGBM matcher
→ F₀.₅ selection (singleton gate + relative threshold, cap 10)
→ batched checkpointed inference → validate → zip
```

```
dataset/*.tsv ──▶ 1. INGEST   chunked pd.read_csv(sep="\t")
                   2. NORMALIZE anyascii → lower → fold + skeleton (lru_cached) + core-name views
                   3. BLOCK    feature bag (n_,b_,k_,d_,dw_,a_,ak_) → Hasher(2^22)
                               → per-country IDF → transposed shards → sparse_dot_topn Kmax=50
                               → adaptive prune → candidate_pairs.tsv
candidates ──▶   4. FEATURES  ~35/pair
                   5. TRAIN    LightGBM, S1-grouped split
                   6. SELECT   p≥thr AND p≥rel·best, cap 10 (grid-searched on macro F₀.₅)
                   7. INFER    100K-S1 batches, checkpointed → matching_results.tsv → validator
```

Suggested layout:

```
code/business_entity_resolution/
├── src/
│   ├── config.py                # paths from ER_DATA_DIR / ER_WORK_DIR
│   ├── cleaning/text_cleaner.py
│   ├── blocking/{build_index,retrieve,evaluate_blocking,ablate_blocking}.py
│   ├── matching/{pair_features,build_training_set,train_matcher,selection,predict_test}.py
│   └── submission/build_outputs.py
├── README.md + requirements.txt  # pandas, numpy, lightgbm, rapidfuzz, scipy, sparse-dot-topn, anyascii
```

K=50 test volume: 1.7M S1 → ~85M raw pairs → adaptive prune → ~34M → float32 streaming (never materialize ~12 GB) → predict → select → 2 TSVs.

---

## 5. Stage Specs

### 5.1 Ingestion (traps kill runs)

```python
df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                 chunksize=250_000, quoting=csv.QUOTE_MINIMAL)
```

- `sep="\t"` always; `keep_default_na=False` (`"NA"` must stay a string); chunked reads mandatory (5M-row full reads segfault/`MemoryError` on 16 GB); keep default CSV quoting (`"""ehpad Club SAS"` occurs).

### 5.2 Normalization (identical train/test, country-agnostic)

1. **Transliterate** with `anyascii` (fast-path pure-ASCII skip). `রাম মার্কেটিং→ram marketing`, `तिरुपति फाइनेंस→tirupati finance`, `Société Générale→societe generale`.
2. Lowercase; split digit/letter runs (`158ND→158 nd`, `Shop4→shop 4`); strip leading zeros (`002050→2050`); strip ordinals (`158th/158nd→158`); word numerals → digits; `0` between letters → `o` (`F0ods→foods`).
3. Canonicalize (fold, don't filter): legal suffixes → core-name view (`Tata Consultancy Services Ltd` → raw + `tata consultancy services`); streets (`street→st`, `road→rd`, `avenue→ave`, `boulevard→blvd`); directions (`north→n`).
4. **Phonetic consonant skeleton** per token, `@lru_cache(500_000)`: `ph→f, bh/v/w→b, kh→k, th→t, g/z→j, m→n`, drop silent `gh`, `tion→sn`, `c→s/k`, drop vowels, collapse repeats. `tirupti phainyans ↔ Tirupati Finance → trpt fynns`. Fold only documented substitutions (short Indic tokens collide otherwise).
5. Emit two name views (raw + core); similarities computed on both. Target ≥25K rec/s on 16 threads.

### 5.3 Blocking (the recall ceiling + efficiency win)

**OR-semantics, not AND-keys.** Conjunctive keys measured 53% recall. One strong shared feature (rare name token, house-number+street) must suffice.

Per-record bag (prefixed, order-independent): `n_` name unigrams · `b_` name bigrams · `k_` name skeleton · `d_` address numbers (zeros stripped) · `dw_` number+next word (`203_26th`, very high precision) · `a_` address words (canonicalized) · `ak_` address-word skeletons.

**Do NOT add concatenation features** (first-2/3 joins, long-token joins): −2.4 pts recall@10 measured.

Weighting: **per-country IDF** from corpus; drop DF > 0.1% within country at index time. Auto-suppresses `pvt/ltd/llc/rue/sarl/road` per geography, satisfying open-country rule.

Index/retrieval (16 GB-safe at 10M+):
1. `FeatureHasher(n_features=2**22)` — zero vocab RAM.
2. Pass 1: stream S2/S3 in 250K chunks → cache sparse chunks → accumulate per-country DF.
3. Pass 1b: regroup per country into ≤2M-row shards, **IDF-weighted + L2-normed + transposed on disk** (query 100s→7s per 10K).
4. Pass 2: batch S1, `sparse_dot_topn` matmul + top-Kmax=50 per shard, merge. Block within country (post-EDA confirm). Then adaptive prune (§2).

Recall@K (measured): K=5:79.6 · 10:88.4 · 20:91.6 · 30:92.9 · **50:93.9** · 75:94.5 · 100:94.9. Kmax=50 default, config-driven. Guard: reordered addresses, missing addresses (name-only must retrieve), non-Latin, number norms.

### 5.4 Pair features (~35, rapidfuzz C++)

Retrieval context: score, rank, score/best, gap-to-best, gap-to-next. Name: `ratio`, `token_sort`, `token_set`, `partial`, `WRatio`, Jaro-Winkler — each on raw, **core**, concatenated-core + token Jaccard + length-diff. Phonetic: same rapidfuzz set on skeletons. Address: `ratio`, `token_set`, `partial`, Jaccard. Numbers (critical): number-set Jaccard, shared count, **`num_conflict_flag`** (both have numbers, zero shared → strong negative; e.g. `104 Main` vs `502 Main`), longest shared number len, postcode +1/−1/0. Flags: empty-address either side, non-Latin candidate, is_S2/is_S3.

**Excluded:** `country` (overfits {US,India}, zero observed gain). Importance ref: score 31%, rank 29%, core_partial 7.9%, num_jacc 7.6%, addr_token_set 5.5%, num_conflict 2.1%, postcode ~0 (subsumed by num_*).

Parallelization (verified 514K pairs: 31.1s serial → 13.5s/8 procs, flat mem): workers get only their record-tuple chunk (no shared big dicts — fork+read = CoW explosion), bounded waves (~2 chunks/worker), return float32 arrays (~120 B/row vs ~1 KB tuples). Serial ∥ parallel identical.

### 5.5 Training set

~150K random train S1 × same Kmax=50 blocking + prune vs full train S2/S3 (~7.5M pairs, ~487K positives). **Negatives from real blocking output** (look-alikes), never random. Label from ground truth. Zero-candidate entities in separate accounting (still count in macro). Seeds fixed; cache pairs (Parquet/float32).

### 5.6 Model: LightGBM (MIT)

Split **by S1 entity 80/20**, never by row (row split leaks + inflates). Config validated:

```python
{'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
 'learning_rate':0.05,'num_leaves':127,'max_depth':-1,'min_child_samples':50,
 'feature_fraction':0.85,'bagging_fraction':0.85,'bagging_freq':5,
 'n_estimators':2500,'random_state':42,'n_jobs':16,'verbose':-1}
```

Early-stop on val log-loss (~iter 1800–2000; observed best 1864, 0.0135), then **refit on 100% entities at 1.1× best iters**. CPU-only. Persist model + feature list + split manifest + OOF val predictions.

### 5.7 Selection (tune on metric, share code val/test)

```python
def select_matches(candidates, tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70, max_matches=10):
    if not candidates: return []
    best = max(c['prob'] for c in candidates)
    if best < tau_singleton: return []          # singleton gate: protect free 1.0
    thr = max(tau_match, best * gamma_rel)
    return [c['entity_id'] for c in candidates if c['prob'] >= thr][:max_matches]
```

Grid-search `(tau_singleton, tau_match, gamma_rel)` maximizing macro F₀.₅ over all val entities incl. zero-candidate. Start `(0.65–0.68, 0.60–0.65, 0.70, 10)` → val 0.9425. Print every run: oracle / all-empty / top-1 baselines.

### 5.8 Test inference

100K-S1 batches: retrieve → prune → features → predict → select → append. **Checkpoint every batch**; resume skips finished. Log per-batch timings (pre-opt ~830s/5M-pair batch ≈ 4h total). Load S2/S3 strings once, lazily per batch. Write candidates = pruned model-input set; matches ⊆ candidates. Validator auto-run + train dress-rehearsal scorer.

---

## 6. Validation & Experiment Protocol

Write the ~40-line macro F₀.₅ scorer first; sanity-check vs PDF example (P=2/3,R=1→0.714). Handle empty/empty=1.0, empty/non-empty=0.0. Mandatory per-run diagnostics: oracle, all-empty, top-1, tuned rule + grid neighborhood (confirm flat optimum), per-country/source slices (India/S3 noisier).

Ablation discipline: one change per experiment (a six-change bundle once net-hurt); fast harness (10K S1 vs true matches + ~1.5M distractors, in-memory) for relative ordering in minutes, confirm winner once at full scale. Any cleaning/feature change invalidates caches — rebuild, version caches by feature-config hash.

Leakage checklist: S1-grouped splits; no `country` feature/rules; rule fit on val only; IDF/hashing fit on indexed corpus only (train index for train, test index for inference — unsupervised, allowed); no external data.

---

## 7. Compute Plan

| Machine | Role |
|---|---|
| Local Ryzen 7/16T, 16 GB (RTX 5060 8 GB reserved) | EDA, ablations on samples, validator, packaging. 250K chunks, ≤2M shards, <9.5 GB. GPU offline for classical path. |
| Kaggle CPU (30 GB, 4 cores, 12h commit runs) | Full index (~7.5 min obs.), train-set build (~13 min/7.5M obs.), training, full inference. Nothing classical needs GPU. |

Env-var paths (`ER_DATA_DIR`, `ER_WORK_DIR`) for local↔Kaggle portability. Every stage idempotent (skip if output exists). Heavy runs = "Save & Run All (Commit)"; drafts die on idle and can lose `/kaggle/working`.

---

## 8. Optional Extensions (ranked; only after classical pipeline is stable + submitted)

1. **One-to-one assignment** — if EDA confirms no S2/S3 claimed by 2 S1 rows: contested S2/S3 → highest-p claimant only. Cuts false merges between look-alike S1s. Else soften/skip.
2. **Error-analysis loop** — split FP/FN by country×source, dump worst cases, one targeted fix at a time.
3. **GPU cross-encoder reranker** — `mdeberta-v3-base` fine-tune on ~487K pairs w/ hard negatives; score top-10/S1 (~17M) or feed as LightGBM feature. Literature (DeepMatcher/Magellan) says boosting stays competitive on noisy structured data — gap-closer for the 3.5-pt classifier gap, not foundation. Trains on 5060 (fp16/short pairs).
4. **Speed/quality trims** — higher LR + fewer trees, drop zero-importance features. Multi-view (name/address union) retrieval: similar recall, ~30% fewer pairs, more complexity — deferred.

Alternatives rejected/deferred: AND-key blocking (53% cap); dense TF-IDF KNN (OOM at 1.7M×10M); cross-encoder-alone (86M infeasible); graph clustering (task is star-shaped, S1 deduped); MinHash-LSH viable fallback if sparse top-K stalls (arXiv 1905.06167); bi-encoder (multilingual-e5) optional semantic recall.

---

## 9. Execution Timeline (~48h from acceptance)

| Window | Goal | Done when |
|---|---|---|
| 0–3h | Scaffold, EDA §3, ingest+clean, format-exact empty baseline | validator PASS; first safe submission |
| 3–10h | Blocking v1 + eval harness (recall@K, missed dumps) | recall@50 ≥90%; pruned `candidate_pairs.tsv`; avg ≤25/S1 |
| 10–18h | Features + train-set + LightGBM + selection grid + scorer | val macro F₀.₅ ≥0.93 + oracle/baselines printed |
| 18–26h | Full Kaggle: index → train → batched infer | real `matching_results.tsv` uploaded |
| 26–36h | Error analysis + one-change fixes + re-tune | val improves or revert |
| 36–44h | One-to-one (§8.1) + stability (seeds/resume/timings) | final outputs from clean caches |
| 44–48h | Freeze; end-to-end rerun from raw; validate; docs; zip | PASS + reproducible from `code/` alone |

Fork priority: **validity > recall > classifier > rule polish > extras.**

---

## 10. Risk Register

| Risk | P | Mitigation |
|---|---|---|
| Format rejection | M | validator every upload; trivial baseline first |
| pandas segfault/MemoryError | H (16 GB) | chunked reads, float32 sparse caches |
| Kaggle draft death | H | commit runs + per-batch checkpoints |
| Stale caches | M | config-hash versioning; rebuild on clean change |
| France failure | M | zero country-conditional logic; per-country IDF; simulate by holding out India |
| Non-Latin collapse | H if ignored | anyascii + skeleton; monitor slice recall |
| Recall ceiling | H | oracle every run; K config; missed dumps |
| Feature parallel blowup | M | bounded waves, chunk-scoped workers, float32 |
| Quoting (`"""...`) | L | default quoting; cleaning tests |
| matches ⊄ candidates | L | assert + validator |

---

## 11. Immediate Action Checklist (when dataset arrives)

```
Dataset ──▶ [EDA verify] ──▶ [Normalizer] ──▶ [Sparse index + Kmax50 + prune]
                                                        │
[Package] ◄── [Selection grid] ◄── [LightGBM] ◄────────┘
```

1. EDA script: counts, singleton %, non-Latin %, cross-country pairs (expect 0%).
2. `text_cleaner.py`: anyascii + skeleton + suffix split; ≥25K/s.
3. `build_index.py` Hasher(2²²); val recall ≥93.5%, avg ≤25; draft `candidate_pairs.tsv`.
4. 35 features → 5-fold S1-grouped CV → thresholds → ≥0.940 → `matching_results.tsv`.
5. Validate + zip per §1.1.

Success gates: licenses MIT/Apache-2.0 ✓ · offline-only ✓ · France generic ✓ · compact candidates ✓ · singleton gate ✓.

---

## 12. References

- Problem statement + candidate-update PDFs (this repo).
- Top-3 pipeline/log for this challenge: `stack-ajit/business_entity_resolution` (`project_log.md`) + Top-3 playbook PDF; Unstop listing.
- Blocking survey arXiv 1905.06167; DeepMatcher (SIGMOD'18); Magellan (CACM); SBERT cross-encoder training; HF reranker blog.
- Models/libs (MIT/Apache): LightGBM · sparse-dot-topn · rapidfuzz · anyascii · microsoft/mdeberta-v3-base.
