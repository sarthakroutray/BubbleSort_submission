# Amazon ML Challenge 2026: Business Entity Resolution — Final Unified Research & Implementation Plan

**Status:** v1.2 — **§3 re-verified against the shipped dataset; implementation NOT started.** Every §3 figure was re-derived from the on-disk data with `scripts/eda.py` on 2026-09-26 and matches this document exactly. The pipeline code under `code/business_entity_resolution/src/` is scaffolding only (66 LOC, almost all docstrings) and the Python environment has **no** dependencies installed — see **§0** before doing anything.
**Sources merged:**
- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (task, schema, metric, fair-play, packaging)
- `6ab66c58d670e_final_submission_candidate_generation.pdf` (candidate-efficiency ranking rule)
- `RESEARCH_GUIDE.md` (operational depth — **cited from a prior synthesis, not present in this repo**; see §12)
- `RESEARCH_AND_IMPLEMENTATION_GUIDE.md` (adaptive pruning, singleton gatekeeper, math, preprocessing code, LightGBM config, selection code — **not present in this repo**; see §12)
- **Shipped dataset EDA** (`scripts/eda.py` output; supersedes all "reported/expected" figures)
**Metric:** macro F₀.₅ (precision-heavy) · **Deliverables:** `output/matching_results.tsv` + `output/candidate_pairs.tsv` + submission zip
**Posture:** validated end-to-end pipeline beats a polished half-pipeline. Format-PASS first, then recall, then classifier.
**Provenance legend:** ★ *measured* on shipped data (reproducible with `scripts/eda.py`) · *(prior log)* inherited from a prior top-3 run, **not** verified locally · unmarked = specification / decision.

---

## 0. Current State (audited 2026-09-26)

| Component | State | Evidence |
|---|---|---|
| Dataset on disk | ✅ present | `dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}` — 7 TSVs, ~2.4 GB (`6ab10eb3b23ba_student_resource.zip`, 1.09 GB, also on disk). Git-ignored via `.gitignore` (`dataset/`, `*.tsv`, `*.zip`). |
| EDA (§3) | ✅ **verified** | `scripts/eda.py` re-run reproduces every §3 figure exactly (row counts, singleton 5.58%, GT histogram, country mix, 100% same-country, 0 contested IDs). |
| Format validator | ✅ usable | `utils/validate_submission.py` (stdlib only). |
| Pipeline code | 🟡 **partial** | `cleaning/text_cleaner.py` (§5.2, 23 tests, 82K rec/s); `blocking/{build_index,retrieve,evaluate_blocking}.py` (§5.3 — full train index 10.32M docs/6 shards, measured recall §3.5); `matching/{pair_features,build_training_set,train_matcher,predict_test,scorer,selection}.py`. `config.py` env-anchored. |
| Python environment | ✅ **ready** | `.venv` (uv-managed CPython 3.14.4); pandas 3.0.6 / numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 / lightgbm 4.7.0 / rapidfuzz 3.14.6 / anyascii 0.3.3 / sparse-dot-topn 1.2.0 / pyarrow 25.0.1 all import cleanly. **API note:** sparse-dot-topn 1.2.0 removed `sparse_dot_topn`; use `awesome_cossim_topn` (deprecated warning) or `sp_matmul_topn`. |
| Blocking (§5.3) | ✅ **measured** | Train index: 10.32M docs, 6 shards (518s). recall@50 = **0.9436**, pruned 0.9355 @ 30.8 cand/S1 (§3.5). |
| Matcher (§5.6) | ✅ **trained** | 6.16M pairs / 647K pos; early-stop @ 500 iters; **val macro F₀.₅ = 0.9412**; oracle 0.9775 / all-empty 0.0549 / top-1 0.6783. |
| Outputs | ✅ **VALIDATED** | Real `output/{matching_results,candidate_pairs}.tsv` for all **1,732,544** test S1 — avg **30.21** candidates/S1, avg **3.09** matches/S1, **6.49%** predicted-empty, max 12 (cap). `utils/validate_submission.py` → **PASS** (exit 0). |
| GPU | ✅ **verified** | RTX 5060 (8 GB) driven via LightGBM's **OpenCL** backend (`device_type='gpu'`; the wheel is built `USE_GPU=ON` but not `USE_CUDA`, and there is no `nvcc` to build CUDA). Full 6.16M-pair training on GPU: val F₀.₅ **0.9403** vs CPU **0.9412**, identical baselines — §7's "CPU fallback identical metrics" holds. |

**§11 items 1–6 are done** (env, baseline, cleaner, blocking+recall, features/model/selection, inference) and the real submission **validates PASS**. §11 item 7 (packaging/zip) remains. §3.5–§3.6 and §0 carry the measured numbers that replace the prior-log targets.

---

## 1. Task Definition (from problem-statement PDF — authoritative)

For each S1 record in `dataset/test/test_source1.tsv`, output the set of S2/S3 record IDs (`test_source2.tsv`, `test_source3.tsv`) for the same real-world business. S1 is deduplicated reference; S2/S3 are noisy. Cardinality per S1: zero (singleton), one, or many — **measured: mean 3.46, max 11** (histogram in §3).

Schema per source file: `entity_id` (prefix `S1-`/`S2-`/`S3-` + **random 1–9-digit numeric suffix, non-sequential**; no separate source column) · `business_name` · `business_address` · `country` (open string set; train = US + India, test adds **France**; measured mix in §3.1).

Ground truth (train only): `train_ground_truth.tsv` → `source1_entity_id ↔ matched_entity_ids` (comma-separated S2/S3, empty = singleton).

### 1.1 Output contracts (exact — rejection if violated)

Both files are **tab-separated**. Always `pd.read_csv(..., sep="\t")`. Without it you silently get one column.

`output/matching_results.tsv` (only scored file; portal upload) and `output/candidate_pairs.tsv` (blocking audit file):

| matching_results.tsv | candidate_pairs.tsv | Rules (both) |
|---|---|---|
| `source1_entity_id` | `source1_entity_id` | exactly one row per test S1 entity |
| `matched_entity_ids` | `candidate_entity_ids` | comma-separated, no quoting, no dups, only S2/S3 IDs existing in test; empty string = none |

Critical semantics (from PDF): `candidate_pairs.tsv` = **exact set fed to the matching model for inference — the last stage before the model scores**, not an early blocking pass. Therefore `matching_results ⊆ candidates` always. Validator warns otherwise. Run before every upload (note the **nested extraction path** — there is no top-level `dataset/test`):

```bash
export ER_DATA_DIR=dataset/6ab10eb3b23ba_student_resource/student_resource/dataset
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$ER_DATA_DIR/test"
```

PASS = exit 0 (format only, not score). Final zip layout:

```
<team>_submission.zip
├── output/matching_results.tsv + candidate_pairs.tsv
├── code/business_entity_resolution/src/ + README.md + requirements.txt
└── Documentation_template.md   # methodology; blocking; model+features; other info; no page limit
```

**Packaging self-containment (PDF rule: "regenerate both output files using only what is in this folder").** ✅ **Resolved.** `eda.py` is copied to `code/business_entity_resolution/eda.py`, the organizer's validator is bundled at `code/business_entity_resolution/utils/validate_submission.py` (stated as unmodified, run against the untouched test set), and every entry point in `README.md` resolves from the package root. `BubbleSort_submission.zip` (`output/` + `code/` + `Documentation_template.md`) is built.

### 1.2 Constraints (disqualification on violation)

1. **No external data lookup of any kind** — no geocoding APIs, registries, internet augmentation, commercial ER APIs. `anyascii` is a local library, allowed.
2. Final model **MIT/Apache-2.0, ≤8B params**. LightGBM (MIT) trivially complies; `mdeberta-v3-base` (~278M, MIT) also complies if used.
3. `country` is open-set. Never one-hot / filter / hard-code `{US, India}`. France must work day one.
4. Every test S1 exactly once; no dup IDs in any list; no S1 self-matches.

### 1.3 Scoring: macro F₀.₅

```
F₀.₅(e) = 1.25·P·R / (0.25·P + R),  macro-averaged over ALL S1 entities
```

- Singleton predicted empty → **1.0**; singleton predicted non-empty → **0.0**. **Measured: 5.58% of train S1 are singletons (123,247 / 2,206,821)** → all-empty baseline = **0.0558**.
- Precision ~2× recall (FP 4× worse than FN under β=0.5). Worked example from PDF: P=2/3,R=1 → 0.714. Conversely P=1.0,R=0.5 → 0.833. **Golden rule: when in doubt, withhold.**
- Top-1-only caps at ~0.68 *(prior log)*. Multi-match mandatory.
- Probabilities matter far more than rule polish (top-10 grid rules within 0.0004) *(prior log)*. Tune threshold conservatively (0.65–0.70).
- Zero-candidate entities still count: correct-empty = 1.0, missed = 0. Blocking recall is a hard ceiling.

Baselines (★ = measured on shipped data; others = prior top-3 log at full scale — trust ratios, re-verify on first full run):

| Stage | Result |
|---|---|
| All-empty (singleton credit only) | ★ 0.0558 (measured) |
| Top-1 (p≥0.5) | 0.6798 *(prior log)* |
| Hashed TF-IDF blocking K=50 recall ceiling | 93.9% (K=100 → 94.9%) *(prior log)* |
| LightGBM + tuned rule val macro F₀.₅ | **0.9425** *(prior log)* |
| Oracle (perfect classifier on our candidates) | 0.9770 *(prior log)* |

Gaps in order: **classifier/selection (~3.5 pts)** then **blocking recall (~2.3 pts)**.

---

## 2. The Decisive Organizer Update (candidate PDF — ranking rule)

> Blocking must scale; candidate generation counts toward final ranking beyond public/private leaderboard. **Smaller candidate set per S1 = ranked higher.**

**Measured test scale: 1,732,544 S1 × 9,969,589 S2/S3.** Fixed K=50 = 86.6M pairs; K=100 = 173M pairs → efficiency penalty. **Final decision: retrieve Kmax=50, then adaptive prune to avg 18–24 (~31–42M pairs).** Keeps the ~93.7–93.9% recall ceiling *(prior log)* while cutting volume ~55%, halving inference, and landing in the top compactness tier.

| Strategy | Avg/S1 | Test pairs | Recall ceiling | Rank |
|---|---|---|---|---|
| Fixed K=100 | 100 | ~173M | 94.9% *(prior log)* | penalized |
| Fixed K=50 | 50 | ~87M | 93.9% *(prior log)* | moderate |
| **Adaptive pruned (adopted)** | **18–24** | **~31–42M** | **~93.7%** *(prior log)* | **best tier** |

Adaptive pruning algorithm (per e₁, scores s₁≥s₂≥…):

1. Retrieve Kmax=50 via sparse dot product.
2. If s₁ > 0.85 (dominant exact): keep sⱼ ≥ 0.50·s₁, cap K=10.
3. Else keep sⱼ ≥ 0.35·s₁, cap K=30.
4. If all sⱼ < 0.15 (likely singleton): keep top-3 or empty.
5. `candidate_pairs.tsv` = this pruned set (the exact model input).

This satisfies both PDFs: candidates are the last pre-model set AND compact.

**⚠ Pruned recall is a target, not a measurement.** The 93.7% ceiling is *(prior log)*; pruning 50→~20 can only remove true matches, and its loss has **never been measured locally**. Before adopting the relative floors, evaluate recall of the *pruned* set on a train sample (per §6 harness). If pruned recall at avg ~20/S1 < 93%, raise the floors (or use an absolute score floor) rather than accept the loss — a smaller candidate set that drops true pairs trades an efficiency rank for real F₀.₅ points, which is a bad trade under a precision-heavy metric.

---

## 3. Data Understanding — EDA COMPLETE (every item below re-verified on the shipped dataset, 2026-09-26)

Reproduce any figure (stdlib-only, streaming, ~1 min, low RAM):

```bash
python3 scripts/eda.py dataset/6ab10eb3b23ba_student_resource/student_resource/dataset
```

The re-run output matches every table, histogram and percentage in §3.1–§3.3 **exactly**.

### 3.1 Measured scale & contents

| File | Rows | Missing name | Missing address | Non-Latin name¹ | Names killed by ASCII-strip² | Non-Latin address |
|---|---|---|---|---|---|---|
| train_source1 | 2,206,821 | 0 | **0** | 0 | 0 | 554 |
| train_source2 | 5,034,616 | 0 | 168,967 (3.4%) | 764,608 (15.2%) | 416,230 (8.3%) | 478,453 (9.5%) |
| train_source3 | 5,285,603 | 0 | 175,916 (3.3%) | 606,737 (11.5%) | 222,288 (4.2%) | 476,588 (9.0%) |
| test_source1 | 1,732,544 | 0 | **0** | 40,789 (2.4%)³ | 0 | 73,800 (4.3%) |
| test_source2 | 4,887,273 | 0 | 129,408 (2.6%) | 928,158 (19.0%)³ | 480,860 (9.8%) | 720,665 (14.7%) |
| test_source3 | 5,082,316 | 0 | 136,098 (2.7%) | 737,515 (14.5%)³ | 256,896 (5.1%) | 729,222 (14.4%) |

¹ any char > U+007F (Indic scripts + accents) · ² name becomes `""` after non-ASCII removal — **6.2% of train, 7.4% of test S2/S3 names** → ASCII-stripping is fatal, transliteration mandatory · ³ includes French accents (test-only country).

- Full cross-products to never materialize: train 2.21M × 10.32M ≈ 2.3×10¹³ · test 1.73M × 9.97M ≈ 1.7×10¹³.
- ID integrity: **0** malformed IDs, **0** duplicate IDs in any file, **0** S1 self-matches, **0** dups within match lists. IDs are random variable-width numerics with **no ordering** — S1 suffixes are **3–9 digits** (`S1-965667` … `S1-925783039`), S2/S3 and matched IDs are **1–9 digits** (the shortest observed is a 1-digit S3 suffix in `test_source3`) — so **"ID gap"/sequential checks are meaningless** on this scheme.
- `business_name` is **never** empty in any file. `business_address` is **never** empty on the S1 side (train and test); only S2/S3 miss addresses — **train S2/S3 3.34%**, **test S2/S3 2.66%**, and **4.41% among GT-matched pairs** (337,018 / 7,638,365). → OR-semantics blocking is required exactly because of the S2/S3 side; name-only retrieval must work.

### 3.2 Ground-truth structure (train_ground_truth.tsv, 2,206,821 rows)

| matches/S1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entities | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |

- Singletons **5.58%**; mean **3.461**; **max 11** (plan assumed max 10 → **selection cap must be ≥11**, use 12).
- Total GT pairs **7,638,365** (S2: 3,693,619 · S3: 3,944,746).
- **Every matched S2/S3 ID is claimed by exactly one S1** (7,638,365 distinct matched IDs = 7,638,365 pairs; 0 claimed twice) → **GT is a perfect matching; one-to-one assignment (§8.1) is legal and validated.**
- Non-Latin among matched S2/S3 names: 1,060,781 / 7,638,365 = **13.9%** (Devanagari, Telugu, …).

### 3.3 Country rule (the decisive check) — CONFIRMED

- **7,638,365 / 7,638,365 train pairs are same-country (100.0%, 0 cross-country).** → block within country at zero recall cost: ~½ the space on train (2 countries), ~⅓ on test (3 countries, France included). Still never hard-code the set — derive the buckets from the data.
- Train S1: US 1,323,633 (60.0%) · India 883,188 (40.0%). Train S2/S3: US ≈60% / India ≈40%.
- Test S1: **India 809,986 (46.7%) · US 663,106 (38.3%) · France 259,452 (15.0%)**. Test S2: India 2,312,565 · US 1,871,330 · France 703,378 (14.4%). Test S3: India 2,405,000 · US 1,945,701 · France 731,615 (14.4%).
- **Distribution shift:** train is US-majority; test is **India-majority** with 15% France. India/S3 slices are noisier *(prior log)* — keep per-country slice metrics.

### 3.4 Noise samples (verified in data)

Corp/Corporation, Pvt/Private, Ltd/Limited · `&`/and · transpositions · transliteration drift · zero-for-o (`indianf0ods.com`, `Hitech F0ods Bakery`) · leading zeros (`0000000305/1A`, `0002/192/A/U-9/4/A`) · ordinals (`2Nd Floor`, `1st Street`) · run-together domains (`wilfordhancock.com`, `primef0od.com`) · CSV quoting inside fields (`"""ehpad…`; `"MUMBAI CITY, NO 13 ROLI(E),` …) · literal `N/A` placeholders in addresses (→ `keep_default_na=False` is mandatory) · landmark refs (`Near SBI ATM`). France vocab drift: `SARL` prefix, `R`/`IMP.` street abbreviations (`26 R DNEIS CORDONNIER`, `2 IMP. LAMARTINE`), accents (`È`, `é`) — handle generically, never country-specific.

---

## 3.5 Blocking recall — LOCALLY MEASURED (supersedes prior-log §5.3 numbers)

Measured 2026-09-26 with `python3 -m src.blocking.evaluate_blocking --sample 40000 --kmax 100`
against the full 10.32M-row train S2/S3 index (hashed bag `n_ b_ k_ d_ dw_ a_ ak_`,
per-country IDF, DF>0.1% dropped), 40,000 random train S1, 137,937 true pairs:

| K | 1 | 3 | 5 | 10 | 20 | 30 | **50** | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| recall@K | 0.257 | 0.641 | 0.819 | 0.901 | 0.926 | 0.935 | **0.9436** | 0.949 | 0.953 |

The measured curve tracks the prior-log targets closely (K=50: 0.9436 vs 0.939 claimed)
— but it is now **measured**, not assumed.

**Prune calibration (this is the important correction).** Applying the §2 prior-log
floors *unchanged* loses ~7 pts of recall: **0.8734 at 11.6 candidates/S1**. A sweep
over 40K train S1 selected a looser rule:

| config | pruned recall | avg cand/S1 |
|---|---|---|
| prior-log floors (0.50/0.35, caps 10/30) | 0.8734 | 11.6 |
| floors 0.30/0.20, caps 20/40 | 0.9291 | 23.9 |
| **adopted: floors 0.28/0.17, caps 30/45** | **0.9355** | **30.8** |
| floors 0.25/0.15, caps 25/50 | 0.9355 | 31.8 |

**Adopted `adaptive_prune(s_hi=0.85, s_lo=0.15, floors=(0.28,0.17), caps=(30,45))`** —
<1 pt below the K=50 ceiling, satisfying the §2 guard ("raise the floors rather than
accept the loss"). Candidate sets are larger than the 18-24/S1 efficiency target;
under a precision-heavy metric the plan explicitly prefers recall over compactness.
Score calibration from the same run: true-match cosine p10=0.457 / p50=0.782;
singleton top-1 p50=0.582 / p95=0.843 (heavy overlap → the classifier, not blocking,
is what separates singletons).

---

## 3.6 End-to-end test run — FINAL (2026-09-27)

Full run on the shipped test set (no sampling), from raw TSVs to validated outputs:

| Stage | Result |
|---|---|
| Test index build | 9,969,589 S2/S3 docs → 6 shards (India 3, US 2, France 1), **579 s** |
| Retrieval | 1,732,544 S1 queries, K=50, adaptive prune (0.28/0.17, caps 30/45) |
| Feature + model inference | `predict_test.py`, per-country/batch, checkpointed — **2,013 s** |
| Candidate volume | avg **30.21** candidates/S1 |
| Predicted matches | avg **3.09** matches/S1; **93.51%** of S1 get ≥1 match; **6.49%** empty; max 12 (cap) |
| Validator | `utils/validate_submission.py` → **PASS** (exit 0) |

Match-count histogram is unimodal at 2–5 (peaks: 3 → 422,875 S1), matching the train
GT shape (mean 3.46, max 11). Predicted-empty rate 6.49% vs the train singleton rate
5.58% — the expected direction, since the singleton gate is tuned precision-heavy.

Model choice: the **CPU-trained model is shipped** (val F₀.₅ 0.9412 vs 0.9403 for the
GPU run); `ER_LGB_DEVICE=gpu` reproduces it to within 0.001 on the RTX 5060.

---

## 4. Architecture (3 stages + decision engine)

```
ingest (chunked TSV) → normalize (transliterate + fold)
→ block (FeatureHasher 2^22 + per-country IDF + sharded sparse top-Kmax=50 + adaptive prune)
→ pair features (~35, rapidfuzz C++) → LightGBM matcher
→ F₀.₅ selection (singleton gate + relative threshold, cap 12)
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
                   6. SELECT   p≥thr AND p≥rel·best, cap 12 (grid-searched on macro F₀.₅)
                   7. INFER    100K-S1 batches, checkpointed → matching_results.tsv → validator
```

Suggested layout:

```
code/business_entity_resolution/
├── src/
│   ├── config.py                # paths from ER_DATA_DIR / ER_WORK_DIR
│   ├── cleaning/text_cleaner.py
│   ├── blocking/{build_index,retrieve,evaluate_blocking,ablate_blocking}.py
│   ├── matching/{pair_features,build_training_set,train_matcher,scorer,selection,predict_test}.py
│   └── submission/build_outputs.py
├── README.md + requirements.txt  # pandas, numpy, scipy, scikit-learn (FeatureHasher), lightgbm, rapidfuzz, sparse-dot-topn, anyascii, pyarrow
```

K=50 test volume: 1.73M S1 → 86.6M raw pairs → adaptive prune → ~36M → float32 streaming (never materialize ~12 GB) → predict → select → 2 TSVs.

---

## 5. Stage Specs

### 5.1 Ingestion (traps kill runs)

```python
df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                 chunksize=250_000, quoting=csv.QUOTE_MINIMAL)
```

- `sep="\t"` always; `keep_default_na=False` (**literal `"N/A"` occurs in addresses** — verified); chunked reads mandatory (5M-row full reads segfault/`MemoryError` on 16 GB); keep default CSV quoting (`"""ehpad Club SAS"` occurs — verified).

### 5.2 Normalization (identical train/test, country-agnostic)

1. **Transliterate** with `anyascii` (fast-path pure-ASCII skip). `রাম মার্কেটিং→ram marketing`, `तिरुपति फाइनेंस→tirupati finance`, `Société Générale→societe generale`. Mandatory: 6.2% of train / 7.4% of test S2/S3 names are entirely non-ASCII.
2. Lowercase; split digit/letter runs (`158ND→158 nd`, `Shop4→shop 4`); strip leading zeros (`002050→2050`); strip ordinals (`158th/2Nd→158`); word numerals → digits; `0` between letters → `o` (`F0ods→foods`).
3. Canonicalize (fold, don't filter): legal suffixes → core-name view (`Tata Consultancy Services Ltd` → raw + `tata consultancy services`); streets (`street→st`, `road→rd`, `avenue→ave`, `boulevard→blvd`); directions (`north→n`).
4. **Phonetic consonant skeleton** per token, `@lru_cache(500_000)`: `ph→f, bh/v/w→b, kh→k, th→t, g/z→j, m→n`, drop silent `gh`, `tion→sn`, `c→s/k`, drop vowels, collapse repeats. `tirupti phainyans ↔ Tirupati Finance → trpt fynns`. Fold only documented substitutions (short Indic tokens collide otherwise).
5. Emit two name views (raw + core); similarities computed on both. Target ≥25K rec/s on 16 threads.

### 5.3 Blocking (the recall ceiling + efficiency win)

**OR-semantics, not AND-keys.** Conjunctive keys measured 53% recall *(prior log)*. One strong shared feature (rare name token, house-number+street) must suffice — required by the measured 4.4% of GT pairs whose S2/S3 side lacks an address.

Per-record bag (prefixed, order-independent): `n_` name unigrams · `b_` name bigrams · `k_` name skeleton · `d_` address numbers (zeros stripped) · `dw_` number+next word (`203_26th`, very high precision) · `a_` address words (canonicalized) · `ak_` address-word skeletons.

**Do NOT add concatenation features** (first-2/3 joins, long-token joins): −2.4 pts recall@10 measured *(prior log)*.

Weighting: **per-country IDF** from corpus; drop DF > 0.1% within country at index time. Auto-suppresses `pvt/ltd/llc/rue/sarl/road` per geography, satisfying open-country rule. **Block within country — validated: 100% of 7.64M GT pairs are same-country.**

Index/retrieval (low-RAM at 10M+ — see §7 budget):
1. `FeatureHasher(n_features=2**22)` — zero vocab RAM.
2. Pass 1: stream S2/S3 in 250K chunks → cache sparse chunks → accumulate per-country DF.
3. Pass 1b: regroup per country into ≤2M-row shards, **IDF-weighted + L2-normed + transposed on disk** (query 100s→7s per 10K).
4. Pass 2: batch S1, `sparse_dot_topn` matmul + top-Kmax=50 per shard, merge. Then adaptive prune (§2).

Recall@K *(prior log)*: K=5:79.6 · 10:88.4 · 20:91.6 · 30:92.9 · **50:93.9** · 75:94.5 · 100:94.9. Kmax=50 default, config-driven. **These are prior-log targets — the first real task in §11 item 5 is to reproduce the recall@K curve locally on a train sample (true matches known) before choosing Kmax or trusting the pruning floors.** Guard: reordered addresses, missing addresses (name-only must retrieve — 4.4% of true pairs), non-Latin, number norms.

### 5.4 Pair features (~35, rapidfuzz C++)

Retrieval context: score, rank, score/best, gap-to-best, gap-to-next. Name: `ratio`, `token_sort`, `token_set`, `partial`, `WRatio`, Jaro-Winkler — each on raw, **core**, concatenated-core + token Jaccard + length-diff. Phonetic: same rapidfuzz set on skeletons. Address: `ratio`, `token_set`, `partial`, Jaccard. Numbers (critical): number-set Jaccard, shared count, **`num_conflict_flag`** (both have numbers, zero shared → strong negative; e.g. `104 Main` vs `502 Main`), longest shared number len, postcode +1/−1/0. Flags: empty-address either side, non-Latin candidate, is_S2/is_S3.

**Excluded:** `country` (overfits {US,India}, zero observed gain). Importance ref *(prior log)*: score 31%, rank 29%, core_partial 7.9%, num_jacc 7.6%, addr_token_set 5.5%, num_conflict 2.1%, postcode ~0 (subsumed by num_*).

Parallelization (verified 514K pairs: 31.1s serial → 13.5s/8 procs, flat mem): workers get only their record-tuple chunk (no shared big dicts — fork+read = CoW explosion), bounded waves (~2 chunks/worker), return float32 arrays (~120 B/row vs ~1 KB tuples). Serial ∥ parallel identical.

### 5.5 Training set

Train S1 is **2.2M rows / 7.64M GT pairs** — sample, don't exhaust: **~150–300K random train S1** (fixed seeds) × same Kmax=50 blocking + prune vs full train S2/S3 → ~7.5–15M raw pairs → **~0.5–1.0M positives** at measured density. **Negatives from real blocking output** (look-alikes), never random. Label from ground truth. Zero-candidate entities in separate accounting (still count in macro). Cache pairs (Parquet/float32). Any cleaning/feature change invalidates caches — version caches by feature-config hash.

### 5.6 Model: LightGBM (MIT)

Split **by S1 entity 80/20**, never by row (row split leaks + inflates). Config validated *(prior log)*:

```python
{'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
 'learning_rate':0.05,'num_leaves':127,'max_depth':-1,'min_child_samples':50,
 'feature_fraction':0.85,'bagging_fraction':0.85,'bagging_freq':5,
 'n_estimators':2500,'random_state':42,'n_jobs':16,'verbose':-1}
```

Early-stop on val log-loss (~iter 1800–2000; observed best 1864, 0.0135), then **refit on 100% entities at 1.1× best iters**. CPU by default; `device_type: 'cuda'` on the local 8 GB GPU if the CUDA build is set up (§7). Persist model + feature list + split manifest + OOF val predictions.

### 5.7 Selection (tune on metric, share code val/test)

```python
def select_matches(candidates, tau_singleton=0.68, tau_match=0.60, gamma_rel=0.70, max_matches=12):
    if not candidates:
        return []
    best = max(c['prob'] for c in candidates)
    if best < tau_singleton:
        return []                               # singleton gate: protect free 1.0
    thr = max(tau_match, best * gamma_rel)
    kept = sorted((c for c in candidates if c['prob'] >= thr),
                  key=lambda c: c['prob'], reverse=True)
    return [c['entity_id'] for c in kept[:max_matches]]
```

**Two must-fix details:**

1. **Sort before capping.** Selection is `p≥thr AND p≥rel·best`; the surviving set must be ordered by descending probability *before* `[:max_matches]`. Blocking order is by block score, not by model probability, so an unsorted slice silently keeps the wrong 12 candidates. The stub sliced in input order — ✅ **fixed in `src/matching/selection.py`.**
2. **Cap must be 12, not 10.** The stub previously read `max_matches=10` (with a "cap 10" docstring) — stale, and directly contradicted the measured GT max of 11 (37 train entities). ✅ **Fixed in `src/matching/selection.py` (now `max_matches=12`, result sorted by probability before the cap).**

**`max_matches=12` (not 10): measured GT max = 11** (37 entities at 11) — a cap of 10 silently truncates them. Grid-search `(tau_singleton, tau_match, gamma_rel)` maximizing macro F₀.₅ over all val entities incl. zero-candidate. Start `(0.65–0.68, 0.60–0.65, 0.70, 12)` → val 0.9425 *(prior log)*. Print every run: oracle / all-empty (0.0558) / top-1 baselines.

### 5.8 Test inference

100K-S1 batches: retrieve → prune → features → predict → select → append. **Checkpoint every batch**; resume skips finished. Log per-batch timings (pre-opt ~830s/5M-pair batch *(prior log)*). Load S2/S3 strings once, lazily per batch. Write candidates = pruned model-input set; matches ⊆ candidates. Validator auto-run + train dress-rehearsal scorer.

---

## 6. Validation & Experiment Protocol

Write the ~40-line macro F₀.₅ scorer first; sanity-check vs PDF example (P=2/3,R=1→0.714). Handle empty/empty=1.0, empty/non-empty=0.0. `src/matching/scorer.py` already implements `f05_entity`/`macro_f05` (17 LOC) — it needs a one-line unit test locking the PDF example (assert `≈0.7143`) plus edge cases (empty/empty=1.0, empty/non-empty=0.0) before it is trusted. Mandatory per-run diagnostics: oracle, all-empty (0.0558 measured), top-1, tuned rule + grid neighborhood (confirm flat optimum), per-country/source slices (**India-majority test; France 15%; India/S3 noisier**).

Ablation discipline: one change per experiment (a six-change bundle once net-hurt); fast harness (10K S1 vs true matches + ~1.5M distractors, in-memory) for relative ordering in minutes, confirm winner once at full scale.

Leakage checklist: S1-grouped splits; no `country` feature/rules; rule fit on val only; IDF/hashing fit on indexed corpus only (train index for train, test index for inference — unsupervised, allowed); no external data.

---

## 7. Compute Plan (hardware: 16 GB RAM + RTX 5060 Laptop 8 GB — both confirmed present)

Measured on this box: **16 GB physical RAM** → **14.4 GiB OS-visible** (`MemTotal` 15,125,468 kB; ~1.5 GB reserved by firmware/iGPU), **~6–7 GiB free** while the desktop is running, **4 GiB swap**; **RTX 5060 Laptop, 8,151 MiB VRAM**; 16 threads; 452 GB free disk. **Sufficient for this pipeline** — the whole design is sparse + chunked + float32 precisely so nothing dense is ever materialized (a dense 1.73M × 9.97M matrix is 1.7×10¹³ cells ≈ 68 TB, impossible on any box). The only thing that must be watched is resident working set during feature building and inference.

| Machine | Role |
|---|---|
| **Primary: local Ryzen 7 (16 threads), 16 GB RAM** | Everything classical: EDA (✅ done, ~1 min), chunked ingest, hashed sparse index (≤2M-row shards, 250K chunks, float32 caches — keep working set **<6 GiB**), pair features (8–16 procs), LightGBM training, full batched test inference. Nothing classical needs VRAM. |
| **GPU: RTX 5060 Laptop, 8 GB VRAM** | ✅ **verified** — CUDA was **not** available (no `nvcc`; the pip LightGBM wheel is `USE_GPU=ON` but not `USE_CUDA`), so the GPU is driven through LightGBM's **OpenCL** backend (`device_type='gpu'`, `ER_LGB_DEVICE=gpu`). Full 6.16M-pair training reproduced the CPU metrics (val F₀.₅ 0.9403 vs 0.9412; identical oracle/all-empty/top-1). Sparse hashed retrieval and rapidfuzz features stay CPU-bound (no GPU path). |
| Kaggle CPU (30 GB, 4 cores, 12h commit runs) | Optional overflow/backfill only — the local box carries the full pipeline. |

**RAM discipline (the one real constraint):** ≤2M-row shards, 250K-row chunks, `float32` sparse caches (`.npz`/Parquet) written to disk rather than held, 2²²-feature shard ≈ 0.6–1.5 GiB, bounded feature-worker waves. If RSS approaches the free budget, spill to disk — do **not** enlarge the in-memory join.

**Python environment (must be created first — currently absent).** Python 3.14.4 with **zero** packages installed and no venv. Create an isolated venv, pin versions, and verify imports before writing pipeline code; if a wheel is missing for 3.14 (`sparse-dot-topn` / `lightgbm` / `scipy` are the likely gaps), fall back to Python 3.12. `requirements.txt` currently lists pandas / numpy / scipy / scikit-learn / lightgbm / rapidfuzz / sparse-dot-topn / anyascii — it still needs **`pyarrow`** added (Parquet pair caches per §5.5). `scikit-learn` is already present and is required for `FeatureHasher`.

Env-var paths (`ER_DATA_DIR`, `ER_WORK_DIR`) for portability. Every stage idempotent (skip if output exists). Heavy runs checkpoint per batch; drafts die on idle and can lose work.

---

## 8. Optional Extensions (ranked; only after classical pipeline is stable + submitted)

1. **One-to-one assignment — legality VALIDATED by EDA** (7,638,365 GT pairs, 0 matched IDs claimed by two S1 rows: GT is a perfect matching). Contested candidate S2/S3 → highest-p claimant only. Cuts false merges between look-alike S1s. Highest-ranked extension; cheap to add in `selection.py`.
2. **Error-analysis loop** — split FP/FN by country×source, dump worst cases, one targeted fix at a time.
3. **GPU cross-encoder reranker** — `mdeberta-v3-base` fine-tune on ~0.5–1M train pairs w/ hard negatives; score top-12/S1 (~21M) or feed as LightGBM feature. **Runs on the local RTX 5060 (fp16/short pairs, 8 GB).** Literature (DeepMatcher/Magellan) says boosting stays competitive on noisy structured data — gap-closer for the 3.5-pt classifier gap, not foundation.
4. **Speed/quality trims** — higher LR + fewer trees, drop zero-importance features. Multi-view (name/address union) retrieval: similar recall, ~30% fewer pairs, more complexity — deferred.

Alternatives rejected/deferred: AND-key blocking (53% cap); dense TF-IDF KNN (OOM at 1.7M×10M); cross-encoder-alone (86M infeasible even on GPU); graph clustering (task is star-shaped, S1 deduped — and GT is a perfect matching, §3.2); MinHash-LSH viable fallback if sparse top-K stalls (arXiv 1905.06167); bi-encoder (multilingual-e5) optional semantic recall.

---

## 9. Execution Order (dependency-ordered, not a time budget)

| # | Goal | Done when | State |
|---|---|---|---|
| 1 | Scaffold, EDA §3, ingest+clean, format-exact empty baseline | validator PASS on full test; first safe submission | EDA ✅ · env ✅ · clean ✅ · baseline ✅ |
| 2 | Blocking v1 + eval harness (recall@K, missed dumps) | **locally measured** recall@50 ≥90% (not prior-log); pruned `candidate_pairs.tsv`; avg ≤25/S1 | ✅ recall@50 = **0.9436** measured; prune 0.9355 @ 30.8/S1 (§3.5); avg>25 is deliberate |
| 3 | Features + train-set + LightGBM + selection grid + scorer | val macro F₀.₅ ≥0.93 + oracle/baselines printed | ✅ val macro F₀.₅ = **0.9412**; oracle 0.9775 / all-empty 0.0549 / top-1 0.6783 |
| 4 | Full run (local 16T / GPU): index → train → batched infer | real `matching_results.tsv` validated | ✅ test index 579s; inference 2,013s; validator **PASS** (§3.6) |
| 5 | Error analysis + one-change fixes + re-tune | val improves or revert | ❌ |
| 6 | One-to-one (§8.1, validated) + stability (seeds/resume/timings) | final outputs from clean caches | ❌ |
| 7 | Freeze; end-to-end rerun from raw; validate; docs; zip | PASS + reproducible from `code/` alone | ✅ end-to-end run from raw + validator PASS; docs filled; `BubbleSort_submission.zip` (333 MB, 36 files) built |

Fork priority: **validity > recall > classifier > rule polish > extras.**

---

## 10. Risk Register

| Risk | P | Mitigation |
|---|---|---|
| Format rejection | M | validator every upload; trivial baseline first |
| pandas segfault/MemoryError (16 GB) | M | chunked reads (250K), ≤2M shards, float32 sparse caches, streaming predict; keep RSS <6 GiB |
| **No deps installed / Python 3.14 wheel gaps** | H | create venv first; verify imports; fall back to 3.12 if `sparse-dot-topn`/`lightgbm`/`scipy` wheels missing |
| Stale caches | M | config-hash versioning; rebuild on clean change |
| France failure (259K test S1 = 15%) | M | zero country-conditional logic; per-country IDF; simulate by holding out India |
| Non-Latin collapse | H if ignored | anyascii + skeleton (measured: 6.2%/7.4% of S2/S3 names vanish under ASCII-strip; 13.9% of matched names non-Latin); monitor slice recall |
| Missing-address pairs (4.4% of GT) | M | OR-semantics; name-only retrieval must work |
| Recall ceiling / pruned-recall loss | H | measure recall@K and pruned recall locally every run; oracle; K config; missed dumps |
| Feature parallel blowup | M | bounded waves, chunk-scoped workers, float32 |
| Quoting (`"""…`) / literal `N/A` | L | default quoting + `keep_default_na=False`; cleaning tests |
| matches ⊄ candidates | L | select only from candidates; assert + validator |
| Cardinality >10 (max 11 measured) | L | selection cap 12 (fix stale stub) |

---

## 11. Immediate Action Checklist (audited 2026-09-26 — item 1 done, everything else pending)

```
Dataset ──▶ [EDA verify ✅] ──▶ [venv + deps] ──▶ [Format-PASS baseline] ──▶ [Normalizer]
                                                                                  │
[Package] ◄── [Selection grid] ◄── [LightGBM] ◄── [Features/train-set] ◄── [Index + Kmax50 + prune]
```

1. ✅ **DONE — EDA** (`scripts/eda.py`): counts (§3.1), singleton 5.58%, matches mean 3.461 / max 11, non-Latin 13.9% of matched / ASCII-strip kills 6.2%, cross-country pairs **0%** (same-country 100%), one-to-one **legal** (0 contested IDs).
2. ✅ **DONE — environment**: `.venv` created (uv-managed CPython 3.14.4 — no wheel gaps, so no 3.12 fallback needed); `requirements.txt` installed (pyarrow already present); all 9 imports verified.
3. ✅ **DONE — format-exact empty baseline**: every test S1 → empty list in both TSVs → validator **PASS** (exit 0) on the full nested test set.
4. ✅ **DONE — `text_cleaner.py`**: anyascii (ASCII fast-path) + lower/ordinal/zero/numeral fold + legal-suffix core view + `@lru_cache` consonant skeleton; country-agnostic. 23 unit tests lock the §3.4 traps (`N/A` + `keep_default_na=False`, `"""` quoting, `F0ods`, `002050`, `2Nd`). Measured **82K rec/s single-thread** (incl. name+address views).
5. ✅ **DONE — blocking + eval** (`build_index.py`, `retrieve.py`, `evaluate_blocking.py`): Hasher(2²²) bag, per-country IDF, 6 shards for 10.32M train docs. **Measured** recall@50 = **0.9436** (target ≥0.935 ✓); pruned recall 0.9355 at 30.8 cand/S1 (avg >25 by deliberate recall-preserving choice, §3.5). Test `candidate_pairs.tsv` produced at inference (item 6).
6. ✅ **DONE — features + model + selection + inference** (`pair_features.py`, `build_training_set.py`, `train_matcher.py`, `selection.py`, `predict_test.py`): 38 features; 200K-S1 train sample → 6.16M pairs / 647K positives; S1-grouped 80/20; LightGBM early-stop @ iter 500; **val macro F₀.₅ = 0.9412** with (τ_single=0.70, τ_match=0.60, γ_rel=0.70, cap 12). Baselines: oracle **0.9775**, all-empty **0.0549**, top-1 **0.6783**. Full test inference (2,013 s, checkpointed per country) → `matching_results.tsv` + `candidate_pairs.tsv`; validator **PASS**.
7. ✅ **DONE — packaging**: `eda.py` copied to `code/business_entity_resolution/eda.py`; the organizer's `utils/validate_submission.py` bundled at `code/business_entity_resolution/utils/` and stated as unmodified; `Documentation_template.md` filled; `BubbleSort_submission.zip` built with `output/` + `code/` + docs. All README entry points resolve from the package root.

Success gates: licenses MIT/Apache-2.0 ✓ · offline-only ✓ · France generic ✓ · compact candidates ✓ · singleton gate ✓ · reproducible from `code/` alone.

---

## 12. References

- Problem statement + candidate-update PDFs (this repo, `docs/`).
- Shipped dataset EDA: `scripts/eda.py` (all §3 figures re-verified, reproducible).
- Top-3 pipeline/log for this challenge: `stack-ajit/business_entity_resolution` (`project_log.md`) + Top-3 playbook PDF; Unstop listing — **source of every *(prior log)* figure; not fetched or verified in this repo.**
- Blocking survey arXiv 1905.06167; DeepMatcher (SIGMOD'18); Magellan (CACM); SBERT cross-encoder training; HF reranker blog.
- Models/libs (MIT/Apache): LightGBM · sparse-dot-topn · rapidfuzz · anyascii · microsoft/mdeberta-v3-base.
- **`RESEARCH_GUIDE.md` / `RESEARCH_AND_IMPLEMENTATION_GUIDE.md` are cited in the header but do not exist in this repository** — they were part of an earlier synthesis. Treat their contents as folded into this plan; do not search for the files.
