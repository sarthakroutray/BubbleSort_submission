# Business Entity Resolution

Amazon ML Challenge 2026 — link every Source-1 record to its matching Source-2/Source-3 records.

## Reproduce end-to-end

```bash
export ER_DATA_DIR=/path/to/dataset   # contains train/ and test/
export ER_WORK_DIR=/path/to/work      # writable caches + outputs

# 1. EDA sanity checks
python3 scripts/eda.py

# 2. Blocking: build index, retrieve Kmax=50, adaptive-prune → candidate_pairs.tsv
python3 -m src.blocking.build_index
python3 -m src.blocking.retrieve

# 3. Matching: features → training set → LightGBM → thresholds
python3 -m src.matching.build_training_set
python3 -m src.matching.train_matcher

# 4. Test inference (batched, checkpointed) → matching_results.tsv
python3 -m src.matching.predict_test

# 5. Validate (must print PASS, exit 0)
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$ER_DATA_DIR/test"
```

Outputs land in `output/`. Full spec: `FINAL_RESEARCH_AND_IMPLEMENTATION_PLAN.md` (repo root).

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── config.py                 # env-var paths
│   ├── cleaning/text_cleaner.py  # normalization (anyascii + skeleton + core name)
│   ├── blocking/                 # index, retrieval, pruning, evaluation
│   ├── matching/                 # features, training, selection, inference, scorer
│   └── submission/build_outputs.py
├── README.md
└── requirements.txt
```
