# Business Entity Resolution

Amazon ML Challenge 2026 — link every Source-1 record to its matching Source-2/Source-3 records.

Deliverables: `output/matching_results.tsv` (scored) + `output/candidate_pairs.tsv`
(blocking audit) — both tab-separated, one row per test S1.

## Reproduce end-to-end

```bash
# Dataset is nested after extraction; export the dir that holds train/ and test/
export ER_DATA_DIR=dataset/6ab10eb3b23ba_student_resource/student_resource/dataset
export ER_WORK_DIR=work          # writable caches (index, train set, checkpoints)
export ER_OUTPUT_DIR=output

# 0. Environment (uv-managed CPython 3.14 works; no 3.12 fallback needed)
uv venv .venv && . .venv/bin/activate     # or: python3 -m venv .venv
pip install -r code/business_entity_resolution/requirements.txt
# All pipeline commands run from the package root so `src` is importable:
cd code/business_entity_resolution

# 1. EDA sanity checks (stdlib only; bundled copy of scripts/eda.py)
python3 eda.py "$ER_DATA_DIR"

# 2. Blocking: hashed index per corpus, then measured recall@K on a train sample
python3 -m src.blocking.build_index --corpus train      # ~9 min, 6 shards
python3 -m src.blocking.evaluate_blocking --sample 40000 --kmax 100

# 3. Matching: training set → LightGBM + selection grid
python3 -m src.matching.build_training_set --sample 200000
python3 -m src.matching.train_matcher                    # add --device gpu to train on the GPU

# 4. Test index + batched, checkpointed inference → both output TSVs
python3 -m src.blocking.build_index --corpus test
python3 -m src.matching.predict_test                     # resumes finished countries

# 5. Validate (must print PASS, exit 0) — bundled copy of the organizer's validator,
#    run against the unmodified test set
python3 utils/validate_submission.py \
  --matching "$ER_OUTPUT_DIR/matching_results.tsv" \
  --candidate "$ER_OUTPUT_DIR/candidate_pairs.tsv" \
  --test-dir "$ER_DATA_DIR/test"

# 6. Unit tests (cleaner traps + F0.5 scorer)
python3 -m unittest discover -s tests -t .
```

GPU: `ER_LGB_DEVICE=gpu` (or `train_matcher --device gpu`) drives LightGBM's OpenCL
backend on the RTX 5060 — verified to reproduce the CPU metrics (val macro F0.5
0.9403 vs 0.9412). CUDA is not compiled into the pip wheel, so `device_type='gpu'`
(OpenCL) is used. Feature building and sparse retrieval stay CPU-bound.

Formatted empty baseline (pins the output contract, no model needed):

```bash
cd code/business_entity_resolution
python3 -m src.submission.build_outputs --empty
```

## Measured results (2026-09-26)

- Blocking recall@50 = **0.9436** (K=100: 0.9527); pruned candidate recall **0.9355**
  at ~30.8 candidates/S1 (40K-sample, full 10.32M-row train index). See plan §3.5.
- Raw recall@K: 1:0.257 · 5:0.819 · 10:0.901 · 20:0.926 · 50:0.9436 · 100:0.9527.

## Layout

```
code/business_entity_resolution/
├── src/
│   ├── config.py                 # env-var paths (ER_DATA_DIR / ER_WORK_DIR / ER_OUTPUT_DIR)
│   ├── cleaning/text_cleaner.py  # normalization (anyascii + skeleton + core name)
│   ├── blocking/                 # build_index, retrieve, evaluate_blocking, ablate_blocking
│   ├── matching/                 # pair_features, build_training_set, train_matcher,
│   │                             #   selection, scorer, predict_test
│   └── submission/build_outputs.py
├── tests/                        # unittest: §3.4 traps + macro F0.5 PDF example
├── README.md
└── requirements.txt
```
