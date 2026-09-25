"""All paths resolve from env vars so the same code runs locally and on Kaggle."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", "work"))

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

CACHE_DIR = WORK_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
BATCH_DIR = CACHE_DIR / "test_batches_k50_b100000"
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", "output"))

CHUNK_SIZE = 250_000
INDEX_SHARD_SIZE = 2_000_000
HASH_FEATURES = 2**22
K_MAX = 50
TEST_BATCH_SIZE = 100_000

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
