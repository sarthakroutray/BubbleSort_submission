"""All paths resolve from env vars so the same code runs locally and on Kaggle.

Defaults are anchored to the project root (not the cwd), so every entry point
works regardless of where it is invoked from.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # <root>/code/business_entity_resolution/src/config.py


def _path(env: str, default: str) -> Path:
    raw = os.environ.get(env, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


DATA_DIR = _path("ER_DATA_DIR", "dataset")
WORK_DIR = _path("ER_WORK_DIR", "work")
OUTPUT_DIR = _path("ER_OUTPUT_DIR", "output")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

CACHE_DIR = WORK_DIR / "cache"
INDEX_DIR = CACHE_DIR / "index"
MODEL_DIR = CACHE_DIR / "model"
BATCH_DIR = CACHE_DIR / "test_batches_k50_b100000"

CHUNK_SIZE = 250_000
INDEX_SHARD_SIZE = 2_000_000
HASH_FEATURES = 2**22
K_MAX = 50
TEST_BATCH_SIZE = 100_000

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"

# Test-corpus country mix (EDA, plan §3): held-out val is reweighted with this so
# the local metric tracks the leaderboard distribution (train val is US-heavy).
TEST_COUNTRY_MIX = {"India": 0.467, "US": 0.383, "France": 0.15}

# LightGBM compute backend. Default is the CUDA tree learner (USE_CUDA build,
# native sm_120 kernels for the RTX 5060); train()/predict fall back to CPU
# automatically if CUDA is unavailable. "gpu" = legacy OpenCL backend (refit
# unstable -> CPU refit is forced in train_matcher). Override via ER_LGB_DEVICE.
LGB_DEVICE = os.environ.get("ER_LGB_DEVICE", "cuda").strip().lower()


def lgb_device_params(device: str = None) -> dict:
    device = (device or LGB_DEVICE or "cpu").lower()
    if device == "cuda":
        return {"device_type": "cuda"}
    if device == "gpu":
        return {"device_type": "gpu", "gpu_platform_id": 0,
                "gpu_device_id": 0, "gpu_use_dp": False}
    return {"device_type": "cpu"}
