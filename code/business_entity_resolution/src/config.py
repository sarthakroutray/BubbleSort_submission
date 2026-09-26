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

# LightGBM device. "gpu" uses the OpenCL backend (the shipped wheel is built with
# USE_GPU=ON but not USE_CUDA), which drives the NVIDIA GPU via the driver's
# OpenCL ICD; "cpu" is the fallback and produces identical metrics (plan §7).
LGB_DEVICE = os.environ.get("ER_LGB_DEVICE", "cpu").strip().lower()


def lgb_device_params(device: str = None) -> dict:
    device = (device or LGB_DEVICE or "cpu").lower()
    if device in ("gpu", "cuda"):
        return {"device_type": "gpu", "gpu_platform_id": 0,
                "gpu_device_id": 0, "gpu_use_dp": False}
    return {"device_type": "cpu"}
