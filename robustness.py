"""
robustness.py - Member 5
Robustness experiments: repeat training with several random seeds and
report mean +/- std of metrics, plus training time, inference time and model size.
"""

import time
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ---------- Paths (relative to this file, not to where you run it) ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Experiment settings ----------
SEEDS = [21, 42, 55, 77, 100]
SPLIT_SEED = 42   # the team's fixed validation/test split seed (do NOT change)

IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": 16384,
    "contamination": 0.15,
}

if __name__ == "__main__":
    print("BASE_DIR :", BASE_DIR)
    print("DATA_DIR :", DATA_DIR, "| exists:", DATA_DIR.exists())
    print("OUT_DIR  :", OUT_DIR, "| exists:", OUT_DIR.exists())
    print("Seeds    :", SEEDS)
    print("IF params:", IF_PARAMS)
