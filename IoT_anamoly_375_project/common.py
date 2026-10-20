"""common.py - shared helpers so every model uses the SAME data split and output format.

The split reproduces isolation_forest.py exactly (same call, same seed):
    test_mixed.csv is cut 50/50, stratified by attack type, into
      VALIDATION  - used to choose hyperparameters / thresholds
      FINAL TEST  - used once, for the reported results

Every model must call save_scores() so evaluate.py can compare them fairly.
Convention: HIGHER score = MORE anomalous.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
SCORES_DIR = BASE_DIR / "scores"
SEED = 42


def load_split():
    """Return a dict with X_train, X_val, X_test, y_val, y_test, t_val, t_test."""
    X_train = pd.read_csv(DATA_DIR / "train_normal.csv")
    test = pd.read_csv(DATA_DIR / "test_mixed.csv")
    types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0].astype(str)
    assert len(types) == len(test), "test_attack_types.csv does not match test_mixed.csv"

    y_all = test["Label"]
    X_all = test.drop(columns=["Label"])[X_train.columns]
    X_val, X_test, y_val, y_test, t_val, t_test = train_test_split(
        X_all, y_all, types, test_size=0.5, stratify=types, random_state=SEED
    )
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_val=y_val, y_test=y_test, t_val=t_val, t_test=t_test)


def save_scores(model_name, train_scores, val_scores, test_scores):
    """Save anomaly scores (higher = more anomalous) for the three data parts."""
    SCORES_DIR.mkdir(exist_ok=True)
    for part, s in (("train", train_scores), ("val", val_scores), ("test", test_scores)):
        np.save(SCORES_DIR / f"{model_name}_{part}.npy", np.asarray(s, dtype=float))
    print(f"Saved scores for '{model_name}' in {SCORES_DIR}")


def save_timings(model_name, train_time_s, infer_time_s, model_path=None, **extra):
    """Save training time, inference time (on the final test set) and model size."""
    SCORES_DIR.mkdir(exist_ok=True)
    size_kb = round(os.path.getsize(model_path) / 1024, 1) if model_path else None
    info = {"train_time_s": train_time_s, "infer_time_s": infer_time_s,
            "model_size_kb": size_kb, **extra}
    with open(SCORES_DIR / f"{model_name}_timings.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Timings:", info)
