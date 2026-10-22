"""
robustness.py
Runs Isolation Forest with 5 different seeds and reports mean +/- std,
plus training time, inference time and model size.
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

# folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# settings
SEEDS = [21, 42, 55, 77, 100]
SPLIT_SEED = 42  # same split as the rest of the team, don't change

# best params from tuning_results.csv
IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": 16384,
    "contamination": 0.15,
}


def load_data():
    # read the csv files
    X_train = pd.read_csv(DATA_DIR / "train_normal.csv")
    test_df = pd.read_csv(DATA_DIR / "test_mixed.csv")
    types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0]

    # keep the same column order as training
    feature_cols = list(X_train.columns)
    X_all = test_df[feature_cols]
    y_all = test_df["Label"].astype(int)

    # split test data into validation and final test
    X_val, X_test, y_val, y_test = train_test_split(
        X_all, y_all,
        test_size=0.5,
        stratify=types,
        random_state=SPLIT_SEED,
    )
    return X_train, X_val, X_test, y_val, y_test


def compute_metrics(y_true, y_pred, scores):
    # 0 = normal, 1 = attack, higher score = more anomalous
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def run_isolation_forest(seed, X_train, X_test, y_test):
    # train and time it
    model = IsolationForest(**IF_PARAMS, random_state=seed, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X_train)
    train_time = time.perf_counter() - t0

    # threshold from training scores only (no test labels)
    train_scores = model.score_samples(X_train)
    threshold = np.percentile(train_scores, 100 * IF_PARAMS["contamination"])

    # score the test set and time it
    t0 = time.perf_counter()
    test_scores = model.score_samples(X_test)
    infer_time = time.perf_counter() - t0

    y_pred = (test_scores < threshold).astype(int)  # below threshold = attack
    anomaly_scores = -test_scores  # flip so higher = more anomalous

    # save to a temp file just to check the size
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(model, path)
        size_mb = path.stat().st_size / (1024 * 1024)

    results = compute_metrics(y_test, y_pred, anomaly_scores)
    results.update({
        "seed": seed,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "infer_ms_per_record": 1000 * infer_time / len(X_test),
        "model_size_mb": size_mb,
    })
    return results


def summarize(per_seed_df):
    # mean and std across seeds (ddof=1 = sample std)
    cols = [c for c in per_seed_df.columns if c != "seed"]
    return pd.DataFrame({
        "mean": per_seed_df[cols].mean(),
        "std": per_seed_df[cols].std(ddof=1),
    })


if __name__ == "__main__":
    X_train, X_val, X_test, y_val, y_test = load_data()

    rows = []
    for seed in SEEDS:
        print(f"Running Isolation Forest, seed {seed} ...")
        rows.append(run_isolation_forest(seed, X_train, X_test, y_test))

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)

    per_seed.to_csv(OUT_DIR / "isolation_forest_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "isolation_forest_summary.csv")

    print("\nPer-seed results:")
    print(per_seed.round(4).to_string(index=False))
    print("\nMean and std over seeds:")
    print(summary.round(4))