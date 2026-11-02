"""
robustness.py

Repeats each model with 5 random seeds on the same fixed validation/test split
and reports mean +/- std of accuracy, balanced accuracy, precision, recall,
F1 and ROC-AUC, plus training time, inference time and model size.
"""
import sys
import json
import time
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tensorflow.keras import Model  # type:ignore
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed  # type:ignore
from tensorflow.keras.callbacks import EarlyStopping  # type:ignore

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "outputs" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [21, 42, 55, 77, 100]
SPLIT_SEED = 42

IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": 16384,
    "contamination": 0.15,
}

THRESH_PCT = 95
BATCH_SIZE = 64
PATIENCE = 15
VAL_FRACTION = 0.10


def load_data():
    X_train = pd.read_csv(DATA_DIR / "train_normal.csv")
    test_df = pd.read_csv(DATA_DIR / "test_mixed.csv")
    types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0]

    feature_cols = list(X_train.columns)
    X_all = test_df[feature_cols]
    y_all = test_df["Label"].astype(int)

    X_val, X_test, y_val, y_test = train_test_split(
        X_all, y_all,
        test_size=0.5,
        stratify=types,
        random_state=SPLIT_SEED,
    )
    return X_train, X_val, X_test, y_val, y_test


def compute_metrics(y_true, y_pred, scores):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def make_row(seed, y_test, y_pred, scores, train_time, infer_time, size_mb):
    row = compute_metrics(y_test, y_pred, scores)
    row.update({
        "seed": seed,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "infer_ms_per_record": 1000 * infer_time / len(y_test),
        "model_size_mb": size_mb,
    })
    return row


def size_of(save_fn, filename):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        save_fn(path)
        return path.stat().st_size / (1024 * 1024)


def run_isolation_forest(seed, X_train, X_test, y_test):
    model = IsolationForest(**IF_PARAMS, random_state=seed, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X_train)
    train_time = time.perf_counter() - t0

    train_scores = model.score_samples(X_train)
    threshold = np.percentile(train_scores, 100 * IF_PARAMS["contamination"])

    t0 = time.perf_counter()
    test_scores = model.score_samples(X_test)
    infer_time = time.perf_counter() - t0

    y_pred = (test_scores < threshold).astype(int)
    size_mb = size_of(lambda p: joblib.dump(model, p), "model.joblib")
    return make_row(seed, y_test, y_pred, -test_scores, train_time, infer_time, size_mb)


def get_ocsvm_nu():
    with open(BASE_DIR / "scores" / "ocsvm_timings.json") as f:
        return json.load(f)["nu"]


def run_ocsvm(seed, nu, X_train, X_test, y_test):
    X_fit = X_train.sample(frac=0.8, random_state=seed)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    t0 = time.perf_counter()
    model.fit(X_fit)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = -model.decision_function(X_test)
    infer_time = time.perf_counter() - t0

    y_pred = (scores > 0).astype(int)
    size_mb = size_of(lambda p: joblib.dump(model, p), "model.joblib")
    return make_row(seed, y_test, y_pred, scores, train_time, infer_time, size_mb)


def get_ae_bottleneck():
    saved = tf.keras.models.load_model(BASE_DIR / "outputs" / "autoencoder.keras")
    return min(l.units for l in saved.layers if isinstance(l, Dense))


def build_autoencoder(n_features, bottleneck):
    inp = Input(shape=(n_features,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    x = Dense(bottleneck, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(n_features, activation="linear")(x)
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m


def build_lstm(n_features):
    inp = Input(shape=(1, n_features))
    x = LSTM(64, return_sequences=True)(inp)
    x = LSTM(32, return_sequences=False)(x)
    x = RepeatVector(1)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m


def run_keras(seed, build_fn, epochs, X_fit, X_valn, X_test, y_test):
    tf.keras.utils.set_random_seed(seed)
    model = build_fn()
    stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    t0 = time.perf_counter()
    model.fit(X_fit, X_fit, validation_data=(X_valn, X_valn), epochs=epochs,
              batch_size=BATCH_SIZE, shuffle=True, callbacks=[stop], verbose=0)
    train_time = time.perf_counter() - t0

    def errors(data):
        rec = model.predict(data, verbose=0)
        return np.mean(np.square(data - rec), axis=tuple(range(1, data.ndim)))

    threshold = np.percentile(errors(X_valn), THRESH_PCT)

    errors(X_test[:100])
    t0 = time.perf_counter()
    test_err = errors(X_test)
    infer_time = time.perf_counter() - t0

    y_pred = (test_err > threshold).astype(int)
    size_mb = size_of(model.save, "model.keras")
    return make_row(seed, y_test, y_pred, test_err, train_time, infer_time, size_mb)


def summarize(per_seed_df):
    cols = [c for c in per_seed_df.columns if c != "seed"]
    return pd.DataFrame({
        "mean": per_seed_df[cols].mean(),
        "std": per_seed_df[cols].std(ddof=1),
    })


def save_results(name, rows):
    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(OUT_DIR / f"{name}_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / f"{name}_summary.csv")
    print(f"\n=== {name}: mean and std over {len(rows)} seeds ===")
    print(summary.round(4))


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["isolation_forest", "autoencoder", "lstm_autoencoder", "ocsvm"]
    X_train, X_val, X_test, y_val, y_test = load_data()

    Xtr = X_train.to_numpy(dtype=np.float32)
    Xte = X_test.to_numpy(dtype=np.float32)
    n_feat = Xtr.shape[1]
    X_fit, X_valn = train_test_split(Xtr, test_size=VAL_FRACTION, random_state=SPLIT_SEED)

    if "isolation_forest" in wanted:
        rows = []
        for s in SEEDS:
            print(f"isolation_forest, seed {s} ...")
            rows.append(run_isolation_forest(s, X_train, X_test, y_test))
        save_results("isolation_forest", rows)

    if "autoencoder" in wanted:
        bn = get_ae_bottleneck()
        print("autoencoder bottleneck size (from saved model):", bn)
        rows = []
        for s in SEEDS:
            print(f"autoencoder, seed {s} ...")
            rows.append(run_keras(s, lambda: build_autoencoder(n_feat, bn), 300,
                                  X_fit, X_valn, Xte, y_test))
        save_results("autoencoder", rows)

    if "lstm" in wanted or "lstm_autoencoder" in wanted:
        to_seq = lambda a: a.reshape(-1, 1, n_feat)
        rows = []
        for s in SEEDS:
            print(f"lstm_autoencoder, seed {s} ...")
            rows.append(run_keras(s, lambda: build_lstm(n_feat), 100,
                                  to_seq(X_fit), to_seq(X_valn), to_seq(Xte), y_test))
        save_results("lstm_autoencoder", rows)

    if "ocsvm" in wanted:
        nu = get_ocsvm_nu()
        print("ocsvm nu (from scores/ocsvm_timings.json):", nu)
        rows = []
        for s in SEEDS:
            print(f"ocsvm, seed {s} ...")
            rows.append(run_ocsvm(s, nu, X_train, X_test, y_test))
        save_results("ocsvm", rows)