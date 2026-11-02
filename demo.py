"""
demo.py
Loads the saved model and tests a few IoT records.
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

IF_PATH = BASE_DIR / "isolation_forest_model.joblib"
OCSVM_PATH = BASE_DIR / "ocsvm_model.joblib"
AE_PATH = BASE_DIR / "outputs" / "autoencoder.keras"
LSTM_PATH = BASE_DIR / "outputs" / "lstm_autoencoder.keras"

SPLIT_SEED = 42
THRESH_PCT = 95
VAL_FRACTION = 0.10
NORMAL_ROWS = 3
MAX_CUSTOM_ROWS = 20


def read_csv_checked(path):
    if not path.exists():
        sys.exit(f"File not found: {path}")
    return pd.read_csv(path)


def pick_samples(feature_cols):
    test_df = read_csv_checked(DATA_DIR / "test_mixed.csv")
    types = read_csv_checked(DATA_DIR / "test_attack_types.csv").iloc[:, 0].astype(str)

    X_all = test_df[feature_cols]
    y_all = test_df["Label"].astype(int)
    _, X_test, _, _, _, t_test = train_test_split(
        X_all, y_all, types,
        test_size=0.5,
        stratify=types,
        random_state=SPLIT_SEED,
    )
    X_test = X_test.reset_index(drop=True)
    t_test = t_test.reset_index(drop=True)

    rng = np.random.RandomState(SPLIT_SEED)
    names = sorted(t_test.unique(), key=lambda n: (n != "Normal", n))
    chosen = []
    for name in names:
        idx = np.flatnonzero(t_test.to_numpy() == name)
        n = NORMAL_ROWS if name == "Normal" else 1
        chosen.extend(rng.choice(idx, size=min(n, len(idx)), replace=False))

    return X_test.iloc[chosen].reset_index(drop=True), t_test.iloc[chosen].reset_index(drop=True)


def load_custom(path, feature_cols):
    df = read_csv_checked(Path(path))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"Your CSV is missing {len(missing)} required column(s), for example: {missing[:3]}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        sys.exit("Your CSV has empty or non-numeric values in the feature columns.")

    if "Label" in df.columns:
        labels = df["Label"].map({0: "Normal", 1: "Attack"}).fillna("?")
    else:
        labels = pd.Series(["?"] * len(df))

    return X.head(MAX_CUSTOM_ROWS).reset_index(drop=True), labels.head(MAX_CUSTOM_ROWS).reset_index(drop=True)


def load_sklearn_model(path):
    model = joblib.load(path)
    return lambda X: model.predict(X) == -1


def load_keras_model(path, sequence, X_valn):
    import tensorflow as tf

    model = tf.keras.models.load_model(path)

    def errors(data):
        arr = np.asarray(data, dtype=np.float32)
        if sequence:
            arr = arr.reshape(-1, 1, arr.shape[1])
        rec = model.predict(arr, verbose=0)
        return np.mean(np.square(arr - rec), axis=tuple(range(1, arr.ndim)))

    threshold = np.percentile(errors(X_valn), THRESH_PCT)
    return lambda X: errors(X) > threshold


def load_all_models(X_train_np):
    _, X_valn = train_test_split(X_train_np, test_size=VAL_FRACTION, random_state=SPLIT_SEED)

    jobs = [
        ("Isolation Forest", IF_PATH, lambda p: load_sklearn_model(p)),
        ("Autoencoder", AE_PATH, lambda p: load_keras_model(p, False, X_valn)),
        ("LSTM Autoencoder", LSTM_PATH, lambda p: load_keras_model(p, True, X_valn)),
        ("One-Class SVM", OCSVM_PATH, lambda p: load_sklearn_model(p)),
    ]

    models = {}
    for name, path, loader in jobs:
        if not path.exists():
            print(f"[skipped] {name}: file not found ({path.name})")
            continue
        try:
            models[name] = loader(path)
        except Exception as err:
            print(f"[skipped] {name}: could not load ({type(err).__name__}: {err})")
    return models


def main():
    train = read_csv_checked(DATA_DIR / "train_normal.csv")
    feature_cols = list(train.drop(columns=["Label"], errors="ignore").columns)
    X_train_np = train[feature_cols].to_numpy(dtype=np.float32)

    if len(sys.argv) > 1:
        X, actual = load_custom(sys.argv[1], feature_cols)
        print(f"\nUsing your file: {sys.argv[1]} ({len(X)} rows)")
    else:
        X, actual = pick_samples(feature_cols)
        print("\nUsing sample rows from the final test set")

    models = load_all_models(X_train_np)
    if not models:
        sys.exit("No models could be loaded, nothing to demo.")

    table = pd.DataFrame({"Row": range(1, len(X) + 1), "Actual": actual})
    flags = {}
    for name, predict in models.items():
        flags[name] = np.asarray(predict(X), dtype=bool)
        table[name] = np.where(flags[name], "ANOMALY", "Normal")

    print("\nIoT Anomaly Detection Demo")
    print("--------------------------")
    print(table.to_string(index=False))

    if (actual != "?").all():
        truth = (actual != "Normal").to_numpy()
        print("\nCorrect predictions on these rows:")
        for name, flag in flags.items():
            print(f"  {name}: {int((flag == truth).sum())} / {len(X)}")

    print("\nNote: each model flags a row when its anomaly score crosses a fixed")
    print("threshold. The scores are not probabilities, and a few rows do not")
    print("measure real-world performance.")


if __name__ == "__main__":
    main()