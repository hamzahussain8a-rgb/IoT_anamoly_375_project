"""export_lstm_scores.py - turn Member 3's saved LSTM autoencoder into score files.

Does NOT retrain or change lstm_autoencoder.py. It loads outputs/lstm_autoencoder.keras
and saves reconstruction errors (higher = more anomalous) in the shared format.
Training time is read from outputs/lstm_summary.csv (written by Member 3's script).

Run AFTER lstm_autoencoder.py:  python export_lstm_scores.py
"""
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore

from common import REPO_DIR, load_split, save_scores, save_timings

model_path = REPO_DIR / "outputs" / "lstm_autoencoder.keras"
model = load_model(model_path)
d = load_split()


def errors(X):
    """Mean squared reconstruction error per record (sequence length 1, as in Member 3's script)."""
    X = X.to_numpy(dtype=np.float32).reshape(-1, 1, X.shape[1])
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=(1, 2))


t0 = time.time()
test_scores = errors(d["X_test"])
infer_time = time.time() - t0

train_time = None
summary = REPO_DIR / "outputs" / "lstm_summary.csv"
if summary.exists():
    train_time = round(float(pd.read_csv(summary)["training_time_seconds"].iloc[0]), 2)

save_scores("lstm_autoencoder", errors(d["X_train"]), errors(d["X_val"]), test_scores)
save_timings("lstm_autoencoder", train_time, round(infer_time, 3), model_path,
             n_test=len(d["X_test"]))

saved = REPO_DIR / "outputs" / "scores_lstm.npy"
if saved.exists():
    print("Test scores match Member 3's saved scores:",
          np.allclose(np.load(saved), test_scores, rtol=1e-3, atol=1e-6))
