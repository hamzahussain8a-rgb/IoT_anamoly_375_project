"""export_ae_scores.py - turn Member 2's saved Autoencoder into score files.

Does NOT retrain or change autoencoder.py. It loads outputs/autoencoder.keras and saves
reconstruction errors (higher = more anomalous) in the shared format for evaluate.py.

Run AFTER autoencoder.py:  python export_ae_scores.py
"""
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore

from common import REPO_DIR, load_split, save_scores, save_timings

model_path = REPO_DIR / "outputs" / "autoencoder.keras"
model = load_model(model_path)
d = load_split()


def errors(X):
    """Mean squared reconstruction error per record."""
    X = X.to_numpy(dtype=np.float32)
    return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)


t0 = time.time()
test_scores = errors(d["X_test"])
infer_time = time.time() - t0

save_scores("autoencoder", errors(d["X_train"]), errors(d["X_val"]), test_scores)
# training time comes from outputs/autoencoder_summary.csv (written by Member 2's updated script)
train_time = None
summary = REPO_DIR / "outputs" / "autoencoder_summary.csv"
if summary.exists():
    train_time = round(float(pd.read_csv(summary)["training_time_seconds"].iloc[0]), 2)
save_timings("autoencoder", train_time, round(infer_time, 3), model_path, n_test=len(d["X_test"]))

# sanity check: same split as Member 2's script?
saved = REPO_DIR / "outputs" / "scores_autoencoder.npy"
if saved.exists():
    ok = np.allclose(np.load(saved), test_scores, rtol=1e-3, atol=1e-6)
    print("Test scores match Member 2's saved scores:", ok)
