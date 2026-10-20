"""export_ae_scores.py - turn Member 2's saved Autoencoder into score files.

Does NOT retrain or change autoencoder.py. It loads outputs/autoencoder.keras and saves
reconstruction errors (higher = more anomalous) in the shared format for evaluate.py.

Run AFTER autoencoder.py:  python export_ae_scores.py
"""
import time

import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

from common import BASE_DIR, load_split, save_scores, save_timings

model_path = BASE_DIR / "outputs" / "autoencoder.keras"
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
# training time is not recorded by autoencoder.py; fill in if Member 2 / 5 measured it
save_timings("autoencoder", None, round(infer_time, 3), model_path, n_test=len(d["X_test"]))

# sanity check: same split as Member 2's script?
saved = BASE_DIR / "outputs" / "scores_autoencoder.npy"
if saved.exists():
    ok = np.allclose(np.load(saved), test_scores, rtol=1e-3, atol=1e-6)
    print("Test scores match Member 2's saved scores:", ok)
