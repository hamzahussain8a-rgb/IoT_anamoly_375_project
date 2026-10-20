"""export_if_scores.py - turn Member 1's saved Isolation Forest into score files.

Does NOT retrain or change isolation_forest.py. It loads isolation_forest_model.joblib
and saves scores in the shared format so evaluate.py can compare it with other models.

Run AFTER isolation_forest.py:  python export_if_scores.py
"""
import time

import joblib

from common import BASE_DIR, load_split, save_scores, save_timings

model_path = BASE_DIR / "isolation_forest_model.joblib"
model = joblib.load(model_path)
d = load_split()


def anomaly_score(X):
    return -model.score_samples(X)           # higher = more anomalous


t0 = time.time()
test_scores = anomaly_score(d["X_test"])
infer_time = time.time() - t0

save_scores("isolation_forest", anomaly_score(d["X_train"]),
            anomaly_score(d["X_val"]), test_scores)
# training time is not stored in the model file; fill it in if Member 1 / 5 measured it
save_timings("isolation_forest", None, round(infer_time, 3), model_path,
             n_test=len(d["X_test"]))
