"""ocsvm.py - One-Class SVM baseline (Member 4).

Trained on NORMAL traffic only. Uses the same validation / final-test split as
isolation_forest.py (via common.py). Labels are used only to choose nu on the
validation half.

Run:  python ocsvm.py
"""
import time

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from common import BASE_DIR, SEED, load_split, save_scores, save_timings

N_TRAIN = 25_000            # OCSVM is slow on big data; lower this (e.g. 10_000) if it takes too long
NU_VALUES = [0.01, 0.05, 0.1, 0.2]

d = load_split()
X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
y_val = d["y_val"]
print(f"train {X_train.shape}, validation {X_val.shape}, final test {X_test.shape}")

X_fit = X_train.sample(n=min(N_TRAIN, len(X_train)), random_state=SEED)


def anomaly_score(model, X):
    return -model.decision_function(X)       # higher = more anomalous


# 1) choose nu by validation ROC-AUC
best_auc, best_nu = -1.0, NU_VALUES[0]
for nu in NU_VALUES:
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(X_fit)
    auc = roc_auc_score(y_val, anomaly_score(m, X_val))
    print(f"nu={nu:<5} validation ROC-AUC = {auc:.4f}")
    if auc > best_auc:
        best_auc, best_nu = auc, nu
print(f"Chosen nu = {best_nu}")

# 2) final model (timed)
t0 = time.time()
model = OneClassSVM(kernel="rbf", gamma="scale", nu=best_nu).fit(X_fit)
train_time = time.time() - t0

t0 = time.time()
test_scores = anomaly_score(model, X_test)
infer_time = time.time() - t0

save_scores("ocsvm", anomaly_score(model, X_train), anomaly_score(model, X_val), test_scores)
model_path = BASE_DIR / "ocsvm_model.joblib"
joblib.dump(model, model_path)
save_timings("ocsvm", round(train_time, 2), round(infer_time, 3), model_path,
             nu=best_nu, n_train=len(X_fit), n_test=len(X_test))
