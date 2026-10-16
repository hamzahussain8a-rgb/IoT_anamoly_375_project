"""
isolation_forest.py
Train an Isolation Forest on NORMAL traffic only, tune
n_estimators / max_samples / contamination, then evaluate on a held-out test set.

Tuning protocol (avoids cheating):
  test_mixed.csv is split 50/50 (stratified by attack type, so both halves contain
  every attack type) into
    - VALIDATION : used to pick the best hyperparameters
    - FINAL TEST : used once, at the end, for the reported results
"""

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
print("Loading prepared datasets...")
X_train = pd.read_csv(DATA_DIR / "train_normal.csv")
test = pd.read_csv(DATA_DIR / "test_mixed.csv")
types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0].astype(str)

y_all = test["Label"]
X_all = test.drop(columns=["Label"])[X_train.columns]      # same column order
assert len(types) == len(test), "test_attack_types.csv does not match test_mixed.csv"

# 50/50 split, stratified by attack type (this also keeps the normal/attack ratio equal)
X_val, X_test, y_val, y_test, t_val, t_test = train_test_split(
    X_all, y_all, types, test_size=0.5, stratify=types, random_state=RANDOM_SEED
)

print(f"Train (normal only): {len(X_train):,}")
print(f"Validation: {len(X_val):,}  (normal: {int((y_val == 0).sum()):,}, attacks: {int(y_val.sum()):,})")
print(f"Final test: {len(X_test):,}  (normal: {int((y_test == 0).sum()):,}, attacks: {int(y_test.sum()):,})")
print(f"Features: {X_train.shape[1]}")
print("Baseline F1 on validation (predict everything = attack):",
      round(f1_score(y_val, [1] * len(y_val)), 4))


def predict_with_contamination(model, X, train_scores, contamination):
    """Same rule scikit-learn uses inside model.predict():
    the threshold is the `contamination` percentile of the TRAINING scores;
    anything scoring below it is an anomaly. 1 = attack, 0 = normal."""
    threshold = np.percentile(train_scores, 100 * contamination)
    return (model.score_samples(X) < threshold).astype(int)


# ----------------------------------------------------------------------
# 2. Hyperparameter grid search on the validation set.
#    * n_estimators and max_samples change the model -> ranked by ROC-AUC
#      (threshold-free).
#    * contamination only moves the decision threshold, not the trees, so each
#      forest is trained ONCE and every contamination value is tried on its scores.
#    * Among equal-AUC settings (every contamination of the same forest), the
#      threshold is picked by BALANCED accuracy.
# ----------------------------------------------------------------------
n_train = len(X_train)
max_samples_options = [m for m in (256, 512, 1024, 2048, 4096, 8192, 16384) if m < n_train]
max_samples_options.append(n_train)          # last value = use every training row

n_estimators_options = [100, 200, 300]
contamination_options = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

n_forests = len(n_estimators_options) * len(max_samples_options)
print(f"\nTuning: {n_forests} forests x {len(contamination_options)} contamination values "
      f"= {n_forests * len(contamination_options)} combinations...\n")

results = []
for n_est, max_s in itertools.product(n_estimators_options, max_samples_options):
    model = IsolationForest(
        n_estimators=n_est,
        max_samples=max_s,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(X_train)                                   # normal traffic only

    train_scores = model.score_samples(X_train)
    auc = roc_auc_score(y_val, -model.score_samples(X_val))   # independent of contamination

    for cont in contamination_options:
        pred = predict_with_contamination(model, X_val, train_scores, cont)
        results.append({
            "n_estimators": n_est,
            "max_samples": max_s,
            "contamination": cont,
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
            "accuracy": accuracy_score(y_val, pred),
            "balanced_acc": balanced_accuracy_score(y_val, pred),
            "roc_auc": auc,
        })
    print(f"n_est={n_est:<4} max_samples={max_s:<6} AUC={auc:.4f}")

results_df = pd.DataFrame(results)
# AUC rounded to 4 decimals so near-identical forests are decided by balanced accuracy
results_df["auc_rank"] = results_df["roc_auc"].round(4)
results_df = (results_df.sort_values(["auc_rank", "balanced_acc"], ascending=False)
              .drop(columns="auc_rank"))
results_df.to_csv(BASE_DIR / "tuning_results.csv", index=False)

print("\n--- TOP 5 CONFIGURATIONS (validation) ---")
print(results_df.head(5).round(4).to_string(index=False))

best = results_df.iloc[0]
best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_samples": int(best["max_samples"]),
    "contamination": float(best["contamination"]),
}
print("\nBest parameters:", best_params)

# ----------------------------------------------------------------------
# 3. Retrain best model and evaluate ONCE on the final test set
# ----------------------------------------------------------------------
final_model = IsolationForest(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
final_model.fit(X_train)

y_pred = (final_model.predict(X_test) == -1).astype(int)     # 1 = attack, 0 = normal

print("\n--- FINAL TEST RESULTS ---")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, -final_model.score_samples(X_test)), 4))

print("\nConfusion matrix (rows = actual [normal, attack], "
      "cols = predicted [normal, attack]):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["Normal", "Attack"], zero_division=0,
))

# ----------------------------------------------------------------------
# 4. Per-attack-type results (final test set)
#    Attack rows: recall = share of that attack type correctly flagged.
#    Normal rows: "recall" is the share correctly left alone (1 - false-alarm rate).
#    AUC per type = that attack type vs ALL normal rows (threshold-free).
# ----------------------------------------------------------------------
anomaly_score = -final_model.score_samples(X_test)
normal_mask = (t_test == "Normal").to_numpy()
rows = []
for t in sorted(t_test.unique(), key=lambda s: (s == "Normal", s)):
    m = (t_test == t).to_numpy()
    row = {"type": t, "rows": int(m.sum()), "correct_rate": float((y_pred[m] == y_test.to_numpy()[m]).mean())}
    if t != "Normal":
        both = m | normal_mask
        row["roc_auc_vs_normal"] = float(roc_auc_score(y_test.to_numpy()[both], anomaly_score[both]))
    rows.append(row)
per_type = pd.DataFrame(rows)
print("\n--- PER-TYPE RESULTS (final test) ---")
print(per_type.round(4).to_string(index=False))
per_type.to_csv(BASE_DIR / "per_attack_results.csv", index=False)

joblib.dump(final_model, BASE_DIR / "isolation_forest_model.joblib")
print("\nSaved model: isolation_forest_model.joblib")
print("Saved tuning table: tuning_results.csv")
print("Saved per-attack table: per_attack_results.csv")
