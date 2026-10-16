"""
diagnose.py - where does the Isolation Forest succeed or fail?
Uses the saved model + test files. Run AFTER isolation_forest.py.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

model = joblib.load(BASE_DIR / "isolation_forest_model.joblib")
test = pd.read_csv(DATA_DIR / "test_mixed.csv")
types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0]

y = test["Label"]
X = test.drop(columns="Label")
score = -model.score_samples(X)          # higher = more anomalous

print(f"Overall ROC-AUC: {roc_auc_score(y, score):.3f}\n")

# 1) AUC per attack type: that attack vs the normal test rows
normal_mask = types == "Normal"
print("ROC-AUC per attack type (0.5 = coin flip, 1.0 = perfect):")
rows = []
for t in sorted(types[~normal_mask].unique()):
    m = normal_mask | (types == t)
    auc = roc_auc_score((types[m] != "Normal").astype(int), score[m])
    rows.append((t, int((types == t).sum()), auc))
for t, n, auc in sorted(rows, key=lambda r: -r[2]):
    print(f"  {t:<12} n={n:<6} AUC={auc:.3f}")

# 2) Single-feature separability (how well does ONE feature split normal vs attack?)
print("\nTop 8 individual features separating normal from attack (AUC, 0.5 = useless):")
single = {c: roc_auc_score(y, X[c]) for c in X.columns}
single = {c: max(a, 1 - a) for c, a in single.items()}
for c, a in sorted(single.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {c:<32} {a:.3f}")