"""evaluate.py - compare every model on the SAME final test set (Member 4).

Needs (made by each model script via common.save_scores):
    scores/<model>_train.npy, scores/<model>_val.npy, scores/<model>_test.npy
Optional: scores/<model>_timings.json

Fair-comparison rule used for EVERY model (same as the Isolation Forest tuning):
    threshold = a percentile of the model's scores on the NORMAL TRAINING data
    the percentile is chosen on the VALIDATION half (best balanced accuracy)
    results are reported on the FINAL TEST half, which was never used for choosing.

Outputs go to ./results/
Run:  python evaluate.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from common import BASE_DIR, SCORES_DIR, load_split

OUT = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

# candidate thresholds = same grid as IF contamination (0.01 ... 0.4) as percentiles
PCT_GRID = [99, 95, 90, 85, 80, 75, 70, 60]
FIXED_PCTS = [90, 95, 99]            # for the threshold-sensitivity table

d = load_split()
y_val, y_test, t_test = d["y_val"].to_numpy(), d["y_test"].to_numpy(), d["t_test"].to_numpy()

names = sorted(p.name[:-len("_test.npy")] for p in SCORES_DIR.glob("*_test.npy"))
if not names:
    raise SystemExit("No scores found. Run the model scripts (and export_if_scores.py) first.")
print("Models found:", ", ".join(names))


def metrics(y, pred):
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Accuracy": accuracy_score(y, pred),
        "Balanced acc": balanced_accuracy_score(y, pred),
    }


table, sweep, per_type, scores, chosen = [], [], {}, {}, {}
for name in names:
    tr = np.load(SCORES_DIR / f"{name}_train.npy")
    va = np.load(SCORES_DIR / f"{name}_val.npy")
    te = np.load(SCORES_DIR / f"{name}_test.npy")
    assert len(va) == len(y_val) and len(te) == len(y_test), f"{name}: wrong number of scores"
    scores[name] = te

    # choose the percentile on VALIDATION
    best_p = max(PCT_GRID, key=lambda p: balanced_accuracy_score(y_val, (va > np.percentile(tr, p)).astype(int)))
    thr = np.percentile(tr, best_p)
    pred = (te > thr).astype(int)
    chosen[name] = thr
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    row = {"Model": name, "Threshold (train pct)": best_p, **metrics(y_test, pred),
           "ROC-AUC": roc_auc_score(y_test, te), "FP": fp, "FN": fn}
    tpath = SCORES_DIR / f"{name}_timings.json"
    if tpath.exists():
        t = json.loads(tpath.read_text())
        row["Train time (s)"] = t.get("train_time_s")
        row["Inference time (s)"] = t.get("infer_time_s")
        row["Model size (KB)"] = t.get("model_size_kb")
    table.append(row)

    for p in FIXED_PCTS:
        pr = (te > np.percentile(tr, p)).astype(int)
        sweep.append({"Model": name, "Train percentile": p, **metrics(y_test, pr)})

    # per attack type: share of that type flagged + AUC vs all normal rows
    normal = t_test == "Normal"
    for t in sorted(set(t_test) - {"Normal"}):
        m = t_test == t
        per_type.setdefault(t, {"Rows": int(m.sum())})
        per_type[t][f"{name} recall"] = float(pred[m].mean())
        both = m | normal
        per_type[t][f"{name} AUC"] = float(roc_auc_score(y_test[both], te[both]))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=[0, 1]),
                           display_labels=["Normal", "Attack"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    fig.savefig(OUT / f"cm_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # score distribution with chosen threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    hi = np.percentile(te, 99.5)
    bins = np.linspace(te.min(), hi, 60)
    ax.hist(np.clip(te[y_test == 0], None, hi), bins=bins, alpha=0.6, density=True, label="Normal")
    ax.hist(np.clip(te[y_test == 1], None, hi), bins=bins, alpha=0.6, density=True, label="Attack")
    ax.axvline(min(thr, hi), color="red", ls="--", label=f"Threshold ({best_p}th pct of train)")
    ax.set_xlabel("Anomaly score (higher = more anomalous)"); ax.set_ylabel("Density")
    ax.set_title(f"Score distribution: {name}"); ax.legend()
    fig.savefig(OUT / f"dist_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# combined ROC curve
fig, ax = plt.subplots(figsize=(6, 5))
for name, s in scores.items():
    fpr, tpr, _ = roc_curve(y_test, s)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, s):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the final test set"); ax.legend()
fig.savefig(OUT / "roc_all.png", dpi=200, bbox_inches="tight")
plt.close(fig)

table = pd.DataFrame(table).round(4)
sweep = pd.DataFrame(sweep).round(4)
per_type = pd.DataFrame(per_type).T.round(4)
per_type.index.name = "Attack type"
table.to_csv(OUT / "comparison_table.csv", index=False)
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
per_type.to_csv(OUT / "per_attack_comparison.csv")

print("\n=== COMPARISON (final test set) ===")
print(table.to_string(index=False))
print("\n=== THRESHOLD SENSITIVITY (fixed percentiles of train scores) ===")
print(sweep.to_string(index=False))
print("\n=== PER ATTACK TYPE ===")
print(per_type.to_string())
print(f"\nSaved to {OUT}")
