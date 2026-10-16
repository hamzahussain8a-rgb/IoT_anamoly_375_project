"""
check_data.py - independent quality check of the preprocessed files.
Run AFTER preprocess.py. It does not use any model.

    python check_data.py            (looks in data/processed next to this file)
    python check_data.py <folder>   (or point it at another folder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data" / "processed"

results = []   # (status, message)


def add(status, msg):
    results.append((status, msg))
    icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[info]"}[status]
    print(f"{icon} {msg}")


# ---------------------------------------------------------------- files
need = ["train_normal.csv", "test_mixed.csv", "test_attack_types.csv", "feature_names.txt"]
missing = [f for f in need if not (D / f).exists()]
if missing:
    print(f"[FAIL] Missing files in {D}: {missing}")
    sys.exit(1)

train = pd.read_csv(D / "train_normal.csv")
test = pd.read_csv(D / "test_mixed.csv")
types = pd.read_csv(D / "test_attack_types.csv").iloc[:, 0].astype(str)
names = (D / "feature_names.txt").read_text().split()

print(f"Checking: {D}\n")

# ---------------------------------------------------------------- structure
if "Label" not in test.columns:
    print("[FAIL] test_mixed.csv has no Label column")
    sys.exit(1)
y = test["Label"]
Xt = test.drop(columns="Label")
extra = [c for c in Xt.columns if not pd.api.types.is_numeric_dtype(Xt[c])]
if extra:
    add("FAIL", f"Non-numeric columns inside the test features: {extra}")
    Xt = Xt.drop(columns=extra)

add("INFO", f"train {train.shape}, test {test.shape} (features + Label), attack-type rows {len(types)}")

if list(train.columns) == list(Xt.columns) == names:
    add("PASS", f"Same {len(names)} feature columns, same order, in train / test / feature_names.txt")
else:
    add("FAIL", "Feature columns differ between train, test and feature_names.txt")

if len(types) == len(test):
    add("PASS", "test_attack_types.csv has one row per test row")
else:
    add("FAIL", f"test_attack_types.csv has {len(types)} rows but test has {len(test)}")

# ---------------------------------------------------------------- cleanliness
bad = int(train.isna().sum().sum() + Xt.isna().sum().sum())
inf = int(np.isinf(train.to_numpy(float)).sum() + np.isinf(Xt.to_numpy(float)).sum())
add("PASS" if bad == 0 else "FAIL", f"Missing values (NaN): {bad}")
add("PASS" if inf == 0 else "FAIL", f"Infinite values: {inf}")

# ---------------------------------------------------------------- scaling
mean_max = float(train.mean().abs().max())
sd = train.std()
if mean_max < 1e-6 and 0.99 < sd.min() and sd.max() < 1.01:
    add("PASS", "Training data is properly scaled (every column mean 0, std 1)")
else:
    add("FAIL", f"Training data not standardised (max |mean|={mean_max:.3g}, std range {sd.min():.3f}-{sd.max():.3f})")

tmax = float(Xt.abs().max().max())
worst = Xt.abs().max().idxmax()
if tmax > 1e4:
    add("FAIL", f"Extreme test value {tmax:.0f} in '{worst}' (broken scaling?)")
elif tmax > 100:
    add("WARN", f"Largest test value is {tmax:.0f} std in '{worst}' (heavy tail, probably fine)")
else:
    add("PASS", f"Largest test value is {tmax:.1f} std ('{worst}'): no extreme outliers")

# ---------------------------------------------------------------- labels
if set(y.unique()) == {0, 1}:
    n_norm, n_att = int((y == 0).sum()), int((y == 1).sum())
    add("PASS", f"Labels are only 0/1: {n_norm:,} normal, {n_att:,} attack ({n_att / len(y):.0%} attacks)")
else:
    add("FAIL", f"Unexpected label values: {sorted(y.unique())}")
    n_norm = n_att = 0

if len(types) == len(test):
    normal_name = types[y == 0].unique()
    consistent = len(normal_name) == 1 and not (types[y == 1] == normal_name[0]).any()
    add("PASS" if consistent else "FAIL", "Attack-type file agrees with Label (normal rows <-> one 'normal' name)")
    counts = types[y == 1].value_counts()
    add("INFO", "Attack types in test: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    rare = counts[counts < 100]
    if len(rare):
        rare_txt = ", ".join(f"{k}={int(v)}" for k, v in rare.items())
        add("WARN", f"Very few rows for: {rare_txt} - per-type results will be unreliable")
    if counts.max() > 2 * counts.min():
        add("WARN", "Attack types are unbalanced (largest > 2x smallest): some types had few unique flows")

# ---------------------------------------------------------------- split sanity
EXPECT_TRAIN, EXPECT_NORMAL, EXPECT_ATTACK = 25_000, 12_500, 12_500
add("PASS" if len(train) == EXPECT_TRAIN else "FAIL", f"Training rows: {len(train):,} (expected {EXPECT_TRAIN:,})")
add("PASS" if n_norm == EXPECT_NORMAL else "FAIL", f"Normal test rows: {n_norm:,} (expected {EXPECT_NORMAL:,})")
add("PASS" if n_att == EXPECT_ATTACK else "FAIL", f"Attack test rows: {n_att:,} (expected {EXPECT_ATTACK:,})")
add("PASS" if len(train) + len(test) == 50_000 else "FAIL", f"Total records: {len(train) + len(test):,} (limit 50,000)")

# ---------------------------------------------------------------- leakage
def row_hash(df):
    return pd.util.hash_pandas_object(df.round(3) + 0.0, index=False).to_numpy()

h_train = row_hash(train)
h_test = row_hash(Xt)
hit = np.isin(h_test, h_train)
n_hit_norm = int(hit[(y == 0).to_numpy()].sum())
n_hit_att = int(hit[(y == 1).to_numpy()].sum())
if n_hit_norm == 0 and n_hit_att == 0:
    add("PASS", "No test row is identical to a training row (no leakage)")
else:
    add("FAIL", f"LEAKAGE: {n_hit_norm} normal and {n_hit_att} attack test rows are copies of training rows")

# ---------------------------------------------------------------- duplicates
vc = pd.Series(h_train).value_counts()
dup_share = 1 - len(vc) / len(train)
add("INFO", f"Training set: {len(vc):,} distinct rows of {len(train):,} ({dup_share:.0%} are repeats)")
if vc.iloc[0] > 50:
    add("FAIL", f"One training pattern repeats {vc.iloc[0]:,} times - it dominates the data")
elif vc.iloc[0] > 25:
    add("WARN", f"Most repeated training pattern appears {vc.iloc[0]} times")
else:
    add("PASS", f"No pattern dominates: the most repeated training row appears {vc.iloc[0]} times")

vt = pd.Series(h_test[(y == 0).to_numpy()]).value_counts()
if len(vt) and vt.iloc[0] > 50:
    add("FAIL", f"One normal TEST pattern repeats {vt.iloc[0]:,} times")

# ---------------------------------------------------------------- similarity of normal train vs normal test
nt = Xt[(y == 0).to_numpy()]
if len(nt):
    shift = ((nt.median() - train.median()).abs() / train.std())
    w = shift.idxmax()
    if shift.max() > 0.5:
        add("WARN", f"Normal test looks different from train in '{w}' (median shift {shift.max():.2f} std)")
    else:
        add("PASS", f"Normal test resembles normal train (largest median shift {shift.max():.2f} std)")

# ---------------------------------------------------------------- verdict
fails = sum(s == "FAIL" for s, _ in results)
warns = sum(s == "WARN" for s, _ in results)
print("\n" + "=" * 60)
if fails:
    print(f"VERDICT: NOT READY - {fails} failed check(s). Fix preprocessing first.")
elif warns:
    print(f"VERDICT: USABLE - all critical checks passed, {warns} note(s) to mention in the report.")
else:
    print("VERDICT: READY - every check passed.")
print("=" * 60)