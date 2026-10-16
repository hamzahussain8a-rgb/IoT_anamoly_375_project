import pandas as pd
tr = pd.read_csv("data/processed/train_normal.csv")
te = pd.read_csv("data/processed/test_mixed.csv")
X = te.drop(columns="Label")
train_set = set(map(tuple, tr.round(6).values))
rows = list(map(tuple, X.round(6).values))
inside = [r in train_set for r in rows]
te["in_train"] = inside
print("Unique training rows:", len(train_set), "of", len(tr))
print("Normal test rows identical to a training row:",
      round(te[te.Label == 0].in_train.mean() * 100, 1), "%")
print("Attack test rows identical to a training row:",
      round(te[te.Label == 1].in_train.mean() * 100, 1), "%")