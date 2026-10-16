"""
preprocess.py - builds the files the models use (NF-ToN-IoT-v2).

  data/processed/train_normal.csv     : 25,000 NORMAL flows (training only, 0 attacks)
  data/processed/test_mixed.csv       : 12,500 normal + 12,500 attack flows, Label column (0/1)
  data/processed/test_attack_types.csv: one row per test row ("Normal" or the attack name)
  data/processed/feature_names.txt, preprocessor.joblib

Total = 50,000 records. Every attack type that exists in the data is represented, as equally
as the data allows (a type with too few unique flows gives everything it has and the other
types share the rest).

Guarantees
  * NO DUPLICATES: inside train, inside test, and between train and test
  * exact counts (the script stops with a clear message if the data cannot supply them)
  * median-fill, log-transform, feature selection and scaling are learned from TRAIN only
"""

import math
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ settings
DATASET = "Nora9029/NF-ToN-IoT-v2"

TRAIN_NORMAL = 25_000        # training set: normal only
TEST_NORMAL = 12_500         # test set: normal part
TEST_ATTACK = 12_500         # test set: attack part (spread over all attack types)

NORMAL_POOL = 42_000         # unique normal flows to collect (37,500 needed + spare for duplicates)
PER_TYPE_KEEP = 5_000        # unique flows remembered per attack type while scanning
SCAN_LIMIT = 2_000_000       # how many rows of the dataset to look through
SEED = 42

CORR_LIMIT = 0.95            # drop a column if it is this correlated with an earlier one
NEAR_CONSTANT = 0.995        # drop a column if one value fills this share of the rows

# never given to the model: labels and identifier-like columns
NOT_FEATURES = {"Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID"}

OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
rng = np.random.RandomState(SEED)


# ------------------------------------------------------ 1. scan and collect
def flow_key(row, cols):
    """Fingerprint of a flow as the model will see it (log scale, rounded)."""
    out = []
    for c in cols:
        try:
            x = float(row[c])
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(None if x != x or abs(x) == math.inf
                   else round(math.log1p(max(x, 0.0)), 2))
    return hash(tuple(out))


kept = defaultdict(list)     # group -> unique rows kept (random sample)
n_unique = defaultdict(int)  # group -> how many unique rows were seen
seen = set()


def keep(group, row, cap):
    n_unique[group] += 1
    if len(kept[group]) < cap:
        kept[group].append(row)
    else:                                   # reservoir sampling: fair over the whole scan
        j = random.randrange(n_unique[group])
        if j < cap:
            kept[group][j] = row


print("Connecting to the dataset (streaming)...")
print(f"Scanning up to {SCAN_LIMIT:,} rows. Press Ctrl+C to stop early and use what was found.")
stream = load_dataset(DATASET, split="train", streaming=True)
stream = stream.shuffle(seed=SEED, buffer_size=20_000)

cols = None
try:
    for n, row in enumerate(stream, 1):
        if cols is None:
            cols = [c for c in row if c not in NOT_FEATURES]
        group = "normal" if int(row["Label"]) == 0 else str(row["Attack"])
        k = hash((group, flow_key(row, cols)))
        if k not in seen:                   # skip flows that look like one already seen
            seen.add(k)
            keep(group, row, NORMAL_POOL if group == "normal" else PER_TYPE_KEEP)
        if n % 200_000 == 0:
            print(f"  scanned {n:,} | unique normal kept {len(kept['normal']):,} | "
                  f"unique attacks seen {sum(v for g, v in n_unique.items() if g != 'normal'):,}",
                  flush=True)
        if n >= SCAN_LIMIT:
            break
except KeyboardInterrupt:
    print("\nStopped early - using what was found so far.")

normal_rows = kept["normal"]
need_normal = TRAIN_NORMAL + TEST_NORMAL
print(f"\nUnique normal flows found: {len(normal_rows):,} (need at least {need_normal:,})")
if len(normal_rows) < need_normal:
    raise SystemExit("Not enough unique normal flows - increase SCAN_LIMIT.")

attack_groups = {g: len(v) for g, v in kept.items() if g != "normal"}
print("Unique attack flows found per type:", dict(sorted(attack_groups.items(), key=lambda kv: -kv[1])))
if sum(attack_groups.values()) < TEST_ATTACK:
    raise SystemExit("Not enough unique attack flows - increase SCAN_LIMIT.")


# ------------------------------------------------------------- 2. clean helpers
def to_numbers(rows):
    df = pd.DataFrame(rows)[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    return df.replace([np.inf, -np.inf], np.nan)


def hashes(df):
    return pd.util.hash_pandas_object(df.round(3) + 0.0, index=False).to_numpy()


def equal_share(available, total):
    """Same number of each attack type; a type with fewer rows gives everything it has
    and the other types share the rest equally."""
    take, left = {}, total
    order = sorted(available, key=available.get)
    for i, t in enumerate(order):
        take[t] = min(available[t], left // (len(order) - i))
        left -= take[t]
    return take


# ------------------------------------------- 3. split normal pool (train side / test side)
normal_X = to_numbers(normal_rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
spare = len(normal_X) - need_normal
train_cand_n = TRAIN_NORMAL + spare // 3            # a little spare on each side for duplicates
train_X = normal_X.iloc[:train_cand_n].copy()
normal_test_X = normal_X.iloc[train_cand_n:].copy()

attack_pool = [r for g in attack_groups for r in kept[g]]
attack_pool_names = np.array([g for g in attack_groups for _ in kept[g]])
attack_X = to_numbers(attack_pool)

# everything below is learned from the TRAINING side only
medians = train_X.median().fillna(0.0)
train_X, normal_test_X, attack_X = (d.fillna(medians) for d in (train_X, normal_test_X, attack_X))

log_cols = [c for c in cols if train_X[c].min() >= 0]          # skewed counters -> log scale
for d in (train_X, normal_test_X, attack_X):
    d[log_cols] = np.log1p(d[log_cols].clip(lower=0))

# ------------------------------------------------- 4. keep only useful features
dropped = {}
for c in cols:
    top_share = train_X[c].value_counts(normalize=True).iloc[0]
    if train_X[c].nunique() <= 1 or top_share >= NEAR_CONSTANT:
        dropped[c] = "constant / almost constant"
remaining = [c for c in cols if c not in dropped]

corr = train_X[remaining].corr().abs()
for i, c in enumerate(remaining):
    for earlier in remaining[:i]:
        if earlier not in dropped and corr.loc[earlier, c] >= CORR_LIMIT:
            dropped[c] = f"duplicates the information in {earlier}"
            break
features = [c for c in cols if c not in dropped]
print(f"\nFeatures kept: {len(features)} of {len(cols)}")
for c, why in dropped.items():
    print(f"  removed {c}: {why}")

# ------------------------------------- 5. final training set: unique rows, exactly 25,000
# (dropping columns can make two flows identical, so uniqueness is re-checked here)
train_X = train_X[features]
train_X = train_X[~pd.Series(hashes(train_X)).duplicated().to_numpy()]
if len(train_X) < TRAIN_NORMAL:
    raise SystemExit(f"Only {len(train_X)} unique training flows after feature selection - "
                     f"increase NORMAL_POOL or SCAN_LIMIT.")
train_X = train_X.iloc[:TRAIN_NORMAL].reset_index(drop=True)

# --------------------------------------------------------------- 6. scale (train only)
scaler = StandardScaler().fit(train_X)
def scale(df):
    return pd.DataFrame(scaler.transform(df[features]), columns=features)

train_S = scale(train_X)
normal_test_S = scale(normal_test_X)
attack_S = scale(attack_X)

# ------------------------------- 7. build the test set: unique, no copies of train, exact counts
train_keys = hashes(train_S)

# normal part
h = hashes(normal_test_S)
ok = ~np.isin(h, train_keys) & ~pd.Series(h).duplicated().to_numpy()
normal_test_S = normal_test_S[ok].reset_index(drop=True)
if len(normal_test_S) < TEST_NORMAL:
    raise SystemExit(f"Only {len(normal_test_S)} unique normal test flows - increase NORMAL_POOL.")
normal_test_S = normal_test_S.iloc[:TEST_NORMAL]

# attack part: unique, not a copy of train or of a normal test flow (that would be a label clash)
h = hashes(attack_S)
banned = np.concatenate([train_keys, hashes(normal_test_S)])
ok = ~np.isin(h, banned) & ~pd.Series(h).duplicated().to_numpy()
attack_S = attack_S[ok].reset_index(drop=True)
attack_names = attack_pool_names[ok]

available = {g: int((attack_names == g).sum()) for g in map(str, np.unique(attack_names))}
take = equal_share(available, TEST_ATTACK)
picked = np.concatenate([rng.choice(np.where(attack_names == g)[0], m, replace=False)
                         for g, m in take.items() if m > 0])
attack_S = attack_S.iloc[picked].reset_index(drop=True)
attack_names = attack_names[picked]
if len(attack_S) != TEST_ATTACK:
    raise SystemExit(f"Could only build {len(attack_S)} attack test flows (wanted {TEST_ATTACK}).")
print("\nAttack flows per type in test:", dict(sorted(take.items(), key=lambda kv: -kv[1])))

# ------------------------------------------------------------------ 8. save
normal_test_S = normal_test_S.assign(Label=0)
attack_S = attack_S.assign(Label=1)
test = pd.concat([normal_test_S, attack_S], ignore_index=True)
types = pd.Series(["Normal"] * len(normal_test_S) + list(attack_names), name="AttackType")
order = rng.permutation(len(test))
test, types = test.iloc[order].reset_index(drop=True), types.iloc[order].reset_index(drop=True)

train_S.to_csv(OUT_DIR / "train_normal.csv", index=False)
test.to_csv(OUT_DIR / "test_mixed.csv", index=False)
types.to_csv(OUT_DIR / "test_attack_types.csv", index=False)
(OUT_DIR / "feature_names.txt").write_text("\n".join(features))
joblib.dump({"scaler": scaler, "medians": medians, "features": features, "log_cols": log_cols},
            OUT_DIR / "preprocessor.joblib")

print("\n--- DONE ---")
print(f"train_normal.csv : {len(train_S):,} normal flows, {len(features)} features")
print(f"test_mixed.csv   : {len(test):,} flows = {len(normal_test_S):,} normal + {len(attack_S):,} attack")
print(f"TOTAL            : {len(train_S) + len(test):,}")
print("Saved to:", OUT_DIR)
