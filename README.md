# Anomaly Detection in IoT Using Unsupervised Learning

Detecting malicious IoT network traffic with an **Isolation Forest** that is trained on **normal traffic only** (no attack examples), using the **NF-ToN-IoT-v2** NetFlow dataset.

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Quick start](#2-quick-start)
3. [Project structure](#3-project-structure)
4. [Dataset and data files](#4-dataset-and-data-files)
5. [How the pipeline works](#5-how-the-pipeline-works)
6. [Script-by-script guide](#6-script-by-script-guide)
7. [Results](#7-results)
8. [Interpretation and known limitations](#8-interpretation-and-known-limitations)
9. [Troubleshooting / FAQ](#10-troubleshooting--faq)

---

## 1. Project summary

| Item | Value |
|---|---|
| Task | Anomaly detection (normal vs. attack network flows) |
| Learning type | Unsupervised: the model never sees an attack during training |
| Model | Isolation Forest (`scikit-learn`) |
| Dataset | [`Nora9029/NF-ToN-IoT-v2`](https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2) (Hugging Face) |
| Total records used | 50,000 (25,000 train + 25,000 test) |
| Features | 32 numeric NetFlow features (after cleaning and selection) |
| Attack types | backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss |

**Idea in one sentence:** learn what "normal" IoT traffic looks like, then flag anything that looks different as an attack.

---

## 2. Quick start

### 2.1 Install the requirements

Open Command Prompt or PowerShell and run:

```bash
pip install datasets pandas numpy scikit-learn joblib
```

Python 3.9 or newer is recommended.

### 2.2 IMPORTANT: do NOT re-run `preprocess.py`

> **Do not run `preprocess.py` again.**
> It streams up to 2,000,000 rows from Hugging Face, takes a very long time, and uses random sampling, so **every person who runs it would get a slightly different dataset** and results would no longer be comparable between teammates.
>
> Use the ready-made CSV files in `data/processed/` instead. Everyone must work from the same files.

### 2.3 Run the project

Make sure the folder looks like [section 3](#3-project-structure) (the CSVs must be inside `data/processed/`), then from the project folder:

```bash
# 1. (optional) verify the data files are healthy
python check_data2.py

# 2. train + tune + evaluate the Isolation Forest
python isolation_forest.py

# 3. see where the model works and where it fails
python diagnose.py

# 4. (optional) confirm there is no train/test duplication
python check_overlap.py
```

Order matters: `diagnose.py` needs the model file that `isolation_forest.py` saves.

---

## 3. Project structure

```
anomalyIOT/
│
├── preprocess.py                 # builds the dataset (already run; do not re-run)
├── isolation_forest.py           # tunes, trains and evaluates the model
├── diagnose.py                   # per-attack and per-feature analysis
├── check_data2.py                # data quality checker
├── check_overlap.py              # train/test duplicate checker
│
├── isolation_forest_model.joblib # saved trained model (output of isolation_forest.py)
├── tuning_results.csv            # all 192 hyperparameter combinations (output)
├── per_attack_results.csv        # per-attack-type results on the final test set (output)
│
└── data/
    └── processed/
        ├── train_normal.csv          # 25,000 normal flows (training only)
        ├── test_mixed.csv            # 12,500 normal + 12,500 attack flows + Label column
        ├── test_attack_types.csv     # attack name for each row of test_mixed.csv
        ├── feature_names.txt         # the 32 feature names, in column order
        └── preprocessor.joblib       # fitted scaler, medians, feature list, log columns
```

All scripts locate files relative to their own folder, so you can run them from anywhere, but the `data/processed/` sub-folder **must exist** with the CSVs inside it.

---

## 4. Dataset and data files

### 4.1 Source

NF-ToN-IoT-v2 is a NetFlow version of the ToN-IoT dataset. Each row is one **network flow** (a summary of a connection: bytes, packets, duration, TCP flags, and so on), labelled normal or one of several attack types.

### 4.2 Files in `data/processed/`

| File | Rows | Contents |
|---|---|---|
| `train_normal.csv` | 25,000 | **Normal flows only**, 32 scaled features. Used to train the model. |
| `test_mixed.csv` | 25,000 | 12,500 normal + 12,500 attack flows, 32 scaled features + `Label` column (`0` = normal, `1` = attack). |
| `test_attack_types.csv` | 25,000 | One column (`AttackType`): `Normal` or the attack name. Row *i* matches row *i* of `test_mixed.csv`. |
| `feature_names.txt` | 32 | The feature names, in the same order as the CSV columns. |
| `preprocessor.joblib` | – | Dictionary with `scaler`, `medians`, `features`, `log_cols`. Use it to transform **new raw data** in exactly the same way. |

### 4.3 Attack types in the test set

| Attack type | Rows |
|---|---|
| ddos | 2,171 |
| password | 2,171 |
| xss | 2,170 |
| scanning | 2,170 |
| injection | 2,170 |
| dos | 1,100 |
| mitm | 393 |
| ransomware | 119 |
| backdoor | 36 |
| **Total attacks** | **12,500** |

The data supplies attack types as equally as possible. Types with few unique flows (backdoor, ransomware, mitm, dos) give everything they have, and the rest share the remaining slots. This is why the table is unbalanced.

### 4.4 The 32 features

`PROTOCOL, L7_PROTO, IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, TCP_FLAGS, SERVER_TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT, MIN_TTL, LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES, RETRANSMITTED_IN_BYTES, RETRANSMITTED_OUT_BYTES, RETRANSMITTED_OUT_PKTS, SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT, NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES, NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, ICMP_TYPE, DNS_QUERY_TYPE, DNS_TTL_ANSWER`

Columns removed on purpose (never given to the model, to avoid leaking identity or labels): `Label`, `Attack`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, `DNS_QUERY_ID`.

---

## 5. How the pipeline works

```
 Hugging Face dataset
        │  (preprocess.py, already done)
        ▼
 unique flows → fill missing → log transform → drop useless features → standardise
        │
        ├──► train_normal.csv   (25,000 normal)
        └──► test_mixed.csv     (12,500 normal + 12,500 attack)
                    │
                    │  isolation_forest.py
                    ▼
        split 50/50 (stratified by attack type)
           ├── VALIDATION (12,500): pick hyperparameters
           └── FINAL TEST (12,500): used once for the reported numbers
                    │
                    ▼
        diagnose.py → which attacks are caught, which features matter
```

### 5.1 What `preprocess.py` guarantees

- **No duplicates** inside train, inside test, and **between train and test**.
- Exact counts (25,000 train, 12,500 + 12,500 test) or the script stops with a clear message.
- Median filling, log transform, feature selection and scaling are **learned from the training data only**, so nothing from the test set leaks into preprocessing.
- Preprocessing steps in order:
  1. Stream and shuffle the dataset, keep only unique flows (fingerprint on log-scale, rounded values).
  2. Convert to numbers, replace `inf` with `NaN`, fill missing values with the **training median**.
  3. Apply `log1p` to every non-negative (skewed counter) column.
  4. Remove constant / near-constant columns (one value fills 99.5% or more of rows).
  5. Remove columns correlated at 0.95 or higher with an earlier column.
  6. Standardise with `StandardScaler` (mean 0, std 1), fitted on training data.
  7. Re-check uniqueness and build the test set.

### 5.2 How `isolation_forest.py` avoids cheating

- `test_mixed.csv` is split **50/50, stratified by attack type**, into a **validation** half and a **final test** half. Both halves contain every attack type.
- Hyperparameters are chosen on the **validation** half only.
- The chosen model is evaluated **once** on the **final test** half.

### 5.3 Hyperparameter search

| Parameter | Values tried |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_samples` | 256, 512, 1024, 2048, 4096, 8192, 16384, 25000 (all rows) |
| `contamination` | 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40 |

That is 24 forests × 8 contamination values = **192 combinations**.

- `n_estimators` and `max_samples` change the trees, so forests are ranked by **ROC-AUC** (does not depend on a threshold).
- `contamination` only moves the decision threshold, so each forest is trained **once** and all contamination values are tried on its scores.
- Ties in AUC (rounded to 4 decimals) are broken by **balanced accuracy**.
- Full results are saved to `tuning_results.csv`.

### 5.4 How a prediction is made

1. The model gives each flow an anomaly score (`-score_samples`; higher means more anomalous).
2. The threshold is the `contamination` percentile of the **training** scores.
3. Flows scoring beyond the threshold are predicted as **attack (1)**, the rest as **normal (0)**.

---

## 6. Script-by-script guide

### `preprocess.py`: dataset builder (do not re-run)
Streams NF-ToN-IoT-v2, cleans it, and writes everything in `data/processed/`. Settings are at the top of the file (`TRAIN_NORMAL`, `TEST_NORMAL`, `TEST_ATTACK`, `SCAN_LIMIT`, `SEED`, `CORR_LIMIT`, `NEAR_CONSTANT`). Only run it if the team deliberately agrees to regenerate **and share** a new dataset.

### `check_data2.py`: data quality check
Independent of any model. Verifies column order, NaN/inf, scaling, labels, split sizes, leakage, duplicates, and train/test similarity. Ends with a verdict (`READY`, `USABLE`, `NOT READY`).

```bash
python check_data2.py                 # uses data/processed next to the script
python check_data2.py <other_folder>  # or point it at another folder
```

Current result: **USABLE** (all critical checks pass). The two notes are expected:
- `backdoor` has only 36 test rows, so its per-type results are unreliable.
- Attack types are unbalanced because some types have few unique flows.

### `isolation_forest.py`: the model
Tunes, trains, evaluates, and saves:
- `isolation_forest_model.joblib` (best model, retrained on all training data)
- `tuning_results.csv` (all 192 combinations, sorted best first)
- `per_attack_results.csv` (results per attack type on the final test half)

Prints the top-5 configurations, final test metrics, the confusion matrix, a classification report and per-type results. Random seed is fixed (`RANDOM_SEED = 42`) so results are reproducible.

### `diagnose.py`: where does it succeed or fail?
Loads the saved model and prints:
1. Overall ROC-AUC.
2. ROC-AUC per attack type (that attack vs. normal rows).
3. The 8 individual features that best separate normal from attack.

> Note: `diagnose.py` scores the **whole** `test_mixed.csv` (25,000 rows), while `isolation_forest.py` reports on the **final-test half** (12,500 rows). Small differences between the two outputs are expected (e.g. overall AUC 0.602 vs 0.6046).

### `check_overlap.py`: duplicate check
Rounds rows to 6 decimals and checks how many test rows are identical to a training row. Expected: **0.0%** for both normal and attack.

---

## 7. Results

### 7.1 Best hyperparameters (chosen on validation)

```
n_estimators  = 200
max_samples   = 16384
contamination = 0.15
```

### 7.2 Final test results (12,500 rows: 6,250 normal, 6,250 attack)

| Metric | Value |
|---|---|
| Accuracy | 0.5981 |
| Balanced accuracy | 0.5981 |
| Precision (attack) | 0.6967 |
| Recall (attack) | 0.3474 |
| F1 (attack) | 0.4636 |
| ROC-AUC | 0.6046 |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred. normal | Pred. attack |
|---|---|---|
| **Actual normal** | 5,305 | 945 |
| **Actual attack** | 4,079 | 2,171 |

### 7.3 Results per attack type (final test)

| Attack | Rows | Correctly flagged | ROC-AUC vs normal |
|---|---|---|---|
| backdoor | 18 | 77.8% | 0.845 |
| ransomware | 59 | 62.7% | 0.845 |
| password | 1,086 | 59.9% | 0.780 |
| mitm | 196 | 60.7% | 0.758 |
| ddos | 1,086 | 58.5% | 0.728 |
| dos | 550 | 38.9% | 0.528 |
| scanning | 1,085 | 12.6% | 0.532 |
| injection | 1,085 | 19.7% | 0.523 |
| xss | 1,085 | 13.9% | 0.454 |
| *Normal (correctly left alone)* | 6,250 | 84.9% | – |

On the full 25,000-row test file, `diagnose.py` gives similar per-type AUCs (e.g. backdoor 0.848, ransomware 0.821, password 0.778, mitm 0.773, ddos 0.729, scanning 0.531, injection 0.525, dos 0.505, xss 0.452).

### 7.4 Most informative single features (AUC, 0.5 = useless)

| Feature | AUC |
|---|---|
| SERVER_TCP_FLAGS | 0.766 |
| TCP_WIN_MAX_IN | 0.749 |
| OUT_BYTES | 0.706 |
| OUT_PKTS | 0.681 |
| TCP_FLAGS | 0.670 |
| LONGEST_FLOW_PKT | 0.669 |
| DST_TO_SRC_SECOND_BYTES | 0.660 |
| TCP_WIN_MAX_OUT | 0.658 |

### 7.5 Data-quality checks

- No NaN or infinite values; training data is standardised (mean 0, std 1).
- 25,000 / 12,500 / 12,500 rows exactly; 50,000 total.
- **No leakage:** 0.0% of test rows are identical to any training row.
- Normal test data closely resembles normal training data (largest median shift 0.07 std).

---

## 8. Interpretation and known limitations

Be honest about these when presenting the work:

- **The model is only moderately better than chance** (ROC-AUC about 0.60, where 0.5 is a coin flip). It detects about 35% of attacks while flagging about 15% of normal flows as false alarms.
- **Detection is very uneven across attack types.** It does reasonably well on backdoor, ransomware, password, mitm and ddos, but is close to a coin flip (or worse) on scanning, injection, dos and especially **xss (AUC 0.454, worse than random)**.
- **Why this happens (likely):** an Isolation Forest flags flows that are *statistically unusual*. Several attacks (xss, injection, scanning) produce flows that look very similar to ordinary traffic at the NetFlow summary level, so they are not "unusual" and slip through.
- **Hyperparameter tuning barely helps.** Validation AUC stays between about 0.54 and 0.60 for every setting (see `tuning_results.csv`). The limit is the model and features, not the settings.
- **`backdoor` (18 final-test rows) and `ransomware` (59 rows) are tiny.** Their scores look good but are statistically unreliable.
- **Test and validation are one split.** Results come from a single 50/50 split with a fixed seed. For stronger claims, repeat with several seeds and report mean and spread.
- **Precision/recall depend on `contamination`.** Lower values give fewer false alarms but miss more attacks (see `tuning_results.csv`). Choose according to whether false alarms or missed attacks are costlier.
- The dataset is a sample (50,000 unique flows), not the full NF-ToN-IoT-v2, and is balanced 50/50, which is **not** the real-world attack rate (attacks are normally rare).

---
## 9. Troubleshooting / FAQ

**`FileNotFoundError: ...data/processed/train_normal.csv`**
The CSVs must be inside `data/processed/` next to the scripts (see [section 3](#3-project-structure)).

**`FileNotFoundError: isolation_forest_model.joblib` when running `diagnose.py`**
Run `python isolation_forest.py` first. It creates the model file.

**`ModuleNotFoundError`**
Run `pip install datasets pandas numpy scikit-learn joblib`.

**My numbers differ slightly from the README.**
Check you are using the shared CSVs (not a re-generated dataset) and a similar `scikit-learn` version. The seed is fixed, but different library versions can change tree building slightly.

**Can I use the model on new traffic?**
Yes, but new raw flows must go through the same steps as training: fill with the saved `medians`, apply `log1p` to `log_cols`, select `features`, then `scaler.transform`. All four are stored in `data/processed/preprocessor.joblib`.

```python
import joblib
import numpy as np
import pandas as pd

pre   = joblib.load("data/processed/preprocessor.joblib")
model = joblib.load("isolation_forest_model.joblib")

# raw_df = DataFrame of raw NetFlow rows with the original column names
X = raw_df[list(pre["medians"].index)].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(pre["medians"])
X[pre["log_cols"]] = np.log1p(X[pre["log_cols"]].clip(lower=0))
X = pd.DataFrame(pre["scaler"].transform(X[pre["features"]]), columns=pre["features"])

is_attack = (model.predict(X) == -1)   # True = flagged as attack
```

**Why is the Isolation Forest's `predict` output `-1` / `1`?**
scikit-learn uses `-1` for anomalies and `1` for normal. The scripts convert this to `1` = attack, `0` = normal.

---

## Credits

- Dataset: NF-ToN-IoT-v2 (via Hugging Face, `Nora9029/NF-ToN-IoT-v2`).
- Libraries: scikit-learn, pandas, NumPy, joblib, Hugging Face `datasets`.
