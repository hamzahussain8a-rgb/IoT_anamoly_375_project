# Anomaly Detection in IoT Using Unsupervised Learning

Detecting malicious IoT network traffic with four unsupervised models (**Isolation Forest**, **Autoencoder**, **LSTM autoencoder** and **One-Class SVM**) that are trained on **normal traffic only** (no attack examples), using the **NF-ToN-IoT-v2** NetFlow dataset.

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Quick start](#2-quick-start)
3. [Project structure](#3-project-structure)
4. [Dataset and data files](#4-dataset-and-data-files)
5. [How the pipeline works](#5-how-the-pipeline-works)
6. [Script-by-script guide](#6-script-by-script-guide)
7. [Results](#7-results)
8. [Robustness experiments](#8-robustness-experiments)
9. [Demo](#9-demo)
10. [Interpretation and known limitations](#10-interpretation-and-known-limitations)
11. [Troubleshooting / FAQ](#11-troubleshooting--faq)

---

## 1. Project summary

| Item | Value |
|---|---|
| Task | Anomaly detection (normal vs. attack network flows) |
| Learning type | Unsupervised: the models never see an attack during training |
| Models | Isolation Forest (`scikit-learn`), dense Autoencoder (`tensorflow`), LSTM autoencoder (`tensorflow`), One-Class SVM (`scikit-learn`) |
| Dataset | [`Nora9029/NF-ToN-IoT-v2`](https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2) (Hugging Face) |
| Total records used | 50,000 (25,000 train + 25,000 test) |
| Features | 32 numeric NetFlow features (after cleaning and selection) |
| Attack types | backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss |

**Idea in one sentence:** learn what "normal" IoT traffic looks like, then flag anything that looks different as an attack.

---

## 2. Quick start

### 2.1 Install the requirements

```bash
pip install -r requirements.txt
```

This installs numpy, pandas, scikit-learn, tensorflow, matplotlib, joblib and datasets. Python 3.9 or newer is recommended. The project was last run with Python 3.13 and TensorFlow 2.21.

### 2.2 IMPORTANT: do NOT re-run `preprocess.py`

> **Do not run `preprocess.py` again.**
> It streams up to 2,000,000 rows from Hugging Face, takes a very long time, and uses random sampling, so **every person who runs it would get a slightly different dataset** and results would no longer be comparable between teammates.
>
> Use the ready-made CSV files in `data/processed/` instead. Everyone must work from the same files.

### 2.3 Run the project

Make sure the folder looks like [section 3](#3-project-structure), then from the project folder:

```bash
# 1. (optional) verify the data files are healthy
python check_data2.py

# 2. train and evaluate the models (saved models are already in the repo)
python isolation_forest.py
python autoencoder.py
python lstm_autoencoder.py
python ocsvm.py

# 3. (optional) analysis of the Isolation Forest
python diagnose.py
python check_overlap.py

# 4. compare all models (reads the saved scores, writes results/)
python evaluate.py

# 5. robustness experiments: 5 seeds per model (takes a long time)
python robustness.py

# 6. demo: load the saved models and classify a few rows
python demo.py
```

The saved models are already in the repository, so you only need to re-run a training script if you want to regenerate that model. `diagnose.py` needs the model file that `isolation_forest.py` saves.

---

## 3. Project structure

```
IoT_anamoly_375_project/
│
├── preprocess.py                  # builds the dataset (already run; do not re-run)
├── common.py                      # shared data split and score/timing saving helpers
├── isolation_forest.py            # tunes, trains and evaluates the Isolation Forest
├── autoencoder.py                 # dense autoencoder (bottleneck experiments)
├── lstm_autoencoder.py            # LSTM autoencoder
├── ocsvm.py                       # One-Class SVM baseline
├── export_if_scores.py            # export anomaly scores of each model to scores/
├── export_ae_scores.py
├── export_lstm_scores.py
├── evaluate.py                    # compares all models, writes results/
├── robustness.py                  # 5-seed robustness experiments for all four models
├── demo.py                        # loads the saved models and predicts sample rows
├── diagnose.py                    # per-attack and per-feature analysis
├── check_data2.py                 # data quality checker
├── check_overlap.py               # train/test duplicate checker
├── requirements.txt
│
├── isolation_forest_model.joblib  # saved Isolation Forest
├── ocsvm_model.joblib             # saved One-Class SVM
├── tuning_results.csv             # all 192 Isolation Forest hyperparameter combinations
├── per_attack_results.csv         # Isolation Forest per-attack results (final test)
│
├── data/
│   └── processed/
│       ├── train_normal.csv       # 25,000 normal flows (training only)
│       ├── test_mixed.csv         # 12,500 normal + 12,500 attack flows + Label column
│       ├── test_attack_types.csv  # attack name for each row of test_mixed.csv
│       ├── feature_names.txt      # the 32 feature names, in column order
│       └── preprocessor.joblib    # fitted scaler, medians, feature list, log columns
│
├── outputs/                       # autoencoder.keras, lstm_autoencoder.keras, training
│   │                              # histories, thresholds and per-attack results
│   └── robustness/                # robustness results (per-seed and summary CSVs)
│
├── scores/                        # anomaly scores and timings of each model
└── results/                       # comparison table, confusion matrices, ROC and
                                   # score-distribution plots from evaluate.py
```

Scripts locate files relative to their own folder, so you can run them from anywhere, but the `data/processed/` sub-folder **must exist** with the CSVs inside it.

---

## 4. Dataset and data files

### 4.1 Source

NF-ToN-IoT-v2 is a NetFlow version of the ToN-IoT dataset. Each row is one **network flow** (a summary of a connection: bytes, packets, duration, TCP flags, and so on), labelled normal or one of several attack types.

### 4.2 Files in `data/processed/`

| File | Rows | Contents |
|---|---|---|
| `train_normal.csv` | 25,000 | **Normal flows only**, 32 scaled features. Used to train the models. |
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

Columns removed on purpose (never given to the models, to avoid leaking identity or labels): `Label`, `Attack`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, `DNS_QUERY_ID`.

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
                    │  every model script uses the same split
                    ▼
        split 50/50 (stratified by attack type, seed 42)
           ├── VALIDATION (12,500): pick hyperparameters / thresholds
           └── FINAL TEST (12,500): used once for the reported numbers
                    │
                    ▼
        evaluate.py → results/        robustness.py → outputs/robustness/
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

### 5.2 How the models avoid cheating

- `test_mixed.csv` is split **50/50, stratified by attack type**, into a **validation** half and a **final test** half. Both halves contain every attack type. All model scripts use the same split (seed 42).
- Hyperparameters and thresholds are chosen on normal-only or validation data, never on the final test labels.
- The chosen model is evaluated **once** on the **final test** half.

### 5.3 Isolation Forest hyperparameter search

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

### 5.4 How each model makes a prediction

All models produce an anomaly score where **higher means more anomalous**. A flow is predicted as **attack (1)** when its score crosses the model's threshold, otherwise **normal (0)**.

| Model | Score | Threshold |
|---|---|---|
| Isolation Forest | `-score_samples` | `contamination` percentile of the **training** scores (0.15) |
| Autoencoder | reconstruction error (mean squared) | 95th percentile of errors on a normal-only validation split |
| LSTM autoencoder | reconstruction error (mean squared) | 95th percentile of errors on a normal-only validation split |
| One-Class SVM | `-decision_function` | 0 (same as `predict() == -1`) |

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

### `isolation_forest.py`: Isolation Forest
Tunes, trains, evaluates, and saves `isolation_forest_model.joblib`, `tuning_results.csv` and `per_attack_results.csv`. Prints the top-5 configurations, final test metrics, the confusion matrix, a classification report and per-type results. The random seed is fixed (`RANDOM_SEED = 42`).

### `autoencoder.py`: dense autoencoder
Trains on normal traffic only. Layout: input → 64 → 32 → bottleneck → 32 → 64 → output. Compares bottleneck sizes 4, 8 and 16 on the mixed validation set (best F1, ROC-AUC breaks ties), up to 300 epochs with early stopping (patience 15). The anomaly threshold is the 95th percentile of reconstruction errors on a normal-only validation split. The selected bottleneck is **8**. Saves `outputs/autoencoder.keras` and result files in `outputs/`.

### `lstm_autoencoder.py`: LSTM autoencoder
Same data and threshold rule as the dense autoencoder, with LSTM layers (64 → 32 → 32 → 64). The sequence length is 1 because the dataset has no time order. Up to 100 epochs, early stopping with patience 15. Saves `outputs/lstm_autoencoder.keras`.

### `ocsvm.py`: One-Class SVM
RBF kernel, `gamma="scale"`. Tries `nu` in 0.01, 0.05, 0.1 and 0.2, picks the best by validation ROC-AUC (chosen: `nu = 0.2`, stored in `scores/ocsvm_timings.json`), then trains on all 25,000 normal rows. Saves `ocsvm_model.joblib`.

### `common.py`: shared helpers
Provides the shared data split (`load_split`) and the helpers that save scores and timings to `scores/`. The scripts that use it find `data/processed/` next to them, or at the folder named in the `IOT_REPO` environment variable.

### `export_if_scores.py`, `export_ae_scores.py`, `export_lstm_scores.py`
Write the anomaly scores of each saved model to `scores/`.

### `evaluate.py`: model comparison
Compares the models using the saved scores and writes `results/`: comparison table, per-attack comparison, threshold sweep, confusion matrices, ROC curves and score-distribution plots.

### `robustness.py`: robustness experiments
See [section 8](#8-robustness-experiments).

### `demo.py`: demo
See [section 9](#9-demo).

### `diagnose.py`: where does the Isolation Forest succeed or fail?
Loads the saved model and prints overall ROC-AUC, ROC-AUC per attack type, and the 8 individual features that best separate normal from attack.

> Note: `diagnose.py` scores the **whole** `test_mixed.csv` (25,000 rows), while `isolation_forest.py` reports on the **final-test half** (12,500 rows). Small differences between the two outputs are expected (e.g. overall AUC 0.602 vs 0.6046).

### `check_overlap.py`: duplicate check
Rounds rows to 6 decimals and checks how many test rows are identical to a training row. Expected: **0.0%** for both normal and attack.

---

## 7. Results

### 7.1 Isolation Forest: best hyperparameters (chosen on validation)

```
n_estimators  = 200
max_samples   = 16384
contamination = 0.15
```

### 7.2 Isolation Forest: final test results (12,500 rows: 6,250 normal, 6,250 attack)

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

### 7.3 Isolation Forest: results per attack type (final test)

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

### 7.6 Comparing all four models

The full model comparison (comparison table, per-attack comparison, confusion matrices, ROC curves) is generated by `evaluate.py` and stored in `results/`. A summary over several seeds is in [section 8](#8-robustness-experiments).

---

## 8. Robustness experiments

`robustness.py` repeats every model with 5 random seeds (21, 42, 55, 77, 100) and evaluates each run on the same fixed final test set (12,500 rows, 50% attacks). The validation/test split always uses seed 42, so **only the model seed changes**. Labels are 0 = normal and 1 = attack. No test labels are used to choose thresholds. Results are the mean and the sample standard deviation (ddof = 1) over the 5 seeds.

### 8.1 Setup per model

| Model | Setup |
|---|---|
| Isolation Forest | n_estimators=200, max_samples=16384, contamination=0.15 (from `tuning_results.csv`). Threshold: 15th percentile of the training scores. |
| Autoencoder | Dense 64-32-8-32-64, bottleneck size read from the saved model (8). Up to 300 epochs, early stopping with patience 15. Threshold: 95th percentile of errors on a normal-only validation split. |
| LSTM autoencoder | Same layout as `lstm_autoencoder.py`. Up to 100 epochs, same early stopping and threshold rule. |
| One-Class SVM | RBF kernel, gamma="scale", nu=0.2 (read from `scores/ocsvm_timings.json`). The model has no randomness, so each seed trains on a different random 80% of the normal rows. |

For the Isolation Forest the seed changes the trees. For the two autoencoders it changes the weight initialisation and the batch shuffling.

### 8.2 Results (mean ± std over 5 seeds)

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Isolation Forest | 0.585 ± 0.017 | 0.449 ± 0.014 | 0.691 ± 0.007 | 0.333 ± 0.014 | 0.592 ± 0.006 |
| Autoencoder | 0.744 ± 0.014 | 0.354 ± 0.027 | 0.832 ± 0.013 | 0.225 ± 0.021 | 0.590 ± 0.010 |
| LSTM autoencoder | 0.718 ± 0.011 | 0.350 ± 0.019 | 0.843 ± 0.008 | 0.221 ± 0.015 | 0.590 ± 0.007 |
| One-Class SVM | 0.486 ± 0.001 | 0.368 ± 0.003 | 0.577 ± 0.003 | 0.270 ± 0.002 | 0.536 ± 0.001 |

The test set is exactly 50/50, so accuracy equals balanced accuracy.

| Model | Training time (s) | Inference, 12,500 rows (s) | Inference per record (ms) | Model size (MB) |
|---|---|---|---|---|
| Isolation Forest | 1.10 ± 0.03 | 0.210 ± 0.011 | 0.017 ± 0.001 | 28.96 ± 0.38 |
| Autoencoder | 163.94 ± 57.70 | 0.501 ± 0.024 | 0.040 ± 0.002 | 0.144 ± 0.000 |
| LSTM autoencoder | 153.39 ± 7.85 | 0.694 ± 0.015 | 0.056 ± 0.001 | 0.884 ± 0.000 |
| One-Class SVM | 9.72 ± 1.04 | 4.850 ± 0.177 | 0.388 ± 0.014 | 1.058 ± 0.004 |

Per-seed results and the summaries are saved in `outputs/robustness/` (`<model>_per_seed.csv` and `<model>_summary.csv`).

### 8.3 How to read these results

- The two autoencoders rank attacks best (ROC-AUC 0.72 to 0.74). The Isolation Forest is much weaker, and the One-Class SVM (0.486) is slightly worse than random guessing.
- A small standard deviation means the result is stable across seeds, not that the model is good. The Isolation Forest and the One-Class SVM are consistently mediocre.
- The autoencoders have high precision but low recall (about 0.22): the 95th-percentile threshold is strict, so they flag few rows and miss most attacks, even though their scores rank attacks fairly well.
- Autoencoder training time varies a lot between seeds because early stopping ends at different epochs.

### 8.4 Method notes

- Training time covers `fit()` only. Inference time is the scoring of the full final test set, measured separately (the Keras models run one warm-up call first). The per-record time is that total divided by 12,500.
- Model size is the size of the model saved to a temporary file, which is deleted afterwards. It can differ slightly from the files saved by the original training scripts.
- Using different seeds on a fixed test set measures how much the model's own randomness matters. It does not measure the variation across independent test samples.
- All four experiments were run in one session on the same Windows PC (CPU only, no GPU), so the timings are comparable.

### 8.5 Run it

```bash
python robustness.py                 # all four models (long: about half an hour or more on CPU)
python robustness.py ocsvm           # a single model: isolation_forest, autoencoder, lstm or ocsvm
```

It needs TensorFlow, `outputs/autoencoder.keras` (to read the bottleneck size) and `scores/ocsvm_timings.json` (to read `nu`). It does not use any saved model for the results: every model is trained again for each seed.

---

## 9. Demo

`demo.py` loads the four saved models and prints **Normal** or **ANOMALY** for each sample row, next to the row's true attack type. Nothing is retrained.

```bash
python demo.py                  # built-in sample rows
python demo.py my_rows.csv      # your own CSV (first 20 rows)
```

- The built-in sample is 3 normal rows plus one row of each attack type, taken from the final test half.
- The autoencoder thresholds are recomputed from the saved models using the same rule as in training (95th percentile of errors on the normal-only validation split).
- A custom CSV must contain the 32 feature columns, already scaled the same way as the files in `data/processed/`. A `Label` column is optional. A missing file, missing columns or non-numeric values stop the script with a clear message. If a model file is missing, that model is skipped with a note.

Required files: `isolation_forest_model.joblib`, `ocsvm_model.joblib`, `outputs/autoencoder.keras`, `outputs/lstm_autoencoder.keras` and `data/processed/`.

The demo shows model outputs on a handful of rows only. It does not measure real-world detection performance, and the scores are not probabilities.

---

## 10. Interpretation and known limitations

Be honest about these when presenting the work:

- **All four models are only moderately better than chance at best.** The best ROC-AUC is about 0.74 (autoencoder), the Isolation Forest is about 0.59, and the One-Class SVM (0.486) is no better than random guessing.
- **The thresholds are conservative.** The autoencoders flag few rows, so precision is high (about 0.83) but recall is low (about 0.22). The Isolation Forest detects about 35% of attacks while flagging about 15% of normal flows as false alarms.
- **Detection is very uneven across attack types.** The Isolation Forest does reasonably well on backdoor, ransomware, password, mitm and ddos, but is close to a coin flip (or worse) on scanning, injection, dos and especially **xss (AUC 0.454, worse than random)**.
- **Why this happens (likely):** these models flag flows that are *statistically unusual*. Several attacks (xss, injection, scanning) produce flows that look very similar to ordinary traffic at the NetFlow summary level, so they are not "unusual" and slip through.
- **Isolation Forest tuning barely helps.** Validation AUC stays between about 0.54 and 0.60 for every setting (see `tuning_results.csv`). The limit is the model and features, not the settings.
- **`backdoor` (18 final-test rows) and `ransomware` (59 rows) are tiny.** Their scores look good but are statistically unreliable.
- **One fixed test set.** The robustness experiments vary only the model's random seed. They show how stable a model is, not how it would behave on different test samples.
- **One-Class SVM seeds.** The One-Class SVM is deterministic, so its seeds only change which 80% of the normal training rows it sees.
- **Precision/recall depend on the threshold.** A stricter threshold gives fewer false alarms but misses more attacks. Choose according to whether false alarms or missed attacks are costlier.
- The dataset is a sample (50,000 unique flows), not the full NF-ToN-IoT-v2, and is balanced 50/50, which is **not** the real-world attack rate (attacks are normally rare).

---

## 11. Troubleshooting / FAQ

**`FileNotFoundError: ...data/processed/train_normal.csv`**
The CSVs must be inside `data/processed/` next to the scripts (see [section 3](#3-project-structure)).

**`FileNotFoundError: isolation_forest_model.joblib` when running `diagnose.py`**
Run `python isolation_forest.py` first. It creates the model file.

**`ModuleNotFoundError`**
Run `pip install -r requirements.txt`. TensorFlow is needed for `autoencoder.py`, `lstm_autoencoder.py`, `robustness.py` and `demo.py`.

**`robustness.py` or `demo.py` cannot find a model file**
`robustness.py` needs `outputs/autoencoder.keras` and `scores/ocsvm_timings.json`. `demo.py` needs the four model files listed in [section 9](#9-demo). Run the matching training script if one is missing.

**My numbers differ slightly from the README.**
Check you are using the shared CSVs (not a re-generated dataset) and similar `scikit-learn` and `tensorflow` versions. The seeds are fixed, but different library versions can change the results slightly. Timings depend on your computer.

**Can I use the models on new traffic?**
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
- Libraries: scikit-learn, pandas, NumPy, joblib, TensorFlow/Keras, matplotlib, Hugging Face `datasets`.