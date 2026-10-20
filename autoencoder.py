"""
autoencoder.py-Member 2 Part
Autoencoder trained on normal traffic only.The reconstruction error (MSE) is
the anomaly score, and the threshold is a percentile of the errors on held-out
normal data.
Input:data/processed/train_normal.csv, test_mixed.csv, test_attack_types.csv
Output:outputs/ (scores, model, threshold, metrics tables, loss plot)
Run:python autoencoder.py
"""

import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")              #save plots to a file,no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras #type:ignore
from tensorflow.keras import layers #type:ignore


#Settings

SEED = 42                      #same results every run
THRESHOLD_PERCENTILE = 95      #chosen threshold (the sweep below tries 95-99)
EPOCHS = 300                  #maximum epochs (early stopping usually ends sooner)
BATCH_SIZE = 64
PATIENCE = 15                  #early stopping: epochs to wait for improvement
VAL_FRACTION = 0.1             #share of normal training data held out for validation

TRAIN_PATH = "data/processed/train_normal.csv"
TEST_PATH = "data/processed/test_mixed.csv"
TYPES_PATH = "data/processed/test_attack_types.csv"
OUT_DIR = "outputs"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


#1.Load the data Member 1 prepared

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    #Every column except the label is a feature.The model never sees 'Label'
    feature_cols = [c for c in train_df.columns if c != "Label"]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test file is missing training features: {missing}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["Label"].to_numpy(dtype=int)    # 0 = normal,1 = attack

    #The data is already cleaned, so just check it instead of changing it
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("NaN or infinite values found. Ask Member 1 to check the data.")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Test labels: normal={int((y_test == 0).sum())}, attack={int((y_test == 1).sum())}")
    print(f"Train value range: {X_train.min():.2f} to {X_train.max():.2f}")
    return X_train, X_test, y_test, feature_cols


#2.Build the autoencoder

def build_autoencoder(n_features):
    """
    n_features -> 64 -> 32 -> bottleneck -> 32 -> 64 -> n_features
    The small bottleneck forces the network to learn a compact summary of
    normal behaviour instead of copying its input
    """
    bottleneck = max(4, n_features // 4)

    inputs = keras.Input(shape=(n_features,))

    #Encoder:compress
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    code = layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)

    #Decoder:rebuild
    x = layers.Dense(32, activation="relu")(code)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_features, activation="linear")(x)  #scaled data can be negative

    model = keras.Model(inputs, outputs, name="dense_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model



#3.Helpers:scores,metrics,per-attack report,loss plot

def reconstruction_error(model, X):
    #Mean squared error between each flow and its reconstruction
    X_rebuilt = model.predict(X, batch_size=1024, verbose=0)
    return np.mean(np.square(X - X_rebuilt), axis=1)


def evaluate_threshold(y_true, scores, threshold):
    #Flag flows with score > threshold as attacks,then compute the metrics
    y_pred = (scores > threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def per_attack_type_report(scores, threshold, out_path):
    #Detection rate per attack type. For 'Normal' it is the false positive rate
    if not os.path.exists(TYPES_PATH):
        print("test_attack_types.csv not found, skipping per-attack-type report.")
        return
    types = pd.read_csv(TYPES_PATH).iloc[:, 0]       # one name per test row
    if len(types) != len(scores):
        print("test_attack_types.csv length does not match test set, skipping.")
        return
    df = pd.DataFrame({"AttackType": types.values, "flagged": scores > threshold})
    report = df.groupby("AttackType")["flagged"].agg(["sum", "count"])
    report.columns = ["flagged", "samples"]
    report["detection_rate"] = report["flagged"] / report["samples"]
    report = report.sort_values("detection_rate", ascending=False)
    report.to_csv(out_path)
    print("\nDetection rate per attack type (Normal row = false positive rate):")
    print(report.round(4).to_string())


def plot_loss(history, path):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


#4.Train, choose threshold, score the test set, save everything

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train, X_test, y_test, feature_cols = load_data()

    #Hold out some NORMAL data.It is used for early stopping and for the
    #threshold, never for training,and the test set is used for neither
    X_fit, X_val = train_test_split(X_train, test_size=VAL_FRACTION, random_state=SEED)

    model = build_autoencoder(X_train.shape[1])
    model.summary()

    #Stop when validation loss stops improving and keep the best weights
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    start = time.time()
    history = model.fit(
        X_fit, X_fit,                         #input and target are the same
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        shuffle=True,
        verbose=2,
    )
    train_time = time.time() - start
    epochs_run = len(history.history["loss"])
    print(f"\nTraining time: {train_time:.1f}s over {epochs_run} epochs")

    #Errors on normal validation data show what "normal" looks like
    val_errors = reconstruction_error(model, X_val)

    #Errors on the test set are the anomaly scores Members 4 and 5 use
    start = time.time()
    test_scores = reconstruction_error(model, X_test)
    inference_time = time.time() - start

    #Try thresholds from the 95th to the 99th percentile of normal errors
    rows = []
    for pct in [95, 96, 97, 98, 99]:
        row = evaluate_threshold(y_test, test_scores, np.percentile(val_errors, pct))
        row["percentile"] = pct
        rows.append(row)
    sweep = pd.DataFrame(rows)[["percentile", "threshold", "accuracy",
                                "precision", "recall", "f1"]]
    print("\nPercentile sweep:")
    print(sweep.round(4).to_string(index=False))

    #Final threshold and its metrics
    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
    metrics = evaluate_threshold(y_test, test_scores, threshold)
    auc = roc_auc_score(y_test, test_scores)         #does not depend on the threshold
    print(f"\nChosen percentile: {THRESHOLD_PERCENTILE} | threshold: {threshold:.6f}")
    print({k: round(v, 4) for k, v in metrics.items()}, f"ROC-AUC: {auc:.4f}")

    per_attack_type_report(test_scores, threshold,
                           os.path.join(OUT_DIR, "autoencoder_per_attack_type.csv"))

    #Save everything Score order matches test_mixed.csv row order
    np.save(os.path.join(OUT_DIR, "scores_autoencoder.npy"), test_scores)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    model.save(os.path.join(OUT_DIR, "autoencoder.keras"))
    sweep.to_csv(os.path.join(OUT_DIR, "autoencoder_percentile_sweep.csv"), index=False)
    plot_loss(history, os.path.join(OUT_DIR, "autoencoder_loss_curve.png"))

    with open(os.path.join(OUT_DIR, "autoencoder_threshold.json"), "w") as f:
        json.dump({
            "percentile": THRESHOLD_PERCENTILE,
            "threshold": threshold,
            "roc_auc": auc,
            "metrics_at_threshold": metrics,
            "epochs_run": epochs_run,
            "train_time_sec": train_time,           #for Member 5's efficiency table
            "inference_time_sec": inference_time,
            "n_test_samples": int(len(X_test)),
            "n_parameters": int(model.count_params()),   #model size
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "seed": SEED,
        }, f, indent=2)

    print(f"\nSaved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()