
"""
autoencoder.py

Train an Autoencoder using normal IoT traffic only.
Use normal validation data to set the anomaly threshold.
Evaluate on the same final test split as isolation_forest.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import Model #type:ignore
from tensorflow.keras.layers import Input, Dense #type:ignore
from tensorflow.keras.callbacks import EarlyStopping #type:ignore



#1.Settings


SEED = 42
THRESHOLD_PERCENTILE = 95
THRESHOLD_PERCENTILES = [95, 96, 97, 98, 99]

EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 15
VAL_FRACTION = 0.10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)



#2.Load the shared processed datasets


print("Loading prepared datasets...")

train_df = pd.read_csv(DATA_DIR / "train_normal.csv")
test_df = pd.read_csv(DATA_DIR / "test_mixed.csv")

attack_types = (
    pd.read_csv(DATA_DIR / "test_attack_types.csv")
    .iloc[:, 0]
    .astype(str)
)

#Training data contains normal traffic only
X_train_df = train_df.drop(columns=["Label"], errors="ignore")

#Keep test features in exactly the same order as training features
X_all_df = test_df.drop(columns=["Label"])[X_train_df.columns]

y_all = test_df["Label"].to_numpy().astype(int)

assert len(attack_types) == len(test_df), (
    "test_attack_types.csv does not match test_mixed.csv"
)

assert set(np.unique(y_all)).issubset({0, 1}), (
    "Labels must be 0 (normal) or 1 (attack)"
)

assert np.isfinite(X_train_df.to_numpy()).all()
assert np.isfinite(X_all_df.to_numpy()).all()

#Use the same 50/50 split as isolation_forest.py
#The final test records are kept separate from model training
X_val_df, X_test_df, y_val, y_test, t_val, t_test = (
    train_test_split(
        X_all_df,
        y_all,
        attack_types,
        test_size=0.5,
        stratify=attack_types,
        random_state=SEED,
    )
)

#Convert features to float32 for TensorFlow.
X_train = X_train_df.to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)

print(f"Training records: {len(X_train):,}")
print(f"Final test records: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Normal test records: {(y_test == 0).sum():,}")
print(f"Attack test records: {(y_test == 1).sum():,}")



#3.Create a normal-only validation set


#This validation data comes from the normal training records
#It is used for early stopping and threshold selection
X_fit, X_val_normal = train_test_split(
    X_train,
    test_size=VAL_FRACTION,
    random_state=SEED,
)

print(f"Training subset: {len(X_fit):,}")
print(f"Normal validation subset: {len(X_val_normal):,}")



#4.Build the Autoencoder


n_features = X_train.shape[1]
bottleneck_size = max(4, n_features // 4)

inputs = Input(shape=(n_features,))

encoded = Dense(64, activation="relu")(inputs)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(bottleneck_size, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
outputs = Dense(n_features, activation="linear")(decoded)

autoencoder = Model(inputs=inputs, outputs=outputs)

autoencoder.compile(
    optimizer="adam",
    loss="mse",
)

autoencoder.summary()



#5.Train using normal traffic only


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
)

print("\nTraining Autoencoder...")

history = autoencoder.fit(
    X_fit,
    X_fit,
    validation_data=(X_val_normal, X_val_normal),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1,
)

print(f"\nTraining stopped after {len(history.history['loss'])} epochs.")



#6.Calculate reconstruction errors


def reconstruction_errors(model, data):
    """Calculate the mean squared reconstruction error per record."""
    reconstructed = model.predict(data, verbose=0)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    return errors


#Use only normal validation records to choose the threshold
normal_validation_errors = reconstruction_errors(
    autoencoder, X_val_normal
)

#Final threshold is fixed at the 95th percentile
threshold = np.percentile(
    normal_validation_errors,
    THRESHOLD_PERCENTILE,
)

print(f"\nSelected percentile: {THRESHOLD_PERCENTILE}")
print(f"Anomaly threshold: {threshold:.6f}")



#7.Evaluate on the shared final test set


test_errors = reconstruction_errors(autoencoder, X_test)

#Higher reconstruction error means more likely to be an attack.
y_pred = (test_errors > threshold).astype(int)

print("\n--- FINAL TEST RESULTS: AUTOENCODER ---")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Recall   :", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))
print("F1-score :", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))
print("ROC-AUC  :", round(
    roc_auc_score(y_test, test_errors), 4
))

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    zero_division=0,
))



#8.Evaluate the threshold percentiles


threshold_rows = []

for percentile in THRESHOLD_PERCENTILES:
    current_threshold = np.percentile(
        normal_validation_errors, percentile
    )

    current_pred = (test_errors > current_threshold).astype(int)

    threshold_rows.append({
        "percentile": percentile,
        "threshold": float(current_threshold),
        "accuracy": accuracy_score(y_test, current_pred),
        "precision": precision_score(
            y_test, current_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, current_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, current_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, test_errors),
    })

threshold_df = pd.DataFrame(threshold_rows)

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_results.csv",
    index=False,
)

print("\n--- THRESHOLD COMPARISON ---")
print(threshold_df.round(4).to_string(index=False))



#9.Results for individual attack types


test_types = t_test.to_numpy()
normal_mask = test_types == "Normal"

per_type_rows = []

for attack_type in sorted(
    np.unique(test_types),
    key=lambda name: (name == "Normal", name),
):
    mask = test_types == attack_type

    correct_rate = float(
        (y_pred[mask] == y_test[mask]).mean()
    )

    row = {
        "type": attack_type,
        "rows": int(mask.sum()),
        "correct_rate": correct_rate,
    }

    if attack_type != "Normal":
        comparison_mask = mask | normal_mask

        row["roc_auc_vs_normal"] = float(
            roc_auc_score(
                y_test[comparison_mask],
                test_errors[comparison_mask],
            )
        )

    per_type_rows.append(row)

per_type_df = pd.DataFrame(per_type_rows)

per_type_df.to_csv(
    OUTPUT_DIR / "per_attack_results_autoencoder.csv",
    index=False,
)

print("\n--- PER-ATTACK RESULTS ---")
print(per_type_df.round(4).to_string(index=False))



#10.Save outputs for the group

#These scores and labels refer to the same final test records
#used by isolation_forest.py.
np.save(
    OUTPUT_DIR / "scores_autoencoder.npy",
    test_errors,
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test,
)

np.save(
    OUTPUT_DIR / "y_pred_autoencoder.npy",
    y_pred,
)

autoencoder.save(
    OUTPUT_DIR / "autoencoder.keras"
)

#Save training history so a loss curve can be plotted later.
pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / "autoencoder_training_history.csv",
    index=False,
)

print("\nSaved files:")
print("outputs/scores_autoencoder.npy")
print("outputs/y_test.npy")
print("outputs/y_pred_autoencoder.npy")
print("outputs/autoencoder.keras")
print("outputs/threshold_results.csv")
print("outputs/per_attack_results_autoencoder.csv")
print("outputs/autoencoder_training_history.csv")
print("\nAutoencoder completed successfully.")
