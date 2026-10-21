
"""
autoencoder.py

Train an Autoencoder using normal IoT traffic only.
Use normal validation data to set the anomaly threshold.
Compare bottleneck sizes using the mixed validation split.
Evaluate the selected model on the same final test split
as isolation_forest.py.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import Model  # type:ignore
from tensorflow.keras.layers import Input, Dense  # type:ignore
from tensorflow.keras.callbacks import EarlyStopping  # type:ignore



#1.Settings


SEED = 42
THRESHOLD_PERCENTILE = 95
THRESHOLD_PERCENTILES = [95, 96, 97, 98, 99]
BOTTLENECK_SIZES = [4, 8, 16]

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



# 2. Load prepared datasets


print("Loading prepared datasets...")

train_df = pd.read_csv(DATA_DIR / "train_normal.csv")
test_df = pd.read_csv(DATA_DIR / "test_mixed.csv")

attack_types = (
    pd.read_csv(DATA_DIR / "test_attack_types.csv")
    .iloc[:, 0]
    .astype(str)
)

X_train_df = train_df.drop(columns=["Label"], errors="ignore")
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
assert "Normal" in set(attack_types), (
    "Expected the exact attack-type value 'Normal'"
)


#Same 50/50 split as isolation_forest.py
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

X_train = X_train_df.to_numpy(dtype=np.float32)
X_val = X_val_df.to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)

y_val = np.asarray(y_val, dtype=int)
y_test = np.asarray(y_test, dtype=int)
t_val = t_val.to_numpy()
t_test = t_test.to_numpy()

print(f"Train (normal only): {len(X_train):,}")
print(
    f"Validation: {len(X_val):,} "
    f"(normal: {int((y_val == 0).sum()):,}, "
    f"attacks: {int((y_val == 1).sum()):,})"
)
print(
    f"Final test: {len(X_test):,} "
    f"(normal: {int((y_test == 0).sum()):,}, "
    f"attacks: {int((y_test == 1).sum()):,})"
)
print(f"Features: {X_train.shape[1]}")



#3.Normal-only training/validation split


#Used for early stopping and calculating the threshold
#The mixed validation set above is used for model comparison

X_fit, X_val_normal = train_test_split(
    X_train,
    test_size=VAL_FRACTION,
    random_state=SEED,
)



#4.Model and evaluation helpers


def build_autoencoder(n_features, bottleneck_size):
    """
    Architecture:
    input -> 64 -> 32 -> bottleneck -> 32 -> 64 -> output
    """

    inputs = Input(shape=(n_features,))

    encoded = Dense(64, activation="relu")(inputs)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(
        bottleneck_size,
        activation="relu",
    )(encoded)

    decoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(decoded)

    outputs = Dense(n_features, activation="linear")(decoded)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    return model


def reconstruction_errors(model, data):
    """Calculate mean squared reconstruction error per record."""

    reconstructed = model.predict(data, verbose=0)

    return np.mean(
        np.square(data - reconstructed),
        axis=1,
    )


def calculate_metrics(y_true, scores, threshold):
    """Calculate metrics from reconstruction errors."""

    predictions = (scores > threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, scores),
    }



#5.Compare bottleneck sizes


print("\n--- BOTTLENECK EXPERIMENTS ---")

experiment_results = []
trained_models = {}

for bottleneck_size in BOTTLENECK_SIZES:

    print(f"\nTraining bottleneck size: {bottleneck_size}")

    #Make experiments repeatable
    tf.keras.utils.set_random_seed(SEED)

    model = build_autoencoder(
        n_features=X_train.shape[1],
        bottleneck_size=bottleneck_size,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    # Measure the model-fitting time.
    start_time = time.time()

    history = model.fit(
        X_fit,
        X_fit,
        validation_data=(X_val_normal, X_val_normal),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1,
    )

    training_time_seconds = time.time() - start_time

    print(
        f"Training time: {training_time_seconds:.2f} seconds"
    )
    print(
        f"Epochs completed: {len(history.history['loss'])}"
    )

    #Calculate threshold using normal-only validation data
    normal_errors = reconstruction_errors(
        model, X_val_normal
    )

    threshold = np.percentile(
        normal_errors,
        THRESHOLD_PERCENTILE,
    )

    #Compare using the same mixed validation records
    validation_errors = reconstruction_errors(
        model, X_val
    )

    metrics = calculate_metrics(
        y_val,
        validation_errors,
        threshold,
    )

    experiment_results.append({
        "bottleneck_size": bottleneck_size,
        "training_time_seconds": training_time_seconds,
        "epochs_completed": len(history.history["loss"]),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": float(threshold),
        **metrics,
    })

    trained_models[bottleneck_size] = {
        "model": model,
        "history": history,
        "threshold": float(threshold),
        "training_time_seconds": training_time_seconds,
    }

    print(
        f"Validation F1: {metrics['f1_score']:.4f}; "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )


#Save comparison table for the report
experiment_df = pd.DataFrame(experiment_results)

experiment_df.to_csv(
    OUTPUT_DIR / "bottleneck_comparison.csv",
    index=False,
)

print("\n--- BOTTLENECK COMPARISON (MIXED VALIDATION) ---")
print(experiment_df.round(4).to_string(index=False))


#Select the highest validation F1-score
#Use ROC-AUC to break ties
ranked_results = experiment_df.sort_values(
    by=["f1_score", "roc_auc"],
    ascending=False,
)

best_bottleneck = int(
    ranked_results.iloc[0]["bottleneck_size"]
)

selected = trained_models[best_bottleneck]
autoencoder = selected["model"]

print(f"\nSelected bottleneck size: {best_bottleneck}")
print(
    "Selection: highest mixed-validation F1-score; "
    "ROC-AUC breaks ties."
)



#6.Compare threshold percentiles on validation data


selected_normal_errors = reconstruction_errors(
    autoencoder,
    X_val_normal,
)

selected_validation_errors = reconstruction_errors(
    autoencoder,
    X_val,
)

threshold_results = []

for percentile in THRESHOLD_PERCENTILES:

    threshold = np.percentile(
        selected_normal_errors,
        percentile,
    )

    metrics = calculate_metrics(
        y_val,
        selected_validation_errors,
        threshold,
    )

    threshold_results.append({
        "percentile": percentile,
        "threshold": float(threshold),
        **metrics,
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_results.csv",
    index=False,
)

print("\n--- THRESHOLD COMPARISON (MIXED VALIDATION) ---")
print(threshold_df.round(4).to_string(index=False))



#7.Final test evaluation


#The final test set is not used to select the bottleneck
#or threshold The 95th-percentile rule remains fixed

final_threshold = np.percentile(
    selected_normal_errors,
    THRESHOLD_PERCENTILE,
)

test_errors = reconstruction_errors(
    autoencoder,
    X_test,
)

y_pred = (test_errors > final_threshold).astype(int)

training_time_seconds = selected["training_time_seconds"]

print("\n--- FINAL TEST RESULTS ---")
print(f"Selected bottleneck: {best_bottleneck}")
print(f"Training time: {training_time_seconds:.2f} seconds")
print(f"Threshold percentile: {THRESHOLD_PERCENTILE}")
print(f"Threshold value: {final_threshold:.8f}")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4),
)
print(
    "Precision:",
    round(precision_score(y_test, y_pred, zero_division=0), 4),
)
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_errors), 4))

print(
    "\nConfusion matrix (rows = actual [normal, attack], "
    "columns = predicted [normal, attack]):"
)
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)



#8.Per-attack-type results


normal_mask = t_test == "Normal"
rows = []

for attack_type in sorted(
    np.unique(t_test),
    key=lambda name: (name == "Normal", name),
):
    mask = t_test == attack_type

    row = {
        "type": attack_type,
        "rows": int(mask.sum()),
        "correct_rate": float(
            (y_pred[mask] == y_test[mask]).mean()
        ),
    }

    if attack_type != "Normal":
        comparison_mask = mask | normal_mask

        row["roc_auc_vs_normal"] = float(
            roc_auc_score(
                y_test[comparison_mask],
                test_errors[comparison_mask],
            )
        )

    rows.append(row)

per_type_df = pd.DataFrame(rows)

print("\n--- PER-TYPE RESULTS (FINAL TEST) ---")
print(per_type_df.round(4).to_string(index=False))

per_type_df.to_csv(
    OUTPUT_DIR / "per_attack_results_autoencoder.csv",
    index=False,
)



#9.Save model and outputs


autoencoder.save(OUTPUT_DIR / "autoencoder.keras")

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

pd.DataFrame(
    selected["history"].history
).to_csv(
    OUTPUT_DIR / "autoencoder_training_history.csv",
    index=False,
)

pd.DataFrame([{
    "selected_bottleneck_size": best_bottleneck,
    "training_time_seconds": training_time_seconds,
    "epochs_completed": len(selected["history"].history["loss"]),
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": float(final_threshold),
}]).to_csv(
    OUTPUT_DIR / "autoencoder_summary.csv",
    index=False,
)

print("\nAll experiments and final evaluation are complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")
