
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

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping


# 1. SETTINGS

SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 15
VAL_FRACTION = 0.10
THRESHOLD_PERCENTILE = 95

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# 2. LOAD DATA

print("Loading IoT dataset...")

train_df = pd.read_csv(DATA_DIR / "train_normal.csv")
test_df = pd.read_csv(DATA_DIR / "test_mixed.csv")
attack_types = pd.read_csv(DATA_DIR / "test_attack_types.csv").iloc[:, 0]

X_train_df = train_df.drop(columns=["Label"], errors="ignore")
X_all_df = test_df.drop(columns=["Label"])[X_train_df.columns]
y_all = test_df["Label"].to_numpy().astype(int)

X_val_df, X_test_df, y_val, y_test, t_val, t_test = train_test_split(
    X_all_df,
    y_all,
    attack_types,
    test_size=0.5,
    stratify=attack_types,
    random_state=SEED,
)

X_train = X_train_df.to_numpy(dtype=np.float32)
X_val = X_val_df.to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)

# Same normal-only training split as Basic Autoencoder
X_fit, X_val_normal = train_test_split(
    X_train,
    test_size=VAL_FRACTION,
    random_state=SEED,
)

n_features = X_train.shape[1]

# Sequence length 1 because the dataset has no preserved time ordering
X_fit = X_fit.reshape(-1, 1, n_features)
X_val_normal = X_val_normal.reshape(-1, 1, n_features)
X_val = X_val.reshape(-1, 1, n_features)
X_test = X_test.reshape(-1, 1, n_features)

print("Training shape:", X_fit.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)


# 3. BUILD LSTM AUTOENCODER

inputs = Input(shape=(1, n_features))

encoded = LSTM(64, return_sequences=True)(inputs)
encoded = LSTM(32, return_sequences=False)(encoded)

repeated = RepeatVector(1)(encoded)

decoded = LSTM(32, return_sequences=True)(repeated)
decoded = LSTM(64, return_sequences=True)(decoded)

outputs = TimeDistributed(Dense(n_features))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

model.summary()


# 4. TRAIN MODEL

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
)

print("\nTraining LSTM Autoencoder...")

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

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# 5. RECONSTRUCTION ERROR

def reconstruction_errors(data):
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed), axis=(1, 2))


normal_errors = reconstruction_errors(X_val_normal)

threshold = np.percentile(
    normal_errors,
    THRESHOLD_PERCENTILE,
)

print(f"Anomaly threshold: {threshold:.6f}")


# 6. VALIDATION METRICS

validation_errors = reconstruction_errors(X_val)
validation_predictions = (validation_errors > threshold).astype(int)

print("\n--- VALIDATION RESULTS ---")
print("F1-score:", f1_score(y_val, validation_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_val, validation_errors))


# 7. FINAL TEST EVALUATION

test_errors = reconstruction_errors(X_test)
y_pred = (test_errors > threshold).astype(int)

print("\n--- LSTM FINAL TEST RESULTS ---")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_errors), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)


# 8. SAVE RESULTS

model.save(OUTPUT_DIR / "lstm_autoencoder.keras")

np.save(OUTPUT_DIR / "scores_lstm.npy", test_errors)
np.save(OUTPUT_DIR / "y_pred_lstm.npy", y_pred)

pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / "lstm_training_history.csv",
    index=False,
)

pd.DataFrame([{
    "training_time_seconds": training_time,
    "epochs_completed": len(history.history["loss"]),
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": float(threshold),
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, test_errors),
}]).to_csv(
    OUTPUT_DIR / "lstm_summary.csv",
    index=False,
)

print("\nLSTM model and results saved successfully!")
