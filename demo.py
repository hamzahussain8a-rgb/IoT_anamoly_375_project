
"""
demo.py
Loads the saved model and tests a few IoT records.
"""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "isolation_forest_model.joblib"


def load_model():
    """Load the saved model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def run_demo(sample_size=10):
    """Show predictions for a few test records."""
    model = load_model()
    test_path = DATA_DIR / "test_mixed.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}"
        )

    test_data = pd.read_csv(test_path)

    # Keep the actual labels separate from the model input.
    X_test = test_data.drop(columns=["Label"])
    X_sample = X_test.head(sample_size)

    # The model marks anomalies as -1 and normal records as 1.
    predictions = model.predict(X_sample)

    results = pd.DataFrame({
        "Sample": range(1, len(X_sample) + 1),
        "Actual": [
            "Attack" if value == 1 else "Normal"
            for value in test_data["Label"].head(len(X_sample))
        ],
        "Prediction": [
            "Anomaly / Attack" if value == -1 else "Normal"
            for value in predictions
        ]
    })

    print("\nIoT Anomaly Detection Demo")
    print("--------------------------")
    print(results.to_string(index=False))


if __name__ == "__main__":
    run_demo()
