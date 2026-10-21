
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models/supervised.joblib"


def predict_supervised(inputs: dict) -> dict:
    if not isinstance(inputs, dict):
        raise TypeError("Input harus berupa dictionary.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan training terlebih dahulu."
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_order = artifact["feature_order"]

    missing = [
        feature for feature in feature_order
        if feature not in inputs
        or inputs[feature] is None
        or inputs[feature] == ""
    ]

    if missing:
        raise ValueError(f"Fitur belum diisi: {missing}")

    row = pd.DataFrame(
        [[inputs[feature] for feature in feature_order]],
        columns=feature_order
    )

    # Pastikan input numerik dan tidak mengandung nilai tak hingga
    row = row.apply(pd.to_numeric, errors="raise")

    import numpy as np

    if np.isinf(row.to_numpy(dtype=float)).any():
        raise ValueError("Input tidak boleh mengandung nilai tak hingga.")

    predicted = pipeline.predict(row)[0]
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["model"].classes_

    probability_dict = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "fault_class": str(predicted),
        "confidence": float(max(probabilities)),
        "probabilities": probability_dict
    }
