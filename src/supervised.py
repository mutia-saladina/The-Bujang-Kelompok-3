from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "supervised.joblib"


def predict_supervised(inputs: dict) -> dict:
    """Predict one steel-plate fault from a mapping of feature names to values."""
    if not isinstance(inputs, dict):
        raise TypeError("Input harus berupa dictionary.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Jalankan python training/train_supervised.py terlebih dahulu."
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact.get("pipeline")
    feature_order = artifact.get("feature_order")

    if pipeline is None or not feature_order:
        raise ValueError(
            "Format model tidak valid: pipeline atau feature_order tidak tersedia."
        )

    missing = [
        feature for feature in feature_order
        if feature not in inputs or inputs[feature] is None or inputs[feature] == ""
    ]
    if missing:
        raise ValueError(f"Fitur belum diisi: {missing}")

    try:
        row = pd.DataFrame(
            [[float(inputs[feature]) for feature in feature_order]],
            columns=feature_order,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Semua nilai fitur harus berupa angka.") from exc

    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Input harus berupa angka terbatas, bukan NaN atau infinity.")

    predicted = pipeline.predict(row)[0]
    classes = pipeline.named_steps["model"].classes_
    probabilities = pipeline.predict_proba(row)[0]

    probability_dict = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "fault_class": str(predicted),
        "confidence": float(max(probabilities)),
        "probabilities": probability_dict,
    }
