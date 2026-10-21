
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "clustering.joblib"


@lru_cache(maxsize=1)
def load_clustering_model():
    """Memuat model sekali, tanpa melatih ulang."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model clustering belum tersedia: {MODEL_PATH}. "
            "Jalankan training/train_clustering.py terlebih dahulu "
            "setelah dataset tersedia."
        )

    return joblib.load(MODEL_PATH)


def predict_clustering(inputs: dict) -> dict:
    """Memprediksi cluster dari fitur input pengguna."""
    model = load_clustering_model()
    feature_order = model["feature_order"]

    # Menjaga urutan fitur sesuai saat training.
    row = pd.DataFrame(
        [{name: inputs.get(name) for name in feature_order}],
        columns=feature_order,
    )

    for column in feature_order:
        row[column] = pd.to_numeric(
            row[column], errors="coerce"
        )

    X_imputed = model["imputer"].transform(row)
    X_scaled = model["scaler"].transform(X_imputed)

    cluster_id = int(model["kmeans"].predict(X_scaled)[0])
    coordinates = model["pca"].transform(X_scaled)[0]

    profiles = model.get("cluster_profiles", {})

    return {
        "cluster_id": cluster_id,
        "profile": profiles.get(str(cluster_id), {}),
        "pca_x": float(coordinates[0]),
        "pca_y": float(coordinates[1]),
    }
