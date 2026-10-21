
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Semua path dihitung dari root proyek
ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "data/processed/steel_faults_clean.csv"
SCHEMA_PATH = ROOT / "config/feature_schema.json"
MODEL_PATH = ROOT / "models/supervised.joblib"
METRICS_PATH = ROOT / "reports/supervised_metrics.json"
CM_PATH = ROOT / "static/img/confusion_matrix.png"


def main():
    # 1. Periksa file input
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV bersih tidak ditemukan: {CSV_PATH}. "
            "Jalankan python src/preprocessing.py terlebih dahulu."
        )

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema tidak ditemukan: {SCHEMA_PATH}. "
            "Jalankan python src/preprocessing.py terlebih dahulu."
        )

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    df = pd.read_csv(CSV_PATH)
    features = schema["feature_order"]
    target = schema["target_column"]

    # 2. Validasi struktur
    assert target in df.columns, f"Target {target} tidak ditemukan"
    assert all(c in df.columns for c in features), (
        "Ada fitur yang tidak ditemukan di CSV"
    )
    assert target not in features, "Target tidak boleh menjadi fitur"

    X = df[features]
    y = df[target]

    # Baris tanpa target tidak dapat dipakai untuk supervised learning
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    print("Ukuran dataset:", df.shape)
    print("Jumlah fitur:", len(features))
    print("\nDistribusi kelas:")
    print(y.value_counts())

    # 3. Bagi data 80:20
    if y.nunique() != 7:
        raise ValueError(
            f"Diharapkan 7 kelas cacat, ditemukan {y.nunique()}."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Setiap kelas perlu minimal 2 sampel untuk stratifikasi."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nJumlah data training:", len(X_train))
    print("Jumlah data testing:", len(X_test))
    print("\nDistribusi kelas training:")
    print(y_train.value_counts())
    print("\nDistribusi kelas testing:")
    print(y_test.value_counts())

    # 4. Pipeline: imputasi + Random Forest
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Fit hanya pada data training
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi pada data testing
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(
            precision_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test, y_pred,
                average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test, y_pred,
                average="weighted", zero_division=0
            )
        ),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution": {
            str(k): int(v)
            for k, v in y.value_counts().items()
        }
    }

    print("\n=== METRIK EVALUASI ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0
    )

    class_results = {
        name: values
        for name, values in report.items()
        if isinstance(values, dict)
        and name not in ("macro avg", "weighted avg")
    }

    if class_results:
        lowest_recall = min(
            class_results,
            key=lambda name: class_results[name]["recall"]
        )
        metrics["lowest_recall_class"] = lowest_recall
        print("Kelas dengan recall terendah:", lowest_recall)

    # 6. Siapkan folder output
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 7. Simpan pipeline dan urutan fitur
    joblib.dump({
        "pipeline": pipeline,
        "feature_order": features
    }, MODEL_PATH)

    # 8. Simpan metrik aktual
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 9. Simpan confusion matrix dari hasil pengujian
    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=pipeline.named_steps["model"].classes_,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== FILE BERHASIL DISIMPAN ===")
    print("Model:", MODEL_PATH)
    print("Metrik:", METRICS_PATH)
    print("Confusion matrix:", CM_PATH)


if __name__ == "__main__":
    main()
