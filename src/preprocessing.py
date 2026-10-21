
from pathlib import Path
import json
import pandas as pd

# Menentukan lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "faults.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "steel_faults_clean.csv"
SCHEMA_PATH = BASE_DIR / "config" / "feature_schema.json"

FAULT_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults"
]


def main():
    print("=== PREPROCESSING STEEL PLATES FAULTS ===")

    # 1. Memeriksa keberadaan dataset
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {RAW_PATH}"
        )

    # 2. Membaca dataset asli
    df = pd.read_csv(RAW_PATH)

    print("Dataset asli:", df.shape)

    # 3. Validasi kolom label
    missing_cols = [
        col for col in FAULT_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Kolom label tidak ditemukan: {missing_cols}"
        )

    labels = df[FAULT_COLS]

    if not labels.isin([0, 1]).all().all():
        raise ValueError("Label harus bernilai 0 atau 1")

    if not labels.sum(axis=1).eq(1).all():
        raise ValueError(
            "Setiap baris harus memiliki tepat satu label"
        )

    # 4. Pemeriksaan kualitas data
    print("Total missing:", df.isnull().sum().sum())
    print("Duplikat:", df.duplicated().sum())

    # 5. Transformasi tujuh label menjadi satu target
    df_clean = df.drop(columns=FAULT_COLS).copy()
    df_clean["fault_type"] = labels.idxmax(axis=1)

    # 6. Menentukan fitur prediktor
    feature_order = [
        col for col in df_clean.columns
        if col != "fault_type"
    ]

    if len(feature_order) != 27:
        raise ValueError(
            f"Jumlah fitur harus 27, ditemukan {len(feature_order)}"
        )

    if not all(
        pd.api.types.is_numeric_dtype(df_clean[col])
        for col in feature_order
    ):
        raise ValueError("Terdapat fitur non-numerik")

    # 7. Membuat feature schema
    defaults = {
        col: float(df_clean[col].median())
        for col in feature_order
    }

    schema = {
        "feature_order": feature_order,
        "target_column": "fault_type",
        "fault_classes": FAULT_COLS,
        "defaults": defaults
    }

    # 8. Menyimpan hasil preprocessing
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(OUTPUT_PATH, index=False)

    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    # 9. Menampilkan ringkasan
    print("\n=== HASIL PREPROCESSING ===")
    print("Dataset bersih:", df_clean.shape)
    print("Jumlah fitur:", len(feature_order))
    print("Jumlah kelas:", df_clean["fault_type"].nunique())
    print("CSV:", OUTPUT_PATH)
    print("Schema:", SCHEMA_PATH)
    print("\nPREPROCESSING BERHASIL!")


if __name__ == "__main__":
    main()
