
from pathlib import Path
import json
import pandas as pd

# Lokasi utama repository
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data/processed/steel_faults_clean.csv"
SCHEMA_PATH = BASE_DIR / "config/feature_schema.json"

# Membaca dataset dan feature schema
df = pd.read_csv(DATA_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)


def test_dataset():
    """Memastikan ukuran dan kualitas dataset."""
    assert len(df) == 1941
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_jumlah_fitur():
    """Memastikan dataset memiliki 27 fitur."""
    assert len(schema["feature_order"]) == 27
    assert df.shape[1] == 28

    assert df.columns.tolist() == (
        schema["feature_order"] + ["fault_type"]
    )


def test_label():
    """Memastikan tujuh jenis cacat tersedia."""
    assert "fault_type" in df.columns
    assert df["fault_type"].nunique() == 7

    assert set(df["fault_type"].unique()) == set(
        schema["fault_classes"]
    )


def test_tipe_data():
    """Memastikan seluruh fitur prediktor numerik."""
    for col in schema["feature_order"]:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_tidak_ada_label_leakage():
    """Memastikan target tidak menjadi fitur model."""
    assert "fault_type" not in schema["feature_order"]

    for label in schema["fault_classes"]:
        assert label not in schema["feature_order"]


def test_feature_schema():
    """Memastikan default tersedia untuk 27 fitur."""
    assert schema["target_column"] == "fault_type"

    assert list(schema["defaults"].keys()) == (
        schema["feature_order"]
    )
