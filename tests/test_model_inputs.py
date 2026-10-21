
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config/feature_schema.json", encoding="utf-8") as f:
    SCHEMA = json.load(f)


@pytest.mark.parametrize(
    "module_name, model_file, function_name",
    [
        ("src.supervised", "supervised.joblib", "predict_supervised"),
        ("src.clustering", "clustering.joblib", "predict_clustering"),
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    ["abc", "", float("nan"), float("inf")],
    ids=["teks", "kosong", "nan", "infinity"],
)
def test_input_tidak_valid(module_name, model_file, function_name, invalid_value):
    # Lewati jika model belum tersedia di branch ini
    if not (ROOT / "models" / model_file).exists():
        pytest.skip("Model belum tersedia")

    module = pytest.importorskip(module_name)

    # Input lengkap dari nilai default schema
    inputs = dict(SCHEMA["defaults"])

    # Ubah satu fitur menjadi nilai tidak valid
    fitur_pertama = SCHEMA["feature_order"][0]
    inputs[fitur_pertama] = invalid_value

    # Model wajib menolak input tidak valid
    with pytest.raises((ValueError, TypeError)):
        getattr(module, function_name)(inputs)
