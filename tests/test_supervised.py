
import json
import pandas as pd

from src.supervised import predict_supervised

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

with open(ROOT / "config/feature_schema.json", encoding="utf-8") as f:
    schema = json.load(f)

df = pd.read_csv(ROOT / "data/processed/steel_faults_clean.csv")
features = schema["feature_order"]

# Pilih satu baris yang memiliki fitur lengkap
sample_df = df[features].dropna()

if sample_df.empty:
    raise ValueError("Tidak ada baris dengan fitur lengkap.")

sample = sample_df.iloc[0].to_dict()

result = predict_supervised(sample)

print("Hasil prediksi:", result)
print("Jumlah probabilitas:", sum(result["probabilities"].values()))

assert result["fault_class"] in result["probabilities"]
assert len(result["probabilities"]) == 7
assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6

print("Uji prediksi berhasil.")
