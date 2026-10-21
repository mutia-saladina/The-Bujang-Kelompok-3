from pathlib import Path
import json

from flask import Flask, render_template, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "config" / "feature_schema.json"


def load_schema():
    """Membaca schema tanpa membuat aplikasi gagal start jika file belum ada."""
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH, encoding="utf-8") as file:
        return json.load(file)


@app.route("/")
def index():
    return render_template("index.html")


# Endpoint "predict" dipertahankan agar url_for('predict') di base.html bekerja.
@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
@app.route("/supervised", methods=["GET", "POST"], endpoint="supervised_page")
def supervised():
    result = None
    error = None
    schema = load_schema()

    if schema is None:
        features = []
        defaults = {}
        error = (
            "File config/feature_schema.json belum ditemukan. "
            "Jalankan preprocessing dan pastikan schema sudah dibuat."
        )
    else:
        features = schema.get("feature_order", [])
        defaults = schema.get("defaults", {})
        if not features:
            error = "Daftar feature_order pada schema kosong atau tidak tersedia."

    if request.method == "POST" and schema is not None and features:
        try:
            from src.supervised import predict_supervised

            inputs = {}
            for feature in features:
                raw_value = request.form.get(feature, "").strip()
                if not raw_value:
                    raise ValueError(f"Fitur '{feature}' wajib diisi.")
                inputs[feature] = float(raw_value)

            result = predict_supervised(inputs)

        except (ValueError, TypeError) as exc:
            error = str(exc)
        except FileNotFoundError as exc:
            error = (
                f"{exc} Pastikan training sudah dijalankan dan "
                "models/supervised.joblib tersedia."
            )
        except Exception as exc:
            error = f"Prediksi gagal: {exc}"

    return render_template(
        "supervised.html",
        features=features,
        defaults=defaults,
        result=result,
        error=error,
    )


@app.route("/cluster", methods=["GET", "POST"])
def cluster():
    result = None
    error = None

    if request.method == "POST":
        error = (
            "Model clustering belum dihubungkan. "
            "Koordinasikan integrasi dengan anggota yang mengerjakan clustering."
        )

    return render_template(
        "clustering.html",
        result=result,
        error=error,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # Jalankan dari root proyek dengan: python app.py
    app.run(debug=True)
