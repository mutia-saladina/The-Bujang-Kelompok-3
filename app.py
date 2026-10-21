from pathlib import Path
import json

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "config" / "feature_schema.json"

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)


def load_schema():
    if not SCHEMA_PATH.is_file():
        return None
    with SCHEMA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@app.route("/")
def index():
    return render_template("index.html")


# Keep endpoint "predict" because base.html and index.html use url_for('predict').
@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
@app.route("/supervised", methods=["GET", "POST"], endpoint="supervised_page")
def supervised():
    schema = load_schema()
    result = None
    error = None

    if schema is None:
        features, defaults = [], {}
        error = "Schema belum tersedia. Jalankan: python src/preprocessing.py"
    else:
        features = schema.get("feature_order", [])
        defaults = schema.get("defaults", {})
        if not features:
            error = "feature_order tidak ditemukan atau kosong di feature_schema.json."

    if request.method == "POST" and schema is not None and features:
        try:
            from src.supervised import predict_supervised

            inputs = {}
            for feature in features:
                raw_value = request.form.get(feature, "").strip()
                if raw_value == "":
                    raise ValueError(f"Fitur '{feature}' wajib diisi.")
                inputs[feature] = float(raw_value)

            result = predict_supervised(inputs)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        except FileNotFoundError as exc:
            error = f"{exc} Jalankan training terlebih dahulu."
        except Exception as exc:
            app.logger.exception("Prediksi supervised gagal")
            error = f"Prediksi gagal: {exc}"

    return render_template(
        "supervised.html",
        features=features,
        defaults=defaults,
        result=result,
        error=error,
    )


@app.route("/cluster")
def cluster():
    metrics_path = ROOT / "reports" / "clustering_metrics.json"
    model_path = ROOT / "models" / "clustering.joblib"
    metrics = {}
    cluster_profiles = {}

    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            app.logger.exception("Tidak dapat membaca metrik clustering")

    if model_path.is_file():
        try:
            import joblib
            artifact = joblib.load(model_path)
            cluster_profiles = artifact.get("cluster_profiles", {})
        except Exception:
            app.logger.exception("Tidak dapat membaca model clustering")

    return render_template(
        "clustering.html",
        metrics=metrics,
        cluster_profiles=cluster_profiles,
    )


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)