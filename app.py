
import json
from pathlib import Path

import joblib
from flask import Flask, render_template


ROOT = Path(__file__).resolve().parent

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/supervised")
def supervised():
    return render_template("supervised.html")


@app.route("/clustering")
def clustering():
    # Muat hasil evaluasi jika training sudah dijalankan.
    report_path = ROOT / "reports" / "clustering_metrics.json"
    model_path = ROOT / "models" / "clustering.joblib"
    image_path = ROOT / "static" / "img" / "cluster_pca.png"

    metrics = {}
    cluster_profiles = {}

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    if model_path.exists():
        model_data = joblib.load(model_path)
        cluster_profiles = model_data.get("cluster_profiles", {})

    plot_available = image_path.exists()

    return render_template(
        "clustering.html",
        metrics=metrics,
        cluster_profiles=cluster_profiles,
        plot_available=plot_available,
    )


if __name__ == "__main__":
    app.run(debug=True)
