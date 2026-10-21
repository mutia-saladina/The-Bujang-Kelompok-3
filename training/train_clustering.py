import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Lokasi folder proyek
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "steel_faults_clean.csv"
SCHEMA_PATH = ROOT / "config" / "feature_schema.json"
MODEL_PATH = ROOT / "models" / "clustering.joblib"
REPORT_PATH = ROOT / "reports" / "clustering_metrics.json"
IMAGE_PATH = ROOT / "static" / "img" / "cluster_pca.png"

# Kolom ini bukan fitur clustering
TARGET_COLUMNS = {
    "fault_type",
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
}


def load_feature_order(df):
    """Baca urutan fitur dari schema jika tersedia."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            schema = json.load(file)

        if isinstance(schema, list):
            features = schema
        elif isinstance(schema, dict):
            features = None

            for key in (
                "feature_order",
                "feature_columns",
                "input_features",
                "features",
            ):
                value = schema.get(key)

                if isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                ):
                    features = value
                    break

            if features is None:
                features = []
        else:
            features = []
    else:
        # Fallback sementara sampai schema Orang 2 tersedia.
        features = [
            column
            for column in df.select_dtypes(include="number").columns
            if column not in TARGET_COLUMNS
        ]

    missing = [column for column in features if column not in df.columns]

    if missing:
        raise ValueError(
            f"Kolom pada schema tidak ditemukan di CSV: {missing}"
        )

    features = [
        column for column in features
        if column not in TARGET_COLUMNS
    ]

    if not features:
        raise ValueError("Tidak ditemukan fitur numerik untuk clustering.")

    return features


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset belum tersedia: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    feature_order = load_feature_order(df)

    X = df[feature_order].apply(pd.to_numeric, errors="coerce")

    # Mengisi nilai kosong dan menyamakan skala fitur.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Dataset perlu setidaknya 3 baris untuk evaluasi silhouette.
    if len(X_scaled) < 3:
        raise ValueError("Dataset membutuhkan minimal 3 baris data.")

    max_k = min(10, len(X_scaled) - 1)
    inertias = {}
    silhouette_scores = {}

    # Evaluasi elbow dan silhouette.
    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)
        inertias[str(k)] = float(model.inertia_)

        if k >= 2:
            silhouette_scores[str(k)] = float(
                silhouette_score(X_scaled, labels)
            )

    # Pilih jumlah cluster dengan silhouette tertinggi.
    best_k = int(
        max(silhouette_scores, key=silhouette_scores.get)
    )

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA untuk visualisasi 2 dimensi.
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    # Profil cluster dalam skala fitur aslinya.
    profile_df = df[feature_order].copy()
    profile_df["cluster"] = cluster_labels

    profiles = profile_df.groupby("cluster")[feature_order].mean()
    cluster_profiles = profiles.round(3).to_dict(orient="index")
    cluster_profiles = {
        str(cluster): profile
        for cluster, profile in cluster_profiles.items()
    }

    # Simpan gambar PCA.
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.75,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Steel Plates — K-Means Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(IMAGE_PATH, dpi=150)
    plt.close()

    # Simpan model beserta preprocessing untuk digunakan aplikasi.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "kmeans": kmeans,
            "pca": pca,
            "feature_order": feature_order,
            "cluster_profiles": cluster_profiles,
            "metrics": {
                "best_k": best_k,
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "explained_variance_ratio": (
                    pca.explained_variance_ratio_.tolist()
                ),
            },
        },
        MODEL_PATH,
    )

    # Simpan ringkasan evaluasi.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "samples": len(df),
                "feature_count": len(feature_order),
                "best_k": best_k,
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "cluster_sizes": {
                    str(int(cluster)): int(count)
                    for cluster, count in pd.Series(
                        cluster_labels
                    ).value_counts().sort_index().items()
                },
            },
            file,
            indent=2,
        )

    print(f"Jumlah data      : {len(df)}")
    print(f"Jumlah fitur     : {len(feature_order)}")
    print(f"Jumlah cluster   : {best_k}")
    print(f"Model tersimpan  : {MODEL_PATH}")
    print(f"Evaluasi tersimpan: {REPORT_PATH}")
    print(f"Visualisasi      : {IMAGE_PATH}")


if __name__ == "__main__":
    main()