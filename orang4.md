# ORANG 4 — ML Engineer Unsupervised, Visualisasi, dan UI

**Navigasi tugas:** [Orang 1 — Koordinator & Web](orang1.md) · [Orang 2 — Data & QA](orang2.md) · [Orang 3 — Supervised](orang3.md) · [Orang 4 — Clustering & UI](orang4.md)

> **Proyek:** Kelompok 3, Mata Kuliah Machine Learning, TRIN — **Steel Plates Faults**.  
> **Dataset:** https://www.kaggle.com/datasets/uciml/faulty-steel-plates  
> **Web:** Flask di `http://127.0.0.1:5000` dengan dua paradigma: *supervised learning* + *unsupervised learning*.  
> **Status izin:** Dalam panduan resmi, Steel Plates Faults dialokasikan ke Kelompok 2, bukan Kelompok 3. **Konfirmasi persetujuan dan ketiadaan duplikasi kepada koordinator/dosen sebelum proyek diserahkan.**


## 1. Identitas dan peran

- **Nama / NIM:** `[Isi orang 4]`
- **Peran:** ML Engineer 2 **merangkap Frontend Developer** untuk visualisasi dan perapian antarmuka (penyesuaian kelompok empat orang).
- **Tugas:** clustering K-Means, pemilihan jumlah cluster, evaluasi Silhouette/DBI/CH, PCA, profil cluster, ekspor artefak, halaman `/clustering`, dan CSS.
- **Ketergantungan:** [kontrak dataset Orang 2](orang2.md#kontrak-output) dan [base layout Orang 1](orang1.md#2-peta-proyek-dan-pembagian-kepemilikan-file).
- **Bukan tugasmu:** klasifikasi kelas cacat sebagai label pelatihan; saat clustering target harus **tidak ada** pada matriks X.

## 2. Output yang menjadi tanggung jawabmu

```text
training/train_clustering.py
training/clustering_experiment.ipynb
src/clustering.py
models/clustering.joblib
templates/clustering.html
static/css/style.css
static/img/cluster_pca.png
reports/clustering_metrics.json
```

## Langkah Hari 3

### Tahap A — data dan persiapan

1. Baca `config/feature_schema.json` dari Orang 2 dan `data/processed/steel_faults_clean.csv`.
2. Buat `X = df[feature_order]`; **jangan** menggunakan `fault_type` maupun kolom one-hot cacat di `X`.
3. Siapkan `SimpleImputer(strategy='median')` dan `StandardScaler()`. Untuk unsupervised, keduanya dilatih pada dataset pengelompokan yang tersedia; dokumentasikan bahwa hasil ini bersifat eksplorasi.
4. Simpan urutan fitur pada artefak agar form Flask memasok fitur dengan susunan identik.

### Tahap B — cari K yang sesuai

5. Uji **Elbow Method** untuk `K=1..10` dan **Silhouette Score** untuk `K=2..10`. Dataset relatif kecil, tetapi gunakan pengaturan sampel evaluasi jika laptop lambat dan tulis keputusan tersebut di laporan.
6. Pilih K berdasarkan Silhouette, pola WCSS, jumlah anggota yang masuk akal, dan kemudahan menjelaskan pola. **Jangan langsung menganggap K=7** karena terdapat tujuh kelas cacat; clustering tidak menggunakan kelas saat training.
7. Latih `KMeans(n_clusters=K_terpilih, random_state=42, n_init=10)`, lalu catat Silhouette, Davies-Bouldin Index, Calinski-Harabasz Index, dan ukuran setiap cluster.
8. Jalankan `PCA(n_components=2)` pada hasil scaling, buat scatter plot berwarna menurut cluster, serta cantumkan varians yang dijelaskan PC1 dan PC2.
9. Buat tabel **rata-rata fitur tiap cluster** (gunakan fitur asli untuk interpretasi) dan deskripsi pendek seperti `cenderung area cacat lebih besar`; jangan memberi label `normal`, `rusak berat`, atau `K_Scatch` secara pasti hanya dari nomor cluster.
10. Opsional: setelah clustering selesai, bandingkan distribusi `fault_type` asli pada tiap cluster untuk mengetahui kecenderungan; catat ini **validasi/interpretasi eksternal**, bukan fitur masukan.

### Tahap C — ekspor dan integrasi

11. Simpan satu artefak `models/clustering.joblib` yang mencakup `imputer`, `scaler`, `kmeans`, `pca`, `feature_order`, `cluster_profiles`, dan metrik evaluasi. Jangan simpan hanya nomor cluster.
12. Buat fungsi `predict_clustering(inputs: dict) -> dict` pada `src/clustering.py`. Input 1 observasi berupa dictionary angka sesuai schema; keluaran minimal `cluster_id`, ringkasan `profile` dari pusat/pola cluster hasil training, dan koordinat PCA (`pca.x`, `pca.y`).
13. Buat `templates/clustering.html` yang *extends* `base.html` Orang 1; halaman harus memperlihatkan input sensor/geometri, nomor cluster, deskripsi pola, dan **gambar PCA asli**, bukan ilustrasi rekaan.
14. Buat `static/css/style.css` untuk konsistensi seluruh halaman (form, hasil, navigasi, mobile). Koordinasikan nama class CSS dengan Orang 1 dan Orang 3 agar tidak ada yang mengganti markup anggota lain diam-diam.
15. Push minimal 3 commit Hari 3, contoh `[FEAT] Uji elbow dan silhouette`, `[FEAT] Simpan pipeline clustering`, `[STYLE] Tata layout modul clustering`.

### Cuplikan konsep training (bukan hasil evaluasi)

```python
import json, pandas as pd, joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

schema = json.load(open('config/feature_schema.json', encoding='utf-8'))
X = pd.read_csv('data/processed/steel_faults_clean.csv')[schema['feature_order']]
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(X))
K = 3  # CONTOH SAJA; ganti dengan K hasil eksperimen
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_scaled)
pca = PCA(n_components=2).fit(X_scaled)
score = silhouette_score(X_scaled, kmeans.labels_)
print('Silhouette aktual:', score)
# Tambahkan profile dan metrics dari perhitungan asli sebelum ekspor.
joblib.dump({'imputer':imputer, 'scaler':scaler, 'kmeans':kmeans,
             'pca':pca, 'feature_order':schema['feature_order']},
            'models/clustering.joblib')
```

## Langkah Hari 4

1. Buka aplikasi Flask milik Orang 1; pastikan `templates/clustering.html` tampil dan CSS tidak merusak halaman supervised.
2. Uji `predict_clustering()` menggunakan satu baris CSV. Pastikan prediksi masuk salah satu indeks cluster `0..K-1`, koordinat PCA numerik, dan teks profil berasal dari data nyata.
3. Tampilkan **minimal satu grafik** di aplikasi, diutamakan PCA cluster; beri judul, sumber output, dan penjelasan singkat. Grafik dapat berupa PNG, tidak harus Chart.js.
4. Bandingkan hasil cluster dengan distribusi `fault_type` (tanpa label leakage), lalu tulis keterbatasan jika pemisahan lemah atau profil sulit dibedakan.
5. Berikan `reports/clustering_metrics.json` dan screenshot tabel hasil pengujian yang sebenarnya kepada Orang 1 untuk laporan progres dan slide.
6. Ikut uji end-to-end dengan Orang 1 dan Orang 3: submit masukan yang sama melalui kedua modul; cek error handling dan respons antarmuka pada ukuran layar berbeda.
7. Buat bahan slide metode clustering, PCA, dan keterbatasan; koordinasi dengan Orang 1 untuk presentasi 8–12 slide dan demo live. Minimal dua anggota siap memaparkan.
8. Push minimal 3 commit Hari 4, contoh `[TEST] Uji prediksi cluster`, `[STYLE] Rapikan responsif web`, `[DOCS] Jelaskan profil tiap cluster`.

## Kontrak output

**Format `models/clustering.joblib`:** dictionary dengan `imputer`, `scaler`, `kmeans`, `pca`, `feature_order`, `cluster_profiles`, `metrics`. Fungsi prediksi menggunakan urutan `feature_order` yang sama, menjalankan **transform tanpa fit baru**, lalu `kmeans.predict()` dan `pca.transform()` pada data hasil scaling.

**Fungsi yang dipakai Flask**:

```python
import json
import pandas as pd
from src.clustering import predict_clustering

schema = json.load(open('config/feature_schema.json', encoding='utf-8'))
df = pd.read_csv('data/processed/steel_faults_clean.csv')
contoh = df[schema['feature_order']].iloc[0].to_dict()
hasil = predict_clustering(contoh)
print(hasil)
# Bentuk output (angka/nama hanyalah ilustrasi):
# {'cluster_id': 2,
#   'profile': 'Area cacat relatif lebih tinggi dibanding rata-rata',
#   'pca': {'x': 0.0, 'y': 0.0}}
```

**Kriteria selesai:**

- [ ] Elbow, Silhouette, dan pilihan K dibuktikan dari eksekusi.
- [ ] Target `fault_type` tidak digunakan sebagai fitur clustering.
- [ ] Model + imputer + scaler + PCA tersedia di `.joblib` dan dapat digunakan untuk data baru.
- [ ] Profil cluster berdasarkan data aktual, bukan dugaan berdasarkan nomor.
- [ ] Halaman clustering dan CSS responsif, terintegrasi dengan Flask.
- [ ] Grafik PCA benar-benar hasil training dan minimal 6 commit total tersedia.

### Aturan kerja bersama

- Seluruh file berada dalam **satu repository** `smart-ml-kelompok3-steel` dan satu folder proyek. Jangan membuat empat proyek terpisah.
- **Setiap anggota** harus melakukan sedikitnya **3 commit pada Hari 3 dan 3 commit pada Hari 4**, lalu push ke branch masing-masing. Commit harus menggambarkan pekerjaan nyata.
- Setiap perubahan masuk ke `dev` melalui *pull request* yang diperiksa anggota lain; `main` hanya untuk versi siap demonstrasi.
- Gunakan format pesan commit seperti `[FEAT] Tambah modul ...`, `[TEST] Uji ...`, atau `[DOCS] Catat ...`.
- Simpan **bukti screenshot** aplikasi/notebook yang dijalankan, tautan commit, hash commit, dan catatan hambatan dalam laporan progres.
- **Jangan mengarang akurasi, Silhouette Score, diagram, ataupun persentase progres.** Gunakan keluaran program asli.
- Jangan pernah memasukkan tujuh kolom label cacat (`Pastry` s.d. `Other_Faults`) ke fitur model atau perhitungan clustering. Tidak ada kelas `Normal` dalam target dataset ini; semua label menunjukkan **jenis cacat**.
- Tulis progres dan materi presentasi dalam bahasa Indonesia baku sesuai PUEBI.

