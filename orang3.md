# ORANG 3 — ML Engineer Supervised Learning

**Navigasi tugas:** [Orang 1 — Koordinator & Web](orang1.md) · [Orang 2 — Data & QA](orang2.md) · [Orang 3 — Supervised](orang3.md) · [Orang 4 — Clustering & UI](orang4.md)

> **Proyek:** Kelompok 3, Mata Kuliah Machine Learning, TRIN — **Steel Plates Faults**.  
> **Dataset:** https://www.kaggle.com/datasets/uciml/faulty-steel-plates  
> **Web:** Flask di `http://127.0.0.1:5000` dengan dua paradigma: *supervised learning* + *unsupervised learning*.  
> **Status izin:** Dalam panduan resmi, Steel Plates Faults dialokasikan ke Kelompok 2, bukan Kelompok 3. **Konfirmasi persetujuan dan ketiadaan duplikasi kepada koordinator/dosen sebelum proyek diserahkan.**


## 1. Identitas dan ruang lingkup

- **Nama / NIM:** `[Isi orang 3]`
- **Tanggung jawab:** klasifikasi **tujuh jenis cacat permukaan** pelat baja menggunakan Random Forest (baseline dapat dibandingkan dengan Logistic Regression); evaluasi dan ekspor model; fungsi prediksi; halaman HTML supervised.
- **Input tunggal:** [CSV dan schema milik Orang 2](orang2.md#kontrak-output).
- **Penerima modulmu:** [Orang 1 — backend dan integrasi](orang1.md#3-kontrak-antarmodul--disepakati-sebelum-mengoding).
- **Tidak dilakukan:** klasifikasi `Normal`/`Defect` karena dataset tidak menyediakan kelas normal. Target adalah **jenis cacat**.

## 2. File yang kamu miliki

```text
training/train_supervised.py
training/supervised_experiment.ipynb
src/supervised.py
templates/supervised.html
models/supervised.joblib
reports/supervised_metrics.json  # angka dari eksekusi sesungguhnya
static/img/confusion_matrix.png  # gambar asli dari run training
```

## Langkah Hari 3

1. Tunggu [konfirmasi data siap](orang2.md#kontrak-output). Baca `data/processed/steel_faults_clean.csv` dan `config/feature_schema.json`. Pastikan target `fault_type` ada serta fitur sama dengan `feature_order`.
2. Ambil `X = df[feature_order]` dan `y = df['fault_type']`. Cetak kelas dan `y.value_counts()`. Jangan memasukkan one-hot kelas asli dalam `X`.
3. Bagi dataset menjadi data latih 80% dan uji 20% menggunakan `train_test_split(..., stratify=y, random_state=42)`; laporkan hasil kelas dan jumlahnya.
4. Buat **Pipeline** Scikit-learn yang mencakup `SimpleImputer(strategy='median')` dan `RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200)`; ini pilihan rancangan untuk tugas, **bukan** ketentuan angka dari dosen. Semua `.fit()` hanya pada data latih.
5. Jalankan baseline dan uji alternatif parameter terbatas bila waktu cukup. Catat **accuracy, macro precision, macro recall, macro F1, weighted F1**, serta `classification_report` dan confusion matrix. Jelaskan kelas dengan recall terendah dan ketidakseimbangan jumlah kelas.
6. Buat grafik confusion matrix dan, jika memungkinkan, feature importance dari model. Simpan gambar yang berasal dari perhitungan model asli.
7. Simpan **seluruh fitted pipeline + `feature_order` + urutan kelas** ke `models/supervised.joblib` menggunakan `joblib.dump`, bukan hanya estimator RF sehingga langkah imputasi tidak hilang saat dipakai Flask.
8. Buat fungsi `predict_supervised(inputs: dict) -> dict` dalam `src/supervised.py`. Pastikan urutan DataFrame 1 baris sesuai `feature_order` dan prediksi menggunakan `pipeline.predict()` serta `pipeline.predict_proba()`.
9. Buat `templates/supervised.html` yang *extends* `base.html` milik Orang 1. Form input berasal dari fitur/schema yang dikirim `app.py` (jangan ketik 27 field dua kali secara manual bila bisa di-*loop*).
10. Uji dengan satu baris contoh dari CSV: prediksi harus menghasilkan salah satu tujuh kelas dan probabilitas berjumlah mendekati 1. Jangan klaim akurasi >85% tanpa hasil uji.
11. Push branch `feature/supervised` dengan sedikitnya 3 commit Hari 3, misalnya `[FEAT] Training Random Forest`, `[FEAT] Simpan pipeline prediksi`, `[FEAT] Tambah halaman supervised`; laporkan hash dan metrik ke Orang 1.

### Contoh bentuk inti pelatihan (boleh disesuaikan dengan hasil aktual)

```python
import json, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

schema = json.load(open('config/feature_schema.json', encoding='utf-8'))
df = pd.read_csv('data/processed/steel_faults_clean.csv')
features = schema['feature_order']
X, y = df[features], df[schema['target_column']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    )),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Macro F1:', f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))
joblib.dump({'pipeline': pipeline, 'feature_order': features}, 'models/supervised.joblib')
```

> Contoh kode menunjukkan kontrak artefak. Pastikan folder `models/` ada dan versi library di lingkungan web kompatibel dengan lingkungan training.

## Langkah Hari 4

1. Pull `dev`, pastikan artefak dapat dimuat melalui `joblib.load()` dari root proyek.
2. Pastikan fungsi prediksi **tidak pernah `.fit()`** pada saat request web.
3. Uji masukan valid, nilai kosong, urutan fitur yang berbeda di dict, kelas yang tidak dikenal, dan kesalahan artefak. Penanganan input kosong merupakan tanggung jawab Flask dan tetap perlu diverifikasi di fungsi prediksi.
4. Tulis laporan metrik aktual ke `reports/supervised_metrics.json`. Sertakan informasi train-test split serta kelas yang paling sering salah prediksi.
5. Pastikan HTML supervised menampilkan prediksi jenis cacat, confidence, dan peluang ketujuh kelas. Istilah confidence adalah skor probabilitas model, **bukan jaminan kebenaran**.
6. Demo route `/supervised` bersama Orang 1, perbaiki kesalahan integrasi, lalu ikut review *pull request* Orang 4.
7. Siapkan 1–2 slide model, metrik, confusion matrix, dan keterbatasan (contohnya bias terhadap kelas mayoritas). Latih penjelasan sekitar 1–2 menit.
8. Push minimal 3 commit Hari 4 dengan bukti, misalnya `[TEST] Uji prediksi tujuh kelas`, `[FIX] Sinkronkan urutan fitur`, `[DOCS] Ringkas hasil klasifikasi`.

## Kontrak output

**Struktur `models/supervised.joblib`:** dictionary dengan kunci `pipeline` (objek Scikit-learn terlatih) dan `feature_order` (list string); `pipeline.classes_` pada estimator dapat diambil dari `pipeline.named_steps['model'].classes_`.

**Antarmuka yang dipanggil Flask:**

```python
import json
import pandas as pd
from src.supervised import predict_supervised

schema = json.load(open('config/feature_schema.json', encoding='utf-8'))
df = pd.read_csv('data/processed/steel_faults_clean.csv')
contoh = df[schema['feature_order']].iloc[0].to_dict()
hasil = predict_supervised(contoh)
print(hasil)
# Output yang diharapkan (struktur, bukan angka contoh hasil):
# {'fault_class': '...', 'confidence': 0.0,
#   'probabilities': {'Pastry': 0.0, 'Z_Scratch': 0.0, ...}}
```

Fungsi memuat pipeline dari artefak, menyusun DataFrame 1 baris sesuai `feature_order`, lalu mengembalikan jenis cacat (`fault_class`), probabilitas kelas teratas (`confidence`), dan kamus `probabilities` untuk **tujuh kelas**.

**Checklist serah-terima:**

- [ ] Pipeline hasil training benar-benar tersimpan dan bisa dimuat kembali.
- [ ] Model hanya melihat fitur, tidak melihat tujuh label input saat prediksi.
- [ ] Terdapat metrik uji, confusion matrix, dan screenshot asli.
- [ ] `predict_supervised()` diuji dengan input dictionary yang memakai schema.
- [ ] Halaman supervised menampilkan hasil dan kesalahan input dengan jelas.
- [ ] Minimal enam commit pribadi dalam dua hari; bukti diberikan ke Orang 1.

### Aturan kerja bersama

- Seluruh file berada dalam **satu repository** `smart-ml-kelompok3-steel` dan satu folder proyek. Jangan membuat empat proyek terpisah.
- **Setiap anggota** harus melakukan sedikitnya **3 commit pada Hari 3 dan 3 commit pada Hari 4**, lalu push ke branch masing-masing. Commit harus menggambarkan pekerjaan nyata.
- Setiap perubahan masuk ke `dev` melalui *pull request* yang diperiksa anggota lain; `main` hanya untuk versi siap demonstrasi.
- Gunakan format pesan commit seperti `[FEAT] Tambah modul ...`, `[TEST] Uji ...`, atau `[DOCS] Catat ...`.
- Simpan **bukti screenshot** aplikasi/notebook yang dijalankan, tautan commit, hash commit, dan catatan hambatan dalam laporan progres.
- **Jangan mengarang akurasi, Silhouette Score, diagram, ataupun persentase progres.** Gunakan keluaran program asli.
- Jangan pernah memasukkan tujuh kolom label cacat (`Pastry` s.d. `Other_Faults`) ke fitur model atau perhitungan clustering. Tidak ada kelas `Normal` dalam target dataset ini; semua label menunjukkan **jenis cacat**.
- Tulis progres dan materi presentasi dalam bahasa Indonesia baku sesuai PUEBI.

