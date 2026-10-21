# ORANG 2 — Data Engineer dan Quality Assurance

**Navigasi tugas:** [Orang 1 — Koordinator & Web](orang1.md) · [Orang 2 — Data & QA](orang2.md) · [Orang 3 — Supervised](orang3.md) · [Orang 4 — Clustering & UI](orang4.md)

> **Proyek:** Kelompok 3, Mata Kuliah Machine Learning, TRIN — **Steel Plates Faults**.  
> **Dataset:** https://www.kaggle.com/datasets/uciml/faulty-steel-plates  
> **Web:** Flask di `http://127.0.0.1:5000` dengan dua paradigma: *supervised learning* + *unsupervised learning*.  
> **Status izin:** Dalam panduan resmi, Steel Plates Faults dialokasikan ke Kelompok 2, bukan Kelompok 3. **Konfirmasi persetujuan dan ketiadaan duplikasi kepada koordinator/dosen sebelum proyek diserahkan.**


## 1. Identitas dan peran

- **Nama / NIM:** `[Isi orang 2]`
- **Peran:** Data Engineer **merangkap QA data**, agar pembagian empat orang tetap mencakup pengujian.
- **Tugas utama:** mengunduh CSV, memeriksa struktur, EDA, menyiapkan label tujuh kelas, memberi kontrak fitur, mendokumentasikan data, membantu pengujian integrasi.
- **Menerima dari:** [Koordinator](orang1.md#kontrak-antarmodul--disepakati-sebelum-mengoding), konfirmasi dataset dan susunan folder.
- **Menyerahkan kepada:** [Supervised](orang3.md#langkah-hari-3), [Clustering](orang4.md#langkah-hari-3), dan [Flask](orang1.md#5-hari-4--tahap-demi-tahap).

## 2. Pemahaman dataset (perlu diverifikasi pada CSV nyata)

Dataset **Steel Plates Faults** ditujukan untuk klasifikasi jenis cacat permukaan pelat baja, bukan penentuan cacat vs tidak cacat. Sumber rujukan menyebut **1.941 observasi, 27 fitur prediktor, dan 7 jenis cacat**. Pada versi CSV yang menyediakan *one-hot labels*, file bisa berisi **34 kolom total** (27 fitur + 7 kolom label). Jangan menyamakan angka 27 fitur dengan jumlah semua kolom. Data sebenarnya belum dilampirkan dalam percakapan ini; semua angka harus dicek ulang sesudah CSV diunduh.

**Daftar tujuh label yang diharapkan (ejaan penting):**

```python
FAULT_COLS = [
    'Pastry', 'Z_Scratch', 'K_Scatch',
    'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'
]
```

**Penting:** Pada beberapa halaman tutorial terdapat ejaan `K_Scratch`, padahal daftar kolom sumber memakai **`K_Scatch`**. Pakai nama yang *benar-benar ada* pada CSV; jangan mengganti diam-diam atau membuang kolom yang tidak dikenali.

## 3. Struktur output milikmu

```text
data/raw/faults.csv
data/processed/steel_faults_clean.csv
config/feature_schema.json
src/preprocessing.py
training/eda_steel_faults.ipynb
tests/test_data.py
data/README_data.md
```

CSV `steel_faults_clean.csv` harus berisi **seluruh fitur asli numerik dalam satuan awal** + satu kolom target teks `fault_type`. Jangan *fit* StandardScaler untuk dataset lengkap pada tahap ini; scaler untuk supervised dilatih **setelah train-test split** di pipeline milik Orang 3. Model clustering membuat scaler-nya sendiri berdasarkan fitur yang sama.

## 4. Langkah Hari 3

### Tahap 1 — Unduh, simpan, dan identifikasi

1. Unduh dataset melalui tautan di atas setelah izin disetujui, lalu ekstrak ke `data/raw/` dan beri nama final `faults.csv` (catat nama file sumber).
2. Buka `training/eda_steel_faults.ipynb`, baca menggunakan `pd.read_csv('data/raw/faults.csv')`.
3. Jalankan pemeriksaan berikut lalu **simpan output sebenarnya**:

   ```python
   import pandas as pd
   df = pd.read_csv('data/raw/faults.csv')
   print(df.shape)
   print(df.dtypes)
   print(df.isna().sum())
   print('Duplikat:', df.duplicated().sum())
   display(df.head())
   ```

4. Pastikan tujuh kolom label tersedia; jika dataset hanya menyediakan kolom target tunggal, **catat skema yang berbeda dan koordinasikan perubahan kontrak terlebih dahulu**.

### Tahap 2 — Periksa integritas tujuh kelas

5. Pastikan tujuh kolom target bernilai 0/1, kemudian periksa total nilai 1 per baris:

   ```python
   fault_columns = ['Pastry','Z_Scratch','K_Scatch',
                    'Stains','Dirtiness','Bumps','Other_Faults']
   assert all(c in df.columns for c in fault_columns)
   assert df[fault_columns].notna().all().all()
   assert df[fault_columns].isin([0, 1]).all().all()
   label_totals = df[fault_columns].sum(axis=1)
   print(label_totals.value_counts(dropna=False))
   assert label_totals.eq(1).all(), 'Ada baris tanpa / lebih dari 1 label'
   ```

6. Bila valid, konversi label satu-hot ke label teks dengan `df['fault_type'] = df[fault_columns].idxmax(axis=1)` dan hapus tujuh kolom one-hot dari masukan model. **Jangan menggunakan `idxmax()` sebelum pemeriksaan** karena baris semua nol akan keliru mendapat label pertama.
7. Tetapkan `feature_order = [c for c in df.columns if c not in fault_columns + ['fault_type']]`; verifikasi setiap fitur numerik dan cocokkan dengan 27 fitur prediktor pada sumber.

### Tahap 3 — EDA dan dokumentasi

8. Buat 3 visualisasi: histogram jumlah setiap jenis cacat, heatmap korelasi prediktor, dan histogram/boxplot beberapa fitur seperti `Pixels_Areas`, `Steel_Plate_Thickness`, `Luminosity_Index`. Jelaskan masing-masing grafik, bukan cuma tempel gambar.
9. Tentukan penanganan missing: **catat berapa banyak**; jangan mengganti nilai secara global bila pipeline pemodelan akan menggunakan median hasil data training. Simpan data bersih dengan nilai kosong numerik tetap mungkin ada dan dokumentasikan agar model melakukan imputasi sendiri.
10. Periksa duplikat: laporkan jumlah dan dampaknya. Bila memutuskan menghapus, catat jumlah baris sebelum-sesudah, alasan, dan lakukan perubahan secara konsisten untuk kedua model.
11. Buat `config/feature_schema.json` (nilai contoh di bawah hanya menunjukkan **struktur**, angka/default harus diisi dari CSV asli):

   ```json
   {
     "feature_order": ["X_Minimum", "X_Maximum", "Pixels_Areas"],
     "target_column": "fault_type",
     "fault_classes": ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"],
     "defaults": {"X_Minimum": 0, "X_Maximum": 0, "Pixels_Areas": 0}
   }
   ```

   **Jangan pakai tiga fitur contoh di atas untuk hasil akhir.** Isi dengan semua kolom numerik aktual sesuai urutan CSV.
12. Buat `src/preprocessing.py` agar proses pembuatan `steel_faults_clean.csv` dapat dijalankan ulang melalui `python src/preprocessing.py` dari root. Jika file mentah belum tersedia, tampilkan kesalahan yang jelas.
13. Lakukan serah-terima versi awal **secepat mungkin**, tanpa menunggu EDA sempurna, agar Orang 3 dan Orang 4 dapat mulai training.

### Tahap 4 — Git dan bukti

14. Gunakan branch `feature/data-preprocessing`, commit bertahap minimal 3 kali dan push. Contoh: `[DATA] Audit struktur Steel Plates`, `[FEAT] Bentuk label fault_type`, `[DOCS] Tulis dokumentasi EDA` (format `[DATA]` boleh diganti `[FEAT]` sesuai kesepakatan panduan).
15. Berikan koordinator screenshot shape, missing/duplikat, distribusi kelas, commit hash, hambatan, dan durasi pekerjaan untuk laporan progres Hari 3.

## 5. Langkah Hari 4

1. Pastikan pembaruan `data/processed/steel_faults_clean.csv` dan `config/feature_schema.json` telah masuk `dev`. Jangan mengubah susunan fitur tanpa memberitahu kedua ML Engineer.
2. Tulis `tests/test_data.py` untuk mengecek: jumlah fitur sesuai schema, semua prediktor numerik, target `fault_type` ada, tujuh kelas dikenali, tidak ada one-hot label di `feature_order`, dan kolom target bukan bagian prediktor.
3. Bantu Orang 1 menguji form web terhadap urutan fitur, input kosong, nilai NaN/tak hingga, dan integritas serialisasi data input.
4. Buat `data/README_data.md` berisi nama CSV, sumber, lisensi jika diketahui, file asli vs hasil transformasi, ukuran dan distribusi aktual, masalah data, dan proses reproduksi.
5. Sinkronkan laporan hasil EDA dan pengujian; ambil screenshot output tabel langsung dari notebook bila dipakai untuk laporan (jangan mengarang tabel angka).
6. Cek klasifikasi tidak melatih pada label leakage; cek clustering tidak memasukkan target. Bantu review *pull request* kedua ML Engineer.
7. Lakukan minimal 3 commit Hari 4: `[TEST] Validasi fitur dan label`, `[DOCS] Lengkapi sumber data`, `[FIX] Rapikan pengecekan input`. Push dan kirim hash kepada Orang 1 sebelum pengumpulan pukul 13.00.

## Kontrak output

**File yang wajib tersedia:**

- `data/raw/faults.csv` — dari Kaggle, dengan catatan sumber serta hak distribusi.
- `data/processed/steel_faults_clean.csv` — 27 prediktor aktual dalam satuan asli + `fault_type` berisi **salah satu dari tujuh** kelas.
- `config/feature_schema.json` — `feature_order`, `target_column`, `fault_classes`, dan `defaults`, mencerminkan data asli.
- `src/preprocessing.py` — dapat mereproduksi data bersih tanpa intervensi manual.
- `training/eda_steel_faults.ipynb`, `data/README_data.md`, `tests/test_data.py`.

**Pesan serah-terima untuk engineer**:

```text
[DATA SIAP]
File data   : data/processed/steel_faults_clean.csv
Schema      : config/feature_schema.json
Fitur       : [jumlah sesuai CSV]
Target      : fault_type, [jumlah kelas aktual]
Missing     : [jumlah yang terukur]
Cara buat   : python src/preprocessing.py
Catatan     : [keputusan duplikat, kolom alternatif, dll.]
Commit      : [hash / PR]
```

## 6. Kriteria selesai

- [ ] Shape, missing, dan distribusi label telah diperiksa dan dibuktikan.
- [ ] Tujuh one-hot target tidak lagi menjadi prediktor.
- [ ] CSV bersih dan schema **mempunyai kolom dan urutan yang sama**.
- [ ] Tidak ada imputasi/scaling dari data testing sebelum split supervised.
- [ ] Kedua ML Engineer berhasil membaca keluaranmu.
- [ ] Pengujian data, dokumentasi, dan minimal 6 commit total selesai.

### Aturan kerja bersama

- Seluruh file berada dalam **satu repository** `smart-ml-kelompok3-steel` dan satu folder proyek. Jangan membuat empat proyek terpisah.
- **Setiap anggota** harus melakukan sedikitnya **3 commit pada Hari 3 dan 3 commit pada Hari 4**, lalu push ke branch masing-masing. Commit harus menggambarkan pekerjaan nyata.
- Setiap perubahan masuk ke `dev` melalui *pull request* yang diperiksa anggota lain; `main` hanya untuk versi siap demonstrasi.
- Gunakan format pesan commit seperti `[FEAT] Tambah modul ...`, `[TEST] Uji ...`, atau `[DOCS] Catat ...`.
- Simpan **bukti screenshot** aplikasi/notebook yang dijalankan, tautan commit, hash commit, dan catatan hambatan dalam laporan progres.
- **Jangan mengarang akurasi, Silhouette Score, diagram, ataupun persentase progres.** Gunakan keluaran program asli.
- Jangan pernah memasukkan tujuh kolom label cacat (`Pastry` s.d. `Other_Faults`) ke fitur model atau perhitungan clustering. Tidak ada kelas `Normal` dalam target dataset ini; semua label menunjukkan **jenis cacat**.
- Tulis progres dan materi presentasi dalam bahasa Indonesia baku sesuai PUEBI.

