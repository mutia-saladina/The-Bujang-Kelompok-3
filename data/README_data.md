# Dokumentasi Dataset Steel Plates Faults

## 1. Informasi Dataset

- **Nama:** Steel Plates Faults
- **Sumber:** UCI Machine Learning Repository melalui Kaggle
- **Link:** https://www.kaggle.com/datasets/uciml/faulty-steel-plates
- **Format:** CSV
- **Jumlah data:** 1.941 observasi
- **Jumlah kolom asli:** 34
- **Jumlah fitur prediktor:** 27
- **Jumlah kelas:** 7
- **Lisensi:** Perlu dikonfirmasi pada sumber dataset

Dataset digunakan untuk mengidentifikasi jenis cacat pada permukaan pelat baja menggunakan Machine Learning.

## 2. Pemeriksaan Kualitas Data

Berdasarkan hasil Exploratory Data Analysis (EDA):

- Total missing value: 0
- Jumlah baris duplikat: 0
- Semua fitur dan label asli bertipe numerik.
- Seluruh label bernilai biner 0 atau 1.
- Setiap observasi memiliki tepat satu label aktif.

## 3. Distribusi Jenis Cacat

| Jenis Cacat  |    Jumlah |
| ------------ | --------: |
| Other_Faults |       673 |
| Bumps        |       402 |
| K_Scatch     |       391 |
| Z_Scratch    |       190 |
| Pastry       |       158 |
| Stains       |        72 |
| Dirtiness    |        55 |
| **Total**    | **1.941** |

Dataset memiliki ketidakseimbangan kelas karena jumlah observasi setiap jenis cacat berbeda.

## 4. Hasil Preprocessing

Tujuh kolom label biner diubah menjadi satu kolom target bernama `fault_type`.

Hasil preprocessing:

- 1.941 baris
- 27 fitur numerik
- 1 kolom target `fault_type`
- Total 28 kolom

Nilai fitur asli tetap dipertahankan. Tidak dilakukan scaling atau imputasi global.

## 5. Struktur File

- `data/raw/faults.csv` — dataset asli.
- `data/processed/steel_faults_clean.csv` — dataset hasil preprocessing.
- `config/feature_schema.json` — urutan fitur, kelas, dan nilai default.
- `src/preprocessing.py` — script preprocessing otomatis.
- `training/eda_steel_faults.ipynb` — notebook eksplorasi data.
- `tests/test_data.py` — pengujian kualitas data.

## 6. Cara Menjalankan

Jalankan perintah berikut dari root repository:

```bash
python src/preprocessing.py
```

Untuk pengujian:

```bash
python -m pytest tests/test_data.py -v
```

Hasil pengujian terakhir: **6 passed**.

## 7. Catatan Integrasi

Dataset bersih digunakan oleh dua modul:

1. **Supervised Learning:** 27 fitur sebagai input dan `fault_type` sebagai target klasifikasi.
2. **Unsupervised Learning:** hanya menggunakan 27 fitur tanpa `fault_type`.

Scaling dan imputasi untuk model supervised dilakukan pada pipeline berdasarkan data training agar tidak terjadi data leakage.

Nilai default pada `feature_schema.json` merupakan referensi pengisian form, bukan hasil preprocessing model.

Dataset ini hanya berisi tujuh jenis cacat dan tidak memiliki kelas Normal.
