# ORANG 1 — Koordinator Proyek dan Backend Integrator

**Navigasi tugas:** [Orang 1 — Koordinator & Web](orang1.md) · [Orang 2 — Data & QA](orang2.md) · [Orang 3 — Supervised](orang3.md) · [Orang 4 — Clustering & UI](orang4.md)

> **Proyek:** Kelompok 3, Mata Kuliah Machine Learning, TRIN — **Steel Plates Faults**.  
> **Dataset:** https://www.kaggle.com/datasets/uciml/faulty-steel-plates  
> **Web:** Flask di `http://127.0.0.1:5000` dengan dua paradigma: *supervised learning* + *unsupervised learning*.  
> **Status izin:** Dalam panduan resmi, Steel Plates Faults dialokasikan ke Kelompok 2, bukan Kelompok 3. **Konfirmasi persetujuan dan ketiadaan duplikasi kepada koordinator/dosen sebelum proyek diserahkan.**


## 1. Identitas dan peran

- **Nama / NIM:** `[Isi orang 1]`
- **Peran utama:** Koordinator / Project Manager **merangkap Backend Developer** (penyesuaian tim beranggotakan empat orang).
- **Tanggung jawab:** pendaftaran dataset, GitHub, kerangka Flask, integrasi dua modul ML, pengujian sistem, laporan progres Hari 3 & 4, serta pemeriksaan kesiapan presentasi.
- **Bukan tugasmu:** menulis ulang fungsi prapemrosesan dan model milik anggota lain. Integrasikan melalui kontrak fungsi yang disepakati.

## 2. Peta proyek dan pembagian kepemilikan file

```text
smart-ml-kelompok3-steel/
├── orang1.md                 # Panduanmu (salin bersama tiga berkas lainnya)
├── orang2.md
├── orang3.md
├── orang4.md
├── app.py                    # Orang 1
├── README.md                 # Orang 1; data/model diisi berdasarkan hasil anggota lain
├── requirements.txt          # Orang 1, disepakati bersama
├── .gitignore                # Orang 1
├── data/
│   ├── raw/faults.csv        # Orang 2: nama final CSV disepakati
│   └── processed/steel_faults_clean.csv
├── config/feature_schema.json   # Orang 2
├── src/
│   ├── __init__.py           # Orang 1
│   ├── preprocessing.py      # Orang 2
│   ├── supervised.py         # Orang 3
│   └── clustering.py         # Orang 4
├── training/
│   ├── train_supervised.py   # Orang 3
│   └── train_clustering.py   # Orang 4
├── models/
│   ├── supervised.joblib     # Orang 3
│   └── clustering.joblib     # Orang 4
├── templates/
│   ├── base.html             # Orang 1
│   ├── index.html            # Orang 1
│   ├── about.html            # Orang 1
│   ├── supervised.html       # Orang 3
│   └── clustering.html       # Orang 4
├── static/
│   ├── css/style.css        # Orang 4
│   └── img/cluster_pca.png  # Orang 4
├── tests/test_web.py        # Orang 1 bersama Orang 2
├── reports/                 # Orang 1 mengompilasi laporan dari semua anggota
└── slides/presentasi_k3.pptx # Dikerjakan bersama; Orang 1 finalisasi
```

**Sumber penghubung:** [Skema fitur / handoff Orang 2](orang2.md#kontrak-output) · [API prediksi Orang 3](orang3.md#kontrak-output) · [API cluster Orang 4](orang4.md#kontrak-output).

## 3. Kontrak antarmodul — disepakati sebelum mengoding

- Orang 2 menyediakan `config/feature_schema.json`: kunci **`feature_order`** (urutan 27 fitur), **`target_column`** = `fault_type`, **`fault_classes`**, dan **`defaults`** (angka representatif untuk form).
- Form Flask mengubah masukan setiap field ke `float` dan menyusun `dict` berdasarkan `feature_order`. Jika kosong atau bukan angka, kembalikan pesan kesalahan, **bukan traceback**.
- Orang 3 menyediakan fungsi `predict_supervised(inputs: dict) -> dict` dalam `src/supervised.py`, dengan keluaran `{ "fault_class": "...", "confidence": 0.0, "probabilities": { ... } }`.
- Orang 4 menyediakan fungsi `predict_clustering(inputs: dict) -> dict` dalam `src/clustering.py`, dengan keluaran `{ "cluster_id": 0, "profile": "...", "pca": {"x": 0.0, "y": 0.0} }`.
- **Semua modul** menerima nama kolom dan unit yang sama. Seluruh pemuatan pipeline/model berasal dari file `.joblib` hasil training yang **benar-benar** dibuat anggota terkait.
- `app.py` adalah satu-satunya aplikasi Flask; anggota lain **tidak** membuat Flask kedua.

## 4. Hari 3 — tahap demi tahap

1. **Konfirmasi dataset (sebelum tenggat pendaftaran 12.00).** Hubungi koordinator kelas dan dosen untuk meminta persetujuan Kelompok 3 menggunakan Steel Plates Faults. Simpan bukti konfirmasi. Jika ditolak, laporkan kepada semua anggota sebelum training dilanjutkan.
2. **Buat repo GitHub** bernama `smart-ml-kelompok3-steel`, set akses dosen sesuai panduan, undang ketiga anggota sebagai kolaborator, buat branch `main` dan `dev`, serta lindungi `main` dengan review bila memungkinkan.
3. **Buat struktur folder** persis seperti bagian 2. Tambahkan placeholder kosong agar folder dikenali Git; buat `.gitignore` untuk `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, secret, dan file sementara. Tentukan apakah CSV dan `.joblib` ukurannya aman serta diizinkan untuk masuk repo.
4. **Persiapkan lingkungan** di VS Code/Anaconda Prompt (Windows):

   ```bat
   py -m venv .venv
   .venv\Scriptsctivate
   python -m pip install flask pandas numpy scikit-learn matplotlib seaborn joblib pytest
   python -m pip freeze > requirements.txt
   ```

5. **Buat Flask dasar** pada `app.py` dengan halaman GET `/`, `/supervised`, `/clustering`, `/about`. Pastikan `python app.py` membuka `http://127.0.0.1:5000` dan halaman tidak 404.
6. **Buat kerangka HTML** `base.html`, `index.html`, `about.html`; pemilik `supervised.html` adalah Orang 3 dan `clustering.html` adalah Orang 4. Jangan menimpa file mereka.
7. **Tambahkan route POST** `/supervised`, `/clustering`, dan opsional `/analisis`. Form membaca daftar field dari `feature_schema.json`; selama model belum siap, tampilkan pesan `Model belum tersedia` tanpa data prediksi palsu.
8. **Daily standup minimal 15 menit**: catat siapa sudah menuntaskan apa, kesulitan, jadwal, dan kebutuhan integrasi.
9. **Kumpulkan bukti dari semua anggota** dan susun `reports/LaporanProgres_Hari3_Kelompok3_TRIN3AEC3.docx` (2–4 halaman, A4, TNR 12, spasi 1,5). Isi ringkasan progres, tugas anggota, commit+hash, screenshot, hambatan, rencana, notulensi, dan tautan dataset. Pastikan terkumpul **pukul 23.59 Hari 3**.
10. **Target 3 commit nyata Hari 3**, contoh `[CHORE] Siapkan struktur repo`, `[FEAT] Tambah halaman Flask`, `[DOCS] Catat progres hari ketiga`.

## 5. Hari 4 — tahap demi tahap

1. Pull `dev` terbaru dan periksa apakah `config/feature_schema.json`, `models/supervised.joblib`, `models/clustering.joblib`, dan API kedua engineer sudah tersedia.
2. **Integrasikan model** tanpa retraining pada setiap request; muat artefak sekali atau cache secara aman. Jangan memanggil `.fit()` pada input pengguna. Pastikan `feature_order` yang dipakai form persis sama dengan artefak model.
3. **Jalankan uji input**: semua angka valid, satu field kosong, huruf di kolom angka, nilai tak hingga, dan model belum tersedia. Input invalid harus mendapatkan pesan ramah.
4. **Uji dua modul dalam satu browser**: membuka `/supervised` memperoleh **jenis cacat dan probabilitas**; membuka `/clustering` memperoleh **ID cluster dan penjelasan pola**. Pastikan keduanya menggunakan fitur yang sama dan tidak melabeli cluster sebagai jenis cacat yang pasti.
5. Tambahkan pengujian ringan `tests/test_web.py`: status 200 untuk empat halaman, validasi input, respons aplikasi ketika artefak belum ditemukan, serta contoh prediksi jika artefak tersedia. Mintalah Orang 2 membantu QA data.
6. **Rapikan README.md**: latar belakang, link dataset, cara instal, menjalankan training, menjalankan Flask, struktur folder, hasil metrik aktual, screenshot aplikasi, kontribusi, dan keterbatasan model.
7. **Siapkan presentasi 8–12 slide**, durasi 10 menit pemaparan + 5 menit tanya jawab; minimal dua anggota berbicara. Ambil grafik dari output model asli.
8. **Kirim laporan Hari 4, tautan repo, dan slide paling lambat 13.00 WIB** sebagai batas aman untuk keluaran wajib; dalam jadwal panduan juga ada rekap final malam hari, tetapi jangan menunda keluaran wajib sampai malam.
9. Presentasi/demo: perlihatkan satu masukan yang menghasilkan output kedua paradigma. Setelah presentasi, tangani masukan dosen bila masih ada waktu.
10. **Target 3 commit nyata Hari 4**, contoh `[FEAT] Integrasi dua prediksi`, `[TEST] Tambah validasi form`, `[DOCS] Finalisasi panduan demo`.

## 6. Kriteria selesai (Definition of Done)

- [ ] Izin dataset Kelompok 3 dikonfirmasi; buktinya tersedia.
- [ ] Repo memiliki `main`, `dev`, dan empat branch kerja; semua anggota aktif.
- [ ] Web Flask dapat dibuka, **empat halaman minimal** tampil.
- [ ] Input angka menghasilkan prediksi supervised **dan** cluster dari artefak model sesungguhnya.
- [ ] Input salah ditolak secara wajar; app tidak crash.
- [ ] Satu grafik nyata tampil di halaman web.
- [ ] Kedua dokumen progres `.docx`, repo, slide, dan bukti commit terkumpul.
- [ ] Minimal 6 commit **per anggota** selama dua hari dan PR telah di-review.

## 7. Format serah-terima yang kamu terima

```text
[HANDOFF]
Dari: Orang [2/3/4]
File: [path]
Cara menjalankan: [perintah]
Bukti: [screenshot/output/log]
Hasil: [angka aktual, bila model]
Masalah terbuka: [jika ada]
Sudah dites oleh: [nama]
```

### Aturan kerja bersama

- Seluruh file berada dalam **satu repository** `smart-ml-kelompok3-steel` dan satu folder proyek. Jangan membuat empat proyek terpisah.
- **Setiap anggota** harus melakukan sedikitnya **3 commit pada Hari 3 dan 3 commit pada Hari 4**, lalu push ke branch masing-masing. Commit harus menggambarkan pekerjaan nyata.
- Setiap perubahan masuk ke `dev` melalui *pull request* yang diperiksa anggota lain; `main` hanya untuk versi siap demonstrasi.
- Gunakan format pesan commit seperti `[FEAT] Tambah modul ...`, `[TEST] Uji ...`, atau `[DOCS] Catat ...`.
- Simpan **bukti screenshot** aplikasi/notebook yang dijalankan, tautan commit, hash commit, dan catatan hambatan dalam laporan progres.
- **Jangan mengarang akurasi, Silhouette Score, diagram, ataupun persentase progres.** Gunakan keluaran program asli.
- Jangan pernah memasukkan tujuh kolom label cacat (`Pastry` s.d. `Other_Faults`) ke fitur model atau perhitungan clustering. Tidak ada kelas `Normal` dalam target dataset ini; semua label menunjukkan **jenis cacat**.
- Tulis progres dan materi presentasi dalam bahasa Indonesia baku sesuai PUEBI.

