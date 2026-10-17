# Bondora Peer-to-Peer (P2P) Lending Analytics

Proyek analisis komprehensif pada dataset pinjaman Peer-to-Peer (P2P) **Bondora** yang mencakup lima pilar analitik data (**Descriptive, Diagnostic, Predictive, Prescriptive, dan Semantic Analytics**). Hasil akhir proyek ini diwujudkan dalam bentuk rangkaian **Notebook Jupyter**, **Laporan Akhir (.docx)**, serta **Dashboard Interaktif Streamlit**.

---

## 1. Lima Jenis Analytics & Cakupan Tahap

1. **Descriptive Analytics** — Meringkas karakteristik historis portofolio pinjaman, profil demografis peminjam, distribusi suku bunga, dan tingkat gagal bayar (*default rate*).
2. **Diagnostic Analytics** — Menganalisis akar penyebab dan hubungan statistik/korelasi faktor-faktor risiko yang membedakan pinjaman lancar dengan pinjaman gagal bayar.
3. **Predictive Analytics**
   - **Klasifikasi Default:** Memprediksi probabilitas gagal bayar pinjaman menggunakan algoritma *machine learning* (`scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`) beserta interpretasi model (`shap`).
   - **Regresi:** Mengestimasi target finansial kontinu pada pinjaman P2P.
4. **Prescriptive Analytics** — Mengoptimalkan pemilihan dan alokasi portofolio pinjaman menggunakan pemrograman linier/optimasi matematis (`pulp`) untuk memaksimalkan imbal hasil pada batas risiko tertentu.
5. **Semantic Analytics** — Memodelkan pengetahuan domain P2P lending ke dalam skema ontologi dan *knowledge graph* (`rdflib`, `networkx`) yang dapat di-query menggunakan SPARQL.

---

## 2. Struktur Folder Proyek

```text
bondora-analytics/
├── data/
│   ├── raw/            # file asli dari Bondora (tidak di-commit)
│   ├── interim/        # hasil antara (tidak di-commit)
│   ├── processed/      # data bersih (tidak di-commit)
│   ├── external/       # data pendukung bila ada (tidak di-commit)
│   └── README.md       # panduan unduh & metadata dataset
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_descriptive.ipynb
│   ├── 03_diagnostic.ipynb
│   ├── 04_predictive_classification.ipynb
│   ├── 05_predictive_regression.ipynb
│   ├── 06_prescriptive.ipynb
│   └── 07_semantic.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/__init__.py
│   ├── features/__init__.py
│   ├── models/__init__.py
│   └── utils/__init__.py
├── models/             # model tersimpan (tidak di-commit)
├── dashboard/          # aplikasi Streamlit (tahap 7)
├── ontology/           # file ontologi & query SPARQL (tahap 6)
├── reports/
│   ├── figures/
│   ├── tables/
│   └── 00_problem_statement.md
├── scripts/
│   └── verify_env.py
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt
├── .gitignore
└── README.md
```

---

## 3. Cara Setup Environment

Pastikan Python telah terpasang di sistem Anda, lalu jalankan langkah-langkah berikut dari direktori root `bondora-analytics/`:

### a. Buat dan Aktifkan Virtual Environment (`.venv`)

**Windows (PowerShell / CMD):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### b. Instalasi Dependensi

```bash
pip install -r requirements-dev.txt
```

*(Opsional: untuk mereproduksi versi persis yang telah dikunci, gunakan `pip install -r requirements.lock.txt`).*

### c. Daftarkan Kernel Jupyter

```bash
python -m ipykernel install --user --name bondora-analytics --display-name "Python (bondora-analytics)"
```

### d. Verifikasi Environment & Struktur Folder

```bash
python scripts/verify_env.py
```

Pastikan output diakhiri dengan `STATUS: OK` dan `pip check` tidak melaporkan konflik.

---

## 4. Instruksi Penempatan Dataset

Dataset **tidak disertakan di dalam repositori Git**. Silakan unduh dataset Bondora secara manual dan simpan ke dalam folder:

```text
data/raw/
```

Lihat panduan lengkap, daftar file yang dibutuhkan (`LoanData.csv` dan opsional `RepaymentsData.csv`), serta tabel pencatatan metadata unduhan pada dokumen [`data/README.md`](data/README.md).

---

## 5. Status Tahap Pengerjaan Proyek

| Tahap | Nama Tahap | Fokus Utama & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Tahap 0** | Persiapan Proyek & Environment | Struktur repo, `.venv`, instalasi paket, konfigurasi, template notebook & dokumen | **Selesai** |
| **Tahap 1** | Data Understanding & Preprocessing | Eksplorasi skema, pembersihan data, penanganan *missing values* & *data leakage* (`01_data_understanding.ipynb`) | **Selesai** |
| **Tahap 2** | Descriptive Analytics | Statistik deskriptif & visualisasi distribusi portofolio (`02_descriptive.ipynb`) | Belum |
| **Tahap 3** | Diagnostic Analytics | Uji hipotesis, analisis korelasi, dan faktor penyebab *default* (`03_diagnostic.ipynb`) | Belum |
| **Tahap 4** | Predictive Analytics | Pemodelan klasifikasi *default* & regresi beserta evaluasi SHAP (`04_*.ipynb`, `05_*.ipynb`) | Belum |
| **Tahap 5** | Prescriptive Analytics | Optimasi alokasi portofolio pinjaman dengan `PuLP` (`06_prescriptive.ipynb`) | Belum |
| **Tahap 6** | Semantic Analytics | Perancangan ontologi RDF/OWL, *knowledge graph*, dan query SPARQL (`07_semantic.ipynb`, `ontology/`) | Belum |
| **Tahap 7** | Dashboard Interaktif | Pembuatan aplikasi web interaktif menggunakan Streamlit (`dashboard/`) | Belum |
| **Tahap 8** | Laporan Akhir & Finalisasi | Penyusunan laporan akhir (`.docx`) dan pemeriksaan reproduktifitas menyeluruh (`reports/`) | Belum |
