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
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── descriptive.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── targets.py
│   │   └── build_clean.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── basic.py
│   ├── models/__init__.py
│   └── utils/
│       ├── __init__.py
│       └── plotting.py
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

## 4. Instruksi Penempatan Dataset & Pembuatan Data Bersih

Dataset **tidak disertakan di dalam repositori Git**. Silakan unduh dataset Bondora secara manual dan simpan ke dalam folder:

```text
data/raw/
```

Lihat panduan lengkap, daftar file yang dibutuhkan (`LoanData.csv` dan opsional `RepaymentsData.csv`), serta tabel pencatatan metadata unduhan pada dokumen [`data/README.md`](data/README.md).

Untuk menjalankan seluruh alur pembersihan data Tahap 1 dan menghasilkan `data/processed/bondora_clean.parquet` beserta `data/processed/schema.json`, jalankan perintah satu baris berikut:

```bash
python -m src.data.build_clean
```

---

## 5. Status Tahap Pengerjaan Proyek

| Tahap | Nama Tahap | Fokus Utama & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Tahap 0** | Persiapan Proyek & Environment | Struktur repo, `.venv`, instalasi paket, konfigurasi, template notebook & dokumen | **Selesai** |
| **Tahap 1** | Data Understanding & Preprocessing | Eksplorasi skema, pembersihan data, penanganan *missing values* & *data leakage* (`01_data_understanding.ipynb`) | **Selesai** |
| **Tahap 2** | Descriptive Analytics | Statistik deskriptif & visualisasi distribusi portofolio (`02_descriptive.ipynb`) | **Selesai** |
| **Tahap 3** | Diagnostic Analytics | Uji hipotesis, analisis korelasi, dan faktor penyebab *default* (`03_diagnostic.ipynb`) | Belum |
| **Tahap 4** | Predictive Analytics | Pemodelan klasifikasi *default* & regresi beserta evaluasi SHAP (`04_*.ipynb`, `05_*.ipynb`) | Belum |
| **Tahap 5** | Prescriptive Analytics | Optimasi alokasi portofolio pinjaman dengan `PuLP` (`06_prescriptive.ipynb`) | Belum |
| **Tahap 6** | Semantic Analytics | Perancangan ontologi RDF/OWL, *knowledge graph*, dan query SPARQL (`07_semantic.ipynb`, `ontology/`) | Belum |
| **Tahap 7** | Dashboard Interaktif | Pembuatan aplikasi web interaktif menggunakan Streamlit (`dashboard/`) | Belum |
| **Tahap 8** | Laporan Akhir & Finalisasi | Penyusunan laporan akhir (`.docx`) dan pemeriksaan reproduktifitas menyeluruh (`reports/`) | Belum |

---

## 6. Daftar Tabel dan Grafik Tahap 2 (Descriptive Analytics)

### a. Tabel Ringkasan (`reports/tables/02_*.csv`)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/tables/02_kpi_portofolio.csv` | Ringkasan KPI portofolio (`n = 179.235`) dan metrik hasil akhir pada subset *resolved* (`n = 113.560`) lengkap dengan nilai antara (pembilang, penyebut, `n`). |
| `reports/tables/02_numeric_summary.csv` | Statistik deskriptif lengkap (n, missing, mean, std, min, p01–p99, max, skewness, kurtosis) untuk 33 variabel numerik. |
| `reports/tables/02_categorical_frequency.csv` | Tabel frekuensi format panjang (`variabel`, `nilai`, `jumlah`, `persen`) untuk 12 variabel kategorikal utama dengan kode `-1 (Not set)` terpisah. |
| `reports/tables/02_yearly_trend.csv` | Tren tahunan (`2009–2021`) jumlah pinjaman, total nominal EUR, serta rata-rata/median `Amount`, `Interest`, `LoanDuration`, dan `IncomeTotal`. |
| `reports/tables/02_outcome_by_year.csv` | Komposisi 4 kelompok hasil pinjaman per tahun pencairan beserta `n_total`, `n_resolved`, dan `pct_resolved`. |
| `reports/tables/02_missing_by_year.csv` | Jumlah dan persentase nilai kosong per tahun (`loan_year`) beserta baris `Total` terverifikasi untuk `MonthlyPayment` (6.690 / 3,73%), `PreviousRepaymentsBeforeLoan` (7.441 / 4,15%), dan `CreditScoreEsMicroL` (30.304 / 16,91%). |
| `reports/tables/02_default_rate_by_segment.csv` | Profil proporsi default, interval kepercayaan Wilson 95%, dan default rate berbobot `Amount` pada subset *resolved* (`n = 113.560`) lintas 12 dimensi segmentasi (termasuk 5 bin `LoanDuration_group`: `<=12`, `13-24`, `25-36`, `37-59`, `60` bulan). |
| `reports/tables/02_return_ratio_summary.csv` | Ringkasan distribusi `return_ratio` (mean, berbobot `Amount`, kuartil/persentil, persentase nol) menurut kelompok hasil, `Country`, `Rating`, dan `loan_year`. |

### b. Grafik Visualisasi (`reports/figures/02_*.png` — 18 Grafik)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/02_outcome_composition.png` | Grafik batang horizontal dan donat komposisi 4 kelompok hasil pada seluruh portofolio (`n = 179.235`). |
| `reports/figures/02_numeric_distributions_pinjaman.png` | Histogram dan boxplot berdampingan untuk `Amount`, `AppliedAmount`, `Interest`, dan `LoanDuration`. |
| `reports/figures/02_numeric_distributions_peminjam.png` | Histogram dan boxplot berdampingan untuk `MonthlyPayment`, `Age` (tanpa umur tidak wajar), `IncomeTotal` (tanpa pendapatan nol), dan `LiabilitiesTotal`. |
| `reports/figures/02_numeric_distributions_rasio_return.png` | Histogram dan boxplot berdampingan untuk `DebtToIncome`, `FreeCash`, `loan_to_income_ratio`, dan `return_ratio` (khusus *resolved*). |
| `reports/figures/02_country_distribution.png` | Grafik batang horizontal distribusi jumlah dan persentase pinjaman menurut negara asal (`EE`, `FI`, `ES`, `SK`). |
| `reports/figures/02_categorical_distributions_1.png` | Grafik batang horizontal proporsi untuk `Rating`, `NewCreditCustomer`, `VerificationType`, `LanguageCode`, `Gender`, dan `Education`. |
| `reports/figures/02_categorical_distributions_2.png` | Grafik batang horizontal proporsi untuk `MaritalStatus`, `EmploymentStatus`, `OccupationArea`, `UseOfLoan`, dan `HomeOwnershipType` (kode `-1 (Not set)` ditandai). |
| `reports/figures/02_loan_volume_trend.png` | Grafik garis perkembangan jumlah pinjaman dan total nominal `Amount` (Juta EUR) per tahun dan per bulan (`2009–2021`). |
| `reports/figures/02_yearly_metrics_trend.png` | Grafik garis tren tahunan rata-rata dan median `Amount`, `Interest`, `LoanDuration`, serta `IncomeTotal`. |
| `reports/figures/02_country_mix_by_year.png` | Grafik batang bertumpuk 100% komposisi negara asal pinjaman (`Country`) per tahun pencairan. |
| `reports/figures/02_sentinel_and_model_version_by_year.png` | Grafik tren persentase kode `-1 (Not set)` dan distribusi `ModelVersion` per tahun pencairan. |
| `reports/figures/02_outcome_mix_by_year.png` | Grafik batang bertumpuk 100% komposisi 4 kelompok hasil per tahun pencairan beserta `pct_resolved`. |
| `reports/figures/02_missing_indicators_by_year.png` | Grafik garis persentase nilai kosong per tahun untuk `MonthlyPayment`, `PreviousRepaymentsBeforeLoan`, dan `CreditScoreEsMicroL`. |
| `reports/figures/02_default_rate_utama.png` | Profil default rate beserta CI Wilson 95% menurut `Country`, `Rating`, `NewCreditCustomer`, `VerificationType`, kelompok `Age`, dan kuartil `IncomeTotal`. |
| `reports/figures/02_default_rate_demografi_pinjaman.png` | Profil default rate beserta CI Wilson 95% menurut 5 kelompok `LoanDuration` (`<=12`, `13-24`, `25-36`, `37-59`, `60` bln), kuartil `Amount`, `Education`, `EmploymentStatus`, `HomeOwnershipType`, dan `UseOfLoan`. |
| `reports/figures/02_rating_pd_vs_observed.png` | Perbandingan urutan peringkat `ProbabilityOfDefault` Bondora (horizon 1 tahun) terhadap default rate teramati (umur pinjaman pada subset *resolved*; level tidak sebanding) per kelas `Rating` (`AA` s.d. `HR`). |
| `reports/figures/02_return_ratio_by_outcome.png` | Histogram dan boxplot distribusi `return_ratio` antara pinjaman `Default` dan `Lunas tanpa default` pada subset *resolved*. |
| `reports/figures/02_principal_recovery_distribution.png` | Proporsi pinjaman default dengan `PrincipalRecovery > 0` dan distribusi rasio pemulihan pokok (`PrincipalRecovery / Amount`). |

### c. Daftar Pilihan (15 Tabel & Grafik) untuk Laporan Utama (`.docx`)
- **4 Tabel:** `02_kpi_portofolio.csv`, `02_outcome_by_year.csv`, `02_default_rate_by_segment.csv`, `02_return_ratio_summary.csv`.
- **11 Grafik:** `02_outcome_composition.png`, `02_numeric_distributions_pinjaman.png`, `02_country_distribution.png`, `02_loan_volume_trend.png`, `02_country_mix_by_year.png`, `02_sentinel_and_model_version_by_year.png`, `02_outcome_mix_by_year.png`, `02_missing_indicators_by_year.png`, `02_default_rate_utama.png`, `02_default_rate_demografi_pinjaman.png`, `02_return_ratio_by_outcome.png` *(catatan: `02_rating_pd_vs_observed.png` ditunda untuk dianalisis dengan horizon 1 tahun yang setara pada Tahap 3/4)*.


