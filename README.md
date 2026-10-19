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
│   │   ├── descriptive.py
│   │   └── diagnostic.py
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

Untuk menjalankan seluruh alur pembersihan data Tahap 1–3 dan menghasilkan `data/processed/bondora_clean.parquet` (`179.235 × 130`, termasuk label gagal bayar berhorizon tetap `observable_12m`, `default_12m`, `observable_24m`, `default_24m`) beserta `data/processed/schema.json` (`44` kandidat fitur `boleh_jadi_fitur = Y`), jalankan perintah satu baris berikut:

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
| **Tahap 3** | Diagnostic Analytics | Label horizon tetap (12m/24m), uji hipotesis, IV, korelasi/VIF, PSI antar-era, Kaplan–Meier, & audit *fairness* (`03_diagnostic.ipynb`) | **Selesai** |
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

---

## 7. Daftar Tabel dan Grafik Tahap 3 (Diagnostic Analytics) & Penjelasan Label Horizon Tetap

### a. Penjelasan Label Gagal Bayar Berhorizon Tetap (`12m` & `24m`)
Untuk mengatasi bias seleksi waktu pengamatan (*right-censoring*) pada pinjaman bertenor panjang (36–60 bulan) dari kohort terbaru yang masih berstatus `Current`, `build_clean.py` menambahkan 4 kolom target/flag horizon tetap berdasarkan `REPORT_DATE = 2021-07-20`:
- **`observable_12m` (`bool`, `148.783` baris `True`) & `default_12m` (`Int8` nullable, `148.369` baris valid):** Bernilai `1` (`43.117` baris; `29,06%`) jika `DefaultDate <= LoanDate + 12 bulan`; `0` (`105.252` baris) jika teramati penuh 12 bulan tanpa default dalam 12 bulan pertama; `<NA>` (`30.866` baris) jika belum teramati 12 bulan (`30.452` baris) atau termasuk `448` pinjaman `Repaid` *write-off* tanpa `DefaultDate` (`414` baris di dalam `observable_12m`).
- **`observable_24m` (`bool`, `98.027` baris `True`) & `default_24m` (`Int8` nullable, `97.878` baris valid):** Uji sensitivitas horizon 24 bulan dengan `44.584` baris `default_24m = 1` (`45,55%`), `53.294` baris `0`, dan `81.357` baris `<NA>`.

### b. Tabel Ringkasan (`reports/tables/03_*.csv` — 12 Tabel)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/tables/03_horizon_label_summary.csv` | Ringkasan per `loan_year` (`2009–2021` + `Total`): `n_total`, `n_observable_12m`, `n_valid_12m`, `n_default_12m`, `rate_12m`, `n_observable_24m`, `rate_24m`, dan perbandingan terhadap `rate_resolved_lifetime`. |
| `reports/tables/03_default_rate_horizon_by_segment.csv` | Profil default rate 12m dan 24m beserta CI Wilson 95% dan bobot `Amount` lintas segmen fitur kandidat dan kolom referensi non-fitur (`Rating`, `Interest_group`, `Gender`). |
| `reports/tables/03_rating_vs_observed_12m.csv` | Perbandingan sebanding horizon 1 tahun antara rata-rata/median `ProbabilityOfDefault` Bondora dan `observed_default_rate_12m` per kelas `Rating` (`AA` s.d. `HR` + `Tidak tersedia`). |
| `reports/tables/03_benchmark_auc.csv` | ROC-AUC benchmark `ProbabilityOfDefault` terhadap `default_12m` beserta CI *bootstrap* 95% (200 iterasi, `RANDOM_SEED = 42`) untuk seluruh populasi teramati (`0,6998`) dan per era (`<=2016`, `2017`, `>=2018`). |
| `reports/tables/03_hypothesis_tests.csv` | Hasil uji Chi-square (kategorikal, Cramér's V) dan Mann–Whitney U (numerik, AUC univariat) terhadap `default_12m` untuk 44 fitur kandidat lengkap dengan koreksi uji berganda Holm (`p_holm`). |
| `reports/tables/03_driver_ranking.csv` | Peringkat eksploratif 44 fitur kandidat menurut *Information Value* (`iv_semua`, `iv_era1` 2009–2017, `iv_era2` 2018–2020), kategori IV, `auc_univariat`, `cramers_v`, dan catatan pergeseran era. |
| `reports/tables/03_return_ratio_drivers.csv` | Asosiasi fitur numerik (korelasi Spearman + `p_holm`) dan kategorikal (Kruskal–Wallis + rentang median antar kelompok) terhadap `return_ratio` pada subset `is_resolved == True` (`n = 113.560`). |
| `reports/tables/03_high_correlation_pairs.csv` | Daftar 19 pasangan fitur numerik kandidat dengan korelasi Spearman $\|\rho_s\| \ge 0,80$ beserta penjelasan domain sebagai acuan penyaringan fitur di Tahap 4. |
| `reports/tables/03_vif.csv` | Nilai *Variance Inflation Factor* (VIF) dan $R^2$ multivariat untuk 31 fitur numerik kandidat (dengan imputasi median sementara khusus perhitungan VIF). |
| `reports/tables/03_psi_by_era.csv` | Nilai *Population Stability Index* (PSI) untuk 44 fitur kandidat dengan kohort referensi `2009–2016` terhadap kohort `2017` dan `2018–2020`. |
| `reports/tables/03_km_cumulative_default.csv` | Tabel *number at risk* (`6, 12, 24, 36` bulan), insiden kumulatif Kaplan–Meier $1 - S(t)$, perbandingan dengan `rate_12m`/`rate_24m` langsung, dan uji *log-rank* terkoreksi Holm (`Country`, `Rating`, `LoanDuration_group`, `cohort_era`). |
| `reports/tables/03_fairness_summary.csv` | Ringkasan transparansi keadilan deskriptif (`Gender` dengan kode asli + `TODO: label resmi`, `Age_group`, `Country`) terhadap `default_12m` beserta selisih mutlak, rasio, dan Cramér's V. |

### c. Grafik Visualisasi (`reports/figures/03_*.png` — 9 Grafik)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/03_horizon_default_rate_by_year.png` | Perbandingan tren tingkat gagal bayar per kohort tahun pencairan: `default_12m` vs `default_24m` vs `resolved lifetime` (Tahap 2). |
| `reports/figures/03_default_rate_horizon_utama.png` | Profil default rate 12 bulan vs 24 bulan (dengan CI Wilson 95%) menurut `Country`, `NewCreditCustomer`, `VerificationType`, `HomeOwnershipType`, `Education`, dan `Rating`. |
| `reports/figures/03_default_rate_horizon_tenor_dan_demografi.png` | Profil default rate 12 bulan vs 24 bulan menurut 5 kelompok tenor (`LoanDuration_group`, disertai pembanding *resolved lifetime*), `Age_group`, `IncomeTotal_quartile`, `Amount_quartile`, `Interest_group`, dan `UseOfLoan` (pra-2018). |
| `reports/figures/03_rating_pd_vs_observed_12m.png` | Evaluasi kalibrasi rata-rata `ProbabilityOfDefault` Bondora vs tingkat gagal bayar 12 bulan teramati per kelas `Rating` dan kurva ROC benchmark per era. |
| `reports/figures/03_driver_ranking_iv.png` | Grafik batang horizontal 20 fitur kandidat teratas menurut *Information Value* (IV) keseluruhan beserta perbandingan IV Era 1 (`2009–2017`) vs Era 2 (`2018–2020`). |
| `reports/figures/03_correlation_heatmap.png` | Peta panas (*heatmap*) matriks korelasi peringkat Spearman antar 31 fitur numerik kandidat (`boleh_jadi_fitur = Y`). |
| `reports/figures/03_psi_by_era.png` | Grafik batang horizontal *Population Stability Index* (PSI) fitur kandidat pada kohort `2017` dan `2018–2020` terhadap referensi `2009–2016`. |
| `reports/figures/03_km_country_and_rating.png` | Kurva insiden kumulatif gagal bayar Kaplan–Meier $1 - S(t)$ hingga 36 bulan menurut `Country` dan kelas `Rating` (`AA` s.d. `HR`). |
| `reports/figures/03_km_tenor_and_cohort.png` | Kurva insiden kumulatif gagal bayar Kaplan–Meier $1 - S(t)$ hingga 36 bulan menurut kelompok tenor (`LoanDuration_group`) dan era kohort pencairan (`cohort_era`). |



