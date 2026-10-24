# Bondora Peer-to-Peer (P2P) Lending Analytics

Proyek analisis komprehensif pada dataset pinjaman Peer-to-Peer (P2P) **Bondora** yang mencakup lima pilar analitik data (**Descriptive, Diagnostic, Predictive, Prescriptive, dan Semantic Analytics**). Hasil akhir proyek ini diwujudkan dalam bentuk rangkaian **Notebook Jupyter**, **Deck Laporan (PPT)**, serta **Dashboard Interaktif Streamlit**.

---

## 1. Lima Jenis Analytics & Cakupan Tahap

1. **Descriptive Analytics** — Meringkas karakteristik historis portofolio pinjaman, profil demografis peminjam, distribusi suku bunga, dan tingkat gagal bayar (*default rate*).
2. **Diagnostic Analytics** — Menganalisis akar penyebab dan hubungan statistik/korelasi faktor-faktor risiko yang membedakan pinjaman lancar dengan pinjaman gagal bayar.
3. **Predictive Analytics**
   - **Klasifikasi Default:** Memprediksi probabilitas gagal bayar pinjaman 12 bulan (`default_12m`) menggunakan algoritma *machine learning* (`scikit-learn`, `lightgbm`) beserta kalibrasi probabilitas dan interpretasi model (`shap`, *permutation importance*).
   - **Regresi:** Mengestimasi rasio pengembalian finansial kontinu (`return_ratio`) pada pinjaman P2P *resolved*.
4. **Prescriptive Analytics** — Mengoptimalkan pemilihan dan alokasi portofolio pinjaman menggunakan pemrograman linier/optimasi matematis (`pulp`) untuk memaksimalkan imbal hasil pada batas risiko tertentu.
5. **Semantic Analytics** — Memodelkan pengetahuan domain P2P lending ke dalam skema ontologi dan *knowledge graph* (`rdflib`, `networkx`) yang dapat di-query menggunakan SPARQL.

---

## 2. Struktur Folder Proyek

```text
bondora-analytics/
├── data/
│   ├── raw/            # file asli dari Bondora (tidak di-commit)
│   ├── interim/        # hasil antara (tidak di-commit)
│   ├── processed/      # data bersih & tabel prediksi parquet (tidak di-commit)
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
│   ├── models/
│   │   ├── __init__.py
│   │   ├── splits.py
│   │   ├── features.py
│   │   ├── pipelines.py
│   │   ├── evaluate.py
│   │   ├── followup.py
│   │   ├── inference.py
│   │   └── train.py
│   ├── semantic/
│   │   ├── __init__.py
│   │   ├── sampling.py
│   │   ├── build_graph.py
│   │   ├── queries.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       └── plotting.py
├── models/             # model tersimpan *.joblib & train_metadata.json (tidak di-commit)
├── dashboard/          # aplikasi Streamlit (tahap 7)
├── ontology/
│   ├── bondora_ontology.ttl
│   └── queries/        # q1.rq .. q8.rq (di-commit)
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── 07_kg_interactive.html
│   └── 00_problem_statement.md
├── scripts/
│   └── verify_env.py
├── tests/
│   ├── test_targets.py
│   ├── test_features.py
│   ├── test_descriptive.py
│   ├── test_diagnostic.py
│   ├── test_models.py
│   ├── test_followup.py
│   ├── test_prescriptive.py
│   └── test_semantic.py
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

## 4. Instruksi Penempatan Dataset, Pembuatan Data Bersih, & Pelatihan Model

Dataset **tidak disertakan di dalam repositori Git**. Silakan unduh dataset Bondora secara manual dan simpan ke dalam folder:

```text
data/raw/
```

Lihat panduan lengkap, daftar file yang dibutuhkan (`LoanData.csv` dan opsional `RepaymentsData.csv`), serta tabel pencatatan metadata unduhan pada dokumen [`data/README.md`](data/README.md).

### a. Pembuatan Data Bersih (Tahap 1–3)
Untuk menjalankan seluruh alur pembersihan data Tahap 1–3 dan menghasilkan `data/processed/bondora_clean.parquet` (`179.235 × 130`, termasuk label gagal bayar berhorizon tetap `observable_12m`, `default_12m`, `observable_24m`, `default_24m`) beserta `data/processed/schema.json` (`44` kandidat fitur `boleh_jadi_fitur = Y`), jalankan perintah satu baris berikut:

```bash
python -m src.data.build_clean
```

### b. Pelatihan & Evaluasi Model Prediktif Satu Perintah (Tahap 4)
Untuk menjalankan seluruh pipeline Tahap 4 (pembagian berbasis waktu → seleksi set fitur A/B/C → pelatihan klasifikasi `LR-A`, `LGBM-A/B/C`, `LGBM-A-acak` → kalibrasi probabilitas → pelatihan regresi `Ridge`, `LGBM-Reg-L2`, `LGBM-Reg-Huber` → evaluasi metrik, SHAP, *permutation importance*, serta penyimpanan model `models/*.joblib`, tabel `reports/tables/04_*.csv` & `05_*.csv`, grafik `reports/figures/04_*.png` & `05_*.png`, dan file prediksi `data/processed/predictions_*.parquet`), jalankan:

```bash
python -m src.models.train
```

Opsi tambahan:
- `--task {classification,regression,all}` (default: `all`)
- `--quick` (mode uji cepat dengan iterasi pencarian hiperparameter lebih sedikit)

> **Catatan Git:** Folder `models/` (`*.joblib`, `train_metadata.json`, `prescriptive_timing.json`) dan berkas prediksi `data/processed/predictions_default12m.parquet` serta `data/processed/predictions_return_ratio.parquet` **tidak di-commit ke Git** (diabaikan melalui `.gitignore`) dan dihasilkan secara deterministik melalui `python -m src.models.train`.

### c. Analisis Preskriptif & Forecast Satu Perintah (Tahap 5)
Untuk menjalankan seluruh pipeline Tahap 5 (penetapan parameter bisnis → fungsi ekonomi & estimasi $L$ pada data latih+validasi → evaluasi 8 strategi investor `S0`–`S5` termasuk pembanding berbatas serakah `S2c` & `S4c` serta optimasi berbatas `S3` → simulasi *what-if*, pengali $p$ `x2,0`/`x3,0`, pengali impas *bisection*, & sensitivitas tornado → skenario kebijakan platform `P1`/`P2` beserta diagnostik sensitivitas level $p$ → prakiraan deret waktu bulanan *seasonal naive* vs ETS, serta penyimpanan tabel `reports/tables/06_*.csv` dan grafik `reports/figures/06_*.png`), jalankan:

```bash
python -m src.analysis.prescriptive_run
```

### d. Pembangunan Knowledge Graph, Query SPARQL, & Inferensi Satu Perintah (Tahap 6)
Untuk menjalankan pipeline Semantic Analytics Tahap 6 (pengambilan sampel terstratifikasi `4.000` pinjaman + `5.000` pinjaman portofolio `S3` + perluasan rantai peminjam hingga `20.000` pinjaman → penyekoran `p_calibrated` via `src.models.inference` → pembangunan graf RDF/OWL/SKOS `data/processed/bondora_graph.ttl.gz` berjumlah `340.461` triple → eksekusi 8 query SPARQL dengan verifikasi silang eksak `pandas` → penutupan deduktif RDFS/OWL-RL via `owlrl.DeductiveClosure` → pembuatan tabel `reports/tables/07_*.csv` dan grafik `reports/figures/07_*.png`), jalankan:

```bash
python -m src.semantic.build_graph
python -m src.semantic.queries
python -m src.semantic.visualize
```

> **Catatan Git:** Berkas graf instans `data/processed/bondora_graph.ttl.gz` (`0,741 MB`), sampel antara `data/processed/semantic_sample.parquet`, dan daftar pinjaman `data/processed/s3_selected_loans.parquet` **tidak di-commit ke Git** (diabaikan melalui `.gitignore`) dan dapat dibangun ulang secara deterministik kapan saja dengan menjalankan `python -m src.semantic.build_graph`. Yang di-commit ke Git adalah ontologi (`ontology/bondora_ontology.ttl`), berkas query (`ontology/queries/q1.rq`..`q8.rq`), kode sumber (`src/semantic/`), tabel (`reports/tables/07_*.csv`), grafik (`reports/figures/07_*.png`, `reports/07_kg_interactive.html`), dan notebook (`notebooks/07_semantic.ipynb`).

---

## 5. Status Tahap Pengerjaan Proyek

| Tahap | Nama Tahap | Fokus Utama & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Tahap 0** | Persiapan Proyek & Environment | Struktur repo, `.venv`, instalasi paket, konfigurasi, template notebook & dokumen | **Selesai** |
| **Tahap 1** | Data Understanding & Preprocessing | Eksplorasi skema, pembersihan data, penanganan *missing values* & *data leakage* (`01_data_understanding.ipynb`) | **Selesai** |
| **Tahap 2** | Descriptive Analytics | Statistik deskriptif & visualisasi distribusi portofolio (`02_descriptive.ipynb`) | **Selesai** |
| **Tahap 3** | Diagnostic Analytics | Label horizon tetap (12m/24m), uji hipotesis, IV, korelasi/VIF, PSI antar-era, Kaplan–Meier, & audit *fairness* (`03_diagnostic.ipynb`) | **Selesai** |
| **Tahap 4** | Predictive Analytics | Pemodelan klasifikasi *default* 12m & regresi `return_ratio`, kalibrasi probabilitas, evaluasi SHAP & subgrup (`04_*.ipynb`, `05_*.ipynb`, `src/models/`) | **Selesai** |
| **Tahap 5** | Prescriptive Analytics | Optimasi alokasi portofolio investor (`S0`–`S5`), simulasi *what-if*, kebijakan platform (`P1`/`P2`), & *forecast* bulanan (`06_prescriptive.ipynb`, `src/analysis/prescriptive*.py`, `src/analysis/forecasting.py`) | **Selesai** |
| **Tahap 6** | Semantic Analytics | Perancangan ontologi OWL/RDFS & taksonomi SKOS (`ontology/bondora_ontology.ttl`), pembangunan *knowledge graph* (`344.904` triple), 8 query SPARQL terverifikasi `pandas`, & inferensi OWL-RL (`07_semantic.ipynb`, `src/semantic/`) | **Selesai** |
| **Tahap 7** | Dashboard Interaktif | Pembuatan aplikasi web interaktif menggunakan Streamlit (`dashboard/`) | Belum |
| **Tahap 8** | Laporan Akhir & Finalisasi | Penyusunan deck laporan (PPT) dan pemeriksaan reproduktifitas menyeluruh (`reports/`) | Belum |

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

### c. Daftar Pilihan (15 Tabel & Grafik) untuk Deck Laporan (PPT)
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
| `reports/tables/03_default_rate_horizon_by_segment.csv` | Profil default rate 12m dan 24m (serta pembanding `LoanDuration_group_lifetime` `n = 113.560`, `default = 71.864`) beserta CI Wilson 95% dan bobot `Amount` lintas segmen fitur kandidat dan kolom referensi non-fitur (`Rating`, `Interest_group`, `Gender`). |
| `reports/tables/03_rating_vs_observed_12m.csv` | Perbandingan sebanding horizon 1 tahun antara rata-rata/median `ProbabilityOfDefault` Bondora dan `observed_default_rate_12m` per kelas `Rating` (`AA` s.d. `HR` + `Tidak tersedia`). |
| `reports/tables/03_benchmark_auc.csv` | ROC-AUC benchmark `ProbabilityOfDefault` terhadap `default_12m` beserta CI *bootstrap* 95% (200 iterasi, `RANDOM_SEED = 42`) untuk seluruh populasi teramati (`0,6998`) dan per era (`<=2016`, `2017`, `>=2018`). |
| `reports/tables/03_hypothesis_tests.csv` | Hasil uji Chi-square (kategorikal, Cramér's V) dan Mann–Whitney U (numerik, AUC univariat) terhadap `default_12m` untuk 44 fitur kandidat lengkap dengan koreksi uji berganda Holm (`p_holm`). |
| `reports/tables/03_driver_ranking.csv` | Peringkat eksploratif 44 fitur kandidat menurut *Information Value* (`iv_semua`, `iv_era1` 2009–2017, `iv_era2` 2018–2020), kategori IV, `auc_univariat`, `cramers_v`, dan catatan pergeseran era. |
| `reports/tables/03_return_ratio_drivers.csv` | Asosiasi fitur numerik (korelasi Spearman + `p_holm`) dan kategorikal (Kruskal–Wallis + rentang median antar kelompok) terhadap `return_ratio` pada subset `is_resolved == True` (`n = 113.560`). |
| `reports/tables/03_high_correlation_pairs.csv` | Daftar 19 pasangan fitur numerik kandidat dengan korelasi Spearman $\|\rho_s\| \ge 0,80$ beserta penjelasan domain sebagai acuan penyaringan fitur di Tahap 4. |
| `reports/tables/03_vif.csv` | Nilai *Variance Inflation Factor* (VIF) dan $R^2$ multivariat untuk 31 fitur numerik kandidat (dengan imputasi median sementara khusus perhitungan VIF). |
| `reports/tables/03_psi_by_era.csv` | Nilai *Population Stability Index* (PSI) untuk 44 fitur kandidat dengan kohort referensi `2009–2016` terhadap kohort `2017` dan `2018–2020`. |
| `reports/tables/03_km_cumulative_default.csv` | Tabel *number at risk* (`6, 12, 24, 36` bulan), insiden kumulatif Kaplan–Meier $1 - S(t)$, perbandingan dengan `rate_12m`/`rate_24m` langsung, dan uji *log-rank* terkoreksi Holm (`Country`, `Rating`, `LoanDuration_group`, `cohort_era`). Keterangan: *"Insiden kumulatif dengan pelunasan dini sebagai penyensoran (cause-specific); bukan probabilitas default aktual. Untuk probabilitas aktual gunakan rate_12m_langsung."* |
| `reports/tables/03_fairness_summary.csv` | Ringkasan transparansi keadilan deskriptif (`Gender` dengan kode asli + `TODO: label resmi`, `Age_group`, `Country`) terhadap `default_12m` beserta selisih mutlak, rasio, dan Cramér's V. |

### c. Grafik Visualisasi (`reports/figures/03_*.png` — 9 Grafik)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/03_horizon_default_rate_by_year.png` | Perbandingan tren tingkat gagal bayar per kohort tahun pencairan dan kelompok tenor (`LoanDuration_group`): `default_12m` vs `default_24m` vs `resolved lifetime` (Tahap 2: `<=12` `7.424 / 27,49%`, `13-24` `7.428 / 36,65%`, `25-36` `36.114 / 65,21%`, `37-59` `8.207 / 66,59%`, `60` `54.387 / 70,03%`; total `n = 113.560`, `default = 71.864`). |
| `reports/figures/03_default_rate_horizon_utama.png` | Profil default rate 12 bulan vs 24 bulan (dengan CI Wilson 95%) menurut `Country`, `NewCreditCustomer`, `VerificationType`, `Age_group`, `IncomeTotal_quartile`, dan `Amount_quartile`. |
| `reports/figures/03_default_rate_horizon_tenor_dan_demografi.png` | Profil default rate 12 bulan vs 24 bulan menurut 5 kelompok tenor (`LoanDuration_group`, disertai pembanding *resolved lifetime* Tahap 2), `Education`, `EmploymentStatus`, `HomeOwnershipType`, `UseOfLoan` (pra-2018), dan `Rating`. |
| `reports/figures/03_rating_pd_vs_observed_12m.png` | Evaluasi kalibrasi rata-rata `ProbabilityOfDefault` Bondora vs tingkat gagal bayar 12 bulan teramati per kelas `Rating` dan kurva ROC benchmark per era. |
| `reports/figures/03_driver_ranking_iv.png` | Grafik batang horizontal 20 fitur kandidat teratas menurut *Information Value* (IV) keseluruhan beserta perbandingan IV Era 1 (`2009–2017`) vs Era 2 (`2018–2020`). |
| `reports/figures/03_correlation_heatmap.png` | Peta panas (*heatmap*) matriks korelasi peringkat Spearman antar 31 fitur numerik kandidat (`boleh_jadi_fitur = Y`). |
| `reports/figures/03_psi_by_era.png` | Grafik batang horizontal *Population Stability Index* (PSI) fitur kandidat pada kohort `2017` dan `2018–2020` terhadap referensi `2009–2016`. |
| `reports/figures/03_km_country_and_rating.png` | Kurva insiden kumulatif gagal bayar Kaplan–Meier $1 - S(t)$ hingga 36 bulan menurut `Country` dan kelas `Rating` (`AA` s.d. `HR`). Keterangan: *"Insiden kumulatif dengan pelunasan dini sebagai penyensoran (cause-specific); bukan probabilitas default aktual. Untuk probabilitas aktual gunakan rate_12m_langsung."* |
| `reports/figures/03_km_tenor_and_cohort.png` | Kurva insiden kumulatif gagal bayar Kaplan–Meier $1 - S(t)$ hingga 36 bulan menurut kelompok tenor (`LoanDuration_group`) dan era kohort pencairan (`cohort_era`). Keterangan: *"Insiden kumulatif dengan pelunasan dini sebagai penyensoran (cause-specific); bukan probabilitas default aktual. Untuk probabilitas aktual gunakan rate_12m_langsung."* |

### d. Daftar Pilihan untuk Deck Laporan (PPT) — Tahap 3
- **Tabel Utama:** `03_horizon_label_summary.csv`, `03_default_rate_horizon_by_segment.csv`, `03_rating_vs_observed_12m.csv`, `03_driver_ranking.csv`, `03_psi_by_era.csv`, `03_km_cumulative_default.csv`, `03_fairness_summary.csv`.
- **Grafik Utama:** `03_horizon_default_rate_by_year.png`, `03_default_rate_horizon_utama.png`, `03_default_rate_horizon_tenor_dan_demografi.png`, `03_rating_pd_vs_observed_12m.png`, `03_driver_ranking_iv.png`, `03_correlation_heatmap.png`, `03_psi_by_era.png`, `03_km_country_and_rating.png`, `03_km_tenor_and_cohort.png`.

---

## 8. Daftar Tabel, Grafik, dan Artefak Tahap 4 (Predictive Analytics)

### a. Tabel Ringkasan Klasifikasi & Regresi (`reports/tables/04_*.csv` & `05_*.csv` — 18 Tabel)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/tables/04_split_summary.csv` | Ringkasan ukuran, proporsi default (`default_12m`), dan rentang tanggal `LoanDate` pada split berbasis waktu (`train <= 2018-06-30`, `val 2018-07-01..2019-06-30`, `test >= 2019-07-01`) dan split acak terstratifikasi `70/15/15`. |
| `reports/tables/04_feature_sets.csv` | Definisi 44 fitur kandidat beserta keanggotaan **Set A** (`36` fitur stabil), **Set B** (`44` fitur lengkap), **Set C** (`39` fitur ablasi + `Rating`, `ProbabilityOfDefault`, `Interest` — tidak menambah AUC; penurunan di uji `-0,0074` kecil dan penyebabnya belum dibuktikan), proporsi nilai dominan latih vs validasi, PSI `2018–2020`, dan alasan keluar 8 fitur *mati-era*. |
| `reports/tables/04_metrics_classification.csv` | Metrik evaluasi klasifikasi (`ROC-AUC` + CI bootstrap 95%, `PR-AUC`, `Gini`, `KS`, `Brier`, `Log Loss`, `ECE` 10-bin, ambang Youden validasi, matriks konfusi, `Precision/Recall` Top-10%/20%/30%, serta penanda `cakupan_evaluasi`: metrik kalibrasi pada validasi bersifat *in-sample*, dan metrik uji yang dipakai untuk pelaporan) untuk semua model dan pembanding (`ProbabilityOfDefault`, `Interest`). |
| `reports/tables/04_auc_comparisons.csv` | Selisih ROC-AUC berpasangan (*paired bootstrap* 200 iterasi, CI 95%) pada data uji antara `LGBM-A` terhadap `LGBM-B`, `LGBM-C`, `LR-A`, `ProbabilityOfDefault`, dan `Interest`. |
| `reports/tables/04_lift_by_decile.csv` | Tabel *lift* dan *cumulative recall/gain* 10 desil pada data uji (`n = 54.283`) untuk model utama `LGBM-A` terkalibrasi. |
| `reports/tables/04_subgroup_performance.csv` | Evaluasi ROC-AUC (dengan CI bootstrap 95%), Brier, KS, dan kalibrasi rata-rata prediksi vs default rate aktual per subgrup (`Country` `EE`, `FI`, `ES` berdampingan dengan pembanding `ProbabilityOfDefault` Bondora dan `Interest` beserta selisih AUC berpasangan + CI 95%, `LoanDuration_group`, `NewCreditCustomer`, dan paruh waktu uji `Jul–Des 2019` vs `Jan–Jul 2020`). |
| `reports/tables/04_level_bias.csv` | Perbandingan rata-rata `p_calibrated` (`LGBM-A`), `p_raw`, dan `PD_bondora` terhadap tingkat gagal bayar 12 bulan aktual pada validasi (*in-sample*: `29,26%` vs `29,26%`) dan uji (*out-of-sample*: `32,96%` vs `26,77%`, level terlalu tinggi `+6,19 pp`), serta pada uji menurut `Country`, paruh waktu (`Jul–Des 2019`, `Jan–Jul 2020`), dan `Country × paruh_waktu_uji` (urutan risiko lebih andal daripada level). |
| `reports/tables/04_nan_signal_shap.csv` | Pemeriksaan `NaN` sebagai sinyal era: perbandingan rata-rata SHAP dan tingkat gagal bayar baris `NaN` vs non-`NaN` pada sampel SHAP uji (`n = 5.000`) dan latih (`n = 5.000`) untuk `PreviousRepaymentsBeforeLoan`, `MonthlyPayment`, `monthly_payment_to_income_ratio`, `CreditScoreFiAsiakasTietoRiskGrade`, dan `CreditScoreEeMini`. |
| `reports/tables/04_nan_signal_refit.csv` | Hasil eksperimen latih ulang `LGBM-A` satu kali dengan hiperparameter terbaik yang sama dan nilai `NaN` pada 5 fitur terkonsentrasi-era diisi konstan tunggal data latih (ROC-AUC uji `0,7164` vs model utama `0,7152`, selisih `+0,0012` `[+0,0005, +0,0019]`; model utama tidak diganti). |
| `reports/tables/04_shap_top15.csv` | 15 fitur teratas menurut rata-rata nilai mutlak SHAP (`mean_abs_shap`) model `LGBM-A` pada sampel acak `5.000` baris uji. |
| `reports/tables/04_permutation_importance.csv` | Peringkat seluruh 36 fitur Set A menurut penurunan ROC-AUC validasi (*permutation importance*, 5 ulangan). |
| `reports/tables/04_lr_top15_coefficients.csv` | 15 koefisien terstandarisasi terbesar pada model linear `LR-A` beserta arah pengaruh terhadap risiko gagal bayar. |
| `reports/tables/04_driver_ranking_train_only.csv` | Perbandingan *Information Value* (IV) yang dihitung murni pada data latih (`<= 2018-06-30`) terhadap peringkat IV eksploratif seluruh populasi dari Tahap 3. |
| `reports/tables/04_local_explanations_examples.csv` | Dekomposisi kontribusi SHAP lokal pada 3 contoh pinjaman uji (risiko rendah `~P10`, risiko menengah `~Median`, dan risiko tinggi `~P90`). |
| `reports/tables/05_split_summary.csv` | Ringkasan ukuran, default rate *lifetime*, serta rata-rata dan kuartil `return_ratio` pada split waktu regresi (`is_resolved == True`, `n = 113.560`). |
| `reports/tables/05_metrics_regression.csv` | Metrik evaluasi regresi (`MAE`, `RMSE`, $R^2$, korelasi peringkat Spearman $\rho_s$, rata-rata Desil 1 vs Desil 10, serta `MAE`/`RMSE` terpisah untuk pinjaman default vs non-default; rata-rata prediksi uji `0,6746` vs aktual `0,5708`, gunakan sebagai urutan) untuk pembanding naif, `ExpectedReturn`, `Ridge-A`, `LGBM-Reg-L2`, dan `LGBM-Reg-Huber`. |
| `reports/tables/05_lift_by_decile.csv` | Tabel *lift* 10 desil prediksi `return_ratio` pada data uji (`n = 34.675`) beserta rata-rata/median `return_ratio` aktual dan tingkat default per desil. |
| `reports/tables/05_shap_top15.csv` | 15 fitur teratas menurut rata-rata nilai mutlak SHAP (`mean_abs_shap`) pada model regresi utama (`LGBM-Reg-Huber`) pada sampel acak `5.000` baris uji. |

### b. Grafik Visualisasi (`reports/figures/04_*.png` & `05_*.png` — 8 Grafik)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/04_roc_curves.png` | Kurva ROC pada data uji untuk `LGBM-A`, `LGBM-B`, `LGBM-C`, `LR-A`, `LGBM-A-acak` (70/15/15), `ProbabilityOfDefault` Bondora, dan suku bunga `Interest`. |
| `reports/figures/04_calibration_curve.png` | Kurva reliabilitas (10 bin) sebelum dan sesudah kalibrasi (Platt vs Isotonik) serta distribusi probabilitas prediksi pada data uji. |
| `reports/figures/04_lift_by_decile.png` | Grafik batang tingkat default aktual vs prediksi per desil risiko dan kurva *cumulative recall* (`LGBM-A` pada data uji). |
| `reports/figures/04_shap_summary_bar.png` | Perbandingan 15 fitur terpenting `LGBM-A` menurut `Mean |SHAP|` (data uji) dan *Permutation Importance* (penurunan AUC validasi). |
| `reports/figures/04_subgroup_performance.png` | Profil ROC-AUC (beserta CI 95%) dan perbandingan rata-rata probabilitas terkalibrasi vs default rate aktual lintas subgrup data uji. |
| `reports/figures/05_pred_vs_actual_and_lift.png` | Diagram pencar (*scatter*) prediksi vs aktual `return_ratio` (sampel uji `n = 3.000`) dan rata-rata `return_ratio` aktual vs prediksi per desil (`D1` tertinggi s.d. `D10` terendah). |
| `reports/figures/05_residuals_by_country_and_rating.png` | Boxplot distribusi residual (`Aktual - Prediksi`) model regresi utama menurut `Country` dan kelas `Rating` pada data uji. |
| `reports/figures/05_shap_top15_regression.png` | Grafik batang horizontal 15 fitur terpenting model regresi utama (`LGBM-Reg-Huber`) menurut rata-rata nilai mutlak SHAP pada sampel uji (`n = 5.000`). |

### c. Artefak Model & Tabel Prediksi (Tidak Di-commit ke Git)
- **Model Tersimpan (`models/`):** `lgbm_default12m_setA.joblib`, `lr_default12m_setA.joblib`, `lgbm_return_ratio_setA.joblib`, `ridge_return_ratio_setA.joblib`, dan `train_metadata.json`.
- **Tabel Prediksi (`data/processed/`):**
  - `data/processed/predictions_default12m.parquet` (`148.369` baris populasi valid `default_12m`; kolom `LoanId`, `split`, konteks pinjaman, `default_12m`, `p_raw`, `p_calibrated`, `p_lr`, `PD_bondora`).
  - `data/processed/predictions_return_ratio.parquet` (`113.560` baris populasi `is_resolved == True`; kolom `LoanId`, `split`, konteks pinjaman, `return_ratio`, `pred_return_ratio`, `pred_baseline_group_median`).

### d. Usulan Pilihan (9 Tabel & Grafik) untuk Deck Laporan (PPT) — Tahap 4
- **3 Tabel:** `04_metrics_classification.csv`, `04_lift_by_decile.csv`, `05_metrics_regression.csv`.
- **6 Grafik:** `04_roc_curves.png`, `04_calibration_curve.png`, `04_lift_by_decile.png`, `04_shap_summary_bar.png`, `04_subgroup_performance.png`, `05_pred_vs_actual_and_lift.png`.

---

## 9. Daftar Tabel dan Grafik Tahap 5 (Prescriptive Analytics + Forecast)

> **Catatan tetap Realisasi 12 bulan (model ekonomi + default aktual):** Bukan laba tunai; default setelah bulan ke-12 dan sisa pokok tidak dihitung; 59,2% portofolio S3 bertenor 60 bulan.

### a. Tabel Ringkasan (`reports/tables/06_*.csv` & `06_run_meta.json` — 23 Berkas)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/tables/06_params.csv` | Daftar lengkap parameter bisnis dan asumsi eksplisit (`TICKET_EUR`, `BUDGET_EUR`, batas konsentrasi/risiko, grid *what-if*, parameter platform, dan *forecast*) beserta alasannya. |
| `reports/tables/06_interest_validation.csv` | Ringkasan validasi rumus cicilan anuitas `PMT * Amount` terhadap kolom `MonthlyPayment` aktual pada himpunan peluang uji (`n = 54.283`, korelasi `0,9619`, median selisih relatif absolut `7,86%`). |
| `reports/tables/06_interest_validation_segments.csv` | Rincian median `Interest`, bunga tahunan implisit dari `MonthlyPayment`, dan selisih relatif `PMT` menurut `Country` dan `Rating`. |
| `reports/tables/06_lgd_estimate.csv` | Estimasi kerugian saat gagal bayar $L = 1 - \text{mean}(\text{PrincipalPaymentsMade}/\text{Amount}) = 0,7311$ (CI bootstrap 95% `[0,7274, 0,7348]`) murni dari pinjaman `default_12m == 1` pada split latih dan validasi (`n = 28.585`). |
| `reports/tables/06_strategy_results.csv` | Evaluasi lengkap 8 strategi investor (`S0`, `S1`, `S2`, `S2c`, `S3`, `S4`, `S4c`, `S5`) pada himpunan peluang uji (`N = 5.000`, `250.000 EUR`): rata-rata $p$ *ex-ante*, tingkat default 12 bulan aktual + CI Wilson 95%, profit & ROI *ex-ante* vs Realisasi 12 bulan (model ekonomi + default aktual) + CI bootstrap 95%, konsentrasi, kepatuhan batasan, dan rekomendasi (`S3`, mengungguli pembanding patuh `S2c` dan `S4c`). |
| `reports/tables/06_s3_cluster_bootstrap.csv` | Bootstrap berpasangan tingkat peminjam (`UserName`, `200` iterasi, `RANDOM_SEED = 42`) vs bootstrap per pinjaman untuk ROI Realisasi 12 bulan dan default 12m aktual pada `S3`, `S2c`, `S4c` beserta selisih `S3 - S2c` (`+4,27%`, CI cluster 95% `[+2,64%; +5,66%]`) dan `S3 - S4c` (`+7,51%`, CI cluster 95% `[+5,89%; +8,79%]`); interval per pinjaman cenderung terlalu sempit karena default berkorelasi kuat dalam peminjam (`97,59%` pinjaman tumpang-tindih berakhir default bersama), namun kedua interval selisih cluster tidak mencakup nol sehingga kesimpulan `S3` unggul atas `S2c` dan `S4c` tetap berlaku. |
| `reports/tables/06_s3_constraints.csv` | Laporan nilai sisi kiri (LHS), batas bawah/atas, *slack* ternormalisasi, dan status mengikat (*binding*) untuk seluruh kendala optimasi portofolio `S3`. |
| `reports/tables/06_s3_composition.csv` | Komposisi portofolio terpilih `S3` dibanding himpunan peluang uji menurut `Country`, `Rating`, dan kelompok tenor beserta rata-rata $p$ dan default rate aktual per segmen. |
| `reports/tables/06_interest_robustness.csv` | Uji ketahanan (*robustness check*) evaluasi strategi `S1`–`S5` dan optimasi ulang `S3_ulang` menggunakan pendapatan bunga dari suku bunga implisit `MonthlyPayment`. |
| `reports/tables/06_whatif_scenarios.csv` | Hasil simulasi *what-if* (`17` skenario $\times$ `3` strategi `S3`, `S2`, `S5` ditambah `2` skenario pengali $p$ `x2,0` dan `x3,0` pada portofolio `S3` = `53` baris) lintas stres probabilitas `p`, pergeseran `L`, skala anggaran, batas negara, dan batas rata-rata risiko. |
| `reports/tables/06_breakeven_multiplier.csv` | Hasil pencarian bagi dua (*bisection search*) pengali impas (*breakeven multiplier* `m* = 1,9454`, `+94,54%`) yang membuat ROI *ex-ante* portofolio `S3` = `0`, beserta perbandingan terhadap skenario `x1,0`, `x2,0` (`ROI ex-ante -1,21%`), dan `x3,0` (`ROI ex-ante -18,64%`). |
| `reports/tables/06_whatif_tornado.csv` | Ringkasan rentang perubahan ROI *ex-ante* `S3` (poin persentase terhadap baseline `20,94%`) untuk nilai rendah dan tinggi tiap parameter sensitivitas. |
| `reports/tables/06_platform_threshold_policy.csv` | Evaluasi kebijakan persetujuan platform (`P1`, $p_{\text{calibrated}} \le \tau$ untuk $\tau \in \{0,10 \dots 0,50\}$): tingkat persetujuan, rata-rata $p$, default rate aktual, total pokok, profit/ROI *ex-ante* & Realisasi 12 bulan (model ekonomi + default aktual), dan rekomendasi ($\tau = 0,25$). |
| `reports/tables/06_platform_breakeven_pricing.csv` | Persentase pinjaman dengan suku bunga di bawah impas (`Interest < r*`) beserta median `Interest`, median `r*`, dan selisih poin persentase menurut `Semua`, `Country`, `Rating`, dan `Tenor` (`P2`). |
| `reports/tables/06_platform_breakeven_check.csv` | Verifikasi akurasi pendekatan linear bunga impas $r^*$ pada `1.000` pinjaman acak dengan menghitung ulang `interest12_frac` eksak pada $r^*$. |
| `reports/tables/06_platform_pricing_scenario.csv` | Simulasi ilustratif kenaikan suku bunga (dibatasi maks. `+10 pp` menuju $r^*$) pada pinjaman di bawah impas menurut kelas `Rating` (dengan peringatan pengabaian elastisitas permintaan & *adverse selection*). |
| `reports/tables/06_platform_level_sensitivity.csv` | Diagnostik sensitivitas level probabilitas untuk `P1` ($\tau$ optimal *ex-ante*) dan `P2` (persentase `Interest < r*` menurut `Country`: `Semua`, `EE`, `ES`, `FI`) pada tiga versi (`tanpa_skala`, `faktor_global` rata-rata pool `26,77%`, dan `faktor_per_negara`), berlabel `"diagnostik sensitivitas; faktor skala memakai data uji sehingga bukan evaluasi model"`. |
| `reports/tables/06_forecast_series.csv` | Deret waktu bulanan penuh (`2016-01` s.d. `2021-06`, `66` bulan) untuk jumlah pinjaman baru (`n_loans`), total nominal pinjaman baru (`amount_loans`), dan arus default baru (`amount_default`). |
| `reports/tables/06_forecast_backtest.csv` | Metrik evaluasi *rolling-origin* 1-langkah pada 6 bulan terakhir (`Jan–Jun 2021`): MAPE, sMAPE, cakupan interval 80% & 95%, dan model terpilih (`ETS` vs `seasonal_naive`). |
| `reports/tables/06_forecast_backtest_detail.csv` | Rincian nilai aktual vs prediksi titik dan interval 80%/95% per bulan pada periode *backtest* `Jan–Jun 2021` untuk kedua model. |
| `reports/tables/06_forecast_values.csv` | Prakiraan ke depan (`Jul–Okt 2021`, dengan `Agu–Okt 2021` sebagai 3 bulan penuh utama) beserta interval prediksi 80% dan 95% untuk ketiga deret bulanan. |
| `reports/tables/06_forecast_decision_link.csv` | Keterkaitan prakiraan jumlah pinjaman baru (`Agu–Okt 2021`) dengan proporsi kelolosan `S3` (`9,21%`) untuk mengestimasi jumlah pinjaman memenuhi syarat per bulan dan durasi *deployment* anggaran. |
| `reports/tables/06_run_meta.json` | Metadata eksekusi pipeline Tahap 5 (bukti angka *preflight*, validasi bunga, estimasi $L$, ringkasan solver `S3`, dan proporsi seleksi). |

### b. Grafik Visualisasi (`reports/figures/06_*.png` — 6 Grafik)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/06_strategy_comparison.png` | Grafik batang perbandingan ROI 12 bulan *ex-ante* vs Realisasi 12 bulan (model ekonomi + default aktual) (dengan CI bootstrap 95%) untuk kedelapan strategi investor (`S0`–`S5`, termasuk `S2c` & `S4c`). |
| `reports/figures/06_strategy_risk_return.png` | Diagram pencar (*risk-return*) tingkat default 12 bulan aktual vs ROI Realisasi 12 bulan (model ekonomi + default aktual) per strategi (strategi patuh batasan ditandai titik penuh). |
| `reports/figures/06_whatif_tornado.png` | Diagram tornado sensitivitas perubahan ROI *ex-ante* strategi `S3` terhadap variasi stres $p$, anggaran, batas negara, pergeseran $L$, dan batas rata-rata risiko. |
| `reports/figures/06_platform_threshold_curve.png` | Kurva profit 12 bulan (*ex-ante* & Realisasi 12 bulan (model ekonomi + default aktual), juta EUR) dan tingkat persetujuan (%) terhadap ambang risiko persetujuan platform $\tau$ beserta penanda rekomendasi $\tau = 0,25$. |
| `reports/figures/06_forecast_volume.png` | Grafik riwayat (`2016-01..2021-06`), 12 bulan terakhir aktual bersanding dengan prakiraan ETS (`Jul–Okt 2021`), dan pita interval prediksi 80%/95% untuk jumlah pinjaman baru dan total `Amount` bulanan (catatan: *ilustrasi metode; interval 80% mencakup 6 dari 6 bulan uji tahan sehingga cenderung lebar*). |
| `reports/figures/06_forecast_defaults.png` | Grafik riwayat, 12 bulan terakhir aktual bersanding dengan prakiraan ETS, dan pita interval prediksi 80%/95% untuk arus default baru bulanan (`Amount` menurut bulan `DefaultDate`; catatan: *ilustrasi metode; interval 80% mencakup 6 dari 6 bulan uji tahan sehingga cenderung lebar*). |

### c. Usulan Pilihan (10 Tabel & Grafik) untuk Deck Laporan (PPT) — Tahap 5
- **4 Tabel:** `06_params.csv`, `06_strategy_results.csv`, `06_s3_composition.csv`, `06_platform_threshold_policy.csv`.
- **6 Grafik:** `06_strategy_comparison.png`, `06_strategy_risk_return.png`, `06_whatif_tornado.png`, `06_platform_threshold_curve.png`, `06_forecast_volume.png`, `06_forecast_defaults.png`.

---

## 10. Daftar Ontologi, Query, Tabel, dan Grafik Tahap 6 (Semantic Analytics)

### a. Skema Ontologi & Berkas Query SPARQL (`ontology/`)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `ontology/bondora_ontology.ttl` | Skema ontologi OWL/RDFS dan taksonomi SKOS (`bo: <http://example.org/bondora#>`, `326` triple, `13` kelas OWL termasuk `3` kelas inferensi, `10` properti objek, `10` properti data termasuk `bo:defaultDate`, `28` individu tetap berlabel Bahasa Indonesia dan penanda `"definisi proyek"`). |
| `ontology/queries/q1.rq` | Query SPARQL Q1: jumlah pinjaman, rata-rata `p_calibrated`, dan rata-rata bunga per tingkat risiko rating (`skos:broader`, termasuk `bo:TingkatTidakTersedia` / `"Tidak tersedia"`) dan negara. |
| `ontology/queries/q2.rq` | Query SPARQL Q2: jumlah teramati, jumlah gagal bayar, dan tingkat gagal bayar 12 bulan (`bo:defaultWithin12m`) per negara dan era pencatatan. |
| `ontology/queries/q3.rq` | Query SPARQL Q3: daftar pinjaman bertingkat risiko rating rendah (`bo:TingkatRendah`: AA/A/B) namun berpita risiko model tinggi (`bo:PitaRisikoTinggi`, $p \ge 0,35$). |
| `ontology/queries/q4.rq` | Query SPARQL Q4: pasangan pinjaman berurutan (`bo:precededBy`) di mana pendahulu langsung berakhir `bo:Default`, dipisahkan menurut `bo:defaultDate` menjadi Varian (a) (`LoanDate > defaultDate` pendahulu — pinjaman ulang setelah default) dan Varian (b) (`LoanDate <= defaultDate` pendahulu — pinjaman tumpang-tindih). |
| `ontology/queries/q5.rq` | Query SPARQL Q5: perhitungan jumlah pendahulu pada rantai peminjaman ulang melalui *property path* transitif `bo:precededBy+` (dengan catatan artefak desain sampel). |
| `ontology/queries/q6.rq` | Query SPARQL Q6: komposisi portofolio rekomendasi `S3` (`bo:selectedBy bo:StrategiS3`, `n = 5.000`) menurut negara, tingkat risiko SKOS, kelas rating, dan pita tenor beserta rata-rata $p$ dan default 12 bulan aktual. |
| `ontology/queries/q7.rq` | Query SPARQL Q7: jumlah pinjaman bertipe `rdf:type bo:HighRiskRatedLoan` menurut negara (sebelum vs sesudah inferensi RDFS/OWL-RL). |
| `ontology/queries/q8.rq` | Query SPARQL Q8: komposisi 4 kelompok hasil pinjaman (`bo:OutcomeGroup`) menurut era pencatatan dan negara asal. |

### b. Tabel Ringkasan (`reports/tables/07_*.csv` — 17 Tabel)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/tables/07_sample_summary.csv` | Ringkasan sampel graf (`20.000` pinjaman, `7.709` peminjam) menurut sumber (`4.000` stratifikasi, `5.000` S3, `11.106` perluasan peminjam), negara, hasil, era, dan 28 strata beserta perbandingan proporsi sampel stratifikasi vs populasi. |
| `reports/tables/07_graph_stats.csv` | Statistik *knowledge graph*: total triple (`344.904`), triple ontologi (`326`), triple instans (`344.578`), verifikasi muat ulang (`344.904`), waktu bangun/muat, ukuran berkas (`0,741 MB`), jumlah individu per kelas, dan jumlah triple per predikat (termasuk `4.438` triple `bo:defaultDate`). |
| `reports/tables/07_query_timings.csv` | Ringkasan waktu eksekusi (`0,17` s.d. `14,68` detik), jumlah baris hasil, dan status verifikasi silang `pandas` (`LULUS`) untuk Q1–Q8. |
| `reports/tables/07_q1_rating_tier_country.csv` | Hasil Q1 beserta baris dan catatan jumlah pinjaman tanpa rating (`"Tidak tersedia"` / `TingkatTidakTersedia`: `n = 126` di `EE`, `0` di `ES/FI/SK`), kolom verifikasi silang `pandas`, dan selisih maksimum ($\le 1,42 \times 10^{-14}$). |
| `reports/tables/07_q2_default12m_country_era.csv` | Hasil Q2 (`n = 17.455` pinjaman teramati 12 bulan di graf) beserta kolom verifikasi silang `pandas`. |
| `reports/tables/07_q3_rating_vs_model_mismatch.csv` | Hasil Q3 (`10` pinjaman rating `A`/`B` dengan `p_calibrated >= 0,35`, di mana `4` pinjaman gagal bayar dalam 12 bulan). |
| `reports/tables/07_q4_repeat_borrower_after_default.csv` | Rincian pasangan transisi Q4 (`1.789` pasangan dengan `bo:defaultDate`: `90` pasangan Varian (a) `a_setelah_default` dan `1.699` pasangan Varian (b) `b_tumpang_tindih`). |
| `reports/tables/07_q4_variants_summary.csv` | Ringkasan Q4 lengkap dengan interval Wilson 95% menurut Varian (a) (`27/90 = 30,00%` `[21,51%; 40,13%]` berakhir `Default`) vs Varian (b) (`1.658/1.699 = 97,59%` `[96,74%; 98,22%]` berakhir `Default`), pembanding tingkat dasar, serta analisis `S3` (`15/67 = 22,39%` `[14,06%; 33,71%]` vs sisa S3 `947/4.933 = 19,20%` `[18,12%; 20,32%]` — tidak ada bukti selisih karena `n` kecil dan interval lebar). |
| `reports/tables/07_q5_repeat_chain_lengths.csv` | Hasil Q5 pada level pinjaman lanjutan (`12.291` baris `bo:precededBy+`) beserta jumlah pendahulu dan panjang rantai. |
| `reports/tables/07_q5_chain_distribution.csv` | Distribusi panjang rantai peminjaman ulang untuk seluruh `7.709` peminjam di graf (`1` s.d. `27` pinjaman, terverifikasi identik antara SPARQL dan `pandas`) disertai catatan bahwa panjang rantai dan persentase peminjam berulang (`41,68%`) adalah artefak desain sampel dan bukan statistik populasi. |
| `reports/tables/07_q5_s3_borrower_concentration.csv` | Distribusi jumlah pinjaman `S3` per peminjam (`5.000` pinjaman dari `4.242` peminjam unik: `1` s.d. `7` pinjaman/peminjam) beserta default 12m aktual dan interval Wilson 95% (`1`: `20,51%` `[19,22%; 21,86%]`; `2`: `17,95%` `[15,79%; 20,34%]`; `3`: `9,85%` `[6,81%; 14,04%]`; subtotal `>= 2`: `15,98%` `[14,15%; 17,99%]`), disertai catatan keterbatasan bahwa default berkorelasi kuat dalam satu peminjam namun hipotesis pembatasan pinjaman per peminjam di S3 tidak didukung data. |
| `reports/tables/07_q6_s3_composition_sparql.csv` | Hasil Q6 (`39` segmen portofolio `S3`, mereproduksi eksak 100% angka Tahap 5: EE `70,00%`, ES `21,62%`, FI `8,38%`; F+HR `22,94%`; Tenor 60 bln `59,20%`; default 12m aktual `19,24%`). |
| `reports/tables/07_q7_high_risk_inferred.csv` | Perbandingan jumlah individu `bo:HighRiskRatedLoan` per negara sebelum inferensi (`0`), sesudah inferensi OWL-RL (`6.960`), via *property path* `skos:broader+` (`6.960`), dan hitungan `pandas` (`6.960`). |
| `reports/tables/07_q8_outcome_era_country.csv` | Hasil Q8 komposisi 4 kelompok hasil menurut era dan negara beserta verifikasi silang `pandas`. |
| `reports/tables/07_inference_summary.csv` | Ringkasan penutupan deduktif `owlrl.DeductiveClosure` pada salinan graf (`344.904 -> 425.705` triple; `bo:DefaultedLoan = 4.460`, `bo:CensoredLoan = 10.024`, `bo:HighRiskRatedLoan = 6.960`). |
| `reports/tables/07_insight_action.csv` | Tabel *Insight & Action* (5 baris) dengan acuan baris 1 & 2 diambil langsung (`assert`) dari `03_default_rate_horizon_by_segment.csv`, `03_rating_vs_observed_12m.csv` (populasi `29,06%`), serta `04_level_bias.csv` & `04_subgroup_performance.csv` (rata-rata `p_calibrated` uji `32,96%` vs aktual `26,77%`: `EE 17,74%`, `FI 46,82%`, `ES 51,65%`), interval Wilson 95% dan kesimpulan tidak ada bukti selisih S3 pada baris 4, serta catatan keterbatasan korelasi default dalam peminjam dan konsentrasi S3 pada baris 5. |
| `reports/tables/07_example_subgraph_loans.csv` | Tabel pendamping subgraf contoh `"Peminjam A"` (`ES`) yang mencantumkan `LoanNumber` (`1571624`, `1752900`, `1933569`), tanggal pencairan, `defaultDate`, dan atribut ketiga pinjaman berurutan. |

### c. Grafik Visualisasi (`reports/figures/07_*.png` & HTML Interaktif)

| Nama File | Deskripsi Satu Baris |
| :--- | :--- |
| `reports/figures/07_ontology_schema.png` | Diagram skema ontologi kelas utama, kelas terdefinisi inferensi OWL-RL, properti objek (`bo:precededBy`, dll.), dan properti data (termasuk `bo:defaultDate`). |
| `reports/figures/07_kg_example_subgraph.png` | Subgraf contoh nyata berlabel generik `"Peminjam A"` (`ES`) dengan rantai 3 pinjaman (`Pinjaman 1 <- Pinjaman 2 <- Pinjaman 3`; `LoanNumber` disebut hanya di tabel pendamping `07_example_subgraph_loans.csv`), taksonomi SKOS, hasil `bo:Default`, dan seleksi `bo:StrategiS3`. |
| `reports/figures/07_taxonomy_rating_risk.png` | Pohon taksonomi SKOS `bo:RatingClass -> skos:broader -> bo:RiskTier` (*definisi proyek*) berdampingan dengan 4 konsep `bo:OutcomeGroup`, 3 `bo:RiskBand`, dan 2 `bo:Era`. |
| `reports/figures/07_triples_by_predicate.png` | Grafik batang horizontal distribusi jumlah triple per predikat pada *knowledge graph* (`344.904` triple, termasuk `bo:defaultDate`). |
| `reports/07_kg_interactive.html` | Visualisasi subgraf jaringan interaktif berbasis `pyvis` (`0,695 MB` $< 5\text{ MB}$) untuk 8 peminjam berulang dan 27 pinjaman. |

### d. Usulan Pilihan (6 Tabel & Grafik) untuk Deck Laporan (PPT) — Tahap 6 (3 Slide)
- **Slide 1 (Ontologi & Taksonomi Semantik):** `07_ontology_schema.png` + `07_taxonomy_rating_risk.png`.
- **Slide 2 (Struktur Knowledge Graph & Contoh Subgraf Rantai Pinjaman):** `07_kg_example_subgraph.png` + `07_graph_stats.csv` (atau `07_triples_by_predicate.png`).
- **Slide 3 (Hasil Query SPARQL, Inferensi OWL-RL, & Insight Relasional):** `07_insight_action.csv` + `07_q4_variants_summary.csv` (bersama `07_q5_s3_borrower_concentration.csv` dan `07_inference_summary.csv`).
