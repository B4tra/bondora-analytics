# 00 — Problem Statement: Bondora Peer-to-Peer Lending Analytics

> Dokumen ini adalah template rumusan masalah dan kerangka analitik proyek Bondora Analytics. Silakan lengkapi bagian bertanda `TODO` sebelum memulai eksplorasi mendalam pada Tahap 1.

## 1. Latar Belakang

- `TODO: Jelaskan konteks bisnis pinjaman Peer-to-Peer (P2P) pada platform Bondora.`
- `TODO: Jelaskan tantangan risiko kredit (gagal bayar / default), asimetri informasi, dan pengembalian investasi (return) bagi investor di platform P2P lending.`
- `TODO: Jelaskan urgensi penggunaan pendekatan analitik menyeluruh (Descriptive hingga Semantic Analytics) pada dataset Bondora.`

## 2. Rumusan Masalah

1. `TODO: Rumusan masalah terkait karakteristik portofolio pinjaman dan tren historis (Descriptive).`
2. `TODO: Rumusan masalah terkait faktor-faktor penyebab terjadinya gagal bayar / default (Diagnostic).`
3. `TODO: Rumusan masalah terkait prediksi probabilitas gagal bayar dan estimasi nilai finansial pinjaman (Predictive — Klasifikasi & Regresi).`
4. `TODO: Rumusan masalah terkait alokasi portofolio optimal dengan batasan risiko tertentu (Prescriptive).`
5. `TODO: Rumusan masalah terkait representasi pengetahuan domain pinjaman P2P menggunakan ontologi dan knowledge graph (Semantic).`

## 3. Tujuan per Jenis Analytics

### 3.1 Descriptive Analytics
- `TODO: Menentukan metrik ringkasan distribusi pinjaman, profil peminjam, tingkat bunga, dan rasio default historis.`

### 3.2 Diagnostic Analytics
- `TODO: Mengidentifikasi faktor-faktor signifikan dan hubungan kausal/korelasional yang membedakan pinjaman lancar vs gagal bayar.`

### 3.3 Predictive Analytics
- **Klasifikasi Default (`default`):** Memprediksi kejadian gagal bayar pinjaman (`default = 1` vs `default = 0`) pada subset pinjaman yang hasil akhirnya telah diketahui (`is_resolved == True`, sebanyak `113.560` baris dengan tingkat gagal bayar `63,28%`):
  - `default = 1` (`71.864` baris): apabila `DefaultDate` tidak kosong (`71.416` baris pada seluruh status) **ATAU** pinjaman berstatus `Repaid` tanpa `DefaultDate` namun memiliki penghapusan buku pokok `PrincipalWriteOffs > 0` (`448` baris, di mana `92,41%` pernah mengalami keterlambatan $\ge 61$ hari).
  - `default = 0` (`41.696` baris): apabila pinjaman berstatus `Repaid`, `DefaultDate` kosong, dan `PrincipalWriteOffs == 0` (termasuk `297` baris yang hanya memiliki `InterestAndPenaltyWriteOffs > 0`).
  - Pinjaman berjalan tanpa `DefaultDate` (`Status == 'Current'` sebanyak `57.611` baris dan `Status == 'Late'` sebanyak `8.064` baris; total `65.675` baris) ditandai `is_resolved = False` (*right-censored*) dan dikeluarkan dari pelatihan/evaluasi model prediktif, namun tetap disimpan di dataset bersih untuk analisis *Descriptive*, *Diagnostic*, dan *Semantic*.
- **Regresi (`return_ratio`):** Memprediksi rasio pengembalian aktual sampai tanggal laporan (`2021-07-20`):
  $$\text{return\_ratio} = \frac{\text{PrincipalPaymentsMade} + \text{InterestAndPenaltyPaymentsMade}}{\text{Amount}}$$
  dengan catatan bahwa `PrincipalRecovery` dan `InterestRecovery` telah termasuk di dalam `PrincipalPaymentsMade` dan `InterestAndPenaltyPaymentsMade`, nilai `Amount <= 0` atau `NaN` menghasilkan `NaN`, serta nilai ini merupakan batas bawah (*lower bound*) bagi pinjaman default yang masih dalam proses penagihan pada tanggal laporan.

### 3.4 Prescriptive Analytics
- `TODO: Merumuskan model optimasi seleksi/alokasi pinjaman untuk memaksimalkan ekspektasi imbal hasil atau meminimalkan risiko portofolio.`

### 3.5 Semantic Analytics
- `TODO: Merancang ontologi domain P2P lending dan membangun knowledge graph yang dapat di-query menggunakan SPARQL.`

## 4. Ruang Lingkup dan Batasan

- **Ruang Lingkup:** Dataset mencakup `179.235` kontrak pinjaman Bondora dengan tanggal pencairan (`LoanDate`) dari `2009-02-28` hingga `2021-07-19` dan tanggal *snapshot* laporan (`REPORT_DATE`) `2021-07-20`, meliputi 4 negara (`EE`, `FI`, `ES`, `SK`).
- **Batasan & Aturan Fitur (Gerbang 1 & Gerbang 2):**
  - Fitur prediktif dibatasi secara ketat pada variabel yang sudah diketahui pada saat pinjaman dicairkan (*origination*) sebelum pembayaran pertama dan bukan merupakan keluaran model internal platform (`Rating`, `ProbabilityOfDefault`, `ExpectedLoss`, `LossGivenDefault`, `ExpectedReturn`, `ModelVersion`, `EL_V*`, `Rating_V*` disimpan sebagai kolom `benchmark`) maupun sinyal pasar investor (`Bids*`) atau `Interest` (`peran = pricing`, digunakan di Tahap 5).
  - Kolom `Gender` ditandai sebagai `atribut_sensitif` (`boleh_jadi_fitur = N`) dan hanya digunakan untuk evaluasi *fairness* pada *Diagnostic Analytics*.
  - Ambang kelayakan nilai kosong (*missing value* $> 50\%$) **dihitung pada populasi relevan** (misalnya per negara untuk skor biro kredit eksternal `CreditScore*` atau setelah penerapan aturan domain riwayat pinjaman sebelumnya sejak perubahan pencatatan Bondora tahun 2018).
  - Dari `126` kolom pada `data/processed/bondora_clean.parquet`, ditetapkan `47` kolom fitur siap pakai (`boleh_jadi_fitur = Y`), terdiri atas `40` fitur asli saat pengajuan, `3` indikator `*_missing` (`MonthlyPayment_missing`, `PreviousRepaymentsBeforeLoan_missing`, `CreditScoreEsMicroL_missing`), dan `4` fitur rasio turunan (`loan_to_income_ratio`, `monthly_payment_to_income_ratio`, `total_liabilities_to_income_ratio`, `avg_previous_loan_amount`).

## 5. Dataset yang Digunakan

- **Dataset Utama:** `data/raw/LoanData.csv` dari Bondora Public Reports (`142,66 MB`, `179.235` baris $\times$ `112` kolom mentah; tanggal laporan `2021-07-20`), yang telah dibersihkan dan disimpan ke `data/processed/bondora_clean.parquet` (`179.235` baris $\times$ `126` kolom) beserta metadata skema `data/processed/schema.json`.
- **Dataset Pendukung (Opsional):** `TODO: RepaymentsData atau sumber eksternal bila digunakan.`

## 6. Metrik Keberhasilan

- **Descriptive & Diagnostic:** `TODO: Kelengkapan visualisasi, uji statistik, dan ketajaman temuan analitis.`
- **Predictive — Klasifikasi:** `TODO: ROC-AUC, PR-AUC, F1-Score, Recall pada kelas default, dan interpretabilitas model (SHAP).`
- **Predictive — Regresi:** `TODO: RMSE, MAE, R-squared.`
- **Prescriptive:** `TODO: Perbandingan performa portofolio hasil optimasi terhadap baseline strategi naif.`
- **Semantic:** `TODO: Validitas skema ontologi RDF/OWL dan keberhasilan eksekusi query kompetensi SPARQL.`
- **Deliverables Akhir:** `TODO: 7 Notebook Jupyter terverifikasi, Laporan Word (.docx), dan Dashboard Streamlit interaktif.`
