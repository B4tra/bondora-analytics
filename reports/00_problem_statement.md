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
- **Klasifikasi Default Berhorizon Tetap (`default_12m` & sensitivitas `default_24m`, tipe *nullable* `Int8` — Ditambahkan pada Tahap 3 untuk Mengurangi Bias Seleksi):**
  - `observable_12m` (`bool`, `148.783` baris `True`): bernilai `True` apabila `LoanDate + DateOffset(months=12) <= REPORT_DATE` (`2021-07-20`).
  - `default_12m` (`Int8` nullable, `148.369` baris non-NA dengan tingkat gagal bayar `29,06%`): bernilai `1` (`43.117` baris) apabila `DefaultDate` tercatat dan `DefaultDate <= LoanDate + 12 bulan`; bernilai `0` (`105.252` baris) apabila pinjaman telah teramati 12 bulan tanpa gagal bayar dalam 12 bulan pertama; bernilai `<NA>` (`30.866` baris) apabila `observable_12m == False` (`30.452` baris), termasuk `448` pinjaman `Repaid` *write-off* tanpa `DefaultDate` (`414` baris di dalam `observable_12m`), atau `DefaultDate < LoanDate` (`0` baris).
  - `observable_24m` (`98.027` baris `True`) dan `default_24m` (`97.878` baris non-NA; `1`: `44.584` baris, `0`: `53.294` baris, `<NA>`: `81.357` baris; tingkat gagal bayar `45,55%`) disediakan sebagai uji sensitivitas horizon 24 bulan.
- **Klasifikasi Default Umur Pinjaman (`default`, tipe *nullable* `Int8` — Subset *Resolved Lifetime*):** Memprediksi kejadian gagal bayar pinjaman (`default = 1` vs `default = 0`) pada subset pinjaman yang hasil akhirnya telah diketahui (`is_resolved == True`, sebanyak `113.560` baris dengan tingkat gagal bayar `63,28%`):
  - `default = 1` (`71.864` baris): apabila `DefaultDate` tidak kosong (`71.416` baris pada seluruh status) **ATAU** pinjaman berstatus `Repaid` tanpa `DefaultDate` namun memiliki penghapusan buku pokok `PrincipalWriteOffs > 0` (`448` baris, di mana `92,41%` pernah mengalami keterlambatan $\ge 61$ hari).
  - `default = 0` (`41.696` baris): apabila pinjaman berstatus `Repaid`, `DefaultDate` kosong, dan `PrincipalWriteOffs == 0` (termasuk `297` baris yang hanya memiliki `InterestAndPenaltyWriteOffs > 0`).
  - `default = <NA>` (`65.675` baris): pinjaman berjalan tanpa `DefaultDate` (`Status == 'Current'` sebanyak `57.611` baris dan `Status == 'Late'` sebanyak `8.064` baris) ditandai `is_resolved = False` (*right-censored*) dan memiliki nilai `default = <NA>`, sehingga dikeluarkan dari pelatihan/evaluasi model prediktif namun tetap disimpan di dataset bersih untuk analisis *Descriptive*, *Diagnostic*, dan *Semantic*.
- **Regresi (`return_ratio`):** Memprediksi rasio pengembalian aktual sampai tanggal laporan (`2021-07-20`):
  $$\text{return\_ratio} = \frac{\text{PrincipalPaymentsMade} + \text{InterestAndPenaltyPaymentsMade}}{\text{Amount}}$$
  dengan catatan bahwa `PrincipalRecovery` dan `InterestRecovery` telah termasuk di dalam `PrincipalPaymentsMade` dan `InterestAndPenaltyPaymentsMade`, nilai `Amount <= 0` atau `NaN` menghasilkan `NaN`, serta nilai ini merupakan batas bawah (*lower bound*) bagi pinjaman default yang masih dalam proses penagihan pada tanggal laporan.

### 3.4 Prescriptive Analytics
- `TODO: Merumuskan model optimasi seleksi/alokasi pinjaman untuk memaksimalkan ekspektasi imbal hasil atau meminimalkan risiko portofolio.`

### 3.5 Semantic Analytics
- `TODO: Merancang ontologi domain P2P lending dan membangun knowledge graph yang dapat di-query menggunakan SPARQL.`

## 4. Ruang Lingkup dan Batasan

- **Ruang Lingkup:** Dataset mencakup `179.235` kontrak pinjaman Bondora dengan tanggal pencairan (`LoanDate`) dari `2009-02-28` hingga `2021-07-19` dan tanggal *snapshot* laporan (`REPORT_DATE`) `2021-07-20`, meliputi 4 negara (`EE`, `FI`, `ES`, `SK`).
- **Batasan & Aturan Fitur (Gerbang 1, Gerbang 2, & Pembaruan Tahap 3–4):**
  - Fitur prediktif dibatasi secara ketat pada variabel yang sudah diketahui pada saat pinjaman dicairkan (*origination*) sebelum pembayaran pertama dan bukan merupakan keluaran model internal platform (`Rating`, `ProbabilityOfDefault`, `ExpectedLoss`, `LossGivenDefault`, `ExpectedReturn`, `ModelVersion`, `EL_V*`, `Rating_V*` disimpan sebagai kolom `benchmark`) maupun sinyal pasar investor (`Bids*`) atau `Interest` (`peran = pricing`, digunakan di Tahap 5).
  - Kolom `Gender` ditandai sebagai `atribut_sensitif` (`boleh_jadi_fitur = N`) dan hanya digunakan untuk evaluasi *fairness* pada *Diagnostic Analytics*.
  - Ambang kelayakan nilai kosong (*missing value* $> 50\%$) **dihitung pada populasi relevan** (misalnya per negara untuk skor biro kredit eksternal `CreditScore*` atau setelah penerapan aturan domain riwayat pinjaman sebelumnya sejak perubahan pencatatan Bondora tahun 2018).
  - Dari `130` kolom pada `data/processed/bondora_clean.parquet`, ditetapkan **`44` kolom fitur kandidat (`boleh_jadi_fitur = Y`)**, terdiri atas `40` fitur asli saat pengajuan dan `4` fitur rasio turunan (`loan_to_income_ratio`, `monthly_payment_to_income_ratio`, `total_liabilities_to_income_ratio`, `avg_previous_loan_amount`). Ketiga indikator `*_missing` (`MonthlyPayment_missing`, `PreviousRepaymentsBeforeLoan_missing`, `CreditScoreEsMicroL_missing`) ditetapkan memiliki `boleh_jadi_fitur = N` (`peran = proxy_era`) karena terbukti terkonsentrasi per era pada Tahap 2. Empat kolom kategorikal berkode `-1 (Not set)` (`UseOfLoan`, `MaritalStatus`, `EmploymentStatus`, `OccupationArea`) tetap berstatus `Y` dengan catatan wajib diuji ablasi pada Tahap 4 karena hampir seluruhnya berkode `"Not set"` sejak 2018.
  - **Pembagian Waktu & Set Fitur Tahap 4 (Predictive Analytics):**
    - Pembagian data utama berbasis waktu pencairan (`LoanDate`): **Latih** `<= 2018-06-30` (`59.459` baris klasifikasi; `54.103` baris regresi), **Validasi** `2018-07-01` s.d. `2019-06-30` (`34.627` baris klasifikasi; `24.782` baris regresi), dan **Uji** `>= 2019-07-01` (`54.283` baris klasifikasi; `34.675` baris regresi).
    - Aturan deteksi fitur **mati-era** (`share_dominan_validasi >= 0,95` dan `share_dominan_latih <= 0,80`) menyaring 8 fitur yang kolaps pada era baru (`UseOfLoan`, `MaritalStatus`, `EmploymentStatus`, `OccupationArea`, `IncomeFromPrincipalEmployer`, `DebtToIncome`, `FreeCash`, `CreditScoreEsMicroL`), menghasilkan **Set A (stabil, utama)** = `36` fitur, **Set B (lengkap)** = `44` fitur, dan **Set C (ablasi)** = `39` fitur (`Set A` + `Rating`, `ProbabilityOfDefault`, `Interest`).

## 5. Dataset yang Digunakan

- **Dataset Utama:** `data/raw/LoanData.csv` dari Bondora Public Reports (`142,66 MB`, `179.235` baris $\times$ `112` kolom mentah; tanggal laporan `2021-07-20`), yang telah dibersihkan dan disimpan ke `data/processed/bondora_clean.parquet` (`179.235` baris $\times$ `130` kolom) beserta metadata skema `data/processed/schema.json`.
- **Dataset Pendukung (Opsional):** `TODO: RepaymentsData atau sumber eksternal bila digunakan.`

## 6. Metrik Keberhasilan

- **Descriptive & Diagnostic:** `TODO: Kelengkapan visualisasi, uji statistik, dan ketajaman temuan analitis.`
- **Predictive — Klasifikasi:** `TODO: ROC-AUC, PR-AUC, F1-Score, Recall pada kelas default, dan interpretabilitas model (SHAP).`
  - *Acuan Pembanding & Realisasi Tahap 4 (`default_12m`, Data Uji Temporal `n = 54.283`, Default Rate `26,77%`):*
    - Pembanding platform: `ProbabilityOfDefault` Bondora mencapai ROC-AUC `0,6746` `[0,6699, 0,6800]` pada data uji (`0,6998` pada seluruh populasi teramati `n = 148.369`), KS `0,2718`, Brier `0,1868`, dan ECE `0,0597`; suku bunga `Interest` sebagai skor peringkat mencapai ROC-AUC `0,6851` `[0,6810, 0,6906]`.
    - Model utama **LGBM-A terkalibrasi isotonik (`Set A`, 36 fitur)** mencapai ROC-AUC uji **`0,7152`** `[0,7107, 0,7196]` (`+0,0406` di atas `ProbabilityOfDefault` dan `+0,0301` di atas `Interest`; dalam-negara: unggul di `EE` `+0,0504` vs PD / `+0,0765` vs `Interest` dan di `ES` `+0,0275` vs PD / `+0,0309` vs `Interest`, sementara di `FI` unggul atas `Interest` `+0,0446` namun tidak berbeda nyata dari PD `-0,0063` `[-0,0211, +0,0090]`), PR-AUC `0,4393` (`0,4468` sebelum kalibrasi), KS `0,3230`, Brier `0,1798` (membaik dari `0,1926` sebelum kalibrasi), ECE `0,0619` (membaik dari `0,1236`), serta *lift* Desil 1 sebesar `2,01x` (menangkap `20,10%` default pada 10% pinjaman risiko tertinggi). Metrik kalibrasi pada validasi bersifat *in-sample* dan metrik uji (*out-of-sample*) yang dipakai untuk pelaporan; level probabilitas uji cenderung terlalu tinggi (rata-rata prediksi `32,96%` vs aktual `26,77%`), sehingga urutan risiko lebih andal daripada level. Model linear `LR-A` mencapai ROC-AUC uji `0,6964`, `LGBM-B` (`44` fitur) mencapai `0,7206`, dan `LGBM-C` (ablasi `Set C`) mencapai `0,7078` (tidak menambah AUC; penurunan di uji `-0,0074` kecil dan penyebabnya belum dibuktikan). Eksperimen latih ulang satu kali dengan `NaN` diisi konstan tunggal menghasilkan ROC-AUC uji `0,7164` (`+0,0012` vs `LGBM-A`). Split acak `70/15/15` menghasilkan ROC-AUC uji `0,7879` (optimisme semu `+0,0728` poin AUC karena mengabaikan pergeseran rezim waktu).
- **Predictive — Regresi:** `TODO: RMSE, MAE, R-squared.`
  - *Acuan Pembanding & Realisasi Tahap 4 (`return_ratio` pada `is_resolved == True`, Data Uji Temporal `n = 34.675`, Mean Aktual `0,5708`):*
    - Pembanding naif: median global latih (`1,0366`) menghasilkan MAE uji `0,5323` ($R^2 = -1,0242$); median per `Country × Rating` menghasilkan MAE uji `0,4697` ($R^2 = -0,4868$, Spearman $\rho_s = 0,1925$); `ExpectedReturn` Bondora berkorelasi negatif pada kohort uji ($\rho_s = -0,1651$).
    - Model utama regresi **LGBM-Reg-Huber (`Set A`, 36 fitur)** mencapai MAE uji **`0,3902`**, RMSE `0,4545`, $R^2 = 0,0252$, dan korelasi peringkat Spearman **$\rho_s = 0,2881$** (model L2 mencapai MAE `0,3934`, $\rho_s = 0,2921$; `Ridge-A` mencapai MAE `0,4159`, $\rho_s = 0,2301$), dengan rata-rata `return_ratio` aktual Desil 1 tertinggi sebesar `0,9014` (`1,58x` rata-rata populasi) dibanding Desil 10 terendah `0,4168` (selisih `+0,4846`). Rata-rata prediksi regresi pada uji adalah `0,6746` vs aktual `0,5708` (level cenderung terlalu tinggi; gunakan sebagai urutan).
- **Prescriptive:** `TODO: Perbandingan performa portofolio hasil optimasi terhadap baseline strategi naif.`
- **Semantic:** `TODO: Validitas skema ontologi RDF/OWL dan keberhasilan eksekusi query kompetensi SPARQL.`
- **Deliverables Akhir:** `TODO: 7 Notebook Jupyter terverifikasi, Deck Laporan (PPT), dan Dashboard Streamlit interaktif.`
