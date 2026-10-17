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
- **Klasifikasi Default:** `TODO: Membangun model klasifikasi untuk memprediksi status gagal bayar (default vs non-default).`
- **Regresi:** `TODO: Membangun model regresi untuk memprediksi target numerik (misalnya return, loss, atau interest rate — ditentukan di Tahap 1).`

### 3.4 Prescriptive Analytics
- `TODO: Merumuskan model optimasi seleksi/alokasi pinjaman untuk memaksimalkan ekspektasi imbal hasil atau meminimalkan risiko portofolio.`

### 3.5 Semantic Analytics
- `TODO: Merancang ontologi domain P2P lending dan membangun knowledge graph yang dapat di-query menggunakan SPARQL.`

## 4. Ruang Lingkup dan Batasan

- **Ruang Lingkup:** `TODO: Jelaskan cakupan data Bondora yang dianalisis (periode waktu, negara, atau segmen pinjaman).`
- **Batasan:** `TODO: Jelaskan batasan data, asumsi pemodelan, dan keterbatasan metodologi.`

## 5. Dataset yang Digunakan

- **Dataset Utama:** `TODO: LoanData dari Bondora Public Reports (isi rincian versi/tanggal unduh sesuai data/README.md).`
- **Dataset Pendukung (Opsional):** `TODO: RepaymentsData atau sumber eksternal bila digunakan.`

## 6. Metrik Keberhasilan

- **Descriptive & Diagnostic:** `TODO: Kelengkapan visualisasi, uji statistik, dan ketajaman temuan analitis.`
- **Predictive — Klasifikasi:** `TODO: ROC-AUC, PR-AUC, F1-Score, Recall pada kelas default, dan interpretabilitas model (SHAP).`
- **Predictive — Regresi:** `TODO: RMSE, MAE, R-squared.`
- **Prescriptive:** `TODO: Perbandingan performa portofolio hasil optimasi terhadap baseline strategi naif.`
- **Semantic:** `TODO: Validitas skema ontologi RDF/OWL dan keberhasilan eksekusi query kompetensi SPARQL.`
- **Deliverables Akhir:** `TODO: 7 Notebook Jupyter terverifikasi, Laporan Word (.docx), dan Dashboard Streamlit interaktif.`
