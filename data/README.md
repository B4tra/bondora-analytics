# Panduan Data — Proyek Bondora Analytics

Dokumen ini menjelaskan sumber data, aturan penyimpanan, dan metadata unduhan untuk proyek **Bondora Analytics**.

## 1. Sumber Data

- **Sumber:** Halaman *Public Reports* Bondora
- **URL Sumber:** `TODO: isi URL sumber`

## 2. File yang Dibutuhkan

1. **LoanData** *(Wajib)* — File utama data pinjaman (`LoanData.csv`, sesuaikan di `src/config.py` apabila nama file unduhan berbeda).
2. **RepaymentsData** *(Opsional)* — File riwayat pembayaran pinjaman (`RepaymentsData.csv`, jika digunakan pada tahap analisis lanjutan).

## 3. Lokasi Penyimpanan

Simpan seluruh file mentah yang diunduh secara manual ke dalam folder:

```text
data/raw/
```

Sementara itu:
- `data/interim/` digunakan untuk hasil pemrosesan antara (intermediate).
- `data/processed/` digunakan untuk dataset bersih yang siap dimodelkan.
- `data/external/` digunakan untuk data pendukung eksternal bila ada.

## 4. Peringatan Penting (Jangan Di-commit!)

> **PERINGATAN:** Seluruh file data di dalam `data/raw/`, `data/interim/`, `data/processed/`, dan `data/external/` **TIDAK BOLEH di-commit ke Git** karena ukurannya besar dan untuk menjaga kebersihan repositori. File `.gitignore` sudah dikonfigurasi untuk mengabaikan isi folder `data/*` kecuali file `.gitkeep`.

## 5. Tabel Metadata Unduhan

Berikut adalah metadata dataset yang telah diverifikasi pada `data/raw/`:

| Tanggal Unduh | Nama File | Ukuran File | Jumlah Baris | Jumlah Kolom | Tanggal Laporan (`REPORT_DATE`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2026-10-17` | `LoanData.csv` | `142,66 MB` (`149.591.227` byte) | `179.235` baris | `112` kolom | `2021-07-20` |
| `TODO: YYYY-MM-DD` | `TODO: RepaymentsData.csv (opsional)` | `TODO: ... MB` | `TODO: ... baris` | `TODO: ... kolom` | `TODO` |

*(Catatan: Hasil pembersihan data Tahap 1 disimpan ke `data/processed/bondora_clean.parquet` berukuran `34,80 MB` dengan `179.235` baris $\times$ `126` kolom beserta `data/processed/schema.json`).*
