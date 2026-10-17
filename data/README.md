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

Silakan lengkapi tabel berikut setelah Anda mengunduh dataset secara manual ke `data/raw/`:

| Tanggal Unduh | Nama File | Ukuran File | Jumlah Baris |
| :--- | :--- | :--- | :--- |
| `TODO: YYYY-MM-DD` | `TODO: LoanData.csv` | `TODO: ... MB` | `TODO: ... baris` |
| `TODO: YYYY-MM-DD` | `TODO: RepaymentsData.csv (opsional)` | `TODO: ... MB` | `TODO: ... baris` |

*(Catatan: Eksplorasi detail mengenai struktur dan isi kolom dataset akan dilakukan pada Tahap 1).*
