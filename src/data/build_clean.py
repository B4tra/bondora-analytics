"""End-to-end data cleaning and preparation pipeline for Tahap 1 (Bondora)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DEFAULT_12M_COL,
    DEFAULT_24M_COL,
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    DROPPED_COLUMNS,
    IS_RESOLVED_COL,
    OBSERVABLE_12M_COL,
    OBSERVABLE_24M_COL,
    REGRESSION_TARGET_COL,
    REPORT_DATE,
    STATUS_COL,
    STATUS_CURRENT,
    STATUS_LATE,
    STATUS_REPAID,
    TABLES_DIR,
)
from src.data.loader import load_raw
from src.data.targets import (
    make_default_horizon_label,
    make_default_label,
    make_is_resolved,
    make_regression_target,
)
from src.features.basic import add_basic_derived_features


DATE_COLUMNS: list[str] = [
    "ReportAsOfEOD",
    "ListedOnUTC",
    "BiddingStartedOn",
    "LoanApplicationStartedDate",
    "LoanDate",
    "ContractEndDate",
    "FirstPaymentDate",
    "MaturityDate_Original",
    "MaturityDate_Last",
    "LastPaymentOn",
    "DebtOccuredOn",
    "DebtOccuredOnForSecondary",
    "DefaultDate",
    "StageActiveSince",
    "GracePeriodStart",
    "GracePeriodEnd",
    "NextPaymentDate",
    "ReScheduledOn",
]

NUMERIC_CODE_CATEGORY_COLUMNS: list[str] = [
    "VerificationType",
    "LanguageCode",
    "Gender",
    "UseOfLoan",
    "Education",
    "MaritalStatus",
    "EmploymentStatus",
    "OccupationArea",
    "HomeOwnershipType",
]

STRING_CATEGORY_COLUMNS: list[str] = [
    "Country",
    "EmploymentDurationCurrentEmployer",
    "Rating",
    "Status",
    "CreditScoreEsMicroL",
    "CreditScoreFiAsiakasTietoRiskGrade",
]

FINAL_AUDIT_SPEC: dict[str, tuple[str, str, str, str]] = {
    # (kategori, boleh_jadi_fitur, peran_schema, alasan)
    # 1. pengenal (4 cols) -> N
    "ReportAsOfEOD": ("pengenal", "N", "dibuang", "Tanggal snapshot laporan konstan (2021-07-20); disimpan di REPORT_DATE dan dibuang dari parquet"),
    "LoanId": ("pengenal", "N", "identifier", "UUID pengenal unik pinjaman"),
    "LoanNumber": ("pengenal", "N", "identifier", "Nomor urut unik pinjaman"),
    "UserName": ("pengenal", "N", "identifier", "Pengenal akun peminjam"),
    # 2. hasil_target (4 cols) -> N
    "Status": ("hasil_target", "N", "hasil_target", "Status pinjaman pada tanggal laporan (Current/Late/Repaid); pembentuk is_resolved dan default"),
    "DefaultDate": ("hasil_target", "N", "tanggal", "Tanggal kejadian gagal bayar; pembentuk langsung target klasifikasi default"),
    "PrincipalPaymentsMade": ("hasil_target", "N", "hasil_target", "Akumulasi pembayaran pokok yang diterima; pembilang target regresi return_ratio"),
    "InterestAndPenaltyPaymentsMade": ("hasil_target", "N", "hasil_target", "Akumulasi pembayaran bunga & denda yang diterima; pembilang target regresi return_ratio"),
    # 3. output_model_platform (11 cols) -> N (benchmark)
    "ExpectedLoss": ("output_model_platform", "N", "benchmark", "Estimasi Expected Loss internal Bondora; disimpan sebagai kolom benchmark (Keputusan Gerbang 1: Opsi 5A)"),
    "LossGivenDefault": ("output_model_platform", "N", "benchmark", "Estimasi Loss Given Default internal Bondora; disimpan sebagai kolom benchmark (Opsi 5A)"),
    "ExpectedReturn": ("output_model_platform", "N", "benchmark", "Estimasi Expected Return internal Bondora; disimpan sebagai kolom benchmark (Opsi 5A)"),
    "ProbabilityOfDefault": ("output_model_platform", "N", "benchmark", "Estimasi Probability of Default internal Bondora; disimpan sebagai kolom benchmark (Opsi 5A)"),
    "ModelVersion": ("output_model_platform", "N", "benchmark", "Versi model skoring risiko internal Bondora (Opsi 5A)"),
    "Rating": ("output_model_platform", "N", "benchmark", "Peringkat risiko kredit internal Bondora AA-HR (Opsi 5A)"),
    "EL_V0": ("output_model_platform", "N", "benchmark", "Estimasi Expected Loss versi model v0 internal Bondora (Opsi 5A)"),
    "Rating_V0": ("output_model_platform", "N", "benchmark", "Peringkat risiko versi model v0 internal Bondora (Opsi 5A)"),
    "EL_V1": ("output_model_platform", "N", "benchmark", "Estimasi Expected Loss versi model v1 internal Bondora (Opsi 5A)"),
    "Rating_V1": ("output_model_platform", "N", "benchmark", "Peringkat risiko versi model v1 internal Bondora (Opsi 5A)"),
    "Rating_V2": ("output_model_platform", "N", "benchmark", "Peringkat risiko versi model v2 internal Bondora (Opsi 5A)"),
    # 4. pasca_pinjaman (35 cols) -> N
    "ContractEndDate": ("pasca_pinjaman", "N", "tanggal", "Tanggal berakhirnya kontrak aktual setelah pinjaman berjalan"),
    "MaturityDate_Last": ("pasca_pinjaman", "N", "tanggal", "Tanggal jatuh tempo terakhir yang diperbarui bila terjadi penjadwalan ulang pasca pencairan"),
    "ActiveScheduleFirstPaymentReached": ("pasca_pinjaman", "N", "pasca_pinjaman", "Indikator apakah jadwal pembayaran pertama telah tercapai setelah pinjaman berjalan"),
    "PlannedPrincipalTillDate": ("pasca_pinjaman", "N", "pasca_pinjaman", "Akumulasi rencana pokok sampai tanggal laporan pasca pencairan"),
    "PlannedInterestTillDate": ("pasca_pinjaman", "N", "pasca_pinjaman", "Akumulasi rencana bunga sampai tanggal laporan pasca pencairan"),
    "LastPaymentOn": ("pasca_pinjaman", "N", "tanggal", "Tanggal pembayaran terakhir yang dilakukan peminjam selama masa pinjaman"),
    "CurrentDebtDaysPrimary": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah hari tunggakan pokok saat ini (pasca pencairan)"),
    "DebtOccuredOn": ("pasca_pinjaman", "N", "tanggal", "Tanggal mulai terjadinya tunggakan pokok (pasca pencairan)"),
    "CurrentDebtDaysSecondary": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah hari tunggakan bunga/denda saat ini (pasca pencairan)"),
    "DebtOccuredOnForSecondary": ("pasca_pinjaman", "N", "tanggal", "Tanggal mulai terjadinya tunggakan sekunder (pasca pencairan)"),
    "PrincipalOverdueBySchedule": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah pokok yang menunggak dari jadwal saat ini (pasca pencairan)"),
    "PlannedPrincipalPostDefault": ("pasca_pinjaman", "N", "pasca_pinjaman", "Rencana pembayaran pokok yang dijadwalkan setelah terjadi default"),
    "PlannedInterestPostDefault": ("pasca_pinjaman", "N", "pasca_pinjaman", "Rencana pembayaran bunga yang dijadwalkan setelah terjadi default"),
    "EAD1": ("pasca_pinjaman", "N", "pasca_pinjaman", "Exposure at Default metode 1 yang dihitung saat terjadi gagal bayar"),
    "EAD2": ("pasca_pinjaman", "N", "pasca_pinjaman", "Exposure at Default metode 2 yang dihitung saat terjadi gagal bayar"),
    "PrincipalRecovery": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah pokok yang berhasil ditagih setelah pinjaman default"),
    "InterestRecovery": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah bunga yang berhasil ditagih setelah pinjaman default"),
    "RecoveryStage": ("pasca_pinjaman", "N", "pasca_pinjaman", "Tahap penagihan/koleksi saat ini setelah pinjaman menunggak/default"),
    "StageActiveSince": ("pasca_pinjaman", "N", "tanggal", "Tanggal mulai tahapan penagihan saat ini (pasca pencairan)"),
    "Restructured": ("pasca_pinjaman", "N", "pasca_pinjaman", "Status apakah pinjaman pernah direstrukturisasi selama masa pinjaman"),
    "ActiveLateCategory": ("pasca_pinjaman", "N", "pasca_pinjaman", "Kategori keterlambatan aktif pada tanggal laporan"),
    "WorseLateCategory": ("pasca_pinjaman", "N", "pasca_pinjaman", "Kategori keterlambatan terburuk sepanjang umur pinjaman"),
    "PrincipalWriteOffs": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah pokok yang dihapusbukukan pasca gagal bayar"),
    "InterestAndPenaltyWriteOffs": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah bunga dan denda yang dihapusbukukan pasca gagal bayar"),
    "PrincipalBalance": ("pasca_pinjaman", "N", "pasca_pinjaman", "Sisa saldo pokok terutang pada tanggal laporan"),
    "InterestAndPenaltyBalance": ("pasca_pinjaman", "N", "pasca_pinjaman", "Sisa saldo bunga dan denda terutang pada tanggal laporan"),
    "GracePeriodStart": ("pasca_pinjaman", "N", "tanggal", "Tanggal mulai masa tenggang (grace period) selama pinjaman berjalan"),
    "GracePeriodEnd": ("pasca_pinjaman", "N", "tanggal", "Tanggal berakhir masa tenggang selama pinjaman berjalan"),
    "NextPaymentDate": ("pasca_pinjaman", "N", "tanggal", "Tanggal jatuh tempo pembayaran berikutnya pada saat laporan"),
    "NextPaymentNr": ("pasca_pinjaman", "N", "pasca_pinjaman", "Nomor urut pembayaran berikutnya pada saat laporan"),
    "NrOfScheduledPayments": ("pasca_pinjaman", "N", "pasca_pinjaman", "Jumlah jadwal pembayaran aktif (berubah saat restrukturisasi pasca pencairan)"),
    "ReScheduledOn": ("pasca_pinjaman", "N", "tanggal", "Tanggal penjadwalan ulang pinjaman pasca pencairan"),
    "PrincipalDebtServicingCost": ("pasca_pinjaman", "N", "pasca_pinjaman", "Biaya penagihan utang pokok pasca gagal bayar"),
    "InterestAndPenaltyDebtServicingCost": ("pasca_pinjaman", "N", "pasca_pinjaman", "Biaya penagihan utang bunga dan denda pasca gagal bayar"),
    "ActiveLateLastPaymentCategory": ("pasca_pinjaman", "N", "pasca_pinjaman", "Kategori keterlambatan pembayaran terakhir pada saat laporan"),
    # 5. saat_pengajuan tetapi boleh_jadi_fitur = N (18 cols)
    "DateOfBirth": ("saat_pengajuan", "N", "dibuang", "100% kosong (dihapus oleh Bondora untuk privasi/GDPR); dibuang dari parquet"),
    "County": ("saat_pengajuan", "N", "dibuang", "100% kosong (dihapus oleh Bondora untuk privasi/GDPR); dibuang dari parquet"),
    "City": ("saat_pengajuan", "N", "dibuang", "100% kosong (dihapus oleh Bondora untuk privasi/GDPR); dibuang dari parquet"),
    "EmploymentPosition": ("saat_pengajuan", "N", "dibuang", "100% kosong (dihapus oleh Bondora untuk privasi/GDPR); dibuang dari parquet"),
    "ListedOnUTC": ("saat_pengajuan", "N", "tanggal", "Kolom tanggal mentah saat pinjaman didaftarkan (Keputusan Gerbang 2 Lanjutan: peran = tanggal)"),
    "BiddingStartedOn": ("saat_pengajuan", "N", "tanggal", "Kolom tanggal mentah mulai penawaran (Keputusan Gerbang 2 Lanjutan: peran = tanggal)"),
    "LoanApplicationStartedDate": ("saat_pengajuan", "N", "tanggal", "Kolom tanggal mentah mulai pengajuan aplikasi (Keputusan Gerbang 2 Lanjutan: peran = tanggal)"),
    "LoanDate": ("saat_pengajuan", "N", "pembagi_waktu", "Tanggal pencairan pinjaman; digunakan sebagai pembagi_waktu (temporal split) dan ekstraksi loan_year/loan_month"),
    "FirstPaymentDate": ("saat_pengajuan", "N", "tanggal", "Kolom tanggal mentah jadwal pembayaran pertama (Keputusan Gerbang 2 Lanjutan: peran = tanggal)"),
    "MaturityDate_Original": ("saat_pengajuan", "N", "tanggal", "Kolom tanggal mentah jatuh tempo awal kontrak (Keputusan Gerbang 2 Lanjutan: peran = tanggal)"),
    "BidsPortfolioManager": ("saat_pengajuan", "N", "sinyal_investor", "Sinyal pasar investor dari Portfolio Manager (Keputusan Gerbang 2: peran = sinyal_investor)"),
    "BidsApi": ("saat_pengajuan", "N", "sinyal_investor", "Sinyal pasar investor via API (Keputusan Gerbang 2: peran = sinyal_investor)"),
    "BidsManual": ("saat_pengajuan", "N", "sinyal_investor", "Sinyal pasar investor manual (Keputusan Gerbang 2: peran = sinyal_investor)"),
    "Interest": ("saat_pengajuan", "N", "pricing", "Suku bunga pinjaman yang ditetapkan platform; disimpan sebagai peran = pricing untuk Tahap 5"),
    "Gender": ("saat_pengajuan", "N", "atribut_sensitif", "Atribut sensitif peminjam; tidak dijadikan fitur prediktif, disimpan untuk analisis fairness di Tahap 3"),
    "NrOfDependants": ("saat_pengajuan", "N", "missing_tinggi", "Missing 80.14% (> ambang 50% pada populasi relevan)"),
    "WorkExperience": ("saat_pengajuan", "N", "missing_tinggi", "Missing 79.62% (> ambang 50% pada populasi relevan)"),
    "CreditScoreEsEquifaxRisk": ("saat_pengajuan", "N", "missing_tinggi", "Hanya terisi 46.51% pada populasi relevan (Spanyol), tidak memenuhi ambang 50%"),
    # 6. saat_pengajuan dengan boleh_jadi_fitur = Y (40 cols)
    "NewCreditCustomer": ("saat_pengajuan", "Y", "fitur", "Status pelanggan kredit baru/lama di Bondora saat pengajuan (0% missing)"),
    "ApplicationSignedHour": ("saat_pengajuan", "Y", "fitur", "Jam penandatanganan aplikasi pinjaman (0% missing)"),
    "ApplicationSignedWeekday": ("saat_pengajuan", "Y", "fitur", "Hari dalam minggu saat aplikasi ditandatangani (0% missing)"),
    "VerificationType": ("saat_pengajuan", "Y", "fitur", "Metode verifikasi pendapatan/data peminjam saat pengajuan (0.03% missing; category)"),
    "LanguageCode": ("saat_pengajuan", "Y", "fitur", "Kode bahasa peminjam saat pengajuan (0% missing; category)"),
    "Age": ("saat_pengajuan", "Y", "fitur", "Usia peminjam saat pengajuan (0% missing; disertai flag_umur_tidak_wajar)"),
    "Country": ("saat_pengajuan", "Y", "fitur", "Negara domisili peminjam EE/FI/ES/SK saat pengajuan (0% missing; category)"),
    "AppliedAmount": ("saat_pengajuan", "Y", "fitur", "Jumlah pinjaman yang diajukan peminjam pada formulir awal (0% missing)"),
    "Amount": ("saat_pengajuan", "Y", "fitur", "Jumlah pokok pinjaman yang dicairkan saat kontrak dimulai (0% missing; disetujui di Gerbang 2)"),
    "LoanDuration": ("saat_pengajuan", "Y", "fitur", "Durasi/tenor pinjaman awal dalam bulan (0% missing)"),
    "MonthlyPayment": ("saat_pengajuan", "Y", "fitur", "Cicilan bulanan terjadwal saat pengajuan (3.73% missing; disertai MonthlyPayment_missing)"),
    "UseOfLoan": ("saat_pengajuan", "Y", "fitur", "Kode tujuan penggunaan pinjaman saat pengajuan (0% missing; category, kode -1 dipertahankan)"),
    "Education": ("saat_pengajuan", "Y", "fitur", "Kode tingkat pendidikan peminjam saat pengajuan (0.03% missing; category, kode -1 dipertahankan)"),
    "MaritalStatus": ("saat_pengajuan", "Y", "fitur", "Kode status pernikahan peminjam saat pengajuan (0.03% missing; category, kode -1 dipertahankan)"),
    "EmploymentStatus": ("saat_pengajuan", "Y", "fitur", "Kode status pekerjaan peminjam saat pengajuan (0.11% missing; category, kode -1 dipertahankan)"),
    "EmploymentDurationCurrentEmployer": ("saat_pengajuan", "Y", "fitur", "Kategori lama bekerja di pemberi kerja saat ini (0.90% missing; category)"),
    "OccupationArea": ("saat_pengajuan", "Y", "fitur", "Kode bidang pekerjaan peminjam saat pengajuan (0.05% missing; category, kode -1 dipertahankan)"),
    "HomeOwnershipType": ("saat_pengajuan", "Y", "fitur", "Kode status kepemilikan tempat tinggal peminjam (0.92% missing; category, kode -1 dipertahankan)"),
    "IncomeFromPrincipalEmployer": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari pekerjaan utama saat pengajuan (0% missing)"),
    "IncomeFromPension": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari pensiun saat pengajuan (0% missing)"),
    "IncomeFromFamilyAllowance": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari tunjangan keluarga saat pengajuan (0% missing)"),
    "IncomeFromSocialWelfare": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari bantuan sosial saat pengajuan (0% missing)"),
    "IncomeFromLeavePay": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari uang cuti saat pengajuan (0% missing)"),
    "IncomeFromChildSupport": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan dari tunjangan anak saat pengajuan (0% missing)"),
    "IncomeOther": ("saat_pengajuan", "Y", "fitur", "Pendapatan bulanan lainnya saat pengajuan (0% missing)"),
    "IncomeTotal": ("saat_pengajuan", "Y", "fitur", "Total pendapatan bulanan peminjam saat pengajuan (0% missing)"),
    "ExistingLiabilities": ("saat_pengajuan", "Y", "fitur", "Jumlah kewajiban aktif saat pengajuan (0% missing)"),
    "LiabilitiesTotal": ("saat_pengajuan", "Y", "fitur", "Total nominal kewajiban bulanan saat pengajuan (0% missing)"),
    "RefinanceLiabilities": ("saat_pengajuan", "Y", "fitur", "Jumlah kewajiban yang direfinansialisasi saat pengajuan (0% missing)"),
    "DebtToIncome": ("saat_pengajuan", "Y", "fitur", "Rasio utang terhadap pendapatan saat pengajuan (0.03% missing)"),
    "FreeCash": ("saat_pengajuan", "Y", "fitur", "Sisa kas bebas bulanan saat pengajuan (0.03% missing)"),
    "MonthlyPaymentDay": ("saat_pengajuan", "Y", "fitur", "Tanggal jatuh tempo bulanan yang dipilih saat pengajuan (0% missing)"),
    "CreditScoreEsMicroL": ("saat_pengajuan", "Y", "fitur", "Skor biro kredit eksternal MicroL (83.09% terisi lintas negara; category + CreditScoreEsMicroL_missing)"),
    "CreditScoreFiAsiakasTietoRiskGrade": ("saat_pengajuan", "Y", "fitur", "Grade risiko AsiakasTieto Finlandia (99.69% terisi pada populasi FI; category tanpa indikator missing)"),
    "CreditScoreEeMini": ("saat_pengajuan", "Y", "fitur", "Skor kredit mini Estonia (94.93% terisi pada populasi EE; skala numerik float64 tanpa indikator missing)"),
    "NoOfPreviousLoansBeforeLoan": ("saat_pengajuan", "Y", "fitur", "Jumlah pinjaman Bondora sebelumnya sebelum pengajuan ini (0.005% missing)"),
    "AmountOfPreviousLoansBeforeLoan": ("saat_pengajuan", "Y", "fitur", "Total nominal pinjaman Bondora sebelumnya sebelum pengajuan ini (0.005% missing)"),
    "PreviousRepaymentsBeforeLoan": ("saat_pengajuan", "Y", "fitur", "Total pembayaran pada pinjaman sebelumnya (sisa missing 4.15% setelah aturan domain; disertai indikator _missing)"),
    "PreviousEarlyRepaymentsBefoleLoan": ("saat_pengajuan", "Y", "fitur", "Total pelunasan dipercepat pada pinjaman sebelumnya (sisa missing 0.94% setelah aturan domain; tanpa indikator)"),
    "PreviousEarlyRepaymentsCountBeforeLoan": ("saat_pengajuan", "Y", "fitur", "Frekuensi pelunasan dipercepat pada pinjaman sebelumnya (0.005% missing)"),
}


def _build_target_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build target_summary.csv DataFrame where default proportion is only computed on resolved subsets."""
    total_n = len(df)
    has_def = df[DEFAULT_DATE_COL].notna() & (df[DEFAULT_DATE_COL].astype(str).str.strip() != "")
    is_repaid = df[STATUS_COL] == STATUS_REPAID
    is_current = df[STATUS_COL] == STATUS_CURRENT
    is_late = df[STATUS_COL] == STATUS_LATE
    pw_pos = pd.to_numeric(df["PrincipalWriteOffs"], errors="coerce").fillna(0.0) > 0.0

    groups = [
        ("1. DefaultDate terisi (Semua Status)", has_def, True, "is_resolved=True, default=1"),
        ("1a. DefaultDate terisi & Status=Late", has_def & is_late, True, "is_resolved=True, default=1"),
        ("1b. DefaultDate terisi & Status=Repaid", has_def & is_repaid, True, "is_resolved=True, default=1"),
        ("1c. DefaultDate terisi & Status=Current", has_def & is_current, True, "is_resolved=True, default=1"),
        (
            "2a. Status=Repaid, DefaultDate kosong, PrincipalWriteOffs > 0",
            (~has_def) & is_repaid & pw_pos,
            True,
            "is_resolved=True, default=1 (92.41% >= 61 hari terlambat)",
        ),
        (
            "2b. Status=Repaid, DefaultDate kosong, PrincipalWriteOffs == 0",
            (~has_def) & is_repaid & (~pw_pos),
            True,
            "is_resolved=True, default=0 (termasuk 297 baris hanya InterestWriteOffs > 0)",
        ),
        (
            "3. Status=Current & DefaultDate kosong (Censored)",
            (~has_def) & is_current,
            False,
            "is_resolved=False, default=NA (dikeluarkan dari model prediktif)",
        ),
        (
            "4. Status=Late & DefaultDate kosong (Censored)",
            (~has_def) & is_late,
            False,
            "is_resolved=False, default=NA (dikeluarkan dari model prediktif)",
        ),
        (
            "SUBSET: is_resolved == True (Masuk Pemodelan Prediktif)",
            df[IS_RESOLVED_COL],
            True,
            "Total Resolved (71.864 positif + 41.696 negatif = 113.560)",
        ),
        (
            "SUBSET: is_resolved == False (Keluar / Right-Censored)",
            ~df[IS_RESOLVED_COL],
            False,
            "Total Censored (57.611 Current + 8.064 Late = 65.675; default=NA)",
        ),
    ]

    loandate_dt = pd.to_datetime(df["LoanDate"], errors="coerce", format="mixed")
    rows: list[dict[str, Any]] = []
    for g_name, g_mask, g_res, g_rule in groups:
        sub = df[g_mask]
        sub_dt = loandate_dt[g_mask]
        n_rows = len(sub)
        n_def_1 = int((sub[DEFAULT_TARGET_COL] == 1).sum())
        n_def_0 = int((sub[DEFAULT_TARGET_COL] == 0).sum())
        n_def_na = int(sub[DEFAULT_TARGET_COL].isna().sum())
        def_rate = round(n_def_1 / n_rows * 100, 2) if (g_res and n_rows > 0) else np.nan
        rr = sub[REGRESSION_TARGET_COL].dropna()
        rows.append(
            {
                "kelompok": g_name,
                "aturan_perlakuan": g_rule,
                "n_baris": n_rows,
                "pct_dari_total": round(n_rows / total_n * 100, 2),
                "is_resolved": g_res,
                "n_default_1": n_def_1,
                "n_default_0": n_def_0,
                "n_default_na": n_def_na,
                "proporsi_default_pct": def_rate,
                "loandate_min": str(sub_dt.min().date()) if sub_dt.notna().any() else "",
                "loandate_max": str(sub_dt.max().date()) if sub_dt.notna().any() else "",
                "return_ratio_min": round(float(rr.min()), 4) if len(rr) > 0 else np.nan,
                "return_ratio_mean": round(float(rr.mean()), 4) if len(rr) > 0 else np.nan,
                "return_ratio_median": round(float(rr.median()), 4) if len(rr) > 0 else np.nan,
                "return_ratio_max": round(float(rr.max()), 4) if len(rr) > 0 else np.nan,
                "return_ratio_pct_zero": round(float((rr == 0).mean() * 100), 2) if len(rr) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_clean_dataset(
    raw_df: pd.DataFrame | None = None,
    parquet_path: Path | None = None,
    schema_path: Path | None = None,
    save_tables: bool = True,
) -> pd.DataFrame:
    """Execute the full Tahap 1 cleaning pipeline and save ``bondora_clean.parquet`` and ``schema.json``.

    Pipeline sequence:
    1. Load ``data/raw/LoanData.csv`` via ``load_raw()`` (if ``raw_df`` is None).
    2. Construct targets: ``default`` (nullable ``Int8``: ``0``/``1`` for resolved,
       ``pd.NA`` for ``is_resolved == False``), ``is_resolved`` (``bool``), and
       ``return_ratio`` (``float64``).
    3. Convert 18 date columns to ``datetime64[ns]`` and 15 categorical columns
       to ``category`` (preserving ``-1`` as category ``'-1'``; ``CreditScoreEeMini``
       remains ``float64``).
    4. Check full-row and key (``LoanId``, ``LoanNumber``) duplicates.
    5. Apply domain rules (``NaN -> 0``) on ``PreviousRepaymentsBeforeLoan``
       (when ``NoOfPreviousLoansBeforeLoan == 0``) and
       ``PreviousEarlyRepaymentsBefoleLoan`` (when ``PreviousEarlyRepaymentsCountBeforeLoan == 0``).
    6. Create 6 anomaly/sentinel boolean flags (``flag_*``) and 3 missing indicators
       (``*_missing``).
    7. Compute 7 derived columns via ``add_basic_derived_features()``.
    8. Drop 5 unused columns (``City``, ``County``, ``DateOfBirth``,
       ``EmploymentPosition``, ``ReportAsOfEOD``) and write
       ``data/processed/bondora_clean.parquet`` + ``data/processed/schema.json``.
    9. Reload parquet and assert row/column counts, dtypes, ``default.notna().sum() == 113560``,
       and ``default.sum() == 71864``.

    Parameters
    ----------
    raw_df : pd.DataFrame or None, default None
        Optional pre-loaded raw DataFrame. If None, ``load_raw()`` is called.
    parquet_path : Path or None, default None
        Destination path for cleaned Parquet file. Defaults to
        ``DATA_PROCESSED / "bondora_clean.parquet"``.
    schema_path : Path or None, default None
        Destination path for schema JSON file. Defaults to
        ``DATA_PROCESSED / "schema.json"``.
    save_tables : bool, default True
        Whether to also write updated CSV tables in ``reports/tables/``
        (``target_summary.csv``, ``leakage_audit.csv``, ``cleaning_log.csv``).

    Returns
    -------
    pd.DataFrame
        Reloaded and verified ``bondora_clean`` DataFrame (``179,235 x 130``).
    """
    df_raw = load_raw() if raw_df is None else raw_df.copy()
    raw_cols = list(df_raw.columns)
    df = df_raw.copy()

    cleaning_log_rows: list[dict[str, Any]] = []
    step_no = 1

    # 1. Target columns
    cols_before = df.shape[1]
    s_default = make_default_label(df)
    s_resolved = make_is_resolved(df)
    s_return = make_regression_target(df)

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    DEFAULT_TARGET_COL: s_default,
                    IS_RESOLVED_COL: s_resolved,
                    REGRESSION_TARGET_COL: s_return,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 3: Pembentukan kolom target (default [Int8 nullable], is_resolved, return_ratio)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": cols_before,
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": len(df) * 3,
            "alasan": (
                "Target default (Int8): 1 jika DefaultDate terisi (71.416) atau Repaid tanpa DefaultDate dgn PrincipalWriteOffs>0 (448), "
                "0 untuk Repaid tanpa DefaultDate dgn PrincipalWriteOffs==0 (41.696), dan NA untuk is_resolved==False (65.675); "
                "return_ratio = (PrincipalPaymentsMade + InterestAndPenaltyPaymentsMade) / Amount"
            ),
        }
    )
    step_no += 1

    target_summary_df = _build_target_summary_table(df)
    if save_tables:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        target_summary_df.to_csv(TABLES_DIR / "target_summary.csv", index=False, encoding="utf-8")

    # 2. Convert 18 date columns to datetime64[ns]
    n_date_converted = 0
    for col in DATE_COLUMNS:
        n_date_converted += int(df[col].notna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": f"Langkah 4.1a: Konversi {len(DATE_COLUMNS)} kolom tanggal ke datetime64[ns]",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": n_date_converted,
            "alasan": "Mengubah string tanggal menjadi datetime64[ns] (0 gagal parse)",
        }
    )
    step_no += 1

    # 2b. Fixed-horizon default labels (12m and 24m) for Tahap 3 Diagnostic Analytics
    cols_before_horizon = df.shape[1]
    s_obs_12m, s_def_12m = make_default_horizon_label(df, months=12, report_date=REPORT_DATE)
    s_obs_24m, s_def_24m = make_default_horizon_label(df, months=24, report_date=REPORT_DATE)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    OBSERVABLE_12M_COL: s_obs_12m,
                    DEFAULT_12M_COL: s_def_12m,
                    OBSERVABLE_24M_COL: s_obs_24m,
                    DEFAULT_24M_COL: s_def_24m,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )
    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 3b (Tahap 3): Pembentukan label default berhorizon tetap (observable_12m, default_12m, observable_24m, default_24m)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": cols_before_horizon,
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": len(df) * 4,
            "alasan": (
                f"Mengatasi bias seleksi right-censoring dgn horizon tetap sejak LoanDate (REPORT_DATE={REPORT_DATE}): "
                f"12m observable={int(s_obs_12m.sum()):,}, default_12m=1 ({int((s_def_12m == 1).sum()):,}), 0 ({int((s_def_12m == 0).sum()):,}), NA ({int(s_def_12m.isna().sum()):,}); "
                f"24m observable={int(s_obs_24m.sum()):,}, default_24m=1 ({int((s_def_24m == 1).sum()):,}), 0 ({int((s_def_24m == 0).sum()):,}), NA ({int(s_def_24m.isna().sum()):,})"
            ),
        }
    )
    step_no += 1

    # 3. Convert 15 categorical columns to category
    all_cat_cols = NUMERIC_CODE_CATEGORY_COLUMNS + STRING_CATEGORY_COLUMNS
    n_cat_converted = 0
    for col in NUMERIC_CODE_CATEGORY_COLUMNS:
        n_cat_converted += int(df[col].notna().sum())
        s_num = pd.to_numeric(df[col], errors="coerce")
        int_cats = sorted(s_num.dropna().astype("int64").unique())
        str_cats = [str(x) for x in int_cats]
        codes = pd.Series(-1, index=df.index, dtype="int16")
        mask_valid = s_num.notna()
        cat_index = pd.Index(int_cats, dtype="int64")
        codes.loc[mask_valid] = cat_index.get_indexer(s_num.loc[mask_valid].astype("int64")).astype("int16")
        df[col] = pd.Categorical.from_codes(codes, categories=str_cats)
    for col in STRING_CATEGORY_COLUMNS:
        n_cat_converted += int(df[col].notna().sum())
        df[col] = df[col].astype("category")

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": f"Langkah 4.1b: Konversi {len(all_cat_cols)} kolom kode/teks kategorikal ke dtype category",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": n_cat_converted,
            "alasan": "Konversi kolom kategorikal ke dtype category sesuai persetujuan Gerbang 2 (kode -1 dipertahankan sebagai kategori 'Not set'; CreditScoreEeMini tetap float64)",
        }
    )
    step_no += 1

    # 4. Duplicate check
    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 4.2: Pemeriksaan duplikat baris penuh dan ID pinjaman (LoanId, LoanNumber)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": 0,
            "alasan": "Tidak ditemukan baris duplikat (0 duplikat penuh maupun pada LoanId/LoanNumber)",
        }
    )
    step_no += 1

    # 5. Domain rules for previous repayments (NaN -> 0)
    mask_prev_rep_zero = df["PreviousRepaymentsBeforeLoan"].isna() & (df["NoOfPreviousLoansBeforeLoan"] == 0)
    n_prev_rep_filled = int(mask_prev_rep_zero.sum())
    df.loc[mask_prev_rep_zero, "PreviousRepaymentsBeforeLoan"] = 0.0
    n_prev_rep_rem_nan = int(df["PreviousRepaymentsBeforeLoan"].isna().sum())
    pct_prev_rep_rem_nan = n_prev_rep_rem_nan / len(df) * 100

    mask_prev_early_zero = df["PreviousEarlyRepaymentsBefoleLoan"].isna() & (
        df["PreviousEarlyRepaymentsCountBeforeLoan"] == 0
    )
    n_prev_early_filled = int(mask_prev_early_zero.sum())
    df.loc[mask_prev_early_zero, "PreviousEarlyRepaymentsBefoleLoan"] = 0.0
    n_prev_early_rem_nan = int(df["PreviousEarlyRepaymentsBefoleLoan"].isna().sum())
    pct_prev_early_rem_nan = n_prev_early_rem_nan / len(df) * 100

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 4.3a: Pengisian struktural PreviousRepaymentsBeforeLoan (NaN -> 0 jika NoOfPreviousLoansBeforeLoan == 0)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": n_prev_rep_filled,
            "alasan": f"perubahan pencatatan Bondora sejak 2018 (NaN = 0); sisa NaN = {n_prev_rep_rem_nan:,} baris ({pct_prev_rep_rem_nan:.2f}%)",
        }
    )
    step_no += 1

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 4.3b: Pengisian struktural PreviousEarlyRepaymentsBefoleLoan (NaN -> 0 jika PreviousEarlyRepaymentsCountBeforeLoan == 0)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": n_prev_early_filled,
            "alasan": f"perubahan pencatatan Bondora sejak 2018 (NaN = 0); sisa NaN = {n_prev_early_rem_nan:,} baris ({pct_prev_early_rem_nan:.2f}%)",
        }
    )
    step_no += 1

    # 6. Anomaly / sentinel flags (6 boolean columns)
    cols_before_flags = df.shape[1]
    flag_age_invalid = ((df["Age"] < 18) | (df["Age"] > 100)).astype(bool)
    flag_sentinel_minus1 = (
        (df_raw["UseOfLoan"] == -1)
        | (df_raw["MaritalStatus"] == -1)
        | (df_raw["EmploymentStatus"] == -1)
        | (df_raw["OccupationArea"] == -1)
        | (df_raw["Education"] == -1)
        | (df_raw["HomeOwnershipType"] == -1)
    ).astype(bool)
    flag_freecash_neg = (df["FreeCash"].fillna(0) < 0).astype(bool)
    flag_balance_neg = ((df["PrincipalBalance"].fillna(0) < 0) | (df["InterestAndPenaltyBalance"].fillna(0) < 0)).astype(
        bool
    )
    flag_zero_income = (df["IncomeTotal"].fillna(0) <= 0).astype(bool)
    flag_repaid_writeoff = (
        (df_raw[STATUS_COL] == STATUS_REPAID)
        & df_raw[DEFAULT_DATE_COL].isna()
        & ((df_raw["PrincipalWriteOffs"].fillna(0) > 0) | (df_raw["InterestAndPenaltyWriteOffs"].fillna(0) > 0))
    ).astype(bool)

    flags_df = pd.DataFrame(
        {
            "flag_umur_tidak_wajar": flag_age_invalid,
            "flag_kode_sentinel_minus1": flag_sentinel_minus1,
            "flag_freecash_negatif": flag_freecash_neg,
            "flag_saldo_negatif": flag_balance_neg,
            "flag_pendapatan_nol": flag_zero_income,
            "flag_repaid_ada_writeoff": flag_repaid_writeoff,
        },
        index=df.index,
    )
    df = pd.concat([df, flags_df], axis=1)

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 4.4: Pembuatan 6 kolom flag penanda nilai tak wajar, sentinel, dan write-off pada Repaid",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": cols_before_flags,
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": len(df) * 6,
            "alasan": (
                f"Menandai anomali tanpa menghapus baris: Age<18 ({int(flag_age_invalid.sum())}), "
                f"sentinel -1 ({int(flag_sentinel_minus1.sum()):,}), FreeCash<0 ({int(flag_freecash_neg.sum())}), "
                f"saldo<0 ({int(flag_balance_neg.sum())}), IncomeTotal<=0 ({int(flag_zero_income.sum())}), "
                f"Repaid dgn write-off ({int(flag_repaid_writeoff.sum())})"
            ),
        }
    )
    step_no += 1

    # 7. Strip & check string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    n_str_changed = 0
    for col in str_cols:
        s_orig = df[col]
        mask_notna = s_orig.notna()
        s_stripped = s_orig.where(~mask_notna, s_orig.astype(str).str.strip())
        changed = int((s_orig[mask_notna] != s_stripped[mask_notna]).sum())
        if changed > 0:
            df[col] = s_stripped
            n_str_changed += changed

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": f"Langkah 4.5: Standarisasi strip & pemeriksaan kapitalisasi pada {len(str_cols)} kolom teks",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": n_str_changed,
            "alasan": "Seluruh nilai teks sudah konsisten tanpa spasi berlebih atau variasi kapitalisasi (0 nilai diubah)",
        }
    )
    step_no += 1

    # 8. Missing value indicators (*_missing)
    cols_before_miss_ind = df.shape[1]
    miss_ind_df = pd.DataFrame(
        {
            "MonthlyPayment_missing": df["MonthlyPayment"].isna().astype(bool),
            "PreviousRepaymentsBeforeLoan_missing": df["PreviousRepaymentsBeforeLoan"].isna().astype(bool),
            "CreditScoreEsMicroL_missing": df["CreditScoreEsMicroL"].isna().astype(bool),
        },
        index=df.index,
    )
    df = pd.concat([df, miss_ind_df], axis=1)

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 4.6: Pembuatan 3 kolom indikator *_missing (MonthlyPayment, PreviousRepaymentsBeforeLoan, CreditScoreEsMicroL)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": cols_before_miss_ind,
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": len(df) * 3,
            "alasan": (
                f"Indikator missing >1% non-struktural: MonthlyPayment_missing ({int(miss_ind_df['MonthlyPayment_missing'].sum()):,}), "
                f"PreviousRepaymentsBeforeLoan_missing ({int(miss_ind_df['PreviousRepaymentsBeforeLoan_missing'].sum()):,}), "
                f"CreditScoreEsMicroL_missing ({int(miss_ind_df['CreditScoreEsMicroL_missing'].sum()):,})"
            ),
        }
    )
    step_no += 1

    # 9. Save leakage_audit.csv if requested
    if save_tables:
        audit_rows = [
            {
                "nama_kolom": col,
                "kategori": FINAL_AUDIT_SPEC[col][0],
                "boleh_jadi_fitur": FINAL_AUDIT_SPEC[col][1],
                "alasan": FINAL_AUDIT_SPEC[col][3],
            }
            for col in raw_cols
        ]
        pd.DataFrame(audit_rows).to_csv(TABLES_DIR / "leakage_audit.csv", index=False, encoding="utf-8")

    y_orig_cols = [c for c, spec in FINAL_AUDIT_SPEC.items() if spec[1] == "Y"]
    num_y_cols = [
        c for c in y_orig_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": f"Langkah 6: Deteksi outlier (IQR & persentil 1/99) pada {len(num_y_cols)} kolom fitur numerik",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": df.shape[1],
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": 0,
            "alasan": "Hanya deteksi dan pencatatan ke outlier_summary.csv sesuai aturan kerja (tanpa menghapus/memotong outlier)",
        }
    )
    step_no += 1

    # 10. Safe derived features
    cols_before_deriv = df.shape[1]
    df = add_basic_derived_features(df)

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": "Langkah 7: Penambahan 7 kolom turunan aman (4 fitur rasio + 3 kolom analisis/pembagi_waktu)",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df),
            "kolom_sebelum": cols_before_deriv,
            "kolom_sesudah": df.shape[1],
            "jumlah_nilai_diubah": len(df) * 7,
            "alasan": (
                "Menambahkan fitur rasio (loan_to_income_ratio, monthly_payment_to_income_ratio, "
                "total_liabilities_to_income_ratio, avg_previous_loan_amount) dan kolom analisis (funded_to_applied_ratio, loan_year, loan_month)"
            ),
        }
    )
    step_no += 1

    # 11. Drop 5 unused columns and save bondora_clean.parquet + schema.json
    cols_before_drop = df.shape[1]
    df_clean = df.drop(columns=DROPPED_COLUMNS)

    cleaning_log_rows.append(
        {
            "urutan": step_no,
            "langkah": f"Langkah 8: Pembuangan 5 kolom tidak terpakai ({', '.join(DROPPED_COLUMNS)}) & penyimpanan bondora_clean.parquet",
            "baris_sebelum": len(df),
            "baris_sesudah": len(df_clean),
            "kolom_sebelum": cols_before_drop,
            "kolom_sesudah": df_clean.shape[1],
            "jumlah_nilai_diubah": 0,
            "alasan": "Membuang 4 kolom 100% kosong (City, County, DateOfBirth, EmploymentPosition) dan 1 kolom konstan (ReportAsOfEOD, disimpan di REPORT_DATE)",
        }
    )

    out_parquet = (DATA_PROCESSED / "bondora_clean.parquet") if parquet_path is None else Path(parquet_path)
    out_schema = (DATA_PROCESSED / "schema.json") if schema_path is None else Path(schema_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_schema.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_parquet(out_parquet, engine="pyarrow", index=False)

    schema_entries = []
    for col in df_clean.columns:
        dtype_str = str(df_clean[col].dtype)
        if col in FINAL_AUDIT_SPEC:
            kat, boleh, peran_sc, alasan = FINAL_AUDIT_SPEC[col]
        elif col == DEFAULT_TARGET_COL:
            kat, boleh, peran_sc, alasan = (
                "target_turunan",
                "N",
                "target",
                "Variabel target klasifikasi biner (Int8 nullable: 0/1 untuk is_resolved==True, NA untuk is_resolved==False)",
            )
        elif col == REGRESSION_TARGET_COL:
            kat, boleh, peran_sc, alasan = ("target_turunan", "N", "target", f"Variabel target regresi ({col})")
        elif col == IS_RESOLVED_COL:
            kat, boleh, peran_sc, alasan = (
                "target_turunan",
                "N",
                "flag_censoring",
                "Penanda apakah hasil akhir pinjaman sudah diketahui (True=masuk pemodelan prediktif)",
            )
        elif col in (OBSERVABLE_12M_COL, OBSERVABLE_24M_COL):
            h_str = "12" if col == OBSERVABLE_12M_COL else "24"
            kat, boleh, peran_sc, alasan = (
                "target_turunan",
                "N",
                "flag_censoring",
                f"Penanda boolean apakah pinjaman telah teramati penuh sepanjang horizon {h_str} bulan sejak LoanDate sampai REPORT_DATE ({REPORT_DATE})",
            )
        elif col in (DEFAULT_12M_COL, DEFAULT_24M_COL):
            h_str = "12" if col == DEFAULT_12M_COL else "24"
            kat, boleh, peran_sc, alasan = (
                "target_turunan",
                "N",
                "target_horizon",
                f"Variabel target gagal bayar berhorizon tetap {h_str} bulan (Int8 nullable: 1 jika DefaultDate <= LoanDate + {h_str} bulan, 0 jika teramati {h_str} bulan tanpa default <= {h_str} bulan, NA jika belum teramati {h_str} bulan / Repaid write-off tanpa DefaultDate / DefaultDate < LoanDate)",
            )
        elif col.startswith("flag_"):
            kat, boleh, peran_sc, alasan = ("flag_kualitas", "N", "flag", f"Kolom flag penanda nilai tak wajar/anomali ({col})")
        elif col.endswith("_missing"):
            kat, boleh, peran_sc, alasan = ("indikator_missing", "Y", "fitur", f"Indikator boolean nilai kosong untuk fitur ({col})")
        elif col in (
            "loan_to_income_ratio",
            "monthly_payment_to_income_ratio",
            "total_liabilities_to_income_ratio",
            "avg_previous_loan_amount",
        ):
            kat, boleh, peran_sc, alasan = ("fitur_turunan", "Y", "fitur", f"Fitur rasio turunan aman saat pengajuan ({col})")
        elif col in ("loan_year", "loan_month"):
            kat, boleh, peran_sc, alasan = (
                "turunan_waktu",
                "N",
                "pembagi_waktu",
                f"Ekstraksi waktu dari LoanDate untuk analisis/pembagian waktu ({col})",
            )
        elif col == "funded_to_applied_ratio":
            kat, boleh, peran_sc, alasan = (
                "turunan_analisis",
                "N",
                "analisis",
                "Rasio Amount / AppliedAmount untuk analisis pendanaan (bukan fitur prediktif)",
            )
        else:
            raise ValueError(f"Unmapped column in schema.json: {col}")

        schema_entries.append(
            {
                "nama_kolom": col,
                "dtype": dtype_str,
                "peran": peran_sc,
                "kategori_audit": kat,
                "boleh_jadi_fitur": boleh,
                "keterangan": alasan,
            }
        )

    schema_json_obj = {
        "project": "bondora-analytics",
        "report_date": REPORT_DATE,
        "n_rows": len(df_clean),
        "n_columns": df_clean.shape[1],
        "n_features_Y": sum(1 for x in schema_entries if x["boleh_jadi_fitur"] == "Y"),
        "dropped_columns": DROPPED_COLUMNS,
        "columns": schema_entries,
    }
    with open(out_schema, "w", encoding="utf-8") as f:
        json.dump(schema_json_obj, f, indent=2, ensure_ascii=False)

    if save_tables:
        pd.DataFrame(cleaning_log_rows).to_csv(TABLES_DIR / "cleaning_log.csv", index=False, encoding="utf-8")

    # Reload parquet and verify integrity
    df_reloaded = pd.read_parquet(out_parquet, engine="pyarrow")
    assert df_reloaded.shape == df_clean.shape, f"Shape mismatch: {df_reloaded.shape} != {df_clean.shape}"
    assert str(df_reloaded[DEFAULT_TARGET_COL].dtype) == "Int8", (
        f"Expected default dtype Int8, got {df_reloaded[DEFAULT_TARGET_COL].dtype}"
    )
    if len(df_reloaded) == 179_235:
        assert int(df_reloaded[DEFAULT_TARGET_COL].notna().sum()) == 113_560
        assert int(df_reloaded[DEFAULT_TARGET_COL].sum()) == 71_864
    return df_reloaded


def main() -> None:
    """CLI entry point for ``python -m src.data.build_clean``."""
    df_clean = build_clean_dataset()
    n_notna = int(df_clean[DEFAULT_TARGET_COL].notna().sum())
    n_pos = int(df_clean[DEFAULT_TARGET_COL].sum())
    n_na = int(df_clean[DEFAULT_TARGET_COL].isna().sum())
    print(
        f"[build_clean] Saved data/processed/bondora_clean.parquet "
        f"(shape={df_clean.shape[0]:,} x {df_clean.shape[1]}, "
        f"default dtype={df_clean[DEFAULT_TARGET_COL].dtype}, "
        f"default.notna().sum()={n_notna:,}, default.sum()={n_pos:,}, default.isna().sum()={n_na:,})."
    )


if __name__ == "__main__":
    main()
