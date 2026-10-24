"""Stratified + S3 + borrower-chain sampling for the Tahap 6 Knowledge Graph."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis.descriptive import OUTCOME_CATEGORIES, make_outcome_group
from src.config import LOAN_DATE_COL, RANDOM_SEED

ERA_CUTOFF_DATE: str = "2017-12-31"
ERA_PRA: str = "EraPraPerubahan"
ERA_PASCA: str = "EraPascaPerubahan"


def make_era_series(loan_date: pd.Series) -> pd.Series:
    """Assign each loan to ``EraPraPerubahan`` (<= 2017-12-31) or ``EraPascaPerubahan`` (>= 2018-01-01)."""
    dt = pd.to_datetime(loan_date)
    vals = np.where(dt <= pd.Timestamp(ERA_CUTOFF_DATE), ERA_PRA, ERA_PASCA)
    return pd.Series(vals, index=loan_date.index, name="era", dtype="object")


def make_tenor_band_series(duration: pd.Series) -> pd.Series:
    """Map ``LoanDuration`` (months) to the 5 project tenor bands used in Tahap 2-3."""
    d = pd.to_numeric(duration, errors="coerce")
    vals = np.select(
        [d <= 12, d <= 24, d <= 36, d <= 59],
        ["Tenor_1_12", "Tenor_13_24", "Tenor_25_36", "Tenor_37_59"],
        default="Tenor_60",
    )
    return pd.Series(vals, index=duration.index, name="tenor_band", dtype="object")


def make_risk_band_series(p_calibrated: pd.Series) -> pd.Series:
    """Map ``p_calibrated`` to the 3 project risk bands: Rendah (<0.15), Sedang (0.15-<0.35), Tinggi (>=0.35)."""
    p = pd.to_numeric(p_calibrated, errors="coerce")
    vals = np.select(
        [p < 0.15, p < 0.35],
        ["PitaRisikoRendah", "PitaRisikoSedang"],
        default="PitaRisikoTinggi",
    )
    return pd.Series(vals, index=p_calibrated.index, name="risk_band", dtype="object")


def allocate_stratified_quotas(
    strata_counts: pd.Series,
    target_n: int = 4_000,
    min_per_stratum: int = 20,
) -> dict[str, int]:
    """Proportional allocation across non-empty strata with a minimum quota per stratum."""
    non_empty = strata_counts[strata_counts > 0].sort_index()
    total_pop = int(non_empty.sum())
    if total_pop <= target_n:
        return {str(k): int(v) for k, v in non_empty.items()}

    n_strata = len(non_empty)
    eff_min = min(min_per_stratum, max(1, target_n // max(1, n_strata)))

    quotas: dict[str, int] = {}
    fixed_keys: set[str] = set()
    for k, cnt in non_empty.items():
        ks = str(k)
        c_int = int(cnt)
        if c_int <= eff_min:
            quotas[ks] = c_int
            fixed_keys.add(ks)

    rem_target = target_n - sum(quotas.values())
    flex_items = [(str(k), int(cnt)) for k, cnt in non_empty.items() if str(k) not in fixed_keys]
    flex_pop = sum(cnt for _, cnt in flex_items)

    raw_quotas: dict[str, float] = {}
    for ks, cnt in flex_items:
        raw = rem_target * (cnt / flex_pop) if flex_pop > 0 else float(eff_min)
        raw_quotas[ks] = raw
        quotas[ks] = min(cnt, max(eff_min, int(np.floor(raw))))

    diff = target_n - sum(quotas.values())
    if diff > 0:
        # Add 1 to strata with largest fractional remainders that still have capacity
        order = sorted(
            flex_items,
            key=lambda item: (-(raw_quotas[item[0]] - np.floor(raw_quotas[item[0]])), item[0]),
        )
        while diff > 0:
            changed = False
            for ks, cnt in order:
                if quotas[ks] < cnt:
                    quotas[ks] += 1
                    diff -= 1
                    changed = True
                    if diff == 0:
                        break
            if not changed:
                break
    elif diff < 0:
        # Subtract 1 from largest flexible strata while keeping >= eff_min
        order = sorted(
            flex_items,
            key=lambda item: (-quotas[item[0]], item[0]),
        )
        while diff < 0:
            changed = False
            for ks, _ in order:
                if quotas[ks] > eff_min:
                    quotas[ks] -= 1
                    diff += 1
                    changed = True
                    if diff == 0:
                        break
            if not changed:
                break

    return quotas


def sample_graph_loans(
    df: pd.DataFrame,
    s3_loan_ids: Sequence[str] | set[str],
    *,
    n_stratified: int = 4_000,
    min_per_stratum: int = 20,
    max_loans: int = 20_000,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Build the Tahap 6 graph sample (Stratified 4,000 + S3 5,000 + borrower expansion <= max_loans)."""
    work = pd.DataFrame(
        {
            "LoanId": df["LoanId"].astype(str),
            "LoanNumber": pd.to_numeric(df["LoanNumber"], errors="coerce").astype("int64"),
            "UserName": df["UserName"].astype(str),
            "Country": df["Country"].astype(str),
            "LoanDate": pd.to_datetime(df[LOAN_DATE_COL]),
            "outcome_group": make_outcome_group(df).astype(str),
            "era": make_era_series(df[LOAN_DATE_COL]).astype(str),
        },
        index=df.index,
    )
    work["stratum"] = work["Country"] + "|" + work["outcome_group"] + "|" + work["era"]

    strata_counts = work["stratum"].value_counts()
    quotas = allocate_stratified_quotas(strata_counts, target_n=n_stratified, min_per_stratum=min_per_stratum)

    rng = np.random.default_rng(seed)
    strat_ids_list: list[str] = []
    for st_key in sorted(quotas.keys()):
        q = quotas[st_key]
        sub_ids = np.sort(work.loc[work["stratum"] == st_key, "LoanId"].to_numpy())
        chosen = rng.choice(sub_ids, size=q, replace=False)
        strat_ids_list.extend(chosen.tolist())

    strat_set = set(strat_ids_list)
    all_loan_ids_set = set(work["LoanId"])
    s3_set = set(str(x) for x in s3_loan_ids) & all_loan_ids_set
    overlap_strat_s3 = strat_set & s3_set
    core_set = strat_set | s3_set

    # Borrowers in core_set
    s3_borrowers = sorted(set(work.loc[work["LoanId"].isin(s3_set), "UserName"]))
    s3_borrowers_set = set(s3_borrowers)
    core_borrowers = sorted(set(work.loc[work["LoanId"].isin(core_set), "UserName"]))
    non_s3_borrowers = sorted(set(core_borrowers) - s3_borrowers_set)

    # Full expansion before cap
    full_expanded_set = set(work.loc[work["UserName"].isin(core_borrowers), "LoanId"])
    n_unconstrained = len(full_expanded_set)

    # Expansion if only S3 borrowers are expanded (non-S3 borrower expansion dropped)
    s3_expanded_only_set = core_set | set(work.loc[work["UserName"].isin(s3_borrowers_set), "LoanId"])
    n_after_drop_non_s3 = len(s3_expanded_only_set)

    # Group extra loans (outside core_set) by borrower
    extra_df = work.loc[~work["LoanId"].isin(core_set) & work["UserName"].isin(core_borrowers), ["UserName", "LoanId"]]
    borrower_extra_map: dict[str, list[str]] = {
        str(b): sorted(g["LoanId"].tolist()) for b, g in extra_df.groupby("UserName", observed=True)
    }

    selected_set = set(core_set)
    expansion_note = "Seluruh pinjaman lain milik peminjam terpilih dimasukkan (di bawah batas max_loans)."

    if n_unconstrained <= max_loans:
        selected_set = full_expanded_set
    else:
        # Step 2.4: reduce step 3 for non-S3 borrowers first, then S3 borrowers if still > max_loans
        s3_b_candidates = [b for b in s3_borrowers if b in borrower_extra_map]
        non_s3_b_candidates = [b for b in non_s3_borrowers if b in borrower_extra_map]

        perm_s3 = rng.permutation(s3_b_candidates).tolist() if s3_b_candidates else []
        perm_non_s3 = rng.permutation(non_s3_b_candidates).tolist() if non_s3_b_candidates else []

        added_s3_borrowers = 0
        added_non_s3_borrowers = 0

        for b in perm_s3:
            extra_loans = borrower_extra_map[b]
            if len(selected_set) + len(extra_loans) <= max_loans:
                selected_set.update(extra_loans)
                added_s3_borrowers += 1

        for b in perm_non_s3:
            extra_loans = borrower_extra_map[b]
            if len(selected_set) + len(extra_loans) <= max_loans:
                selected_set.update(extra_loans)
                added_non_s3_borrowers += 1

        expansion_note = (
            f"Perluasan penuh menghasilkan {n_unconstrained:,} pinjaman (> batas {max_loans:,}); "
            f"perluasan peminjam di luar S3 dikurangi ({added_non_s3_borrowers}/{len(non_s3_b_candidates)} peminjam non-S3 "
            f"dan {added_s3_borrowers}/{len(s3_b_candidates)} peminjam S3 dengan pinjaman tambahan disertakan utuh per peminjam "
            f"secara acak dengan seed={seed}) sehingga total = {len(selected_set):,} pinjaman."
        )

    expansion_only_set = selected_set - core_set

    # Tag each row in sample DataFrame
    sample_df = df.loc[df["LoanId"].astype(str).isin(selected_set)].copy()
    sample_ids = sample_df["LoanId"].astype(str)
    sample_df["in_stratified"] = sample_ids.isin(strat_set)
    sample_df["in_s3"] = sample_ids.isin(s3_set)
    sample_df["in_expansion_only"] = sample_ids.isin(expansion_only_set)
    sample_df["outcome_group"] = work.loc[sample_df.index, "outcome_group"]
    sample_df["era"] = work.loc[sample_df.index, "era"]
    sample_df["tenor_band"] = make_tenor_band_series(sample_df["LoanDuration"])

    summary_df = _build_sample_summary_table(
        work=work,
        strat_set=strat_set,
        s3_set=s3_set,
        overlap_strat_s3=overlap_strat_s3,
        core_set=core_set,
        expansion_only_set=expansion_only_set,
        selected_set=selected_set,
        n_unconstrained=n_unconstrained,
        n_after_drop_non_s3=n_after_drop_non_s3,
        expansion_note=expansion_note,
    )

    return {
        "sample_df": sample_df,
        "summary_df": summary_df,
        "stratified_ids": strat_set,
        "s3_ids": s3_set,
        "expansion_only_ids": expansion_only_set,
        "selected_ids": selected_set,
        "n_unconstrained": n_unconstrained,
        "n_after_drop_non_s3": n_after_drop_non_s3,
        "expansion_note": expansion_note,
        "quotas": quotas,
    }


def _build_sample_summary_table(
    *,
    work: pd.DataFrame,
    strat_set: set[str],
    s3_set: set[str],
    overlap_strat_s3: set[str],
    core_set: set[str],
    expansion_only_set: set[str],
    selected_set: set[str],
    n_unconstrained: int,
    n_after_drop_non_s3: int,
    expansion_note: str,
) -> pd.DataFrame:
    """Create ``07_sample_summary.csv`` comparing sample sources, dimensions, and strata vs population."""
    n_pop = len(work)
    n_strat = len(strat_set)
    n_sel = len(selected_set)

    sub_sel = work.loc[work["LoanId"].isin(selected_set)]
    sub_strat = work.loc[work["LoanId"].isin(strat_set)]

    rows: list[dict[str, Any]] = [
        {
            "kategori": "Total",
            "subkategori": "Pinjaman unik di graf (final <= 20.000)",
            "n_sampel_graf": n_sel,
            "pct_sampel_graf": 100.0,
            "n_stratifikasi_4000": n_strat,
            "pct_stratifikasi_4000": 100.0,
            "n_populasi": n_pop,
            "pct_populasi": 100.0,
            "selisih_proporsi_strat_vs_pop_pp": 0.0,
            "catatan": expansion_note,
        },
        {
            "kategori": "Total",
            "subkategori": "Peminjam unik (UserName)",
            "n_sampel_graf": int(sub_sel["UserName"].nunique()),
            "pct_sampel_graf": float(sub_sel["UserName"].nunique() / max(1, work["UserName"].nunique()) * 100.0),
            "n_stratifikasi_4000": int(sub_strat["UserName"].nunique()),
            "pct_stratifikasi_4000": float(sub_strat["UserName"].nunique() / max(1, work["UserName"].nunique()) * 100.0),
            "n_populasi": int(work["UserName"].nunique()),
            "pct_populasi": 100.0,
            "selisih_proporsi_strat_vs_pop_pp": np.nan,
            "catatan": "Rasio terhadap total peminjam unik populasi",
        },
        {
            "kategori": "Sumber",
            "subkategori": "1. Sampel terstratifikasi (Country x OutcomeGroup x Era)",
            "n_sampel_graf": n_strat,
            "pct_sampel_graf": float(n_strat / n_sel * 100.0),
            "n_stratifikasi_4000": n_strat,
            "pct_stratifikasi_4000": 100.0,
            "n_populasi": n_pop,
            "pct_populasi": float(n_strat / n_pop * 100.0),
            "selisih_proporsi_strat_vs_pop_pp": 0.0,
            "catatan": "Alokasi proporsional dengan minimum 20 per strata tidak kosong",
        },
        {
            "kategori": "Sumber",
            "subkategori": "2. Portofolio S3 (Tahap 5)",
            "n_sampel_graf": len(s3_set),
            "pct_sampel_graf": float(len(s3_set) / n_sel * 100.0),
            "n_stratifikasi_4000": len(overlap_strat_s3),
            "pct_stratifikasi_4000": float(len(overlap_strat_s3) / max(1, n_strat) * 100.0),
            "n_populasi": len(s3_set),
            "pct_populasi": float(len(s3_set) / n_pop * 100.0),
            "selisih_proporsi_strat_vs_pop_pp": np.nan,
            "catatan": f"Irisan antara sampel stratifikasi dan S3 = {len(overlap_strat_s3)} pinjaman",
        },
        {
            "kategori": "Sumber",
            "subkategori": "Gabungan butir 1 U 2 (Inti sebelum perluasan peminjam)",
            "n_sampel_graf": len(core_set),
            "pct_sampel_graf": float(len(core_set) / n_sel * 100.0),
            "n_stratifikasi_4000": n_strat,
            "pct_stratifikasi_4000": 100.0,
            "n_populasi": n_pop,
            "pct_populasi": float(len(core_set) / n_pop * 100.0),
            "selisih_proporsi_strat_vs_pop_pp": np.nan,
            "catatan": f"Peminjam unik pada gabungan inti = {work.loc[work['LoanId'].isin(core_set), 'UserName'].nunique():,}",
        },
        {
            "kategori": "Sumber",
            "subkategori": "3. Perluasan peminjam (pinjaman tambahan di luar butir 1 U 2)",
            "n_sampel_graf": len(expansion_only_set),
            "pct_sampel_graf": float(len(expansion_only_set) / n_sel * 100.0),
            "n_stratifikasi_4000": 0,
            "pct_stratifikasi_4000": 0.0,
            "n_populasi": n_unconstrained - len(core_set),
            "pct_populasi": float((n_unconstrained - len(core_set)) / n_pop * 100.0),
            "selisih_proporsi_strat_vs_pop_pp": np.nan,
            "catatan": (
                f"Sebelum batas 20.000: total {n_unconstrained:,} pinjaman; "
                f"setelah hapus perluasan non-S3: {n_after_drop_non_s3:,} pinjaman; final: {n_sel:,}"
            ),
        },
    ]

    for dim_label, col_name, order_vals in [
        ("Country", "Country", ["EE", "FI", "ES", "SK"]),
        ("OutcomeGroup", "outcome_group", OUTCOME_CATEGORIES),
        ("Era", "era", [ERA_PRA, ERA_PASCA]),
    ]:
        for val in order_vals:
            c_sel = int((sub_sel[col_name] == val).sum())
            c_str = int((sub_strat[col_name] == val).sum())
            c_pop = int((work[col_name] == val).sum())
            if c_pop == 0 and c_sel == 0:
                continue
            p_sel = c_sel / max(1, n_sel) * 100.0
            p_str = c_str / max(1, n_strat) * 100.0
            p_pop = c_pop / max(1, n_pop) * 100.0
            rows.append(
                {
                    "kategori": dim_label,
                    "subkategori": str(val),
                    "n_sampel_graf": c_sel,
                    "pct_sampel_graf": p_sel,
                    "n_stratifikasi_4000": c_str,
                    "pct_stratifikasi_4000": p_str,
                    "n_populasi": c_pop,
                    "pct_populasi": p_pop,
                    "selisih_proporsi_strat_vs_pop_pp": p_str - p_pop,
                    "catatan": "Perbandingan proporsi sampel stratifikasi vs populasi",
                }
            )

    for st_val in sorted(work["stratum"].unique()):
        c_sel = int((sub_sel["stratum"] == st_val).sum())
        c_str = int((sub_strat["stratum"] == st_val).sum())
        c_pop = int((work["stratum"] == st_val).sum())
        p_sel = c_sel / max(1, n_sel) * 100.0
        p_str = c_str / max(1, n_strat) * 100.0
        p_pop = c_pop / max(1, n_pop) * 100.0
        rows.append(
            {
                "kategori": "Strata (Country|Outcome|Era)",
                "subkategori": str(st_val),
                "n_sampel_graf": c_sel,
                "pct_sampel_graf": p_sel,
                "n_stratifikasi_4000": c_str,
                "pct_stratifikasi_4000": p_str,
                "n_populasi": c_pop,
                "pct_populasi": p_pop,
                "selisih_proporsi_strat_vs_pop_pp": p_str - p_pop,
                "catatan": "Diambil seluruhnya (< 20)" if c_pop < 20 else "Alokasi proporsional (>= 20)",
            }
        )

    return pd.DataFrame(rows)
