"""Diagnostic analytics functions for Tahap 3 (Bondora): hypothesis tests, IV/WoE, AUC, VIF, PSI, and Kaplan-Meier."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import (
    DEFAULT_12M_COL,
    DEFAULT_DATE_COL,
    LOAN_DATE_COL,
    PRINCIPAL_WRITEOFFS_COL,
    RANDOM_SEED,
    REPORT_DATE,
    STATUS_COL,
    STATUS_REPAID,
)


def holm_bonferroni(p_values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Apply step-down Holm-Bonferroni multiple-testing correction to a family of p-values.

    Parameters
    ----------
    p_values : array-like of float
        Raw p-values (may contain ``np.nan``).

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values in ``[0.0, 1.0]`` with ``p_holm >= p_value`` and
        monotonic ordering matching the sorted raw p-values.
    """
    p_arr = np.asarray(p_values, dtype="float64")
    out = np.full_like(p_arr, np.nan, dtype="float64")
    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]
    m = len(valid_p)
    if m == 0:
        return out

    order = np.argsort(valid_p, kind="mergesort")
    sorted_p = valid_p[order]
    mult = np.arange(m, 0, -1, dtype="float64")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * mult))
    adj_sorted = np.maximum(adj_sorted, sorted_p)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    out[valid_mask] = adj_sorted[inv_order]
    return out


def categorize_cramers_v(v: float) -> str:
    """Categorize Cramér's V effect size according to Tahap 3 conventions."""
    if np.isnan(v):
        return "-"
    if v < 0.10:
        return "Diabaikan (< 0,10)"
    if v <= 0.30:
        return "Kecil-sedang (0,10-0,30)"
    return "Kuat (> 0,30)"


def categorize_auc_univariate(auc_val: float) -> str:
    """Categorize symmetric univariate AUC effect size (in [0.5, 1.0])."""
    if np.isnan(auc_val):
        return "-"
    sym_auc = max(float(auc_val), 1.0 - float(auc_val))
    if sym_auc < 0.55:
        return "Lemah (0,50-0,55)"
    if sym_auc <= 0.65:
        return "Sedang (0,55-0,65)"
    return "Kuat (> 0,65)"


def categorize_iv(iv: float) -> str:
    """Categorize Information Value (IV) according to standard credit-scoring thresholds."""
    if np.isnan(iv):
        return "-"
    if iv < 0.02:
        return "Tidak berguna (< 0,02)"
    if iv < 0.10:
        return "Lemah (0,02-0,10)"
    if iv < 0.30:
        return "Sedang (0,10-0,30)"
    if iv <= 0.50:
        return "Kuat (0,30-0,50)"
    return "Mencurigakan (> 0,50)"


def categorize_psi(psi: float) -> str:
    """Categorize Population Stability Index (PSI) shift magnitude."""
    if np.isnan(psi):
        return "-"
    if psi < 0.10:
        return "Stabil (< 0,10)"
    if psi <= 0.25:
        return "Bergeser sedang (0,10-0,25)"
    return "Bergeser besar (> 0,25)"


def compute_cramers_v(
    x: pd.Series,
    y: pd.Series,
    min_expected: float = 5.0,
    bias_correction: bool = False,
) -> dict[str, Any]:
    """Compute Chi-square test of independence and Cramér's V between two categorical series.

    Categories in ``x`` whose minimum expected cell count is ``< min_expected``
    are merged into ``"Lainnya"`` before computing the final test.

    Parameters
    ----------
    x : pd.Series
        Categorical / grouping feature series.
    y : pd.Series
        Binary or categorical target series.
    min_expected : float, default 5.0
        Minimum expected cell count threshold for category merging. Set to 0 to disable.
    bias_correction : bool, default False
        If ``True``, returns Bergsma (2013) bias-corrected Cramér's V as ``cramers_v``;
        otherwise returns standard Cramér's V as ``cramers_v`` (both are always included
        in the returned dict under ``cramers_v_standard`` and ``cramers_v_corrected``).

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``n``, ``chi2``, ``p_value``, ``dof``, ``min_expected``,
        ``cramers_v``, ``cramers_v_standard``, ``cramers_v_corrected``,
        ``contingency_table``, ``expected_table``, and ``merged_categories``.
    """
    mask = x.notna() & y.notna()
    s_x = x.loc[mask].astype(str)
    s_y = y.loc[mask]
    n = int(len(s_x))
    if n == 0:
        raise ValueError("No valid non-null pairs found for compute_cramers_v.")

    ct_init = pd.crosstab(s_x, s_y)
    merged_cats: list[str] = []

    if min_expected > 0 and ct_init.shape[0] > 2 and ct_init.shape[1] >= 2:
        col_totals = ct_init.sum(axis=0).to_numpy(dtype="float64")
        min_col_prop = float(col_totals.min()) / float(n) if n > 0 else 0.0
        if min_col_prop > 0:
            row_totals = ct_init.sum(axis=1)
            small_rows = row_totals[row_totals * min_col_prop < min_expected].index.tolist()
            if small_rows and len(small_rows) < ct_init.shape[0]:
                # If only 1 row is small and merging it alone keeps it identical, merge with smallest remaining row
                if len(small_rows) == 1:
                    rem_rows = row_totals.drop(index=small_rows).sort_values()
                    if len(rem_rows) > 1:
                        small_rows.append(str(rem_rows.index[0]))
                merged_cats = [str(c) for c in small_rows]
                s_x = s_x.where(~s_x.isin(merged_cats), "Lainnya")

    ct = pd.crosstab(s_x, s_y)
    r, c = ct.shape
    if r < 2 or c < 2:
        return {
            "n": n,
            "chi2": 0.0,
            "p_value": 1.0,
            "dof": 0,
            "min_expected": float(ct.values.min()) if ct.size > 0 else 0.0,
            "cramers_v": 0.0,
            "cramers_v_standard": 0.0,
            "cramers_v_corrected": 0.0,
            "contingency_table": ct,
            "expected_table": ct.astype("float64"),
            "merged_categories": merged_cats,
        }

    chi2_val, p_val, dof, expected = stats.chi2_contingency(ct.to_numpy(), correction=False)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    min_exp_val = float(expected.min())

    denom_std = float(n * min(r - 1, c - 1))
    v_std = float(np.sqrt(max(0.0, chi2_val / denom_std))) if denom_std > 0 else 0.0
    v_std = min(1.0, v_std)

    if n > 1:
        phi2 = chi2_val / float(n)
        phi2_corr = max(0.0, phi2 - float((r - 1) * (c - 1)) / float(n - 1))
        r_corr = float(r) - float((r - 1) ** 2) / float(n - 1)
        c_corr = float(c) - float((c - 1) ** 2) / float(n - 1)
        denom_corr = min(r_corr - 1.0, c_corr - 1.0)
        v_corr = float(np.sqrt(phi2_corr / denom_corr)) if denom_corr > 0 else 0.0
        v_corr = min(1.0, v_corr)
    else:
        v_corr = v_std

    return {
        "n": n,
        "chi2": float(chi2_val),
        "p_value": float(p_val),
        "dof": int(dof),
        "min_expected": min_exp_val,
        "cramers_v": v_corr if bias_correction else v_std,
        "cramers_v_standard": v_std,
        "cramers_v_corrected": v_corr,
        "contingency_table": ct,
        "expected_table": expected_df,
        "merged_categories": merged_cats,
    }


def mannwhitney_auc(
    df: pd.DataFrame,
    col: str,
    target_col: str = DEFAULT_12M_COL,
) -> dict[str, Any]:
    """Compute Mann-Whitney U test and univariate AUC between default=1 and default=0.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Numeric feature column name.
    target_col : str, default DEFAULT_12M_COL
        Binary target column name (1 = default, 0 = non-default).

    Returns
    -------
    dict[str, Any]
        Dictionary with ``n``, ``n_default``, ``n_non_default``, ``median_default``,
        ``median_non_default``, ``mean_default``, ``mean_non_default``, ``u_stat``,
        ``p_value``, ``auc_raw``, ``auc_univariat``, ``rank_biserial``, and ``arah``.
    """
    for c_name in (col, target_col):
        if c_name not in df.columns:
            raise KeyError(f"Column '{c_name}' not found in DataFrame.")

    s_num = pd.to_numeric(df[col], errors="coerce")
    s_tgt = df[target_col]
    mask = s_num.notna() & s_tgt.notna()

    s_num_valid = s_num.loc[mask].astype("float64")
    s_tgt_valid = s_tgt.loc[mask].astype("int64")

    x_def = s_num_valid.loc[s_tgt_valid == 1].to_numpy()
    x_non = s_num_valid.loc[s_tgt_valid == 0].to_numpy()
    n1 = int(len(x_def))
    n0 = int(len(x_non))
    n_total = n1 + n0

    if n1 == 0 or n0 == 0:
        raise ValueError(f"Both classes must have at least 1 observation for '{col}'.")

    res = stats.mannwhitneyu(x_def, x_non, alternative="two-sided")
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    auc_raw = u_stat / (float(n1) * float(n0))
    auc_sym = max(auc_raw, 1.0 - auc_raw)
    rank_biserial = 2.0 * auc_raw - 1.0

    if abs(auc_raw - 0.5) < 1e-6:
        arah = "netral"
    elif auc_raw > 0.5:
        arah = "positif (nilai lebih tinggi berasosiasi dgn default)"
    else:
        arah = "negatif (nilai lebih rendah berasosiasi dgn default)"

    return {
        "n": n_total,
        "n_default": n1,
        "n_non_default": n0,
        "median_default": float(np.median(x_def)),
        "median_non_default": float(np.median(x_non)),
        "mean_default": float(np.mean(x_def)),
        "mean_non_default": float(np.mean(x_non)),
        "u_stat": u_stat,
        "p_value": p_val,
        "auc_raw": auc_raw,
        "auc_univariat": auc_sym,
        "rank_biserial": rank_biserial,
        "arah": arah,
    }


def run_hypothesis_tests(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = DEFAULT_12M_COL,
    min_expected: float = 5.0,
) -> pd.DataFrame:
    """Run Chi-square (categorical/bool) and Mann-Whitney U (numeric) tests with Holm correction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame filtered to observable rows (or containing ``target_col``).
    feature_cols : Sequence[str]
        Candidate feature column names (``boleh_jadi_fitur == 'Y'``).
    target_col : str, default DEFAULT_12M_COL
        Binary target column name.
    min_expected : float, default 5.0
        Minimum expected cell count for Chi-square category merging.

    Returns
    -------
    pd.DataFrame
        Table with columns: ``variabel``, ``tipe``, ``uji``, ``n``, ``statistik``,
        ``p_value``, ``p_holm``, ``ukuran_efek``, ``jenis_efek``, ``arah``, ``catatan``.
    """
    sub = df.loc[df[target_col].notna()].copy()
    rows: list[dict[str, Any]] = []

    for col in feature_cols:
        s = sub[col]
        is_cat_or_bool = (
            isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s.dtype)
            or pd.api.types.is_object_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype)
        )
        if is_cat_or_bool:
            res = compute_cramers_v(s, sub[target_col], min_expected=min_expected, bias_correction=False)
            # Determine direction description for categorical/binary
            ct = res["contingency_table"]
            if 1 in ct.columns:
                rates = (ct[1] / ct.sum(axis=1)).sort_values(ascending=False)
                top_cat = f"{rates.index[0]} ({rates.iloc[0]*100:.1f}%)"
                low_cat = f"{rates.index[-1]} ({rates.iloc[-1]*100:.1f}%)"
                arah_str = f"tertinggi: {top_cat}; terendah: {low_cat}"
            else:
                arah_str = "kategorikal (multi-kelompok)"

            merge_note = (
                f"digabung ke Lainnya: {', '.join(res['merged_categories'])}; "
                if res["merged_categories"]
                else ""
            )
            catatan_str = (
                f"{merge_note}df={res['dof']}, min_expected={res['min_expected']:.2f}, "
                f"V_bias_corr={res['cramers_v_corrected']:.4f} ({categorize_cramers_v(res['cramers_v'])})"
            )
            rows.append(
                {
                    "variabel": col,
                    "tipe": "kategorikal" if not pd.api.types.is_bool_dtype(s.dtype) else "boolean",
                    "uji": "Chi-square independensi",
                    "n": res["n"],
                    "statistik": res["chi2"],
                    "p_value": res["p_value"],
                    "p_holm": np.nan,
                    "ukuran_efek": res["cramers_v"],
                    "jenis_efek": "Cramér's V",
                    "arah": arah_str,
                    "catatan": catatan_str,
                    "_family": "cat",
                }
            )
        else:
            res = mannwhitney_auc(sub, col, target_col=target_col)
            catatan_str = (
                f"median default={res['median_default']:.4g} vs non-default={res['median_non_default']:.4g}; "
                f"AUC_raw={res['auc_raw']:.4f}, r_rb={res['rank_biserial']:+.4f} "
                f"({categorize_auc_univariate(res['auc_univariat'])})"
            )
            rows.append(
                {
                    "variabel": col,
                    "tipe": "numerik",
                    "uji": "Mann-Whitney U",
                    "n": res["n"],
                    "statistik": res["u_stat"],
                    "p_value": res["p_value"],
                    "p_holm": np.nan,
                    "ukuran_efek": res["auc_univariat"],
                    "jenis_efek": "AUC univariat",
                    "arah": res["arah"],
                    "catatan": catatan_str,
                    "_family": "num",
                }
            )

    out_df = pd.DataFrame(rows)
    for fam in ("cat", "num"):
        fam_mask = out_df["_family"] == fam
        if fam_mask.any():
            out_df.loc[fam_mask, "p_holm"] = holm_bonferroni(out_df.loc[fam_mask, "p_value"].to_numpy())

    out_df = out_df.drop(columns=["_family"]).sort_values(
        by=["jenis_efek", "ukuran_efek"], ascending=[True, False]
    ).reset_index(drop=True)
    return out_df


def compute_iv_woe(
    df: pd.DataFrame,
    col: str,
    target_col: str = DEFAULT_12M_COL,
    n_bins: int = 10,
    min_cat_n: int = 200,
    smoothing: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Compute Weight of Evidence (WoE) table and Information Value (IV) for a feature.

    Rules:
    - Numeric features: ``n_bins`` quantile bins (``duplicates="drop"``), with missing
      values placed in a separate ``"(Missing)"`` bin.
    - Categorical / boolean features: each category as a bin; categories with
      ``n < min_cat_n`` are merged into ``"Lainnya"``; missing values placed in ``"(Missing)"``.
    - Empty default/non-default cells receive ``smoothing`` (default 0.5) count adjustment.
    - ``WoE = ln(%non_default / %default)`` and ``IV = sum((%non_default - %default) * WoE)``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Feature column name.
    target_col : str, default DEFAULT_12M_COL
        Binary target column name (1 = default, 0 = non-default).
    n_bins : int, default 10
        Number of quantile bins for numeric features.
    min_cat_n : int, default 200
        Minimum sample size for categorical bins before merging into ``"Lainnya"``.
    smoothing : float, default 0.5
        Cell count smoothing applied when a bin has 0 defaults or 0 non-defaults.

    Returns
    -------
    tuple[float, pd.DataFrame]
        ``(total_iv, woe_table_df)``.
    """
    for c_name in (col, target_col):
        if c_name not in df.columns:
            raise KeyError(f"Column '{c_name}' not found in DataFrame.")

    mask = df[target_col].notna()
    s = df.loc[mask, col]
    y = df.loc[mask, target_col].astype("int64")

    is_cat_or_bool = (
        isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(s.dtype)
        or pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
    )

    if is_cat_or_bool:
        bins_str = s.astype("object").where(s.notna(), "(Missing)").astype(str)
        if min_cat_n > 0:
            vc = bins_str.value_counts(dropna=False)
            small_cats = set(vc[vc < min_cat_n].index) - {"(Missing)"}
            if small_cats:
                bins_str = bins_str.where(~bins_str.isin(small_cats), "Lainnya")
    else:
        s_num = pd.to_numeric(s, errors="coerce")
        valid_num = s_num.dropna()
        if len(valid_num) > 0 and valid_num.nunique() > 1:
            q_bins = pd.qcut(valid_num, q=min(n_bins, valid_num.nunique()), duplicates="drop")
            bins_str = pd.Series("(Missing)", index=s.index, dtype="object")
            bins_str.loc[valid_num.index] = q_bins.astype(str)
        elif len(valid_num) > 0:
            bins_str = pd.Series("(Missing)", index=s.index, dtype="object")
            bins_str.loc[valid_num.index] = str(valid_num.iloc[0])
        else:
            bins_str = pd.Series("(Missing)", index=s.index, dtype="object")

    tmp = pd.DataFrame({"bin": bins_str, "target": y}, index=s.index)
    agg = (
        tmp.groupby("bin", observed=True, sort=False)
        .agg(
            n=("target", "count"),
            n_default=("target", "sum"),
        )
        .reset_index()
    )
    agg["n_non_default"] = agg["n"] - agg["n_default"]

    total_def = float(agg["n_default"].sum())
    total_non = float(agg["n_non_default"].sum())
    if total_def == 0 or total_non == 0:
        agg["dist_non_default"] = np.nan
        agg["dist_default"] = np.nan
        agg["woe"] = 0.0
        agg["iv_bin"] = 0.0
        return 0.0, agg

    # Apply 0.5 smoothing on empty cells (where n_non_default == 0 or n_default == 0)
    has_zero = (agg["n_non_default"] == 0).any() or (agg["n_default"] == 0).any()
    if has_zero:
        c_non = np.where(agg["n_non_default"] == 0, smoothing, agg["n_non_default"].astype("float64"))
        c_def = np.where(agg["n_default"] == 0, smoothing, agg["n_default"].astype("float64"))
        p_non = c_non / float(c_non.sum())
        p_def = c_def / float(c_def.sum())
    else:
        p_non = agg["n_non_default"].astype("float64").to_numpy() / total_non
        p_def = agg["n_default"].astype("float64").to_numpy() / total_def

    woe = np.log(p_non / p_def)
    iv_bin = (p_non - p_def) * woe

    agg["default_rate"] = agg["n_default"] / agg["n"]
    agg["dist_non_default"] = p_non
    agg["dist_default"] = p_def
    agg["woe"] = woe
    agg["iv_bin"] = iv_bin

    total_iv = float(np.sum(iv_bin))
    return total_iv, agg


def compute_driver_ranking(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = DEFAULT_12M_COL,
    year_col: str = "loan_year",
    n_bins: int = 10,
    min_cat_n: int = 200,
) -> pd.DataFrame:
    """Compute overall and per-era IV, univariate AUC, and Cramér's V for a list of features.

    Era split:
    - ``era1``: ``loan_year`` in 2009..2017
    - ``era2``: ``loan_year`` in 2018..2020

    Returns
    -------
    pd.DataFrame
        Table with columns: ``fitur``, ``tipe``, ``n``, ``iv_semua``, ``iv_era1``,
        ``iv_era2``, ``kategori_iv``, ``auc_univariat``, ``cramers_v``, ``arah``, ``catatan_era``.
    """
    sub = df.loc[df[target_col].notna()].copy()
    sub_era1 = sub.loc[sub[year_col] <= 2017]
    sub_era2 = sub.loc[(sub[year_col] >= 2018) & (sub[year_col] <= 2020)]

    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        s = sub[col]
        n_valid = int(s.notna().sum())
        is_cat_or_bool = (
            isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s.dtype)
            or pd.api.types.is_object_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype)
        )

        iv_all, _ = compute_iv_woe(sub, col, target_col=target_col, n_bins=n_bins, min_cat_n=min_cat_n)
        iv_e1, _ = compute_iv_woe(sub_era1, col, target_col=target_col, n_bins=n_bins, min_cat_n=max(50, min_cat_n // 2))
        iv_e2, _ = compute_iv_woe(sub_era2, col, target_col=target_col, n_bins=n_bins, min_cat_n=max(50, min_cat_n // 2))

        cat_all = categorize_iv(iv_all)
        cat_e1 = categorize_iv(iv_e1)
        cat_e2 = categorize_iv(iv_e2)
        diff_era = abs(iv_e1 - iv_e2)

        if is_cat_or_bool:
            cv_res = compute_cramers_v(s, sub[target_col], min_expected=5.0)
            cramers_v_val = cv_res["cramers_v"]
            auc_val = np.nan
            ct = cv_res["contingency_table"]
            if 1 in ct.columns:
                rates = (ct[1] / ct.sum(axis=1)).sort_values(ascending=False)
                arah_str = f"tertinggi: {rates.index[0]} ({rates.iloc[0]*100:.1f}%)"
            else:
                arah_str = "kategorikal"
            tipe_str = "boolean" if pd.api.types.is_bool_dtype(s.dtype) else "kategorikal"
        else:
            mw_res = mannwhitney_auc(sub, col, target_col=target_col)
            auc_val = mw_res["auc_univariat"]
            cramers_v_val = np.nan
            arah_str = mw_res["arah"]
            tipe_str = "numerik"

        notes: list[str] = []
        if iv_all > 0.50:
            notes.append("IV > 0,50 (mencurigakan: periksa penanda era/negara)")
        if diff_era > 0.10 or cat_e1 != cat_e2:
            notes.append(f"Bergeser antar-era (|delta_IV|={diff_era:.4f}: {cat_e1} -> {cat_e2})")
        else:
            notes.append(f"Stabil antar-era (|delta_IV|={diff_era:.4f})")

        rows.append(
            {
                "fitur": col,
                "tipe": tipe_str,
                "n": n_valid,
                "iv_semua": iv_all,
                "iv_era1": iv_e1,
                "iv_era2": iv_e2,
                "kategori_iv": cat_all,
                "auc_univariat": auc_val,
                "cramers_v": cramers_v_val,
                "arah": arah_str,
                "catatan_era": "; ".join(notes),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("iv_semua", ascending=False).reset_index(drop=True)
    return out_df


def bootstrap_roc_auc(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_boot: int = 200,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    """Compute ROC-AUC and 95% percentile bootstrap confidence interval.

    Parameters
    ----------
    y_true : array-like
        Binary true labels (0 or 1).
    y_score : array-like
        Continuous predicted probability or risk score.
    n_boot : int, default 200
        Number of bootstrap resamples.
    seed : int, default RANDOM_SEED
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float | int]
        Dictionary with ``n``, ``n_default``, ``roc_auc``, ``ci_low``, ``ci_high``, ``std_boot``.
    """
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    ys = pd.to_numeric(pd.Series(y_score), errors="coerce")
    mask = yt.notna() & ys.notna()

    y_arr = yt.loc[mask].to_numpy(dtype="int64")
    s_arr = ys.loc[mask].to_numpy(dtype="float64")
    n = int(len(y_arr))
    n_def = int(y_arr.sum())

    if n == 0 or n_def == 0 or n_def == n:
        return {
            "n": n,
            "n_default": n_def,
            "roc_auc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std_boot": np.nan,
        }

    auc_point = float(roc_auc_score(y_arr, s_arr))
    rng = np.random.default_rng(seed)
    boot_aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        if 0 < yb.sum() < n:
            boot_aucs.append(float(roc_auc_score(yb, s_arr[idx])))

    boot_arr = np.asarray(boot_aucs, dtype="float64")
    return {
        "n": n,
        "n_default": n_def,
        "roc_auc": auc_point,
        "ci_low": float(np.percentile(boot_arr, 2.5)),
        "ci_high": float(np.percentile(boot_arr, 97.5)),
        "std_boot": float(np.std(boot_arr, ddof=1)),
    }


def compute_vif_table(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for numeric features after temporary median imputation.

    Uses the inverse/pseudoinverse of the standardized correlation matrix R:
    ``VIF_j = diag(R^{-1})_j`` (mathematically identical to ``1 / (1 - R_j^2)``
    from OLS regression with an intercept, while being fast and numerically stable).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : Sequence[str]
        Numeric feature column names.

    Returns
    -------
    pd.DataFrame
        Table with columns: ``fitur``, ``n_terisi``, ``pct_missing_diisi_median_sementara``,
        ``median_imputasi_sementara``, ``vif``, ``r2_multivariat``, ``kategori_vif``.
    """
    work = pd.DataFrame(index=df.index)
    meta_rows: list[dict[str, Any]] = []

    for col in numeric_cols:
        s_num = pd.to_numeric(df[col], errors="coerce").astype("float64")
        n_valid = int(s_num.notna().sum())
        pct_miss = float(s_num.isna().mean() * 100.0)
        med_val = float(s_num.median()) if n_valid > 0 else 0.0
        work[col] = s_num.fillna(med_val)
        meta_rows.append(
            {
                "fitur": col,
                "n_terisi": n_valid,
                "pct_missing_diisi_median_sementara": round(pct_miss, 4),
                "median_imputasi_sementara": med_val,
            }
        )

    x_mat = work.to_numpy(dtype="float64")
    means = x_mat.mean(axis=0)
    stds = x_mat.std(axis=0, ddof=1)
    stds_safe = np.where(stds > 0, stds, 1.0)
    z_mat = (x_mat - means) / stds_safe

    # OLS with intercept via QR / lstsq or correlation matrix inverse
    n_obs, p_cols = z_mat.shape
    vifs = np.ones(p_cols, dtype="float64")
    r2s = np.zeros(p_cols, dtype="float64")

    for j in range(p_cols):
        if stds[j] == 0:
            vifs[j] = np.nan
            r2s[j] = np.nan
            continue
        y_j = z_mat[:, j]
        x_other = np.delete(z_mat, j, axis=1)
        beta, _, _, _ = np.linalg.lstsq(x_other, y_j, rcond=None)
        y_hat = x_other @ beta
        ss_res = float(np.sum((y_j - y_hat) ** 2))
        ss_tot = float(np.sum(y_j ** 2))
        r2_j = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
        r2s[j] = r2_j
        vifs[j] = float("inf") if r2_j >= 1.0 - 1e-12 else 1.0 / (1.0 - r2_j)

    out_df = pd.DataFrame(meta_rows)
    out_df["vif"] = vifs
    out_df["r2_multivariat"] = r2s

    def _cat_vif(v: float) -> str:
        if np.isnan(v):
            return "-"
        if v < 5.0:
            return "Rendah (< 5)"
        if v < 10.0:
            return "Sedang (5-10)"
        return "Tinggi (>= 10, multikolinearitas kuat)"

    out_df["kategori_vif"] = [_cat_vif(v) for v in vifs]
    return out_df.sort_values("vif", ascending=False).reset_index(drop=True)


def compute_psi(
    ref_series: pd.Series,
    comp_series: pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, pd.DataFrame]:
    """Compute Population Stability Index (PSI) of ``comp_series`` relative to ``ref_series``.

    For numeric variables, decile bin cutpoints are learned strictly from ``ref_series``
    (with ``-inf`` and ``+inf`` outer bounds), and missing values are placed in a separate
    ``"(Missing)"`` bin when present. For categorical/boolean variables, fixed category
    levels are compared directly.

    Parameters
    ----------
    ref_series : pd.Series
        Reference baseline distribution (e.g., 2009-2016 cohort).
    comp_series : pd.Series
        Comparison target distribution (e.g., 2017 or 2018-2020 cohort).
    n_bins : int, default 10
        Number of quantile bins for numeric features.
    epsilon : float, default 1e-4
        Floor proportion replacing zero bin proportions to prevent ``log(0)``.

    Returns
    -------
    tuple[float, pd.DataFrame]
        ``(total_psi, psi_bin_table)``.
    """
    n_ref = len(ref_series)
    n_comp = len(comp_series)
    if n_ref == 0 or n_comp == 0:
        raise ValueError("Both ref_series and comp_series must be non-empty.")

    is_cat_or_bool = (
        isinstance(ref_series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(ref_series.dtype)
        or pd.api.types.is_object_dtype(ref_series.dtype)
        or pd.api.types.is_string_dtype(ref_series.dtype)
    )

    has_missing = bool(ref_series.isna().any() or comp_series.isna().any())

    if is_cat_or_bool:
        r_str = ref_series.astype("object").where(ref_series.notna(), "(Missing)").astype(str)
        c_str = comp_series.astype("object").where(comp_series.notna(), "(Missing)").astype(str)
        all_bins = list(dict.fromkeys(list(r_str.unique()) + list(c_str.unique())))
        r_counts = r_str.value_counts().reindex(all_bins, fill_value=0)
        c_counts = c_str.value_counts().reindex(all_bins, fill_value=0)
    else:
        r_num = pd.to_numeric(ref_series, errors="coerce")
        c_num = pd.to_numeric(comp_series, errors="coerce")
        r_valid = r_num.dropna().to_numpy(dtype="float64")

        if len(r_valid) > 0 and len(np.unique(r_valid)) > 1:
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(r_valid, qs))
            edges[0] = -np.inf
            edges[-1] = np.inf
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
        else:
            edges = np.array([-np.inf, np.inf])

        r_binned = pd.cut(r_num, bins=edges, include_lowest=True)
        c_binned = pd.cut(c_num, bins=edges, include_lowest=True)
        bin_labels = [str(cat) for cat in r_binned.cat.categories]

        r_str = r_binned.astype("object").where(r_num.notna(), "(Missing)").astype(str)
        c_str = c_binned.astype("object").where(c_num.notna(), "(Missing)").astype(str)

        all_bins = bin_labels + (["(Missing)"] if has_missing else [])
        r_counts = r_str.value_counts().reindex(all_bins, fill_value=0)
        c_counts = c_str.value_counts().reindex(all_bins, fill_value=0)

    p_raw = r_counts.to_numpy(dtype="float64") / float(n_ref)
    q_raw = c_counts.to_numpy(dtype="float64") / float(n_comp)

    p_safe = np.where(p_raw == 0.0, epsilon, p_raw)
    q_safe = np.where(q_raw == 0.0, epsilon, q_raw)

    psi_bins = (q_safe - p_safe) * np.log(q_safe / p_safe)
    # Exact 0 when raw proportions are identical
    psi_bins = np.where(np.isclose(p_raw, q_raw, atol=1e-15), 0.0, psi_bins)

    tbl = pd.DataFrame(
        {
            "bin": all_bins,
            "n_ref": r_counts.to_numpy(dtype="int64"),
            "n_comp": c_counts.to_numpy(dtype="int64"),
            "prop_ref": p_raw,
            "prop_comp": q_raw,
            "psi_bin": psi_bins,
        }
    )
    return float(np.sum(psi_bins)), tbl


def prepare_survival_data(
    df: pd.DataFrame,
    report_date: str | pd.Timestamp = REPORT_DATE,
    days_per_month: float = 30.4375,
) -> pd.DataFrame:
    """Construct time-to-default survival dataset excluding unknown-date write-offs.

    Rules (Langkah 9.1):
    - Exclude the 448 ``Repaid`` loans without ``DefaultDate`` that have ``PrincipalWriteOffs > 0``
      and any rows with ``DefaultDate < LoanDate`` or missing ``LoanDate``.
    - ``event = 1`` if ``DefaultDate`` is present; ``duration_months = (DefaultDate - LoanDate) / days_per_month``.
    - ``event = 0`` (censored) otherwise:
      ``end_date = ContractEndDate`` when ``Status == 'Repaid'`` and ``ContractEndDate`` is present
      and ``<= report_date``, else ``report_date``.
      ``duration_months = (end_date - LoanDate) / days_per_month``.

    Returns
    -------
    pd.DataFrame
        Copy of valid subset with appended columns ``event`` (int64),
        ``duration_days`` (float64), and ``duration_months`` (float64).
    """
    loan_dt = pd.to_datetime(df[LOAN_DATE_COL], errors="coerce", format="mixed")
    def_dt = pd.to_datetime(df[DEFAULT_DATE_COL], errors="coerce", format="mixed")
    cend_dt = (
        pd.to_datetime(df["ContractEndDate"], errors="coerce", format="mixed")
        if "ContractEndDate" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    rep_dt = pd.Timestamp(report_date)

    has_def = def_dt.notna()
    invalid_def = has_def & loan_dt.notna() & (def_dt < loan_dt)
    is_repaid = df[STATUS_COL].astype(str).str.strip() == STATUS_REPAID

    if PRINCIPAL_WRITEOFFS_COL in df.columns:
        pw = pd.to_numeric(df[PRINCIPAL_WRITEOFFS_COL], errors="coerce").fillna(0.0)
        repaid_wo = is_repaid & (~has_def) & (pw > 0.0)
    else:
        repaid_wo = pd.Series(False, index=df.index)

    valid_mask = loan_dt.notna() & (~repaid_wo) & (~invalid_def)
    sub = df.loc[valid_mask].copy()

    l_dt = loan_dt.loc[valid_mask]
    d_dt = def_dt.loc[valid_mask]
    c_dt = cend_dt.loc[valid_mask]
    rep_mask = is_repaid.loc[valid_mask]

    event = d_dt.notna().astype("int64")

    # End date for censored observations: ContractEndDate if Repaid and valid (<= rep_dt and >= l_dt), else rep_dt
    use_cend = (~d_dt.notna()) & rep_mask & c_dt.notna() & (c_dt >= l_dt) & (c_dt <= rep_dt)
    end_dt = pd.Series(rep_dt, index=sub.index, dtype="datetime64[ns]")
    end_dt.loc[d_dt.notna()] = d_dt.loc[d_dt.notna()].astype("datetime64[ns]")
    end_dt.loc[use_cend] = c_dt.loc[use_cend].astype("datetime64[ns]")

    dur_days = (end_dt - l_dt.astype("datetime64[ns]")).dt.total_seconds() / 86400.0
    dur_days = np.maximum(0.0, dur_days.to_numpy(dtype="float64"))
    dur_months = dur_days / float(days_per_month)

    sub["event"] = event.to_numpy(dtype="int64")
    sub["duration_days"] = dur_days
    sub["duration_months"] = dur_months
    return sub


def kaplan_meier_curve(
    durations: Sequence[float] | pd.Series | np.ndarray,
    events: Sequence[int] | pd.Series | np.ndarray,
    max_horizon: float | None = 36.0,
) -> pd.DataFrame:
    """Compute product-limit Kaplan-Meier survival and cumulative incidence curve.

    Parameters
    ----------
    durations : array-like of float
        Observed event or censoring durations (e.g., in months).
    events : array-like of int
        Event indicator (``1`` = event/default, ``0`` = right-censored).
    max_horizon : float or None, default 36.0
        If provided, truncates output curve table to ``time <= max_horizon``.

    Returns
    -------
    pd.DataFrame
        Curve table starting at ``time = 0.0`` with columns: ``time``, ``n_at_risk``,
        ``n_events``, ``n_censored``, ``survival``, ``cumulative_incidence``,
        ``ci_low_incidence``, ``ci_high_incidence``.
    """
    t_arr = np.asarray(durations, dtype="float64")
    e_arr = np.asarray(events, dtype="int64")
    valid = ~np.isnan(t_arr) & ~np.isnan(e_arr)
    t_arr = t_arr[valid]
    e_arr = e_arr[valid]

    n_total = len(t_arr)
    if n_total == 0:
        raise ValueError("Cannot compute Kaplan-Meier curve on empty array.")

    uniq_times = np.unique(t_arr)
    if max_horizon is not None:
        uniq_times = uniq_times[uniq_times <= float(max_horizon)]

    rows: list[dict[str, float | int]] = [
        {
            "time": 0.0,
            "n_at_risk": int(n_total),
            "n_events": 0,
            "n_censored": 0,
            "survival": 1.0,
            "cumulative_incidence": 0.0,
            "ci_low_incidence": 0.0,
            "ci_high_incidence": 0.0,
        }
    ]

    surv = 1.0
    greenwood_sum = 0.0
    z = 1.959963984540054

    for t in uniq_times:
        if t <= 0.0:
            # Handle any events/censoring at t == 0
            at_t = t_arr == t
            d_j = int(np.sum(at_t & (e_arr == 1)))
            c_j = int(np.sum(at_t & (e_arr == 0)))
            if d_j > 0:
                n_j = int(np.sum(t_arr >= t))
                surv *= 1.0 - float(d_j) / float(n_j)
                if n_j > d_j:
                    greenwood_sum += float(d_j) / (float(n_j) * float(n_j - d_j))
                se_s = surv * np.sqrt(greenwood_sum)
                inc = 1.0 - surv
                rows[0] = {
                    "time": 0.0,
                    "n_at_risk": n_j,
                    "n_events": d_j,
                    "n_censored": c_j,
                    "survival": surv,
                    "cumulative_incidence": inc,
                    "ci_low_incidence": float(np.clip(inc - z * se_s, 0.0, 1.0)),
                    "ci_high_incidence": float(np.clip(inc + z * se_s, 0.0, 1.0)),
                }
            continue

        n_j = int(np.sum(t_arr >= t))
        if n_j == 0:
            break
        at_t = t_arr == t
        d_j = int(np.sum(at_t & (e_arr == 1)))
        c_j = int(np.sum(at_t & (e_arr == 0)))

        if d_j > 0:
            surv *= 1.0 - float(d_j) / float(n_j)
            if n_j > d_j:
                greenwood_sum += float(d_j) / (float(n_j) * float(n_j - d_j))

        se_s = surv * np.sqrt(greenwood_sum)
        inc = 1.0 - surv
        rows.append(
            {
                "time": float(t),
                "n_at_risk": n_j,
                "n_events": d_j,
                "n_censored": c_j,
                "survival": float(surv),
                "cumulative_incidence": float(inc),
                "ci_low_incidence": float(np.clip(inc - z * se_s, 0.0, 1.0)),
                "ci_high_incidence": float(np.clip(inc + z * se_s, 0.0, 1.0)),
            }
        )

    return pd.DataFrame(rows)


def km_summary_at_horizons(
    durations: Sequence[float] | pd.Series | np.ndarray,
    events: Sequence[int] | pd.Series | np.ndarray,
    horizons: Sequence[float] = (6.0, 12.0, 24.0, 36.0),
) -> pd.DataFrame:
    """Evaluate Kaplan-Meier number at risk and cumulative incidence at fixed horizons.

    Parameters
    ----------
    durations : array-like of float
        Observed durations in months.
    events : array-like of int
        Event indicators (1 = default, 0 = censored).
    horizons : Sequence[float], default (6.0, 12.0, 24.0, 36.0)
        Target time points in months.

    Returns
    -------
    pd.DataFrame
        Summary rows per horizon with ``horizon_m``, ``n_total``, ``n_at_risk``,
        ``cum_events``, ``survival``, ``cumulative_incidence``, ``ci_low_incidence``,
        ``ci_high_incidence``.
    """
    t_arr = np.asarray(durations, dtype="float64")
    e_arr = np.asarray(events, dtype="int64")
    max_h = max(horizons)
    curve = kaplan_meier_curve(t_arr, e_arr, max_horizon=max_h)

    rows: list[dict[str, float | int]] = []
    for h in horizons:
        sub_c = curve.loc[curve["time"] <= float(h)]
        last_row = sub_c.iloc[-1]
        n_at_risk_h = int(np.sum(t_arr > float(h)))
        cum_ev_h = int(np.sum((t_arr <= float(h)) & (e_arr == 1)))
        rows.append(
            {
                "horizon_m": int(h) if float(h).is_integer() else float(h),
                "n_total": int(len(t_arr)),
                "n_at_risk": n_at_risk_h,
                "cum_events": cum_ev_h,
                "survival": float(last_row["survival"]),
                "cumulative_incidence": float(last_row["cumulative_incidence"]),
                "ci_low_incidence": float(last_row["ci_low_incidence"]),
                "ci_high_incidence": float(last_row["ci_high_incidence"]),
            }
        )
    return pd.DataFrame(rows)


def multigroup_logrank_test(
    durations: Sequence[float] | pd.Series | np.ndarray,
    events: Sequence[int] | pd.Series | np.ndarray,
    groups: Sequence[Any] | pd.Series | np.ndarray,
    max_horizon: float | None = 36.0,
) -> dict[str, Any]:
    """Perform multi-group Mantel-Haenszel log-rank test across survival groups.

    Parameters
    ----------
    durations : array-like of float
        Observed durations.
    events : array-like of int
        Event indicators (1 = event, 0 = censored).
    groups : array-like
        Group labels for each subject.
    max_horizon : float or None, default 36.0
        If set, administratively censors observations after ``max_horizon`` so the
        test matches the plotted horizon window.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``chi2_stat``, ``dof``, ``p_value``, ``n_groups``, ``n_total``.
    """
    t_arr = np.asarray(durations, dtype="float64").copy()
    e_arr = np.asarray(events, dtype="int64").copy()
    g_ser = pd.Series(groups)

    valid = (~np.isnan(t_arr)) & (~np.isnan(e_arr)) & g_ser.notna().to_numpy()
    t_arr = t_arr[valid]
    e_arr = e_arr[valid]
    g_arr = g_ser.loc[valid].astype(str).to_numpy()

    if max_horizon is not None:
        beyond = t_arr > float(max_horizon)
        t_arr[beyond] = float(max_horizon)
        e_arr[beyond] = 0

    uniq_groups = np.unique(g_arr)
    k_groups = len(uniq_groups)
    if k_groups < 2:
        raise ValueError("multigroup_logrank_test requires at least 2 distinct groups.")

    # Round durations to 4 decimals (approx ~4 minutes) to group simultaneous daily events cleanly
    t_round = np.round(t_arr, 6)
    event_times = np.unique(t_round[e_arr == 1])

    if len(event_times) == 0:
        return {
            "chi2_stat": 0.0,
            "dof": k_groups - 1,
            "p_value": 1.0,
            "n_groups": k_groups,
            "n_total": int(len(t_arr)),
        }

    # Pre-index group membership matrices (N x K)
    g_ind = np.column_stack([g_arr == g for g in uniq_groups])

    # Sort by time descending to compute cumulative risk sets in O(N + M*K)
    order_desc = np.argsort(-t_round, kind="mergesort")
    t_sorted = t_round[order_desc]
    e_sorted = e_arr[order_desc]
    g_sorted = g_ind[order_desc]

    obs_minus_exp = np.zeros(k_groups, dtype="float64")
    cov_mat = np.zeros((k_groups, k_groups), dtype="float64")

    # Since event_times has at most ~3,000 unique daily values, aggregate by unique time
    df_ev = pd.DataFrame(g_sorted, columns=uniq_groups)
    df_ev["_t"] = t_sorted
    df_ev["_e"] = e_sorted

    # Group by _t ascending
    grp_t = df_ev.groupby("_t", sort=True)
    n_by_t = grp_t[list(uniq_groups)].sum().to_numpy(dtype="float64")
    # events at each _t per group
    ev_only = df_ev.loc[df_ev["_e"] == 1].groupby("_t", sort=True)[list(uniq_groups)].sum()
    t_index = grp_t.size().index
    d_by_t = ev_only.reindex(t_index, fill_value=0).to_numpy(dtype="float64")

    # Risk set at time t_j is reverse cumulative sum of n_by_t
    risk_by_t = np.cumsum(n_by_t[::-1], axis=0)[::-1]

    for j in range(len(t_index)):
        d_vec = d_by_t[j]
        d_tot = float(d_vec.sum())
        if d_tot <= 0.0:
            continue
        n_vec = risk_by_t[j]
        n_tot = float(n_vec.sum())
        if n_tot <= 1.0:
            continue

        p_vec = n_vec / n_tot
        e_vec = d_tot * p_vec
        obs_minus_exp += d_vec - e_vec

        factor = d_tot * (n_tot - d_tot) / (n_tot - 1.0)
        cov_j = factor * (np.diag(p_vec) - np.outer(p_vec, p_vec))
        cov_mat += cov_j

    z_sub = obs_minus_exp[:-1]
    cov_sub = cov_mat[:-1, :-1]
    chi2_stat = float(z_sub.T @ np.linalg.pinv(cov_sub) @ z_sub)
    chi2_stat = max(0.0, chi2_stat)
    dof = k_groups - 1
    p_val = float(stats.chi2.sf(chi2_stat, dof))

    return {
        "chi2_stat": chi2_stat,
        "dof": int(dof),
        "p_value": p_val,
        "n_groups": int(k_groups),
        "n_total": int(len(t_arr)),
    }
