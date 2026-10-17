"""Pure aggregation and descriptive statistics functions for Tahap 2 (Bondora)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COL,
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    INTEREST_PENALTY_PAYMENTS_COL,
    IS_RESOLVED_COL,
    PRINCIPAL_PAYMENTS_COL,
    REGRESSION_TARGET_COL,
    STATUS_COL,
    STATUS_CURRENT,
    STATUS_LATE,
)
from src.data.targets import make_default_label, make_regression_target

OUTCOME_CATEGORIES: list[str] = [
    "Default",
    "Lunas tanpa default",
    "Berjalan (Current)",
    "Terlambat belum default",
]


def make_outcome_group(df: pd.DataFrame) -> pd.Series:
    """Classify every loan row into exactly one of the 4 mutually exclusive outcome groups.

    Groups:
    1. ``"Default"``: ``default == 1``
    2. ``"Lunas tanpa default"``: ``default == 0``
    3. ``"Berjalan (Current)"``: ``Status == 'Current'`` and ``DefaultDate`` is missing (``default`` is NA)
    4. ``"Terlambat belum default"``: ``Status == 'Late'`` and ``DefaultDate`` is missing (``default`` is NA)

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame containing ``Status`` and `` default`` (or ``DefaultDate``).

    Returns
    -------
    pd.Series
        Ordered Categorical Series named ``outcome_group`` with no missing values.
    """
    if DEFAULT_TARGET_COL in df.columns:
        s_def = df[DEFAULT_TARGET_COL]
    else:
        s_def = make_default_label(df)

    status_str = df[STATUS_COL].astype(str).str.strip()
    is_def_1 = s_def.notna() & (s_def == 1)
    is_def_0 = s_def.notna() & (s_def == 0)
    is_unresolved = s_def.isna()

    is_current_unres = is_unresolved & (status_str == STATUS_CURRENT)
    is_late_unres = is_unresolved & (status_str == STATUS_LATE)

    result = pd.Series(index=df.index, dtype="object", name="outcome_group")
    result.loc[is_def_1] = "Default"
    result.loc[is_def_0] = "Lunas tanpa default"
    result.loc[is_current_unres] = "Berjalan (Current)"
    result.loc[is_late_unres] = "Terlambat belum default"

    if result.isna().any():
        n_unmapped = int(result.isna().sum())
        raise ValueError(f"make_outcome_group found {n_unmapped} rows that did not map to any outcome group.")

    return pd.Series(
        pd.Categorical(result, categories=OUTCOME_CATEGORIES, ordered=True),
        index=df.index,
        name="outcome_group",
    )


def numeric_summary(
    df: pd.DataFrame,
    cols: Sequence[str],
    resolved_only_cols: Sequence[str] = (REGRESSION_TARGET_COL,),
) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics for numeric columns.

    For columns listed in ``resolved_only_cols`` (by default ``return_ratio``),
    statistics are computed strictly on ``is_resolved == True`` rows when
    ``is_resolved`` is present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : Sequence[str]
        List of numeric column names to summarize.
    resolved_only_cols : Sequence[str], default ("return_ratio",)
        Columns whose outcome statistics must be restricted to resolved loans.

    Returns
    -------
    pd.DataFrame
        Summary table with count, missing, mean, std, min, percentiles
        (1, 5, 25, 50, 75, 95, 99), max, skewness, and kurtosis.
    """
    resolved_set = set(resolved_only_cols)
    rows: list[dict[str, Any]] = []

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        if col in resolved_set and IS_RESOLVED_COL in df.columns:
            sub_series = df.loc[df[IS_RESOLVED_COL] == True, col]  # noqa: E712
            pop_label = "is_resolved == True"
        else:
            sub_series = df[col]
            pop_label = "seluruh_portofolio"

        n_total_pop = len(sub_series)
        s_num = pd.to_numeric(sub_series, errors="coerce").astype("float64")
        s_valid = s_num.dropna()
        n_valid = len(s_valid)
        n_missing = n_total_pop - n_valid
        pct_missing = (n_missing / n_total_pop * 100.0) if n_total_pop > 0 else 0.0

        if n_valid > 0:
            q = s_valid.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            rows.append(
                {
                    "nama_kolom": col,
                    "populasi": pop_label,
                    "n": n_valid,
                    "missing": n_missing,
                    "pct_missing": round(pct_missing, 4),
                    "mean": round(float(s_valid.mean()), 4),
                    "std": round(float(s_valid.std(ddof=1)), 4) if n_valid > 1 else 0.0,
                    "min": round(float(s_valid.min()), 4),
                    "p01": round(float(q.loc[0.01]), 4),
                    "p05": round(float(q.loc[0.05]), 4),
                    "p25": round(float(q.loc[0.25]), 4),
                    "p50": round(float(q.loc[0.50]), 4),
                    "p75": round(float(q.loc[0.75]), 4),
                    "p95": round(float(q.loc[0.95]), 4),
                    "p99": round(float(q.loc[0.99]), 4),
                    "max": round(float(s_valid.max()), 4),
                    "skewness": round(float(s_valid.skew()), 4) if n_valid > 2 else np.nan,
                    "kurtosis": round(float(s_valid.kurt()), 4) if n_valid > 3 else np.nan,
                }
            )
        else:
            rows.append(
                {
                    "nama_kolom": col,
                    "populasi": pop_label,
                    "n": 0,
                    "missing": n_missing,
                    "pct_missing": round(pct_missing, 4),
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                }
            )

    return pd.DataFrame(rows)


def freq_table(
    df: pd.DataFrame,
    col: str,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute descending frequency table with count, percentage, and cumulative percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Categorical or discrete column name.
    dropna : bool, default False
        If ``False``, missing values are included under label ``"(Missing)"``
        so percentages sum to 100% across all rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``variabel``, ``nilai``, ``jumlah``, ``persen``,
        ``persen_kumulatif``.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    s = df[col]
    if dropna:
        s_clean = s.dropna().astype(str)
    else:
        s_clean = s.astype("object").where(s.notna(), "(Missing)").astype(str)

    total = len(s_clean)
    vc = s_clean.value_counts(dropna=False)

    out = pd.DataFrame(
        {
            "variabel": col,
            "nilai": vc.index.astype(str),
            "jumlah": vc.values.astype("int64"),
        }
    )
    if total > 0:
        out["persen"] = out["jumlah"] / total * 100.0
        out["persen_kumulatif"] = out["persen"].cumsum()
    else:
        out["persen"] = 0.0
        out["persen_kumulatif"] = 0.0

    return out.reset_index(drop=True)


def wilson_ci(
    k: pd.Series | np.ndarray | int,
    n: pd.Series | np.ndarray | int,
    z: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 95% Wilson score confidence interval for binomial proportions in [0, 1].

    Parameters
    ----------
    k : array-like or int
        Number of positive events (defaults).
    n : array-like or int
        Number of trials (resolved loans).
    z : float, default 1.959963984540054
        Normal critical value for 95% confidence level.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ci_low, ci_high)`` clipped to ``[0.0, 1.0]``.
    """
    k_arr = np.asarray(k, dtype="float64")
    n_arr = np.asarray(n, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_arr > 0, k_arr / n_arr, np.nan)
        z2 = z * z
        denom = 1.0 + z2 / n_arr
        center = (p + z2 / (2.0 * n_arr)) / denom
        rad = np.sqrt(np.maximum(0.0, (p * (1.0 - p) / n_arr) + (z2 / (4.0 * n_arr * n_arr))))
        half = (z * rad) / denom
        ci_low = np.clip(center - half, 0.0, 1.0)
        ci_high = np.clip(center + half, 0.0, 1.0)
        # Ensure exact bracketing against floating-point rounding at boundary 0 or 1
        ci_low = np.minimum(ci_low, p)
        ci_high = np.maximum(ci_high, p)

    return ci_low, ci_high


def default_rate_by(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str = AMOUNT_COL,
    min_n: int = 200,
) -> pd.DataFrame:
    """Compute default rate, 95% Wilson CI, and amount-weighted default rate on resolved loans.

    Strictly filters to rows where ``default`` is non-NA (and ``is_resolved == True``
    if present). Groups with ``n < min_n`` are merged into ``"Lainnya"``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_col : str
        Column name to segment by.
    weight_col : str, default "Amount"
        Column name used for amount-weighted default rate.
    min_n : int, default 200
        Minimum sample size ``n`` required to keep a segment separate; smaller
        segments are combined into ``"Lainnya"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``variabel``, ``segmen``, ``n``, ``n_default``,
        ``default_rate``, ``ci_low``, ``ci_high``, ``default_rate_by_amount``.
        All rates and CI bounds are proportions in ``[0, 1]``.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")

    if DEFAULT_TARGET_COL in df.columns:
        s_def = df[DEFAULT_TARGET_COL]
    else:
        s_def = make_default_label(df)

    mask_res = s_def.notna()
    if IS_RESOLVED_COL in df.columns:
        mask_res = mask_res & (df[IS_RESOLVED_COL] == True)  # noqa: E712

    sub = df.loc[mask_res].copy()
    sub["_def_int"] = s_def.loc[mask_res].astype("int64")

    raw_grp = sub[group_col]
    grp_str = raw_grp.astype("object").where(raw_grp.notna(), "(Missing)").astype(str)

    # Identify small groups (< min_n) and merge into 'Lainnya'
    counts = grp_str.value_counts(dropna=False)
    small_cats = set(counts[counts < min_n].index)
    if small_cats:
        grp_str = grp_str.where(~grp_str.isin(small_cats), "Lainnya")

    sub["_segmen"] = grp_str

    has_weight = weight_col in sub.columns
    if has_weight:
        sub["_weight"] = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0.0)
        sub["_def_weight"] = np.where(sub["_def_int"] == 1, sub["_weight"], 0.0)
        agg = (
            sub.groupby("_segmen", observed=True, sort=False)
            .agg(
                n=("_def_int", "count"),
                n_default=("_def_int", "sum"),
                total_weight=("_weight", "sum"),
                def_weight=("_def_weight", "sum"),
            )
            .reset_index()
        )
    else:
        agg = (
            sub.groupby("_segmen", observed=True, sort=False)
            .agg(
                n=("_def_int", "count"),
                n_default=("_def_int", "sum"),
            )
            .reset_index()
        )

    agg = agg.rename(columns={"_segmen": "segmen"})
    agg["variabel"] = group_col
    agg["n"] = agg["n"].astype("int64")
    agg["n_default"] = agg["n_default"].astype("int64")
    agg["default_rate"] = np.where(agg["n"] > 0, agg["n_default"] / agg["n"], np.nan)

    ci_low, ci_high = wilson_ci(agg["n_default"].to_numpy(), agg["n"].to_numpy())
    agg["ci_low"] = ci_low
    agg["ci_high"] = ci_high

    if has_weight:
        agg["default_rate_by_amount"] = np.where(
            agg["total_weight"] > 0,
            agg["def_weight"] / agg["total_weight"],
            np.nan,
        )
    else:
        agg["default_rate_by_amount"] = np.nan

    # Sort descending by default_rate (keeping 'Lainnya' at bottom if present)
    agg["_is_lainnya"] = agg["segmen"] == "Lainnya"
    agg = agg.sort_values(by=["_is_lainnya", "default_rate", "n"], ascending=[True, False, False]).reset_index(drop=True)

    cols_order = [
        "variabel",
        "segmen",
        "n",
        "n_default",
        "default_rate",
        "ci_low",
        "ci_high",
        "default_rate_by_amount",
    ]
    return agg[cols_order]


def outcome_mix_by(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute counts and row percentages of the 4 outcome groups per ``group_col`` value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (full portfolio).
    group_col : str
        Grouping column name (e.g. ``"loan_year"`` or ``"Country"``).

    Returns
    -------
    pd.DataFrame
        Table with counts and percentages for each of the 4 outcome groups,
        plus ``n_total``, ``n_resolved``, and ``pct_resolved``.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")

    outcomes = make_outcome_group(df)
    grp_series = df[group_col]

    tmp = pd.DataFrame({group_col: grp_series, "outcome_group": outcomes}, index=df.index)
    ct_count = pd.crosstab(tmp[group_col], tmp["outcome_group"], dropna=False)

    for cat in OUTCOME_CATEGORIES:
        if cat not in ct_count.columns:
            ct_count[cat] = 0
    ct_count = ct_count[OUTCOME_CATEGORIES]

    n_total = ct_count.sum(axis=1)
    n_resolved = ct_count["Default"] + ct_count["Lunas tanpa default"]
    pct_resolved = np.where(n_total > 0, n_resolved / n_total * 100.0, 0.0)

    out = pd.DataFrame({group_col: ct_count.index})
    out["n_total"] = n_total.to_numpy(dtype="int64")
    out["n_resolved"] = n_resolved.to_numpy(dtype="int64")
    out["pct_resolved"] = np.round(pct_resolved, 2)

    col_map = {
        "Default": ("n_default", "pct_default"),
        "Lunas tanpa default": ("n_lunas_tanpa_default", "pct_lunas_tanpa_default"),
        "Berjalan (Current)": ("n_berjalan", "pct_berjalan"),
        "Terlambat belum default": ("n_terlambat_belum_default", "pct_terlambat_belum_default"),
    }
    for cat, (n_col, pct_col) in col_map.items():
        cnt = ct_count[cat].to_numpy(dtype="int64")
        out[n_col] = cnt
        out[pct_col] = np.round(np.where(out["n_total"] > 0, cnt / out["n_total"] * 100.0, 0.0), 2)

    return out.reset_index(drop=True)


def weighted_return_ratio(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute amount-weighted and simple average return_ratio strictly on resolved rows.

    Formula:
        ``weighted_return_ratio = sum(PrincipalPaymentsMade + InterestAndPenaltyPaymentsMade) / sum(Amount)``

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame.

    Returns
    -------
    dict[str, float | int]
        Dictionary containing ``n_resolved``, ``numerator_total_payments``,
        ``denominator_total_amount``, ``weighted_return_ratio``,
        ``simple_mean_return_ratio``, and ``median_return_ratio``.
    """
    if IS_RESOLVED_COL in df.columns:
        mask_res = df[IS_RESOLVED_COL] == True  # noqa: E712
    elif DEFAULT_TARGET_COL in df.columns:
        mask_res = df[DEFAULT_TARGET_COL].notna()
    else:
        mask_res = make_default_label(df).notna()

    sub = df.loc[mask_res]
    n_res = int(len(sub))

    p_paid = pd.to_numeric(sub[PRINCIPAL_PAYMENTS_COL], errors="coerce").fillna(0.0)
    i_paid = pd.to_numeric(sub[INTEREST_PENALTY_PAYMENTS_COL], errors="coerce").fillna(0.0)
    amt = pd.to_numeric(sub[AMOUNT_COL], errors="coerce")

    valid_amt_mask = amt.notna() & (amt > 0)
    num_total = float((p_paid.loc[valid_amt_mask] + i_paid.loc[valid_amt_mask]).sum())
    den_total = float(amt.loc[valid_amt_mask].sum())
    w_ratio = num_total / den_total if den_total > 0 else float("nan")

    if REGRESSION_TARGET_COL in sub.columns:
        rr = pd.to_numeric(sub[REGRESSION_TARGET_COL], errors="coerce").dropna()
    else:
        rr = make_regression_target(sub).dropna()

    return {
        "n_resolved": n_res,
        "numerator_total_payments": num_total,
        "denominator_total_amount": den_total,
        "weighted_return_ratio": float(w_ratio),
        "simple_mean_return_ratio": float(rr.mean()) if len(rr) > 0 else float("nan"),
        "median_return_ratio": float(rr.median()) if len(rr) > 0 else float("nan"),
    }
