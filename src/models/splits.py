"""Time-based and stratified random data splitting utilities for Tahap 4 (Predictive Analytics)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_12M_COL,
    DEFAULT_TARGET_COL,
    IS_RESOLVED_COL,
    LOAN_DATE_COL,
    RANDOM_SEED,
    REGRESSION_TARGET_COL,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
)

SplitName = Literal["train", "val", "test"]


def assign_temporal_split(
    df: pd.DataFrame,
    date_col: str = LOAN_DATE_COL,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
) -> pd.Series:
    """Assign each row to ``'train'``, ``'val'``, or ``'test'`` based on ``LoanDate``.

    Rules
    -----
    - ``train``: ``LoanDate <= train_end`` (default ``2018-06-30``)
    - ``val``  : ``train_end < LoanDate <= val_end`` (default ``2018-07-01`` to ``2019-06-30``)
    - ``test`` : ``LoanDate > val_end`` (default ``>= 2019-07-01``)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing ``date_col``.
    date_col : str, default LOAN_DATE_COL
        Date column name.
    train_end : str, default SPLIT_TRAIN_END
        Inclusive upper date bound for the training split (``YYYY-MM-DD``).
    val_end : str, default SPLIT_VAL_END
        Inclusive upper date bound for the validation split (``YYYY-MM-DD``).

    Returns
    -------
    pd.Series
        String series of ``'train'``, ``'val'``, or ``'test'`` indexed like ``df``.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Column '{date_col}' contains {int(dates.isna().sum())} missing/unparseable dates.")

    train_ts = pd.Timestamp(train_end)
    val_ts = pd.Timestamp(val_end)
    if val_ts <= train_ts:
        raise ValueError(f"val_end ({val_end}) must be strictly after train_end ({train_end}).")

    split_arr = np.where(
        dates <= train_ts,
        "train",
        np.where(dates <= val_ts, "val", "test"),
    )
    return pd.Series(split_arr, index=df.index, name="split", dtype="object")


def make_classification_splits(
    df: pd.DataFrame,
    target_col: str = DEFAULT_12M_COL,
    date_col: str = LOAN_DATE_COL,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
    min_rows: int = 10_000,
    min_minority_rate: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter valid ``default_12m`` rows, assign temporal splits, and build split summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    target_col : str, default DEFAULT_12M_COL
        Binary classification target column.
    date_col : str, default LOAN_DATE_COL
        Origination date column.
    train_end : str, default SPLIT_TRAIN_END
        Inclusive end date for train split.
    val_end : str, default SPLIT_VAL_END
        Inclusive end date for validation split.
    min_rows : int, default 10_000
        Minimum acceptable row count per split.
    min_minority_rate : float, default 0.10
        Minimum acceptable minority class proportion per split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_cls_with_split, split_summary_df)``
    """
    mask_valid = df[target_col].notna()
    df_cls = df.loc[mask_valid].copy()
    df_cls["split"] = assign_temporal_split(df_cls, date_col=date_col, train_end=train_end, val_end=val_end)

    rows = []
    n_total_pop = len(df_cls)
    for sp_code, sp_label in [
        ("train", f"Latih (<= {train_end})"),
        ("val", f"Validasi (> {train_end} s.d. {val_end})"),
        ("test", f"Uji (> {val_end})"),
    ]:
        sub = df_cls.loc[df_cls["split"] == sp_code]
        n_sp = int(len(sub))
        n_def = int((sub[target_col] == 1).sum())
        n_nondef = int((sub[target_col] == 0).sum())
        def_rate = float(n_def / n_sp) if n_sp > 0 else np.nan
        min_class_rate = min(def_rate, 1.0 - def_rate) if n_sp > 0 else 0.0

        if min_rows > 0 and n_sp < min_rows:
            raise ValueError(f"Split '{sp_code}' has {n_sp} rows (< min_rows={min_rows}).")
        if min_minority_rate > 0 and min_class_rate < min_minority_rate:
            raise ValueError(
                f"Split '{sp_code}' has minority class rate {min_class_rate:.4f} (< {min_minority_rate})."
            )

        dt_min = pd.to_datetime(sub[date_col]).min()
        dt_max = pd.to_datetime(sub[date_col]).max()
        rows.append(
            {
                "skema_split": "temporal",
                "split": sp_code,
                "deskripsi": sp_label,
                "n": n_sp,
                "proporsi_populasi": round(n_sp / n_total_pop, 6),
                "n_default": n_def,
                "n_non_default": n_nondef,
                "default_rate": round(def_rate, 6),
                "loan_date_min": str(dt_min.date()) if pd.notna(dt_min) else "",
                "loan_date_max": str(dt_max.date()) if pd.notna(dt_max) else "",
            }
        )

    # Add Total row for temporal split
    n_def_all = int((df_cls[target_col] == 1).sum())
    dt_all_min = pd.to_datetime(df_cls[date_col]).min()
    dt_all_max = pd.to_datetime(df_cls[date_col]).max()
    rows.append(
        {
            "skema_split": "temporal",
            "split": "Total",
            "deskripsi": "Seluruh Populasi Valid default_12m",
            "n": n_total_pop,
            "proporsi_populasi": 1.0,
            "n_default": n_def_all,
            "n_non_default": int(n_total_pop - n_def_all),
            "default_rate": round(n_def_all / n_total_pop, 6),
            "loan_date_min": str(dt_all_min.date()),
            "loan_date_max": str(dt_all_max.date()),
        }
    )

    return df_cls, pd.DataFrame(rows)


def make_random_stratified_split(
    df_cls: pd.DataFrame,
    target_col: str = DEFAULT_12M_COL,
    date_col: str = LOAN_DATE_COL,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.Series, pd.DataFrame]:
    """Create a 70/15/15 stratified random split on the classification population.

    Used strictly in Step 6 to quantify optimism from ignoring temporal regime shifts.

    Parameters
    ----------
    df_cls : pd.DataFrame
        Classification DataFrame (rows with non-null ``target_col``).
    target_col : str, default DEFAULT_12M_COL
        Binary target column name.
    date_col : str, default LOAN_DATE_COL
        Date column for reporting date ranges.
    random_state : int, default RANDOM_SEED
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(random_split_series, random_split_summary_df)``
    """
    y = df_cls[target_col].astype("int8")
    idx_all = df_cls.index.to_numpy()

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all,
        y.to_numpy(),
        test_size=0.30,
        stratify=y.to_numpy(),
        random_state=random_state,
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    split_s = pd.Series("train", index=df_cls.index, name="split_random", dtype="object")
    split_s.loc[idx_val] = "val"
    split_s.loc[idx_test] = "test"

    n_total = len(df_cls)
    rows = []
    for sp_code, sp_label in [
        ("train", "Acak Terstratifikasi Latih (70%)"),
        ("val", "Acak Terstratifikasi Validasi (15%)"),
        ("test", "Acak Terstratifikasi Uji (15%)"),
    ]:
        sub = df_cls.loc[split_s == sp_code]
        n_sp = int(len(sub))
        n_def = int((sub[target_col] == 1).sum())
        dt_min = pd.to_datetime(sub[date_col]).min()
        dt_max = pd.to_datetime(sub[date_col]).max()
        rows.append(
            {
                "skema_split": "random_70_15_15",
                "split": sp_code,
                "deskripsi": sp_label,
                "n": n_sp,
                "proporsi_populasi": round(n_sp / n_total, 6),
                "n_default": n_def,
                "n_non_default": int(n_sp - n_def),
                "default_rate": round(n_def / n_sp, 6),
                "loan_date_min": str(dt_min.date()) if pd.notna(dt_min) else "",
                "loan_date_max": str(dt_max.date()) if pd.notna(dt_max) else "",
            }
        )

    return split_s, pd.DataFrame(rows)


def make_regression_splits(
    df: pd.DataFrame,
    resolved_col: str = IS_RESOLVED_COL,
    target_col: str = REGRESSION_TARGET_COL,
    default_col: str = DEFAULT_TARGET_COL,
    date_col: str = LOAN_DATE_COL,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
    min_test_rows: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter ``is_resolved == True`` rows, assign temporal splits, and summarize regression splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    resolved_col : str, default IS_RESOLVED_COL
        Boolean resolved indicator column.
    target_col : str, default REGRESSION_TARGET_COL
        Continuous regression target (``return_ratio``).
    default_col : str, default DEFAULT_TARGET_COL
        Lifetime default indicator column on resolved loans.
    date_col : str, default LOAN_DATE_COL
        Origination date column.
    train_end : str, default SPLIT_TRAIN_END
        Inclusive end date for train split.
    val_end : str, default SPLIT_VAL_END
        Inclusive end date for validation split.
    min_test_rows : int, default 5_000
        Minimum required rows in test split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_reg_with_split, reg_split_summary_df)``
    """
    mask_valid = (df[resolved_col] == True) & df[target_col].notna()  # noqa: E712
    df_reg = df.loc[mask_valid].copy()
    df_reg["split"] = assign_temporal_split(df_reg, date_col=date_col, train_end=train_end, val_end=val_end)

    n_test = int((df_reg["split"] == "test").sum())
    if min_test_rows > 0 and n_test < min_test_rows:
        raise ValueError(f"Regression test split has only {n_test} rows (< min_test_rows={min_test_rows}).")

    n_total = len(df_reg)
    rows = []
    for sp_code, sp_label in [
        ("train", f"Latih Resolved (<= {train_end})"),
        ("val", f"Validasi Resolved (> {train_end} s.d. {val_end})"),
        ("test", f"Uji Resolved (> {val_end})"),
    ]:
        sub = df_reg.loc[df_reg["split"] == sp_code]
        n_sp = int(len(sub))
        rr = pd.to_numeric(sub[target_col], errors="coerce")
        n_def = int((sub[default_col] == 1).sum()) if default_col in sub.columns else 0
        dt_min = pd.to_datetime(sub[date_col]).min()
        dt_max = pd.to_datetime(sub[date_col]).max()
        rows.append(
            {
                "split": sp_code,
                "deskripsi": sp_label,
                "n": n_sp,
                "proporsi_populasi": round(n_sp / n_total, 6),
                "n_default_lifetime": n_def,
                "default_rate_lifetime": round(n_def / n_sp, 6) if n_sp > 0 else np.nan,
                "mean_return_ratio": round(float(rr.mean()), 6),
                "std_return_ratio": round(float(rr.std(ddof=1)), 6),
                "p25_return_ratio": round(float(rr.quantile(0.25)), 6),
                "median_return_ratio": round(float(rr.median()), 6),
                "p75_return_ratio": round(float(rr.quantile(0.75)), 6),
                "loan_date_min": str(dt_min.date()) if pd.notna(dt_min) else "",
                "loan_date_max": str(dt_max.date()) if pd.notna(dt_max) else "",
            }
        )

    rr_all = pd.to_numeric(df_reg[target_col], errors="coerce")
    n_def_all = int((df_reg[default_col] == 1).sum()) if default_col in df_reg.columns else 0
    dt_all_min = pd.to_datetime(df_reg[date_col]).min()
    dt_all_max = pd.to_datetime(df_reg[date_col]).max()
    rows.append(
        {
            "split": "Total",
            "deskripsi": "Seluruh Populasi Resolved (is_resolved == True)",
            "n": n_total,
            "proporsi_populasi": 1.0,
            "n_default_lifetime": n_def_all,
            "default_rate_lifetime": round(n_def_all / n_total, 6),
            "mean_return_ratio": round(float(rr_all.mean()), 6),
            "std_return_ratio": round(float(rr_all.std(ddof=1)), 6),
            "p25_return_ratio": round(float(rr_all.quantile(0.25)), 6),
            "median_return_ratio": round(float(rr_all.median()), 6),
            "p75_return_ratio": round(float(rr_all.quantile(0.75)), 6),
            "loan_date_min": str(dt_all_min.date()),
            "loan_date_max": str(dt_all_max.date()),
        }
    )

    return df_reg, pd.DataFrame(rows)
