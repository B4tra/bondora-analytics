"""Feature set definitions (Set A, Set B, Set C) and data-driven era-dead feature detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, TABLES_DIR

# Roles strictly forbidden as candidate features (Rule 2)
FORBIDDEN_ROLES: frozenset[str] = frozenset(
    {
        "benchmark",
        "pricing",
        "atribut_sensitif",
        "proxy_era",
        "pasca_pinjaman",
        "hasil_target",
        "target",
        "target_horizon",
        "tanggal",
        "pembagi_waktu",
        "analisis",
        "sinyal_investor",
        "identifier",
        "flag",
    }
)

# Explicit column names strictly forbidden in Set A and Set B
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "Rating",
        "ProbabilityOfDefault",
        "ExpectedLoss",
        "LossGivenDefault",
        "ExpectedReturn",
        "ModelVersion",
        "EL_V0",
        "Rating_V0",
        "EL_V1",
        "Rating_V1",
        "EL_V2",
        "Rating_V2",
        "Interest",
        "Gender",
        "MonthlyPayment_missing",
        "PreviousRepaymentsBeforeLoan_missing",
        "CreditScoreEsMicroL_missing",
        "default",
        "return_ratio",
        "is_resolved",
        "observable_12m",
        "default_12m",
        "observable_24m",
        "default_24m",
        "loan_year",
        "loan_month",
        "funded_to_applied_ratio",
        "LoanId",
        "LoanNumber",
        "UserName",
    }
)

# Additional benchmark/pricing columns permitted ONLY in Set C (ablation experiment)
SET_C_ABLATION_COLS: tuple[str, ...] = ("Rating", "ProbabilityOfDefault", "Interest")


def load_candidate_features_from_schema(
    schema_path: Path = DATA_PROCESSED / "schema.json",
) -> list[dict[str, Any]]:
    """Load the 44 candidate feature metadata records (``boleh_jadi_fitur == 'Y'``) from ``schema.json``.

    Raises
    ------
    AssertionError
        If any column with ``boleh_jadi_fitur == 'Y'`` has a forbidden role or is in ``FORBIDDEN_COLUMNS``.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    candidates: list[dict[str, Any]] = []
    for col_meta in schema_data.get("columns", []):
        if col_meta.get("boleh_jadi_fitur") == "Y":
            c_name = col_meta["nama_kolom"]
            c_role = col_meta.get("peran", "")
            if c_role in FORBIDDEN_ROLES or c_name in FORBIDDEN_COLUMNS:
                raise AssertionError(
                    f"Forbidden column '{c_name}' (role='{c_role}') marked as boleh_jadi_fitur == 'Y'!"
                )
            candidates.append(col_meta)

    return candidates


def _classify_feature_type(series: pd.Series) -> str:
    """Classify a feature series into ``'numerik'``, ``'kategorikal'``, or ``'boolean'``."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return "kategorikal"
    return "numerik"


def compute_dominant_value_stats(
    train_s: pd.Series,
    val_s: pd.Series,
) -> dict[str, Any]:
    """Compute dominant value share in validation and its corresponding share in training.

    Missing values (``NaN``/``<NA>``), ``-1`` (``"Not set"``), and ``0`` are treated as distinct
    values when computing value frequency proportions over all rows (``dropna=False``).

    Returns
    -------
    dict[str, Any]
        ``{"val_dom_value": ..., "share_dominan_validasi": float, "share_dominan_latih": float}``
    """
    def _normalize_tokens(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            out = num.astype("object")
            out = out.where(num.notna(), "(Missing)")
            return out
        s_obj = s.astype("object").where(s.notna(), "(Missing)").astype(str).str.strip()
        s_obj = s_obj.replace({"-1.0": "-1", "0.0": "0"})
        return s_obj

    t_norm = _normalize_tokens(train_s)
    v_norm = _normalize_tokens(val_s)

    if len(v_norm) == 0 or len(t_norm) == 0:
        return {
            "val_dom_value": None,
            "share_dominan_validasi": 0.0,
            "share_dominan_latih": 0.0,
        }

    v_vc = v_norm.value_counts(dropna=False, normalize=True)
    dom_val = v_vc.index[0]
    val_dom_share = float(v_vc.iloc[0])

    # Proportion of that same validation-dominant value in training, and max dominant share in training
    t_vc = t_norm.value_counts(dropna=False, normalize=True)
    train_same_share = float(t_vc.get(dom_val, 0.0))

    return {
        "val_dom_value": str(dom_val),
        "share_dominan_validasi": val_dom_share,
        "share_dominan_latih": train_same_share,
    }


def build_feature_sets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    candidate_cols: list[str] | None = None,
    psi_csv_path: Path | None = TABLES_DIR / "03_psi_by_era.csv",
    val_dom_threshold: float = 0.95,
    train_dom_max_threshold: float = 0.80,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Define Set A (stable), Set B (all 44), and Set C (Set A + Rating, PD, Interest) from train/val data.

    Era-Dead Rule (Step 2.3)
    ------------------------
    A feature is declared ``mati-era`` if:
    - ``share_dominan_validasi >= val_dom_threshold`` (default ``0.95``), AND
    - ``share_dominan_latih <= train_dom_max_threshold`` (default ``0.80``).

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split DataFrame.
    df_val : pd.DataFrame
        Validation split DataFrame.
    candidate_cols : list[str] or None, default None
        Candidate feature names. If None, loaded from ``schema.json`` (44 features).
    psi_csv_path : Path or None
        Path to ``03_psi_by_era.csv`` for supporting PSI 2018-2020 evidence.
    val_dom_threshold : float, default 0.95
        Threshold on validation dominant value share.
    train_dom_max_threshold : float, default 0.80
        Maximum training share of that dominant value to qualify as newly collapsed (era-dead).

    Returns
    -------
    tuple[dict[str, list[str]], pd.DataFrame]
        ``({"A": set_a_cols, "B": set_b_cols, "C": set_c_cols}, feature_sets_table)``
    """
    if candidate_cols is None:
        cand_meta = load_candidate_features_from_schema()
        candidate_cols = [m["nama_kolom"] for m in cand_meta]

    # Verify no forbidden columns in candidate_cols
    overlap_forbidden = set(candidate_cols) & FORBIDDEN_COLUMNS
    if overlap_forbidden:
        raise AssertionError(f"Candidate features contain forbidden columns: {sorted(overlap_forbidden)}")

    psi_map: dict[str, float] = {}
    if psi_csv_path is not None and Path(psi_csv_path).exists():
        psi_df = pd.read_csv(psi_csv_path)
        if "fitur" in psi_df.columns and "psi_2018_2020" in psi_df.columns:
            psi_map = dict(zip(psi_df["fitur"], psi_df["psi_2018_2020"]))

    set_b: list[str] = list(candidate_cols)
    set_a: list[str] = []
    rows: list[dict[str, Any]] = []

    for col in set_b:
        ftype = _classify_feature_type(df_train[col])
        dom_stats = compute_dominant_value_stats(df_train[col], df_val[col])
        sh_train = dom_stats["share_dominan_latih"]
        sh_val = dom_stats["share_dominan_validasi"]
        dom_val_str = dom_stats["val_dom_value"]
        psi_val = float(psi_map.get(col, np.nan))

        is_era_dead = (sh_val >= val_dom_threshold) and (sh_train <= train_dom_max_threshold)
        if not is_era_dead:
            set_a.append(col)
            alasan = "Aktif & stabil (masuk Set A)"
        else:
            dom_label = "-1 (Not set)" if dom_val_str == "-1" else dom_val_str
            alasan = (
                f"Mati-era: nilai dominan '{dom_label}' mencapai {sh_val*100:.2f}% di validasi "
                f"(vs {sh_train*100:.2f}% di latih; PSI 2018-2020 = {psi_val:.4f})"
            )

        rows.append(
            {
                "fitur": col,
                "tipe": ftype,
                "di_set_A": not is_era_dead,
                "di_set_B": True,
                "di_set_C": not is_era_dead,
                "alasan_keluar": alasan,
                "nilai_dominan_validasi": dom_val_str,
                "share_dominan_latih": round(sh_train, 6),
                "share_dominan_validasi": round(sh_val, 6),
                "psi_2018_2020": round(psi_val, 4) if not np.isnan(psi_val) else np.nan,
            }
        )

    set_c: list[str] = list(set_a)
    for extra_col in SET_C_ABLATION_COLS:
        if extra_col in df_train.columns:
            set_c.append(extra_col)
            ftype = _classify_feature_type(df_train[extra_col])
            dom_stats = compute_dominant_value_stats(df_train[extra_col], df_val[extra_col])
            rows.append(
                {
                    "fitur": extra_col,
                    "tipe": ftype,
                    "di_set_A": False,
                    "di_set_B": False,
                    "di_set_C": True,
                    "alasan_keluar": "Hanya untuk eksperimen ablasi Set C (benchmark/pricing platform)",
                    "nilai_dominan_validasi": dom_stats["val_dom_value"],
                    "share_dominan_latih": round(dom_stats["share_dominan_latih"], 6),
                    "share_dominan_validasi": round(dom_stats["share_dominan_validasi"], 6),
                    "psi_2018_2020": np.nan,
                }
            )

    tbl = pd.DataFrame(rows)
    return {"A": set_a, "B": set_b, "C": set_c}, tbl
