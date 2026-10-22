"""Tahap 4 follow-up checks (evaluation only).

The primary ``LGBM-A`` model saved in ``models/lgbm_default12m_setA.joblib`` is **never retrained or
overwritten** here. The module re-runs evaluation from saved artifacts
(``data/processed/predictions_default12m.parquet`` + the saved model) and produces:

1. ``reports/tables/04_subgroup_performance.csv`` -- within-country ROC-AUC of LGBM-A next to the
   ``ProbabilityOfDefault`` and ``Interest`` baselines on the same test rows (+ paired AUC difference CI).
2. ``reports/tables/04_level_bias.csv`` -- mean predicted probability vs actual 12-month default rate.
3. ``reports/tables/04_nan_signal_shap.csv`` and ``reports/tables/04_nan_signal_refit.csv`` -- NaN-as-signal
   check (SHAP of NaN vs non-NaN rows, plus a single report-only refit with NaN filled by one constant).
4. A ``cakupan_evaluasi`` column in ``reports/tables/04_metrics_classification.csv`` marking validation
   metrics as in-sample and test metrics as the reported ones.

Usage:
    python -m src.models.followup [--skip-refit]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import (
    DATA_PROCESSED,
    DEFAULT_12M_COL,
    MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    ensure_dirs,
)
from src.models.evaluate import (
    bootstrap_auc_ci,
    compute_ece,
    compute_ks_statistic,
    compute_subgroup_performance,
    paired_bootstrap_auc_diff,
)
from src.models.pipelines import CategoryAligner, ValidationProbabilityCalibrator
from src.models.splits import make_classification_splits

COUNTRIES: tuple[str, ...] = ("EE", "FI", "ES")
HALF_LABELS: tuple[str, str] = ("Jul–Des 2019", "Jan–Jul 2020")
HALF_CUTOFF: str = "2019-12-31"

# NaN-concentration rule: NaN share >= NAN_SHARE_MIN in at least one split AND the spread of NaN share
# across train/val/test >= NAN_RANGE_MIN_PP percentage points. The two features named by the reviewer
# are always included.
NAN_SHARE_MIN: float = 0.01
NAN_RANGE_MIN_PP: float = 5.0
FORCED_NAN_FEATURES: tuple[str, ...] = ("PreviousRepaymentsBeforeLoan", "MonthlyPayment")

SCOPE_VAL: str = (
    "validasi (in-sample): penyetelan hiperparameter, early stopping, kalibrator, dan ambang Youden "
    "dipelajari dari data ini; bukan untuk pelaporan"
)
SCOPE_TEST: str = "uji (out-of-sample): metrik yang dipakai untuk pelaporan"


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def assign_test_half(loan_date: pd.Series) -> pd.Series:
    """Label each date as ``'Jul–Des 2019'`` (<= 2019-12-31) or ``'Jan–Jul 2020'``."""
    dt = pd.to_datetime(loan_date)
    arr = np.where(dt <= pd.Timestamp(HALF_CUTOFF), HALF_LABELS[0], HALF_LABELS[1])
    return pd.Series(arr, index=loan_date.index, dtype="object")


# --------------------------------------------------------------------------------------
# 1. Within-country baselines
# --------------------------------------------------------------------------------------
def build_subgroup_with_baselines(df_cls: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Recompute LGBM-A subgroup performance and add PD/Interest rows (+ AUC differences) per Country.

    Parameters
    ----------
    df_cls : pd.DataFrame
        Classification population with ``split`` (output of ``make_classification_splits``).
    pred : pd.DataFrame
        ``predictions_default12m.parquet`` content (``p_raw``, ``p_calibrated``, ``PD_bondora``,
        ``Interest`` joined on ``LoanId``).
    """
    df_test = df_cls.loc[df_cls["split"] == "test"].copy()
    pred_idx = pred.set_index("LoanId")
    ids = df_test["LoanId"].astype(str).to_numpy()
    p_raw = pred_idx.loc[ids, "p_raw"].to_numpy(dtype=np.float64)
    p_cal = pred_idx.loc[ids, "p_calibrated"].to_numpy(dtype=np.float64)
    pd_b = pred_idx.loc[ids, "PD_bondora"].to_numpy(dtype=np.float64)
    interest = pred_idx.loc[ids, "Interest"].to_numpy(dtype=np.float64)
    y = df_test[DEFAULT_12M_COL].astype("int32").to_numpy()

    base = compute_subgroup_performance(df_test, y, p_raw, p_cal)
    base.insert(2, "kode_model", "LGBM-A")

    # Integrity check: recomputed LGBM-A rows must equal what the saved CSV already holds.
    path = TABLES_DIR / "04_subgroup_performance.csv"
    if path.exists():
        prev = pd.read_csv(path)
        if "kode_model" in prev.columns:
            prev = prev.loc[prev["kode_model"] == "LGBM-A"]
        merged = base.merge(prev, on=["dimensi", "subgrup"], suffixes=("", "_prev"))
        for col in ("n", "n_default", "roc_auc", "auc_ci_low_95", "auc_ci_high_95", "brier_score"):
            ok = np.allclose(
                merged[col].to_numpy(dtype=float),
                merged[f"{col}_prev"].to_numpy(dtype=float),
                atol=1e-6,
                equal_nan=True,
            )
            if not ok:
                raise AssertionError(f"Recomputed LGBM-A subgroup column '{col}' differs from saved CSV.")

    country_arr = df_test["Country"].astype(str).to_numpy()
    baseline_specs = [
        ("BASE-PD-Bondora", pd_b, True),
        ("BASE-Interest", interest, False),
    ]

    out_rows: list[pd.DataFrame] = []
    for _, row in base.iterrows():
        out_rows.append(row.to_frame().T)
        if row["dimensi"] == "Country" and str(row["subgrup"]) in COUNTRIES:
            c = str(row["subgrup"])
            m = country_arr == c
            y_c, p_c = y[m], p_cal[m]
            for code, score, is_prob in baseline_specs:
                s_c = score[m]
                if np.isnan(s_c).any():
                    raise ValueError(f"{code} has NaN rows within Country={c}; same-row comparison is not valid.")
                auc, lo, hi = bootstrap_auc_ci(y_c, s_c, n_bootstrap=200, random_state=RANDOM_SEED)
                diff = paired_bootstrap_auc_diff(y_c, p_c, s_c, n_bootstrap=200, random_state=RANDOM_SEED)
                n_def = int(y_c.sum())
                out_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "dimensi": "Country",
                                "subgrup": c,
                                "kode_model": code,
                                "n": int(m.sum()),
                                "n_default": n_def,
                                "observed_default_rate": round(n_def / int(m.sum()), 6),
                                "roc_auc": round(auc, 6),
                                "auc_ci_low_95": round(lo, 6),
                                "auc_ci_high_95": round(hi, 6),
                                "pr_auc": round(float(average_precision_score(y_c, s_c)), 6),
                                "ks": round(compute_ks_statistic(y_c, s_c), 6),
                                "brier_score": round(float(brier_score_loss(y_c, np.clip(s_c, 0, 1))), 6)
                                if is_prob
                                else np.nan,
                                "selisih_auc_lgbm_a_minus_pembanding": diff["diff_auc"],
                                "selisih_auc_ci_low_95": diff["ci_low_95"],
                                "selisih_auc_ci_high_95": diff["ci_high_95"],
                            }
                        ]
                    )
                )

    result = pd.concat(out_rows, ignore_index=True)
    for col in ("selisih_auc_lgbm_a_minus_pembanding", "selisih_auc_ci_low_95", "selisih_auc_ci_high_95"):
        if col not in result.columns:
            result[col] = np.nan
    numeric_cols = [c for c in result.columns if c not in ("dimensi", "subgrup", "kode_model")]
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in ("n", "n_default"):
        result[col] = result[col].astype("int64")
    return result


# --------------------------------------------------------------------------------------
# 2. Level bias
# --------------------------------------------------------------------------------------
def compute_level_bias_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted probability vs actual 12-month default rate (validation, test, test by segment).

    ``pred`` needs ``split``, ``Country``, ``LoanDate``, ``default_12m``, ``p_raw``, ``p_calibrated`` and
    ``PD_bondora``. Validation rows are flagged in-sample because the calibrator was fit on them.
    """
    work = pred.copy()
    work["paruh_waktu_uji"] = assign_test_half(work["LoanDate"])
    rows: list[dict[str, Any]] = []

    def _add(cakupan: str, split: str, subgrup: str, sub: pd.DataFrame) -> None:
        n = int(len(sub))
        if n == 0:
            return
        actual = float(sub["default_12m"].mean())
        mean_cal = float(sub["p_calibrated"].mean())
        mean_raw = float(sub["p_raw"].mean())
        mean_pd = float(pd.to_numeric(sub["PD_bondora"], errors="coerce").mean())
        rows.append(
            {
                "cakupan": cakupan,
                "split": split,
                "subgrup": subgrup,
                "n": n,
                "n_default": int(sub["default_12m"].sum()),
                "default_12m_aktual": round(actual, 6),
                "mean_p_raw": round(mean_raw, 6),
                "mean_p_calibrated": round(mean_cal, 6),
                "selisih_cal_minus_aktual_pp": round((mean_cal - actual) * 100.0, 4),
                "rasio_cal_per_aktual": round(mean_cal / actual, 4) if actual > 0 else np.nan,
                "mean_pd_bondora": round(mean_pd, 6),
                "selisih_pd_minus_aktual_pp": round((mean_pd - actual) * 100.0, 4),
                "sifat_metrik": SCOPE_VAL if split == "val" else SCOPE_TEST,
            }
        )

    for sp in ("val", "test"):
        _add("Seluruh", sp, "Seluruh", work.loc[work["split"] == sp])

    test = work.loc[work["split"] == "test"]
    for c in COUNTRIES:
        _add("Country", "test", c, test.loc[test["Country"].astype(str) == c])
    for h in HALF_LABELS:
        _add("paruh_waktu_uji", "test", h, test.loc[test["paruh_waktu_uji"] == h])
    for c in COUNTRIES:
        for h in HALF_LABELS:
            _add(
                "Country x paruh_waktu_uji",
                "test",
                f"{c} | {h}",
                test.loc[(test["Country"].astype(str) == c) & (test["paruh_waktu_uji"] == h)],
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. NaN as signal
# --------------------------------------------------------------------------------------
def nan_concentration_table(df_cls: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """NaN share per split and the rule flag ``memenuhi_aturan_konsentrasi`` for each feature."""
    rows = []
    for f in features:
        shares = {sp: float(df_cls.loc[df_cls["split"] == sp, f].isna().mean()) for sp in ("train", "val", "test")}
        spread_pp = (max(shares.values()) - min(shares.values())) * 100.0
        rule = bool(max(shares.values()) >= NAN_SHARE_MIN and spread_pp >= NAN_RANGE_MIN_PP)
        rows.append(
            {
                "fitur": f,
                "nan_share_latih": round(shares["train"], 6),
                "nan_share_validasi": round(shares["val"], 6),
                "nan_share_uji": round(shares["test"], 6),
                "rentang_nan_pp": round(spread_pp, 4),
                "memenuhi_aturan_konsentrasi": rule,
                "dipaksa_masuk_permintaan": f in FORCED_NAN_FEATURES,
            }
        )
    return pd.DataFrame(rows)


def select_nan_features(conc: pd.DataFrame) -> list[str]:
    """Features that meet the concentration rule or were explicitly requested."""
    mask = conc["memenuhi_aturan_konsentrasi"] | conc["dipaksa_masuk_permintaan"]
    return conc.loc[mask, "fitur"].tolist()


def shap_by_nan_group(
    contrib: np.ndarray,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    sample_label: str,
    selected: list[str],
) -> pd.DataFrame:
    """Mean signed / absolute SHAP (log-odds) for NaN vs non-NaN rows of each feature.

    Rows are produced for every selected feature (even with zero NaN rows in the sample) and for any other
    feature that has at least one NaN row in the sample.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    rows = []
    for j, f in enumerate(features):
        is_nan = X_raw[f].isna().to_numpy()
        n_nan = int(is_nan.sum())
        if n_nan == 0 and f not in selected:
            continue
        s = contrib[:, j]
        non = ~is_nan

        def _mean(arr: np.ndarray, mask: np.ndarray) -> float:
            return float(arr[mask].mean()) if mask.any() else np.nan

        mean_nan = _mean(s, is_nan)
        mean_non = _mean(s, non)
        rows.append(
            {
                "fitur": f,
                "sampel": sample_label,
                "dipilih_untuk_isian_konstan": f in selected,
                "n_sampel": int(len(s)),
                "n_nan": n_nan,
                "share_nan": round(n_nan / len(s), 6),
                "mean_shap_nan": round(mean_nan, 6) if not np.isnan(mean_nan) else np.nan,
                "mean_shap_non_nan": round(mean_non, 6) if not np.isnan(mean_non) else np.nan,
                "selisih_shap_nan_minus_non_nan": round(mean_nan - mean_non, 6)
                if not np.isnan(mean_nan)
                else np.nan,
                "mean_abs_shap_nan": round(_mean(np.abs(s), is_nan), 6) if n_nan else np.nan,
                "mean_abs_shap_non_nan": round(_mean(np.abs(s), non), 6),
                "default_rate_nan": round(_mean(y_arr, is_nan), 6) if n_nan else np.nan,
                "default_rate_non_nan": round(_mean(y_arr, non), 6),
            }
        )
    return pd.DataFrame(rows)


def fit_fill_values(df_train: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    """Single constant per feature learned from the TRAIN split only (median / most frequent category)."""
    fill: dict[str, Any] = {}
    for c in cols:
        s = df_train[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            fill[c] = str(s.dropna().astype(str).mode().iloc[0])
        else:
            fill[c] = float(pd.to_numeric(s, errors="coerce").median())
    return fill


def apply_fill_values(df: pd.DataFrame, fill: dict[str, Any]) -> pd.DataFrame:
    """Fill NaN in the given columns with the supplied constants (returns a copy)."""
    out = df.copy()
    for c, v in fill.items():
        s = out[c]
        if isinstance(v, str):
            out[c] = s.astype("object").where(s.notna(), v)
        else:
            out[c] = pd.to_numeric(s, errors="coerce").fillna(v)
    return out


def _fit_lgbm_same_params(model_params: dict[str, Any], df_tr: pd.DataFrame, y_tr: np.ndarray,
                          df_va: pd.DataFrame, y_va: np.ndarray, feats: list[str]):
    """Fit one LGBMClassifier with the saved best params (no tuning), early stopping 50 on validation AUC."""
    import lightgbm as lgb

    aligner = CategoryAligner()
    X_tr = aligner.fit_transform(df_tr[feats])
    X_va = aligner.transform(df_va[feats])
    clf = lgb.LGBMClassifier(**model_params)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return aligner, clf


def _evaluate_refit(label: str, aligner: Any, clf: Any, df_va: pd.DataFrame, y_va: np.ndarray,
                    df_te: pd.DataFrame, y_te: np.ndarray, feats: list[str],
                    p_main_cal_test: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    p_va_raw = clf.predict_proba(aligner.transform(df_va[feats]))[:, 1]
    p_te_raw = clf.predict_proba(aligner.transform(df_te[feats]))[:, 1]
    cal = ValidationProbabilityCalibrator(random_state=RANDOM_SEED).fit(p_va_raw, y_va)
    p_te_cal = cal.transform(p_te_raw)
    auc_raw, lo_raw, hi_raw = bootstrap_auc_ci(y_te, p_te_raw, n_bootstrap=200, random_state=RANDOM_SEED)
    auc_cal, lo_cal, hi_cal = bootstrap_auc_ci(y_te, p_te_cal, n_bootstrap=200, random_state=RANDOM_SEED)
    diff = paired_bootstrap_auc_diff(y_te, p_te_cal, p_main_cal_test, n_bootstrap=200, random_state=RANDOM_SEED)
    return {
        "varian": label,
        "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
        "val_auc_raw": round(float(roc_auc_score(y_va, p_va_raw)), 6),
        "test_auc_raw": round(auc_raw, 6),
        "test_auc_raw_ci_low_95": round(lo_raw, 6),
        "test_auc_raw_ci_high_95": round(hi_raw, 6),
        "metode_kalibrasi": cal.selected_method_,
        "test_auc_calibrated": round(auc_cal, 6),
        "test_auc_calibrated_ci_low_95": round(lo_cal, 6),
        "test_auc_calibrated_ci_high_95": round(hi_cal, 6),
        "selisih_auc_vs_lgbm_a_utama": diff["diff_auc"],
        "selisih_ci_low_95": diff["ci_low_95"],
        "selisih_ci_high_95": diff["ci_high_95"],
        "test_brier_calibrated": round(float(brier_score_loss(y_te, p_te_cal)), 6),
        "test_ece_calibrated": round(compute_ece(y_te, p_te_cal, n_bins=10)[0], 6),
    }


# --------------------------------------------------------------------------------------
# 4. Metrics scope annotation
# --------------------------------------------------------------------------------------
def annotate_metrics_scope(path=TABLES_DIR / "04_metrics_classification.csv") -> pd.DataFrame:
    """Add ``cakupan_evaluasi`` (validation = in-sample, test = reported) to the metrics table."""
    m = pd.read_csv(path)
    m["cakupan_evaluasi"] = np.where(m["split"].isin(["val", "val_random"]), SCOPE_VAL, SCOPE_TEST)
    m.to_csv(path, index=False)
    return m


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_followup(df: pd.DataFrame | None = None, skip_refit: bool = False) -> dict[str, Any]:
    """Run all follow-up evaluations from saved artifacts and write the tables."""
    t0 = time.perf_counter()
    ensure_dirs()
    if df is None:
        df = pd.read_parquet(DATA_PROCESSED / "bondora_clean.parquet")
    df_cls, _ = make_classification_splits(df)
    pred = pd.read_parquet(DATA_PROCESSED / "predictions_default12m.parquet")
    if not pred["LoanId"].is_unique or len(pred) != len(df_cls):
        raise AssertionError("predictions_default12m.parquet does not match the classification population.")
    art = joblib.load(MODELS_DIR / "lgbm_default12m_setA.joblib")
    feats: list[str] = list(art["features"])
    summary: dict[str, Any] = {}

    # (1) within-country baselines
    print("[Follow-up 1] Within-country PD / Interest baselines...")
    sub = build_subgroup_with_baselines(df_cls, pred)
    sub.to_csv(TABLES_DIR / "04_subgroup_performance.csv", index=False)
    summary["subgroup_rows"] = int(len(sub))

    # (2) level bias
    print("[Follow-up 2] Level bias table...")
    lvl = compute_level_bias_table(pred)
    lvl.to_csv(TABLES_DIR / "04_level_bias.csv", index=False)
    summary["level_bias_rows"] = int(len(lvl))

    # (3) NaN as signal
    print("[Follow-up 3] NaN-as-signal: SHAP by NaN group...")
    conc = nan_concentration_table(df_cls, feats)
    selected = select_nan_features(conc)
    summary["nan_selected_features"] = selected

    df_tr = df_cls.loc[df_cls["split"] == "train"]
    df_va = df_cls.loc[df_cls["split"] == "val"]
    df_te = df_cls.loc[df_cls["split"] == "test"]
    y_tr = df_tr[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_va = df_va[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_te = df_te[DEFAULT_12M_COL].astype("int32").to_numpy()

    model = art["model"]
    aligner = art["aligner"]
    shap_frames = []
    for label, d_part, y_part in (("uji (sampel SHAP n=5.000, seed sama dengan 04_shap_top15)", df_te, y_te),
                                   ("latih (sampel acak n=5.000)", df_tr, y_tr)):
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(len(d_part), size=min(5000, len(d_part)), replace=False))
        d_s = d_part.iloc[idx]
        contrib = np.asarray(model.predict(aligner.transform(d_s[feats]), pred_contrib=True), dtype=np.float64)[:, :-1]
        if label.startswith("uji"):
            j = feats.index("Country")
            ref = pd.read_csv(TABLES_DIR / "04_shap_top15.csv").set_index("fitur").loc["Country", "mean_abs_shap"]
            got = float(np.abs(contrib[:, j]).mean())
            if abs(got - float(ref)) > 1e-3:
                raise AssertionError(f"SHAP sample cross-check failed for Country: {got:.6f} vs {ref:.6f}")
            summary["shap_crosscheck_country_mean_abs"] = round(got, 6)
        shap_frames.append(shap_by_nan_group(contrib, d_s[feats], y_part[idx], feats, label, selected))
    shap_tbl = pd.concat(shap_frames, ignore_index=True).merge(conc, on="fitur", how="left")
    shap_tbl.to_csv(TABLES_DIR / "04_nan_signal_shap.csv", index=False)

    # (3b) single report-only refit with NaN filled by one constant
    if not skip_refit:
        print("[Follow-up 3b] Report-only refit (same params, no tuning)...")
        params = model.get_params()
        p_main_cal_te = pred.set_index("LoanId").loc[df_te["LoanId"].astype(str), "p_calibrated"].to_numpy(dtype=float)
        p_main_raw_te = pred.set_index("LoanId").loc[df_te["LoanId"].astype(str), "p_raw"].to_numpy(dtype=float)

        a_main, lo_m, hi_m = bootstrap_auc_ci(y_te, p_main_cal_te, n_bootstrap=200, random_state=RANDOM_SEED)
        r_main, rlo_m, rhi_m = bootstrap_auc_ci(y_te, p_main_raw_te, n_bootstrap=200, random_state=RANDOM_SEED)
        rows = [
            {
                "varian": "LGBM-A utama (tersimpan, NaN apa adanya)",
                "best_iteration": int(art["best_iteration"]),
                "test_auc_raw": round(r_main, 6),
                "test_auc_raw_ci_low_95": round(rlo_m, 6),
                "test_auc_raw_ci_high_95": round(rhi_m, 6),
                "metode_kalibrasi": art["calibrator"].selected_method_,
                "test_auc_calibrated": round(a_main, 6),
                "test_auc_calibrated_ci_low_95": round(lo_m, 6),
                "test_auc_calibrated_ci_high_95": round(hi_m, 6),
                "selisih_auc_vs_lgbm_a_utama": 0.0,
                "test_brier_calibrated": round(float(brier_score_loss(y_te, p_main_cal_te)), 6),
                "test_ece_calibrated": round(compute_ece(y_te, p_main_cal_te, n_bins=10)[0], 6),
                "fitur_diisi": "-",
                "konstanta_pengisi": "-",
            }
        ]

        al_c, clf_c = _fit_lgbm_same_params(params, df_tr, y_tr, df_va, y_va, feats)
        ctrl = _evaluate_refit("Kontrol refit (NaN apa adanya, parameter sama)", al_c, clf_c, df_va, y_va, df_te, y_te,
                               feats, p_main_cal_te)
        ctrl.update({"fitur_diisi": "-", "konstanta_pengisi": "-"})
        rows.append(ctrl)

        fill = fit_fill_values(df_tr, selected)
        d_tr_f = df_tr[feats].copy()
        d_va_f = df_va[feats].copy()
        d_te_f = df_te[feats].copy()
        d_tr_f = apply_fill_values(d_tr_f, fill)
        d_va_f = apply_fill_values(d_va_f, fill)
        d_te_f = apply_fill_values(d_te_f, fill)
        al_f, clf_f = _fit_lgbm_same_params(params, d_tr_f, y_tr, d_va_f, y_va, feats)
        filled = _evaluate_refit("Refit NaN diisi konstanta tunggal (median / modus data latih)", al_f, clf_f, d_va_f,
                                 y_va, d_te_f, y_te, feats, p_main_cal_te)
        filled.update(
            {
                "fitur_diisi": ", ".join(selected),
                "konstanta_pengisi": "; ".join(f"{k}={v:.6g}" if not isinstance(v, str) else f"{k}={v}" for k, v in fill.items()),
            }
        )
        rows.append(filled)

        refit_tbl = pd.DataFrame(rows)
        refit_tbl["selisih_auc_vs_kontrol_refit"] = np.nan
        ctrl_auc = ctrl["test_auc_calibrated"]
        refit_tbl.loc[refit_tbl["varian"].str.startswith("Refit NaN"), "selisih_auc_vs_kontrol_refit"] = round(
            filled["test_auc_calibrated"] - ctrl_auc, 6
        )
        refit_tbl.to_csv(TABLES_DIR / "04_nan_signal_refit.csv", index=False)
        summary["refit"] = refit_tbl[["varian", "test_auc_raw", "test_auc_calibrated", "selisih_auc_vs_lgbm_a_utama"]].to_dict(
            "records"
        )

    # (4) metrics scope annotation
    annotate_metrics_scope()

    summary["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[Follow-up] done in {summary['elapsed_seconds']:.2f}s")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Tahap 4 follow-up evaluation (does not retrain the primary model).")
    parser.add_argument("--skip-refit", action="store_true", help="Skip the report-only NaN-filled refit.")
    args = parser.parse_args(argv)
    return run_followup(skip_refit=args.skip_refit)


if __name__ == "__main__":
    main()
