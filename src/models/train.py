"""End-to-end training, calibration, evaluation, interpretation, and prediction pipeline for Tahap 4.

Usage:
    python -m src.models.train [--task {classification,regression,all}] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from src.analysis.diagnostic import compute_iv_woe
from src.config import (
    DATA_PROCESSED,
    DEFAULT_12M_COL,
    DEFAULT_TARGET_COL,
    FIGURES_DIR,
    LOAN_DATE_COL,
    MODELS_DIR,
    RANDOM_SEED,
    REGRESSION_TARGET_COL,
    REPORT_DATE,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    TABLES_DIR,
    ensure_dirs,
)
from src.models.evaluate import (
    bootstrap_auc_ci,
    compute_classification_lift_table,
    compute_ece,
    compute_regression_lift_table,
    compute_subgroup_performance,
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    find_youden_threshold,
    paired_bootstrap_auc_diff,
)
from src.models.features import build_feature_sets, load_candidate_features_from_schema
from src.models.pipelines import (
    ValidationProbabilityCalibrator,
    train_lgbm_classifier_with_tuning,
    train_lgbm_regressor_with_tuning,
    train_logistic_regression_classifier,
    train_ridge_regressor,
)
from src.models.splits import (
    make_classification_splits,
    make_random_stratified_split,
    make_regression_splits,
)
from src.utils.plotting import fmt_id, save_fig, set_style


def run_preflight_checks(df: pd.DataFrame, schema_meta: dict[str, Any]) -> dict[str, Any]:
    """Verify exact reference numbers required by Tahap 4 Preflight."""
    n_rows, n_cols = df.shape
    n_obs_12m = int(df["observable_12m"].sum())
    n_val_12m = int(df[DEFAULT_12M_COL].notna().sum())
    n_def_12m = int((df[DEFAULT_12M_COL] == 1).sum())
    rate_12m_pct = round(float(n_def_12m / n_val_12m * 100.0), 2)
    n_feat_y = int(schema_meta["n_features_Y"])

    assert (n_rows, n_cols) == (179_235, 130), f"Unexpected shape: {(n_rows, n_cols)}"
    assert n_obs_12m == 148_783, f"Unexpected observable_12m: {n_obs_12m}"
    assert n_val_12m == 148_369, f"Unexpected valid default_12m: {n_val_12m}"
    assert n_def_12m == 43_117, f"Unexpected default_12m sum: {n_def_12m}"
    assert rate_12m_pct == 29.06, f"Unexpected default_12m rate: {rate_12m_pct}%"
    assert n_feat_y == 44, f"Unexpected n_features_Y: {n_feat_y}"
    assert REPORT_DATE == "2021-07-20", f"Unexpected REPORT_DATE: {REPORT_DATE}"

    cand_list = load_candidate_features_from_schema()
    assert len(cand_list) == 44

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "observable_12m": n_obs_12m,
        "valid_default_12m": n_val_12m,
        "sum_default_12m": n_def_12m,
        "rate_default_12m_pct": rate_12m_pct,
        "n_features_Y": n_feat_y,
        "report_date": REPORT_DATE,
    }


def _compute_lgbm_shap_and_examples(
    lgbm_model: Any,
    aligner: Any,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    p_cal_test: np.ndarray,
    sample_size: int = 5000,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compute global SHAP values on 5,000 random test rows and 3 local loan explanations."""
    rng = np.random.default_rng(random_state)
    n_test = len(df_test)
    take_n = min(sample_size, n_test)
    sample_idx = np.sort(rng.choice(n_test, size=take_n, replace=False))

    X_test_al = aligner.transform(df_test[feature_cols])
    X_sample = X_test_al.iloc[sample_idx]

    shap_method_note = "shap.TreeExplainer"
    try:
        import shap

        explainer = shap.TreeExplainer(lgbm_model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            shap_mat = np.asarray(sv[1], dtype=np.float64)
        elif sv.ndim == 3:
            shap_mat = np.asarray(sv[:, :, 1], dtype=np.float64)
        else:
            shap_mat = np.asarray(sv, dtype=np.float64)
    except Exception as exc:
        shap_method_note = f"LightGBM pred_contrib=True (fallback: {exc.__class__.__name__})"
        contrib = lgbm_model.predict(X_sample, pred_contrib=True)
        shap_mat = np.asarray(contrib, dtype=np.float64)[:, :-1]

    mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
    mean_signed_shap = np.mean(shap_mat, axis=0)

    shap_df = (
        pd.DataFrame(
            {
                "fitur": feature_cols,
                "mean_abs_shap": np.round(mean_abs_shap, 6),
                "mean_signed_shap": np.round(mean_signed_shap, 6),
                "n_sampel_uji": take_n,
                "metode": shap_method_note,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df["peringkat_shap"] = np.arange(1, len(shap_df) + 1)
    shap_df = shap_df[
        ["peringkat_shap", "fitur", "mean_abs_shap", "mean_signed_shap", "n_sampel_uji", "metode"]
    ]

    # Pick 3 representative loans on test split: low risk (~p10), median risk (~p50), high risk (~p90)
    q10, q50, q90 = np.quantile(p_cal_test, [0.10, 0.50, 0.90])
    idx_low = int(np.argmin(np.abs(p_cal_test - q10)))
    idx_mid = int(np.argmin(np.abs(p_cal_test - q50)))
    idx_high = int(np.argmin(np.abs(p_cal_test - q90)))

    X_three = X_test_al.iloc[[idx_low, idx_mid, idx_high]]
    contrib_three = np.asarray(lgbm_model.predict(X_three, pred_contrib=True), dtype=np.float64)

    example_rows: list[dict[str, Any]] = []
    for pos_i, (label_profil, row_idx) in enumerate(
        [
            ("Risiko Rendah (~P10)", idx_low),
            ("Risiko Menengah (~Median)", idx_mid),
            ("Risiko Tinggi (~P90)", idx_high),
        ]
    ):
        loan_row = df_test.iloc[row_idx]
        c_vec = contrib_three[pos_i, :-1]
        base_val = float(contrib_three[pos_i, -1])
        top_feat_idx = np.argsort(-np.abs(c_vec))[:8]
        for rank_f, f_i in enumerate(top_feat_idx, start=1):
            fname = feature_cols[f_i]
            fval = loan_row[fname]
            example_rows.append(
                {
                    "profil_pinjaman": label_profil,
                    "LoanId": str(loan_row["LoanId"]),
                    "Country": str(loan_row["Country"]),
                    "Rating": str(loan_row["Rating"]),
                    "default_12m_aktual": int(loan_row[DEFAULT_12M_COL]),
                    "p_calibrated": round(float(p_cal_test[row_idx]), 4),
                    "PD_bondora": round(float(loan_row["ProbabilityOfDefault"]), 4)
                    if pd.notna(loan_row["ProbabilityOfDefault"])
                    else np.nan,
                    "base_value_log_odds": round(base_val, 4),
                    "peringkat_kontribusi": rank_f,
                    "fitur": fname,
                    "nilai_fitur": str(fval),
                    "shap_log_odds": round(float(c_vec[f_i]), 4),
                    "arah_risiko": "Meningkatkan risiko (+)" if c_vec[f_i] > 0 else "Menurunkan risiko (-)",
                }
            )

    return shap_df, pd.DataFrame(example_rows), shap_method_note


def _compute_permutation_importance_val(
    lgbm_model: Any,
    aligner: Any,
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: list[str],
    n_repeats: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute permutation importance (AUC drop on validation set over 5 repeats)."""
    X_val_al = aligner.transform(df_val[feature_cols])
    p_base = lgbm_model.predict_proba(X_val_al)[:, 1]
    base_auc = float(roc_auc_score(y_val, p_base))

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    n_val = len(X_val_al)

    for col in feature_cols:
        drops: list[float] = []
        orig_col = X_val_al[col].copy()
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_val)
            X_val_al[col] = orig_col.iloc[perm_idx].to_numpy()
            if isinstance(orig_col.dtype, pd.CategoricalDtype):
                X_val_al[col] = pd.Categorical(
                    X_val_al[col], categories=orig_col.cat.categories, ordered=False
                )
            p_perm = lgbm_model.predict_proba(X_val_al)[:, 1]
            auc_perm = float(roc_auc_score(y_val, p_perm))
            drops.append(base_auc - auc_perm)
        X_val_al[col] = orig_col

        rows.append(
            {
                "fitur": col,
                "baseline_val_auc": round(base_auc, 6),
                "auc_drop_mean": round(float(np.mean(drops)), 6),
                "auc_drop_std": round(float(np.std(drops, ddof=1)), 6) if n_repeats > 1 else 0.0,
                "n_repeats": n_repeats,
            }
        )

    perm_df = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    perm_df["peringkat_permutation"] = np.arange(1, len(perm_df) + 1)
    return perm_df[
        ["peringkat_permutation", "fitur", "auc_drop_mean", "auc_drop_std", "baseline_val_auc", "n_repeats"]
    ]


def _compute_train_only_iv_comparison(
    df_train: pd.DataFrame,
    candidate_cols: list[str],
    set_a_cols: list[str],
) -> pd.DataFrame:
    """Compute IV strictly on training split and compare top 15 ranks with Tahap 3 full-population IV."""
    t3_iv_path = TABLES_DIR / "03_driver_ranking.csv"
    t3_df = pd.read_csv(t3_iv_path)
    t3_df["peringkat_tahap3"] = np.arange(1, len(t3_df) + 1)
    t3_map_iv = dict(zip(t3_df["fitur"], t3_df["iv_semua"]))
    t3_map_rank = dict(zip(t3_df["fitur"], t3_df["peringkat_tahap3"]))

    set_a_set = set(set_a_cols)
    rows: list[dict[str, Any]] = []
    for col in candidate_cols:
        iv_tr, _ = compute_iv_woe(df_train, col, target_col=DEFAULT_12M_COL)
        rows.append(
            {
                "fitur": col,
                "di_set_A": col in set_a_set,
                "iv_train_only": round(float(iv_tr), 4),
                "iv_tahap3_semua": round(float(t3_map_iv.get(col, np.nan)), 4),
                "peringkat_tahap3": int(t3_map_rank.get(col, 999)),
            }
        )

    iv_df = pd.DataFrame(rows).sort_values("iv_train_only", ascending=False).reset_index(drop=True)
    iv_df["peringkat_train_only"] = np.arange(1, len(iv_df) + 1)
    iv_df["perubahan_peringkat_vs_tahap3"] = iv_df["peringkat_tahap3"] - iv_df["peringkat_train_only"]
    return iv_df[
        [
            "peringkat_train_only",
            "peringkat_tahap3",
            "perubahan_peringkat_vs_tahap3",
            "fitur",
            "di_set_A",
            "iv_train_only",
            "iv_tahap3_semua",
        ]
    ]


def run_classification_pipeline(
    df: pd.DataFrame,
    quick: bool = False,
) -> dict[str, Any]:
    """Execute Steps 1-7 & 9a: classification splits, feature sets, training, calibration, evaluation, and SHAP."""
    t0 = time.perf_counter()
    set_style()

    # Step 1: Temporal and Stratified Random splits
    df_cls, temporal_summary = make_classification_splits(df)
    rand_split_s, rand_summary = make_random_stratified_split(df_cls)
    df_cls["split_random"] = rand_split_s

    split_summary_all = pd.concat([temporal_summary, rand_summary], ignore_index=True)
    split_summary_all.to_csv(TABLES_DIR / "04_split_summary.csv", index=False)

    df_train = df_cls.loc[df_cls["split"] == "train"].copy()
    df_val = df_cls.loc[df_cls["split"] == "val"].copy()
    df_test = df_cls.loc[df_cls["split"] == "test"].copy()

    y_train = df_train[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_val = df_val[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_test = df_test[DEFAULT_12M_COL].astype("int32").to_numpy()

    # Step 2: Feature Sets (Set A, Set B, Set C)
    fsets, fsets_tbl = build_feature_sets(df_train, df_val)
    fsets_tbl.to_csv(TABLES_DIR / "04_feature_sets.csv", index=False)

    set_a = fsets["A"]
    set_b = fsets["B"]
    set_c = fsets["C"]
    print(f"[Step 2] Feature sets: Set A = {len(set_a)}, Set B = {len(set_b)}, Set C = {len(set_c)}")

    n_iter_lgbm = 5 if quick else 20
    n_est_lgbm = 600 if quick else 2000

    # Step 3 & 4: Train Models on Temporal Split
    print("[Step 3] Training LR-A (Logistic Regression on Set A)...")
    lr_a_res = train_logistic_regression_classifier(
        df_train[set_a], y_train, df_val[set_a], y_val
    )
    p_val_lr_raw = lr_a_res["model"].predict_proba(lr_a_res["preprocessor"].transform(df_val[set_a]))[:, 1]
    p_test_lr_raw = lr_a_res["model"].predict_proba(lr_a_res["preprocessor"].transform(df_test[set_a]))[:, 1]
    cal_lr_a = ValidationProbabilityCalibrator(random_state=RANDOM_SEED)
    cal_lr_a.fit(p_val_lr_raw, y_val)
    p_val_lr_cal = cal_lr_a.transform(p_val_lr_raw)
    p_test_lr_cal = cal_lr_a.transform(p_test_lr_raw)

    print(f"[Step 3] Training LGBM-A (LightGBM on Set A, n_iter={n_iter_lgbm})...")
    lgbm_a_res = train_lgbm_classifier_with_tuning(
        df_train[set_a],
        y_train,
        df_val[set_a],
        y_val,
        n_iter=n_iter_lgbm,
        n_estimators=n_est_lgbm,
        random_state=RANDOM_SEED,
        check_unbalance=True,
    )
    p_val_lgbm_a_raw = lgbm_a_res["model"].predict_proba(lgbm_a_res["aligner"].transform(df_val[set_a]))[:, 1]
    p_test_lgbm_a_raw = lgbm_a_res["model"].predict_proba(lgbm_a_res["aligner"].transform(df_test[set_a]))[:, 1]

    cal_lgbm_a = ValidationProbabilityCalibrator(random_state=RANDOM_SEED)
    cal_lgbm_a.fit(p_val_lgbm_a_raw, y_val)
    p_val_lgbm_a_cal = cal_lgbm_a.transform(p_val_lgbm_a_raw)
    p_test_lgbm_a_cal = cal_lgbm_a.transform(p_test_lgbm_a_raw)
    p_test_lgbm_a_platt = cal_lgbm_a.transform(p_test_lgbm_a_raw, method="sigmoid")
    p_test_lgbm_a_iso = cal_lgbm_a.transform(p_test_lgbm_a_raw, method="isotonic")

    print(f"[Step 3] Training LGBM-B (LightGBM on Set B, n_iter={n_iter_lgbm})...")
    lgbm_b_res = train_lgbm_classifier_with_tuning(
        df_train[set_b],
        y_train,
        df_val[set_b],
        y_val,
        n_iter=n_iter_lgbm,
        n_estimators=n_est_lgbm,
        random_state=RANDOM_SEED,
    )
    p_val_lgbm_b_raw = lgbm_b_res["model"].predict_proba(lgbm_b_res["aligner"].transform(df_val[set_b]))[:, 1]
    p_test_lgbm_b_raw = lgbm_b_res["model"].predict_proba(lgbm_b_res["aligner"].transform(df_test[set_b]))[:, 1]
    cal_lgbm_b = ValidationProbabilityCalibrator(random_state=RANDOM_SEED).fit(p_val_lgbm_b_raw, y_val)
    p_val_lgbm_b_cal = cal_lgbm_b.transform(p_val_lgbm_b_raw)
    p_test_lgbm_b_cal = cal_lgbm_b.transform(p_test_lgbm_b_raw)

    print(f"[Step 3] Training LGBM-C (LightGBM on Set C Ablation, n_iter={n_iter_lgbm})...")
    lgbm_c_res = train_lgbm_classifier_with_tuning(
        df_train[set_c],
        y_train,
        df_val[set_c],
        y_val,
        n_iter=n_iter_lgbm,
        n_estimators=n_est_lgbm,
        random_state=RANDOM_SEED,
    )
    p_val_lgbm_c_raw = lgbm_c_res["model"].predict_proba(lgbm_c_res["aligner"].transform(df_val[set_c]))[:, 1]
    p_test_lgbm_c_raw = lgbm_c_res["model"].predict_proba(lgbm_c_res["aligner"].transform(df_test[set_c]))[:, 1]
    cal_lgbm_c = ValidationProbabilityCalibrator(random_state=RANDOM_SEED).fit(p_val_lgbm_c_raw, y_val)
    p_val_lgbm_c_cal = cal_lgbm_c.transform(p_val_lgbm_c_raw)
    p_test_lgbm_c_cal = cal_lgbm_c.transform(p_test_lgbm_c_raw)

    # Random split optimism check: LGBM-A-acak
    print(f"[Step 5] Training LGBM-A-acak (Random 70/15/15 split on Set A)...")
    df_tr_rnd = df_cls.loc[df_cls["split_random"] == "train"]
    df_va_rnd = df_cls.loc[df_cls["split_random"] == "val"]
    df_te_rnd = df_cls.loc[df_cls["split_random"] == "test"]
    y_tr_rnd = df_tr_rnd[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_va_rnd = df_va_rnd[DEFAULT_12M_COL].astype("int32").to_numpy()
    y_te_rnd = df_te_rnd[DEFAULT_12M_COL].astype("int32").to_numpy()

    lgbm_rnd_res = train_lgbm_classifier_with_tuning(
        df_tr_rnd[set_a],
        y_tr_rnd,
        df_va_rnd[set_a],
        y_va_rnd,
        n_iter=n_iter_lgbm,
        n_estimators=n_est_lgbm,
        random_state=RANDOM_SEED,
    )
    p_va_rnd_raw = lgbm_rnd_res["model"].predict_proba(lgbm_rnd_res["aligner"].transform(df_va_rnd[set_a]))[:, 1]
    p_te_rnd_raw = lgbm_rnd_res["model"].predict_proba(lgbm_rnd_res["aligner"].transform(df_te_rnd[set_a]))[:, 1]
    cal_rnd = ValidationProbabilityCalibrator(random_state=RANDOM_SEED).fit(p_va_rnd_raw, y_va_rnd)
    p_va_rnd_cal = cal_rnd.transform(p_va_rnd_raw)
    p_te_rnd_cal = cal_rnd.transform(p_te_rnd_raw)

    # Step 5: Evaluate all models & baselines on validation and test splits
    thr_lr_a = find_youden_threshold(y_val, p_val_lr_cal)
    thr_lgbm_a_raw = find_youden_threshold(y_val, p_val_lgbm_a_raw)
    thr_lgbm_a = find_youden_threshold(y_val, p_val_lgbm_a_cal)
    thr_lgbm_b = find_youden_threshold(y_val, p_val_lgbm_b_cal)
    thr_lgbm_c = find_youden_threshold(y_val, p_val_lgbm_c_cal)
    thr_rnd = find_youden_threshold(y_va_rnd, p_va_rnd_cal)

    metrics_rows: list[dict[str, Any]] = [
        # Validation metrics
        evaluate_classification_predictions(
            y_val, p_val_lr_cal, thr_lr_a, "LR-A", "Logistic Regression (Terkalibrasi)", "Set A (36 fitur)", "val"
        ),
        evaluate_classification_predictions(
            y_val, p_val_lgbm_a_raw, thr_lgbm_a_raw, "LGBM-A-raw", "LightGBM Utama (Sebelum Kalibrasi)", "Set A (36 fitur)", "val"
        ),
        evaluate_classification_predictions(
            y_val, p_val_lgbm_a_cal, thr_lgbm_a, "LGBM-A", "LightGBM Utama (Terkalibrasi)", "Set A (36 fitur)", "val"
        ),
        evaluate_classification_predictions(
            y_val, p_val_lgbm_b_cal, thr_lgbm_b, "LGBM-B", "LightGBM Lengkap (Terkalibrasi)", "Set B (44 fitur)", "val"
        ),
        evaluate_classification_predictions(
            y_val, p_val_lgbm_c_cal, thr_lgbm_c, "LGBM-C", "LightGBM Ablasi + Rating/PD/Interest", "Set C (39 fitur)", "val"
        ),
        evaluate_classification_predictions(
            y_va_rnd, p_va_rnd_cal, thr_rnd, "LGBM-A-acak", "LightGBM Split Acak 70/15/15", "Set A (36 fitur)", "val_random"
        ),
        # Test metrics (full test split n = 54,283)
        evaluate_classification_predictions(
            y_test, p_test_lr_raw, find_youden_threshold(y_val, p_val_lr_raw), "LR-A-raw", "Logistic Regression (Sebelum Kalibrasi)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lr_cal, thr_lr_a, "LR-A", "Logistic Regression (Terkalibrasi)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_a_raw, thr_lgbm_a_raw, "LGBM-A-raw", "LightGBM Utama (Sebelum Kalibrasi)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_a_platt, find_youden_threshold(y_val, cal_lgbm_a.transform(p_val_lgbm_a_raw, "sigmoid")), "LGBM-A-sigmoid", "LightGBM Utama (Kalibrasi Platt/Sigmoid)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_a_iso, find_youden_threshold(y_val, cal_lgbm_a.transform(p_val_lgbm_a_raw, "isotonic")), "LGBM-A-isotonic", "LightGBM Utama (Kalibrasi Isotonik)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_a_cal, thr_lgbm_a, "LGBM-A", "LightGBM Utama (Terkalibrasi Terpilih)", "Set A (36 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_b_cal, thr_lgbm_b, "LGBM-B", "LightGBM Lengkap (Terkalibrasi)", "Set B (44 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_test, p_test_lgbm_c_cal, thr_lgbm_c, "LGBM-C", "LightGBM Ablasi + Rating/PD/Interest", "Set C (39 fitur)", "test"
        ),
        evaluate_classification_predictions(
            y_te_rnd, p_te_rnd_cal, thr_rnd, "LGBM-A-acak", "LightGBM Split Acak 70/15/15", "Set A (36 fitur)", "test_random"
        ),
    ]

    # Baselines on common intersection of test rows where PD and Interest are non-null
    mask_common_test = df_test["ProbabilityOfDefault"].notna() & df_test["Interest"].notna()
    mask_common_val = df_val["ProbabilityOfDefault"].notna() & df_val["Interest"].notna()
    y_test_comm = y_test[mask_common_test.to_numpy()]
    pd_test_comm = pd.to_numeric(df_test.loc[mask_common_test, "ProbabilityOfDefault"], errors="coerce").to_numpy()
    int_test_comm = pd.to_numeric(df_test.loc[mask_common_test, "Interest"], errors="coerce").to_numpy()
    train_base_prob = np.full(len(y_test_comm), float(y_train.mean()), dtype=np.float64)

    pd_val_comm = pd.to_numeric(df_val.loc[mask_common_val, "ProbabilityOfDefault"], errors="coerce").to_numpy()
    thr_pd_val = find_youden_threshold(y_val[mask_common_val.to_numpy()], pd_val_comm)

    metrics_rows.extend(
        [
            evaluate_classification_predictions(
                y_test_comm,
                train_base_prob,
                0.5,
                "BASE-Prior",
                "Pembanding: Proporsi Default Data Latih (0,3103)",
                "Tanpa Fitur",
                "test_irisan_benchmark",
                is_probability=True,
                n_bootstrap=0,
            ),
            evaluate_classification_predictions(
                y_test_comm,
                pd_test_comm,
                thr_pd_val,
                "BASE-PD-Bondora",
                "Pembanding: ProbabilityOfDefault Bondora",
                "Benchmark Platform",
                "test_irisan_benchmark",
                is_probability=True,
                n_bootstrap=200,
            ),
            evaluate_classification_predictions(
                y_test_comm,
                int_test_comm,
                np.nan,
                "BASE-Interest",
                "Pembanding: Suku Bunga Interest (Skor Peringkat)",
                "Pricing Platform",
                "test_irisan_benchmark",
                is_probability=False,
                n_bootstrap=200,
            ),
            evaluate_classification_predictions(
                y_test_comm,
                p_test_lgbm_a_cal[mask_common_test.to_numpy()],
                thr_lgbm_a,
                "LGBM-A-irisan",
                "LightGBM Utama pada Baris Irisan Benchmark",
                "Set A (36 fitur)",
                "test_irisan_benchmark",
                is_probability=True,
                n_bootstrap=200,
            ),
        ]
    )

    metrics_cls_df = pd.DataFrame(metrics_rows)
    metrics_cls_df.to_csv(TABLES_DIR / "04_metrics_classification.csv", index=False)

    # Paired bootstrap AUC comparisons (Step 6.3 & 6.4)
    auc_comparisons = [
        {
            "perbandingan": "LGBM-A (Set A) vs LGBM-B (Set B Lengkap)",
            "populasi_uji": "Seluruh Uji (n = 54.283)",
            "n": int(len(y_test)),
            **paired_bootstrap_auc_diff(y_test, p_test_lgbm_a_cal, p_test_lgbm_b_cal),
        },
        {
            "perbandingan": "LGBM-C (Set A + Rating/PD/Interest) vs LGBM-A (Set A)",
            "populasi_uji": "Seluruh Uji (n = 54.283)",
            "n": int(len(y_test)),
            **paired_bootstrap_auc_diff(y_test, p_test_lgbm_c_cal, p_test_lgbm_a_cal),
        },
        {
            "perbandingan": "LGBM-A (Set A) vs LR-A (Logistic Regression Set A)",
            "populasi_uji": "Seluruh Uji (n = 54.283)",
            "n": int(len(y_test)),
            **paired_bootstrap_auc_diff(y_test, p_test_lgbm_a_cal, p_test_lr_cal),
        },
        {
            "perbandingan": "LGBM-A (Set A) vs ProbabilityOfDefault Bondora",
            "populasi_uji": f"Irisan Benchmark Uji (n = {int(mask_common_test.sum())})",
            "n": int(mask_common_test.sum()),
            **paired_bootstrap_auc_diff(
                y_test_comm, p_test_lgbm_a_cal[mask_common_test.to_numpy()], pd_test_comm
            ),
        },
        {
            "perbandingan": "LGBM-A (Set A) vs Suku Bunga Interest",
            "populasi_uji": f"Irisan Benchmark Uji (n = {int(mask_common_test.sum())})",
            "n": int(mask_common_test.sum()),
            **paired_bootstrap_auc_diff(
                y_test_comm, p_test_lgbm_a_cal[mask_common_test.to_numpy()], int_test_comm
            ),
        },
    ]
    auc_comp_df = pd.DataFrame(auc_comparisons)
    auc_comp_df.to_csv(TABLES_DIR / "04_auc_comparisons.csv", index=False)

    # Decile lift table for LGBM-A on test split
    lift_cls_df = compute_classification_lift_table(y_test, p_test_lgbm_a_cal, n_deciles=10)
    lift_cls_df.to_csv(TABLES_DIR / "04_lift_by_decile.csv", index=False)

    # Step 6.1: Subgroup performance on test split
    subgrp_df = compute_subgroup_performance(df_test, y_test, p_test_lgbm_a_raw, p_test_lgbm_a_cal)
    subgrp_df.to_csv(TABLES_DIR / "04_subgroup_performance.csv", index=False)

    # Step 7: SHAP (5,000 test rows), Permutation Importance (validation), LR-A coefficients, Train-only IV
    print("[Step 7] Computing SHAP values, permutation importance, LR coefficients, and train-only IV...")
    shap_cls_df, local_ex_df, shap_note = _compute_lgbm_shap_and_examples(
        lgbm_a_res["model"],
        lgbm_a_res["aligner"],
        df_test,
        set_a,
        p_test_lgbm_a_cal,
        sample_size=5000,
        random_state=RANDOM_SEED,
    )
    shap_cls_df.head(15).to_csv(TABLES_DIR / "04_shap_top15.csv", index=False)
    local_ex_df.to_csv(TABLES_DIR / "04_local_explanations_examples.csv", index=False)

    perm_df = _compute_permutation_importance_val(
        lgbm_a_res["model"],
        lgbm_a_res["aligner"],
        df_val,
        y_val,
        set_a,
        n_repeats=5,
        random_state=RANDOM_SEED,
    )
    perm_df.to_csv(TABLES_DIR / "04_permutation_importance.csv", index=False)

    lr_feat_names = lr_a_res["preprocessor"].get_feature_names_out()
    lr_coefs = lr_a_res["model"].coef_.ravel()
    lr_coef_df = (
        pd.DataFrame(
            {
                "fitur_transformed": lr_feat_names,
                "koefisien_terstandarisasi": np.round(lr_coefs, 6),
                "abs_koefisien": np.round(np.abs(lr_coefs), 6),
                "arah": np.where(lr_coefs >= 0, "Positif (+ risiko default)", "Negatif (- risiko default)"),
            }
        )
        .sort_values("abs_koefisien", ascending=False)
        .reset_index(drop=True)
    )
    lr_coef_df["peringkat"] = np.arange(1, len(lr_coef_df) + 1)
    lr_coef_df = lr_coef_df[
        ["peringkat", "fitur_transformed", "koefisien_terstandarisasi", "abs_koefisien", "arah"]
    ]
    lr_coef_df.head(15).to_csv(TABLES_DIR / "04_lr_top15_coefficients.csv", index=False)

    iv_train_df = _compute_train_only_iv_comparison(df_train, set_b, set_a)
    iv_train_df.to_csv(TABLES_DIR / "04_driver_ranking_train_only.csv", index=False)

    # Step 9a: Save predictions_default12m.parquet for all 148,369 valid rows
    print("[Step 9a] Generating data/processed/predictions_default12m.parquet...")
    X_all_lgbm = lgbm_a_res["aligner"].transform(df_cls[set_a])
    p_all_raw = lgbm_a_res["model"].predict_proba(X_all_lgbm)[:, 1]
    p_all_cal = cal_lgbm_a.transform(p_all_raw)

    X_all_lr = lr_a_res["preprocessor"].transform(df_cls[set_a])
    p_all_lr = cal_lr_a.transform(lr_a_res["model"].predict_proba(X_all_lr)[:, 1])

    pred_cls_df = pd.DataFrame(
        {
            "LoanId": df_cls["LoanId"].astype(str),
            "split": df_cls["split"].astype(str),
            "LoanDate": df_cls["LoanDate"],
            "loan_year": df_cls["loan_year"].astype("int32"),
            "Country": df_cls["Country"].astype(str),
            "Rating": df_cls["Rating"].astype("object").where(df_cls["Rating"].notna(), None),
            "Amount": pd.to_numeric(df_cls["Amount"], errors="coerce").astype("float64"),
            "AppliedAmount": pd.to_numeric(df_cls["AppliedAmount"], errors="coerce").astype("float64"),
            "Interest": pd.to_numeric(df_cls["Interest"], errors="coerce").astype("float64"),
            "LoanDuration": pd.to_numeric(df_cls["LoanDuration"], errors="coerce").astype("int32"),
            "default_12m": df_cls[DEFAULT_12M_COL].astype("int8"),
            "p_raw": p_all_raw.astype("float64"),
            "p_calibrated": p_all_cal.astype("float64"),
            "p_lr": p_all_lr.astype("float64"),
            "PD_bondora": pd.to_numeric(df_cls["ProbabilityOfDefault"], errors="coerce").astype("float64"),
        },
        index=df_cls.index,
    )
    assert pred_cls_df["LoanId"].is_unique, "LoanId must be unique in predictions_default12m.parquet"
    assert ((pred_cls_df["p_raw"] >= 0.0) & (pred_cls_df["p_raw"] <= 1.0)).all()
    assert ((pred_cls_df["p_calibrated"] >= 0.0) & (pred_cls_df["p_calibrated"] <= 1.0)).all()
    assert ((pred_cls_df["p_lr"] >= 0.0) & (pred_cls_df["p_lr"] <= 1.0)).all()
    assert pred_cls_df[["p_raw", "p_calibrated", "p_lr"]].isna().sum().sum() == 0
    pred_cls_df.to_parquet(DATA_PROCESSED / "predictions_default12m.parquet", index=False)

    # Save classification models to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_code": "LGBM-A",
            "feature_set": "A",
            "features": set_a,
            "aligner": lgbm_a_res["aligner"],
            "model": lgbm_a_res["model"],
            "calibrator": cal_lgbm_a,
            "best_params": lgbm_a_res["best_params"],
            "best_iteration": lgbm_a_res["best_iteration"],
            "threshold_youden_val": thr_lgbm_a,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "lgbm_default12m_setA.joblib",
    )
    joblib.dump(
        {
            "model_code": "LR-A",
            "feature_set": "A",
            "features": set_a,
            "preprocessor": lr_a_res["preprocessor"],
            "model": lr_a_res["model"],
            "calibrator": cal_lr_a,
            "best_C": lr_a_res["best_C"],
            "threshold_youden_val": thr_lr_a,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "lr_default12m_setA.joblib",
    )

    # Generate Classification Figures (04_*.png)
    _plot_classification_figures(
        y_test=y_test,
        y_test_comm=y_test_comm,
        p_test_lgbm_a_raw=p_test_lgbm_a_raw,
        p_test_lgbm_a_cal=p_test_lgbm_a_cal,
        p_test_lgbm_a_platt=p_test_lgbm_a_platt,
        p_test_lgbm_a_iso=p_test_lgbm_a_iso,
        p_test_lgbm_b_cal=p_test_lgbm_b_cal,
        p_test_lgbm_c_cal=p_test_lgbm_c_cal,
        p_test_lr_cal=p_test_lr_cal,
        pd_test_comm=pd_test_comm,
        int_test_comm=int_test_comm,
        y_te_rnd=y_te_rnd,
        p_te_rnd_cal=p_te_rnd_cal,
        lift_cls_df=lift_cls_df,
        subgrp_df=subgrp_df,
        shap_cls_df=shap_cls_df,
        perm_df=perm_df,
    )

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": round(elapsed, 2),
        "n_set_A": len(set_a),
        "n_set_B": len(set_b),
        "n_set_C": len(set_c),
        "era_dead_features": fsets_tbl.loc[~fsets_tbl["di_set_A"] & fsets_tbl["di_set_B"], "fitur"].tolist(),
        "lr_a_best_C": lr_a_res["best_C"],
        "lr_a_grid": lr_a_res["grid_search"],
        "lgbm_a_best_params": lgbm_a_res["best_params"],
        "lgbm_a_best_iteration": lgbm_a_res["best_iteration"],
        "lgbm_a_unbalance_comparison": lgbm_a_res["unbalance_comparison"],
        "lgbm_a_calibration_cv": cal_lgbm_a.cv_results_,
        "lgbm_a_selected_calibration": cal_lgbm_a.selected_method_,
        "lgbm_b_best_params": lgbm_b_res["best_params"],
        "lgbm_b_best_iteration": lgbm_b_res["best_iteration"],
        "lgbm_c_best_params": lgbm_c_res["best_params"],
        "lgbm_c_best_iteration": lgbm_c_res["best_iteration"],
        "lgbm_rnd_best_params": lgbm_rnd_res["best_params"],
        "lgbm_rnd_best_iteration": lgbm_rnd_res["best_iteration"],
        "shap_method": shap_note,
    }


def _plot_classification_figures(
    y_test: np.ndarray,
    y_test_comm: np.ndarray,
    p_test_lgbm_a_raw: np.ndarray,
    p_test_lgbm_a_cal: np.ndarray,
    p_test_lgbm_a_platt: np.ndarray,
    p_test_lgbm_a_iso: np.ndarray,
    p_test_lgbm_b_cal: np.ndarray,
    p_test_lgbm_c_cal: np.ndarray,
    p_test_lr_cal: np.ndarray,
    pd_test_comm: np.ndarray,
    int_test_comm: np.ndarray,
    y_te_rnd: np.ndarray,
    p_te_rnd_cal: np.ndarray,
    lift_cls_df: pd.DataFrame,
    subgrp_df: pd.DataFrame,
    shap_cls_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> None:
    """Create and save all 04_*.png visual artifacts."""
    # 1. 04_roc_curves.png
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    curve_specs = [
        (y_test, p_test_lgbm_a_cal, "LGBM-A (Utama, Set A)", "#1f77b4", "-", 2.4),
        (y_test, p_test_lgbm_b_cal, "LGBM-B (Lengkap, Set B)", "#2ca02c", "--", 1.9),
        (y_test, p_test_lgbm_c_cal, "LGBM-C (Ablasi + Rating/PD/Interest)", "#9467bd", "-.", 2.0),
        (y_test, p_test_lr_cal, "LR-A (Logistic Regression, Set A)", "#ff7f0e", "-", 1.8),
        (y_te_rnd, p_te_rnd_cal, "LGBM-A-acak (Split Acak 70/15/15 — Optimistis)", "#8c564b", ":", 1.9),
        (y_test_comm, pd_test_comm, "Benchmark: ProbabilityOfDefault Bondora", "#c44e52", "--", 2.0),
        (y_test_comm, int_test_comm, "Benchmark: Suku Bunga (Interest)", "#e377c2", ":", 1.8),
    ]
    for yt, yp, lbl, col, ls, lw in curve_specs:
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_v = float(roc_auc_score(yt, yp))
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f"{lbl} (AUC = {fmt_id(auc_v, 4)})")

    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.1, label="Acak (AUC = 0,5000)")
    ax.set_title("Perbandingan Kurva ROC pada Data Uji (Klasifikasi Gagal Bayar 12 Bulan / default_12m)")
    ax.set_xlabel("False Positive Rate (1 - Spesifisitas)")
    ax.set_ylabel("True Positive Rate (Sensitivitas / Recall)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=8.8)
    fig.tight_layout()
    save_fig(fig, "04_roc_curves.png", close=True)

    # 2. 04_calibration_curve.png
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    ax0, ax1 = axes[0], axes[1]
    cal_specs = [
        (y_test, p_test_lgbm_a_raw, "LGBM-A Sebelum Kalibrasi (Raw)", "#ff7f0e", "--", "o"),
        (y_test, p_test_lgbm_a_platt, "LGBM-A Kalibrasi Platt (Sigmoid)", "#2ca02c", ":", "^"),
        (y_test, p_test_lgbm_a_iso, "LGBM-A Kalibrasi Isotonik (Terpilih)", "#1f77b4", "-", "s"),
        (y_test, p_test_lr_cal, "LR-A Terkalibrasi", "#9467bd", "-.", "d"),
        (y_test_comm, pd_test_comm, "ProbabilityOfDefault Bondora", "#c44e52", ":", "x"),
    ]
    ax0.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Kalibrasi Sempurna (Diagonal)")
    for yt, yp, lbl, col, ls, mk in cal_specs:
        ece_v, tbl_b = compute_ece(yt, yp, n_bins=10)
        brier_v = float(np.mean((yt - yp) ** 2))
        valid_b = tbl_b.loc[tbl_b["n"] >= 20]
        ax0.plot(
            valid_b["mean_predicted_prob"],
            valid_b["observed_default_rate"],
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=1.9,
            label=f"{lbl}\n(Brier={fmt_id(brier_v, 4)}, ECE={fmt_id(ece_v, 4)})",
        )
    ax0.set_title("Kurva Reliabilitas (Kalibrasi) pada Data Uji (10 Bin)")
    ax0.set_xlabel("Rata-rata Probabilitas Prediksi")
    ax0.set_ylabel("Proporsi Default Teramati (default_12m)")
    ax0.set_xlim(0, 1.0)
    ax0.set_ylim(0, 1.0)
    ax0.legend(loc="upper left", fontsize=8.0)

    ax1.hist(p_test_lgbm_a_raw, bins=35, alpha=0.55, color="#ff7f0e", label="LGBM-A Sebelum Kalibrasi (Raw)", density=True)
    ax1.hist(p_test_lgbm_a_cal, bins=35, alpha=0.55, color="#1f77b4", label="LGBM-A Sesudah Kalibrasi", density=True)
    ax1.axvline(float(y_test.mean()), color="#c44e52", linestyle="--", linewidth=1.8, label=f"Default Rate Uji Aktual ({fmt_id(y_test.mean()*100, 2)}%)")
    ax1.set_title("Distribusi Probabilitas Prediksi LGBM-A pada Data Uji")
    ax1.set_xlabel("Probabilitas Prediksi Default 12 Bulan")
    ax1.set_ylabel("Kerapatan (Density)")
    ax1.legend(loc="upper right", fontsize=8.8)

    fig.tight_layout()
    save_fig(fig, "04_calibration_curve.png", close=True)

    # 3. 04_lift_by_decile.png
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))
    ax_l0, ax_l1 = axes[0], axes[1]
    x_d = lift_cls_df["desil"].to_numpy()
    obs_pct = lift_cls_df["observed_default_rate"].to_numpy() * 100
    pred_pct = lift_cls_df["mean_predicted_prob"].to_numpy() * 100
    base_pct = float(y_test.mean() * 100)

    w = 0.38
    b_obs = ax_l0.bar(x_d - w / 2, obs_pct, width=w, color="#c44e52", label="Default Rate Aktual (%)")
    ax_l0.bar(x_d + w / 2, pred_pct, width=w, color="#1f77b4", label="Rata-rata Probabilitas Terkalibrasi (%)")
    ax_l0.axhline(base_pct, color="black", linestyle="--", linewidth=1.3, label=f"Rata-rata Populasi Uji ({fmt_id(base_pct, 2)}%)")
    for bar, lift_v in zip(b_obs, lift_cls_df["lift_desil"]):
        ax_l0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{fmt_id(lift_v, 2)}x",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
    ax_l0.set_xticks(x_d)
    ax_l0.set_xticklabels([f"D{d}" for d in x_d])
    ax_l0.set_ylim(0, max(obs_pct.max(), pred_pct.max()) * 1.18)
    ax_l0.set_title("Tingkat Default Aktual vs Prediksi per Desil Risiko (D1 = Tertinggi)")
    ax_l0.set_xlabel("Desil Risiko Prediksi (LGBM-A pada Data Uji, n = 54.283)")
    ax_l0.set_ylabel("Default Rate 12 Bulan (%)")
    ax_l0.legend(loc="upper right", fontsize=8.5)

    pop_frac = np.arange(1, 11) * 10
    cum_rec_pct = lift_cls_df["cum_recall_captured"].to_numpy() * 100
    ax_l1.plot(pop_frac, cum_rec_pct, marker="o", linewidth=2.3, color="#1f77b4", label="Kurva Cumulative Gain (LGBM-A)")
    ax_l1.plot([0, 100], [0, 100], linestyle="--", color="#666666", label="Garis Acak (Baseline)")
    for pf, cr in zip(pop_frac[:3], cum_rec_pct[:3]):
        ax_l1.annotate(
            f"Top {pf}%: {fmt_id(cr, 1)}% default",
            xy=(pf, cr),
            xytext=(pf + 4, cr - 6),
            fontsize=8.3,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.0},
        )
    ax_l1.set_title("Kurva Cumulative Recall (Proporsi Default Tertangkap per Desil)")
    ax_l1.set_xlabel("Persentase Pinjaman Risiko Tertinggi yang Dipilih (%)")
    ax_l1.set_ylabel("Persentase Kejadian Default yang Tertangkap (%)")
    ax_l1.set_xlim(0, 105)
    ax_l1.set_ylim(0, 105)
    ax_l1.legend(loc="lower right", fontsize=8.8)

    fig.tight_layout()
    save_fig(fig, "04_lift_by_decile.png", close=True)

    # 4. 04_shap_summary_bar.png (Top 15 SHAP + Top 15 Permutation Importance)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    top_s = shap_cls_df.head(15).iloc[::-1]
    axes[0].barh(top_s["fitur"], top_s["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.88)
    for idx_b, (f_v, m_v) in enumerate(zip(top_s["fitur"], top_s["mean_abs_shap"])):
        axes[0].text(m_v + top_s["mean_abs_shap"].max() * 0.015, idx_b, fmt_id(m_v, 4), va="center", fontsize=8.2)
    axes[0].set_xlim(0, top_s["mean_abs_shap"].max() * 1.18)
    axes[0].set_title("15 Fitur Terpenting LGBM-A menurut Mean |SHAP| (Sampel Uji n = 5.000)")
    axes[0].set_xlabel("Rata-rata |Kontribusi SHAP| (Skala Log-Odds)")

    top_p = perm_df.head(15).iloc[::-1]
    axes[1].barh(
        top_p["fitur"],
        top_p["auc_drop_mean"],
        xerr=top_p["auc_drop_std"],
        capsize=2.5,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.88,
    )
    for idx_b, m_v in enumerate(top_p["auc_drop_mean"]):
        axes[1].text(max(m_v, 0) + top_p["auc_drop_mean"].max() * 0.02, idx_b, fmt_id(m_v, 4), va="center", fontsize=8.2)
    axes[1].set_xlim(min(0.0, float(top_p["auc_drop_mean"].min()) * 1.1), top_p["auc_drop_mean"].max() * 1.22)
    axes[1].set_title("15 Fitur Terpenting LGBM-A menurut Permutation Importance (Data Validasi)")
    axes[1].set_xlabel("Penurunan ROC-AUC Validasi saat Fitur Diacak (5 Ulangan)")

    fig.tight_layout()
    save_fig(fig, "04_shap_summary_bar.png", close=True)

    # 5. 04_subgroup_performance.png
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    labels_sg = [f"{r['dimensi']}: {r['subgrup']}\n(n={fmt_id(r['n'], is_int=True)})" for _, r in subgrp_df.iterrows()][::-1]
    aucs_sg = subgrp_df["roc_auc"].to_numpy()[::-1]
    ci_l_sg = subgrp_df["auc_ci_low_95"].to_numpy()[::-1]
    ci_h_sg = subgrp_df["auc_ci_high_95"].to_numpy()[::-1]
    y_sg = np.arange(len(subgrp_df))

    err_sg = np.vstack([np.maximum(0, aucs_sg - ci_l_sg), np.maximum(0, ci_h_sg - aucs_sg)])
    axes[0].barh(y_sg, aucs_sg, xerr=err_sg, capsize=2.5, color="#1f77b4", alpha=0.88)
    axes[0].axvline(0.5, color="#666666", linestyle="--", linewidth=1.1)
    axes[0].set_yticks(y_sg)
    axes[0].set_yticklabels(labels_sg, fontsize=8.0)
    axes[0].set_xlim(0.45, 0.90)
    axes[0].set_title("ROC-AUC LGBM-A per Subgrup pada Data Uji (CI Bootstrap 95%)")
    axes[0].set_xlabel("ROC-AUC")

    obs_sg = subgrp_df["observed_default_rate"].to_numpy()[::-1] * 100
    cal_sg = subgrp_df["mean_pred_prob_calibrated"].to_numpy()[::-1] * 100
    h_sg = 0.36
    axes[1].barh(y_sg + h_sg / 2, obs_sg, height=h_sg, color="#c44e52", label="Default Rate Aktual (%)")
    axes[1].barh(y_sg - h_sg / 2, cal_sg, height=h_sg, color="#1f77b4", label="Rata-rata Prediksi Terkalibrasi (%)")
    axes[1].set_yticks(y_sg)
    axes[1].set_yticklabels(labels_sg, fontsize=8.0)
    axes[1].set_title("Kalibrasi Rata-rata Prediksi vs Default Rate Aktual per Subgrup Uji")
    axes[1].set_xlabel("Default Rate / Probabilitas Prediksi (%)")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    save_fig(fig, "04_subgroup_performance.png", close=True)


def run_regression_pipeline(
    df: pd.DataFrame,
    set_a_cols: list[str],
    quick: bool = False,
) -> dict[str, Any]:
    """Execute Steps 8 & 9b: regression splits, baselines, Ridge & LightGBM (L2 vs Huber), evaluation, and SHAP."""
    t0 = time.perf_counter()
    set_style()

    df_reg, reg_split_summary = make_regression_splits(df)
    reg_split_summary.to_csv(TABLES_DIR / "05_split_summary.csv", index=False)

    df_tr = df_reg.loc[df_reg["split"] == "train"].copy()
    df_va = df_reg.loc[df_reg["split"] == "val"].copy()
    df_te = df_reg.loc[df_reg["split"] == "test"].copy()

    y_tr = pd.to_numeric(df_tr[REGRESSION_TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    y_va = pd.to_numeric(df_va[REGRESSION_TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    y_te = pd.to_numeric(df_te[REGRESSION_TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    def_va = df_va[DEFAULT_TARGET_COL].astype("int32").to_numpy()
    def_te = df_te[DEFAULT_TARGET_COL].astype("int32").to_numpy()

    # Step 8.3: Baselines learned strictly from training split
    global_train_med = float(np.median(y_tr))
    rat_tr_str = df_tr["Rating"].astype("object").where(df_tr["Rating"].notna(), "(Missing)").astype(str)
    rat_med_map = df_tr.groupby(rat_tr_str, observed=False)[REGRESSION_TARGET_COL].median().to_dict()

    c_tr_str = df_tr["Country"].astype(str)
    cr_med_map = (
        df_tr.assign(_c=c_tr_str, _r=rat_tr_str)
        .groupby(["_c", "_r"], observed=False)[REGRESSION_TARGET_COL]
        .median()
        .to_dict()
    )
    c_med_map = df_tr.groupby(c_tr_str, observed=False)[REGRESSION_TARGET_COL].median().to_dict()

    def _predict_group_medians(sub_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        r_s = sub_df["Rating"].astype("object").where(sub_df["Rating"].notna(), "(Missing)").astype(str)
        c_s = sub_df["Country"].astype(str)
        pred_r = np.array([float(rat_med_map.get(rv, global_train_med)) for rv in r_s], dtype=np.float64)
        pred_cr = np.array(
            [
                float(cr_med_map.get((cv, rv), c_med_map.get(cv, global_train_med)))
                for cv, rv in zip(c_s, r_s)
            ],
            dtype=np.float64,
        )
        return pred_r, pred_cr

    pred_va_med_r, pred_va_med_cr = _predict_group_medians(df_va)
    pred_te_med_r, pred_te_med_cr = _predict_group_medians(df_te)

    # ExpectedReturn Bondora correlation on test rows where non-null
    er_te_series = pd.to_numeric(df_te["ExpectedReturn"], errors="coerce")
    mask_er_te = er_te_series.notna()
    er_spearman_test = float(
        stats.spearmanr(y_te[mask_er_te.to_numpy()], er_te_series.loc[mask_er_te].to_numpy()).statistic
    )

    n_iter_reg = 4 if quick else 15
    n_est_reg = 500 if quick else 1500

    print("[Step 8.4] Training Ridge Regressor on Set A...")
    ridge_res = train_ridge_regressor(df_tr[set_a_cols], y_tr, df_va[set_a_cols], y_va)
    pred_va_ridge = np.clip(ridge_res["model"].predict(ridge_res["preprocessor"].transform(df_va[set_a_cols])), 0.0, None)
    pred_te_ridge = np.clip(ridge_res["model"].predict(ridge_res["preprocessor"].transform(df_te[set_a_cols])), 0.0, None)

    print(f"[Step 8.4] Training LightGBM Regressor (L2 loss) on Set A (n_iter={n_iter_reg})...")
    lgbm_l2_res = train_lgbm_regressor_with_tuning(
        df_tr[set_a_cols],
        y_tr,
        df_va[set_a_cols],
        y_va,
        objective="regression",
        n_iter=n_iter_reg,
        n_estimators=n_est_reg,
        random_state=RANDOM_SEED,
    )
    pred_va_l2 = np.clip(lgbm_l2_res["model"].predict(lgbm_l2_res["aligner"].transform(df_va[set_a_cols])), 0.0, None)
    pred_te_l2 = np.clip(lgbm_l2_res["model"].predict(lgbm_l2_res["aligner"].transform(df_te[set_a_cols])), 0.0, None)

    print(f"[Step 8.4] Training LightGBM Regressor (Huber loss) on Set A (n_iter={n_iter_reg})...")
    lgbm_huber_res = train_lgbm_regressor_with_tuning(
        df_tr[set_a_cols],
        y_tr,
        df_va[set_a_cols],
        y_va,
        objective="huber",
        n_iter=n_iter_reg,
        n_estimators=n_est_reg,
        random_state=RANDOM_SEED,
    )
    pred_va_huber = np.clip(lgbm_huber_res["model"].predict(lgbm_huber_res["aligner"].transform(df_va[set_a_cols])), 0.0, None)
    pred_te_huber = np.clip(lgbm_huber_res["model"].predict(lgbm_huber_res["aligner"].transform(df_te[set_a_cols])), 0.0, None)

    # Select primary regression model based on validation MAE (L2 vs Huber)
    if lgbm_huber_res["best_val_mae"] <= lgbm_l2_res["best_val_mae"]:
        primary_reg_res = lgbm_huber_res
        primary_reg_code = "LGBM-Reg-Huber"
        pred_va_primary = pred_va_huber
        pred_te_primary = pred_te_huber
    else:
        primary_reg_res = lgbm_l2_res
        primary_reg_code = "LGBM-Reg-L2"
        pred_va_primary = pred_va_l2
        pred_te_primary = pred_te_l2

    print(
        f"[Step 8.4] Selected primary regression model by validation MAE: {primary_reg_code} "
        f"(val MAE Huber={lgbm_huber_res['best_val_mae']:.4f} vs L2={lgbm_l2_res['best_val_mae']:.4f})"
    )

    reg_metrics_rows = [
        # Validation
        evaluate_regression_predictions(
            y_va, np.full_like(y_va, global_train_med), def_va, "BASE-Med-Global", "Median Global Data Latih", "val"
        ),
        evaluate_regression_predictions(
            y_va, pred_va_med_r, def_va, "BASE-Med-Rating", "Median per Rating (Data Latih)", "val"
        ),
        evaluate_regression_predictions(
            y_va, pred_va_med_cr, def_va, "BASE-Med-CountryRating", "Median per Country x Rating (Data Latih)", "val"
        ),
        evaluate_regression_predictions(
            y_va, pred_va_ridge, def_va, "Ridge-A", "Ridge Regression (Set A)", "val"
        ),
        evaluate_regression_predictions(
            y_va, pred_va_l2, def_va, "LGBM-Reg-L2", "LightGBM Regressor L2 (Set A)", "val"
        ),
        evaluate_regression_predictions(
            y_va, pred_va_huber, def_va, "LGBM-Reg-Huber", "LightGBM Regressor Huber (Set A)", "val"
        ),
        # Test
        evaluate_regression_predictions(
            y_te, np.full_like(y_te, global_train_med), def_te, "BASE-Med-Global", "Median Global Data Latih", "test"
        ),
        evaluate_regression_predictions(
            y_te, pred_te_med_r, def_te, "BASE-Med-Rating", "Median per Rating (Data Latih)", "test"
        ),
        evaluate_regression_predictions(
            y_te, pred_te_med_cr, def_te, "BASE-Med-CountryRating", "Median per Country x Rating (Data Latih)", "test"
        ),
        evaluate_regression_predictions(
            y_te[mask_er_te.to_numpy()],
            er_te_series.loc[mask_er_te].to_numpy(),
            def_te[mask_er_te.to_numpy()],
            "BASE-ExpectedReturn",
            "ExpectedReturn Bondora (Skor Benchmark)",
            "test_irisan_er",
        ),
        evaluate_regression_predictions(
            y_te, pred_te_ridge, def_te, "Ridge-A", "Ridge Regression (Set A)", "test"
        ),
        evaluate_regression_predictions(
            y_te, pred_te_l2, def_te, "LGBM-Reg-L2", "LightGBM Regressor L2 (Set A)", "test"
        ),
        evaluate_regression_predictions(
            y_te, pred_te_huber, def_te, "LGBM-Reg-Huber", "LightGBM Regressor Huber (Set A)", "test"
        ),
    ]
    reg_metrics_df = pd.DataFrame(reg_metrics_rows)
    reg_metrics_df.to_csv(TABLES_DIR / "05_metrics_regression.csv", index=False)

    # Regression decile lift table for primary model on test split
    reg_lift_df = compute_regression_lift_table(y_te, pred_te_primary, default_flag=def_te, n_deciles=10)
    reg_lift_df.to_csv(TABLES_DIR / "05_lift_by_decile.csv", index=False)

    # Step 8.7: Regression SHAP Top 15 on 5,000 random test rows
    rng = np.random.default_rng(RANDOM_SEED)
    take_n = min(5000, len(df_te))
    s_idx = np.sort(rng.choice(len(df_te), size=take_n, replace=False))
    X_te_reg_al = primary_reg_res["aligner"].transform(df_te[set_a_cols])
    X_te_sample = X_te_reg_al.iloc[s_idx]

    reg_shap_note = "shap.TreeExplainer"
    try:
        import shap

        expl_reg = shap.TreeExplainer(primary_reg_res["model"])
        sv_reg = np.asarray(expl_reg.shap_values(X_te_sample), dtype=np.float64)
    except Exception as exc:
        reg_shap_note = f"LightGBM pred_contrib=True (fallback: {exc.__class__.__name__})"
        sv_reg = np.asarray(primary_reg_res["model"].predict(X_te_sample, pred_contrib=True), dtype=np.float64)[:, :-1]

    mean_abs_reg_shap = np.mean(np.abs(sv_reg), axis=0)
    mean_signed_reg_shap = np.mean(sv_reg, axis=0)
    reg_shap_df = (
        pd.DataFrame(
            {
                "fitur": set_a_cols,
                "mean_abs_shap": np.round(mean_abs_reg_shap, 6),
                "mean_signed_shap": np.round(mean_signed_reg_shap, 6),
                "n_sampel_uji": take_n,
                "model_regresi": primary_reg_code,
                "metode": reg_shap_note,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    reg_shap_df["peringkat_shap"] = np.arange(1, len(reg_shap_df) + 1)
    reg_shap_df = reg_shap_df[
        ["peringkat_shap", "fitur", "mean_abs_shap", "mean_signed_shap", "n_sampel_uji", "model_regresi", "metode"]
    ]
    reg_shap_df.head(15).to_csv(TABLES_DIR / "05_shap_top15.csv", index=False)

    # Step 9b: Save predictions_return_ratio.parquet for all 113,560 resolved rows
    print("[Step 9b] Generating data/processed/predictions_return_ratio.parquet...")
    X_all_reg = primary_reg_res["aligner"].transform(df_reg[set_a_cols])
    pred_all_reg = np.clip(primary_reg_res["model"].predict(X_all_reg), 0.0, None)
    _, pred_all_med_cr = _predict_group_medians(df_reg)

    pred_reg_df = pd.DataFrame(
        {
            "LoanId": df_reg["LoanId"].astype(str),
            "split": df_reg["split"].astype(str),
            "LoanDate": df_reg["LoanDate"],
            "loan_year": df_reg["loan_year"].astype("int32"),
            "Country": df_reg["Country"].astype(str),
            "Rating": df_reg["Rating"].astype("object").where(df_reg["Rating"].notna(), None),
            "Amount": pd.to_numeric(df_reg["Amount"], errors="coerce").astype("float64"),
            "AppliedAmount": pd.to_numeric(df_reg["AppliedAmount"], errors="coerce").astype("float64"),
            "Interest": pd.to_numeric(df_reg["Interest"], errors="coerce").astype("float64"),
            "LoanDuration": pd.to_numeric(df_reg["LoanDuration"], errors="coerce").astype("int32"),
            "default": df_reg[DEFAULT_TARGET_COL].astype("int8"),
            "return_ratio": pd.to_numeric(df_reg[REGRESSION_TARGET_COL], errors="coerce").astype("float64"),
            "pred_return_ratio": pred_all_reg.astype("float64"),
            "pred_baseline_group_median": pred_all_med_cr.astype("float64"),
        },
        index=df_reg.index,
    )
    assert pred_reg_df["LoanId"].is_unique, "LoanId must be unique in predictions_return_ratio.parquet"
    assert pred_reg_df[["pred_return_ratio", "pred_baseline_group_median"]].isna().sum().sum() == 0
    pred_reg_df.to_parquet(DATA_PROCESSED / "predictions_return_ratio.parquet", index=False)

    # Save regression models to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_code": primary_reg_code,
            "objective": primary_reg_res["objective"],
            "feature_set": "A",
            "features": set_a_cols,
            "aligner": primary_reg_res["aligner"],
            "model": primary_reg_res["model"],
            "best_params": primary_reg_res["best_params"],
            "best_iteration": primary_reg_res["best_iteration"],
            "val_mae_l2": lgbm_l2_res["best_val_mae"],
            "val_mae_huber": lgbm_huber_res["best_val_mae"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "lgbm_return_ratio_setA.joblib",
    )
    joblib.dump(
        {
            "model_code": "Ridge-A",
            "feature_set": "A",
            "features": set_a_cols,
            "preprocessor": ridge_res["preprocessor"],
            "model": ridge_res["model"],
            "best_alpha": ridge_res["best_alpha"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "ridge_return_ratio_setA.joblib",
    )

    # Plot Regression Figures (05_*.png)
    _plot_regression_figures(
        df_te=df_te,
        y_te=y_te,
        def_te=def_te,
        pred_te_primary=pred_te_primary,
        primary_reg_code=primary_reg_code,
        reg_lift_df=reg_lift_df,
        reg_shap_df=reg_shap_df,
    )

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": round(elapsed, 2),
        "primary_regression_model": primary_reg_code,
        "ridge_best_alpha": ridge_res["best_alpha"],
        "val_mae_ridge": round(ridge_res["best_val_mae"], 6),
        "val_mae_lgbm_l2": round(lgbm_l2_res["best_val_mae"], 6),
        "val_mae_lgbm_huber": round(lgbm_huber_res["best_val_mae"], 6),
        "lgbm_l2_best_params": lgbm_l2_res["best_params"],
        "lgbm_l2_best_iteration": lgbm_l2_res["best_iteration"],
        "lgbm_huber_best_params": lgbm_huber_res["best_params"],
        "lgbm_huber_best_iteration": lgbm_huber_res["best_iteration"],
        "expected_return_spearman_test": round(er_spearman_test, 6),
        "global_train_median_return_ratio": round(global_train_med, 6),
    }


def _plot_regression_figures(
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    def_te: np.ndarray,
    pred_te_primary: np.ndarray,
    primary_reg_code: str,
    reg_lift_df: pd.DataFrame,
    reg_shap_df: pd.DataFrame,
) -> None:
    """Create and save all 05_*.png visual artifacts."""
    # 1. 05_pred_vs_actual_and_lift.png
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    ax0, ax1 = axes[0], axes[1]

    rng = np.random.default_rng(RANDOM_SEED)
    idx_s = rng.choice(len(y_te), size=min(3000, len(y_te)), replace=False)
    y_s = y_te[idx_s]
    p_s = pred_te_primary[idx_s]
    d_s = def_te[idx_s]

    m_non = d_s == 0
    m_def = d_s == 1
    ax0.scatter(p_s[m_non], y_s[m_non], alpha=0.28, s=14, color="#2ca02c", label="Lunas tanpa Default (default=0)")
    ax0.scatter(p_s[m_def], y_s[m_def], alpha=0.28, s=14, color="#c44e52", label="Default (default=1)")
    ax0.plot([0, 1.8], [0, 1.8], color="black", linestyle="--", linewidth=1.3, label="Garis Ideal (Prediksi = Aktual)")
    ax0.set_xlim(0.0, 1.55)
    ax0.set_ylim(-0.05, 2.2)
    ax0.set_title(f"Prediksi vs Aktual return_ratio pada Sampel Uji (n = 3.000; {primary_reg_code})")
    ax0.set_xlabel("Prediksi return_ratio")
    ax0.set_ylabel("Aktual return_ratio (is_resolved == True)")
    ax0.legend(loc="upper left", fontsize=8.5)

    x_d = reg_lift_df["desil"].to_numpy()
    act_m = reg_lift_df["mean_actual_return_ratio"].to_numpy()
    pred_m = reg_lift_df["mean_predicted_return_ratio"].to_numpy()
    w = 0.38
    b_act = ax1.bar(x_d - w / 2, act_m, width=w, color="#2ca02c", label="Rata-rata Aktual return_ratio")
    ax1.bar(x_d + w / 2, pred_m, width=w, color="#1f77b4", label="Rata-rata Prediksi return_ratio")
    ax1.axhline(float(y_te.mean()), color="black", linestyle="--", linewidth=1.2, label=f"Rata-rata Aktual Uji ({fmt_id(y_te.mean(), 4)})")
    for bar, v in zip(b_act, act_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, fmt_id(v, 2), ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x_d)
    ax1.set_xticklabels([f"D{d}" for d in x_d])
    ax1.set_ylim(0, max(act_m.max(), pred_m.max()) * 1.18)
    ax1.set_title(f"Lift per Desil Prediksi return_ratio pada Data Uji (D1 = Tertinggi)")
    ax1.set_xlabel(f"Desil Prediksi {primary_reg_code} (Data Uji Resolved, n = 34.675)")
    ax1.set_ylabel("Rata-rata return_ratio")
    ax1.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    save_fig(fig, "05_pred_vs_actual_and_lift.png", close=True)

    # 2. 05_residuals_by_country_and_rating.png
    residuals = y_te - pred_te_primary
    res_df = pd.DataFrame(
        {
            "Country": df_te["Country"].astype(str).to_numpy(),
            "Rating": df_te["Rating"].astype("object").where(df_te["Rating"].notna(), "(Missing)").astype(str).to_numpy(),
            "residual": residuals,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    countries = [c for c in ["EE", "FI", "ES", "SK"] if c in res_df["Country"].unique()]
    data_c = [res_df.loc[res_df["Country"] == c, "residual"].to_numpy() for c in countries]
    axes[0].boxplot(
        data_c,
        tick_labels=countries,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#8ebad9", "edgecolor": "black"},
        medianprops={"color": "#c44e52", "linewidth": 2.0},
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Distribusi Residual (Aktual - Prediksi) menurut Country (Uji, {primary_reg_code})")
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Residual return_ratio (Tanpa Pencilan Ekstrem)")

    ratings = [r for r in ["AA", "A", "B", "C", "D", "E", "F", "HR"] if r in res_df["Rating"].unique()]
    data_r = [res_df.loc[res_df["Rating"] == r, "residual"].to_numpy() for r in ratings]
    axes[1].boxplot(
        data_r,
        tick_labels=ratings,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#a1d99b", "edgecolor": "black"},
        medianprops={"color": "#c44e52", "linewidth": 2.0},
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title(f"Distribusi Residual (Aktual - Prediksi) menurut Rating (Uji, {primary_reg_code})")
    axes[1].set_xlabel("Rating Bondora [Kolom Referensi]")
    axes[1].set_ylabel("Residual return_ratio (Tanpa Pencilan Ekstrem)")

    fig.tight_layout()
    save_fig(fig, "05_residuals_by_country_and_rating.png", close=True)

    # 3. 05_shap_top15_regression.png
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    top_r = reg_shap_df.head(15).iloc[::-1]
    ax.barh(top_r["fitur"], top_r["mean_abs_shap"], color="#2ca02c", edgecolor="black", alpha=0.88)
    for idx_b, m_v in enumerate(top_r["mean_abs_shap"]):
        ax.text(m_v + top_r["mean_abs_shap"].max() * 0.015, idx_b, fmt_id(m_v, 4), va="center", fontsize=8.5)
    ax.set_xlim(0, top_r["mean_abs_shap"].max() * 1.18)
    ax.set_title(f"15 Fitur Terpenting Regresi return_ratio menurut Mean |SHAP| ({primary_reg_code}, n = 5.000)")
    ax.set_xlabel("Rata-rata |Kontribusi SHAP| pada Prediksi return_ratio")
    fig.tight_layout()
    save_fig(fig, "05_shap_top15_regression.png", close=True)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Entry point for ``python -m src.models.train``."""
    parser = argparse.ArgumentParser(description="Train Tahap 4 predictive models (Bondora).")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "all"],
        default="all",
        help="Which modeling task to run (default: all).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with fewer random search iterations for fast verification.",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    ensure_dirs()

    print("=== Tahap 4 Predictive Analytics Pipeline (`src.models.train`) ===")
    df = pd.read_parquet(DATA_PROCESSED / "bondora_clean.parquet")
    with open(DATA_PROCESSED / "schema.json", "r", encoding="utf-8") as f:
        schema_meta = json.load(f)

    preflight_info = run_preflight_checks(df, schema_meta)
    print(f"[Preflight OK] {preflight_info}")

    cls_meta: dict[str, Any] = {}
    reg_meta: dict[str, Any] = {}

    if args.task in ("classification", "all"):
        cls_meta = run_classification_pipeline(df, quick=args.quick)

    if args.task in ("regression", "all"):
        if not cls_meta:
            df_cls_tmp, _ = make_classification_splits(df)
            fsets_tmp, _ = build_feature_sets(
                df_cls_tmp.loc[df_cls_tmp["split"] == "train"],
                df_cls_tmp.loc[df_cls_tmp["split"] == "val"],
            )
            set_a_cols = fsets_tmp["A"]
        else:
            fsets_tbl = pd.read_csv(TABLES_DIR / "04_feature_sets.csv")
            set_a_cols = fsets_tbl.loc[fsets_tbl["di_set_A"] == True, "fitur"].tolist()  # noqa: E712

        reg_meta = run_regression_pipeline(df, set_a_cols=set_a_cols, quick=args.quick)

    if args.task in ("classification", "all"):
        # Evaluation-only follow-up (within-country baselines, level bias, NaN-signal check); does not
        # retrain or overwrite the primary model. Can also be run standalone: python -m src.models.followup
        from src.models.followup import run_followup

        run_followup(df)

    total_elapsed = round(time.perf_counter() - t_start, 2)
    meta_out = {
        "random_seed": RANDOM_SEED,
        "split_train_end": SPLIT_TRAIN_END,
        "split_val_end": SPLIT_VAL_END,
        "quick_mode": bool(args.quick),
        "task": args.task,
        "total_elapsed_seconds": total_elapsed,
        "preflight": preflight_info,
        "classification": cls_meta,
        "regression": reg_meta,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print(f"=== Pipeline finished in {total_elapsed:.2f}s ===")
    return meta_out


if __name__ == "__main__":
    main()
