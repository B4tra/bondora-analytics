"""Evaluation metrics, calibration diagnostics, decile lift tables, and subgroup checks for Tahap 4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import RANDOM_SEED


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic ``max |TPR - FPR|`` for binary classification."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_arr)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_arr, p_arr)
    return float(np.max(np.abs(tpr - fpr)))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Compute Expected Calibration Error (ECE) over ``n_bins`` uniform probability bins in ``[0, 1]``.

    Returns
    -------
    tuple[float, pd.DataFrame]
        ``(ece_value, calibration_bins_df)``
    """
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n_tot = len(y_arr)
    if n_tot == 0:
        return np.nan, pd.DataFrame()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign bin indices in 0 .. n_bins-1
    bin_idx = np.digitize(p_arr, edges[1:-1], right=False)

    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b > 0:
            mean_p = float(p_arr[mask].mean())
            mean_y = float(y_arr[mask].mean())
            abs_gap = abs(mean_y - mean_p)
            ece += (n_b / n_tot) * abs_gap
        else:
            mean_p = np.nan
            mean_y = np.nan
            abs_gap = np.nan
        rows.append(
            {
                "bin": b + 1,
                "bin_low": round(float(edges[b]), 2),
                "bin_high": round(float(edges[b + 1]), 2),
                "n": n_b,
                "mean_predicted_prob": mean_p,
                "observed_default_rate": mean_y,
                "abs_gap": abs_gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def find_youden_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Find the probability threshold on validation data maximizing Youden's J = TPR - FPR."""
    y_arr = np.asarray(y_val, dtype=np.int32)
    p_arr = np.asarray(p_val, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    thr = float(thresholds[best_idx])
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr, 1e-6, 1.0 - 1e-6))


def top_k_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_fracs: tuple[float, ...] = (0.10, 0.20, 0.30),
) -> dict[str, float]:
    """Compute precision and recall when selecting the top ``k%`` highest-risk loans."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    s_arr = np.asarray(y_score, dtype=np.float64)
    n_tot = len(y_arr)
    n_pos_tot = int(y_arr.sum())

    # Stable descending sort by score
    order = np.argsort(-s_arr, kind="mergesort")
    y_sorted = y_arr[order]

    out: dict[str, float] = {}
    for frac in top_fracs:
        pct_int = int(round(frac * 100))
        k = max(1, int(round(n_tot * frac)))
        tp_k = int(y_sorted[:k].sum())
        prec_k = float(tp_k / k)
        rec_k = float(tp_k / n_pos_tot) if n_pos_tot > 0 else np.nan
        out[f"precision_top{pct_int}pct"] = prec_k
        out[f"recall_top{pct_int}pct"] = rec_k
    return out


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Compute point ROC-AUC and 95% bootstrap confidence interval ``(auc, ci_low, ci_high)``."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    s_arr = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_arr)) < 2:
        return np.nan, np.nan, np.nan

    point_auc = float(roc_auc_score(y_arr, s_arr))
    if n_bootstrap <= 0:
        return point_auc, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    n = len(y_arr)
    boot_aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        if 0 < yb.sum() < n:
            boot_aucs.append(float(roc_auc_score(yb, s_arr[idx])))

    if not boot_aucs:
        return point_auc, np.nan, np.nan
    ci_low, ci_high = np.quantile(boot_aucs, [0.025, 0.975])
    return point_auc, float(ci_low), float(ci_high)


def paired_bootstrap_auc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Compute paired bootstrap difference in ROC-AUC ``AUC(score_a) - AUC(score_b)`` and 95% CI."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    sa = np.asarray(score_a, dtype=np.float64)
    sb = np.asarray(score_b, dtype=np.float64)

    auc_a = float(roc_auc_score(y_arr, sa))
    auc_b = float(roc_auc_score(y_arr, sb))
    diff_point = auc_a - auc_b

    rng = np.random.default_rng(random_state)
    n = len(y_arr)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        if 0 < yb.sum() < n:
            diffs.append(float(roc_auc_score(yb, sa[idx]) - roc_auc_score(yb, sb[idx])))

    ci_l, ci_h = np.quantile(diffs, [0.025, 0.975]) if diffs else (np.nan, np.nan)
    return {
        "auc_a": round(auc_a, 6),
        "auc_b": round(auc_b, 6),
        "diff_auc": round(diff_point, 6),
        "ci_low_95": round(float(ci_l), 6),
        "ci_high_95": round(float(ci_h), 6),
    }


def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_val: float,
    model_code: str,
    model_name: str,
    feature_set: str,
    split_name: str,
    is_probability: bool = True,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compute full suite of classification metrics on a given evaluation split."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    s_arr = np.asarray(y_prob, dtype=np.float64)
    n_eval = int(len(y_arr))
    n_def = int(y_arr.sum())

    # Constant baseline check
    is_constant = bool(np.allclose(s_arr, s_arr[0]))
    if is_constant:
        auc_val, ci_l, ci_h = 0.5, 0.5, 0.5
        gini_val = 0.0
        ks_val = 0.0
    else:
        auc_val, ci_l, ci_h = bootstrap_auc_ci(
            y_arr,
            s_arr,
            n_bootstrap=n_bootstrap if split_name.startswith("test") else 0,
            random_state=random_state,
        )
        gini_val = 2.0 * auc_val - 1.0
        ks_val = compute_ks_statistic(y_arr, s_arr)

    pr_auc_val = float(average_precision_score(y_arr, s_arr))
    topk = top_k_precision_recall(y_arr, s_arr, top_fracs=(0.10, 0.20, 0.30))

    if is_probability:
        p_clip = np.clip(s_arr, 1e-15, 1.0 - 1e-15)
        brier_val = float(brier_score_loss(y_arr, np.clip(s_arr, 0.0, 1.0)))
        logloss_val = float(log_loss(y_arr, p_clip))
        ece_val, _ = compute_ece(y_arr, np.clip(s_arr, 0.0, 1.0), n_bins=10)

        y_pred = (s_arr >= threshold_val).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        prec_y = float(precision_score(y_arr, y_pred, zero_division=0))
        rec_y = float(recall_score(y_arr, y_pred, zero_division=0))
        f1_y = float(f1_score(y_arr, y_pred, zero_division=0))
    else:
        brier_val = np.nan
        logloss_val = np.nan
        ece_val = np.nan
        tn = fp = fn = tp = np.nan
        prec_y = rec_y = f1_y = np.nan

    return {
        "kode_model": model_code,
        "nama_model": model_name,
        "set_fitur": feature_set,
        "split": split_name,
        "n": n_eval,
        "n_default": n_def,
        "default_rate": round(n_def / n_eval, 6),
        "roc_auc": round(auc_val, 6),
        "auc_ci_low_95": round(ci_l, 6) if not np.isnan(ci_l) else np.nan,
        "auc_ci_high_95": round(ci_h, 6) if not np.isnan(ci_h) else np.nan,
        "pr_auc": round(pr_auc_val, 6),
        "gini": round(gini_val, 6),
        "ks": round(ks_val, 6),
        "brier_score": round(brier_val, 6) if not np.isnan(brier_val) else np.nan,
        "log_loss": round(logloss_val, 6) if not np.isnan(logloss_val) else np.nan,
        "ece_10bin": round(ece_val, 6) if not np.isnan(ece_val) else np.nan,
        "threshold_youden_val": round(threshold_val, 6) if is_probability else np.nan,
        "tn": int(tn) if not np.isnan(tn) else np.nan,
        "fp": int(fp) if not np.isnan(fp) else np.nan,
        "fn": int(fn) if not np.isnan(fn) else np.nan,
        "tp": int(tp) if not np.isnan(tp) else np.nan,
        "precision_youden": round(prec_y, 6) if not np.isnan(prec_y) else np.nan,
        "recall_youden": round(rec_y, 6) if not np.isnan(rec_y) else np.nan,
        "f1_youden": round(f1_y, 6) if not np.isnan(f1_y) else np.nan,
        "precision_top10pct": round(topk["precision_top10pct"], 6),
        "recall_top10pct": round(topk["recall_top10pct"], 6),
        "precision_top20pct": round(topk["precision_top20pct"], 6),
        "recall_top20pct": round(topk["recall_top20pct"], 6),
        "precision_top30pct": round(topk["precision_top30pct"], 6),
        "recall_top30pct": round(topk["recall_top30pct"], 6),
    }


def compute_classification_lift_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Build a 10-decile lift and cumulative gain table sorted from highest risk (Decile 1) to lowest (Decile 10)."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    n_tot = len(y_arr)
    tot_def = int(y_arr.sum())
    base_rate = float(tot_def / n_tot)

    order = np.argsort(-p_arr, kind="mergesort")
    y_s = y_arr[order]
    p_s = p_arr[order]

    chunks = np.array_split(np.arange(n_tot), n_deciles)
    rows = []
    cum_n = 0
    cum_def = 0

    for d_idx, idx_chunk in enumerate(chunks, start=1):
        yc = y_s[idx_chunk]
        pc = p_s[idx_chunk]
        n_d = int(len(idx_chunk))
        def_d = int(yc.sum())
        obs_rate = float(def_d / n_d) if n_d > 0 else 0.0
        mean_p = float(pc.mean()) if n_d > 0 else np.nan
        min_p = float(pc.min()) if n_d > 0 else np.nan
        max_p = float(pc.max()) if n_d > 0 else np.nan
        lift_d = obs_rate / base_rate if base_rate > 0 else np.nan

        cum_n += n_d
        cum_def += def_d
        cum_rate = float(cum_def / cum_n)
        cum_lift = cum_rate / base_rate if base_rate > 0 else np.nan
        cum_capture = float(cum_def / tot_def) if tot_def > 0 else np.nan

        rows.append(
            {
                "desil": d_idx,
                "n": n_d,
                "n_default": def_d,
                "observed_default_rate": round(obs_rate, 6),
                "mean_predicted_prob": round(mean_p, 6),
                "min_predicted_prob": round(min_p, 6),
                "max_predicted_prob": round(max_p, 6),
                "lift_desil": round(lift_d, 4),
                "cum_n": cum_n,
                "cum_default": cum_def,
                "cum_default_rate": round(cum_rate, 6),
                "cum_recall_captured": round(cum_capture, 6),
                "cum_lift": round(cum_lift, 4),
            }
        )

    return pd.DataFrame(rows)


def compute_subgroup_performance(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
) -> pd.DataFrame:
    """Compute test-set performance per subgroup: Country, LoanDuration_group, NewCreditCustomer, and half-year."""
    work = df_test[["Country", "LoanDuration", "NewCreditCustomer", "LoanDate"]].copy()
    work["y"] = np.asarray(y_true, dtype=np.int32)
    work["p_raw"] = np.asarray(p_raw, dtype=np.float64)
    work["p_cal"] = np.asarray(p_cal, dtype=np.float64)

    dur_bins = [-np.inf, 12, 24, 36, 59, np.inf]
    dur_labels = ["<= 12 bulan", "13–24 bulan", "25–36 bulan", "37–59 bulan", "60 bulan"]
    work["LoanDuration_group"] = pd.cut(work["LoanDuration"], bins=dur_bins, labels=dur_labels)

    dt = pd.to_datetime(work["LoanDate"])
    work["paruh_waktu_uji"] = np.where(
        dt <= pd.Timestamp("2019-12-31"),
        "Jul–Des 2019",
        "Jan–Jul 2020",
    )

    dims = [
        ("Country", ["EE", "FI", "ES", "SK"]),
        ("LoanDuration_group", dur_labels),
        ("NewCreditCustomer", [True, False]),
        ("paruh_waktu_uji", ["Jul–Des 2019", "Jan–Jul 2020"]),
    ]

    rows: list[dict[str, Any]] = []
    for dim_col, order_vals in dims:
        for val in order_vals:
            sub = work.loc[work[dim_col] == val]
            n_sub = int(len(sub))
            if n_sub == 0:
                continue
            ys = sub["y"].to_numpy()
            pr = sub["p_raw"].to_numpy()
            pc = sub["p_cal"].to_numpy()
            n_def = int(ys.sum())
            obs_r = float(n_def / n_sub)
            mean_pr = float(pr.mean())
            mean_pc = float(pc.mean())

            if 0 < n_def < n_sub:
                auc_v, ci_l, ci_h = bootstrap_auc_ci(ys, pc, n_bootstrap=200, random_state=RANDOM_SEED)
                pr_auc_v = float(average_precision_score(ys, pc))
                ks_v = compute_ks_statistic(ys, pc)
            else:
                auc_v = ci_l = ci_h = pr_auc_v = ks_v = np.nan

            brier_v = float(brier_score_loss(ys, pc))
            rows.append(
                {
                    "dimensi": dim_col,
                    "subgrup": str(val),
                    "n": n_sub,
                    "n_default": n_def,
                    "observed_default_rate": round(obs_r, 6),
                    "mean_pred_prob_raw": round(mean_pr, 6),
                    "mean_pred_prob_calibrated": round(mean_pc, 6),
                    "selisih_pred_cal_minus_obs": round(mean_pc - obs_r, 6),
                    "roc_auc": round(auc_v, 6) if not np.isnan(auc_v) else np.nan,
                    "auc_ci_low_95": round(ci_l, 6) if not np.isnan(ci_l) else np.nan,
                    "auc_ci_high_95": round(ci_h, 6) if not np.isnan(ci_h) else np.nan,
                    "pr_auc": round(pr_auc_v, 6) if not np.isnan(pr_auc_v) else np.nan,
                    "ks": round(ks_v, 6) if not np.isnan(ks_v) else np.nan,
                    "brier_score": round(brier_v, 6),
                }
            )

    return pd.DataFrame(rows)


def evaluate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default_flag: np.ndarray | None,
    model_code: str,
    model_name: str,
    split_name: str,
) -> dict[str, Any]:
    """Compute regression evaluation metrics (MAE, RMSE, R2, Spearman, default/non-default split, decile spread)."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_pred, dtype=np.float64)
    n_eval = int(len(y_arr))

    mae = float(mean_absolute_error(y_arr, p_arr))
    rmse = float(np.sqrt(mean_squared_error(y_arr, p_arr)))
    r2 = float(r2_score(y_arr, p_arr))

    if np.allclose(p_arr, p_arr[0]):
        rho_s, p_val_s = 0.0, 1.0
    else:
        res_sp = stats.spearmanr(y_arr, p_arr, nan_policy="omit")
        rho_s = float(res_sp.statistic)
        p_val_s = float(res_sp.pvalue)

    # Top decile vs bottom decile actual mean return_ratio
    order = np.argsort(-p_arr, kind="mergesort")
    y_sorted = y_arr[order]
    chunks = np.array_split(y_sorted, 10)
    top_dec_actual = float(chunks[0].mean()) if len(chunks[0]) > 0 else np.nan
    bot_dec_actual = float(chunks[-1].mean()) if len(chunks[-1]) > 0 else np.nan

    mae_def = rmse_def = mean_act_def = mean_pred_def = np.nan
    mae_nondef = rmse_nondef = mean_act_nondef = mean_pred_nondef = np.nan

    if default_flag is not None:
        d_arr = np.asarray(default_flag)
        m_def = d_arr == 1
        m_non = d_arr == 0
        if m_def.any():
            mae_def = float(mean_absolute_error(y_arr[m_def], p_arr[m_def]))
            rmse_def = float(np.sqrt(mean_squared_error(y_arr[m_def], p_arr[m_def])))
            mean_act_def = float(y_arr[m_def].mean())
            mean_pred_def = float(p_arr[m_def].mean())
        if m_non.any():
            mae_nondef = float(mean_absolute_error(y_arr[m_non], p_arr[m_non]))
            rmse_nondef = float(np.sqrt(mean_squared_error(y_arr[m_non], p_arr[m_non])))
            mean_act_nondef = float(y_arr[m_non].mean())
            mean_pred_nondef = float(p_arr[m_non].mean())

    return {
        "kode_model": model_code,
        "nama_model": model_name,
        "split": split_name,
        "n": n_eval,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "spearman_rho": round(rho_s, 6),
        "spearman_p_value": p_val_s,
        "mean_actual": round(float(y_arr.mean()), 6),
        "mean_predicted": round(float(p_arr.mean()), 6),
        "mean_actual_decile1_tertinggi": round(top_dec_actual, 6),
        "mean_actual_decile10_terendah": round(bot_dec_actual, 6),
        "selisih_decile1_minus_decile10": round(top_dec_actual - bot_dec_actual, 6),
        "mae_default": round(mae_def, 6) if not np.isnan(mae_def) else np.nan,
        "rmse_default": round(rmse_def, 6) if not np.isnan(rmse_def) else np.nan,
        "mean_actual_default": round(mean_act_def, 6) if not np.isnan(mean_act_def) else np.nan,
        "mean_pred_default": round(mean_pred_def, 6) if not np.isnan(mean_pred_def) else np.nan,
        "mae_non_default": round(mae_nondef, 6) if not np.isnan(mae_nondef) else np.nan,
        "rmse_non_default": round(rmse_nondef, 6) if not np.isnan(rmse_nondef) else np.nan,
        "mean_actual_non_default": round(mean_act_nondef, 6) if not np.isnan(mean_act_nondef) else np.nan,
        "mean_pred_non_default": round(mean_pred_nondef, 6) if not np.isnan(mean_pred_nondef) else np.nan,
    }


def compute_regression_lift_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default_flag: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Build a 10-decile regression lift table ordered from highest predicted return_ratio (Decile 1) to lowest (Decile 10)."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_pred, dtype=np.float64)
    d_arr = np.asarray(default_flag, dtype=np.float64) if default_flag is not None else np.full_like(y_arr, np.nan)

    n_tot = len(y_arr)
    overall_mean = float(y_arr.mean())
    order = np.argsort(-p_arr, kind="mergesort")
    y_s = y_arr[order]
    p_s = p_arr[order]
    d_s = d_arr[order]

    chunks = np.array_split(np.arange(n_tot), n_deciles)
    rows = []
    for d_idx, idx_c in enumerate(chunks, start=1):
        yc = y_s[idx_c]
        pc = p_s[idx_c]
        dc = d_s[idx_c]
        n_d = int(len(idx_c))
        mean_act = float(yc.mean()) if n_d > 0 else np.nan
        med_act = float(np.median(yc)) if n_d > 0 else np.nan
        mean_pr = float(pc.mean()) if n_d > 0 else np.nan
        def_rate = float(np.nanmean(dc)) if n_d > 0 and not np.all(np.isnan(dc)) else np.nan
        lift_ratio = mean_act / overall_mean if overall_mean > 0 else np.nan

        rows.append(
            {
                "desil": d_idx,
                "n": n_d,
                "mean_predicted_return_ratio": round(mean_pr, 6),
                "mean_actual_return_ratio": round(mean_act, 6),
                "median_actual_return_ratio": round(med_act, 6),
                "observed_default_rate_lifetime": round(def_rate, 6) if not np.isnan(def_rate) else np.nan,
                "lift_vs_overall_mean": round(lift_ratio, 4) if not np.isnan(lift_ratio) else np.nan,
            }
        )

    return pd.DataFrame(rows)
