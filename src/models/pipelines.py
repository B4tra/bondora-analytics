"""Leakage-free preprocessing pipelines, model training, hyperparameter tuning, and calibration (Tahap 4)."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_SEED

warnings.filterwarnings("ignore", message=".*eval_set.*")


class LeakageFreeLinearPreprocessor(BaseEstimator, TransformerMixin):
    """Fit all preprocessing statistics strictly on training data for linear models (LR / Ridge).

    Steps learned from ``X_train`` only:
    1. Numeric columns:
       - Identify columns with any ``NaN`` in ``X_train`` -> create binary ``{col}_isnan`` indicators.
       - Compute training medians ``medians_`` for imputation.
       - Identify non-negative numeric columns (``min >= 0`` in ``X_train``) with training
         skewness ``> 2.0`` (``log1p_cols_``) and apply ``np.log1p(np.maximum(x, 0.0))``.
       - Compute 1st and 99th percentiles (``p01_``, ``p99_``) on ``X_train`` for winsorizing.
       - Compute training ``mean_`` and ``std_`` (replaced by 1.0 if 0.0) for standardization.
    2. Categorical / boolean columns:
       - Convert ``NaN`` to ``"(Missing)"`` and preserve ``"-1"`` as its own category.
       - Fit ``OneHotEncoder(min_frequency=200, handle_unknown='infrequent_if_exist', sparse_output=False)``
         strictly on ``X_train``.
    """

    def __init__(self, min_frequency: int = 200, skew_threshold: float = 2.0) -> None:
        self.min_frequency = min_frequency
        self.skew_threshold = skew_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LeakageFreeLinearPreprocessor":
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []

        for col in X.columns:
            s = X[col]
            if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                self.cat_cols_.append(col)
            else:
                self.num_cols_.append(col)

        self.has_nan_cols_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.log1p_cols_: set[str] = set()
        self.p01_: dict[str, float] = {}
        self.p99_: dict[str, float] = {}
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}

        for col in self.num_cols_:
            s_num = pd.to_numeric(X[col], errors="coerce").astype("float64")
            if bool(s_num.isna().any()):
                self.has_nan_cols_.append(col)

            med = float(s_num.median()) if s_num.notna().any() else 0.0
            self.medians_[col] = med
            s_imp = s_num.fillna(med)

            # Check non-negative and skewness > skew_threshold on training data
            min_v = float(s_imp.min()) if len(s_imp) > 0 else 0.0
            skew_v = float(stats.skew(s_imp.to_numpy(), nan_policy="omit")) if len(s_imp) > 2 else 0.0
            if min_v >= 0.0 and np.isfinite(skew_v) and skew_v > self.skew_threshold:
                self.log1p_cols_.add(col)
                s_proc = np.log1p(np.maximum(s_imp.to_numpy(), 0.0))
            else:
                s_proc = s_imp.to_numpy()

            q01, q99 = np.quantile(s_proc, [0.01, 0.99]) if len(s_proc) > 0 else (0.0, 0.0)
            self.p01_[col] = float(q01)
            self.p99_[col] = float(q99)

            s_win = np.clip(s_proc, q01, q99)
            mu = float(np.mean(s_win))
            sd = float(np.std(s_win, ddof=0))
            if sd < 1e-12:
                sd = 1.0
            self.means_[col] = mu
            self.stds_[col] = sd

        if self.cat_cols_:
            X_cat_clean = self._prepare_cats(X[self.cat_cols_])
            self.ohe_ = OneHotEncoder(
                min_frequency=self.min_frequency,
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                dtype=np.float64,
            )
            self.ohe_.fit(X_cat_clean)
            ohe_names = list(self.ohe_.get_feature_names_out(self.cat_cols_))
        else:
            self.ohe_ = None
            ohe_names = []

        nan_ind_names = [f"{col}_isnan" for col in self.has_nan_cols_]
        self.feature_names_out_: list[str] = list(self.num_cols_) + nan_ind_names + ohe_names
        return self

    @staticmethod
    def _prepare_cats(df_cat: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in df_cat.columns:
            s = df_cat[col]
            s_str = s.astype("object").where(s.notna(), "(Missing)").astype(str).str.strip()
            s_str = s_str.replace({"-1.0": "-1"})
            out[col] = s_str
        return pd.DataFrame(out, index=df_cat.index)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        n_rows = len(X)
        num_arrays: list[np.ndarray] = []

        for col in self.num_cols_:
            s_num = pd.to_numeric(X[col], errors="coerce").astype("float64")
            s_imp = s_num.fillna(self.medians_[col]).to_numpy()
            if col in self.log1p_cols_:
                s_proc = np.log1p(np.maximum(s_imp, 0.0))
            else:
                s_proc = s_imp
            s_win = np.clip(s_proc, self.p01_[col], self.p99_[col])
            s_std = (s_win - self.means_[col]) / self.stds_[col]
            num_arrays.append(s_std.reshape(n_rows, 1))

        for col in self.has_nan_cols_:
            s_num = pd.to_numeric(X[col], errors="coerce")
            ind = s_num.isna().to_numpy(dtype=np.float64).reshape(n_rows, 1)
            num_arrays.append(ind)

        if self.cat_cols_ and self.ohe_ is not None:
            X_cat_clean = self._prepare_cats(X[self.cat_cols_])
            cat_mat = self.ohe_.transform(X_cat_clean)
            num_arrays.append(cat_mat)

        if not num_arrays:
            return np.empty((n_rows, 0), dtype=np.float64)
        return np.hstack(num_arrays)

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out_)


class CategoryAligner(BaseEstimator, TransformerMixin):
    """Align categorical columns for LightGBM using category lists learned strictly from ``X_train``.

    Categories not seen in ``X_train`` automatically become ``NaN`` in validation/test splits.
    Numeric columns are converted to ``float64`` with ``NaN`` preserved.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoryAligner":
        self.cat_categories_: dict[str, list[str]] = {}
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.feature_Order_: list[str] = list(X.columns)

        for col in X.columns:
            s = X[col]
            if (
                isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
            ):
                self.cat_cols_.append(col)
                s_clean = s.astype("object").where(s.notna(), np.nan)
                valid_vals = sorted({str(v).strip() for v in s_clean.dropna().unique()})
                self.cat_categories_[col] = valid_vals
            else:
                self.num_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_cols: dict[str, pd.Series] = {}
        for col in self.feature_Order_:
            s = X[col]
            if col in self.cat_categories_:
                cats = self.cat_categories_[col]
                s_str = s.astype("object").where(s.notna(), np.nan)
                s_str = s_str.map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
                s_str = s_str.where(s_str.isin(cats), np.nan)
                out_cols[col] = pd.Categorical(s_str, categories=cats, ordered=False)
            elif pd.api.types.is_bool_dtype(s):
                out_cols[col] = s.astype("float64")
            else:
                out_cols[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        return pd.DataFrame(out_cols, index=X.index)


class ValidationProbabilityCalibrator:
    """Calibrate raw model probabilities on validation data comparing Platt (sigmoid) vs Isotonic via 5-fold CV."""

    def __init__(self, random_state: int = RANDOM_SEED, n_splits: int = 5) -> None:
        self.random_state = random_state
        self.n_splits = n_splits
        self.selected_method_: str = "isotonic"
        self.cv_results_: dict[str, float] = {}
        self.platt_lr_: LogisticRegression | None = None
        self.iso_reg_: IsotonicRegression | None = None

    @staticmethod
    def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
        return np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)

    def fit(self, p_val_raw: np.ndarray, y_val: np.ndarray) -> "ValidationProbabilityCalibrator":
        p_arr = np.asarray(p_val_raw, dtype=np.float64)
        y_arr = np.asarray(y_val, dtype=np.int32)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_platt = np.zeros_like(p_arr)
        oof_iso = np.zeros_like(p_arr)

        logits_all = self._to_logit(p_arr)

        for tr_idx, te_idx in skf.split(logits_all, y_arr):
            # Platt (sigmoid on logit)
            lr_cal = LogisticRegression(C=1e5, solver="lbfgs", random_state=self.random_state)
            lr_cal.fit(logits_all[tr_idx], y_arr[tr_idx])
            oof_platt[te_idx] = lr_cal.predict_proba(logits_all[te_idx])[:, 1]

            # Isotonic
            iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso_cal.fit(p_arr[tr_idx], y_arr[tr_idx])
            oof_iso[te_idx] = iso_cal.transform(p_arr[te_idx])

        brier_raw = float(brier_score_loss(y_arr, p_arr))
        brier_platt = float(brier_score_loss(y_arr, oof_platt))
        brier_iso = float(brier_score_loss(y_arr, oof_iso))

        self.cv_results_ = {
            "brier_val_raw": brier_raw,
            "brier_val_cv5_sigmoid": brier_platt,
            "brier_val_cv5_isotonic": brier_iso,
        }

        # Fit both on full validation data and select winner by 5-fold CV Brier score
        self.platt_lr_ = LogisticRegression(C=1e5, solver="lbfgs", random_state=self.random_state)
        self.platt_lr_.fit(logits_all, y_arr)

        self.iso_reg_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso_reg_.fit(p_arr, y_arr)

        if brier_platt <= brier_iso:
            self.selected_method_ = "sigmoid"
        else:
            self.selected_method_ = "isotonic"

        return self

    def transform(self, p_raw: np.ndarray, method: str | None = None) -> np.ndarray:
        use_method = method or self.selected_method_
        p_arr = np.asarray(p_raw, dtype=np.float64)
        if use_method == "sigmoid":
            assert self.platt_lr_ is not None
            return np.clip(self.platt_lr_.predict_proba(self._to_logit(p_arr))[:, 1], 0.0, 1.0)
        if use_method == "isotonic":
            assert self.iso_reg_ is not None
            return np.clip(self.iso_reg_.transform(p_arr), 0.0, 1.0)
        raise ValueError(f"Unknown calibration method: {use_method}")


def train_logistic_regression_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fit ``LeakageFreeLinearPreprocessor`` on ``X_train`` and tune L2 ``LogisticRegression`` on ``X_val``."""
    prep = LeakageFreeLinearPreprocessor(min_frequency=200, skew_threshold=2.0)
    X_tr_mat = prep.fit_transform(X_train)
    X_va_mat = prep.transform(X_val)

    y_tr = np.asarray(y_train, dtype=np.int32)
    y_va = np.asarray(y_val, dtype=np.int32)

    best_auc = -1.0
    best_c = c_grid[0]
    best_model: LogisticRegression | None = None
    grid_records: list[dict[str, float]] = []

    for c_val in c_grid:
        clf = LogisticRegression(
            C=c_val,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(X_tr_mat, y_tr)
        p_va = clf.predict_proba(X_va_mat)[:, 1]
        auc_va = float(roc_auc_score(y_va, p_va))
        brier_va = float(brier_score_loss(y_va, p_va))
        grid_records.append({"C": c_val, "val_roc_auc": round(auc_va, 6), "val_brier": round(brier_va, 6)})
        if auc_va > best_auc:
            best_auc = auc_va
            best_c = c_val
            best_model = clf

    assert best_model is not None
    return {
        "preprocessor": prep,
        "model": best_model,
        "best_C": best_c,
        "best_val_auc": best_auc,
        "grid_search": grid_records,
    }


def train_lgbm_classifier_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    n_iter: int = 20,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_SEED,
    check_unbalance: bool = False,
) -> dict[str, Any]:
    """Tune and fit ``LGBMClassifier`` using randomized search on validation ROC-AUC with early stopping."""
    import lightgbm as lgb

    aligner = CategoryAligner()
    X_tr_al = aligner.fit_transform(X_train)
    X_va_al = aligner.transform(X_val)

    y_tr = np.asarray(y_train, dtype=np.int32)
    y_va = np.asarray(y_val, dtype=np.int32)

    param_dist = {
        "num_leaves": [15, 31, 63, 95, 127],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "min_child_samples": [50, 100, 200, 350, 500],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 5.0, 10.0, 25.0],
    }

    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    best_auc = -1.0
    best_params: dict[str, Any] = {}
    best_model: Any = None
    best_iteration: int = 0
    search_history: list[dict[str, Any]] = []

    for idx_s, params in enumerate(sampler, start=1):
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            subsample_freq=1,
            random_state=random_state + idx_s,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        clf.fit(
            X_tr_al,
            y_tr,
            eval_set=[(X_va_al, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        p_va = clf.predict_proba(X_va_al)[:, 1]
        auc_va = float(roc_auc_score(y_va, p_va))
        b_iter = int(getattr(clf, "best_iteration_", n_estimators) or n_estimators)
        rec = {
            "iter": idx_s,
            **params,
            "best_iteration": b_iter,
            "val_roc_auc": round(auc_va, 6),
        }
        search_history.append(rec)

        if auc_va > best_auc:
            best_auc = auc_va
            best_params = dict(params)
            best_model = clf
            best_iteration = b_iter

    unbalance_comparison: dict[str, Any] | None = None
    if check_unbalance and best_params:
        clf_unb = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            subsample_freq=1,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **best_params,
        )
        clf_unb.fit(
            X_tr_al,
            y_tr,
            eval_set=[(X_va_al, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        p_va_unb = clf_unb.predict_proba(X_va_al)[:, 1]
        unbalance_comparison = {
            "val_auc_standard": round(best_auc, 6),
            "val_auc_is_unbalance_true": round(float(roc_auc_score(y_va, p_va_unb)), 6),
            "val_brier_standard": round(float(brier_score_loss(y_va, best_model.predict_proba(X_va_al)[:, 1])), 6),
            "val_brier_is_unbalance_true": round(float(brier_score_loss(y_va, p_va_unb)), 6),
        }

    return {
        "aligner": aligner,
        "model": best_model,
        "best_params": best_params,
        "best_iteration": best_iteration,
        "best_val_auc": best_auc,
        "search_history": search_history,
        "unbalance_comparison": unbalance_comparison,
    }


def train_ridge_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fit ``LeakageFreeLinearPreprocessor`` on ``X_train`` and tune ``Ridge`` regression on validation MAE."""
    prep = LeakageFreeLinearPreprocessor(min_frequency=200, skew_threshold=2.0)
    X_tr_mat = prep.fit_transform(X_train)
    X_va_mat = prep.transform(X_val)

    y_tr = np.asarray(y_train, dtype=np.float64)
    y_va = np.asarray(y_val, dtype=np.float64)

    best_mae = float("inf")
    best_alpha = alpha_grid[0]
    best_model: Ridge | None = None
    grid_records: list[dict[str, float]] = []

    for alpha in alpha_grid:
        reg = Ridge(alpha=alpha, random_state=random_state)
        reg.fit(X_tr_mat, y_tr)
        pred_va = reg.predict(X_va_mat)
        mae_va = float(mean_absolute_error(y_va, pred_va))
        grid_records.append({"alpha": alpha, "val_mae": round(mae_va, 6)})
        if mae_va < best_mae:
            best_mae = mae_va
            best_alpha = alpha
            best_model = reg

    assert best_model is not None
    return {
        "preprocessor": prep,
        "model": best_model,
        "best_alpha": best_alpha,
        "best_val_mae": best_mae,
        "grid_search": grid_records,
    }


def train_lgbm_regressor_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    objective: str = "regression",
    n_iter: int = 15,
    n_estimators: int = 1500,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Tune and fit ``LGBMRegressor`` (``objective='regression'`` L2 or ``'huber'``) on validation MAE."""
    import lightgbm as lgb

    aligner = CategoryAligner()
    X_tr_al = aligner.fit_transform(X_train)
    X_va_al = aligner.transform(X_val)

    y_tr = np.asarray(y_train, dtype=np.float64)
    y_va = np.asarray(y_val, dtype=np.float64)

    param_dist = {
        "num_leaves": [15, 31, 63, 95],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "min_child_samples": [50, 100, 200, 400],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 15.0],
    }

    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    best_mae = float("inf")
    best_params: dict[str, Any] = {}
    best_model: Any = None
    best_iteration: int = 0
    search_history: list[dict[str, Any]] = []

    for idx_s, params in enumerate(sampler, start=1):
        reg = lgb.LGBMRegressor(
            objective=objective,
            n_estimators=n_estimators,
            subsample_freq=1,
            random_state=random_state + idx_s,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        reg.fit(
            X_tr_al,
            y_tr,
            eval_set=[(X_va_al, y_va)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        pred_va = reg.predict(X_va_al)
        mae_va = float(mean_absolute_error(y_va, pred_va))
        b_iter = int(getattr(reg, "best_iteration_", n_estimators) or n_estimators)
        rec = {
            "iter": idx_s,
            "objective": objective,
            **params,
            "best_iteration": b_iter,
            "val_mae": round(mae_va, 6),
        }
        search_history.append(rec)

        if mae_va < best_mae:
            best_mae = mae_va
            best_params = dict(params)
            best_model = reg
            best_iteration = b_iter

    return {
        "aligner": aligner,
        "model": best_model,
        "objective": objective,
        "best_params": best_params,
        "best_iteration": best_iteration,
        "best_val_mae": best_mae,
        "search_history": search_history,
    }
