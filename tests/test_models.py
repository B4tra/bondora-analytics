"""Unit tests for Tahap 4 predictive analytics modules (tests/test_models.py)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import (
    compute_classification_lift_table,
    compute_ece,
    compute_ks_statistic,
    evaluate_regression_predictions,
)
from src.models.features import (
    FORBIDDEN_COLUMNS,
    FORBIDDEN_ROLES,
    SET_C_ABLATION_COLS,
    build_feature_sets,
    load_candidate_features_from_schema,
)
from src.models.pipelines import CategoryAligner, LeakageFreeLinearPreprocessor
from src.models.splits import (
    assign_temporal_split,
    make_classification_splits,
    make_random_stratified_split,
    make_regression_splits,
)


def test_time_based_split_boundaries_no_overlap_and_na_exclusion() -> None:
    """Verify temporal splits have exact date cutoffs, zero index overlap, and exclude NA target rows."""
    df_mini = pd.DataFrame(
        {
            "LoanId": [f"L{i}" for i in range(1, 11)],
            "LoanDate": pd.to_datetime(
                [
                    "2017-01-15",
                    "2018-06-30",  # exact upper boundary of train
                    "2018-07-01",  # exact lower boundary of val
                    "2019-06-30",  # exact upper boundary of val
                    "2019-07-01",  # exact lower boundary of test
                    "2020-02-10",
                    "2018-03-10",  # NA default_12m -> must be excluded from classification splits
                    "2019-01-20",
                    "2019-11-05",
                    "2016-05-01",
                ]
            ),
            "default_12m": pd.Series([0, 1, 0, 1, 0, 1, pd.NA, 0, 1, 0], dtype="Int8"),
            "is_resolved": [True, True, True, True, True, False, True, True, True, True],
            "return_ratio": [1.10, 0.40, 1.05, 0.25, 1.15, np.nan, 0.95, 1.02, 0.30, 1.08],
            "default": pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype="Int8"),
        }
    )

    df_cls, summary_cls = make_classification_splits(
        df_mini,
        target_col="default_12m",
        date_col="LoanDate",
        train_end="2018-06-30",
        val_end="2019-06-30",
        min_rows=1,
        min_minority_rate=0.10,
    )

    # Row L7 with NA default_12m must be excluded -> 9 rows left
    assert len(df_cls) == 9
    assert "L7" not in set(df_cls["LoanId"])

    idx_train = set(df_cls.loc[df_cls["split"] == "train", "LoanId"])
    idx_val = set(df_cls.loc[df_cls["split"] == "val", "LoanId"])
    idx_test = set(df_cls.loc[df_cls["split"] == "test", "LoanId"])

    assert idx_train == {"L1", "L2", "L10"}
    assert idx_val == {"L3", "L4", "L8"}
    assert idx_test == {"L5", "L6", "L9"}

    # Disjoint sets
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)

    # Check regression split excludes is_resolved == False (L6)
    df_reg, _ = make_regression_splits(
        df_mini,
        resolved_col="is_resolved",
        target_col="return_ratio",
        default_col="default",
        date_col="LoanDate",
        train_end="2018-06-30",
        val_end="2019-06-30",
        min_test_rows=1,
    )
    assert len(df_reg) == 9
    assert "L6" not in set(df_reg["LoanId"])


def test_feature_sets_no_forbidden_columns_and_era_dead_detection() -> None:
    """Verify Set A/B have no forbidden columns, Set C adds only the 3 benchmark cols, and era-dead rule works."""
    rng = np.random.default_rng(42)
    n_tr, n_va = 100, 100

    # Synthetic candidate features:
    # - feat_stable_num: normal distribution in both train & val -> masuk_set_A == True
    # - feat_stable_cat: balanced categories in both train & val -> masuk_set_A == True
    # - feat_dead_era: diverse in train (50% 'A', 50% 'B'), but 98% '(Missing)' in val -> mati-era!
    # - feat_always_constant: 97% '0' in train AND 98% '0' in val -> NOT newly era-dead (train share > 0.80)
    df_train = pd.DataFrame(
        {
            "feat_stable_num": rng.normal(10.0, 2.0, size=n_tr),
            "feat_stable_cat": ["EE"] * 50 + ["FI"] * 50,
            "feat_dead_era": ["Cat1"] * 50 + ["Cat2"] * 50,
            "feat_always_constant": ["0"] * 97 + ["1"] * 3,
            "Rating": ["C"] * n_tr,
            "ProbabilityOfDefault": rng.uniform(0.1, 0.5, size=n_tr),
            "Interest": rng.uniform(15.0, 35.0, size=n_tr),
        }
    )
    df_val = pd.DataFrame(
        {
            "feat_stable_num": rng.normal(10.5, 2.0, size=n_va),
            "feat_stable_cat": ["EE"] * 45 + ["FI"] * 55,
            "feat_dead_era": [np.nan] * 98 + ["Cat1"] * 2,
            "feat_always_constant": ["0"] * 98 + ["1"] * 2,
            "Rating": ["C"] * n_va,
            "ProbabilityOfDefault": rng.uniform(0.1, 0.5, size=n_va),
            "Interest": rng.uniform(15.0, 35.0, size=n_va),
        }
    )

    cand_cols = ["feat_stable_num", "feat_stable_cat", "feat_dead_era", "feat_always_constant"]
    fsets, ftable = build_feature_sets(
        df_train,
        df_val,
        candidate_cols=cand_cols,
        psi_csv_path=None,
        val_dom_threshold=0.95,
        train_dom_max_threshold=0.80,
    )

    assert "feat_dead_era" not in fsets["A"]
    assert "feat_dead_era" in fsets["B"]
    assert "feat_stable_num" in fsets["A"]
    assert "feat_stable_cat" in fsets["A"]
    assert "feat_always_constant" in fsets["A"]

    # Check Set C equals Set A + (Rating, ProbabilityOfDefault, Interest)
    assert fsets["C"] == fsets["A"] + list(SET_C_ABLATION_COLS)

    # Also verify real schema.json candidate features contain zero forbidden roles/columns
    schema_candidates = load_candidate_features_from_schema()
    assert len(schema_candidates) == 44
    for meta in schema_candidates:
        assert meta["nama_kolom"] not in FORBIDDEN_COLUMNS
        assert meta["peran"] not in FORBIDDEN_ROLES


def test_leakage_free_preprocessor_uses_only_train_statistics() -> None:
    """Verify LeakageFreeLinearPreprocessor and CategoryAligner learn stats exclusively from X_train."""
    X_train = pd.DataFrame(
        {
            "num_a": [10.0, 20.0, 30.0, np.nan, 50.0, 25.0, 15.0, 35.0, 40.0, 22.0],
            "cat_a": ["EE", "EE", "FI", "FI", "ES", "EE", "FI", "ES", "-1", np.nan],
        }
    )
    # Validation has extreme values and unseen category 'ZZ' that would shift median/p99 if leaked
    X_val = pd.DataFrame(
        {
            "num_a": [10_000.0, 20_000.0, np.nan, 50_000.0],
            "cat_a": ["ZZ", "EE", "-1", np.nan],
        }
    )

    prep = LeakageFreeLinearPreprocessor(min_frequency=1, skew_threshold=100.0)
    prep.fit(X_train)

    expected_train_median = float(X_train["num_a"].median())
    expected_train_p99 = float(np.quantile(X_train["num_a"].fillna(expected_train_median), 0.99))

    assert np.isclose(prep.medians_["num_a"], expected_train_median)
    assert np.isclose(prep.p99_["num_a"], expected_train_p99)
    assert prep.p99_["num_a"] <= 50.0

    X_val_trans = prep.transform(X_val)
    assert X_val_trans.shape[0] == len(X_val)
    assert not np.isnan(X_val_trans).any()

    # Check that row 0 and row 1 of X_val (10,000 and 20,000) both get winsorized to train's p99
    assert np.isclose(X_val_trans[0, 0], X_val_trans[1, 0])

    # Check CategoryAligner maps unseen category 'ZZ' to NaN while preserving '-1'
    aligner = CategoryAligner().fit(X_train)
    X_val_aligned = aligner.transform(X_val)
    assert pd.isna(X_val_aligned["cat_a"].iloc[0])  # 'ZZ' unseen in train -> NaN
    assert X_val_aligned["cat_a"].iloc[2] == "-1"


def test_ks_ece_and_decile_lift_manual_calculations() -> None:
    """Verify compute_ks_statistic, compute_ece, and compute_classification_lift_table on manual examples."""
    # 1. Manual KS test: perfect separation -> KS = 1.0
    y_perf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_perf = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert np.isclose(compute_ks_statistic(y_perf, p_perf), 1.0)

    # 2. Manual ECE test with 2 bins ([0.0, 0.5), [0.5, 1.0]):
    # Bin 1: 4 items with p=0.2, y=[0, 0, 0, 1] -> mean_p=0.20, mean_y=0.25 -> |gap|=0.05
    # Bin 2: 6 items with p=0.8, y=[1, 1, 1, 0, 0, 0] -> mean_p=0.80, mean_y=0.50 -> |gap|=0.30
    # ECE = (4/10)*0.05 + (6/10)*0.30 = 0.02 + 0.18 = 0.20
    y_ece = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    p_ece = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    ece_val, ece_tbl = compute_ece(y_ece, p_ece, n_bins=2)
    assert np.isclose(ece_val, 0.20, atol=1e-12)
    assert len(ece_tbl) == 2

    # 3. Manual Decile Lift test (10 items, 1 per decile, 4 defaults total -> base_rate = 0.40)
    # Top 2 items (D1, D2) are defaults (obs_rate = 1.0 -> lift = 1.0 / 0.4 = 2.5)
    y_lift = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    p_lift = np.linspace(0.95, 0.05, 10)
    lift_tbl = compute_classification_lift_table(y_lift, p_lift, n_deciles=10)
    assert len(lift_tbl) == 10
    assert np.isclose(lift_tbl.loc[0, "lift_desil"], 2.5)
    assert np.isclose(lift_tbl.loc[1, "cum_recall_captured"], 0.50)  # 2 of 4 defaults in top 2 deciles
    assert np.isclose(lift_tbl.loc[9, "cum_recall_captured"], 1.00)


def test_prediction_table_schema_and_bounds() -> None:
    """Verify prediction table validation rules on synthetic and generated prediction artifacts."""
    df_pred_mini = pd.DataFrame(
        {
            "LoanId": ["L1", "L2", "L3"],
            "split": ["train", "val", "test"],
            "LoanDate": pd.to_datetime(["2017-05-01", "2018-10-01", "2019-10-01"]),
            "loan_year": [2017, 2018, 2019],
            "Country": ["EE", "FI", "ES"],
            "Rating": ["B", "C", "HR"],
            "Amount": [1000.0, 2000.0, 1500.0],
            "AppliedAmount": [1000.0, 2000.0, 1500.0],
            "Interest": [18.0, 25.0, 45.0],
            "LoanDuration": [36, 36, 60],
            "default_12m": [0, 0, 1],
            "p_raw": [0.15, 0.32, 0.78],
            "p_calibrated": [0.12, 0.30, 0.74],
            "p_lr": [0.18, 0.35, 0.69],
            "PD_bondora": [0.14, 0.28, 0.65],
        }
    )

    req_cols = {
        "LoanId",
        "split",
        "LoanDate",
        "loan_year",
        "Country",
        "Rating",
        "Amount",
        "AppliedAmount",
        "Interest",
        "LoanDuration",
        "default_12m",
        "p_raw",
        "p_calibrated",
        "p_lr",
        "PD_bondora",
    }
    assert req_cols.issubset(df_pred_mini.columns)
    assert df_pred_mini["LoanId"].is_unique
    for p_col in ["p_raw", "p_calibrated", "p_lr"]:
        assert df_pred_mini[p_col].notna().all()
        assert ((df_pred_mini[p_col] >= 0.0) & (df_pred_mini[p_col] <= 1.0)).all()

    # Also verify generated prediction parquet artifacts on disk
    pred_cls_path = PROJECT_ROOT / "data" / "processed" / "predictions_default12m.parquet"
    pred_reg_path = PROJECT_ROOT / "data" / "processed" / "predictions_return_ratio.parquet"
    assert pred_cls_path.exists() and pred_reg_path.exists()

    df_p_cls = pd.read_parquet(pred_cls_path)
    assert len(df_p_cls) == 148_369
    assert req_cols.issubset(df_p_cls.columns)
    assert df_p_cls["LoanId"].is_unique
    for p_col in ["p_raw", "p_calibrated", "p_lr"]:
        assert df_p_cls[p_col].notna().all()
        assert ((df_p_cls[p_col] >= 0.0) & (df_p_cls[p_col] <= 1.0)).all()

    df_p_reg = pd.read_parquet(pred_reg_path)
    req_reg_cols = {
        "LoanId",
        "split",
        "LoanDate",
        "loan_year",
        "Country",
        "Rating",
        "Amount",
        "AppliedAmount",
        "Interest",
        "LoanDuration",
        "return_ratio",
        "pred_return_ratio",
        "pred_baseline_group_median",
    }
    assert len(df_p_reg) == 113_560
    assert req_reg_cols.issubset(df_p_reg.columns)
    assert df_p_reg["LoanId"].is_unique
    assert df_p_reg["pred_return_ratio"].notna().all()
    assert df_p_reg["pred_baseline_group_median"].notna().all()


def test_regression_metrics_manual_hand_calculation() -> None:
    """Verify evaluate_regression_predictions matches manual MAE, RMSE, R2, and Spearman calculation."""
    y_true = np.array([1.0, 0.5, 0.0, 1.5])
    y_pred = np.array([0.8, 0.5, 0.2, 1.3])
    def_flag = np.array([0, 1, 1, 0])

    # Absolute errors: |1.0-0.8|=0.2, |0.5-0.5|=0.0, |0.0-0.2|=0.2, |1.5-1.3|=0.2
    # MAE = (0.2 + 0.0 + 0.2 + 0.2) / 4 = 0.15
    # Squared errors: 0.04, 0.0, 0.04, 0.04 -> MSE = 0.12 / 4 = 0.03 -> RMSE = sqrt(0.03)
    # Ranks of y_true: [3, 2, 1, 4]; ranks of y_pred: [3, 2, 1, 4] -> Spearman rho = 1.0
    res = evaluate_regression_predictions(
        y_true=y_true,
        y_pred=y_pred,
        default_flag=def_flag,
        model_code="Manual-Test",
        model_name="Manual Regression Test",
        split_name="test",
    )

    assert np.isclose(res["mae"], 0.15, atol=1e-6)
    assert np.isclose(res["rmse"], np.sqrt(0.03), atol=1e-6)
    assert np.isclose(res["spearman_rho"], 1.0, atol=1e-6)
    # Default subset (indices 1, 2): errors 0.0 and 0.2 -> MAE_default = 0.10
    assert np.isclose(res["mae_default"], 0.10, atol=1e-6)
    # Non-default subset (indices 0, 3): errors 0.2 and 0.2 -> MAE_non_default = 0.20
    assert np.isclose(res["mae_non_default"], 0.20, atol=1e-6)
