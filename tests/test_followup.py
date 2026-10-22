"""Unit tests for Tahap 4 follow-up checks (tests/test_followup.py)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.followup import (
    FORCED_NAN_FEATURES,
    apply_fill_values,
    assign_test_half,
    compute_level_bias_table,
    fit_fill_values,
    nan_concentration_table,
    select_nan_features,
    shap_by_nan_group,
)

TABLES = PROJECT_ROOT / "reports" / "tables"


def _mini_pred() -> pd.DataFrame:
    """Hand-made prediction table: 2 validation rows and 6 test rows across Country and half-year."""
    return pd.DataFrame(
        {
            "LoanId": [f"L{i}" for i in range(8)],
            "split": ["val", "val", "test", "test", "test", "test", "test", "test"],
            "Country": ["EE", "EE", "EE", "EE", "FI", "FI", "ES", "ES"],
            "LoanDate": pd.to_datetime(
                [
                    "2018-09-01",
                    "2018-10-01",
                    "2019-08-01",  # EE, Jul-Des 2019
                    "2020-02-01",  # EE, Jan-Jul 2020
                    "2019-12-31",  # FI, Jul-Des 2019 (boundary)
                    "2020-01-01",  # FI, Jan-Jul 2020 (boundary)
                    "2019-09-01",  # ES
                    "2019-10-01",  # ES
                ]
            ),
            "default_12m": [0, 1, 0, 0, 1, 0, 1, 1],
            "p_raw": [0.3, 0.5, 0.2, 0.2, 0.6, 0.4, 0.8, 0.8],
            "p_calibrated": [0.2, 0.6, 0.1, 0.3, 0.5, 0.3, 0.7, 0.8],
            "PD_bondora": [0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        }
    )


def test_assign_test_half_boundaries() -> None:
    """2019-12-31 belongs to 'Jul–Des 2019'; 2020-01-01 belongs to 'Jan–Jul 2020'."""
    s = assign_test_half(pd.Series(pd.to_datetime(["2019-07-01", "2019-12-31", "2020-01-01", "2020-07-20"])))
    assert s.tolist() == ["Jul–Des 2019", "Jul–Des 2019", "Jan–Jul 2020", "Jan–Jul 2020"]


def test_level_bias_table_hand_calculation() -> None:
    """Mean predicted vs actual rate must match hand calculation overall, by country and by half-year."""
    tbl = compute_level_bias_table(_mini_pred()).set_index(["cakupan", "split", "subgrup"])

    # Test overall: mean p_cal = (0.1+0.3+0.5+0.3+0.7+0.8)/6 = 0.45 ; actual = 3/6 = 0.5
    row = tbl.loc[("Seluruh", "test", "Seluruh")]
    assert row["n"] == 6
    assert np.isclose(row["mean_p_calibrated"], 0.45)
    assert np.isclose(row["default_12m_aktual"], 0.5)
    assert np.isclose(row["selisih_cal_minus_aktual_pp"], -5.0)
    assert np.isclose(row["rasio_cal_per_aktual"], 0.9)

    # Validation overall is flagged as in-sample.
    val = tbl.loc[("Seluruh", "val", "Seluruh")]
    assert np.isclose(val["mean_p_calibrated"], 0.4)
    assert "in-sample" in val["sifat_metrik"]
    assert "out-of-sample" in row["sifat_metrik"]

    # Country EE (test): p_cal mean (0.1 + 0.3)/2 = 0.2 ; actual 0
    ee = tbl.loc[("Country", "test", "EE")]
    assert ee["n"] == 2 and np.isclose(ee["mean_p_calibrated"], 0.2) and np.isclose(ee["default_12m_aktual"], 0.0)

    # Half-year (test): Jul-Des 2019 rows = EE(0.1), FI(0.5), ES(0.7), ES(0.8) -> mean 0.525, actual 3/4
    h1 = tbl.loc[("paruh_waktu_uji", "test", "Jul–Des 2019")]
    assert h1["n"] == 4 and np.isclose(h1["mean_p_calibrated"], 0.525) and np.isclose(h1["default_12m_aktual"], 0.75)

    # Crossed segment exists and is consistent.
    fi2 = tbl.loc[("Country x paruh_waktu_uji", "test", "FI | Jan–Jul 2020")]
    assert fi2["n"] == 1 and np.isclose(fi2["mean_p_calibrated"], 0.3)


def test_nan_concentration_rule_and_selection() -> None:
    """A feature whose NaN share collapses between eras is selected; a uniformly sparse one is not."""
    n = 100
    df = pd.DataFrame(
        {
            "split": ["train"] * n + ["val"] * n + ["test"] * n,
            # 20% NaN in train, 0% in val/test -> spread 20 pp -> concentrated
            "era_nan": [np.nan] * 20 + [1.0] * 80 + [1.0] * 2 * n,
            # 10% NaN everywhere -> spread 0 pp -> not concentrated
            "flat_nan": ([np.nan] * 10 + [1.0] * 90) * 3,
            # never NaN
            "clean": [1.0] * 3 * n,
            "PreviousRepaymentsBeforeLoan": [1.0] * 3 * n,
        }
    )
    conc = nan_concentration_table(df, ["era_nan", "flat_nan", "clean", "PreviousRepaymentsBeforeLoan"]).set_index("fitur")
    assert bool(conc.loc["era_nan", "memenuhi_aturan_konsentrasi"])
    assert np.isclose(conc.loc["era_nan", "rentang_nan_pp"], 20.0)
    assert not bool(conc.loc["flat_nan", "memenuhi_aturan_konsentrasi"])
    assert not bool(conc.loc["clean", "memenuhi_aturan_konsentrasi"])

    selected = select_nan_features(conc.reset_index())
    assert "era_nan" in selected
    assert "flat_nan" not in selected and "clean" not in selected
    # Requested features are always included even if they do not satisfy the rule.
    assert "PreviousRepaymentsBeforeLoan" in FORCED_NAN_FEATURES
    assert "PreviousRepaymentsBeforeLoan" in selected


def test_shap_by_nan_group_hand_calculation() -> None:
    """Mean SHAP of NaN vs non-NaN rows, including a selected feature with zero NaN rows."""
    X = pd.DataFrame({"f1": [np.nan, np.nan, 1.0, 2.0], "f2": [1.0, 2.0, 3.0, 4.0]})
    contrib = np.array([[0.5, 0.1], [0.3, 0.1], [-0.2, 0.2], [-0.4, 0.2]])
    y = np.array([1, 1, 0, 0])
    tbl = shap_by_nan_group(contrib, X, y, ["f1", "f2"], "uji", selected=["f1", "f2"]).set_index("fitur")

    assert tbl.loc["f1", "n_nan"] == 2
    assert np.isclose(tbl.loc["f1", "mean_shap_nan"], 0.4)
    assert np.isclose(tbl.loc["f1", "mean_shap_non_nan"], -0.3)
    assert np.isclose(tbl.loc["f1", "selisih_shap_nan_minus_non_nan"], 0.7)
    assert np.isclose(tbl.loc["f1", "default_rate_nan"], 1.0)
    assert np.isclose(tbl.loc["f1", "default_rate_non_nan"], 0.0)

    # f2 is selected but has no NaN rows in the sample -> NaN-group stats are NaN, non-NaN stats still present.
    assert tbl.loc["f2", "n_nan"] == 0
    assert np.isnan(tbl.loc["f2", "mean_shap_nan"])
    assert np.isclose(tbl.loc["f2", "mean_shap_non_nan"], 0.15)


def test_fill_values_use_train_statistics_only() -> None:
    """Constants are the TRAIN median / mode, never influenced by validation or test values."""
    df_train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, np.nan],
            "cat": pd.Categorical(["a", "b", "b", None]),
        }
    )
    df_other = pd.DataFrame(
        {
            "num": [1000.0, 2000.0, np.nan],
            "cat": pd.Categorical(["a", "a", None], categories=["a", "b"]),
        }
    )
    fill = fit_fill_values(df_train, ["num", "cat"])
    assert fill["num"] == pytest.approx(2.0)  # median of [1, 2, 3], not influenced by 1000/2000
    assert fill["cat"] == "b"

    filled = apply_fill_values(df_other, fill)
    assert filled["num"].tolist() == [1000.0, 2000.0, 2.0]
    assert filled["cat"].tolist() == ["a", "a", "b"]
    assert not filled.isna().any().any()
    # Original frame is untouched.
    assert df_other["num"].isna().sum() == 1


def test_generated_followup_tables_schema() -> None:
    """Generated follow-up artifacts: baselines per country, level bias values, refit variants."""
    sub = pd.read_csv(TABLES / "04_subgroup_performance.csv")
    assert {"kode_model", "selisih_auc_lgbm_a_minus_pembanding", "selisih_auc_ci_low_95", "selisih_auc_ci_high_95"}.issubset(
        sub.columns
    )
    for c in ("EE", "FI", "ES"):
        block = sub.loc[(sub["dimensi"] == "Country") & (sub["subgrup"] == c)].set_index("kode_model")
        assert {"LGBM-A", "BASE-PD-Bondora", "BASE-Interest"} == set(block.index)
        assert block.loc["LGBM-A", "n"] == block.loc["BASE-PD-Bondora", "n"] == block.loc["BASE-Interest", "n"]
        for code in ("BASE-PD-Bondora", "BASE-Interest"):
            d = block.loc[code]
            # The reported difference must equal LGBM-A AUC minus the baseline AUC (point estimates).
            assert d["selisih_auc_lgbm_a_minus_pembanding"] == pytest.approx(
                block.loc["LGBM-A", "roc_auc"] - d["roc_auc"], abs=2e-6
            )
            assert d["selisih_auc_ci_low_95"] <= d["selisih_auc_lgbm_a_minus_pembanding"] <= d["selisih_auc_ci_high_95"]

    lvl = pd.read_csv(TABLES / "04_level_bias.csv").set_index(["cakupan", "split", "subgrup"])
    test_all = lvl.loc[("Seluruh", "test", "Seluruh")]
    assert test_all["n"] == 54_283
    assert test_all["mean_p_calibrated"] == pytest.approx(0.32959, abs=5e-5)
    assert test_all["default_12m_aktual"] == pytest.approx(0.267708, abs=5e-6)
    val_all = lvl.loc[("Seluruh", "val", "Seluruh")]
    assert val_all["mean_p_calibrated"] == pytest.approx(val_all["default_12m_aktual"], abs=1e-6)  # in-sample isotonic

    refit = pd.read_csv(TABLES / "04_nan_signal_refit.csv")
    assert len(refit) == 3
    assert refit["varian"].str.contains("utama").any() and refit["varian"].str.contains("Kontrol").any()
    assert refit["varian"].str.startswith("Refit NaN").any()

    metrics = pd.read_csv(TABLES / "04_metrics_classification.csv")
    assert "cakupan_evaluasi" in metrics.columns
    assert metrics.loc[metrics["split"] == "val", "cakupan_evaluasi"].str.contains("in-sample").all()
    assert metrics.loc[metrics["split"] == "test", "cakupan_evaluasi"].str.contains("out-of-sample").all()
