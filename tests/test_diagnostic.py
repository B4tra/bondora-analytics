"""Unit tests for Tahap 3 diagnostic analytics functions (tests/test_diagnostic.py)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.diagnostic import (
    compute_cramers_v,
    compute_iv_woe,
    compute_psi,
    holm_bonferroni,
    kaplan_meier_curve,
    km_summary_at_horizons,
    mannwhitney_auc,
    multigroup_logrank_test,
)


def test_cramers_v_perfect_and_independent() -> None:
    """Verify Cramér's V equals 1.0 for perfect association and ~0.0 for independent variables."""
    # Perfect association: A always 0, B always 1
    x_perf = pd.Series(["A"] * 20 + ["B"] * 20)
    y_perf = pd.Series([0] * 20 + [1] * 20)
    res_perf = compute_cramers_v(x_perf, y_perf, min_expected=5.0, bias_correction=False)
    assert np.isclose(res_perf["cramers_v"], 1.0)
    assert np.isclose(res_perf["cramers_v_corrected"], 1.0)

    # Independent 2x2 table: 10 A-0, 10 A-1, 10 B-0, 10 B-1
    x_ind = pd.Series(["A"] * 20 + ["B"] * 20)
    y_ind = pd.Series([0, 1] * 20)
    res_ind = compute_cramers_v(x_ind, y_ind, min_expected=5.0, bias_correction=False)
    assert np.isclose(res_ind["cramers_v"], 0.0, atol=1e-12)
    assert np.isclose(res_ind["chi2"], 0.0, atol=1e-12)
    assert np.isclose(res_ind["p_value"], 1.0, atol=1e-12)


def test_iv_manual_example_and_random_near_zero() -> None:
    """Verify compute_iv_woe matches manual calculation and gives ~0 for random noise."""
    # Group A: 30 non-default, 10 default
    # Group B: 10 non-default, 30 default
    # Total non-default = 40, Total default = 40
    # Group A: p_non = 30/40 = 0.75, p_def = 10/40 = 0.25 -> WoE_A = ln(3) -> IV_A = (0.75 - 0.25)*ln(3) = 0.5*ln(3)
    # Group B: p_non = 10/40 = 0.25, p_def = 30/40 = 0.75 -> WoE_B = -ln(3) -> IV_B = (0.25 - 0.75)*(-ln(3)) = 0.5*ln(3)
    # Total IV = ln(3) = 1.0986122886681098
    df_manual = pd.DataFrame(
        {
            "grp": ["A"] * 40 + ["B"] * 40,
            "default_12m": pd.Series([0] * 30 + [1] * 10 + [0] * 10 + [1] * 30, dtype="Int8"),
        }
    )
    iv_val, woe_tbl = compute_iv_woe(df_manual, "grp", target_col="default_12m", min_cat_n=1)
    assert np.isclose(iv_val, np.log(3.0), atol=1e-10)
    assert len(woe_tbl) == 2

    # Random independent feature -> IV close to 0 (< 0.02)
    rng = np.random.default_rng(42)
    df_rand = pd.DataFrame(
        {
            "rand_num": rng.normal(0.0, 1.0, size=5000),
            "default_12m": pd.Series(rng.integers(0, 2, size=5000), dtype="Int8"),
        }
    )
    iv_rand, _ = compute_iv_woe(df_rand, "rand_num", target_col="default_12m", n_bins=10)
    assert iv_rand < 0.02


def test_psi_identical_zero_and_shifted_large() -> None:
    """Verify compute_psi is 0.0 for identical distributions and > 0.25 for heavily shifted distributions."""
    rng = np.random.default_rng(123)
    ref_num = pd.Series(rng.normal(100.0, 15.0, size=2000))
    psi_same, _ = compute_psi(ref_num, ref_num.copy(), n_bins=10)
    assert np.isclose(psi_same, 0.0, atol=1e-12)

    comp_shifted = pd.Series(rng.normal(130.0, 15.0, size=2000))
    psi_shift, _ = compute_psi(ref_num, comp_shifted, n_bins=10)
    assert psi_shift > 0.25

    # Also verify categorical PSI
    ref_cat = pd.Series(["EE"] * 500 + ["FI"] * 300 + ["ES"] * 200)
    psi_cat_same, _ = compute_psi(ref_cat, ref_cat.copy())
    assert np.isclose(psi_cat_same, 0.0, atol=1e-12)

    comp_cat_shift = pd.Series(["EE"] * 100 + ["FI"] * 200 + ["ES"] * 700)
    psi_cat_shift, _ = compute_psi(ref_cat, comp_cat_shift)
    assert psi_cat_shift > 0.25


def test_kaplan_meier_manual_hand_calculation() -> None:
    """Verify Kaplan-Meier survival and cumulative incidence match hand calculation on a 5-subject example."""
    # 5 subjects:
    # t = 2: 1 event (n_at_risk = 5, d = 1 -> S(2) = 1 * (1 - 1/5) = 0.80, inc = 0.20)
    # t = 4: 1 censored (n_at_risk = 4, d = 0 -> S(4) = 0.80, inc = 0.20)
    # t = 6: 1 event (n_at_risk = 3, d = 1 -> S(6) = 0.80 * (1 - 1/3) = 8/15 = 0.533333..., inc = 7/15 = 0.466667...)
    # t = 8: 2 censored (n_at_risk = 2, d = 0 -> S(8) = 8/15, inc = 7/15)
    durations = [2.0, 4.0, 6.0, 8.0, 8.0]
    events = [1, 0, 1, 0, 0]

    curve = kaplan_meier_curve(durations, events, max_horizon=12.0)

    row_0 = curve.loc[curve["time"] == 0.0].iloc[0]
    assert np.isclose(row_0["survival"], 1.0)
    assert np.isclose(row_0["cumulative_incidence"], 0.0)

    row_2 = curve.loc[curve["time"] == 2.0].iloc[0]
    assert row_2["n_at_risk"] == 5
    assert row_2["n_events"] == 1
    assert np.isclose(row_2["survival"], 0.80)
    assert np.isclose(row_2["cumulative_incidence"], 0.20)

    row_6 = curve.loc[curve["time"] == 6.0].iloc[0]
    assert row_6["n_at_risk"] == 3
    assert row_6["n_events"] == 1
    assert np.isclose(row_6["survival"], 8.0 / 15.0)
    assert np.isclose(row_6["cumulative_incidence"], 7.0 / 15.0)

    # Check summary at horizons (3.0 and 7.0)
    h_sum = km_summary_at_horizons(durations, events, horizons=(3.0, 7.0))
    assert np.isclose(h_sum.loc[h_sum["horizon_m"] == 3, "cumulative_incidence"].iloc[0], 0.20)
    assert np.isclose(h_sum.loc[h_sum["horizon_m"] == 7, "cumulative_incidence"].iloc[0], 7.0 / 15.0)


def test_holm_bonferroni_monotonic_and_greater_equal_raw() -> None:
    """Verify Holm-Bonferroni adjusted p-values satisfy p_holm >= p_value and preserve rank order."""
    raw_p = np.array([0.04, 0.001, 0.015, 0.25, 0.005])
    p_holm = holm_bonferroni(raw_p)

    # Manual check for sorted [0.001, 0.005, 0.015, 0.04, 0.25] with m=5:
    # i=0 (0.001): 5 * 0.001 = 0.005
    # i=1 (0.005): max(0.005, 4 * 0.005) = 0.020
    # i=2 (0.015): max(0.020, 3 * 0.015) = 0.045
    # i=3 (0.040): max(0.045, 2 * 0.040) = 0.080
    # i=4 (0.250): max(0.080, 1 * 0.250) = 0.250
    expected = np.array([0.080, 0.005, 0.045, 0.250, 0.020])
    assert np.allclose(p_holm, expected)
    assert np.all(p_holm >= raw_p)
    assert np.all((p_holm >= 0.0) & (p_holm <= 1.0))

    order = np.argsort(raw_p)
    assert np.all(np.diff(p_holm[order]) >= 0.0)


def test_mannwhitney_auc_direction_and_logrank() -> None:
    """Verify mannwhitney_auc and multigroup_logrank_test on separated groups."""
    df_mw = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            "default_12m": pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype="Int8"),
        }
    )
    res_mw = mannwhitney_auc(df_mw, "score", target_col="default_12m")
    assert np.isclose(res_mw["auc_raw"], 1.0)
    assert np.isclose(res_mw["auc_univariat"], 1.0)
    assert "positif" in res_mw["arah"]

    # Log-rank test between a high-hazard group and a zero-event group
    lr = multigroup_logrank_test(
        durations=[2, 3, 4, 5, 12, 12, 12, 12],
        events=[1, 1, 1, 1, 0, 0, 0, 0],
        groups=["High"] * 4 + ["Low"] * 4,
    )
    assert lr["chi2_stat"] > 3.84
    assert lr["p_value"] < 0.05
