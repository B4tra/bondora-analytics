"""Unit tests for Tahap 2 descriptive analytics helper functions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.descriptive import (
    OUTCOME_CATEGORIES,
    default_rate_by,
    freq_table,
    make_outcome_group,
    numeric_summary,
    outcome_mix_by,
    weighted_return_ratio,
    wilson_ci,
)
from src.utils.plotting import fmt_id


@pytest.fixture()
def mini_df() -> pd.DataFrame:
    """Create a 12-row synthetic loan DataFrame for unit testing."""
    return pd.DataFrame(
        {
            "Status": [
                "Late",
                "Repaid",
                "Late",
                "Repaid",
                "Repaid",
                "Repaid",
                "Current",
                "Current",
                "Current",
                "Late",
                "Late",
                "Repaid",
            ],
            "DefaultDate": [
                "2020-01-10",
                "2020-02-15",
                "2020-03-20",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "2020-05-01",
            ],
            "default": pd.Series(
                [1, 1, 1, 0, 0, 0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1],
                dtype="Int8",
            ),
            "is_resolved": [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            "Country": [
                "EE",
                "EE",
                "EE",
                "EE",
                "FI",
                "FI",
                "EE",
                "FI",
                "ES",
                "ES",
                "SK",
                "SK",
            ],
            "Amount": [
                1000.0,
                2000.0,
                1000.0,
                1000.0,
                2000.0,
                1000.0,
                1500.0,
                2500.0,
                3000.0,
                1200.0,
                800.0,
                2000.0,
            ],
            "PrincipalPaymentsMade": [
                200.0,
                500.0,
                100.0,
                1000.0,
                2000.0,
                1000.0,
                300.0,
                400.0,
                500.0,
                100.0,
                50.0,
                400.0,
            ],
            "InterestAndPenaltyPaymentsMade": [
                50.0,
                100.0,
                50.0,
                200.0,
                400.0,
                150.0,
                60.0,
                80.0,
                90.0,
                20.0,
                10.0,
                100.0,
            ],
            "return_ratio": [
                0.25,
                0.30,
                0.15,
                1.20,
                1.20,
                1.15,
                0.24,
                0.192,
                0.1967,
                0.10,
                0.075,
                0.25,
            ],
        }
    )


def test_make_outcome_group_exact_partition(mini_df: pd.DataFrame) -> None:
    """Verify every row maps to exactly one outcome group with expected counts."""
    grp = make_outcome_group(mini_df)
    assert len(grp) == len(mini_df)
    assert grp.isna().sum() == 0
    assert list(grp.cat.categories) == OUTCOME_CATEGORIES

    counts = grp.value_counts()
    assert counts["Default"] == 4
    assert counts["Lunas tanpa default"] == 3
    assert counts["Berjalan (Current)"] == 3
    assert counts["Terlambat belum default"] == 2
    assert int(counts.sum()) == 12


def test_default_rate_by_ignores_na_and_merges_small_groups(mini_df: pd.DataFrame) -> None:
    """Verify default_rate_by excludes NA default rows and merges n < min_n into 'Lainnya'."""
    # Resolved rows in mini_df:
    # EE: 4 rows (3 default=1 with Amount 1000,2000,1000; 1 default=0 with Amount 1000)
    # FI: 2 rows (0 default=1; 2 default=0 with Amount 2000,1000)
    # SK: 1 row (1 default=1 with Amount 2000) -> n=1 < min_n=2 -> merged into 'Lainnya'
    res = default_rate_by(mini_df, group_col="Country", weight_col="Amount", min_n=2)

    assert int(res["n"].sum()) == 7  # Only the 7 resolved rows are included
    assert set(res["segmen"]) == {"EE", "FI", "Lainnya"}

    ee_row = res.loc[res["segmen"] == "EE"].iloc[0]
    assert ee_row["n"] == 4
    assert ee_row["n_default"] == 3
    assert np.isclose(ee_row["default_rate"], 3 / 4)
    # Amount-weighted default rate for EE: (1000+2000+1000) / (1000+2000+1000+1000) = 4000/5000 = 0.8
    assert np.isclose(ee_row["default_rate_by_amount"], 0.8)

    fi_row = res.loc[res["segmen"] == "FI"].iloc[0]
    assert fi_row["n"] == 2
    assert fi_row["n_default"] == 0
    assert np.isclose(fi_row["default_rate"], 0.0)

    lainnya_row = res.loc[res["segmen"] == "Lainnya"].iloc[0]
    assert lainnya_row["n"] == 1
    assert lainnya_row["n_default"] == 1
    assert np.isclose(lainnya_row["default_rate"], 1.0)


def test_wilson_ci_bounds_and_bracketing(mini_df: pd.DataFrame) -> None:
    """Verify Wilson 95% confidence intervals lie within [0, 1] and bracket default_rate."""
    res = default_rate_by(mini_df, group_col="Country", min_n=1)
    assert (res["ci_low"] >= 0.0).all()
    assert (res["ci_high"] <= 1.0).all()
    assert (res["ci_low"] <= res["default_rate"]).all()
    assert (res["default_rate"] <= res["ci_high"]).all()

    # Also test edge proportions directly
    low, high = wilson_ci(np.array([0, 5, 10]), np.array([10, 10, 10]))
    p = np.array([0.0, 0.5, 1.0])
    assert np.all(low >= 0.0) and np.all(high <= 1.0)
    assert np.all(low <= p) and np.all(p <= high)


def test_weighted_return_ratio_matches_manual_calculation(mini_df: pd.DataFrame) -> None:
    """Verify weighted_return_ratio matches manual numerator/denominator sum on resolved rows."""
    stats = weighted_return_ratio(mini_df)
    # Resolved rows indices: 0, 1, 2, 3, 4, 5, 11 (7 rows)
    # PrincipalPaymentsMade sum: 200 + 500 + 100 + 1000 + 2000 + 1000 + 400 = 5200
    # InterestAndPenaltyPaymentsMade sum: 50 + 100 + 50 + 200 + 400 + 150 + 100 = 1050
    # Total numerator = 6250.0
    # Total Amount on resolved rows: 1000 + 2000 + 1000 + 1000 + 2000 + 1000 + 2000 = 10000.0
    # Weighted return ratio = 6250 / 10000 = 0.625
    # Simple mean return_ratio = (0.25 + 0.30 + 0.15 + 1.20 + 1.20 + 1.15 + 0.25) / 7 = 4.50 / 7
    assert stats["n_resolved"] == 7
    assert np.isclose(stats["numerator_total_payments"], 6250.0)
    assert np.isclose(stats["denominator_total_amount"], 10000.0)
    assert np.isclose(stats["weighted_return_ratio"], 0.625)
    assert np.isclose(stats["simple_mean_return_ratio"], 4.50 / 7.0)


def test_freq_table_sums_to_100_percent(mini_df: pd.DataFrame) -> None:
    """Verify freq_table percentages sum to 100% and are sorted descending."""
    ft = freq_table(mini_df, "Country")
    assert int(ft["jumlah"].sum()) == len(mini_df)
    assert np.isclose(ft["persen"].sum(), 100.0)
    assert np.isclose(ft["persen_kumulatif"].iloc[-1], 100.0)
    assert ft["jumlah"].is_monotonic_decreasing


def test_numeric_summary_and_outcome_mix_and_fmt_id(mini_df: pd.DataFrame) -> None:
    """Verify numeric_summary restricts return_ratio to resolved rows and fmt_id formats numbers."""
    ns = numeric_summary(mini_df, ["Amount", "return_ratio"])
    amt_row = ns.loc[ns["nama_kolom"] == "Amount"].iloc[0]
    rr_row = ns.loc[ns["nama_kolom"] == "return_ratio"].iloc[0]
    assert amt_row["n"] == 12
    assert rr_row["n"] == 7  # Only resolved rows

    mix = outcome_mix_by(mini_df, "Country")
    assert int(mix["n_total"].sum()) == 12
    assert int(mix["n_resolved"].sum()) == 7

    assert fmt_id(179235, is_int=True) == "179.235"
    assert fmt_id(1234.56, decimals=2) == "1.234,56"
