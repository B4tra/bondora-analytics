"""Unit tests for basic derived features in src/features/basic.py."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.basic import (
    add_basic_derived_features,
    compute_avg_previous_loan_amount,
    compute_funded_to_applied_ratio,
    compute_loan_to_income_ratio,
    compute_loan_year_and_month,
    compute_monthly_payment_to_income_ratio,
    compute_total_liabilities_to_income_ratio,
    safe_divide,
)


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create a 6-row synthetic DataFrame covering normal, zero-denominator, and NaN cases."""
    return pd.DataFrame(
        {
            "AppliedAmount": [3000.0, 2000.0, 1500.0, 0.0, np.nan, 4000.0],
            "Amount": [2400.0, 2000.0, 1500.0, 0.0, 1000.0, 3000.0],
            "MonthlyPayment": [150.0, 100.0, np.nan, 80.0, 90.0, 200.0],
            "IncomeTotal": [1500.0, 0.0, -100.0, np.nan, 2000.0, 1000.0],
            "LiabilitiesTotal": [450.0, 300.0, 200.0, 100.0, np.nan, 0.0],
            "NoOfPreviousLoansBeforeLoan": [2.0, 0.0, 0.0, np.nan, 3.0, -1.0],
            "AmountOfPreviousLoansBeforeLoan": [4000.0, 0.0, 500.0, 1000.0, np.nan, 1000.0],
            "LoanDate": [
                "2019-03-15",
                "2020-11-02",
                np.nan,
                "2021-07-19",
                "invalid-date",
                "2015-01-31",
            ],
        }
    )


def test_safe_divide_no_inf() -> None:
    """Verify safe_divide never produces inf/-inf on zero, negative, or NaN denominators."""
    num = pd.Series([100.0, 0.0, 50.0, np.nan])
    den = pd.Series([0.0, 0.0, -5.0, 10.0])

    res_std = safe_divide(num, den, zero_when_both_zero=False)
    assert not np.isinf(res_std).any()
    assert res_std.isna().all()

    res_zero = safe_divide(num, den, zero_when_both_zero=True)
    assert not np.isinf(res_zero).any()
    assert np.isnan(res_zero.iloc[0])
    assert np.isclose(res_zero.iloc[1], 0.0)


def test_income_based_ratios_handle_zero_and_nan(sample_features_df: pd.DataFrame) -> None:
    """Verify ratios dividing by IncomeTotal return NaN for <=0 or NaN without inf."""
    lti = compute_loan_to_income_ratio(sample_features_df)
    mpti = compute_monthly_payment_to_income_ratio(sample_features_df)
    tlti = compute_total_liabilities_to_income_ratio(sample_features_df)

    for s in (lti, mpti, tlti):
        assert len(s) == len(sample_features_df)
        assert not np.isinf(s).any()
        # Rows 1 (IncomeTotal=0), 2 (IncomeTotal=-100), 3 (IncomeTotal=NaN) must be NaN
        assert np.isnan(s.iloc[1])
        assert np.isnan(s.iloc[2])
        assert np.isnan(s.iloc[3])

    # Row 0: AppliedAmount=3000, MonthlyPayment=150, LiabilitiesTotal=450, IncomeTotal=1500
    assert np.isclose(lti.iloc[0], 2.0)
    assert np.isclose(mpti.iloc[0], 0.1)
    assert np.isclose(tlti.iloc[0], 0.3)


def test_avg_previous_loan_amount_zero_over_zero_and_anom(sample_features_df: pd.DataFrame) -> None:
    """Verify avg_previous_loan_amount handles 0/0 -> 0.0 and anomalous >0/0 -> NaN."""
    avg_prev = compute_avg_previous_loan_amount(sample_features_df)

    assert len(avg_prev) == len(sample_features_df)
    assert not np.isinf(avg_prev).any()

    # Row 0: 4000 / 2 = 2000.0
    assert np.isclose(avg_prev.iloc[0], 2000.0)
    # Row 1: 0 / 0 -> 0.0 (first-time borrower)
    assert np.isclose(avg_prev.iloc[1], 0.0)
    # Row 2: 500 / 0 -> NaN (anomalous positive amount with 0 count)
    assert np.isnan(avg_prev.iloc[2])
    # Rows 3, 4, 5 -> NaN
    assert np.isnan(avg_prev.iloc[3])
    assert np.isnan(avg_prev.iloc[4])
    assert np.isnan(avg_prev.iloc[5])


def test_funded_to_applied_ratio_and_loan_year_month(sample_features_df: pd.DataFrame) -> None:
    """Verify funded_to_applied_ratio and compute_loan_year_and_month outputs and row counts."""
    fta = compute_funded_to_applied_ratio(sample_features_df)
    assert len(fta) == len(sample_features_df)
    assert not np.isinf(fta).any()
    assert np.isclose(fta.iloc[0], 0.8)
    assert np.isnan(fta.iloc[3])  # AppliedAmount == 0
    assert np.isnan(fta.iloc[4])  # AppliedAmount is NaN

    loan_year, loan_month = compute_loan_year_and_month(sample_features_df)
    assert len(loan_year) == len(sample_features_df)
    assert len(loan_month) == len(sample_features_df)
    assert loan_year.iloc[0] == 2019
    assert loan_month.iloc[0] == 3
    assert pd.isna(loan_year.iloc[2])
    assert pd.isna(loan_year.iloc[4])


def test_add_basic_derived_features_preserves_rows(sample_features_df: pd.DataFrame) -> None:
    """Verify add_basic_derived_features appends all 7 derived columns and preserves row count."""
    out = add_basic_derived_features(sample_features_df)
    assert len(out) == len(sample_features_df)
    assert out.shape[1] == sample_features_df.shape[1] + 7
    expected_new_cols = {
        "loan_to_income_ratio",
        "monthly_payment_to_income_ratio",
        "total_liabilities_to_income_ratio",
        "avg_previous_loan_amount",
        "funded_to_applied_ratio",
        "loan_year",
        "loan_month",
    }
    assert expected_new_cols.issubset(set(out.columns))
