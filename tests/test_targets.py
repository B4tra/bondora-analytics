"""Unit and integration tests for target construction and clean dataset pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, LOANDATA_FILENAME
from src.data.build_clean import build_clean_dataset
from src.data.targets import (
    make_default_label,
    make_is_resolved,
    make_regression_target,
)


@pytest.fixture
def sample_loans_df() -> pd.DataFrame:
    """Create an 8-row synthetic DataFrame covering all target & edge cases."""
    return pd.DataFrame(
        {
            "Status": [
                "Late",      # 0: DefaultDate present -> default=1, is_resolved=True
                "Repaid",    # 1: DefaultDate present -> default=1, is_resolved=True
                "Current",   # 2: DefaultDate present -> default=1, is_resolved=True
                "Repaid",    # 3: DefaultDate NaN, PrincipalWriteOffs > 0 -> default=1, is_resolved=True
                "Repaid",    # 4: DefaultDate NaN, only InterestWriteOffs > 0 -> default=0, is_resolved=True
                "Repaid",    # 5: DefaultDate NaN, no write-offs -> default=0, is_resolved=True
                "Current",   # 6: DefaultDate NaN -> default=NA, is_resolved=False (censored), Amount=0
                "Late",      # 7: DefaultDate NaN -> default=NA, is_resolved=False (censored), Amount=NaN
            ],
            "DefaultDate": [
                "2020-05-10",
                "2019-11-01",
                "2020-01-15",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "PrincipalWriteOffs": [0.0, 0.0, 0.0, 150.0, 0.0, np.nan, 0.0, 0.0],
            "InterestAndPenaltyWriteOffs": [0.0, 0.0, 0.0, 20.0, 45.0, 0.0, 0.0, 0.0],
            "Amount": [1000.0, 2000.0, 1500.0, 1000.0, 1000.0, 500.0, 0.0, np.nan],
            "PrincipalPaymentsMade": [200.0, 2000.0, 500.0, 850.0, 1000.0, 500.0, 100.0, np.nan],
            "InterestAndPenaltyPaymentsMade": [50.0, 300.0, 100.0, 100.0, 150.0, 80.0, 20.0, 10.0],
        }
    )


def test_make_default_label_values_and_rules(sample_loans_df: pd.DataFrame) -> None:
    """Verify make_default_label outputs nullable Int8 with 0/1 for resolved and NA for unresolved."""
    labels = make_default_label(sample_loans_df)

    assert len(labels) == len(sample_loans_df)
    assert str(labels.dtype) == "Int8"
    assert set(labels.dropna().unique()).issubset({0, 1})

    assert labels.iloc[:6].tolist() == [1, 1, 1, 1, 0, 0]
    assert pd.isna(labels.iloc[6])
    assert pd.isna(labels.iloc[7])


def test_default_na_matches_unresolved_exactly(sample_loans_df: pd.DataFrame) -> None:
    """Verify default is NA if and only if is_resolved == False."""
    labels = make_default_label(sample_loans_df)
    resolved = make_is_resolved(sample_loans_df)

    assert labels.isna().equals(~resolved)
    assert labels.notna().equals(resolved)


def test_make_is_resolved_rules(sample_loans_df: pd.DataFrame) -> None:
    """Verify make_is_resolved identifies resolved vs right-censored loans."""
    resolved = make_is_resolved(sample_loans_df)

    assert len(resolved) == len(sample_loans_df)
    assert resolved.dtype == bool

    expected = [True, True, True, True, True, True, False, False]
    assert resolved.tolist() == expected


def test_make_regression_target_no_inf_and_exact_ratio(sample_loans_df: pd.DataFrame) -> None:
    """Verify make_regression_target handles Amount=0 and NaN without inf/-inf."""
    ratios = make_regression_target(sample_loans_df)

    assert len(ratios) == len(sample_loans_df)
    assert not np.isinf(ratios).any()

    # Row 0: (200 + 50) / 1000 = 0.25
    assert np.isclose(ratios.iloc[0], 0.25)
    # Row 1: (2000 + 300) / 2000 = 1.15
    assert np.isclose(ratios.iloc[1], 1.15)
    # Row 6 (Amount == 0) and Row 7 (Amount is NaN) must be NaN
    assert np.isnan(ratios.iloc[6])
    assert np.isnan(ratios.iloc[7])


@pytest.mark.skipif(
    not (DATA_RAW / LOANDATA_FILENAME).exists(),
    reason="Raw dataset data/raw/LoanData.csv is not available.",
)
def test_build_clean_dataset_integration(tmp_path: Path) -> None:
    """Integration test verifying build_clean_dataset produces 179,235 x 126 and exact Int8 default counts."""
    out_parquet = tmp_path / "bondora_clean.parquet"
    out_schema = tmp_path / "schema.json"

    df_clean = build_clean_dataset(
        parquet_path=out_parquet,
        schema_path=out_schema,
        save_tables=False,
    )

    assert df_clean.shape == (179_235, 126)
    assert str(df_clean["default"].dtype) == "Int8"
    assert int(df_clean["default"].notna().sum()) == 113_560
    assert int(df_clean["default"].sum()) == 71_864
    assert int(df_clean["default"].isna().sum()) == 65_675
