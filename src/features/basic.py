"""Safe derived features computed strictly from origination-time columns."""

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    zero_when_both_zero: bool = False,
) -> pd.Series:
    """Safely divide two numeric Series without producing ``inf`` or ``-inf``.

    Any division where ``denominator <= 0`` or ``NaN`` returns ``NaN``,
    except when ``zero_when_both_zero=True`` and both ``numerator == 0``
    and ``denominator == 0``, which returns ``0.0``.

    Parameters
    ----------
    numerator : pd.Series
        Numerator series.
    denominator : pd.Series
        Denominator series (must be strictly positive for standard division).
    zero_when_both_zero : bool, default False
        If ``True``, rows where ``numerator == 0`` and ``denominator == 0``
        are assigned ``0.0`` instead of ``NaN``.

    Returns
    -------
    pd.Series
        Resulting float64 Series free of ``inf`` and ``-inf``.
    """
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")

    valid_den = den.where(den > 0.0, np.nan)
    result = num / valid_den

    if zero_when_both_zero:
        both_zero = (num == 0.0) & (den == 0.0)
        result = result.where(~both_zero, 0.0)

    return result.astype("float64")


def compute_loan_to_income_ratio(
    df: pd.DataFrame,
    applied_amount_col: str = "AppliedAmount",
    income_total_col: str = "IncomeTotal",
) -> pd.Series:
    """Compute ratio of applied loan amount to monthly total income.

    Formula:
        ``loan_to_income_ratio = AppliedAmount / IncomeTotal``
        Returns ``NaN`` if ``IncomeTotal <= 0`` or ``NaN``.
    """
    res = safe_divide(df[applied_amount_col], df[income_total_col], zero_when_both_zero=False)
    return res.rename("loan_to_income_ratio")


def compute_monthly_payment_to_income_ratio(
    df: pd.DataFrame,
    monthly_payment_col: str = "MonthlyPayment",
    income_total_col: str = "IncomeTotal",
) -> pd.Series:
    """Compute ratio of scheduled monthly loan payment to monthly total income.

    Formula:
        ``monthly_payment_to_income_ratio = MonthlyPayment / IncomeTotal``
        Returns ``NaN`` if ``IncomeTotal <= 0`` or ``NaN``.
    """
    res = safe_divide(df[monthly_payment_col], df[income_total_col], zero_when_both_zero=False)
    return res.rename("monthly_payment_to_income_ratio")


def compute_total_liabilities_to_income_ratio(
    df: pd.DataFrame,
    liabilities_total_col: str = "LiabilitiesTotal",
    income_total_col: str = "IncomeTotal",
) -> pd.Series:
    """Compute ratio of total monthly liabilities to monthly total income.

    Formula:
        ``total_liabilities_to_income_ratio = LiabilitiesTotal / IncomeTotal``
        Returns ``NaN`` if ``IncomeTotal <= 0`` or ``NaN``.
    """
    res = safe_divide(df[liabilities_total_col], df[income_total_col], zero_when_both_zero=False)
    return res.rename("total_liabilities_to_income_ratio")


def compute_avg_previous_loan_amount(
    df: pd.DataFrame,
    amount_prev_col: str = "AmountOfPreviousLoansBeforeLoan",
    count_prev_col: str = "NoOfPreviousLoansBeforeLoan",
) -> pd.Series:
    """Compute average amount of previous Bondora loans prior to this application.

    Formula:
        ``avg_previous_loan_amount = AmountOfPreviousLoansBeforeLoan / NoOfPreviousLoansBeforeLoan``
        Returns ``0.0`` when both numerator and denominator are ``0``, and
        ``NaN`` if denominator ``<= 0`` with positive numerator or if either is ``NaN``.
    """
    res = safe_divide(df[amount_prev_col], df[count_prev_col], zero_when_both_zero=True)
    return res.rename("avg_previous_loan_amount")


def compute_funded_to_applied_ratio(
    df: pd.DataFrame,
    amount_col: str = "Amount",
    applied_amount_col: str = "AppliedAmount",
) -> pd.Series:
    """Compute ratio of disbursed amount to applied amount (for analysis, not model feature).

    Formula:
        ``funded_to_applied_ratio = Amount / AppliedAmount``
        Returns ``NaN`` if ``AppliedAmount <= 0`` or ``NaN``.
    """
    res = safe_divide(df[amount_col], df[applied_amount_col], zero_when_both_zero=False)
    return res.rename("funded_to_applied_ratio")


def compute_loan_year_and_month(
    df: pd.DataFrame,
    loan_date_col: str = "LoanDate",
) -> tuple[pd.Series, pd.Series]:
    """Extract calendar year and month from ``LoanDate`` (for temporal splitting/analysis).

    Formula:
        ``loan_year = pd.to_datetime(LoanDate).dt.year``
        ``loan_month = pd.to_datetime(LoanDate).dt.month``
    """
    dt = pd.to_datetime(df[loan_date_col], errors="coerce", format="mixed")
    loan_year = dt.dt.year.astype("Int64").rename("loan_year")
    loan_month = dt.dt.month.astype("Int64").rename("loan_month")
    return loan_year, loan_month


def add_basic_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all 7 approved derived columns to a copy of ``df``.

    Model features (``boleh_jadi_fitur = Y``):
    - ``loan_to_income_ratio``
    - ``monthly_payment_to_income_ratio``
    - ``total_liabilities_to_income_ratio``
    - ``avg_previous_loan_amount``

    Analysis / temporal split columns (``boleh_jadi_fitur = N``):
    - ``loan_year``
    - ``loan_month``
    - ``funded_to_applied_ratio``

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required origination columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the 7 derived columns appended.
    """
    loan_year, loan_month = compute_loan_year_and_month(df)
    derived = pd.DataFrame(
        {
            "loan_to_income_ratio": compute_loan_to_income_ratio(df),
            "monthly_payment_to_income_ratio": compute_monthly_payment_to_income_ratio(df),
            "total_liabilities_to_income_ratio": compute_total_liabilities_to_income_ratio(df),
            "avg_previous_loan_amount": compute_avg_previous_loan_amount(df),
            "funded_to_applied_ratio": compute_funded_to_applied_ratio(df),
            "loan_year": loan_year,
            "loan_month": loan_month,
        },
        index=df.index,
    )
    return pd.concat([df, derived], axis=1)
