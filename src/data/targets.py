"""Target variable and resolution flag definitions based on Gate 1 and Gate 2 decisions."""

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COL,
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    INTEREST_PENALTY_PAYMENTS_COL,
    IS_RESOLVED_COL,
    PRINCIPAL_PAYMENTS_COL,
    PRINCIPAL_WRITEOFFS_COL,
    REGRESSION_TARGET_COL,
    STATUS_COL,
    STATUS_REPAID,
)


def make_default_label(
    df: pd.DataFrame,
    default_date_col: str = DEFAULT_DATE_COL,
    status_col: str = STATUS_COL,
    principal_writeoffs_col: str = PRINCIPAL_WRITEOFFS_COL,
    repaid_status: str = STATUS_REPAID,
) -> pd.Series:
    """Compute binary default classification target (Option 2B + Gate 2 refinement).

    Rules:
    1. ``DefaultDate`` is non-missing / non-empty -> ``default = 1``.
    2. ``Status == 'Repaid'`` with missing ``DefaultDate`` but
       ``PrincipalWriteOffs > 0`` -> ``default = 1`` (448 loans where 92.41%
       reached >= 61 days late and suffered principal write-off).
    3. All other rows (including ``Repaid`` with only interest write-offs) -> ``default = 0``.

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame.
    default_date_col : str, default DEFAULT_DATE_COL
        Column name recording the date of default.
    status_col : str, default STATUS_COL
        Column name recording loan status.
    principal_writeoffs_col : str, default PRINCIPAL_WRITEOFFS_COL
        Column name recording principal write-offs.
    repaid_status : str, default STATUS_REPAID
        Status string indicating a repaid loan.

    Returns
    -------
    pd.Series
        Integer Series (``int64``) named ``default`` with values ``0`` or ``1``.
    """
    if default_date_col not in df.columns:
        raise KeyError(f"Column '{default_date_col}' not found in DataFrame.")

    s_def = df[default_date_col]
    has_default_date = s_def.notna() & (s_def.astype(str).str.strip() != "")

    if status_col in df.columns and principal_writeoffs_col in df.columns:
        is_repaid = df[status_col].astype(str).str.strip() == repaid_status
        pw = pd.to_numeric(df[principal_writeoffs_col], errors="coerce").fillna(0.0)
        repaid_with_principal_writeoff = is_repaid & (~has_default_date) & (pw > 0.0)
        is_default = has_default_date | repaid_with_principal_writeoff
    else:
        is_default = has_default_date

    return is_default.astype("int64").rename(DEFAULT_TARGET_COL)


def make_is_resolved(
    df: pd.DataFrame,
    status_col: str = STATUS_COL,
    default_date_col: str = DEFAULT_DATE_COL,
    repaid_status: str = STATUS_REPAID,
) -> pd.Series:
    """Compute whether a loan's final default outcome is resolved (Option 3A).

    Rules:
    - ``DefaultDate`` is present (any status) -> ``is_resolved = True``
    - ``Status == 'Repaid'`` and ``DefaultDate`` is missing -> ``is_resolved = True``
    - ``Status == 'Current'`` and ``DefaultDate`` is missing -> ``is_resolved = False``
    - ``Status == 'Late'`` and ``DefaultDate`` is missing -> ``is_resolved = False``

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame.
    status_col : str, default STATUS_COL
        Column name recording loan status.
    default_date_col : str, default DEFAULT_DATE_COL
        Column name recording the date of default.
    repaid_status : str, default STATUS_REPAID
        Status string indicating a fully repaid loan.

    Returns
    -------
    pd.Series
        Boolean Series (``bool``) named ``is_resolved``.
    """
    for col in (status_col, default_date_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    s_def = df[default_date_col]
    has_default_date = s_def.notna() & (s_def.astype(str).str.strip() != "")
    is_repaid = df[status_col].astype(str).str.strip() == repaid_status
    resolved = (has_default_date | is_repaid).astype(bool)
    return resolved.rename(IS_RESOLVED_COL)


def make_regression_target(
    df: pd.DataFrame,
    principal_paid_col: str = PRINCIPAL_PAYMENTS_COL,
    interest_penalty_paid_col: str = INTEREST_PENALTY_PAYMENTS_COL,
    amount_col: str = AMOUNT_COL,
) -> pd.Series:
    """Compute cumulative return ratio up to the report date (Option 4A).

    Formula:
        ``return_ratio = (PrincipalPaymentsMade + InterestAndPenaltyPaymentsMade) / Amount``

    Note: This represents realized cumulative return up to ``REPORT_DATE``
    (a lower bound for defaulted loans that are still in active recovery).
    Division by ``Amount <= 0`` or ``NaN`` safely yields ``NaN`` (never ``inf``).

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame.
    principal_paid_col : str, default PRINCIPAL_PAYMENTS_COL
        Column name for principal payments made.
    interest_penalty_paid_col : str, default INTEREST_PENALTY_PAYMENTS_COL
        Column name for interest and penalty payments made.
    amount_col : str, default AMOUNT_COL
        Column name for disbursed loan amount.

    Returns
    -------
    pd.Series
        Float Series (``float64``) named ``return_ratio``.
    """
    for col in (principal_paid_col, interest_penalty_paid_col, amount_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    principal_paid = pd.to_numeric(df[principal_paid_col], errors="coerce")
    interest_paid = pd.to_numeric(df[interest_penalty_paid_col], errors="coerce")
    amount = pd.to_numeric(df[amount_col], errors="coerce")

    safe_amount = amount.where(amount > 0, np.nan)
    total_paid = principal_paid + interest_paid
    ratio = (total_paid / safe_amount).astype("float64")
    return ratio.rename(REGRESSION_TARGET_COL)
