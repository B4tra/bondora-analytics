"""Target variable and resolution flag definitions based on Gate 1 and Gate 2 decisions."""

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COL,
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    INTEREST_PENALTY_PAYMENTS_COL,
    IS_RESOLVED_COL,
    LOAN_DATE_COL,
    PRINCIPAL_PAYMENTS_COL,
    PRINCIPAL_WRITEOFFS_COL,
    REGRESSION_TARGET_COL,
    REPORT_DATE,
    STATUS_COL,
    STATUS_REPAID,
)


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


def make_default_label(
    df: pd.DataFrame,
    default_date_col: str = DEFAULT_DATE_COL,
    status_col: str = STATUS_COL,
    principal_writeoffs_col: str = PRINCIPAL_WRITEOFFS_COL,
    repaid_status: str = STATUS_REPAID,
) -> pd.Series:
    """Compute nullable binary default classification target (``Int8``).

    Rules:
    1. Right-censored loans (``is_resolved == False``, i.e. ``Status`` in
       ``{'Current', 'Late'}`` with missing ``DefaultDate``) -> ``pd.NA``.
    2. ``DefaultDate`` is non-missing / non-empty -> ``default = 1``.
    3. ``Status == 'Repaid'`` with missing ``DefaultDate`` and
       ``PrincipalWriteOffs > 0`` -> ``default = 1`` (448 loans where 92.41%
       reached >= 61 days late and suffered principal write-off).
    4. ``Status == 'Repaid'`` with missing ``DefaultDate`` and
       ``PrincipalWriteOffs == 0`` (including 297 loans with only interest
       write-offs) -> ``default = 0``.

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
        Nullable integer Series (``Int8``) named ``default`` with values
        ``0``, ``1``, or ``pd.NA`` (for unresolved loans).
    """
    for col in (default_date_col, status_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    s_def = df[default_date_col]
    has_default_date = s_def.notna() & (s_def.astype(str).str.strip() != "")
    is_repaid = df[status_col].astype(str).str.strip() == repaid_status
    is_resolved = has_default_date | is_repaid

    if principal_writeoffs_col in df.columns:
        pw = pd.to_numeric(df[principal_writeoffs_col], errors="coerce").fillna(0.0)
        repaid_with_principal_writeoff = is_repaid & (~has_default_date) & (pw > 0.0)
        is_default = has_default_date | repaid_with_principal_writeoff
    else:
        is_default = has_default_date

    out = pd.Series(pd.NA, index=df.index, dtype="Int8", name=DEFAULT_TARGET_COL)
    out.loc[is_resolved & is_default] = 1
    out.loc[is_resolved & (~is_default)] = 0
    return out


def make_default_horizon_label(
    df: pd.DataFrame,
    months: int = 12,
    report_date: str | pd.Timestamp = REPORT_DATE,
    loan_date_col: str = LOAN_DATE_COL,
    default_date_col: str = DEFAULT_DATE_COL,
    status_col: str = STATUS_COL,
    principal_writeoffs_col: str = PRINCIPAL_WRITEOFFS_COL,
    repaid_status: str = STATUS_REPAID,
) -> tuple[pd.Series, pd.Series]:
    """Compute fixed-horizon observability flag and default label (e.g., 12m or 24m).

    Rules:
    - ``observable_{H}m`` (``bool``): ``LoanDate + DateOffset(months=H) <= report_date``.
    - ``default_{H}m`` (nullable ``Int8``):
      * ``pd.NA`` if not ``observable_{H}m``, OR if ``Status == 'Repaid'`` without
        ``DefaultDate`` but ``PrincipalWriteOffs > 0`` (448 loans whose exact default
        date is unknown), OR if ``DefaultDate < LoanDate`` (anomalous timestamp).
      * ``1`` if ``DefaultDate`` is present and ``LoanDate <= DefaultDate <= LoanDate + DateOffset(months=H)``.
      * ``0`` otherwise (observable loans with no ``DefaultDate`` or with
        ``DefaultDate > LoanDate + DateOffset(months=H)``).

    Parameters
    ----------
    df : pd.DataFrame
        Input loan DataFrame.
    months : int, default 12
        Fixed observation horizon in calendar months.
    report_date : str or pd.Timestamp, default REPORT_DATE
        Dataset snapshot cutoff date.
    loan_date_col : str, default LOAN_DATE_COL
        Column name for loan origination date.
    default_date_col : str, default DEFAULT_DATE_COL
        Column name for default date.
    status_col : str, default STATUS_COL
        Column name for loan status.
    principal_writeoffs_col : str, default PRINCIPAL_WRITEOFFS_COL
        Column name for principal write-offs.
    repaid_status : str, default STATUS_REPAID
        Status string indicating a repaid loan.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(observable_{H}m, default_{H}m)`` where ``observable_{H}m`` is ``bool``
        and ``default_{H}m`` is nullable ``Int8``.
    """
    if months <= 0:
        raise ValueError(f"months must be positive, got {months}.")
    for col in (loan_date_col, default_date_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    loan_dt = pd.to_datetime(df[loan_date_col], errors="coerce", format="mixed")
    def_dt = pd.to_datetime(df[default_date_col], errors="coerce", format="mixed")
    rep_dt = pd.Timestamp(report_date)

    horizon_dt = loan_dt + pd.DateOffset(months=int(months))
    observable = (loan_dt.notna() & (horizon_dt <= rep_dt)).astype(bool)
    observable = observable.rename(f"observable_{months}m")

    has_def_date = def_dt.notna()
    invalid_def_date = has_def_date & loan_dt.notna() & (def_dt < loan_dt)

    if status_col in df.columns and principal_writeoffs_col in df.columns:
        is_repaid = df[status_col].astype(str).str.strip() == repaid_status
        pw = pd.to_numeric(df[principal_writeoffs_col], errors="coerce").fillna(0.0)
        repaid_wo_no_defdate = is_repaid & (~has_def_date) & (pw > 0.0)
    else:
        repaid_wo_no_defdate = pd.Series(False, index=df.index)

    excluded = (~observable) | repaid_wo_no_defdate | invalid_def_date
    default_within_h = has_def_date & (def_dt >= loan_dt) & (def_dt <= horizon_dt)

    default_h = pd.Series(pd.NA, index=df.index, dtype="Int8", name=f"default_{months}m")
    default_h.loc[(~excluded) & default_within_h] = 1
    default_h.loc[(~excluded) & (~default_within_h)] = 0

    return observable, default_h


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
