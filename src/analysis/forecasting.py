"""Tahap 5 light forecast: monthly loan volume and new-default flow with prediction intervals.

Two candidate models per series: a seasonal naive baseline (lag 12) and an additive-error ETS with damped
additive trend (optionally with additive 12-month seasonality, chosen by AIC). The final model per series is
chosen by rolling-origin (one-step-ahead) backtest on the last months. Intervals for the seasonal naive model are
Gaussian with the std of the in-sample lag-12 differences, widened by ``sqrt(ceil(h / 12))``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BACKTEST_MONTHS,
    FORECAST_HORIZON,
    FORECAST_LAST_FULL_MONTH,
    FORECAST_START,
)

LEVELS: tuple[float, float] = (0.80, 0.95)

SERIES_LABELS: dict[str, str] = {
    "n_loans": "Jumlah pinjaman baru per bulan",
    "amount_loans": "Total Amount pinjaman baru per bulan (EUR)",
    "amount_default": "Arus default baru: total Amount menurut bulan DefaultDate (EUR)",
}


def build_monthly_series(
    df: pd.DataFrame,
    start: str = FORECAST_START,
    last_full: str = FORECAST_LAST_FULL_MONTH,
) -> pd.DataFrame:
    """Monthly series on full months only (``start`` .. ``last_full``).

    Columns: ``n_loans`` and ``amount_loans`` by ``LoanDate`` month, ``amount_default`` = sum of ``Amount`` of
    loans whose ``DefaultDate`` falls in the month (new default flow). Missing months are filled with 0.
    """
    idx = pd.date_range(start=start, end=last_full, freq="MS")
    loans = df[["LoanDate", "Amount"]].dropna(subset=["LoanDate"]).copy()
    loans["month"] = loans["LoanDate"].dt.to_period("M").dt.to_timestamp()
    by_loan = loans.groupby("month")["Amount"].agg(n_loans="size", amount_loans="sum")

    dflt = df.loc[df["DefaultDate"].notna(), ["DefaultDate", "Amount"]].copy()
    dflt["month"] = dflt["DefaultDate"].dt.to_period("M").dt.to_timestamp()
    by_default = dflt.groupby("month")["Amount"].sum().rename("amount_default")

    out = pd.concat([by_loan, by_default], axis=1, sort=True).reindex(idx).fillna(0.0)
    out.index.name = "month"
    out["n_loans"] = out["n_loans"].astype("float64")
    return out


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def _z(level: float) -> float:
    return float(stats.norm.ppf(0.5 + level / 2.0))


def seasonal_naive_forecast(y: pd.Series, horizon: int, season: int = 12) -> pd.DataFrame:
    """Seasonal naive point forecast with Gaussian intervals from in-sample lag-``season`` differences."""
    vals = y.to_numpy(dtype="float64")
    if len(vals) < season + 2:
        raise ValueError("Series too short for seasonal naive.")
    diffs = vals[season:] - vals[:-season]
    sigma = float(np.std(diffs, ddof=1))
    steps = np.arange(1, horizon + 1)
    point = np.array([vals[len(vals) - season + ((k - 1) % season)] for k in steps])
    widen = np.sqrt(np.ceil(steps / season))
    out = {"mean": point}
    for lev in LEVELS:
        half = _z(lev) * sigma * widen
        out[f"lo{int(lev * 100)}"] = np.maximum(point - half, 0.0)
        out[f"hi{int(lev * 100)}"] = point + half
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.DataFrame(out, index=idx)


def ets_forecast(
    y: pd.Series, horizon: int, seasonal: bool | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Additive-error ETS with damped additive trend; seasonality chosen by AIC when ``seasonal is None``.

    Returns the forecast frame (mean, 80% and 95% intervals) and fit info (seasonal flag, AIC values).
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = y.astype("float64").copy()
    y.index = pd.DatetimeIndex(y.index, freq="MS")

    def _fit(use_seasonal: bool):
        kwargs: dict[str, Any] = {
            "error": "add",
            "trend": "add",
            "damped_trend": True,
            "seasonal": "add" if use_seasonal else None,
        }
        if use_seasonal:
            kwargs["seasonal_periods"] = 12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ETSModel(y, **kwargs).fit(disp=False, maxiter=2000)

    fits: dict[str, Any] = {}
    aic: dict[str, float] = {}
    candidates = [False] if len(y) < 36 else [False, True]
    if seasonal is not None:
        candidates = [bool(seasonal)]
    for flag in candidates:
        try:
            res = _fit(flag)
            fits[str(flag)] = res
            aic[str(flag)] = float(res.aic)
        except Exception:  # noqa: BLE001 - a failed candidate is simply skipped
            continue
    if not fits:
        raise RuntimeError("ETS fit failed for all candidates.")
    best_key = min(aic, key=aic.get)
    res = fits[best_key]
    start = len(y)
    pred = res.get_prediction(start=start, end=start + horizon - 1)
    out = {}
    for lev in LEVELS:
        sf = pred.summary_frame(alpha=1.0 - lev)
        out["mean"] = sf["mean"].to_numpy()
        out[f"lo{int(lev * 100)}"] = np.maximum(sf["pi_lower"].to_numpy(), 0.0)
        out[f"hi{int(lev * 100)}"] = sf["pi_upper"].to_numpy()
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    frame = pd.DataFrame(out, index=idx)
    frame["mean"] = np.maximum(frame["mean"], 0.0)
    return frame, {"seasonal": best_key == "True", "aic": aic}


def forecast_with_model(y: pd.Series, horizon: int, model: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch to ``seasonal_naive`` or ``ets``."""
    if model == "seasonal_naive":
        return seasonal_naive_forecast(y, horizon), {}
    if model == "ets":
        return ets_forecast(y, horizon)
    raise ValueError(f"Unknown model: {model}")


# --------------------------------------------------------------------------------------
# Metrics and backtest
# --------------------------------------------------------------------------------------


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute percentage error (percent); zero actuals are excluded."""
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(pred, dtype="float64")
    m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100.0) if m.any() else float("nan")


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE (percent), ``mean(2|a-f| / (|a|+|f|))``; pairs with both zero are excluded."""
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(pred, dtype="float64")
    denom = np.abs(a) + np.abs(f)
    m = denom != 0
    return float(np.mean(2.0 * np.abs(a[m] - f[m]) / denom[m]) * 100.0) if m.any() else float("nan")


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actual values that fall inside ``[lower, upper]`` (inclusive)."""
    a = np.asarray(actual, dtype="float64")
    lo = np.asarray(lower, dtype="float64")
    hi = np.asarray(upper, dtype="float64")
    return float(np.mean((a >= lo) & (a <= hi)))


def rolling_origin_backtest(
    y: pd.Series, model: str, n_test: int = BACKTEST_MONTHS
) -> pd.DataFrame:
    """One-step-ahead rolling-origin evaluation on the last ``n_test`` months."""
    rows = []
    for k in range(n_test, 0, -1):
        train = y.iloc[: len(y) - k]
        actual = float(y.iloc[len(y) - k])
        fc, _ = forecast_with_model(train, 1, model)
        r = fc.iloc[0]
        rows.append(
            {
                "month": y.index[len(y) - k],
                "actual": actual,
                "pred": float(r["mean"]),
                "lo80": float(r["lo80"]),
                "hi80": float(r["hi80"]),
                "lo95": float(r["lo95"]),
                "hi95": float(r["hi95"]),
            }
        )
    return pd.DataFrame(rows)


def run_forecasts(
    series_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    n_test: int = BACKTEST_MONTHS,
) -> dict[str, pd.DataFrame]:
    """Backtest both models per series, choose by MAPE, then forecast ``horizon`` months ahead.

    Returns ``{"backtest": ..., "backtest_detail": ..., "forecast": ...}``.
    """
    bt_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    fc_rows: list[pd.DataFrame] = []
    for col in series_df.columns:
        y = series_df[col]
        scores: dict[str, float] = {}
        for model in ("seasonal_naive", "ets"):
            det = rolling_origin_backtest(y, model, n_test)
            m_mape = mape(det["actual"], det["pred"])
            scores[model] = m_mape
            _, info = ets_forecast(y.iloc[: len(y) - n_test], 1) if model == "ets" else (None, {})
            bt_rows.append(
                {
                    "serie": col,
                    "model": model,
                    "n_uji": n_test,
                    "mape_pct": m_mape,
                    "smape_pct": smape(det["actual"], det["pred"]),
                    "cakupan_80": interval_coverage(det["actual"], det["lo80"], det["hi80"]),
                    "cakupan_95": interval_coverage(det["actual"], det["lo95"], det["hi95"]),
                    "ets_musiman_aic": info.get("seasonal", np.nan) if model == "ets" else np.nan,
                }
            )
            det.insert(0, "model", model)
            det.insert(0, "serie", col)
            detail_rows.append(det)
        chosen = min(scores, key=lambda k: (np.nan_to_num(scores[k], nan=np.inf), k))
        for r in bt_rows:
            if r["serie"] == col:
                r["dipilih"] = r["model"] == chosen
        fc, info = forecast_with_model(y, horizon, chosen)
        fc = fc.reset_index().rename(columns={"index": "month"})
        fc.insert(0, "model", chosen)
        fc.insert(0, "serie", col)
        fc["ets_musiman_aic"] = info.get("seasonal", np.nan) if chosen == "ets" else np.nan
        fc_rows.append(fc)
    return {
        "backtest": pd.DataFrame(bt_rows),
        "backtest_detail": pd.concat(detail_rows, ignore_index=True),
        "forecast": pd.concat(fc_rows, ignore_index=True),
    }
