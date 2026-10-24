"""Unit tests for Tahap 5 prescriptive economics, strategies, optimisation and forecasting (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import forecasting as fc
from src.analysis import prescriptive as pr


# ---------------------------------------------------------------------------
# interest12_frac
# ---------------------------------------------------------------------------


def test_interest12_zero_rate_is_zero() -> None:
    out = pr.interest12_frac([0.0, 0.0], [12, 36])
    assert np.allclose(out, 0.0)


def test_interest12_hand_calculation_12_month_loan() -> None:
    # 12% annual, 12 months: i = 0.01, PMT = 0.01 / (1 - 1.01^-12); interest = 12*PMT - 1 (loan fully repaid)
    i = 0.01
    pmt = i / (1 - (1 + i) ** -12)
    expected = 12 * pmt - 1.0
    got = pr.interest12_frac([12.0], [12])[0]
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.0661853, abs=1e-6)


def test_interest12_short_tenor_uses_n_not_12() -> None:
    i = 0.02
    n = 6
    pmt = i / (1 - (1 + i) ** -n)
    expected = n * pmt - 1.0  # fully repaid after 6 months
    got = pr.interest12_frac([24.0], [n])[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_interest12_long_tenor_is_partial_and_below_total() -> None:
    got = pr.interest12_frac([24.0], [60])[0]
    i = 0.02
    pmt = i / (1 - (1 + i) ** -60)
    total_interest = 60 * pmt - 1.0
    assert 0 < got < total_interest
    # first-year interest on a 60m annuity is close to but below 12 * i * (average outstanding)
    assert got < 0.24


def test_interest12_nan_propagates() -> None:
    out = pr.interest12_frac([np.nan, 10.0, 10.0], [12, np.nan, 0])
    assert np.isnan(out).all()


def test_payment_close_to_monthly_payment_column() -> None:
    amount = np.array([1000.0, 5000.0, 2500.0])
    interest = np.array([20.0, 30.0, 12.0])
    n = np.array([36, 60, 24])
    mp = pr.annuity_payment_frac(interest, n) * amount * np.array([1.001, 0.999, 1.0])
    v = pr.validate_payment_formula(amount, interest, n, mp)
    assert v["korelasi"] > 0.9999
    assert v["median_selisih_relatif_abs"] < 0.01
    assert v["share_dalam_5persen"] == 1.0


def test_implied_rate_round_trip() -> None:
    amount = np.array([1000.0, 4000.0])
    rate = np.array([18.0, 45.0])
    n = np.array([36, 60])
    mp = pr.annuity_payment_frac(rate, n) * amount
    assert np.allclose(pr.implied_annual_rate_pct(amount, mp, n), rate, atol=1e-6)


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


def test_unit_profit_manual_and_limits() -> None:
    f, L = 0.10, 0.60
    p = np.array([0.0, 0.25, 1.0])
    u = pr.unit_profit(p, [f, f, f], L)
    assert u[0] == pytest.approx(f)  # p = 0 -> full interest
    assert u[1] == pytest.approx(0.75 * 0.10 - 0.25 * 0.60)
    assert u[2] == pytest.approx(-L)  # p = 1 -> lose L
    d = np.array([0, 1])
    u_real = pr.unit_profit(d, [f, f], L)
    assert u_real[0] == pytest.approx(f) and u_real[1] == pytest.approx(-L)


def test_portfolio_profit() -> None:
    assert pr.portfolio_profit(50.0, np.array([0.1, -0.2, 0.3])) == pytest.approx(50 * 0.2)


def test_lgd_estimate_and_ci() -> None:
    paid = np.array([0.0, 100.0, 50.0, 0.0])
    amount = np.array([100.0, 100.0, 100.0, 100.0])
    est = pr.estimate_loss_given_default(paid, amount, n_boot=200, seed=1)
    assert est["L"] == pytest.approx(1 - 0.375)
    assert est["ci_low"] <= est["L"] <= est["ci_high"]
    assert est["n"] == 4


def test_wilson_ci_contains_estimate() -> None:
    lo, hi = pr.wilson_ci(30, 100)
    assert lo < 0.30 < hi


# ---------------------------------------------------------------------------
# Strategies and optimisation
# ---------------------------------------------------------------------------


def _toy_opp(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.6, n)
    interest = 10 + 60 * p + rng.normal(0, 2, n)
    opp = pd.DataFrame(
        {
            "p": p,
            "Interest": interest,
            "Country": rng.choice(["EE", "FI", "ES"], n, p=[0.6, 0.25, 0.15]),
            "Rating": rng.choice(["AA", "B", "D", "F", "HR"], n),
            "LoanDuration": rng.choice([24, 36, 60], n),
            "Amount": rng.uniform(500, 5000, n),
            "default_12m": (rng.uniform(size=n) < p).astype(int),
        }
    )
    opp["interest12_frac"] = pr.interest12_frac(opp["Interest"], opp["LoanDuration"])
    return pr.add_unit_profits(opp, 0.6)


LIMITS = {
    "min_loans": 40,
    "max_avg_p": 0.25,
    "max_share_country": 0.6,
    "max_share_worst": 0.3,
    "max_share_tenor60": 0.5,
}


def _optimise(opp: pd.DataFrame, n_select: int = 60, **over: float) -> dict:
    lim = {**LIMITS, **over}
    return pr.optimize_portfolio(
        opp["p"].to_numpy(),
        opp["u_exante"].to_numpy(),
        opp["Country"].to_numpy(),
        opp["Rating"].to_numpy(),
        opp["LoanDuration"].to_numpy(),
        n_select=n_select,
        min_loans=int(lim["min_loans"]),
        max_avg_p=lim["max_avg_p"],
        max_share_country=lim["max_share_country"],
        max_share_worst=lim["max_share_worst"],
        max_share_tenor60=lim["max_share_tenor60"],
        time_limit=30,
    )


def test_optimizer_satisfies_all_constraints() -> None:
    opp = _toy_opp()
    res = _optimise(opp)
    assert res["status"].startswith("Optimal")
    sel = res["selected"]
    assert len(sel) == 60 >= LIMITS["min_loans"]
    comp = pr.portfolio_composition(sel, opp)
    flags = pr.constraint_flags(
        comp,
        min_loans=LIMITS["min_loans"],
        max_avg_p=LIMITS["max_avg_p"],
        max_share_country=LIMITS["max_share_country"],
        max_share_worst=LIMITS["max_share_worst"],
        max_share_tenor60=LIMITS["max_share_tenor60"],
    )
    assert all(flags.values())
    assert res["all_constraints_ok"]
    assert len(set(sel.tolist())) == len(sel)


def test_optimizer_matches_known_solution_small_case() -> None:
    # 4 loans, choose 2, no real constraints: best two u values are known.
    p = np.array([0.1, 0.1, 0.1, 0.1])
    u = np.array([0.05, 0.20, 0.10, 0.15])
    res = pr.optimize_portfolio(
        p,
        u,
        np.array(["EE", "EE", "FI", "FI"]),
        np.array(["A", "A", "B", "B"]),
        np.array([36, 36, 36, 36]),
        n_select=2,
        min_loans=2,
        max_avg_p=0.5,
        max_share_country=1.0,
        max_share_worst=1.0,
        max_share_tenor60=1.0,
    )
    assert sorted(res["selected"].tolist()) == [1, 3]
    assert res["objective"] == pytest.approx(0.35)


def test_optimizer_beats_or_matches_greedy_on_u_while_feasible() -> None:
    opp = _toy_opp()
    res = _optimise(opp)
    greedy = np.argsort(-opp["u_exante"].to_numpy())[:60]
    # greedy ignores constraints, so its u can be higher; optimiser must beat a random feasible-ish choice
    rand = pr.select_random_draws(len(opp), 60, n_draws=1, seed=3)[0]
    assert res["objective"] >= opp["u_exante"].to_numpy()[rand].sum() - 1e-9
    assert res["objective"] <= opp["u_exante"].to_numpy()[greedy].sum() + 1e-9


def test_infeasible_case_detected_with_conflict_report() -> None:
    opp = _toy_opp()
    # Require avg p <= 0.001 although min possible p is ~0.02 -> infeasible
    res = _optimise(opp, max_avg_p=0.001)
    assert res["status"] == "Infeasible"
    assert len(res["selected"]) == 0
    assert any("rata-rata p" in k for k in res["conflicts"])
    assert all(v > 0 for v in res["conflicts"].values())


def test_random_strategy_reproducible() -> None:
    a = pr.select_random_draws(100, 10, n_draws=3, seed=7)
    b = pr.select_random_draws(100, 10, n_draws=3, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = pr.select_random_draws(100, 10, n_draws=3, seed=8)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_s2_selects_lowest_p() -> None:
    p = np.array([0.5, 0.1, 0.3, 0.05, 0.9])
    assert sorted(pr.select_lowest_p(p, 2).tolist()) == [1, 3]


def test_s1_selects_best_ratings() -> None:
    rating = np.array(["HR", "AA", "B", "A", "Tanpa rating"], dtype=object)
    assert sorted(pr.select_by_rating(rating, 2).tolist()) == [1, 3]


def test_constrained_greedy_satisfies_all_constraints_s2c_and_s4c() -> None:
    opp = _toy_opp(400, seed=11)
    opp["pred_return_ratio"] = opp["u_exante"] + 0.05 * np.random.default_rng(11).normal(size=len(opp))
    for score, asc in (
        (opp["p"].to_numpy(), True),
        (opp["pred_return_ratio"].to_numpy(), False),
    ):
        sel = pr.select_constrained_greedy(
            score,
            opp["p"].to_numpy(),
            opp["Country"].to_numpy(),
            opp["Rating"].to_numpy(),
            opp["LoanDuration"].to_numpy(),
            n_select=60,
            min_loans=int(LIMITS["min_loans"]),
            max_avg_p=LIMITS["max_avg_p"],
            max_share_country=LIMITS["max_share_country"],
            max_share_worst=LIMITS["max_share_worst"],
            max_share_tenor60=LIMITS["max_share_tenor60"],
            ascending=asc,
        )
        assert len(sel) == 60
        comp = pr.portfolio_composition(sel, opp)
        flags = pr.constraint_flags(
            comp,
            min_loans=LIMITS["min_loans"],
            max_avg_p=LIMITS["max_avg_p"],
            max_share_country=LIMITS["max_share_country"],
            max_share_worst=LIMITS["max_share_worst"],
            max_share_tenor60=LIMITS["max_share_tenor60"],
        )
        assert all(flags.values())


def test_find_breakeven_multiplier_bisection() -> None:
    p = np.array([0.10, 0.15, 0.20, 0.25])
    f = np.array([0.20, 0.22, 0.25, 0.28])
    L = 0.60
    res = pr.find_breakeven_multiplier(p, f, L)
    assert res["pengali_impas"] > 1.0
    assert abs(res["roi_exante_pada_impas"]) < 1e-9
    p_be = np.clip(res["pengali_impas"] * p, 0.0, 1.0)
    assert pr.unit_profit(p_be, f, L).mean() == pytest.approx(0.0, abs=1e-9)


def test_evaluate_portfolio_matches_manual() -> None:
    opp = _toy_opp(50)
    sel = np.arange(10)
    m = pr.evaluate_portfolio(sel, opp, ticket=50.0, limits={**LIMITS, "min_loans": 5}, n_boot=10)
    assert m["profit_exante_eur"] == pytest.approx(50.0 * opp["u_exante"].iloc[:10].sum())
    assert m["profit_expost_eur"] == pytest.approx(50.0 * opp["u_expost"].iloc[:10].sum())
    assert m["roi_exante"] == pytest.approx(m["profit_exante_eur"] / 500.0)
    assert m["profit_expost_ci_low"] <= m["profit_expost_ci_high"]


# ---------------------------------------------------------------------------
# Breakeven pricing and threshold policy
# ---------------------------------------------------------------------------


def test_breakeven_interest_makes_expected_profit_zero_under_linear_model() -> None:
    p = np.array([0.1, 0.3])
    phi = np.array([0.5, 0.6])
    L = 0.6
    r_star = pr.breakeven_interest_pct(p, L, phi)
    f = phi * r_star / 100.0  # linear approximation
    u = pr.unit_profit(p, f, L)
    assert np.allclose(u, 0.0, atol=1e-12)


def test_breakeven_exact_check_small_error_for_short_loan() -> None:
    # exact f at r*: error should be small relative to the unit profit scale for a moderate rate
    p = np.array([0.1])
    L = 0.6
    n = np.array([36])
    interest = np.array([20.0])
    f0 = pr.interest12_frac(interest, n)
    phi = f0 / (interest / 100)
    r_star = pr.breakeven_interest_pct(p, L, phi)
    u = pr.unit_profit(p, pr.interest12_frac(r_star, n), L)
    assert abs(u[0]) < 0.02


def test_threshold_policy_monotone_approval() -> None:
    opp = _toy_opp()
    tab = pr.threshold_policy_table(opp, [0.1, 0.3, 0.6], 0.6)
    assert tab["n_disetujui"].is_monotonic_increasing
    assert tab["tingkat_persetujuan"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def _synthetic_series(n: int = 60) -> pd.Series:
    idx = pd.date_range("2016-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    vals = 1000 + 5 * np.arange(n) + 80 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 10, n)
    return pd.Series(vals, index=idx)


@pytest.mark.parametrize("model", ["seasonal_naive", "ets"])
def test_forecast_length_and_interval_order(model: str) -> None:
    y = _synthetic_series()
    out, _ = fc.forecast_with_model(y, 4, model)
    assert len(out) == 4
    assert (out["hi95"] >= out["lo95"]).all() and (out["hi80"] >= out["lo80"]).all()
    assert (out["hi95"] >= out["hi80"] - 1e-9).all() and (out["lo95"] <= out["lo80"] + 1e-9).all()
    assert (out["mean"] >= out["lo80"] - 1e-9).all() and (out["mean"] <= out["hi80"] + 1e-9).all()
    assert out.index[0] == y.index[-1] + pd.offsets.MonthBegin(1)


def test_seasonal_naive_repeats_last_year() -> None:
    y = _synthetic_series()
    out = fc.seasonal_naive_forecast(y, 3)
    assert np.allclose(out["mean"].to_numpy(), y.iloc[-12:-9].to_numpy())


def test_interval_coverage_computed_correctly() -> None:
    actual = np.array([1.0, 5.0, 10.0, 20.0])
    lo = np.array([0.0, 6.0, 9.0, 0.0])
    hi = np.array([2.0, 8.0, 11.0, 15.0])
    assert fc.interval_coverage(actual, lo, hi) == pytest.approx(0.5)


def test_mape_smape_known_values() -> None:
    assert fc.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert fc.smape([100, 200], [100, 200]) == pytest.approx(0.0)
    assert fc.smape([100], [200]) == pytest.approx(2 * 100 / 300 * 100)


def test_build_monthly_series_full_months_only() -> None:
    df = pd.DataFrame(
        {
            "LoanDate": pd.to_datetime(["2016-01-05", "2016-01-20", "2016-02-03", "2021-07-10"]),
            "DefaultDate": pd.to_datetime(["2016-03-01", None, None, None]),
            "Amount": [100.0, 200.0, 300.0, 400.0],
        }
    )
    s = fc.build_monthly_series(df, start="2016-01", last_full="2016-03")
    assert list(s.index.strftime("%Y-%m")) == ["2016-01", "2016-02", "2016-03"]
    assert s["n_loans"].tolist() == [2.0, 1.0, 0.0]
    assert s["amount_loans"].tolist() == [300.0, 300.0, 0.0]
    assert s["amount_default"].tolist() == [0.0, 0.0, 100.0]
