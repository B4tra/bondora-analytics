"""Tahap 5 prescriptive analytics: unit economics, investor strategies, constrained optimization.

The decision framework follows the course slides: prediction (+ uncertainty) combined with business rules
and constraints yields a recommended action, supported by what-if simulation.

Conventions
-----------
* ``p``  : calibrated 12-month default probability from Tahap 4 (``p_calibrated``).
* ``f``  : 12-month interest income as a fraction of principal if the loan performs (``interest12_frac``).
* ``L``  : loss given default as a fraction of principal (constant, estimated on train+val only).
* ``u``  : expected 12-month profit per EUR invested, ``(1 - p) * f - p * L`` (ex-ante) or with the actual
  default flag ``d`` in place of ``p`` (ex-post).
* Ex-ante numbers are model-based (what the decision maker sees); ex-post numbers use realised outcomes and
  are for evaluation only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from src.config import (
    N_BOOTSTRAP,
    N_RANDOM_DRAWS,
    OPT_TIME_LIMIT_S,
    RANDOM_SEED,
    RATING_ORDER,
    WORST_RATINGS,
)

# --------------------------------------------------------------------------------------
# Unit economics
# --------------------------------------------------------------------------------------


def interest12_frac(
    interest_pct: Sequence[float] | np.ndarray | pd.Series,
    n_months: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = 12,
) -> np.ndarray:
    """Interest paid within ``horizon`` months as a fraction of principal for an annuity loan.

    With monthly rate ``i = Interest / 100 / 12``, tenor ``n`` and principal 1, the instalment is
    ``PMT = i / (1 - (1 + i)^-n)`` (``1 / n`` if ``i == 0``). Over ``m = min(horizon, n)`` months the borrower
    pays ``m * PMT`` of which ``1 - balance_m`` is principal, hence interest = ``m * PMT - (1 - balance_m)``
    with ``balance_m = (1 + i)^m - PMT * ((1 + i)^m - 1) / i``. Fully vectorised; NaN in -> NaN out.
    """
    r = np.asarray(interest_pct, dtype="float64")
    n = np.asarray(n_months, dtype="float64")
    valid = np.isfinite(r) & np.isfinite(n) & (n > 0)
    i = np.where(valid, r / 100.0 / 12.0, np.nan)
    n_safe = np.where(valid, n, np.nan)
    m = np.minimum(float(horizon), n_safe)
    zero = valid & (i == 0.0)
    i_safe = np.where(zero, 1.0, i)  # placeholder to avoid 0-division; overwritten below
    pmt = np.where(zero, 1.0 / n_safe, i_safe / (1.0 - (1.0 + i_safe) ** (-n_safe)))
    growth = (1.0 + i_safe) ** m
    balance = np.where(zero, 1.0 - pmt * m, growth - pmt * (growth - 1.0) / i_safe)
    interest = m * pmt - (1.0 - balance)
    interest = np.where(zero, 0.0, interest)
    return np.where(valid, interest, np.nan)


def annuity_payment_frac(
    interest_pct: Sequence[float] | np.ndarray | pd.Series,
    n_months: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Monthly instalment as a fraction of principal for an annuity loan (``PMT`` for principal 1)."""
    r = np.asarray(interest_pct, dtype="float64")
    n = np.asarray(n_months, dtype="float64")
    valid = np.isfinite(r) & np.isfinite(n) & (n > 0)
    i = np.where(valid, r / 100.0 / 12.0, np.nan)
    n_safe = np.where(valid, n, np.nan)
    zero = valid & (i == 0.0)
    i_safe = np.where(zero, 1.0, i)
    pmt = np.where(zero, 1.0 / n_safe, i_safe / (1.0 - (1.0 + i_safe) ** (-n_safe)))
    return np.where(valid, pmt, np.nan)


def validate_payment_formula(
    amount: Sequence[float] | pd.Series,
    interest_pct: Sequence[float] | pd.Series,
    n_months: Sequence[float] | pd.Series,
    monthly_payment: Sequence[float] | pd.Series,
) -> dict[str, float]:
    """Compare computed ``PMT * Amount`` with the reported ``MonthlyPayment`` column."""
    amt = np.asarray(amount, dtype="float64")
    mp = np.asarray(monthly_payment, dtype="float64")
    calc = annuity_payment_frac(interest_pct, n_months) * amt
    ok = np.isfinite(calc) & np.isfinite(mp) & (mp > 0) & (amt > 0)
    rel = (calc[ok] - mp[ok]) / mp[ok]
    return {
        "n_dibandingkan": int(ok.sum()),
        "korelasi": float(np.corrcoef(calc[ok], mp[ok])[0, 1]),
        "median_selisih_relatif": float(np.median(rel)),
        "median_selisih_relatif_abs": float(np.median(np.abs(rel))),
        "p95_selisih_relatif_abs": float(np.percentile(np.abs(rel), 95)),
        "share_dalam_5persen": float(np.mean(np.abs(rel) <= 0.05)),
    }


def implied_annual_rate_pct(
    amount: Sequence[float] | pd.Series,
    monthly_payment: Sequence[float] | pd.Series,
    n_months: Sequence[float] | pd.Series,
    upper: float = 300.0,
) -> np.ndarray:
    """Annual nominal rate (percent) that reproduces ``MonthlyPayment`` under the annuity formula (bisection)."""
    target = np.asarray(monthly_payment, dtype="float64") / np.asarray(amount, dtype="float64")
    n = np.asarray(n_months, dtype="float64")
    lo = np.zeros(len(target))
    hi = np.full(len(target), float(upper))
    for _ in range(60):
        mid = (lo + hi) / 2.0
        below = annuity_payment_frac(mid, n) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = (lo + hi) / 2.0
    return np.where(np.isfinite(target) & (target > 0), out, np.nan)


def estimate_loss_given_default(
    principal_paid: Sequence[float] | pd.Series,
    amount: Sequence[float] | pd.Series,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Estimate ``L = 1 - mean(PrincipalPaymentsMade / Amount)`` on defaulted loans with a bootstrap CI.

    Callers must pass only train+validation loans with ``default_12m == 1``. The ratio is clipped to [0, 1].
    The estimate is conservative (collections on freshly defaulted loans are still in progress) and ignores
    interest received before default.
    """
    ratio = np.clip(
        np.asarray(principal_paid, dtype="float64") / np.asarray(amount, dtype="float64"), 0.0, 1.0
    )
    ratio = ratio[np.isfinite(ratio)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratio), size=(n_boot, len(ratio)))
    boot = 1.0 - ratio[idx].mean(axis=1)
    return {
        "L": float(1.0 - ratio.mean()),
        "n": int(len(ratio)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "recovery_mean": float(ratio.mean()),
        "recovery_median": float(np.median(ratio)),
        "recovery_p10": float(np.percentile(ratio, 10)),
        "recovery_p90": float(np.percentile(ratio, 90)),
        "share_zero_recovery": float(np.mean(ratio == 0.0)),
    }


def unit_profit(
    prob_or_flag: Sequence[float] | np.ndarray | pd.Series,
    interest12: Sequence[float] | np.ndarray | pd.Series,
    loss_given_default: float,
) -> np.ndarray:
    """12-month profit per EUR invested: ``(1 - q) * f - q * L`` with ``q`` = p (ex-ante) or d (ex-post)."""
    q = np.asarray(prob_or_flag, dtype="float64")
    f = np.asarray(interest12, dtype="float64")
    return (1.0 - q) * f - q * float(loss_given_default)


def portfolio_profit(ticket: float | np.ndarray, u: np.ndarray) -> float:
    """Portfolio profit ``sum(ticket_i * u_i)``."""
    return float(np.sum(np.asarray(ticket, dtype="float64") * np.asarray(u, dtype="float64")))


def wilson_ci(successes: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = successes / n
    denom = 1.0 + z**2 / n
    centre = (ph + z**2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


# --------------------------------------------------------------------------------------
# Constraint bookkeeping
# --------------------------------------------------------------------------------------


def _masks(
    country: np.ndarray, rating: np.ndarray, duration: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    countries = {str(c): (country == c).astype(float) for c in pd.unique(country)}
    worst = np.isin(rating, list(WORST_RATINGS)).astype(float)
    ten60 = (duration == 60).astype(float)
    return countries, worst, ten60


def portfolio_composition(
    sel: np.ndarray,
    opp: pd.DataFrame,
) -> dict[str, float]:
    """Exposure shares (equal tickets => shares by count) used by the constraints."""
    n = len(sel)
    sub = opp.iloc[sel]
    country_share = sub["Country"].astype(str).value_counts(normalize=True)
    return {
        "n": float(n),
        "p_mean": float(sub["p"].mean()),
        "share_country_max": float(country_share.max()),
        "country_max_name": str(country_share.idxmax()),
        "share_rating_FHR": float(sub["Rating"].isin(list(WORST_RATINGS)).mean()),
        "share_tenor60": float((sub["LoanDuration"] == 60).mean()),
    }


def constraint_flags(
    comp: dict[str, float],
    *,
    min_loans: int,
    max_avg_p: float,
    max_share_country: float,
    max_share_worst: float,
    max_share_tenor60: float,
    tol: float = 1e-9,
) -> dict[str, bool]:
    """Evaluate each business constraint on a portfolio composition."""
    return {
        "ok_min_loans": bool(comp["n"] >= min_loans - tol),
        "ok_avg_p": bool(comp["p_mean"] <= max_avg_p + tol),
        "ok_country": bool(comp["share_country_max"] <= max_share_country + tol),
        "ok_rating_FHR": bool(comp["share_rating_FHR"] <= max_share_worst + tol),
        "ok_tenor60": bool(comp["share_tenor60"] <= max_share_tenor60 + tol),
    }


# --------------------------------------------------------------------------------------
# Optimisation (S3)
# --------------------------------------------------------------------------------------


def _build_constraints(
    p: np.ndarray,
    country: np.ndarray,
    rating: np.ndarray,
    duration: np.ndarray,
    *,
    n_select: int,
    min_loans: int,
    max_avg_p: float,
    max_share_country: float,
    max_share_worst: float,
    max_share_tenor60: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (A, lb, ub, names) for the linear constraints ``lb <= A x <= ub``."""
    countries, worst, ten60 = _masks(country, rating, duration)
    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []
    names: list[str] = []
    n = len(p)

    rows.append(np.ones(n))
    lbs.append(float(n_select))
    ubs.append(float(n_select))
    names.append("anggaran: jumlah pinjaman = N")

    rows.append(np.ones(n))
    lbs.append(float(min_loans))
    ubs.append(np.inf)
    names.append("jumlah pinjaman >= MIN_LOANS")

    rows.append(p - max_avg_p)
    lbs.append(-np.inf)
    ubs.append(0.0)
    names.append("rata-rata p <= MAX_AVG_P")

    for cname, mask in countries.items():
        rows.append(mask)
        lbs.append(-np.inf)
        ubs.append(max_share_country * n_select)
        names.append(f"eksposur negara {cname} <= MAX_SHARE_COUNTRY")

    rows.append(worst)
    lbs.append(-np.inf)
    ubs.append(max_share_worst * n_select)
    names.append("eksposur rating F+HR <= MAX_SHARE_WORST_RATINGS")

    rows.append(ten60)
    lbs.append(-np.inf)
    ubs.append(max_share_tenor60 * n_select)
    names.append("eksposur tenor 60 bulan <= MAX_SHARE_TENOR60")

    return np.vstack(rows), np.asarray(lbs), np.asarray(ubs), names


def _constraint_report(
    x: np.ndarray, a: np.ndarray, lb: np.ndarray, ub: np.ndarray, names: list[str], n_select: int
) -> pd.DataFrame:
    """Report LHS, bounds, slack (normalised by N) and binding status for a solution ``x``."""
    lhs = a @ x
    rows = []
    for k, nm in enumerate(names):
        slack_lo = lhs[k] - lb[k] if np.isfinite(lb[k]) else np.inf
        slack_hi = ub[k] - lhs[k] if np.isfinite(ub[k]) else np.inf
        slack = float(min(slack_lo, slack_hi))
        rows.append(
            {
                "kendala": nm,
                "lhs": float(lhs[k]),
                "batas_bawah": float(lb[k]) if np.isfinite(lb[k]) else np.nan,
                "batas_atas": float(ub[k]) if np.isfinite(ub[k]) else np.nan,
                "slack_normal": slack / n_select,
                "mengikat": bool(slack / n_select <= 1e-3),
                "terpenuhi": bool(slack >= -1e-6),
            }
        )
    return pd.DataFrame(rows)


def diagnose_infeasibility(
    a: np.ndarray, lb: np.ndarray, ub: np.ndarray, names: list[str], n_select: int
) -> dict[str, float]:
    """Minimum relaxation (elastic LP) that makes the constraint set feasible.

    Slacks are expressed in units of ``N`` (share points / probability points) so that rows are comparable.
    Returns ``{constraint_name: required_relaxation}`` for the constraints that must be relaxed.
    """
    n_var = a.shape[1]
    n_con = a.shape[0]
    # variables: x (n_var), s_lo (n_con), s_hi (n_con); minimise sum(s) / N
    c = np.concatenate([np.zeros(n_var), np.ones(2 * n_con) / n_select])
    rows_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for k in range(n_con):
        if np.isfinite(ub[k]):
            r = np.zeros(n_var + 2 * n_con)
            r[:n_var] = a[k]
            r[n_var + n_con + k] = -1.0
            rows_ub.append(r)
            b_ub.append(ub[k])
        if np.isfinite(lb[k]):
            r = np.zeros(n_var + 2 * n_con)
            r[:n_var] = -a[k]
            r[n_var + k] = -1.0
            rows_ub.append(r)
            b_ub.append(-lb[k])
    bounds = [(0.0, 1.0)] * n_var + [(0.0, None)] * (2 * n_con)
    res = linprog(c, A_ub=np.vstack(rows_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    out: dict[str, float] = {}
    if res.status != 0:
        return {"(diagnosis gagal)": float("nan")}
    s = res.x[n_var:]
    for k, nm in enumerate(names):
        need = float(s[k] + s[n_con + k]) / n_select
        if need > 1e-9:
            out[nm] = need
    return out


def optimize_portfolio(
    p: np.ndarray,
    u: np.ndarray,
    country: np.ndarray,
    rating: np.ndarray,
    duration: np.ndarray,
    *,
    n_select: int,
    min_loans: int,
    max_avg_p: float,
    max_share_country: float,
    max_share_worst: float,
    max_share_tenor60: float,
    pool_size: int | None = None,
    time_limit: float = OPT_TIME_LIMIT_S,
    mip_rel_gap: float = 1e-4,
) -> dict[str, Any]:
    """Maximise ``sum(u_i x_i)`` over binary ``x`` subject to the business constraints (S3).

    Solver: HiGHS through SciPy (``linprog`` / ``milp``) because the installed ``pulp`` 4.x no longer bundles
    CBC. The LP relaxation is solved first; if its optimum is integral it is also optimal for the binary
    problem. Otherwise variables already at 0/1 are fixed and the small remaining MILP over the fractional
    variables is solved (the gap to the LP bound is reported). A direct binary solve of the full 54k-variable
    problem was avoided because it exceeded the 3-minute limit. ``pool_size`` optionally restricts candidates
    to the top-``pool_size`` loans by ``u`` (a heuristic; ``None`` = use all loans).

    Returns a dict with ``selected`` (positional indices into the inputs), ``status``, ``objective``
    (sum of u over selected loans), ``constraint_report`` and, if infeasible, ``conflicts``.
    """
    p = np.asarray(p, dtype="float64")
    u = np.asarray(u, dtype="float64")
    country = np.asarray(country)
    rating = np.asarray(rating)
    duration = np.asarray(duration)
    n_all = len(p)

    if pool_size is not None and pool_size < n_all:
        cand = np.argsort(-u, kind="stable")[:pool_size]
        cand.sort()
    else:
        cand = np.arange(n_all)

    a, lb, ub, names = _build_constraints(
        p[cand],
        country[cand],
        rating[cand],
        duration[cand],
        n_select=n_select,
        min_loans=min_loans,
        max_avg_p=max_avg_p,
        max_share_country=max_share_country,
        max_share_worst=max_share_worst,
        max_share_tenor60=max_share_tenor60,
    )
    n_var = len(cand)
    result: dict[str, Any] = {
        "n_variables": int(n_var),
        "pool_size": None if pool_size is None else int(pool_size),
        "n_fractional": 0,
    }

    lp_obj, x_lp, lp_status = _solve_lp_relaxation(u[cand], a, lb, ub)
    x: np.ndarray | None = None
    if lp_status == 2:
        result["status"] = "Infeasible"
    elif lp_status != 0 or x_lp is None:
        result["status"] = f"Solver gagal (status LP {lp_status})"
    else:
        tol = 1e-7
        frac = (x_lp > tol) & (x_lp < 1.0 - tol)
        result["n_fractional"] = int(frac.sum())
        result["lp_bound"] = float(lp_obj)
        if not frac.any():
            # An integral LP optimum is an optimal binary solution (LP value is an upper bound on the MILP).
            x = np.round(x_lp)
            result["status"] = "Optimal (relaksasi LP bersifat integral)"
        else:
            # Fix variables already at 0/1 and solve the small MILP over the fractional ones (HiGHS).
            x = np.round(x_lp)
            idx_free = np.where(frac)[0]
            x[idx_free] = 0.0
            fixed_lhs = a @ x
            sub = milp(
                c=-u[cand][idx_free],
                constraints=LinearConstraint(
                    sparse.csr_matrix(a[:, idx_free]), lb - fixed_lhs, ub - fixed_lhs
                ),
                integrality=np.ones(len(idx_free)),
                bounds=Bounds(0.0, 1.0),
                options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap), "disp": False},
            )
            if sub.x is not None and sub.status in (0, 1):
                x[idx_free] = np.round(sub.x)
                gap = (lp_obj - float(u[cand] @ x)) / max(abs(lp_obj), 1e-12)
                result["status"] = (
                    f"Optimal (LP + MILP pada {len(idx_free)} variabel pecahan; gap ke batas LP {gap:.2e})"
                    if sub.status == 0
                    else f"Feasible (LP + MILP batas waktu; gap ke batas LP {gap:.2e})"
                )
            else:
                x = None
                result["status"] = "LP relaksasi layak, MILP sisa gagal"

    if x is None:
        result["selected"] = np.array([], dtype=int)
        result["objective"] = float("nan")
        result["constraint_report"] = pd.DataFrame()
        if result["status"] == "Infeasible":
            result["conflicts"] = diagnose_infeasibility(a, lb, ub, names, n_select)
        return result

    sel = cand[x > 0.5]
    result["selected"] = sel
    result["objective"] = float(u[sel].sum())
    report = _constraint_report(x, a, lb, ub, names, n_select)
    result["constraint_report"] = report
    # Safety net: verify constraints on the returned solution.
    result["all_constraints_ok"] = bool(report["terpenuhi"].all())
    return result


def _solve_lp_relaxation(
    u: np.ndarray, a: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[float, np.ndarray | None, int]:
    """Solve ``max u'x`` s.t. ``lb <= Ax <= ub``, ``0 <= x <= 1`` with HiGHS; returns (objective, x, status)."""
    finite_ub = np.isfinite(ub)
    finite_lb = np.isfinite(lb)
    eq = finite_ub & finite_lb & (ub == lb)
    ineq_ub = finite_ub & ~eq
    ineq_lb = finite_lb & ~eq
    a_ub = np.vstack([a[ineq_ub], -a[ineq_lb]])
    b_ub = np.concatenate([ub[ineq_ub], -lb[ineq_lb]])
    lp = linprog(
        -u,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a[eq] if eq.any() else None,
        b_eq=ub[eq] if eq.any() else None,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if lp.status != 0:
        return float("nan"), None, int(lp.status)
    return float(-lp.fun), lp.x, 0



# --------------------------------------------------------------------------------------
# Strategies S0-S5
# --------------------------------------------------------------------------------------


def rating_rank(rating: np.ndarray | pd.Series) -> np.ndarray:
    """Numeric rank of ratings (AA=0 ... HR=7); missing/unknown ratings rank last."""
    order = {r: k for k, r in enumerate(RATING_ORDER)}
    return np.array([order.get(r, len(RATING_ORDER)) for r in np.asarray(rating, dtype=object)])


def _rank_select(score: np.ndarray, n_select: int, rng: np.random.Generator, ascending: bool) -> np.ndarray:
    """Top-``n_select`` by score with random tie-breaking."""
    tie = rng.random(len(score))
    key = score if ascending else -score
    order = np.lexsort((tie, key))
    return np.sort(order[:n_select])


def select_random_draws(
    n_total: int, n_select: int, n_draws: int = N_RANDOM_DRAWS, seed: int = RANDOM_SEED
) -> list[np.ndarray]:
    """S0: ``n_draws`` independent random portfolios of ``n_select`` loans."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_total, size=n_select, replace=False)) for _ in range(n_draws)]


def select_by_rating(rating: np.ndarray, n_select: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """S1: best Bondora ratings first (AA -> HR), random tie-breaking."""
    rng = np.random.default_rng(seed)
    return _rank_select(rating_rank(rating).astype(float), n_select, rng, ascending=True)


def select_lowest_p(p: np.ndarray, n_select: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """S2: lowest calibrated default probability, no concentration limits."""
    rng = np.random.default_rng(seed)
    return _rank_select(np.asarray(p, dtype="float64"), n_select, rng, ascending=True)


def select_top_score(score: np.ndarray, n_select: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """S4 / S5: highest score (predicted return ratio or interest rate), random tie-breaking."""
    rng = np.random.default_rng(seed)
    return _rank_select(np.asarray(score, dtype="float64"), n_select, rng, ascending=False)


# --------------------------------------------------------------------------------------
# Portfolio evaluation
# --------------------------------------------------------------------------------------


def evaluate_portfolio(
    sel: np.ndarray,
    opp: pd.DataFrame,
    *,
    ticket: float,
    limits: dict[str, float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    u_col: str = "u_exante",
    u_real_col: str = "u_expost",
) -> dict[str, Any]:
    """Ex-ante / ex-post metrics, concentration and constraint flags of one portfolio.

    ``limits`` keys: ``min_loans, max_avg_p, max_share_country, max_share_worst, max_share_tenor60``.
    Ex-post ROI/profit carry 95% bootstrap CIs over the selected loans; the default rate carries a Wilson CI.
    """
    sub = opp.iloc[sel]
    n = len(sub)
    invest = n * ticket
    u_ex = sub[u_col].to_numpy()
    u_re = sub[u_real_col].to_numpy()
    d = sub["default_12m"].to_numpy().astype(float)
    profit_ex = float(ticket * u_ex.sum())
    profit_re = float(ticket * u_re.sum())

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_profit = ticket * u_re[boot_idx].sum(axis=1)
    wl, wh = wilson_ci(d.sum(), n)
    comp = portfolio_composition(sel, opp)
    flags = constraint_flags(
        comp,
        min_loans=limits["min_loans"],
        max_avg_p=limits["max_avg_p"],
        max_share_country=limits["max_share_country"],
        max_share_worst=limits["max_share_worst"],
        max_share_tenor60=limits["max_share_tenor60"],
    )
    return {
        "n_pinjaman": int(n),
        "total_investasi_eur": float(invest),
        "p_rata2_exante": comp["p_mean"],
        "default_rate_expost": float(d.mean()),
        "default_rate_ci_low": wl,
        "default_rate_ci_high": wh,
        "profit_exante_eur": profit_ex,
        "roi_exante": profit_ex / invest,
        "profit_expost_eur": profit_re,
        "profit_expost_ci_low": float(np.percentile(boot_profit, 2.5)),
        "profit_expost_ci_high": float(np.percentile(boot_profit, 97.5)),
        "roi_expost": profit_re / invest,
        "roi_expost_ci_low": float(np.percentile(boot_profit, 2.5)) / invest,
        "roi_expost_ci_high": float(np.percentile(boot_profit, 97.5)) / invest,
        "share_country_maks": comp["share_country_max"],
        "country_maks": comp["country_max_name"],
        "share_rating_FHR": comp["share_rating_FHR"],
        "share_tenor60": comp["share_tenor60"],
        **flags,
        "memenuhi_batasan": bool(all(flags.values())),
    }


def evaluate_random_baseline(
    draws: list[np.ndarray],
    opp: pd.DataFrame,
    *,
    ticket: float,
    limits: dict[str, float],
    u_col: str = "u_exante",
    u_real_col: str = "u_expost",
) -> dict[str, Any]:
    """S0: average the portfolio metrics over random draws; CIs are 2.5-97.5 percentiles across draws."""
    rows = [
        evaluate_portfolio(
            s, opp, ticket=ticket, limits=limits, n_boot=2, u_col=u_col, u_real_col=u_real_col
        )
        for s in draws
    ]
    df = pd.DataFrame(rows)
    num = [
        "n_pinjaman",
        "total_investasi_eur",
        "p_rata2_exante",
        "default_rate_expost",
        "profit_exante_eur",
        "roi_exante",
        "profit_expost_eur",
        "roi_expost",
        "share_country_maks",
        "share_rating_FHR",
        "share_tenor60",
    ]
    out: dict[str, Any] = {c: float(df[c].mean()) for c in num}
    out["n_pinjaman"] = int(round(out["n_pinjaman"]))
    out["default_rate_ci_low"] = float(df["default_rate_expost"].quantile(0.025))
    out["default_rate_ci_high"] = float(df["default_rate_expost"].quantile(0.975))
    out["profit_expost_ci_low"] = float(df["profit_expost_eur"].quantile(0.025))
    out["profit_expost_ci_high"] = float(df["profit_expost_eur"].quantile(0.975))
    out["roi_expost_ci_low"] = float(df["roi_expost"].quantile(0.025))
    out["roi_expost_ci_high"] = float(df["roi_expost"].quantile(0.975))
    out["country_maks"] = str(df["country_maks"].mode().iloc[0])
    comp = {
        "n": float(out["n_pinjaman"]),
        "p_mean": out["p_rata2_exante"],
        "share_country_max": out["share_country_maks"],
        "share_rating_FHR": out["share_rating_FHR"],
        "share_tenor60": out["share_tenor60"],
    }
    flags = constraint_flags(
        comp,
        min_loans=limits["min_loans"],
        max_avg_p=limits["max_avg_p"],
        max_share_country=limits["max_share_country"],
        max_share_worst=limits["max_share_worst"],
        max_share_tenor60=limits["max_share_tenor60"],
    )
    out.update(flags)
    out["memenuhi_batasan"] = bool(all(flags.values()))
    return out


def add_unit_profits(
    opp: pd.DataFrame,
    loss_given_default: float,
    p_col: str = "p",
) -> pd.DataFrame:
    """Return a copy of ``opp`` with ``u_exante`` and ``u_expost`` columns for the given ``L``."""
    out = opp.copy()
    out["u_exante"] = unit_profit(out[p_col], out["interest12_frac"], loss_given_default)
    out["u_expost"] = unit_profit(out["default_12m"], out["interest12_frac"], loss_given_default)
    return out


# --------------------------------------------------------------------------------------
# Platform scenarios
# --------------------------------------------------------------------------------------


def threshold_policy_table(
    opp: pd.DataFrame, taus: Sequence[float], loss_given_default: float
) -> pd.DataFrame:
    """P1: approve loans with ``p <= tau``; exposure = full ``Amount``; economics from Step 2."""
    amount = opp["Amount"].to_numpy()
    u_ex = opp["u_exante"].to_numpy()
    u_re = opp["u_expost"].to_numpy()
    p = opp["p"].to_numpy()
    d = opp["default_12m"].to_numpy().astype(float)
    rows = []
    for tau in taus:
        m = p <= tau + 1e-12
        n_ok = int(m.sum())
        if n_ok == 0:
            rows.append({"tau": tau, "n_disetujui": 0})
            continue
        principal = float(amount[m].sum())
        pe = float((amount[m] * u_ex[m]).sum())
        pr = float((amount[m] * u_re[m]).sum())
        rows.append(
            {
                "tau": float(tau),
                "n_disetujui": n_ok,
                "tingkat_persetujuan": n_ok / len(opp),
                "p_rata2_disetujui": float(p[m].mean()),
                "default_rate_expost": float(d[m].mean()),
                "total_pokok_eur": principal,
                "profit_exante_eur": pe,
                "roi_exante": pe / principal,
                "profit_expost_eur": pr,
                "roi_expost": pr / principal,
            }
        )
    return pd.DataFrame(rows)


def breakeven_interest_pct(
    p: np.ndarray, loss_given_default: float, phi: np.ndarray
) -> np.ndarray:
    """Breakeven annual interest (percent) under the linear approximation ``f = phi * Interest / 100``.

    ``r* = p * L / ((1 - p) * phi)`` (decimal) so that ``(1 - p) * f - p * L = 0``; returned in percent.
    """
    p = np.asarray(p, dtype="float64")
    phi = np.asarray(phi, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p * loss_given_default / ((1.0 - p) * phi)
    return r * 100.0


def tenor_bin(duration: pd.Series | np.ndarray) -> pd.Categorical:
    """Tenor buckets used in the breakeven pricing segmentation."""
    bins = [0, 12, 24, 36, 48, 59, 60]
    labels = ["<=12", "13-24", "25-36", "37-48", "49-59", "60"]
    return pd.cut(pd.Series(duration), bins=bins, labels=labels, right=True)
