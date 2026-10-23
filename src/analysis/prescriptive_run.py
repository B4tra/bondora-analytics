"""Tahap 5 pipeline: ``python -m src.analysis.prescriptive_run``.

Runs Steps 1-7 of the Tahap 5 plan: parameters, unit economics, investor strategies (S0-S5) with constrained
optimisation (S3), what-if scenarios, platform scenarios (P1/P2) and light monthly forecasts. All tables are
written to ``reports/tables/06_*.csv`` and figures to ``reports/figures/06_*.png``.

All business parameters live in ``src/config.py`` and are fixed before the backtest runs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import forecasting as fc
from src.analysis import prescriptive as pr
from src.config import (
    BACKTEST_MONTHS,
    BUDGET_EUR,
    BUDGET_SCALE_GRID,
    DATA_PROCESSED,
    FIGURES_DIR,
    FORECAST_HORIZON,
    INTEREST_RAISE_CAP_PP,
    LGD_SHIFT_GRID,
    MAX_AVG_P,
    MAX_AVG_P_GRID,
    MAX_SHARE_COUNTRY,
    MAX_SHARE_COUNTRY_GRID,
    MAX_SHARE_TENOR60,
    MAX_SHARE_WORST_RATINGS,
    MIN_LOANS,
    MODELS_DIR,
    N_BOOTSTRAP,
    N_RANDOM_DRAWS,
    OPT_TIME_LIMIT_S,
    PD_STRESS_GRID,
    PLATFORM_MAX_AVG_P,
    PLATFORM_MIN_APPROVAL,
    RANDOM_SEED,
    TABLES_DIR,
    TAU_GRID,
    TICKET_EUR,
    ensure_dirs,
)
from src.utils.plotting import fmt_id, save_fig, set_style

EXPECTED_CLS_ROWS = 148_369
EXPECTED_TEST_ROWS = 54_283
EXPECTED_REG_ROWS = 113_560

STRATEGY_DESC: dict[str, str] = {
    "S0": "Acak (baseline), rata-rata 100 undian",
    "S1": "Ikuti rating Bondora (AA -> HR)",
    "S2": "Risiko terendah (p_calibrated terendah, tanpa batas konsentrasi)",
    "S3": "Optimasi berbatas (maksimalkan profit ex-ante di bawah semua batasan)",
    "S4": "Peringkat pengembalian (pred_return_ratio tertinggi)",
    "S5": "Imbal hasil tertinggi (Interest tertinggi; strategi naif)",
}


# --------------------------------------------------------------------------------------
# Data loading and preflight
# --------------------------------------------------------------------------------------


def base_limits(
    n_select_scale: float = 1.0,
    max_avg_p: float = MAX_AVG_P,
    max_share_country: float = MAX_SHARE_COUNTRY,
) -> dict[str, float]:
    """Constraint limits (defaults from config; scaled ``MIN_LOANS`` for budget scenarios)."""
    return {
        "min_loans": int(round(MIN_LOANS * n_select_scale)),
        "max_avg_p": max_avg_p,
        "max_share_country": max_share_country,
        "max_share_worst": MAX_SHARE_WORST_RATINGS,
        "max_share_tenor60": MAX_SHARE_TENOR60,
    }


def load_inputs() -> dict[str, Any]:
    """Load clean data and Tahap 4 prediction tables and verify the reference numbers (preflight)."""
    cls = pd.read_parquet(DATA_PROCESSED / "predictions_default12m.parquet")
    reg = pd.read_parquet(DATA_PROCESSED / "predictions_return_ratio.parquet")
    clean = pd.read_parquet(DATA_PROCESSED / "bondora_clean.parquet")

    n_test = int((cls["split"] == "test").sum())
    pre = {
        "n_pred_default12m": int(len(cls)),
        "n_pred_default12m_test": n_test,
        "n_pred_return_ratio": int(len(reg)),
        "p_calibrated_min": float(cls["p_calibrated"].min()),
        "p_calibrated_max": float(cls["p_calibrated"].max()),
        "p_calibrated_nan": int(cls["p_calibrated"].isna().sum()),
        "test_loandate_min": str(cls.loc[cls["split"] == "test", "LoanDate"].min().date()),
        "test_loandate_max": str(cls.loc[cls["split"] == "test", "LoanDate"].max().date()),
    }
    problems = []
    if pre["n_pred_default12m"] != EXPECTED_CLS_ROWS:
        problems.append(f"predictions_default12m rows {pre['n_pred_default12m']} != {EXPECTED_CLS_ROWS}")
    if n_test != EXPECTED_TEST_ROWS:
        problems.append(f"test rows {n_test} != {EXPECTED_TEST_ROWS}")
    if pre["n_pred_return_ratio"] != EXPECTED_REG_ROWS:
        problems.append(f"predictions_return_ratio rows {pre['n_pred_return_ratio']} != {EXPECTED_REG_ROWS}")
    if pre["p_calibrated_nan"] or pre["p_calibrated_min"] < 0 or pre["p_calibrated_max"] > 1:
        problems.append("p_calibrated has NaN or lies outside [0, 1]")
    if problems:
        raise AssertionError("Preflight gagal: " + "; ".join(problems))
    return {"cls": cls, "reg": reg, "clean": clean, "preflight": pre}


def build_opportunity_set(inputs: dict[str, Any]) -> pd.DataFrame:
    """Test-split loans with predictions, interest fraction and predicted return ratio (all loans)."""
    cls, reg, clean = inputs["cls"], inputs["reg"], inputs["clean"]
    opp = cls.loc[cls["split"] == "test"].reset_index(drop=True).copy()
    opp["Rating"] = opp["Rating"].astype("object").where(opp["Rating"].notna(), "Tanpa rating")
    opp["p"] = opp["p_calibrated"].astype("float64")
    opp["default_12m"] = opp["default_12m"].astype("int8")

    extra = clean.set_index("LoanId")[["MonthlyPayment", "PrincipalPaymentsMade"]]
    opp = opp.join(extra, on="LoanId")

    # Predicted return ratio for ALL opportunity loans: reuse the saved Tahap 4 model; cross-check against
    # the stored predictions where they overlap (resolved loans only).
    bundle = joblib.load(MODELS_DIR / "lgbm_return_ratio_setA.joblib")
    cl = clean.set_index("LoanId").loc[opp["LoanId"], bundle["features"]]
    pred = np.clip(bundle["model"].predict(bundle["aligner"].transform(cl)), 0.0, None)
    opp["pred_return_ratio"] = pred
    ref = reg.set_index("LoanId")["pred_return_ratio"]
    both = opp["LoanId"].isin(ref.index)
    max_diff = float(np.abs(opp.loc[both, "pred_return_ratio"].to_numpy() - ref.loc[opp.loc[both, "LoanId"]].to_numpy()).max())
    if max_diff > 1e-6:
        raise AssertionError(f"Saved regression model disagrees with predictions_return_ratio (max diff {max_diff}).")
    opp.attrs["return_ratio_crosscheck_n"] = int(both.sum())
    opp.attrs["return_ratio_crosscheck_maxdiff"] = max_diff

    opp["interest12_frac"] = pr.interest12_frac(opp["Interest"], opp["LoanDuration"])
    if opp["Amount"].min() < TICKET_EUR:
        raise AssertionError("Some loans have Amount below TICKET_EUR.")
    if opp["interest12_frac"].isna().any():
        raise AssertionError("NaN interest12_frac in opportunity set.")
    return opp


# --------------------------------------------------------------------------------------
# Step 1: parameters
# --------------------------------------------------------------------------------------


def params_table() -> pd.DataFrame:
    """Parameters and assumptions with their rationale (written to ``06_params.csv``)."""
    rows = [
        ("TICKET_EUR", TICKET_EUR, "Nominal yang didanai per pinjaman; investor mendanai porsi, bukan seluruh pinjaman."),
        ("BUDGET_EUR", BUDGET_EUR, "Anggaran investor; setara sekitar 5.000 pinjaman pada tiket 50 EUR."),
        ("N_PINJAMAN", int(BUDGET_EUR / TICKET_EUR), "Jumlah pinjaman yang didanai = BUDGET_EUR / TICKET_EUR."),
        ("MAX_AVG_P", MAX_AVG_P, "Batas rata-rata p_calibrated portofolio (selera risiko investor)."),
        ("MAX_SHARE_COUNTRY", MAX_SHARE_COUNTRY, "Batas konsentrasi per negara (diversifikasi, bukan diskriminasi individu)."),
        ("MAX_SHARE_WORST_RATINGS", MAX_SHARE_WORST_RATINGS, "Batas eksposur rating F dan HR (rating tersedia saat memilih)."),
        ("MAX_SHARE_TENOR60", MAX_SHARE_TENOR60, "Batas eksposur tenor 60 bulan (risiko jangka panjang)."),
        ("MIN_LOANS", MIN_LOANS, "Jumlah pinjaman minimum untuk diversifikasi (analog service level pada slide)."),
        ("PD_STRESS_GRID", "; ".join(f"{x:+.0%}" for x in PD_STRESS_GRID), "Pengali pada p untuk simulasi what-if."),
        ("LGD_SHIFT_GRID", "; ".join(f"{x:+.2f}" for x in LGD_SHIFT_GRID), "Pergeseran kerugian saat default (pecahan pokok; -10/0/+10 poin)."),
        ("BUDGET_SCALE_GRID", "; ".join(str(x) for x in BUDGET_SCALE_GRID), "Skala anggaran; MIN_LOANS disesuaikan proporsional."),
        ("MAX_SHARE_COUNTRY_GRID", "; ".join(str(x) for x in MAX_SHARE_COUNTRY_GRID), "Menunjukkan biaya diversifikasi."),
        ("MAX_AVG_P_GRID", "; ".join(str(x) for x in MAX_AVG_P_GRID), "Sensitivitas terhadap selera risiko."),
        ("N_RANDOM_DRAWS", N_RANDOM_DRAWS, "Undian S0 yang dirata-ratakan."),
        ("N_BOOTSTRAP", N_BOOTSTRAP, "Iterasi bootstrap CI ex-post."),
        ("RANDOM_SEED", RANDOM_SEED, "Seed untuk semua proses acak."),
        ("OPT_TIME_LIMIT_S", OPT_TIME_LIMIT_S, "Batas waktu solusi biner sebelum relaksasi LP."),
        ("TAU_GRID", "; ".join(f"{x:.2f}" for x in TAU_GRID), "Kandidat batas risiko persetujuan platform (p <= tau)."),
        ("PLATFORM_MIN_APPROVAL", PLATFORM_MIN_APPROVAL, "Tingkat persetujuan minimum platform (analog service level)."),
        ("PLATFORM_MAX_AVG_P", PLATFORM_MAX_AVG_P, "Batas rata-rata p pinjaman disetujui."),
        ("INTEREST_RAISE_CAP_PP", INTEREST_RAISE_CAP_PP, "Skenario ilustratif harga: kenaikan bunga maksimum (poin) menuju impas."),
        ("ASUMSI_bunga", "sederhana, tanpa reinvestasi", "interest12_frac dari anuitas, horizon min(12, tenor) bulan."),
        ("ASUMSI_biaya", "tanpa biaya platform", "Tidak ada biaya layanan/pajak; keuntungan kotor."),
        ("ASUMSI_kerugian", "konstan L", "Kerugian saat default konstan; bunga sebelum default diabaikan (konservatif)."),
        ("ASUMSI_atribut_sensitif", "Gender tidak dipakai", "Country hanya batas konsentrasi (diversifikasi), bukan alat diskriminasi."),
    ]
    return pd.DataFrame(rows, columns=["parameter", "nilai", "alasan"]).astype({"nilai": "string"})


# --------------------------------------------------------------------------------------
# Steps 3-4: strategies
# --------------------------------------------------------------------------------------


def run_strategies(opp: pd.DataFrame, n_select: int, limits: dict[str, float]) -> dict[str, Any]:
    """Build S0-S5, evaluate them and mark the recommendation (best ex-ante profit among compliant options)."""
    selections: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    draws = pr.select_random_draws(len(opp), n_select)
    selections["S1"] = pr.select_by_rating(opp["Rating"].to_numpy(), n_select)
    selections["S2"] = pr.select_lowest_p(opp["p"].to_numpy(), n_select)

    opt = pr.optimize_portfolio(
        opp["p"].to_numpy(),
        opp["u_exante"].to_numpy(),
        opp["Country"].astype(str).to_numpy(),
        opp["Rating"].to_numpy(),
        opp["LoanDuration"].to_numpy(),
        n_select=n_select,
        min_loans=limits["min_loans"],
        max_avg_p=limits["max_avg_p"],
        max_share_country=limits["max_share_country"],
        max_share_worst=limits["max_share_worst"],
        max_share_tenor60=limits["max_share_tenor60"],
    )
    opt["solve_seconds"] = round(time.perf_counter() - t0, 2)
    selections["S3"] = opt["selected"]
    selections["S4"] = pr.select_top_score(opp["pred_return_ratio"].to_numpy(), n_select)
    selections["S5"] = pr.select_top_score(opp["Interest"].to_numpy(), n_select)

    rows: list[dict[str, Any]] = []
    for code in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        if code == "S0":
            m = pr.evaluate_random_baseline(draws, opp, ticket=TICKET_EUR, limits=limits)
        elif len(selections[code]) == 0:
            continue
        else:
            m = pr.evaluate_portfolio(selections[code], opp, ticket=TICKET_EUR, limits=limits)
        rows.append({"opsi": code, "deskripsi": STRATEGY_DESC[code], **m})
    res = pd.DataFrame(rows)
    ok = res["memenuhi_batasan"].astype(bool)
    res["rekomendasi"] = False
    if ok.any():
        best = res.loc[ok, "profit_exante_eur"].idxmax()
        res.loc[best, "rekomendasi"] = True
    res["rekomendasi"] = res["rekomendasi"].map({True: "Ya", False: "Tidak"})
    return {"results": res, "selections": selections, "optimization": opt, "draws": draws}


def interest_validation_segments(opp: pd.DataFrame) -> pd.DataFrame:
    """Where does the annuity PMT disagree with ``MonthlyPayment``? Median gaps by Country and Rating.

    ``r_implisit`` is the annual nominal rate that reproduces ``MonthlyPayment`` for the loan's Amount and tenor.
    """
    d = opp.copy()
    d["pmt_calc"] = pr.annuity_payment_frac(d["Interest"], d["LoanDuration"]) * d["Amount"]
    d["selisih_rel"] = d["pmt_calc"] / d["MonthlyPayment"] - 1.0
    d["r_implisit"] = pr.implied_annual_rate_pct(d["Amount"], d["MonthlyPayment"], d["LoanDuration"])
    d["selisih_pp"] = d["r_implisit"] - d["Interest"]
    rows = []
    for dim in ("Country", "Rating"):
        for seg, g in d.groupby(d[dim].astype(str), observed=True):
            rows.append(
                {
                    "dimensi": dim,
                    "segmen": seg,
                    "n": len(g),
                    "interest_median": float(g["Interest"].median()),
                    "r_implisit_median": float(g["r_implisit"].median()),
                    "selisih_pp_median": float(g["selisih_pp"].median()),
                    "selisih_rel_pmt_median": float(g["selisih_rel"].median()),
                }
            )
    return pd.DataFrame(rows)


def run_interest_robustness(
    opp: pd.DataFrame, L: float, strat: dict[str, Any], n_select: int, limits: dict[str, float]
) -> pd.DataFrame:
    """Re-evaluate every strategy (and re-optimise S3) with interest income computed from the MonthlyPayment-implied rate."""
    alt = opp.copy()
    r_impl = pr.implied_annual_rate_pct(alt["Amount"], alt["MonthlyPayment"], alt["LoanDuration"])
    ok = np.isfinite(r_impl)
    alt["interest12_frac"] = np.where(
        ok, pr.interest12_frac(np.where(ok, r_impl, 0.0), alt["LoanDuration"]), alt["interest12_frac"]
    )
    alt = pr.add_unit_profits(alt, L)
    sels = dict(strat["selections"])
    opt = pr.optimize_portfolio(
        alt["p"].to_numpy(),
        alt["u_exante"].to_numpy(),
        alt["Country"].astype(str).to_numpy(),
        alt["Rating"].to_numpy(),
        alt["LoanDuration"].to_numpy(),
        n_select=n_select,
        min_loans=limits["min_loans"],
        max_avg_p=limits["max_avg_p"],
        max_share_country=limits["max_share_country"],
        max_share_worst=limits["max_share_worst"],
        max_share_tenor60=limits["max_share_tenor60"],
    )
    rows = []
    base_res = strat["results"].set_index("opsi")
    for code, sel in {**sels, "S3_ulang": opt["selected"]}.items():
        if len(sel) == 0:
            continue
        m = pr.evaluate_portfolio(sel, alt, ticket=TICKET_EUR, limits=limits, n_boot=20)
        base_code = "S3" if code == "S3_ulang" else code
        rows.append(
            {
                "opsi": code,
                "roi_exante_Interest": float(base_res.loc[base_code, "roi_exante"]) if code != "S3_ulang" else np.nan,
                "roi_exante_r_implisit": m["roi_exante"],
                "roi_expost_r_implisit": m["roi_expost"],
                "memenuhi_batasan": m["memenuhi_batasan"],
                "irisan_dengan_S3_share": float(len(np.intersect1d(sel, sels["S3"])) / len(sels["S3"])),
                "catatan": "S3 dioptimasi ulang dengan r implisit" if code == "S3_ulang" else "seleksi tetap, hanya penilaian diganti",
            }
        )
    out = pd.DataFrame(rows)
    out["opsi_terbaik_exante_patuh"] = ""
    ok_mask = out["memenuhi_batasan"].astype(bool)
    if ok_mask.any():
        out.loc[out.loc[ok_mask, "roi_exante_r_implisit"].idxmax(), "opsi_terbaik_exante_patuh"] = "Ya"
    return out


def composition_table(opp: pd.DataFrame, sel: np.ndarray) -> pd.DataFrame:
    """S3 composition by Country, Rating and tenor vs the opportunity set."""
    sub = opp.iloc[sel]
    rows = []
    dims = {
        "Country": (opp["Country"].astype(str), sub["Country"].astype(str)),
        "Rating": (opp["Rating"].astype(str), sub["Rating"].astype(str)),
        "Tenor": (pr.tenor_bin(opp["LoanDuration"]).astype(str), pr.tenor_bin(sub["LoanDuration"]).astype(str)),
    }
    for dim, (full, part) in dims.items():
        cnt_full = full.value_counts()
        cnt_part = part.value_counts()
        for seg in sorted(cnt_full.index):
            n_p = int(cnt_part.get(seg, 0))
            sub_seg = sub.loc[part.to_numpy() == seg] if n_p else sub.iloc[:0]
            rows.append(
                {
                    "dimensi": dim,
                    "segmen": seg,
                    "n_peluang": int(cnt_full[seg]),
                    "share_peluang": float(cnt_full[seg] / len(opp)),
                    "n_S3": n_p,
                    "share_S3": n_p / len(sub),
                    "p_rata2_S3": float(sub_seg["p"].mean()) if n_p else np.nan,
                    "default_rate_expost_S3": float(sub_seg["default_12m"].mean()) if n_p else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Step 5: what-if
# --------------------------------------------------------------------------------------


def _scenario_rows(
    opp: pd.DataFrame,
    L: float,
    *,
    skenario: str,
    nilai: str,
    p_stress: float = 0.0,
    L_shift: float = 0.0,
    budget_scale: float = 1.0,
    max_share_country: float = MAX_SHARE_COUNTRY,
    max_avg_p: float = MAX_AVG_P,
) -> list[dict[str, Any]]:
    """Run S3 (re-optimised), S2 and S5 under one scenario and return rows with the best option flagged."""
    L_use = float(np.clip(L + L_shift, 0.0, 1.0))
    sc = opp.copy()
    sc["p"] = np.clip(sc["p"] * (1.0 + p_stress), 0.0, 1.0)
    sc = pr.add_unit_profits(sc, L_use)
    n_select = int(round(BUDGET_EUR * budget_scale / TICKET_EUR))
    limits = base_limits(budget_scale, max_avg_p, max_share_country)

    opt = pr.optimize_portfolio(
        sc["p"].to_numpy(),
        sc["u_exante"].to_numpy(),
        sc["Country"].astype(str).to_numpy(),
        sc["Rating"].to_numpy(),
        sc["LoanDuration"].to_numpy(),
        n_select=n_select,
        min_loans=limits["min_loans"],
        max_avg_p=limits["max_avg_p"],
        max_share_country=limits["max_share_country"],
        max_share_worst=limits["max_share_worst"],
        max_share_tenor60=limits["max_share_tenor60"],
    )
    sels = {
        "S3": opt["selected"],
        "S2": pr.select_lowest_p(sc["p"].to_numpy(), n_select),
        "S5": pr.select_top_score(sc["Interest"].to_numpy(), n_select),
    }
    rows: list[dict[str, Any]] = []
    for code, sel in sels.items():
        base = {
            "skenario": skenario,
            "nilai_parameter": nilai,
            "opsi": code,
            "status_solver": opt["status"] if code == "S3" else "n/a (aturan peringkat)",
            "catatan": "",
        }
        if len(sel) == 0:
            conflicts = opt.get("conflicts", {})
            base.update(
                {
                    "n_pinjaman": 0,
                    "profit_exante_eur": np.nan,
                    "roi_exante": np.nan,
                    "profit_expost_eur": np.nan,
                    "roi_expost": np.nan,
                    "memenuhi_batasan": False,
                    "catatan": "Tidak layak; pelonggaran minimum (satuan N): "
                    + "; ".join(f"{k}: {v:.4f}" for k, v in conflicts.items()),
                }
            )
        else:
            m = pr.evaluate_portfolio(sel, sc, ticket=TICKET_EUR, limits=limits, n_boot=20)
            base.update(
                {
                    "n_pinjaman": m["n_pinjaman"],
                    "profit_exante_eur": m["profit_exante_eur"],
                    "roi_exante": m["roi_exante"],
                    "profit_expost_eur": m["profit_expost_eur"],
                    "roi_expost": m["roi_expost"],
                    "memenuhi_batasan": m["memenuhi_batasan"],
                    "p_rata2_exante": m["p_rata2_exante"],
                    "share_country_maks": m["share_country_maks"],
                }
            )
        rows.append(base)
    df = pd.DataFrame(rows)
    compliant = df["memenuhi_batasan"].astype(bool)
    best = df.loc[compliant, "profit_exante_eur"].idxmax() if compliant.any() else None
    label = df.loc[best, "opsi"] if best is not None else "tidak ada yang memenuhi batasan"
    df["opsi_terbaik"] = label
    return df.to_dict("records")


def run_whatif(opp: pd.DataFrame, L: float) -> pd.DataFrame:
    """All Step-5 scenarios for S3 (recommended), S2 and S5."""
    rows: list[dict[str, Any]] = []
    for s in PD_STRESS_GRID:
        rows += _scenario_rows(opp, L, skenario="stres_p", nilai=f"{s:+.0%}", p_stress=s)
    for sh in LGD_SHIFT_GRID:
        rows += _scenario_rows(opp, L, skenario="geser_L", nilai=f"{sh * 100:+.0f} poin", L_shift=sh)
    for b in BUDGET_SCALE_GRID:
        rows += _scenario_rows(opp, L, skenario="anggaran", nilai=f"{b:.0%} dari BUDGET_EUR", budget_scale=b)
    for c in MAX_SHARE_COUNTRY_GRID:
        rows += _scenario_rows(opp, L, skenario="batas_negara", nilai=f"{c:.2f}", max_share_country=c)
    for a in MAX_AVG_P_GRID:
        rows += _scenario_rows(opp, L, skenario="batas_risiko", nilai=f"{a:.2f}", max_avg_p=a)
    cols = [
        "skenario",
        "nilai_parameter",
        "opsi",
        "n_pinjaman",
        "profit_exante_eur",
        "roi_exante",
        "profit_expost_eur",
        "roi_expost",
        "status_solver",
        "memenuhi_batasan",
        "opsi_terbaik",
        "p_rata2_exante",
        "share_country_maks",
        "catatan",
    ]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------------------
# Step 6: platform scenarios
# --------------------------------------------------------------------------------------


def run_platform_policy(opp: pd.DataFrame, L: float) -> pd.DataFrame:
    """P1: threshold policy table with feasibility flags and recommendation."""
    tab = pr.threshold_policy_table(opp, TAU_GRID, L)
    tab["memenuhi_batasan"] = (tab["tingkat_persetujuan"] >= PLATFORM_MIN_APPROVAL) & (
        tab["p_rata2_disetujui"] <= PLATFORM_MAX_AVG_P
    )
    tab["rekomendasi"] = "Tidak"
    ok = tab["memenuhi_batasan"]
    if ok.any():
        tab.loc[tab.loc[ok, "profit_exante_eur"].idxmax(), "rekomendasi"] = "Ya"
    return tab


def run_platform_pricing(opp: pd.DataFrame, L: float, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """P2: breakeven pricing table, linear-approximation check and illustrative repricing scenario."""
    interest = opp["Interest"].to_numpy()
    phi = opp["interest12_frac"].to_numpy() / (interest / 100.0)
    rstar = pr.breakeven_interest_pct(opp["p"].to_numpy(), L, phi)
    d = opp.assign(phi=phi, r_star=rstar)
    d["di_bawah_impas"] = d["Interest"] < d["r_star"]
    d["tenor_bin"] = pr.tenor_bin(d["LoanDuration"]).astype(str).to_numpy()

    rows = []
    rows.append(
        {
            "dimensi": "Semua",
            "segmen": "Semua",
            "n_pinjaman": len(d),
            "pct_di_bawah_impas": float(d["di_bawah_impas"].mean()),
            "interest_median": float(d["Interest"].median()),
            "r_star_median": float(d["r_star"].median()),
            "selisih_median_pp": float((d["r_star"] - d["Interest"]).median()),
        }
    )
    for dim, col in [("Country", "Country"), ("Rating", "Rating"), ("Tenor", "tenor_bin")]:
        for seg, g in d.groupby(d[col].astype(str), observed=True):
            rows.append(
                {
                    "dimensi": dim,
                    "segmen": seg,
                    "n_pinjaman": len(g),
                    "pct_di_bawah_impas": float(g["di_bawah_impas"].mean()),
                    "interest_median": float(g["Interest"].median()),
                    "r_star_median": float(g["r_star"].median()),
                    "selisih_median_pp": float((g["r_star"] - g["Interest"]).median()),
                }
            )
    breakeven_tab = pd.DataFrame(rows)

    # Verify the linear approximation: recompute f at r* and the resulting ex-ante unit profit (should be ~0).
    rng = np.random.default_rng(seed)
    ok = np.where(np.isfinite(rstar) & (d["p"].to_numpy() < 0.999))[0]
    idx = rng.choice(ok, size=min(1000, len(ok)), replace=False)
    f_at_r = pr.interest12_frac(rstar[idx], d["LoanDuration"].to_numpy()[idx])
    u_at_r = pr.unit_profit(d["p"].to_numpy()[idx], f_at_r, L)
    check = pd.DataFrame(
        {
            "n_sampel": [len(idx)],
            "galat_u_rata2_abs": [float(np.mean(np.abs(u_at_r)))],
            "galat_u_median_abs": [float(np.median(np.abs(u_at_r)))],
            "galat_u_p95_abs": [float(np.percentile(np.abs(u_at_r), 95))],
            "galat_u_rata2_bertanda": [float(np.mean(u_at_r))],
        }
    )

    # Illustrative repricing: raise Interest toward r* (capped); ignores demand and default-risk effects.
    new_int = np.where(d["di_bawah_impas"], np.minimum(d["r_star"], d["Interest"] + INTEREST_RAISE_CAP_PP), d["Interest"])
    f_new = pr.interest12_frac(new_int, d["LoanDuration"].to_numpy())
    u_old = d["u_exante"].to_numpy()
    u_new = pr.unit_profit(d["p"].to_numpy(), f_new, L)
    amt = d["Amount"].to_numpy()
    d["delta_profit"] = amt * (u_new - u_old)
    d["profit_lama"] = amt * u_old
    d["profit_baru"] = amt * u_new
    scen_rows = [
        {
            "segmen": "Semua",
            "n_dinaikkan": int(d["di_bawah_impas"].sum()),
            "profit_exante_lama_eur": float(d["profit_lama"].sum()),
            "profit_exante_baru_eur": float(d["profit_baru"].sum()),
            "delta_profit_eur": float(d["delta_profit"].sum()),
            "total_pokok_eur": float(amt.sum()),
        }
    ]
    for rating, g in d.groupby(d["Rating"].astype(str), observed=True):
        scen_rows.append(
            {
                "segmen": f"Rating {rating}",
                "n_dinaikkan": int(g["di_bawah_impas"].sum()),
                "profit_exante_lama_eur": float(g["profit_lama"].sum()),
                "profit_exante_baru_eur": float(g["profit_baru"].sum()),
                "delta_profit_eur": float(g["delta_profit"].sum()),
                "total_pokok_eur": float(g["Amount"].sum()),
            }
        )
    return {"breakeven": breakeven_tab, "check": check, "scenario": pd.DataFrame(scen_rows)}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def plot_strategy_figures(res: pd.DataFrame) -> None:
    """06_strategy_comparison.png and 06_strategy_risk_return.png."""
    set_style()
    order = res["opsi"].tolist()
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, res["roi_exante"] * 100, w, label="ROI ex-ante (berbasis model)", color="#4c72b0")
    err = np.vstack(
        [
            (res["roi_expost"] - res["roi_expost_ci_low"]).to_numpy() * 100,
            (res["roi_expost_ci_high"] - res["roi_expost"]).to_numpy() * 100,
        ]
    )
    ax.bar(x + w / 2, res["roi_expost"] * 100, w, yerr=err, capsize=4, label="ROI ex-post (hasil aktual, CI 95%)", color="#dd8452")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o}{'' if m else ' *'}" for o, m in zip(order, res["memenuhi_batasan"])])
    ax.set_ylabel("ROI 12 bulan (%)")
    ax.set_title("Perbandingan ROI ex-ante vs ex-post per strategi investor")
    ax.legend()
    ax.text(0.01, 0.02, "* melanggar minimal satu batasan (tidak boleh direkomendasikan)", transform=ax.transAxes, fontsize=8.5)
    save_fig(fig, "06_strategy_comparison")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, r in res.iterrows():
        filled = bool(r["memenuhi_batasan"])
        ax.scatter(
            r["default_rate_expost"] * 100,
            r["roi_expost"] * 100,
            s=140,
            facecolors="#4c72b0" if filled else "none",
            edgecolors="#c44e52" if not filled else "#4c72b0",
            linewidths=2,
        )
        ax.annotate(r["opsi"], (r["default_rate_expost"] * 100, r["roi_expost"] * 100), xytext=(7, 6), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Tingkat default 12 bulan aktual (%)")
    ax.set_ylabel("ROI ex-post (%)")
    ax.set_title("Risiko vs imbal hasil per strategi (penuh = memenuhi batasan)")
    save_fig(fig, "06_strategy_risk_return")


def plot_tornado(whatif: pd.DataFrame) -> pd.DataFrame:
    """06_whatif_tornado.png: change in S3 ex-ante ROI vs baseline for the low/high value of each parameter."""
    set_style()
    s3 = whatif[whatif["opsi"] == "S3"].copy()
    base = s3[(s3["skenario"] == "stres_p") & (s3["nilai_parameter"] == "+0%")].iloc[0]["roi_exante"]
    spec = [
        ("stres_p", "Stres p (-20% .. +50%)", "-20%", "+50%"),
        ("geser_L", "Pergeseran L (-10 .. +10 poin)", "-10 poin", "+10 poin"),
        ("anggaran", "Anggaran (50% .. 200%)", "50% dari BUDGET_EUR", "200% dari BUDGET_EUR"),
        ("batas_negara", "Batas negara (0,5 .. 1,0)", "0.50", "1.00"),
        ("batas_risiko", "Batas rata-rata p (0,15 .. 0,30)", "0.15", "0.30"),
    ]
    rows = []
    for key, label, lo, hi in spec:
        sub = s3[s3["skenario"] == key].set_index("nilai_parameter")
        rows.append(
            {
                "parameter": label,
                "roi_low": sub.loc[lo, "roi_exante"],
                "roi_high": sub.loc[hi, "roi_exante"],
                "status_low": sub.loc[lo, "status_solver"],
                "status_high": sub.loc[hi, "status_solver"],
            }
        )
    t = pd.DataFrame(rows)
    t["delta_low_pp"] = (t["roi_low"] - base) * 100
    t["delta_high_pp"] = (t["roi_high"] - base) * 100
    t["rentang"] = t[["delta_low_pp", "delta_high_pp"]].abs().max(axis=1)
    t = t.sort_values("rentang")

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(t))
    ax.barh(y, t["delta_low_pp"].fillna(0), color="#55a868", label="nilai rendah")
    ax.barh(y, t["delta_high_pp"].fillna(0), color="#c44e52", alpha=0.85, label="nilai tinggi")
    for k, (_, r) in enumerate(t.iterrows()):
        for col, st, off in (("delta_low_pp", "status_low", -1), ("delta_high_pp", "status_high", 1)):
            if pd.isna(r[col]):
                ax.text(0, k, " tidak layak ", va="center", ha="center", fontsize=8.5, color="#7f0000")
    ax.set_yticks(y)
    ax.set_yticklabels(t["parameter"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Perubahan ROI ex-ante S3 vs baseline ({base * 100:.2f}%), poin persentase")
    ax.set_title("Analisis sensitivitas (tornado) strategi S3")
    ax.legend(loc="lower right")
    save_fig(fig, "06_whatif_tornado")
    return t.drop(columns=["rentang"])


def plot_platform_curve(pol: pd.DataFrame) -> None:
    """06_platform_threshold_curve.png."""
    set_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(pol["tau"], pol["profit_exante_eur"] / 1e6, "o-", color="#4c72b0", label="Profit ex-ante (juta EUR)")
    ax1.plot(pol["tau"], pol["profit_expost_eur"] / 1e6, "s--", color="#dd8452", label="Profit ex-post (juta EUR)")
    ax1.set_xlabel("Batas risiko persetujuan tau (p <= tau)")
    ax1.set_ylabel("Profit 12 bulan (juta EUR)")
    ax1.axhline(0, color="black", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(pol["tau"], pol["tingkat_persetujuan"] * 100, "^:", color="#55a868", label="Tingkat persetujuan (%)")
    ax2.axhline(PLATFORM_MIN_APPROVAL * 100, color="#55a868", lw=0.8, ls="--")
    ax2.set_ylabel("Tingkat persetujuan (%)")
    ax2.grid(False)
    rec = pol[pol["rekomendasi"] == "Ya"]
    if len(rec):
        ax1.axvline(float(rec["tau"].iloc[0]), color="#c44e52", lw=1.2, ls="-.")
        ax1.text(float(rec["tau"].iloc[0]), ax1.get_ylim()[1] * 0.92, " tau terpilih", color="#c44e52")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left")
    ax1.set_title("Kebijakan persetujuan platform: profit dan tingkat persetujuan vs tau")
    save_fig(fig, "06_platform_threshold_curve")


def _plot_series(ax: plt.Axes, y: pd.Series, f: pd.DataFrame, title: str, ylabel: str) -> None:
    ax.plot(y.index, y.values, color="#4c72b0", label="Riwayat (bulan penuh)")
    ax.plot(f["month"], f["mean"], "o-", color="#dd8452", label=f"Prakiraan ({f['model'].iloc[0]})")
    ax.fill_between(f["month"], f["lo95"], f["hi95"], color="#dd8452", alpha=0.18, label="Interval 95%")
    ax.fill_between(f["month"], f["lo80"], f["hi80"], color="#dd8452", alpha=0.35, label="Interval 80%")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_forecasts(series: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """06_forecast_volume.png and 06_forecast_defaults.png."""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, col, title, yl in [
        (axes[0], "n_loans", "Prakiraan jumlah pinjaman baru per bulan", "Jumlah pinjaman"),
        (axes[1], "amount_loans", "Prakiraan total Amount pinjaman baru per bulan", "EUR"),
    ]:
        _plot_series(ax, series[col], forecast[forecast["serie"] == col], title, yl)
    axes[0].legend(loc="upper left", fontsize=8.5)
    save_fig(fig, "06_forecast_volume")

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_series(
        ax,
        series["amount_default"],
        forecast[forecast["serie"] == "amount_default"],
        "Prakiraan arus default baru (total Amount menurut bulan DefaultDate)",
        "EUR",
    )
    ax.legend(loc="upper left", fontsize=8.5)
    save_fig(fig, "06_forecast_defaults")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_forecast_step(clean: pd.DataFrame, s3_share: float, n_select: int) -> dict[str, Any]:
    """Step 7: monthly series, models, backtest, forecast and the link to the investment decision."""
    series = fc.build_monthly_series(clean)
    out = fc.run_forecasts(series, horizon=FORECAST_HORIZON, n_test=BACKTEST_MONTHS)
    forecast = out["forecast"]
    ser = series.reset_index()
    ser.to_csv(TABLES_DIR / "06_forecast_series.csv", index=False)
    forecast.to_csv(TABLES_DIR / "06_forecast_values.csv", index=False)
    out["backtest"].to_csv(TABLES_DIR / "06_forecast_backtest.csv", index=False)
    out["backtest_detail"].to_csv(TABLES_DIR / "06_forecast_backtest_detail.csv", index=False)

    vol = forecast[forecast["serie"] == "n_loans"].copy()
    vol = vol[vol["month"] >= pd.Timestamp("2021-08-01")]
    link = pd.DataFrame(
        {
            "bulan": vol["month"].dt.strftime("%Y-%m").to_numpy(),
            "prakiraan_pinjaman": vol["mean"].to_numpy(),
            "lo80": vol["lo80"].to_numpy(),
            "hi80": vol["hi80"].to_numpy(),
            "share_lolos_S3": s3_share,
            "lolos_per_bulan": vol["mean"].to_numpy() * s3_share,
            "lolos_lo80": vol["lo80"].to_numpy() * s3_share,
            "lolos_hi80": vol["hi80"].to_numpy() * s3_share,
            "MIN_LOANS": MIN_LOANS,
            "N_target": n_select,
        }
    )
    link["bulan_untuk_mencapai_N"] = n_select / link["lolos_per_bulan"]
    link["bulan_untuk_mencapai_MIN_LOANS"] = MIN_LOANS / link["lolos_per_bulan"]
    link.to_csv(TABLES_DIR / "06_forecast_decision_link.csv", index=False)
    plot_forecasts(series, forecast)
    return {"series": series, "forecast": forecast, "backtest": out["backtest"], "link": link}


def main() -> dict[str, Any]:
    """Run the full Tahap 5 pipeline and write all ``06_*`` tables and figures."""
    t_start = time.perf_counter()
    ensure_dirs()
    set_style()
    meta: dict[str, Any] = {"random_seed": RANDOM_SEED}

    print("=== Tahap 5 Prescriptive Analytics (`src.analysis.prescriptive_run`) ===")
    inputs = load_inputs()
    meta["preflight"] = inputs["preflight"]
    print(f"[Preflight OK] {inputs['preflight']}")

    # Step 1
    params_table().to_csv(TABLES_DIR / "06_params.csv", index=False)

    # Opportunity set and Step 2 (economics)
    opp = build_opportunity_set(inputs)
    meta["return_ratio_crosscheck"] = {
        "n": opp.attrs["return_ratio_crosscheck_n"],
        "max_abs_diff": opp.attrs["return_ratio_crosscheck_maxdiff"],
    }
    val = pr.validate_payment_formula(opp["Amount"], opp["Interest"], opp["LoanDuration"], opp["MonthlyPayment"])
    meta["interest_validation"] = val
    pd.DataFrame([val]).to_csv(TABLES_DIR / "06_interest_validation.csv", index=False)
    print(f"[Validasi PMT vs MonthlyPayment] {val}")

    cls = inputs["cls"]
    clean = inputs["clean"]
    tv_ids = cls.loc[cls["split"].isin(["train", "val"]) & (cls["default_12m"] == 1), "LoanId"]
    tv = clean.set_index("LoanId").loc[tv_ids, ["PrincipalPaymentsMade", "Amount"]]
    lgd = pr.estimate_loss_given_default(tv["PrincipalPaymentsMade"], tv["Amount"])
    meta["lgd"] = lgd
    pd.DataFrame([lgd]).to_csv(TABLES_DIR / "06_lgd_estimate.csv", index=False)
    L = lgd["L"]
    print(f"[L] {lgd}")

    opp = pr.add_unit_profits(opp, L)
    meta["opportunity_set"] = {
        "n": int(len(opp)),
        "default_rate_actual": float(opp["default_12m"].mean()),
        "p_mean": float(opp["p"].mean()),
        "u_exante_mean": float(opp["u_exante"].mean()),
        "u_expost_mean": float(opp["u_expost"].mean()),
        "interest12_frac_mean": float(opp["interest12_frac"].mean()),
    }

    # Steps 3-4
    n_select = int(BUDGET_EUR / TICKET_EUR)
    limits = base_limits()
    strat = run_strategies(opp, n_select, limits)
    res = strat["results"]
    res.to_csv(TABLES_DIR / "06_strategy_results.csv", index=False)
    opt = strat["optimization"]
    rep = opt["constraint_report"]
    rep.to_csv(TABLES_DIR / "06_s3_constraints.csv", index=False)
    meta["s3_solver"] = {
        "status": opt["status"],
        "objective_sum_u": opt["objective"],
        "n_variables": opt["n_variables"],
        "pool_size": opt["pool_size"],
        "n_fractional": opt["n_fractional"],
        "all_constraints_ok": opt.get("all_constraints_ok"),
        "solver": "HiGHS via scipy.optimize.linprog/milp (LP dulu, lalu MILP sisa); pulp 4.x tidak menyertakan CBC",
        "n_selected": int(len(opt["selected"])),
        "conflicts": opt.get("conflicts", {}),
    }
    print(f"[S3 solver] {meta['s3_solver']} in {opt['solve_seconds']}s")
    print(res[["opsi", "n_pinjaman", "p_rata2_exante", "default_rate_expost", "roi_exante", "roi_expost", "memenuhi_batasan", "rekomendasi"]].to_string())
    composition_table(opp, strat["selections"]["S3"]).to_csv(TABLES_DIR / "06_s3_composition.csv", index=False)
    interest_validation_segments(opp).to_csv(TABLES_DIR / "06_interest_validation_segments.csv", index=False)
    run_interest_robustness(opp, L, strat, n_select, limits).to_csv(
        TABLES_DIR / "06_interest_robustness.csv", index=False
    )
    plot_strategy_figures(res)

    # Step 5
    print("[What-if] running scenarios...")
    whatif = run_whatif(opp, L)
    whatif.to_csv(TABLES_DIR / "06_whatif_scenarios.csv", index=False)
    tornado = plot_tornado(whatif)
    tornado.to_csv(TABLES_DIR / "06_whatif_tornado.csv", index=False)

    # Step 6
    pol = run_platform_policy(opp, L)
    pol.to_csv(TABLES_DIR / "06_platform_threshold_policy.csv", index=False)
    plot_platform_curve(pol)
    pricing = run_platform_pricing(opp, L)
    pricing["breakeven"].to_csv(TABLES_DIR / "06_platform_breakeven_pricing.csv", index=False)
    pricing["check"].to_csv(TABLES_DIR / "06_platform_breakeven_check.csv", index=False)
    pricing["scenario"].to_csv(TABLES_DIR / "06_platform_pricing_scenario.csv", index=False)

    # Step 7
    s3_share = float(len(opt["selected"]) / len(opp))
    meta["s3_share_of_opportunity_set"] = s3_share
    fcast = run_forecast_step(clean, s3_share, n_select)

    with open(TABLES_DIR / "06_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=float)

    elapsed = round(time.perf_counter() - t_start, 2)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "prescriptive_timing.json", "w", encoding="utf-8") as f:
        json.dump({"total_elapsed_seconds": elapsed}, f)
    print(f"=== Pipeline finished in {elapsed:.2f}s ===")
    return {"meta": meta, "strategy": res, "whatif": whatif, "policy": pol, "pricing": pricing, "forecast": fcast}


if __name__ == "__main__":
    main()
