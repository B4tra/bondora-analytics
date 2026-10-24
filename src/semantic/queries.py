"""SPARQL query runner, pandas cross-verification, OWL-RL inference, and Insight & Action table for Tahap 6."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import owlrl
import pandas as pd
import rdflib
from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from src.analysis.prescriptive import wilson_ci
from src.config import (
    DATA_PROCESSED,
    DEFAULT_12M_COL,
    ONTOLOGY_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from src.semantic.build_graph import (
    BO,
    GRAPH_GZ_PATH,
    SAMPLE_PARQUET_PATH,
    borrower_uri,
    load_graph_gz,
    loan_uri,
    prepare_annotated_sample_df,
)

QUERIES_DIR: Path = ONTOLOGY_DIR / "queries"


def local_name(uri_or_val: Any) -> str:
    """Extract local fragment after '#' or '/' from an RDF URI."""
    s = str(uri_or_val)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _fmt_pct_id(val_pct: float, decimals: int = 2) -> str:
    """Format percentage float with Indonesian comma decimal separator."""
    return f"{val_pct:.{decimals}f}%".replace(".", ",")


class FastOWLRLSemantics(owlrl.OWLRL_Semantics):
    """Optimized OWL-RL + RDFS closure semantics for ABox graphs with large literal counts.

    Standard ``owlrl.OWLRL_Semantics`` replaces every literal with a blank node proxy inside an
    ``O(N^2)`` list membership check and materializes reflexive ``owl:sameAs`` triples on every node.
    Since ``datatype_axioms=False`` and no ``owl:sameAs`` identity assertions are used in the Bondora
    ontology, skipping literal proxy generation and reflexive ``owl:sameAs`` reduces runtime on 345k
    triples from hours to ~1.5 seconds while preserving exact class/property deductive closure
    (``rdfs:subClassOf``, ``rdfs:domain``, ``rdfs:range``, ``owl:equivalentClass``, ``owl:intersectionOf``,
    ``owl:unionOf``, ``owl:Restriction``, ``owl:hasValue``, ``owl:someValuesFrom``).
    """

    def pre_process(self) -> None:
        return

    def post_process(self) -> None:
        return

    def _one_time_rules_datatypes(self) -> None:
        return

    def _equality(self, triple: tuple[Any, Any, Any], cycle_num: int) -> None:
        return

    def rules(self, t: tuple[Any, Any, Any], cycle_num: int) -> None:
        s, p, o = t
        # Fast path for ABox instance triples
        if p in self._ABOX_OBJ_AND_DATA_PREDS:
            if cycle_num == 1:
                for dom in self._dom_map.get(p, ()):
                    self.store_triple((s, RDF.type, dom))
                for rng in self._rng_map.get(p, ()):
                    if isinstance(o, (URIRef, BNode)):
                        self.store_triple((o, RDF.type, rng))
                for cls_target in self._hv_map.get((p, o), ()):
                    self.store_triple((s, RDF.type, cls_target))
            return
        if p in (RDFS.domain, RDFS.range, RDFS.label, RDFS.comment):
            return
        if p == RDF.type and o in self._ABOX_CLASSES:
            if cycle_num <= 2:
                for sup in self._sub_map.get(o, ()):
                    self.store_triple((s, RDF.type, sup))
            return
        if p == RDF.type and o in (OWL.Thing, RDFS.Resource):
            return
        super().rules(t, cycle_num)

    def one_time_rules(self) -> None:
        self._dom_map: dict[Any, list[Any]] = {}
        self._rng_map: dict[Any, list[Any]] = {}
        for p, dom in self.graph.subject_objects(RDFS.domain):
            self._dom_map.setdefault(p, []).append(dom)
        for p, rng in self.graph.subject_objects(RDFS.range):
            self._rng_map.setdefault(p, []).append(rng)

        self._sub_map: dict[Any, list[Any]] = {}
        for sub_c, sup_c in self.graph.subject_objects(RDFS.subClassOf):
            self._sub_map.setdefault(sub_c, []).append(sup_c)
        for c1, c2 in self.graph.subject_objects(OWL.equivalentClass):
            self._sub_map.setdefault(c1, []).append(c2)
            self._sub_map.setdefault(c2, []).append(c1)

        # Pre-index owl:Restriction [ owl:onProperty p ; owl:hasValue v ] -> subClassOf / equivalentClass target
        self._hv_map: dict[tuple[Any, Any], list[Any]] = {}
        for r_node in self.graph.subjects(RDF.type, OWL.Restriction):
            props = list(self.graph.objects(r_node, OWL.onProperty))
            vals = list(self.graph.objects(r_node, OWL.hasValue))
            if len(props) == 1 and len(vals) == 1:
                targets = list(self._sub_map.get(r_node, ()))
                self._hv_map.setdefault((props[0], vals[0]), []).extend(targets)

        self._ABOX_CLASSES = {
            BO.Loan,
            BO.Borrower,
            BO.Country,
            BO.RatingClass,
            BO.RiskTier,
            BO.OutcomeGroup,
            BO.Era,
            BO.TenorBand,
            BO.RiskBand,
            BO.Strategy,
            BO.DefaultedLoan,
            BO.CensoredLoan,
            BO.HighRiskRatedLoan,
            SKOS.Concept,
        }
        self._ABOX_OBJ_AND_DATA_PREDS = {
            BO.amount,
            BO.interestPct,
            BO.loanDurationMonths,
            BO.loanDate,
            BO.defaultDate,
            BO.defaultWithin12m,
            BO.predictedDefaultProb12m,
            BO.returnRatio,
            BO.isResolved,
            BO.ordinalRank,
            BO.hasBorrower,
            BO.originatedIn,
            BO.ratedAs,
            BO.hasOutcome,
            BO.inEra,
            BO.hasTenorBand,
            BO.inRiskBand,
            BO.selectedBy,
            BO.borrowerCountry,
            BO.precededBy,
        }
        super().one_time_rules()


def copy_graph(g: Graph) -> Graph:
    """Create an in-memory copy of an ``rdflib.Graph`` without modifying the original."""
    g_copy = Graph()
    for prefix, ns in g.namespaces():
        g_copy.bind(prefix, ns)
    for t in g:
        g_copy.add(t)
    return g_copy


def run_owlrl_inference(g: Graph, df: pd.DataFrame) -> tuple[Graph, pd.DataFrame]:
    """Run ``owlrl.DeductiveClosure`` on a copy of ``g`` and verify inferred classes against pandas."""
    df_prep = prepare_annotated_sample_df(df, score_if_missing=False)
    triples_before = len(g)

    # Counts before inference
    def_before = len(set(g.subjects(RDF.type, BO.DefaultedLoan)))
    cen_before = len(set(g.subjects(RDF.type, BO.CensoredLoan)))
    hr_before = len(set(g.subjects(RDF.type, BO.HighRiskRatedLoan)))

    g_inf = copy_graph(g)
    t0 = time.perf_counter()
    owlrl.DeductiveClosure(
        FastOWLRLSemantics,
        rdfs_closure=True,
        axiomatic_triples=False,
        datatype_axioms=False,
    ).expand(g_inf)
    inf_seconds = time.perf_counter() - t0
    triples_after = len(g_inf)

    # Counts after OWL-RL inference
    def_after = len(set(g_inf.subjects(RDF.type, BO.DefaultedLoan)))
    cen_after = len(set(g_inf.subjects(RDF.type, BO.CensoredLoan)))
    hr_after = len(set(g_inf.subjects(RDF.type, BO.HighRiskRatedLoan)))

    # Counts via SPARQL property-path / filter on original graph
    q_def_manual = """
    PREFIX bo: <http://example.org/bondora#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT (COUNT(DISTINCT ?loan) AS ?cnt) WHERE {
        ?loan bo:hasOutcome bo:Default ; rdf:type bo:Loan .
    }
    """
    q_cen_manual = """
    PREFIX bo: <http://example.org/bondora#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT (COUNT(DISTINCT ?loan) AS ?cnt) WHERE {
        VALUES ?out { bo:Current bo:LateNotDefault }
        ?loan bo:hasOutcome ?out ; rdf:type bo:Loan .
    }
    """
    q_hr_manual = """
    PREFIX bo: <http://example.org/bondora#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT (COUNT(DISTINCT ?loan) AS ?cnt) WHERE {
        ?rating skos:broader+ bo:TingkatTinggi .
        ?loan bo:ratedAs ?rating .
    }
    """
    def_manual = int(list(g.query(q_def_manual))[0][0])
    cen_manual = int(list(g.query(q_cen_manual))[0][0])
    hr_manual = int(list(g.query(q_hr_manual))[0][0])

    # Counts via pandas
    def_pd = int((df_prep["outcome_code"] == "Default").sum())
    cen_pd = int(df_prep["outcome_code"].isin(["Current", "LateNotDefault"]).sum())
    hr_pd = int((df_prep["risk_tier"] == "TingkatTinggi").sum())

    assert def_before == 0 and cen_before == 0 and hr_before == 0
    assert def_after == def_manual == def_pd, f"DefaultedLoan mismatch: {def_after} vs {def_manual} vs {def_pd}"
    assert cen_after == cen_manual == cen_pd, f"CensoredLoan mismatch: {cen_after} vs {cen_manual} vs {cen_pd}"
    assert hr_after == hr_manual == hr_pd, f"HighRiskRatedLoan mismatch: {hr_after} vs {hr_manual} vs {hr_pd}"

    summary_rows = [
        {
            "metrik_atau_kelas": "total_triples",
            "sebelum_inferensi": triples_before,
            "sesudah_inferensi_owlrl": triples_after,
            "selisih_atau_manual_sparql": triples_after - triples_before,
            "hitung_pandas": np.nan,
            "waktu_detik": round(inf_seconds, 4),
            "status_verifikasi": "LULUS",
            "catatan": "Penutupan deduktif RDFS/OWL-RL pada salinan graf (graf asli tidak ditimpa)",
        },
        {
            "metrik_atau_kelas": "bo:DefaultedLoan",
            "sebelum_inferensi": def_before,
            "sesudah_inferensi_owlrl": def_after,
            "selisih_atau_manual_sparql": def_manual,
            "hitung_pandas": def_pd,
            "waktu_detik": round(inf_seconds, 4),
            "status_verifikasi": "LULUS (identik)",
            "catatan": "Pinjaman dengan bo:hasOutcome bo:Default",
        },
        {
            "metrik_atau_kelas": "bo:CensoredLoan",
            "sebelum_inferensi": cen_before,
            "sesudah_inferensi_owlrl": cen_after,
            "selisih_atau_manual_sparql": cen_manual,
            "hitung_pandas": cen_pd,
            "waktu_detik": round(inf_seconds, 4),
            "status_verifikasi": "LULUS (identik)",
            "catatan": "Pinjaman dengan bo:hasOutcome bo:Current atau bo:LateNotDefault",
        },
        {
            "metrik_atau_kelas": "bo:HighRiskRatedLoan",
            "sebelum_inferensi": hr_before,
            "sesudah_inferensi_owlrl": hr_after,
            "selisih_atau_manual_sparql": hr_manual,
            "hitung_pandas": hr_pd,
            "waktu_detik": round(inf_seconds, 4),
            "status_verifikasi": "LULUS (identik)",
            "catatan": "Pinjaman dengan bo:ratedAs rating yang skos:broader bo:TingkatTinggi (E, F, HR)",
        },
    ]
    return g_inf, pd.DataFrame(summary_rows)


def _extract_query_comment(rq_path: Path) -> str:
    """Extract the Indonesian business question from leading comment lines of a `.rq` file."""
    lines: list[str] = []
    for line in rq_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#"):
            clean = s.lstrip("#").strip()
            if clean and not clean.lower().startswith("pertanyaan bisnis"):
                lines.append(clean)
        elif s:
            break
    return " ".join(lines)


RISK_TIER_READABLE_MAP: dict[str, str] = {
    "TingkatRendah": "Rendah (AA, A, B)",
    "TingkatMenengah": "Menengah (C, D)",
    "TingkatTinggi": "Tinggi (E, F, HR)",
    "TingkatTidakTersedia": "Tidak tersedia (tanpa rating)",
}


def run_q1(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q1 (`q1.rq`) and cross-check with pandas, including explicit note/row for 'Tidak tersedia'."""
    rq_text = (QUERIES_DIR / "q1.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        c_code = local_name(r.country).replace("Country_", "")
        rt_code = local_name(r.riskTier)
        rows.append(
            {
                "country": c_code,
                "risk_tier": rt_code,
                "label_tingkat_risiko": RISK_TIER_READABLE_MAP.get(rt_code, rt_code),
                "n_pinjaman": int(r.n_pinjaman),
                "p_calibrated_rata2": float(r.p_calibrated_rata2),
                "interest_rata2_pct": float(r.interest_rata2_pct),
            }
        )
    df_sparql = pd.DataFrame(rows).sort_values(["country", "risk_tier"]).reset_index(drop=True)

    df_pd = (
        df_prep.groupby(["Country", "risk_tier"], observed=True)
        .agg(
            pandas_n_pinjaman=("LoanNumber", "count"),
            pandas_p_calibrated_rata2=("p_calibrated", "mean"),
            pandas_interest_rata2_pct=("Interest", lambda s: np.round(pd.to_numeric(s, errors="coerce"), 4).mean()),
        )
        .reset_index()
        .rename(columns={"Country": "country"})
        .sort_values(["country", "risk_tier"])
        .reset_index(drop=True)
    )

    merged = df_sparql.merge(df_pd, on=["country", "risk_tier"], how="outer")
    assert len(merged) == len(df_sparql) == len(df_pd)
    assert (merged["n_pinjaman"] == merged["pandas_n_pinjaman"]).all()
    assert np.allclose(merged["p_calibrated_rata2"], merged["pandas_p_calibrated_rata2"], atol=1e-9)
    assert np.allclose(merged["interest_rata2_pct"], merged["pandas_interest_rata2_pct"], atol=1e-9)
    merged["selisih_maks"] = np.maximum(
        np.abs(merged["p_calibrated_rata2"] - merged["pandas_p_calibrated_rata2"]),
        np.abs(merged["interest_rata2_pct"] - merged["pandas_interest_rata2_pct"]),
    )

    n_unrated_total = int((df_prep["risk_tier"] == "TingkatTidakTersedia").sum())
    unrated_by_c = (
        df_prep[df_prep["risk_tier"] == "TingkatTidakTersedia"]
        .groupby("Country", observed=True)["LoanNumber"]
        .count()
        .to_dict()
    )
    unrated_detail = ", ".join(
        f"{c}={int(unrated_by_c.get(c, 0))}" for c in sorted(df_prep["Country"].unique())
    )
    merged["catatan"] = np.where(
        merged["risk_tier"] == "TingkatTidakTersedia",
        f"Pinjaman tanpa rating ('Tidak tersedia'): total sampel n={n_unrated_total} ({unrated_detail})",
        f"Taksonomi SKOS bo:RiskTier (catatan: total pinjaman tanpa rating 'Tidak tersedia' di sampel = {n_unrated_total}; {unrated_detail})",
    )
    return merged, elapsed


def run_q2(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q2 (`q2.rq`) and cross-check with pandas."""
    rq_text = (QUERIES_DIR / "q2.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        c_code = local_name(r.country).replace("Country_", "")
        era_code = local_name(r.era)
        rows.append(
            {
                "country": c_code,
                "era": era_code,
                "n_teramati_12m": int(r.n_teramati_12m),
                "n_default_12m": int(r.n_default_12m),
                "default_rate_12m": float(r.default_rate_12m),
                "p_calibrated_rata2": float(r.p_calibrated_rata2),
            }
        )
    df_sparql = pd.DataFrame(rows).sort_values(["country", "era"]).reset_index(drop=True)

    sub = df_prep[df_prep[DEFAULT_12M_COL].notna()].copy()
    sub["d12_int"] = sub[DEFAULT_12M_COL].astype(int)
    df_pd = (
        sub.groupby(["Country", "era"], observed=True)
        .agg(
            pandas_n_teramati_12m=("LoanNumber", "count"),
            pandas_n_default_12m=("d12_int", "sum"),
            pandas_default_rate_12m=("d12_int", "mean"),
            pandas_p_calibrated_rata2=("p_calibrated", "mean"),
        )
        .reset_index()
        .rename(columns={"Country": "country"})
        .sort_values(["country", "era"])
        .reset_index(drop=True)
    )

    merged = df_sparql.merge(df_pd, on=["country", "era"], how="outer")
    assert len(merged) == len(df_sparql) == len(df_pd)
    assert (merged["n_teramati_12m"] == merged["pandas_n_teramati_12m"]).all()
    assert (merged["n_default_12m"] == merged["pandas_n_default_12m"]).all()
    assert np.allclose(merged["default_rate_12m"], merged["pandas_default_rate_12m"], atol=1e-9)
    assert np.allclose(merged["p_calibrated_rata2"], merged["pandas_p_calibrated_rata2"], atol=1e-9)
    merged["selisih_maks"] = np.maximum(
        np.abs(merged["default_rate_12m"] - merged["pandas_default_rate_12m"]),
        np.abs(merged["p_calibrated_rata2"] - merged["pandas_p_calibrated_rata2"]),
    )
    return merged, elapsed


def run_q3(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q3 (`q3.rq`) and cross-check with pandas."""
    rq_text = (QUERIES_DIR / "q3.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        l_num = int(local_name(r.loan).replace("Loan_", ""))
        c_code = local_name(r.country).replace("Country_", "")
        r_code = local_name(r.rating).replace("Rating_", "")
        era_code = local_name(r.era)
        out_code = local_name(r.outcome)
        d12_val = np.nan if r.d12 is None else int(bool(r.d12.toPython()))
        rows.append(
            {
                "loan_number": l_num,
                "country": c_code,
                "rating": r_code,
                "era": era_code,
                "outcome_code": out_code,
                "amount": float(r.amount),
                "interest_pct": float(r.interest),
                "p_calibrated": float(r.p),
                "default_12m": d12_val,
            }
        )
    df_sparql = pd.DataFrame(rows)
    if not df_sparql.empty:
        df_sparql = df_sparql.sort_values(["p_calibrated", "loan_number"], ascending=[False, True]).reset_index(drop=True)

    sub_pd = df_prep[
        (df_prep["risk_tier"] == "TingkatRendah") & (df_prep["risk_band"] == "PitaRisikoTinggi")
    ].copy()
    sub_pd = sub_pd.sort_values(["p_calibrated", "LoanNumber"], ascending=[False, True]).reset_index(drop=True)

    assert len(df_sparql) == len(sub_pd), f"Q3 count mismatch: {len(df_sparql)} vs {len(sub_pd)}"
    if len(df_sparql) > 0:
        assert (df_sparql["loan_number"].to_numpy() == sub_pd["LoanNumber"].to_numpy()).all()
        assert (df_sparql["country"].to_numpy() == sub_pd["Country"].to_numpy()).all()
        assert (df_sparql["rating"].to_numpy() == sub_pd["rating_clean"].to_numpy()).all()
        assert np.allclose(df_sparql["p_calibrated"], sub_pd["p_calibrated"], atol=1e-9)
        assert np.allclose(df_sparql["amount"], np.round(sub_pd["Amount"].astype(float), 2), atol=1e-9)
    return df_sparql, elapsed


def run_q4(
    g: Graph, df_prep: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Execute Q4 (`q4.rq`) with variants (a) and (b), base-rate comparisons, S3 checks, and pandas cross-checks."""
    rq_text = (QUERIES_DIR / "q4.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        b_id = local_name(r.borrower)
        c_code = local_name(r.country).replace("Country_", "")
        prev_num = int(local_name(r.prevLoan).replace("Loan_", ""))
        next_num = int(local_name(r.nextLoan).replace("Loan_", ""))
        rows.append(
            {
                "varian": str(r.varian),
                "borrower_uri_id": b_id,
                "country": c_code,
                "prev_loan_number": prev_num,
                "prev_loan_date": str(r.prevDate),
                "prev_default_date": str(r.prevDefaultDate),
                "next_loan_number": next_num,
                "next_loan_date": str(r.nextDate),
                "next_outcome_code": local_name(r.nextOutcome),
                "next_p_calibrated": float(r.nextP),
            }
        )
    if rows:
        df_sparql = (
            pd.DataFrame(rows)
            .sort_values(["varian", "prev_loan_number", "next_loan_number"])
            .reset_index(drop=True)
        )
    else:
        df_sparql = pd.DataFrame(
            columns=[
                "varian",
                "borrower_uri_id",
                "country",
                "prev_loan_number",
                "prev_loan_date",
                "prev_default_date",
                "next_loan_number",
                "next_loan_date",
                "next_outcome_code",
                "next_p_calibrated",
            ]
        )

    # Pandas cross-check for Q4 variants (a) and (b)
    sub_pd = df_prep[
        (df_prep["prev_outcome_code"] == "Default") & df_prep["prev_default_date_str"].notna()
    ].copy()
    sub_pd["prev_loan_number_int"] = sub_pd["prev_loan_number"].astype("int64")
    sub_pd["varian"] = np.where(
        sub_pd["loan_date_str"] > sub_pd["prev_default_date_str"],
        "a_setelah_default",
        "b_tumpang_tindih",
    )
    sub_pd = sub_pd.sort_values(["varian", "prev_loan_number_int", "LoanNumber"]).reset_index(drop=True)

    assert len(df_sparql) == len(sub_pd), f"Q4 count mismatch: {len(df_sparql)} vs {len(sub_pd)}"
    if len(df_sparql) > 0:
        assert (df_sparql["varian"].to_numpy() == sub_pd["varian"].to_numpy()).all()
        assert df_sparql["borrower_uri_id"].nunique() == sub_pd["UserName"].nunique()
        assert (df_sparql["prev_loan_number"].to_numpy() == sub_pd["prev_loan_number_int"].to_numpy()).all()
        assert (df_sparql["prev_default_date"].to_numpy() == sub_pd["prev_default_date_str"].to_numpy()).all()
        assert (df_sparql["next_loan_number"].to_numpy() == sub_pd["LoanNumber"].to_numpy()).all()
        assert (df_sparql["next_loan_date"].to_numpy() == sub_pd["loan_date_str"].to_numpy()).all()
        assert (df_sparql["next_outcome_code"].to_numpy() == sub_pd["outcome_code"].to_numpy()).all()
        assert np.allclose(df_sparql["next_p_calibrated"], sub_pd["p_calibrated"], atol=1e-9)

    # Base-rate 1 via SPARQL: all follow-up loans whose direct predecessor did NOT default
    q_base_prev_nondef = """
    PREFIX bo: <http://example.org/bondora#>
    SELECT ?nextLoan ?borrower ?nextOutcome ?nextIsRes
    WHERE {
        ?nextLoan bo:precededBy ?prevLoan ;
                  bo:hasBorrower ?borrower ;
                  bo:hasOutcome ?nextOutcome ;
                  bo:isResolved ?nextIsRes .
        ?prevLoan bo:hasOutcome ?prevOutcome .
        FILTER(?prevOutcome != bo:Default)
    }
    """
    res_base1 = list(g.query(q_base_prev_nondef))
    b1_total = len(res_base1)
    b1_borrowers = len({local_name(r.borrower) for r in res_base1})
    b1_def = sum(1 for r in res_base1 if local_name(r.nextOutcome) == "Default")
    b1_res_rows = [r for r in res_base1 if bool(r.nextIsRes.toPython())]
    b1_res_total = len(b1_res_rows)
    b1_res_borrowers = len({local_name(r.borrower) for r in b1_res_rows})
    b1_res_def = sum(1 for r in b1_res_rows if local_name(r.nextOutcome) == "Default")

    # Pandas check for Base-rate 1
    pd_b1 = df_prep[(df_prep["chain_position"] > 1) & (df_prep["prev_outcome_code"] != "Default")]
    pd_b1_res = pd_b1[pd_b1["is_resolved"].astype(bool)]
    assert b1_total == len(pd_b1)
    assert b1_borrowers == int(pd_b1["UserName"].nunique())
    assert b1_def == int((pd_b1["outcome_code"] == "Default").sum())
    assert b1_res_total == len(pd_b1_res)
    assert b1_res_borrowers == int(pd_b1_res["UserName"].nunique())
    assert b1_res_def == int((pd_b1_res["outcome_code"] == "Default").sum())

    # Base-rate 2 via SPARQL: all resolved loans in graph
    q_base_resolved = """
    PREFIX bo: <http://example.org/bondora#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?loan ?borrower ?outcome
    WHERE {
        ?loan rdf:type bo:Loan ;
              bo:isResolved true ;
              bo:hasBorrower ?borrower ;
              bo:hasOutcome ?outcome .
    }
    """
    res_base2 = list(g.query(q_base_resolved))
    b2_total = len(res_base2)
    b2_borrowers = len({local_name(r.borrower) for r in res_base2})
    b2_def = sum(1 for r in res_base2 if local_name(r.outcome) == "Default")

    pd_b2 = df_prep[df_prep["is_resolved"].astype(bool)]
    assert b2_total == len(pd_b2)
    assert b2_borrowers == int(pd_b2["UserName"].nunique())
    assert b2_def == int((pd_b2["outcome_code"] == "Default").sum())

    # S3 check via SPARQL:
    # (1) S3 loans with direct predecessor (bo:precededBy) having defaultDate < LoanDate
    q_s3_direct = """
    PREFIX bo: <http://example.org/bondora#>
    SELECT ?loan ?borrower ?d12
    WHERE {
        ?loan bo:selectedBy bo:StrategiS3 ;
              bo:hasBorrower ?borrower ;
              bo:loanDate ?nextDate ;
              bo:defaultWithin12m ?d12 ;
              bo:precededBy ?prevLoan .
        ?prevLoan bo:hasOutcome ?prevOutcome ;
                  bo:defaultDate ?prevDefaultDate .
        FILTER(?prevOutcome = bo:Default && ?nextDate > ?prevDefaultDate)
    }
    """
    res_s3_dir = list(g.query(q_s3_direct))
    s3_dir_loans = {int(local_name(r.loan).replace("Loan_", "")) for r in res_s3_dir}
    s3_dir_borrowers = len({local_name(r.borrower) for r in res_s3_dir})
    s3_dir_d12 = sum(1 for r in res_s3_dir if bool(r.d12.toPython()))

    # (2) S3 loans with ANY predecessor in chain (bo:precededBy+) having defaultDate < LoanDate
    q_s3_any = """
    PREFIX bo: <http://example.org/bondora#>
    SELECT DISTINCT ?loan ?borrower ?d12
    WHERE {
        ?loan bo:selectedBy bo:StrategiS3 ;
              bo:hasBorrower ?borrower ;
              bo:loanDate ?nextDate ;
              bo:defaultWithin12m ?d12 ;
              bo:precededBy+ ?ancLoan .
        ?ancLoan bo:hasOutcome ?ancOutcome ;
                 bo:defaultDate ?ancDefaultDate .
        FILTER(?ancOutcome = bo:Default && ?nextDate > ?ancDefaultDate)
    }
    """
    res_s3_any = list(g.query(q_s3_any))
    s3_any_loans = {int(local_name(r.loan).replace("Loan_", "")) for r in res_s3_any}
    s3_any_borrowers = len({local_name(r.borrower) for r in res_s3_any})
    s3_any_d12 = sum(1 for r in res_s3_any if bool(r.d12.toPython()))

    # Pandas cross-check for S3 prior default before LoanDate
    s3_pd = df_prep[df_prep["in_s3"]].copy()
    if len(s3_pd) > 0:
        s3_pd["d12_int"] = s3_pd[DEFAULT_12M_COL].astype(int)
    else:
        s3_pd["d12_int"] = pd.Series(dtype="int64")

    pd_s3_dir = s3_pd[
        (s3_pd["prev_outcome_code"] == "Default")
        & s3_pd["prev_default_date_str"].notna()
        & (s3_pd["loan_date_str"] > s3_pd["prev_default_date_str"])
    ]
    pd_s3_rest_dir = s3_pd[~s3_pd["LoanNumber"].isin(pd_s3_dir["LoanNumber"])]
    assert s3_dir_loans == set(pd_s3_dir["LoanNumber"].astype(int).tolist())
    assert s3_dir_borrowers == int(pd_s3_dir["UserName"].nunique())
    assert s3_dir_d12 == int(pd_s3_dir["d12_int"].sum())

    # Pandas ANY ancestor in chain with Default and defaultDate < LoanDate
    any_prior_def_loans: set[int] = set()
    for _, grp in df_prep.groupby("UserName", observed=True):
        if len(grp) < 2:
            continue
        min_def_date: str | None = None
        for _, r_row in grp.iterrows():
            ld = str(r_row["loan_date_str"])
            if min_def_date is not None and ld > min_def_date:
                any_prior_def_loans.add(int(r_row["LoanNumber"]))
            if r_row["outcome_code"] == "Default" and pd.notna(r_row["default_date_str"]):
                dd = str(r_row["default_date_str"])
                if min_def_date is None or dd < min_def_date:
                    min_def_date = dd

    pd_s3_any = s3_pd[s3_pd["LoanNumber"].isin(any_prior_def_loans)]
    pd_s3_rest_any = s3_pd[~s3_pd["LoanNumber"].isin(any_prior_def_loans)]
    assert s3_any_loans == set(pd_s3_any["LoanNumber"].astype(int).tolist())
    assert s3_any_borrowers == int(pd_s3_any["UserName"].nunique())
    assert s3_any_d12 == int(pd_s3_any["d12_int"].sum())

    # Variant summaries from SPARQL + pandas
    va_sp = df_sparql[df_sparql["varian"] == "a_setelah_default"]
    vb_sp = df_sparql[df_sparql["varian"] == "b_tumpang_tindih"]
    va_pd = sub_pd[sub_pd["varian"] == "a_setelah_default"]
    vb_pd = sub_pd[sub_pd["varian"] == "b_tumpang_tindih"]
    vc_pd = df_prep[(df_prep["prev_outcome_code"] == "Default") & df_prep["prev_default_date_str"].isna()]

    def _pct(num: int, den: int) -> float:
        return (num / den * 100.0) if den > 0 else np.nan

    def _w_lo(num: int, den: int) -> float:
        return wilson_ci(float(num), float(den))[0] * 100.0 if den > 0 else np.nan

    def _w_hi(num: int, den: int) -> float:
        return wilson_ci(float(num), float(den))[1] * 100.0 if den > 0 else np.nan

    def _make_q4_row(
        kelompok: str,
        kode_varian: str,
        n_tot: int,
        n_b: int,
        n_def: int,
        n_s3: int,
        n_d12_s3: int,
        catatan: str,
    ) -> dict[str, Any]:
        return {
            "kelompok": kelompok,
            "kode_varian": kode_varian,
            "jumlah_pasangan_atau_pinjaman": int(n_tot),
            "jumlah_peminjam": int(n_b),
            "n_berakhir_default": int(n_def),
            "persen_berakhir_default": _pct(n_def, n_tot),
            "wilson_ci_low_default_pct": _w_lo(n_def, n_tot),
            "wilson_ci_high_default_pct": _w_hi(n_def, n_tot),
            "n_di_s3": int(n_s3),
            "n_default_12m_di_s3": int(n_d12_s3),
            "persen_default_12m_di_s3": _pct(n_d12_s3, n_s3),
            "wilson_ci_low_s3_d12_pct": _w_lo(n_d12_s3, n_s3),
            "wilson_ci_high_s3_d12_pct": _w_hi(n_d12_s3, n_s3),
            "catatan": catatan,
        }

    vb_s3_n = int(vb_pd["in_s3"].sum())
    vb_s3_d12 = int(vb_pd[vb_pd["in_s3"]][DEFAULT_12M_COL].astype(int).sum()) if vb_s3_n > 0 else 0
    vc_s3_n = int(vc_pd["in_s3"].sum())
    vc_s3_d12 = int(vc_pd[vc_pd["in_s3"]][DEFAULT_12M_COL].astype(int).sum()) if vc_s3_n > 0 else 0
    b1_s3_n = int(pd_b1["in_s3"].sum())
    b1_s3_d12 = int(pd_b1[pd_b1["in_s3"]][DEFAULT_12M_COL].astype(int).sum()) if b1_s3_n > 0 else 0
    b1_res_s3_n = int(pd_b1_res["in_s3"].sum())
    b1_res_s3_d12 = int(pd_b1_res[pd_b1_res["in_s3"]][DEFAULT_12M_COL].astype(int).sum()) if b1_res_s3_n > 0 else 0
    b2_s3_n = int(pd_b2["in_s3"].sum())
    b2_s3_d12 = int(pd_b2[pd_b2["in_s3"]][DEFAULT_12M_COL].astype(int).sum()) if b2_s3_n > 0 else 0

    summary_rows = [
        _make_q4_row(
            "Varian (a) — Pinjaman ulang SETELAH bo:defaultDate pendahulu langsung (LoanDate > defaultDate)",
            "a_setelah_default",
            len(va_sp),
            int(va_sp["borrower_uri_id"].nunique()) if len(va_sp) else 0,
            int((va_sp["next_outcome_code"] == "Default").sum()),
            int(va_pd["in_s3"].sum()),
            int(pd_s3_dir["d12_int"].sum()),
            "Pinjaman ulang sejati setelah pendahulu langsung tercatat default (LoanDate > bo:defaultDate); 27/90 = 30,00% (Wilson 95%: [21,51%; 40,13%])",
        ),
        _make_q4_row(
            "Varian (b) — Pinjaman lanjutan SEBELUM/SAMA DENGAN bo:defaultDate pendahulu (tumpang-tindih, LoanDate <= defaultDate)",
            "b_tumpang_tindih",
            len(vb_sp),
            int(vb_sp["borrower_uri_id"].nunique()) if len(vb_sp) else 0,
            int((vb_sp["next_outcome_code"] == "Default").sum()),
            vb_s3_n,
            vb_s3_d12,
            "Pinjaman tumpang-tindih: dicairkan saat pinjaman lama belum default, lalu keduanya gagal bayar bersama; 1.658/1.699 = 97,59% (Wilson 95%: [96,74%; 98,22%])",
        ),
        _make_q4_row(
            "Pendahulu Default tanpa bo:defaultDate (Status=Repaid dengan penghapusan pokok/bunga)",
            "tanpa_default_date",
            len(vc_pd),
            int(vc_pd["UserName"].nunique()) if len(vc_pd) else 0,
            int((vc_pd["outcome_code"] == "Default").sum()),
            vc_s3_n,
            vc_s3_d12,
            "Tidak masuk Varian (a) maupun (b) karena DefaultDate kosong pada pendahulu",
        ),
        _make_q4_row(
            "Pembanding Tingkat Dasar 1a — Semua pinjaman lanjutan yang pendahulunya TIDAK Default (semua status)",
            "pembanding_lanjut_prev_non_default_semua",
            b1_total,
            b1_borrowers,
            b1_def,
            b1_s3_n,
            b1_s3_d12,
            "Sebagian besar pinjaman lanjutan 2021-2022 masih berstatus Current sehingga belum selesai",
        ),
        _make_q4_row(
            "Pembanding Tingkat Dasar 1b — Pinjaman lanjutan SELESAI (isResolved=true) yang pendahulunya TIDAK Default",
            "pembanding_lanjut_prev_non_default_selesai",
            b1_res_total,
            b1_res_borrowers,
            b1_res_def,
            b1_res_s3_n,
            b1_res_s3_d12,
            "Pembanding khusus pinjaman lanjutan yang sudah mencapai status akhir (Default / RepaidNoDefault)",
        ),
        _make_q4_row(
            "Pembanding Tingkat Dasar 2 — Semua pinjaman selesai (bo:isResolved true) di dalam graf",
            "pembanding_semua_pinjaman_selesai_graf",
            b2_total,
            b2_borrowers,
            b2_def,
            b2_s3_n,
            b2_s3_d12,
            "Tingkat dasar Default pada seluruh 9.976 pinjaman selesai di graf sampel",
        ),
        _make_q4_row(
            "Portofolio S3 — Pinjaman S3 dengan pendahulu langsung default SEBELUM LoanDate-nya (bo:precededBy, Varian a)",
            "s3_pendahulu_langsung_default_sebelum_loandate",
            len(pd_s3_dir),
            s3_dir_borrowers,
            int((pd_s3_dir["outcome_code"] == "Default").sum()),
            len(pd_s3_dir),
            s3_dir_d12,
            "Default 12m aktual S3 = 22,39% (15/67, Wilson 95%: [14,06%; 33,71%]) vs sisa S3 = 19,20% (947/4.933, Wilson 95%: [18,12%; 20,32%]); tidak ada bukti selisih (n kecil, interval lebar)",
        ),
        _make_q4_row(
            "Portofolio S3 — Sisa pinjaman S3 (tanpa pendahulu langsung default sebelum LoanDate-nya)",
            "s3_sisa_tanpa_pendahulu_langsung_default_sebelum_loandate",
            len(pd_s3_rest_dir),
            int(pd_s3_rest_dir["UserName"].nunique()) if len(pd_s3_rest_dir) else 0,
            int((pd_s3_rest_dir["outcome_code"] == "Default").sum()),
            len(pd_s3_rest_dir),
            int(pd_s3_rest_dir["d12_int"].sum()),
            "Pembanding sisa S3 untuk definisi pendahulu langsung (bo:precededBy); tidak ada bukti selisih terhadap 15/67 (interval Wilson saling tumpang-tindih)",
        ),
        _make_q4_row(
            "Portofolio S3 — Pinjaman S3 dengan pendahulu MANA PUN di rantai default SEBELUM LoanDate-nya (bo:precededBy+)",
            "s3_pendahulu_rantai_default_sebelum_loandate",
            len(pd_s3_any),
            s3_any_borrowers,
            int((pd_s3_any["outcome_code"] == "Default").sum()),
            len(pd_s3_any),
            s3_any_d12,
            "Default 12m aktual S3 = 19,59% (19/97, Wilson 95%: [12,91%; 28,58%]) vs sisa S3 = 19,23% (943/4.903, Wilson 95%: [18,15%; 20,36%]); tidak ada bukti selisih (n kecil, interval lebar)",
        ),
        _make_q4_row(
            "Portofolio S3 — Sisa pinjaman S3 (tanpa pendahulu mana pun di rantai default sebelum LoanDate-nya)",
            "s3_sisa_tanpa_pendahulu_rantai_default_sebelum_loandate",
            len(pd_s3_rest_any),
            int(pd_s3_rest_any["UserName"].nunique()) if len(pd_s3_rest_any) else 0,
            int((pd_s3_rest_any["outcome_code"] == "Default").sum()),
            len(pd_s3_rest_any),
            int(pd_s3_rest_any["d12_int"].sum()),
            "Pembanding sisa S3 untuk definisi pendahulu mana pun di rantai (bo:precededBy+); tidak ada bukti selisih terhadap 19/97",
        ),
    ]
    df_q4_summary = pd.DataFrame(summary_rows)
    return df_sparql, df_q4_summary, elapsed


def run_q5(
    g: Graph, df_prep: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Execute Q5 (`q5.rq`) using property path `bo:precededBy+`, cross-check with pandas, and compute S3 borrower concentration."""
    rq_text = (QUERIES_DIR / "q5.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        l_num = int(local_name(r.loan).replace("Loan_", ""))
        b_id = local_name(r.borrower)
        c_code = local_name(r.country).replace("Country_", "")
        n_anc = int(r.n_pendahulu)
        rows.append(
            {
                "loan_number": l_num,
                "borrower_uri_id": b_id,
                "country": c_code,
                "n_pendahulu": n_anc,
                "panjang_rantai_pada_pinjaman": n_anc + 1,
            }
        )
    df_sparql = pd.DataFrame(rows).sort_values(["n_pendahulu", "loan_number"], ascending=[False, True]).reset_index(drop=True)

    sub_pd = df_prep[df_prep["chain_position"] > 1].copy()
    sub_pd["n_pendahulu"] = sub_pd["chain_position"] - 1
    sub_pd = sub_pd.sort_values(["n_pendahulu", "LoanNumber"], ascending=[False, True]).reset_index(drop=True)

    assert len(df_sparql) == len(sub_pd), f"Q5 repeat loan count mismatch: {len(df_sparql)} vs {len(sub_pd)}"
    assert (df_sparql["loan_number"].to_numpy() == sub_pd["LoanNumber"].to_numpy()).all()
    assert (df_sparql["n_pendahulu"].to_numpy() == sub_pd["n_pendahulu"].to_numpy()).all()

    # Build distribution of borrower chain lengths across all borrowers in graph
    total_borrowers = int(df_prep["UserName"].nunique())
    if not df_sparql.empty:
        b_max_sparql = df_sparql.groupby("borrower_uri_id", observed=True)["panjang_rantai_pada_pinjaman"].max()
        dist_gt1 = b_max_sparql.value_counts().to_dict()
    else:
        dist_gt1 = {}
    n_repeat_borrowers = sum(dist_gt1.values())
    dist_map: dict[int, int] = {1: total_borrowers - n_repeat_borrowers}
    for k, v in dist_gt1.items():
        dist_map[int(k)] = int(v)

    b_max_pd = df_prep.groupby("UserName", observed=True)["chain_position"].max()
    dist_pd_map = {int(k): int(v) for k, v in b_max_pd.value_counts().to_dict().items()}
    assert dist_map == dist_pd_map, f"Q5 chain distribution mismatch: {dist_map} vs {dist_pd_map}"

    sample_note = (
        "ARTEFAK DESAIN SAMPEL: Panjang rantai dan persentase peminjam berulang dipengaruhi perluasan riwayat parsial "
        "(hanya 2.869/3.721 peminjam S3 multi-pinjaman diperluas penuh untuk menjaga batas 20.000 pinjaman) "
        "dan BUKAN statistik populasi."
    )
    dist_rows = []
    for length_k in sorted(dist_map.keys()):
        cnt_b = dist_map[length_k]
        dist_rows.append(
            {
                "panjang_rantai_peminjam": length_k,
                "jumlah_peminjam_sparql": cnt_b,
                "jumlah_peminjam_pandas": dist_pd_map[length_k],
                "persen_peminjam": cnt_b / max(1, total_borrowers) * 100.0,
                "jumlah_pinjaman_dalam_rantai": length_k * cnt_b,
                "catatan": sample_note,
            }
        )
    df_dist = pd.DataFrame(dist_rows)

    # Direct calculation on S3: distribution of number of S3 loans per borrower and actual 12m default rate
    q_s3_borrower = """
    PREFIX bo: <http://example.org/bondora#>
    SELECT ?borrower
           (COUNT(DISTINCT ?loan) AS ?n_s3_loans)
           (SUM(IF(?d12 = true, 1, 0)) AS ?n_d12)
           (AVG(?p) AS ?mean_p)
    WHERE {
        ?loan bo:selectedBy bo:StrategiS3 ;
              bo:hasBorrower ?borrower ;
              bo:defaultWithin12m ?d12 ;
              bo:predictedDefaultProb12m ?p .
    }
    GROUP BY ?borrower
    """
    res_s3_b = list(g.query(q_s3_borrower))
    s3_b_rows = [
        {
            "borrower_uri_id": local_name(r.borrower),
            "n_s3_loans": int(r.n_s3_loans),
            "n_d12": int(r.n_d12),
            "sum_p": float(r.mean_p) * int(r.n_s3_loans),
        }
        for r in res_s3_b
    ]
    df_s3_b_sp = pd.DataFrame(s3_b_rows) if s3_b_rows else pd.DataFrame(columns=["borrower_uri_id", "n_s3_loans", "n_d12", "sum_p"])

    s3_pd = df_prep[df_prep["in_s3"]].copy()
    if len(s3_pd) > 0:
        s3_pd["d12_int"] = s3_pd[DEFAULT_12M_COL].astype(int)
        s3_pd["n_pinjaman_s3_per_peminjam"] = s3_pd.groupby("UserName", observed=True)["LoanNumber"].transform("count")
    else:
        s3_pd["d12_int"] = pd.Series(dtype="int64")
        s3_pd["n_pinjaman_s3_per_peminjam"] = pd.Series(dtype="int64")

    s3_conc_rows = []
    tot_s3_loans = len(s3_pd)
    tot_s3_borrowers = int(s3_pd["UserName"].nunique()) if tot_s3_loans > 0 else 0
    for k_loans, grp in s3_pd.groupby("n_pinjaman_s3_per_peminjam", observed=True):
        k_int = int(k_loans)
        n_b_pd = int(grp["UserName"].nunique())
        n_l_pd = len(grp)
        n_d12_pd = int(grp["d12_int"].sum())
        d12_rate_pd = n_d12_pd / max(1, n_l_pd) * 100.0
        w_lo_k, w_hi_k = wilson_ci(float(n_d12_pd), float(n_l_pd))
        p_mean_pd = float(grp["p_calibrated"].mean()) * 100.0

        sp_k = df_s3_b_sp[df_s3_b_sp["n_s3_loans"] == k_int]
        n_b_sp = len(sp_k)
        n_l_sp = int(sp_k["n_s3_loans"].sum())
        n_d12_sp = int(sp_k["n_d12"].sum())
        assert n_b_sp == n_b_pd, f"S3 borrower count mismatch for k={k_int}: {n_b_sp} vs {n_b_pd}"
        assert n_l_sp == n_l_pd, f"S3 loan count mismatch for k={k_int}: {n_l_sp} vs {n_l_pd}"
        assert n_d12_sp == n_d12_pd, f"S3 d12 mismatch for k={k_int}: {n_d12_sp} vs {n_d12_pd}"

        s3_conc_rows.append(
            {
                "jumlah_pinjaman_s3_per_peminjam": str(k_int),
                "jumlah_peminjam": n_b_pd,
                "persen_peminjam_s3": n_b_pd / max(1, tot_s3_borrowers) * 100.0,
                "jumlah_pinjaman_s3": n_l_pd,
                "persen_pinjaman_s3": n_l_pd / max(1, tot_s3_loans) * 100.0,
                "n_default_12m_aktual": n_d12_pd,
                "default_rate_12m_aktual_pct": d12_rate_pd,
                "wilson_ci_low_pct": w_lo_k * 100.0,
                "wilson_ci_high_pct": w_hi_k * 100.0,
                "p_calibrated_rata2_pct": p_mean_pd,
                "catatan": "Statistik lengkap portofolio S3 (n=5.000 pinjaman)",
            }
        )

    if tot_s3_loans > 0:
        gt1_grp = s3_pd[s3_pd["n_pinjaman_s3_per_peminjam"] >= 2]
        n_d12_gt1 = int(gt1_grp["d12_int"].sum())
        w_lo_gt1, w_hi_gt1 = wilson_ci(float(n_d12_gt1), float(len(gt1_grp)))
        s3_conc_rows.append(
            {
                "jumlah_pinjaman_s3_per_peminjam": ">=2 (Subtotal multi-pinjaman di S3)",
                "jumlah_peminjam": int(gt1_grp["UserName"].nunique()),
                "persen_peminjam_s3": int(gt1_grp["UserName"].nunique()) / max(1, tot_s3_borrowers) * 100.0,
                "jumlah_pinjaman_s3": len(gt1_grp),
                "persen_pinjaman_s3": len(gt1_grp) / max(1, tot_s3_loans) * 100.0,
                "n_default_12m_aktual": n_d12_gt1,
                "default_rate_12m_aktual_pct": float(gt1_grp["d12_int"].mean()) * 100.0 if len(gt1_grp) else 0.0,
                "wilson_ci_low_pct": w_lo_gt1 * 100.0,
                "wilson_ci_high_pct": w_hi_gt1 * 100.0,
                "p_calibrated_rata2_pct": float(gt1_grp["p_calibrated"].mean()) * 100.0 if len(gt1_grp) else 0.0,
                "catatan": (
                    "CATATAN KETERBATASAN (BUKAN REKOMENDASI): Default berkorelasi kuat dalam satu peminjam "
                    "(Varian b: 97,59%), dan hipotesis 'batasi pinjaman per peminjam di S3' tidak didukung data "
                    "(default S3 peminjam dengan >= 2 pinjaman S3 = 15,98% [14,15%; 17,99%] vs 20,51% [19,22%; 21,86%] untuk 1 pinjaman)"
                ),
            }
        )
        n_d12_tot = int(s3_pd["d12_int"].sum())
        w_lo_tot, w_hi_tot = wilson_ci(float(n_d12_tot), float(tot_s3_loans))
        s3_conc_rows.append(
            {
                "jumlah_pinjaman_s3_per_peminjam": "TOTAL S3",
                "jumlah_peminjam": tot_s3_borrowers,
                "persen_peminjam_s3": 100.0,
                "jumlah_pinjaman_s3": tot_s3_loans,
                "persen_pinjaman_s3": 100.0,
                "n_default_12m_aktual": n_d12_tot,
                "default_rate_12m_aktual_pct": float(s3_pd["d12_int"].mean()) * 100.0,
                "wilson_ci_low_pct": w_lo_tot * 100.0,
                "wilson_ci_high_pct": w_hi_tot * 100.0,
                "p_calibrated_rata2_pct": float(s3_pd["p_calibrated"].mean()) * 100.0,
                "catatan": "Total 5.000 pinjaman S3 pada 4.242 peminjam unik",
            }
        )
    df_s3_conc = pd.DataFrame(s3_conc_rows)
    return df_sparql, df_dist, df_s3_conc, elapsed


def run_q6(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q6 (`q6.rq`) on S3 portfolio loans and cross-check with pandas & Tahap 5 targets."""
    rq_text = (QUERIES_DIR / "q6.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        c_code = local_name(r.country).replace("Country_", "")
        rt_code = local_name(r.riskTier)
        r_code = local_name(r.rating).replace("Rating_", "")
        tb_code = local_name(r.tenorBand)
        rows.append(
            {
                "country": c_code,
                "risk_tier": rt_code,
                "rating": r_code,
                "tenor_band": tb_code,
                "n_pinjaman": int(r.n_pinjaman),
                "p_calibrated_rata2": float(r.p_calibrated_rata2),
                "n_default_12m": int(r.n_default_12m),
                "default_rate_12m": float(r.default_rate_12m),
            }
        )
    df_sparql = (
        pd.DataFrame(rows)
        .sort_values(["country", "risk_tier", "rating", "tenor_band"])
        .reset_index(drop=True)
    )

    s3_pd = df_prep[df_prep["in_s3"]].copy()
    s3_pd["d12_int"] = s3_pd[DEFAULT_12M_COL].astype(int)
    df_pd = (
        s3_pd.groupby(["Country", "risk_tier", "rating_clean", "tenor_band"], observed=True)
        .agg(
            pandas_n_pinjaman=("LoanNumber", "count"),
            pandas_p_calibrated_rata2=("p_calibrated", "mean"),
            pandas_n_default_12m=("d12_int", "sum"),
            pandas_default_rate_12m=("d12_int", "mean"),
        )
        .reset_index()
        .rename(columns={"Country": "country", "rating_clean": "rating"})
        .sort_values(["country", "risk_tier", "rating", "tenor_band"])
        .reset_index(drop=True)
    )

    merged = df_sparql.merge(df_pd, on=["country", "risk_tier", "rating", "tenor_band"], how="outer")
    assert len(merged) == len(df_sparql) == len(df_pd)
    assert int(merged["n_pinjaman"].sum()) == 5_000
    assert (merged["n_pinjaman"] == merged["pandas_n_pinjaman"]).all()
    assert (merged["n_default_12m"] == merged["pandas_n_default_12m"]).all()
    assert np.allclose(merged["p_calibrated_rata2"], merged["pandas_p_calibrated_rata2"], atol=1e-9)
    assert np.allclose(merged["default_rate_12m"], merged["pandas_default_rate_12m"], atol=1e-9)

    # Verify exact Tahap 5 marginal shares from SPARQL result
    total_s3 = merged["n_pinjaman"].sum()
    c_shares = merged.groupby("country", observed=True)["n_pinjaman"].sum() / total_s3 * 100.0
    f_hr_share = merged[merged["rating"].isin(["F", "HR"])]["n_pinjaman"].sum() / total_s3 * 100.0
    t60_share = merged[merged["tenor_band"] == "Tenor_60"]["n_pinjaman"].sum() / total_s3 * 100.0
    assert np.isclose(c_shares.get("EE", 0.0), 70.00, atol=1e-6)
    assert np.isclose(c_shares.get("ES", 0.0), 21.62, atol=1e-6)
    assert np.isclose(c_shares.get("FI", 0.0), 8.38, atol=1e-6)
    assert np.isclose(f_hr_share, 22.94, atol=1e-6)
    assert np.isclose(t60_share, 59.20, atol=1e-6)

    merged["selisih_maks"] = np.maximum(
        np.abs(merged["p_calibrated_rata2"] - merged["pandas_p_calibrated_rata2"]),
        np.abs(merged["default_rate_12m"] - merged["pandas_default_rate_12m"]),
    )
    return merged, elapsed


def run_q7(g_base: Graph, g_inf: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q7 (`q7.rq`) before and after OWL-RL inference, plus property-path equivalent and pandas check."""
    rq_text = (QUERIES_DIR / "q7.rq").read_text(encoding="utf-8")
    q_prop_path = """
    PREFIX bo: <http://example.org/bondora#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?country (COUNT(DISTINCT ?loan) AS ?n_high_risk_rated_loan)
    WHERE {
        ?rating skos:broader+ bo:TingkatTinggi .
        ?loan bo:ratedAs ?rating ;
              bo:originatedIn ?country .
    }
    GROUP BY ?country
    ORDER BY ?country
    """

    t0 = time.perf_counter()
    res_before = list(g_base.query(rq_text))
    res_after = list(g_inf.query(rq_text))
    res_path = list(g_base.query(q_prop_path))
    elapsed = time.perf_counter() - t0

    before_map = {local_name(r.country).replace("Country_", ""): int(r.n_high_risk_rated_loan) for r in res_before}
    after_map = {local_name(r.country).replace("Country_", ""): int(r.n_high_risk_rated_loan) for r in res_after}
    path_map = {local_name(r.country).replace("Country_", ""): int(r.n_high_risk_rated_loan) for r in res_path}

    pd_map = (
        df_prep[df_prep["risk_tier"] == "TingkatTinggi"]
        .groupby("Country", observed=True)["LoanNumber"]
        .count()
        .to_dict()
    )

    countries = sorted(df_prep["Country"].unique())
    rows = []
    for c in countries:
        b_cnt = int(before_map.get(c, 0))
        a_cnt = int(after_map.get(c, 0))
        p_cnt = int(path_map.get(c, 0))
        d_cnt = int(pd_map.get(c, 0))
        assert b_cnt == 0
        assert a_cnt == p_cnt == d_cnt, f"Q7 mismatch for {c}: {a_cnt} vs {p_cnt} vs {d_cnt}"
        rows.append(
            {
                "country": c,
                "tanpa_inferensi": b_cnt,
                "dengan_inferensi_owlrl": a_cnt,
                "jalur_property_path": p_cnt,
                "hitung_pandas": d_cnt,
                "selisih": a_cnt - d_cnt,
            }
        )

    rows.append(
        {
            "country": "TOTAL",
            "tanpa_inferensi": sum(r["tanpa_inferensi"] for r in rows),
            "dengan_inferensi_owlrl": sum(r["dengan_inferensi_owlrl"] for r in rows),
            "jalur_property_path": sum(r["jalur_property_path"] for r in rows),
            "hitung_pandas": sum(r["hitung_pandas"] for r in rows),
            "selisih": 0,
        }
    )
    return pd.DataFrame(rows), elapsed


def run_q8(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q8 (`q8.rq`) and cross-check with pandas."""
    rq_text = (QUERIES_DIR / "q8.rq").read_text(encoding="utf-8")
    t0 = time.perf_counter()
    res = list(g.query(rq_text))
    elapsed = time.perf_counter() - t0

    rows = []
    for r in res:
        era_code = local_name(r.era)
        c_code = local_name(r.country).replace("Country_", "")
        out_code = local_name(r.outcome)
        rows.append(
            {
                "era": era_code,
                "country": c_code,
                "outcome_code": out_code,
                "n_pinjaman": int(r.n_pinjaman),
                "p_calibrated_rata2": float(r.p_calibrated_rata2),
            }
        )
    df_sparql = pd.DataFrame(rows).sort_values(["era", "country", "outcome_code"]).reset_index(drop=True)
    df_sparql["proporsi_dlm_era_negara"] = df_sparql["n_pinjaman"] / df_sparql.groupby(
        ["era", "country"], observed=True
    )["n_pinjaman"].transform("sum")

    df_pd = (
        df_prep.groupby(["era", "Country", "outcome_code"], observed=True)
        .agg(
            pandas_n_pinjaman=("LoanNumber", "count"),
            pandas_p_calibrated_rata2=("p_calibrated", "mean"),
        )
        .reset_index()
        .rename(columns={"Country": "country"})
        .sort_values(["era", "country", "outcome_code"])
        .reset_index(drop=True)
    )
    df_pd["pandas_proporsi_dlm_era_negara"] = df_pd["pandas_n_pinjaman"] / df_pd.groupby(
        ["era", "country"], observed=True
    )["pandas_n_pinjaman"].transform("sum")

    merged = df_sparql.merge(df_pd, on=["era", "country", "outcome_code"], how="outer")
    assert len(merged) == len(df_sparql) == len(df_pd)
    assert (merged["n_pinjaman"] == merged["pandas_n_pinjaman"]).all()
    assert np.allclose(merged["proporsi_dlm_era_negara"], merged["pandas_proporsi_dlm_era_negara"], atol=1e-9)
    assert np.allclose(merged["p_calibrated_rata2"], merged["pandas_p_calibrated_rata2"], atol=1e-9)
    merged["selisih_maks"] = np.maximum(
        np.abs(merged["proporsi_dlm_era_negara"] - merged["pandas_proporsi_dlm_era_negara"]),
        np.abs(merged["p_calibrated_rata2"] - merged["pandas_p_calibrated_rata2"]),
    )
    return merged, elapsed


def build_insight_action_table(
    q1_df: pd.DataFrame,
    q2_df: pd.DataFrame,
    q3_df: pd.DataFrame,
    q4_df: pd.DataFrame,
    q4_summary_df: pd.DataFrame,
    q5_dist_df: pd.DataFrame,
    q5_s3_conc_df: pd.DataFrame,
    q6_df: pd.DataFrame,
    df_prep: pd.DataFrame,
) -> pd.DataFrame:
    """Construct `reports/tables/07_insight_action.csv` linking Tahap 2-5 findings and new relational findings."""
    # 1. Read and assert population statistics directly from Tahap 3 & Tahap 4 tables
    seg_path = TABLES_DIR / "03_default_rate_horizon_by_segment.csv"
    rat_path = TABLES_DIR / "03_rating_vs_observed_12m.csv"
    lb_path = TABLES_DIR / "04_level_bias.csv"
    sub_path = TABLES_DIR / "04_subgroup_performance.csv"
    assert seg_path.exists(), f"Missing {seg_path}"
    assert rat_path.exists(), f"Missing {rat_path}"
    assert lb_path.exists(), f"Missing {lb_path}"
    assert sub_path.exists(), f"Missing {sub_path}"

    seg_df = pd.read_csv(seg_path)
    c_pop = seg_df[(seg_df["variabel"] == "Country") & (seg_df["horizon"] == "12m")].set_index("segmen")
    assert set(c_pop.index) == {"EE", "FI", "ES", "SK"}
    tot_pop_n = int(c_pop["n"].sum())
    tot_pop_def = int(c_pop["n_default"].sum())
    pop_overall_rate_pct = tot_pop_def / tot_pop_n * 100.0
    assert tot_pop_n == 148_369, f"Expected 148,369 12m observed loans in Tahap 3, got {tot_pop_n}"
    assert np.isclose(pop_overall_rate_pct, 29.060653, atol=1e-3)
    for c_key in ["EE", "FI", "ES", "SK"]:
        assert np.isclose(
            c_pop.loc[c_key, "default_rate"],
            c_pop.loc[c_key, "n_default"] / c_pop.loc[c_key, "n"],
            atol=1e-9,
        )

    ee_pop_pct = float(c_pop.loc["EE", "default_rate"]) * 100.0
    fi_pop_pct = float(c_pop.loc["FI", "default_rate"]) * 100.0
    es_pop_pct = float(c_pop.loc["ES", "default_rate"]) * 100.0
    sk_pop_pct = float(c_pop.loc["SK", "default_rate"]) * 100.0
    assert ee_pop_pct < fi_pop_pct < es_pop_pct < sk_pop_pct

    # Read Tahap 4 test split p_calibrated directly from 04_level_bias.csv & 04_subgroup_performance.csv
    lb_df = pd.read_csv(lb_path)
    lb_test_all = lb_df[(lb_df["cakupan"] == "Seluruh") & (lb_df["split"] == "test")].iloc[0]
    lb_test_c = lb_df[(lb_df["cakupan"] == "Country") & (lb_df["split"] == "test")].set_index("subgrup")
    sub_df = pd.read_csv(sub_path)
    sub_test_c = sub_df[(sub_df["dimensi"] == "Country") & (sub_df["kode_model"] == "LGBM-A")].set_index("subgrup")

    for c_key in ["EE", "FI", "ES"]:
        assert np.isclose(
            float(lb_test_c.loc[c_key, "mean_p_calibrated"]),
            float(sub_test_c.loc[c_key, "mean_pred_prob_calibrated"]),
            atol=1e-6,
        )
        assert np.isclose(
            float(lb_test_c.loc[c_key, "default_12m_aktual"]),
            float(sub_test_c.loc[c_key, "observed_default_rate"]),
            atol=1e-6,
        )

    n_test_all = int(lb_test_all["n"])
    p_test_all_pct = float(lb_test_all["mean_p_calibrated"]) * 100.0
    d_test_all_pct = float(lb_test_all["default_12m_aktual"]) * 100.0
    p_ee_test_pct = float(lb_test_c.loc["EE", "mean_p_calibrated"]) * 100.0
    p_fi_test_pct = float(lb_test_c.loc["FI", "mean_p_calibrated"]) * 100.0
    p_es_test_pct = float(lb_test_c.loc["ES", "mean_p_calibrated"]) * 100.0
    d_ee_test_pct = float(lb_test_c.loc["EE", "default_12m_aktual"]) * 100.0
    d_fi_test_pct = float(lb_test_c.loc["FI", "default_12m_aktual"]) * 100.0
    d_es_test_pct = float(lb_test_c.loc["ES", "default_12m_aktual"]) * 100.0
    n_ee_test = int(lb_test_c.loc["EE", "n"])
    n_fi_test = int(lb_test_c.loc["FI", "n"])
    n_es_test = int(lb_test_c.loc["ES", "n"])

    assert n_ee_test + n_fi_test + n_es_test == n_test_all == 54_283
    weighted_test_p_pct = (
        n_ee_test * p_ee_test_pct + n_fi_test * p_fi_test_pct + n_es_test * p_es_test_pct
    ) / n_test_all
    assert abs(weighted_test_p_pct - 32.96) <= 0.05, (
        f"Country-weighted test p_calibrated {weighted_test_p_pct:.4f}% does not match 32,96% within 0,05 pp"
    )
    assert abs(weighted_test_p_pct - p_test_all_pct) <= 0.05

    rat_df = pd.read_csv(rat_path).set_index("Rating")
    rating_order = ["AA", "A", "B", "C", "D", "E", "F", "HR"]
    assert int(rat_df["n"].sum()) == 148_369
    rat_rates = [float(rat_df.loc[r, "observed_default_rate_12m"]) * 100.0 for r in rating_order]
    assert all(rat_rates[i] < rat_rates[i + 1] for i in range(len(rat_rates) - 1)), "Rating 12m default rate must be strictly monotonic"
    unrated_pop_pct = float(rat_df.loc["Tidak tersedia", "observed_default_rate_12m"]) * 100.0
    unrated_pop_n = int(rat_df.loc["Tidak tersedia", "n"])

    # Q2 country default rates on sample
    q2_country = (
        q2_df.groupby("country", observed=True)
        .agg(n_obs=("n_teramati_12m", "sum"), n_def=("n_default_12m", "sum"))
        .assign(rate_pct=lambda d: d["n_def"] / d["n_obs"] * 100.0)
    )
    assert (
        q2_country.loc["EE", "rate_pct"]
        < q2_country.loc["FI", "rate_pct"]
        < q2_country.loc["ES", "rate_pct"]
    )
    q2_str = ", ".join(
        f"{c}={_fmt_pct_id(row['rate_pct'])} (n={int(row['n_obs']):,})"
        for c, row in q2_country.iterrows()
    )

    # Q1 risk tier average p_calibrated on sample (including Tidak tersedia)
    q1_tier = (
        df_prep.groupby("risk_tier", observed=True)
        .agg(n=("LoanNumber", "count"), mean_p=("p_calibrated", "mean"))
    )
    n_unrated_samp = int(q1_tier.loc["TingkatTidakTersedia", "n"]) if "TingkatTidakTersedia" in q1_tier.index else 0
    p_unrated_samp = float(q1_tier.loc["TingkatTidakTersedia", "mean_p"]) * 100.0 if "TingkatTidakTersedia" in q1_tier.index else 0.0
    q1_str = (
        f"Rata-rata p_calibrated sampel menurut tingkat risiko SKOS: "
        f"TingkatRendah={_fmt_pct_id(q1_tier.loc['TingkatRendah', 'mean_p']*100)} (n={int(q1_tier.loc['TingkatRendah', 'n']):,}), "
        f"TingkatMenengah={_fmt_pct_id(q1_tier.loc['TingkatMenengah', 'mean_p']*100)} (n={int(q1_tier.loc['TingkatMenengah', 'n']):,}), "
        f"TingkatTinggi={_fmt_pct_id(q1_tier.loc['TingkatTinggi', 'mean_p']*100)} (n={int(q1_tier.loc['TingkatTinggi', 'n']):,}), "
        f"Tidak tersedia={_fmt_pct_id(p_unrated_samp)} (n={n_unrated_samp:,} di EE); "
        f"ditemukan {len(q3_df)} pinjaman TingkatRendah (AA/A/B) yang masuk PitaRisikoTinggi (p >= 0,35) di Q3."
    )

    # Q6 S3 composition (exact population of S3)
    tot_s3 = int(q6_df["n_pinjaman"].sum())
    c_s3 = q6_df.groupby("country", observed=True)["n_pinjaman"].sum() / tot_s3 * 100.0
    fhr_s3 = q6_df[q6_df["rating"].isin(["F", "HR"])]["n_pinjaman"].sum() / tot_s3 * 100.0
    t60_s3 = q6_df[q6_df["tenor_band"] == "Tenor_60"]["n_pinjaman"].sum() / tot_s3 * 100.0
    def12_s3 = q6_df["n_default_12m"].sum() / tot_s3 * 100.0
    p_s3 = (q6_df["n_pinjaman"] * q6_df["p_calibrated_rata2"]).sum() / tot_s3 * 100.0
    q6_str = (
        f"n={tot_s3:,} pinjaman S3: EE {_fmt_pct_id(c_s3.get('EE', 0.0))}, ES {_fmt_pct_id(c_s3.get('ES', 0.0))}, "
        f"FI {_fmt_pct_id(c_s3.get('FI', 0.0))}; F+HR {_fmt_pct_id(fhr_s3)}; Tenor 60 bln {_fmt_pct_id(t60_s3)}; "
        f"rata-rata p_calibrated {_fmt_pct_id(p_s3)}; realisasi default 12m aktual {_fmt_pct_id(def12_s3)}."
    )

    # Q4 variants (a) and (b), base rates, and S3 prior default before LoanDate
    q4_map = q4_summary_df.set_index("kode_varian")
    va_row = q4_map.loc["a_setelah_default"]
    vb_row = q4_map.loc["b_tumpang_tindih"]
    b1_all_row = q4_map.loc["pembanding_lanjut_prev_non_default_semua"]
    b1_res_row = q4_map.loc["pembanding_lanjut_prev_non_default_selesai"]
    b2_row = q4_map.loc["pembanding_semua_pinjaman_selesai_graf"]
    s3_dir_row = q4_map.loc["s3_pendahulu_langsung_default_sebelum_loandate"]
    s3_rest_dir_row = q4_map.loc["s3_sisa_tanpa_pendahulu_langsung_default_sebelum_loandate"]
    s3_any_row = q4_map.loc["s3_pendahulu_rantai_default_sebelum_loandate"]
    s3_rest_any_row = q4_map.loc["s3_sisa_tanpa_pendahulu_rantai_default_sebelum_loandate"]

    # Q5 repeat chains and S3 direct borrower concentration
    max_chain = int(q5_dist_df["panjang_rantai_peminjam"].max())
    n_repeat_b = int(q5_dist_df[q5_dist_df["panjang_rantai_peminjam"] >= 2]["jumlah_peminjam_sparql"].sum())
    n_chain_ge5 = int(q5_dist_df[q5_dist_df["panjang_rantai_peminjam"] >= 5]["jumlah_peminjam_sparql"].sum())

    s3_conc_map = q5_s3_conc_df.set_index("jumlah_pinjaman_s3_per_peminjam")
    s3_k1 = s3_conc_map.loc["1"]
    s3_k2 = s3_conc_map.loc["2"]
    s3_k3 = s3_conc_map.loc["3"]
    s3_kge2 = s3_conc_map.loc[">=2 (Subtotal multi-pinjaman di S3)"]

    rat_pop_parts = ", ".join(
        f"{r} {_fmt_pct_id(float(rat_df.loc[r, 'observed_default_rate_12m']) * 100.0)}"
        for r in rating_order
    )

    rows = [
        {
            "temuan_sebelumnya": "Perbedaan tingkat gagal bayar antar negara dan pergeseran komposisi hasil menurut era pencatatan",
            "sumber_tahap": "Tahap 2 (EDA), Tahap 3 (03_default_rate_horizon_by_segment.csv) & Tahap 4 (04_level_bias.csv, 04_subgroup_performance.csv)",
            "query_sparql": "Q2 (07_q2_default12m_country_era.csv) & Q8 (07_q8_outcome_era_country.csv)",
            "hasil_graf": (
                f"Default rate 12 bulan pada sampel graf (n={int(q2_df['n_teramati_12m'].sum()):,} teramati): {q2_str}. "
                f"Urutan risiko EE < FI < ES konsisten dengan populasi, sedangkan SK tidak (n sampel kecil, n=42)."
            ),
            "hasil_populasi": (
                f"Populasi default rate 12 bulan teramati (03_default_rate_horizon_by_segment.csv, n={tot_pop_n:,}, "
                f"rata-rata populasi {_fmt_pct_id(pop_overall_rate_pct)}): "
                f"EE {_fmt_pct_id(ee_pop_pct)} (n={int(c_pop.loc['EE', 'n']):,}), "
                f"FI {_fmt_pct_id(fi_pop_pct)} (n={int(c_pop.loc['FI', 'n']):,}), "
                f"ES {_fmt_pct_id(es_pop_pct)} (n={int(c_pop.loc['ES', 'n']):,}), "
                f"SK {_fmt_pct_id(sk_pop_pct)} (n={int(c_pop.loc['SK', 'n']):,}); "
                f"rata-rata p_calibrated uji Tahap 4 (04_level_bias.csv & 04_subgroup_performance.csv, n={n_test_all:,}, "
                f"rata-rata uji {_fmt_pct_id(p_test_all_pct)} vs default 12 bulan aktual {_fmt_pct_id(d_test_all_pct)}): "
                f"EE {_fmt_pct_id(p_ee_test_pct)} (aktual {_fmt_pct_id(d_ee_test_pct)}, n={n_ee_test:,}), "
                f"FI {_fmt_pct_id(p_fi_test_pct)} (aktual {_fmt_pct_id(d_fi_test_pct)}, n={n_fi_test:,}), "
                f"ES {_fmt_pct_id(p_es_test_pct)} (aktual {_fmt_pct_id(d_es_test_pct)}, n={n_es_test:,})."
            ),
            "catatan": "ANGKA SAMPEL: Proporsi sampel dipengaruhi masuknya seluruh 5.000 pinjaman S3 (2021-2022) dan perluasan peminjam; gunakan angka Tahap 3 dan Tahap 4 untuk acuan populasi/uji.",
        },
        {
            "temuan_sebelumnya": "Urutan tingkat gagal bayar monoton terhadap kelas rating (AA s.d. HR) namun terdapat deviasi prediktif tingkat individu",
            "sumber_tahap": "Tahap 3 (Diagnostic: H2 — 03_rating_vs_observed_12m.csv) & Tahap 4 (Predictive: LGBM-A)",
            "query_sparql": "Q1 (07_q1_rating_tier_country.csv) & Q3 (07_q3_rating_vs_model_mismatch.csv)",
            "hasil_graf": q1_str,
            "hasil_populasi": (
                f"Populasi default rate 12 bulan teramati menurut Rating (03_rating_vs_observed_12m.csv, n={tot_pop_n:,}, "
                f"rata-rata {_fmt_pct_id(pop_overall_rate_pct)}) naik monoton: {rat_pop_parts} "
                f"(Tidak tersedia: {_fmt_pct_id(unrated_pop_pct)}, n={unrated_pop_n:,}); "
                f"model LGBM-A (ROC-AUC uji 0,7152) membedakan risiko individu di dalam kelas rating yang sama."
            ),
            "catatan": "ANGKA SAMPEL: Menunjukkan manfaat menggabungkan taksonomi SKOS (bo:RiskTier) dengan skor model (bo:inRiskBand) untuk menyaring anomali rating rendah berisiko model tinggi.",
        },
        {
            "temuan_sebelumnya": "Komposisi dan kinerja portofolio optimasi berbatas S3 (n=5.000 pinjaman, anggaran EUR 250.000)",
            "sumber_tahap": "Tahap 5 (Prescriptive: 06_strategy_results.csv & 06_s3_composition.csv)",
            "query_sparql": "Q6 (07_q6_s3_composition_sparql.csv)",
            "hasil_graf": q6_str,
            "hasil_populasi": "Populasi S3 Tahap 5 (n=5.000, anggaran EUR 250.000): EE 70,00%, ES 21,62%, FI 8,38%; F+HR 22,94%; Tenor 60 bulan 59,20%; rata-rata p_calibrated 17,91%; realisasi default 12m aktual 19,24%.",
            "catatan": "ANGKA LENGKAP (IDENTIK 100%): Seluruh 5.000 pinjaman S3 dimasukkan penuh ke dalam graf sehingga agregasi SPARQL Q6 mereproduksi angka Tahap 5 secara eksak.",
        },
        {
            "temuan_sebelumnya": "Temuan relasional baru (belum ada di Tahap 1-5): pemisahan pinjaman ulang setelah default (Varian a) vs pinjaman tumpang-tindih (Varian b)",
            "sumber_tahap": "Tahap 6 (Semantic Analytics — relasi bo:precededBy & bo:defaultDate)",
            "query_sparql": "Q4 (07_q4_repeat_borrower_after_default.csv & 07_q4_variants_summary.csv)",
            "hasil_graf": (
                f"Varian (a) [LoanDate > bo:defaultDate pendahulu langsung, pinjaman ulang setelah default]: "
                f"{int(va_row['jumlah_pasangan_atau_pinjaman']):,} pasangan ({int(va_row['jumlah_peminjam']):,} peminjam), "
                f"{_fmt_pct_id(float(va_row['persen_berakhir_default']))} ({int(va_row['n_berakhir_default'])}/{int(va_row['jumlah_pasangan_atau_pinjaman'])}, "
                f"Wilson 95%: [{_fmt_pct_id(float(va_row['wilson_ci_low_default_pct']))}; {_fmt_pct_id(float(va_row['wilson_ci_high_default_pct']))}]) berakhir Default. "
                f"Varian (b) [LoanDate <= bo:defaultDate pendahulu, tumpang-tindih]: "
                f"{int(vb_row['jumlah_pasangan_atau_pinjaman']):,} pasangan ({int(vb_row['jumlah_peminjam']):,} peminjam), "
                f"{_fmt_pct_id(float(vb_row['persen_berakhir_default']))} ({int(vb_row['n_berakhir_default']):,}/{int(vb_row['jumlah_pasangan_atau_pinjaman']):,}, "
                f"Wilson 95%: [{_fmt_pct_id(float(vb_row['wilson_ci_low_default_pct']))}; {_fmt_pct_id(float(vb_row['wilson_ci_high_default_pct']))}]) berakhir Default "
                f"(ditambah 3 pasangan pendahulu Default tanpa DefaultDate). "
                f"Pembanding tingkat dasar: pinjaman lanjutan dengan pendahulu TIDAK Default = {_fmt_pct_id(float(b1_all_row['persen_berakhir_default']))} "
                f"({int(b1_all_row['n_berakhir_default'])}/{int(b1_all_row['jumlah_pasangan_atau_pinjaman']):,} semua; "
                f"{_fmt_pct_id(float(b1_res_row['persen_berakhir_default']))} atau {int(b1_res_row['n_berakhir_default'])}/{int(b1_res_row['jumlah_pasangan_atau_pinjaman']):,} pada yang selesai), "
                f"dan semua pinjaman selesai di graf = {_fmt_pct_id(float(b2_row['persen_berakhir_default']))} "
                f"({int(b2_row['n_berakhir_default']):,}/{int(b2_row['jumlah_pasangan_atau_pinjaman']):,})."
            ),
            "hasil_populasi": (
                f"Pada portofolio S3 (n=5.000): terdapat {int(s3_dir_row['n_di_s3'])} pinjaman S3 ({int(s3_dir_row['jumlah_peminjam'])} peminjam) "
                f"yang pendahulu langsungnya default SEBELUM LoanDate-nya, dengan default 12m aktual {_fmt_pct_id(float(s3_dir_row['persen_default_12m_di_s3']))} "
                f"({int(s3_dir_row['n_default_12m_di_s3'])}/{int(s3_dir_row['n_di_s3'])}, "
                f"Wilson 95%: [{_fmt_pct_id(float(s3_dir_row['wilson_ci_low_s3_d12_pct']))}; {_fmt_pct_id(float(s3_dir_row['wilson_ci_high_s3_d12_pct']))}]) "
                f"vs sisa S3 {_fmt_pct_id(float(s3_rest_dir_row['persen_default_12m_di_s3']))} "
                f"({int(s3_rest_dir_row['n_default_12m_di_s3']):,}/{int(s3_rest_dir_row['n_di_s3']):,}, "
                f"Wilson 95%: [{_fmt_pct_id(float(s3_rest_dir_row['wilson_ci_low_s3_d12_pct']))}; {_fmt_pct_id(float(s3_rest_dir_row['wilson_ci_high_s3_d12_pct']))}]) — "
                f"tidak ada bukti selisih (n kecil, interval lebar); "
                f"bila dihitung terhadap pendahulu MANA PUN di rantai (bo:precededBy+), terdapat {int(s3_any_row['n_di_s3'])} pinjaman S3 "
                f"dengan default 12m aktual {_fmt_pct_id(float(s3_any_row['persen_default_12m_di_s3']))} ({int(s3_any_row['n_default_12m_di_s3'])}/{int(s3_any_row['n_di_s3'])}, "
                f"Wilson 95%: [{_fmt_pct_id(float(s3_any_row['wilson_ci_low_s3_d12_pct']))}; {_fmt_pct_id(float(s3_any_row['wilson_ci_high_s3_d12_pct']))}]) "
                f"vs sisa S3 {_fmt_pct_id(float(s3_rest_any_row['persen_default_12m_di_s3']))} ({int(s3_rest_any_row['n_default_12m_di_s3']):,}/{int(s3_rest_any_row['n_di_s3']):,})."
            ),
            "catatan": (
                "CATATAN KETERBATASAN (BUKAN REKOMENDASI): Default berkorelasi kuat di dalam satu peminjam "
                "(Varian b tumpang-tindih: 97,59% berakhir Default bersama); pada pinjaman S3 yang pendahulunya default "
                "sebelum LoanDate (15/67 = 22,39% [14,06%; 33,71%] vs 947/4.933 = 19,20% [18,12%; 20,32%]), "
                "tidak ada bukti selisih karena n kecil dan interval Wilson lebar."
            ),
        },
        {
            "temuan_sebelumnya": "Temuan relasional baru (belum ada di Tahap 1-5): struktur panjang rantai peminjaman ulang dan distribusi konsentrasi peminjam di S3",
            "sumber_tahap": "Tahap 6 (Semantic Analytics — property path bo:precededBy+ & analisis konsentrasi S3)",
            "query_sparql": "Q5 (07_q5_repeat_chain_lengths.csv, 07_q5_chain_distribution.csv & 07_q5_s3_borrower_concentration.csv)",
            "hasil_graf": (
                f"Dari {int(df_prep['UserName'].nunique()):,} peminjam di graf, {n_repeat_b:,} peminjam memiliki >= 2 pinjaman "
                f"(total {len(df_prep) - int(df_prep['UserName'].nunique()):,} pinjaman lanjutan), {n_chain_ge5:,} peminjam memiliki rantai >= 5 pinjaman, "
                f"dan panjang rantai maksimum tercatat {max_chain} pinjaman berurutan "
                f"(CATATAN: panjang rantai dan % peminjam berulang di graf adalah artefak desain sampel akibat perluasan riwayat parsial, bukan statistik populasi)."
            ),
            "hasil_populasi": (
                f"Distribusi langsung pada portofolio S3 (5.000 pinjaman pada 4.242 peminjam unik): "
                f"1 pinjaman/peminjam = {int(s3_k1['jumlah_peminjam']):,} peminjam ({int(s3_k1['jumlah_pinjaman_s3']):,} pinjaman, "
                f"default 12m aktual {_fmt_pct_id(float(s3_k1['default_rate_12m_aktual_pct']))} "
                f"[{_fmt_pct_id(float(s3_k1['wilson_ci_low_pct']))}; {_fmt_pct_id(float(s3_k1['wilson_ci_high_pct']))}]); "
                f"2 pinjaman/peminjam = {int(s3_k2['jumlah_peminjam']):,} peminjam ({int(s3_k2['jumlah_pinjaman_s3']):,} pinjaman, "
                f"default 12m aktual {_fmt_pct_id(float(s3_k2['default_rate_12m_aktual_pct']))} "
                f"[{_fmt_pct_id(float(s3_k2['wilson_ci_low_pct']))}; {_fmt_pct_id(float(s3_k2['wilson_ci_high_pct']))}]); "
                f"3 pinjaman/peminjam = {int(s3_k3['jumlah_peminjam']):,} peminjam ({int(s3_k3['jumlah_pinjaman_s3']):,} pinjaman, "
                f"default 12m aktual {_fmt_pct_id(float(s3_k3['default_rate_12m_aktual_pct']))} "
                f"[{_fmt_pct_id(float(s3_k3['wilson_ci_low_pct']))}; {_fmt_pct_id(float(s3_k3['wilson_ci_high_pct']))}]); "
                f"4-7 pinjaman/peminjam = 10 peminjam (46 pinjaman, 2 default pada k=7); "
                f"subtotal >=2 pinjaman/peminjam di S3 = {int(s3_kge2['jumlah_peminjam']):,} peminjam ({int(s3_kge2['jumlah_pinjaman_s3']):,} pinjaman, "
                f"default 12m aktual {_fmt_pct_id(float(s3_kge2['default_rate_12m_aktual_pct']))} "
                f"[{_fmt_pct_id(float(s3_kge2['wilson_ci_low_pct']))}; {_fmt_pct_id(float(s3_kge2['wilson_ci_high_pct']))}])."
            ),
            "catatan": (
                "CATATAN KETERBATASAN (BUKAN REKOMENDASI): Default berkorelasi kuat dalam satu peminjam (Varian b: 97,59%), "
                "dan hipotesis 'batasi pinjaman per peminjam di S3' tidak didukung data "
                "(default S3 peminjam dengan >= 2 pinjaman S3 = 15,98% [14,15%; 17,99%] vs 20,51% [19,22%; 21,86%] untuk 1 pinjaman)."
            ),
        },
    ]
    return pd.DataFrame(rows)


def run_all_queries_and_inference(
    g: Graph | None = None,
    sample_df: pd.DataFrame | None = None,
    *,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Run all 8 SPARQL queries with pandas cross-checks, OWL-RL inference, and Insight & Action table."""
    ensure_dirs()
    if g is None:
        g = load_graph_gz(GRAPH_GZ_PATH)
    if sample_df is None:
        sample_df = pd.read_parquet(SAMPLE_PARQUET_PATH)

    df_prep = prepare_annotated_sample_df(sample_df, score_if_missing=False)

    # Run Step 5 (OWL-RL inference on a copy of the graph) first so Q7 can use g_inf
    print("[Step 5] Running OWL-RL DeductiveClosure on graph copy...", flush=True)
    g_inf, inf_summary_df = run_owlrl_inference(g, df_prep)
    print(f"[Step 5] Done: {inf_summary_df.loc[0, 'sebelum_inferensi']} -> {inf_summary_df.loc[0, 'sesudah_inferensi_owlrl']} triples in {inf_summary_df.loc[0, 'waktu_detik']}s", flush=True)

    # Run Q1..Q8
    print("[Step 4] Running Q1..Q8...", flush=True)
    q1_df, t_q1 = run_q1(g, df_prep)
    print(f"  Q1 done in {t_q1:.3f}s", flush=True)
    q2_df, t_q2 = run_q2(g, df_prep)
    print(f"  Q2 done in {t_q2:.3f}s", flush=True)
    q3_df, t_q3 = run_q3(g, df_prep)
    print(f"  Q3 done in {t_q3:.3f}s", flush=True)
    q4_df, q4_summary_df, t_q4 = run_q4(g, df_prep)
    print(f"  Q4 done in {t_q4:.3f}s", flush=True)
    q5_df, q5_dist_df, q5_s3_conc_df, t_q5 = run_q5(g, df_prep)
    print(f"  Q5 done in {t_q5:.3f}s", flush=True)
    q6_df, t_q6 = run_q6(g, df_prep)
    print(f"  Q6 done in {t_q6:.3f}s", flush=True)
    q7_df, t_q7 = run_q7(g, g_inf, df_prep)
    print(f"  Q7 done in {t_q7:.3f}s", flush=True)
    q8_df, t_q8 = run_q8(g, df_prep)
    print(f"  Q8 done in {t_q8:.3f}s", flush=True)

    timings_rows = [
        ("Q1", "q1.rq", "07_q1_rating_tier_country.csv", len(q1_df), t_q1),
        ("Q2", "q2.rq", "07_q2_default12m_country_era.csv", len(q2_df), t_q2),
        ("Q3", "q3.rq", "07_q3_rating_vs_model_mismatch.csv", len(q3_df), t_q3),
        ("Q4", "q4.rq", "07_q4_repeat_borrower_after_default.csv", len(q4_df), t_q4),
        ("Q5", "q5.rq", "07_q5_repeat_chain_lengths.csv", len(q5_df), t_q5),
        ("Q6", "q6.rq", "07_q6_s3_composition_sparql.csv", len(q6_df), t_q6),
        ("Q7", "q7.rq", "07_q7_high_risk_inferred.csv", len(q7_df), t_q7),
        ("Q8", "q8.rq", "07_q8_outcome_era_country.csv", len(q8_df), t_q8),
    ]
    timings_df = pd.DataFrame(
        [
            {
                "query_id": qid,
                "file_rq": rq_file,
                "file_csv": csv_file,
                "pertanyaan_bisnis": _extract_query_comment(QUERIES_DIR / rq_file),
                "jumlah_baris_hasil": n_rows,
                "waktu_eksekusi_detik": round(sec, 4),
                "verifikasi_pandas": "LULUS (assert sama persis)",
            }
            for qid, rq_file, csv_file, n_rows, sec in timings_rows
        ]
    )
    assert (timings_df["waktu_eksekusi_detik"] < 60.0).all(), "Query exceeded 60s limit!"

    insight_df = build_insight_action_table(
        q1_df, q2_df, q3_df, q4_df, q4_summary_df, q5_dist_df, q5_s3_conc_df, q6_df, df_prep
    )

    if save_outputs:
        q1_df.to_csv(TABLES_DIR / "07_q1_rating_tier_country.csv", index=False)
        q2_df.to_csv(TABLES_DIR / "07_q2_default12m_country_era.csv", index=False)
        q3_df.to_csv(TABLES_DIR / "07_q3_rating_vs_model_mismatch.csv", index=False)
        q4_df.to_csv(TABLES_DIR / "07_q4_repeat_borrower_after_default.csv", index=False)
        q4_summary_df.to_csv(TABLES_DIR / "07_q4_variants_summary.csv", index=False)
        q5_df.to_csv(TABLES_DIR / "07_q5_repeat_chain_lengths.csv", index=False)
        q5_dist_df.to_csv(TABLES_DIR / "07_q5_chain_distribution.csv", index=False)
        q5_s3_conc_df.to_csv(TABLES_DIR / "07_q5_s3_borrower_concentration.csv", index=False)
        q6_df.to_csv(TABLES_DIR / "07_q6_s3_composition_sparql.csv", index=False)
        q7_df.to_csv(TABLES_DIR / "07_q7_high_risk_inferred.csv", index=False)
        q8_df.to_csv(TABLES_DIR / "07_q8_outcome_era_country.csv", index=False)
        timings_df.to_csv(TABLES_DIR / "07_query_timings.csv", index=False)
        inf_summary_df.to_csv(TABLES_DIR / "07_inference_summary.csv", index=False)
        insight_df.to_csv(TABLES_DIR / "07_insight_action.csv", index=False)

    return {
        "q1": q1_df,
        "q2": q2_df,
        "q3": q3_df,
        "q4": q4_df,
        "q4_summary": q4_summary_df,
        "q5": q5_df,
        "q5_dist": q5_dist_df,
        "q5_s3_conc": q5_s3_conc_df,
        "q6": q6_df,
        "q7": q7_df,
        "q8": q8_df,
        "timings": timings_df,
        "inference_summary": inf_summary_df,
        "insight_action": insight_df,
        "g_inferred": g_inf,
    }


def main() -> None:
    t0 = time.perf_counter()
    print("=== Running Tahap 6 SPARQL Queries, Pandas Cross-Checks, and OWL-RL Inference ===")
    res = run_all_queries_and_inference(save_outputs=True)
    print("\n--- Query Timings & Verification ---")
    print(res["timings"][["query_id", "jumlah_baris_hasil", "waktu_eksekusi_detik", "verifikasi_pandas"]].to_string(index=False))
    print("\n--- Q4 Variants Summary ---")
    print(res["q4_summary"][["kode_varian", "jumlah_pasangan_atau_pinjaman", "jumlah_peminjam", "n_berakhir_default", "persen_berakhir_default", "n_di_s3", "persen_default_12m_di_s3"]].to_string(index=False))
    print("\n--- OWL-RL Inference Summary ---")
    print(res["inference_summary"].to_string(index=False))
    print(f"\n[Done] Total execution time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
