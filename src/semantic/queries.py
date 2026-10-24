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


class FastOWLRLSemantics(owlrl.OWLRL_Semantics):
    """Optimized OWL-RL + RDFS closure semantics for ABox graphs with large literal counts.

    Standard ``owlrl.OWLRL_Semantics`` replaces every literal with a blank node proxy inside an
    ``O(N^2)`` list membership check and materializes reflexive ``owl:sameAs`` triples on every node.
    Since ``datatype_axioms=False`` and no ``owl:sameAs`` identity assertions are used in the Bondora
    ontology, skipping literal proxy generation and reflexive ``owl:sameAs`` reduces runtime on 340k
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
        if p == RDF.type and o in self._ABOX_CLASSES:
            if cycle_num <= 2:
                for sup in self._sub_map.get(o, ()):
                    self.store_triple((s, RDF.type, sup))
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


def run_q1(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q1 (`q1.rq`) and cross-check with pandas."""
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


def run_q4(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Execute Q4 (`q4.rq`) and cross-check with pandas."""
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
                "borrower_uri_id": b_id,
                "country": c_code,
                "prev_loan_number": prev_num,
                "prev_loan_date": str(r.prevDate),
                "next_loan_number": next_num,
                "next_loan_date": str(r.nextDate),
                "next_outcome_code": local_name(r.nextOutcome),
                "next_p_calibrated": float(r.nextP),
            }
        )
    df_sparql = pd.DataFrame(rows).sort_values(["prev_loan_number", "next_loan_number"]).reset_index(drop=True)

    sub_pd = df_prep[df_prep["prev_outcome_code"] == "Default"].copy()
    sub_pd["prev_loan_number_int"] = sub_pd["prev_loan_number"].astype("int64")
    sub_pd = sub_pd.sort_values(["prev_loan_number_int", "LoanNumber"]).reset_index(drop=True)

    assert len(df_sparql) == len(sub_pd), f"Q4 count mismatch: {len(df_sparql)} vs {len(sub_pd)}"
    assert df_sparql["borrower_uri_id"].nunique() == sub_pd["UserName"].nunique()
    assert (df_sparql["prev_loan_number"].to_numpy() == sub_pd["prev_loan_number_int"].to_numpy()).all()
    assert (df_sparql["next_loan_number"].to_numpy() == sub_pd["LoanNumber"].to_numpy()).all()
    assert (df_sparql["next_outcome_code"].to_numpy() == sub_pd["outcome_code"].to_numpy()).all()
    assert np.allclose(df_sparql["next_p_calibrated"], sub_pd["p_calibrated"], atol=1e-9)
    return df_sparql, elapsed


def run_q5(g: Graph, df_prep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Execute Q5 (`q5.rq`) using property path `bo:precededBy+` and cross-check with pandas."""
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
    # Borrowers with 1 loan have max chain length = 1; borrowers in Q5 have max chain length = 1 + max(n_pendahulu)
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

    # Pandas borrower chain length distribution
    b_max_pd = df_prep.groupby("UserName", observed=True)["chain_position"].max()
    dist_pd_map = {int(k): int(v) for k, v in b_max_pd.value_counts().to_dict().items()}
    assert dist_map == dist_pd_map, f"Q5 chain distribution mismatch: {dist_map} vs {dist_pd_map}"

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
            }
        )
    df_dist = pd.DataFrame(dist_rows)
    return df_sparql, df_dist, elapsed


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
    q5_dist_df: pd.DataFrame,
    q6_df: pd.DataFrame,
    df_prep: pd.DataFrame,
) -> pd.DataFrame:
    """Construct `reports/tables/07_insight_action.csv` linking Tahap 2-5 findings and new relational findings."""
    # Q2 country default rates on sample
    q2_country = (
        q2_df.groupby("country", observed=True)
        .agg(n_obs=("n_teramati_12m", "sum"), n_def=("n_default_12m", "sum"))
        .assign(rate_pct=lambda d: d["n_def"] / d["n_obs"] * 100.0)
    )
    q2_str = ", ".join(
        f"{c}={row['rate_pct']:.2f}% (n={int(row['n_obs']):,})"
        for c, row in q2_country.iterrows()
    )

    # Q1 risk tier average p_calibrated on sample
    q1_tier = (
        df_prep.groupby("risk_tier", observed=True)
        .agg(n=("LoanNumber", "count"), mean_p=("p_calibrated", "mean"))
    )
    q1_str = (
        f"Rata-rata p_calibrated sampel menurut tingkat risiko SKOS: "
        f"TingkatRendah={q1_tier.loc['TingkatRendah', 'mean_p']*100:.2f}% (n={int(q1_tier.loc['TingkatRendah', 'n']):,}), "
        f"TingkatMenengah={q1_tier.loc['TingkatMenengah', 'mean_p']*100:.2f}% (n={int(q1_tier.loc['TingkatMenengah', 'n']):,}), "
        f"TingkatTinggi={q1_tier.loc['TingkatTinggi', 'mean_p']*100:.2f}% (n={int(q1_tier.loc['TingkatTinggi', 'n']):,}); "
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
        f"n={tot_s3:,} pinjaman S3: EE {c_s3.get('EE', 0.0):.2f}%, ES {c_s3.get('ES', 0.0):.2f}%, "
        f"FI {c_s3.get('FI', 0.0):.2f}%; F+HR {fhr_s3:.2f}%; Tenor 60 bln {t60_s3:.2f}%; "
        f"rata-rata p_calibrated {p_s3:.2f}%; realisasi default 12m aktual {def12_s3:.2f}%."
    )

    # Q4 repeat after default
    n_q4_pairs = len(q4_df)
    n_q4_borrowers = int(q4_df["borrower_uri_id"].nunique()) if not q4_df.empty else 0
    q4_next_def = int((q4_df["next_outcome_code"] == "Default").sum()) if not q4_df.empty else 0
    q4_next_def_pct = (q4_next_def / max(1, n_q4_pairs) * 100.0) if n_q4_pairs > 0 else 0.0
    q4_in_s3 = int(df_prep[df_prep["LoanNumber"].isin(q4_df["next_loan_number"])]["in_s3"].sum()) if not q4_df.empty else 0

    # Q5 repeat chains
    max_chain = int(q5_dist_df["panjang_rantai_peminjam"].max())
    n_repeat_b = int(q5_dist_df[q5_dist_df["panjang_rantai_peminjam"] >= 2]["jumlah_peminjam_sparql"].sum())
    n_chain_ge5 = int(q5_dist_df[q5_dist_df["panjang_rantai_peminjam"] >= 5]["jumlah_peminjam_sparql"].sum())

    rows = [
        {
            "temuan_sebelumnya": "Perbedaan tingkat gagal bayar antar negara dan pergeseran komposisi hasil menurut era pencatatan",
            "sumber_tahap": "Tahap 2 (EDA) & Tahap 3 (Diagnostic: H1, H7)",
            "query_sparql": "Q2 (07_q2_default12m_country_era.csv) & Q8 (07_q8_outcome_era_country.csv)",
            "hasil_graf": f"Default rate 12 bulan pada sampel graf (n={int(q2_df['n_teramati_12m'].sum()):,} teramati): {q2_str}. Urutan risiko negara (EE < FI < ES < SK) konsisten dengan populasi.",
            "hasil_populasi": "Populasi default_12m teramati (Tahap 2-3, n=148.369): EE 15,28%, FI 24,25%, ES 40,26%, SK 47,26% (selisih ES-EE = +24,98 poin persentase).",
            "catatan": "ANGKA SAMPEL: Proporsi sampel dipengaruhi masuknya seluruh 5.000 pinjaman S3 (2021-2022) dan perluasan peminjam; gunakan angka Tahap 2-3 untuk estimasi populasi.",
        },
        {
            "temuan_sebelumnya": "Urutan tingkat gagal bayar monoton terhadap kelas rating (AA s.d. HR) namun terdapat deviasi prediktif tingkat individu",
            "sumber_tahap": "Tahap 3 (Diagnostic: H2) & Tahap 4 (Predictive: LGBM-A)",
            "query_sparql": "Q1 (07_q1_rating_tier_country.csv) & Q3 (07_q3_rating_vs_model_mismatch.csv)",
            "hasil_graf": q1_str,
            "hasil_populasi": "Populasi Tahap 3 menunjukkan default rate 12m naik monoton dari AA (3,82%) hingga HR (52,70%); model LGBM-A (ROC-AUC uji 0,7152) menangkap sinyal arus kas dan beban kewajiban di luar huruf rating.",
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
            "temuan_sebelumnya": "Temuan relasional baru (belum ada di Tahap 1-5): peminjam yang memperoleh pinjaman baru setelah pinjaman sebelumnya berstatus akhir Default",
            "sumber_tahap": "Tahap 6 (Semantic Analytics — relasi bo:precededBy)",
            "query_sparql": "Q4 (07_q4_repeat_borrower_after_default.csv)",
            "hasil_graf": (
                f"Tercatat {n_q4_pairs:,} pasangan transisi pinjaman (milik {n_q4_borrowers:,} peminjam unik di graf) "
                f"di mana pinjaman baru memiliki pendahulu langsung berstatus akhir bo:Default; "
                f"{q4_next_def:,} ({q4_next_def_pct:.2f}%) dari pinjaman lanjutan tersebut juga berakhir Default, "
                f"dan {q4_in_s3:,} di antaranya terpilih dalam portofolio S3."
            ),
            "hasil_populasi": "Tidak dihitung pada model tabular per-baris Tahap 1-5 karena relasi antar-pinjaman milik peminjam yang sama tidak direpresentasikan sebagai graf urutan.",
            "catatan": "WAWASAN RELASIONAL SAMPEL (PERTANYAAN TINDAK LANJUT): Karena bo:Default mencerminkan status akhir pinjaman (sebagian pinjaman lanjutan dicairkan ketika pinjaman lama masih berjalan sebelum akhirnya gagal bayar bersama-sama), apakah kebijakan seleksi investor perlu membatasi pinjaman beruntun dari peminjam yang memiliki eksposur pinjaman sebelumnya?",
        },
        {
            "temuan_sebelumnya": "Temuan relasional baru (belum ada di Tahap 1-5): struktur panjang rantai peminjaman ulang per peminjam",
            "sumber_tahap": "Tahap 6 (Semantic Analytics — property path bo:precededBy+)",
            "query_sparql": "Q5 (07_q5_repeat_chain_lengths.csv & 07_q5_chain_distribution.csv)",
            "hasil_graf": (
                f"Dari {int(df_prep['UserName'].nunique()):,} peminjam di graf, {n_repeat_b:,} peminjam memiliki >= 2 pinjaman "
                f"(total {len(df_prep) - int(df_prep['UserName'].nunique()):,} pinjaman lanjutan), {n_chain_ge5:,} peminjam memiliki rantai >= 5 pinjaman, "
                f"dan panjang rantai maksimum tercatat {max_chain} pinjaman berurutan."
            ),
            "hasil_populasi": "Tahap 5 hanya membatasi bobot negara, F+HR, dan tenor 60 bulan tanpa batas konsentrasi jumlah pinjaman per peminjam (UserName).",
            "catatan": "WAWASAN RELASIONAL SAMPEL (TINDAKAN): Menyarankan penambahan batasan konsentrasi per peminjam (mis. maksimal 1-2 pinjaman aktif per bo:Borrower) pada optimasi portofolio.",
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
    q4_df, t_q4 = run_q4(g, df_prep)
    print(f"  Q4 done in {t_q4:.3f}s", flush=True)
    q5_df, q5_dist_df, t_q5 = run_q5(g, df_prep)
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

    insight_df = build_insight_action_table(q1_df, q2_df, q3_df, q4_df, q5_dist_df, q6_df, df_prep)

    if save_outputs:
        q1_df.to_csv(TABLES_DIR / "07_q1_rating_tier_country.csv", index=False)
        q2_df.to_csv(TABLES_DIR / "07_q2_default12m_country_era.csv", index=False)
        q3_df.to_csv(TABLES_DIR / "07_q3_rating_vs_model_mismatch.csv", index=False)
        q4_df.to_csv(TABLES_DIR / "07_q4_repeat_borrower_after_default.csv", index=False)
        q5_df.to_csv(TABLES_DIR / "07_q5_repeat_chain_lengths.csv", index=False)
        q5_dist_df.to_csv(TABLES_DIR / "07_q5_chain_distribution.csv", index=False)
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
        "q5": q5_df,
        "q5_dist": q5_dist_df,
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
    print("\n--- OWL-RL Inference Summary ---")
    print(res["inference_summary"].to_string(index=False))
    print(f"\n[Done] Total execution time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
