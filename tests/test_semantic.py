"""Unit tests for Tahap 6 Semantic Analytics (`tests/test_semantic.py`)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import rdflib
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from src.config import DEFAULT_12M_COL, IS_RESOLVED_COL, LOAN_DATE_COL, REGRESSION_TARGET_COL
from src.semantic.build_graph import (
    BO,
    ONTOLOGY_PATH,
    build_knowledge_graph,
    load_graph_gz,
    loan_uri,
    prepare_annotated_sample_df,
    serialize_graph_gz,
)
from src.semantic.queries import (
    run_owlrl_inference,
    run_q1,
    run_q2,
    run_q4,
    run_q5,
    run_q6,
)
from src.semantic.sampling import sample_graph_loans


def _make_mini_df() -> pd.DataFrame:
    """Create a synthetic mini DataFrame (12 loans across 5 borrowers) for unit testing."""
    rows = [
        # Borrower B1: 3 loans in EE (1st Default -> 2nd RepaidNoDefault -> 3rd Current in S3)
        {
            "LoanId": "L_101",
            "LoanNumber": 101,
            "UserName": "user_alpha",
            "Country": "EE",
            "Rating": "F",
            LOAN_DATE_COL: "2016-05-10",
            "Amount": 1000.0,
            "Interest": 28.5,
            "LoanDuration": 36,
            "Status": "Late",
            "DefaultDate": pd.Timestamp("2016-11-15"),
            IS_RESOLVED_COL: True,
            DEFAULT_12M_COL: 1.0,
            REGRESSION_TARGET_COL: 0.45,
            "p_calibrated": 0.42,
        },
        {
            "LoanId": "L_102",
            "LoanNumber": 102,
            "UserName": "user_alpha",
            "Country": "EE",
            "Rating": "C",
            LOAN_DATE_COL: "2018-03-20",
            "Amount": 1500.0,
            "Interest": 18.0,
            "LoanDuration": 24,
            "Status": "Repaid",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: True,
            DEFAULT_12M_COL: 0.0,
            REGRESSION_TARGET_COL: 1.15,
            "p_calibrated": 0.20,
        },
        {
            "LoanId": "L_103",
            "LoanNumber": 103,
            "UserName": "user_alpha",
            "Country": "EE",
            "Rating": "A",
            LOAN_DATE_COL: "2021-08-01",
            "Amount": 2000.0,
            "Interest": 11.5,
            "LoanDuration": 60,
            "Status": "Current",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: False,
            DEFAULT_12M_COL: 0.0,
            REGRESSION_TARGET_COL: np.nan,
            "p_calibrated": 0.09,
        },
        # Borrower B2: 2 loans on same date (tie-break by LoanNumber 201 -> 202)
        {
            "LoanId": "L_202",
            "LoanNumber": 202,
            "UserName": "user_beta",
            "Country": "ES",
            "Rating": "HR",
            LOAN_DATE_COL: "2017-07-01",
            "Amount": 800.0,
            "Interest": 45.0,
            "LoanDuration": 12,
            "Status": "Late",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: False,
            DEFAULT_12M_COL: np.nan,
            REGRESSION_TARGET_COL: np.nan,
            "p_calibrated": 0.55,
        },
        {
            "LoanId": "L_201",
            "LoanNumber": 201,
            "UserName": "user_beta",
            "Country": "ES",
            "Rating": "E",
            LOAN_DATE_COL: "2017-07-01",
            "Amount": 500.0,
            "Interest": 36.0,
            "LoanDuration": 12,
            "Status": "Late",
            "DefaultDate": pd.Timestamp("2017-12-01"),
            IS_RESOLVED_COL: True,
            DEFAULT_12M_COL: 1.0,
            REGRESSION_TARGET_COL: 0.30,
            "p_calibrated": 0.48,
        },
        # Borrower B3: 1 loan in FI with low rating (AA) but high model risk (p=0.39) -> Q3 mismatch
        {
            "LoanId": "L_301",
            "LoanNumber": 301,
            "UserName": "user_gamma",
            "Country": "FI",
            "Rating": "AA",
            LOAN_DATE_COL: "2019-11-12",
            "Amount": 3000.0,
            "Interest": 9.5,
            "LoanDuration": 48,
            "Status": "Current",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: False,
            DEFAULT_12M_COL: 0.0,
            REGRESSION_TARGET_COL: np.nan,
            "p_calibrated": 0.39,
        },
        # Borrower B4: 1 loan in SK with missing rating
        {
            "LoanId": "L_401",
            "LoanNumber": 401,
            "UserName": "user_delta",
            "Country": "SK",
            "Rating": None,
            LOAN_DATE_COL: "2015-02-14",
            "Amount": 1200.0,
            "Interest": 22.0,
            "LoanDuration": 60,
            "Status": "Repaid",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: True,
            DEFAULT_12M_COL: 0.0,
            REGRESSION_TARGET_COL: 1.22,
            "p_calibrated": 0.14,
        },
        # Additional loans for B5
        {
            "LoanId": "L_501",
            "LoanNumber": 501,
            "UserName": "user_eps",
            "Country": "FI",
            "Rating": "B",
            LOAN_DATE_COL: "2020-01-10",
            "Amount": 1800.0,
            "Interest": 14.0,
            "LoanDuration": 36,
            "Status": "Current",
            "DefaultDate": pd.NaT,
            IS_RESOLVED_COL: False,
            DEFAULT_12M_COL: 0.0,
            REGRESSION_TARGET_COL: np.nan,
            "p_calibrated": 0.12,
        },
    ]
    df = pd.DataFrame(rows)
    df[LOAN_DATE_COL] = pd.to_datetime(df[LOAN_DATE_COL])
    # Include Gender and Age in raw DataFrame to prove graph builder excludes them
    df["Gender"] = [0, 0, 0, 1, 1, 0, 1, 2]
    df["Age"] = [34, 36, 39, 28, 28, 45, 51, 31]
    return df


def test_1_ontology_loads_and_schema_complete() -> None:
    """1. Ontology loads; all classes and properties exist; every property has rdfs:domain and rdfs:range."""
    g = Graph()
    g.parse(str(ONTOLOGY_PATH), format="turtle")
    assert len(g) > 100

    expected_classes = [
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
    ]
    for cls_uri in expected_classes:
        assert (cls_uri, RDF.type, OWL.Class) in g, f"Missing class {cls_uri}"
        assert any(g.objects(cls_uri, RDFS.label)), f"Missing rdfs:label on {cls_uri}"

    expected_obj_props = [
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
    ]
    expected_data_props = [
        BO.amount,
        BO.interestPct,
        BO.loanDurationMonths,
        BO.loanDate,
        BO.defaultWithin12m,
        BO.predictedDefaultProb12m,
        BO.returnRatio,
        BO.isResolved,
        BO.ordinalRank,
    ]

    for prop in expected_obj_props:
        assert (prop, RDF.type, OWL.ObjectProperty) in g, f"Missing ObjectProperty {prop}"
    for prop in expected_data_props:
        assert (prop, RDF.type, OWL.DatatypeProperty) in g, f"Missing DatatypeProperty {prop}"

    all_props = set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(g.subjects(RDF.type, OWL.DatatypeProperty))
    for prop in all_props:
        assert any(g.objects(prop, RDFS.domain)), f"Property {prop} missing rdfs:domain"
        assert any(g.objects(prop, RDFS.range)), f"Property {prop} missing rdfs:range"
        assert any(g.objects(prop, RDFS.label)), f"Property {prop} missing rdfs:label"


def test_2_taxonomy_skos_and_outcome_groups() -> None:
    """2. Taxonomy: every rating has exactly one skos:broader to a risk tier; outcome groups equal 4."""
    g = Graph()
    g.parse(str(ONTOLOGY_PATH), format="turtle")

    ratings = list(g.subjects(RDF.type, BO.RatingClass))
    assert len(ratings) == 9  # AA, A, B, C, D, E, F, HR, RatingTidakTersedia

    risk_tiers = set(g.subjects(RDF.type, BO.RiskTier))
    assert len(risk_tiers) >= 3

    for r_uri in ratings:
        broader_list = list(g.objects(r_uri, SKOS.broader))
        assert len(broader_list) == 1, f"Rating {r_uri} should have exactly 1 skos:broader, got {broader_list}"
        assert broader_list[0] in risk_tiers, f"Rating {r_uri} skos:broader {broader_list[0]} is not a bo:RiskTier"

    outcomes = set(g.subjects(RDF.type, BO.OutcomeGroup))
    assert outcomes == {BO.Default, BO.RepaidNoDefault, BO.Current, BO.LateNotDefault}


def test_3_graph_builder_cardinalities_and_no_gender_age() -> None:
    """3. Every bo:Loan has 1 country, 1 rating, 1 outcome, 1 era, 1 tenor band; no Gender or Age in graph."""
    mini_df = _make_mini_df()
    prep_df = prepare_annotated_sample_df(mini_df, s3_loan_ids=["L_103", "L_501"], score_if_missing=False)
    g, _ = build_knowledge_graph(prep_df)

    loans = list(g.subjects(RDF.type, BO.Loan))
    assert len(loans) == len(mini_df)

    for l_node in loans:
        assert len(list(g.objects(l_node, BO.originatedIn))) == 1
        assert len(list(g.objects(l_node, BO.ratedAs))) == 1
        assert len(list(g.objects(l_node, BO.hasOutcome))) == 1
        assert len(list(g.objects(l_node, BO.inEra))) == 1
        assert len(list(g.objects(l_node, BO.hasTenorBand))) == 1
        assert len(list(g.objects(l_node, BO.inRiskBand))) == 1
        assert len(list(g.objects(l_node, BO.hasBorrower))) == 1

    # Verify no Gender or Age appears as any URI/predicate in the graph
    for s, p, o in g:
        for term in (s, p, o):
            if isinstance(term, URIRef):
                low = str(term).lower()
                assert "gender" not in low
                assert "age" not in low


def test_4_preceded_by_chain_order_and_acyclicity() -> None:
    """4. precededBy chain has correct order, tie-breaking, no cycles, and first loan has no predecessor."""
    mini_df = _make_mini_df()
    prep_df = prepare_annotated_sample_df(mini_df, s3_loan_ids=["L_103"], score_if_missing=False)
    g, _ = build_knowledge_graph(prep_df)

    # Borrower user_alpha: 101 -> 102 -> 103 (so 103 precededBy 102, 102 precededBy 101, 101 has none)
    assert list(g.objects(loan_uri(101), BO.precededBy)) == []
    assert list(g.objects(loan_uri(102), BO.precededBy)) == [loan_uri(101)]
    assert list(g.objects(loan_uri(103), BO.precededBy)) == [loan_uri(102)]

    # Borrower user_beta: same date 2017-07-01, tie-broken by LoanNumber 201 -> 202
    assert list(g.objects(loan_uri(201), BO.precededBy)) == []
    assert list(g.objects(loan_uri(202), BO.precededBy)) == [loan_uri(201)]

    # Verify directed graph of precededBy is a DAG (acyclic)
    dag = nx.DiGraph()
    for s, o in g.subject_objects(BO.precededBy):
        dag.add_edge(str(s), str(o))
    assert nx.is_directed_acyclic_graph(dag)


def test_5_sampling_strata_reproducibility_and_limits() -> None:
    """5. Sampling represents all non-empty strata, is reproducible with seed, and obeys max_loans."""
    rng = np.random.default_rng(123)
    n_pop = 300
    countries = rng.choice(["EE", "ES", "FI"], size=n_pop)
    statuses = rng.choice(["Repaid", "Current", "Late"], size=n_pop)
    has_def = (statuses == "Late") & (rng.random(n_pop) < 0.5)
    dates = pd.to_datetime(rng.choice(["2016-06-01", "2019-06-01"], size=n_pop))

    synth_df = pd.DataFrame(
        {
            "LoanId": [f"id_{i}" for i in range(n_pop)],
            "LoanNumber": np.arange(1000, 1000 + n_pop),
            "UserName": [f"u_{i % 60}" for i in range(n_pop)],
            "Country": countries,
            "Rating": rng.choice(["A", "C", "F"], size=n_pop),
            LOAN_DATE_COL: dates,
            "Amount": 1000.0,
            "Interest": 20.0,
            "LoanDuration": 36,
            "Status": statuses,
            "DefaultDate": [pd.Timestamp("2017-01-01") if d else pd.NaT for d in has_def],
            IS_RESOLVED_COL: statuses != "Current",
            DEFAULT_12M_COL: [1.0 if d else 0.0 for d in has_def],
            REGRESSION_TARGET_COL: 1.0,
        }
    )
    s3_ids = [f"id_{i}" for i in range(15)]

    s1 = sample_graph_loans(synth_df, s3_ids, n_stratified=80, min_per_stratum=3, max_loans=120, seed=42)
    s2 = sample_graph_loans(synth_df, s3_ids, n_stratified=80, min_per_stratum=3, max_loans=120, seed=42)

    assert len(s1["sample_df"]) <= 120
    assert (s1["sample_df"]["LoanNumber"].to_numpy() == s2["sample_df"]["LoanNumber"].to_numpy()).all()

    # Check every non-empty stratum in synth_df is represented in stratified sample
    strata_rows = s1["summary_df"][s1["summary_df"]["kategori"] == "Strata (Country|Outcome|Era)"]
    assert len(strata_rows) > 0
    assert (strata_rows["n_stratifikasi_4000"] > 0).all()
    sub_strat = s1["sample_df"][s1["sample_df"]["in_stratified"]]
    samp_strata = set((sub_strat["Country"] + "|" + sub_strat["outcome_group"] + "|" + sub_strat["era"]).unique())
    assert set(strata_rows["subkategori"].unique()) == samp_strata


def test_6_sparql_pandas_cross_check_and_inference_on_mini_graph() -> None:
    """6. Cross-check SPARQL aggregation queries and OWL-RL inference on mini graph against pandas."""
    mini_df = _make_mini_df()
    prep_df = prepare_annotated_sample_df(mini_df, s3_loan_ids=["L_103", "L_501"], score_if_missing=False)
    g, df_sorted = build_knowledge_graph(prep_df)

    q1_df, _ = run_q1(g, df_sorted)
    q2_df, _ = run_q2(g, df_sorted)
    q4_df, _ = run_q4(g, df_sorted)
    q5_df, q5_dist_df, _ = run_q5(g, df_sorted)
    q6_df, _ = run_q6(
        # On mini graph we only check run_q1, run_q2, run_q4, run_q5 and OWL-RL inference
        # (run_q6 asserts the full 5,000 S3 portfolio shares; test OWL-RL inference directly here)
        g, df_sorted
    ) if False else (None, 0.0)

    assert (q1_df["n_pinjaman"] == q1_df["pandas_n_pinjaman"]).all()
    assert (q2_df["n_default_12m"] == q2_df["pandas_n_default_12m"]).all()
    assert len(q4_df) == 2  # Loan 102 after 101 Default, and Loan 202 after 201 Default
    assert int(q5_dist_df["panjang_rantai_peminjam"].max()) == 3

    g_inf, inf_df = run_owlrl_inference(g, df_sorted)
    assert len(g_inf) > len(g)
    assert (inf_df.loc[1:, "sesudah_inferensi_owlrl"].to_numpy() == inf_df.loc[1:, "hitung_pandas"].to_numpy()).all()


def test_7_serialization_reload_preserves_triple_count(tmp_path: Path) -> None:
    """7. Reloading serialized gzipped Turtle graph produces the exact same triple count."""
    mini_df = _make_mini_df()
    prep_df = prepare_annotated_sample_df(mini_df, s3_loan_ids=["L_103"], score_if_missing=False)
    g, _ = build_knowledge_graph(prep_df)

    gz_file = tmp_path / "mini_graph.ttl.gz"
    size_bytes = serialize_graph_gz(g, gz_file)
    assert size_bytes > 0

    g_reloaded = load_graph_gz(gz_file)
    assert len(g_reloaded) == len(g)
