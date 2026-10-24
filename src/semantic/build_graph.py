"""Knowledge Graph builder for Tahap 6 (`python -m src.semantic.build_graph`)."""

from __future__ import annotations

import gzip
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rdflib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

from src.analysis.descriptive import make_outcome_group
from src.config import (
    DATA_PROCESSED,
    DEFAULT_12M_COL,
    IS_RESOLVED_COL,
    LOAN_DATE_COL,
    ONTOLOGY_DIR,
    RANDOM_SEED,
    REGRESSION_TARGET_COL,
    TABLES_DIR,
    ensure_dirs,
)
from src.models.inference import predict_default12m
from src.semantic.sampling import (
    make_era_series,
    make_risk_band_series,
    make_tenor_band_series,
    sample_graph_loans,
)

BO_NS_STR: str = "http://example.org/bondora#"
BO: Namespace = Namespace(BO_NS_STR)

ONTOLOGY_PATH: Path = ONTOLOGY_DIR / "bondora_ontology.ttl"
GRAPH_GZ_PATH: Path = DATA_PROCESSED / "bondora_graph.ttl.gz"
SAMPLE_PARQUET_PATH: Path = DATA_PROCESSED / "semantic_sample.parquet"

OUTCOME_URI_MAP: dict[str, URIRef] = {
    "Default": BO.Default,
    "Lunas tanpa default": BO.RepaidNoDefault,
    "Berjalan (Current)": BO.Current,
    "Terlambat belum default": BO.LateNotDefault,
}

RATING_URI_MAP: dict[str, URIRef] = {
    "AA": BO.Rating_AA,
    "A": BO.Rating_A,
    "B": BO.Rating_B,
    "C": BO.Rating_C,
    "D": BO.Rating_D,
    "E": BO.Rating_E,
    "F": BO.Rating_F,
    "HR": BO.Rating_HR,
}

RISK_TIER_LABEL_MAP: dict[str, str] = {
    "AA": "TingkatRendah",
    "A": "TingkatRendah",
    "B": "TingkatRendah",
    "C": "TingkatMenengah",
    "D": "TingkatMenengah",
    "E": "TingkatTinggi",
    "F": "TingkatTinggi",
    "HR": "TingkatTinggi",
}


def safe_borrower_local_id(username: str) -> str:
    """Convert raw ``UserName`` into a collision-free, valid Turtle local identifier."""
    s = str(username).strip()
    if re.match(r"^[A-Za-z0-9_]+$", s):
        return s
    out: list[str] = []
    for ch in s:
        if re.match(r"^[A-Za-z0-9_]$", ch):
            out.append(ch)
        else:
            out.append(f"_x{ord(ch):04x}_")
    return "".join(out)


def loan_uri(loan_number: int | str) -> URIRef:
    """Return ``bo:Loan_<LoanNumber>`` URI."""
    return BO[f"Loan_{int(loan_number)}"]


def borrower_uri(username: str) -> URIRef:
    """Return ``bo:Borrower_<safe_UserName>`` URI."""
    return BO[f"Borrower_{safe_borrower_local_id(username)}"]


def prepare_annotated_sample_df(
    df: pd.DataFrame,
    s3_loan_ids: Sequence[str] | set[str] | None = None,
    *,
    score_if_missing: bool = True,
) -> pd.DataFrame:
    """Ensure ``df`` has all derived columns needed for RDF triple generation and pandas cross-checks."""
    out = df.copy()
    if "LoanId" not in out.columns:
        out["LoanId"] = [f"id_{i}" for i in range(len(out))]
    out["LoanId"] = out["LoanId"].astype(str)
    out["LoanNumber"] = pd.to_numeric(out["LoanNumber"], errors="coerce").astype("int64")
    out["UserName"] = out["UserName"].astype(str)
    out["Country"] = out["Country"].astype(str)
    out[LOAN_DATE_COL] = pd.to_datetime(out[LOAN_DATE_COL])
    if "DefaultDate" in out.columns:
        out["DefaultDate"] = pd.to_datetime(out["DefaultDate"], errors="coerce")
    else:
        out["DefaultDate"] = pd.NaT

    if "p_calibrated" not in out.columns:
        if score_if_missing:
            out["p_calibrated"] = np.round(predict_default12m(out).to_numpy(dtype="float64"), 6)
        else:
            raise ValueError("Column 'p_calibrated' is missing and score_if_missing=False.")
    else:
        out["p_calibrated"] = np.round(pd.to_numeric(out["p_calibrated"], errors="coerce").to_numpy(dtype="float64"), 6)

    if "in_s3" not in out.columns:
        s3_set = set(str(x) for x in (s3_loan_ids or []))
        out["in_s3"] = out["LoanId"].isin(s3_set)
    else:
        out["in_s3"] = out["in_s3"].astype(bool)

    out["outcome_group"] = make_outcome_group(out).astype(str)
    out["outcome_code"] = out["outcome_group"].map(
        {
            "Default": "Default",
            "Lunas tanpa default": "RepaidNoDefault",
            "Berjalan (Current)": "Current",
            "Terlambat belum default": "LateNotDefault",
        }
    )
    out["era"] = make_era_series(out[LOAN_DATE_COL]).astype(str)
    out["tenor_band"] = make_tenor_band_series(out["LoanDuration"]).astype(str)
    out["risk_band"] = make_risk_band_series(out["p_calibrated"]).astype(str)

    raw_rating = out["Rating"].astype("object").where(out["Rating"].notna(), None)
    out["rating_clean"] = [str(r).strip() if r is not None and str(r).strip() in RATING_URI_MAP else "TidakTersedia" for r in raw_rating]
    out["risk_tier"] = [RISK_TIER_LABEL_MAP.get(r, "TingkatTidakTersedia") for r in out["rating_clean"]]

    # Compute precededBy predecessor LoanNumber on the sorted sample
    out = out.sort_values(["UserName", LOAN_DATE_COL, "LoanNumber"], kind="mergesort").reset_index(drop=True)
    out["loan_date_str"] = pd.to_datetime(out[LOAN_DATE_COL]).dt.strftime("%Y-%m-%d")
    def_dt = pd.to_datetime(out["DefaultDate"], errors="coerce")
    out["default_date_str"] = def_dt.dt.strftime("%Y-%m-%d").where(def_dt.notna(), None)
    out["prev_loan_number"] = out.groupby("UserName", observed=True)["LoanNumber"].shift(1)
    out["prev_outcome_code"] = out.groupby("UserName", observed=True)["outcome_code"].shift(1)
    out["prev_default_date"] = out.groupby("UserName", observed=True)["DefaultDate"].shift(1)
    out["prev_default_date_str"] = out.groupby("UserName", observed=True)["default_date_str"].shift(1)
    out["chain_position"] = out.groupby("UserName", observed=True).cumcount() + 1
    out["borrower_loan_count_in_graph"] = out.groupby("UserName", observed=True)["LoanNumber"].transform("count")

    return out


def build_knowledge_graph(
    sample_df: pd.DataFrame,
    ontology_path: Path = ONTOLOGY_PATH,
) -> tuple[Graph, pd.DataFrame]:
    """Construct the RDF Knowledge Graph from an annotated sample DataFrame and the ontology."""
    g = Graph()
    g.bind("bo", BO)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("xsd", XSD)
    g.parse(str(ontology_path), format="turtle")

    df_sorted = prepare_annotated_sample_df(sample_df, score_if_missing=False)

    # Add Borrower individuals (country from earliest loan in graph)
    first_per_borrower = df_sorted.drop_duplicates(subset=["UserName"], keep="first")
    for uname, c_code in zip(first_per_borrower["UserName"], first_per_borrower["Country"]):
        b_node = borrower_uri(str(uname))
        g.add((b_node, RDF.type, BO.Borrower))
        g.add((b_node, BO.borrowerCountry, BO[f"Country_{c_code}"]))

    # Pre-extract arrays for fast triple population
    loan_nums = df_sorted["LoanNumber"].to_numpy(dtype="int64")
    usernames = df_sorted["UserName"].astype(str).to_numpy()
    countries = df_sorted["Country"].astype(str).to_numpy()
    ratings = df_sorted["rating_clean"].astype(str).to_numpy()
    outcomes = df_sorted["outcome_group"].astype(str).to_numpy()
    eras = df_sorted["era"].astype(str).to_numpy()
    tenors = df_sorted["tenor_band"].astype(str).to_numpy()
    risk_bands = df_sorted["risk_band"].astype(str).to_numpy()
    in_s3_arr = df_sorted["in_s3"].to_numpy(dtype=bool)
    prev_nums = df_sorted["prev_loan_number"].to_numpy()

    amounts = pd.to_numeric(df_sorted["Amount"], errors="coerce").to_numpy(dtype="float64")
    interests = pd.to_numeric(df_sorted["Interest"], errors="coerce").to_numpy(dtype="float64")
    durations = pd.to_numeric(df_sorted["LoanDuration"], errors="coerce").to_numpy(dtype="int64")
    loan_dates = pd.to_datetime(df_sorted[LOAN_DATE_COL]).dt.strftime("%Y-%m-%d").to_numpy()
    def_dates_dt = pd.to_datetime(df_sorted["DefaultDate"], errors="coerce")
    def_dates_notna = def_dates_dt.notna().to_numpy(dtype=bool)
    def_dates_str = def_dates_dt.dt.strftime("%Y-%m-%d").to_numpy()
    p_cals = df_sorted["p_calibrated"].to_numpy(dtype="float64")
    is_res_arr = df_sorted[IS_RESOLVED_COL].astype(bool).to_numpy()
    def12_arr = df_sorted[DEFAULT_12M_COL].to_numpy()
    ret_arr = pd.to_numeric(df_sorted[REGRESSION_TARGET_COL], errors="coerce").to_numpy(dtype="float64")

    for i in range(len(df_sorted)):
        l_node = loan_uri(int(loan_nums[i]))
        b_node = borrower_uri(usernames[i])
        r_code = ratings[i]
        r_uri = RATING_URI_MAP.get(r_code, BO.RatingTidakTersedia)

        g.add((l_node, RDF.type, BO.Loan))
        g.add((l_node, BO.hasBorrower, b_node))
        g.add((l_node, BO.originatedIn, BO[f"Country_{countries[i]}"]))
        g.add((l_node, BO.ratedAs, r_uri))
        g.add((l_node, BO.hasOutcome, OUTCOME_URI_MAP[outcomes[i]]))
        g.add((l_node, BO.inEra, BO[eras[i]]))
        g.add((l_node, BO.hasTenorBand, BO[tenors[i]]))
        g.add((l_node, BO.inRiskBand, BO[risk_bands[i]]))

        if in_s3_arr[i]:
            g.add((l_node, BO.selectedBy, BO.StrategiS3))

        if not pd.isna(prev_nums[i]):
            g.add((l_node, BO.precededBy, loan_uri(int(prev_nums[i]))))

        g.add((l_node, BO.amount, Literal(Decimal(f"{amounts[i]:.2f}"), datatype=XSD.decimal)))
        g.add((l_node, BO.interestPct, Literal(Decimal(f"{interests[i]:.4f}"), datatype=XSD.decimal)))
        g.add((l_node, BO.loanDurationMonths, Literal(int(durations[i]), datatype=XSD.integer)))
        g.add((l_node, BO.loanDate, Literal(str(loan_dates[i]), datatype=XSD.date)))
        if def_dates_notna[i]:
            g.add((l_node, BO.defaultDate, Literal(str(def_dates_str[i]), datatype=XSD.date)))
        g.add((l_node, BO.predictedDefaultProb12m, Literal(Decimal(f"{p_cals[i]:.6f}"), datatype=XSD.decimal)))
        g.add((l_node, BO.isResolved, Literal(bool(is_res_arr[i]), datatype=XSD.boolean)))

        if not pd.isna(def12_arr[i]):
            g.add((l_node, BO.defaultWithin12m, Literal(bool(int(def12_arr[i]) == 1), datatype=XSD.boolean)))

        if is_res_arr[i] and not np.isnan(ret_arr[i]):
            g.add((l_node, BO.returnRatio, Literal(Decimal(f"{ret_arr[i]:.6f}"), datatype=XSD.decimal)))

    return g, df_sorted


def serialize_graph_gz(g: Graph, path: Path = GRAPH_GZ_PATH) -> int:
    """Serialize ``g`` as gzip-compressed Turtle and return the compressed byte size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ttl_bytes = g.serialize(format="turtle").encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(ttl_bytes)
    return int(path.stat().st_size)


def load_graph_gz(path: Path = GRAPH_GZ_PATH) -> Graph:
    """Load gzip-compressed Turtle graph into a new ``rdflib.Graph``."""
    g = Graph()
    with gzip.open(path, "rb") as f:
        g.parse(file=f, format="turtle")
    return g


def compute_graph_stats_table(
    g: Graph,
    *,
    ontology_triples: int,
    reloaded_triples: int,
    build_seconds: float,
    reload_seconds: float,
    file_bytes: int,
) -> pd.DataFrame:
    """Build ``07_graph_stats.csv`` with summary metrics, class individual counts, and predicate triple counts."""
    total_triples = len(g)
    rows: list[dict[str, Any]] = [
        {
            "bagian": "Ringkasan",
            "item": "total_triples",
            "jumlah": total_triples,
            "persen_dari_total": 100.0,
            "catatan": "Total triple dalam graf (ontologi + instans)",
        },
        {
            "bagian": "Ringkasan",
            "item": "ontology_triples",
            "jumlah": ontology_triples,
            "persen_dari_total": ontology_triples / max(1, total_triples) * 100.0,
            "catatan": "Triple dari ontology/bondora_ontology.ttl",
        },
        {
            "bagian": "Ringkasan",
            "item": "instance_triples",
            "jumlah": total_triples - ontology_triples,
            "persen_dari_total": (total_triples - ontology_triples) / max(1, total_triples) * 100.0,
            "catatan": "Triple instans pinjaman dan peminjam",
        },
        {
            "bagian": "Ringkasan",
            "item": "reloaded_triples",
            "jumlah": reloaded_triples,
            "persen_dari_total": reloaded_triples / max(1, total_triples) * 100.0,
            "catatan": "Jumlah triple saat dimuat ulang dari bondora_graph.ttl.gz (harus sama persis)",
        },
        {
            "bagian": "Kinerja & File",
            "item": "waktu_bangun_detik",
            "jumlah": round(build_seconds, 3),
            "persen_dari_total": np.nan,
            "catatan": "Waktu pembangunan + serialisasi graf (detik)",
        },
        {
            "bagian": "Kinerja & File",
            "item": "waktu_muat_ulang_detik",
            "jumlah": round(reload_seconds, 3),
            "persen_dari_total": np.nan,
            "catatan": "Waktu muat ulang dari bondora_graph.ttl.gz (detik)",
        },
        {
            "bagian": "Kinerja & File",
            "item": "ukuran_file_bytes",
            "jumlah": file_bytes,
            "persen_dari_total": np.nan,
            "catatan": f"Ukuran bondora_graph.ttl.gz = {file_bytes / (1024 * 1024):.3f} MB",
        },
    ]

    # Count individuals per class
    class_list = [
        ("bo:Loan", BO.Loan),
        ("bo:Borrower", BO.Borrower),
        ("bo:Country", BO.Country),
        ("bo:RatingClass", BO.RatingClass),
        ("bo:RiskTier", BO.RiskTier),
        ("bo:OutcomeGroup", BO.OutcomeGroup),
        ("bo:Era", BO.Era),
        ("bo:TenorBand", BO.TenorBand),
        ("bo:RiskBand", BO.RiskBand),
        ("bo:Strategy", BO.Strategy),
    ]
    for c_label, c_uri in class_list:
        cnt = len(set(g.subjects(RDF.type, c_uri)))
        rows.append(
            {
                "bagian": "Individu per Kelas",
                "item": c_label,
                "jumlah": cnt,
                "persen_dari_total": np.nan,
                "catatan": f"Jumlah individu bertipe {c_label}",
            }
        )

    # Count triples per predicate
    pred_counts: dict[str, int] = {}
    for _, p, _ in g:
        p_str = g.namespace_manager.normalizeUri(p)
        pred_counts[p_str] = pred_counts.get(p_str, 0) + 1

    for p_label, p_cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], x[0])):
        rows.append(
            {
                "bagian": "Triple per Predikat",
                "item": p_label,
                "jumlah": p_cnt,
                "persen_dari_total": p_cnt / max(1, total_triples) * 100.0,
                "catatan": "Jumlah triple dengan predikat ini",
            }
        )

    return pd.DataFrame(rows)


def main() -> dict[str, Any]:
    """Run Step 2 (sampling) and Step 3 (graph build, serialization, reload check, and stats tables)."""
    ensure_dirs()
    t0 = time.perf_counter()

    print("=== Tahap 6 Knowledge Graph Builder (`src.semantic.build_graph`) ===")
    clean_df = pd.read_parquet(DATA_PROCESSED / "bondora_clean.parquet")
    s3_df = pd.read_parquet(DATA_PROCESSED / "s3_selected_loans.parquet")
    assert clean_df.shape == (179_235, 130)
    assert len(s3_df) == 5_000

    # Step 2: Stratified + S3 + borrower expansion sample
    samp = sample_graph_loans(clean_df, s3_df["LoanId"].tolist(), n_stratified=4_000, max_loans=20_000, seed=RANDOM_SEED)
    samp["summary_df"].to_csv(TABLES_DIR / "07_sample_summary.csv", index=False)
    print(f"[Sampling] {samp['expansion_note']}")

    # Score all sampled loans via src/models/inference.py
    sample_df = samp["sample_df"].copy()
    sample_df["p_calibrated"] = np.round(predict_default12m(sample_df).to_numpy(dtype="float64"), 6)

    # Ontology triple count
    g_ont = Graph()
    g_ont.parse(str(ONTOLOGY_PATH), format="turtle")
    ontology_triples = len(g_ont)

    # Step 3: Build graph and serialize
    t_build_start = time.perf_counter()
    g, annotated_df = build_knowledge_graph(sample_df, ontology_path=ONTOLOGY_PATH)
    file_bytes = serialize_graph_gz(g, GRAPH_GZ_PATH)
    build_seconds = time.perf_counter() - t_build_start

    annotated_df.to_parquet(SAMPLE_PARQUET_PATH, index=False)

    # Reload verification
    t_reload_start = time.perf_counter()
    g_reloaded = load_graph_gz(GRAPH_GZ_PATH)
    reload_seconds = time.perf_counter() - t_reload_start
    assert len(g_reloaded) == len(g), f"Reloaded triple count {len(g_reloaded)} != {len(g)}"
    assert len(annotated_df) <= 20_000 and len(g) <= 600_000

    stats_df = compute_graph_stats_table(
        g,
        ontology_triples=ontology_triples,
        reloaded_triples=len(g_reloaded),
        build_seconds=build_seconds,
        reload_seconds=reload_seconds,
        file_bytes=file_bytes,
    )
    stats_df.to_csv(TABLES_DIR / "07_graph_stats.csv", index=False)

    total_elapsed = time.perf_counter() - t0
    print(
        f"[Graph Built] loans={len(annotated_df):,}, borrowers={annotated_df['UserName'].nunique():,}, "
        f"triples={len(g):,} (reloaded={len(g_reloaded):,}), "
        f"build={build_seconds:.2f}s, reload={reload_seconds:.2f}s, "
        f"size={file_bytes / (1024 * 1024):.2f} MB, total={total_elapsed:.2f}s"
    )
    return {
        "graph": g,
        "sample_df": annotated_df,
        "sample_summary": samp["summary_df"],
        "stats_df": stats_df,
    }


if __name__ == "__main__":
    main()
