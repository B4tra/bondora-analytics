"""Visualizations for Tahap 6 Semantic Analytics (`reports/figures/07_*.png` + `reports/07_kg_interactive.html`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    LOAN_DATE_COL,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from src.semantic.build_graph import SAMPLE_PARQUET_PATH, prepare_annotated_sample_df


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str = "#e8f1fa",
    edgecolor: str = "#1f4e78",
    fontsize: float = 9.5,
    fontweight: str = "bold",
    textcolor: str = "#10253f",
) -> None:
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=4,
    )


def _draw_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str = "",
    *,
    rad: float = 0.0,
    color: str = "#334155",
    linestyle: str = "-",
    label_offset: tuple[float, float] = (0.0, 0.0),
    fontsize: float = 8.2,
) -> None:
    arrow = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6,head_width=3.8",
        color=color,
        linestyle=linestyle,
        linewidth=1.35,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cbd5e1", alpha=0.92, lw=0.7),
            zorder=5,
        )


def plot_ontology_schema(output_path: Path = FIGURES_DIR / "07_ontology_schema.png") -> Path:
    """1. Plot `07_ontology_schema.png`: Ontology class, property, and defined-class schema diagram."""
    fig, ax = plt.subplots(figsize=(14.5, 9.0), dpi=160)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Build a networkx representation to verify structural connectivity of the schema
    G_schema = nx.DiGraph()
    schema_edges = [
        ("bo:Loan", "bo:Borrower", "bo:hasBorrower"),
        ("bo:Loan", "bo:Country", "bo:originatedIn"),
        ("bo:Borrower", "bo:Country", "bo:borrowerCountry"),
        ("bo:Loan", "bo:RatingClass", "bo:ratedAs"),
        ("bo:RatingClass", "bo:RiskTier", "skos:broader"),
        ("bo:Loan", "bo:OutcomeGroup", "bo:hasOutcome"),
        ("bo:Loan", "bo:Era", "bo:inEra"),
        ("bo:Loan", "bo:TenorBand", "bo:hasTenorBand"),
        ("bo:Loan", "bo:RiskBand", "bo:inRiskBand"),
        ("bo:Loan", "bo:Strategy", "bo:selectedBy"),
        ("bo:Loan", "bo:Loan", "bo:precededBy"),
        ("bo:DefaultedLoan", "bo:Loan", "rdfs:subClassOf"),
        ("bo:CensoredLoan", "bo:Loan", "rdfs:subClassOf"),
        ("bo:HighRiskRatedLoan", "bo:Loan", "rdfs:subClassOf"),
    ]
    for u, v, lbl in schema_edges:
        G_schema.add_edge(u, v, label=lbl)

    # Coordinates for manual layout
    pos = {
        "bo:Loan": (0.48, 0.52),
        "bo:Borrower": (0.15, 0.68),
        "bo:Country": (0.15, 0.36),
        "bo:RatingClass": (0.82, 0.78),
        "bo:RiskTier": (0.82, 0.92),
        "bo:OutcomeGroup": (0.82, 0.58),
        "bo:Era": (0.82, 0.42),
        "bo:TenorBand": (0.82, 0.26),
        "bo:RiskBand": (0.82, 0.11),
        "bo:Strategy": (0.48, 0.14),
        # Defined classes (top)
        "bo:DefaultedLoan": (0.24, 0.88),
        "bo:CensoredLoan": (0.48, 0.88),
        "bo:HighRiskRatedLoan": (0.48, 0.74),
    }

    # Draw Core Classes
    _draw_box(
        ax,
        *pos["bo:Loan"],
        0.22,
        0.11,
        "bo:Loan\n(Pinjaman P2P)\n[n = 20.000 di graf]",
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        fontsize=10.5,
    )
    _draw_box(
        ax,
        *pos["bo:Borrower"],
        0.21,
        0.09,
        "bo:Borrower\n(Peminjam Pseudonim)\n[tanpa Gender & Age]",
        facecolor="#e0f2fe",
        edgecolor="#0369a1",
        fontsize=9.2,
    )
    _draw_box(
        ax,
        *pos["bo:Country"],
        0.19,
        0.085,
        "bo:Country\n(EE, ES, FI, SK)",
        facecolor="#dcfce7",
        edgecolor="#15803d",
        fontsize=9.2,
    )
    _draw_box(
        ax,
        *pos["bo:RatingClass"],
        0.23,
        0.085,
        "bo:RatingClass\n(AA, A, B, C, D, E, F, HR,\nRatingTidakTersedia)",
        facecolor="#fef3c7",
        edgecolor="#b45309",
        fontsize=8.8,
    )
    _draw_box(
        ax,
        *pos["bo:RiskTier"],
        0.25,
        0.075,
        "bo:RiskTier (skos:Concept)\n(Rendah / Menengah / Tinggi)\n*definisi proyek*",
        facecolor="#fde68a",
        edgecolor="#92400e",
        fontsize=8.6,
    )
    _draw_box(
        ax,
        *pos["bo:OutcomeGroup"],
        0.24,
        0.085,
        "bo:OutcomeGroup (skos:Concept)\n(Default, RepaidNoDefault,\nCurrent, LateNotDefault)",
        facecolor="#fce7f3",
        edgecolor="#be185d",
        fontsize=8.5,
    )
    _draw_box(
        ax,
        *pos["bo:Era"],
        0.24,
        0.08,
        "bo:Era (Pra/Pasca 2018)\n*definisi proyek*",
        facecolor="#f3e8ff",
        edgecolor="#6d28d9",
        fontsize=8.8,
    )
    _draw_box(
        ax,
        *pos["bo:TenorBand"],
        0.24,
        0.08,
        "bo:TenorBand (5 Pita Tenor)\n*definisi proyek*",
        facecolor="#f3e8ff",
        edgecolor="#6d28d9",
        fontsize=8.8,
    )
    _draw_box(
        ax,
        *pos["bo:RiskBand"],
        0.24,
        0.085,
        "bo:RiskBand (p_calibrated)\n(<0,15 / 0,15–<0,35 / ≥0,35)\n*definisi proyek*",
        facecolor="#f3e8ff",
        edgecolor="#6d28d9",
        fontsize=8.5,
    )
    _draw_box(
        ax,
        *pos["bo:Strategy"],
        0.22,
        0.08,
        "bo:Strategy\n(bo:StrategiS3 Tahap 5)",
        facecolor="#ccfbf1",
        edgecolor="#0f766e",
        fontsize=9.0,
    )

    # Draw Defined Classes (OWL-RL Inference)
    _draw_box(
        ax,
        *pos["bo:DefaultedLoan"],
        0.21,
        0.075,
        "bo:DefaultedLoan\n(≡ Loan ⊓ hasOutcome Default)",
        facecolor="#fee2e2",
        edgecolor="#b91c1c",
        fontsize=8.3,
    )
    _draw_box(
        ax,
        *pos["bo:CensoredLoan"],
        0.22,
        0.075,
        "bo:CensoredLoan\n(≡ Loan ⊓ Current/LateNotDefault)",
        facecolor="#fee2e2",
        edgecolor="#b91c1c",
        fontsize=8.3,
    )
    _draw_box(
        ax,
        *pos["bo:HighRiskRatedLoan"],
        0.24,
        0.075,
        "bo:HighRiskRatedLoan\n(≡ Loan ⊓ ratedAs E/F/HR)",
        facecolor="#fee2e2",
        edgecolor="#b91c1c",
        fontsize=8.3,
    )

    # Draw Datatype Properties box attached to bo:Loan
    _draw_box(
        ax,
        0.16,
        0.13,
        0.26,
        0.19,
        "Properti Data (owl:DatatypeProperty)\npada bo:Loan:\n"
        "• bo:amount (xsd:decimal)\n"
        "• bo:interestPct (xsd:decimal)\n"
        "• bo:loanDurationMonths (xsd:integer)\n"
        "• bo:loanDate (xsd:date)\n"
        "• bo:predictedDefaultProb12m (xsd:decimal)\n"
        "• bo:defaultWithin12m (xsd:boolean, opsional)\n"
        "• bo:returnRatio (xsd:decimal, selesai)\n"
        "• bo:isResolved (xsd:boolean)",
        facecolor="#f8fafc",
        edgecolor="#475569",
        fontsize=8.1,
        fontweight="normal",
    )

    # Draw arrows
    _draw_arrow(ax, (0.37, 0.555), (0.255, 0.65), "bo:hasBorrower", label_offset=(-0.015, 0.015))
    _draw_arrow(ax, (0.37, 0.49), (0.245, 0.38), "bo:originatedIn", label_offset=(-0.01, -0.02))
    _draw_arrow(ax, (0.15, 0.635), (0.15, 0.405), "bo:borrowerCountry", label_offset=(-0.01, 0.0))
    _draw_arrow(ax, (0.59, 0.56), (0.705, 0.76), "bo:ratedAs", label_offset=(-0.01, 0.03))
    _draw_arrow(ax, (0.82, 0.823), (0.82, 0.882), "skos:broader", label_offset=(0.06, 0.0))
    _draw_arrow(ax, (0.59, 0.535), (0.70, 0.58), "bo:hasOutcome", label_offset=(0.0, 0.02))
    _draw_arrow(ax, (0.59, 0.50), (0.70, 0.42), "bo:inEra", label_offset=(0.0, 0.015))
    _draw_arrow(ax, (0.59, 0.48), (0.70, 0.27), "bo:hasTenorBand", label_offset=(-0.01, -0.01))
    _draw_arrow(ax, (0.56, 0.465), (0.70, 0.13), "bo:inRiskBand", label_offset=(-0.02, -0.03))
    _draw_arrow(ax, (0.48, 0.465), (0.48, 0.18), "bo:selectedBy", label_offset=(0.0, 0.01))
    _draw_arrow(ax, (0.37, 0.47), (0.29, 0.22), "properti literal", linestyle=":", color="#64748b")

    # Self-loop for bo:precededBy on bottom-left of bo:Loan
    _draw_arrow(
        ax,
        (0.41, 0.465),
        (0.45, 0.465),
        "bo:precededBy\n(rantai peminjaman ulang)",
        rad=-1.4,
        color="#0f766e",
        label_offset=(-0.06, -0.075),
    )

    # SubClassOf dashed arrows from Defined Classes to bo:Loan
    _draw_arrow(ax, (0.28, 0.84), (0.42, 0.575), "rdfs:subClassOf (OWL-RL)", linestyle="--", color="#b91c1c", label_offset=(-0.055, 0.035))
    _draw_arrow(ax, (0.48, 0.84), (0.48, 0.78), "", linestyle="--", color="#b91c1c")
    _draw_arrow(ax, (0.48, 0.70), (0.48, 0.575), "rdfs:subClassOf (OWL-RL)", linestyle="--", color="#b91c1c", label_offset=(0.075, 0.0))

    ax.set_title(
        "Skema Ontologi Bondora (bo: http://example.org/bondora#) — Kelas, Relasi Objek, Properti Data, dan Kelas Inferensi",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def select_example_borrower_loans(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Select an informative borrower with >= 3 loans (ideally 3-4 loans), >= 1 Default, and >= 1 S3 loan."""
    # Look for borrowers with 3 or 4 loans in graph, at least 1 Default followed by another loan, and at least 1 S3 loan
    counts = df_prep.groupby("UserName", observed=True)["LoanNumber"].count()
    cand_users = counts[(counts >= 3) & (counts <= 4)].index.tolist()
    sub = df_prep[df_prep["UserName"].isin(cand_users)].copy()

    best_user = None
    for uname, grp in sub.groupby("UserName", observed=True):
        has_def_prev = (grp["prev_outcome_code"] == "Default").any()
        has_s3 = grp["in_s3"].any()
        if has_def_prev and has_s3:
            best_user = str(uname)
            break

    if best_user is None:
        for uname, grp in sub.groupby("UserName", observed=True):
            if (grp["outcome_code"] == "Default").any():
                best_user = str(uname)
                break

    assert best_user is not None, "Could not find borrower with >= 3 loans and >= 1 Default"
    res = df_prep[df_prep["UserName"] == best_user].sort_values([LOAN_DATE_COL, "LoanNumber"]).reset_index(drop=True)
    return res


def plot_example_subgraph(
    df_prep: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "07_kg_example_subgraph.png",
) -> dict[str, Any]:
    """2. Plot `07_kg_example_subgraph.png`: Subgraph of 1 borrower with >= 3 loans and >= 1 Default."""
    b_loans = select_example_borrower_loans(df_prep)
    uname = str(b_loans["UserName"].iloc[0])
    c_code = str(b_loans["Country"].iloc[0])
    loan_numbers = [int(x) for x in b_loans["LoanNumber"].tolist()]

    fig, ax = plt.subplots(figsize=(14.5, 8.5), dpi=160)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Borrower and Country in dedicated left column
    b_pos = (0.105, 0.74)
    c_pos = (0.105, 0.30)
    _draw_box(
        ax,
        *b_pos,
        0.18,
        0.09,
        f"bo:Borrower_{uname}\n(rdf:type bo:Borrower)",
        facecolor="#e0f2fe",
        edgecolor="#0369a1",
        fontsize=8.8,
    )
    _draw_box(
        ax,
        *c_pos,
        0.17,
        0.08,
        f"bo:Country_{c_code}\n(Negara: {c_code})",
        facecolor="#dcfce7",
        edgecolor="#15803d",
        fontsize=8.8,
    )
    _draw_arrow(ax, (b_pos[0], b_pos[1] - 0.045), (c_pos[0], c_pos[1] + 0.04), "bo:borrowerCountry", fontsize=7.8)

    n_loans = len(b_loans)
    xs = np.linspace(0.35, 0.87, n_loans)
    loan_y = 0.50

    prev_pos = None
    for idx, row in b_loans.iterrows():
        x = float(xs[idx])
        l_num = int(row["LoanNumber"])
        l_date = pd.to_datetime(row[LOAN_DATE_COL]).strftime("%Y-%m-%d")
        amt = float(row["Amount"])
        rate_pct = float(row["Interest"])
        p_cal = float(row["p_calibrated"])
        r_code = str(row["rating_clean"])
        rt_code = str(row["risk_tier"])
        out_code = str(row["outcome_code"])
        rb_code = str(row["risk_band"])
        in_s3 = bool(row["in_s3"])

        # Loan box
        _draw_box(
            ax,
            x,
            loan_y,
            0.20,
            0.13,
            f"bo:Loan_{l_num}\n"
            f"Tanggal: {l_date}\n"
            f"Pokok: EUR {amt:,.0f} | Bunga: {rate_pct:.1f}%\n"
            f"p_calibrated: {p_cal:.3f}",
            facecolor="#dbeafe",
            edgecolor="#1d4ed8",
            fontsize=8.3,
        )

        # Link to Borrower and Country (only label first loan to avoid clutter, draw subtle lines for others)
        _draw_arrow(
            ax,
            (x - 0.06, loan_y + 0.065),
            (b_pos[0] + 0.09, b_pos[1] - 0.01),
            "bo:hasBorrower" if idx == 0 else "",
            rad=-0.06,
            fontsize=7.4,
            label_offset=(0.0, 0.02),
        )
        _draw_arrow(
            ax,
            (x - 0.06, loan_y - 0.065),
            (c_pos[0] + 0.085, c_pos[1] + 0.01),
            "bo:originatedIn" if idx == 0 else "",
            rad=0.06,
            fontsize=7.4,
            label_offset=(0.0, -0.02),
        )

        # Link precededBy to previous loan
        if prev_pos is not None:
            _draw_arrow(
                ax,
                (x - 0.10, loan_y),
                (prev_pos[0] + 0.10, loan_y),
                "bo:precededBy",
                color="#0f766e",
                label_offset=(0.0, 0.032),
                fontsize=8.0,
            )
        prev_pos = (x, loan_y)

        # Rating & RiskTier directly above loan
        r_y = 0.75
        rt_y = 0.90
        _draw_box(
            ax,
            x,
            r_y,
            0.15,
            0.065,
            f"bo:Rating_{r_code}",
            facecolor="#fef3c7",
            edgecolor="#b45309",
            fontsize=8.2,
        )
        _draw_box(
            ax,
            x,
            rt_y,
            0.17,
            0.065,
            f"bo:{rt_code}\n(SKOS RiskTier)",
            facecolor="#fde68a",
            edgecolor="#92400e",
            fontsize=7.8,
        )
        _draw_arrow(ax, (x, loan_y + 0.065), (x, r_y - 0.033), "bo:ratedAs", fontsize=7.4)
        _draw_arrow(ax, (x, r_y + 0.033), (x, rt_y - 0.033), "skos:broader", fontsize=7.4)

        # Outcome, RiskBand, and Strategy below
        out_color = "#fee2e2" if out_code == "Default" else "#fce7f3"
        out_edge = "#b91c1c" if out_code == "Default" else "#be185d"
        _draw_box(
            ax,
            x - 0.05,
            0.27,
            0.14,
            0.065,
            f"bo:{out_code}\n(Outcome)",
            facecolor=out_color,
            edgecolor=out_edge,
            fontsize=7.7,
        )
        _draw_arrow(ax, (x - 0.03, loan_y - 0.065), (x - 0.05, 0.27 + 0.033), "bo:hasOutcome", fontsize=7.2)

        _draw_box(
            ax,
            x + 0.05,
            0.14,
            0.15,
            0.065,
            f"bo:{rb_code}\n(RiskBand)",
            facecolor="#f3e8ff",
            edgecolor="#6d28d9",
            fontsize=7.6,
        )
        _draw_arrow(ax, (x + 0.03, loan_y - 0.065), (x + 0.05, 0.14 + 0.033), "bo:inRiskBand", fontsize=7.2)

        if in_s3:
            _draw_box(
                ax,
                x - 0.05,
                0.04,
                0.16,
                0.058,
                "bo:StrategiS3\n(Portofolio S3)",
                facecolor="#ccfbf1",
                edgecolor="#0f766e",
                fontsize=7.6,
            )
            _draw_arrow(ax, (x - 0.05, 0.27 - 0.033), (x - 0.05, 0.04 + 0.03), "", linestyle=":", color="#94a3b8")
            _draw_arrow(ax, (x, loan_y - 0.065), (x - 0.05, 0.04 + 0.03), "bo:selectedBy", fontsize=7.2, label_offset=(-0.02, -0.05))

    ax.set_title(
        f"Contoh Subgraf Knowledge Graph — Peminjam bo:Borrower_{uname} ({n_loans} Pinjaman: LoanNumber {loan_numbers})\n"
        "Menampilkan Rantai Peminjaman Ulang (bo:precededBy), Taksonomi Rating SKOS, Hasil Pinjaman, dan Seleksi S3",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "username": uname,
        "country": c_code,
        "loan_numbers": loan_numbers,
        "n_loans": n_loans,
        "loans_df": b_loans[["LoanNumber", LOAN_DATE_COL, "Country", "rating_clean", "risk_tier", "outcome_code", "p_calibrated", "in_s3", "prev_loan_number"]],
    }


def plot_taxonomy_rating_risk(output_path: Path = FIGURES_DIR / "07_taxonomy_rating_risk.png") -> Path:
    """3. Plot `07_taxonomy_rating_risk.png`: SKOS Rating -> RiskTier taxonomy tree and 4 OutcomeGroup concepts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.2), dpi=160, gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Left panel: SKOS Rating -> RiskTier
    ax1.set_xlim(-0.02, 1.04)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")

    _draw_box(
        ax1,
        0.50,
        0.90,
        0.62,
        0.09,
        "Skema Taksonomi Tingkat Risiko Rating (skos:Concept)\n*Definisi Proyek mengikuti urutan huruf rating Bondora*",
        facecolor="#fef3c7",
        edgecolor="#92400e",
        fontsize=9.0,
    )

    tiers = [
        (
            "bo:TingkatRendah\n(Risiko Rendah)",
            0.17,
            0.62,
            0.19,
            "#dcfce7",
            "#15803d",
            [("bo:Rating_AA\n(rank 1)", 0.05, 0.28), ("bo:Rating_A\n(rank 2)", 0.17, 0.14), ("bo:Rating_B\n(rank 3)", 0.29, 0.28)],
        ),
        (
            "bo:TingkatMenengah\n(Risiko Menengah)",
            0.44,
            0.62,
            0.19,
            "#fef9c3",
            "#a16207",
            [("bo:Rating_C\n(rank 4)", 0.39, 0.14), ("bo:Rating_D\n(rank 5)", 0.50, 0.28)],
        ),
        (
            "bo:TingkatTinggi\n(Risiko Tinggi)",
            0.70,
            0.62,
            0.19,
            "#fee2e2",
            "#b91c1c",
            [("bo:Rating_E\n(rank 6)", 0.61, 0.14), ("bo:Rating_F\n(rank 7)", 0.71, 0.28), ("bo:Rating_HR\n(rank 8)", 0.81, 0.14)],
        ),
        (
            "bo:Tingkat\nTidakTersedia",
            0.93,
            0.62,
            0.15,
            "#e2e8f0",
            "#475569",
            [("bo:Rating\nTidakTersedia\n(rank 9)", 0.93, 0.28)],
        ),
    ]

    for idx_t, (t_label, tx, ty, tw, t_fc, t_ec, children) in enumerate(tiers):
        _draw_box(ax1, tx, ty, tw, 0.10, t_label, facecolor=t_fc, edgecolor=t_ec, fontsize=8.0)
        _draw_arrow(
            ax1,
            (tx, ty + 0.05),
            (0.50, 0.855),
            "skos:inScheme" if idx_t in (0, 3) else "",
            linestyle=":",
            color="#64748b",
            fontsize=7.2,
        )
        for idx_c, (c_label, cx, cy) in enumerate(children):
            _draw_box(ax1, cx, cy, 0.105, 0.10, c_label, facecolor="#ffffff", edgecolor=t_ec, fontsize=7.6)
            _draw_arrow(
                ax1,
                (cx, cy + 0.05),
                (tx, ty - 0.05),
                "skos:broader" if idx_c == 0 else "",
                color=t_ec,
                fontsize=7.1,
                label_offset=(0.0, 0.02),
            )

    ax1.set_title("A. Pohon Taksonomi SKOS: bo:RatingClass → bo:RiskTier", fontsize=11.0, fontweight="bold")

    # Right panel: 4 OutcomeGroup + 3 RiskBand + 2 Era
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.axis("off")

    _draw_box(
        ax2,
        0.50,
        0.90,
        0.78,
        0.085,
        "bo:OutcomeGroup (4 Kelompok Hasil Pinjaman — Tahap 2)",
        facecolor="#fce7f3",
        edgecolor="#be185d",
        fontsize=9.2,
    )
    outcomes = [
        ("bo:Default\n(Gagal bayar: DefaultDate ada)", 0.26, 0.72, "#fee2e2", "#b91c1c"),
        ("bo:RepaidNoDefault\n(Lunas tanpa default)", 0.74, 0.72, "#dcfce7", "#15803d"),
        ("bo:Current\n(Berjalan lancar, tersensor)", 0.26, 0.55, "#e0f2fe", "#0369a1"),
        ("bo:LateNotDefault\n(Terlambat belum default, tersensor)", 0.74, 0.55, "#fef3c7", "#b45309"),
    ]
    for o_lbl, ox, oy, ofc, oec in outcomes:
        _draw_box(ax2, ox, oy, 0.42, 0.095, o_lbl, facecolor=ofc, edgecolor=oec, fontsize=8.2)

    _draw_box(
        ax2,
        0.50,
        0.36,
        0.82,
        0.075,
        "bo:RiskBand (Pita Skor Model p_calibrated — *Definisi Proyek*)",
        facecolor="#f3e8ff",
        edgecolor="#6d28d9",
        fontsize=8.8,
    )
    rbands = [
        ("bo:PitaRisikoRendah\n(p < 0,15)", 0.19, 0.22),
        ("bo:PitaRisikoSedang\n(0,15 ≤ p < 0,35)", 0.50, 0.22),
        ("bo:PitaRisikoTinggi\n(p ≥ 0,35)", 0.81, 0.22),
    ]
    for r_lbl, rx, ry in rbands:
        _draw_box(ax2, rx, ry, 0.27, 0.085, r_lbl, facecolor="#faf5ff", edgecolor="#6d28d9", fontsize=8.0)

    _draw_box(
        ax2,
        0.50,
        0.07,
        0.86,
        0.08,
        "bo:Era Pencatatan (*Definisi Proyek*): bo:EraPraPerubahan (≤2017) & bo:EraPascaPerubahan (≥2018)",
        facecolor="#f1f5f9",
        edgecolor="#334155",
        fontsize=8.2,
    )

    ax2.set_title("B. Kelompok Hasil (bo:OutcomeGroup), Pita Risiko Model, dan Era", fontsize=11.0, fontweight="bold")

    fig.suptitle(
        "Taksonomi Semantik Bondora: Hirarki Tingkat Risiko Rating (SKOS) dan Pengelompokan Kontekstual",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_triples_by_predicate(
    stats_df: pd.DataFrame | None = None,
    output_path: Path = FIGURES_DIR / "07_triples_by_predicate.png",
) -> Path:
    """4. Plot `07_triples_by_predicate.png`: Horizontal bar chart of triple counts per RDF predicate."""
    if stats_df is None:
        stats_df = pd.read_csv(TABLES_DIR / "07_graph_stats.csv")

    sub = stats_df[stats_df["bagian"] == "Triple per Predikat"].copy()
    sub["jumlah"] = sub["jumlah"].astype(int)
    sub = sub.sort_values("jumlah", ascending=True).reset_index(drop=True)

    total_triples = int(stats_df.loc[stats_df["item"] == "total_triples", "jumlah"].iloc[0])

    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=160)
    colors = ["#1d4ed8" if str(x).startswith("bo:") else "#64748b" for x in sub["item"]]
    bars = ax.barh(sub["item"], sub["jumlah"], color=colors, edgecolor="#1e293b", height=0.72)

    max_val = int(sub["jumlah"].max())
    ax.set_xlim(0, max_val * 1.18)
    for bar, cnt, pct in zip(bars, sub["jumlah"], sub["persen_dari_total"]):
        ax.text(
            bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(cnt):,} ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#0f172a",
        )

    ax.set_xlabel("Jumlah Triple dalam Knowledge Graph", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Predikat RDF / OWL / SKOS / RDFS", fontsize=10.5, fontweight="bold")
    ax.set_title(
        f"Distribusi Jumlah Triple menurut Predikat pada Knowledge Graph Bondora\n"
        f"(Total = {total_triples:,} Triple dari 20.000 Pinjaman dan 7.709 Peminjam)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.45)

    legend_handles = [
        mpatches.Patch(color="#1d4ed8", label="Properti Domain Bondora (bo:*)"),
        mpatches.Patch(color="#64748b", label="Predikat Standar RDF / RDFS / OWL / SKOS"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_interactive_pyvis_html(
    df_prep: pd.DataFrame,
    output_path: Path = REPORTS_DIR / "07_kg_interactive.html",
) -> dict[str, Any]:
    """5. Optional: Build interactive `reports/07_kg_interactive.html` (< 5 MB) using `pyvis`."""
    from pyvis.network import Network

    # Select top 8 informative repeat borrowers who have at least 1 Default and >= 3 loans
    counts = df_prep.groupby("UserName", observed=True)["LoanNumber"].count()
    cand_users = counts[(counts >= 3) & (counts <= 5)].index.tolist()
    sub = df_prep[df_prep["UserName"].isin(cand_users)].copy()

    chosen_users: list[str] = []
    for uname, grp in sub.groupby("UserName", observed=True):
        if (grp["prev_outcome_code"] == "Default").any() and grp["in_s3"].any():
            chosen_users.append(str(uname))
            if len(chosen_users) >= 8:
                break

    sub_df = df_prep[df_prep["UserName"].isin(chosen_users)].sort_values(["UserName", LOAN_DATE_COL, "LoanNumber"]).copy()

    net = Network(
        height="780px",
        width="100%",
        directed=True,
        bgcolor="#f8fafc",
        font_color="#0f172a",
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-3500, central_gravity=0.3, spring_length=130)

    added_nodes: set[str] = set()

    def add_node_once(node_id: str, label: str, title: str, color: str, shape: str = "dot", size: int = 18) -> None:
        if node_id not in added_nodes:
            net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size)
            added_nodes.add(node_id)

    # Taxonomy RiskTier nodes
    for rt, col in [("bo:TingkatRendah", "#22c55e"), ("bo:TingkatMenengah", "#eab308"), ("bo:TingkatTinggi", "#ef4444")]:
        add_node_once(rt, rt, f"SKOS RiskTier: {rt} (definisi proyek)", col, shape="diamond", size=22)

    add_node_once("bo:StrategiS3", "bo:StrategiS3", "Portofolio Optimasi Berbatas S3 (Tahap 5)", "#14b8a6", shape="star", size=24)

    for _, row in sub_df.iterrows():
        uname = str(row["UserName"])
        c_code = str(row["Country"])
        l_num = int(row["LoanNumber"])
        r_code = str(row["rating_clean"])
        rt_code = f"bo:{row['risk_tier']}"
        out_code = f"bo:{row['outcome_code']}"
        l_date = pd.to_datetime(row[LOAN_DATE_COL]).strftime("%Y-%m-%d")
        p_cal = float(row["p_calibrated"])

        b_id = f"bo:Borrower_{uname}"
        c_id = f"bo:Country_{c_code}"
        l_id = f"bo:Loan_{l_num}"
        r_id = f"bo:Rating_{r_code}"

        add_node_once(b_id, b_id, f"Peminjam: {b_id} (Negara: {c_code})", "#0284c7", shape="triangle", size=20)
        add_node_once(c_id, c_id, f"Negara: {c_code}", "#16a34a", shape="box", size=20)
        add_node_once(
            l_id,
            f"Loan_{l_num}\n({l_date})",
            f"bo:Loan_{l_num} | Tanggal: {l_date} | Hasil: {out_code} | p_calibrated: {p_cal:.3f}",
            "#dc2626" if out_code == "bo:Default" else "#2563eb",
            shape="dot",
            size=18,
        )
        add_node_once(r_id, r_id, f"Kelas Rating: {r_id}", "#f59e0b", shape="ellipse", size=16)
        add_node_once(out_code, out_code, f"Kelompok Hasil: {out_code}", "#db2777", shape="box", size=18)

        net.add_edge(l_id, b_id, label="hasBorrower", color="#64748b")
        net.add_edge(b_id, c_id, label="borrowerCountry", color="#16a34a")
        net.add_edge(l_id, r_id, label="ratedAs", color="#d97706")
        if rt_code in added_nodes:
            net.add_edge(r_id, rt_code, label="skos:broader", color="#b45309")
        net.add_edge(l_id, out_code, label="hasOutcome", color="#be185d")

        if bool(row["in_s3"]):
            net.add_edge(l_id, "bo:StrategiS3", label="selectedBy", color="#0d9488")

        if not pd.isna(row["prev_loan_number"]):
            prev_id = f"bo:Loan_{int(row['prev_loan_number'])}"
            net.add_edge(l_id, prev_id, label="precededBy", color="#0f766e", width=2.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_str = net.generate_html()
    output_path.write_text(html_str, encoding="utf-8")
    size_bytes = int(output_path.stat().st_size)
    assert size_bytes < 5 * 1024 * 1024, f"Interactive HTML size {size_bytes} >= 5 MB"
    return {
        "path": str(output_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "n_borrowers": len(chosen_users),
        "n_loans": len(sub_df),
    }


def generate_all_semantic_visualizations(
    sample_df: pd.DataFrame | None = None,
    stats_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate all 4 mandatory figures (`07_*.png`) and the optional interactive HTML (`07_kg_interactive.html`)."""
    ensure_dirs()
    if sample_df is None:
        sample_df = pd.read_parquet(SAMPLE_PARQUET_PATH)
    df_prep = prepare_annotated_sample_df(sample_df, score_if_missing=False)

    p1 = plot_ontology_schema()
    sub_info = plot_example_subgraph(df_prep)
    p3 = plot_taxonomy_rating_risk()
    p4 = plot_triples_by_predicate(stats_df=stats_df)
    html_info = generate_interactive_pyvis_html(df_prep)

    return {
        "fig_ontology_schema": str(p1),
        "fig_example_subgraph": str(FIGURES_DIR / "07_kg_example_subgraph.png"),
        "example_subgraph_info": sub_info,
        "fig_taxonomy": str(p3),
        "fig_triples_by_predicate": str(p4),
        "interactive_html_info": html_info,
    }


if __name__ == "__main__":
    info = generate_all_semantic_visualizations()
    print("[Visualizations Generated]")
    print("Example Borrower:", info["example_subgraph_info"]["username"])
    print("Example LoanNumbers:", info["example_subgraph_info"]["loan_numbers"])
    print("Interactive HTML:", info["interactive_html_info"])
