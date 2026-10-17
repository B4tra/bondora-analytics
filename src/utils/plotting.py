"""Plotting style, Indonesian number formatting, and figure export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

# Consistent color palette for the 4 loan outcome groups across all charts
OUTCOME_PALETTE: dict[str, str] = {
    "Default": "#c44e52",
    "Lunas tanpa default": "#2ca02c",
    "Berjalan (Current)": "#1f77b4",
    "Berjalan": "#1f77b4",
    "Terlambat belum default": "#ff7f0e",
}

DEFAULT_FIGSIZE: tuple[float, float] = (10.0, 5.5)
DEFAULT_DPI: int = 150


def set_style() -> dict[str, str]:
    """Configure a clean seaborn/matplotlib theme with 10x5.5 inches and dpi=150.

    Returns
    -------
    dict[str, str]
        The standard ``OUTCOME_PALETTE`` mapping outcome group names to hex colors.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "grid.alpha": 0.35,
        }
    )
    return OUTCOME_PALETTE.copy()


def fmt_id(
    x: Any,
    decimals: int = 2,
    is_int: bool | None = None,
) -> str:
    """Format a number using Indonesian conventions (dot thousands, comma decimals).

    Parameters
    ----------
    x : int, float, or number-like
        Numeric value to format.
    decimals : int, default 2
        Number of decimal digits when formatting floating-point values.
    is_int : bool or None, default None
        If ``True``, formats as integer without decimals. If ``None``,
        automatically detects integer types.

    Returns
    -------
    str
        Formatted string (e.g. ``179.235`` or ``1.234,56``), or ``"-"`` for NaN/None.
    """
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) or pd.isna(x):
        return "-"

    if is_int is None:
        is_int = isinstance(x, (int, np.integer)) and not isinstance(x, bool)

    if is_int or decimals == 0:
        val_int = int(round(float(x)))
        return f"{val_int:,}".replace(",", ".")

    val_float = float(x)
    us_str = f"{val_float:,.{decimals}f}"
    # Swap ',' -> '.' and '.' -> ','
    return us_str.replace(",", "_TMP_").replace(".", ",").replace("_TMP_", ".")


def save_fig(
    fig: plt.Figure,
    name: str,
    figures_dir: Path = FIGURES_DIR,
    close: bool = True,
) -> Path:
    """Save a matplotlib Figure to ``reports/figures/`` with ``02_`` prefix and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    name : str
        Filename (with or without ``02_`` prefix and ``.png`` extension).
    figures_dir : Path, default FIGURES_DIR
        Target directory for saved figures.
    close : bool, default True
        Whether to call ``plt.close(fig)`` after saving.

    Returns
    -------
    Path
        Full path to the saved PNG file.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_name = name.strip()
    if not clean_name.startswith("02_"):
        clean_name = f"02_{clean_name}"
    if not clean_name.lower().endswith(".png"):
        clean_name = f"{clean_name}.png"

    out_path = figures_dir / clean_name
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path
