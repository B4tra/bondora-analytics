"""Data loading and schema inspection utilities."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_RAW, LOANDATA_FILENAME


def load_raw(path: str | Path | None = None, **kwargs: Any) -> pd.DataFrame:
    """Load a raw CSV dataset with automatic encoding fallback.

    Parameters
    ----------
    path : str | Path | None, optional
        Path to the CSV file. If ``None``, defaults to
        ``DATA_RAW / LOANDATA_FILENAME`` from ``src.config``.
    **kwargs : Any
        Additional keyword arguments forwarded to ``pandas.read_csv``.
        Defaults to ``low_memory=False`` if not explicitly specified.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the target CSV file does not exist.
    UnicodeDecodeError
        If none of the candidate encodings can decode the file.
    """
    target_path = Path(path) if path is not None else (DATA_RAW / LOANDATA_FILENAME)
    if not target_path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at: {target_path}")

    read_kwargs: dict[str, Any] = {"low_memory": False}
    read_kwargs.update(kwargs)

    if "encoding" in read_kwargs:
        encodings_to_try = [read_kwargs.pop("encoding")]
    else:
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]

    last_error: UnicodeDecodeError | None = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(target_path, encoding=enc, **read_kwargs)
            print(
                f"[load_raw] Successfully loaded '{target_path.name}' "
                f"using encoding='{enc}' (shape={df.shape[0]:,} rows x {df.shape[1]} cols)."
            )
            return df
        except UnicodeDecodeError as err:
            last_error = err

    assert last_error is not None
    raise last_error


def describe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column schema summary table for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        ``nama_kolom``, ``dtype``, ``n_missing``, ``pct_missing``,
        ``n_unique``, and ``contoh_nilai`` (first 3 non-null values).
    """
    n_rows = len(df)
    records: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        n_miss = int(series.isna().sum())
        pct_miss = round((n_miss / n_rows) * 100.0, 4) if n_rows > 0 else 0.0
        n_uniq = int(series.nunique(dropna=True))
        sample_vals = series.dropna().head(3).tolist()
        sample_str = " | ".join(str(v) for v in sample_vals)

        records.append(
            {
                "nama_kolom": str(col),
                "dtype": str(series.dtype),
                "n_missing": n_miss,
                "pct_missing": pct_miss,
                "n_unique": n_uniq,
                "contoh_nilai": sample_str,
            }
        )

    return pd.DataFrame(records)
