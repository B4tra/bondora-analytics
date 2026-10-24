"""Model inference helper for scoring loans without retraining models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR

DEFAULT12M_MODEL_PATH: Path = MODELS_DIR / "lgbm_default12m_setA.joblib"
_CACHED_BUNDLE: dict[str, Any] | None = None
_CACHED_PATH: Path | None = None


def load_default12m_bundle(model_path: Path | None = None) -> dict[str, Any]:
    """Load the saved calibrated LGBM-A classification bundle."""
    global _CACHED_BUNDLE, _CACHED_PATH
    target_path = Path(model_path) if model_path is not None else DEFAULT12M_MODEL_PATH
    if _CACHED_BUNDLE is None or _CACHED_PATH != target_path:
        _CACHED_BUNDLE = joblib.load(target_path)
        _CACHED_PATH = target_path
    return _CACHED_BUNDLE


def predict_default12m(df: pd.DataFrame, model_path: Path | None = None) -> pd.Series:
    """Compute calibrated 12-month default probability ``p_calibrated`` for rows in ``df``."""
    bundle = load_default12m_bundle(model_path)
    features: list[str] = list(bundle["features"])
    aligner = bundle["aligner"]
    model = bundle["model"]
    calibrator = bundle["calibrator"]

    X_aligned = aligner.transform(df[features])
    p_raw = np.asarray(model.predict_proba(X_aligned)[:, 1], dtype="float64")
    p_cal = np.asarray(calibrator.transform(p_raw), dtype="float64")
    return pd.Series(p_cal, index=df.index, name="p_calibrated", dtype="float64")
