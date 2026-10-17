"""Central configuration module for the Bondora Analytics project."""

from pathlib import Path

# Project root resolved relative to this file's location
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_EXTERNAL: Path = DATA_DIR / "external"

# Output and artifact directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
ONTOLOGY_DIR: Path = PROJECT_ROOT / "ontology"
DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Reproducibility seed
RANDOM_SEED: int = 42

# Raw dataset filenames
# TODO: konfirmasi nama file sesuai unduhan pengguna
LOANDATA_FILENAME: str = "LoanData.csv"
REPAYMENTS_FILENAME: str = "RepaymentsData.csv"


def ensure_dirs() -> None:
    """Create all standard project directories if they do not already exist."""
    directories = [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_EXTERNAL,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        ONTOLOGY_DIR,
        DASHBOARD_DIR,
        NOTEBOOKS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
