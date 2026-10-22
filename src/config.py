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

# Dataset snapshot report date (from ReportAsOfEOD)
REPORT_DATE: str = "2021-07-20"

# Temporal split boundary dates for Tahap 4 (Predictive Analytics)
SPLIT_TRAIN_END: str = "2018-06-30"
SPLIT_VAL_END: str = "2019-06-30"

# Raw dataset filenames
# TODO: konfirmasi nama file sesuai unduhan pengguna
LOANDATA_FILENAME: str = "LoanData.csv"
REPAYMENTS_FILENAME: str = "RepaymentsData.csv"

# Core column names and status values for target definitions (Gate 1 & Gate 2 decisions)
STATUS_COL: str = "Status"
DEFAULT_DATE_COL: str = "DefaultDate"
LOAN_DATE_COL: str = "LoanDate"
AMOUNT_COL: str = "Amount"
APPLIED_AMOUNT_COL: str = "AppliedAmount"
PRINCIPAL_PAYMENTS_COL: str = "PrincipalPaymentsMade"
INTEREST_PENALTY_PAYMENTS_COL: str = "InterestAndPenaltyPaymentsMade"
PRINCIPAL_WRITEOFFS_COL: str = "PrincipalWriteOffs"
INTEREST_PENALTY_WRITEOFFS_COL: str = "InterestAndPenaltyWriteOffs"

STATUS_REPAID: str = "Repaid"
STATUS_CURRENT: str = "Current"
STATUS_LATE: str = "Late"

# Output target column names
DEFAULT_TARGET_COL: str = "default"
REGRESSION_TARGET_COL: str = "return_ratio"
IS_RESOLVED_COL: str = "is_resolved"
OBSERVABLE_12M_COL: str = "observable_12m"
DEFAULT_12M_COL: str = "default_12m"
OBSERVABLE_24M_COL: str = "observable_24m"
DEFAULT_24M_COL: str = "default_24m"

# Columns dropped from bondora_clean.parquet (4 columns 100% missing + 1 constant snapshot column)
DROPPED_COLUMNS: list[str] = [
    "City",
    "County",
    "DateOfBirth",
    "EmploymentPosition",
    "ReportAsOfEOD",
]


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
