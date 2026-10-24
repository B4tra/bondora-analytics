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


# ---------------------------------------------------------------------------
# Tahap 5 (Prescriptive Analytics + Forecast) parameters.
# All values are fixed BEFORE running the backtest; they must not be tuned on test outcomes.
# ---------------------------------------------------------------------------
TICKET_EUR: float = 50.0  # nominal funded per loan (investor funds a slice, not the whole loan)
BUDGET_EUR: float = 250_000.0  # investor budget (about 5,000 loans)
MAX_AVG_P: float = 0.20  # cap on portfolio mean calibrated default probability
MAX_SHARE_COUNTRY: float = 0.70  # max exposure share per country
MAX_SHARE_WORST_RATINGS: float = 0.25  # max exposure share on ratings F and HR
WORST_RATINGS: tuple[str, ...] = ("F", "HR")
MAX_SHARE_TENOR60: float = 0.70  # max exposure share on 60-month loans
MIN_LOANS: int = 2_000  # minimum number of funded loans (diversification)
PD_STRESS_GRID: tuple[float, ...] = (-0.20, 0.0, 0.20, 0.50)  # multiplicative shock on p
LGD_SHIFT_GRID: tuple[float, ...] = (-0.10, 0.0, 0.10)  # additive shift on loss given default
BUDGET_SCALE_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)
MAX_SHARE_COUNTRY_GRID: tuple[float, ...] = (0.5, 0.7, 1.0)
MAX_AVG_P_GRID: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
N_RANDOM_DRAWS: int = 100  # draws averaged for the random baseline (S0)
N_BOOTSTRAP: int = 200  # bootstrap iterations for ex-post profit CIs
RATING_ORDER: tuple[str, ...] = ("AA", "A", "B", "C", "D", "E", "F", "HR")
OPT_TIME_LIMIT_S: float = 180.0  # binary solve time limit before falling back to LP relaxation

# Platform scenarios (exposure = full Amount of each approved loan)
TAU_GRID: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PLATFORM_MIN_APPROVAL: float = 0.40  # service-level analogue: minimum approval rate
PLATFORM_MAX_AVG_P: float = 0.25  # cap on mean p of approved loans
INTEREST_RAISE_CAP_PP: float = 10.0  # illustrative scenario: max interest increase (pp) toward breakeven

# Light forecast
FORECAST_START: str = "2016-01"
FORECAST_LAST_FULL_MONTH: str = "2021-06"  # July 2021 is partial (data ends 2021-07-20)
FORECAST_HORIZON: int = 4  # Jul..Oct 2021; headline months are Aug-Oct 2021
BACKTEST_MONTHS: int = 6  # rolling-origin evaluation on the last 6 full months


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
