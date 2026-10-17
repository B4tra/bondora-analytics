"""Environment and project structure verification script for Bondora Analytics."""

import importlib
import sys
from pathlib import Path

# Add project root to sys.path so src.config can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_EXTERNAL,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES_DIR,
    LOANDATA_FILENAME,
    MODELS_DIR,
    ONTOLOGY_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)

# List of (import_name, display_name) for required packages
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("sklearn", "scikit-learn (sklearn)"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("imblearn", "imbalanced-learn (imblearn)"),
    ("shap", "shap"),
    ("pulp", "pulp"),
    ("rdflib", "rdflib"),
    ("networkx", "networkx"),
    ("streamlit", "streamlit"),
    ("joblib", "joblib"),
    ("docx", "python-docx (docx)"),
]

REQUIRED_DIRS: list[Path] = [
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_EXTERNAL,
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "data",
    PROJECT_ROOT / "src" / "features",
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "src" / "utils",
    MODELS_DIR,
    PROJECT_ROOT / "dashboard",
    ONTOLOGY_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    PROJECT_ROOT / "scripts",
]


def check_packages() -> list[str]:
    """Import required packages, print version table, and return failed imports."""
    print("=" * 60)
    print(f"{'Package':<30} | {'Version / Status':<25}")
    print("-" * 60)

    failed_packages: list[str] = []
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{display_name:<30} | {version:<25}")
        except Exception as exc:
            print(f"{display_name:<30} | FAILED ({exc.__class__.__name__})")
            failed_packages.append(f"{display_name}: {exc}")

    print("=" * 60)
    return failed_packages


def check_directories() -> list[str]:
    """Verify that all target project directories exist."""
    missing_dirs: list[str] = []
    print("\nChecking project directory structure...")
    for directory in REQUIRED_DIRS:
        rel_path = directory.relative_to(PROJECT_ROOT)
        if not directory.is_dir():
            missing_dirs.append(str(rel_path))
            print(f"  [MISSING] {rel_path}")
    if not missing_dirs:
        print(f"  All {len(REQUIRED_DIRS)} required directories are present.")
    return missing_dirs


def check_raw_data() -> None:
    """Check if raw dataset exists in data/raw/ and print a warning if missing."""
    print("\nChecking raw dataset availability...")
    loandata_path = DATA_RAW / LOANDATA_FILENAME
    if loandata_path.is_file():
        print(f"  [FOUND] {loandata_path.relative_to(PROJECT_ROOT)}")
    else:
        print(
            f"  [WARNING] File dataset '{LOANDATA_FILENAME}' belum ditemukan di "
            f"'{DATA_RAW.relative_to(PROJECT_ROOT)}/'."
        )
        print(
            "            Silakan unduh dataset Bondora secara manual dan simpan ke "
            "`data/raw/` (lihat `data/README.md` untuk panduan lengkap)."
        )


def main() -> int:
    """Run all verification checks and return an appropriate exit code."""
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Project root      : {PROJECT_ROOT}\n")

    failed_packages = check_packages()
    missing_dirs = check_directories()
    check_raw_data()

    print("\n" + "=" * 60)
    if failed_packages or missing_dirs:
        print("STATUS: PROBLEMS DETECTED")
        if failed_packages:
            print("  Failed package imports:")
            for item in failed_packages:
                print(f"    - {item}")
        if missing_dirs:
            print("  Missing directories:")
            for item in missing_dirs:
                print(f"    - {item}")
        print("=" * 60)
        return 1

    print("STATUS: OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
