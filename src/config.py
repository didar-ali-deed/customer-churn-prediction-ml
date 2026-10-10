"""Central paths and reproducibility settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/telco.csv"
PROCESSED = ROOT / "data/processed/clean.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
SEED = 42


def ensure_dirs() -> None:
    """Create artifact directories without touching existing files."""
    for path in (PROCESSED.parent, REPORTS, FIGURES, MODELS):
        path.mkdir(parents=True, exist_ok=True)
