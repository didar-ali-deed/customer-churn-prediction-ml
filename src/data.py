"""Read and inspect the genuine Telco CSV using pandas and numpy."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RAW

LOGGER = logging.getLogger(__name__)


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Load raw strings as supplied; fail with an actionable missing-file error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Download Kaggle blastchar/telco-customer-churn and place the CSV at {path}"
        )
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict[str, Any]:
    """Return shape, types, examples, missingness, duplicates, and target balance."""
    numeric_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    result = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient="records"),
        "missing": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["customerID"].duplicated().sum()),
        "unique_counts": df.nunique(dropna=False).to_dict(),
        "target_counts": df["Churn"].value_counts().to_dict(),
        "target_proportions": df["Churn"].value_counts(normalize=True).to_dict(),
        "total_charges_blank": int(df["TotalCharges"].astype("str").str.strip().eq("").sum()),
        "total_charges_numeric_missing": int(numeric_charges.isna().sum()),
        "blank_charge_tenures": df.loc[numeric_charges.isna(), "tenure"].tolist(),
    }
    LOGGER.info("Data audit: %s", result)
    return result
